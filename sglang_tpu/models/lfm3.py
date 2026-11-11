# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Native LFM3 inference with SGLang KV caches and TPU ShortConv state.

Based on LiquidAI/LFM3-1.7B's modeling_lfm3.py at revision
1da06bec6a819da10b54c67e5c90972b46d81871. Attention gates, sandwich norms and
the full-head frequency grid for partial RoPE are specific to LFM3.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.lfm2 import Lfm2ShortConv
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers
from sglang_tpu.configs.lfm3 import Lfm3Config


class Lfm3RotaryEmbedding(nn.Module):
    """Compute the checkpoint's zero-tail RoPE once per attention type/forward."""

    def __init__(self, config: Lfm3Config):
        super().__init__()
        self.layer_types = tuple(
            t for t in config._attention_layer_types if t in config.layer_types
        )
        for layer_type in self.layer_types:
            params = config.rope_parameters[layer_type]
            dim = config.head_dim
            rotated_pairs = int(params.get("partial_rotary_factor", 1.0) * dim) // 2
            # Keep the denominator and rotate-half pairing of a FULL head.
            # Shrinking rotary_dim changes both and produces a different model.
            inv_freq = 1.0 / (
                params["rope_theta"]
                ** (torch.arange(0, 2 * rotated_pairs, 2, dtype=torch.float32) / dim)
            )
            inv_freq = torch.cat(
                (inv_freq, inv_freq.new_zeros(dim // 2 - rotated_pairs))
            )
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> dict:
        embeddings = {}
        for layer_type in self.layer_types:
            inv_freq = getattr(self, f"{layer_type}_inv_freq")
            angles = positions.float().unsqueeze(-1) * inv_freq.float().unsqueeze(0)
            angles = torch.cat((angles, angles), dim=-1)
            # The HF model rounds cos/sin to the activation dtype before doing
            # the actual rotation in FP32. Preserve that order.
            embeddings[layer_type] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return embeddings


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.float().chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return (
        x.float() * cos[:, None, :].float() + rotated * sin[:, None, :].float()
    ).to(x.dtype)


class Lfm3Attention(nn.Module):
    def __init__(self, config: Lfm3Config, layer_id: int, prefix: str):
        super().__init__()
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads,
            bias=False,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            prefix=add_prefix("o_proj", prefix),
        )
        window = (
            config.sliding_window - 1
            if config.layer_types[layer_id] == "sliding_attention"
            else -1
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=window,
            prefix=add_prefix("attn", prefix),
        )

    def forward(self, hidden_states, position_embeddings, forward_batch):
        tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(
            tokens, self.num_heads, self.head_dim
        )
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(
            tokens, self.num_kv_heads, self.head_dim
        )
        cos, sin = position_embeddings
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        output = self.attn(
            q.reshape(tokens, q_size), k.reshape(tokens, kv_size), v, forward_batch
        )
        gate, _ = self.gate_proj(hidden_states)
        output = output.reshape(tokens, self.num_heads, self.head_dim)
        output = output * gate.sigmoid().unsqueeze(-1)
        output, _ = self.o_proj(output.reshape(tokens, q_size))
        return output


class Lfm3ShortConv(Lfm2ShortConv):
    """The same gated depthwise convolution, including its compiled state contract.

The LFM3 checkpoint uses width 7. The TPU implementation derives the history
length from the weights/config, so it also handles this wider window.
"""


class Lfm3MLP(nn.Module):
    def __init__(self, config: Lfm3Config, prefix: str):
        super().__init__()
        self.w1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            prefix=add_prefix("w1", prefix),
        )
        self.w3 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            prefix=add_prefix("w3", prefix),
        )
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            prefix=add_prefix("w2", prefix),
        )

    def forward(self, hidden_states):
        gate, _ = self.w1(hidden_states)
        up, _ = self.w3(hidden_states)
        output, _ = self.w2(F.silu(gate) * up)
        return output


class Lfm3DecoderLayer(nn.Module):
    def __init__(self, config: Lfm3Config, layer_id: int, prefix: str):
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type in config._attention_layer_types
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.is_attention_layer:
            self.self_attn = Lfm3Attention(
                config, layer_id, add_prefix("self_attn", prefix)
            )
        else:
            self.conv = Lfm3ShortConv(
                config, layer_id, prefix=add_prefix("conv", prefix)
            )
        self.feed_forward = Lfm3MLP(config, add_prefix("feed_forward", prefix))
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states, position_embeddings, forward_batch):
        if forward_batch.forward_mode.is_idle():
            return hidden_states
        residual = hidden_states
        hidden_states = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            hidden_states = self.self_attn(
                hidden_states, position_embeddings[self.layer_type], forward_batch
            )
        else:
            hidden_states = self.conv(hidden_states, forward_batch)
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states = residual + hidden_states
        ffn_out = self.feed_forward(self.ffn_norm(hidden_states))
        if self.use_sandwich_norm:
            ffn_out = self.post_ffn_norm(ffn_out)
        return hidden_states + ffn_out


class Lfm3Model(nn.Module):
    def __init__(self, config: Lfm3Config, prefix: str):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.rotary_emb = Lfm3RotaryEmbedding(config)
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Lfm3DecoderLayer(config, idx, prefix),
            prefix=add_prefix("layers", prefix),
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        position_embeddings = self.rotary_emb(positions, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, forward_batch)
        return self.embedding_norm(hidden_states)


class Lfm3ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, config: Lfm3Config, quant_config=None, prefix: str = ""):
        super().__init__()
        if not isinstance(config, Lfm3Config):
            raise ValueError(
                "Use --model-config-parser lfm3 for the native TPU LFM3 model."
            )
        if not current_platform.is_out_of_tree() or not current_platform.is_tpu():
            raise ValueError("Native TPU LFM3 requires SGLANG_PLATFORM=tpu_oot.")
        if get_parallel().tp_size != 1 or get_parallel().pp_size != 1:
            raise ValueError("Native TPU LFM3 currently requires TP=1 and PP=1.")
        if quant_config is not None:
            raise ValueError("Native TPU LFM3 currently requires unquantized weights.")
        config.validate_architecture()
        config.validate_rope()
        self.config = config
        self.model = Lfm3Model(config, prefix=add_prefix("model", prefix))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def get_num_kv_cache_layers(self) -> int:
        return len(self.config.full_attention_layer_ids)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_mtp: bool = False,
    ) -> set[str]:
        if is_mtp:
            raise ValueError("LFM3 does not implement MTP weight loading.")
        params = dict(self.named_parameters())
        loaded = set()
        qkv_shards = {}
        for name, weight in weights:
            if ".rotary_emb." in name:
                continue
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                # The embedding parameter is the shared head; safetensors normally
                # stores only model.embed_tokens.weight for this checkpoint.
                continue
            if ".conv.conv.weight" in name:
                name = name.replace(".conv.conv.weight", ".conv.conv_weight")
                weight = weight.squeeze(1)
            elif ".conv.conv.bias" in name:
                name = name.replace(".conv.conv.bias", ".conv.conv_bias")

            shard = None
            for projection, shard_id in (
                ("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")
            ):
                if f".self_attn.{projection}." in name:
                    name = name.replace(f".{projection}.", ".qkv_proj.")
                    shard = shard_id
                    break
            if name not in params:
                raise ValueError(f"Unexpected LFM3 checkpoint parameter: {name}.")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard)
                shards = qkv_shards.setdefault(name, set())
                shards.add(shard)
                if shards == {"q", "k", "v"}:
                    loaded.add(name)
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Missing LFM3 checkpoint parameters: {sorted(missing)}.")
        return loaded


EntryClass = Lfm3ForCausalLM
