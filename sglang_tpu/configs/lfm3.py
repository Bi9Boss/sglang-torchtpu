# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Team. All rights reserved.
"""LFM3 configuration and its SGLang recurrent-state contract.

The checkpoint schema follows LiquidAI/LFM3-1.7B at revision
1da06bec6a819da10b54c67e5c90972b46d81871. Loading it through the ``lfm3``
config parser uses this class in every process, without remote-code patches.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from huggingface_hub.dataclasses import strict
from transformers import PretrainedConfig

if TYPE_CHECKING:
    from sglang.srt.configs.mamba_utils import Mamba2CacheParams


@strict
class Lfm3Config(PretrainedConfig):
    model_type = "lfm3"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1_000_000.0
    _attention_layer_types = ("full_attention", "sliding_attention")

    vocab_size: int = 128_000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = None
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 124_893
    bos_token_id: int | None = 124_894
    eos_token_id: int | list[int] | None = 124_900
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    conv_bias: bool = False
    conv_L_cache: int = 7
    layer_types: list[str] | None = None
    sliding_window: int | None = 256
    use_sandwich_norm: bool = True

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            if self.num_attention_heads <= 0:
                raise ValueError("LFM3 num_attention_heads must be positive.")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if i % 4 == 2
                else "sliding_attention" if i > 0 and i % 4 == 0 else "conv"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "conv": None,
            }
        self.tie_word_embeddings = kwargs.pop(
            "tie_embedding", self.tie_word_embeddings
        )
        if self.architectures is None:
            self.architectures = ["Lfm3ForCausalLM"]
        super().__post_init__(**kwargs)

    def standardize_rope_params(self):
        # Transformers' generic normalizer can mix scalar defaults into this
        # mapping or index a missing `conv` entry while constructing defaults.
        # LFM3's schema is exclusively per-layer-type; conv has no rotation.
        if not isinstance(self.rope_parameters, dict):
            raise ValueError("LFM3 rope_parameters must be a per-layer-type mapping.")
        rope = {}
        for layer_type in self._attention_layer_types:
            params = self.rope_parameters.get(layer_type)
            if params is None and layer_type not in self.layer_types:
                continue
            if not isinstance(params, dict):
                raise ValueError(f"Missing LFM3 RoPE parameters for {layer_type}.")
            rope[layer_type] = dict(params)
            rope[layer_type].setdefault("rope_type", "default")
        rope["conv"] = None
        self.rope_parameters = rope

    def validate_rope(self):
        # LFM3 rotates a prefix of the full-head frequency grid. Generic partial
        # RoPE validation/normalization does not express that convention.
        for layer_type in self._attention_layer_types:
            if layer_type not in self.layer_types:
                continue
            params = self.rope_parameters[layer_type]
            if params.get("rope_type", "default") != "default":
                raise ValueError(
                    "The TPU LFM3 implementation currently requires default RoPE."
                )
            theta = params.get("rope_theta")
            factor = params.get("partial_rotary_factor", 1.0)
            if (
                not isinstance(theta, (int, float))
                or not math.isfinite(theta)
                or theta <= 0
            ):
                raise ValueError(
                    f"Invalid LFM3 rope_theta for {layer_type}: {theta}."
                )
            if not isinstance(factor, (int, float)) or not 0 < factor <= 1:
                raise ValueError(f"Invalid LFM3 partial_rotary_factor: {factor}.")

    def validate_architecture(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("LFM3 layer_types must have num_hidden_layers entries.")
        if any(
            t not in (*self._attention_layer_types, "conv") for t in self.layer_types
        ):
            raise ValueError(f"Unsupported LFM3 layer_types: {self.layer_types}.")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window < 2
        ):
            raise ValueError("LFM3 sliding attention requires sliding_window >= 2.")
        if self.conv_L_cache < 2:
            raise ValueError("LFM3 ShortConv requires conv_L_cache >= 2.")
        if not self.linear_layer_ids or not self.full_attention_layer_ids:
            raise ValueError(
                "Native TPU LFM3 currently requires both conv and attention layers."
            )
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("LFM3 head_dim must be positive and even.")
        if (
            self.num_attention_heads <= 0
            or self.num_key_value_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError(
                "LFM3 query heads must be positive and divisible by its KV heads."
            )

    @property
    def full_attention_layer_ids(self) -> list[int]:
        # This hybrid-state interface means "uses a KV cache", not "unbounded
        # attention". Both attention types share the full KV pool; each RadixAttention
        # layer supplies its own window. No SWA eviction is enabled in this version.
        return [
            i
            for i, layer_type in enumerate(self.layer_types)
            if layer_type in self._attention_layer_types
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, layer_type in enumerate(self.layer_types)
            if layer_type == "conv"
        ]

    @property
    def mamba_chunk_size(self) -> int:
        return 1

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams | None:
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
            mamba2_state_dtype,
        )
        from sglang.srt.runtime_context import get_parallel

        if not self.linear_layer_ids:
            return None
        tp_size = get_parallel().attn_tp_size
        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=self.hidden_size,
            n_groups=1,
            num_heads=tp_size,
            head_dim=self.hidden_size,
            state_size=0,
            conv_kernel=self.conv_L_cache,
        )
        return Mamba2CacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )
