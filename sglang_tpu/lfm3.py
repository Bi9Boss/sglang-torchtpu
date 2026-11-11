# SPDX-License-Identifier: Apache-2.0
"""Register LFM3 through SGLang's config, hybrid-state and override registries."""

from __future__ import annotations

from functools import cache

from sglang.srt.configs.model_config_parser_registry import ModelConfigParserBase
from sglang_tpu.configs.lfm3 import Lfm3Config


class Lfm3ConfigParser(ModelConfigParserBase):
    def parse(self, model, trust_remote_code, revision=None, **kwargs):
        # Read JSON directly: AutoConfig with trust_remote_code=True would select
        # the checkpoint's class instead of our state-aware native configuration.
        config_dict, remaining = Lfm3Config.get_config_dict(
            model, revision=revision, **kwargs
        )
        if config_dict.get("model_type") != "lfm3":
            raise ValueError("The lfm3 config parser requires an LFM3 checkpoint.")
        config = Lfm3Config.from_dict(config_dict, **remaining)
        config._name_or_path = str(model)
        return config


def _lfm3_overrides(server_args, hf_config):
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.platforms import current_platform

    cfg = resolving_view(server_args)
    if not isinstance(hf_config, Lfm3Config):
        raise ValueError("Native TPU LFM3 requires --model-config-parser lfm3.")
    if not current_platform.is_out_of_tree() or not current_platform.is_tpu():
        raise ValueError("Native TPU LFM3 requires SGLANG_PLATFORM=tpu_oot.")
    if any(
        getattr(cfg, name) != 1
        for name in (
            "tp_size", "pp_size", "dp_size", "attn_dp_size", "attn_cp_size", "dcp_size"
        )
    ):
        raise ValueError("Native TPU LFM3 currently requires a single TPU worker.")
    if cfg.speculative_algorithm is not None:
        raise ValueError("Native TPU LFM3 does not implement speculative decoding.")
    if cfg.quantization is not None or cfg.dtype not in ("auto", "bfloat16"):
        raise ValueError("Native TPU LFM3 currently requires unquantized BF16 weights.")
    if cfg.kv_cache_dtype not in ("auto", "bf16", "bfloat16"):
        raise ValueError("Native TPU LFM3 currently requires a BF16 KV cache.")
    if cfg.model_impl not in ("auto", "sglang"):
        raise ValueError("Native TPU LFM3 requires --model-impl auto or sglang.")
    if cfg.disaggregation_mode != "null":
        raise ValueError("Native TPU LFM3 does not implement disaggregated serving.")
    backends = (
        cfg.attention_backend,
        cfg.prefill_attention_backend,
        cfg.decode_attention_backend,
    )
    if any(
        b not in (None, "pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched")
        for b in backends
    ):
        raise ValueError(
            "Native TPU LFM3 requires a Pallas RPA attention backend "
            "for sliding attention."
        )
    # This version retains every layer's KV and owns one conv-history row per
    # request. Prefix snapshots and their restoration are intentionally absent.
    return {"disable_radix_cache": True, "dtype": "bfloat16"}


@cache
def register_lfm3() -> None:
    # Called by the general plugin in launchers and spawned workers, before
    # ServerArgs/ModelConfig. Hardware initialization is not needed here.
    from sglang.srt.arg_groups.model_override_base import register_model_override
    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        register_linear_attn_model,
    )
    from sglang.srt.configs.model_config_parser_registry import (
        register_model_config_parser,
    )

    register_model_config_parser("lfm3")(Lfm3ConfigParser)
    register_linear_attn_model(
        LinearAttnModelSpec(
            config_class=Lfm3Config,
            backend_class_name=(
                "sglang_tpu.attention_short_conv.TpuShortConvAttnBackend"
            ),
            hybrid_backend_class_name=(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "ShortConvHybridAttnBackend"
            ),
            arch_names=["Lfm3ForCausalLM"],
            uses_mamba_radix_cache=False,
            support_mamba_cache=False,
        )
    )
    register_model_override("Lfm3ForCausalLM")(_lfm3_overrides)
