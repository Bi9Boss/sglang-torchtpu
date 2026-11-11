"""LFM3 config validation must preserve the checkpoint's ShortConv layer names.

Transformers' legacy-name migration changed ``conv`` to ``linear_attention``
after construction, dropping the recurrent-state layer IDs and causing model
initialization to reject a previously loaded config.
"""

import json
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.configs.lfm3 import Lfm3Config

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLfm3Config(CustomTestCase):
    def test_shortconv_layers_survive_validation_and_serialization(self):
        cases = (
            (Lfm3Config(), [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
            (
                Lfm3Config.from_dict(
                    {
                        "model_type": "lfm3",
                        "num_hidden_layers": 5,
                        "layer_types": [
                            "conv",
                            "conv",
                            "full_attention",
                            "conv",
                            "sliding_attention",
                        ],
                    }
                ),
                [0, 1, 3],
            ),
        )
        for config, conv_layer_ids in cases:
            with self.subTest(num_hidden_layers=config.num_hidden_layers):
                # Exercise both explicit revalidation and from_pretrained's
                # logging/serialization path, which constructs default configs.
                config.validate_layer_type()
                config.validate_architecture()
                serialized = json.loads(config.to_json_string())
                restored = Lfm3Config.from_dict(serialized)
                for candidate in (config, restored):
                    self.assertEqual(candidate.linear_layer_ids, conv_layer_ids)
                    self.assertNotIn("linear_attention", candidate.layer_types)
                    self.assertEqual(
                        len(candidate.full_attention_layer_ids) + len(conv_layer_ids),
                        candidate.num_hidden_layers,
                    )
                    self.assertIsNone(candidate.rope_parameters["conv"])

    def test_invalid_layer_types_and_counts_are_still_rejected(self):
        for invalid in ("linear_attention", "mamba", "unknown"):
            with self.subTest(layer_type=invalid):
                with self.assertRaisesRegex(
                    StrictDataclassClassValidationError, "Unsupported LFM3 layer_types"
                ):
                    Lfm3Config(
                        num_hidden_layers=3,
                        layer_types=["conv", "full_attention", invalid],
                    )
        with self.assertRaisesRegex(
            StrictDataclassClassValidationError, "num_hidden_layers entries"
        ):
            Lfm3Config(num_hidden_layers=3, layer_types=["conv", "full_attention"])


if __name__ == "__main__":
    unittest.main()
