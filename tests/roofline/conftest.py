# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared model fixtures for roofline unit tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

QWEN35_CFG: dict[str, object] = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    },
}

SMALL_HYBRID_MOE_CFG: dict[str, object] = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 4096,
        "num_hidden_layers": 8,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "attn_output_gate": True,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "dtype": "bfloat16",
    },
}


def load_qwen3_8_bringup_config() -> dict[str, object]:
    """Load authentic Qwen3.8 fixture with TPU bring-up ple_layer_ids=[] override."""
    fixture_path = (
        Path(__file__).resolve().parent / "fixtures" / "qwen3_8_flash_next_fp8.json"
    )
    cfg = copy.deepcopy(json.loads(fixture_path.read_text()))
    cfg["text_config"]["ple_layer_ids"] = []
    return cfg
