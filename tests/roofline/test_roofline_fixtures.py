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
"""Authentic HuggingFace fixture tests for canonical model identity and geometry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.roofline.conftest import load_qwen3_8_bringup_config
from vllm_torchtpu.roofline import (
    HF_REPO_TO_CANONICAL_ID,
    SUPPORTED_MODELS,
    count_layer_types,
    from_hf_config,
    resolve_model_identity,
)
from vllm_torchtpu.roofline.geometry import UnsupportedArchitectureError, supports

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

SUPPORTED_FIXTURES: tuple[tuple[str, str, str], ...] = (
    ("qwen3_5_397b_fp8.json", "qwen3_5_moe", "Qwen/Qwen3.5-397B-A17B-FP8"),
    ("qwen3_5_397b_nvfp4.json", "qwen3_5_moe", "nvidia/Qwen3.5-397B-A17B-NVFP4"),
    ("glm_5_2_fp8.json", "glm_moe_dsa", "zai-org/GLM-5.2-FP8"),
    ("glm_5_2_nvfp4.json", "glm_moe_dsa", "nvidia/GLM-5.2-NVFP4"),
    (
        "qwen3_8_flash_next_fp8_bringup",
        "qwen4_exp",
        "Qwen/Qwen3.8-Flash-Next-FP8",
    ),
)

NEGATIVE_FIXTURE_FILES: tuple[tuple[str, str], ...] = (
    ("qwen3_8_flash_next_fp8.json", "ple_layer_ids"),
    ("deepseek_v4_pro.json", "deepseek_v4"),
    ("deepseek_v4_1_flash.json", "deepseek_v41"),
    ("minimax_m3_1_flash.json", "minimax_m3_vl"),
    ("muse_spark_open_dflash2.json", "Non-causal"),
)


def _load_fixture(name: str) -> dict[str, object]:
    """Load one checked-in authentic HuggingFace config JSON fixture."""
    if name == "qwen3_8_flash_next_fp8_bringup":
        return load_qwen3_8_bringup_config()
    return json.loads((FIXTURES_DIR / name).read_text())


def test_every_supported_model_has_authentic_fixture_and_repo_mapping() -> None:
    """Verify every SUPPORTED_MODELS entry has checked-in authentic fixtures."""
    covered_ids = {canonical_id for _, canonical_id, _ in SUPPORTED_FIXTURES}
    assert covered_ids == set(SUPPORTED_MODELS)
    for spec in SUPPORTED_MODELS.values():
        assert spec.hf_repos
        for repo in spec.hf_repos:
            assert HF_REPO_TO_CANONICAL_ID[repo.lower()] == spec.canonical_id


@pytest.mark.parametrize(
    ("fixture_name", "expected_id", "hf_repo"),
    SUPPORTED_FIXTURES,
)
def test_supported_authentic_hf_fixtures_resolve_and_extract_geometry(
    fixture_name: str,
    expected_id: str,
    hf_repo: str,
) -> None:
    """Verify canonical identity, repo matching, and geometry extraction."""
    cfg = _load_fixture(fixture_name)
    identity = resolve_model_identity(cfg, model_ref=hf_repo)
    assert identity.canonical_id == expected_id
    assert identity.hf_repo == hf_repo

    geom = from_hf_config(cfg, model_ref=hf_repo)
    assert geom.model_id == expected_id
    assert geom.num_layers > 0
    assert geom.hidden > 0

    wrong_repo = (
        "zai-org/GLM-5.2-FP8"
        if expected_id != "glm_moe_dsa"
        else "Qwen/Qwen3.5-397B-A17B-FP8"
    )
    with pytest.raises(UnsupportedArchitectureError, match="does not match"):
        resolve_model_identity(cfg, model_ref=wrong_repo)


def test_glm_5_2_authentic_fixture_parses_index_share_and_mlp_layer_types() -> None:
    """Verify GLM-5.2 parses IndexShare indexer_types and mlp_layer_types."""
    expected_layers = 78
    expected_full_idx = 21
    expected_shared_idx = 57
    expected_dense_ffn = 3
    expected_moe_ffn = 75
    geom = from_hf_config(
        _load_fixture("glm_5_2_fp8.json"),
        model_ref="zai-org/GLM-5.2-FP8",
    )
    assert geom.num_layers == expected_layers
    assert len(geom.indexer_types) == expected_layers
    assert geom.indexer_types.count("full") == expected_full_idx
    assert geom.indexer_types.count("shared") == expected_shared_idx
    assert geom.index_share_for_mtp_iteration is True
    assert geom.ffn_layer_counts() == (expected_moe_ffn, expected_dense_ffn)


def test_qwen3_8_bringup_fixture_parses_hc_lowrank_and_zeroes_stripped_mtp() -> None:
    """Verify Qwen3.8 bring-up matches TPU origin/main dense GQA + stripped MTP."""
    expected_layers = 48
    expected_gdn = 36
    expected_gqa = 12
    expected_hc_mult = 4
    expected_hc_lowrank = 320
    expected_qsa_budget = 2048
    geom = from_hf_config(
        _load_fixture("qwen3_8_flash_next_fp8_bringup"),
        model_ref="Qwen/Qwen3.8-Flash-Next-FP8",
    )
    counts = count_layer_types(geom)
    assert geom.num_layers == expected_layers
    assert (counts.gdn, counts.gqa) == (expected_gdn, expected_gqa)
    assert geom.hc_mult == expected_hc_mult
    assert geom.hc_lowrank == expected_hc_lowrank
    assert geom.qsa_budget == expected_qsa_budget
    assert geom.ple_layer_ids == ()
    assert geom.mtp_layers == 0


@pytest.mark.parametrize(("fixture_name", "match_text"), NEGATIVE_FIXTURE_FILES)
def test_negative_control_authentic_fixtures_fail_closed(
    fixture_name: str,
    match_text: str,
) -> None:
    """Verify unmodeled models and unsupported features fail closed."""
    cfg = _load_fixture(fixture_name)
    with pytest.raises(UnsupportedArchitectureError, match=match_text):
        from_hf_config(cfg)


@pytest.mark.parametrize(
    ("cfg", "match_text"),
    [
        (
            {
                "model_type": "muse_spark",
                "architectures": ["MuseSparkForCausalLM"],
                "hidden_size": 8192,
                "num_hidden_layers": 62,
                "num_attention_heads": 128,
            },
            "muse_spark",
        ),
        (
            {
                "model_type": "llama4_text",
                "hidden_size": 5120,
                "num_hidden_layers": 48,
                "num_attention_heads": 40,
            },
            "llama4_text",
        ),
        (
            {
                "model_type": "gemma2",
                "hidden_size": 3584,
                "num_hidden_layers": 42,
                "num_attention_heads": 16,
            },
            "gemma2",
        ),
        (
            {
                "model_type": "gemma4_text",
                "hidden_size": 4096,
                "num_hidden_layers": 48,
                "num_attention_heads": 32,
            },
            "gemma4_text",
        ),
        (
            {
                "model_type": "qwen2",
                "hidden_size": 3584,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
            },
            "qwen2",
        ),
        (
            {
                "hidden_size": 4096,
                "num_hidden_layers": 8,
                "num_attention_heads": 32,
            },
            "missing required 'model_type'",
        ),
        (
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
            },
            "architectures",
        ),
    ],
)
def test_non_cdk_omitted_model_type_and_bad_architecture_fail_closed(
    cfg: dict[str, object],
    match_text: str,
) -> None:
    """Verify non-CDK model_types, omitted model_type, and mismatched arch fail."""
    with pytest.raises(UnsupportedArchitectureError, match=match_text):
        from_hf_config(cfg)


def test_unlisted_repo_conflicting_aliases_interval_and_dedicated_mtp_fail_closed() -> (
    None
):
    """Verify unlisted repos, alias conflicts, bad intervals, and ded MTP fail."""
    q35 = _load_fixture("qwen3_5_397b_fp8.json")
    with pytest.raises(UnsupportedArchitectureError, match="Unlisted HuggingFace repo"):
        from_hf_config(q35, model_ref="someorg/NotARealRepo")

    glm = _load_fixture("glm_5_2_nvfp4.json")
    bad_alias = dict(glm, num_experts=128)
    with pytest.raises(ValueError, match="Conflicting HF config alias keys"):
        from_hf_config(bad_alias)

    q38 = load_qwen3_8_bringup_config()
    bad_interval = {
        **q38,
        "text_config": {**dict(q38["text_config"]), "full_attention_interval": 2},
    }
    with pytest.raises(ValueError, match="full_attention_interval"):
        from_hf_config(bad_interval)

    bad_mtp_emb = {
        **q35,
        "text_config": {
            **dict(q35["text_config"]),
            "mtp_use_dedicated_embeddings": True,
        },
    }
    with pytest.raises(UnsupportedArchitectureError, match="Dedicated MTP embeddings"):
        from_hf_config(bad_mtp_emb)


def test_spoofed_sliding_window_and_kda_under_listed_model_type_fail_closed() -> None:
    """Verify per-family structural allowlist blocks spoofed SWA or KDA configs."""
    spoofed_swa = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "layer_types": ["sliding_attention", "full_attention"] * 2,
        "sliding_window": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        supports(spoofed_swa)
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        from_hf_config(spoofed_swa)

    spoofed_kda = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "full_attention_interval": 4,
        "linear_attn_config": {
            "num_heads": 16,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
        "intermediate_size": 4096,
        "vocab_size": 32000,
    }
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        supports(spoofed_kda)
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        from_hf_config(spoofed_kda)

    q35_raw = _load_fixture("qwen3_5_397b_fp8.json")
    q35_text = q35_raw["text_config"]
    assert isinstance(q35_text, dict)
    spoofed_mla = {**q35_raw, "text_config": {**q35_text, "kv_lora_rank": 512}}
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        supports(spoofed_mla)
    with pytest.raises(
        UnsupportedArchitectureError, match="does not allow layer kinds"
    ):
        from_hf_config(spoofed_mla)
