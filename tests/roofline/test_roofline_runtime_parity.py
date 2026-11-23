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
"""Runtime parity tests against live vllm_torchtpu validators and backends."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.roofline.conftest import load_qwen3_8_bringup_config
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    SolveOptions,
    TrafficScenario,
    default_rpa_page_size,
    dtypes_from_hf_config,
    from_hf_config,
    solve_serving_plan,
    tpu_min_page_size,
)
from vllm_torchtpu.roofline.geometry import UnsupportedArchitectureError
from vllm_torchtpu.roofline.solver import (
    _candidate_decode_parallels,
    _candidate_prefill_parallels,
)

try:
    from vllm.model_executor.models.config import (
        Qwen4ExpForCausalLMConfig as _Qwen4ExpRuntimeProbe,
    )
except ImportError as _exc:
    _Qwen4ExpRuntimeProbe = None
    PallasAttentionBackend = None
    PallasMLAttentionBackend = None
    get_tpu_min_page_size = None
    TPUQwen4ExpConfig = None
    PcpStaticSupportValidator = None
    validate_pipeline_parallel_config = None
    _RUNTIME_IMPORT_ERROR: Exception | None = _exc
else:
    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackend,
        PallasMLAttentionBackend,
        get_tpu_min_page_size,
    )
    from vllm_torchtpu.models.qwen4_exp.config import TPUQwen4ExpConfig
    from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
    from vllm_torchtpu.platforms.pp_validation import (
        validate_pipeline_parallel_config,
    )

    _RUNTIME_IMPORT_ERROR = None

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _require_runtime() -> None:
    """Skip only when the local Python env lacks container vllm runtime modules."""
    if _RUNTIME_IMPORT_ERROR is not None or _Qwen4ExpRuntimeProbe is None:
        pytest.skip(f"Runtime vllm environment unavailable: {_RUNTIME_IMPORT_ERROR}")


def _make_vllm_config(**overrides: object) -> SimpleNamespace:
    """Build a minimal VllmConfig-shaped namespace for live validator parity."""
    parallel_config = SimpleNamespace(
        tensor_parallel_size=overrides.get("tp", 1),
        pipeline_parallel_size=overrides.get("pp", 1),
        data_parallel_size=overrides.get("dp", 1),
        prefill_context_parallel_size=overrides.get("pcp", 1),
        decode_context_parallel_size=overrides.get("dcp", 1),
        enable_expert_parallel=overrides.get("ep", False),
        is_moe_model=overrides.get("is_moe", True),
        cp_kv_cache_interleave_size=overrides.get("interleave", 128),
        enable_dbo=False,
        ubatch_size=1,
        use_sequence_parallel_moe=overrides.get("use_sequence_parallel_moe", False),
    )
    raw_text = overrides.get("text_cfg")
    text_fields: dict[str, object] = (
        dict(raw_text) if isinstance(raw_text, dict) else {}
    )
    text_fields.setdefault("hc_count", overrides.get("hc_count", 4))
    text_fields["layer_types"] = overrides.get(
        "layer_types", text_fields.get("layer_types")
    )
    text_fields["ple_layer_ids"] = overrides.get(
        "ple_layer_ids", text_fields.get("ple_layer_ids", [])
    )
    text_fields["indexer_budget"] = overrides.get(
        "indexer_budget", text_fields.get("indexer_budget", 2048)
    )
    text_fields.setdefault("mamba_ssm_dtype", overrides.get("mamba_ssm_dtype"))
    rope_raw = overrides.get("rope_parameters", text_fields.get("rope_parameters"))
    text_fields["rope_parameters"] = (
        dict(rope_raw) if isinstance(rope_raw, dict) else {}
    )
    hf_text_config = SimpleNamespace(**text_fields)
    model_config = SimpleNamespace(
        max_model_len=overrides.get("max_model_len", 2048),
        architecture=overrides.get(
            "architecture", "Qwen3_5MoeForConditionalGeneration"
        ),
        hf_text_config=hf_text_config,
        hf_config=SimpleNamespace(
            architectures=[
                overrides.get("architecture", "Qwen3_5MoeForConditionalGeneration")
            ],
            text_config=hf_text_config,
        ),
        multimodal_config=None,
        is_moe=overrides.get("is_moe", True),
    )
    scheduler_config = SimpleNamespace(
        async_scheduling=overrides.get("async_scheduling", False),
        max_num_seqs=overrides.get("max_num_seqs", 64),
    )
    cache_config = SimpleNamespace(mamba_ssm_cache_dtype="auto")
    return SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        speculative_config=overrides.get("speculative_config"),
        kv_transfer_config=overrides.get("kv_transfer_config"),
        additional_config=overrides.get("additional_config", {}),
        compilation_config=SimpleNamespace(compile_sizes=[1, 8, 16, 64]),
    )


@pytest.mark.parametrize("ctx_len", [512, 2048, 4096, 8192, 8193, 32768, 131072])
def test_default_rpa_and_mla_page_size_parity_with_live_backends(
    ctx_len: int,
) -> None:
    """Assert default_rpa_page_size matches PallasAttentionBackend & MLA backend."""
    _require_runtime()
    assert PallasAttentionBackend is not None
    assert PallasMLAttentionBackend is not None
    vcfg = _make_vllm_config(max_model_len=ctx_len)
    live_rpa = PallasAttentionBackend.get_page_size(vcfg)
    live_mla = PallasMLAttentionBackend.get_page_size(vcfg)
    assert default_rpa_page_size(ctx_len, is_mla=False) == live_rpa
    assert default_rpa_page_size(ctx_len, is_mla=True) == live_mla


@pytest.mark.parametrize(
    ("ctx_len", "dcp", "max_num_seqs", "pcp"),
    [
        (8192, 1, 64, 1),
        (65536, 1, 256, 1),
        (131072, 1, 512, 1),
        (131072, 4, 256, 1),
        (65536, 1, 64, 2),
    ],
)
def test_tpu_min_page_size_parity_with_live_attention_backend(
    ctx_len: int,
    dcp: int,
    max_num_seqs: int,
    pcp: int,
) -> None:
    """Assert tpu_min_page_size matches live get_tpu_min_page_size exactly."""
    _require_runtime()
    assert get_tpu_min_page_size is not None
    vcfg = _make_vllm_config(
        max_model_len=ctx_len,
        dcp=dcp,
        max_num_seqs=max_num_seqs,
        pcp=pcp,
    )
    live_min = get_tpu_min_page_size(vcfg)
    roofline_min = tpu_min_page_size(
        ctx_len,
        dcp,
        max_num_seqs,
        pcp=pcp,
        page_table_bytes=CHIPS["tpu7x"].smem_page_table_bytes,
    )
    assert roofline_min == live_min


def test_pp_validation_parity_with_solver_prefill_legality() -> None:
    """Assert solver PP legality matches live validate_pipeline_parallel_config."""
    _require_runtime()
    assert validate_pipeline_parallel_config is not None
    q35_cfg = json.loads((_FIXTURES_DIR / "qwen3_5_397b_fp8.json").read_text())
    q35_geom = from_hf_config(q35_cfg, model_ref="Qwen/Qwen3.5-397B-A17B-FP8")
    topo = TOPOLOGIES["2x2x1"]

    # Live validator rejects PP>1 + TP>1, PP>1 + async_scheduling, and PP>1 + spec k=2.
    with pytest.raises(NotImplementedError):
        validate_pipeline_parallel_config(_make_vllm_config(pp=2, tp=4))
    with pytest.raises(NotImplementedError):
        validate_pipeline_parallel_config(
            _make_vllm_config(pp=2, tp=1, async_scheduling=True)
        )
    with pytest.raises(NotImplementedError):
        validate_pipeline_parallel_config(
            _make_vllm_config(
                pp=2,
                tp=1,
                speculative_config=SimpleNamespace(
                    method="mtp", num_speculative_tokens=2
                ),
                kv_transfer_config=SimpleNamespace(
                    is_kv_producer=True, kv_connector="TPURaidenConnector"
                ),
            )
        )

    # Solver matches: no PP>1 with TP>1, PCP>1, or spec_k=2; allows PP>1 at k=1 disagg.
    pars_k2 = _candidate_prefill_parallels(
        q35_geom, topo, for_disagg=True, spec_k=2, chip=CHIPS["tpu7x"]
    )
    assert all(p.pp == 1 for p in pars_k2)
    pars_k1 = _candidate_prefill_parallels(
        q35_geom, topo, for_disagg=True, spec_k=1, chip=CHIPS["tpu7x"]
    )
    assert all(not (p.pp > 1 and (p.tp > 1 or p.pcp > 1)) for p in pars_k1)
    assert any(p.pp > 1 and p.tp == 1 and p.pcp == 1 for p in pars_k1)


def test_pcp_interleave_128_satisfies_live_pcp_validator() -> None:
    """Assert interleave=128 satisfies PcpStaticSupportValidator for HND and GDN."""
    _require_runtime()
    assert PcpStaticSupportValidator is not None
    vcfg_ok = _make_vllm_config(pcp=2, ep=True, interleave=128)
    PcpStaticSupportValidator.validate_platform_config(vcfg_ok, kv_cache_layout="HND")
    vcfg_bad = _make_vllm_config(pcp=2, ep=True, interleave=64)
    with pytest.raises(ValueError, match="128"):
        PcpStaticSupportValidator.validate_platform_config(
            vcfg_bad, kv_cache_layout="HND"
        )

    # PCP MTP requires num_speculative_tokens=1 (pcp_validation.py:151-154).
    vcfg_k1 = _make_vllm_config(
        pcp=2,
        ep=True,
        interleave=128,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
    )
    PcpStaticSupportValidator.validate_platform_config(vcfg_k1, kv_cache_layout="HND")
    vcfg_k3 = _make_vllm_config(
        pcp=2,
        ep=True,
        interleave=128,
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3),
    )
    with pytest.raises(NotImplementedError, match="num_speculative_tokens=1"):
        PcpStaticSupportValidator.validate_platform_config(
            vcfg_k3, kv_cache_layout="HND"
        )
    q35_cfg = json.loads((_FIXTURES_DIR / "qwen3_5_397b_fp8.json").read_text())
    q35_geom = from_hf_config(q35_cfg, model_ref="Qwen/Qwen3.5-397B-A17B-FP8")
    pars_k3 = _candidate_prefill_parallels(
        q35_geom, TOPOLOGIES["2x2x1"], for_disagg=True, spec_k=3, chip=CHIPS["tpu7x"]
    )
    assert all(p.pcp == 1 for p in pars_k3)


def test_qwen4_exp_config_parity_with_live_runtime_guards() -> None:
    """Assert roofline qwen4_exp guards match live TPUQwen4ExpConfig checks."""
    _require_runtime()
    assert TPUQwen4ExpConfig is not None
    raw_cfg = json.loads((_FIXTURES_DIR / "qwen3_8_flash_next_fp8.json").read_text())
    bringup_cfg = load_qwen3_8_bringup_config()

    # transformers relabels full_attention -> indexed_attention on Qwen4Exp load.
    relabelled_layers = [
        "indexed_attention" if lt == "full_attention" else lt
        for lt in bringup_cfg["text_config"]["layer_types"]
    ]

    # Bring-up config within indexer_budget passes live TPUQwen4ExpConfig.
    vcfg_ok = _make_vllm_config(
        architecture="Qwen4ExpForConditionalGeneration",
        text_cfg=bringup_cfg["text_config"],
        layer_types=relabelled_layers,
        ple_layer_ids=[],
        indexer_budget=2048,
        max_model_len=2048,
    )
    TPUQwen4ExpConfig.verify_and_update_config(vcfg_ok)

    # Unmodified ple_layer_ids=[2] rejected by both live runtime and roofline.
    vcfg_ple = _make_vllm_config(
        architecture="Qwen4ExpForConditionalGeneration",
        text_cfg=bringup_cfg["text_config"],
        layer_types=relabelled_layers,
        ple_layer_ids=raw_cfg["text_config"]["ple_layer_ids"],
        indexer_budget=raw_cfg["text_config"]["indexer_budget"],
        max_model_len=2048,
    )
    with pytest.raises(NotImplementedError, match="ple_layer_ids"):
        TPUQwen4ExpConfig.verify_and_update_config(vcfg_ple)
    with pytest.raises(UnsupportedArchitectureError, match="ple_layer_ids"):
        from_hf_config(raw_cfg, model_ref="Qwen/Qwen3.8-Flash-Next-FP8")

    # Bring-up ple_layer_ids=[] rejected when max_model_len > indexer_budget (2048).
    vcfg_over = _make_vllm_config(
        architecture="Qwen4ExpForConditionalGeneration",
        text_cfg=bringup_cfg["text_config"],
        layer_types=relabelled_layers,
        ple_layer_ids=[],
        indexer_budget=2048,
        max_model_len=4096,
    )
    with pytest.raises(NotImplementedError, match="indexer_budget"):
        TPUQwen4ExpConfig.verify_and_update_config(vcfg_over)

    geom = from_hf_config(bringup_cfg, model_ref="Qwen/Qwen3.8-Flash-Next-FP8")
    dt = dtypes_from_hf_config(bringup_cfg)
    scen_over = TrafficScenario(
        name="over-budget",
        isl=2048,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    plan_over = solve_serving_plan(
        geom, dt, CHIPS["tpu7x"], scen_over, SolveOptions(max_chips=4)
    )
    assert plan_over.recommended is None
    assert any("indexer_budget" in w for w in plan_over.warnings)

    # Bring-up rejects pp>1, pcp>1, dcp>1, and spec_k>0 on both runtime and roofline.
    for kw, match_msg in (
        ({"pp": 2}, "pipeline_parallel_size"),
        ({"pcp": 2}, "context parallelism"),
        ({"dcp": 2}, "context parallelism"),
        (
            {
                "speculative_config": SimpleNamespace(
                    method="mtp", num_speculative_tokens=1
                )
            },
            "speculative",
        ),
    ):
        vcfg_bad = _make_vllm_config(
            architecture="Qwen4ExpForConditionalGeneration",
            text_cfg=bringup_cfg["text_config"],
            layer_types=relabelled_layers,
            ple_layer_ids=[],
            indexer_budget=2048,
            max_model_len=1024,
            **kw,
        )
        with pytest.raises(NotImplementedError, match=match_msg):
            TPUQwen4ExpConfig.verify_and_update_config(vcfg_bad)

    dec_pars = _candidate_decode_parallels(
        geom, TOPOLOGIES["2x2x1"], chip=CHIPS["tpu7x"]
    )
    pre_pars = _candidate_prefill_parallels(
        geom, TOPOLOGIES["2x2x1"], chip=CHIPS["tpu7x"]
    )
    assert all(p.kv_split == 1 for p in dec_pars)
    assert all(p.pp == 1 and p.pcp == 1 for p in pre_pars)
    assert (
        _candidate_prefill_parallels(
            geom, TOPOLOGIES["2x2x1"], spec_k=1, chip=CHIPS["tpu7x"]
        )
        == []
    )
