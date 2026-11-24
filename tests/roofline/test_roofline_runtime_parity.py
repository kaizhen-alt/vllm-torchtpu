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

import importlib
import json
from dataclasses import dataclass
from operator import methodcaller
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.roofline.conftest import load_qwen3_8_bringup_config
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    Parallel,
    ParetoPoint,
    PoolMeshPlan,
    PpStageOptions,
    SeqKvLayout,
    ServingPlan,
    SolveOptions,
    StepShape,
    TrafficScenario,
    count_layer_types,
    default_rpa_page_size,
    dtypes_from_hf_config,
    emit_launch_args,
    from_hf_config,
    hbm_breakdown_per_core_bytes,
    partition_pp_layers,
    pass_costs,
    point_launch_args,
    pool_cli_args,
    resolve_model_profile,
    solve_serving_plan,
    tpu_min_page_size,
)
from vllm_torchtpu.roofline.geometry import UnsupportedArchitectureError
from vllm_torchtpu.roofline.solver import (
    _candidate_decode_parallels,
    _candidate_prefill_parallels,
    _valid_raiden_kv_pair,
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
    _torch_tpu_builds_stage_meshes = None
    validate_pipeline_parallel_config = None
    TpuPlatform = None
    tpu_connector_mod = None
    TPURaidenConnectorWorker = None
    derive_gdn_head_geometry = None
    VllmConfigCls = None
    ModelConfigCls = None
    CacheConfigCls = None
    KVTransferConfigCls = None
    torch_mod = None
    _RUNTIME_IMPORT_ERROR: Exception | None = _exc
else:
    import torch as torch_mod
    from vllm.config import (
        CacheConfig as CacheConfigCls,
    )
    from vllm.config import (
        KVTransferConfig as KVTransferConfigCls,
    )
    from vllm.config import (
        ModelConfig as ModelConfigCls,
    )
    from vllm.config import (
        VllmConfig as VllmConfigCls,
    )

    from vllm_torchtpu.distributed.kv_transfer import (
        tpu_connector as tpu_connector_mod,
    )
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
        TPURaidenConnectorWorker,
    )
    from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackend,
        PallasMLAttentionBackend,
        get_tpu_min_page_size,
    )
    from vllm_torchtpu.models.qwen4_exp.config import TPUQwen4ExpConfig
    from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
    from vllm_torchtpu.platforms.pp_validation import (
        _torch_tpu_builds_stage_meshes,
        validate_pipeline_parallel_config,
    )
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

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

    # Solver matches: no PP>1 with TP>1 or spec_k=2; allows PP>1 and PP x PCP at k=1.
    pars_k2 = _candidate_prefill_parallels(
        q35_geom, topo, for_disagg=True, spec_k=2, chip=CHIPS["tpu7x"]
    )
    assert all(p.pp == 1 for p in pars_k2)
    pars_k1 = _candidate_prefill_parallels(
        q35_geom, topo, for_disagg=True, spec_k=1, chip=CHIPS["tpu7x"]
    )
    assert all(not (p.pp > 1 and p.tp > 1) for p in pars_k1)
    assert any(p.pp > 1 and p.pcp > 1 for p in pars_k1)
    assert any(p.pp > 1 and p.tp == 1 and p.pcp == 1 for p in pars_k1)
    assert _torch_tpu_builds_stage_meshes is not None
    pp_pcp_vcfg = _make_vllm_config(
        pp=2,
        tp=1,
        pcp=4,
        ep=True,
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=True, kv_connector="TPURaidenConnector"
        ),
    )
    if _torch_tpu_builds_stage_meshes():
        validate_pipeline_parallel_config(pp_pcp_vcfg)
    else:
        with pytest.raises(NotImplementedError):
            validate_pipeline_parallel_config(pp_pcp_vcfg)


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


def test_emitted_pcp_pp_and_raiden_flags_match_live_runtime_validators() -> None:
    """Assert emitted PCP, PP, MTP, and Raiden connector flags pass live validators."""
    _require_runtime()
    assert PcpStaticSupportValidator is not None
    assert validate_pipeline_parallel_config is not None

    pcp_pool = PoolMeshPlan(
        role="prefill",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=4,
        dp=1,
        ep=8,
        pp=1,
        pcp=2,
        kv_split=1,
        mesh_axes=(2, 2, 1),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="ag_ep",
    )
    pcp_flags = pool_cli_args(pcp_pool, kv_dtype="fp8", spec_k=1)
    interleave = int(pcp_flags[pcp_flags.index("--cp-kv-cache-interleave-size") + 1])
    spec_cfg = json.loads(pcp_flags[pcp_flags.index("--speculative-config") + 1])
    PcpStaticSupportValidator.validate_platform_config(
        _make_vllm_config(
            pcp=2,
            ep=True,
            interleave=interleave,
            speculative_config=SimpleNamespace(**spec_cfg),
        ),
        kv_cache_layout="HND",
    )

    pp_pool = PoolMeshPlan(
        role="prefill",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=1,
        dp=1,
        ep=1,
        pp=8,
        pcp=1,
        kv_split=1,
        mesh_axes=(1,),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="allreduce",
    )
    pp_flags = pool_cli_args(pp_pool, kv_dtype="fp8")
    assert "--no-async-scheduling" in pp_flags
    kv_cfg = json.loads(pp_flags[pp_flags.index("--kv-transfer-config") + 1])
    conn_mod = importlib.import_module(kv_cfg["kv_connector_module_path"])
    assert kv_cfg["kv_connector"] in vars(conn_mod)
    validate_pipeline_parallel_config(
        _make_vllm_config(
            pp=8,
            tp=1,
            async_scheduling=False,
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=True, kv_connector=kv_cfg["kv_connector"]
            ),
        )
    )
    with pytest.raises(NotImplementedError, match="TPURaidenConnector"):
        validate_pipeline_parallel_config(
            _make_vllm_config(
                pp=8,
                tp=1,
                async_scheduling=False,
                kv_transfer_config=SimpleNamespace(
                    is_kv_producer=True, kv_connector="TpuRaidenConnector"
                ),
            )
        )


def test_qwen4_exp_moe_sp_rejected_by_live_config_and_emit() -> None:
    """Verify MoE sequence parallel is rejected on qwen4_exp by runtime and emit."""
    _require_runtime()
    assert TPUQwen4ExpConfig is not None
    bringup_cfg = load_qwen3_8_bringup_config()
    relabelled = [
        "indexed_attention" if lt == "full_attention" else lt
        for lt in bringup_cfg["text_config"]["layer_types"]
    ]
    vcfg_sp = _make_vllm_config(
        architecture="Qwen4ExpForConditionalGeneration",
        text_cfg=bringup_cfg["text_config"],
        layer_types=relabelled,
        ple_layer_ids=[],
        indexer_budget=2048,
        max_model_len=1024,
        use_sequence_parallel_moe=True,
    )
    with pytest.raises(NotImplementedError, match="TPU_MOE_SEQUENCE_PARALLEL"):
        TPUQwen4ExpConfig.verify_and_update_config(vcfg_sp)
    fused_pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=8,
        dp=1,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 1),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="fused_rs",
    )
    sp_pt = ParetoPoint(
        mode="mixed",
        concurrency=4,
        total_chips=4,
        prefill_pool=None,
        decode_pool=fused_pool,
        est_ttft_ms=50.0,
        est_tpot_ms=5.0,
        est_interactivity_tok_s=200.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=500.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
    )
    sp_scen = TrafficScenario(
        name="sp-parity",
        isl=512,
        osl=64,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
    )
    with pytest.raises(ValueError, match="TPU_MOE_SEQUENCE_PARALLEL"):
        emit_launch_args(
            ServingPlan(
                model_layers=48,
                chip="tpu7x",
                scenario=sp_scen,
                mode="mixed",
                recommended=sp_pt,
                frontier=(sp_pt,),
                warnings=(),
                model_id="qwen4_exp",
            ),
            model="Qwen/Qwen3.8-Flash-Next-FP8",
        )


def _build_platform_vcfg(
    *,
    is_hybrid: bool,
    kv_transfer_cfg: dict[str, object],
) -> object:
    """Build a mocked VllmConfig for live TpuPlatform validation checks."""
    assert VllmConfigCls is not None
    assert ModelConfigCls is not None
    assert CacheConfigCls is not None
    assert KVTransferConfigCls is not None
    assert torch_mod is not None
    vcfg = MagicMock(spec=VllmConfigCls)
    vcfg.model_config = MagicMock(spec=ModelConfigCls)
    vcfg.model_config.dtype = torch_mod.bfloat16
    vcfg.model_config.is_hybrid = is_hybrid
    vcfg.model_config.is_attention_free = False
    vcfg.model_config.architecture = (
        "Qwen3_5MoeForConditionalGeneration" if is_hybrid else "GlmMoeDsaForCausalLM"
    )
    vcfg.model_config.architectures = [vcfg.model_config.architecture]
    vcfg.model_config.hf_config = None
    vcfg.model_config.runner_type = "generate"
    vcfg.model_config.get_head_size = lambda: 128
    vcfg.cache_config = CacheConfigCls()
    vcfg.cache_config.block_size = 128
    vcfg.cache_config.enable_prefix_caching = False
    vcfg.cache_config.mamba_cache_mode = None
    vcfg.attention_config = MagicMock()
    vcfg.attention_config.backend = None
    vcfg.compilation_config = MagicMock()
    vcfg.compilation_config.mode = MagicMock()
    vcfg.compilation_config.compile_sizes = [16, 32]
    vcfg.compilation_config.splitting_ops = []
    vcfg.scheduler_config = MagicMock()
    vcfg.scheduler_config.max_num_batched_tokens = 32
    vcfg.scheduler_config.max_num_seqs = 16
    vcfg.scheduler_config.max_num_scheduled_tokens = None
    vcfg.scheduler_config.enable_chunked_prefill = True
    vcfg.scheduler_config.is_multimodal_model = False
    vcfg.scheduler_config.async_scheduling = False
    vcfg.speculative_config = None
    vcfg.lora_config = None
    vcfg.parallel_config = MagicMock(
        prefill_context_parallel_size=1,
        nnodes=1,
        node_rank=0,
        world_size=1,
        world_size_across_dp=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        cp_kv_cache_interleave_size=1,
        decode_context_parallel_size=1,
    )
    raw_extra = kv_transfer_cfg.get("kv_connector_extra_config")
    raw_mod = kv_transfer_cfg.get("kv_connector_module_path")
    vcfg.kv_transfer_config = KVTransferConfigCls(
        kv_connector=str(kv_transfer_cfg["kv_connector"]),
        kv_connector_module_path=str(raw_mod) if raw_mod is not None else None,
        kv_role=str(kv_transfer_cfg.get("kv_role", "kv_both")),
        kv_connector_extra_config=(
            dict(raw_extra) if isinstance(raw_extra, dict) else {}
        ),
    )
    vcfg.additional_config = {}
    return vcfg


def test_kv_offload_and_hybrid_pp_stage_parity_with_live_tpu_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert offload/multi connectors and hybrid PP partitions pass live runtime."""
    _require_runtime()
    assert TpuPlatform is not None
    assert validate_pipeline_parallel_config is not None

    offload_pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=8,
        dp=1,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 1),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="ag_ep",
        hbm_state_gb=2.0,
        kv_offload_gb=32.0,
    )
    offload_flags = pool_cli_args(offload_pool, kv_dtype="fp8", model_id="qwen3_5_moe")
    assert "--kv-offloading-size" not in offload_flags
    offload_cfg = json.loads(
        offload_flags[offload_flags.index("--kv-transfer-config") + 1]
    )
    disagg_pool = PoolMeshPlan(
        role="prefill",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=1,
        dp=1,
        ep=1,
        pp=8,
        pcp=1,
        kv_split=1,
        mesh_axes=(1,),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="allreduce",
        kv_offload_gb=32.0,
    )
    disagg_flags = pool_cli_args(disagg_pool, kv_dtype="fp8", model_id="qwen3_5_moe")
    multi_cfg = json.loads(disagg_flags[disagg_flags.index("--kv-transfer-config") + 1])

    with (
        patch("vllm_torchtpu.patch_registry.apply"),
        patch(
            "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env"
        ),
    ):
        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "1")
        TpuPlatform.check_and_update_config(
            _build_platform_vcfg(is_hybrid=True, kv_transfer_cfg=offload_cfg)
        )
        TpuPlatform.check_and_update_config(
            _build_platform_vcfg(is_hybrid=True, kv_transfer_cfg=multi_cfg)
        )
        with pytest.raises(AssertionError, match="OffloadingConnector"):
            TpuPlatform.check_and_update_config(
                _build_platform_vcfg(
                    is_hybrid=False,
                    kv_transfer_cfg={
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                    },
                )
            )
        monkeypatch.setenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", "0")
        with pytest.raises(ValueError, match="TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1"):
            TpuPlatform.check_and_update_config(
                _build_platform_vcfg(is_hybrid=True, kv_transfer_cfg=offload_cfg)
            )

    # Live validate_pipeline_parallel_config rejects a hybrid stage with 0 full layers,
    # while roofline partition_pp_layers(..., require_full_attn_per_stage=True) yields
    # a VLLM_PP_LAYER_PARTITION that passes.
    layers = ["linear_attention"] * 6 + ["full_attention"] * 2
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with pytest.raises(NotImplementedError, match="no full-attention layer"):
        validate_pipeline_parallel_config(
            _make_vllm_config(
                pp=2,
                tp=1,
                async_scheduling=False,
                layer_types=layers,
                kv_transfer_config=SimpleNamespace(
                    is_kv_producer=True,
                    kv_connector=offload_cfg["kv_connector"],
                ),
            )
        )
    asym_cfg = {
        **json.loads(
            (_FIXTURES_DIR / "qwen3_5_397b_fp8.json").read_text(encoding="utf-8")
        ),
    }
    asym_text = dict(asym_cfg["text_config"])
    asym_text.pop("full_attention_interval", None)
    asym_text["num_hidden_layers"] = 8
    asym_text["layer_types"] = layers
    asym_cfg["text_config"] = asym_text
    geom_asym = from_hf_config(asym_cfg)
    dt_asym = dtypes_from_hf_config(asym_cfg)
    part = partition_pp_layers(
        geom_asym,
        dt_asym,
        Parallel(tp=1, pp=2, moe="tp_slice", ranks_per_chip=2),
        (4096, 1),
        options=PpStageOptions(require_full_attn_per_stage=True),
    )
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(str(x) for x in part))
    validate_pipeline_parallel_config(
        _make_vllm_config(
            pp=2,
            tp=1,
            async_scheduling=False,
            layer_types=layers,
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=True,
                kv_connector=offload_cfg["kv_connector"],
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class _RaidenTopologySpec:
    """Worker and mesh parameters for Raiden admission parity tests."""

    architecture: str = "Qwen3_5MoeForConditionalGeneration"
    is_producer: bool = False
    tp: int = 1
    dp: int = 1
    pp: int = 1
    pcp: int = 1
    dcp: int = 1
    tp_rank: int = 0
    block_size: int = 128
    interleave: int = 128
    extra_config: dict[str, object] | None = None


def _glm_admission(worker: object) -> str:
    """Return runtime GLM Raiden admission topology for a parity worker."""
    return str(methodcaller("_raiden_glm_admission_topology")(worker))


def _qwen35_admission(worker: object) -> str:
    """Return runtime Qwen3.5 Raiden admission topology for a parity worker."""
    return str(methodcaller("_raiden_qwen35_admission_topology")(worker))


def _make_parity_raiden_worker(spec: _RaidenTopologySpec) -> object:
    """Construct a TPURaidenConnectorWorker with mocked device/network handles."""
    assert TPURaidenConnectorWorker is not None
    cfg = MagicMock()
    cfg.speculative_config = None
    cfg.additional_config = {}
    cfg.kv_transfer_config.kv_connector = "TPURaidenConnector"
    cfg.kv_transfer_config.kv_role = (
        "kv_producer" if spec.is_producer else "kv_consumer"
    )
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    cfg.kv_transfer_config.is_kv_producer = spec.is_producer
    cfg.kv_transfer_config.kv_connector_extra_config = dict(spec.extra_config or {})
    cfg.cache_config.block_size = spec.block_size
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.architecture = spec.architecture
    cfg.parallel_config.data_parallel_rank = 0
    cfg.parallel_config.data_parallel_size = spec.dp
    cfg.parallel_config.tensor_parallel_size = spec.tp
    cfg.parallel_config.pipeline_parallel_size = spec.pp
    cfg.parallel_config.prefill_context_parallel_size = spec.pcp
    cfg.parallel_config.decode_context_parallel_size = spec.dcp
    cfg.parallel_config.cp_kv_cache_interleave_size = spec.interleave
    conn_mod = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    with (
        patch(f"{conn_mod}.get_tensor_model_parallel_rank", return_value=spec.tp_rank),
        patch(f"{conn_mod}.get_tensor_model_parallel_world_size", return_value=spec.tp),
        patch(f"{conn_mod}.dist_utils.get_node_id", return_value=0),
        patch(f"{conn_mod}.dist_utils.get_host_ip", return_value="127.0.0.1"),
        patch(f"{conn_mod}.dist_utils.get_kv_transfer_port", return_value=9100),
    ):
        return TPURaidenConnectorWorker(cfg)


def test_raiden_glm_admission_topology_parity_with_solver() -> None:
    """Assert _raiden_glm_admission_topology matches _valid_raiden_kv_pair."""
    _require_runtime()
    glm_cfg = json.loads((_FIXTURES_DIR / "glm_5_2_fp8.json").read_text("utf-8"))
    glm_geom = from_hf_config(glm_cfg)
    conn_mod = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"

    # TP16/DCP16 -> TP16/DCP16 admitted on both runtime and roofline.
    p_dcp16 = _make_parity_raiden_worker(
        _RaidenTopologySpec(
            is_producer=True,
            architecture="GlmMoeDsaForCausalLM",
            tp=16,
            dcp=16,
            tp_rank=3,
            block_size=256,
            interleave=4,
            extra_config={
                "destination_decode_context_parallel_size": 16,
                "destination_cp_kv_cache_interleave_size": 4,
            },
        )
    )
    d_dcp16 = _make_parity_raiden_worker(
        _RaidenTopologySpec(
            is_producer=False,
            architecture="GlmMoeDsaForCausalLM",
            tp=16,
            dcp=16,
            tp_rank=3,
            block_size=256,
            interleave=4,
        )
    )
    with (
        patch(f"{conn_mod}._raiden_dst_shards", return_value=16),
        patch("vllm_torchtpu.distributed.dcp.get_dcp_rank", return_value=3),
    ):
        assert _glm_admission(p_dcp16) == "tp16dcp16_prefill"
        assert _glm_admission(d_dcp16) == "tp16dcp16_decode"
    assert _valid_raiden_kv_pair(
        glm_geom, Parallel(tp=16, kv_split=16), Parallel(tp=16, kv_split=16)
    )

    # TP16/DCP1 -> TP16/DCP16 rejected on both.
    p_dcp1 = _make_parity_raiden_worker(
        _RaidenTopologySpec(
            is_producer=True,
            architecture="GlmMoeDsaForCausalLM",
            tp=16,
            dcp=1,
            block_size=256,
            interleave=4,
            extra_config={
                "destination_decode_context_parallel_size": 16,
                "destination_cp_kv_cache_interleave_size": 4,
            },
        )
    )
    with (
        patch(f"{conn_mod}._raiden_dst_shards", return_value=16),
        pytest.raises(ValueError, match="pipeline-parallel TP1 producer"),
    ):
        _glm_admission(p_dcp1)
    assert not _valid_raiden_kv_pair(
        glm_geom, Parallel(tp=16, kv_split=1), Parallel(tp=16, kv_split=16)
    )

    # TP16/DCP8 -> TP16/DCP8 rejected on both.
    p_dcp8 = _make_parity_raiden_worker(
        _RaidenTopologySpec(
            is_producer=True,
            architecture="GlmMoeDsaForCausalLM",
            tp=16,
            dcp=8,
            block_size=256,
            interleave=4,
            extra_config={
                "destination_decode_context_parallel_size": 8,
                "destination_cp_kv_cache_interleave_size": 4,
            },
        )
    )
    with pytest.raises(ValueError, match="tensor_parallel_size =="):
        _glm_admission(p_dcp8)
    assert not _valid_raiden_kv_pair(
        glm_geom, Parallel(tp=16, kv_split=8), Parallel(tp=16, kv_split=8)
    )

    # PP8 -> TP16/PP2 rejected on both.
    d_tp16_pp2 = _make_parity_raiden_worker(
        _RaidenTopologySpec(
            is_producer=False,
            architecture="GlmMoeDsaForCausalLM",
            tp=16,
            pp=2,
            dcp=1,
            block_size=256,
        )
    )
    with pytest.raises(ValueError, match="consumer tensor_parallel_size=16"):
        _glm_admission(d_tp16_pp2)
    assert not _valid_raiden_kv_pair(
        glm_geom, Parallel(pp=8, tp=1), Parallel(tp=16, pp=2)
    )


def test_raiden_qwen35_admission_and_hnd_emit_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert Qwen3.5 Raiden admission aligns with solver pairing and HND emission."""
    _require_runtime()
    assert tpu_connector_mod is not None
    qwen_cfg = json.loads((_FIXTURES_DIR / "qwen3_5_397b_fp8.json").read_text("utf-8"))
    qwen_geom = from_hf_config(qwen_cfg)

    def _pool(role: str, spec: _RaidenTopologySpec) -> PoolMeshPlan:
        """Build a synthetic Qwen3.5 Raiden pool mesh plan for launch emission."""
        return PoolMeshPlan(
            role=role,
            instances=1,
            topology="2x2x1",
            chips_per_instance=4,
            tp=spec.tp,
            dp=spec.dp,
            ep=max(1, spec.tp * spec.dp),
            pp=spec.pp,
            pcp=spec.pcp,
            kv_split=spec.dcp,
            mesh_axes=(2, 2, 1) if spec.tp > 1 else (1,),
            hbm_used_gb=20.0,
            hbm_limit_gb=80.0,
            moe_dispatch="ag_ep" if spec.tp > 1 else "allreduce",
        )

    monkeypatch.setattr(tpu_connector_mod, "_raiden_seq_on_lane_layout", lambda: True)
    expected_pair_commands = 2

    # PP8 -> TP8 admits runtime seq-on-lane decode and emits HND on BOTH pools.
    assert _valid_raiden_kv_pair(qwen_geom, Parallel(pp=8, tp=1), Parallel(tp=8, dp=1))
    pp8_spec = _RaidenTopologySpec(is_producer=True, pp=8)
    tp8_spec = _RaidenTopologySpec(is_producer=False, tp=8)
    pt_pp8_tp8 = ParetoPoint(
        concurrency=16,
        mode="disagg",
        total_chips=8,
        prefill_pool=_pool("prefill", pp8_spec),
        decode_pool=_pool("decode", tp8_spec),
        est_ttft_ms=100.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=200.0,
        token_throughput_per_chip=600.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
    )
    launch_pp8_tp8 = point_launch_args(
        pt_pp8_tp8,
        kv_dtype="fp8",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_id="qwen3_5_moe",
        is_hybrid=True,
    )
    assert len(launch_pp8_tp8["commands"]) == expected_pair_commands
    assert all("VLLM_KV_CACHE_LAYOUT=HND" in c for c in launch_pp8_tp8["commands"])
    w_pp8 = _make_parity_raiden_worker(pp8_spec)
    w_tp8 = _make_parity_raiden_worker(tp8_spec)
    assert _qwen35_admission(w_pp8) == "pp8_prefill"
    assert _qwen35_admission(w_tp8) == "tp8dcp1_decode"

    # PP4 x PCP2 -> TP8 x DCP4 admits runtime seq-on-lane and emits HND on BOTH pools.
    assert _valid_raiden_kv_pair(
        qwen_geom, Parallel(pp=4, pcp=2, tp=1), Parallel(tp=8, kv_split=4)
    )
    pp4_pcp2_spec = _RaidenTopologySpec(is_producer=True, pp=4, pcp=2)
    tp8_dcp4_spec = _RaidenTopologySpec(is_producer=False, tp=8, dcp=4)
    pt_pcp_dcp = ParetoPoint(
        concurrency=16,
        mode="disagg",
        total_chips=8,
        prefill_pool=_pool("prefill", pp4_pcp2_spec),
        decode_pool=_pool("decode", tp8_dcp4_spec),
        est_ttft_ms=100.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=200.0,
        token_throughput_per_chip=600.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
    )
    launch_pcp_dcp = point_launch_args(
        pt_pcp_dcp,
        kv_dtype="fp8",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_id="qwen3_5_moe",
        is_hybrid=True,
    )
    assert len(launch_pcp_dcp["commands"]) == expected_pair_commands
    assert all("VLLM_KV_CACHE_LAYOUT=HND" in c for c in launch_pcp_dcp["commands"])
    w_pp4_pcp2 = _make_parity_raiden_worker(pp4_pcp2_spec)
    w_tp8_dcp4 = _make_parity_raiden_worker(tp8_dcp4_spec)
    assert _qwen35_admission(w_pp4_pcp2) == "pp4pcp2_prefill"
    assert _qwen35_admission(w_tp8_dcp4) == "tp8dcp4_decode"

    # TP4 x DP2 x DCP2 fails solver pairing, omits HND, and admits as dp2tp4_decode.
    assert not _valid_raiden_kv_pair(
        qwen_geom, Parallel(pp=8, tp=1), Parallel(tp=4, dp=2, kv_split=2)
    )
    dp_dcp_spec = _RaidenTopologySpec(is_producer=False, tp=4, dp=2, dcp=2)
    pt_dp_dcp = ParetoPoint(
        concurrency=16,
        mode="disagg",
        total_chips=8,
        prefill_pool=_pool("prefill", pp8_spec),
        decode_pool=_pool("decode", dp_dcp_spec),
        est_ttft_ms=100.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=200.0,
        token_throughput_per_chip=600.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
    )
    launch_dp_dcp = point_launch_args(
        pt_dp_dcp,
        kv_dtype="fp8",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_id="qwen3_5_moe",
        is_hybrid=True,
    )
    assert not any("VLLM_KV_CACHE_LAYOUT=HND" in c for c in launch_dp_dcp["commands"])
    w_dp_dcp = _make_parity_raiden_worker(dp_dcp_spec)
    assert _qwen35_admission(w_dp_dcp) == "dp2tp4_decode"


def test_kv_cache_dtype_admission_parity_with_tpu_platform() -> None:
    """Assert TpuPlatform.is_kv_cache_dtype_supported matches main profile admission."""
    _require_runtime()
    assert TpuPlatform is not None
    qwen_cfg = json.loads((_FIXTURES_DIR / "qwen3_5_397b_fp8.json").read_text("utf-8"))
    main_prof = resolve_model_profile("qwen3_5_moe", "main")
    assert set(main_prof.kv_dtypes) == {"bf16", "fp8"}
    mock_mcfg = MagicMock()

    for supported_tok in ("bfloat16", "fp8"):
        assert TpuPlatform.is_kv_cache_dtype_supported(supported_tok, mock_mcfg)
        dt = dtypes_from_hf_config(
            {
                **qwen_cfg,
                "quantization_config": {
                    **dict(qwen_cfg["quantization_config"]),
                    "kv_cache_scheme": supported_tok,
                },
            },
            runtime_profile="main",
        )
        assert dt.kv_dtype in ("bfloat16", "bf16", "fp8")

    for unsupported_tok in ("fp4", "turboquant_4bit_nc", "int4_per_token_head", "int8"):
        assert not TpuPlatform.is_kv_cache_dtype_supported(unsupported_tok, mock_mcfg)
        with pytest.raises(ValueError, match="Unsupported"):
            dtypes_from_hf_config(
                {
                    **qwen_cfg,
                    "quantization_config": {
                        **dict(qwen_cfg["quantization_config"]),
                        "kv_cache_scheme": unsupported_tok,
                    },
                },
                runtime_profile="main",
            )


def test_gdn_state_bytes_parity_with_derive_gdn_head_geometry() -> None:
    """Assert GDN HBM and I/O bytes match derive_gdn_head_geometry across tp*pcp."""
    _require_runtime()
    assert derive_gdn_head_geometry is not None
    qwen_cfg = json.loads((_FIXTURES_DIR / "qwen3_5_397b_fp8.json").read_text("utf-8"))
    geom = from_hf_config(qwen_cfg)
    dt = dtypes_from_hf_config(qwen_cfg)
    counts = count_layer_types(geom)

    for tp, pcp in ((1, 1), (2, 2), (8, 2), (16, 2), (4, 8), (32, 1)):
        head_geo = derive_gdn_head_geometry(
            geom.linear_k_heads, geom.linear_v_heads, tp * pcp
        )
        rec_bytes = (
            float(head_geo.local_num_v_heads * geom.linear_k_dim * geom.linear_v_dim)
            * dt.state
        )
        conv_bytes = (
            float(
                head_geo.local_conv_dim(geom.linear_k_dim, geom.linear_v_dim)
                * max(0, geom.conv_kernel - 1)
            )
            * dt.conv
        )
        expected_state_per_seq = float(counts.gdn) * (rec_bytes + conv_bytes)

        par = Parallel(tp=tp, pcp=pcp)
        bd = hbm_breakdown_per_core_bytes(geom, dt, par, SeqKvLayout(1, 8192))
        assert bd.state == pytest.approx(expected_state_per_seq, rel=1e-12)

        hit_mods = {
            m.name: m
            for m in pass_costs(
                geom,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=8192,
                    prefill_tokens=8192,
                    prefix_hit_rate=0.5,
                ),
                dt,
            )
        }
        miss_mods = {
            m.name: m
            for m in pass_costs(
                geom,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=8192,
                    prefill_tokens=8192,
                    prefix_hit_rate=0.0,
                ),
                dt,
            )
        }
        assert hit_mods["gdn"].hbm_bytes - miss_mods["gdn"].hbm_bytes == pytest.approx(
            expected_state_per_seq, rel=1e-12
        )
