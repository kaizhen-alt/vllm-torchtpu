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
"""Unit tests for mesh/prefill-decode solver and CLI solve subcommand."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from dataclasses import replace as dc_replace
from operator import methodcaller
from pathlib import Path
from typing import Literal

import pytest

import vllm_torchtpu.roofline as rf
from tests.roofline.conftest import (
    SMALL_HYBRID_MOE_CFG,
    cold_crossover_sweep,
    load_qwen3_8_bringup_config,
)
from vllm_torchtpu.roofline import (
    CHIPS,
    TOPOLOGIES,
    Dtypes,
    FloorOptions,
    KvFitInput,
    LaunchOptions,
    Parallel,
    ParetoPoint,
    PoolMeshPlan,
    ScenarioOverrides,
    SeqKvLayout,
    ServingPlan,
    SolveOptions,
    SpecShape,
    StepShape,
    TrafficScenario,
    attach_reference_report,
    build_example_scenario,
    build_solve_payload,
    cli,
    dsa_indexer_layer_weight_elems,
    dtypes_from_hf_config,
    emit_launch_args,
    explain_layout_rejection,
    floor_throughput_per_chip,
    format_parallelism,
    from_hf_config,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    pareto_skyline_by_concurrency,
    parse_layout,
    point_launch_args,
    reference_layout_report,
    render_html,
    scale_pp_stage_floor,
    solve_serving_plan,
    top_layouts,
    x_bound,
)
from vllm_torchtpu.roofline import constants as rf_constants
from vllm_torchtpu.roofline import crossover as rf_crossover
from vllm_torchtpu.roofline import solver as rf_solver
from vllm_torchtpu.roofline.constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    SLO_VERDICT_RULED_OUT,
    SLO_VERDICT_UNDECIDED,
    UNIT_GIGA,
    ZERO_THINK_TIME_WARNING,
)
from vllm_torchtpu.roofline.crossover import (
    CandidateState,
    EpLoadSpec,
    _CrossoverPlanner,
    _derive_winner_range_and_crossover,
    _eval_grid_disagg_ttft,
    _kv_parts_for_pool,
    _pool_sig,
    ep_z_unit,
    eval_grid_mix_candidates,
    max_ep_load_ratio,
    scale_ep_moe_floor,
    scaled_ep_serial_s,
    setup_crossover_sweep,
    sweep_prefix_hit_crossovers,
)
from vllm_torchtpu.roofline.geometry import (
    UnsupportedArchitectureError,
    _from_hf_config_dry_run,
)
from vllm_torchtpu.roofline.memory import (
    compute_w_attn,
    compute_w_ffn,
    default_rpa_page_size,
    kv_capacity_and_hit_rate,
    smem_max_seqs,
)
from vllm_torchtpu.roofline.partition import (
    count_full_attn_layers,
)
from vllm_torchtpu.roofline.schema import _pool_key
from vllm_torchtpu.roofline.solver import (
    EXAMPLE_SCENARIOS,
    _apply_capacity_ceiling_verdict,
    _build_l1_decode_input,
    _candidate_decode_parallels,
    _candidate_moe_dispatches,
    _candidate_prefill_parallels,
    _decode_capacity_profile,
    _find_prefill_options,
    _pareto_filter,
    _prefill_stage_step_floor,
    _run_isolated_tarski_loop,
    _SolveContext,
    _valid_gdn_degree,
    _valid_head_tp,
    _valid_raiden_gdn_pair,
    _valid_raiden_kv_pair,
)
from vllm_torchtpu.roofline.step import (
    combine_modules,
    spec_cycle_floor,
    step_floor,
)

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


def _check_dp_moe_decode(chip: rf.ChipSpec, q35_geom: rf.ModelGeometry) -> None:
    """Verify that DP > 1 MoE decode candidates price cross-DP collectives."""
    dp_cands = [
        p
        for p in _candidate_decode_parallels(q35_geom, rf.TOPOLOGIES["2x2x1"])
        if p.dp > 1
    ]
    assert dp_cands
    ep_degree = 8
    for par_dp in dp_cands:
        assert par_dp.moe == "ep"
        assert par_dp.ep == ep_degree
        assert par_dp.moe_dispatch in ("ag_ep", "a2a_push")
        opts = FloorOptions(dt=rf.Dtypes(kv=1.0), topology=rf.TOPOLOGIES["2x2x1"])
        sf_dp = rf.step_floor(
            q35_geom,
            chip,
            par_dp,
            rf.StepShape(batch=4, context=1024),
            opts,
        )
        sf_dp_ar = rf.step_floor(
            q35_geom,
            chip,
            dc_replace(par_dp, moe_dispatch="allreduce"),
            rf.StepShape(batch=4, context=1024),
            opts,
        )
        assert sf_dp.chain_s > 0.0
        if par_dp.moe_dispatch == "ag_ep":
            assert sf_dp_ar.chain_s == pytest.approx(sf_dp.chain_s)


def _check_mixed_prefill_dp2(chip: rf.ChipSpec) -> None:
    """Verify that mixed prefill work scales as 1/dp per replica."""
    dp2_geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32768,
        }
    )
    dp2_scen = rf.TrafficScenario(
        name="custom",
        isl=4096,
        osl=512,
        concurrencies=(8,),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    expected_dp_2x2x1 = 8
    dp2_plan = rf.solve_serving_plan(
        dp2_geom,
        rf.Dtypes(kv=1.0),
        chip,
        dp2_scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert dp2_plan.recommended is not None
    assert dp2_plan.recommended.decode_pool.dp == expected_dp_2x2x1
    par_dp2 = rf.Parallel(tp=1, dp=8, mesh_axes=(2, 2, 2, 1))
    opts_dp2 = FloorOptions(dt=rf.Dtypes(kv=1.0), topology=rf.TOPOLOGIES["2x2x1"])
    sf_d_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=1, context=4352),
        opts_dp2,
    )
    sf_m_dp2 = rf.step_floor(
        dp2_geom,
        chip,
        par_dp2,
        rf.StepShape(batch=1, context=4352, prefill_tokens=2048),
        opts_dp2,
    )
    assert sf_d_dp2.serial_estimate_s > 0.0
    assert sf_m_dp2.serial_estimate_s > 0.0
    # 1 seq/replica * 2 prefill chunks amortized over osl=512 decode steps.
    exp_p_mix = min(1.0, float(1 * 2) / float(dp2_scen.osl))
    assert dp2_plan.recommended.p_mixed == pytest.approx(exp_p_mix, rel=1e-5)


def _check_cli_solve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that CLI solve subcommand produces structured JSON payload."""
    cfg_path = tmp_path / "q35_config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "solve",
            "--model",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "auto",
            "--max-chips",
            "8",
            "--example-slo-tier",
            "interactive",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended"] is not None


def _check_pp_prefill_ttft(chip: rf.ChipSpec) -> None:
    """Verify that PP > 1 prefill TTFT includes pipeline bubble stage fills."""
    pp_geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32768,
        }
    )
    for p_chunk, n_chk in ((2048, 1), (256, 4)):
        chk_scen = rf.TrafficScenario(
            name="custom",
            isl=1024,
            osl=256,
            concurrencies=(1,),
            prefill_chunk=p_chunk,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
        )
        chk_disagg = rf.solve_serving_plan(
            pp_geom,
            rf.Dtypes(kv=1.0),
            chip,
            chk_scen,
            SolveOptions(mode="disagg", max_chips=8),
        )
        pp_pts = [
            p
            for p in chk_disagg.candidates
            if p.prefill_pool is not None and p.prefill_pool.pp > 1
        ]
        assert pp_pts
        p0 = pp_pts[0]
        assert p0.prefill_pool is not None
        sf_st = rf.step_floor(
            pp_geom,
            chip,
            rf.Parallel(
                tp=p0.prefill_pool.tp,
                pp=p0.prefill_pool.pp,
                pcp=p0.prefill_pool.pcp,
                moe="ep" if p0.prefill_pool.ep > 1 else "tp_slice",
                ep=p0.prefill_pool.ep,
                mesh_axes=p0.prefill_pool.mesh_axes,
            ),
            rf.StepShape(
                batch=0, rows_per_seq=0, context=0, state_writes=0, prefill_tokens=1024
            ),
            FloorOptions(
                dt=rf.Dtypes(kv=1.0),
                topology=rf.TOPOLOGIES[p0.prefill_pool.topology],
            ),
        )
        bubble_mult = 1.0 + float(p0.prefill_pool.pp - 1) / float(n_chk)
        chk_disagg_inf_dcn = rf.solve_serving_plan(
            pp_geom,
            rf.Dtypes(kv=1.0),
            dc_replace(chip, dcn_bw=math.inf),
            chk_scen,
            SolveOptions(mode="disagg", max_chips=8),
        )
        p0_inf = next(
            p
            for p in chk_disagg_inf_dcn.candidates
            if p.prefill_pool is not None
            and p.prefill_pool.pp == p0.prefill_pool.pp
            and p.prefill_pool.pcp == p0.prefill_pool.pcp
            and p.prefill_pool.topology == p0.prefill_pool.topology
            and p.decode_pool.topology == p0.decode_pool.topology
        )
        if p0.prefill_pool.ep <= 1:
            assert p0_inf.est_ttft_ms == pytest.approx(
                sf_st.serial_estimate_s * bubble_mult * 1e3, rel=1e-4
            )
        assert p0.est_ttft_ms > p0_inf.est_ttft_ms


def test_solver_hybrid_moe_mixed_vs_disagg_and_crossover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test mixed vs disaggregated solver behavior and MTP cycle pricing."""
    q35_geom = rf.from_hf_config(QWEN35_CFG)
    chip = rf.CHIPS["tpu7x"]
    scen = EXAMPLE_SCENARIOS["chat-8k1k"]
    mixed_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=8),
    )
    disagg_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=8),
    )
    min_conc = 16
    min_mixed_fraction = 0.10
    mixed_c16 = [p for p in mixed_plan.frontier if p.concurrency >= min_conc]
    assert mixed_c16
    assert any(p.p_mixed >= min_mixed_fraction for p in mixed_c16)
    assert disagg_plan.recommended is not None
    assert disagg_plan.recommended.p_mixed == 0.0

    # coding-agent (accept_len=3.4) with 1 MTP layer prices k=3 window.
    ca_scen = rf.build_example_scenario(
        "coding-agent", overrides=ScenarioOverrides(concurrencies=(1,))
    )
    ca_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        ca_scen,
        SolveOptions(mode="disagg", max_chips=8),
    )
    assert ca_plan.recommended is not None
    rec_d = ca_plan.recommended.decode_pool
    sf_k3 = rf.spec_cycle_floor(
        q35_geom,
        chip,
        rf.Parallel(
            tp=rec_d.tp,
            dp=rec_d.dp,
            moe="ep" if rec_d.ep > 1 else "tp_slice",
            ep=rec_d.ep,
            moe_dispatch=rec_d.moe_dispatch,
            kv_split=rec_d.kv_split,
            mesh_axes=rec_d.mesh_axes,
        ),
        SpecShape(
            batch=max(1, (1 + rec_d.dp - 1) // rec_d.dp),
            context=ca_scen.isl + ca_scen.osl // 2,
            k=3,
        ),
        FloorOptions(
            dt=rf.Dtypes(kv=1.0),
            topology=rf.TOPOLOGIES[rec_d.topology],
        ),
    )
    assert ca_plan.recommended.est_tpot_ms == pytest.approx(
        sf_k3.serial_estimate_s * 1e3 / ca_scen.accept_len, rel=1e-4
    )

    _check_dp_moe_decode(chip, q35_geom)

    _check_mixed_prefill_dp2(chip)

    _check_pp_prefill_ttft(chip)

    hybrid_2t_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 8192,
            "num_hidden_layers": 92,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 23,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "attn_output_gate": True,
            "linear_num_key_heads": 32,
            "linear_num_value_heads": 128,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 2048,
            "shared_expert_intermediate_size": 2048,
            "vocab_size": 248320,
        },
    }
    hybrid_2t_geom = rf.from_hf_config(hybrid_2t_cfg)
    hybrid_2t_dt = rf.Dtypes(
        w_dense=1.0, w_expert=0.5625, w_aux=1.0, kv=1.0, state=2.0, lm_head=2.0
    )
    topo_32 = rf.TOPOLOGIES["2x2x4"]
    par_tp32 = rf.Parallel(
        tp=32, dp=1, moe="ep", ep=32, moe_dispatch="ag_ep", kv_split=8
    )
    par_tp8_dp4 = rf.Parallel(
        tp=8, dp=4, moe="ep", ep=32, moe_dispatch="ag_ep", kv_split=2
    )
    opts_32 = FloorOptions(dt=hybrid_2t_dt, topology=topo_32)
    sf_tp32_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=1, context=131072),
        opts_32,
    )
    sf_dp4_c4 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=1, context=131072),
        opts_32,
    )
    assert sf_tp32_c4.serial_estimate_s < sf_dp4_c4.serial_estimate_s

    sf_tp32_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp32,
        rf.StepShape(batch=256, context=8192),
        opts_32,
    )
    sf_dp4_c256 = rf.step_floor(
        hybrid_2t_geom,
        chip,
        par_tp8_dp4,
        rf.StepShape(batch=64, context=8192),
        opts_32,
    )
    assert sf_dp4_c256.serial_estimate_s < sf_tp32_c256.serial_estimate_s

    nvfp4_path = (
        Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_nvfp4.json"
    )
    nvfp4_cfg = json.loads(nvfp4_path.read_text())
    nvfp4_geom = rf.from_hf_config(
        nvfp4_cfg, model_ref="nvidia/Qwen3.5-397B-A17B-NVFP4"
    )
    nvfp4_dt = rf.dtypes_from_hf_config(nvfp4_cfg)
    max_nvfp4_chips = 16
    nvfp4_disagg = rf.solve_serving_plan(
        nvfp4_geom,
        nvfp4_dt,
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=max_nvfp4_chips),
    )
    assert nvfp4_disagg.recommended is not None
    assert nvfp4_disagg.recommended.total_chips <= max_nvfp4_chips
    assert any(
        p.prefill_pool is not None
        and (
            p.prefill_pool.instances != p.decode_pool.instances
            or p.prefill_pool.topology != p.decode_pool.topology
        )
        for p in nvfp4_disagg.frontier
    )

    infeasible_scen = rf.TrafficScenario(
        name="custom",
        isl=8192,
        osl=1024,
        concurrencies=(1, 4),
        tpot_slo_ms=0.05,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    inf_plan = rf.solve_serving_plan(
        q35_geom,
        rf.Dtypes(kv=1.0),
        chip,
        infeasible_scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert any("INFEASIBLE_SLO" in w for w in inf_plan.warnings)

    _check_cli_solve(tmp_path, capsys)


def test_physical_kv_page_capacity_and_unpruned_regimes() -> None:
    """Test physical KV page table capacity scaling and unpruned disagg regimes."""
    chip = rf.CHIPS["tpu7x"]
    q35_geom = rf.from_hf_config(QWEN35_CFG)
    q35_dt = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )

    hbm_diff_threshold = 0.80e9
    hbm_mtp1 = rf.hbm_footprint_per_core_bytes(
        q35_geom, q35_dt, rf.Parallel(tp=8, kv_split=1, moe="ep"), SeqKvLayout(0, 0)
    )
    hbm_mtp0 = rf.hbm_footprint_per_core_bytes(
        dc_replace(q35_geom, mtp_layers=0),
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        SeqKvLayout(0, 0),
    )
    assert (hbm_mtp1 - hbm_mtp0) > hbm_diff_threshold
    hbm_phys = rf.hbm_footprint_per_core_bytes(
        q35_geom,
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        SeqKvLayout(1, 8192, physical_kv_layout=True),
    )
    hbm_log = rf.hbm_footprint_per_core_bytes(
        q35_geom,
        q35_dt,
        rf.Parallel(tp=8, kv_split=1, moe="ep"),
        SeqKvLayout(1, 8192, physical_kv_layout=False),
    )
    assert (hbm_phys - hbm_log) == pytest.approx(15.0 * 8192.0 * 512.0, abs=2.0)

    accept_len = 3.0
    target_concurrency = 64
    min_pp_stages = 8
    scen = rf.TrafficScenario(
        name="custom",
        isl=1024,
        osl=1024,
        concurrencies=(target_concurrency,),
        prefix_hit_rate=0.5,
        accept_len=accept_len,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    disagg = rf.solve_serving_plan(
        q35_geom,
        q35_dt,
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=12),
    )
    assert all(p.concurrency == target_concurrency for p in disagg.frontier)
    # Disagg prefill runs spec_k <= 1 so PP>1 is admitted even when decode spec_k=2;
    # direct _candidate_prefill_parallels(..., spec_k=2) still rejects PP>1.
    assert any(
        p.prefill_pool is not None and p.prefill_pool.pp >= min_pp_stages
        for p in disagg.candidates
    )
    assert all(
        p.pp == 1
        for p in _candidate_prefill_parallels(
            q35_geom, rf.TOPOLOGIES["2x2x1"], for_disagg=True, spec_k=2, chip=chip
        )
    )
    disagg_k1 = rf.solve_serving_plan(
        q35_geom,
        q35_dt,
        chip,
        dc_replace(scen, accept_len=1.5, spec_tokens=1),
        SolveOptions(mode="disagg", max_chips=12),
    )
    assert any(
        p.prefill_pool is not None and p.prefill_pool.pp >= min_pp_stages
        for p in disagg_k1.candidates
    )
    assert any(
        p.decode_pool.moe_dispatch == "a2a_push" and p.decode_pool.dp in (4, 8)
        for p in disagg.candidates
    )

    dsa_78l_cfg = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "layer_types": ["dsa"] * 78,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "shared_expert_intermediate_size": 2048,
        "first_k_dense_replace": 3,
        "dense_intermediate_size": 12288,
        "vocab_size": 154880,
        "num_nextn_predict_layers": 1,
    }
    dsa_78l_geom = rf.from_hf_config(dsa_78l_cfg)
    dsa_78l_dt = rf.Dtypes(
        w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    accept_len = 3.0
    tp_16 = 16
    kv_split_16 = 16
    dsa_c48 = rf.solve_serving_plan(
        dsa_78l_geom,
        dsa_78l_dt,
        chip,
        rf.TrafficScenario(
            name="dsa-single-turn",
            isl=137168,
            osl=894,
            concurrencies=(48,),
            prefix_hit_rate=0.95,
            accept_len=accept_len,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
        ),
        SolveOptions(mode="mixed", max_chips=8, gpu_memory_utilization=0.85),
    )
    expected_c48 = 48
    tp16_pts = [
        p
        for p in dsa_c48.candidates
        if p.decode_pool.tp == tp_16
        and p.decode_pool.ep == 1
        and p.decode_pool.max_num_seqs == expected_c48
    ]
    best_tp16 = min(tp16_pts, key=lambda p: p.est_tpot_ms)
    assert best_tp16.decode_pool.kv_split == kv_split_16

    w4_dsa_geom = dc_replace(
        dsa_78l_geom,
        hidden=7168,
        num_layers=61,
        layer_types=("dsa",) * 61,
        num_heads=128,
        num_kv_heads=128,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        v_head_dim=128,
        index_n_heads=64,
        vocab=129280,
    )
    w4_dsa_dt = rf.Dtypes(
        w_dense=1.0, w_expert=0.5625, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0
    )
    w4_dsa_c64 = rf.solve_serving_plan(
        w4_dsa_geom,
        w4_dsa_dt,
        chip,
        rf.TrafficScenario(
            name="w4-dsa-c64",
            isl=131072,
            osl=935,
            concurrencies=(64,),
            prefix_hit_rate=0.95,
            accept_len=accept_len,
            think_time_s=0.0,
            kv_working_set_turns=1.0,
            spec_tokens=3,
            kv_offload_gb=48.0,
        ),
        SolveOptions(mode="mixed", max_chips=8, gpu_memory_utilization=0.85),
    )
    expected_tp_2 = 2
    expected_dp_8 = 8
    expected_dp_16 = 16
    ep_degree_16 = 16
    tp2_dp8 = [
        p
        for p in w4_dsa_c64.candidates
        if p.decode_pool.tp == expected_tp_2
        and p.decode_pool.dp == expected_dp_8
        and p.decode_pool.ep == ep_degree_16
    ]
    tp1_dp16 = [
        p
        for p in w4_dsa_c64.candidates
        if p.decode_pool.tp == 1
        and p.decode_pool.dp == expected_dp_16
        and p.decode_pool.ep == ep_degree_16
    ]
    assert tp2_dp8
    assert tp1_dp16
    assert min(p.est_ttft_ms for p in tp2_dp8) < min(p.est_ttft_ms for p in tp1_dp16)
    assert min(p.est_tpot_ms for p in tp1_dp16) < min(p.est_tpot_ms for p in tp2_dp8)


def test_engine_memory_utilization_and_explicit_scenario_inputs(tmp_path: Path) -> None:
    """Test engine GPU memory utilization clipping and CLI explicit scenario flags."""
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(kv=1.0, w_aux=2.0)

    with pytest.raises(TypeError):
        rf.TrafficScenario(name="x", isl=1024, osl=128)

    util_045 = 0.45
    util_092 = 0.92
    plan_45 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        rf.TrafficScenario(
            name="x",
            isl=1024,
            osl=128,
            prefix_hit_rate=0.5,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1,),
            kv_working_set_turns=1.0,
        ),
        SolveOptions(mode="mixed", max_chips=4, gpu_memory_utilization=util_045),
    )
    assert plan_45.gpu_memory_utilization == util_045
    assert len(plan_45.frontier) == 0

    plan_92 = rf.solve_serving_plan(
        geom,
        dt,
        chip,
        rf.TrafficScenario(
            name="x",
            isl=1024,
            osl=128,
            prefix_hit_rate=0.5,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(1,),
            kv_working_set_turns=1.0,
        ),
        SolveOptions(mode="mixed", max_chips=4, gpu_memory_utilization=util_092),
    )
    assert plan_92.gpu_memory_utilization == util_092
    assert len(plan_92.frontier) > 0

    scen_c0 = rf.TrafficScenario(
        name="closed",
        isl=4096,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(128,),
        kv_working_set_turns=1.0,
    )
    scen_think = rf.TrafficScenario(
        name="open",
        isl=4096,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=15.0,
        concurrencies=(128,),
        kv_working_set_turns=1.0,
    )
    p_c0 = rf.solve_serving_plan(
        geom, dt, chip, scen_c0, SolveOptions(mode="mixed", max_chips=4)
    )
    p_think = rf.solve_serving_plan(
        geom, dt, chip, scen_think, SolveOptions(mode="mixed", max_chips=4)
    )
    assert p_c0.recommended is not None
    assert p_think.recommended is not None
    assert p_think.recommended.est_tpot_ms <= p_c0.recommended.est_tpot_ms

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["solve", "--config", str(cfg_path)])
    assert excinfo.value.code != 0

    rc_ex = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_ex == 0

    rc_exp = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "1024",
            "--osl",
            "128",
            "--prefix-hit-rate",
            "0.5",
            "--accept-len",
            "1.0",
            "--think-time-s",
            "0.0",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_exp == 0


def test_runtime_legality_heads_pp_tp_pcp_ep_gdn_and_smem_cap() -> None:
    """Test parallelism legality checks across attention heads, GDN, and SMEM."""
    assert count_full_attn_layers(rf.from_hf_config(QWEN35_CFG)) > 0
    assert _valid_head_tp(rf.from_hf_config(QWEN35_CFG), 1)
    assert _pareto_filter(()) == ()
    uneven_geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 3584,
            "num_hidden_layers": 8,
            "num_attention_heads": 28,
            "num_key_value_heads": 6,
            "head_dim": 128,
            "intermediate_size": 7168,
            "vocab_size": 32000,
        }
    )
    invalid_tp_4 = 4
    invalid_tp_8 = 8
    dec_pars = _candidate_decode_parallels(uneven_geom, rf.TOPOLOGIES["2x2x1"])
    assert all(28 % p.tp == 0 and (6 % p.tp == 0 or p.tp % 6 == 0) for p in dec_pars)
    assert all(p.tp not in (invalid_tp_4, invalid_tp_8) for p in dec_pars)

    pre_pars = _candidate_prefill_parallels(uneven_geom, rf.TOPOLOGIES["2x2x1"])
    assert all(not (p.pp > 1 and p.tp > 1) for p in pre_pars)
    assert any(p.pp > 1 and p.pcp > 1 for p in pre_pars)
    assert any(p.pp > 1 and p.tp == 1 and p.pcp == 1 for p in pre_pars)

    hybrid_8l = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32000,
        }
    )
    disagg_pre = _candidate_prefill_parallels(
        hybrid_8l, rf.TOPOLOGIES["2x2x1"], for_disagg=True
    )
    hybrid_full_attn = 2
    assert all(
        not (p.pp > 1 and p.tp > 1) and p.pp <= hybrid_full_attn for p in disagg_pre
    )
    assert any(p.pp > 1 and p.pcp > 1 for p in disagg_pre)
    assert not _valid_gdn_degree(dc_replace(hybrid_8l, linear_v_heads=64), 6)
    assert not _valid_raiden_gdn_pair(
        hybrid_8l, rf.Parallel(tp=2, pcp=1), rf.Parallel(tp=8, pcp=1)
    )
    assert _valid_raiden_gdn_pair(
        hybrid_8l, rf.Parallel(tp=1, pcp=2, pp=2), rf.Parallel(tp=8, pcp=1)
    )

    moe_geom = rf.from_hf_config(QWEN35_CFG)
    moe_pre = _candidate_prefill_parallels(moe_geom, rf.TOPOLOGIES["2x2x1"])
    for p in moe_pre:
        if p.pcp > 1:
            assert p.moe == "ep"
            assert p.ep == p.tp * p.pcp

    isl_262144 = 262144
    page_size_128 = 128
    page_size_1024 = 1024
    smem_seqs_64 = 64
    assert default_rpa_page_size(isl_262144) == page_size_128
    assert default_rpa_page_size(isl_262144, is_mla=True) == page_size_1024
    assert smem_max_seqs(isl_262144, 1, page_size_128) == smem_seqs_64


def test_disagg_capacity_clamped_hit_rate_and_prefill_dispatch() -> None:
    """Test disagg capacity clamping, prefix hit degradation, and prefill dispatch."""
    chip = rf.CHIPS["tpu7x"]
    chat_scen = EXAMPLE_SCENARIOS["chat-8k1k"]
    cap_init, h_init = kv_capacity_and_hit_rate(
        rf.from_hf_config(QWEN35_CFG),
        rf.Dtypes(kv=1.0),
        rf.Parallel(tp=1),
        1,
        KvFitInput(
            b_ws=1,
            ctx_full=1024,
            hbm_limit_b=1e12,
            w_base_b=0.0,
            b_kv_phys_core=1.0,
            offload_b_per_core=0.0,
            prefix_hit_rate=chat_scen.prefix_hit_rate,
            kv_working_set_turns=chat_scen.kv_working_set_turns,
        ),
    )
    assert cap_init > 0.0
    assert h_init == chat_scen.prefix_hit_rate
    moe_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 32000,
    }
    geom = rf.from_hf_config(moe_cfg)
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scen_small = rf.TrafficScenario(
        name="small-ws",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=0.01,
    )
    scen_huge = dc_replace(scen_small, name="huge-ws", kv_working_set_turns=5000.0)
    plan_small = rf.solve_serving_plan(
        geom, dt, chip, scen_small, SolveOptions(mode="disagg", max_chips=8)
    )
    plan_huge = rf.solve_serving_plan(
        geom, dt, chip, scen_huge, SolveOptions(mode="disagg", max_chips=8)
    )
    assert plan_small.recommended is not None
    assert plan_huge.recommended is not None
    by_pair_small = {
        (
            p.prefill_pool.topology,
            p.prefill_pool.tp,
            p.prefill_pool.pcp,
            p.prefill_pool.pp,
            p.decode_pool.topology,
            p.decode_pool.tp,
            p.decode_pool.dp,
            p.decode_pool.kv_split,
        ): p
        for p in plan_small.candidates
        if p.prefill_pool is not None
    }
    matched = 0
    for pt_h in plan_huge.candidates:
        if pt_h.prefill_pool is None:
            continue
        k = (
            pt_h.prefill_pool.topology,
            pt_h.prefill_pool.tp,
            pt_h.prefill_pool.pcp,
            pt_h.prefill_pool.pp,
            pt_h.decode_pool.topology,
            pt_h.decode_pool.tp,
            pt_h.decode_pool.dp,
            pt_h.decode_pool.kv_split,
        )
        pt_s = by_pair_small.get(k)
        if pt_s is not None:
            matched += 1
            assert pt_s.kv_cache_capacity_tokens is not None
            assert pt_h.effective_prefix_hit_rate < pt_s.effective_prefix_hit_rate
            assert pt_h.est_ttft_ms > pt_s.est_ttft_ms
    assert matched >= 1
    pcp_ep_pts = [
        p
        for p in plan_small.candidates
        if p.prefill_pool is not None
        and p.prefill_pool.pcp > 1
        and p.prefill_pool.tp == 1
        and p.prefill_pool.ep > 1
    ]
    assert pcp_ep_pts
    assert all(
        p.prefill_pool.moe_dispatch == "ag_ep" for p in pcp_ep_pts if p.prefill_pool
    )

    scen_think = dc_replace(
        scen_small,
        concurrencies=(64,),
        think_time_s=5.0,
        ttft_slo_ms=5000.0,
        tpot_slo_ms=200.0,
    )
    plan_think = rf.solve_serving_plan(
        geom, dt, chip, scen_think, SolveOptions(mode="auto", max_chips=4)
    )
    assert plan_think.recommended is not None
    assert plan_think.recommended.slo_met

    scen_capped_ws = dc_replace(
        scen_small,
        concurrencies=(64,),
        kv_working_set_turns=200.0,
        max_num_seqs=1,
    )
    scen_single_ws = dc_replace(
        scen_small,
        concurrencies=(1,),
        kv_working_set_turns=200.0,
        max_num_seqs=1,
    )
    expected_tp_8 = 8
    pt_c64 = next(
        p
        for p in rf.solve_serving_plan(
            geom,
            dt,
            chip,
            scen_capped_ws,
            SolveOptions(mode="mixed", max_chips=4),
        ).candidates
        if p.decode_pool.tp == expected_tp_8 and p.decode_pool.dp == 1
    )
    pt_c1 = next(
        p
        for p in rf.solve_serving_plan(
            geom,
            dt,
            chip,
            scen_single_ws,
            SolveOptions(mode="mixed", max_chips=4),
        ).candidates
        if p.decode_pool.tp == expected_tp_8 and p.decode_pool.dp == 1
    )
    assert pt_c64.effective_prefix_hit_rate < pt_c1.effective_prefix_hit_rate


def test_solver_excludes_impossible_dp1_ep_and_gdn_dcp() -> None:
    """Reject cross-DP EP dispatches at dp=1 and kv_split>1 on KV-less pure GDN."""
    assert "a2a_push" not in _candidate_moe_dispatches("ep", dp=1)
    assert "ag_ep" not in _candidate_moe_dispatches("ep", dp=1)
    pure_gdn = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 0,
            "head_dim": 128,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    dec = _candidate_decode_parallels(pure_gdn, rf.TOPOLOGIES["2x2x1"])
    assert dec
    assert all(p.kv_split == 1 for p in dec)


def test_solver_rejects_subhost_max_chips() -> None:
    """Reject max_chips below minimum provisionable 2x2x1 slice (4 chips)."""
    geom = rf.from_hf_config(QWEN35_CFG)
    scen = rf.build_example_scenario(overrides=ScenarioOverrides(concurrencies=(1,)))
    for bad_chips in (1, 2):
        with pytest.raises(ValueError, match="2x2x1"):
            rf.solve_serving_plan(
                geom,
                rf.Dtypes(),
                rf.CHIPS["tpu7x"],
                scen,
                SolveOptions(max_chips=bad_chips),
            )


def test_swa_kv_capacity_not_linearized_beyond_window() -> None:
    """Verify SWA KV stays capped at min(ctx, sliding_window) in capacity fit."""
    sw_window = 512
    ctx_long = 131_072
    isl_long = 130_048
    osl_long = 1024
    tp_8 = 8
    conc_one = 1
    hit_rate_target = 0.8
    no_offload = 0.0
    single_turn = 1.0
    hybrid_swa_geom = _from_hf_config_dry_run(
        {
            "model_type": "synthetic_swa",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "layer_types": ["sliding_attention"] * 15 + ["full_attention"],
            "sliding_window": sw_window,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    pure_swa_geom = dc_replace(
        hybrid_swa_geom,
        layer_types=("sliding_attention",) * 16,
    )
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    chip = rf.CHIPS["tpu7x"]
    par = rf.Parallel(tp=tp_8, ranks_per_chip=chip.cores_per_chip)
    w_base_b = hbm_footprint_per_core_bytes(hybrid_swa_geom, dt, par, SeqKvLayout(0, 0))
    kv_at_sw = (
        hbm_footprint_per_core_bytes(
            hybrid_swa_geom, dt, par, SeqKvLayout(conc_one, sw_window)
        )
        - w_base_b
    )
    kv_past_sw = (
        hbm_footprint_per_core_bytes(
            hybrid_swa_geom, dt, par, SeqKvLayout(conc_one, sw_window + conc_one)
        )
        - w_base_b
    )
    full_kv_per_tok = kv_past_sw - kv_at_sw
    swa_fixed_per_seq = kv_at_sw - float(sw_window) * full_kv_per_tok
    hbm_limit_b = DEFAULT_GPU_MEMORY_UTILIZATION * float(chip.hbm_bytes)
    scen = rf.TrafficScenario(
        name="swa-long",
        isl=isl_long,
        osl=osl_long,
        concurrencies=(conc_one,),
        prefix_hit_rate=hit_rate_target,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=single_turn,
    )
    plan = rf.solve_serving_plan(
        hybrid_swa_geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=chip.cores_per_chip * 2),
    )
    pt_tp8 = next(
        p
        for p in plan.candidates
        if p.decode_pool.tp == tp_8
        and p.decode_pool.dp == conc_one
        and p.decode_pool.kv_split == conc_one
    )
    # Full-context capacity subtracts fixed SWA bytes from HBM budget.
    expected_cap_tok = int(
        (hbm_limit_b - w_base_b - swa_fixed_per_seq) / full_kv_per_tok
    )
    assert pt_tp8.kv_cache_capacity_tokens == expected_cap_tok

    # Pure SWA has zero full-context KV per token so prefix capacity stays unbounded.
    pure_plan = rf.solve_serving_plan(
        pure_swa_geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=chip.cores_per_chip * 2),
    )
    pure_pt = next(
        p
        for p in pure_plan.candidates
        if p.decode_pool.tp == tp_8
        and p.decode_pool.dp == conc_one
        and p.decode_pool.kv_split == conc_one
    )
    assert pure_pt.kv_cache_capacity_tokens is None
    assert pure_pt.effective_prefix_hit_rate == pytest.approx(hit_rate_target)

    # Direct kv_capacity_and_hit_rate preserves capacity from sw_window to ctx_long.
    cap_short, h_short = kv_capacity_and_hit_rate(
        hybrid_swa_geom,
        dt,
        par,
        conc_one,
        KvFitInput(
            b_ws=conc_one,
            ctx_full=sw_window,
            hbm_limit_b=hbm_limit_b,
            w_base_b=w_base_b,
            b_kv_phys_core=full_kv_per_tok,
            offload_b_per_core=no_offload,
            prefix_hit_rate=hit_rate_target,
            kv_working_set_turns=single_turn,
        ),
        fixed_kv_per_seq_bytes=swa_fixed_per_seq,
    )
    cap_long, h_long = kv_capacity_and_hit_rate(
        hybrid_swa_geom,
        dt,
        par,
        conc_one,
        KvFitInput(
            b_ws=conc_one,
            ctx_full=ctx_long,
            hbm_limit_b=hbm_limit_b,
            w_base_b=w_base_b,
            b_kv_phys_core=full_kv_per_tok,
            offload_b_per_core=no_offload,
            prefix_hit_rate=hit_rate_target,
            kv_working_set_turns=single_turn,
        ),
        fixed_kv_per_seq_bytes=swa_fixed_per_seq,
    )
    assert cap_long == pytest.approx(cap_short)
    assert h_long == pytest.approx(h_short)


def test_prefill_options_keep_all_legal_variants() -> None:
    """Verify _find_prefill_options keeps all legal variants in the same class."""
    geom = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    dt = rf.Dtypes(w_dense=1.0, kv=1.0)
    chip = rf.CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    scen = rf.build_example_scenario("chat-8k1k")
    hbm_limit_b = DEFAULT_GPU_MEMORY_UTILIZATION * float(chip.hbm_bytes)
    ctx_full = scen.isl + scen.osl
    eval_context = _SolveContext(
        geom=geom,
        dtypes=dt,
        chip=chip,
        scenario=scen,
        mode="disagg",
        max_chips=topo.num_chips,
        max_instances=None,
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        eff_al=1.0,
        spec_k=0,
        ctx_full=ctx_full,
        ctx_mid=scen.isl + scen.osl // 2,
        eff_isl=scen.isl,
        n_chunks=max(1, scen.isl // scen.prefill_chunk),
        page_size=default_rpa_page_size(ctx_full),
        c_max=max(scen.concurrencies),
    )
    options, _ = _find_prefill_options(eval_context, [topo])
    span_target = topo.num_cores
    single_stage_variants = [
        opt
        for opt in options
        if opt.par.tp * opt.par.pcp == span_target and opt.par.pp == 1
    ]
    expected_pcp_tp_pairs = {(8, 1), (4, 2), (2, 4), (1, 8)}
    assert {(opt.par.tp, opt.par.pcp) for opt in single_stage_variants} == (
        expected_pcp_tp_pairs
    )
    assert len(single_stage_variants) == len(expected_pcp_tp_pairs)


def test_prefill_enumerates_both_ep_and_legal_tp_slice_moe() -> None:
    """Verify prefill enumerates EP plus legal TP-slice MoE when PCP is disabled."""
    base_moe = rf.from_hf_config(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 32000,
        }
    )
    topo = rf.TOPOLOGIES["2x2x1"]
    tp_8 = 8
    tp_4 = 4
    pcp_2 = 2
    block_128 = 128
    moe_div = tp_8 * block_128
    # Partial checkpoint block (not divisible by 128) cannot be TP-sharded (fp8.py).
    moe_partial_block = moe_div + 64

    geom_both = dc_replace(
        base_moe, moe_inter=moe_div, weight_block_size=(block_128, block_128)
    )
    pars_both = _candidate_prefill_parallels(geom_both, topo)
    modes_tp8 = {
        (p.moe, p.ep) for p in pars_both if p.pp == 1 and p.pcp == 1 and p.tp == tp_8
    }
    assert modes_tp8 == {("ep", tp_8), ("tp_slice", 1)}

    geom_ep_only = dc_replace(
        base_moe, moe_inter=moe_partial_block, weight_block_size=(block_128, block_128)
    )
    pars_ep_only = _candidate_prefill_parallels(geom_ep_only, topo)
    modes_ep_only = {
        (p.moe, p.ep) for p in pars_ep_only if p.pp == 1 and p.pcp == 1 and p.tp == tp_8
    }
    assert modes_ep_only == {("ep", tp_8)}
    dec_ep_only = _candidate_decode_parallels(geom_ep_only, topo)
    dec_tp8 = {(p.moe, p.ep) for p in dec_ep_only if p.tp == tp_8 and p.dp == 1}
    assert dec_tp8 == {("ep", tp_8)}

    geom_pcp_div = dc_replace(
        base_moe,
        num_experts=tp_4 * pcp_2,
        moe_inter=moe_div,
        weight_block_size=(block_128, block_128),
    )
    pars_pcp_div = _candidate_prefill_parallels(geom_pcp_div, topo)
    modes_pcp_div = {
        (p.moe, p.ep)
        for p in pars_pcp_div
        if p.pp == 1 and p.pcp == pcp_2 and p.tp == tp_4
    }
    assert modes_pcp_div == {("ep", tp_4 * pcp_2)}

    geom_pcp_indiv = dc_replace(
        base_moe,
        num_experts=tp_4 * pcp_2 + pcp_2 * 2,
        moe_inter=moe_div,
        weight_block_size=(block_128, block_128),
    )
    pars_pcp_indiv = _candidate_prefill_parallels(geom_pcp_indiv, topo)
    modes_pcp_indiv = {
        (p.moe, p.ep)
        for p in pars_pcp_indiv
        if p.pp == 1 and p.pcp == pcp_2 and p.tp == tp_4
    }
    assert modes_pcp_indiv == set()


def test_solver_think_time_preserves_working_set_and_sets_max_num_seqs() -> None:
    """Verify KvFitInput uses c/dp working set and always populates max_num_seqs."""
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    base_scen = rf.TrafficScenario(
        name="think-ws",
        isl=32768,
        osl=512,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(64,),
        kv_working_set_turns=1.0,
    )
    think_scen = dc_replace(base_scen, think_time_s=60.0)
    plan_tt0 = rf.solve_serving_plan(
        geom, dt, chip, base_scen, SolveOptions(mode="mixed", max_chips=4)
    )
    plan_tt60 = rf.solve_serving_plan(
        geom, dt, chip, think_scen, SolveOptions(mode="mixed", max_chips=4)
    )
    expected_tp = 8
    pt_tt0 = next(
        p
        for p in plan_tt0.candidates
        if p.decode_pool.tp == expected_tp and p.decode_pool.dp == 1
    )
    pt_tt60 = next(
        p
        for p in plan_tt60.candidates
        if p.decode_pool.tp == expected_tp and p.decode_pool.dp == 1
    )
    assert pt_tt0.decode_pool.max_num_seqs is not None
    assert pt_tt60.decode_pool.max_num_seqs is not None
    assert pt_tt60.effective_prefix_hit_rate is not None
    assert pt_tt0.effective_prefix_hit_rate is not None
    assert pt_tt60.effective_prefix_hit_rate == pytest.approx(
        pt_tt0.effective_prefix_hit_rate, rel=1e-2
    )


def test_solver_prunes_hybrid_multihost_and_enforces_qwen4_exp_limits() -> None:
    """Verify multi-host native mp Mode A/B filter and TPUQwen4ExpConfig bounds."""
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    topo_4host = rf.TOPOLOGIES["2x2x4"]
    hosts = topo_4host.num_chips // chip.chips_per_host
    cores_per_host = chip.chips_per_host * topo_4host.cores_per_chip
    dec_pars = _candidate_decode_parallels(geom, topo_4host, chip=chip)
    assert dec_pars
    for par in dec_pars:
        assert par.dp == 1 or (par.dp % hosts == 0 and par.tp <= cores_per_host)
    # Hybrid 1 < dp < hosts (dp=2 on 4 hosts) must be pruned.
    hybrid_dp = 2
    assert not any(par.dp == hybrid_dp for par in dec_pars)

    qwen4_cfg = load_qwen3_8_bringup_config()
    qwen4_geom = rf.from_hf_config(qwen4_cfg, model_ref="Qwen/Qwen3.8-Flash-Next-FP8")
    qwen4_dt = rf.dtypes_from_hf_config(qwen4_cfg)
    qwen4_dec = _candidate_decode_parallels(
        qwen4_geom, rf.TOPOLOGIES["2x2x1"], chip=chip
    )
    assert qwen4_dec
    assert all(p.kv_split == 1 for p in qwen4_dec)
    qwen4_pre = _candidate_prefill_parallels(
        qwen4_geom, rf.TOPOLOGIES["2x2x1"], chip=chip
    )
    assert qwen4_pre
    assert all(p.pp == 1 and p.pcp == 1 for p in qwen4_pre)

    # Exceeding qsa_budget (2048) returns no candidates + explicit warning.
    over_budget_scen = rf.TrafficScenario(
        name="qwen4-over-budget",
        isl=2048,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    over_plan = rf.solve_serving_plan(
        qwen4_geom, qwen4_dt, chip, over_budget_scen, SolveOptions(max_chips=4)
    )
    assert over_plan.recommended is None
    assert any("indexer_budget" in w for w in over_plan.warnings)


def test_smem_page_size_escalation_lifts_resident_cap_at_long_context() -> None:
    """Verify automatic tpu_min_page_size escalation lifts SMEM cap at 64K."""
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = rf.Dtypes(w_dense=1.0, w_expert=1.0, w_aux=2.0, kv=1.0, state=2.0, lm_head=2.0)
    ctx_64k = 65536
    # Use a tighter synthetic SMEM table budget so unescalated page_size=128 caps
    # at 16 (below the 43-seq HBM cap) while escalated page_size=256+ lifts it.
    tight_chip = dc_replace(chip, smem_page_table_bytes=32768)
    cap_unescalated = smem_max_seqs(
        ctx_64k, 1, 128, page_table_bytes=tight_chip.smem_page_table_bytes
    )
    scen = rf.TrafficScenario(
        name="smem-escalate-64k",
        isl=ctx_64k - 512,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(cap_unescalated + 32,),
        kv_working_set_turns=1.0,
    )
    plan_auto = rf.solve_serving_plan(
        geom, dt, tight_chip, scen, SolveOptions(mode="mixed", max_chips=4)
    )
    tp_target = 8
    tp8_auto = [
        pt
        for pt in plan_auto.candidates
        if pt.decode_pool.tp == tp_target
        and pt.decode_pool.dp == 1
        and pt.decode_pool.kv_split == 1
    ]
    assert tp8_auto
    assert max(pt.decode_pool.max_num_seqs or 0 for pt in tp8_auto) > cap_unescalated


def test_disagg_think_time_preserves_decode_working_set_hit_rate_clamp() -> None:
    """Verify disagg decode KvFitInput clamps hit rate by c/(y_d*dp) on GLM-5.2."""
    glm_path = Path(__file__).resolve().parent / "fixtures" / "glm_5_2_fp8.json"
    glm_cfg = json.loads(glm_path.read_text())
    chip = rf.CHIPS["tpu7x"]
    geom = rf.from_hf_config(glm_cfg)
    dt = rf.dtypes_from_hf_config(glm_cfg)
    base_scen = rf.TrafficScenario(
        name="disagg-think-ws",
        isl=32768,
        osl=256,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(256,),
        kv_working_set_turns=1.0,
    )
    think_scen = dc_replace(base_scen, think_time_s=60.0)
    opts = SolveOptions(mode="disagg", max_chips=16)
    plan_tt0 = rf.solve_serving_plan(geom, dt, chip, base_scen, opts)
    plan_tt60 = rf.solve_serving_plan(geom, dt, chip, think_scen, opts)
    clamped_ceiling = 0.85

    def _pair_key(pt: rf.ParetoPoint) -> tuple[object, ...] | None:
        """Build a full disagg pool identity key for exact tt=0 vs tt=60 matching."""
        if pt.prefill_pool is None:
            return None
        d, p = pt.decode_pool, pt.prefill_pool
        return (
            d.tp,
            d.dp,
            d.ep,
            d.pp,
            d.pcp,
            d.kv_split,
            d.instances,
            d.moe_dispatch,
            p.tp,
            p.dp,
            p.pp,
            p.pcp,
            p.instances,
        )

    tt60_by_key = {
        key: pt.effective_prefix_hit_rate
        for pt in plan_tt60.candidates
        if (key := _pair_key(pt)) is not None
        and pt.effective_prefix_hit_rate is not None
    }
    matched_pairs = [
        (pt.effective_prefix_hit_rate, tt60_by_key[key])
        for pt in plan_tt0.candidates
        if (key := _pair_key(pt)) is not None
        and key in tt60_by_key
        and pt.effective_prefix_hit_rate is not None
        and pt.effective_prefix_hit_rate < clamped_ceiling
    ]
    assert matched_pairs
    for h0, h60 in matched_pairs:
        assert h60 == pytest.approx(h0, abs=1e-6)


def test_moe_tp_slice_hbm_pads_to_128_and_pp_uses_heaviest_stage() -> None:
    """Verify MoE TP-slice HBM pads shard width to 128 and PP HBM uses ceil(L/pp)."""
    q35_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    q35_cfg = json.loads(q35_path.read_text())
    geom = rf.from_hf_config(q35_cfg)
    dt = rf.dtypes_from_hf_config(q35_cfg)
    # Qwen3.5 moe_inter=1024: TP8=128, TP16=64 padded to 128 -> equal routed HBM.
    w_tp8, _ = compute_w_ffn(geom, dt, rf.Parallel(tp=8, moe="tp_slice"))
    w_tp16, _ = compute_w_ffn(geom, dt, rf.Parallel(tp=16, moe="tp_slice"))
    min_padded_ratio = 0.9
    assert w_tp16 > min_padded_ratio * w_tp8

    # Qwen3.5 L=60 at PP8: heaviest stage holds ceil(60/8)=8 layers (8/60, not 7.5/60).
    w_attn_pp1 = compute_w_attn(
        geom, dt, rf.Parallel(tp=1, pp=1), rf.count_layer_types(geom)
    )
    w_attn_pp8 = compute_w_attn(
        geom, dt, rf.Parallel(tp=1, pp=8), rf.count_layer_types(geom)
    )
    expected_pp8_scale = 8.0 / 60.0
    assert w_attn_pp8 == pytest.approx(w_attn_pp1 * expected_pp8_scale, rel=1e-9)


def test_pcp_rejected_when_spec_k_exceeds_one() -> None:
    """Verify solver gates out PCP > 1 whenever spec_k > 1 (pcp_validation.py)."""
    q35_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    q35_cfg = json.loads(q35_path.read_text())
    geom = rf.from_hf_config(q35_cfg, model_ref="Qwen/Qwen3.5-397B-A17B-FP8")
    dt = rf.dtypes_from_hf_config(q35_cfg)
    chip = rf.CHIPS["tpu7x"]
    scen_k3 = rf.TrafficScenario(
        name="pcp-spec-k3",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=3.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    plan_k3 = rf.solve_serving_plan(
        geom, dt, chip, scen_k3, SolveOptions(mode="disagg", max_chips=16)
    )
    assert plan_k3.candidates
    assert all(pt.decode_pool.pcp == 1 for pt in plan_k3.candidates)
    assert any(
        pt.prefill_pool is not None and pt.prefill_pool.pcp > 1
        for pt in plan_k3.candidates
    )
    plan_mixed_k3 = rf.solve_serving_plan(
        geom, dt, chip, scen_k3, SolveOptions(mode="mixed", max_chips=16)
    )
    assert plan_mixed_k3.candidates
    assert all(
        pt.prefill_pool is None and pt.decode_pool.pcp == 1
        for pt in plan_mixed_k3.candidates
    )

    scen_k1 = dc_replace(scen_k3, accept_len=1.5, spec_tokens=1)
    plan_k1 = rf.solve_serving_plan(
        geom, dt, chip, scen_k1, SolveOptions(mode="disagg", max_chips=16)
    )
    assert any(
        pt.prefill_pool is not None and pt.prefill_pool.pcp > 1
        for pt in plan_k1.candidates
    )


def test_hbm_breakdown_matches_footprint() -> None:
    """HbmBreakdown total matches hbm_footprint_per_core_bytes across CDK fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    glm_cfg = json.loads(
        (fixtures_dir / "glm_5_2_fp8.json").read_text(encoding="utf-8")
    )
    qwen3_8_cfg = load_qwen3_8_bringup_config()
    geoms = [
        rf.from_hf_config(QWEN35_CFG),
        rf.from_hf_config(glm_cfg),
        rf.from_hf_config(qwen3_8_cfg),
    ]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    par = Parallel(tp=4, dp=1, ep=1, pp=1, pcp=1)
    seq_layout = SeqKvLayout(seqs_per_dp=16, context=2048)

    for geom in geoms:
        bd = hbm_breakdown_per_core_bytes(geom, dt, par, seq_layout)
        footprint = hbm_footprint_per_core_bytes(geom, dt, par, seq_layout)
        assert bd.total == footprint
        assert math.isclose(bd.weights + bd.kv + bd.state, bd.total, rel_tol=1e-9)
        assert bd.weights > 0
        assert bd.kv >= 0
        if geom.linear_k_heads:
            assert bd.state > 0
        else:
            assert bd.state == 0.0


@pytest.mark.parametrize("mode", ["mixed", "disagg", "auto"])
def test_search_stats_funnel_is_monotone(
    mode: Literal["mixed", "disagg", "auto"],
) -> None:
    """Search funnel is monotonically non-increasing and frontier is positive."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scenario = dc_replace(
        EXAMPLE_SCENARIOS["chat-8k1k"],
        concurrencies=(1, 4),
        ttft_slo_ms=5000.0,
        tpot_slo_ms=500.0,
    )
    max_chips = 8 if mode == "disagg" else 4
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode=mode, max_chips=max_chips),
    )
    assert splan.search is not None
    stats = splan.search
    assert stats.enumerated >= stats.hbm_fit >= stats.slo_met >= stats.frontier >= 1
    assert stats.frontier == len(splan.frontier)


def test_decode_floor_matches_binding() -> None:
    """Every candidate has decode_floor matching its binding resource."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scenario = dc_replace(
        EXAMPLE_SCENARIOS["chat-8k1k"],
        concurrencies=(1, 4),
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    for pt in splan.candidates:
        assert pt.decode_floor is not None
        df = pt.decode_floor
        expected_hard = max(
            df.hbm_ms, df.matmul_ms, df.ici_ms, df.chain_ms, df.hop_latency_ms
        )
        assert df.hard_floor_ms <= expected_hard + 1e-9
        if df.binding == "hbm":
            assert df.hard_floor_ms <= df.hbm_ms + 1e-9
        elif df.binding == "matmul":
            assert math.isclose(df.matmul_ms, df.hard_floor_ms, rel_tol=1e-5)


def test_explicit_gqa_token_stays_gqa_when_kv_lora_rank_positive() -> None:
    """Verify explicit GQA layer token is sized as GQA and indexer weights replicate."""
    dsa_base = {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "layer_types": ["dsa"] * 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 64,
        "kv_lora_rank": 512,
        "q_lora_rank": 1024,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "intermediate_size": 8192,
        "vocab_size": 32000,
    }
    geom_idx = _from_hf_config_dry_run(dsa_base)
    geom_no_idx = dc_replace(geom_idx, index_n_heads=0)
    geom_gqa_tok = dc_replace(geom_idx, layer_types=("gqa",) * 4)
    dt = rf.Dtypes(w_dense=1.0, kv=1.0)
    par = rf.Parallel(tp=4, dp=1, ep=1, pp=1, pcp=1)
    seq_layout = SeqKvLayout(4, 2048)

    bd_idx = hbm_breakdown_per_core_bytes(geom_idx, dt, par, seq_layout)
    bd_no_idx = hbm_breakdown_per_core_bytes(geom_no_idx, dt, par, seq_layout)
    bd_gqa_tok = hbm_breakdown_per_core_bytes(geom_gqa_tok, dt, par, seq_layout)

    expected_idx_w = (
        float(geom_idx.num_layers * dsa_indexer_layer_weight_elems(geom_idx))
        * dt.w_dense
    )
    assert bd_idx.weights - bd_no_idx.weights == pytest.approx(expected_idx_w, rel=1e-6)
    assert bd_gqa_tok.weights < bd_idx.weights
    assert bd_gqa_tok.kv > 0.0


def test_search_stats_tracks_core_grid_before_arch_pruning() -> None:
    """SearchStats.core_grid counts core-partitioned tuples before arch legality."""
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    scenario = rf.TrafficScenario(
        name="core-grid-check",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 4),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        geom,
        Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0),
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert plan.search is not None
    # 8-core (PP, TP, DP, PCP) partitions weighted by DCP|TP: (10*1 + 6*2 + 3*3 + 1*4),
    # times 4 EP divisors {1, 2, 4, 8} and 3 decode MoE dispatches per concurrency.
    expected_core_grid = (
        (10 * 1 + 6 * 2 + 3 * 3 + 1 * 4) * 4 * 3 * len(scenario.concurrencies)
    )
    assert plan.search.core_grid == expected_core_grid
    assert plan.search.core_grid > plan.search.enumerated


def test_decode_capacity_profile_subtracts_logit_peak_from_b_hbm_max() -> None:
    """_decode_capacity_profile reserves logit_peak_b when computing b_hbm_max."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = rf.dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = rf.CHIPS["tpu7x"]
    scen = rf.TrafficScenario(
        name="logit-cap-check",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=1.0,
    )
    par = rf.Parallel(tp=2, dp=4, ep=8, moe="ep", ranks_per_chip=chip.cores_per_chip)
    ctx_full = scen.isl + scen.osl
    hbm_limit_b = float(chip.hbm_bytes) * rf.constants.DEFAULT_GPU_MEMORY_UTILIZATION
    eval_context = _SolveContext(
        geom=geom,
        dtypes=dt,
        chip=chip,
        scenario=scen,
        mode="mixed",
        max_chips=4,
        max_instances=None,
        gpu_memory_utilization=rf.constants.DEFAULT_GPU_MEMORY_UTILIZATION,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / rf.constants.UNIT_GIGA,
        eff_al=1.0,
        spec_k=0,
        ctx_full=ctx_full,
        ctx_mid=scen.isl + scen.osl // 2,
        eff_isl=scen.isl,
        n_chunks=1,
        page_size=rf.default_rpa_page_size(ctx_full, is_mla=geom.kv_lora_rank > 0),
        c_max=max(scen.concurrencies),
    )
    w_base_0, b_max_0, _, _, _, hbm_c1 = _decode_capacity_profile(
        eval_context, par, logit_peak_b=0.0
    )
    per_seq_b = hbm_c1 - w_base_0
    reserved_seqs = 4
    assert b_max_0 > reserved_seqs
    large_logit_peak = float(reserved_seqs) * per_seq_b
    w_base_1, b_max_1, _, _, _, _ = _decode_capacity_profile(
        eval_context, par, logit_peak_b=large_logit_peak
    )
    assert w_base_1 == pytest.approx(w_base_0, rel=0, abs=0)
    assert b_max_1 <= b_max_0 - reserved_seqs
    assert (
        w_base_1 + float(b_max_1) * per_seq_b + large_logit_peak <= hbm_limit_b + 1e-6
    )


def test_pp_dp_accounts_for_logit_peak_on_last_stage() -> None:
    """DP shifts layers off the last stage when logit_peak_b pushes it above limit."""
    cfg = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "glm_5_2_fp8.json").read_text(
            encoding="utf-8"
        )
    )
    geom = rf.from_hf_config(cfg)
    dt = rf.dtypes_from_hf_config(cfg)
    par = Parallel(tp=1, pp=4, moe="tp_slice", ranks_per_chip=2)

    part_zero, _bd_zero, last_zero, ok_zero = rf.pp_stage_breakdowns(
        geom,
        dt,
        par,
        (4096, 1),
        options=rf.PpStageOptions(disagg=True, logit_peak_b=0.0),
    )
    assert ok_zero
    logit_peak_b = 12.0 * float(UNIT_GIGA)
    part_peak, _bd_peak, last_peak, ok_peak = rf.pp_stage_breakdowns(
        geom,
        dt,
        par,
        (4096, 1),
        options=rf.PpStageOptions(disagg=True, logit_peak_b=logit_peak_b),
    )
    assert ok_peak
    assert sum(part_peak) == geom.num_layers
    assert part_peak[-1] < part_zero[-1]
    assert last_peak + logit_peak_b < last_zero + logit_peak_b


def test_pp_dp_rejects_zero_full_attention_stage_for_hybrid_raiden() -> None:
    """Hybrid Raiden partitioning requires >= 1 full-attention layer on every stage."""
    asym_hybrid_cfg = {
        **SMALL_HYBRID_MOE_CFG,
        "text_config": {
            **SMALL_HYBRID_MOE_CFG["text_config"],
            "num_hidden_layers": 8,
            "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2,
        },
    }
    geom = rf.from_hf_config(asym_hybrid_cfg)
    dt = rf.dtypes_from_hf_config(asym_hybrid_cfg)
    par_pp2 = Parallel(tp=1, pp=2, moe="tp_slice", ranks_per_chip=2)
    part_pp2 = rf.partition_pp_layers(
        geom,
        dt,
        par_pp2,
        (4096, 1),
        options=rf.PpStageOptions(require_full_attn_per_stage=True),
    )
    assert part_pp2 == (7, 1)
    assert "full_attention" in geom.layer_types[: part_pp2[0]]
    assert "full_attention" in geom.layer_types[part_pp2[0] :]

    par_pp3 = Parallel(tp=1, pp=3, moe="tp_slice", ranks_per_chip=2)
    part_pp3, _bd3, _last3, ok_pp3 = rf.pp_stage_breakdowns(
        geom,
        dt,
        par_pp3,
        (4096, 1),
        options=rf.PpStageOptions(require_full_attn_per_stage=True),
    )
    assert not ok_pp3
    assert part_pp3 == ()
    with pytest.raises(ValueError, match="Cannot partition"):
        rf.partition_pp_layers(
            geom,
            dt,
            par_pp3,
            (4096, 1),
            options=rf.PpStageOptions(require_full_attn_per_stage=True),
        )


def test_uneven_pp_scales_hybrid_kv_by_exact_stage_attention_layers() -> None:
    """Uneven PP hybrid KV uses exact stage full-attention layers, not layer ratio."""
    asym_hybrid_cfg = {
        **SMALL_HYBRID_MOE_CFG,
        "text_config": {
            **SMALL_HYBRID_MOE_CFG["text_config"],
            "num_hidden_layers": 8,
            "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2,
        },
    }
    geom = rf.from_hf_config(asym_hybrid_cfg)
    dt = rf.dtypes_from_hf_config(asym_hybrid_cfg)
    par_pp2 = Parallel(tp=1, pp=2, moe="tp_slice", ranks_per_chip=2)
    lin_stage, fix_stage = rf.pp_stage_kv_parts(
        geom, dt, par_pp2, ((7, 1), 4096), chip=CHIPS["tpu7x"]
    )
    lin_even, _ = rf.memory.kv_bytes_per_core_parts(
        geom, dt, par_pp2, 4096, chip=CHIPS["tpu7x"]
    )
    # Both stages hold exactly 1 full-attention layer, equal to even 2/2 == 1 layer,
    # NOT inflated by 7/4 = 1.75x.
    assert lin_stage == pytest.approx(lin_even, rel=1e-9)
    assert fix_stage == 0.0

    # When stage 0 holds 2 full-attention layers and stage 1 holds 0, exact max stage
    # KV equals 2x the even 1-layer average.
    skewed_cfg = {
        **SMALL_HYBRID_MOE_CFG,
        "text_config": {
            **SMALL_HYBRID_MOE_CFG["text_config"],
            "num_hidden_layers": 8,
            "layer_types": ["full_attention"] * 2 + ["linear_attention"] * 6,
        },
    }
    skewed_geom = rf.from_hf_config(skewed_cfg)
    lin_skewed, _ = rf.pp_stage_kv_parts(
        skewed_geom, dt, par_pp2, ((3, 5), 4096), chip=CHIPS["tpu7x"]
    )
    assert lin_skewed == pytest.approx(2.0 * lin_even, rel=1e-9)


def test_disagg_dcn_and_pcie_offload_affect_ttft_on_cdk_fixtures() -> None:
    """Verify DCN bandwidth and host PCIe offload affect disagg and mixed TTFT."""
    cfg = json.loads(
        (
            Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
        ).read_text(encoding="utf-8")
    )
    geom = rf.from_hf_config(cfg)
    dt = rf.dtypes_from_hf_config(cfg)
    chip = CHIPS["tpu7x"]
    scen_off = rf.TrafficScenario(
        name="q35-offload-dcn",
        isl=16384,
        osl=256,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=20.0,
        kv_offload_gb=64.0,
    )
    plan_fast = solve_serving_plan(
        geom, dt, chip, scen_off, SolveOptions(mode="mixed", max_chips=8)
    )
    plan_slow_pcie = solve_serving_plan(
        geom,
        dt,
        dc_replace(chip, pcie_bw=4.0e9),
        scen_off,
        SolveOptions(mode="mixed", max_chips=8),
    )
    assert plan_fast.recommended is not None
    assert plan_slow_pcie.recommended is not None
    assert plan_slow_pcie.recommended.est_ttft_ms > plan_fast.recommended.est_ttft_ms

    scen_dcn = rf.TrafficScenario(
        name="q35-dcn",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    disagg_fast = solve_serving_plan(
        geom, dt, chip, scen_dcn, SolveOptions(mode="disagg", max_chips=16)
    )
    disagg_slow_dcn = solve_serving_plan(
        geom,
        dt,
        dc_replace(chip, dcn_bw=1.0e9),
        scen_dcn,
        SolveOptions(mode="disagg", max_chips=16),
    )
    assert disagg_fast.recommended is not None
    assert disagg_slow_dcn.recommended is not None
    assert disagg_slow_dcn.recommended.est_ttft_ms > disagg_fast.recommended.est_ttft_ms


def test_disagg_prefill_uses_min_one_spec_k_and_allows_pp_with_pcp() -> None:
    """Verify disagg clamps prefill spec_k to 1, admits PPxPCP, and sets HND."""
    cfg_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    geom = from_hf_config(cfg)
    dt = dtypes_from_hf_config(cfg)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="q35-disagg-spec3",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.5,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    max_disagg_chips = 32
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=max_disagg_chips),
    )
    assert plan.candidates
    pre_axes = {
        (p.prefill_pool.pp, p.prefill_pool.pcp, p.prefill_pool.tp)
        for p in plan.candidates
        if p.prefill_pool is not None
    }
    assert (8, 1, 1) in pre_axes
    assert (4, 2, 1) in pre_axes
    assert (1, 4, 2) in pre_axes

    def _launched_spec(cli_args: list[str]) -> int:
        """Read num_speculative_tokens from emitted --speculative-config JSON flag."""
        idx = cli_args.index("--speculative-config") + 1
        return int(json.loads(cli_args[idx])["num_speculative_tokens"])

    def _find_candidate(
        *,
        pre_match: tuple[int, int, int] | None = None,
        dec_match: tuple[int, int] | None = None,
    ) -> rf.ParetoPoint:
        """Return first disagg candidate matching prefill (pp,pcp,tp) or decode."""
        for cand in plan.candidates:
            if cand.prefill_pool is None:
                continue
            pre = (cand.prefill_pool.pp, cand.prefill_pool.pcp, cand.prefill_pool.tp)
            dec = (cand.decode_pool.tp, cand.decode_pool.kv_split)
            if (pre_match is None or pre == pre_match) and (
                dec_match is None or dec == dec_match
            ):
                return cand
        msg = f"Missing candidate pre={pre_match}, dec={dec_match}"
        raise AssertionError(msg)

    expected_pre_k = 1
    expected_dec_k = 3
    launch_opts = LaunchOptions(
        kv_dtype="fp8",
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        chip=chip,
        model_id=plan.model_id,
        spec_k=expected_dec_k,
        spec_tokens=expected_dec_k,
        mtp_layers=geom.mtp_layers,
        is_hybrid=True,
    )
    for pre_key in ((8, 1, 1), (4, 2, 1), (1, 4, 2), (1, 1, 8)):
        pt = _find_candidate(pre_match=pre_key)
        launched = point_launch_args(pt, launch_opts)
        assert _launched_spec(launched["prefill_cli_args"]) == expected_pre_k
        assert _launched_spec(launched["cli_args"]) == expected_dec_k
        if pre_key != (8, 1, 1):
            assert "VLLM_KV_CACHE_LAYOUT=HND" in launched["commands"][0]

    pt_dec_dcp = _find_candidate(dec_match=(8, 4))
    launched_dcp = point_launch_args(pt_dec_dcp, launch_opts)
    assert "VLLM_KV_CACHE_LAYOUT=HND" in launched_dcp["commands"][-1]


def test_uniform_pp_stage_floor_prices_heterogeneous_layer_mix() -> None:
    """Verify uniform PP layer counts preserve per-stage max floor scaling."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    q35_cfg = json.loads((fixtures_dir / "qwen3_5_397b_fp8.json").read_text())
    glm_cfg = json.loads((fixtures_dir / "glm_5_2_fp8.json").read_text())
    base_chip = CHIPS["tpu7x"]
    ms_per_s = 1000.0
    unit_topo = rf.SliceTopology("s1x1x1", (1, 1, 1))
    large_hbm_chip = dc_replace(base_chip, hbm_bytes=512 * 1024**3)

    cases = (
        (
            q35_cfg,
            Parallel(tp=1, pp=4, pcp=2),
            (15, 15, 15, 15),
            TOPOLOGIES["2x2x1"],
            base_chip,
        ),
        (glm_cfg, Parallel(tp=1, pp=2, pcp=1), (39, 39), unit_topo, large_hbm_chip),
    )
    for cfg, target_par, exp_part, topo, case_chip in cases:
        geom = from_hf_config(cfg)
        dt = dtypes_from_hf_config(cfg)
        scen = TrafficScenario(
            name=f"pp-mix-{geom.model_id}",
            isl=4096,
            osl=256,
            prefix_hit_rate=0.5,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(16,),
            kv_working_set_turns=1.0,
        )
        ctx = _SolveContext(
            geom=geom,
            dtypes=dt,
            chip=case_chip,
            scenario=scen,
            mode="disagg",
            max_chips=8,
            max_instances=None,
            gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
            hbm_limit_b=case_chip.hbm_bytes * DEFAULT_GPU_MEMORY_UTILIZATION,
            hbm_limit_gb=(case_chip.hbm_bytes * DEFAULT_GPU_MEMORY_UTILIZATION)
            / UNIT_GIGA,
            eff_al=1.0,
            spec_k=0,
            ctx_full=scen.isl + scen.osl,
            ctx_mid=scen.isl + scen.osl // 2,
            eff_isl=2048,
            n_chunks=1,
            page_size=default_rpa_page_size(
                scen.isl + scen.osl, is_mla=geom.kv_lora_rank > 0
            ),
            c_max=16,
        )
        pre_opts, _ = _find_prefill_options(ctx, [topo])
        pp_opts = [
            opt
            for opt in pre_opts
            if opt.par.pp == target_par.pp
            and opt.par.pcp == target_par.pcp
            and opt.par.tp == target_par.tp
        ]
        assert pp_opts
        assert all(opt.pp_part is None for opt in pp_opts)

        shape = StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=scen.isl,
            prefix_hit_rate=scen.prefix_hit_rate,
        )
        exact_sf = scale_pp_stage_floor(
            geom,
            case_chip,
            pp_opts[0].par,
            (shape, FloorOptions(dt=dt, topology=topo)),
            exp_part,
        )
        avg_sf = _prefill_stage_step_floor(
            ctx, topo, pp_opts[0].par, scen.prefix_hit_rate, None
        )
        assert exact_sf.hard_floor_s != pytest.approx(avg_sf.hard_floor_s, rel=1e-6)
        assert pp_opts[0].bound_ttft_ms == pytest.approx(
            exact_sf.hard_floor_s * ms_per_s, rel=1e-9
        )

    # Full solve_serving_plan on qwen3_5 keeps PoolMeshPlan.pp_layer_partition as None.
    q35_geom = from_hf_config(q35_cfg)
    q35_dt = dtypes_from_hf_config(q35_cfg)
    q35_scen = TrafficScenario(
        name="pp4-uniform-check",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        q35_geom, q35_dt, base_chip, q35_scen, SolveOptions(mode="disagg", max_chips=16)
    )
    target_pp = 4
    matched = [
        pt
        for pt in plan.candidates
        if pt.prefill_pool is not None and pt.prefill_pool.pp == target_pp
    ]
    assert matched
    assert all(
        pt.prefill_pool.pp_layer_partition is None
        for pt in matched
        if pt.prefill_pool is not None
    )


def test_glm_raiden_main_admits_tp_match_dcp_and_blocks_mismatched_pairs() -> None:
    """Verify GLM-5.2 main profile enforces Raiden P/D rules vs hardware profile."""
    cfg_path = Path(__file__).resolve().parent / "fixtures" / "glm_5_2_fp8.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    geom_main = from_hf_config(cfg)
    geom_hw = from_hf_config(cfg, runtime_profile="hardware")
    dt = dtypes_from_hf_config(cfg)
    chip = CHIPS["tpu7x"]

    assert _valid_raiden_kv_pair(
        geom_main, Parallel(tp=16, kv_split=16), Parallel(tp=16, kv_split=16)
    )
    assert not _valid_raiden_kv_pair(
        geom_main, Parallel(tp=16, kv_split=1), Parallel(tp=16, kv_split=16)
    )
    assert not _valid_raiden_kv_pair(
        geom_main, Parallel(tp=16, kv_split=8), Parallel(tp=16, kv_split=8)
    )
    assert _valid_raiden_kv_pair(
        geom_hw, Parallel(tp=16, kv_split=8), Parallel(tp=16, kv_split=8)
    )

    scen = TrafficScenario(
        name="glm-raiden-dcp",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    opts = SolveOptions(mode="disagg", max_chips=16)
    plan_main = solve_serving_plan(geom_main, dt, chip, scen, opts)
    main_pairs = {
        (
            p.prefill_pool.tp,
            p.prefill_pool.kv_split,
            p.decode_pool.tp,
            p.decode_pool.kv_split,
        )
        for p in plan_main.candidates
        if p.prefill_pool is not None
    }
    assert (16, 16, 16, 16) in main_pairs
    assert (16, 1, 16, 16) not in main_pairs
    assert (16, 8, 16, 8) not in main_pairs

    reason = explain_layout_rejection(
        geom_main,
        {
            "prefill": {"tp": 16, "dp": 1, "pp": 1, "pcp": 1, "kv_split": 16},
            "decode": {"tp": 16, "dp": 1, "pp": 1, "pcp": 1, "kv_split": 16},
        },
        [TOPOLOGIES["2x2x2"]],
        opts,
        chip=chip,
    )
    assert "DP1 and DCP1" not in reason

    plan_hw = solve_serving_plan(geom_hw, dt, chip, scen, opts)
    hw_pairs = {
        (
            p.prefill_pool.tp,
            p.prefill_pool.kv_split,
            p.decode_pool.tp,
            p.decode_pool.kv_split,
        )
        for p in plan_hw.candidates
        if p.prefill_pool is not None
    }
    assert (16, 8, 16, 8) in hw_pairs


def test_qwen4_exp_main_vs_hardware_profile_and_fp4_kv_gate() -> None:
    """Verify qwen4_exp bring-up limits on main vs unlocked hardware + fp4 KV."""
    raw_path = (
        Path(__file__).resolve().parent / "fixtures" / "qwen3_8_flash_next_fp8.json"
    )
    raw_cfg = json.loads(raw_path.read_text(encoding="utf-8"))
    with pytest.raises(UnsupportedArchitectureError, match="ple_layer_ids"):
        from_hf_config(raw_cfg)

    bringup_cfg = load_qwen3_8_bringup_config()
    geom_main = from_hf_config(bringup_cfg)
    dt_main = dtypes_from_hf_config(bringup_cfg)
    chip = CHIPS["tpu7x"]

    over_ctx = TrafficScenario(
        name="q38-over-ctx",
        isl=2048,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
    )
    plan_over = solve_serving_plan(
        geom_main, dt_main, chip, over_ctx, SolveOptions(mode="mixed", max_chips=4)
    )
    assert plan_over.recommended is None
    assert any("indexer_budget" in w for w in plan_over.warnings)

    with_spec = TrafficScenario(
        name="q38-spec3",
        isl=512,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.5,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    plan_spec = solve_serving_plan(
        geom_main, dt_main, chip, with_spec, SolveOptions(mode="mixed", max_chips=4)
    )
    assert plan_spec.recommended is None
    assert any("speculative" in w.lower() for w in plan_spec.warnings)

    for bad_kv_scheme in ("fp4", "int4_per_token_head"):
        with pytest.raises(ValueError, match="Unsupported 4-bit/FP4 KV"):
            dtypes_from_hf_config(
                {
                    **raw_cfg,
                    "quantization_config": {
                        **dict(raw_cfg["quantization_config"]),
                        "kv_cache_scheme": bad_kv_scheme,
                    },
                }
            )
    for bad_scheme in (
        "int8",
        "float32",
        {"type": "int8", "num_bits": 8},
    ):
        for prof in ("main", "hardware"):
            with pytest.raises(ValueError, match="Unsupported"):
                dtypes_from_hf_config(
                    {
                        **raw_cfg,
                        "quantization_config": {
                            **dict(raw_cfg["quantization_config"]),
                            "kv_cache_scheme": bad_scheme,
                        },
                    },
                    runtime_profile=prof,
                )

    for fp8_tok in (
        "fp8_e4m3fn",
        "fp8_e5m2fn",
        {"dynamic": False, "num_bits": 8, "type": "float"},
    ):
        for prof in ("main", "hardware"):
            dt_fp8 = dtypes_from_hf_config(
                {
                    **raw_cfg,
                    "quantization_config": {
                        **dict(raw_cfg["quantization_config"]),
                        "kv_cache_scheme": fp8_tok,
                    },
                },
                runtime_profile=prof,
            )
            assert dt_fp8.kv == pytest.approx(1.0)
            assert dt_fp8.kv_dtype == "fp8"

    fp4_dt = Dtypes(
        w_dense=dt_main.w_dense,
        w_expert=dt_main.w_expert,
        kv=0.5,
        kv_dtype="fp4",
        w_dense_dtype=dt_main.w_dense_dtype,
        w_expert_dtype=dt_main.w_expert_dtype,
    )
    short_scen = dc_replace(with_spec, spec_tokens=0, accept_len=1.0)
    plan_fp4_blocked = solve_serving_plan(
        geom_main, fp4_dt, chip, short_scen, SolveOptions(mode="mixed", max_chips=4)
    )
    assert plan_fp4_blocked.recommended is None
    assert any("fp4" in w.lower() for w in plan_fp4_blocked.warnings)


def test_qwen4_exp_hardware_profile_unlocks_fp4_kv_and_launch_args() -> None:
    """Verify qwen4_exp hardware profile unlocks full model, 100K ctx, and FP4 KV."""
    raw_path = (
        Path(__file__).resolve().parent / "fixtures" / "qwen3_8_flash_next_fp8.json"
    )
    raw_cfg = json.loads(raw_path.read_text(encoding="utf-8"))
    chip = CHIPS["tpu7x"]
    fp4_cfg = {
        **raw_cfg,
        "quantization_config": {
            **dict(raw_cfg["quantization_config"]),
            "kv_cache_scheme": "fp4",
        },
    }

    # Explicit hardware profile unlocks full model, 100K ctx, MTP, PP8->TP8xDCP4, FP4.
    geom_hw = from_hf_config(raw_cfg, runtime_profile="hardware")
    assert geom_hw.runtime_profile == "hardware"
    assert geom_hw.ple_layer_ids == (2,)
    assert geom_hw.mtp_layers == 1
    expected_fp4_bytes = 0.5
    for kv_scheme in ("fp4", "int4_per_token_head"):
        dt_hw_fp4_tok = dtypes_from_hf_config(
            {
                **raw_cfg,
                "quantization_config": {
                    **dict(raw_cfg["quantization_config"]),
                    "kv_cache_scheme": kv_scheme,
                },
            },
            runtime_profile="hardware",
        )
        assert dt_hw_fp4_tok.kv == pytest.approx(expected_fp4_bytes)
        assert dt_hw_fp4_tok.kv_dtype == "fp4"
    dt_hw_fp4 = dtypes_from_hf_config(fp4_cfg, runtime_profile="hardware")
    assert dt_hw_fp4.kv == pytest.approx(expected_fp4_bytes)
    assert dt_hw_fp4.kv_dtype == "fp4"

    hw_scen = TrafficScenario(
        name="q38-hw-100k",
        isl=98304,
        osl=4096,
        prefix_hit_rate=0.5,
        accept_len=1.5,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    plan_hw = solve_serving_plan(
        geom_hw,
        dt_hw_fp4,
        chip,
        hw_scen,
        SolveOptions(mode="disagg", max_chips=16),
    )
    assert plan_hw.recommended is not None
    assert plan_hw.runtime_profile == "hardware"
    expected_prefill_pp = 8
    expected_decode_tp = 8
    expected_decode_dcp = 4
    assert any(
        p.prefill_pool is not None
        and p.prefill_pool.pp == expected_prefill_pp
        and p.decode_pool.tp == expected_decode_tp
        and p.decode_pool.kv_split == expected_decode_dcp
        for p in plan_hw.candidates
    )
    launched = emit_launch_args(
        plan_hw, dtypes=dt_hw_fp4, model="Qwen/Qwen3.8-Flash-Next-FP8"
    )
    assert "--hf-overrides" not in launched["cli_args"]
    assert "--hf-overrides" not in launched["prefill_cli_args"]
    assert "--kv-cache-dtype" in launched["cli_args"]
    kv_idx = launched["cli_args"].index("--kv-cache-dtype") + 1
    assert launched["cli_args"][kv_idx] == "fp4"


def test_raiden_kv_pairing_guards_and_pcp_explain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce Raiden GLM/Qwen3.5 pairing constraints and pcp_allows_pp messaging."""
    fixtures = Path(__file__).resolve().parent / "fixtures"
    glm_geom = from_hf_config(
        json.loads((fixtures / "glm_5_2_fp8.json").read_text(encoding="utf-8"))
    )
    qwen_geom = from_hf_config(
        json.loads((fixtures / "qwen3_5_397b_fp8.json").read_text(encoding="utf-8"))
    )
    qwen4_geom = from_hf_config(load_qwen3_8_bringup_config())

    assert _valid_raiden_kv_pair(
        glm_geom, Parallel(tp=16, kv_split=16), Parallel(tp=16, kv_split=16)
    )
    assert not _valid_raiden_kv_pair(
        glm_geom, Parallel(pp=8, tp=1), Parallel(tp=16, pp=2, kv_split=16)
    )
    assert not _valid_raiden_kv_pair(
        glm_geom, Parallel(pp=8, tp=1), Parallel(tp=16, pp=2)
    )

    assert _valid_raiden_kv_pair(qwen_geom, Parallel(pp=8, tp=1), Parallel(tp=8, dp=1))
    assert _valid_raiden_kv_pair(
        qwen_geom, Parallel(pp=4, pcp=2, tp=1), Parallel(tp=8, kv_split=4)
    )
    assert not _valid_raiden_kv_pair(
        qwen_geom, Parallel(pp=8, tp=1), Parallel(tp=4, dp=2, kv_split=2)
    )
    assert not _valid_raiden_kv_pair(
        qwen_geom, Parallel(pp=1, tp=1, dp=4), Parallel(tp=1, dp=4)
    )

    topos = tuple(TOPOLOGIES.values())
    qwen4_reason = explain_layout_rejection(
        qwen4_geom,
        {"prefill": {"pcp": 2, "pp": 2, "tp": 1}, "decode": {"tp": 8}},
        topos,
        SolveOptions(mode="disagg"),
    )
    assert "PP > 1, " in qwen4_reason

    pcp_pp_profile = rf_constants.RuntimeProfile(
        name="main",
        parallel=rf_constants.ParallelRules(
            pcp_allows_pp=True,
            max_pcp=2,
            raiden_pair_rule="qwen_gdn",
        ),
    )
    monkeypatch.setattr(rf_solver, "resolve_geom_profile", lambda _g: pcp_pp_profile)
    pcp_pp_reason = explain_layout_rejection(
        qwen_geom,
        {"prefill": {"pcp": 4, "pp": 2, "tp": 1}, "decode": {"tp": 8}},
        topos,
        SolveOptions(mode="disagg"),
    )
    assert "PP > 1, " not in pcp_pp_reason
    assert pcp_pp_reason == "PCP > 1 is not permitted with this profile or spec_k > 1"


def test_fast_pass_memo_matches_step_and_spec_cycle_floors() -> None:
    """Verify crossover planner memoization matches step and spec cycle floors."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    scen = TrafficScenario(
        name="memo-test",
        isl=2048,
        osl=256,
        concurrencies=(16,),
        prefix_hit_rate=0.5,
        accept_len=2.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        spec_tokens=2,
    )
    planner = _CrossoverPlanner(geom, dt, chip, scen)
    pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        tp=2,
        dp=2,
        pp=2,
        pcp=1,
        ep=4,
        kv_split=1,
        chips_per_instance=4,
        topology="2x2x1",
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=10.0,
        hbm_limit_gb=94.76,
        pp_layer_partition=(32, 28),
    )
    pid = planner.get_pool_id(pool)
    topo, par, pp_part = planner.pool_plans[pid]
    assert pp_part == (32, 28)

    opts = FloorOptions(dt=dt, topology=topo)
    shape_pre = StepShape(
        batch=0,
        rows_per_seq=0,
        context=0,
        state_writes=0,
        prefill_tokens=scen.isl,
        prefix_hit_rate=0.5,
    )
    pre_sf = planner.compute_pre_sf(pid, pool, 0.5)
    sf_pre_direct = rf.scale_pp_stage_floor(geom, chip, par, (shape_pre, opts), pp_part)
    assert pre_sf == pytest.approx(sf_pre_direct.serial_estimate_s)

    batch_res = 4
    dec_sf, draft_s = planner.get_dec_sf(pid, batch_res)
    ver_shape = StepShape(
        batch=batch_res,
        rows_per_seq=planner.spec_k + 1,
        context=planner.ctx_mid,
        state_writes=1,
    )
    sf_dec_even = spec_cycle_floor(
        geom,
        chip,
        par,
        SpecShape(batch=batch_res, context=planner.ctx_mid, k=planner.spec_k),
        opts,
    )
    sf_ver_direct = rf.scale_pp_stage_floor(geom, chip, par, (ver_shape, opts), pp_part)
    draft_mods = tuple(m for m in sf_dec_even.modules if m.name.startswith("draft_"))
    sf_dec_direct = combine_modules(
        (*sf_ver_direct.modules, *draft_mods),
        chip,
        topology=topo,
        parallel=par,
    )
    tot_tok = batch_res * (planner.spec_k + 1) * par.dp
    ep_spec = EpLoadSpec(geom=geom, par=par, global_tokens=tot_tok)
    sf_dec_direct = scale_ep_moe_floor(sf_dec_direct, chip, ep_spec, topology=topo)
    sf_ver_direct = scale_ep_moe_floor(sf_ver_direct, chip, ep_spec, topology=topo)
    exp_draft = max(
        0.0, sf_dec_direct.serial_estimate_s - sf_ver_direct.serial_estimate_s
    )
    assert dec_sf == pytest.approx(sf_dec_direct.serial_estimate_s)
    assert draft_s == pytest.approx(exp_draft)

    chunk_tok = 512
    mix_sf = planner.compute_mix_sf(pid, batch_res, chunk_tok)
    shape_mix = StepShape(
        batch=batch_res,
        rows_per_seq=planner.spec_k + 1,
        context=planner.ctx_mid,
        state_writes=1,
        prefill_tokens=chunk_tok,
        prefix_hit_rate=0.0,
    )
    sf_mix_direct = rf.scale_pp_stage_floor(geom, chip, par, (shape_mix, opts), pp_part)
    assert mix_sf == pytest.approx(sf_mix_direct.serial_estimate_s)


def test_prefix_hit_crossover_sweep_and_pcie_offload() -> None:
    """Verify crossover sweep over prefix hit rates and Raiden host PCIe offload."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    hit_90 = 0.90
    hit_95 = 0.95
    scen_offload = TrafficScenario(
        name="offload-compare",
        isl=32768,
        osl=512,
        prefix_hit_rate=hit_90,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=20.0,
        kv_offload_gb=64.0,
    )
    plan_off = solve_serving_plan(
        geom, dt, chip, scen_offload, SolveOptions(mode="auto", max_chips=8)
    )
    assert len(plan_off.candidates) > len(plan_off.frontier)
    win_range, cross_rate = cold_crossover_sweep(geom, dt, chip, plan_off)
    assert win_range is not None
    assert 0.0 <= win_range[0] <= hit_90 <= win_range[1] <= hit_95
    assert plan_off.winner_prefix_hit_range == win_range
    assert plan_off.prefix_hit_crossover_rate == cross_rate
    solve_payload = build_solve_payload(plan_off, chip=chip, dtypes=dt)
    assert "sweep_candidates" not in solve_payload
    assert solve_payload["winner_prefix_hit_range"] == win_range
    assert solve_payload["prefix_hit_crossover_rate"] == cross_rate

    emitted = emit_launch_args(plan_off, dtypes=dt, model="Qwen/Qwen3.5-397B-A17B-FP8")
    assert "sweep_candidates" not in emitted["serving_plan"]
    assert "--kv-offloading-size" not in emitted["cli_args"]


def test_prefix_hit_crossover_sweep_slo_probe_and_examples() -> None:
    """Verify crossover sweep detects TTFT SLO switches across example presets."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    probe_geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    hit_95 = 0.95
    tp_high = 8
    scen_probe = TrafficScenario(
        name="slo-crossover-probe",
        isl=16384,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8, 32),
        kv_working_set_turns=1.0,
    )
    probe_h0 = solve_serving_plan(
        probe_geom, dt, chip, scen_probe, SolveOptions(mode="mixed", max_chips=4)
    )
    assert probe_h0.recommended is not None
    tp_high_min_ttft = min(
        cand.est_ttft_ms
        for cand in probe_h0.candidates
        if cand.decode_pool.tp == tp_high
    )
    mid_ttft_slo = 0.5 * (tp_high_min_ttft + probe_h0.recommended.est_ttft_ms)
    scen_slo = dc_replace(scen_probe, ttft_slo_ms=mid_ttft_slo)
    plan_slo_h0 = solve_serving_plan(
        probe_geom, dt, chip, scen_slo, SolveOptions(mode="mixed", max_chips=4)
    )
    plan_slo_h95 = solve_serving_plan(
        probe_geom,
        dt,
        chip,
        dc_replace(scen_slo, prefix_hit_rate=hit_95),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert plan_slo_h0.recommended is not None
    assert plan_slo_h95.recommended is not None
    assert (
        plan_slo_h0.recommended.decode_pool.tp
        != plan_slo_h95.recommended.decode_pool.tp
    )
    assert plan_slo_h0.prefix_hit_crossover_rate is not None
    assert plan_slo_h0.winner_prefix_hit_range is not None
    assert plan_slo_h0.winner_prefix_hit_range[1] < hit_95

    for ex_name, n_chips in (
        ("coding-agent", 8),
        ("chat-8k1k", 16),
        ("reasoning-1k8k", 16),
    ):
        ex_scen = build_example_scenario(ex_name)
        ex_plan = solve_serving_plan(
            geom, dt, chip, ex_scen, SolveOptions(mode="auto", max_chips=n_chips)
        )
        ex_range, _ex_cross = cold_crossover_sweep(geom, dt, chip, ex_plan)
        assert ex_range is not None
        if ex_name == "chat-8k1k":
            ex_uncached = dc_replace(ex_plan, sweep_candidates=())
            fb_range, fb_cross = cold_crossover_sweep(
                geom, dt, chip, ex_uncached, repeats=1
            )
            assert (fb_range, fb_cross) == (
                ex_plan.winner_prefix_hit_range,
                ex_plan.prefix_hit_crossover_rate,
            )


def test_ep_expected_maximum_matches_quadrature() -> None:
    """Verify ep_z_unit matches exact Gaussian quadrature and max_ep_load_ratio."""
    ep4, ep8, ep16 = 4, 8, 16
    assert ep_z_unit(ep4) == pytest.approx(1.029375373003964, rel=1e-12)
    assert ep_z_unit(ep8) == pytest.approx(1.4236003060452778, rel=1e-12)
    assert ep_z_unit(ep16) == pytest.approx(1.765991393054788, rel=1e-12)
    assert max_ep_load_ratio(0, ep8) == 1.0
    assert max_ep_load_ratio(64, 1) == 1.0
    assert max_ep_load_ratio(16, ep8) > max_ep_load_ratio(4096, ep8) > 1.0


def test_sweep_matches_main_path_ttft_with_offload() -> None:
    """Verify crossover sweep matches main-path TTFT and throughput with offload."""
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes()
    scen = dc_replace(
        build_example_scenario("coding-agent"),
        kv_offload_gb=32.0,
        prefix_hit_rate=0.5,
        think_time_s=1.0,
    )
    plan_res = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=64)
    )
    ep_pts = [p for p in plan_res.candidates if p.decode_pool.ep > 1]
    assert ep_pts
    pt = ep_pts[0]
    pools, eval_context = setup_crossover_sweep(geom, dt, chip, scen, [pt])
    state = CandidateState()
    eval_grid_mix_candidates(
        scen.prefix_hit_rate, scen, pools.mix_meta, eval_context, state
    )
    entry = state.near_slo[0] if state.near_slo else state.near_all[0]
    sweep_tput = entry[0]
    meta = pools.mix_meta[0]
    h_eff = scen.prefix_hit_rate * meta.cap_ratio
    pcie_s = (
        float(scen.isl)
        * max(0.0, h_eff - scen.prefix_hit_rate * meta.hbm_ratio)
        * meta.pcie_slope
        if meta.pcie_slope > 0.0
        else 0.0
    )
    sf_pre_s = eval_context.compute_pre_sf(meta.d_pid, meta.d_pool, h_eff)
    sweep_ttft_ms = (sf_pre_s + pcie_s) * 1e3
    assert sweep_ttft_ms == pytest.approx(meta.pt.est_ttft_ms, rel=1e-3)
    assert sweep_tput == pytest.approx(meta.pt.token_throughput_per_chip, rel=0.03)


def test_crossover_ep_decode_passes_slice_topology() -> None:
    """Verify get_dec_sf preserves slice topology and scales global tokens by DP."""
    geom = rf.from_hf_config(QWEN35_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes()
    scen = dc_replace(build_example_scenario("coding-agent"), spec_tokens=3)
    planner = _CrossoverPlanner(geom, dt, chip, scen)
    pool = PoolMeshPlan(
        role="decode",
        instances=1,
        tp=4,
        dp=2,
        pp=1,
        pcp=1,
        ep=8,
        kv_split=1,
        chips_per_instance=4,
        topology="2x2x1",
        mesh_axes=(2, 2, 1, 1),
        hbm_used_gb=40.0,
        hbm_limit_gb=94.76,
    )
    pid = planner.get_pool_id(pool)
    topo, par, _ = planner.pool_plans[pid]
    batch_res = 2
    dec_sf_s, _ = planner.get_dec_sf(pid, batch_res)
    sf_raw = spec_cycle_floor(
        geom,
        chip,
        par,
        SpecShape(batch=batch_res, context=planner.ctx_mid, k=planner.spec_k),
        FloorOptions(dt=dt, topology=topo),
    )
    ep_spec = EpLoadSpec(
        geom=geom,
        par=par,
        global_tokens=batch_res * (planner.spec_k + 1) * pool.dp,
    )
    sf_expected = scale_ep_moe_floor(sf_raw, chip, ep_spec, topology=topo)
    assert dec_sf_s == pytest.approx(sf_expected.serial_estimate_s, rel=1e-12)


def test_prune_sweep_candidates_retains_low_concurrency_slo_fallback() -> None:
    """Verify sweep pruning retains lower concurrency points when SLO is constrained."""
    chip = CHIPS["tpu7x"]
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0)
    low_conc = 8
    high_conc = 16
    scen = TrafficScenario(
        name="prune-slo-fallback",
        isl=8192,
        osl=512,
        prefix_hit_rate=0.95,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(low_conc, high_conc),
        kv_working_set_turns=1.0,
        tpot_slo_ms=6.5,
    )
    plan = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=4)
    )
    assert any(p.concurrency == low_conc for p in plan.sweep_candidates)
    pruned_res = sweep_prefix_hit_crossovers(geom, dt, chip, plan)
    unpruned_res = sweep_prefix_hit_crossovers(
        geom, dt, chip, dc_replace(plan, sweep_candidates=plan.candidates)
    )
    assert pruned_res == unpruned_res


def test_ep_scaled_serial_time_includes_the_vector_term() -> None:
    """Include per-module vector time when scaling serial step latency under EP."""
    geometry = rf.from_hf_config(QWEN35_CFG)
    vector_bound_chip = dc_replace(CHIPS["tpu7x"], vector_flops=1.0e8)
    parallel = rf.Parallel(
        tp=1,
        dp=8,
        ep=8,
        moe="ep",
        moe_dispatch="ag_ep",
        ranks_per_chip=vector_bound_chip.cores_per_chip,
    )
    base_floor = step_floor(
        geometry,
        vector_bound_chip,
        parallel,
        rf.StepShape(batch=1, rows_per_seq=1, context=1024),
        rf.FloorOptions(
            dt=Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0, act="a8"),
            topology=rf.TOPOLOGIES["2x2x1"],
        ),
    )
    global_tokens = 8
    scaled_floor = scale_ep_moe_floor(
        base_floor,
        vector_bound_chip,
        EpLoadSpec(geom=geometry, par=parallel, global_tokens=global_tokens),
        topology=rf.TOPOLOGIES["2x2x1"],
    )
    scaled_serial = scaled_ep_serial_s(
        base_floor,
        vector_bound_chip,
        geometry,
        parallel,
        global_tokens,
    )
    independent_vector_floor_s = (
        sum(module.vector_flops for module in base_floor.modules)
        / vector_bound_chip.vector_flops
        + base_floor.chain_s
    )
    assert (
        independent_vector_floor_s
        > base_floor.hbm_s + base_floor.matmul_s + base_floor.chain_s
    )
    assert scaled_serial >= independent_vector_floor_s
    assert scaled_serial >= base_floor.serial_estimate_s
    assert scaled_serial == pytest.approx(scaled_floor.serial_estimate_s, rel=1e-12)


def test_crossover_dcn_key_preserves_distinct_chip_counts() -> None:
    """Keep distinct chip counts separate in disaggregated sweep candidate pruning."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = rf.dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="crossover-dcn-key",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=1.0,
    )
    pre_pool = PoolMeshPlan(
        role="prefill",
        instances=1,
        topology="2x2x4",
        chips_per_instance=16,
        tp=8,
        dp=4,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 4),
        hbm_used_gb=20.0,
        hbm_limit_gb=94.0,
        moe_dispatch="ag_ep",
    )
    dec_4 = PoolMeshPlan(
        role="decode",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=4,
        dp=2,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=20.0,
        hbm_limit_gb=94.0,
        moe_dispatch="ag_ep",
    )
    dec_8 = PoolMeshPlan(
        role="decode",
        instances=1,
        topology="2x2x2",
        chips_per_instance=8,
        tp=8,
        dp=2,
        ep=16,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 2),
        hbm_used_gb=20.0,
        hbm_limit_gb=94.0,
        moe_dispatch="ag_ep",
    )
    pt_small = rf.ParetoPoint(
        mode="disagg",
        concurrency=32,
        total_chips=20,
        prefill_pool=pre_pool,
        decode_pool=dec_4,
        est_ttft_ms=50.0,
        est_tpot_ms=4.0,
        est_interactivity_tok_s=250.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=900.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
    )
    pt_large = rf.ParetoPoint(
        mode="disagg",
        concurrency=32,
        total_chips=24,
        prefill_pool=pre_pool,
        decode_pool=dec_8,
        est_ttft_ms=50.0,
        est_tpot_ms=4.0,
        est_interactivity_tok_s=250.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=900.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
    )
    kept = rf_crossover.prune_sweep_candidates(
        geom, dt, chip, scen, [pt_small, pt_large]
    )
    expected_chips = {20, 24}
    assert {pt.total_chips for pt in kept} == expected_chips


def test_crossover_disagg_sweep_uses_prefill_pool_batch_and_isl() -> None:
    """Disagg crossover KV ratios use prefill pool, prefill batch, and ISL only."""
    cfg = json.loads(
        (
            Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
        ).read_text(encoding="utf-8")
    )
    geom = rf.from_hf_config(cfg)
    dt = rf.dtypes_from_hf_config(cfg)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="disagg-prefill-kv",
        isl=8192,
        osl=8192,
        prefix_hit_rate=0.85,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=12.0,
        kv_offload_gb=64.0,
    )
    p_pool = PoolMeshPlan(
        role="prefill",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=8,
        dp=1,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=60.0,
        hbm_limit_gb=78.0,
        moe_dispatch="ag_ep",
        hbm_weights_gb=68.0,
        hbm_kv_gb=2.0,
        hbm_state_gb=0.2,
        kv_offload_gb=64.0,
    )
    d_pool = PoolMeshPlan(
        role="decode",
        instances=4,
        topology="2x2x2",
        chips_per_instance=8,
        tp=8,
        dp=2,
        ep=16,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 2),
        hbm_used_gb=25.0,
        hbm_limit_gb=94.0,
        moe_dispatch="ag_ep",
        hbm_weights_gb=20.0,
        hbm_kv_gb=4.0,
        hbm_state_gb=1.0,
        max_num_seqs=4,
        kv_offload_gb=0.0,
    )
    pt = ParetoPoint(
        mode="disagg",
        concurrency=32,
        total_chips=36,
        prefill_pool=p_pool,
        decode_pool=d_pool,
        est_ttft_ms=120.0,
        est_tpot_ms=6.0,
        est_interactivity_tok_s=166.7,
        output_throughput_per_chip=50.0,
        token_throughput_per_chip=100.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
        kv_cache_capacity_tokens=1_500_000,
    )
    pools, _ctx = setup_crossover_sweep(geom, dt, chip, scen, [pt])
    ((_p_pid, cap_ratio), (_grp_pool, _bub, items)) = next(
        iter(pools.dis_ttft_groups.items())
    )
    assert len(items) == 1
    item = items[0]
    p_par = Parallel(tp=8, pp=1, pcp=1, kv_split=1, ranks_per_chip=chip.cores_per_chip)
    b_kv_p, fixed_p = rf.memory.kv_bytes_per_core_parts(
        geom, dt, p_par, scen.isl, chip=chip
    )
    w_tok_p = scen.kv_working_set_turns * float(scen.concurrencies[0] * scen.isl)
    expected_cap = min(1.0, float(pt.kv_cache_capacity_tokens or 0) / w_tok_p)
    avail_p = max(
        0.0,
        (p_pool.hbm_limit_gb - p_pool.hbm_weights_gb - p_pool.hbm_state_gb)
        * float(UNIT_GIGA)
        - fixed_p,
    )
    expected_hbm = min(1.0, (avail_p / b_kv_p) / w_tok_p)
    assert cap_ratio == pytest.approx(expected_cap, rel=1e-9)
    assert item.hbm_ratio == pytest.approx(expected_hbm, rel=1e-9)
    half_ratio = 0.5
    assert item.hbm_ratio < half_ratio


def test_crossover_grid_disagg_includes_pcie_and_dcn_cap() -> None:
    """Disagg grid evaluator folds PCIe offload time into rate_p and caps by DCN."""
    cfg = json.loads(
        (
            Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
        ).read_text(encoding="utf-8")
    )
    geom = rf.from_hf_config(cfg)
    dt = rf.dtypes_from_hf_config(cfg)
    slow_chip = dc_replace(CHIPS["tpu7x"], dcn_bw=0.5e9, pcie_bw=2.0e9)
    scen = TrafficScenario(
        name="disagg-dcn-pcie-grid",
        isl=8192,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(32,),
        kv_working_set_turns=15.0,
        kv_offload_gb=64.0,
    )
    plan_slow = solve_serving_plan(
        geom, dt, slow_chip, scen, SolveOptions(mode="disagg", max_chips=16)
    )
    assert plan_slow.recommended is not None
    pt = plan_slow.recommended
    pools, ctx = setup_crossover_sweep(geom, dt, slow_chip, scen, [pt])
    state = CandidateState()
    _eval_grid_disagg_ttft(
        scen.prefix_hit_rate, scen, pools.dis_ttft_groups, ctx, state
    )
    entry = state.near_slo[0] if state.near_slo else state.near_all[0]
    grid_tput = entry[0]

    fast_dcn_chip = dc_replace(slow_chip, dcn_bw=100.0e9)
    pools_fast, ctx_fast = setup_crossover_sweep(geom, dt, fast_dcn_chip, scen, [pt])
    state_fast = CandidateState()
    _eval_grid_disagg_ttft(
        scen.prefix_hit_rate, scen, pools_fast.dis_ttft_groups, ctx_fast, state_fast
    )
    entry_fast = (
        state_fast.near_slo[0] if state_fast.near_slo else state_fast.near_all[0]
    )
    assert grid_tput < entry_fast[0]
    assert grid_tput == pytest.approx(pt.token_throughput_per_chip, rel=0.05)


def test_crossover_kv_slope_separates_fixed_swa_window() -> None:
    """Crossover KV ratios separate fixed SWA window bytes from linear KV slope."""
    swa_window = 512
    long_isl = 8192
    swa_geom = _from_hf_config_dry_run(
        {
            "model_type": "swa_synth_test",
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "layer_types": ["sliding_attention", "full_attention"] * 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "sliding_window": swa_window,
            "intermediate_size": 4096,
            "vocab_size": 32000,
        }
    )
    dt = Dtypes(w_dense=2.0, kv=1.0)
    chip = CHIPS["tpu7x"]
    par = Parallel(tp=4, pp=1, ranks_per_chip=chip.cores_per_chip)
    pool = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=4,
        dp=1,
        ep=1,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="allreduce",
        hbm_weights_gb=10.0,
        hbm_state_gb=0.0,
        max_num_seqs=16,
    )
    scen = TrafficScenario(
        name="swa-slope",
        isl=long_isl,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=4.0,
    )
    planner = _CrossoverPlanner(swa_geom, dt, chip, scen)
    ctx_full = scen.isl + scen.osl
    lin_b, fix_b = planner.get_b_kv_parts(pool, ctx_full)
    exp_lin, exp_fix = rf.memory.kv_bytes_per_core_parts(
        swa_geom, dt, par, ctx_full, chip=chip
    )
    assert lin_b == pytest.approx(exp_lin, rel=1e-12)
    assert fix_b == pytest.approx(exp_fix, rel=1e-12)
    assert fix_b > 0.0
    # Naive footprint(1, 1) - footprint(1, 0) folds SWA layers into linear slope (2x).
    naive_slope = rf.hbm_footprint_per_core_bytes(
        swa_geom, dt, par, rf.SeqKvLayout(1, 1), chip=chip
    ) - rf.hbm_footprint_per_core_bytes(
        swa_geom, dt, par, rf.SeqKvLayout(1, 0), chip=chip
    )
    assert naive_slope > lin_b * 1.5


def test_crossover_mixed_mkey_preserves_distinct_offload_ratios() -> None:
    """Mixed prune mkey preserves distinct HBM spill ratios when cap_ratio ties."""
    geom = rf.from_hf_config(QWEN35_CFG)
    dt = Dtypes(kv=1.0, state=2.0)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="mkey-offload",
        isl=16384,
        osl=512,
        prefix_hit_rate=0.9,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=20.0,
        kv_offload_gb=64.0,
    )
    pool_light = PoolMeshPlan(
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
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=45.0,
        hbm_limit_gb=80.0,
        moe_dispatch="ag_ep",
        hbm_weights_gb=35.0,
        hbm_state_gb=1.0,
        max_num_seqs=16,
        kv_offload_gb=64.0,
    )
    pool_heavy = dc_replace(pool_light, hbm_weights_gb=68.0, hbm_used_gb=72.0)
    pt_a = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=4,
        prefill_pool=None,
        decode_pool=pool_light,
        est_ttft_ms=100.0,
        est_tpot_ms=8.0,
        est_interactivity_tok_s=125.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=800.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
        kv_cache_capacity_tokens=50_000_000,
    )
    pt_b = dc_replace(pt_a, decode_pool=pool_heavy, token_throughput_per_chip=750.0)
    kept = rf_crossover.prune_sweep_candidates(geom, dt, chip, scen, [pt_a, pt_b])
    expected_kept = 2
    assert len(kept) == expected_kept


def test_crossover_winner_range_returns_none_when_idx0_differs() -> None:
    """Return None for winner_prefix_hit_range when grid winner at h0 differs."""
    scen = TrafficScenario(
        name="winner-mismatch",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    pool_a = PoolMeshPlan(
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
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=40.0,
        hbm_limit_gb=80.0,
        moe_dispatch="ag_ep",
    )
    pool_b = dc_replace(pool_a, tp=4, dp=2)
    pt_rec = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=4,
        prefill_pool=None,
        decode_pool=pool_a,
        est_ttft_ms=100.0,
        est_tpot_ms=8.0,
        est_interactivity_tok_s=125.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=800.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
    )
    pt_other = dc_replace(pt_rec, decode_pool=pool_b)
    plan_obj = ServingPlan(
        model_layers=8,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=pt_rec,
        frontier=(pt_rec, pt_other),
        warnings=(),
    )
    sig_other = (_pool_sig(pool_b), None, 4, 800.0, pt_other)
    winners = [(0.0, sig_other), (0.5, sig_other), (0.95, sig_other)]
    win_range, cross = _derive_winner_range_and_crossover(scen, plan_obj, winners)
    assert win_range is None
    assert cross is None


def test_crossover_mix_sf_matches_solver_and_ep_scales_by_dp() -> None:
    """compute_mix_sf matches step_floor directly and get_dec_sf scales EP by DP."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0, state=2.0)
    scen = TrafficScenario(
        name="ep-dp-mix-match",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    planner = _CrossoverPlanner(geom, dt, chip, scen)
    pool_dp2 = PoolMeshPlan(
        role="mixed",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=4,
        dp=2,
        ep=8,
        pp=1,
        pcp=1,
        kv_split=1,
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=40.0,
        hbm_limit_gb=94.76,
        moe_dispatch="ag_ep",
    )
    pid = planner.get_pool_id(pool_dp2)
    topo, par, _ = planner.pool_plans[pid]
    opts = FloorOptions(dt=dt, topology=topo)
    b_res = 2
    chunk_tok = 512

    mix_sf = planner.compute_mix_sf(pid, b_res, chunk_tok)
    sf_mix_direct = step_floor(
        geom,
        chip,
        par,
        StepShape(
            batch=b_res,
            rows_per_seq=1,
            context=planner.ctx_mid,
            state_writes=1,
            prefill_tokens=chunk_tok,
            prefix_hit_rate=0.0,
        ),
        opts,
    )
    assert mix_sf == pytest.approx(sf_mix_direct.serial_estimate_s, rel=1e-12)

    dec_sf, _ = planner.get_dec_sf(pid, b_res)
    sf_dec_direct = step_floor(
        geom,
        chip,
        par,
        StepShape(batch=b_res, rows_per_seq=1, context=planner.ctx_mid),
        opts,
    )
    exp_with_dp = scaled_ep_serial_s(
        sf_dec_direct, chip, geom, par, b_res * pool_dp2.dp
    )
    exp_without_dp = scaled_ep_serial_s(sf_dec_direct, chip, geom, par, b_res)
    assert dec_sf == pytest.approx(exp_with_dp, rel=1e-12)
    assert dec_sf < exp_without_dp


def test_crossover_mixed_x_capacity_scales_by_instances() -> None:
    """Mixed crossover closed-loop capacity scales linearly with pool instances."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = rf.dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="mixed-instances",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(64,),
        kv_working_set_turns=1.0,
    )
    pool_1 = PoolMeshPlan(
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
        mesh_axes=(2, 2, 2, 1),
        hbm_used_gb=20.0,
        hbm_limit_gb=94.0,
        moe_dispatch="ag_ep",
        max_num_seqs=16,
    )
    pool_2 = dc_replace(pool_1, instances=2)
    pt_1 = ParetoPoint(
        mode="mixed",
        concurrency=32,
        total_chips=4,
        prefill_pool=None,
        decode_pool=pool_1,
        est_ttft_ms=50.0,
        est_tpot_ms=4.0,
        est_interactivity_tok_s=250.0,
        output_throughput_per_chip=100.0,
        token_throughput_per_chip=500.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
        slo_verdict="estimate_meets",
    )
    pt_2 = dc_replace(pt_1, concurrency=64, total_chips=8, decode_pool=pool_2)
    pools_1, ctx_1 = setup_crossover_sweep(geom, dt, chip, scen, [pt_1])
    pools_2, ctx_2 = setup_crossover_sweep(geom, dt, chip, scen, [pt_2])
    state_1 = CandidateState()
    state_2 = CandidateState()
    eval_grid_mix_candidates(
        scen.prefix_hit_rate, scen, pools_1.mix_meta, ctx_1, state_1
    )
    eval_grid_mix_candidates(
        scen.prefix_hit_rate, scen, pools_2.mix_meta, ctx_2, state_2
    )
    tput_1 = state_1.near_slo[0][0]
    tput_2 = state_2.near_slo[0][0]
    assert pools_2.mix_meta[0].res_cap == pytest.approx(
        2.0 * pools_1.mix_meta[0].res_cap, rel=1e-12
    )
    assert tput_2 == pytest.approx(tput_1, rel=1e-6)


def test_scale_ep_moe_floor_preserves_collective_latency() -> None:
    """Verify scale_ep_moe_floor preserves collective_latency_s in chain_s."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = rf.dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    topo = rf.TOPOLOGIES["2x2x1"]
    par = Parallel(tp=4, dp=2, ep=8, moe="ep", moe_dispatch="ag_ep")
    coll_lat_s = 1e-4
    sf_base = step_floor(
        geom,
        chip,
        par,
        StepShape(batch=2, rows_per_seq=1, context=1024),
        FloorOptions(dt=dt, topology=topo, collective_latency_s=coll_lat_s),
    )
    sf_scaled = scale_ep_moe_floor(
        sf_base,
        chip,
        EpLoadSpec(geom=geom, par=par, global_tokens=4),
        topology=topo,
        collective_latency_s=coll_lat_s,
    )
    assert sf_scaled.chain_s == pytest.approx(sf_base.chain_s, rel=1e-12)


def _load_qwen3_5_397b_fp8() -> tuple[rf.ModelGeometry, rf.Dtypes, rf.ChipSpec]:
    """Load the checked-in Qwen3.5 397B FP8 geometry, dtypes, and v7x chip spec."""
    cfg_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return from_hf_config(cfg), dtypes_from_hf_config(cfg), CHIPS["tpu7x"]


def _mixed_key(pt: ParetoPoint) -> tuple[object, ...]:
    """Build a structural candidate key for comparing mixed-mode solver plans."""
    pool = pt.decode_pool
    return (
        pool.topology,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.moe_dispatch,
    )


def _assert_unknown_think_l1_and_report(
    model_ctx: tuple[rf.ModelGeometry, rf.Dtypes, rf.ChipSpec],
    scenarios: tuple[TrafficScenario, TrafficScenario, TrafficScenario],
    plan_none: ServingPlan,
    tmp_path: Path,
) -> None:
    """Assert sound X/throughput bounds, HTML rendering, and sweep."""
    geom, dt, chip = model_ctx
    scen_none, scen_zero, scen_15 = scenarios
    tp_deg, dp_deg, ep_deg = 4, 2, 8
    rescued_par = Parallel(
        tp=tp_deg, dp=dp_deg, ep=ep_deg, moe="ep", moe_dispatch="a2a_push"
    )
    l1_opts = {
        "decode_par": rescued_par,
        "decode_topo": TOPOLOGIES["2x2x1"],
        "concurrency": 256,
    }
    bound_in, b_cap_single = _build_l1_decode_input(geom, dt, chip, scen_none, l1_opts)
    tarski_b, _ = _run_isolated_tarski_loop(bound_in, 256, b_cap_single, dp_deg, 1e6)
    assert tarski_b == pytest.approx(1.0)

    tput_none = floor_throughput_per_chip(geom, dt, chip, scen_none, **l1_opts)
    tput_zero = floor_throughput_per_chip(geom, dt, chip, scen_zero, **l1_opts)
    tput_15 = floor_throughput_per_chip(geom, dt, chip, scen_15, **l1_opts)
    assert tput_none == pytest.approx(tput_zero, rel=1e-12)
    assert tput_none >= tput_15

    xb_none = x_bound(geom, dt, chip, scen_none, **l1_opts)
    xb_zero = x_bound(geom, dt, chip, scen_zero, **l1_opts)
    xb_15 = x_bound(geom, dt, chip, scen_15, **l1_opts)
    assert xb_none == pytest.approx(xb_zero, rel=1e-12)
    assert xb_none >= xb_15

    payload = build_solve_payload(
        plan_none, chip=chip, dtypes=dt, model="Qwen/Qwen3.5-397B-FP8"
    )
    html = render_html(payload)
    assert "Qwen3.5" in html
    out_html = tmp_path / "unknown_think.html"
    out_html.write_text(html, encoding="utf-8")
    assert out_html.is_file()
    hit_range, crossover_hit = sweep_prefix_hit_crossovers(geom, dt, chip, plan_none)
    range_bounds_len = 2
    assert hit_range is None or len(hit_range) == range_bounds_len
    assert crossover_hit is None or 0.0 <= crossover_hit <= 1.0


def test_unknown_think_time_skips_capacity_and_tarski_false_prune_with_sound_x_bound(
    tmp_path: Path,
) -> None:
    """Verify None think time avoids false SLO pruning while keeping sound X bounds."""
    geom, dt, chip = _load_qwen3_5_397b_fp8()
    scen_none = TrafficScenario(
        "chat-unknown",
        4096,
        512,
        0.0,
        1.0,
        None,
        (256,),
        1.0,
        ttft_slo_ms=600.0,
        tpot_slo_ms=15.0,
    )
    scen_zero = dc_replace(scen_none, name="chat-zero", think_time_s=0.0)
    scen_15 = dc_replace(scen_none, name="chat-15s", think_time_s=15.0)
    scen_open = dc_replace(
        scen_none, name="chat-open", ttft_slo_ms=None, tpot_slo_ms=None
    )

    with pytest.raises(ValueError, match="think_time_s"):
        TrafficScenario("neg", 1024, 128, 0.0, 1.0, -0.1, (1,), 1.0)

    opts = SolveOptions(mode="mixed", max_chips=16, max_instances=1)
    plan_none = solve_serving_plan(geom, dt, chip, scen_none, opts)
    plan_zero = solve_serving_plan(geom, dt, chip, scen_zero, opts)
    plan_15 = solve_serving_plan(geom, dt, chip, scen_15, opts)
    plan_open = solve_serving_plan(geom, dt, chip, scen_open, opts)

    unknown_warn = rf.UNKNOWN_THINK_TIME_WARNING
    assert unknown_warn in plan_none.warnings
    assert ZERO_THINK_TIME_WARNING not in plan_none.warnings
    assert unknown_warn not in plan_open.warnings

    assert (
        _apply_capacity_ceiling_verdict(
            scen_none,
            SLO_VERDICT_UNDECIDED,
            concurrency=256,
            pool_capacity_req_s=1e-6,
        )
        == SLO_VERDICT_UNDECIDED
    )
    assert (
        _apply_capacity_ceiling_verdict(
            scen_zero,
            SLO_VERDICT_UNDECIDED,
            concurrency=256,
            pool_capacity_req_s=1e-6,
        )
        == SLO_VERDICT_RULED_OUT
    )

    target_key = ("2x2x1", 4, 2, 8, 1, 1, 1, "a2a_push")
    by_zero = {_mixed_key(pt): pt for pt in plan_zero.candidates}
    by_15 = {_mixed_key(pt): pt for pt in plan_15.candidates}
    by_none = {_mixed_key(pt): pt for pt in plan_none.candidates}
    assert target_key not in by_zero
    assert target_key in by_15
    assert target_key in by_none
    assert by_none[target_key].min_decode_seqs == pytest.approx(1.0)
    _assert_unknown_think_l1_and_report(
        (geom, dt, chip),
        (scen_none, scen_zero, scen_15),
        plan_none,
        tmp_path,
    )


def _find_2x2x1_tp8(candidates: tuple[ParetoPoint, ...], instances: int) -> ParetoPoint:
    """Select the 2x2x1 TP8 EP8 DCP1 mixed point for one instance count."""
    tp_deg, ep_deg, unit_axis = 8, 8, 1
    matches = [
        pt
        for pt in candidates
        if pt.decode_pool.topology == "2x2x1"
        and pt.decode_pool.instances == instances
        and pt.decode_pool.tp == tp_deg
        and pt.decode_pool.dp == unit_axis
        and pt.decode_pool.ep == ep_deg
        and pt.decode_pool.pp == unit_axis
        and pt.decode_pool.pcp == unit_axis
        and pt.decode_pool.kv_split == unit_axis
    ]
    assert matches, f"missing {instances}x 2x2x1 TP8 candidate"
    return max(matches, key=lambda pt: pt.token_throughput_per_chip)


def _assert_multi_instance_cli_and_disagg(
    model_ctx: tuple[rf.ModelGeometry, rf.Dtypes, rf.ChipSpec],
    scen_32: TrafficScenario,
    plan_multi: ServingPlan,
    pts_by_inst: tuple[ParetoPoint, ParetoPoint],
) -> None:
    """Assert multi-instance CLI formatting, layout parsing, attach, and disagg."""
    geom, dt, chip = model_ctx
    pt_2x, pt_4x = pts_by_inst
    inst_2, inst_4, tp_deg, ep_deg, pp_deg = 2, 4, 8, 8, 8
    chips_4x, single_inst = 16, 1

    cmds_2x = point_launch_args(pt_2x, LaunchOptions(), model="Qwen/Qwen3.5-397B-FP8")[
        "commands"
    ]
    cmds_4x = point_launch_args(pt_4x, LaunchOptions(), model="Qwen/Qwen3.5-397B-FP8")[
        "commands"
    ]
    assert any("2x " in cmd for cmd in cmds_2x)
    assert any("4x " in cmd for cmd in cmds_4x)

    parsed_4x = parse_layout("4x TP8 x EP8")
    assert parsed_4x["decode"]["instances"] == inst_4
    assert parsed_4x["decode"]["tp"] == tp_deg
    assert parsed_4x["decode"]["ep"] == ep_deg
    parsed_2x_topo = parse_layout("2x 2x2x1 TP8")
    assert parsed_2x_topo["decode"]["instances"] == inst_2
    assert parsed_2x_topo["decode"]["tp"] == tp_deg

    top_pts = top_layouts(plan_multi, 5)
    ref = reference_layout_report(plan_multi, top_pts, "4x TP8 x EP8", "outside")
    assert ref["found"] is True
    assert ref["layout"] == "4x TP8 x EP8"
    assert isinstance(ref["point"], dict)
    assert ref["point"]["total_chips"] == chips_4x
    assert ref["point"]["decode_pool"]["instances"] == inst_4

    payload = build_solve_payload(
        plan_multi, chip=chip, dtypes=dt, model="Qwen/Qwen3.5-397B-FP8"
    )
    attached = attach_reference_report(
        payload,
        plan_multi,
        ref,
        launch_fn=lambda pt: point_launch_args(
            pt, LaunchOptions(), model="Qwen/Qwen3.5-397B-FP8"
        ),
    )
    ref_out = attached["reference"]
    assert ref_out["layout"] == "4x TP8 x EP8"
    assert ref_out["launch"] is not None
    assert any("4x " in cmd for cmd in ref_out["launch"]["commands"])

    parsed_disagg = parse_layout("P: PP8 / D: TP8 x EP8")
    assert "instances" not in parsed_disagg["prefill"]
    assert "instances" not in parsed_disagg["decode"]

    plan_disagg = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_32,
        SolveOptions(mode="disagg", max_chips=64, max_instances=4),
    )
    disagg_multi = [
        pt
        for pt in plan_disagg.candidates
        if pt.prefill_pool is not None
        and pt.prefill_pool.pp == pp_deg
        and pt.decode_pool.tp == tp_deg
        and pt.decode_pool.ep == ep_deg
        and (
            pt.prefill_pool.instances > single_inst
            or pt.decode_pool.instances > single_inst
        )
    ]
    assert disagg_multi
    ref_disagg = reference_layout_report(
        plan_disagg,
        top_layouts(plan_disagg, 5),
        "P: PP8 / D: TP8 x EP8",
        "outside",
    )
    assert ref_disagg["found"] is True
    assert isinstance(ref_disagg["point"], dict)
    assert (
        ref_disagg["point"]["prefill_pool"]["instances"] > single_inst
        or ref_disagg["point"]["decode_pool"]["instances"] > single_inst
    )


def test_multi_instance_mixed_enumerates_and_scales_throughput_and_cli() -> None:
    """Verify multi-instance mixed replication enumerates, scales, and emits CLI."""
    geom, dt, chip = _load_qwen3_5_397b_fp8()
    base = build_example_scenario("chat-8k1k")
    scen_32 = dc_replace(base, think_time_s=0.0, concurrencies=(32,))
    scen_16 = dc_replace(base, think_time_s=0.0, concurrencies=(16,))
    scen_8 = dc_replace(base, think_time_s=0.0, concurrencies=(8,))

    plan_multi = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_32,
        SolveOptions(mode="mixed", max_chips=16, max_instances=4),
    )
    plan_c16 = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_16,
        SolveOptions(mode="mixed", max_chips=16, max_instances=1),
    )
    plan_c8 = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_8,
        SolveOptions(mode="mixed", max_chips=16, max_instances=1),
    )

    pt_1x = _find_2x2x1_tp8(plan_multi.candidates, 1)
    pt_2x = _find_2x2x1_tp8(plan_multi.candidates, 2)
    pt_4x = _find_2x2x1_tp8(plan_multi.candidates, 4)
    chips_1x, chips_2x, chips_4x, expected_distinct = 4, 8, 16, 3
    assert pt_1x.total_chips == chips_1x
    assert pt_2x.total_chips == chips_2x
    assert pt_4x.total_chips == chips_4x

    pool_keys = {
        _pool_key(pt_1x.decode_pool),
        _pool_key(pt_2x.decode_pool),
        _pool_key(pt_4x.decode_pool),
    }
    pool_sigs = {
        _pool_sig(pt_1x.decode_pool),
        _pool_sig(pt_2x.decode_pool),
        _pool_sig(pt_4x.decode_pool),
    }
    assert len(pool_keys) == expected_distinct
    assert len(pool_sigs) == expected_distinct

    ref_c16 = _find_2x2x1_tp8(plan_c16.candidates, 1)
    ref_c8 = _find_2x2x1_tp8(plan_c8.candidates, 1)
    for lhs, rhs in ((pt_2x, ref_c16), (pt_4x, ref_c8)):
        assert lhs.token_throughput_per_chip == pytest.approx(
            rhs.token_throughput_per_chip, rel=1e-6
        )
        assert lhs.output_throughput_per_chip == pytest.approx(
            rhs.output_throughput_per_chip, rel=1e-6
        )
        assert lhs.floor_throughput_per_chip == pytest.approx(
            rhs.floor_throughput_per_chip, rel=1e-6
        )

    tp8_par = Parallel(
        tp=8,
        dp=1,
        ep=8,
        moe="ep",
        moe_dispatch=pt_4x.decode_pool.moe_dispatch,
    )
    l1_tput_4x = floor_throughput_per_chip(
        geom,
        dt,
        chip,
        scen_32,
        decode_par=tp8_par,
        decode_topo=TOPOLOGIES["2x2x1"],
        concurrency=32,
        y_d=4,
    )
    l1_tput_1x_c8 = floor_throughput_per_chip(
        geom,
        dt,
        chip,
        scen_8,
        decode_par=tp8_par,
        decode_topo=TOPOLOGIES["2x2x1"],
        concurrency=8,
        y_d=1,
    )
    assert l1_tput_4x == pytest.approx(l1_tput_1x_c8, rel=1e-9)
    _assert_multi_instance_cli_and_disagg(
        (geom, dt, chip), scen_32, plan_multi, (pt_2x, pt_4x)
    )


def test_reference_layout_report_diagnoses_ruled_out_and_family_skyline() -> None:
    """Verify reference report surfaces ruled-out SLO bounds and skyline peers."""
    geom, dt, chip = _load_qwen3_5_397b_fp8()
    tpot_slo_limit = 1.0
    scen_tight = TrafficScenario(
        "tight-dual-slo",
        4096,
        512,
        0.0,
        1.0,
        0.0,
        (32,),
        1.0,
        ttft_slo_ms=500.0,
        tpot_slo_ms=tpot_slo_limit,
    )
    plan_tight = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_tight,
        SolveOptions(mode="mixed", max_chips=16, max_instances=1),
    )
    assert plan_tight.candidates
    assert all(pt.slo_verdict != SLO_VERDICT_RULED_OUT for pt in plan_tight.candidates)
    assert plan_tight.ruled_out_candidates
    assert all(
        pt.slo_verdict == SLO_VERDICT_RULED_OUT
        for pt in plan_tight.ruled_out_candidates
    )

    ref_ruled = reference_layout_report(
        plan_tight,
        top_layouts(plan_tight, 5),
        "TP2 x DP4 x EP8",
        "outside Pareto frontier",
    )
    assert ref_ruled["found"] is False
    assert ref_ruled["ruled_out"] is True
    assert ref_ruled["slo_verdict"] == SLO_VERDICT_RULED_OUT
    assert isinstance(ref_ruled["bound_tpot_ms"], float)
    assert ref_ruled["bound_tpot_ms"] > tpot_slo_limit
    assert isinstance(ref_ruled["bound_ttft_ms"], float)
    assert ref_ruled["bound_ttft_ms"] > 0.0
    assert isinstance(ref_ruled["hbm_used_gb"], float)
    assert ref_ruled["hbm_used_gb"] > 0.0
    reason = str(ref_ruled["rejection_reason"])
    assert reason == str(ref_ruled["rejected_by"])
    assert reason
    assert "outside Pareto frontier" not in reason
    assert "TPOT" in reason
    assert "1.0" in reason

    scen_open = dc_replace(
        build_example_scenario("chat-8k1k"),
        think_time_s=0.0,
        concurrencies=(32,),
    )
    plan_open = solve_serving_plan(
        geom,
        dt,
        chip,
        scen_open,
        SolveOptions(mode="mixed", max_chips=16, max_instances=1),
    )
    sky = pareto_skyline_by_concurrency(plan_open, k=5)[32]
    tp_deg, ep_deg, dcp_sky, dcp_base = 16, 16, 8, 1
    assert any(
        pt.decode_pool.tp == tp_deg
        and pt.decode_pool.ep == ep_deg
        and pt.decode_pool.kv_split == dcp_sky
        for pt in sky
    )
    assert not any(
        pt.decode_pool.tp == tp_deg
        and pt.decode_pool.ep == ep_deg
        and pt.decode_pool.kv_split == dcp_base
        for pt in sky
    )

    ref_fam = reference_layout_report(
        plan_open,
        top_layouts(plan_open, 5),
        "TP16 x EP16",
        "outside",
    )
    assert ref_fam["found"] is True
    assert ref_fam["ruled_out"] is False
    assert ref_fam["family_in_skyline_k"] is True
    assert "DCP8" in str(ref_fam["skyline_family_peer"])


def test_reference_layout_report_flags_all_ruled_out_matching_candidate() -> None:
    """Flag ruled_out=True when all matching candidates miss SLO."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = rf.dtypes_from_hf_config(SMALL_HYBRID_MOE_CFG)
    chip = CHIPS["tpu7x"]
    scen = dc_replace(
        build_example_scenario("chat-8k1k"),
        ttft_slo_ms=0.01,
        tpot_slo_ms=0.01,
        concurrencies=(8,),
    )
    plan = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=4)
    )
    assert plan.recommended is None
    assert plan.candidates
    assert all(pt.slo_verdict == SLO_VERDICT_RULED_OUT for pt in plan.candidates)

    cand = plan.candidates[0]
    ref = reference_layout_report(
        plan,
        top_layouts(plan, 5),
        format_parallelism(asdict(cand.decode_pool)),
        "not enumerated",
    )
    assert ref["found"] is True
    assert ref["ruled_out"] is True
    assert ref["slo_verdict"] == SLO_VERDICT_RULED_OUT
    assert ref["rejected_by"]
    assert ref["rejection_reason"] == ref["rejected_by"]
    assert "exceeds" in str(ref["rejected_by"])


def test_crossover_uniform_pp_stage_floor_matches_solver_ttft() -> None:
    """Resolve uniform PP partitions in crossover planner matching solver floor."""
    fixtures = Path(__file__).resolve().parent / "fixtures"
    cfg = json.loads((fixtures / "qwen3_5_397b_fp8.json").read_text(encoding="utf-8"))
    geom = from_hf_config(cfg)
    dt = dtypes_from_hf_config(cfg)
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="q35-pp4-pcp2",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8,),
        kv_working_set_turns=1.0,
    )
    plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(
            mode="disagg",
            prefill_topology="2x2x1",
            decode_topology="2x2x1",
            max_chips=16,
        ),
    )
    expected_pp = 4
    expected_pcp = 2
    pp4_pcp2_pts = [
        pt
        for pt in (*plan.candidates, *plan.ruled_out_candidates)
        if pt.prefill_pool is not None
        and pt.prefill_pool.pp == expected_pp
        and pt.prefill_pool.pcp == expected_pcp
        and pt.prefill_pool.pp_layer_partition is None
    ]
    assert pp4_pcp2_pts
    pt = pp4_pcp2_pts[0]
    assert pt.prefill_pool is not None

    planner = _CrossoverPlanner(geom, dt, chip, scen)
    p_pid = planner.get_pool_id(pt.prefill_pool)
    topo, par, pp_part = planner.pool_plans[p_pid]
    assert pp_part == (15, 15, 15, 15)

    shape = StepShape(
        batch=0,
        rows_per_seq=0,
        context=0,
        state_writes=0,
        prefill_tokens=scen.isl,
        prefix_hit_rate=scen.prefix_hit_rate,
    )
    opts = FloorOptions(dt=dt, topology=topo)
    stage_sf = methodcaller("_eval_stage_or_step_floor", par, shape, opts, pp_part)(
        planner
    )
    solver_stage_sf = scale_pp_stage_floor(
        geom, chip, par, (shape, opts), (15, 15, 15, 15)
    )
    assert stage_sf.hard_floor_s == pytest.approx(
        solver_stage_sf.hard_floor_s, rel=1e-12
    )
    ms_per_s = 1000.0
    assert stage_sf.hard_floor_s * ms_per_s <= pt.bound_ttft_ms + 1e-9
    assert planner.compute_pre_sf(
        p_pid, pt.prefill_pool, scen.prefix_hit_rate
    ) == pytest.approx(solver_stage_sf.serial_estimate_s, rel=1e-12)

    pools, eval_ctx = setup_crossover_sweep(geom, dt, chip, scen, [pt])
    state = CandidateState()
    _eval_grid_disagg_ttft(
        scen.prefix_hit_rate, scen, pools.dis_ttft_groups, eval_ctx, state
    )
    chosen = state.near_slo[0] if state.near_slo else state.near_all[0]
    assert chosen[0] == pytest.approx(pt.token_throughput_per_chip, rel=1e-6)


def test_explain_mixed_instance_budget_guards() -> None:
    """Apply mixed instance budget checks only when no prefill pool is specified."""
    geom = rf.from_hf_config(SMALL_HYBRID_MOE_CFG)
    topos = tuple(TOPOLOGIES.values())

    no_max_chips = explain_layout_rejection(
        geom,
        {"decode": {"instances": 2, "tp": 8}},
        topos,
        SolveOptions(mode="auto", max_chips=None),
    )
    assert "max_chips" in no_max_chips

    over_default_inst = explain_layout_rejection(
        geom,
        {"decode": {"instances": 20, "tp": 8}},
        topos,
        SolveOptions(mode="mixed", max_chips=256, max_instances=None),
    )
    assert "max_instances=16" in over_default_inst

    over_chip_budget = explain_layout_rejection(
        geom,
        {"decode": {"instances": 3, "tp": 8}},
        topos,
        SolveOptions(mode="mixed", max_chips=8),
    )
    assert "max_chips=8" in over_chip_budget

    disagg_auto = explain_layout_rejection(
        geom,
        {"prefill": {"pp": 2}, "decode": {"instances": 2, "tp": 8}},
        topos,
        SolveOptions(mode="auto", max_chips=None),
    )
    assert "mixed instances" not in disagg_auto


def test_crossover_uniform_pp_kv_parts_matches_stage_peak() -> None:
    """Resolve uniform PP stage KV via max-stage peak rather than layer average."""
    geom = from_hf_config(QWEN35_CFG)
    dt = Dtypes()
    chip = CHIPS["tpu7x"]
    pool = PoolMeshPlan(
        role="prefill",
        instances=1,
        topology="2x2x1",
        chips_per_instance=4,
        tp=1,
        dp=1,
        ep=1,
        pp=4,
        pcp=1,
        kv_split=1,
        mesh_axes=(4,),
        hbm_used_gb=20.0,
        hbm_limit_gb=80.0,
        moe_dispatch="allreduce",
        pp_layer_partition=None,
    )
    par = Parallel(tp=1, pp=4, ranks_per_chip=chip.cores_per_chip)
    uniform_part = (geom.num_layers // pool.pp,) * pool.pp
    expected_lin, expected_fix = rf.pp_stage_kv_parts(
        geom, dt, par, (uniform_part, 2048), chip=chip
    )
    got_lin, got_fix = _kv_parts_for_pool(geom, dt, chip, pool, 2048)
    assert got_lin == pytest.approx(4096.0)
    assert got_lin == pytest.approx(expected_lin)
    assert got_fix == pytest.approx(expected_fix)
