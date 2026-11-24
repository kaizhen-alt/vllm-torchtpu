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
"""Verification tests for DeepSeek-V4.1-Flash priced-only model onboarding."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import time
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import (
    constants,
    models,
    roofline,
    support,
    vmem,
    workbench_engine,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
SNAPSHOT_PATH = DATA_DIR / "hf_configs" / "deepseek-v4.1-flash.json"
CONTRACT_PATH = DATA_DIR / "contracts" / "deepseek_v4_1_flash_runtime_contract.json"

GIB = float(1024**3)
EXPECTED_NUM_LAYERS = 40
EXPECTED_ATTN_LAYER_PARAMS = 126_615_552
EXPECTED_ROUTED_GIB = 268.9453125
EXPECTED_LINEAR_KV_BYTES = 2148
EXPECTED_SWA_FIXED_BYTES = 5_242_880
EXPECTED_RING_STATE_BYTES = 12_288
EXPECTED_CANDIDATE_SCAN_CAP = 16_384
EXPECTED_CAPPED_INDEX_OWNERS = 4
EXPECTED_MHC_TOTAL_PARAMS = 39_321_600
EXPECTED_ENGRAM_H2D_BOTH = 12_672.0
EXPECTED_ENGRAM_H2D_FIRST = 6_336.0
EXPECTED_ENGRAM_L1_RANK_GIB = 101_377_628_352 / (8 * GIB)
EXPECTED_CONTRACT_RULES = 11
EXPECTED_CED_BOUNDARY = 20
DISPLAY_TOL_US = 0.015


def _run_engine_runner(
    js_runtime: str,
    runner_path: Path,
    in_json: Path,
    out_json: Path,
    timeout_s: float = 60.0,
) -> None:
    """Execute advisor engine runner script via POSIX spawn without shell."""
    argv = [js_runtime, str(runner_path), str(in_json), str(out_json)]
    pid = os.posix_spawnp(js_runtime, argv, os.environ)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        wpid, status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            rc = os.waitstatus_to_exitcode(status)
            if rc != 0:
                msg = f"JS runner failed with return code {rc}"
                raise RuntimeError(msg)
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    msg = f"JS runner timed out after {timeout_s}s"
    raise TimeoutError(msg)


def _scope_by_name(
    report: advisor.AdvisorReport, scope_name: str
) -> advisor.ScopeEstimate:
    """Return one ScopeEstimate from an AdvisorReport by scope name."""
    return next(scope for scope in report.scopes if scope.scope == scope_name)


def test_dsv41_flash_snapshot_matches_preset_and_manifest_totals() -> None:
    """Verify authentic DeepSeek-V4.1-Flash snapshot matches preset and totals."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    assert resolved.name == "deepseek-v4.1-flash"
    assert preset.num_layers == EXPECTED_NUM_LAYERS
    assert preset.num_moe_layers == EXPECTED_NUM_LAYERS
    assert preset.expert_dtype == "mxfp4"
    assert preset.weight_dtype == "fp8"
    assert preset.ced_boundary == EXPECTED_CED_BOUNDARY
    assert models.hyper_connection_total_params(preset) == EXPECTED_MHC_TOTAL_PARAMS

    attn_params = (
        preset.hidden_size * preset.q_lora_rank
        + preset.q_lora_rank * preset.num_attention_heads * preset.head_dim
        + preset.hidden_size * preset.num_kv_heads * preset.head_dim
        + preset.num_attention_heads * preset.head_dim * preset.o_lora_rank
        + preset.o_groups * preset.o_lora_rank * preset.hidden_size
    )
    assert attn_params == EXPECTED_ATTN_LAYER_PARAMS

    routed_params = (
        preset.num_moe_layers
        * preset.num_experts
        * 3
        * preset.moe_intermediate_size
        * preset.hidden_size
    )
    routed_gib = routed_params * constants.DTYPE_BYTES["mxfp4"] / GIB
    assert routed_gib == pytest.approx(EXPECTED_ROUTED_GIB, rel=1e-5)

    sa = preset.sparse_attn
    assert sa is not None
    assert sa.kind == "csa"
    linear_kv = sum(
        (o.record_elems * o.elem_bytes + o.record_overhead_bytes) / o.record_tokens
        for o in (*sa.kv_owners, *sa.index_owners)
    )
    assert int(linear_kv) == EXPECTED_LINEAR_KV_BYTES
    swa_fixed = len(sa.window_layers) * sa.window_tokens * sa.window_bytes_per_token
    assert int(swa_fixed) == EXPECTED_SWA_FIXED_BYTES
    assert sa.kv_owner_ring_bytes_per_req == EXPECTED_RING_STATE_BYTES
    capped_owners = [o.max_scan_records for o in sa.index_owners if o.max_scan_records]
    assert capped_owners == [EXPECTED_CANDIDATE_SCAN_CAP] * EXPECTED_CAPPED_INDEX_OWNERS

    tier, _ = support.classify_model_support_tier(preset)
    assert tier == "priced_only"


def test_dsv41_flash_engram_hbm_first_and_ced_fusion_breaks() -> None:
    """Verify Engram CPU/HBM switch, layer-group partition, and CED prefill cap."""
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    chip = advisor.resolve_chip_spec("tpu7x")
    cpu_wl = advisor.WorkloadSpec(
        phase="decode", batch=8, context=4096, tp=1, dp=8, ep=8
    )
    hbm_first_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=8,
        context=4096,
        tp=1,
        dp=8,
        ep=8,
        host_table_hbm_overrides=(("engram_l1", True),),
    )

    cpu_overrides = dict(cpu_wl.host_table_hbm_overrides)
    cpu_h2d = sum(
        t.h2d_bytes_per_token
        for t in preset.host_tables
        if not cpu_overrides.get(t.name, t.hbm_resident)
    )
    assert cpu_h2d == pytest.approx(EXPECTED_ENGRAM_H2D_BOTH, rel=1e-6)
    cpu_breaks = vmem.active_fusion_break_layers(preset, cpu_wl)
    assert cpu_breaks == (1, 14)
    assert vmem.fusion_break_layer_groups(preset.num_layers, cpu_breaks) == (1, 13, 26)

    first_overrides = dict(hbm_first_wl.host_table_hbm_overrides)
    first_h2d = sum(
        t.h2d_bytes_per_token
        for t in preset.host_tables
        if not first_overrides.get(t.name, t.hbm_resident)
    )
    assert first_h2d == pytest.approx(EXPECTED_ENGRAM_H2D_FIRST, rel=1e-6)
    assert vmem.active_fusion_break_layers(preset, hbm_first_wl) == (14,)
    hbm_delta_gib = (
        roofline.build_scope_context(preset, hbm_first_wl, chip).hbm_used
        - roofline.build_scope_context(preset, cpu_wl, chip).hbm_used
    ) / GIB
    assert hbm_delta_gib == pytest.approx(EXPECTED_ENGRAM_L1_RANK_GIB, rel=1e-6)

    pref_std = advisor.WorkloadSpec(
        phase="prefill", batch=8, isl=1024, tp=1, dp=8, ep=8, execution="full"
    )
    pref_ced = advisor.WorkloadSpec(
        phase="prefill", batch=8, isl=1024, tp=1, dp=8, ep=8, execution="ced"
    )
    std_breaks = vmem.active_fusion_break_layers(preset, pref_std)
    ced_breaks = vmem.active_fusion_break_layers(preset, pref_ced)
    assert EXPECTED_CED_BOUNDARY not in std_breaks
    assert EXPECTED_CED_BOUNDARY in ced_breaks
    std_rep = advisor.advise(preset, pref_std, chip)
    ced_rep = advisor.advise(preset, pref_ced, chip)
    assert _scope_by_name(ced_rep, "per_op").step_bound_us < (
        _scope_by_name(std_rep, "per_op").step_bound_us
    )


def test_dsv41_flash_priced_only_verdict_and_blockers() -> None:
    """Verify TP1 x DP8 x EP8 recommends per_op with exact honest blockers."""
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="verify", batch=8, spec_tokens=5, context=4096, tp=1, dp=8, ep=8
    )
    rep = advisor.advise(preset, wl, chip)
    assert rep.recommended_scope == "per_op"
    ws = _scope_by_name(rep, "whole_step")
    assert ws.support == "hypothetical"
    blockers = set(ws.blockers)
    for expected in (
        "fusion_break",
        "all_to_all_not_in_core",
        "family_not_in_core",
        "dtype_not_in_core",
    ):
        assert expected in blockers, f"Missing {expected} in {blockers}"
    assert "unmodeled_layer_family" not in blockers
    assert "planner_rejected" not in blockers

    sweep = advisor.advise_regimes(preset, chip)
    html_text = advisor.render_advisor_html(sweep)
    assert "deepseek-v4.1-flash" in html_text


def test_dsv41_flash_recorded_contract_parity() -> None:
    """Verify recorded DeepSeek-V4.1-Flash runtime contract matches advisor model."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rules = contract["rules"]
    assert len(rules) == EXPECTED_CONTRACT_RULES
    assert contract["served_execution"] == "ced"
    assert any("dcp > 1" in entry for entry in contract["not_enumerated"])
    summaries = "\n".join(rule["summary"] for rule in rules)
    for needle in (
        "PP must be 1",
        "prefill-context-parallel-size == 1",
        "TP == 1",
        "Indexer cache dtype must be fp8",
        "SWA_PAGE_BYTES = 64 * 512 * 2",
        "ced_boundary = num_hidden_layers // 2 (20)",
        "EP >= 16",
    ):
        assert needle in summaries, f"Missing contract phrase: {needle}"


def test_dsv41_flash_js_workbench_parity(tmp_path: Path) -> None:
    """Verify Python-vs-JS workbench parity for DeepSeek-V4.1-Flash regimes."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    preset = models.resolve_model_profile("deepseek-v4.1-flash")
    prof_dict = dataclasses.asdict(preset)
    chip_dict = dataclasses.asdict(chip_spec)

    workloads = [
        advisor.WorkloadSpec(
            phase="verify", batch=8, spec_tokens=5, context=4096, tp=1, dp=8, ep=8
        ),
        advisor.WorkloadSpec(
            phase="verify",
            batch=8,
            spec_tokens=5,
            context=4096,
            tp=1,
            dp=8,
            ep=8,
            host_table_hbm_overrides=(("engram_l1", True),),
        ),
        advisor.WorkloadSpec(
            phase="verify", batch=256, spec_tokens=5, context=4096, tp=1, dp=8, ep=8
        ),
        advisor.WorkloadSpec(
            phase="prefill", batch=8, isl=1024, tp=1, dp=8, ep=8, execution="ced"
        ),
        advisor.WorkloadSpec(
            phase="prefill", batch=8, isl=4096, tp=1, dp=8, ep=8, execution="ced"
        ),
    ]
    payload_items = [
        {
            "params": {
                "only_validate": False,
                "chip": chip_dict,
                "profile": prof_dict,
                "wl": dataclasses.asdict(wl),
            }
        }
        for wl in workloads
    ]
    in_json = tmp_path / "dsv41_parity_in.json"
    out_json = tmp_path / "dsv41_parity_out.json"
    in_json.write_text(
        json.dumps(
            {
                "engine_path": str(engine_path),
                "seed_data": {
                    "chip_spec": chip_dict,
                    **workbench_engine.build_engine_seed_defaults(),
                },
                "items": payload_items,
            }
        ),
        encoding="utf-8",
    )
    _run_engine_runner(js_runtime, runner_path, in_json, out_json)
    records = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(records) == len(workloads)

    for wl, rec in zip(workloads, records, strict=True):
        got = rec["eval"]
        py_rep = advisor.advise(preset, wl, chip_spec)
        py_raw = advisor.compute_scope_raw_step_us(preset, wl, chip=chip_spec.name)
        assert got["full"]["recScope"] == py_rep.recommended_scope
        for py_scope in py_rep.scopes:
            js_scope = got["scopes"][py_scope.scope]
            assert js_scope["support"] == py_scope.support
            assert list(js_scope["blockers"]) == list(py_scope.blockers)
            assert js_scope["rawStepUs"] == pytest.approx(
                py_raw[py_scope.scope], rel=1e-6, abs=1e-9
            )
            assert js_scope["step_bound_us"] == pytest.approx(
                py_scope.step_bound_us, abs=DISPLAY_TOL_US
            )
            assert js_scope["step_floor_strict_us"] == pytest.approx(
                py_scope.step_floor_strict_us, abs=DISPLAY_TOL_US
            )
            assert js_scope["step_achievable_us"] == pytest.approx(
                py_scope.step_achievable_us, abs=DISPLAY_TOL_US
            )
