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
"""Verification tests for Qwen3.8-Flash-Next-FP8 declared megakernel onboarding."""

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
    models,
    roofline,
    support,
    workbench_engine,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
SNAPSHOT_PATH = DATA_DIR / "hf_configs" / "qwen3.8-flash-next.json"
CONTRACT_PATH = DATA_DIR / "contracts" / "qwen3_8_flash_next_fp8_runtime_contract.json"

GIB = float(1024**3)
EXPECTED_NUM_LAYERS = 48
EXPECTED_GDN_LAYERS = 36
EXPECTED_QSA_LAYERS = 12
EXPECTED_QSA_MAX_KEYS = 2051
EXPECTED_HC_TOTAL_PARAMS = 639_631_360
EXPECTED_PLE_TOTAL_BYTES = 51_200_245_760
EXPECTED_PLE_TP8_GIB = EXPECTED_PLE_TOTAL_BYTES / (8 * GIB)
EXPECTED_PLE_H2D_BYTES = 2560
EXPECTED_PLE_STATE_BYTES = 184_320
EXPECTED_MAIN_KV_BYTES = 1024
EXPECTED_COMPRESSED_K_BYTES = 128
EXPECTED_CONTRACT_RULES = 22
EXPECTED_QSA_MAX_SPEC_TOKENS = 5
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


def test_qwen38_flash_snapshot_matches_preset_and_manifest_totals() -> None:
    """Verify authentic Qwen3.8-Flash-Next snapshot matches preset and totals."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    assert resolved.name == "qwen3.8-flash-next"
    assert preset.num_layers == EXPECTED_NUM_LAYERS
    assert preset.num_linear_layers == EXPECTED_GDN_LAYERS
    assert preset.num_attention_layers == EXPECTED_QSA_LAYERS
    assert preset.state_dtype == "fp32"
    assert preset.weight_dtype == "bf16"
    assert preset.expert_dtype == "fp8"
    assert preset.expert_act_dtype == "fp8"
    assert models.hyper_connection_total_params(preset) == EXPECTED_HC_TOTAL_PARAMS
    assert preset.sparse_attn is not None
    assert preset.sparse_attn.kind == "qsa"
    assert preset.sparse_attn.select_tokens == EXPECTED_QSA_MAX_KEYS
    assert len(preset.host_tables) == 1
    ple = preset.host_tables[0]
    assert ple.name == "ple"
    assert ple.layers == (1,)
    assert ple.hbm_resident is True
    assert ple.table_bytes_total == EXPECTED_PLE_TOTAL_BYTES
    assert ple.table_bytes_total / (8 * GIB) == pytest.approx(
        EXPECTED_PLE_TP8_GIB, rel=1e-6
    )
    tier, _ = support.classify_model_support_tier(preset)
    assert tier == "declared"


def test_qwen38_flash_ple_offload_toggles_fusion_break_and_hbm() -> None:
    """Verify PLE HBM residency vs host offload toggles HBM bytes and fusion break."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    chip = advisor.resolve_chip_spec("tpu7x")
    resident_wl = advisor.WorkloadSpec(
        phase="decode", batch=8, context=4096, tp=8, ep=8, dcp=4
    )
    offload_wl = advisor.WorkloadSpec(
        phase="decode",
        batch=8,
        context=4096,
        tp=8,
        ep=8,
        dcp=4,
        host_table_hbm_overrides=(("ple", False),),
    )

    resident_report = advisor.advise(preset, resident_wl, chip)
    offload_report = advisor.advise(preset, offload_wl, chip)

    resident_ws = _scope_by_name(resident_report, "whole_step")
    offload_ws = _scope_by_name(offload_report, "whole_step")
    assert resident_ws.support == "declared"
    assert "fusion_break" not in resident_ws.blockers
    assert offload_ws.support == "hypothetical"
    assert "fusion_break" in offload_ws.blockers

    resident_hbm = roofline.build_scope_context(preset, resident_wl, chip).hbm_used
    offload_hbm = roofline.build_scope_context(preset, offload_wl, chip).hbm_used
    hbm_delta_gib = (resident_hbm - offload_hbm) / GIB
    assert hbm_delta_gib == pytest.approx(EXPECTED_PLE_TP8_GIB, rel=1e-6)


def test_qwen38_flash_declared_megakernel_and_microbatch_crossover() -> None:
    """Verify declared TP8xEP8xDCP4 and colocated layouts plus microbatch split."""
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    chip = advisor.resolve_chip_spec("tpu7x")

    low_wl = advisor.WorkloadSpec(
        phase="verify", batch=8, spec_tokens=3, context=4096, tp=8, ep=8, dcp=4
    )
    low_report = advisor.advise(preset, low_wl, chip)
    low_ws = _scope_by_name(low_report, "whole_step")
    assert low_ws.support == "declared"
    assert low_ws.n_splits == 1
    assert low_report.recommended_scope == "whole_step"

    split_wl = advisor.WorkloadSpec(
        phase="verify", batch=64, spec_tokens=3, context=4096, tp=8, ep=8, dcp=4
    )
    split_report = advisor.advise(preset, split_wl, chip)
    split_ws = _scope_by_name(split_report, "whole_step")
    split_po = _scope_by_name(split_report, "per_op")
    assert split_ws.support == "declared"
    assert split_ws.n_splits > 1
    assert split_ws.step_achievable_us > split_po.step_achievable_us
    assert split_report.recommended_scope == "per_op"

    for dcp_width in (1, 2):
        coloc_wl = advisor.WorkloadSpec(
            phase="verify",
            batch=8,
            spec_tokens=3,
            context=4096,
            tp=8,
            ep=8,
            dcp=dcp_width,
        )
        coloc_report = advisor.advise(preset, coloc_wl, chip)
        coloc_ws = _scope_by_name(coloc_report, "whole_step")
        assert coloc_ws.support == "declared"


def test_qwen38_flash_recorded_contract_parity() -> None:
    """Verify recorded Qwen3.8-Flash-Next runtime contract matches advisor profile."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rules = contract["rules"]
    assert len(rules) == EXPECTED_CONTRACT_RULES
    rule_by_name = {rule["name"]: rule for rule in rules}
    assert "main_kv_dcp_env" in rule_by_name
    assert "VLLM_TPU_QSA_MAIN_KV_DCP" in rule_by_name["main_kv_dcp_env"]["rule"]
    assert (
        str(EXPECTED_QSA_MAX_SPEC_TOKENS)
        in (rule_by_name["qsa_ring_rows_and_max_spec"]["rule"])
    )
    assert (
        "index + 1 in t.ple_layer_ids"
        in (rule_by_name["ple_indexing_rows_state_and_h2d"]["verbatim"])
    )

    verified = contract["verified_layout_bytes"]
    assert verified["main_kv_b_per_tok_layer_chip"] == EXPECTED_MAIN_KV_BYTES
    assert verified["compressed_k_live_b_per_tok_layer"] == EXPECTED_COMPRESSED_K_BYTES
    assert verified["ple_total_bytes"] == EXPECTED_PLE_TOTAL_BYTES
    assert verified["ple_h2d_fp8_b_per_tok"] == EXPECTED_PLE_H2D_BYTES
    assert verified["ple_state_bytes"] == EXPECTED_PLE_STATE_BYTES

    preset = models.resolve_model_profile("qwen3.8-flash-next")
    ple = preset.host_tables[0]
    assert ple.layers == (1,)
    assert ple.table_bytes_total == verified["ple_total_bytes"]
    assert int(ple.h2d_bytes_per_token) == verified["ple_h2d_fp8_b_per_tok"]
    assert ple.state_bytes_per_request == verified["ple_state_bytes"]


def test_qwen38_flash_js_workbench_parity(tmp_path: Path) -> None:
    """Verify Python-vs-JS workbench parity for Qwen3.8-Flash-Next regimes."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    preset = models.resolve_model_profile("qwen3.8-flash-next")
    prof_dict = dataclasses.asdict(preset)
    chip_dict = dataclasses.asdict(chip_spec)

    workloads = [
        advisor.WorkloadSpec(
            phase="verify", batch=8, spec_tokens=3, context=4096, tp=8, ep=8, dcp=4
        ),
        advisor.WorkloadSpec(
            phase="verify", batch=64, spec_tokens=3, context=4096, tp=8, ep=8, dcp=4
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=8,
            context=4096,
            tp=8,
            ep=8,
            dcp=4,
            host_table_hbm_overrides=(("ple", False),),
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
    in_json = tmp_path / "q38_parity_in.json"
    out_json = tmp_path / "q38_parity_out.json"
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
