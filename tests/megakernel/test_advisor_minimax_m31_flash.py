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
"""Verification tests for MiniMax-M3.1-Flash priced-only model onboarding."""

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
    workbench_engine,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "advisor"
SNAPSHOT_PATH = DATA_DIR / "hf_configs" / "minimax-m3.1-flash.json"
CONTRACT_PATH = DATA_DIR / "contracts" / "minimax_m3_1_flash_runtime_contract.json"

GIB = float(1024**3)
EXPECTED_NUM_LAYERS = 60
EXPECTED_MOE_LAYERS = 57
EXPECTED_DENSE_INTER = 12_288
EXPECTED_MOE_INTER = 3072
EXPECTED_SHARED_INTER = 3072
EXPECTED_ROUTED_GIB = 216.421875
EXPECTED_IDX_PROJ_PARAMS = 235_929_600
EXPECTED_MSA_ATTENDED_TOKENS = 2176
EXPECTED_CONTRACT_RULES = 24
EXPECTED_BLOCK_SIZE = 128
EXPECTED_INT4_G1 = 228
EXPECTED_INT4_G2 = 356
EXPECTED_INT4_G4 = 612
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


def test_minimax_m31_flash_snapshot_matches_preset_and_manifest_totals() -> None:
    """Verify authentic MiniMax-M3.1-Flash snapshot matches preset and totals."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    resolved = models.model_profile_from_hf_config(snapshot["config"])
    preset = models.resolve_model_profile("minimax-m3.1-flash")
    assert resolved.name == "minimax-m3.1-flash"
    assert preset.num_layers == EXPECTED_NUM_LAYERS
    assert preset.num_moe_layers == EXPECTED_MOE_LAYERS
    assert preset.intermediate_size == EXPECTED_DENSE_INTER
    assert preset.moe_intermediate_size == EXPECTED_MOE_INTER
    assert preset.shared_expert_intermediate_size == EXPECTED_SHARED_INTER
    assert preset.weight_dtype == "fp8"
    assert preset.expert_dtype == "nvfp4"
    assert preset.weight_block_k == 0
    assert preset.dcp_span == "tp_group"

    routed_params = (
        preset.num_moe_layers
        * preset.num_experts
        * 3
        * preset.moe_intermediate_size
        * preset.hidden_size
    )
    routed_gib = routed_params * constants.DTYPE_BYTES["nvfp4"] / GIB
    assert routed_gib == pytest.approx(EXPECTED_ROUTED_GIB, rel=1e-5)

    sa = preset.sparse_attn
    assert sa is not None
    assert sa.kind == "msa"
    assert sa.window_tokens + sa.select_tokens == EXPECTED_MSA_ATTENDED_TOKENS
    idx_params = (
        preset.num_layers
        * (sa.index_heads + sa.index_kv_heads)
        * sa.index_dim
        * preset.hidden_size
    )
    assert idx_params == EXPECTED_IDX_PROJ_PARAMS

    tier, _ = support.classify_model_support_tier(preset)
    assert tier == "priced_only"


def test_minimax_m31_flash_wide_dcp_and_hadamard_int4_cache_bytes() -> None:
    """Verify exact per-rank MSA KV bytes across G in {1, 2, 4} and KV dtypes."""
    preset = models.resolve_model_profile("minimax-m3.1-flash")
    chip = advisor.resolve_chip_spec("tpu7x")
    expected_table = (
        (8, 1, 2, "int4_per_token_head", 228.0),
        (8, 1, 4, "int4_per_token_head", 356.0),
        (8, 1, 8, "int4_per_token_head", 612.0),
        (8, 1, 2, "fp8", 384.0),
        (8, 1, 4, "fp8", 640.0),
        (8, 1, 8, "fp8", 1152.0),
        (8, 1, 2, "bf16", 768.0),
        (8, 1, 4, "bf16", 1280.0),
        (8, 1, 8, "bf16", 2304.0),
        (4, 2, 1, "fp8", 384.0),
    )
    for tp, dp, dcp, kv_dt, expected_per_layer_b in expected_table:
        prof_kv = models.finalize_model_profile(
            dataclasses.replace(preset, kv_dtype=kv_dt)
        )
        wl_lo = advisor.WorkloadSpec(
            phase="decode",
            batch=dp,
            context=1024,
            tp=tp,
            dp=dp,
            ep=8,
            dcp=dcp,
        )
        wl_hi = advisor.WorkloadSpec(
            phase="decode",
            batch=dp,
            context=2048,
            tp=tp,
            dp=dp,
            ep=8,
            dcp=dcp,
        )
        # Delta over +1024 context tokens across 60 layers (1 seq/rep) divided by dcp.
        delta_hbm = (
            roofline.build_scope_context(prof_kv, wl_hi, chip).hbm_used
            - roofline.build_scope_context(prof_kv, wl_lo, chip).hbm_used
        )
        # Each rank stores (context / dcp) tokens per sequence across 60 layers.
        tokens_stored_delta = 1024.0 / float(dcp)
        per_stored_tok_layer = delta_hbm / (preset.num_layers * tokens_stored_delta)
        assert per_stored_tok_layer == pytest.approx(expected_per_layer_b, rel=1e-6), (
            f"Mismatch for tp={tp} dp={dp} dcp={dcp} kv={kv_dt}: {per_stored_tok_layer}"
        )

    prof_int4 = models.finalize_model_profile(
        dataclasses.replace(preset, kv_dtype="int4_per_token_head")
    )
    prof_fp8 = models.finalize_model_profile(
        dataclasses.replace(preset, kv_dtype="fp8")
    )
    for dcp in (2, 4, 8):
        wl_cmp = advisor.WorkloadSpec(
            phase="decode",
            batch=8,
            context=8192,
            tp=8,
            ep=8,
            dcp=dcp,
        )
        assert (
            roofline.build_scope_context(prof_int4, wl_cmp, chip).hbm_used
            < roofline.build_scope_context(prof_fp8, wl_cmp, chip).hbm_used
        )


def test_minimax_m31_flash_recorded_contract_parity() -> None:
    """Verify recorded MiniMax-M3.1-Flash runtime contract matches advisor model."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rules = contract["rules"]
    assert len(rules) == EXPECTED_CONTRACT_RULES
    by_name = {rule["name"]: rule for rule in rules}
    block_rule = by_name["block_size_equals_sparse_block_size"]["rule"]
    assert str(EXPECTED_BLOCK_SIZE) in block_rule
    assert "4" in by_name["min_tp_equals_num_kv_heads"]["rule"]
    assert "PCP > 1" in by_name["no_pcp"]["rule"]
    assert "PP > 1" in by_name["no_pp"]["rule"]
    assert "replicas" in by_name["dcp_tp_group_divisibility"]["verbatim"]
    assert "int4_per_token_head" in by_name["allowed_kv_cache_dtypes"]["rule"]
    assert "turboquant_4bit_nc" not in by_name["allowed_kv_cache_dtypes"]["rule"]
    assert "7 draft" in by_name["msa_topk_selection_and_verify_tile"]["rule"]

    cache_table = contract["cache_bytes_per_stored_token_per_layer_per_rank"]
    assert cache_table["kv_heads_1"]["int4_per_token_head"] == EXPECTED_INT4_G1
    assert cache_table["kv_heads_2"]["int4_per_token_head"] == EXPECTED_INT4_G2
    assert cache_table["kv_heads_4"]["int4_per_token_head"] == EXPECTED_INT4_G4


def test_glm52_wide_dcp_pcp2_dcp16_resolves_cleanly() -> None:
    """Verify GLM-5.2 Wide-DCP TP8 x EP8 x PCP2 x DCP16 validates without DCP error."""
    glm = models.resolve_model_profile("glm-5.2")
    chip = advisor.resolve_chip_spec("tpu7x")
    wl = advisor.WorkloadSpec(
        phase="decode", batch=8, context=4096, tp=8, ep=8, pcp=2, dcp=16
    )
    reasons = advisor.validate_workload(glm, chip, wl)
    assert "dcp_not_dividing_tp" not in reasons
    report = advisor.advise(glm, wl, chip)
    assert report.recommended_scope == "per_op"


def test_minimax_m31_flash_js_workbench_parity(tmp_path: Path) -> None:
    """Verify Python-vs-JS workbench parity for MiniMax-M3.1-Flash Wide-DCP/4-bit."""
    js_runtime = shutil.which("node") or shutil.which("gjs")
    if js_runtime is None:
        msg = "Neither node nor gjs runtime found; install nodejs"
        raise RuntimeError(msg)

    runner_path = Path(__file__).with_name("advisor_engine_runner.js")
    engine_path = Path(models.__file__).parent / "assets" / "workbench_engine.js"
    chip_spec = advisor.resolve_chip_spec("tpu7x")
    preset = models.resolve_model_profile("minimax-m3.1-flash")
    chip_dict = dataclasses.asdict(chip_spec)

    cases = [
        (
            "fp8",
            advisor.WorkloadSpec(
                phase="verify",
                batch=8,
                spec_tokens=7,
                context=8192,
                tp=8,
                ep=8,
                dcp=2,
            ),
        ),
        (
            "int4_per_token_head",
            advisor.WorkloadSpec(
                phase="verify",
                batch=8,
                spec_tokens=7,
                context=8192,
                tp=8,
                ep=8,
                dcp=4,
            ),
        ),
        (
            "int4_per_token_head",
            advisor.WorkloadSpec(
                phase="verify",
                batch=16,
                spec_tokens=7,
                context=32768,
                tp=8,
                ep=8,
                dcp=8,
            ),
        ),
    ]
    prof_cases = [
        (models.finalize_model_profile(dataclasses.replace(preset, kv_dtype=kv_dt)), wl)
        for kv_dt, wl in cases
    ]
    payload_items = [
        {
            "params": {
                "only_validate": False,
                "chip": chip_dict,
                "profile": dataclasses.asdict(prof_kv),
                "wl": dataclasses.asdict(wl),
            }
        }
        for prof_kv, wl in prof_cases
    ]
    in_json = tmp_path / "mm31_parity_in.json"
    out_json = tmp_path / "mm31_parity_out.json"
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
    assert len(records) == len(prof_cases)

    for (prof_kv, wl), rec in zip(prof_cases, records, strict=True):
        got = rec["eval"]
        py_rep = advisor.advise(prof_kv, wl, chip_spec)
        py_raw = advisor.compute_scope_raw_step_us(prof_kv, wl, chip=chip_spec.name)
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
