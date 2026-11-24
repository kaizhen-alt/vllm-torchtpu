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
"""CPU unit tests for vLLM serve flag emission and Pareto ceiling export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Literal

import pytest

from tests.roofline.conftest import SMALL_HYBRID_MOE_CFG
from vllm_torchtpu.roofline import (
    CHIPS,
    SLO_VERDICT_RULED_OUT,
    SOLVE_SCHEMA,
    Dtypes,
    ParetoPoint,
    PoolMeshPlan,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    build_solve_payload,
    cli,
    emit_ceiling_rows,
    emit_launch_args,
    explain_layout_rejection,
    format_parallelism,
    format_submodule_parallelism,
    from_hf_config,
    pareto_skyline_by_concurrency,
    parse_layout,
    point_launch_args,
    pool_cli_args,
    pool_label,
    reference_layout_report,
    solve_serving_plan,
    top_layouts,
)
from vllm_torchtpu.roofline.constants import NO_FEASIBLE_LAYOUT_WARNING, UNIT_GIBI
from vllm_torchtpu.roofline.emit import (
    _MODEL_PLACEHOLDER,
    _UNKNOWN_MODEL_WARNING,
)
from vllm_torchtpu.roofline.families import _decode_group_key
from vllm_torchtpu.roofline.hf_loader import dtypes_from_hf_config as _dtypes_from_cfg
from vllm_torchtpu.roofline.memory import (
    SeqKvLayout as _SeqKvLayout,
)
from vllm_torchtpu.roofline.memory import (
    hbm_breakdown_per_core_bytes as _hbm_breakdown,
)
from vllm_torchtpu.roofline.modules import Parallel as _Parallel
from vllm_torchtpu.roofline.schema import _pool_key


@dataclass(frozen=True)
class _PoolConfig:
    """Typed synthetic pool configuration for flag-emission tests."""

    instances: int = 1
    topology: str = "2x2x2"
    chips: int = 8
    tp: int = 2
    dp: int = 4
    ep: int = 8
    pp: int = 1
    pcp: int = 1
    kv_split: int = 1
    moe_dispatch: str = "allreduce"


def _toy_moe_config_dict() -> dict[str, object]:
    """Return sample MoE model configuration dictionary for CLI emission tests."""
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1536,
        "shared_expert_intermediate_size": 3072,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
        "dtype": "bfloat16",
        "quantization_config": {"quant_method": "fp8"},
    }


def _sample_pool(
    role: Literal["mixed", "prefill", "decode"] = "decode",
    cfg: _PoolConfig | None = None,
) -> PoolMeshPlan:
    """Build synthetic PoolMeshPlan fixture with typed parallel dimensions."""
    spec = cfg or _PoolConfig()
    return PoolMeshPlan(
        role=role,
        instances=spec.instances,
        topology=spec.topology,
        chips_per_instance=spec.chips,
        tp=spec.tp,
        dp=spec.dp,
        ep=spec.ep,
        pp=spec.pp,
        pcp=spec.pcp,
        kv_split=spec.kv_split,
        mesh_axes=(2, 2, 2, 2),
        hbm_used_gb=42.0,
        hbm_limit_gb=83.25,
        moe_dispatch=spec.moe_dispatch,
    )


def test_pool_cli_args_vllm_layouts() -> None:
    """Verify mapping from PoolMeshPlan topologies to vLLM server CLI arguments."""
    dp_ep_pool = _sample_pool(cfg=_PoolConfig(tp=2, dp=4, ep=8, pp=2))
    vllm_flags = pool_cli_args(
        dp_ep_pool,
        kv_dtype="fp8",
        gpu_memory_utilization=0.9,
    )
    assert "--kv-transfer-config" in vllm_flags
    kv_idx = vllm_flags.index("--kv-transfer-config")
    kv_cfg = json.loads(vllm_flags[kv_idx + 1])
    raiden_mod = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    assert kv_cfg == {
        "kv_connector": "TPURaidenConnector",
        "kv_connector_module_path": raiden_mod,
        "kv_role": "kv_consumer",
    }
    assert vllm_flags[:kv_idx] == [
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "4",
        "--kv-cache-dtype",
        "fp8",
        "--pipeline-parallel-size",
        "2",
        "--no-async-scheduling",
        "--enable-expert-parallel",
        "--gpu-memory-utilization",
        "0.9",
    ]

    cp_pool = _sample_pool(
        role="mixed", cfg=_PoolConfig(tp=8, dp=1, ep=1, pcp=2, kv_split=4)
    )
    cp_vllm = pool_cli_args(cp_pool, kv_dtype="auto")
    assert cp_vllm == [
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "1",
        "--kv-cache-dtype",
        "auto",
        "--prefill-context-parallel-size",
        "2",
        "--cp-kv-cache-interleave-size",
        "128",
        "--decode-context-parallel-size",
        "4",
    ]


def test_emit_launch_args_mixed_and_disagg() -> None:
    """Verify emission of launch commands for mixed and disaggregated serving plans."""
    scen = TrafficScenario(
        name="chat-8k1k",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16, 64),
        kv_working_set_turns=1.0,
    )
    mixed_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=8, dp=1, ep=8),
    )
    mixed_pt = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=4,
        prefill_pool=None,
        decode_pool=mixed_pool,
        est_ttft_ms=180.0,
        est_tpot_ms=12.0,
        est_interactivity_tok_s=83.3,
        output_throughput_per_chip=320.0,
        token_throughput_per_chip=2880.0,
        p_mixed=0.15,
        binding="hbm",
        slo_met=True,
    )
    mixed_plan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=mixed_pt,
        frontier=(mixed_pt,),
        warnings=(),
        gpu_memory_utilization=0.9,
    )
    mixed_out = emit_launch_args(
        mixed_plan,
        dtypes=Dtypes(kv=1.0),
        model="Qwen/Qwen3.5-397B-A17B-FP8",
    )
    assert mixed_out["prefill_cli_args"] == []
    assert len(mixed_out["commands"]) == 1
    assert mixed_out["commands"][0].startswith(
        "vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --tensor-parallel-size 8"
    )

    prefill_pool = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(
            instances=1, topology="2x2x1", chips=4, tp=4, dp=1, ep=1, pcp=2
        ),
    )
    decode_pool = _sample_pool(
        role="decode",
        cfg=_PoolConfig(instances=2, topology="2x2x2", chips=8, tp=2, dp=4, ep=8),
    )
    disagg_pt = ParetoPoint(
        mode="disagg",
        concurrency=64,
        total_chips=20,
        prefill_pool=prefill_pool,
        decode_pool=decode_pool,
        est_ttft_ms=140.0,
        est_tpot_ms=9.5,
        est_interactivity_tok_s=105.2,
        output_throughput_per_chip=336.8,
        token_throughput_per_chip=3031.2,
        p_mixed=0.0,
        binding="ici",
        slo_met=True,
    )
    disagg_plan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="disagg",
        recommended=disagg_pt,
        frontier=(disagg_pt,),
        warnings=(),
        gpu_memory_utilization=0.9,
    )
    disagg_out = emit_launch_args(
        disagg_plan,
        dtypes=Dtypes(kv=1.0),
        model="Qwen/Qwen3.5-397B-A17B-FP8",
    )
    expected_commands = 3
    assert "--prefill-context-parallel-size" in disagg_out["prefill_cli_args"]
    assert "--cp-kv-cache-interleave-size" in disagg_out["prefill_cli_args"]
    assert "--kv-transfer-config" in disagg_out["prefill_cli_args"]
    assert "--kv-transfer-config" in disagg_out["cli_args"]
    prefill_kv = json.loads(
        disagg_out["prefill_cli_args"][
            disagg_out["prefill_cli_args"].index("--kv-transfer-config") + 1
        ]
    )
    decode_kv = json.loads(
        disagg_out["cli_args"][disagg_out["cli_args"].index("--kv-transfer-config") + 1]
    )
    assert prefill_kv["kv_connector"] == "TPURaidenConnector"
    assert (
        prefill_kv["kv_connector_module_path"]
        == "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
    )
    assert prefill_kv["kv_role"] == "kv_producer"
    assert decode_kv["kv_connector"] == "TPURaidenConnector"
    assert decode_kv["kv_role"] == "kv_consumer"
    assert any("Raiden controller" in w for w in disagg_out["warnings"])
    assert len(disagg_out["commands"]) == expected_commands
    assert disagg_out["commands"][0].startswith("# prefill (1x 2x2x1): vllm serve")
    assert disagg_out["commands"][1].startswith(
        "# decode (2x 2x2x2, host 0): vllm serve"
    )
    assert "--data-parallel-size-local 2" in disagg_out["commands"][1]
    assert "--data-parallel-start-rank 0" in disagg_out["commands"][1]
    assert disagg_out["commands"][2].startswith(
        "# decode (2x 2x2x2, host 1): vllm serve"
    )
    assert "--data-parallel-start-rank 2" in disagg_out["commands"][2]
    assert "--headless" in disagg_out["commands"][2]


def test_emit_launch_args_infeasible_plan_returns_empty_and_cli_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that infeasible SLO plans exit non-zero and write stderr warnings."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_toy_moe_config_dict()), encoding="utf-8")

    rc_emit = cli.main(
        [
            "emit",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "16",
            "--tpot-slo-ms",
            "0.001",
        ]
    )
    captured = capsys.readouterr()
    assert rc_emit == 1
    assert captured.out == ""
    assert "WARNING:" in captured.err

    rc_ceil = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "16",
            "--tpot-slo-ms",
            "0.001",
            "--json",
        ]
    )
    captured_ceil = capsys.readouterr()
    assert rc_ceil == 1
    assert captured_ceil.out.strip() == "[]"
    assert "WARNING:" in captured_ceil.err


def test_emit_ceiling_rows_and_cli_subcommands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify Pareto ceiling row formatting and CLI subcommand JSON output."""
    scen = TrafficScenario(
        name="chat-8k1k",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 8, 32),
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        from_hf_config(_toy_moe_config_dict()),
        Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0),
        CHIPS["tpu7x"],
        scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    rows = emit_ceiling_rows(splan)
    assert len(rows) >= 1
    assert list(rows[0].keys())[:5] == [
        "est_interactivity_tok_s",
        "token_throughput_per_chip",
        "output_throughput_per_chip",
        "est_ttft_ms",
        "est_tpot_ms",
    ]

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_toy_moe_config_dict()), encoding="utf-8")

    rc_vllm = cli.main(
        [
            "emit",
            "--model",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "8,32",
            "--json",
        ]
    )
    out_vllm = json.loads(capsys.readouterr().out)
    assert rc_vllm == 0
    assert "--tensor-parallel-size" in out_vllm["cli_args"]
    assert "--gpu-memory-utilization" in out_vllm["cli_args"]
    assert "env" not in out_vllm

    rc_ceil = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "8,32",
            "--json",
        ]
    )
    out_ceil = json.loads(capsys.readouterr().out)
    assert rc_ceil == 0
    assert len(out_ceil) >= 1
    assert "est_interactivity_tok_s" in out_ceil[0]


_MIXED_SCENARIO_ARGS: tuple[str, ...] = (
    "--example-scenario",
    "chat-8k1k",
    "--mode",
    "mixed",
    "--max-chips",
    "4",
    "--concurrency",
    "8,32",
)


@pytest.mark.parametrize("model_flag", ["--config", "--model"])
def test_emit_cli_prints_placeholder_when_model_source_is_a_config_file(
    model_flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print <MODEL>, not the config.json path, and warn on stderr."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_toy_moe_config_dict()), encoding="utf-8")

    exit_code = cli.main(["emit", model_flag, str(cfg_path), *_MIXED_SCENARIO_ARGS])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"vllm serve {_MODEL_PLACEHOLDER} --tensor-parallel-size" in captured.out
    assert str(cfg_path) not in captured.out
    assert f"WARNING: {_UNKNOWN_MODEL_WARNING}" in captured.err


def test_emit_cli_prints_model_directory_in_serve_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print a model directory given to --model as the model to serve."""
    (tmp_path / "config.json").write_text(
        json.dumps(_toy_moe_config_dict()), encoding="utf-8"
    )

    exit_code = cli.main(["emit", "--model", str(tmp_path), *_MIXED_SCENARIO_ARGS])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"vllm serve {tmp_path} --tensor-parallel-size" in captured.out
    assert _UNKNOWN_MODEL_WARNING not in captured.err


def test_ceiling_cli_warns_when_no_layout_fits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Write the no-feasible-layout warning to stderr when no layout fits in HBM."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_toy_moe_config_dict()), encoding="utf-8")
    # Smallest 4-chip (8-core) weight shard is TP=8; divide by 100 so no layout fits.
    toy_cfg = _toy_moe_config_dict()
    min_weight_bytes = _hbm_breakdown(
        from_hf_config(toy_cfg),
        _dtypes_from_cfg(toy_cfg),
        _Parallel(tp=8, ep=1, moe="tp_slice"),
        _SeqKvLayout(seqs_per_dp=1, context=1),
    ).weights
    no_layout_fits_memory_fraction = (
        f"{(min_weight_bytes / CHIPS['tpu7x'].hbm_bytes) / 100.0:.4f}"
    )

    exit_code = cli.main(
        [
            "ceiling",
            "--config",
            str(cfg_path),
            *_MIXED_SCENARIO_ARGS,
            "--gpu-memory-utilization",
            no_layout_fits_memory_fraction,
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.strip() == "[]"
    assert f"WARNING: {NO_FEASIBLE_LAYOUT_WARNING}" in captured.err


def test_resident_batch_cap_emits_max_num_seqs_without_tput_inflation() -> None:
    """Verify resident batch cap adds --max-num-seqs without inflating tput."""
    dense_cfg = {
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
    geom = from_hf_config(dense_cfg)
    dt = Dtypes(kv=1.0)
    chip = CHIPS["tpu7x"]
    cap_concurrency = 15
    overflow_concurrency = 256
    expected_tp = 8
    expected_max_model_len = 131328
    scen = TrafficScenario(
        name="long-ctx-cap",
        isl=131072,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(cap_concurrency, overflow_concurrency),
        kv_working_set_turns=1.0,
        page_size=16,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    dp1_pts = {
        p.concurrency: p
        for p in splan.candidates
        if p.decode_pool.tp == expected_tp
        and p.decode_pool.dp == 1
        and p.decode_pool.pp == 1
        and p.decode_pool.pcp == 1
        and p.decode_pool.kv_split == 1
    }
    assert cap_concurrency in dp1_pts
    assert overflow_concurrency in dp1_pts
    pt_15 = dp1_pts[cap_concurrency]
    pt_256 = dp1_pts[overflow_concurrency]
    assert pt_256.decode_pool.max_num_seqs == cap_concurrency
    assert pt_256.decode_pool.max_model_len == expected_max_model_len
    flags = pool_cli_args(pt_256.decode_pool, kv_dtype="fp8")
    assert "--max-model-len" in flags
    assert flags[flags.index("--max-model-len") + 1] == str(expected_max_model_len)
    assert "--max-num-seqs" in flags
    assert flags[flags.index("--max-num-seqs") + 1] == str(cap_concurrency)
    assert pt_256.token_throughput_per_chip == pytest.approx(
        pt_15.token_throughput_per_chip, rel=1e-6
    )
    assert pt_256.est_ttft_ms > pt_15.est_ttft_ms


def test_emit_multihost_mode_a_mode_b_and_hybrid_rejection() -> None:
    """Verify multi-host Mode A/B flags and fail-closed hybrid 1 < dp < hosts."""
    scen = TrafficScenario(
        name="mh",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )

    def _plan_for(pool: PoolMeshPlan, *, model_id: str = "qwen3_5_moe") -> ServingPlan:
        """Build a minimal SLO-met ServingPlan wrapping ``pool``."""
        pt = ParetoPoint(
            mode="mixed",
            concurrency=16,
            total_chips=pool.chips_per_instance,
            prefill_pool=None,
            decode_pool=pool,
            est_ttft_ms=100.0,
            est_tpot_ms=10.0,
            est_interactivity_tok_s=100.0,
            output_throughput_per_chip=200.0,
            token_throughput_per_chip=1000.0,
            p_mixed=0.0,
            binding="hbm",
            slo_met=True,
        )
        return ServingPlan(
            model_layers=60,
            chip="tpu7x",
            scenario=scen,
            mode="mixed",
            recommended=pt,
            frontier=(pt,),
            warnings=(),
            model_id=model_id,
        )

    # Mode A: 2 hosts (8 chips), dp=1, tp=16 -> --nnodes 2 + --headless on host 1.
    mode_a_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=16, dp=1, ep=16),
    )
    out_a = emit_launch_args(
        _plan_for(mode_a_pool),
        dtypes=Dtypes(kv=1.0),
        model="Qwen/Qwen3.5-397B-A17B-FP8",
    )
    expected_hosts_2 = 2
    assert len(out_a["commands"]) == expected_hosts_2
    assert "--nnodes 2 --node-rank 0" in out_a["commands"][0]
    assert "--headless" not in out_a["commands"][0]
    assert "--nnodes 2 --node-rank 1" in out_a["commands"][1]
    assert "--headless" in out_a["commands"][1]

    # Mode B: 2 hosts (2x2x2, 8 chips), dp=16, tp=1 -> dp_local=8, start_rank=0 / 8.
    mode_b_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=1, dp=16, ep=16),
    )
    out_b = emit_launch_args(
        _plan_for(mode_b_pool, model_id="glm_moe_dsa"),
        dtypes=Dtypes(kv=1.0),
        model="zai-org/GLM-5.2-FP8",
    )
    assert len(out_b["commands"]) == expected_hosts_2
    assert (
        "--data-parallel-size-local 8 --data-parallel-start-rank 0"
        in out_b["commands"][0]
    )
    assert (
        "--data-parallel-size-local 8 --data-parallel-start-rank 8"
        in out_b["commands"][1]
    )
    assert "--headless" in out_b["commands"][1]

    # Hybrid 1 < dp < hosts (dp=2 on 4 hosts 2x2x4, 16 chips) must fail closed.
    hybrid_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x4", chips=16, tp=16, dp=2, ep=32),
    )
    with pytest.raises(ValueError, match="Unsupported hybrid multi-host"):
        emit_launch_args(
            _plan_for(hybrid_pool),
            dtypes=Dtypes(kv=1.0),
            model="Qwen/Qwen3.5-397B-A17B-FP8",
        )


def test_emit_moe_sp_guard_qwen4_overrides_and_unconditional_warnings() -> None:
    """Verify MoE sequence-parallel guard, qwen4 overrides, and SLO warnings."""
    qwen4_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=8, dp=1, ep=8),
    )
    qwen4_flags = pool_cli_args(qwen4_pool, kv_dtype="fp8", model_id="qwen4_exp")
    assert "--hf-overrides" in qwen4_flags
    assert '{"text_config": {"ple_layer_ids": []}}' in qwen4_flags

    # fused_rs allowed only on qwen3_5_moe with pp == 1.
    scen = TrafficScenario(
        name="sp",
        isl=1024,
        osl=128,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(4,),
        kv_working_set_turns=1.0,
    )
    fused_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(
            topology="2x2x1", chips=4, tp=8, dp=1, ep=8, moe_dispatch="fused_rs"
        ),
    )
    pt = ParetoPoint(
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
    ok_out = emit_launch_args(
        ServingPlan(
            model_layers=60,
            chip="tpu7x",
            scenario=scen,
            mode="mixed",
            recommended=pt,
            frontier=(pt,),
            warnings=(),
            model_id="qwen3_5_moe",
        ),
        dtypes=Dtypes(kv=1.0),
        model="Qwen/Qwen3.5-397B-A17B-FP8",
    )
    assert "TPU_MOE_SEQUENCE_PARALLEL=1" in ok_out["commands"][0]

    with pytest.raises(ValueError, match="TPU_MOE_SEQUENCE_PARALLEL"):
        emit_launch_args(
            ServingPlan(
                model_layers=78,
                chip="tpu7x",
                scenario=scen,
                mode="mixed",
                recommended=pt,
                frontier=(pt,),
                warnings=(),
                model_id="glm_moe_dsa",
            ),
            dtypes=Dtypes(kv=1.0),
            model="zai-org/GLM-5.2-FP8",
        )

    # SLO_UNMET is appended even when INFEASIBLE_SLO already fired, and
    # recommended stays None (never promotes an unmet candidate).
    infeasible_plan = solve_serving_plan(
        from_hf_config(_toy_moe_config_dict()),
        Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0),
        CHIPS["tpu7x"],
        TrafficScenario(
            name="infeasible-slo",
            isl=4096,
            osl=256,
            prefix_hit_rate=0.0,
            accept_len=1.0,
            think_time_s=0.0,
            concurrencies=(4,),
            kv_working_set_turns=1.0,
            tpot_slo_ms=0.001,
        ),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert infeasible_plan.recommended is None
    assert any("INFEASIBLE_SLO" in w for w in infeasible_plan.warnings)
    assert any("SLO_UNMET" in w for w in infeasible_plan.warnings)


def test_emit_speculative_config_for_mtp_scenario_and_omit_when_zero() -> None:
    """Verify --speculative-config is emitted iff the solved plan prices spec_k > 0."""
    q35_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    q35_cfg = json.loads(q35_path.read_text())
    geom = from_hf_config(q35_cfg)
    dt = _dtypes_from_cfg(q35_cfg)
    spec_scen = TrafficScenario(
        name="mtp-spec-test",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=3.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
        spec_tokens=3,
    )
    spec_plan = solve_serving_plan(
        geom, dt, CHIPS["tpu7x"], spec_scen, SolveOptions(mode="mixed", max_chips=16)
    )
    assert spec_plan.recommended is not None
    expected_spec_k = spec_scen.spec_tokens
    assert expected_spec_k is not None
    assert spec_plan.spec_k == expected_spec_k
    out_spec = emit_launch_args(
        spec_plan, dtypes=dt, model="Qwen/Qwen3.5-397B-A17B-FP8"
    )
    assert "--speculative-config" in out_spec["cli_args"]
    spec_val = json.loads(
        out_spec["cli_args"][out_spec["cli_args"].index("--speculative-config") + 1]
    )
    assert spec_val == {"method": "mtp", "num_speculative_tokens": expected_spec_k}

    no_spec_scen = TrafficScenario(
        name="no-spec-test",
        isl=4096,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
        spec_tokens=0,
    )
    no_spec_plan = solve_serving_plan(
        geom, dt, CHIPS["tpu7x"], no_spec_scen, SolveOptions(mode="mixed", max_chips=16)
    )
    out_plain = emit_launch_args(
        no_spec_plan, dtypes=dt, model="Qwen/Qwen3.5-397B-A17B-FP8"
    )
    assert "--speculative-config" not in out_plain["cli_args"]

    # Disaggregated MTP clamps prefill spec_k=1 (allowing PCP > 1) and decode=3.
    disagg_spec = solve_serving_plan(
        geom, dt, CHIPS["tpu7x"], spec_scen, SolveOptions(mode="disagg", max_chips=16)
    )
    out_disagg = emit_launch_args(
        disagg_spec, dtypes=dt, model="Qwen/Qwen3.5-397B-A17B-FP8"
    )
    assert "--prefill-context-parallel-size" not in out_disagg["cli_args"]
    p_cli = out_disagg["prefill_cli_args"]
    d_cli = out_disagg["cli_args"]
    prefill_spec = json.loads(p_cli[p_cli.index("--speculative-config") + 1])
    decode_spec = json.loads(d_cli[d_cli.index("--speculative-config") + 1])
    assert prefill_spec == {"method": "mtp", "num_speculative_tokens": 1}
    assert decode_spec == {"method": "mtp", "num_speculative_tokens": expected_spec_k}


def test_top_layouts_distinct_and_led_by_recommended() -> None:
    """Top-K layouts are distinct, SLO-met, sorted by throughput, led by recommended."""
    chip = CHIPS["tpu7x"]
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scenario = TrafficScenario(
        name="chat-8k1k",
        isl=8192,
        osl=1024,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 2, 4),
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    top_k = 3
    top = top_layouts(splan, k=top_k)
    assert 1 <= len(top) <= top_k
    assert splan.recommended is not None
    assert top[0] == splan.recommended

    keys: set[tuple[object, ...]] = set()
    for pt in top:
        assert pt.slo_met
        key = (
            pt.mode,
            _pool_key(pt.prefill_pool),
            _pool_key(pt.decode_pool),
        )
        assert key not in keys
        keys.add(key)

    tputs = [pt.token_throughput_per_chip for pt in top]
    assert tputs == sorted(tputs, reverse=True)


def test_top_layouts_empty_without_slo_met() -> None:
    """Top-K layouts return empty tuple if no candidate meets SLOs."""
    chip = CHIPS["tpu7x"]
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scenario = TrafficScenario(
        name="custom",
        isl=8192,
        osl=1024,
        concurrencies=(1,),
        tpot_slo_ms=0.0001,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        scenario,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert top_layouts(splan, k=3) == ()


def test_solve_json_envelope_and_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Solve --json outputs versioned schema with provenance, stats, and launches."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(SMALL_HYBRID_MOE_CFG), encoding="utf-8")

    rc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--chip",
            "tpu7x",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--concurrency",
            "1,2",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == SOLVE_SCHEMA
    assert out["provenance"]["calibrated"] is False
    assert "L1 analytical step floors" in out["provenance"]["basis"]
    assert "uncalibrated serial-phase" in out["provenance"]["basis"]
    assert out["search"]["enumerated"] >= 1
    assert len(out["top_k"]) >= 1
    assert out["top_k"][0]["launch"]["commands"][0].startswith("vllm serve")
    assert out["single_host_fit"]["feasible"] is True
    expected_max_chips = 4
    assert out["chip_budget"]["max_chips"] == expected_max_chips
    assert "recommended" in out
    assert "frontier" in out
    assert "candidates" in out


def test_top_layouts_structural_diversity_and_interactivity_anchor() -> None:
    """Verify structural diversity across top layouts with interactivity anchor."""
    scen = TrafficScenario(
        name="chat-8k1k",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 16, 64),
        kv_working_set_turns=1.0,
    )
    p_a1 = dc_replace(
        _sample_pool(
            role="mixed",
            cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8, ep=16),
        ),
        moe_dispatch="ag_ep",
    )
    p_a2 = dc_replace(
        _sample_pool(
            role="mixed",
            cfg=_PoolConfig(topology="2x2x2", chips=8, tp=2, dp=8, ep=16),
        ),
        moe_dispatch="a2a_push",
    )
    p_b = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=4, dp=4, ep=16),
    )
    p_c = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=8, dp=2, ep=16),
    )
    p_d = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x2", chips=8, tp=16, dp=1, ep=16),
    )

    def _mk_pt(pool: PoolMeshPlan, conc: int, tput: float, inter: float) -> ParetoPoint:
        """Create a synthetic Pareto point for testing layout selection."""
        return ParetoPoint(
            mode="mixed",
            concurrency=conc,
            total_chips=8,
            prefill_pool=None,
            decode_pool=pool,
            est_ttft_ms=100.0,
            est_tpot_ms=1000.0 / inter,
            est_interactivity_tok_s=inter,
            output_throughput_per_chip=tput * 0.1,
            token_throughput_per_chip=tput,
            p_mixed=0.05,
            binding="hbm",
            slo_met=True,
        )

    a1 = _mk_pt(p_a1, 64, 5000.0, 40.0)
    a2 = _mk_pt(p_a2, 64, 4900.0, 39.0)
    b = _mk_pt(p_b, 32, 4200.0, 65.0)
    c = _mk_pt(p_c, 16, 3800.0, 80.0)
    d = _mk_pt(p_d, 1, 2500.0, 130.0)
    splan = ServingPlan(
        model_layers=32,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=a1,
        frontier=(d, c, b, a1),
        warnings=(),
        candidates=(a1, a2, b, c, d),
    )

    top_k = 3
    top = top_layouts(splan, k=top_k)
    assert top == (a1, b, d)
    cmds = {
        tuple(point_launch_args(pt, kv_dtype="fp8", model="m")["commands"])
        for pt in top
    }
    assert len(cmds) == top_k

    payload = build_solve_payload(
        splan,
        chip=CHIPS["tpu7x"],
        dtypes=Dtypes(kv=1.0),
        model="m",
        top_k=top_k,
    )
    roles = [entry["frontier_role"] for entry in payload["top_k"]]
    margins = [entry["margin_vs_rank1_pct"] for entry in payload["top_k"]]
    assert roles == ["max_throughput", "balanced_knee", "max_interactivity"]
    assert margins == [0.0, -16.0, -50.0]


def test_single_host_fit_excludes_ruled_out_candidates() -> None:
    """Single-host fit rejects ruled-out single-host layouts."""
    chip = CHIPS["tpu7x"]
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    impossible_scen = TrafficScenario(
        name="impossible-slo",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1,),
        kv_working_set_turns=1.0,
        tpot_slo_ms=0.0001,
    )
    splan = solve_serving_plan(
        geom,
        dt,
        chip,
        impossible_scen,
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert splan.recommended is None
    assert splan.candidates
    payload = build_solve_payload(splan, chip=chip, dtypes=dt)
    assert payload["single_host_fit"]["feasible"] is False
    assert payload["single_host_fit"]["status"] == "REJECTED"

    host_pool = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=8, dp=1, ep=8),
    )
    ruled_out_pt = ParetoPoint(
        mode="mixed",
        concurrency=1,
        total_chips=4,
        prefill_pool=None,
        decode_pool=host_pool,
        est_ttft_ms=50.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=50.0,
        token_throughput_per_chip=200.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=False,
        slo_verdict=SLO_VERDICT_RULED_OUT,
    )
    synthetic_plan = ServingPlan(
        model_layers=8,
        chip="tpu7x",
        scenario=impossible_scen,
        mode="mixed",
        recommended=None,
        frontier=(ruled_out_pt,),
        warnings=(),
        candidates=(ruled_out_pt,),
        has_moe=True,
        budget=4,
    )
    synth_payload = build_solve_payload(synthetic_plan, chip=chip, dtypes=dt)
    assert synth_payload["single_host_fit"]["feasible"] is False
    assert synth_payload["single_host_fit"]["status"] == "REJECTED"


def test_solve_payload_preserves_pd_ratio_and_partial_pins() -> None:
    """Solve payload preserves pd_ratio key and emits partial pins independently."""
    chip = CHIPS["tpu7x"]
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes(w_dense=1.0, w_expert=1.0, kv=1.0)
    scen = TrafficScenario(
        name="pin-check",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 4),
        kv_working_set_turns=1.0,
    )
    full_pin_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(
            mode="disagg",
            prefill_topology="2x2x1",
            decode_topology="2x2x1",
            pd_ratio=(1, 2),
        ),
    )
    assert full_pin_plan.pins == {
        "prefill_topology": "2x2x1",
        "decode_topology": "2x2x1",
        "pd_ratio": "1:2",
    }
    expected_pinned_chips = 12
    assert full_pin_plan.budget == expected_pinned_chips
    full_payload = build_solve_payload(full_pin_plan, chip=chip, dtypes=dt)
    assert full_payload["chip_budget"]["pd_ratio"] == "1:2"
    assert full_payload["chip_budget"]["prefill_topology"] == "2x2x1"
    assert full_payload["chip_budget"]["decode_topology"] == "2x2x1"
    assert full_payload["pins"] == full_pin_plan.pins

    partial_pin_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="mixed", max_chips=8, decode_topology="2x2x1"),
    )
    assert partial_pin_plan.pins == {"decode_topology": "2x2x1"}
    partial_payload = build_solve_payload(partial_pin_plan, chip=chip, dtypes=dt)
    assert partial_payload["chip_budget"]["decode_topology"] == "2x2x1"
    assert partial_payload["chip_budget"]["prefill_topology"] is None
    assert partial_payload["chip_budget"]["pd_ratio"] is None
    assert all(pt.decode_pool.topology == "2x2x1" for pt in partial_pin_plan.candidates)

    ratio_only_plan = solve_serving_plan(
        geom,
        dt,
        chip,
        scen,
        SolveOptions(mode="disagg", max_chips=12, pd_ratio=(1, 1)),
    )
    assert ratio_only_plan.pins == {"pd_ratio": "1:1"}
    ratio_payload = build_solve_payload(ratio_only_plan, chip=chip, dtypes=dt)
    assert ratio_payload["chip_budget"]["pd_ratio"] == "1:1"
    assert all(
        pt.prefill_pool is not None
        and pt.prefill_pool.instances == 1
        and pt.decode_pool.instances == 1
        for pt in ratio_only_plan.candidates
    )


def test_pool_label_joins_axes_with_x_and_omits_ones() -> None:
    """Verify format_parallelism and pool_label omit size-1 axes and join with x."""
    sample_pool = {
        "instances": 1,
        "topology": "2x2x1",
        "chips_per_instance": 4,
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "pp": 1,
        "pcp": 1,
        "kv_split": 4,
    }
    assert format_parallelism(sample_pool) == "TP8 x EP8 x DCP4"
    label = pool_label(sample_pool)
    assert label.endswith(": TP8 x EP8 x DCP4")

    all_one_pool = {
        "instances": 1,
        "topology": "1x1x1",
        "chips_per_instance": 1,
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "pcp": 1,
        "kv_split": 1,
    }
    assert format_parallelism(all_one_pool) == "TP1"


def test_format_submodule_parallelism_splits_attn_and_moe() -> None:
    """format_submodule_parallelism separates Attention, PP, and MoE sharding."""
    pcp_ep_pool = {"tp": 1, "dp": 1, "ep": 8, "pp": 1, "pcp": 8, "kv_split": 1}
    assert (
        format_submodule_parallelism(pcp_ep_pool, has_moe=True)
        == "Attn: PCP8 · MoE: EP8"
    )

    tp_sliced_pool = {"tp": 8, "dp": 1, "ep": 1, "pp": 1, "pcp": 1, "kv_split": 4}
    assert (
        format_submodule_parallelism(tp_sliced_pool, has_moe=True)
        == "Attn: TP8 x DCP4 · MoE: TP8"
    )

    pp_pcp_ep_pool = {"tp": 1, "dp": 1, "ep": 2, "pp": 4, "pcp": 2, "kv_split": 1}
    assert (
        format_submodule_parallelism(pp_pcp_ep_pool, has_moe=True)
        == "PP4 · Attn: PCP2 · MoE: EP2"
    )

    dense_tp_pool = {"tp": 8, "dp": 1, "ep": 1, "pp": 1, "pcp": 1, "kv_split": 1}
    assert format_submodule_parallelism(dense_tp_pool, has_moe=False) == "TP8"


def test_decode_group_key_distinguishes_prefill_mesh_families() -> None:
    """Disagg layouts with distinct prefill meshes must not collide in decode groups."""
    scen = TrafficScenario(
        name="disagg-prefill-diversity",
        isl=4096,
        osl=512,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(16,),
        kv_working_set_turns=1.0,
    )
    decode_pool = _sample_pool(
        role="decode",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=8, dp=1, ep=8),
    )
    prefill_pp = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=2, pp=4, pcp=2),
    )
    prefill_pcp = _sample_pool(
        role="prefill",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=1, dp=1, ep=8, pp=1, pcp=8),
    )
    pt_pp = ParetoPoint(
        mode="disagg",
        concurrency=16,
        total_chips=8,
        prefill_pool=prefill_pp,
        decode_pool=decode_pool,
        est_ttft_ms=120.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=180.0,
        token_throughput_per_chip=1600.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
    )
    pt_pcp = ParetoPoint(
        mode="disagg",
        concurrency=16,
        total_chips=8,
        prefill_pool=prefill_pcp,
        decode_pool=decode_pool,
        est_ttft_ms=110.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=175.0,
        token_throughput_per_chip=1550.0,
        p_mixed=0.0,
        binding="hbm",
        slo_met=True,
    )
    assert _decode_group_key(pt_pp) != _decode_group_key(pt_pcp)

    splan = ServingPlan(
        model_layers=16,
        chip="tpu7x",
        scenario=scen,
        mode="disagg",
        recommended=pt_pp,
        frontier=(pt_pp, pt_pcp),
        warnings=(),
        candidates=(pt_pp, pt_pcp),
        has_moe=True,
        budget=8,
    )
    skyline = pareto_skyline_by_concurrency(splan, k=3)
    expected_surviving = 2
    assert len(skyline[16]) == expected_surviving
    assert set(skyline[16]) == {pt_pp, pt_pcp}


def test_reference_layout_rank_compares_same_concurrency_peers() -> None:
    """Reference layout ranking compares strictly against same-concurrency peers."""
    scen = TrafficScenario(
        name="ref-rank-same-conc",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(1, 128),
        kv_working_set_turns=1.0,
    )
    pool_a = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=8, dp=1, ep=8),
    )
    pool_b = _sample_pool(
        role="mixed",
        cfg=_PoolConfig(topology="2x2x1", chips=4, tp=4, dp=2, ep=8),
    )

    def _pt(pool: PoolMeshPlan, conc: int, tput: float, tpot_ms: float) -> ParetoPoint:
        """Create a synthetic mixed ParetoPoint at one concurrency level."""
        return ParetoPoint(
            mode="mixed",
            concurrency=conc,
            total_chips=4,
            prefill_pool=None,
            decode_pool=pool,
            est_ttft_ms=50.0,
            est_tpot_ms=tpot_ms,
            est_interactivity_tok_s=1000.0 / tpot_ms,
            output_throughput_per_chip=tput * 0.2,
            token_throughput_per_chip=tput,
            p_mixed=0.0,
            binding="hbm",
            slo_met=True,
        )

    # Layout A only exists at c=1 and beats Layout B at c=1 on both throughput and TPOT,
    # even though Layout B achieves much higher throughput at c=128.
    pt_a_c1 = _pt(pool_a, 1, tput=400.0, tpot_ms=5.0)
    pt_b_c1 = _pt(pool_b, 1, tput=300.0, tpot_ms=8.0)
    pt_b_c128 = _pt(pool_b, 128, tput=9000.0, tpot_ms=12.0)

    splan = ServingPlan(
        model_layers=8,
        chip="tpu7x",
        scenario=scen,
        mode="mixed",
        recommended=pt_b_c128,
        frontier=(pt_a_c1, pt_b_c128),
        warnings=(),
        candidates=(pt_a_c1, pt_b_c1, pt_b_c128),
        has_moe=True,
        budget=4,
    )
    report = reference_layout_report(splan, (pt_b_c128, pt_a_c1), "TP8 x EP8", "")
    assert report["found"] is True
    expected_peer_layouts = 2
    assert report["candidates"] == expected_peer_layouts
    assert report["rank_by_estimated_throughput"] == 1
    assert report["rank_by_tpot_bound"] == 1
    assert report["in_top_k"] is True


def test_families_and_cli_reference_layout_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify parse_layout, bound families, explain_layout_rejection, and CLI flag."""
    parsed = parse_layout("P: PP4 x EP1 / D: TP8 x EP8 x DCP2")
    expected_pp = 4
    expected_tp = 8
    expected_dcp = 2
    assert parsed["prefill"]["pp"] == expected_pp
    assert parsed["decode"]["tp"] == expected_tp
    assert parsed["decode"]["kv_split"] == expected_dcp

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(SMALL_HYBRID_MOE_CFG), encoding="utf-8")
    rc = cli.main(
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
            "1,4",
            "--reference-layout",
            "TP8 x EP8",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "bound_families_by_concurrency" in out
    assert "1" in out["bound_families_by_concurrency"]
    assert "concurrency_top_k" in out
    assert out["reference"]["layout"] == "TP8 x EP8"
    assert out["reference"]["found"] is True
    assert out["reference"]["launch"] is not None

    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    reason = explain_layout_rejection(
        geom,
        parse_layout("P: PP4 / D: TP8 x EP8"),
        tuple(CHIPS["tpu7x"] and []),
        SolveOptions(mode="mixed", max_chips=4),
    )
    assert "disaggregated mode" in reason


def test_emit_kv_offload_uses_tpu_raiden_offloading_connector_and_unified_pool() -> (
    None
):
    """Emit TPURaidenOffloadingConnector, TPUMultiConnector, and unified pool env."""
    offload_gib = 64.0
    expected_cpu_bytes = round(offload_gib * UNIT_GIBI)
    mixed_hybrid_pool = PoolMeshPlan(
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
        hbm_used_gb=40.0,
        hbm_limit_gb=80.0,
        moe_dispatch="ag_ep",
        hbm_state_gb=2.0,
        max_num_seqs=16,
        max_model_len=8192,
        kv_offload_gb=offload_gib,
    )
    mixed_flags = pool_cli_args(
        mixed_hybrid_pool, kv_dtype="fp8", model_id="qwen3_5_moe"
    )
    assert "--kv-offloading-size" not in mixed_flags
    assert "--kv-transfer-config" in mixed_flags
    mixed_kv = json.loads(mixed_flags[mixed_flags.index("--kv-transfer-config") + 1])
    assert mixed_kv["kv_connector"] == "TPURaidenOffloadingConnector"
    assert (
        mixed_kv["kv_connector_module_path"] == "vllm_torchtpu.offload.raiden_connector"
    )
    assert mixed_kv["kv_role"] == "kv_both"
    assert mixed_kv["kv_connector_extra_config"] == {
        "cpu_bytes_to_use": expected_cpu_bytes
    }

    hybrid_pt = ParetoPoint(
        mode="mixed",
        concurrency=16,
        total_chips=4,
        prefill_pool=None,
        decode_pool=mixed_hybrid_pool,
        est_ttft_ms=100.0,
        est_tpot_ms=10.0,
        est_interactivity_tok_s=100.0,
        output_throughput_per_chip=200.0,
        token_throughput_per_chip=600.0,
        p_mixed=0.1,
        binding="hbm",
        slo_met=True,
    )
    hybrid_launch = point_launch_args(
        hybrid_pt,
        kv_dtype="fp8",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_id="qwen3_5_moe",
        is_hybrid=True,
    )
    assert any(
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1" in cmd
        for cmd in hybrid_launch["commands"]
    )
    assert not any("--kv-offloading-size" in cmd for cmd in hybrid_launch["commands"])

    pure_mla_pool = dc_replace(mixed_hybrid_pool, hbm_state_gb=0.0)
    mla_pt = dc_replace(hybrid_pt, decode_pool=pure_mla_pool)
    mla_launch = point_launch_args(
        mla_pt,
        kv_dtype="fp8",
        model="zai-org/GLM-5.2-FP8",
        model_id="glm_moe_dsa",
        is_hybrid=False,
    )
    assert not any(
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1" in cmd for cmd in mla_launch["commands"]
    )

    prefill_pool = PoolMeshPlan(
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
        hbm_used_gb=30.0,
        hbm_limit_gb=80.0,
        moe_dispatch="allreduce",
        hbm_state_gb=1.0,
        max_model_len=8192,
        pp_layer_partition=(8, 8, 8, 8, 8, 8, 8, 4),
        kv_offload_gb=offload_gib,
    )
    decode_pool = dc_replace(mixed_hybrid_pool, role="decode", kv_offload_gb=0.0)
    disagg_pt = dc_replace(
        hybrid_pt, mode="disagg", prefill_pool=prefill_pool, decode_pool=decode_pool
    )
    disagg_launch = point_launch_args(
        disagg_pt,
        kv_dtype="fp8",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_id="qwen3_5_moe",
        is_hybrid=True,
    )
    pre_args = disagg_launch["prefill_cli_args"]
    assert "--kv-offloading-size" not in pre_args
    multi_cfg = json.loads(pre_args[pre_args.index("--kv-transfer-config") + 1])
    assert multi_cfg["kv_connector"] == "TPUMultiConnector"
    assert (
        multi_cfg["kv_connector_module_path"]
        == "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
    )
    assert multi_cfg["kv_role"] == "kv_producer"
    children = multi_cfg["kv_connector_extra_config"]["connectors"]
    assert [c["kv_connector"] for c in children] == [
        "TPURaidenConnector",
        "TPURaidenOffloadingConnector",
    ]
    assert children[0]["kv_role"] == "kv_producer"
    assert children[1]["kv_role"] == "kv_both"
    assert children[1]["kv_connector_extra_config"] == {
        "cpu_bytes_to_use": expected_cpu_bytes
    }
    assert any(
        "VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4" in cmd
        and "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1" in cmd
        for cmd in disagg_launch["commands"]
    )
