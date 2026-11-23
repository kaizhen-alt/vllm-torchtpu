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
from pathlib import Path
from typing import Literal

import pytest

from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    ParetoPoint,
    PoolMeshPlan,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    cli,
    emit_ceiling_rows,
    emit_launch_args,
    from_hf_config,
    pool_cli_args,
    solve_serving_plan,
)
from vllm_torchtpu.roofline.constants import NO_FEASIBLE_LAYOUT_WARNING
from vllm_torchtpu.roofline.emit import (
    _MODEL_PLACEHOLDER,
    _UNKNOWN_MODEL_WARNING,
)
from vllm_torchtpu.roofline.hf_loader import dtypes_from_hf_config as _dtypes_from_cfg
from vllm_torchtpu.roofline.memory import (
    SeqKvLayout as _SeqKvLayout,
)
from vllm_torchtpu.roofline.memory import (
    hbm_breakdown_per_core_bytes as _hbm_breakdown,
)
from vllm_torchtpu.roofline.modules import Parallel as _Parallel


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
        p90_ttft_ms=180.0,
        p90_tpot_ms=12.0,
        p90_interactivity_tok_s=83.3,
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
        p90_ttft_ms=140.0,
        p90_tpot_ms=9.5,
        p90_interactivity_tok_s=105.2,
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
        "p90_interactivity_tok_s",
        "token_throughput_per_chip",
        "output_throughput_per_chip",
        "p90_ttft_ms",
        "p90_tpot_ms",
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
    assert "p90_interactivity_tok_s" in out_ceil[0]


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
    assert pt_256.p90_ttft_ms > pt_15.p90_ttft_ms


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
            p90_ttft_ms=100.0,
            p90_tpot_ms=10.0,
            p90_interactivity_tok_s=100.0,
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
        p90_ttft_ms=50.0,
        p90_tpot_ms=5.0,
        p90_interactivity_tok_s=200.0,
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

    # Disaggregated MTP with spec_k=3 must not emit PCP > 1 prefill pools.
    disagg_spec = solve_serving_plan(
        geom, dt, CHIPS["tpu7x"], spec_scen, SolveOptions(mode="disagg", max_chips=16)
    )
    out_disagg = emit_launch_args(
        disagg_spec, dtypes=dt, model="Qwen/Qwen3.5-397B-A17B-FP8"
    )
    assert "--prefill-context-parallel-size" not in out_disagg["prefill_cli_args"]
