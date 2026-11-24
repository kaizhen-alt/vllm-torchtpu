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
"""CLI unit tests for plan, explain, solve, emit, and ceiling subcommands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from tests.roofline.conftest import (
    SMALL_HYBRID_MOE_CFG,
    load_qwen3_8_bringup_config,
)
from vllm_torchtpu.roofline import (
    CHIPS,
    Dtypes,
    Parallel,
    ServingRequest,
    SolveOptions,
    TrafficScenario,
    UnsupportedArchitectureError,
    cli,
    from_hf_config,
    kv_cli_dtype_str,
    load_hf_config_with_identity,
    plan,
    solve_serving_plan,
    supports,
    top_layouts,
)
from vllm_torchtpu.roofline.cli import (
    _has_quantization_config,
    _parse_serve_cmd,
    _render_solve_ascii_table,
    _SolveRenderContext,
)
from vllm_torchtpu.roofline.constants import FP4_BYTES, UNIT_GIBI
from vllm_torchtpu.roofline.families import reference_layout_report

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


def _iter_subcommand_options(
    parser: argparse.ArgumentParser,
) -> list[tuple[str, tuple[str, ...], str | None]]:
    """Collect ``(subcommand, option_strings, help)`` across root and subparsers."""
    items: list[tuple[str, tuple[str, ...], str | None]] = []

    def _walk(cmd_name: str, subparser: argparse.ArgumentParser) -> None:
        """Walk parser actions via stdlib attributes lacking public accessors."""
        actions: Sequence[argparse.Action] = getattr(subparser, "_actions", ())
        for action in actions:
            if action.option_strings:
                items.append((cmd_name, tuple(action.option_strings), action.help))
            choices = getattr(action, "choices", None)
            if isinstance(choices, Mapping):
                for sub_name, child in choices.items():
                    if isinstance(child, argparse.ArgumentParser):
                        _walk(str(sub_name), child)

    _walk("root", parser)
    return items


def test_cli_plan_json_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify JSON output schema and hard floor for plan command."""
    expected_concurrency_rows = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "1024",
            "--concurrency",
            "1,32",
            "--json",
        ]
    )
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == expected_concurrency_rows
    expected_keys = {
        "c",
        "rows",
        "binding",
        "hard_floor_ms",
        "serial_estimate_ms",
        "tpot_floor_ms",
        "interactivity_ceil_tok_s",
        "output_tput_ceil_per_chip",
        "flag",
    }
    assert set(rows[0].keys()) == expected_keys
    assert rows[0]["c"] == 1
    assert rows[0]["hard_floor_ms"] <= rows[1]["hard_floor_ms"]


def test_cli_explain_spec_window_includes_draft_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command includes speculative draft modules."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "90036",
            "--spec-window",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    mods = json.loads(capsys.readouterr().out)
    names = [m["name"] for m in mods]
    assert "gqa" in names
    assert "draft_gqa" in names
    assert "draft_fc" in names


def test_cli_rejects_non_positive_tp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command exits with usage error on non-positive --tp."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["explain", "--config", str(cfg_path), "--tp", "0"])
    expected_usage_exit = 2
    assert exc_info.value.code == expected_usage_exit
    stderr_text = capsys.readouterr().err
    assert "--tp" in stderr_text
    assert ">= 1" in stderr_text


def test_cli_plan_axes_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify explain command with MoE expert parallel and FP8 KV cache."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    rc = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--context",
            "90036",
            "--moe",
            "ep",
            "--kv-dtype",
            "fp8",
            "--json",
        ]
    )
    assert rc == 0
    mods = json.loads(capsys.readouterr().out)
    total_phys = sum(m["hbm_bytes"] for m in mods)
    mod_map = {m["name"]: m["hbm_bytes"] for m in mods}
    expected_phys = (
        mod_map["gqa"] + mod_map["gdn"] + mod_map["moe"] + mod_map["lm_head"]
    )
    assert total_phys == pytest.approx(expected_phys, rel=1e-06)
    gqa_mod = next(m for m in mods if m["name"] == "gqa")
    assert gqa_mod.get("hbm_bytes_min") is not None
    expected_logical = total_phys - (gqa_mod["hbm_bytes"] - gqa_mod["hbm_bytes_min"])
    logical_total = total_phys - (gqa_mod["hbm_bytes"] - gqa_mod["hbm_bytes_min"])
    assert logical_total == pytest.approx(expected_logical, rel=1e-06)


def test_plan_requires_acceptance_length_for_spec_window(tmp_path: Path) -> None:
    """Verify acceptance length requirement when speculative window is positive."""
    geom = from_hf_config(QWEN35_CFG)
    with pytest.raises(
        ValueError, match="acceptance_length is required when spec_window > 0"
    ):
        plan(
            geom,
            CHIPS["tpu7x"],
            Parallel(tp=8),
            Dtypes(),
            ServingRequest(context=1024, concurrencies=[1], spec_window=3),
        )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "plan",
                "--config",
                str(cfg_path),
                "--tp",
                "8",
                "--spec-window",
                "3",
                "--context",
                "1024",
            ]
        )
    assert excinfo.value.code != 0


def test_cli_module_dunder_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure CLI help output is emitted and exits cleanly."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "usage:" in out.lower()


def test_cli_plan_and_explain_auto_resolve_hf_dtypes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify plan/explain inherit dtypes from HF config when flags omitted."""
    cfg_dict = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "torch_dtype": "float16",
        "quantization_config": {"quant_method": "fp8"},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")
    rc_plan = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--context",
            "1024",
            "--concurrency",
            "1,4",
            "--json",
        ]
    )
    assert rc_plan == 0
    plan_auto = json.loads(capsys.readouterr().out)
    expected_concs = 2
    assert len(plan_auto) == expected_concs
    rc_plan_fp8 = cli.main(
        [
            "plan",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--context",
            "1024",
            "--concurrency",
            "1,4",
            "--weight-dtype",
            "fp8",
            "--json",
        ]
    )
    assert rc_plan_fp8 == 0
    plan_fp8 = json.loads(capsys.readouterr().out)
    assert plan_auto[0]["tpot_floor_ms"] == pytest.approx(plan_fp8[0]["tpot_floor_ms"])
    rc_explain = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--batch",
            "4",
            "--context",
            "1024",
            "--json",
        ]
    )
    assert rc_explain == 0
    explain_auto = json.loads(capsys.readouterr().out)
    rc_explain_bf16 = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--batch",
            "4",
            "--context",
            "1024",
            "--weight-dtype",
            "bf16",
            "--json",
        ]
    )
    assert rc_explain_bf16 == 0
    explain_bf16 = json.loads(capsys.readouterr().out)
    auto_hbm = sum(float(row["hbm_bytes"]) for row in explain_auto)
    bf16_hbm = sum(float(row["hbm_bytes"]) for row in explain_bf16)
    assert auto_hbm < bf16_hbm * 0.75
    bf16_cfg = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "torch_dtype": "bfloat16",
    }
    bf16_cfg_path = tmp_path / "bf16_config.json"
    bf16_cfg_path.write_text(json.dumps(bf16_cfg), encoding="utf-8")
    rc_explain_bf16_auto = cli.main(
        [
            "explain",
            "--config",
            str(bf16_cfg_path),
            "--chip",
            "tpu7x",
            "--tp",
            "4",
            "--batch",
            "4",
            "--context",
            "1024",
            "--json",
        ]
    )
    assert rc_explain_bf16_auto == 0
    explain_bf16_auto = json.loads(capsys.readouterr().out)
    assert sum(float(row["hbm_bytes"]) for row in explain_bf16_auto) == pytest.approx(
        bf16_hbm
    )


def test_cli_chip_spec_hf_overrides_clean_error_and_dtype_label_preservation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --chip-spec, --hf-overrides, clean exit-2 error, and replace() dtypes."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    qwen38_path = fixtures_dir / "qwen3_8_flash_next_fp8.json"
    nvfp4_path = fixtures_dir / "qwen3_5_397b_nvfp4.json"

    # Without --hf-overrides, unmodified Qwen3.8 exits cleanly with code 2.
    rc_fail = cli.main(["plan", "--config", str(qwen38_path), "--tp", "8", "--json"])
    expected_error_rc = 2
    assert rc_fail == expected_error_rc
    assert "ple_layer_ids" in capsys.readouterr().err
    rc_missing = cli.main(
        ["plan", "--config", str(tmp_path / "missing.json"), "--tp", "8"]
    )
    assert rc_missing == expected_error_rc
    assert "error:" in capsys.readouterr().err.lower()

    # With --hf-overrides and --chip-spec, plan succeeds end-to-end.
    spec_path = tmp_path / "custom_chip.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
                "provenance": "projected",
                "base": "tpu7x",
                "name": "tpu7x_custom",
                "hbm_bw": 1e12,
            }
        )
    )
    rc_ok = cli.main(
        [
            "plan",
            "--config",
            str(qwen38_path),
            "--hf-overrides",
            '{"text_config": {"ple_layer_ids": []}}',
            "--chip-spec",
            str(spec_path),
            "--tp",
            "8",
            "--concurrency",
            "1",
            "--json",
        ]
    )
    assert rc_ok == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1

    # Partial CLI override preserves untouched NVFP4 expert dtype label via replace().
    nvfp4_cfg = json.loads(nvfp4_path.read_text())
    parsed = argparse.Namespace(
        weight_dtype=None,
        expert_weight_dtype=None,
        aux_dtype=None,
        kv_dtype="bf16",
        state_dtype=None,
        act=None,
    )
    dt_resolved = cli.resolve_dtypes_from_args(parsed, nvfp4_cfg)
    assert dt_resolved.w_expert_dtype == "nvfp4"
    assert dt_resolved.kv_dtype == "bf16"
    assert dt_resolved.act == "a16"


def test_cli_help_clarifies_l1_pruning_vs_uncalibrated_shortlisting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI help strings distinguish L1 pruning from uncalibrated shortlists."""
    for argv, expected_substrings in (
        (["--help"], ("L1 physical lower bounds", "uncalibrated serial-phase")),
        (
            ["solve", "--help"],
            ("Prune parallel layouts via L1 HBM bounds", "uncalibrated"),
        ),
        (["ceiling", "--help"], ("uncalibrated", "L1 candidate")),
        (["plan", "--help"], ("hard L1 physical step floors", "uncalibrated")),
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(argv)
        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out
        assert "winning" not in help_text.lower()
        for expected in expected_substrings:
            assert expected in help_text


def test_cli_solve_and_emit_support_chip_spec_hf_overrides_and_dtype_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify solve and emit honor --chip-spec, --hf-overrides, and dtype flags."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    qwen38_path = fixtures_dir / "qwen3_8_flash_next_fp8.json"
    spec_path = tmp_path / "custom_chip.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
                "provenance": "projected",
                "base": "tpu7x",
                "name": "tpu7x_override_test",
            }
        )
    )
    rc_emit = cli.main(
        [
            "emit",
            "--config",
            str(qwen38_path),
            "--hf-overrides",
            '{"text_config": {"ple_layer_ids": []}}',
            "--chip-spec",
            str(spec_path),
            "--kv-dtype",
            "fp8",
            "--act",
            "a16",
            "--isl",
            "1024",
            "--osl",
            "128",
            "--prefix-hit-rate",
            "0.0",
            "--accept-len",
            "1.0",
            "--think-time-s",
            "0.0",
            "--concurrency",
            "4",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--json",
        ]
    )
    assert rc_emit == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["serving_plan"]["chip"] == "tpu7x_override_test"
    assert "--hf-overrides" in payload["cli_args"]
    assert (
        payload["cli_args"][payload["cli_args"].index("--kv-cache-dtype") + 1] == "fp8"
    )


def _iter_subcommand_options(
    parser: argparse.ArgumentParser,
) -> list[tuple[str, tuple[str, ...], str | None]]:
    """Collect (subcommand, option_strings, help) tuples across parser."""
    out: list[tuple[str, tuple[str, ...], str | None]] = []
    actions: Sequence[argparse.Action] = getattr(parser, "_actions", ())
    for action in actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, Mapping) and all(
            isinstance(val, argparse.ArgumentParser) for val in choices.values()
        ):
            for sub_name, sub_parser in choices.items():
                sub_actions: Sequence[argparse.Action] = getattr(
                    sub_parser, "_actions", ()
                )
                out.extend(
                    (str(sub_name), tuple(sub_act.option_strings), sub_act.help)
                    for sub_act in sub_actions
                )
        else:
            out.append(("root", tuple(action.option_strings), action.help))
    return out


def test_cli_from_serve_and_bench_cmd_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI solve roundtrip from serve and bench command strings."""
    expected_isl = 4096
    expected_osl = 512
    max_chips = 4
    offload_gib = 32
    offload_bytes = offload_gib * int(UNIT_GIBI)
    raiden_offload_cfg = json.dumps(
        {
            "kv_connector": "TPURaidenOffloadingConnector",
            "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": offload_bytes},
        }
    )
    bf16_cfg_path = tmp_path / "synth_bf16_config.json"
    bf16_cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    cfg_path = tmp_path / "synth_config.json"
    cfg_path.write_text(
        json.dumps({**QWEN35_CFG, "quantization_config": {"quant_method": "fp8"}}),
        encoding="utf-8",
    )
    html_path = tmp_path / "roundtrip.html"
    rc = cli.main(
        [
            "solve",
            "--from-serve-cmd",
            (
                f"vllm serve {bf16_cfg_path} --quantization fp8 "
                "--kv-cache-dtype fp8 --tensor-parallel-size 8 "
                "--data-parallel-size 1 --enable-expert-parallel "
                "--gpu-memory-utilization 0.90 "
                f"--kv-transfer-config '{raiden_offload_cfg}'"
            ),
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name random "
                "--random-prefix-len 3072 --random-input-len 1024 "
                "--random-output-len 512 --max-concurrency 16"
            ),
            "--dcn-gbit-per-s",
            "200",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["isl"] == expected_isl
    assert payload["scenario"]["osl"] == expected_osl
    assert payload["scenario"]["prefix_hit_rate"] == pytest.approx(0.75)
    assert payload["scenario"]["concurrencies"] == [16]
    assert payload["scenario"]["kv_offload_gb"] == pytest.approx(float(offload_gib))
    assert payload["gpu_memory_utilization"] == pytest.approx(0.90)
    assert payload["chip_spec"]["dcn_bw"] == pytest.approx(25.0e9)
    assert payload["recommended"]["total_chips"] <= max_chips
    assert payload["winner_prefix_hit_range"] is not None
    assert "sweep_candidates" not in payload
    top_cmd = payload["top_k"][0]["launch"]["commands"][0]
    assert "--kv-offloading-size" not in top_cmd
    assert "TPURaidenOffloadingConnector" in top_cmd
    assert str(offload_bytes) in top_cmd
    html_text = html_path.read_text(encoding="utf-8")
    assert "sweep_candidates" not in html_text
    assert "Prefix-Hit Regime" in html_text
    assert ("crossover=" in html_text) or ("stable" in html_text)

    sonnet_isl = 550
    rc_sonnet = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name sonnet "
                "--sonnet-input-len 550 --sonnet-prefix-len 200 "
                "--sonnet-output-len 150 --max-concurrency 4"
            ),
            "--kv-offload-gb",
            "16",
            "--max-chips",
            "4",
            "--mode",
            "mixed",
            "--json",
        ]
    )
    assert rc_sonnet == 0
    sonnet_payload = json.loads(capsys.readouterr().out)
    assert sonnet_payload["scenario"]["isl"] == sonnet_isl
    assert sonnet_payload["scenario"]["prefix_hit_rate"] == pytest.approx(200.0 / 550.0)
    assert sonnet_payload["scenario"]["kv_offload_gb"] == pytest.approx(16.0)


def test_cli_forwards_spec_tokens_and_token_caps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify CLI forwards spec_tokens and token caps from serve cmd."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    expected_spec_tokens = 3
    expected_prefill_chunk = 2048
    spec_json = json.dumps({"method": "mtp", "num_speculative_tokens": 3})
    cmd = [
        "solve",
        "--config",
        str(cfg_path),
        "--from-serve-cmd",
        (
            f"vllm serve {cfg_path} --tensor-parallel-size 8 --quantization fp8 "
            "--max-model-len 16384 --max-num-batched-tokens 2048 "
            f"--speculative-config '{spec_json}'"
        ),
        "--json",
    ]
    rc = cli.main(cmd)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scenario"]["spec_tokens"] == expected_spec_tokens
    assert payload["scenario"]["prefill_chunk"] == expected_prefill_chunk
    launch_args = payload["top_k"][0]["launch"]["cli_args"]
    assert "--max-model-len" in launch_args
    assert "16384" in launch_args
    assert "--speculative-config" in launch_args


def test_every_cli_flag_has_help(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify every CLI flag has help text with unit/scope, aliases, and assumptions."""
    parser = cli.build_parser()
    sub_opts = _iter_subcommand_options(parser)
    assert len(sub_opts) > 0
    for sub_name, option_strings, raw_help in sub_opts:
        if not option_strings or "-h" in option_strings:
            continue
        assert raw_help is not None, f"{sub_name} {option_strings} missing help"
        help_str = raw_help.strip()
        assert bool(help_str), f"{sub_name} {option_strings} empty help"
        assert help_str != "==SUPPRESS=="
        assert "scope:" in help_str.lower()
        assert "unit:" in help_str.lower()

    rc_assump = cli.main(["--dcn-gbps", "200", "--assumptions", "--json"])
    assert rc_assump == 0
    assump_payload = json.loads(capsys.readouterr().out)
    assert "catalog" in assump_payload
    assert assump_payload["catalog"]["dcn_bw"]["value"] == pytest.approx(25.0e9)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")
    rc_alias = cli.main(
        [
            "explain",
            "--config",
            str(cfg_path),
            "--tp",
            "8",
            "--ici-bw-gbytes-per-s",
            "250",
            "--json",
        ]
    )
    assert rc_alias == 0
    _ = capsys.readouterr()


def test_solve_exits_nonzero_on_empty_frontier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve exits 1 on empty frontier and 2 on invalid subhost or loader err."""
    expected_error_exit = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(QWEN35_CFG), encoding="utf-8")

    rc_empty = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_empty = capsys.readouterr()
    assert rc_empty == 1
    assert "WARNING:" in captured_empty.err

    rc_subhost = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--max-chips",
            "1",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_subhost = capsys.readouterr()
    assert rc_subhost == expected_error_exit
    assert "error:" in captured_subhost.err

    rc_err = cli.main(
        [
            "solve",
            "--config",
            str(tmp_path / "missing_config.json"),
            "--max-chips",
            "4",
            "--example-scenario",
            "chat-8k1k",
        ]
    )
    captured_err = capsys.readouterr()
    assert rc_err == expected_error_exit
    err_lines = captured_err.err.strip().splitlines()
    assert len(err_lines) == 1
    assert err_lines[0].startswith("error:")


def test_cli_solve_top_k_ascii_table_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify --top-k defaults to 5, drives JSON and ASCII tables, and rejects <=0."""
    expected_default_k = 5
    expected_k_2 = 2
    expected_err_code = 2
    cfg_path = tmp_path / "small_qwen35_fp8.json"
    raw_text_cfg = SMALL_HYBRID_MOE_CFG["text_config"]
    assert isinstance(raw_text_cfg, dict)
    cfg_path.write_text(
        json.dumps(
            {
                **SMALL_HYBRID_MOE_CFG,
                "text_config": {
                    **raw_text_cfg,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        ),
        encoding="utf-8",
    )

    rc_5 = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "coding-agent",
            "--mode",
            "disagg",
            "--max-chips",
            "8",
            "--kv-offload-gb",
            "64",
            "--json",
        ]
    )
    assert rc_5 == 0
    payload_5 = json.loads(capsys.readouterr().out)
    assert 1 <= len(payload_5["top_k"]) <= expected_default_k
    assert all(
        entry["frontier_role"] != "off_frontier_alt" for entry in payload_5["top_k"]
    )

    rc_2_txt = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "8",
            "--top-k",
            "2",
        ]
    )
    assert rc_2_txt == 0
    out_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert "rank" in out_lines[0]
    assert "role" in out_lines[0]
    assert len(out_lines) == 1 + expected_k_2

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "solve",
                "--config",
                str(cfg_path),
                "--example-scenario",
                "coding-agent",
                "--max-chips",
                "8",
                "--top-k",
                "0",
            ]
        )
    assert excinfo.value.code == expected_err_code
    _ = capsys.readouterr()


def test_cli_solve_rejects_nonpositive_max_instances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify solve rejects --max-instances <= 0 with parser error exit code."""
    expected_err_code = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(SMALL_HYBRID_MOE_CFG), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "solve",
                "--config",
                str(cfg_path),
                "--chip",
                "tpu7x",
                "--example-scenario",
                "coding-agent",
                "--max-instances",
                "0",
            ]
        )
    assert excinfo.value.code == expected_err_code
    assert "--max-instances" in capsys.readouterr().err


def test_cli_solve_ascii_and_html_and_reference_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify ASCII table, HTML report, and --reference-layout ranking."""
    config_path = tmp_path / "qwen35_fp8.json"
    raw_text_config = QWEN35_CFG["text_config"]
    assert isinstance(raw_text_config, dict)
    config_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text_config,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        ),
        encoding="utf-8",
    )
    html_path = tmp_path / "report.html"
    rc_match = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--isl",
            "105000",
            "--osl",
            "1000",
            "--prefix-hit-rate",
            "0.95",
            "--accept-len",
            "3.39",
            "--think-time-s",
            "0.0",
            "--concurrency",
            "8",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--top-k",
            "1",
            "--reference-layout",
            "TP8 x EP8",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc_match == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reference"]["found"] is True
    assert "Search Budget:" in html_path.read_text(encoding="utf-8")

    rc_rej = cli.main(
        [
            "solve",
            "--config",
            str(config_path),
            "--isl",
            "105000",
            "--osl",
            "1000",
            "--prefix-hit-rate",
            "0.95",
            "--accept-len",
            "3.39",
            "--think-time-s",
            "0.0",
            "--concurrency",
            "8",
            "--mode",
            "mixed",
            "--max-chips",
            "4",
            "--reference-layout",
            "TP3",
            "--json",
        ]
    )
    assert rc_rej == 0
    rej_payload = json.loads(capsys.readouterr().out)
    assert rej_payload["reference"]["found"] is False
    assert "TP3" in rej_payload["reference"]["rejected_by"]


def test_assumptions_catalog_and_collective_latency_calibration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify --assumptions catalog and --collective-latency-us calibration scope."""
    rc_json = cli.main(["--assumptions", "--json"])
    assert rc_json == 0
    assump_json = json.loads(capsys.readouterr().out)
    assert assump_json["catalog"]
    assert assump_json["solver_assumptions"]

    rc_text = cli.main(["--assumptions"])
    assert rc_text == 0
    assert "Solver assumptions" in capsys.readouterr().out

    raw_text = SMALL_HYBRID_MOE_CFG["text_config"]
    assert isinstance(raw_text, dict)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                **SMALL_HYBRID_MOE_CFG,
                "text_config": {
                    **raw_text,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        ),
        encoding="utf-8",
    )
    base_argv = [
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
        "4",
        "--json",
    ]
    assert cli.main(base_argv) == 0
    base_payload = json.loads(capsys.readouterr().out)
    assert base_payload["provenance"]["calibrated"] is False
    assert base_payload["provenance"]["supplied_values"] == []

    cal_us = 50.0
    assert (
        cli.main([*base_argv[:-1], "--collective-latency-us", str(cal_us), "--json"])
        == 0
    )
    cal_payload = json.loads(capsys.readouterr().out)
    assert cal_payload["provenance"]["calibrated"] is True
    base_rec = base_payload["recommended"]
    base_pool = base_rec["decode_pool"]
    cal_peer = next(
        cand
        for cand in cal_payload["candidates"]
        if cand["concurrency"] == base_rec["concurrency"]
        and cand["decode_pool"]["tp"] == base_pool["tp"]
        and cand["decode_pool"]["dp"] == base_pool["dp"]
        and cand["decode_pool"]["ep"] == base_pool["ep"]
        and cand["decode_pool"]["kv_split"] == base_pool["kv_split"]
        and cand["decode_pool"]["moe_dispatch"] == base_pool["moe_dispatch"]
    )
    assert cal_peer["bound_tpot_ms"] == pytest.approx(
        base_rec["bound_tpot_ms"], rel=1e-12
    )
    assert cal_peer["est_tpot_ms"] >= base_rec["est_tpot_ms"]


def test_cli_invalid_scenario_and_overrides_exit_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid --concurrency or --hf-overrides exit 2 cleanly without tracebacks."""
    expected_error_rc = 2
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(SMALL_HYBRID_MOE_CFG), encoding="utf-8")

    rc_bad_conc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "4",
            "--concurrency",
            "abc",
        ]
    )
    assert rc_bad_conc == expected_error_rc
    assert "error:" in capsys.readouterr().err.lower()

    rc_zero_conc = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "4",
            "--concurrency",
            "0",
        ]
    )
    assert rc_zero_conc == expected_error_rc
    assert "error:" in capsys.readouterr().err.lower()

    rc_bad_ovr = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--hf-overrides",
            "123",
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "4",
        ]
    )
    assert rc_bad_ovr == expected_error_rc
    assert "error:" in capsys.readouterr().err.lower()


def test_cli_emit_uses_from_bench_cmd_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Emit uses --from-bench-cmd --model instead of falling back to <MODEL>."""
    model_dir = tmp_path / "qwen35_local"
    model_dir.mkdir()
    raw_text = SMALL_HYBRID_MOE_CFG["text_config"]
    assert isinstance(raw_text, dict)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                **SMALL_HYBRID_MOE_CFG,
                "text_config": {
                    **raw_text,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        ),
        encoding="utf-8",
    )
    rc = cli.main(
        [
            "emit",
            "--from-bench-cmd",
            (
                f"vllm bench serve --model {model_dir} "
                "--random-input-len 1024 --random-output-len 128 --max-concurrency 8"
            ),
            "--max-chips",
            "4",
            "--mode",
            "mixed",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["commands"]
    assert all("<MODEL>" not in cmd for cmd in payload["commands"])
    assert any(str(model_dir) in cmd for cmd in payload["commands"])


def test_cli_parse_serve_cmd_roundtrips_tpu_flags_and_raiden_offload() -> None:
    """_parse_serve_cmd parses TPU flags, Raiden offload, and TPUMultiConnector."""
    offload_gib = 10
    offload_bytes = offload_gib * int(UNIT_GIBI)
    expected_tp = 8
    expected_pcp = 2
    expected_dcp = 4
    expected_interleave = 128
    expected_nnodes = 2

    standalone_kv = json.dumps(
        {
            "kv_connector": "TPURaidenOffloadingConnector",
            "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": offload_bytes},
        }
    )
    standalone_cmd = (
        "vllm serve Qwen/Qwen3.5-397B-A17B-FP8 -tp 8 -ep -pcp 2 "
        "--decode-context-parallel-size 4 --cp-kv-cache-interleave-size 128 "
        "--no-async-scheduling --nnodes 2 --headless "
        f"--kv-transfer-config '{standalone_kv}'"
    )
    parsed = _parse_serve_cmd(standalone_cmd)
    assert parsed["tp"] == expected_tp
    assert parsed["enable_expert_parallel"] is True
    assert parsed["pcp"] == expected_pcp
    assert parsed["dcp"] == expected_dcp
    assert parsed["cp_kv_cache_interleave_size"] == expected_interleave
    assert parsed["async_scheduling"] is False
    assert parsed["nnodes"] == expected_nnodes
    assert parsed["headless"] is True
    assert parsed["kv_offload_gb"] == pytest.approx(float(offload_gib))

    multi_gib = 24
    multi_bytes = multi_gib * int(UNIT_GIBI)
    multi_kv = json.dumps(
        {
            "kv_connector": "TPUMultiConnector",
            "kv_connector_module_path": (
                "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
            ),
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "connectors": [
                    {
                        "kv_connector": "TPURaidenConnector",
                        "kv_connector_module_path": (
                            "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
                        ),
                        "kv_role": "kv_producer",
                    },
                    {
                        "kv_connector": "TPURaidenOffloadingConnector",
                        "kv_connector_module_path": (
                            "vllm_torchtpu.offload.raiden_connector"
                        ),
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"cpu_bytes_to_use": multi_bytes},
                    },
                ]
            },
        }
    )
    multi_cmd = (
        "vllm serve Qwen/Qwen3.5-397B-A17B-FP8 -tp 8 --enable-expert-parallel "
        f"--kv-transfer-config '{multi_kv}'"
    )
    parsed_multi = _parse_serve_cmd(multi_cmd)
    assert parsed_multi["kv_connector"] == "TPUMultiConnector"
    assert parsed_multi["kv_role"] == "kv_producer"
    assert parsed_multi["kv_offload_gb"] == pytest.approx(float(multi_gib))


def test_cli_inspect_fail_closed_and_canonical_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inspect emits canonical identity, supports ModelOpt + overrides, fails closed."""
    expected_error_rc = 2
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    glm_nvfp4 = fixtures_dir / "glm_5_2_nvfp4.json"
    qwen38_path = fixtures_dir / "qwen3_8_flash_next_fp8.json"
    dsv4_path = fixtures_dir / "deepseek_v4_pro.json"

    # Canonical ModelOpt NVFP4 fixture passes tier0_ready with canonical_id.
    rc_glm = cli.main(["inspect", "--config", str(glm_nvfp4), "--json"])
    assert rc_glm == 0
    glm_payload = json.loads(capsys.readouterr().out)
    assert glm_payload["onboarding_tier"] == "tier0_ready"
    assert glm_payload["canonical_id"] == "glm_moe_dsa"
    assert glm_payload["unmapped_structural_keys"] == []
    assert _has_quantization_config(json.loads(glm_nvfp4.read_text()))

    # Qwen3.8 fails closed unmodified and becomes tier0_ready with --hf-overrides.
    rc_q38_raw = cli.main(["inspect", "--config", str(qwen38_path), "--json"])
    assert rc_q38_raw == expected_error_rc
    q38_raw = json.loads(capsys.readouterr().out)
    assert q38_raw["onboarding_tier"] != "tier0_ready"

    rc_q38_ovr = cli.main(
        [
            "inspect",
            "--config",
            str(qwen38_path),
            "--hf-overrides",
            '{"text_config": {"ple_layer_ids": []}}',
            "--json",
        ]
    )
    assert rc_q38_ovr == 0
    q38_ovr = json.loads(capsys.readouterr().out)
    assert q38_ovr["onboarding_tier"] == "tier0_ready"
    assert q38_ovr["canonical_id"] == "qwen4_exp"

    # Unlisted / unsupported model fails closed with exit 2 and zero bypass flag.
    rc_dsv4 = cli.main(["inspect", "--config", str(dsv4_path), "--json"])
    assert rc_dsv4 == expected_error_rc
    dsv4_payload = json.loads(capsys.readouterr().out)
    assert dsv4_payload["onboarding_tier"] != "tier0_ready"
    assert dsv4_payload["canonical_id"] is None

    parser = cli.build_parser()
    all_opts = {
        opt
        for _sub, opt_strs, _help in _iter_subcommand_options(parser)
        for opt in opt_strs
    }
    assert "--allow-unlisted-model-type" not in all_opts


def _kv_flag_value(cli_args: Sequence[str]) -> str:
    """Return value following ``--kv-cache-dtype`` in an emitted flag list."""
    return str(cli_args[cli_args.index("--kv-cache-dtype") + 1])


def test_cli_runtime_profile_hardware_enables_qwen4_and_fp4_kv_while_main_blocks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify --runtime-profile gates 4-bit KV and experimental bringup across CLI."""
    expected_error_rc = 2
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    qwen38_path = fixtures_dir / "qwen3_8_flash_next_fp8.json"
    qwen35_path = tmp_path / "qwen35_fp8.json"
    raw_text = QWEN35_CFG["text_config"]
    assert isinstance(raw_text, dict)
    qwen35_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        ),
        encoding="utf-8",
    )

    # Default 'main' profile blocks unmodified qwen4_exp on inspect/plan/solve.
    assert (
        cli.main(["inspect", "--config", str(qwen38_path), "--json"])
        == expected_error_rc
    )
    assert json.loads(capsys.readouterr().out)["onboarding_tier"] != "tier0_ready"
    for blocked_argv in (
        ["plan", "--config", str(qwen38_path), "--tp", "8", "--concurrency", "1"],
        [
            "solve",
            "--config",
            str(qwen38_path),
            "--example-scenario",
            "chat-8k1k",
            "--max-chips",
            "4",
        ],
    ):
        assert cli.main(blocked_argv) == expected_error_rc
        assert "error:" in capsys.readouterr().err.lower()

    # Hardware profile admits unmodified qwen4_exp across all 6 subcommands.
    assert (
        cli.main(
            [
                "inspect",
                "--config",
                str(qwen38_path),
                "--runtime-profile",
                "hardware",
                "--json",
            ]
        )
        == 0
    )
    insp_hw = json.loads(capsys.readouterr().out)
    assert insp_hw["onboarding_tier"] == "tier0_ready"
    assert insp_hw["canonical_id"] == "qwen4_exp"
    assert insp_hw["runtime_profile"] == "hardware"

    for hw_argv in (
        [
            "plan",
            "--config",
            str(qwen38_path),
            "--runtime-profile",
            "hardware",
            "--tp",
            "8",
            "--concurrency",
            "1",
            "--json",
        ],
        [
            "explain",
            "--config",
            str(qwen38_path),
            "--runtime-profile",
            "hardware",
            "--tp",
            "8",
            "--json",
        ],
        [
            "solve",
            "--config",
            str(qwen38_path),
            "--runtime-profile",
            "hardware",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
            "--json",
        ],
        [
            "emit",
            "--config",
            str(qwen38_path),
            "--runtime-profile",
            "hardware",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
            "--json",
        ],
        [
            "ceiling",
            "--config",
            str(qwen38_path),
            "--runtime-profile",
            "hardware",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
            "--json",
        ],
    ):
        assert cli.main(hw_argv) == 0
        assert json.loads(capsys.readouterr().out)

    # 4-bit KV dtype is blocked on 'main' (both --kv-dtype and --from-serve-cmd).
    for fp4_blocked in (
        [
            "plan",
            "--config",
            str(qwen35_path),
            "--kv-dtype",
            "fp4",
            "--tp",
            "8",
            "--concurrency",
            "1",
        ],
        [
            "explain",
            "--config",
            str(qwen35_path),
            "--kv-dtype",
            "fp4",
            "--tp",
            "8",
        ],
        [
            "solve",
            "--config",
            str(qwen35_path),
            "--kv-dtype",
            "fp4",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
        ],
        [
            "emit",
            "--config",
            str(qwen35_path),
            "--kv-dtype",
            "fp4",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
        ],
        [
            "ceiling",
            "--config",
            str(qwen35_path),
            "--kv-dtype",
            "fp4",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
        ],
        [
            "solve",
            "--from-serve-cmd",
            (
                f"vllm serve {qwen35_path} --tensor-parallel-size 8 "
                "--kv-cache-dtype turboquant_4bit_nc"
            ),
            "--example-scenario",
            "coding-agent",
        ],
    ):
        assert cli.main(fp4_blocked) == expected_error_rc
        assert "not admitted" in capsys.readouterr().err.lower()

    # Under --runtime-profile hardware, fp4 KV emits 'fp4' without clamping to 'fp8'.
    assert (
        cli.main(
            [
                "emit",
                "--config",
                str(qwen35_path),
                "--runtime-profile",
                "hardware",
                "--kv-dtype",
                "fp4",
                "--example-scenario",
                "coding-agent",
                "--max-chips",
                "4",
                "--json",
            ]
        )
        == 0
    )
    emit_fp4 = json.loads(capsys.readouterr().out)
    assert _kv_flag_value(emit_fp4["cli_args"]) == "fp4"

    assert (
        cli.main(
            [
                "solve",
                "--runtime-profile",
                "hardware",
                "--from-serve-cmd",
                (
                    f"vllm serve {qwen35_path} --tensor-parallel-size 8 "
                    "--kv-cache-dtype turboquant_4bit_nc"
                ),
                "--example-scenario",
                "coding-agent",
                "--json",
            ]
        )
        == 0
    )
    solve_tq = json.loads(capsys.readouterr().out)
    assert solve_tq["top_k"]
    assert _kv_flag_value(solve_tq["top_k"][0]["launch"]["cli_args"]) == "fp4"

    dt_hw = cli.resolve_dtypes_from_args(
        argparse.Namespace(kv_dtype="fp4"),
        json.loads(qwen35_path.read_text(encoding="utf-8")),
        runtime_profile="hardware",
    )
    assert dt_hw.kv == pytest.approx(FP4_BYTES)
    assert dt_hw.kv_dtype == "fp4"

    for bad_prof_argv in (
        ["inspect", "--config", str(qwen35_path), "--runtime-profile", "unknown"],
        [
            "solve",
            "--config",
            str(qwen35_path),
            "--runtime-profile",
            "unknown",
            "--example-scenario",
            "coding-agent",
            "--max-chips",
            "4",
        ],
    ):
        assert cli.main(bad_prof_argv) == expected_error_rc
        assert "runtime_profile" in capsys.readouterr().err


def test_cli_omitted_think_time_uses_none_open_loop_without_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify omitting --think-time-s passes think_time_s=None and renders cleanly."""
    cfg_path = tmp_path / "qwen35_fp8.json"
    raw_text = QWEN35_CFG["text_config"]
    assert isinstance(raw_text, dict)
    cfg_path.write_text(
        json.dumps(
            {
                **QWEN35_CFG,
                "text_config": {
                    **raw_text,
                    "quantization_config": {"quant_method": "fp8"},
                },
            }
        ),
        encoding="utf-8",
    )
    html_path = tmp_path / "open_loop.html"
    rc_custom = cli.main(
        [
            "solve",
            "--config",
            str(cfg_path),
            "--isl",
            "2048",
            "--osl",
            "256",
            "--prefix-hit-rate",
            "0.5",
            "--accept-len",
            "1.0",
            "--concurrency",
            "8",
            "--max-chips",
            "4",
            "--html",
            str(html_path),
            "--json",
        ]
    )
    assert rc_custom == 0
    custom_payload = json.loads(capsys.readouterr().out)
    assert custom_payload["scenario"]["think_time_s"] is None
    assert custom_payload["frontier"]
    assert html_path.read_text(encoding="utf-8")

    rc_cmd = cli.main(
        [
            "solve",
            "--from-serve-cmd",
            f"vllm serve {cfg_path} --tensor-parallel-size 8",
            "--from-bench-cmd",
            (
                "vllm bench serve --dataset-name random "
                "--random-input-len 1024 --random-output-len 128 --max-concurrency 8"
            ),
            "--json",
        ]
    )
    assert rc_cmd == 0
    cmd_payload = json.loads(capsys.readouterr().out)
    assert cmd_payload["scenario"]["think_time_s"] is None
    assert cmd_payload["frontier"]


def test_cli_reference_layout_prints_ruled_out_not_unenumerated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print 'enumerated, ruled out by SLO' when ref_report is ruled_out."""
    q35_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_5_397b_fp8.json"
    cfg_path = tmp_path / "small_hybrid.json"
    cfg_path.write_text(json.dumps(SMALL_HYBRID_MOE_CFG), encoding="utf-8")

    # Partial SLO miss: TP2 x DP4 x EP8 ruled out while frontier stays non-empty.
    rc = cli.main(
        [
            "solve",
            "--config",
            str(q35_path),
            "--isl",
            "8192",
            "--osl",
            "1024",
            "--prefix-hit-rate",
            "0.5",
            "--accept-len",
            "1.0",
            "--concurrency",
            "32",
            "--tpot-slo-ms",
            "1.0",
            "--max-chips",
            "16",
            "--max-instances",
            "1",
            "--mode",
            "mixed",
            "--reference-layout",
            "TP2 x DP4 x EP8",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "enumerated, ruled out by SLO" in out
    assert "not enumerated" not in out

    # All-ruled-out frontier: found=True and ruled_out=True must still print ruled out.
    geom = from_hf_config(SMALL_HYBRID_MOE_CFG)
    dt = Dtypes()
    chip = CHIPS["tpu7x"]
    scen = TrafficScenario(
        name="ruled-out-ascii",
        isl=2048,
        osl=256,
        prefix_hit_rate=0.5,
        accept_len=1.0,
        think_time_s=0.0,
        concurrencies=(8,),
        ttft_slo_ms=0.01,
        tpot_slo_ms=0.01,
        kv_working_set_turns=1.0,
    )
    splan = solve_serving_plan(
        geom, dt, chip, scen, SolveOptions(mode="mixed", max_chips=4)
    )
    assert splan.candidates
    ref = reference_layout_report(
        splan, top_layouts(splan, 5), "TP1 x DP8 x EP8", "not enumerated"
    )
    assert ref["found"] is True
    assert ref["ruled_out"] is True
    rctx = _SolveRenderContext(
        src=str(cfg_path),
        chip=chip,
        dtypes=dt,
        topologies=cli.topologies_for_chip(chip),
    )
    _render_solve_ascii_table(splan, 5, rctx, ref_report=ref)
    ascii_out = capsys.readouterr().out
    assert "enumerated, ruled out by SLO" in ascii_out
    assert "not enumerated" not in ascii_out


def test_geometry_supports_and_loader_accept_hardware_runtime_profile(
    tmp_path: Path,
) -> None:
    """Thread runtime_profile cleanly through supports, loader, and kv_cli_dtype_str."""
    q38_path = Path(__file__).resolve().parent / "fixtures" / "qwen3_8_flash_fp8.json"
    if not q38_path.exists():
        q38_path = tmp_path / "qwen3_8.json"
        raw_cfg = load_qwen3_8_bringup_config()
        raw_cfg["text_config"]["ple_layer_ids"] = [2]
        q38_path.write_text(json.dumps(raw_cfg), encoding="utf-8")
    else:
        raw_cfg = json.loads(q38_path.read_text(encoding="utf-8"))

    with pytest.raises(UnsupportedArchitectureError, match="ple_layer_ids"):
        supports(
            raw_cfg,
            model_ref="Qwen/Qwen3.8-Flash-Next-FP8",
            runtime_profile="main",
        )
    supports(
        raw_cfg,
        model_ref="Qwen/Qwen3.8-Flash-Next-FP8",
        runtime_profile="hardware",
    )

    cfg, identity = load_hf_config_with_identity(
        str(q38_path),
        runtime_profile="hardware",
    )
    assert identity.canonical_id == "qwen4_exp"
    assert isinstance(cfg, Mapping)
    assert kv_cli_dtype_str(Dtypes(kv=2.0), default_bf16="bfloat16") == "bfloat16"
    assert kv_cli_dtype_str(Dtypes(kv=1.0), default_bf16="bfloat16") == "fp8"
    assert (
        kv_cli_dtype_str(Dtypes(kv=FP4_BYTES, kv_dtype="fp4"), default_bf16="bfloat16")
        == "fp4"
    )
