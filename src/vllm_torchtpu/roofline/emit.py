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
"""Render vLLM serve CLI flags and uncalibrated L1 candidate curves from plans."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict

from .chips import CHIPS
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_PCP_KV_INTERLEAVE_SIZE,
    KV_DTYPE_BYTES,
    NO_FEASIBLE_LAYOUT_WARNING,
)
from .solver import classify_multihost_mp_mode

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chips import ChipSpec
    from .modules import Dtypes
    from .solver import PoolMeshPlan, ServingPlan

# Printed in a command in place of a model that ``vllm serve`` cannot load.
_MODEL_PLACEHOLDER = "<MODEL>"
_HOST0_ADDR_PLACEHOLDER = "<HOST_0_INTERNAL_IP>"
_DEFAULT_MASTER_PORT = "29500"
_DEFAULT_DP_RPC_PORT = "13345"
_QWEN4_BRINGUP_OVERRIDES = '{"text_config": {"ple_layer_ids": []}}'
_RAIDEN_CONNECTOR_NAME = "TPURaidenConnector"
_RAIDEN_MODULE_PATH = "vllm_torchtpu.distributed.kv_transfer.tpu_connector"
_DISAGG_WIRING_WARNING = (
    "Disaggregated launch requires a Raiden controller "
    "(TPU_USE_RAIDEN_KV_CACHE_MANAGER=1, TPU_RAIDEN_CONTROLLER_ADDRESS) "
    "and a proxy router; see docs/user_guide/disaggregated_serving.md."
)
_UNKNOWN_MODEL_WARNING = (
    f"The model to serve is not known: replace {_MODEL_PLACEHOLDER} in each "
    "command with a model ID or a model directory."
)


class LaunchPayload(TypedDict):
    """Structured vLLM serve CLI payload emitted for the top-ranked L1 candidate."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]
    warnings: list[str]
    serving_plan: Mapping[str, object]


class CeilingRow(TypedDict):
    """Single uncalibrated L1 candidate row exported by ``emit_ceiling_rows``."""

    p90_interactivity_tok_s: float
    token_throughput_per_chip: float
    output_throughput_per_chip: float
    p90_ttft_ms: float
    p90_tpot_ms: float
    concurrency: int
    total_chips: int
    mode: str
    binding_unit: str


def _kv_transfer_flag_value(role: str) -> str | None:
    """Return JSON value for --kv-transfer-config on disagg pools."""
    if role == "prefill":
        kv_role = "kv_producer"
    elif role == "decode":
        kv_role = "kv_consumer"
    else:
        return None
    return json.dumps(
        {
            "kv_connector": _RAIDEN_CONNECTOR_NAME,
            "kv_connector_module_path": _RAIDEN_MODULE_PATH,
            "kv_role": kv_role,
        }
    )


def _append_pool_runtime_flags(
    args: list[str],
    pool: PoolMeshPlan,
    *,
    gpu_memory_utilization: float | None,
    model_id: str,
    spec_k: int,
) -> None:
    """Append memory, override, speculative, and disagg transfer flags."""
    if pool.max_model_len is not None:
        args.extend(["--max-model-len", str(pool.max_model_len)])
    if pool.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(pool.max_num_seqs)])
    if gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", f"{gpu_memory_utilization:g}"])
    if model_id == "qwen4_exp":
        args.extend(["--hf-overrides", _QWEN4_BRINGUP_OVERRIDES])
    if spec_k > 0:
        spec_json = json.dumps({"method": "mtp", "num_speculative_tokens": spec_k})
        args.extend(["--speculative-config", spec_json])
    kv_transfer = _kv_transfer_flag_value(pool.role)
    if kv_transfer is not None:
        args.extend(["--kv-transfer-config", kv_transfer])


def pool_cli_args(
    pool: PoolMeshPlan,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    model_id: str = "",
    spec_k: int = 0,
) -> list[str]:
    """Translate one ``PoolMeshPlan`` into ``vllm serve`` CLI flags."""
    args = [
        "--tensor-parallel-size",
        str(pool.tp),
        "--data-parallel-size",
        str(pool.dp),
        "--kv-cache-dtype",
        kv_dtype,
    ]
    if pool.pp > 1:
        args.extend(
            [
                "--pipeline-parallel-size",
                str(pool.pp),
                "--no-async-scheduling",
            ]
        )
    if pool.pcp > 1:
        args.extend(
            [
                "--prefill-context-parallel-size",
                str(pool.pcp),
                "--cp-kv-cache-interleave-size",
                str(DEFAULT_PCP_KV_INTERLEAVE_SIZE),
            ]
        )
    if pool.kv_split > 1:
        args.extend(["--decode-context-parallel-size", str(pool.kv_split)])
    if pool.ep > 1 or pool.moe_dispatch == "fused_rs":
        args.append("--enable-expert-parallel")
    _append_pool_runtime_flags(
        args,
        pool,
        gpu_memory_utilization=gpu_memory_utilization,
        model_id=model_id,
        spec_k=spec_k,
    )
    return args


def _servable_model(source: str | None) -> str | None:
    """Return ``source`` unless it is None or names a ``.json`` config file."""
    if source is None or source.endswith(".json"):
        return None
    return source


def _env_prefix(pool: PoolMeshPlan, *, model_id: str = "") -> str:
    """Return validated runtime environment variable prefix for ``pool``."""
    env_parts: list[str] = []
    if pool.moe_dispatch == "a2a_push":
        env_parts.append("USE_MOE_FUSED_EP_KERNEL=1")
    elif pool.moe_dispatch == "fused_rs":
        if model_id != "qwen3_5_moe" or pool.pp != 1:
            msg = (
                "TPU_MOE_SEQUENCE_PARALLEL=1 is supported only for qwen3_5_moe "
                f"with pp == 1; got model_id={model_id!r}, pp={pool.pp}."
            )
            raise ValueError(msg)
        env_parts.append("TPU_MOE_SEQUENCE_PARALLEL=1 USE_MOE_FUSED_EP_KERNEL=1")
    return " ".join(env_parts)


def _format_server_cmd(
    cli_args: list[str],
    model: str | None,
    *,
    env_prefix: str = "",
) -> str:
    """Format launch CLI arguments as a standalone bash command string."""
    joined = shlex.join(cli_args)
    prefix = f"vllm serve {_servable_model(model) or _MODEL_PLACEHOLDER}"
    cmd = f"{prefix} {joined}".strip()
    return f"{env_prefix} {cmd}" if env_prefix else cmd


def _mode_a_host_args(
    base_args: list[str],
    hosts: int,
) -> list[tuple[int, list[str]]]:
    """Expand Mode A multi-node TP/PP/PCP flags (--nnodes / --node-rank)."""
    per_host: list[tuple[int, list[str]]] = []
    for rank in range(hosts):
        extra = [
            "--nnodes",
            str(hosts),
            "--node-rank",
            str(rank),
            "--master-addr",
            _HOST0_ADDR_PLACEHOLDER,
            "--master-port",
            _DEFAULT_MASTER_PORT,
        ]
        if rank > 0:
            extra.append("--headless")
        per_host.append((rank, [*base_args, *extra]))
    return per_host


def _mode_b_host_args(
    base_args: list[str],
    hosts: int,
    dp: int,
) -> list[tuple[int, list[str]]]:
    """Expand Mode B multi-host DP flags (--data-parallel-size-local)."""
    dp_local = dp // hosts
    per_host: list[tuple[int, list[str]]] = []
    for rank in range(hosts):
        extra = [
            "--data-parallel-size-local",
            str(dp_local),
            "--data-parallel-start-rank",
            str(rank * dp_local),
            "--data-parallel-address",
            _HOST0_ADDR_PLACEHOLDER,
            "--data-parallel-rpc-port",
            _DEFAULT_DP_RPC_PORT,
        ]
        if rank > 0:
            extra.append("--headless")
        per_host.append((rank, [*base_args, *extra]))
    return per_host


def _multihost_host_arg_lists(
    pool: PoolMeshPlan,
    base_args: list[str],
    chip: ChipSpec | None,
) -> list[tuple[int, list[str]]]:
    """Expand multi-host ``pool`` into per-host Mode A or Mode B CLI arg lists."""
    chips_per_host = chip.chips_per_host if chip is not None else DEFAULT_CHIPS_PER_HOST
    cores_per_chip = chip.cores_per_chip if chip is not None else DEFAULT_CORES_PER_CHIP
    mode = classify_multihost_mp_mode(
        pool.chips_per_instance,
        (pool.tp, pool.dp, pool.pp, pool.pcp),
        chips_per_host=chips_per_host,
        cores_per_chip=cores_per_chip,
    )
    if mode == "single_host":
        return [(0, list(base_args))]
    hosts = pool.chips_per_instance // chips_per_host
    if mode == "mode_a":
        return _mode_a_host_args(base_args, hosts)
    if mode == "mode_b":
        return _mode_b_host_args(base_args, hosts, pool.dp)

    msg = (
        "Unsupported hybrid multi-host multiprocessing layout: "
        f"chips_per_instance={pool.chips_per_instance} ({hosts} hosts), "
        f"dp={pool.dp}, tp={pool.tp}, pp={pool.pp}, pcp={pool.pcp}. "
        "Native mp supports only Mode A (dp == 1) or Mode B "
        "(dp % hosts == 0 and tp * pp * pcp <= cores_per_host)."
    )
    raise ValueError(msg)


def _format_pool_commands(
    pool: PoolMeshPlan,
    base_args: list[str],
    model: str | None,
    *,
    chip: ChipSpec | None = None,
    model_id: str = "",
) -> list[str]:
    """Render single-host or per-host Mode A/B launch commands for one pool."""
    prefix = _env_prefix(pool, model_id=model_id)
    host_args = _multihost_host_arg_lists(pool, base_args, chip)
    is_disagg = pool.role in ("prefill", "decode")
    is_multihost = len(host_args) > 1
    commands: list[str] = []
    for rank, args in host_args:
        cmd = _format_server_cmd(args, model, env_prefix=prefix)
        if is_disagg and is_multihost:
            commands.append(
                f"# {pool.role} ({pool.instances}x {pool.topology}, host {rank}): {cmd}"
            )
        elif is_disagg:
            commands.append(f"# {pool.role} ({pool.instances}x {pool.topology}): {cmd}")
        elif is_multihost:
            commands.append(f"# host {rank} ({pool.topology}): {cmd}")
        else:
            commands.append(cmd)
    return commands


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
    chip: ChipSpec | None = None,
) -> LaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for the top-ranked L1 candidate."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    eff_chip = chip if chip is not None else CHIPS.get(plan.chip)
    warnings = list(plan.warnings)
    rec = plan.recommended
    if rec is None or not rec.slo_met:
        if not warnings:
            warnings.append(NO_FEASIBLE_LAYOUT_WARNING)
        return {
            "mode": plan.mode,
            "cli_args": [],
            "prefill_cli_args": [],
            "commands": [],
            "warnings": warnings,
            "serving_plan": asdict(plan),
        }

    decode_args = pool_cli_args(
        rec.decode_pool,
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model_id=plan.model_id,
        spec_k=plan.spec_k,
    )
    prefill_args = (
        pool_cli_args(
            rec.prefill_pool,
            kv_dtype=kv_str,
            gpu_memory_utilization=plan.gpu_memory_utilization,
            model_id=plan.model_id,
            spec_k=plan.spec_k,
        )
        if rec.prefill_pool is not None
        else []
    )
    commands: list[str] = []
    if rec.prefill_pool is not None:
        commands.extend(
            _format_pool_commands(
                rec.prefill_pool,
                prefill_args,
                model,
                chip=eff_chip,
                model_id=plan.model_id,
            )
        )
    commands.extend(
        _format_pool_commands(
            rec.decode_pool,
            decode_args,
            model,
            chip=eff_chip,
            model_id=plan.model_id,
        )
    )

    if rec.prefill_pool is not None:
        warnings.append(_DISAGG_WIRING_WARNING)
    if _servable_model(model) is None:
        warnings.append(_UNKNOWN_MODEL_WARNING)
    return {
        "mode": rec.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
        "warnings": warnings,
        "serving_plan": asdict(plan),
    }


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export uncalibrated L1 candidate rows by p90 interactivity and throughput."""
    return [
        {
            "p90_interactivity_tok_s": pt.p90_interactivity_tok_s,
            "token_throughput_per_chip": pt.token_throughput_per_chip,
            "output_throughput_per_chip": pt.output_throughput_per_chip,
            "p90_ttft_ms": pt.p90_ttft_ms,
            "p90_tpot_ms": pt.p90_tpot_ms,
            "concurrency": pt.concurrency,
            "total_chips": pt.total_chips,
            "mode": pt.mode,
            "binding_unit": pt.binding,
        }
        for pt in plan.frontier
        if pt.slo_met
    ]
