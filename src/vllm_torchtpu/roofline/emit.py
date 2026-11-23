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
"""Render vLLM serve CLI flags, top-K anchor shortlists, and solve JSON."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, TypedDict, Unpack

from .chips import CHIPS, JAX_VERSION, chip_catalog, topologies_for_chip
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_PCP_KV_INTERLEAVE_SIZE,
    DEFAULT_TOP_K,
    FRONTIER_2D_TIE_TOLERANCE,
    KV_DTYPE_BYTES,
    MIN_TOP_LAYOUTS_FOR_FASTEST,
    NO_FEASIBLE_LAYOUT_WARNING,
    PERCENT_SCALE,
    SLO_VERDICT_RANK,
    SLO_VERDICT_RULED_OUT,
    SOLVE_SCHEMA,
    UNIT_GIGA,
)
from .families import (
    attach_reference_report,
    build_bound_families_by_concurrency,
    pareto_skyline_by_concurrency,
    parse_layout,
    reference_layout_report,
    strict_rank,
)
from .schema import (
    CeilingRow,
    LaunchPayload,
    ParetoPoint,
    PointLaunchPayload,
    PoolMeshPlan,
    PoolStructKey,
    ProvenancePayload,
    ServingPlan,
    SolvePayload,
    TopKEntry,
    _point_struct_key,
    effective_slo_verdict,
    format_host_count,
    format_parallelism,
    format_submodule_parallelism,
    pool_label,
    pool_slice_summary,
)
from .solver import classify_multihost_mp_mode

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology
    from .modules import Dtypes

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

__all__ = [
    "FRONTIER_2D_TIE_TOLERANCE",
    "CeilingRow",
    "LaunchOptions",
    "LaunchPayload",
    "PointLaunchPayload",
    "ProvenancePayload",
    "SolvePayload",
    "TopKEntry",
    "attach_reference_report",
    "build_bound_families_by_concurrency",
    "build_solve_payload",
    "emit_ceiling_rows",
    "emit_launch_args",
    "format_host_count",
    "format_parallelism",
    "format_solve_json",
    "format_submodule_parallelism",
    "pareto_skyline_by_concurrency",
    "parse_layout",
    "point_launch_args",
    "pool_cli_args",
    "pool_label",
    "pool_slice_summary",
    "reference_layout_report",
    "strict_rank",
    "top_layout_roles",
    "top_layouts",
]


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Typed launch command emission options for ``point_launch_args``."""

    kv_dtype: str = "auto"
    gpu_memory_utilization: float | None = None
    model: str | None = None
    chip: ChipSpec | None = None
    model_id: str = ""
    spec_k: int = 0


class _PointLaunchKwargs(TypedDict, total=False):
    """Keyword-argument overrides accepted by ``point_launch_args``."""

    kv_dtype: str
    gpu_memory_utilization: float | None
    model: str | None
    chip: ChipSpec | None
    model_id: str
    spec_k: int


class _BuildSolvePayloadOpts(TypedDict, total=False):
    """Optional arguments for ``build_solve_payload``."""

    dtypes: Dtypes | None
    model: str | None
    top_k: int
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None


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


def _frontier_2d_v1(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return upper envelope over modeled (interactivity, throughput) estimates."""
    ordered = sorted(
        points,
        key=lambda point: (
            -point.p90_interactivity_tok_s,
            -point.token_throughput_per_chip,
        ),
    )
    envelope: list[ParetoPoint] = []
    best_throughput = -1.0
    for point in ordered:
        if (
            point.token_throughput_per_chip
            > best_throughput + FRONTIER_2D_TIE_TOLERANCE
        ):
            envelope.append(point)
            best_throughput = point.token_throughput_per_chip
    envelope.sort(key=lambda point: point.p90_interactivity_tok_s)
    return envelope


def _is_2d_dominated(
    point: ParetoPoint,
    peers: Sequence[ParetoPoint],
) -> bool:
    """Return True if a peer beats point on both interactivity and throughput."""
    return any(
        other is not point
        and other.p90_interactivity_tok_s >= point.p90_interactivity_tok_s
        and other.token_throughput_per_chip >= point.token_throughput_per_chip
        and (
            other.p90_interactivity_tok_s > point.p90_interactivity_tok_s
            or other.token_throughput_per_chip > point.token_throughput_per_chip
        )
        for other in peers
    )


def top_layouts(plan: ServingPlan, k: int = DEFAULT_TOP_K) -> tuple[ParetoPoint, ...]:
    """Build a heuristic family-collapsed anchor shortlist of up to k layouts.

    Collapses modeled candidates across concurrencies by structural pool axes
    and selects uncalibrated serial-phase throughput/interactivity anchors
    that satisfy SLO targets.
    """
    if k <= 0:
        return ()
    best: dict[tuple[str, PoolStructKey | None, PoolStructKey | None], ParetoPoint] = {}
    slo_cands: list[ParetoPoint] = []
    frontier_slo = [pt for pt in plan.frontier if pt.slo_met]
    source_points = _frontier_2d_v1(frontier_slo) or [
        pt for pt in plan.candidates if pt.slo_met
    ]
    for pt in source_points:
        if not pt.slo_met:
            continue
        slo_cands.append(pt)
        key = _point_struct_key(pt)
        cur = best.get(key)
        if cur is None or (
            pt.token_throughput_per_chip,
            pt.p90_interactivity_tok_s,
        ) > (
            cur.token_throughput_per_chip,
            cur.p90_interactivity_tok_s,
        ):
            best[key] = pt
    ranked = sorted(best.values(), key=lambda p: -p.token_throughput_per_chip)
    if len(ranked) <= 1 or k < MIN_TOP_LAYOUTS_FOR_FASTEST:
        return tuple(ranked[:k])
    selected = list(ranked[:k])
    fastest = max(
        slo_cands,
        key=lambda p: (p.p90_interactivity_tok_s, p.token_throughput_per_chip),
    )
    f_key = _point_struct_key(fastest)
    sel_keys = [_point_struct_key(p) for p in selected]
    if f_key not in sel_keys:
        selected[-1] = fastest
    elif sel_keys.index(f_key) > 0:
        selected[sel_keys.index(f_key)] = fastest
    selected.sort(key=lambda p: -p.token_throughput_per_chip)
    return tuple(selected)


def top_layout_roles(
    top_pts: tuple[ParetoPoint, ...],
    frontier: Sequence[ParetoPoint] = (),
) -> list[tuple[ParetoPoint, str, float]]:
    """Annotate selected Top-K points with role labels and Rank-1 throughput margins."""
    top1_tput = top_pts[0].token_throughput_per_chip if top_pts else 0.0
    fastest_idx = (
        max(
            range(len(top_pts)),
            key=lambda i: (
                top_pts[i].p90_interactivity_tok_s,
                top_pts[i].token_throughput_per_chip,
            ),
        )
        if len(top_pts) > 1
        else -1
    )
    eligible_frontier = [
        pt for pt in frontier if effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT
    ]
    dominance_pool: tuple[ParetoPoint, ...] = (*top_pts, *eligible_frontier)
    out: list[tuple[ParetoPoint, str, float]] = []
    for idx, pt in enumerate(top_pts):
        if idx == 0:
            role = "max_throughput"
        elif idx == fastest_idx:
            role = "max_interactivity"
        elif _is_2d_dominated(pt, dominance_pool):
            role = "off_frontier_alt"
        else:
            role = "balanced_knee"
        margin = round(
            PERCENT_SCALE
            * (pt.token_throughput_per_chip - top1_tput)
            / max(1e-9, top1_tput),
            2,
        )
        out.append((pt, role, margin))
    return out


def point_launch_args(
    pt: ParetoPoint,
    options: LaunchOptions | None = None,
    **kwargs: Unpack[_PointLaunchKwargs],
) -> PointLaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for a single ``ParetoPoint``."""
    base = options or LaunchOptions()
    kv_dtype = kwargs.get("kv_dtype", base.kv_dtype)
    gpu_memory_utilization = kwargs.get(
        "gpu_memory_utilization", base.gpu_memory_utilization
    )
    model = kwargs.get("model", base.model)
    chip = kwargs.get("chip", base.chip)
    model_id = kwargs.get("model_id", base.model_id)
    spec_k = kwargs.get("spec_k", base.spec_k)
    decode_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        model_id=model_id,
        spec_k=spec_k,
    )
    prefill_args = (
        pool_cli_args(
            pt.prefill_pool,
            kv_dtype=kv_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            model_id=model_id,
            spec_k=spec_k,
        )
        if pt.prefill_pool is not None
        else []
    )
    commands: list[str] = []
    if pt.prefill_pool is not None:
        commands.extend(
            _format_pool_commands(
                pt.prefill_pool,
                prefill_args,
                model,
                chip=chip,
                model_id=model_id,
            )
        )
    commands.extend(
        _format_pool_commands(
            pt.decode_pool,
            decode_args,
            model,
            chip=chip,
            model_id=model_id,
        )
    )
    return {
        "mode": pt.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
    }


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
    if (
        rec is None
        or not rec.slo_met
        or effective_slo_verdict(rec) == SLO_VERDICT_RULED_OUT
    ):
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

    launch = point_launch_args(
        rec,
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
        chip=eff_chip,
        model_id=plan.model_id,
        spec_k=plan.spec_k,
    )
    if rec.prefill_pool is not None:
        warnings.append(_DISAGG_WIRING_WARNING)
    if _servable_model(model) is None:
        warnings.append(_UNKNOWN_MODEL_WARNING)
    return {
        "mode": launch["mode"],
        "cli_args": launch["cli_args"],
        "prefill_cli_args": launch["prefill_cli_args"],
        "commands": launch["commands"],
        "warnings": warnings,
        "serving_plan": asdict(plan),
    }


def _resolve_single_host_topology(
    chips_per_host: int,
    chip: ChipSpec | None,
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None,
) -> str:
    """Return single-host slice topology name matching ``chips_per_host``."""
    if topologies is not None:
        topo_seq = (
            list(topologies.values())
            if isinstance(topologies, Mapping)
            else list(topologies)
        )
    elif chip is not None:
        topo_seq = list(topologies_for_chip(chip).values())
    else:
        topo_seq = list(topologies_for_chip("tpu7x").values())
    return next(
        (topo.name for topo in topo_seq if topo.num_chips == chips_per_host),
        f"{chips_per_host}chip",
    )


def _compute_single_host_fit(
    plan: ServingPlan,
    *,
    chip: ChipSpec | None = None,
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None = None,
) -> dict[str, object]:
    """Compute single-host slice fit verdict, capacity, and status."""
    eff_chip = chip if chip is not None else CHIPS.get(plan.chip)
    chips_per_host = (
        eff_chip.chips_per_host if eff_chip is not None else DEFAULT_CHIPS_PER_HOST
    )
    cores_per_chip = (
        eff_chip.cores_per_chip if eff_chip is not None else DEFAULT_CORES_PER_CHIP
    )
    cores_per_host = chips_per_host * cores_per_chip
    host_topo = _resolve_single_host_topology(chips_per_host, eff_chip, topologies)
    host_desc = f"(1 host, {chips_per_host} chips, {cores_per_host} TensorCores)"
    host_points = [
        pt
        for pt in (*plan.frontier, *plan.candidates)
        if pt.total_chips == chips_per_host
        and pt.decode_pool.topology == host_topo
        and pt.prefill_pool is None
        and effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT
    ]
    if host_points:
        best_pt = max(
            host_points,
            key=lambda point: (
                SLO_VERDICT_RANK[effective_slo_verdict(point)],
                point.token_throughput_per_chip,
            ),
        )
        pool = best_pt.decode_pool
        return {
            "feasible": True,
            "status": "FEASIBLE",
            "reason": f"Fits single-host {host_topo} {host_desc}",
            "topology": host_topo,
            "chips_per_host": chips_per_host,
            "cores_per_host": cores_per_host,
            "hbm_used_gb": round(pool.hbm_used_gb, 2),
            "hbm_limit_gb": round(pool.hbm_limit_gb, 2),
            "best_parallelism": format_parallelism(asdict(pool)),
            "best_submodule_parallelism": format_submodule_parallelism(
                asdict(pool), has_moe=plan.has_moe
            ),
        }
    per_core_bytes = eff_chip.hbm_bytes if eff_chip else CHIPS["tpu7x"].hbm_bytes
    limit_gb = round(
        max(0.0, per_core_bytes * plan.gpu_memory_utilization) / float(UNIT_GIGA),
        2,
    )
    if plan.mode == "disagg":
        reason = (
            f"Disaggregated mode selected; single-host {host_topo} requires mixed mode "
            f"{host_desc}"
        )
    elif 0 < plan.budget < chips_per_host:
        reason = (
            f"Chip budget ({plan.budget}) is below 1 host "
            f"({host_topo} requires {chips_per_host} chips, "
            f"{cores_per_host} TensorCores)"
        )
    else:
        reason = (
            f"No feasible {host_topo} layout within HBM, SMEM, head-divisibility, or "
            f"SLO bounds {host_desc}"
        )
    return {
        "feasible": False,
        "status": "REJECTED",
        "reason": reason,
        "topology": host_topo,
        "chips_per_host": chips_per_host,
        "cores_per_host": cores_per_host,
        "hbm_used_gb": 0.0,
        "hbm_limit_gb": limit_gb,
        "best_parallelism": None,
        "best_submodule_parallelism": None,
    }


def build_solve_payload(
    plan: ServingPlan,
    *,
    chip: ChipSpec,
    **opts: Unpack[_BuildSolvePayloadOpts],
) -> SolvePayload:
    """Build versioned solve JSON payload with L1 provenance and anchor shortlist."""
    dtypes = opts.get("dtypes")
    model = opts.get("model")
    top_k = opts.get("top_k", DEFAULT_TOP_K)
    topologies = opts.get("topologies")
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    top_pts = top_layouts(plan, top_k)
    launch_opts = LaunchOptions(
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
        chip=chip,
        model_id=plan.model_id,
        spec_k=plan.spec_k,
    )
    top_entries: list[TopKEntry] = [
        {
            "frontier_role": role,
            "margin_vs_rank1_pct": margin,
            "point": asdict(pt),
            "launch": point_launch_args(pt, launch_opts),
        }
        for pt, role, margin in top_layout_roles(top_pts, plan.frontier)
    ]
    pins_map = plan.pins or {}
    chip_dict: dict[str, object] = dict(asdict(chip).items())
    chip_dict["jax_version"] = JAX_VERSION
    chip_dict["catalog"] = chip_catalog(chip)
    target_chips = (
        plan.recommended.total_chips if plan.recommended is not None else plan.budget
    )
    payload: SolvePayload = {
        "model_layers": plan.model_layers,
        "chip": plan.chip,
        "scenario": asdict(plan.scenario),
        "mode": plan.mode,
        "recommended": (
            asdict(plan.recommended) if plan.recommended is not None else None
        ),
        "frontier": tuple(asdict(p) for p in plan.frontier),
        "warnings": plan.warnings,
        "candidates": tuple(asdict(p) for p in plan.candidates),
        "gpu_memory_utilization": plan.gpu_memory_utilization,
        "search": asdict(plan.search) if plan.search is not None else None,
        "schema": SOLVE_SCHEMA,
        "provenance": {
            "calibrated": False,
            "basis": (
                "L1 analytical step floors and uncalibrated serial-phase "
                "p90 estimates; modeled, not measured"
            ),
        },
        "chip_spec": chip_dict,
        "model": model,
        "top_k": top_entries,
        "has_moe": plan.has_moe,
        "single_host_fit": _compute_single_host_fit(
            plan, chip=chip, topologies=topologies
        ),
        "target_chips": target_chips,
        "budget": plan.budget,
        "budget_source": plan.budget_source,
        "max_instances": plan.max_instances,
        "pins": plan.pins,
        "chip_budget": {
            "max_chips": plan.budget,
            "budget_source": plan.budget_source,
            "max_instances": plan.max_instances,
            "prefill_topology": pins_map.get("prefill_topology"),
            "decode_topology": pins_map.get("decode_topology"),
            "pd_ratio": pins_map.get("pd_ratio", pins_map.get("pd")),
        },
    }
    skyline_by_conc = pareto_skyline_by_concurrency(plan, top_k)
    payload["concurrency_top_k"] = {
        str(conc): [
            {
                "frontier_role": role,
                "margin_vs_rank1_pct": margin,
                "point": asdict(pt),
                "launch": point_launch_args(pt, launch_opts),
            }
            for pt, role, margin in top_layout_roles(conc_pts)
        ]
        for conc, conc_pts in skyline_by_conc.items()
    }
    payload["bound_families_by_concurrency"] = build_bound_families_by_concurrency(
        plan,
        launch_fn=lambda pt: point_launch_args(pt, launch_opts),
    )
    if dtypes is not None:
        payload["dtypes"] = asdict(dtypes)
    return payload


def format_solve_json(payload: SolvePayload) -> str:
    """Serialize a ``SolvePayload`` mapping as a JSON string."""
    return json.dumps(payload)


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
