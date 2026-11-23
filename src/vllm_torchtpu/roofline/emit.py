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

from .chips import (
    CHIPS,
    JAX_VERSION,
    chip_catalog,
    get_solver_assumptions,
    topologies_for_chip,
)
from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_HBM_RESERVE_GB,
    DEFAULT_PCP_KV_INTERLEAVE_SIZE,
    DEFAULT_TOP_K,
    FRONTIER_2D_TIE_TOLERANCE,
    KV_DTYPE_BYTES,
    MIN_TOP_LAYOUTS_FOR_BOUND_BEST,
    MIN_TOP_LAYOUTS_FOR_FASTEST,
    NO_FEASIBLE_LAYOUT_WARNING,
    PERCENT_SCALE,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RANK,
    SLO_VERDICT_RULED_OUT,
    SOLVE_SCHEMA,
    UNIT_GIBI,
    UNIT_GIGA,
    UNIT_MICRO,
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
_RAIDEN_OFFLOAD_CONNECTOR_NAME = "TPURaidenOffloadingConnector"
_RAIDEN_OFFLOAD_MODULE_PATH = "vllm_torchtpu.offload.raiden_connector"
_MULTI_CONNECTOR_NAME = "TPUMultiConnector"
_MULTI_CONNECTOR_MODULE_PATH = (
    "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
)
_HYBRID_MODEL_IDS = frozenset({"qwen3_5_moe", "qwen4_exp"})
_DISAGG_WIRING_WARNING = (
    "Disaggregated launch requires a Raiden controller "
    "(TPU_USE_RAIDEN_KV_CACHE_MANAGER=1, TPU_RAIDEN_CONTROLLER_ADDRESS) "
    "and a proxy router; see docs/user_guide/disaggregated_serving.md."
)
_UNKNOWN_MODEL_WARNING = (
    f"The model to serve is not known: replace {_MODEL_PLACEHOLDER} in each "
    "command with a model ID or a model directory."
)
_DEFAULT_ACTIVATION_REGIME = "bernoulli"


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
    "frontier_2d_points",
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
    spec_tokens: int = 0
    mtp_layers: int | None = None
    hbm_reserve_gb: float = DEFAULT_HBM_RESERVE_GB
    topologies: Mapping[str, SliceTopology] | None = None
    is_hybrid: bool = False


class _PointLaunchKwargs(TypedDict, total=False):
    """Keyword-argument overrides accepted by ``point_launch_args``."""

    kv_dtype: str
    gpu_memory_utilization: float | None
    model: str | None
    chip: ChipSpec | None
    model_id: str
    spec_k: int
    spec_tokens: int
    mtp_layers: int | None
    hbm_reserve_gb: float
    topologies: Mapping[str, SliceTopology] | None
    is_hybrid: bool


class _BuildSolvePayloadOpts(TypedDict, total=False):
    """Optional arguments for ``build_solve_payload``."""

    dtypes: Dtypes | None
    model: str | None
    top_k: int
    topologies: Mapping[str, SliceTopology] | Sequence[SliceTopology] | None


def _kv_transfer_flag_value(pool: PoolMeshPlan) -> str | None:
    """Return JSON value for --kv-transfer-config covering disagg and offload."""
    if pool.role == "prefill":
        kv_role: str | None = "kv_producer"
    elif pool.role == "decode":
        kv_role = "kv_consumer"
    else:
        kv_role = None
    has_offload = pool.kv_offload_gb > 0.0
    if kv_role is None and not has_offload:
        return None
    raiden_disagg = (
        {
            "kv_connector": _RAIDEN_CONNECTOR_NAME,
            "kv_connector_module_path": _RAIDEN_MODULE_PATH,
            "kv_role": kv_role,
        }
        if kv_role is not None
        else None
    )
    if not has_offload:
        return json.dumps(raiden_disagg)
    cpu_bytes = round(pool.kv_offload_gb * UNIT_GIBI)
    raiden_offload: dict[str, object] = {
        "kv_connector": _RAIDEN_OFFLOAD_CONNECTOR_NAME,
        "kv_connector_module_path": _RAIDEN_OFFLOAD_MODULE_PATH,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
    }
    if raiden_disagg is None:
        return json.dumps(raiden_offload)
    return json.dumps(
        {
            "kv_connector": _MULTI_CONNECTOR_NAME,
            "kv_connector_module_path": _MULTI_CONNECTOR_MODULE_PATH,
            "kv_role": kv_role,
            "kv_connector_extra_config": {
                "connectors": [raiden_disagg, raiden_offload]
            },
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
    kv_transfer = _kv_transfer_flag_value(pool)
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


def _env_prefix(
    pool: PoolMeshPlan,
    *,
    model_id: str = "",
    is_hybrid: bool = False,
) -> str:
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
    if pool.pp_layer_partition is not None and len(set(pool.pp_layer_partition)) > 1:
        part_str = ",".join(str(x) for x in pool.pp_layer_partition)
        env_parts.append(f"VLLM_PP_LAYER_PARTITION={part_str}")
    hybrid_pool = is_hybrid or pool.hbm_state_gb > 0.0 or model_id in _HYBRID_MODEL_IDS
    if pool.kv_offload_gb > 0.0 and hybrid_pool:
        env_parts.append("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1")
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
    model_meta: tuple[str, bool] = ("", False),
) -> list[str]:
    """Render single-host or per-host Mode A/B launch commands for one pool."""
    model_id, is_hybrid = model_meta
    prefix = _env_prefix(pool, model_id=model_id, is_hybrid=is_hybrid)
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


def _filter_eligible(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return non-ruled-out points, preferring ``estimate_meets`` when present."""
    eligible = [
        pt for pt in points if effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT
    ]
    meets = [
        pt for pt in eligible if effective_slo_verdict(pt) == SLO_VERDICT_ESTIMATE_MEETS
    ]
    return meets if meets else eligible


def frontier_2d_points(points: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """Return non-dominated 2D upper envelope in (interactivity, throughput)."""
    ordered = sorted(
        points,
        key=lambda point: (
            -point.est_interactivity_tok_s,
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
    envelope.sort(key=lambda point: point.est_interactivity_tok_s)
    return envelope


def _is_2d_dominated(
    point: ParetoPoint,
    peers: Sequence[ParetoPoint],
) -> bool:
    """Return True if a peer beats point on both interactivity and throughput."""
    return any(
        other is not point
        and other.est_interactivity_tok_s >= point.est_interactivity_tok_s
        and other.token_throughput_per_chip >= point.token_throughput_per_chip
        and (
            other.est_interactivity_tok_s > point.est_interactivity_tok_s
            or other.token_throughput_per_chip > point.token_throughput_per_chip
        )
        for other in peers
    )


def _select_compact_top_layouts(
    ranked: list[ParetoPoint],
    fastest: ParetoPoint,
    slots: int,
) -> list[ParetoPoint]:
    """Select top throughput layouts plus the fastest interactivity anchor."""
    if len(ranked) <= 1 or slots < MIN_TOP_LAYOUTS_FOR_FASTEST:
        return list(ranked[:slots])
    selected = list(ranked[:slots])
    f_key = _point_struct_key(fastest)
    sel_keys = [_point_struct_key(p) for p in selected]
    if f_key not in sel_keys:
        selected[-1] = fastest
    elif sel_keys.index(f_key) > 0:
        selected[sel_keys.index(f_key)] = fastest
    selected.sort(key=lambda p: -p.token_throughput_per_chip)
    return selected


def top_layouts(
    plan: ServingPlan,
    k: int = DEFAULT_TOP_K,
    *,
    near_tie_rel: float = 0.0,
) -> tuple[ParetoPoint, ...]:
    """Select up to k structurally distinct eligible layouts spanning Pareto roles."""
    if k <= 0:
        return ()
    eligible = _filter_eligible(plan.candidates)
    if not eligible:
        return ()
    frontier_2d = frontier_2d_points(_filter_eligible(plan.frontier))
    source_points = frontier_2d if (near_tie_rel <= 0.0 and frontier_2d) else eligible
    best: dict[tuple[str, PoolStructKey | None, PoolStructKey | None], ParetoPoint] = {}
    for pt in source_points:
        key = _point_struct_key(pt)
        cur = best.get(key)
        if cur is None or (
            pt.token_throughput_per_chip,
            pt.est_interactivity_tok_s,
        ) > (cur.token_throughput_per_chip, cur.est_interactivity_tok_s):
            best[key] = pt
    ranked = sorted(best.values(), key=lambda p: -p.token_throughput_per_chip)
    fastest = max(
        source_points,
        key=lambda p: (p.est_interactivity_tok_s, p.token_throughput_per_chip),
    )
    selected = _select_compact_top_layouts(ranked, fastest, k)
    if near_tie_rel > 0.0 and len(selected) >= k and len(ranked) > k:
        kth_tput = selected[k - 1].token_throughput_per_chip
        sel_keys = {_point_struct_key(p) for p in selected}
        selected.extend(
            p
            for p in ranked[k:]
            if _point_struct_key(p) not in sel_keys
            and (kth_tput - p.token_throughput_per_chip)
            / max(FRONTIER_2D_TIE_TOLERANCE, kth_tput)
            <= near_tie_rel
        )
    if k >= MIN_TOP_LAYOUTS_FOR_BOUND_BEST:
        lowest_bound = min(pt.bound_tpot_ms for pt in eligible)
        best_bnd = [pt for pt in eligible if pt.bound_tpot_ms == lowest_bound]
        sel_keys = {_point_struct_key(pt) for pt in selected}
        if not any(_point_struct_key(pt) in sel_keys for pt in best_bnd):
            bound_pt = max(best_bnd, key=lambda pt: pt.token_throughput_per_chip)
            if len(selected) >= k:
                selected = _select_compact_top_layouts(ranked, fastest, k - 1)
            selected.append(bound_pt)
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
                top_pts[i].est_interactivity_tok_s,
                top_pts[i].token_throughput_per_chip,
            ),
        )
        if len(top_pts) > 1
        else -1
    )
    lowest_bound = min((pt.bound_tpot_ms for pt in top_pts), default=0.0)
    bound_idx = next(
        (i for i, pt in enumerate(top_pts) if pt.bound_tpot_ms == lowest_bound), -1
    )
    if any(
        top_pts[i].bound_tpot_ms == lowest_bound
        for i in (0, fastest_idx)
        if 0 <= i < len(top_pts)
    ):
        bound_idx = -1
    eligible_frontier = _filter_eligible(frontier) if frontier else ()
    dominance_pool: tuple[ParetoPoint, ...] = (*top_pts, *eligible_frontier)
    out: list[tuple[ParetoPoint, str, float]] = []
    for idx, pt in enumerate(top_pts):
        if idx == 0:
            role = "max_throughput"
        elif idx == fastest_idx:
            role = "max_interactivity"
        elif idx == bound_idx:
            role = "min_tpot_bound"
        elif _is_2d_dominated(pt, dominance_pool):
            role = "off_frontier_alt"
        else:
            role = "balanced_knee"
        margin = round(
            PERCENT_SCALE
            * (pt.token_throughput_per_chip - top1_tput)
            / max(FRONTIER_2D_TIE_TOLERANCE, top1_tput),
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
    spec_tokens = kwargs.get("spec_tokens", base.spec_tokens)
    mtp_layers = kwargs.get("mtp_layers", base.mtp_layers)
    is_hybrid = kwargs.get("is_hybrid", base.is_hybrid)
    raw_spec = spec_tokens if spec_tokens > 0 else spec_k
    eff_spec = 0 if (mtp_layers is not None and mtp_layers <= 0) else raw_spec
    decode_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        model_id=model_id,
        spec_k=eff_spec,
    )
    prefill_args = (
        pool_cli_args(
            pt.prefill_pool,
            kv_dtype=kv_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            model_id=model_id,
            spec_k=eff_spec,
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
                model_meta=(model_id, is_hybrid),
            )
        )
    commands.extend(
        _format_pool_commands(
            pt.decode_pool,
            decode_args,
            model,
            chip=chip,
            model_meta=(model_id, is_hybrid),
        )
    )
    return {
        "mode": pt.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
    }


def _plan_dict(plan: ServingPlan) -> dict[str, object]:
    """Convert ``ServingPlan`` to dict with serialized ``activation_regime``."""
    data: dict[str, object] = dict(asdict(plan).items())
    data.pop("activation_model", None)
    data["activation_regime"] = plan.activation_model.regime
    return data


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
    chip: ChipSpec | None = None,
    topologies: Mapping[str, SliceTopology] | None = None,
) -> LaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for the top-ranked L1 candidate."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    eff_chip = chip if chip is not None else CHIPS.get(plan.chip)
    warnings = list(plan.warnings)
    rec = plan.recommended
    if rec is None or effective_slo_verdict(rec) == SLO_VERDICT_RULED_OUT:
        if not warnings:
            warnings.append(NO_FEASIBLE_LAYOUT_WARNING)
        return {
            "mode": plan.mode,
            "cli_args": [],
            "prefill_cli_args": [],
            "commands": [],
            "warnings": warnings,
            "serving_plan": _plan_dict(plan),
        }

    eff_spec = (
        plan.spec_tokens
        if plan.spec_tokens > 0
        else (plan.spec_k if plan.spec_k > 0 else (plan.scenario.spec_tokens or 0))
    )
    if eff_spec > 0 and plan.mtp_layers <= 0 and plan.spec_tokens > 0:
        warnings.append(
            "Model declares zero MTP draft layers; omitting --speculative-config."
        )
    mtp_guard = plan.mtp_layers if plan.spec_tokens > 0 else None
    launch = point_launch_args(
        rec,
        LaunchOptions(
            kv_dtype=kv_str,
            gpu_memory_utilization=plan.gpu_memory_utilization,
            model=model,
            chip=eff_chip,
            model_id=plan.model_id,
            spec_k=eff_spec,
            spec_tokens=eff_spec,
            mtp_layers=mtp_guard,
            hbm_reserve_gb=plan.hbm_reserve_gb,
            topologies=topologies,
            is_hybrid=plan.is_hybrid,
        ),
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
        "serving_plan": _plan_dict(plan),
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
    effective_limit_bytes = max(
        0.0,
        per_core_bytes * plan.gpu_memory_utilization
        - plan.hbm_reserve_gb * float(UNIT_GIGA),
    )
    limit_gb = round(effective_limit_bytes / float(UNIT_GIGA), 2)
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


def _supplied_entry(
    name: str,
    flag: str,
    kind: str,
    value: object,
    unit: str,
) -> dict[str, object]:
    """Build one structured supplied-override provenance record."""
    return {"name": name, "flag": flag, "kind": kind, "value": value, "unit": unit}


def _build_provenance_payload(plan: ServingPlan) -> ProvenancePayload:
    """Build provenance block recording active calibrations and supplied constraints."""
    active_calibrations: list[str] = []
    supplied_values: list[dict[str, object]] = []
    if plan.collective_latency_s > 0.0:
        supplied_values.append(
            _supplied_entry(
                "collective_latency_us",
                "--collective-latency-us",
                "calibration",
                plan.collective_latency_s / UNIT_MICRO,
                "us",
            )
        )
        active_calibrations.append("collective_latency_s")
    if plan.hbm_reserve_gb != DEFAULT_HBM_RESERVE_GB:
        supplied_values.append(
            _supplied_entry(
                "hbm_reserve_gb",
                "--hbm-reserve-gb",
                "constraint",
                plan.hbm_reserve_gb,
                "GB",
            )
        )
    if plan.activation_model.regime != _DEFAULT_ACTIVATION_REGIME:
        is_cal = plan.activation_model.prunes_on_envelope
        supplied_values.append(
            _supplied_entry(
                "moe_activation",
                "--moe-activation",
                "calibration" if is_cal else "constraint",
                plan.activation_model.regime,
                "mode",
            )
        )
        if is_cal:
            active_calibrations.append("moe_activation")
    return {
        "calibrated": bool(active_calibrations),
        "supplied_values": supplied_values,
        "ranking_basis": (
            "modeled serial_estimate_s estimate ranking for recommended, frontier, "
            "and top_k; active TPOT lower bound ranking for "
            "bound_families_by_concurrency"
        ),
        "basis": (
            "L1 analytical step floors and uncalibrated serial-phase "
            "roofline estimates; est_* values are modeled estimates, bound_* "
            "values are physical lower bounds, and envelope_* values reflect "
            "the active activation_regime (empirical_hull drives SLO ruling out)"
        ),
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
    eff_spec = (
        plan.spec_tokens
        if plan.spec_tokens > 0
        else (plan.spec_k if plan.spec_k > 0 else (plan.scenario.spec_tokens or 0))
    )
    topo_map: Mapping[str, SliceTopology] | None = (
        {topo.name: topo for topo in topologies}
        if isinstance(topologies, Sequence)
        else topologies
    )
    launch_opts = LaunchOptions(
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
        chip=chip,
        model_id=plan.model_id,
        spec_k=eff_spec,
        spec_tokens=eff_spec,
        mtp_layers=plan.mtp_layers if plan.spec_tokens > 0 else None,
        hbm_reserve_gb=plan.hbm_reserve_gb,
        topologies=topo_map,
        is_hybrid=plan.is_hybrid,
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
        "provenance": _build_provenance_payload(plan),
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
    payload["solver_assumptions"] = get_solver_assumptions(chip)
    payload["activation_regime"] = plan.activation_model.regime
    payload["spec_tokens"] = plan.spec_tokens
    payload["mtp_layers"] = plan.mtp_layers
    payload["hbm_reserve_gb"] = plan.hbm_reserve_gb
    if dtypes is not None:
        payload["dtypes"] = asdict(dtypes)
    return payload


def format_solve_json(payload: SolvePayload) -> str:
    """Serialize a ``SolvePayload`` mapping as a JSON string."""
    return json.dumps(payload)


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export Pareto ceiling rows leading with interactivity and throughput."""
    return [
        {
            "est_interactivity_tok_s": pt.est_interactivity_tok_s,
            "token_throughput_per_chip": pt.token_throughput_per_chip,
            "output_throughput_per_chip": pt.output_throughput_per_chip,
            "est_ttft_ms": pt.est_ttft_ms,
            "est_tpot_ms": pt.est_tpot_ms,
            "bound_tpot_ms": pt.bound_tpot_ms,
            "concurrency": pt.concurrency,
            "total_chips": pt.total_chips,
            "mode": pt.mode,
            "binding_unit": pt.binding,
            "min_decode_seqs": pt.min_decode_seqs,
            "floor_throughput_per_chip": pt.floor_throughput_per_chip,
        }
        for pt in plan.frontier
        if effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT
    ]
