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
"""L1 roofline bound pruning and heuristic shortlist solver."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

from .chips import TOPOLOGIES, ChipSpec, SliceTopology, topologies_for_chip
from .collectives import submesh_for_group
from .constants import (
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_INSTANCES,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    LINEAR_ATTENTION_LAYER_TOKENS,
    NO_FEASIBLE_LAYOUT_WARNING,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RULED_OUT,
    UNIT_GIGA,
    UNIT_MICRO,
    UNIT_MILLIS_PER_SECOND,
)
from .memory import (
    HbmBreakdown,
    KvFitInput,
    SeqKvLayout,
    default_rpa_page_size,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    kv_bytes_per_core_parts,
    kv_capacity_and_hit_rate,
    smem_max_seqs,
    tpu_min_page_size,
)
from .modules import Dtypes, Parallel, StepShape
from .partition import (
    count_full_attn_layers,
)
from .schema import (
    FloorTimes,
    ParetoPoint,
    PoolMeshPlan,
    ScenarioOverrides,
    SearchStats,
    ServingPlan,
    TrafficScenario,
)
from .step import FloorOptions, SpecShape, StepFloor, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .geometry import ModelGeometry


EXAMPLE_SCENARIOS: dict[str, TrafficScenario] = {
    "chat-8k1k": TrafficScenario(
        name="example:chat-8k1k",
        isl=8192,
        osl=1024,
        concurrencies=EXAMPLE_CONCURRENCIES,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "reasoning-1k8k": TrafficScenario(
        name="example:reasoning-1k8k",
        isl=1024,
        osl=8192,
        concurrencies=EXAMPLE_CONCURRENCIES,
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "coding-agent": TrafficScenario(
        name="example:coding-agent",
        isl=16384,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128),
        prefix_hit_rate=0.92,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
        spec_tokens=3,
    ),
    "custom": TrafficScenario(
        name="example:custom",
        isl=4096,
        osl=1024,
        concurrencies=(1, 4, 16, 64),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
}


def build_example_scenario(
    preset: str = "chat-8k1k",
    *,
    slo_tier: str = "none",
    overrides: ScenarioOverrides | None = None,
) -> TrafficScenario:
    """Resolve a TrafficScenario from a labeled preset and typed overrides."""
    if preset not in EXAMPLE_SCENARIOS:
        msg = (
            f"Unknown example scenario preset {preset!r}; "
            f"known: {sorted(EXAMPLE_SCENARIOS)}"
        )
        raise ValueError(msg)
    if slo_tier not in EXAMPLE_SLO_TIERS:
        msg = f"Unknown SLO tier {slo_tier!r}; known: {sorted(EXAMPLE_SLO_TIERS)}"
        raise ValueError(msg)
    base = EXAMPLE_SCENARIOS[preset]
    ovr = overrides or ScenarioOverrides()
    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[slo_tier]
    return replace(
        base,
        isl=base.isl if ovr.isl is None else ovr.isl,
        osl=base.osl if ovr.osl is None else ovr.osl,
        concurrencies=(
            base.concurrencies if ovr.concurrencies is None else ovr.concurrencies
        ),
        prefix_hit_rate=(
            base.prefix_hit_rate if ovr.prefix_hit_rate is None else ovr.prefix_hit_rate
        ),
        accept_len=base.accept_len if ovr.accept_len is None else ovr.accept_len,
        think_time_s=(
            base.think_time_s if ovr.think_time_s is None else ovr.think_time_s
        ),
        prefill_chunk=(
            base.prefill_chunk if ovr.prefill_chunk is None else ovr.prefill_chunk
        ),
        ttft_slo_ms=(
            ovr.ttft_slo_ms
            if ovr.ttft_slo_ms is not None
            else (tier_ttft if tier_ttft is not None else base.ttft_slo_ms)
        ),
        tpot_slo_ms=(
            ovr.tpot_slo_ms
            if ovr.tpot_slo_ms is not None
            else (tier_tpot if tier_tpot is not None else base.tpot_slo_ms)
        ),
        kv_working_set_turns=(
            base.kv_working_set_turns
            if ovr.kv_working_set_turns is None
            else ovr.kv_working_set_turns
        ),
    )


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Search mode, slice budget, pins, and HBM headroom options."""

    mode: Literal["auto", "mixed", "disagg"] = "auto"
    max_chips: int | None = None
    max_instances: int | None = None
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    topologies: tuple[SliceTopology, ...] | None = None
    collective_latency_s: float = 0.0
    prefill_topology: str | None = None
    decode_topology: str | None = None
    pd_ratio: tuple[int, int] | None = None
    budget_source: str = "flag"


def _divisors(n: int) -> tuple[int, ...]:
    """Return all integer divisors of n in ascending order."""
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _valid_head_tp(geom: ModelGeometry, tp: int) -> bool:
    """Check hardware attention weight and KV head divisibility for TP degree."""
    if geom.num_heads > 0 and (tp > geom.num_heads or geom.num_heads % tp != 0):
        return False
    if geom.num_kv_heads <= 0:
        return True
    return geom.num_kv_heads % tp == 0 or tp % geom.num_kv_heads == 0


def _has_recurrent_layers(geom: ModelGeometry) -> bool:
    """Check whether any layer type in geometry uses recurrent linear attention."""
    return any(tok in LINEAR_ATTENTION_LAYER_TOKENS for tok in geom.layer_types)


def _valid_gdn_degree(geom: ModelGeometry, deg: int) -> bool:
    """Check hardware and runtime GDN head divisibility for TP*PCP degree."""
    if deg <= 1 or not _has_recurrent_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    # Hardware sharding divisibility across value and key heads.
    if geom.linear_v_heads % deg != 0:
        return False
    uq = min(geom.linear_k_heads, deg)
    if geom.linear_k_heads % uq != 0 or deg % uq != 0:
        return False
    return (geom.linear_v_heads // deg) % (geom.linear_k_heads // uq) == 0


def _valid_raiden_gdn_pair(
    geom: ModelGeometry,
    par_p: Parallel,
    par_d: Parallel,
) -> bool:
    """Check runtime disaggregated prefill/decode GDN state transfer compatibility."""
    if not _has_recurrent_layers(geom):
        return True
    if geom.linear_v_heads <= 0 or geom.linear_k_heads <= 0:
        return True
    deg_p = par_p.tp * par_p.pcp
    deg_d = par_d.tp * par_d.pcp
    if not (_valid_gdn_degree(geom, deg_p) and _valid_gdn_degree(geom, deg_d)):
        return False
    # Runtime constraint: PP+PCP prefill allows integer decode multiples.
    if par_p.pp > 1 and deg_p > 1 and deg_d > deg_p and deg_d % deg_p == 0:
        return True
    kq = geom.linear_k_heads
    if max(1, deg_p // kq) != max(1, deg_d // kq):
        return False
    return deg_p == 1 or deg_p % deg_d == 0


def _valid_raiden_kv_pair(
    geom: ModelGeometry,
    prefill_parallel: Parallel,
    decode_parallel: Parallel,
) -> bool:
    """Check runtime disaggregated KV cache shard compatibility between P and D."""
    if geom.num_kv_heads <= 0:
        return True
    deg_p = max(1, prefill_parallel.tp * prefill_parallel.pcp)
    deg_d = max(1, decode_parallel.tp * decode_parallel.pcp)
    return deg_p % deg_d == 0 or deg_d % deg_p == 0


def classify_multihost_mp_mode(
    chips_per_instance: int,
    axes: tuple[int, int, int, int],
    *,
    chips_per_host: int,
    cores_per_chip: int,
) -> Literal["single_host", "mode_a", "mode_b"] | None:
    """Classify native mp layout as single_host, mode_a, mode_b, or None."""
    tp, dp, pp, pcp = axes
    if chips_per_instance <= chips_per_host:
        return "single_host"
    if chips_per_instance % chips_per_host != 0:
        return None
    hosts = chips_per_instance // chips_per_host
    cores_per_host = chips_per_host * cores_per_chip
    if dp == 1:
        return "mode_a"
    if dp > 1 and dp % hosts == 0 and (tp * pp * pcp) <= cores_per_host:
        return "mode_b"
    return None


def _valid_multihost_mp_layout(
    topo: SliceTopology,
    axes: tuple[int, int, int, int],
    chip: ChipSpec | None = None,
) -> bool:
    """Check native multiprocessing multi-host Mode A (dp==1) and Mode B rules."""
    chips_per_host = chip.chips_per_host if chip is not None else DEFAULT_CHIPS_PER_HOST
    return (
        classify_multihost_mp_mode(
            topo.num_chips,
            axes,
            chips_per_host=chips_per_host,
            cores_per_chip=topo.cores_per_chip,
        )
        is not None
    )


def _valid_moe_block_quant_hidden(geom: ModelGeometry) -> bool:
    """Check FP8 serialized block-quant hidden divisibility."""
    if geom.num_experts <= 0 or not geom.weight_block_size:
        return True
    return all(geom.hidden % blk == 0 for blk in geom.weight_block_size)


def _valid_moe_tp_slice(geom: ModelGeometry, tp: int) -> bool:
    """Check load-time 128-padding and FP8 block-quant MoE TP-slice rules."""
    if geom.num_experts <= 0 or geom.moe_inter <= 0:
        return True
    if tp > 1 and geom.moe_inter % tp != 0:
        return False
    if geom.weight_block_size:
        if not _valid_moe_block_quant_hidden(geom):
            return False
        partial_block = any(geom.moe_inter % blk != 0 for blk in geom.weight_block_size)
        if partial_block and tp > 1:
            return False
    return True


def _candidate_moe_dispatches(
    moe_mode: Literal["tp_slice", "ep"],
    dp: int,
) -> list[Literal["allreduce", "ag_ep", "a2a_push"]]:
    """Return legal MoE decode dispatch collectives for mode and DP degree."""
    if moe_mode == "ep" and dp > 1:
        return ["ag_ep", "a2a_push"]
    return ["allreduce"]


# Disaggregated serving requires at least 1 prefill + 1 decode instance.
_MIN_DISAGG_INSTANCES = 2
# Four disjoint core-partitioning axes: PP, TP, DP, PCP.
_ATTN_CORE_PARTITION_AXES = 4
# Three decode MoE collective dispatches: allreduce, ag_ep, a2a_push.
_CORE_GRID_DISPATCH_MODES = 3


def _count_core_valid_layouts(cores: int, *, has_moe: bool) -> int:
    """Count core-partitioned (PP, TP, DP, PCP, DCP, EP, dispatch) tuples."""
    divisors = _divisors(cores)
    moe_multiplier = len(divisors) * _CORE_GRID_DISPATCH_MODES if has_moe else 1
    total = 0
    for pp_degree, tp_degree, dp_degree, pcp_degree in itertools.product(
        divisors, repeat=_ATTN_CORE_PARTITION_AXES
    ):
        if pp_degree * tp_degree * dp_degree * pcp_degree == cores:
            total += len(_divisors(tp_degree)) * moe_multiplier
    return total


def _count_disagg_ratio_slots(
    eval_context: _SolveContext,
    prefill_topology: SliceTopology,
    decode_topology: SliceTopology,
) -> int:
    """Count valid (prefill, decode) instance ratio slots for one slice pair."""
    if eval_context.pd_ratio is not None:
        xp_pin, yd_pin = eval_context.pd_ratio
        total_chips = (
            xp_pin * prefill_topology.num_chips + yd_pin * decode_topology.num_chips
        )
        if eval_context.max_chips is not None and total_chips > eval_context.max_chips:
            return 0
        if (
            eval_context.max_instances is not None
            and (xp_pin + yd_pin) > eval_context.max_instances
        ):
            return 0
        return 1
    effective_max_instances = (
        eval_context.max_instances
        if eval_context.max_instances is not None
        else DEFAULT_MAX_INSTANCES
    )
    if effective_max_instances < _MIN_DISAGG_INSTANCES:
        return 0
    max_single_pool = effective_max_instances - 1
    if eval_context.max_chips is not None:
        max_prefill_instances = min(
            max_single_pool, eval_context.max_chips // prefill_topology.num_chips
        )
        max_decode_instances = min(
            max_single_pool, eval_context.max_chips // decode_topology.num_chips
        )
    else:
        max_prefill_instances = max_single_pool
        max_decode_instances = max_single_pool
    slots = 0
    for prefill_instances in range(1, max_prefill_instances + 1):
        max_decode_for_prefill = min(
            max_decode_instances, effective_max_instances - prefill_instances
        )
        for decode_instances in range(1, max_decode_for_prefill + 1):
            total_chips = (
                prefill_instances * prefill_topology.num_chips
                + decode_instances * decode_topology.num_chips
            )
            if (
                eval_context.max_chips is not None
                and total_chips > eval_context.max_chips
            ):
                continue
            slots += 1
    return slots


def _count_core_grid(
    eval_context: _SolveContext,
    topologies: Sequence[SliceTopology],
) -> int:
    """Count core-partitioned (layout, concurrency) tuples across searched pools."""
    has_moe = eval_context.geom.num_experts > 0
    concurrency_count = len(eval_context.scenario.concurrencies)
    total = 0
    if eval_context.mode in ("auto", "mixed") and (
        eval_context.max_instances is None or eval_context.max_instances >= 1
    ):
        for topology in topologies:
            if (
                eval_context.decode_topology is not None
                and topology.name != eval_context.decode_topology
            ):
                continue
            if (
                eval_context.max_chips is not None
                and topology.num_chips > eval_context.max_chips
            ):
                continue
            total += (
                _count_core_valid_layouts(topology.num_cores, has_moe=has_moe)
                * concurrency_count
            )
    if eval_context.mode in ("auto", "disagg"):
        for decode_topology in topologies:
            if (
                eval_context.decode_topology is not None
                and decode_topology.name != eval_context.decode_topology
            ):
                continue
            decode_layout_count = _count_core_valid_layouts(
                decode_topology.num_cores, has_moe=has_moe
            )
            for prefill_topology in topologies:
                if (
                    eval_context.prefill_topology is not None
                    and prefill_topology.name != eval_context.prefill_topology
                ):
                    continue
                prefill_layout_count = _count_core_valid_layouts(
                    prefill_topology.num_cores, has_moe=has_moe
                )
                ratio_slots = _count_disagg_ratio_slots(
                    eval_context, prefill_topology, decode_topology
                )
                total += (
                    prefill_layout_count
                    * decode_layout_count
                    * ratio_slots
                    * concurrency_count
                )
    return total


def _candidate_decode_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
    *,
    chip: ChipSpec | None = None,
) -> list[Parallel]:
    """Enumerate legal decode parallelism configurations for a given topology."""
    if not _valid_moe_block_quant_hidden(geom):
        return []
    cores = topo.num_cores
    out: list[Parallel] = []
    has_kv_cache = geom.num_kv_heads > 0 or geom.kv_lora_rank > 0
    disallow_dcp = geom.model_id == "qwen4_exp"
    for tp in _divisors(cores):
        if not _valid_head_tp(geom, tp) or not _valid_gdn_degree(geom, tp):
            continue
        dp = cores // tp
        if not _valid_multihost_mp_layout(topo, (tp, dp, 1, 1), chip=chip):
            continue
        if not has_kv_cache or disallow_dcp:
            kv_splits: tuple[int, ...] = (1,)
        else:
            # Hardware KV group divisibility for decode head sharding.
            kv_group = (
                max(1, tp // geom.num_kv_heads)
                if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
                else tp
            )
            kv_splits = _divisors(kv_group) if kv_group > 1 else (1,)
        moe_mode: Literal["tp_slice", "ep"] = (
            "ep"
            if geom.num_experts > 0 and cores > 1 and geom.num_experts >= cores
            else "tp_slice"
        )
        # Runtime guardrail: multi-DP decode MoE requires EP over all DP*TP cores.
        if (
            geom.num_experts > 0
            and moe_mode == "tp_slice"
            and (dp > 1 or not _valid_moe_tp_slice(geom, tp))
        ):
            continue
        ep_val = cores if moe_mode == "ep" else 1
        mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
        modes: list[
            tuple[
                Literal["tp_slice", "ep"],
                int,
                Literal["allreduce", "ag_ep", "a2a_push"],
            ]
        ] = [
            (moe_mode, ep_val, disp) for disp in _candidate_moe_dispatches(moe_mode, dp)
        ]
        # Single-DP TP-sliced MoE uses load-time 128-padding + block-quant checks.
        if moe_mode == "ep" and dp == 1 and _valid_moe_tp_slice(geom, tp):
            modes.append(("tp_slice", 1, "allreduce"))
        for kvs in kv_splits:
            for m_mode, m_ep, m_disp in modes:
                out.append(
                    Parallel(
                        tp=tp,
                        kv_split=kvs,
                        moe=m_mode,
                        moe_dispatch=m_disp,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=dp,
                        ep=m_ep,
                        pp=1,
                        pcp=1,
                        mesh_axes=mesh_ax,
                    ),
                )
    return out


def _prefill_pp_allowed(
    geom: ModelGeometry,
    pp: int,
    n_full_attn: int,
    *,
    for_disagg: bool,
    spec_k: int,
) -> bool:
    """Check runtime PP stage count, disagg full-attention, and spec rules."""
    if geom.num_layers < pp or (geom.model_id == "qwen4_exp" and pp > 1):
        return False
    if for_disagg and 0 < n_full_attn < geom.num_layers and pp > n_full_attn:
        return False
    return not (pp > 1 and spec_k > 0 and not (for_disagg and spec_k == 1))


def _pcp_allowed(
    geom: ModelGeometry,
    pp: int,
    pcp: int,
    spec_k: int,
) -> bool:
    """Check PCP legality against runtime PP, model, and MTP spec_k rules."""
    if pcp <= 1:
        return True
    return geom.model_id != "qwen4_exp" and pp == 1 and spec_k <= 1


def _prefill_axes_allowed(
    geom: ModelGeometry,
    topo: SliceTopology,
    axes: tuple[int, int, int],
    *,
    spec_k: int = 0,
    chip: ChipSpec | None = None,
) -> bool:
    """Check TP/PP/PCP legality against runtime validators and head divisibility."""
    tp, pp, pcp = axes
    if not _pcp_allowed(geom, pp, pcp, spec_k):
        return False
    if tp < 1 or (pp > 1 and tp > 1):
        return False
    if not _valid_multihost_mp_layout(topo, (tp, 1, pp, pcp), chip=chip):
        return False
    return _valid_head_tp(geom, tp) and _valid_gdn_degree(geom, tp * pcp)


def _candidate_prefill_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
    *,
    for_disagg: bool = False,
    spec_k: int = 0,
    chip: ChipSpec | None = None,
) -> list[Parallel]:
    """Enumerate legal prefill parallelism configurations for a given topology."""
    if not _valid_moe_block_quant_hidden(geom) or (
        geom.model_id == "qwen4_exp" and spec_k > 0
    ):
        return []
    cores = topo.num_cores
    n_full_attn = count_full_attn_layers(geom)
    out: list[Parallel] = []
    for pp in _divisors(cores):
        if not _prefill_pp_allowed(
            geom, pp, n_full_attn, for_disagg=for_disagg, spec_k=spec_k
        ):
            continue
        rem = cores // pp
        for pcp in _divisors(rem):
            tp = rem // pcp
            if not _prefill_axes_allowed(
                geom, topo, (tp, pp, pcp), spec_k=spec_k, chip=chip
            ):
                continue
            mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
            for moe_mode, ep_val in _candidate_prefill_moe_modes(geom, tp, pcp):
                dispatches: tuple[Literal["allreduce", "ag_ep"], ...] = (
                    ("ag_ep",)
                    if (moe_mode == "ep" and tp == 1)
                    else (
                        ("allreduce", "ag_ep") if moe_mode == "ep" else ("allreduce",)
                    )
                )
                out.extend(
                    Parallel(
                        tp=tp,
                        kv_split=1,
                        moe=moe_mode,
                        moe_dispatch=disp,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=1,
                        ep=ep_val,
                        pp=pp,
                        pcp=pcp,
                        mesh_axes=mesh_ax,
                    )
                    for disp in dispatches
                )
    return out


def _candidate_prefill_moe_modes(
    geom: ModelGeometry,
    tp: int,
    pcp: int,
) -> list[tuple[Literal["tp_slice", "ep"], int]]:
    """Return legal (moe_mode, ep_degree) pairs for a prefill configuration."""
    if pcp > 1 and geom.num_experts > 0:
        ep_span = tp * pcp
        # Runtime + hardware constraint: PCP>1 requires EP spanning all TP*PCP ranks.
        if geom.num_experts < ep_span or geom.num_experts % ep_span != 0:
            return []
        return [("ep", ep_span)]
    if tp == 1 or geom.num_experts == 0:
        return [("tp_slice", 1)]
    modes: list[tuple[Literal["tp_slice", "ep"], int]] = []
    if geom.num_experts >= tp:
        modes.append(("ep", tp))
    # Unified MoE TP-slice check: load-time 128-padding + block-quant rules.
    if _valid_moe_tp_slice(geom, tp):
        modes.append(("tp_slice", 1))
    return modes


def _pareto_filter(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Filter points to non-dominated Pareto frontier sorted by concurrency."""
    if not points:
        return ()
    frontier: list[ParetoPoint] = []
    for p in points:
        dominated = any(
            q.concurrency <= p.concurrency
            and q.output_throughput_per_chip >= p.output_throughput_per_chip
            and q.p90_interactivity_tok_s >= p.p90_interactivity_tok_s
            and (
                q.concurrency < p.concurrency
                or q.output_throughput_per_chip > p.output_throughput_per_chip
                or q.p90_interactivity_tok_s > p.p90_interactivity_tok_s
            )
            for q in points
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(
        key=lambda x: (
            x.concurrency,
            -x.output_throughput_per_chip,
            -x.p90_interactivity_tok_s,
        ),
    )
    return tuple(frontier)


@dataclass
class _SolveContext:
    """Internal solver context capturing model, hardware, and scenario constants."""

    geom: ModelGeometry
    dtypes: Dtypes
    chip: ChipSpec
    scenario: TrafficScenario
    mode: str
    max_chips: int | None
    max_instances: int | None
    gpu_memory_utilization: float
    hbm_limit_b: float
    hbm_limit_gb: float
    eff_al: float
    spec_k: int
    ctx_full: int
    ctx_mid: int
    eff_isl: int
    n_chunks: int
    page_size: int
    c_max: int
    collective_latency_s: float = 0.0
    prefill_topology: str | None = None
    decode_topology: str | None = None
    pd_ratio: tuple[int, int] | None = None
    enumerated: int = 0
    hbm_fit: int = 0


@dataclass(frozen=True)
class _PrefillOption:
    """Feasible prefill configuration for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    ttft_ms: float
    rate_p: float
    hbm_gb: float
    hbm_bd: HbmBreakdown


@dataclass(frozen=True)
class _DisaggDecodeState:
    """Decode configuration and memory bounds for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    w_base_b: float
    b_hbm_max: int
    b_smem_max: int
    b_kv_phys: float
    offload_b_per_core: float
    fixed_kv_per_seq_b: float = 0.0


@dataclass(frozen=True)
class _MixedPointBounds:
    """Precomputed base weights, capacity limits, and TTFT floor for mixed search."""

    w_base_b: float
    b_kv_phys_core: float
    b_hbm_max: int
    b_smem_max: int
    sf_ttft: StepFloor
    fixed_kv_per_seq_b: float = 0.0


@dataclass(frozen=True)
class _DisaggRatioTarget:
    """Concurrency and instance counts (xP, yD) for a disaggregated point."""

    c: int
    x_p: int
    y_d: int
    total_chips: int


@dataclass(frozen=True)
class _DisaggBatchStats:
    """Resolved decode batch size and effective prefill rate for a ratio target."""

    b_res: int
    b_local: int
    ttft_eff_ms: float
    rate_p_eff: float


@dataclass(frozen=True)
class _PoolPlanInput:
    """Inputs needed to materialize one PoolMeshPlan instance pool."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topo: SliceTopology
    par: Parallel
    hbm_used_gb: float
    hbm_limit_gb: float
    max_num_seqs: int | None
    max_model_len: int
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0


def _floor_times(sf: StepFloor) -> FloorTimes:
    """Extract decode-step resource times (ms) from a StepFloor."""
    return FloorTimes(
        hbm_ms=sf.hbm_s * UNIT_MILLIS_PER_SECOND,
        matmul_ms=sf.matmul_s * UNIT_MILLIS_PER_SECOND,
        ici_ms=sf.ici_s * UNIT_MILLIS_PER_SECOND,
        chain_ms=sf.chain_s * UNIT_MILLIS_PER_SECOND,
        hop_latency_ms=sf.latency_assumed_s * UNIT_MILLIS_PER_SECOND,
        hard_floor_ms=sf.hard_floor_s * UNIT_MILLIS_PER_SECOND,
        binding=sf.binding,
    )


def _build_pool_mesh_plan(spec: _PoolPlanInput) -> PoolMeshPlan:
    """Construct a PoolMeshPlan from topology, Parallel, and HBM usage."""
    axes = (
        spec.par.mesh_axes
        if spec.par.mesh_axes is not None
        else submesh_for_group(spec.par.tp, spec.topo, spec.topo.cores_per_chip)
    )
    return PoolMeshPlan(
        role=spec.role,
        instances=spec.instances,
        topology=spec.topo.name,
        chips_per_instance=spec.topo.num_chips,
        tp=spec.par.tp,
        dp=spec.par.dp,
        ep=spec.par.ep,
        pp=spec.par.pp,
        pcp=spec.par.pcp,
        kv_split=spec.par.kv_split,
        mesh_axes=axes,
        hbm_used_gb=spec.hbm_used_gb,
        hbm_limit_gb=spec.hbm_limit_gb,
        moe_dispatch=spec.par.moe_dispatch,
        hbm_weights_gb=spec.hbm_weights_gb,
        hbm_kv_gb=spec.hbm_kv_gb,
        hbm_state_gb=spec.hbm_state_gb,
        max_num_seqs=spec.max_num_seqs,
        max_model_len=spec.max_model_len,
    )


def _eval_decode_floor(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    batch: int,
) -> StepFloor:
    """Evaluate single-step or speculative-cycle floor at midpoint context."""
    opts = FloorOptions(
        dt=eval_context.dtypes,
        collective_latency_s=eval_context.collective_latency_s,
        topology=topo,
    )
    if eval_context.spec_k > 0:
        return spec_cycle_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            SpecShape(batch=batch, context=eval_context.ctx_mid, k=eval_context.spec_k),
            opts,
        )
    return step_floor(
        eval_context.geom,
        eval_context.chip,
        par,
        StepShape(batch=batch, context=eval_context.ctx_mid),
        opts,
    )


def _prefill_step_floor(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    prefix_hit_rate: float,
) -> StepFloor:
    """Evaluate prompt prefill floor for a given topology, parallel, and hit rate."""
    return step_floor(
        eval_context.geom,
        eval_context.chip,
        par,
        StepShape(
            batch=0,
            rows_per_seq=0,
            context=0,
            state_writes=0,
            prefill_tokens=eval_context.scenario.isl,
            prefix_hit_rate=prefix_hit_rate,
        ),
        FloorOptions(
            dt=eval_context.dtypes,
            collective_latency_s=eval_context.collective_latency_s,
            topology=topo,
        ),
    )


def _clamp_resident_batch(
    eval_context: _SolveContext,
    par: Parallel,
    b_start: int,
) -> tuple[int, HbmBreakdown] | None:
    """Decrease resident batch until HBM usage fits within hbm_limit_b."""
    b_res = b_start
    hbm_bd = hbm_breakdown_per_core_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        SeqKvLayout(b_res, eval_context.ctx_full),
        chip=eval_context.chip,
    )
    while b_res > 1 and hbm_bd.total > eval_context.hbm_limit_b:
        b_res -= 1
        hbm_bd = hbm_breakdown_per_core_bytes(
            eval_context.geom,
            eval_context.dtypes,
            par,
            SeqKvLayout(b_res, eval_context.ctx_full),
            chip=eval_context.chip,
        )
    if hbm_bd.total > eval_context.hbm_limit_b:
        return None
    return b_res, hbm_bd


def _find_prefill_options(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[_PrefillOption], float]:
    """Enumerate all legal, HBM-feasible prefill configurations without collapsing."""
    prefill_options: list[_PrefillOption] = []
    min_req_ttft_ms = math.inf
    for topo in topos:
        if (
            eval_context.prefill_topology is not None
            and topo.name != eval_context.prefill_topology
        ):
            continue
        for par in _candidate_prefill_parallels(
            eval_context.geom,
            topo,
            for_disagg=True,
            spec_k=eval_context.spec_k,
            chip=eval_context.chip,
        ):
            hbm_bd = hbm_breakdown_per_core_bytes(
                eval_context.geom,
                eval_context.dtypes,
                par,
                SeqKvLayout(1, eval_context.scenario.isl),
                chip=eval_context.chip,
            )
            hbm_b = hbm_bd.total
            if hbm_b > eval_context.hbm_limit_b:
                continue
            sf_p = _prefill_step_floor(
                eval_context, topo, par, eval_context.scenario.prefix_hit_rate
            )
            pp_factor = 1.0 + float(par.pp - 1) / float(max(1, eval_context.n_chunks))
            ttft_ms = sf_p.serial_estimate_s * pp_factor * UNIT_MILLIS_PER_SECOND
            floor_ttft_ms = sf_p.hard_floor_s * pp_factor * UNIT_MILLIS_PER_SECOND
            min_req_ttft_ms = min(min_req_ttft_ms, floor_ttft_ms)
            rate_p = 1.0 / max(1e-9, sf_p.serial_estimate_s)
            prefill_options.append(
                _PrefillOption(
                    topo=topo,
                    par=par,
                    ttft_ms=ttft_ms,
                    rate_p=rate_p,
                    hbm_gb=hbm_b / UNIT_GIGA,
                    hbm_bd=hbm_bd,
                )
            )
    return prefill_options, min_req_ttft_ms


def _solve_mixed_think_batch(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    sf_ttft: StepFloor,
) -> int:
    """Resolve Little's-law active local batch size under non-zero think time."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_local = max(1, math.ceil(c / max(1, par.dp)))
    if eval_context.scenario.think_time_s <= 0.0:
        return b_local
    for _ in range(2):
        sf_dec_tmp = _eval_decode_floor(eval_context, topo, par, b_local)
        service_time_s = (
            sf_ttft.serial_estimate_s + dec_steps_per_req * sf_dec_tmp.serial_estimate_s
        )
        active_concurrency = (
            float(c)
            * service_time_s
            / (service_time_s + eval_context.scenario.think_time_s)
        )
        b_local = max(1, math.ceil(active_concurrency / max(1, par.dp)))
    return b_local


def _eval_mixed_step_floors(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    b_res: int,
    h_eff: float,
) -> tuple[int, StepFloor, StepFloor, float]:
    """Evaluate decode and mixed-chunk step floors at effective hit rate."""
    eff_isl_c = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
    cached_prefix_c = max(0, eval_context.scenario.isl - eff_isl_c)
    n_chunks_c = max(
        1, math.ceil(eff_isl_c / max(1, eval_context.scenario.prefill_chunk))
    )
    chunk_tok_c = min(eff_isl_c, eval_context.scenario.prefill_chunk)
    avg_prior = cached_prefix_c + CAUSAL_MIDPOINT_HALF * float(
        max(0, eff_isl_c - chunk_tok_c)
    )
    mix_span = avg_prior + float(chunk_tok_c)
    mix_pre_total = round(mix_span)
    mix_hit_rate = avg_prior / mix_span if mix_span > 0.0 else 0.0
    floor_opts = FloorOptions(
        dt=eval_context.dtypes,
        collective_latency_s=eval_context.collective_latency_s,
        topology=topo,
    )
    sf_dec = _eval_decode_floor(eval_context, topo, par, b_res)
    sf_ver_only = (
        step_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            StepShape(
                batch=b_res,
                rows_per_seq=eval_context.spec_k + 1,
                context=eval_context.ctx_mid,
                state_writes=1,
            ),
            floor_opts,
        )
        if eval_context.spec_k > 0
        else sf_dec
    )
    draft_serial_s = max(0.0, sf_dec.serial_estimate_s - sf_ver_only.serial_estimate_s)
    sf_mix = step_floor(
        eval_context.geom,
        eval_context.chip,
        par,
        StepShape(
            batch=b_res,
            rows_per_seq=eval_context.spec_k + 1,
            context=eval_context.ctx_mid,
            state_writes=1,
            prefill_tokens=mix_pre_total,
            prefix_hit_rate=mix_hit_rate,
        ),
        floor_opts,
    )
    return n_chunks_c, sf_dec, sf_mix, sf_mix.serial_estimate_s + draft_serial_s


def _evaluate_mixed_point(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    bounds: _MixedPointBounds,
) -> ParetoPoint | None:
    """Evaluate one concurrency target for a single-pool mixed serving configuration."""
    if bounds.b_hbm_max < 1 or bounds.b_smem_max < 1:
        return None
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    offload_b_per_core = (
        float(eval_context.scenario.kv_offload_gb)
        * UNIT_GIGA
        / float(max(1, topo.num_cores))
    )
    b_local = _solve_mixed_think_batch(eval_context, topo, par, c, bounds.sf_ttft)
    b_start = min(b_local, bounds.b_hbm_max, bounds.b_smem_max)
    if eval_context.scenario.max_num_seqs is not None:
        b_start = min(b_start, eval_context.scenario.max_num_seqs)
    clamped = _clamp_resident_batch(eval_context, par, b_start)
    if clamped is None:
        return None
    b_res, hbm_bd = clamped
    hbm_b = hbm_bd.total
    eval_context.hbm_fit += 1
    b_ws = max(1, math.ceil(c / max(1, par.dp)))
    cap_tok, h_eff = kv_capacity_and_hit_rate(
        eval_context.geom,
        eval_context.dtypes,
        par,
        b_res,
        KvFitInput(
            b_ws=b_ws,
            ctx_full=eval_context.ctx_full,
            hbm_limit_b=eval_context.hbm_limit_b,
            w_base_b=bounds.w_base_b,
            b_kv_phys_core=bounds.b_kv_phys_core,
            offload_b_per_core=offload_b_per_core,
            prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
            kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
            fixed_kv_per_seq_bytes=bounds.fixed_kv_per_seq_b,
        ),
        chip=eval_context.chip,
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    sf_ttft_c = (
        _prefill_step_floor(eval_context, topo, par, h_eff)
        if h_eff != eval_context.scenario.prefix_hit_rate
        else bounds.sf_ttft
    )
    n_chunks_c, sf_dec, sf_mix, mix_serial_s = _eval_mixed_step_floors(
        eval_context, topo, par, b_res, h_eff
    )
    dec_slots_per_req = dec_steps_per_req / float(max(1, b_res))
    if float(b_res * n_chunks_c) <= dec_steps_per_req:
        replica_req_time_s = (
            dec_slots_per_req - float(n_chunks_c)
        ) * sf_dec.serial_estimate_s + float(n_chunks_c) * mix_serial_s
    else:
        eff_isl_c = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
        cached_prefix_c = max(0, eval_context.scenario.isl - eff_isl_c)
        chunk_tok_c = min(eff_isl_c, eval_context.scenario.prefill_chunk)
        avg_prior = cached_prefix_c + CAUSAL_MIDPOINT_HALF * float(
            max(0, eff_isl_c - chunk_tok_c)
        )
        mix_span = avg_prior + float(chunk_tok_c)
        sf_pure_chunk = step_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=round(mix_span),
                prefix_hit_rate=avg_prior / mix_span if mix_span > 0.0 else 0.0,
            ),
            FloorOptions(
                dt=eval_context.dtypes,
                collective_latency_s=eval_context.collective_latency_s,
                topology=topo,
            ),
        )
        replica_req_time_s = (
            dec_slots_per_req * mix_serial_s
            + (float(n_chunks_c) - dec_slots_per_req) * sf_pure_chunk.serial_estimate_s
        )
    total_req_work_s = float(b_res) * replica_req_time_s
    res_cap = float(b_res * max(1, par.dp))
    c_served = min(float(c), res_cap)
    server_cap_rate = res_cap / max(1e-9, total_req_work_s)
    service_time_final_s = (
        sf_ttft_c.serial_estimate_s + dec_steps_per_req * sf_dec.serial_estimate_s
    )
    rate_req_s = (
        min(
            server_cap_rate,
            float(c)
            / max(1e-9, service_time_final_s + eval_context.scenario.think_time_s),
        )
        if eval_context.scenario.think_time_s > 0.0
        else c_served / max(1e-9, total_req_work_s)
    )
    p_mix = min(1.0, float(b_res * n_chunks_c) / float(max(1.0, dec_steps_per_req)))
    p90_step_s = (1.0 - p_mix) * sf_dec.serial_estimate_s + p_mix * mix_serial_s
    p90_tpot_ms = p90_step_s * UNIT_MILLIS_PER_SECOND / eval_context.eff_al
    c_demand = min(float(c), float(b_local * max(1, par.dp)))
    queue_wait_s = (
        max(0.0, (c_demand / max(1e-9, res_cap)) - 1.0) * service_time_final_s
    )
    ttft_base_ms = (sf_ttft_c.serial_estimate_s + queue_wait_s) * UNIT_MILLIS_PER_SECOND
    p90_ttft_ms = ttft_base_ms + (
        sf_mix.serial_estimate_s * UNIT_MILLIS_PER_SECOND if c > 1 else 0.0
    )
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, p90_tpot_ms)
    out_tput = rate_req_s * float(eval_context.scenario.osl) / float(topo.num_chips)
    tot_tput = (
        rate_req_s
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(topo.num_chips)
    )
    slo_ok = (
        eval_context.scenario.ttft_slo_ms is None
        or p90_ttft_ms <= eval_context.scenario.ttft_slo_ms
    ) and (
        eval_context.scenario.tpot_slo_ms is None
        or p90_tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    max_seqs_flag = (
        min(b_res, eval_context.scenario.max_num_seqs)
        if eval_context.scenario.max_num_seqs is not None
        else b_res
    )
    pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="mixed",
            instances=1,
            topo=topo,
            par=par,
            hbm_used_gb=hbm_b / UNIT_GIGA,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            max_num_seqs=max_seqs_flag,
            max_model_len=eval_context.ctx_full,
            hbm_weights_gb=hbm_bd.weights / UNIT_GIGA,
            hbm_kv_gb=hbm_bd.kv / UNIT_GIGA,
            hbm_state_gb=hbm_bd.state / UNIT_GIGA,
        )
    )
    binding = (
        f"dec:{sf_dec.binding},mix:{sf_mix.binding}" if p_mix > 0.0 else sf_dec.binding
    )
    return ParetoPoint(
        mode="mixed",
        concurrency=c,
        total_chips=topo.num_chips,
        prefill_pool=None,
        decode_pool=pool,
        p90_ttft_ms=p90_ttft_ms,
        p90_tpot_ms=p90_tpot_ms,
        p90_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=p_mix,
        binding=binding,
        slo_met=slo_ok,
        slo_verdict=(SLO_VERDICT_ESTIMATE_MEETS if slo_ok else SLO_VERDICT_RULED_OUT),
        decode_floor=_floor_times(sf_dec),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
    )


def _decode_capacity_profile(
    eval_context: _SolveContext,
    par: Parallel,
) -> tuple[float, int, int, float, float, float]:
    """Compute base weights, batch limits, and separated KV byte components."""
    w_base_b = hbm_footprint_per_core_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        SeqKvLayout(0, 0),
        chip=eval_context.chip,
    )
    hbm_c1 = hbm_footprint_per_core_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        SeqKvLayout(1, eval_context.ctx_full),
        chip=eval_context.chip,
    )
    per_seq_b = hbm_c1 - w_base_b
    b_hbm_max = (
        int((eval_context.hbm_limit_b - w_base_b) // per_seq_b)
        if (hbm_c1 <= eval_context.hbm_limit_b and per_seq_b > 0.0)
        else (eval_context.c_max if hbm_c1 <= eval_context.hbm_limit_b else 0)
    )
    eff_page_size = eval_context.page_size
    if eval_context.scenario.page_size is None and b_hbm_max >= 1:
        target_seqs = min(
            b_hbm_max,
            max(1, math.ceil(eval_context.c_max / max(1, par.dp))),
        )
        if eval_context.scenario.max_num_seqs is not None:
            target_seqs = min(target_seqs, eval_context.scenario.max_num_seqs)
        eff_page_size = max(
            eff_page_size,
            tpu_min_page_size(
                eval_context.ctx_full,
                par.kv_split,
                target_seqs,
                pcp=par.pcp,
                page_table_bytes=eval_context.chip.smem_page_table_bytes,
            ),
        )
    b_smem_max = smem_max_seqs(
        eval_context.ctx_full,
        par.kv_split,
        eff_page_size,
        page_table_bytes=eval_context.chip.smem_page_table_bytes,
    )
    b_kv_phys_core, fixed_kv_per_seq_b = kv_bytes_per_core_parts(
        eval_context.geom,
        eval_context.dtypes,
        par,
        eval_context.ctx_full,
        chip=eval_context.chip,
    )
    return (
        w_base_b,
        b_hbm_max,
        b_smem_max,
        b_kv_phys_core,
        fixed_kv_per_seq_b,
        hbm_c1,
    )


def _evaluate_mixed_candidates(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[ParetoPoint], float, float]:
    """Evaluate all single-pool mixed candidate parallelisms across topologies."""
    candidates: list[ParetoPoint] = []
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf
    for topo in topos:
        if (
            eval_context.decode_topology is not None
            and topo.name != eval_context.decode_topology
        ):
            continue
        for par in _candidate_decode_parallels(
            eval_context.geom, topo, chip=eval_context.chip
        ):
            (
                w_base_b,
                b_hbm_max,
                b_smem_max,
                b_kv_phys_core,
                fixed_kv_per_seq_b,
                hbm_c1,
            ) = _decode_capacity_profile(eval_context, par)
            if hbm_c1 <= eval_context.hbm_limit_b and b_smem_max >= 1:
                sf_c1 = _eval_decode_floor(eval_context, topo, par, 1)
                min_c1_tpot_ms = min(
                    min_c1_tpot_ms,
                    sf_c1.hard_floor_s * UNIT_MILLIS_PER_SECOND / eval_context.eff_al,
                )
            if eval_context.mode not in ("auto", "mixed"):
                continue
            sf_ttft = _prefill_step_floor(
                eval_context, topo, par, eval_context.scenario.prefix_hit_rate
            )
            min_req_ttft_ms = min(
                min_req_ttft_ms, sf_ttft.hard_floor_s * UNIT_MILLIS_PER_SECOND
            )
            bounds = _MixedPointBounds(
                w_base_b=w_base_b,
                b_kv_phys_core=b_kv_phys_core,
                b_hbm_max=b_hbm_max,
                b_smem_max=b_smem_max,
                sf_ttft=sf_ttft,
                fixed_kv_per_seq_b=fixed_kv_per_seq_b,
            )
            for c in eval_context.scenario.concurrencies:
                eval_context.enumerated += 1
                pt = _evaluate_mixed_point(eval_context, topo, par, c, bounds)
                if pt is not None:
                    candidates.append(pt)
    return candidates, min_c1_tpot_ms, min_req_ttft_ms


def _compute_disagg_metrics(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    target: _DisaggRatioTarget,
    batch_stats: _DisaggBatchStats,
    sf_d: StepFloor,
) -> tuple[float, float, float, float, float]:
    """Compute serving latencies and throughputs for disaggregated configuration."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    p90_tpot_ms = sf_d.serial_estimate_s * UNIT_MILLIS_PER_SECOND / eval_context.eff_al
    interactivity = UNIT_MILLIS_PER_SECOND / max(1e-6, p90_tpot_ms)
    res_cap_d = float(batch_stats.b_res * max(1, target.y_d * dec.par.dp))
    c_served_d = min(float(target.c), res_cap_d)
    rate_d_cap = c_served_d / (dec_steps_per_req * max(1e-9, sf_d.serial_estimate_s))
    c_demand_d = min(
        float(target.c), float(batch_stats.b_local * max(1, target.y_d * dec.par.dp))
    )
    queue_wait_d_ms = (
        max(0.0, (c_demand_d / max(1e-9, res_cap_d)) - 1.0)
        * (dec_steps_per_req * sf_d.serial_estimate_s)
        * UNIT_MILLIS_PER_SECOND
    )
    p90_ttft_d_ms = batch_stats.ttft_eff_ms + queue_wait_d_ms
    if eval_context.scenario.think_time_s > 0.0:
        client_rate = float(target.c) / max(
            1e-9,
            batch_stats.ttft_eff_ms * UNIT_MICRO * UNIT_MILLIS_PER_SECOND
            + dec_steps_per_req * sf_d.serial_estimate_s
            + eval_context.scenario.think_time_s,
        )
        coupled_rate = min(target.x_p * batch_stats.rate_p_eff, rate_d_cap, client_rate)
    else:
        coupled_rate = min(target.x_p * batch_stats.rate_p_eff, rate_d_cap)
    out_tput = (
        coupled_rate * float(eval_context.scenario.osl) / float(target.total_chips)
    )
    tot_tput = (
        coupled_rate
        * float(eval_context.scenario.isl + eval_context.scenario.osl)
        / float(target.total_chips)
    )
    return p90_ttft_d_ms, p90_tpot_ms, interactivity, out_tput, tot_tput


def _solve_disagg_think_batch(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
) -> int:
    """Resolve Little's-law active local batch for disaggregated decode pool."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_local = max(1, math.ceil(target.c / max(1, target.y_d * dec.par.dp)))
    if eval_context.scenario.think_time_s <= 0.0:
        return b_local
    for _ in range(2):
        decode_step_floor = _eval_decode_floor(eval_context, dec.topo, dec.par, b_local)
        service_time_s = (
            pref.ttft_ms / UNIT_MILLIS_PER_SECOND
            + dec_steps_per_req * decode_step_floor.serial_estimate_s
        )
        active_concurrency = (
            float(target.c)
            * service_time_s
            / (service_time_s + eval_context.scenario.think_time_s)
        )
        b_local = max(
            1, math.ceil(active_concurrency / max(1, target.y_d * dec.par.dp))
        )
    return b_local


def _resolve_effective_prefill(
    eval_context: _SolveContext,
    pref: _PrefillOption,
    h_eff: float,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> tuple[float, float]:
    """Recompute cached prefill (ttft_ms, rate_p) when prefix hit rate is clamped."""
    if h_eff >= eval_context.scenario.prefix_hit_rate:
        return pref.ttft_ms, pref.rate_p
    p_key = (pref.topo.name, pref.par, round(h_eff, 6))
    cached_p = cache.get(p_key)
    if cached_p is not None:
        return cached_p
    eff_isl_p = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
    n_chunks_p = max(
        1, math.ceil(eff_isl_p / max(1, eval_context.scenario.prefill_chunk))
    )
    sf_p_eff = _prefill_step_floor(eval_context, pref.topo, pref.par, h_eff)
    ttft_p_eff = (
        sf_p_eff.serial_estimate_s
        * (1.0 + float(pref.par.pp - 1) / float(max(1, n_chunks_p)))
        * UNIT_MILLIS_PER_SECOND
    )
    rate_p_eff = 1.0 / max(1e-9, sf_p_eff.serial_estimate_s)
    cached_p = (ttft_p_eff, rate_p_eff)
    cache[p_key] = cached_p
    return cached_p


def _evaluate_disagg_instance_pair(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> ParetoPoint | None:
    """Evaluate one instance ratio pair (x_p, y_d) for disaggregated serving."""
    if (
        eval_context.max_chips is not None
        and target.total_chips > eval_context.max_chips
    ):
        return None
    if dec.b_hbm_max < 1 or dec.b_smem_max < 1:
        return None
    b_local = _solve_disagg_think_batch(eval_context, dec, pref, target)
    b_start = min(b_local, dec.b_hbm_max, dec.b_smem_max)
    if eval_context.scenario.max_num_seqs is not None:
        b_start = min(b_start, eval_context.scenario.max_num_seqs)
    clamped = _clamp_resident_batch(eval_context, dec.par, b_start)
    if clamped is None:
        return None
    b_res, hbm_bd_d = clamped
    hbm_d = hbm_bd_d.total
    eval_context.hbm_fit += 1
    b_ws_d = max(1, math.ceil(target.c / max(1, target.y_d * dec.par.dp)))
    cap_tok, h_eff = kv_capacity_and_hit_rate(
        eval_context.geom,
        eval_context.dtypes,
        dec.par,
        b_res,
        KvFitInput(
            b_ws=b_ws_d,
            ctx_full=eval_context.ctx_full,
            hbm_limit_b=eval_context.hbm_limit_b,
            w_base_b=dec.w_base_b,
            b_kv_phys_core=dec.b_kv_phys,
            offload_b_per_core=dec.offload_b_per_core,
            prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
            kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
            fixed_kv_per_seq_bytes=dec.fixed_kv_per_seq_b,
        ),
        chip=eval_context.chip,
    )
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    ttft_eff_ms, rate_p_eff = _resolve_effective_prefill(
        eval_context, pref, h_eff, cache
    )
    sf_d = _eval_decode_floor(eval_context, dec.topo, dec.par, b_res)
    p90_ttft_d_ms, p90_tpot_ms, interactivity, out_tput, tot_tput = (
        _compute_disagg_metrics(
            eval_context,
            dec,
            target,
            _DisaggBatchStats(b_res, b_local, ttft_eff_ms, rate_p_eff),
            sf_d,
        )
    )
    slo_ok = (
        eval_context.scenario.ttft_slo_ms is None
        or p90_ttft_d_ms <= eval_context.scenario.ttft_slo_ms
    ) and (
        eval_context.scenario.tpot_slo_ms is None
        or p90_tpot_ms <= eval_context.scenario.tpot_slo_ms
    )
    max_seqs_d = (
        min(b_res, eval_context.scenario.max_num_seqs)
        if eval_context.scenario.max_num_seqs is not None
        else b_res
    )
    p_pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="prefill",
            instances=target.x_p,
            topo=pref.topo,
            par=pref.par,
            hbm_used_gb=pref.hbm_gb,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            max_num_seqs=None,
            max_model_len=eval_context.ctx_full,
            hbm_weights_gb=pref.hbm_bd.weights / UNIT_GIGA,
            hbm_kv_gb=pref.hbm_bd.kv / UNIT_GIGA,
            hbm_state_gb=pref.hbm_bd.state / UNIT_GIGA,
        )
    )
    d_pool = _build_pool_mesh_plan(
        _PoolPlanInput(
            role="decode",
            instances=target.y_d,
            topo=dec.topo,
            par=dec.par,
            hbm_used_gb=hbm_d / UNIT_GIGA,
            hbm_limit_gb=eval_context.hbm_limit_gb,
            max_num_seqs=max_seqs_d,
            max_model_len=eval_context.ctx_full,
            hbm_weights_gb=hbm_bd_d.weights / UNIT_GIGA,
            hbm_kv_gb=hbm_bd_d.kv / UNIT_GIGA,
            hbm_state_gb=hbm_bd_d.state / UNIT_GIGA,
        )
    )
    return ParetoPoint(
        mode="disagg",
        concurrency=target.c,
        total_chips=target.total_chips,
        prefill_pool=p_pool,
        decode_pool=d_pool,
        p90_ttft_ms=p90_ttft_d_ms,
        p90_tpot_ms=p90_tpot_ms,
        p90_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=0.0,
        binding=sf_d.binding,
        slo_met=slo_ok,
        slo_verdict=(SLO_VERDICT_ESTIMATE_MEETS if slo_ok else SLO_VERDICT_RULED_OUT),
        decode_floor=_floor_times(sf_d),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
    )


def _search_disagg_instance_ratios(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    c: int,
    cache: dict[tuple[str, Parallel, float], tuple[float, float]],
) -> list[ParetoPoint]:
    """Search instance multiplier ratios (x_p, y_d) for one prefill-decode pair."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_one = math.ceil(c / max(1, dec.par.dp))
    sf_d_one = _eval_decode_floor(eval_context, dec.topo, dec.par, b_one)
    rate_d_one = float(c) / (dec_steps_per_req * max(1e-9, sf_d_one.serial_estimate_s))
    eff_max_inst = (
        eval_context.max_instances
        if eval_context.max_instances is not None
        else DEFAULT_MAX_INSTANCES
    )
    if eval_context.max_chips is None:
        ratio_xp = math.ceil(rate_d_one / max(1e-9, pref.rate_p)) + 1
        ratio_yd = math.ceil(pref.rate_p / max(1e-9, rate_d_one)) + 1
        max_xp = min(eff_max_inst, max(1, ratio_xp))
        max_yd = min(eff_max_inst, max(1, ratio_yd))
    else:
        max_xp = min(
            eff_max_inst, max(1, eval_context.max_chips // pref.topo.num_chips)
        )
        max_yd = min(eff_max_inst, max(1, eval_context.max_chips // dec.topo.num_chips))
    points: list[ParetoPoint] = []
    ratio_pairs = (
        [eval_context.pd_ratio]
        if eval_context.pd_ratio is not None
        else [
            (x_p, y_d) for x_p in range(1, max_xp + 1) for y_d in range(1, max_yd + 1)
        ]
    )
    for x_p, y_d in ratio_pairs:
        total_chips = x_p * pref.topo.num_chips + y_d * dec.topo.num_chips
        if eval_context.max_chips is not None and total_chips > eval_context.max_chips:
            continue
        if (
            eval_context.max_instances is not None
            and (x_p + y_d) > eval_context.max_instances
        ):
            continue
        eval_context.enumerated += 1
        pt = _evaluate_disagg_instance_pair(
            eval_context,
            dec,
            pref,
            _DisaggRatioTarget(c=c, x_p=x_p, y_d=y_d, total_chips=total_chips),
            cache,
        )
        if pt is not None:
            points.append(pt)
    return points


def _evaluate_disaggregated_candidates(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
    prefill_options: list[_PrefillOption],
) -> list[ParetoPoint]:
    """Evaluate all unconstrained disaggregated serving candidate configurations."""
    if eval_context.mode not in ("auto", "disagg") or not prefill_options:
        return []
    candidates: list[ParetoPoint] = []
    cache: dict[tuple[str, Parallel, float], tuple[float, float]] = {}
    for topo_d in topos:
        if (
            eval_context.decode_topology is not None
            and topo_d.name != eval_context.decode_topology
        ):
            continue
        offload_b = (
            float(eval_context.scenario.kv_offload_gb)
            * UNIT_GIGA
            / float(max(1, topo_d.num_cores))
        )
        for par_d in _candidate_decode_parallels(
            eval_context.geom, topo_d, chip=eval_context.chip
        ):
            w_base_d, b_hbm_max_d, b_smem_max_d, b_kv_phys_d, fixed_kv_d, _ = (
                _decode_capacity_profile(eval_context, par_d)
            )
            dec_state = _DisaggDecodeState(
                topo=topo_d,
                par=par_d,
                w_base_b=w_base_d,
                b_hbm_max=b_hbm_max_d,
                b_smem_max=b_smem_max_d,
                b_kv_phys=b_kv_phys_d,
                offload_b_per_core=offload_b,
                fixed_kv_per_seq_b=fixed_kv_d,
            )
            for c in eval_context.scenario.concurrencies:
                for pref in prefill_options:
                    if not _valid_raiden_gdn_pair(
                        eval_context.geom, pref.par, par_d
                    ) or not _valid_raiden_kv_pair(eval_context.geom, pref.par, par_d):
                        continue
                    candidates.extend(
                        _search_disagg_instance_ratios(
                            eval_context, dec_state, pref, c, cache
                        ),
                    )
    return candidates


def _resolve_solve_topologies(
    opts: SolveOptions,
    chip: ChipSpec | None = None,
) -> list[SliceTopology]:
    """Resolve provisioned topologies within max_chips budget, rejecting subhost."""
    if opts.topologies is not None:
        topologies = list(opts.topologies)
    elif chip is not None:
        topologies = list(topologies_for_chip(chip).values())
    else:
        topologies = list(TOPOLOGIES.values())
    if opts.max_chips is not None:
        if topologies:
            min_topo = min(topologies, key=lambda topo: topo.num_chips)
            if opts.max_chips < min_topo.num_chips:
                msg = (
                    f"max_chips={opts.max_chips} is below minimum provisionable TPU "
                    f"topology {min_topo.name!r} ({min_topo.num_chips} chips)"
                )
                raise ValueError(msg)
        return [t for t in topologies if t.num_chips <= opts.max_chips]
    return topologies


def _qwen4_scenario_warnings(
    geom: ModelGeometry,
    ctx_full: int,
    spec_k: int,
) -> tuple[str, ...]:
    """Enforce TPUQwen4ExpConfig scenario limits (context budget and spec decode)."""
    if geom.model_id != "qwen4_exp":
        return ()
    msgs: list[str] = []
    if geom.qsa_budget > 0 and ctx_full > geom.qsa_budget:
        msgs.append(
            f"UNSUPPORTED_CONFIG: qwen4_exp max_model_len={ctx_full} exceeds "
            f"indexer_budget={geom.qsa_budget} on dense bring-up path."
        )
    if spec_k > 0:
        msgs.append(
            "UNSUPPORTED_CONFIG: qwen4_exp does not support speculative decoding "
            f"(spec_tokens={spec_k})."
        )
    return tuple(msgs)


def _explain_mode_rejection(
    layout: Mapping[str, Mapping[str, int]],
    mode: str,
) -> str | None:
    """Return rejection reason if layout contradicts solver mode."""
    if mode == "mixed" and "prefill" in layout:
        return "a prefill and decode layout is only searched in disaggregated mode"
    if mode == "disagg" and "prefill" not in layout:
        return "a single-pool layout is only searched in mixed mode"
    return None


def _explain_decode_tp_rejection(
    geom: ModelGeometry,
    decode_tp: int,
    *,
    ep_degree: int = 1,
) -> str | None:
    """Return rejection reason if decode tensor-parallel degree is invalid."""
    if not _valid_moe_block_quant_hidden(geom):
        return "model hidden size does not divide FP8 weight_block_size"
    if not _valid_head_tp(geom, decode_tp):
        return f"TP{decode_tp} does not divide the attention heads"
    if not _valid_gdn_degree(geom, decode_tp):
        return f"TP{decode_tp} is not a valid linear-attention shard degree"
    if ep_degree <= 1 and not _valid_moe_tp_slice(geom, decode_tp):
        return f"TP{decode_tp} does not satisfy MoE TP-slice divisibility rules"
    return None


def _explain_decode_rejection(
    geom: ModelGeometry,
    decode_axes: Mapping[str, int],
    decode_topologies: Sequence[SliceTopology],
    *,
    chip: ChipSpec | None = None,
) -> str | None:
    """Return rejection reason if decode layout is invalid or not enumerated."""
    reason: str | None = None
    if decode_axes.get("pp", 1) > 1 or decode_axes.get("pcp", 1) > 1:
        reason = "decode pools are enumerated with PP1 and PCP1"
    elif geom.model_id == "qwen4_exp" and decode_axes.get("kv_split", 1) > 1:
        reason = "qwen4_exp bring-up path does not support DCP > 1"
    if reason is not None:
        return reason

    decode_tp = decode_axes.get("tp", 1)
    reference_ep = decode_axes.get("ep", 1)
    tp_reason = _explain_decode_tp_rejection(geom, decode_tp, ep_degree=reference_ep)
    if tp_reason is not None:
        return tp_reason

    decode_dp = decode_axes.get("dp", 1)
    needed_cores = decode_tp * decode_dp
    core_matching_topologies = [
        topology for topology in decode_topologies if topology.num_cores == needed_cores
    ]
    if not core_matching_topologies:
        return (
            f"TP{decode_tp} x DP{decode_dp} needs {needed_cores} cores and no searched"
            " slice has that many"
        )

    if not any(
        _valid_multihost_mp_layout(
            slice_topology, (decode_tp, decode_dp, 1, 1), chip=chip
        )
        for slice_topology in core_matching_topologies
    ):
        return "multi-host multiprocessing requires Mode A (DP1) or Mode B"

    reference_dcp = decode_axes.get("kv_split", 1)
    decode_found = any(
        parallel_candidate.tp == decode_tp
        and parallel_candidate.dp == decode_dp
        and parallel_candidate.ep == reference_ep
        and parallel_candidate.kv_split == reference_dcp
        for slice_topology in core_matching_topologies
        for parallel_candidate in _candidate_decode_parallels(
            geom, slice_topology, chip=chip
        )
    )
    return (
        None
        if decode_found
        else f"this decode layout is not enumerated on a {needed_cores}-core slice"
    )


@dataclass(frozen=True, slots=True)
class _PrefillExplainSpec:
    """Grouped runtime inputs for prefill layout rejection diagnosis."""

    topologies: Sequence[SliceTopology]
    chip: ChipSpec | None = None
    spec_k: int = 0


def _explain_prefill_rejection(
    geom: ModelGeometry,
    prefill_axes: Mapping[str, int],
    decode_axes: Mapping[str, int],
    spec: _PrefillExplainSpec,
) -> str | None:
    """Return rejection reason if prefill layout is invalid or not enumerated."""
    prefill_tp = prefill_axes.get("tp", 1)
    prefill_pp = prefill_axes.get("pp", 1)
    prefill_pcp = prefill_axes.get("pcp", 1)
    prefill_ep = prefill_axes.get("ep", 1)
    if not _pcp_allowed(geom, prefill_pp, prefill_pcp, spec.spec_k):
        return "PCP > 1 is not permitted with PP > 1, qwen4_exp, or spec_k > 1"
    n_full_attn = count_full_attn_layers(geom)
    if not _prefill_pp_allowed(
        geom, prefill_pp, n_full_attn, for_disagg=True, spec_k=spec.spec_k
    ):
        return f"PP{prefill_pp} is not permitted for this model and scenario"
    prefill_found = any(
        parallel_candidate.tp == prefill_tp
        and parallel_candidate.pp == prefill_pp
        and parallel_candidate.pcp == prefill_pcp
        and parallel_candidate.ep == prefill_ep
        for slice_topology in spec.topologies
        for parallel_candidate in _candidate_prefill_parallels(
            geom,
            slice_topology,
            for_disagg=True,
            spec_k=spec.spec_k,
            chip=spec.chip,
        )
    )
    if not prefill_found:
        return "this prefill layout is not enumerated on any searched slice"

    prefill_parallel = Parallel(
        tp=prefill_tp,
        dp=prefill_axes.get("dp", 1),
        ep=prefill_ep,
        pp=prefill_pp,
        pcp=prefill_pcp,
        kv_split=prefill_axes.get("kv_split", 1),
    )
    decode_parallel = Parallel(
        tp=decode_axes.get("tp", 1),
        dp=decode_axes.get("dp", 1),
        ep=decode_axes.get("ep", 1),
        pp=decode_axes.get("pp", 1),
        pcp=decode_axes.get("pcp", 1),
        kv_split=decode_axes.get("kv_split", 1),
    )
    if not (
        _valid_raiden_gdn_pair(geom, prefill_parallel, decode_parallel)
        and _valid_raiden_kv_pair(geom, prefill_parallel, decode_parallel)
    ):
        return "prefill and decode degrees do not form a legal KV handoff pair"
    return None


class _ExplainOptions(TypedDict, total=False):
    """Optional chip and speculative decode overrides for rejection checks."""

    chip: ChipSpec | None
    spec_k: int


def explain_layout_rejection(
    geom: ModelGeometry,
    layout: Mapping[str, Mapping[str, int]],
    topologies: Sequence[SliceTopology],
    opts: SolveOptions | None = None,
    **explain_opts: Unpack[_ExplainOptions],
) -> str:
    """Name the first search rule that leaves a reference layout without candidates."""
    chip = explain_opts.get("chip")
    spec_k = explain_opts.get("spec_k", 0)
    options = opts or SolveOptions()
    decode_topologies = [
        topology
        for topology in topologies
        if options.decode_topology is None or topology.name == options.decode_topology
    ]
    prefill_topologies = [
        topology
        for topology in topologies
        if options.prefill_topology is None or topology.name == options.prefill_topology
    ]
    mode_reason = _explain_mode_rejection(layout, options.mode)
    if mode_reason is not None:
        return mode_reason

    if "prefill" in layout:
        prefill_axes = layout["prefill"]
        if prefill_axes.get("dp", 1) > 1 or prefill_axes.get("kv_split", 1) > 1:
            return "prefill pools are enumerated with DP1 and DCP1"

    decode_reason = _explain_decode_rejection(
        geom, layout["decode"], decode_topologies, chip=chip
    )
    if decode_reason is not None:
        return decode_reason

    if "prefill" in layout:
        prefill_reason = _explain_prefill_rejection(
            geom,
            layout["prefill"],
            layout["decode"],
            _PrefillExplainSpec(
                topologies=prefill_topologies,
                chip=chip,
                spec_k=spec_k,
            ),
        )
        if prefill_reason is not None:
            return prefill_reason

    return (
        "enumerated, but no searched point with this layout passed the "
        "memory fit, the chip budget, and the latency targets"
    )


def _build_pins(opts: SolveOptions) -> dict[str, object] | None:
    """Build canonical pin dictionary preserving any partial or full pins."""
    pins: dict[str, object] = {}
    if opts.prefill_topology is not None:
        pins["prefill_topology"] = opts.prefill_topology
    if opts.decode_topology is not None:
        pins["decode_topology"] = opts.decode_topology
    if opts.pd_ratio is not None:
        x_p, y_d = opts.pd_ratio
        if x_p < 1 or y_d < 1:
            msg = f"pd_ratio must have positive counts, got {opts.pd_ratio}"
            raise ValueError(msg)
        pins["pd_ratio"] = f"{x_p}:{y_d}"
    return pins or None


def _resolve_budget_max_chips(
    opts: SolveOptions,
    *,
    chip: ChipSpec,
) -> int | None:
    """Resolve effective max_chips when full disagg pins are set."""
    if (
        opts.pd_ratio is None
        or opts.prefill_topology is None
        or opts.decode_topology is None
    ):
        return opts.max_chips
    topo_map = (
        {t.name: t for t in opts.topologies}
        if opts.topologies is not None
        else topologies_for_chip(chip)
    )
    if opts.prefill_topology not in topo_map or opts.decode_topology not in topo_map:
        return opts.max_chips
    pinned_chips = (
        opts.pd_ratio[0] * topo_map[opts.prefill_topology].num_chips
        + opts.pd_ratio[1] * topo_map[opts.decode_topology].num_chips
    )
    if opts.max_chips is not None and opts.max_chips != pinned_chips:
        msg = (
            f"Pinned disagg chips ({pinned_chips}) conflicts with "
            f"max_chips={opts.max_chips}"
        )
        raise ValueError(msg)
    return pinned_chips


def _active_mode_min_ttft(
    mode: str,
    min_prefill_ttft_ms: float,
    min_mixed_ttft_ms: float,
) -> float:
    """Return the single-request prefill floor for the active solver mode."""
    if mode == "mixed":
        return min_mixed_ttft_ms
    if mode == "disagg":
        return min_prefill_ttft_ms
    return min(min_prefill_ttft_ms, min_mixed_ttft_ms)


def solve_serving_plan(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    options: SolveOptions | None = None,
) -> ServingPlan:
    """Search single-pool mixed and unconstrained xP/yD disagg plans."""
    opts = options or SolveOptions()
    if not (0.0 < opts.gpu_memory_utilization <= 1.0):
        msg = (
            f"gpu_memory_utilization must be in (0, 1], got "
            f"{opts.gpu_memory_utilization}"
        )
        raise ValueError(msg)
    if opts.collective_latency_s < 0.0:
        msg = f"collective_latency_s must be >= 0, got {opts.collective_latency_s}"
        raise ValueError(msg)
    pins = _build_pins(opts)
    resolved_max_chips = _resolve_budget_max_chips(opts, chip=chip)
    opts = replace(opts, max_chips=resolved_max_chips)
    topos = _resolve_solve_topologies(opts, chip=chip)
    hbm_limit_b = float(opts.gpu_memory_utilization) * float(chip.hbm_bytes)
    eff_al = max(1.0, scenario.accept_len)
    spec_k = (
        scenario.spec_tokens
        if scenario.spec_tokens is not None
        else (max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0)
    )
    ctx_full = scenario.isl + scenario.osl
    eff_isl = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks = max(1, math.ceil(eff_isl / max(1, scenario.prefill_chunk)))
    page_size = (
        scenario.page_size
        if scenario.page_size is not None
        else default_rpa_page_size(ctx_full, is_mla=geom.kv_lora_rank > 0)
    )

    eval_context = _SolveContext(
        geom=geom,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        mode=opts.mode,
        max_chips=opts.max_chips,
        max_instances=opts.max_instances,
        gpu_memory_utilization=opts.gpu_memory_utilization,
        hbm_limit_b=hbm_limit_b,
        hbm_limit_gb=hbm_limit_b / UNIT_GIGA,
        eff_al=eff_al,
        spec_k=spec_k,
        ctx_full=ctx_full,
        ctx_mid=scenario.isl + scenario.osl // 2,
        eff_isl=eff_isl,
        n_chunks=n_chunks,
        page_size=page_size,
        c_max=max(scenario.concurrencies),
        collective_latency_s=opts.collective_latency_s,
        prefill_topology=opts.prefill_topology,
        decode_topology=opts.decode_topology,
        pd_ratio=opts.pd_ratio,
    )

    qwen4_warnings = _qwen4_scenario_warnings(geom, ctx_full, spec_k)
    if qwen4_warnings:
        return ServingPlan(
            model_layers=geom.num_layers,
            chip=chip.name,
            scenario=scenario,
            mode=opts.mode,
            recommended=None,
            frontier=(),
            warnings=qwen4_warnings,
            candidates=(),
            gpu_memory_utilization=opts.gpu_memory_utilization,
            collective_latency_s=opts.collective_latency_s,
            spec_k=spec_k,
            model_id=geom.model_id,
            has_moe=geom.num_experts > 0,
            budget=opts.max_chips if opts.max_chips is not None else 0,
            budget_source=opts.budget_source,
            max_instances=opts.max_instances,
            pins=pins,
        )
    prefill_options, min_prefill_ttft_ms = _find_prefill_options(eval_context, topos)
    mixed_candidates, min_c1_tpot_ms, min_mixed_ttft_ms = _evaluate_mixed_candidates(
        eval_context,
        topos,
    )
    disagg_candidates = _evaluate_disaggregated_candidates(
        eval_context, topos, prefill_options
    )
    candidates = mixed_candidates + disagg_candidates

    warnings: list[str] = []
    if geom.max_position_embeddings > 0 and ctx_full > geom.max_position_embeddings:
        warnings.append(
            f"CONTEXT_EXCEEDS_MAX_POSITION_EMBEDDINGS: ctx_full={ctx_full} > "
            f"max_position_embeddings={geom.max_position_embeddings}"
        )
    min_req_ttft_ms = _active_mode_min_ttft(
        opts.mode, min_prefill_ttft_ms, min_mixed_ttft_ms
    )
    if (
        scenario.tpot_slo_ms is not None
        and math.isfinite(min_c1_tpot_ms)
        and scenario.tpot_slo_ms < min_c1_tpot_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: tpot_slo_ms={scenario.tpot_slo_ms:.2f} < "
            f"single-request floor {min_c1_tpot_ms:.2f} ms",
        )
    if (
        scenario.ttft_slo_ms is not None
        and math.isfinite(min_req_ttft_ms)
        and scenario.ttft_slo_ms < min_req_ttft_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: ttft_slo_ms={scenario.ttft_slo_ms:.2f} < "
            f"single-request prefill floor {min_req_ttft_ms:.2f} ms",
        )

    slo_candidates = [p for p in candidates if p.slo_met]
    pool_for_frontier = slo_candidates if slo_candidates else candidates
    if not candidates:
        warnings.append(NO_FEASIBLE_LAYOUT_WARNING)
    elif not slo_candidates:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")

    frontier = _pareto_filter(pool_for_frontier)
    slo_frontier = [p for p in frontier if p.slo_met]
    recommended = (
        max(slo_frontier, key=lambda p: p.token_throughput_per_chip)
        if slo_frontier
        else None
    )
    return ServingPlan(
        model_layers=geom.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=opts.mode,
        recommended=recommended,
        frontier=frontier,
        warnings=tuple(warnings),
        candidates=tuple(pool_for_frontier),
        gpu_memory_utilization=opts.gpu_memory_utilization,
        collective_latency_s=opts.collective_latency_s,
        spec_k=spec_k,
        model_id=geom.model_id,
        has_moe=geom.num_experts > 0,
        search=SearchStats(
            enumerated=eval_context.enumerated,
            core_grid=_count_core_grid(eval_context, topos),
            hbm_fit=eval_context.hbm_fit,
            slo_met=len(slo_candidates),
            frontier=len(frontier),
        ),
        budget=opts.max_chips if opts.max_chips is not None else 0,
        budget_source=opts.budget_source,
        max_instances=opts.max_instances,
        pins=pins,
    )
