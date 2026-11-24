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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, Literal, TypedDict, Unpack

from .chips import TOPOLOGIES, ChipSpec, SliceTopology, topologies_for_chip
from .collectives import submesh_for_group
from .constants import (
    ACTIVATION_BYTES_BF16,
    CAUSAL_MIDPOINT_HALF,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_HBM_RESERVE_GB,
    DEFAULT_MAX_INSTANCES,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    FP4_BYTES,
    FP8_BYTES,
    LINEAR_ATTENTION_LAYER_TOKENS,
    MIN_DISAGG_INSTANCES,
    NO_FEASIBLE_LAYOUT_WARNING,
    SLO_VERDICT_ESTIMATE_MEETS,
    SLO_VERDICT_RANK,
    SLO_VERDICT_RULED_OUT,
    SLO_VERDICT_UNDECIDED,
    UNDECIDED_SLO_WARNING,
    UNIT_GIBI,
    UNIT_GIGA,
    UNIT_MILLIS_PER_SECOND,
    ZERO_THINK_TIME_WARNING,
    ParallelRules,
    SloVerdict,
    canonicalize_kv_dtype,
    disagg_prefill_spec_k,
)
from .geometry import resolve_geom_profile
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
from .occupancy import ExpertActivationModel
from .partition import (
    PpStageOptions,
    count_full_attn_layers,
    pp_stage_breakdowns,
    pp_stage_kv_parts,
    scale_pp_stage_floor,
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
from .step import (
    FloorOptions,
    SpecShape,
    StepFloor,
    UnpaddedFloorOptions,
    spec_cycle_floor,
    step_floor,
    unpadded_strict_resource_times,
)

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
    "agentic-64k": TrafficScenario(
        name="example:agentic-64k",
        isl=65536,
        osl=1024,
        concurrencies=EXAMPLE_CONCURRENCIES,
        prefix_hit_rate=0.88,
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
        kv_offload_gb=(
            base.kv_offload_gb if ovr.kv_offload_gb is None else ovr.kv_offload_gb
        ),
    )


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Search mode, slice budget, and HBM headroom options for solve_serving_plan."""

    mode: Literal["auto", "mixed", "disagg"] = "auto"
    max_chips: int | None = None
    max_instances: int | None = None
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    topologies: tuple[SliceTopology, ...] | None = None
    collective_latency_s: float = 0.0
    hbm_reserve_gb: float = DEFAULT_HBM_RESERVE_GB
    prefill_topology: str | None = None
    decode_topology: str | None = None
    pd_ratio: tuple[int, int] | None = None
    budget_source: str = "flag"
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )

    def __post_init__(self) -> None:
        """Validate non-negative HBM workspace reserve constraint."""
        if self.hbm_reserve_gb < 0.0:
            msg = f"hbm_reserve_gb must be >= 0, got {self.hbm_reserve_gb}"
            raise ValueError(msg)


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


def _valid_glm_mla_raiden_pair(
    prefill_parallel: Parallel,
    decode_parallel: Parallel,
) -> bool:
    """Enforce exact GLM-MLA Raiden P/D pairing rules matching runtime tpu_connector."""
    if prefill_parallel.dp != 1 or prefill_parallel.pcp != 1:
        return False
    pp_p = prefill_parallel.pp
    tp_p = prefill_parallel.tp
    dcp_p = prefill_parallel.kv_split
    tp_d = decode_parallel.tp
    dp_d = decode_parallel.dp
    dcp_d = decode_parallel.kv_split
    tp_dcp_match = (
        pp_p == 1 and tp_p == dcp_p > 1 and dp_d == 1 and tp_d == dcp_d == tp_p
    )
    pp_to_dcp = (
        pp_p > 1
        and tp_p == 1
        and dcp_p == 1
        and dp_d == 1
        and tp_d > 1
        and tp_d == dcp_d
    )
    tp_to_dp = pp_p == 1 and dcp_p == 1 and tp_d == 1 and dcp_d == 1
    return tp_dcp_match or pp_to_dcp or tp_to_dp


def _valid_raiden_kv_pair(
    geom: ModelGeometry,
    prefill_parallel: Parallel,
    decode_parallel: Parallel,
) -> bool:
    """Check runtime disaggregated KV cache shard compatibility between P and D."""
    if geom.num_kv_heads <= 0:
        return True
    rule = resolve_geom_profile(geom).parallel.raiden_pair_rule
    if rule == "glm_mla":
        return _valid_glm_mla_raiden_pair(prefill_parallel, decode_parallel)
    deg_p = max(1, prefill_parallel.tp * prefill_parallel.pcp)
    deg_d = max(1, decode_parallel.tp * decode_parallel.pcp)
    divisible = deg_p % deg_d == 0 or deg_d % deg_p == 0
    if rule == "qwen_gdn":
        return prefill_parallel.kv_split == 1 and divisible
    dcp_match = prefill_parallel.kv_split in {1, decode_parallel.kv_split}
    return dcp_match and divisible


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
    rules = resolve_geom_profile(geom).parallel
    disallow_dcp = rules.max_dcp is not None and rules.max_dcp <= 1
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
                if geom.num_kv_heads > 0
                and geom.kv_lora_rank == 0
                and rules.dcp_span != "tp_group"
                else tp
            )
            raw_splits = _divisors(kv_group) if kv_group > 1 else (1,)
            kv_splits = (
                tuple(k for k in raw_splits if k <= rules.max_dcp)
                if rules.max_dcp is not None
                else raw_splits
            )
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
    rules = resolve_geom_profile(geom).parallel
    if geom.num_layers < pp or (rules.max_pp is not None and pp > rules.max_pp):
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
    rules = resolve_geom_profile(geom).parallel
    if rules.max_pcp is not None and pcp > rules.max_pcp:
        return False
    if pp > 1 and not rules.pcp_allows_pp:
        return False
    return spec_k <= 1


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


def _candidate_prefill_kv_splits(
    geom: ModelGeometry,
    axes: tuple[int, int, int],
    rules: ParallelRules,
    *,
    for_disagg: bool,
) -> tuple[int, ...]:
    """Return legal prefill DCP degrees governed by ``rules.prefill_dcp_mode``."""
    tp, pp, pcp = axes
    has_kv_cache = geom.num_kv_heads > 0 or geom.kv_lora_rank > 0
    if (
        not for_disagg
        or not has_kv_cache
        or rules.prefill_dcp_mode == "none"
        or pp > 1
        or pcp > 1
        or tp <= 1
    ):
        return (1,)
    if rules.prefill_dcp_mode == "tp_match":
        if rules.max_dcp is not None and tp > rules.max_dcp:
            return (1,)
        return (1, tp)
    kv_group = (
        max(1, tp // geom.num_kv_heads)
        if geom.num_kv_heads > 0
        and geom.kv_lora_rank == 0
        and rules.dcp_span != "tp_group"
        else tp
    )
    raw_splits = _divisors(kv_group) if kv_group > 1 else (1,)
    if rules.max_dcp is None:
        return raw_splits
    return tuple(k for k in raw_splits if k <= rules.max_dcp)


def _append_prefill_candidates(
    out: list[Parallel],
    topo: SliceTopology,
    axes: tuple[int, int, int],
    kv_splits: tuple[int, ...],
    moe_modes: Sequence[tuple[Literal["tp_slice", "ep"], int]],
) -> None:
    """Expand prefill candidates across legal DCP and MoE dispatch combinations."""
    tp, pp, pcp = axes
    mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
    for moe_mode, ep_val in moe_modes:
        dispatches: tuple[Literal["allreduce", "ag_ep"], ...] = (
            ("ag_ep",)
            if (moe_mode == "ep" and tp == 1)
            else (("allreduce", "ag_ep") if moe_mode == "ep" else ("allreduce",))
        )
        for kvs in kv_splits:
            out.extend(
                Parallel(
                    tp=tp,
                    kv_split=kvs,
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


def _candidate_prefill_parallels(
    geom: ModelGeometry,
    topo: SliceTopology,
    *,
    for_disagg: bool = False,
    spec_k: int = 0,
    chip: ChipSpec | None = None,
) -> list[Parallel]:
    """Enumerate legal prefill parallelism configurations for a given topology."""
    rules = resolve_geom_profile(geom).parallel
    spec_disallowed = (
        rules.max_spec_tokens is not None and spec_k > rules.max_spec_tokens
    )
    if not _valid_moe_block_quant_hidden(geom) or spec_disallowed:
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
            kv_splits = _candidate_prefill_kv_splits(
                geom, (tp, pp, pcp), rules, for_disagg=for_disagg
            )
            moe_modes = _candidate_prefill_moe_modes(geom, tp, pcp)
            _append_prefill_candidates(out, topo, (tp, pp, pcp), kv_splits, moe_modes)
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
    ordered = sorted(
        points,
        key=lambda x: (
            -SLO_VERDICT_RANK[x.slo_verdict],
            x.concurrency,
            -x.output_throughput_per_chip,
            -x.est_interactivity_tok_s,
        ),
    )
    frontier: list[ParetoPoint] = []
    for p in ordered:
        p_rank = SLO_VERDICT_RANK[p.slo_verdict]
        dominated = any(
            SLO_VERDICT_RANK[q.slo_verdict] >= p_rank
            and q.concurrency <= p.concurrency
            and q.output_throughput_per_chip >= p.output_throughput_per_chip
            and q.est_interactivity_tok_s >= p.est_interactivity_tok_s
            and (
                SLO_VERDICT_RANK[q.slo_verdict] > p_rank
                or q.concurrency < p.concurrency
                or q.output_throughput_per_chip > p.output_throughput_per_chip
                or q.est_interactivity_tok_s > p.est_interactivity_tok_s
            )
            for q in frontier
        )
        if not dominated:
            frontier.append(p)
    frontier.sort(
        key=lambda x: (
            x.concurrency,
            -x.output_throughput_per_chip,
            -x.est_interactivity_tok_s,
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
    hbm_reserve_gb: float = DEFAULT_HBM_RESERVE_GB
    prefill_topology: str | None = None
    decode_topology: str | None = None
    pd_ratio: tuple[int, int] | None = None
    activation_model: ExpertActivationModel = field(
        default_factory=ExpertActivationModel
    )
    enumerated: int = 0
    hbm_fit: int = 0
    decode_floor_cache: dict[tuple[str, Parallel, int], StepFloor] = field(
        default_factory=dict
    )
    kv_fit_cache: dict[tuple[Parallel, int, int, float], tuple[float, float]] = field(
        default_factory=dict
    )
    clamp_batch_cache: dict[tuple[Parallel, int], tuple[int, HbmBreakdown] | None] = (
        field(default_factory=dict)
    )
    xfer_cache: dict[tuple[str, str, int], float] = field(default_factory=dict)


@dataclass(frozen=True)
class _PrefillOption:
    """Feasible prefill configuration for disaggregated serving."""

    topo: SliceTopology
    par: Parallel
    ttft_ms: float
    rate_p: float
    hbm_gb: float
    hbm_bd: HbmBreakdown
    pp_part: tuple[int, ...] | None = None
    bound_ttft_ms: float = 0.0
    envelope_ttft_ms: float = 0.0
    w_base_b: float = 0.0
    b_kv_phys: float = 0.0
    offload_b_per_core: float = 0.0
    logit_peak_b: float = 0.0
    fixed_kv_per_seq_b: float = 0.0
    pp_stage_partition: tuple[int, ...] | None = None


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
    h_eff: float = 0.0


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
    pp_layer_partition: tuple[int, ...] | None = None
    kv_offload_gb: float = 0.0


type _StepDurationKind = Literal["serial_estimate", "hard_floor", "envelope_floor"]


@dataclass(frozen=True, slots=True)
class _MixedLoopInput:
    """Inputs for solving the mixed prefill-decode Little's-law fixed point."""

    b_res: int
    n_chunks_c: int
    chunk_tok_c: int
    ttft_service_s: float
    pcie_read_s: float = 0.0
    eff_isl_c: int = 0
    duration_kind: _StepDurationKind = "serial_estimate"
    ttft_upper_s: float | None = None


def _step_duration_s(
    step_floor_val: StepFloor, duration_kind: _StepDurationKind
) -> float:
    """Select the modeled, hard-floor, or envelope-floor step duration in seconds."""
    if duration_kind == "hard_floor":
        return step_floor_val.hard_floor_s
    if duration_kind == "envelope_floor":
        return step_floor_val.envelope_floor_s
    return step_floor_val.serial_estimate_s


@dataclass(frozen=True, slots=True)
class _MixedLoopOutput:
    """Solved fixed-point mixed step floors, latencies, rate, and residency."""

    sf_dec: StepFloor
    sf_mix: StepFloor
    p_mix: float
    est_ttft_ms: float
    est_tpot_ms: float
    x_rate: float
    decode_seqs: float = 0.0


@dataclass(frozen=True, slots=True)
class _DisaggMetricsOutput:
    """Solved disaggregated latency, throughput, residency, and floor times."""

    est_ttft_ms: float
    est_tpot_ms: float
    interactivity: float
    out_tput: float
    tot_tput: float
    sf_d: StepFloor
    bound_sf_d: StepFloor
    min_decode_seqs: float
    floor_throughput_per_chip: float
    envelope_step_floor: StepFloor
    envelope_decode_seqs: float
    x_cap_pool: float = 0.0


@dataclass(frozen=True, slots=True)
class _EffectivePrefill:
    """Cached prefill latency, throughput rate, hard bound, and envelope bound."""

    ttft_ms: float
    rate_p: float
    bound_ttft_ms: float
    envelope_ttft_ms: float


def _floor_times(step_floor_val: StepFloor) -> FloorTimes:
    """Decode-step resource times (ms) behind a point's binding unit."""
    hbm_span_s = (
        step_floor_val.hbm_bound_s
        if step_floor_val.hbm_bound_s > 0.0
        else step_floor_val.hbm_s
    )
    matmul_span_s = (
        step_floor_val.matmul_bound_s
        if step_floor_val.matmul_bound_s > 0.0
        else step_floor_val.matmul_s
    )
    vector_span_s = (
        step_floor_val.vector_bound_s
        if step_floor_val.vector_bound_s > 0.0
        else step_floor_val.vector_s
    )
    coll_span_s = (
        step_floor_val.coll_bound_s
        if step_floor_val.coll_bound_s > 0.0
        else step_floor_val.chain_s + step_floor_val.latency_assumed_s
    )
    return FloorTimes(
        hbm_ms=hbm_span_s * UNIT_MILLIS_PER_SECOND,
        matmul_ms=matmul_span_s * UNIT_MILLIS_PER_SECOND,
        ici_ms=step_floor_val.ici_s * UNIT_MILLIS_PER_SECOND,
        chain_ms=step_floor_val.chain_s * UNIT_MILLIS_PER_SECOND,
        hop_latency_ms=step_floor_val.latency_assumed_s * UNIT_MILLIS_PER_SECOND,
        hard_floor_ms=step_floor_val.hard_floor_s * UNIT_MILLIS_PER_SECOND,
        binding=step_floor_val.binding,
        hbm_bound_ms=hbm_span_s * UNIT_MILLIS_PER_SECOND,
        matmul_bound_ms=matmul_span_s * UNIT_MILLIS_PER_SECOND,
        vector_bound_ms=vector_span_s * UNIT_MILLIS_PER_SECOND,
        coll_bound_ms=coll_span_s * UNIT_MILLIS_PER_SECOND,
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
        pp_layer_partition=spec.pp_layer_partition,
        kv_offload_gb=spec.kv_offload_gb,
    )


def _peak_prefill_logit_bytes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    chunk_rows: int,
    *,
    sampled_rows: int = 1,
) -> float:
    """Return estimated peak HBM buffer bytes for prefill workspace."""
    if (chunk_rows <= 0 and sampled_rows <= 0) or geom.vocab <= 0:
        return 0.0
    tp = max(1, parallel.tp)
    rows_local = math.ceil(max(0, chunk_rows) / max(1, parallel.pcp))
    n_samp = max(1, sampled_rows) if chunk_rows > 0 else max(0, sampled_rows)
    act_b = ACTIVATION_BYTES_BF16 if dtypes.act == "a16" else FP8_BYTES
    logit_b = max(act_b, dtypes.lm_head)
    hc = float(geom.hc_mult)
    return (
        float(n_samp) * (float(geom.vocab) / float(tp)) * logit_b
        + float(rows_local) * float(geom.hidden) * act_b * hc
    )


def _within_slo_bounds(
    scen: TrafficScenario,
    ttft_ms: float,
    tpot_ms: float,
) -> bool:
    """Return True when both TTFT and TPOT stay within any configured SLOs."""
    ttft_ok = scen.ttft_slo_ms is None or ttft_ms <= scen.ttft_slo_ms
    tpot_ok = scen.tpot_slo_ms is None or tpot_ms <= scen.tpot_slo_ms
    return ttft_ok and tpot_ok


_TARSKI_MAX_ITERS: Final[int] = 12
_FIXED_POINT_ITERS: Final[int] = 3
_HIT_RATE_KEY_DECIMALS: Final[int] = 6
_TARSKI_TOL: Final[float] = 1e-6
_RATE_FLOOR_EPS: Final[float] = 1e-9
_CAP_DENOM_EPS: Final[float] = 1e-12
_UNPADDED_BUCKET_COUNT: Final[int] = 4


def _resolve_slo_verdict(
    scen: TrafficScenario,
    est_ttft_ms: float,
    est_tpot_ms: float,
    bound_ttft_ms: float,
    bound_tpot_ms: float,
) -> SloVerdict:
    """Classify lower-bound vs modeled-estimate SLO feasibility verdict."""
    if not _within_slo_bounds(scen, bound_ttft_ms, bound_tpot_ms):
        return SLO_VERDICT_RULED_OUT
    return (
        SLO_VERDICT_ESTIMATE_MEETS
        if _within_slo_bounds(scen, est_ttft_ms, est_tpot_ms)
        else SLO_VERDICT_UNDECIDED
    )


def _apply_capacity_ceiling_verdict(
    scen: TrafficScenario,
    verdict: SloVerdict,
    *,
    concurrency: int = 0,
    pool_capacity_req_s: float | None = None,
) -> SloVerdict:
    """Rule out dual-SLO workloads exceeding pool capacity ceiling."""
    if (
        verdict == SLO_VERDICT_RULED_OUT
        or pool_capacity_req_s is None
        or concurrency <= 0
        or scen.ttft_slo_ms is None
        or scen.tpot_slo_ms is None
    ):
        return verdict
    x_min = float(concurrency) / max(
        _RATE_FLOOR_EPS,
        scen.ttft_slo_ms / UNIT_MILLIS_PER_SECOND
        + float(scen.osl) * (scen.tpot_slo_ms / UNIT_MILLIS_PER_SECOND)
        + scen.think_time_s,
    )
    return SLO_VERDICT_RULED_OUT if pool_capacity_req_s < x_min else verdict


class _L1BoundOpts(TypedDict, total=False):
    """Keyword options for Layer-1 physical bound entry points."""

    prefill_par: Parallel | None
    prefill_topo: SliceTopology | None
    decode_par: Parallel | None
    decode_topo: SliceTopology | None
    b_cap_single: int
    concurrency: int
    x_p: int
    y_d: int
    spec_k: int | None
    activation_model: ExpertActivationModel | None
    total_chips: int | None
    pp_part: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class _BoundDecodeInput:
    """Pure Layer-1 parameters for evaluating decode step/spec floors."""

    geom: ModelGeometry
    dtypes: Dtypes
    chip: ChipSpec
    scenario: TrafficScenario
    decode_par: Parallel
    decode_topo: SliceTopology
    spec_k: int
    activation_model: ExpertActivationModel
    ctx_mid: int
    eff_al: float


def _resolve_spec_k(
    geom: ModelGeometry,
    scenario: TrafficScenario,
    spec_k: int | None = None,
) -> int:
    """Resolve effective speculative draft step count without touching estimates."""
    if spec_k is not None:
        return spec_k
    if scenario.spec_tokens is not None:
        return scenario.spec_tokens
    eff_al = max(1.0, scenario.accept_len)
    if eff_al > 1.0:
        return max(geom.mtp_layers, math.ceil(eff_al) - 1)
    return 0


def _eval_bound_decode_floor(bound_in: _BoundDecodeInput, batch: int) -> StepFloor:
    """Evaluate a pure Layer-1 decode step floor without latency calibration."""
    opts = FloorOptions(
        dt=bound_in.dtypes,
        topology=bound_in.decode_topo,
        activation_model=bound_in.activation_model,
    )
    if bound_in.spec_k > 0:
        return spec_cycle_floor(
            bound_in.geom,
            bound_in.chip,
            bound_in.decode_par,
            SpecShape(batch=batch, context=bound_in.ctx_mid, k=bound_in.spec_k),
            opts,
        )
    return step_floor(
        bound_in.geom,
        bound_in.chip,
        bound_in.decode_par,
        StepShape(batch=batch, context=bound_in.ctx_mid),
        opts,
    )


def _disagg_kv_xfer_s(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    topos: tuple[SliceTopology, SliceTopology],
    uncached_tokens: int,
) -> float:
    """Return transfer time in seconds to migrate KV cache over DCN between pools."""
    if chip.dcn_bw <= 0.0 or not math.isfinite(chip.dcn_bw):
        return 0.0
    bd_one = hbm_breakdown_per_core_bytes(
        geom,
        dtypes,
        Parallel(tp=1, pp=1, ranks_per_chip=chip.cores_per_chip),
        SeqKvLayout(1, max(0, uncached_tokens), physical_kv_layout=False),
        chip=chip,
    )
    wire_b = bd_one.kv + bd_one.state
    topo_p, topo_d = topos
    denom = float(max(1, min(topo_p.num_chips, topo_d.num_chips))) * chip.dcn_bw
    return wire_b / denom


def _pcie_read_s(
    eval_context: _SolveContext,
    b_kv_phys: float,
    h_eff: float,
    h_hbm_only: float,
) -> float:
    """Compute host PCIe offload read time in seconds for spilled KV prefix tokens."""
    offload_read_b = (
        max(0.0, h_eff - h_hbm_only) * float(eval_context.scenario.isl) * b_kv_phys
    )
    pcie_core_bw = eval_context.chip.pcie_bw / float(
        max(1, eval_context.chip.cores_per_chip)
    )
    if offload_read_b > 0.0 and pcie_core_bw > 0.0 and math.isfinite(pcie_core_bw):
        return offload_read_b / pcie_core_bw
    return 0.0


def _eval_bound_prefill_floor(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    opts: _L1BoundOpts,
) -> StepFloor:
    """Evaluate a pure Layer-1 prefill step floor at scenario prefix_hit_rate."""
    pref_par = opts.get("prefill_par")
    pref_topo = opts.get("prefill_topo")
    if pref_par is None or pref_topo is None:
        msg = "prefill_par and prefill_topo are required for TTFT bounds"
        raise ValueError(msg)
    act_model = opts.get("activation_model") or ExpertActivationModel()
    shape = StepShape(
        batch=0,
        rows_per_seq=0,
        context=0,
        state_writes=0,
        prefill_tokens=scenario.isl,
        prefix_hit_rate=scenario.prefix_hit_rate,
    )
    floor_opts = FloorOptions(
        dt=dtypes,
        topology=pref_topo,
        activation_model=act_model,
    )
    pp_part = opts.get("pp_part")
    if pp_part:
        return scale_pp_stage_floor(geom, chip, pref_par, (shape, floor_opts), pp_part)
    return step_floor(geom, chip, pref_par, shape, floor_opts)


def bound_ttft_ms(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return physical TTFT lower bound (ms) strictly at scenario prefix_hit_rate."""
    sf = _eval_bound_prefill_floor(geom, dtypes, chip, scenario, opts)
    return sf.hard_floor_s * UNIT_MILLIS_PER_SECOND


def envelope_ttft_ms(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return activation-envelope TTFT bound (ms) at scenario prefix_hit_rate."""
    sf = _eval_bound_prefill_floor(geom, dtypes, chip, scenario, opts)
    return sf.envelope_floor_s * UNIT_MILLIS_PER_SECOND


def _eval_unpadded_prefill_res(
    bound_in: _BoundDecodeInput,
    pref_par: Parallel,
    pref_topo: SliceTopology,
) -> tuple[float, float, float, float]:
    """Return unpadded prefill resource times for X_cap_pool work bounds."""
    scenario = bound_in.scenario
    eff_isl_floor = max(
        0,
        math.floor((1.0 - scenario.prefix_hit_rate) * float(scenario.isl)),
    )
    if eff_isl_floor <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return unpadded_strict_resource_times(
        bound_in.geom,
        bound_in.chip,
        pref_par,
        StepShape(
            batch=0,
            rows_per_seq=1,
            context=0,
            prefill_tokens=scenario.isl,
            prefix_hit_rate=scenario.prefix_hit_rate,
        ),
        UnpaddedFloorOptions(
            dt=bound_in.dtypes,
            topology=pref_topo,
            zero_weights=True,
        ),
    )


def _eval_unpadded_decode_res(
    bound_in: _BoundDecodeInput,
    b_cap_single: int,
) -> tuple[float, float, float, float]:
    """Return unpadded decode resource times at b_cap_single for X_cap_pool."""
    return unpadded_strict_resource_times(
        bound_in.geom,
        bound_in.chip,
        bound_in.decode_par,
        StepShape(
            batch=b_cap_single,
            rows_per_seq=bound_in.spec_k + 1,
            context=bound_in.ctx_mid,
            state_writes=1,
        ),
        UnpaddedFloorOptions(
            dt=bound_in.dtypes,
            topology=bound_in.decode_topo,
            spec_k=bound_in.spec_k,
            zero_weights=False,
        ),
    )


def _mixed_capacity_work_bound(
    bound_in: _BoundDecodeInput,
    b_cap_single: int,
) -> float:
    """Return isolated mixed-pool X_cap_pool ceiling (req/s) via unpadded times."""
    dec_steps_per_req = max(1.0, float(bound_in.scenario.osl) / bound_in.eff_al)
    pre_res = _eval_unpadded_prefill_res(
        bound_in, bound_in.decode_par, bound_in.decode_topo
    )
    dec_res = _eval_unpadded_decode_res(bound_in, b_cap_single)
    dec_scale = dec_steps_per_req / float(max(1, b_cap_single))
    req_time_s = max(
        pre_res[i] + dec_scale * dec_res[i] for i in range(_UNPADDED_BUCKET_COUNT)
    )
    return float(max(1, bound_in.decode_par.dp)) / max(_CAP_DENOM_EPS, req_time_s)


def _disagg_capacity_work_bound(
    bound_in: _BoundDecodeInput,
    pref_par: Parallel,
    pref_topo: SliceTopology,
    ratio_cap: tuple[int, int, int],
) -> float:
    """Return isolated disagg-pool X_cap_pool ceiling (req/s) across P and D pools."""
    x_p, y_d, b_cap_single = ratio_cap
    dec_steps_per_req = max(1.0, float(bound_in.scenario.osl) / bound_in.eff_al)
    pre_res = _eval_unpadded_prefill_res(bound_in, pref_par, pref_topo)
    x_cap_pref = (
        float(x_p) / max(_CAP_DENOM_EPS, *pre_res) if any(pre_res) else math.inf
    )
    dec_res = _eval_unpadded_decode_res(bound_in, b_cap_single)
    dec_scale = dec_steps_per_req / float(max(1, b_cap_single))
    x_cap_dec = float(max(1, y_d * bound_in.decode_par.dp)) / max(
        _CAP_DENOM_EPS, *(dec_scale * val for val in dec_res)
    )
    chip = bound_in.chip
    eff_isl_floor = max(
        0,
        math.floor(
            (1.0 - bound_in.scenario.prefix_hit_rate) * float(bound_in.scenario.isl)
        ),
    )
    if eff_isl_floor > 0 and math.isfinite(chip.dcn_bw) and chip.dcn_bw > 0.0:
        xfer_one_s = _disagg_kv_xfer_s(
            bound_in.geom,
            bound_in.dtypes,
            chip,
            (pref_topo, bound_in.decode_topo),
            eff_isl_floor,
        )
        if xfer_one_s > 0.0:
            min_one = float(min(pref_topo.num_chips, bound_in.decode_topo.num_chips))
            min_pool = float(
                min(x_p * pref_topo.num_chips, y_d * bound_in.decode_topo.num_chips)
            )
            x_cap_dcn = (min_pool / max(1.0, min_one)) / xfer_one_s
        else:
            x_cap_dcn = math.inf
    else:
        x_cap_dcn = math.inf
    return min(x_cap_pref, x_cap_dec, x_cap_dcn)


def _context_to_bound_input(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
) -> _BoundDecodeInput:
    """Extract pure Layer-1 bound input from a solver evaluation context."""
    return _BoundDecodeInput(
        geom=eval_context.geom,
        dtypes=eval_context.dtypes,
        chip=eval_context.chip,
        scenario=eval_context.scenario,
        decode_par=par,
        decode_topo=topo,
        spec_k=eval_context.spec_k,
        activation_model=eval_context.activation_model,
        ctx_mid=eval_context.ctx_mid,
        eff_al=eval_context.eff_al,
    )


def _compute_mixed_capacity_bound(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    b_cap_single: int,
) -> float:
    """Isolated unpadded work-conservation ceiling X_cap_pool (req/s) for mixed."""
    return _mixed_capacity_work_bound(
        _context_to_bound_input(eval_context, topo, par), b_cap_single
    )


def _compute_disagg_capacity_bound(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    b_cap_single: int,
) -> float:
    """Isolated unpadded work-conservation ceiling X_cap_pool (req/s) for disagg."""
    return _disagg_capacity_work_bound(
        _context_to_bound_input(eval_context, dec.topo, dec.par),
        pref.par,
        pref.topo,
        (target.x_p, target.y_d, b_cap_single),
    )


def _run_isolated_tarski_loop(
    bound_in: _BoundDecodeInput,
    concurrency: int,
    b_cap_single: int,
    idp_val: int,
    x_cap_eff: float,
) -> tuple[float, StepFloor]:
    """Run bottom-up Tarski pure-decode hard_floor residency fixed-point loop."""
    scenario = bound_in.scenario
    ttft_slo_ms = scenario.ttft_slo_ms
    if ttft_slo_ms is None:
        return (1.0, _eval_bound_decode_floor(bound_in, 1))
    ttft_upper_s = ttft_slo_ms / UNIT_MILLIS_PER_SECOND
    osl = float(scenario.osl)
    think = scenario.think_time_s
    max_cap_seqs = float(min(concurrency, b_cap_single * idp_val))
    b_dec = 1.0
    for _ in range(_TARSKI_MAX_ITERS):
        b_rep = max(1, min(b_cap_single, math.floor(b_dec / float(idp_val))))
        sf_d = _eval_bound_decode_floor(bound_in, b_rep)
        t_dec_s = osl * (sf_d.hard_floor_s / bound_in.eff_al)
        x_rate = min(
            x_cap_eff,
            float(concurrency) / max(_RATE_FLOOR_EPS, ttft_upper_s + t_dec_s + think),
        )
        b_next = max(1.0, min(max_cap_seqs, x_rate * t_dec_s))
        if abs(b_next - b_dec) <= _TARSKI_TOL:
            b_dec = b_next
            break
        b_dec = b_next
    floor_b_replica = max(1, min(b_cap_single, math.floor(b_dec / float(idp_val))))
    return (b_dec, _eval_bound_decode_floor(bound_in, floor_b_replica))


def _build_l1_decode_input(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    opts: _L1BoundOpts,
) -> tuple[_BoundDecodeInput, int]:
    """Validate and construct Layer-1 decode input + effective b_cap_single."""
    decode_par = opts.get("decode_par")
    decode_topo = opts.get("decode_topo")
    if decode_par is None or decode_topo is None:
        msg = "decode_par and decode_topo are required for Layer-1 decode bounds"
        raise ValueError(msg)
    b_cap_single = max(1, opts.get("b_cap_single", 1))
    spec_k = _resolve_spec_k(geom, scenario, opts.get("spec_k"))
    act_model = opts.get("activation_model") or ExpertActivationModel()
    eff_al = max(1.0, scenario.accept_len)
    ctx_mid = scenario.isl + scenario.osl // 2
    bound_in = _BoundDecodeInput(
        geom=geom,
        dtypes=dtypes,
        chip=chip,
        scenario=scenario,
        decode_par=decode_par,
        decode_topo=decode_topo,
        spec_k=spec_k,
        activation_model=act_model,
        ctx_mid=ctx_mid,
        eff_al=eff_al,
    )
    return (bound_in, b_cap_single)


def _resolve_l1_x_cap(
    bound_in: _BoundDecodeInput,
    b_cap_single: int,
    opts: _L1BoundOpts,
) -> tuple[float, Parallel, SliceTopology, int, int]:
    """Return (x_cap_pool, ttft_par, ttft_topo, x_p, y_d) for L1 bounds."""
    pref_par = opts.get("prefill_par")
    pref_topo = opts.get("prefill_topo")
    if pref_par is not None and pref_topo is not None:
        x_p = max(1, opts.get("x_p", 1))
        y_d = max(1, opts.get("y_d", 1))
        x_cap_val = _disagg_capacity_work_bound(
            bound_in, pref_par, pref_topo, (x_p, y_d, b_cap_single)
        )
        return (x_cap_val, pref_par, pref_topo, x_p, y_d)
    return (
        _mixed_capacity_work_bound(bound_in, b_cap_single),
        bound_in.decode_par,
        bound_in.decode_topo,
        1,
        1,
    )


def x_cap_pool(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return isolated unpadded Layer-1 pool request capacity ceiling (req/s)."""
    bound_in, b_cap_single = _build_l1_decode_input(geom, dtypes, chip, scenario, opts)
    return _resolve_l1_x_cap(bound_in, b_cap_single, opts)[0]


X_cap_pool = x_cap_pool


def _resolve_l1_tarski_floor(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    opts: _L1BoundOpts,
) -> tuple[_BoundDecodeInput, StepFloor]:
    """Compute Layer-1 Tarski decode step floor for mixed or disagg bounds."""
    bound_in, b_cap_single = _build_l1_decode_input(geom, dtypes, chip, scenario, opts)
    conc = max(1, opts.get("concurrency", 1))
    x_cap_val, ttft_par, ttft_topo, x_p, y_d = _resolve_l1_x_cap(
        bound_in, b_cap_single, opts
    )
    if opts.get("prefill_par") is not None and opts.get("prefill_topo") is not None:
        b_ttft_ms = bound_ttft_ms(
            geom,
            dtypes,
            chip,
            scenario,
            prefill_par=ttft_par,
            prefill_topo=ttft_topo,
            activation_model=bound_in.activation_model,
            pp_part=opts.get("pp_part"),
        )
        rate_p_floor = float(x_p) / max(
            _RATE_FLOOR_EPS, b_ttft_ms / UNIT_MILLIS_PER_SECOND
        )
        x_cap_val = min(x_cap_val, rate_p_floor)
    _, sf = _run_isolated_tarski_loop(
        bound_in, conc, b_cap_single, max(1, y_d * bound_in.decode_par.dp), x_cap_val
    )
    return (bound_in, sf)


def bound_tpot_ms(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return physical Layer-1 TPOT lower bound (ms) under isolated Tarski residency."""
    bound_in, sf = _resolve_l1_tarski_floor(geom, dtypes, chip, scenario, opts)
    return (sf.hard_floor_s * UNIT_MILLIS_PER_SECOND) / bound_in.eff_al


def envelope_tpot_ms(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return activation-envelope TPOT bound (ms) at isolated Tarski residency."""
    bound_in, sf = _resolve_l1_tarski_floor(geom, dtypes, chip, scenario, opts)
    return (sf.envelope_floor_s * UNIT_MILLIS_PER_SECOND) / bound_in.eff_al


def x_bound(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return isolated Layer-1 request rate upper bound X_bound (req/s)."""
    bound_in, b_cap_single = _build_l1_decode_input(geom, dtypes, chip, scenario, opts)
    conc = max(1, opts.get("concurrency", 1))
    x_cap_val, ttft_par, ttft_topo, _, _ = _resolve_l1_x_cap(
        bound_in, b_cap_single, opts
    )
    b_ttft_s = (
        bound_ttft_ms(
            geom,
            dtypes,
            chip,
            scenario,
            prefill_par=ttft_par,
            prefill_topo=ttft_topo,
            activation_model=bound_in.activation_model,
            pp_part=opts.get("pp_part"),
        )
        / UNIT_MILLIS_PER_SECOND
    )
    sf_b1 = _eval_bound_decode_floor(bound_in, 1)
    bound_tpot_b1_s = sf_b1.hard_floor_s / bound_in.eff_al
    return min(
        x_cap_val,
        float(conc)
        / max(
            _RATE_FLOOR_EPS,
            b_ttft_s + float(scenario.osl) * bound_tpot_b1_s + scenario.think_time_s,
        ),
    )


X_bound = x_bound


def floor_throughput_per_chip(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    **opts: Unpack[_L1BoundOpts],
) -> float:
    """Return isolated Layer-1 per-chip token throughput ceiling (tok/s/chip)."""
    x_rate_bound = x_bound(geom, dtypes, chip, scenario, **opts)
    total_chips = opts.get("total_chips")
    if total_chips is None:
        dec_topo = opts.get("decode_topo")
        pref_topo = opts.get("prefill_topo")
        if dec_topo is None:
            msg = "decode_topo or total_chips is required"
            raise ValueError(msg)
        if opts.get("prefill_par") is not None and pref_topo is not None:
            x_p = max(1, opts.get("x_p", 1))
            y_d = max(1, opts.get("y_d", 1))
            total_chips = x_p * pref_topo.num_chips + y_d * dec_topo.num_chips
        else:
            total_chips = dec_topo.num_chips
    return (x_rate_bound * float(scenario.isl + scenario.osl)) / float(
        max(1, total_chips)
    )


def _eval_decode_floor(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    batch: int,
) -> StepFloor:
    """Evaluate single-step or speculative-cycle floor at midpoint context."""
    key = (topo.name, par, batch)
    cached = eval_context.decode_floor_cache.get(key)
    if cached is not None:
        return cached
    opts = FloorOptions(
        dt=eval_context.dtypes,
        collective_latency_s=eval_context.collective_latency_s,
        topology=topo,
        activation_model=eval_context.activation_model,
    )
    sf = (
        spec_cycle_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            SpecShape(batch=batch, context=eval_context.ctx_mid, k=eval_context.spec_k),
            opts,
        )
        if eval_context.spec_k > 0
        else step_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            StepShape(batch=batch, context=eval_context.ctx_mid),
            opts,
        )
    )
    eval_context.decode_floor_cache[key] = sf
    return sf


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
            activation_model=eval_context.activation_model,
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


def _prefill_stage_step_floor(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    hit_rate: float,
    pp_part: tuple[int, ...] | None = None,
) -> StepFloor:
    """Evaluate prefill step floor, using exact stage slices for uneven PP."""
    if not pp_part or par.pp <= 1:
        return _prefill_step_floor(eval_context, topo, par, hit_rate)
    shape = StepShape(
        batch=0,
        rows_per_seq=0,
        context=0,
        state_writes=0,
        prefill_tokens=eval_context.scenario.isl,
        prefix_hit_rate=hit_rate,
    )
    opts = FloorOptions(
        dt=eval_context.dtypes,
        collective_latency_s=eval_context.collective_latency_s,
        topology=topo,
        activation_model=eval_context.activation_model,
    )
    return scale_pp_stage_floor(
        eval_context.geom, eval_context.chip, par, (shape, opts), pp_part
    )


def _resolve_prefill_pp_breakdown(
    eval_context: _SolveContext,
    par: Parallel,
    logit_peak_b: float,
) -> tuple[HbmBreakdown, float, tuple[int, ...] | None, tuple[int, ...] | None] | None:
    """Resolve prefill stage HBM breakdown, stage partition, and uneven partition."""
    if par.pp <= 1:
        hbm_bd = hbm_breakdown_per_core_bytes(
            eval_context.geom,
            eval_context.dtypes,
            par,
            SeqKvLayout(1, eval_context.scenario.isl),
            chip=eval_context.chip,
        )
        return (hbm_bd, hbm_bd.total, None, None)
    even_cap = max(1, math.ceil(eval_context.geom.num_layers / par.pp))
    is_hybrid = (
        0 < count_full_attn_layers(eval_context.geom) < eval_context.geom.num_layers
    )
    pp_part, hbm_bd, last_stage_b, pp_feas = pp_stage_breakdowns(
        eval_context.geom,
        eval_context.dtypes,
        par,
        (eval_context.scenario.isl, 1),
        options=PpStageOptions(
            disagg=True,
            max_stage_layers=even_cap,
            logit_peak_b=logit_peak_b,
            require_full_attn_per_stage=is_hybrid,
        ),
        chip=eval_context.chip,
    )
    if (
        not pp_feas
        or hbm_bd.total > eval_context.hbm_limit_b
        or (last_stage_b + logit_peak_b) > eval_context.hbm_limit_b
    ):
        pp_part, hbm_bd, last_stage_b, pp_feas = pp_stage_breakdowns(
            eval_context.geom,
            eval_context.dtypes,
            par,
            (eval_context.scenario.isl, 1),
            options=PpStageOptions(
                disagg=True,
                logit_peak_b=logit_peak_b,
                require_full_attn_per_stage=is_hybrid,
            ),
            chip=eval_context.chip,
        )
    if not pp_feas or not pp_part:
        return None
    pp_part_opt = pp_part if len(set(pp_part)) > 1 else None
    return (hbm_bd, last_stage_b, pp_part, pp_part_opt)


def _find_prefill_options(
    eval_context: _SolveContext,
    topos: list[SliceTopology],
) -> tuple[list[_PrefillOption], float]:
    """Enumerate all legal, HBM-feasible prefill configurations without collapsing."""
    prefill_options: list[_PrefillOption] = []
    min_req_ttft_ms = math.inf
    pre_chunk_rows = min(eval_context.eff_isl, eval_context.scenario.prefill_chunk)
    eff_isl_floor = max(
        0,
        math.floor(
            (1.0 - eval_context.scenario.prefix_hit_rate)
            * float(eval_context.scenario.isl)
        ),
    )
    prefill_spec_k = disagg_prefill_spec_k(
        eval_context.spec_k, resolve_geom_profile(eval_context.geom).parallel
    )
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
            spec_k=prefill_spec_k,
            chip=eval_context.chip,
        ):
            logit_peak_b = _peak_prefill_logit_bytes(
                eval_context.geom,
                eval_context.dtypes,
                par,
                pre_chunk_rows,
                sampled_rows=1,
            )
            stage_info = _resolve_prefill_pp_breakdown(eval_context, par, logit_peak_b)
            if stage_info is None:
                continue
            hbm_bd, last_stage_b, pp_part, pp_part_opt = stage_info
            hbm_b = hbm_bd.total
            if (
                hbm_b > eval_context.hbm_limit_b
                or (last_stage_b + logit_peak_b) > eval_context.hbm_limit_b
            ):
                continue
            sf_p = _prefill_stage_step_floor(
                eval_context,
                topo,
                par,
                eval_context.scenario.prefix_hit_rate,
                pp_part,
            )
            pp_factor = 1.0 + float(par.pp - 1) / float(max(1, eval_context.n_chunks))
            ttft_ms = sf_p.serial_estimate_s * pp_factor * UNIT_MILLIS_PER_SECOND
            xfer_min_ms = (
                0.0
                if eval_context.mode in ("auto", "mixed") or eff_isl_floor <= 0
                else _disagg_kv_xfer_s(
                    eval_context.geom,
                    eval_context.dtypes,
                    eval_context.chip,
                    (topo, topo),
                    eff_isl_floor,
                )
                * UNIT_MILLIS_PER_SECOND
            )
            bound_ttft_val = sf_p.hard_floor_s * UNIT_MILLIS_PER_SECOND
            min_req_ttft_ms = min(min_req_ttft_ms, bound_ttft_val + xfer_min_ms)
            rate_p = 1.0 / max(_RATE_FLOOR_EPS, sf_p.serial_estimate_s)
            pref_w_base_b = hbm_bd.total - hbm_bd.kv - hbm_bd.state
            if par.pp > 1 and pp_part:
                pref_b_kv_phys, pref_fixed_kv_b = pp_stage_kv_parts(
                    eval_context.geom,
                    eval_context.dtypes,
                    par,
                    (pp_part, eval_context.scenario.isl),
                    chip=eval_context.chip,
                )
            else:
                pref_b_kv_phys, pref_fixed_kv_b = kv_bytes_per_core_parts(
                    eval_context.geom,
                    eval_context.dtypes,
                    par,
                    eval_context.scenario.isl,
                    chip=eval_context.chip,
                )
            pref_offload_b = (
                float(eval_context.scenario.kv_offload_gb)
                * UNIT_GIBI
                / float(max(1, par.tp * par.pp * par.dp * par.pcp))
            )
            prefill_options.append(
                _PrefillOption(
                    topo=topo,
                    par=par,
                    ttft_ms=ttft_ms,
                    rate_p=rate_p,
                    hbm_gb=hbm_b / UNIT_GIGA,
                    hbm_bd=hbm_bd,
                    pp_part=pp_part_opt,
                    bound_ttft_ms=bound_ttft_val,
                    envelope_ttft_ms=sf_p.envelope_floor_s * UNIT_MILLIS_PER_SECOND,
                    w_base_b=pref_w_base_b,
                    b_kv_phys=pref_b_kv_phys,
                    offload_b_per_core=pref_offload_b,
                    logit_peak_b=logit_peak_b,
                    fixed_kv_per_seq_b=pref_fixed_kv_b,
                    pp_stage_partition=pp_part,
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


def _fit_mixed_batch_with_logits(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    bounds: _MixedPointBounds,
) -> tuple[int, int, HbmBreakdown] | None:
    """Compute active local batch and HBM-fitting resident batch with logit peak."""
    b_local = _solve_mixed_think_batch(eval_context, topo, par, c, bounds.sf_ttft)
    b_res = min(b_local, bounds.b_hbm_max, bounds.b_smem_max)
    if eval_context.scenario.max_num_seqs is not None:
        b_res = min(b_res, eval_context.scenario.max_num_seqs)
    pre_chunk_rows = min(eval_context.eff_isl, eval_context.scenario.prefill_chunk)
    row_mul = max(1, eval_context.spec_k + 1)
    hbm_bd = hbm_breakdown_per_core_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        SeqKvLayout(b_res, eval_context.ctx_full),
        chip=eval_context.chip,
    )
    logit_peak_b = _peak_prefill_logit_bytes(
        eval_context.geom,
        eval_context.dtypes,
        par,
        pre_chunk_rows,
        sampled_rows=b_res * row_mul,
    )
    while b_res > 1 and (hbm_bd.total + logit_peak_b) > eval_context.hbm_limit_b:
        b_res -= 1
        hbm_bd = hbm_breakdown_per_core_bytes(
            eval_context.geom,
            eval_context.dtypes,
            par,
            SeqKvLayout(b_res, eval_context.ctx_full),
            chip=eval_context.chip,
        )
        logit_peak_b = _peak_prefill_logit_bytes(
            eval_context.geom,
            eval_context.dtypes,
            par,
            pre_chunk_rows,
            sampled_rows=b_res * row_mul,
        )
    if (hbm_bd.total + logit_peak_b) > eval_context.hbm_limit_b:
        return None
    return (b_local, b_res, hbm_bd)


def _solve_mixed_decode_fixed_point(
    eval_context: _SolveContext,
    topo: SliceTopology,
    par: Parallel,
    c: int,
    loop_in: _MixedLoopInput,
) -> _MixedLoopOutput:
    """Solve 3-step Little's-law fixed point for mixed prefill-decode serving."""
    eff_al = eval_context.eff_al
    think = eval_context.scenario.think_time_s
    osl = float(eval_context.scenario.osl)
    dec_steps_per_req = max(1.0, osl / eff_al)
    dp_val = max(1, par.dp)
    max_decode_seqs = float(loop_in.b_res * dp_val)
    b_dec = max(1.0, min(float(min(c, int(max_decode_seqs))), float(c)))
    opts = FloorOptions(
        dt=eval_context.dtypes,
        collective_latency_s=eval_context.collective_latency_s,
        topology=topo,
        activation_model=eval_context.activation_model,
    )
    eff_isl_c = (
        loop_in.eff_isl_c
        if loop_in.eff_isl_c > 0
        else max(1, loop_in.chunk_tok_c * loop_in.n_chunks_c)
    )
    cached_prefix_c = max(0, eval_context.scenario.isl - eff_isl_c)
    avg_prior = float(cached_prefix_c) + CAUSAL_MIDPOINT_HALF * float(
        max(0, eff_isl_c - loop_in.chunk_tok_c)
    )
    mix_pre_total = max(1, round(avg_prior + float(loop_in.chunk_tok_c)))
    mix_hit_rate = min(1.0, max(0.0, avg_prior / float(mix_pre_total)))

    sf_dec = _eval_decode_floor(eval_context, topo, par, 1)
    sf_mix = sf_dec
    p_mix = 0.0
    tpot_s = 0.0
    t_dec_s = 0.0
    x_rate = 0.0
    for _ in range(_FIXED_POINT_ITERS):
        b_replica = max(1, min(loop_in.b_res, math.ceil(b_dec / float(dp_val))))
        sf_dec = _eval_decode_floor(eval_context, topo, par, b_replica)
        sf_ver_only = (
            step_floor(
                eval_context.geom,
                eval_context.chip,
                par,
                StepShape(
                    batch=b_replica,
                    rows_per_seq=eval_context.spec_k + 1,
                    context=eval_context.ctx_mid,
                    state_writes=1,
                ),
                opts,
            )
            if eval_context.spec_k > 0
            else sf_dec
        )
        dec_step_s = _step_duration_s(sf_dec, loop_in.duration_kind)
        ver_step_s = _step_duration_s(sf_ver_only, loop_in.duration_kind)
        draft_step_s = max(0.0, dec_step_s - ver_step_s)
        sf_mix = step_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            StepShape(
                batch=b_replica,
                rows_per_seq=eval_context.spec_k + 1,
                context=eval_context.ctx_mid,
                state_writes=1,
                prefill_tokens=mix_pre_total,
                prefix_hit_rate=mix_hit_rate,
            ),
            opts,
        )
        mix_step_s = _step_duration_s(sf_mix, loop_in.duration_kind) + draft_step_s
        sf_pure_chunk = step_floor(
            eval_context.geom,
            eval_context.chip,
            par,
            StepShape(
                batch=0,
                rows_per_seq=0,
                context=0,
                state_writes=0,
                prefill_tokens=mix_pre_total,
                prefix_hit_rate=mix_hit_rate,
            ),
            opts,
        )
        pure_chunk_s = _step_duration_s(sf_pure_chunk, loop_in.duration_kind)
        bn = float(b_replica * loop_in.n_chunks_c)
        p_mix = min(1.0, bn / max(1.0, dec_steps_per_req))
        effective_step_s = (1.0 - p_mix) * dec_step_s + p_mix * mix_step_s
        tpot_s = effective_step_s / eff_al
        t_dec_s = osl * tpot_s
        dec_per_res = dec_steps_per_req / float(max(1, loop_in.b_res))
        n_ch = float(loop_in.n_chunks_c)
        if dec_per_res >= n_ch:
            replica_req_time_s = (
                (dec_per_res - n_ch) * dec_step_s
                + n_ch * mix_step_s
                + loop_in.pcie_read_s
            )
        else:
            replica_req_time_s = (
                dec_per_res * mix_step_s
                + (n_ch - dec_per_res) * pure_chunk_s
                + loop_in.pcie_read_s
            )
        x_capacity = float(dp_val) / max(_RATE_FLOOR_EPS, replica_req_time_s)
        prefill_wait_s = (
            loop_in.ttft_upper_s
            if loop_in.ttft_upper_s is not None
            else loop_in.ttft_service_s
        )
        x_rate = min(
            x_capacity,
            float(c) / max(_RATE_FLOOR_EPS, prefill_wait_s + t_dec_s + think),
        )
        b_dec = max(1.0, min(float(min(c, int(max_decode_seqs))), x_rate * t_dec_s))
    ttft_est_s = float(c) / max(_RATE_FLOOR_EPS, x_rate) - t_dec_s - think
    return _MixedLoopOutput(
        sf_dec=sf_dec,
        sf_mix=sf_mix,
        p_mix=p_mix,
        est_ttft_ms=ttft_est_s * UNIT_MILLIS_PER_SECOND,
        est_tpot_ms=tpot_s * UNIT_MILLIS_PER_SECOND,
        x_rate=x_rate,
        decode_seqs=b_dec,
    )


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
    fitted = _fit_mixed_batch_with_logits(eval_context, topo, par, c, bounds)
    if fitted is None:
        return None
    _b_local, b_res, hbm_bd = fitted
    hbm_b = hbm_bd.total
    eval_context.hbm_fit += 1
    offload_b_per_core = (
        float(eval_context.scenario.kv_offload_gb)
        * UNIT_GIBI
        / float(max(1, par.tp * par.pp * par.dp * par.pcp))
    )
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
    if offload_b_per_core > 0.0:
        _, h_hbm_only = kv_capacity_and_hit_rate(
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
                offload_b_per_core=0.0,
                prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
                kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
                fixed_kv_per_seq_bytes=bounds.fixed_kv_per_seq_b,
            ),
            chip=eval_context.chip,
        )
    else:
        h_hbm_only = h_eff
    pcie_read_s = _pcie_read_s(eval_context, bounds.b_kv_phys_core, h_eff, h_hbm_only)
    sf_ttft_c = (
        _prefill_step_floor(eval_context, topo, par, h_eff)
        if h_eff != eval_context.scenario.prefix_hit_rate
        else bounds.sf_ttft
    )
    ttft_service_s = sf_ttft_c.serial_estimate_s + pcie_read_s
    bound_ttft_val = bounds.sf_ttft.hard_floor_s * UNIT_MILLIS_PER_SECOND
    envelope_ttft_val = bounds.sf_ttft.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    eff_isl_c = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
    n_chunks_c = max(
        1, math.ceil(eff_isl_c / max(1, eval_context.scenario.prefill_chunk))
    )
    chunk_tok_c = min(eff_isl_c, eval_context.scenario.prefill_chunk)
    solved = _solve_mixed_decode_fixed_point(
        eval_context,
        topo,
        par,
        c,
        _MixedLoopInput(
            b_res=b_res,
            n_chunks_c=n_chunks_c,
            chunk_tok_c=chunk_tok_c,
            ttft_service_s=ttft_service_s,
            pcie_read_s=pcie_read_s,
            eff_isl_c=eff_isl_c,
        ),
    )
    interactivity = UNIT_MILLIS_PER_SECOND / max(_RATE_FLOOR_EPS, solved.est_tpot_ms)
    out_tput = (solved.x_rate * float(eval_context.scenario.osl)) / float(
        topo.num_chips
    )
    tot_tput = (
        solved.x_rate * float(eval_context.scenario.isl + eval_context.scenario.osl)
    ) / float(topo.num_chips)
    b_cap_single = max(
        1,
        min(
            bounds.b_hbm_max,
            bounds.b_smem_max,
            (
                eval_context.scenario.max_num_seqs
                if eval_context.scenario.max_num_seqs is not None
                else bounds.b_hbm_max
            ),
        ),
    )
    x_cap_val = _compute_mixed_capacity_bound(eval_context, topo, par, b_cap_single)
    bound_in = _context_to_bound_input(eval_context, topo, par)
    sf_dec_b1 = _eval_bound_decode_floor(bound_in, 1)
    x_bound_val = min(
        x_cap_val,
        float(c)
        / max(
            _RATE_FLOOR_EPS,
            bound_ttft_val / UNIT_MILLIS_PER_SECOND
            + float(eval_context.scenario.osl)
            * (sf_dec_b1.hard_floor_s / eval_context.eff_al)
            + eval_context.scenario.think_time_s,
        ),
    )
    floor_tput = (
        x_bound_val * float(eval_context.scenario.isl + eval_context.scenario.osl)
    ) / float(topo.num_chips)
    min_decode_seqs, bound_sf_dec = _run_isolated_tarski_loop(
        bound_in, c, b_cap_single, max(1, par.dp), x_cap_val
    )
    bound_tpot_val = (
        bound_sf_dec.hard_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.eff_al
    envelope_tpot_val = (
        bound_sf_dec.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.eff_al
    use_envelope = eval_context.activation_model.prunes_on_envelope
    verdict = _resolve_slo_verdict(
        eval_context.scenario,
        solved.est_ttft_ms,
        solved.est_tpot_ms,
        envelope_ttft_val if use_envelope else bound_ttft_val,
        envelope_tpot_val if use_envelope else bound_tpot_val,
    )
    verdict = _apply_capacity_ceiling_verdict(
        eval_context.scenario,
        verdict,
        concurrency=c,
        pool_capacity_req_s=x_cap_val,
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
            pp_layer_partition=None,
            kv_offload_gb=eval_context.scenario.kv_offload_gb,
        )
    )
    binding = (
        f"dec:{solved.sf_dec.binding},mix:{solved.sf_mix.binding}"
        if solved.p_mix > 0.0
        else solved.sf_dec.binding
    )
    return ParetoPoint(
        mode="mixed",
        concurrency=c,
        total_chips=topo.num_chips,
        prefill_pool=None,
        decode_pool=pool,
        est_ttft_ms=solved.est_ttft_ms,
        est_tpot_ms=solved.est_tpot_ms,
        est_interactivity_tok_s=interactivity,
        output_throughput_per_chip=out_tput,
        token_throughput_per_chip=tot_tput,
        p_mixed=solved.p_mix,
        binding=binding,
        slo_met=verdict == SLO_VERDICT_ESTIMATE_MEETS,
        bound_tpot_ms=bound_tpot_val,
        bound_ttft_ms=bound_ttft_val,
        slo_verdict=verdict,
        decode_floor=_floor_times(bound_sf_dec),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
        min_decode_seqs=min_decode_seqs,
        floor_throughput_per_chip=floor_tput,
        envelope_tpot_ms=envelope_tpot_val,
        envelope_ttft_ms=envelope_ttft_val,
        envelope_decode_seqs=min_decode_seqs,
    )


def _decode_capacity_profile(
    eval_context: _SolveContext,
    par: Parallel,
    *,
    logit_peak_b: float = 0.0,
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
    fits_c1 = (hbm_c1 + logit_peak_b) <= eval_context.hbm_limit_b
    b_hbm_max = (
        int(max(0.0, eval_context.hbm_limit_b - logit_peak_b - w_base_b) // per_seq_b)
        if (fits_c1 and per_seq_b > 0.0)
        else (eval_context.c_max if fits_c1 else 0)
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
    pre_chunk_rows = min(eval_context.eff_isl, eval_context.scenario.prefill_chunk)
    c1_rows = max(1, eval_context.spec_k + 1)
    if eval_context.max_instances is not None and eval_context.max_instances < 1:
        return (candidates, min_c1_tpot_ms, min_req_ttft_ms)
    for topo in topos:
        if (
            eval_context.decode_topology is not None
            and topo.name != eval_context.decode_topology
        ):
            continue
        if (
            eval_context.max_chips is not None
            and topo.num_chips > eval_context.max_chips
        ):
            continue
        for par in _candidate_decode_parallels(
            eval_context.geom, topo, chip=eval_context.chip
        ):
            logit_peak_mix_b = _peak_prefill_logit_bytes(
                eval_context.geom,
                eval_context.dtypes,
                par,
                pre_chunk_rows,
                sampled_rows=c1_rows,
            )
            (
                w_base_b,
                b_hbm_max,
                b_smem_max,
                b_kv_phys_core,
                fixed_kv_per_seq_b,
                hbm_c1,
            ) = _decode_capacity_profile(
                eval_context, par, logit_peak_b=logit_peak_mix_b
            )
            if (
                hbm_c1 + logit_peak_mix_b
            ) <= eval_context.hbm_limit_b and b_smem_max >= 1:
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


def _run_disagg_decode_loop(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    target: _DisaggRatioTarget,
    batch_stats: _DisaggBatchStats,
    loop_ctrl: tuple[float, _StepDurationKind, float],
) -> tuple[StepFloor, float, float, float, float]:
    """Solve 3-step Little's-law fixed point for a disaggregated decode pool."""
    prefill_wait_s, duration_kind, rate_dcn_cap = loop_ctrl
    eff_al = eval_context.eff_al
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eff_al)
    think = eval_context.scenario.think_time_s
    osl = float(eval_context.scenario.osl)
    idp_val = max(1, target.y_d * dec.par.dp)
    max_decode_seqs = float(batch_stats.b_res * idp_val)
    b_dec = max(1.0, min(float(min(target.c, int(max_decode_seqs))), float(target.c)))
    sf_d = _eval_decode_floor(eval_context, dec.topo, dec.par, 1)
    tpot_s = 0.0
    t_dec_s = 0.0
    x_rate = 0.0
    for _ in range(_FIXED_POINT_ITERS):
        b_replica = max(1, min(batch_stats.b_res, math.ceil(b_dec / float(idp_val))))
        sf_d = _eval_decode_floor(eval_context, dec.topo, dec.par, b_replica)
        step_s = _step_duration_s(sf_d, duration_kind)
        tpot_s = step_s / eff_al
        t_dec_s = osl * tpot_s
        rate_d_cap = float(batch_stats.b_res * idp_val) / (
            dec_steps_per_req * max(_RATE_FLOOR_EPS, step_s)
        )
        x_capacity = min(
            float(target.x_p) * batch_stats.rate_p_eff, rate_d_cap, rate_dcn_cap
        )
        x_rate = min(
            x_capacity,
            float(target.c) / max(_RATE_FLOOR_EPS, prefill_wait_s + t_dec_s + think),
        )
        b_dec = max(
            1.0, min(float(min(target.c, int(max_decode_seqs))), x_rate * t_dec_s)
        )
    return (sf_d, tpot_s, t_dec_s, x_rate, b_dec)


def _compute_disagg_metrics(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    batch_stats: _DisaggBatchStats,
) -> _DisaggMetricsOutput:
    """Solve coupled Little's-law fixed point for a disaggregated candidate."""
    think = eval_context.scenario.think_time_s
    osl = float(eval_context.scenario.osl)
    uncached_d_tok = max(
        0, math.ceil((1.0 - batch_stats.h_eff) * eval_context.scenario.isl)
    )
    if uncached_d_tok > 0:
        xfer_key = (pref.topo.name, dec.topo.name, uncached_d_tok)
        xfer_s = eval_context.xfer_cache.get(xfer_key)
        if xfer_s is None:
            xfer_s = _disagg_kv_xfer_s(
                eval_context.geom,
                eval_context.dtypes,
                eval_context.chip,
                (pref.topo, dec.topo),
                uncached_d_tok,
            )
            eval_context.xfer_cache[xfer_key] = xfer_s
    else:
        xfer_s = 0.0
    if xfer_s > 0.0:
        min_one = float(min(pref.topo.num_chips, dec.topo.num_chips))
        min_pool = float(
            min(target.x_p * pref.topo.num_chips, target.y_d * dec.topo.num_chips)
        )
        rate_dcn_cap = (min_pool / max(1.0, min_one)) / xfer_s
    else:
        rate_dcn_cap = math.inf
    kv_xfer_ms = xfer_s * UNIT_MILLIS_PER_SECOND
    ttft_service_s = (batch_stats.ttft_eff_ms + kv_xfer_ms) / UNIT_MILLIS_PER_SECOND
    sf_d, tpot_s, t_dec_s, x_rate, _ = _run_disagg_decode_loop(
        eval_context,
        dec,
        target,
        batch_stats,
        (ttft_service_s, "serial_estimate", rate_dcn_cap),
    )
    ttft_est_s = float(target.c) / max(_RATE_FLOOR_EPS, x_rate) - t_dec_s - think
    est_ttft_d_ms = ttft_est_s * UNIT_MILLIS_PER_SECOND
    est_tpot_ms = tpot_s * UNIT_MILLIS_PER_SECOND
    interactivity = UNIT_MILLIS_PER_SECOND / max(_RATE_FLOOR_EPS, est_tpot_ms)
    out_tput = (x_rate * osl) / float(target.total_chips)
    tot_tput = (
        x_rate * float(eval_context.scenario.isl + eval_context.scenario.osl)
    ) / float(target.total_chips)
    b_cap_single = max(
        1,
        min(
            dec.b_hbm_max,
            dec.b_smem_max,
            (
                eval_context.scenario.max_num_seqs
                if eval_context.scenario.max_num_seqs is not None
                else dec.b_hbm_max
            ),
        ),
    )
    x_cap_val = _compute_disagg_capacity_bound(
        eval_context, dec, pref, target, b_cap_single
    )
    bound_in = _context_to_bound_input(eval_context, dec.topo, dec.par)
    sf_dec_b1 = _eval_bound_decode_floor(bound_in, 1)
    x_bound_val = min(
        x_cap_val,
        float(target.c)
        / max(
            _RATE_FLOOR_EPS,
            pref.bound_ttft_ms / UNIT_MILLIS_PER_SECOND
            + osl * (sf_dec_b1.hard_floor_s / eval_context.eff_al)
            + think,
        ),
    )
    floor_tput = (
        x_bound_val * float(eval_context.scenario.isl + eval_context.scenario.osl)
    ) / float(target.total_chips)
    rate_p_floor = float(target.x_p) / max(
        _RATE_FLOOR_EPS, pref.bound_ttft_ms / UNIT_MILLIS_PER_SECOND
    )
    min_decode_seqs, bound_sf_d = _run_isolated_tarski_loop(
        bound_in,
        target.c,
        b_cap_single,
        max(1, target.y_d * dec.par.dp),
        min(x_cap_val, rate_p_floor),
    )
    return _DisaggMetricsOutput(
        est_ttft_ms=est_ttft_d_ms,
        est_tpot_ms=est_tpot_ms,
        interactivity=interactivity,
        out_tput=out_tput,
        tot_tput=tot_tput,
        sf_d=sf_d,
        bound_sf_d=bound_sf_d,
        min_decode_seqs=min_decode_seqs,
        floor_throughput_per_chip=floor_tput,
        envelope_step_floor=bound_sf_d,
        envelope_decode_seqs=min_decode_seqs,
        x_cap_pool=x_cap_val,
    )


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
    cache: dict[tuple[str, Parallel, float], _EffectivePrefill],
) -> _EffectivePrefill:
    """Recompute cached prefill TTFT, rate, hard bound, and envelope bound."""
    if h_eff >= eval_context.scenario.prefix_hit_rate:
        return _EffectivePrefill(
            ttft_ms=pref.ttft_ms,
            rate_p=pref.rate_p,
            bound_ttft_ms=pref.bound_ttft_ms,
            envelope_ttft_ms=pref.envelope_ttft_ms,
        )
    p_key = (pref.topo.name, pref.par, round(h_eff, _HIT_RATE_KEY_DECIMALS))
    cached_p = cache.get(p_key)
    if cached_p is not None:
        return cached_p
    eff_isl_p = max(1, math.ceil((1.0 - h_eff) * eval_context.scenario.isl))
    n_chunks_p = max(
        1, math.ceil(eff_isl_p / max(1, eval_context.scenario.prefill_chunk))
    )
    sf_p_eff = _prefill_stage_step_floor(
        eval_context, pref.topo, pref.par, h_eff, pref.pp_stage_partition
    )
    ttft_p_eff = (
        sf_p_eff.serial_estimate_s
        * (1.0 + float(pref.par.pp - 1) / float(max(1, n_chunks_p)))
        * UNIT_MILLIS_PER_SECOND
    )
    rate_p_eff = 1.0 / max(_RATE_FLOOR_EPS, sf_p_eff.serial_estimate_s)
    cached_p = _EffectivePrefill(
        ttft_ms=ttft_p_eff,
        rate_p=rate_p_eff,
        bound_ttft_ms=sf_p_eff.hard_floor_s * UNIT_MILLIS_PER_SECOND,
        envelope_ttft_ms=sf_p_eff.envelope_floor_s * UNIT_MILLIS_PER_SECOND,
    )
    cache[p_key] = cached_p
    return cached_p


def _fit_disagg_decode_batch(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
) -> tuple[int, int, HbmBreakdown] | None:
    """Resolve active local batch and HBM-fitting resident decode batch."""
    b_local = _solve_disagg_think_batch(eval_context, dec, pref, target)
    b_start = min(b_local, dec.b_hbm_max, dec.b_smem_max)
    if eval_context.scenario.max_num_seqs is not None:
        b_start = min(b_start, eval_context.scenario.max_num_seqs)
    clamp_key = (dec.par, b_start)
    if clamp_key in eval_context.clamp_batch_cache:
        clamped = eval_context.clamp_batch_cache[clamp_key]
    else:
        clamped = _clamp_resident_batch(eval_context, dec.par, b_start)
        eval_context.clamp_batch_cache[clamp_key] = clamped
    if clamped is None:
        return None
    return (b_local, clamped[0], clamped[1])


def _evaluate_disagg_instance_pair(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    target: _DisaggRatioTarget,
    cache: dict[tuple[str, Parallel, float], _EffectivePrefill],
) -> ParetoPoint | None:
    """Evaluate one instance ratio pair (x_p, y_d) for disaggregated serving."""
    if (
        eval_context.max_chips is not None
        and target.total_chips > eval_context.max_chips
    ):
        return None
    if dec.b_hbm_max < 1 or dec.b_smem_max < 1:
        return None
    fitted_d = _fit_disagg_decode_batch(eval_context, dec, pref, target)
    if fitted_d is None:
        return None
    b_local, b_res, hbm_bd_d = fitted_d
    hbm_d = hbm_bd_d.total
    eval_context.hbm_fit += 1
    b_ws_p = max(1, math.ceil(target.c / max(1, target.x_p * pref.par.dp)))
    pref_hbm_limit_b = max(0.0, eval_context.hbm_limit_b - pref.logit_peak_b)
    kv_key = (pref.par, -1, b_ws_p, pref.offload_b_per_core)
    cached_kv = eval_context.kv_fit_cache.get(kv_key)
    if cached_kv is None:
        cached_kv = kv_capacity_and_hit_rate(
            eval_context.geom,
            eval_context.dtypes,
            pref.par,
            1,
            KvFitInput(
                b_ws=b_ws_p,
                ctx_full=eval_context.scenario.isl,
                hbm_limit_b=pref_hbm_limit_b,
                w_base_b=pref.w_base_b,
                b_kv_phys_core=pref.b_kv_phys,
                offload_b_per_core=pref.offload_b_per_core,
                prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
                kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
                fixed_kv_per_seq_bytes=pref.fixed_kv_per_seq_b,
            ),
            chip=eval_context.chip,
        )
        eval_context.kv_fit_cache[kv_key] = cached_kv
    cap_tok, h_eff = cached_kv
    cap_tok_int = None if math.isinf(cap_tok) else int(cap_tok)
    if pref.offload_b_per_core > 0.0:
        kv_key_hbm = (pref.par, -1, b_ws_p, 0.0)
        cached_kv_hbm = eval_context.kv_fit_cache.get(kv_key_hbm)
        if cached_kv_hbm is None:
            cached_kv_hbm = kv_capacity_and_hit_rate(
                eval_context.geom,
                eval_context.dtypes,
                pref.par,
                1,
                KvFitInput(
                    b_ws=b_ws_p,
                    ctx_full=eval_context.scenario.isl,
                    hbm_limit_b=pref_hbm_limit_b,
                    w_base_b=pref.w_base_b,
                    b_kv_phys_core=pref.b_kv_phys,
                    offload_b_per_core=0.0,
                    prefix_hit_rate=eval_context.scenario.prefix_hit_rate,
                    kv_working_set_turns=eval_context.scenario.kv_working_set_turns,
                    fixed_kv_per_seq_bytes=pref.fixed_kv_per_seq_b,
                ),
                chip=eval_context.chip,
            )
            eval_context.kv_fit_cache[kv_key_hbm] = cached_kv_hbm
        _, h_hbm_p = cached_kv_hbm
    else:
        h_hbm_p = h_eff
    pcie_p_ms = (
        _pcie_read_s(eval_context, pref.b_kv_phys, h_eff, h_hbm_p)
        * UNIT_MILLIS_PER_SECOND
    )
    eff_prefill = _resolve_effective_prefill(eval_context, pref, h_eff, cache)
    rate_p_eff = 1.0 / max(
        _RATE_FLOOR_EPS,
        1.0 / max(_RATE_FLOOR_EPS, eff_prefill.rate_p)
        + pcie_p_ms / UNIT_MILLIS_PER_SECOND,
    )
    bound_ttft_val = pref.bound_ttft_ms
    envelope_ttft_val = pref.envelope_ttft_ms
    disagg_metrics = _compute_disagg_metrics(
        eval_context,
        dec,
        pref,
        target,
        _DisaggBatchStats(
            b_res=b_res,
            b_local=b_local,
            ttft_eff_ms=eff_prefill.ttft_ms + pcie_p_ms,
            rate_p_eff=rate_p_eff,
            h_eff=h_eff,
        ),
    )
    bound_tpot_val = (
        disagg_metrics.bound_sf_d.hard_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.eff_al
    envelope_tpot_val = (
        disagg_metrics.envelope_step_floor.envelope_floor_s * UNIT_MILLIS_PER_SECOND
    ) / eval_context.eff_al
    use_envelope = eval_context.activation_model.prunes_on_envelope
    verdict = _resolve_slo_verdict(
        eval_context.scenario,
        disagg_metrics.est_ttft_ms,
        disagg_metrics.est_tpot_ms,
        envelope_ttft_val if use_envelope else bound_ttft_val,
        envelope_tpot_val if use_envelope else bound_tpot_val,
    )
    verdict = _apply_capacity_ceiling_verdict(
        eval_context.scenario,
        verdict,
        concurrency=target.c,
        pool_capacity_req_s=disagg_metrics.x_cap_pool,
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
            pp_layer_partition=pref.pp_part,
            kv_offload_gb=eval_context.scenario.kv_offload_gb,
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
            pp_layer_partition=None,
            kv_offload_gb=0.0,
        )
    )
    return ParetoPoint(
        mode="disagg",
        concurrency=target.c,
        total_chips=target.total_chips,
        prefill_pool=p_pool,
        decode_pool=d_pool,
        est_ttft_ms=disagg_metrics.est_ttft_ms,
        est_tpot_ms=disagg_metrics.est_tpot_ms,
        est_interactivity_tok_s=disagg_metrics.interactivity,
        output_throughput_per_chip=disagg_metrics.out_tput,
        token_throughput_per_chip=disagg_metrics.tot_tput,
        p_mixed=0.0,
        binding=disagg_metrics.sf_d.binding,
        slo_met=verdict == SLO_VERDICT_ESTIMATE_MEETS,
        bound_tpot_ms=bound_tpot_val,
        bound_ttft_ms=bound_ttft_val,
        slo_verdict=verdict,
        decode_floor=_floor_times(disagg_metrics.bound_sf_d),
        kv_cache_capacity_tokens=cap_tok_int,
        effective_prefix_hit_rate=h_eff,
        min_decode_seqs=disagg_metrics.min_decode_seqs,
        floor_throughput_per_chip=disagg_metrics.floor_throughput_per_chip,
        envelope_tpot_ms=envelope_tpot_val,
        envelope_ttft_ms=envelope_ttft_val,
        envelope_decode_seqs=disagg_metrics.envelope_decode_seqs,
    )


def _search_disagg_instance_ratios(
    eval_context: _SolveContext,
    dec: _DisaggDecodeState,
    pref: _PrefillOption,
    c: int,
    cache: dict[tuple[str, Parallel, float], _EffectivePrefill],
) -> list[ParetoPoint]:
    """Search instance multiplier ratios (x_p, y_d) for one prefill-decode pair."""
    dec_steps_per_req = max(1.0, eval_context.scenario.osl / eval_context.eff_al)
    b_one = max(1, min(math.ceil(c / max(1, dec.par.dp)), dec.b_hbm_max))
    sf_d_one = _eval_decode_floor(eval_context, dec.topo, dec.par, b_one)
    rate_d_one = float(c) / (dec_steps_per_req * max(1e-9, sf_d_one.serial_estimate_s))
    eff_max_inst = (
        eval_context.max_instances
        if eval_context.max_instances is not None
        else DEFAULT_MAX_INSTANCES
    )
    if eff_max_inst < MIN_DISAGG_INSTANCES:
        return []
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
    cache: dict[tuple[str, Parallel, float], _EffectivePrefill] = {}
    for topo_d in topos:
        if (
            eval_context.decode_topology is not None
            and topo_d.name != eval_context.decode_topology
        ):
            continue
        for par_d in _candidate_decode_parallels(
            eval_context.geom, topo_d, chip=eval_context.chip
        ):
            offload_b = (
                float(eval_context.scenario.kv_offload_gb)
                * UNIT_GIBI
                / float(max(1, par_d.tp * par_d.pp * par_d.dp * par_d.pcp))
            )
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
                    if (
                        eval_context.prefill_topology is not None
                        and pref.topo.name != eval_context.prefill_topology
                    ):
                        continue
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


def _finalize_plan_selection(
    scenario: TrafficScenario,
    candidates: list[ParetoPoint],
    warnings: list[str],
) -> tuple[
    list[ParetoPoint],
    list[ParetoPoint],
    tuple[ParetoPoint, ...],
    ParetoPoint | None,
]:
    """Filter ruled-out points, build frontier, pick winner, and add warnings."""
    not_ruled_out = [
        cand for cand in candidates if cand.slo_verdict != SLO_VERDICT_RULED_OUT
    ]
    pool_for_frontier = not_ruled_out if not_ruled_out else candidates
    if not candidates:
        warnings.append(NO_FEASIBLE_LAYOUT_WARNING)
    elif not not_ruled_out:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")
    frontier = _pareto_filter(pool_for_frontier)
    rec_pool = [p for p in frontier if p.slo_verdict == SLO_VERDICT_ESTIMATE_MEETS] or [
        p for p in frontier if p.slo_verdict != SLO_VERDICT_RULED_OUT
    ]
    recommended = (
        max(rec_pool, key=lambda p: p.token_throughput_per_chip) if rec_pool else None
    )
    if recommended is not None and recommended.slo_verdict == SLO_VERDICT_UNDECIDED:
        warnings.append(UNDECIDED_SLO_WARNING)
    if scenario.think_time_s <= 0.0:
        warnings.append(ZERO_THINK_TIME_WARNING)
    return (not_ruled_out, pool_for_frontier, frontier, recommended)


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


def _profile_scenario_warnings(
    geom: ModelGeometry,
    dtypes: Dtypes,
    ctx_full: int,
    spec_k: int,
) -> tuple[str, ...]:
    """Enforce profile-driven context, spec-decode, and KV dtype limits."""
    profile = resolve_geom_profile(geom)
    rules = profile.parallel
    model_label = geom.model_id or "model"
    msgs: list[str] = []
    if (
        rules.cap_context_at_indexer_budget
        and geom.qsa_budget > 0
        and ctx_full > geom.qsa_budget
    ):
        msgs.append(
            f"UNSUPPORTED_CONFIG: {model_label} max_model_len={ctx_full} exceeds "
            f"indexer_budget={geom.qsa_budget} on dense bring-up path."
        )
    if rules.max_spec_tokens is not None and spec_k > rules.max_spec_tokens:
        msgs.append(
            f"UNSUPPORTED_CONFIG: {model_label} does not support speculative decoding "
            f"(spec_tokens={spec_k})."
        )
    canonical_kv = canonicalize_kv_dtype(dtypes.kv_dtype)
    if canonical_kv not in profile.kv_dtypes or (
        dtypes.kv <= FP4_BYTES and "fp4" not in profile.kv_dtypes
    ):
        msgs.append(
            f"UNSUPPORTED_CONFIG: kv_dtype={dtypes.kv_dtype!r} ({dtypes.kv:g} B) "
            f"is not supported under runtime_profile={profile.name!r}."
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
    rules = resolve_geom_profile(geom).parallel
    if decode_axes.get("pp", 1) > 1 or decode_axes.get("pcp", 1) > 1:
        reason = "decode pools are enumerated with PP1 and PCP1"
    elif rules.max_dcp is not None and decode_axes.get("kv_split", 1) > rules.max_dcp:
        reason = (
            f"{geom.model_id or 'model'} profile {geom.runtime_profile!r} limits "
            f"DCP <= {rules.max_dcp}"
        )
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
    rules = resolve_geom_profile(geom).parallel
    eff_pre_spec = disagg_prefill_spec_k(spec.spec_k, rules)
    prefill_tp = prefill_axes.get("tp", 1)
    prefill_pp = prefill_axes.get("pp", 1)
    prefill_pcp = prefill_axes.get("pcp", 1)
    prefill_ep = prefill_axes.get("ep", 1)
    prefill_dcp = prefill_axes.get("kv_split", 1)
    if not _pcp_allowed(geom, prefill_pp, prefill_pcp, eff_pre_spec):
        return "PCP > 1 is not permitted with PP > 1, this profile, or spec_k > 1"
    n_full_attn = count_full_attn_layers(geom)
    if not _prefill_pp_allowed(
        geom, prefill_pp, n_full_attn, for_disagg=True, spec_k=eff_pre_spec
    ):
        return f"PP{prefill_pp} is not permitted for this model and scenario"
    prefill_found = any(
        parallel_candidate.tp == prefill_tp
        and parallel_candidate.pp == prefill_pp
        and parallel_candidate.pcp == prefill_pcp
        and parallel_candidate.ep == prefill_ep
        and parallel_candidate.kv_split == prefill_dcp
        for slice_topology in spec.topologies
        for parallel_candidate in _candidate_prefill_parallels(
            geom,
            slice_topology,
            for_disagg=True,
            spec_k=eff_pre_spec,
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
        rules = resolve_geom_profile(geom).parallel
        dcp_disallowed = (
            prefill_axes.get("kv_split", 1) > 1 and rules.prefill_dcp_mode == "none"
        )
        if prefill_axes.get("dp", 1) > 1 or dcp_disallowed:
            return (
                "prefill pools are enumerated with DP1 and DCP1"
                if rules.prefill_dcp_mode == "none"
                else "prefill pools are enumerated with DP1"
            )

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
    hbm_limit_b = max(
        0.0,
        float(opts.gpu_memory_utilization) * float(chip.hbm_bytes)
        - opts.hbm_reserve_gb * UNIT_GIGA,
    )
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
        hbm_reserve_gb=opts.hbm_reserve_gb,
        prefill_topology=opts.prefill_topology,
        decode_topology=opts.decode_topology,
        pd_ratio=opts.pd_ratio,
        activation_model=opts.activation_model,
    )

    profile_warnings = _profile_scenario_warnings(geom, dtypes, ctx_full, spec_k)
    if profile_warnings:
        return ServingPlan(
            model_layers=geom.num_layers,
            chip=chip.name,
            scenario=scenario,
            mode=opts.mode,
            recommended=None,
            frontier=(),
            warnings=profile_warnings,
            candidates=(),
            gpu_memory_utilization=opts.gpu_memory_utilization,
            collective_latency_s=opts.collective_latency_s,
            spec_k=spec_k,
            spec_tokens=spec_k,
            mtp_layers=geom.mtp_layers,
            hbm_reserve_gb=opts.hbm_reserve_gb,
            model_id=geom.model_id,
            has_moe=geom.num_experts > 0,
            is_hybrid=0 < count_full_attn_layers(geom) < geom.num_layers,
            budget=opts.max_chips if opts.max_chips is not None else 0,
            budget_source=opts.budget_source,
            max_instances=opts.max_instances,
            pins=pins,
            activation_model=opts.activation_model,
            runtime_profile=geom.runtime_profile,
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
    if dtypes.w_dense_source == "default" and dtypes.w_expert_source == "default":
        warnings.append(
            "No quantization_config block in config.json; weight and expert "
            f"dtypes defaulted to {dtypes.w_dense_dtype} "
            f"({dtypes.w_dense:.1f} B/param)."
        )
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

    not_ruled_out_candidates, pool_for_frontier, frontier, recommended = (
        _finalize_plan_selection(scenario, candidates, warnings)
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
        spec_tokens=spec_k,
        mtp_layers=geom.mtp_layers,
        hbm_reserve_gb=opts.hbm_reserve_gb,
        model_id=geom.model_id,
        has_moe=geom.num_experts > 0,
        is_hybrid=0 < count_full_attn_layers(geom) < geom.num_layers,
        search=SearchStats(
            enumerated=eval_context.enumerated,
            core_grid=_count_core_grid(eval_context, topos),
            hbm_fit=eval_context.hbm_fit,
            not_ruled_out=len(not_ruled_out_candidates),
            slo_met=sum(1 for candidate in candidates if candidate.slo_met),
            frontier=len(frontier),
            envelope_feasible=sum(
                1
                for candidate in candidates
                if _within_slo_bounds(
                    scenario,
                    candidate.envelope_ttft_ms,
                    candidate.envelope_tpot_ms,
                )
            ),
        ),
        budget=opts.max_chips if opts.max_chips is not None else 0,
        budget_source=opts.budget_source,
        max_instances=opts.max_instances,
        pins=pins,
        activation_model=opts.activation_model,
        runtime_profile=geom.runtime_profile,
    )
