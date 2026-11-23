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
"""Pipeline-parallel layer partitioning, stage KV sizing, and stage step floors."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, TypedDict, Unpack

from .constants import NON_FULL_ATTN_TOKENS
from .geometry import layer_ffn_kinds
from .memory import (
    HbmBreakdown,
    SeqKvLayout,
    _compute_w_head,
    hbm_breakdown_per_core_bytes,
    kv_bytes_per_core_parts,
)
from .modules import ModuleCost, pass_costs
from .step import (
    FloorOptions,
    StepFloor,
    StepShape,
    combine_modules,
    step_floor,
)

if TYPE_CHECKING:
    from .chips import ChipSpec
    from .geometry import ModelGeometry
    from .modules import Dtypes, Parallel

_UNIFORM_TIE_TOL: Final[float] = 1e-6


def _extract_pp_handoff_mods(
    raw_mods: tuple[ModuleCost, ...],
) -> list[ModuleCost]:
    """Extract pipeline P2P handoff collectives into a dedicated stage module."""
    p2p_colls = tuple(
        coll for mod in raw_mods for coll in mod.collectives if coll.kind == "p2p"
    )
    p2p_ucolls = tuple(
        (coll, count)
        for mod in raw_mods
        for coll, count in (mod.unique_collectives or ())
        if coll.kind == "p2p"
    )
    if not p2p_colls:
        return []
    return [
        ModuleCost(
            name="pp_p2p",
            hbm_bytes=0.0,
            fp8_flops=0.0,
            bf16_flops=0.0,
            collectives=p2p_colls,
            unique_collectives=p2p_ucolls or None,
        )
    ]


@dataclass(frozen=True, slots=True)
class PpStageOptions:
    """Options controlling stage feasibility, logit peak, and max layer count."""

    disagg: bool = False
    max_stage_layers: int | None = None
    logit_peak_b: float = 0.0
    require_full_attn_per_stage: bool = False


@dataclass(frozen=True, slots=True)
class _PpLayerPrefixes:
    """Prefix sums of per-layer memory footprints and full-attention counts."""

    weights: list[float]
    kv: list[float]
    state: list[float]
    full: list[int]
    embed_b: float
    lm_b: float


@dataclass(frozen=True, slots=True)
class _PpDpProblem:
    """DP state bounds and prefix sums for contiguous pipeline partitioning."""

    pp: int
    n_l: int
    max_stage_layers: int | None
    min_full: int
    logit_peak_b: float
    pref: _PpLayerPrefixes


@dataclass(frozen=True, slots=True)
class _DpTables:
    """Dynamic programming cost and backpointer tables."""

    cost: list[list[float]]
    prev: list[list[int]]


def count_full_attn_layers(geom: ModelGeometry) -> int:
    """Count standard full-attention layers excluding linear and sliding variants."""
    return sum(1 for tok in geom.layer_types if tok not in NON_FULL_ATTN_TOKENS)


def stage_slice_geometry(
    geom: ModelGeometry,
    start: int,
    end: int,
    *,
    include_mtp: bool = False,
) -> ModelGeometry:
    """Build a contiguous stage sub-geometry covering layers ``[start, end)``."""
    cnt = max(0, end - start)
    layer_types = geom.layer_types[start:end] if geom.layer_types else ()
    cr_slice = (
        geom.compress_ratios[start:end] if len(geom.compress_ratios) >= end else ()
    )
    ffn_all = layer_ffn_kinds(geom)
    ffn_slice = ffn_all[start:end] if len(ffn_all) >= end else ("dense",) * cnt
    n_moe = sum(1 for kind in ffn_slice if kind == "moe")
    n_dense = sum(1 for kind in ffn_slice if kind == "dense")
    idx_slice = geom.indexer_types[start:end] if len(geom.indexer_types) >= end else ()
    return replace(
        geom,
        num_layers=cnt,
        layer_types=layer_types,
        compress_ratios=cr_slice,
        indexer_types=idx_slice,
        ffn_kinds=ffn_slice,
        n_moe_layers=n_moe,
        n_dense_layers=n_dense,
        first_k_dense_replace=0 if n_moe > 0 else n_dense,
        mtp_layers=geom.mtp_layers if include_mtp else 0,
    )


def _pp_layer_prefixes(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par1: Parallel,
    layout: SeqKvLayout,
    *,
    chip: ChipSpec | None = None,
) -> _PpLayerPrefixes:
    """Compute prefix sums of weights, KV, state, and full-attention counts."""
    n_l = geom.num_layers
    tp = max(1, par1.tp)
    emb_single, lm_single = _compute_w_head(geom, dtypes, tp)
    w_head_single = emb_single + lm_single
    if geom.mtp_layers > 0:
        bd_with_mtp = hbm_breakdown_per_core_bytes(
            geom, dtypes, par1, SeqKvLayout(0, 0), chip=chip
        )
        bd_no_mtp = hbm_breakdown_per_core_bytes(
            replace(geom, mtp_layers=0), dtypes, par1, SeqKvLayout(0, 0), chip=chip
        )
        w_mtp_total = max(0.0, bd_with_mtp.weights - bd_no_mtp.weights)
    else:
        w_mtp_total = 0.0
    w_embed = emb_single
    w_lm = lm_single + w_mtp_total

    layer_w: list[float] = []
    layer_kv: list[float] = []
    layer_st: list[float] = []
    is_full: list[int] = []
    for idx in range(n_l):
        g_i = stage_slice_geometry(geom, idx, idx + 1, include_mtp=False)
        bd_i = hbm_breakdown_per_core_bytes(g_i, dtypes, par1, layout, chip=chip)
        layer_w.append(max(0.0, bd_i.weights - w_head_single))
        layer_kv.append(bd_i.kv)
        layer_st.append(bd_i.state)
        tok = geom.layer_types[idx] if idx < len(geom.layer_types) else "full_attention"
        is_full.append(0 if tok in NON_FULL_ATTN_TOKENS else 1)

    pref_w = [0.0] * (n_l + 1)
    pref_kv = [0.0] * (n_l + 1)
    pref_st = [0.0] * (n_l + 1)
    pref_full = [0] * (n_l + 1)
    for i in range(n_l):
        pref_w[i + 1] = pref_w[i] + layer_w[i]
        pref_kv[i + 1] = pref_kv[i] + layer_kv[i]
        pref_st[i + 1] = pref_st[i] + layer_st[i]
        pref_full[i + 1] = pref_full[i] + is_full[i]
    return _PpLayerPrefixes(
        weights=pref_w,
        kv=pref_kv,
        state=pref_st,
        full=pref_full,
        embed_b=w_embed,
        lm_b=w_lm,
    )


def _check_uniform_pp(
    unif: tuple[int, ...],
    prob: _PpDpProblem,
    target_cost: float,
) -> bool:
    """Check if uniform layer partition is legal and cost-competitive."""
    pp = len(unif)
    pref = prob.pref
    u_max = 0.0
    pos = 0
    for s, cnt in enumerate(unif):
        if prob.max_stage_layers is not None and cnt > prob.max_stage_layers:
            return False
        nxt = pos + cnt
        if prob.min_full > 0 and (pref.full[nxt] - pref.full[pos]) < prob.min_full:
            return False
        extra_stage_bytes = (pref.embed_b if s == 0 else 0.0) + (
            (pref.lm_b + prob.logit_peak_b) if s == pp - 1 else 0.0
        )
        u_max = max(
            u_max,
            (pref.weights[nxt] - pref.weights[pos])
            + (pref.kv[nxt] - pref.kv[pos])
            + (pref.state[nxt] - pref.state[pos])
            + extra_stage_bytes,
        )
        pos = nxt
    return u_max <= target_cost * (1.0 + _UNIFORM_TIE_TOL)


def _relax_stage_dp(
    stage_idx: int,
    extra: float,
    prob: _PpDpProblem,
    dp: _DpTables,
) -> None:
    """Relax DP transitions for one pipeline stage index."""
    inf = math.inf
    pref = prob.pref
    rem_stages = prob.pp - stage_idx
    for b in range(stage_idx, prob.n_l - rem_stages + 1):
        best_c = inf
        best_a = -1
        a_min = (
            max(stage_idx - 1, b - prob.max_stage_layers)
            if prob.max_stage_layers is not None
            else (stage_idx - 1)
        )
        for a in range(a_min, b):
            prev_cost = dp.cost[stage_idx - 1][a]
            if prev_cost == inf:
                continue
            if prob.min_full > 0 and (pref.full[b] - pref.full[a]) < prob.min_full:
                continue
            seg = (
                (pref.weights[b] - pref.weights[a])
                + (pref.kv[b] - pref.kv[a])
                + (pref.state[b] - pref.state[a])
                + extra
            )
            cand = max(prev_cost, seg)
            if cand < best_c:
                best_c = cand
                best_a = a
        dp.cost[stage_idx][b] = best_c
        dp.prev[stage_idx][b] = best_a


def _dp_solve_pp(prob: _PpDpProblem) -> tuple[tuple[int, ...], bool]:
    """Solve DP partitioning for contiguous pipeline stages."""
    if prob.pp > prob.n_l:
        return ((), False)
    inf = math.inf
    dp = _DpTables(
        cost=[[inf] * (prob.n_l + 1) for _ in range(prob.pp + 1)],
        prev=[[-1] * (prob.n_l + 1) for _ in range(prob.pp + 1)],
    )
    dp.cost[0][0] = 0.0

    for s in range(1, prob.pp + 1):
        extra = (prob.pref.embed_b if s == 1 else 0.0) + (
            (prob.pref.lm_b + prob.logit_peak_b) if s == prob.pp else 0.0
        )
        _relax_stage_dp(s, extra, prob, dp)

    if dp.prev[prob.pp][prob.n_l] < 0:
        return ((), False)

    rev: list[int] = []
    cur = prob.n_l
    for s in range(prob.pp, 0, -1):
        prv = dp.prev[s][cur]
        rev.append(cur - prv)
        cur = prv
    part = tuple(reversed(rev))

    if prob.n_l % prob.pp == 0:
        unif = (prob.n_l // prob.pp,) * prob.pp
        if _check_uniform_pp(unif, prob, dp.cost[prob.pp][prob.n_l]):
            part = unif
    return (part, True)


def _pp_part_breakdown(
    part: tuple[int, ...],
    pref: _PpLayerPrefixes,
    pp: int,
) -> tuple[HbmBreakdown, float]:
    """Calculate the peak stage HBM breakdown and last-stage footprint."""
    best_bd = HbmBreakdown(0.0, 0.0, 0.0, -1.0)
    last_tot = 0.0
    pos = 0
    for s, cnt in enumerate(part):
        nxt = pos + cnt
        ex_w = (pref.embed_b if s == 0 else 0.0) + (pref.lm_b if s == pp - 1 else 0.0)
        sw = (pref.weights[nxt] - pref.weights[pos]) + ex_w
        skv = pref.kv[nxt] - pref.kv[pos]
        sst = pref.state[nxt] - pref.state[pos]
        stot = sw + skv + sst
        if s == pp - 1:
            last_tot = stot
        if stot > best_bd.total:
            best_bd = HbmBreakdown(weights=sw, kv=skv, state=sst, total=stot)
        pos = nxt
    return (best_bd, last_tot)


class _PpBreakdownsKwargs(TypedDict, total=False):
    """Optional keyword arguments for ``pp_stage_breakdowns``."""

    options: PpStageOptions | None
    chip: ChipSpec | None


class _PpPartitionKwargs(TypedDict, total=False):
    """Optional keyword arguments for ``partition_pp_layers``."""

    options: PpStageOptions | None
    disagg: bool
    chip: ChipSpec | None


def pp_stage_breakdowns(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    workload: tuple[int, int] = (0, 1),
    **kwargs: Unpack[_PpBreakdownsKwargs],
) -> tuple[tuple[int, ...], HbmBreakdown, float, bool]:
    """Dynamic-programming contiguous PP layer partition and HBM breakdown."""
    context, seqs_per_dp = workload
    opts = kwargs.get("options") or PpStageOptions()
    chip = kwargs.get("chip")
    pp = max(1, parallel.pp)
    n_l = geom.num_layers
    layout = SeqKvLayout(seqs_per_dp=seqs_per_dp, context=context)
    if pp <= 1 or n_l <= 0:
        bd = hbm_breakdown_per_core_bytes(geom, dtypes, parallel, layout, chip=chip)
        return ((n_l,), bd, bd.total, True)
    if pp > n_l:
        return ((), HbmBreakdown(0.0, 0.0, 0.0, 0.0), 0.0, False)
    par1 = replace(parallel, pp=1)
    pref = _pp_layer_prefixes(geom, dtypes, par1, layout, chip=chip)
    req_full = (opts.disagg or opts.require_full_attn_per_stage) and (
        0 < pref.full[n_l] < n_l
    )
    prob = _PpDpProblem(
        pp=pp,
        n_l=n_l,
        max_stage_layers=opts.max_stage_layers,
        min_full=1 if req_full else 0,
        logit_peak_b=max(0.0, opts.logit_peak_b),
        pref=pref,
    )
    part, feasible = _dp_solve_pp(prob)
    if not feasible or not part:
        return ((), HbmBreakdown(0.0, 0.0, 0.0, 0.0), 0.0, False)
    best_bd, last_tot = _pp_part_breakdown(part, pref, pp)
    return (part, best_bd, last_tot, True)


def partition_pp_layers(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    workload: tuple[int, int] = (0, 1),
    **kwargs: Unpack[_PpPartitionKwargs],
) -> tuple[int, ...]:
    """Dynamic-programming contiguous PP layer partition minimizing max-stage HBM."""
    options = kwargs.get("options")
    disagg = bool(kwargs.get("disagg", False))
    chip = kwargs.get("chip")
    eff_opts = (
        options
        if options is not None
        else PpStageOptions(disagg=disagg, require_full_attn_per_stage=disagg)
    )
    part, _, _, feasible = pp_stage_breakdowns(
        geom,
        dtypes,
        parallel,
        workload=workload,
        options=eff_opts,
        chip=chip,
    )
    if not feasible or not part:
        msg = (
            f"Cannot partition {geom.num_layers} layers into {max(1, parallel.pp)} "
            "pipeline stages satisfying full-attention and non-empty stage constraints"
        )
        raise ValueError(msg)
    return part


def pp_stage_kv_parts(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    stage_workload: tuple[tuple[int, ...], int],
    *,
    chip: ChipSpec | None = None,
) -> tuple[float, float]:
    """Return max-KV-stage (linear_kv_per_token_bytes, fixed_window_kv_bytes)."""
    pp_part, ctx_tokens = stage_workload
    par1 = replace(parallel, pp=1)
    ctx_scale = float(max(1, ctx_tokens))
    best_pair = (0.0, 0.0)
    best_total = -1.0
    pos = 0
    for cnt in pp_part:
        nxt = pos + cnt
        g_stage = stage_slice_geometry(geom, pos, nxt, include_mtp=False)
        lin_b, fix_b = kv_bytes_per_core_parts(
            g_stage, dtypes, par1, ctx_tokens, chip=chip
        )
        stage_tot = lin_b * ctx_scale + fix_b
        if stage_tot > best_total:
            best_total = stage_tot
            best_pair = (lin_b, fix_b)
        pos = nxt
    return best_pair


def scale_pp_stage_floor(
    geom: ModelGeometry,
    chip: ChipSpec,
    parallel: Parallel,
    workload: tuple[StepShape, FloorOptions],
    pp_part: tuple[int, ...] | None,
) -> StepFloor:
    """Evaluate exact per-stage step floors across a contiguous PP layer partition."""
    shape, options = workload
    if pp_part is None or len(pp_part) <= 1 or parallel.pp <= 1:
        return step_floor(geom, chip, parallel, shape, options)
    if sum(pp_part) != geom.num_layers or any(cnt <= 0 for cnt in pp_part):
        msg = (
            f"Invalid pp_part {pp_part} for num_layers={geom.num_layers}, "
            f"pp={parallel.pp}"
        )
        raise ValueError(msg)

    fp8_ratio = chip.fp8_flops / chip.bf16_flops if chip.fp8_flops > 0.0 else 0.0
    raw_mods = pass_costs(
        geom,
        parallel,
        shape,
        options.dt,
        mxu_m_tile=chip.mxu_m_tile,
        mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
        mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
        activation_model=options.activation_model,
        fp8_to_bf16_ratio=fp8_ratio,
        chip=chip,
    )
    handoff_mods = _extract_pp_handoff_mods(raw_mods)
    par_stage = replace(parallel, pp=1)
    n_stages = len(pp_part)
    stage_floors: list[StepFloor] = []
    pos = 0
    for stage_idx, cnt in enumerate(pp_part):
        nxt = pos + cnt
        is_first = stage_idx == 0
        is_last = stage_idx == n_stages - 1
        g_stage = stage_slice_geometry(geom, pos, nxt, include_mtp=is_last)
        mods = pass_costs(
            g_stage,
            par_stage,
            shape,
            options.dt,
            mxu_m_tile=chip.mxu_m_tile,
            mxu_lhs_rows_bf16=chip.mxu_lhs_rows_bf16,
            mxu_lhs_rows_fp8=chip.mxu_lhs_rows_fp8,
            activation_model=options.activation_model,
            fp8_to_bf16_ratio=fp8_ratio,
            chip=chip,
        )
        filtered = [
            mod
            for mod in mods
            if (mod.name != "embed" or is_first) and (mod.name != "lm_head" or is_last)
        ]
        filtered.extend(handoff_mods)
        stage_floors.append(
            combine_modules(
                tuple(filtered),
                chip,
                options.collective_latency_s,
                topology=options.topology,
                parallel=parallel,
            )
        )
        pos = nxt

    peak_floor = max(
        stage_floors, key=lambda sf: (sf.hard_floor_s, sf.serial_estimate_s)
    )
    peak = max(stage_floors, key=lambda sf: (sf.serial_estimate_s, sf.hard_floor_s))
    return replace(
        peak,
        binding=peak_floor.binding,
        hbm_s=max(sf.hbm_s for sf in stage_floors),
        matmul_s=max(sf.matmul_s for sf in stage_floors),
        vector_s=max(sf.vector_s for sf in stage_floors),
        ici_s=max(sf.ici_s for sf in stage_floors),
        chain_s=max(sf.chain_s for sf in stage_floors),
        latency_assumed_s=max(sf.latency_assumed_s for sf in stage_floors),
        hbm_bound_s=max(sf.hbm_bound_s for sf in stage_floors),
        matmul_bound_s=max(sf.matmul_bound_s for sf in stage_floors),
        vector_bound_s=max(sf.vector_bound_s for sf in stage_floors),
        coll_bound_s=max(sf.coll_bound_s for sf in stage_floors),
        hard_floor_s=max(sf.hard_floor_s for sf in stage_floors),
        envelope_floor_s=max(sf.envelope_floor_s for sf in stage_floors),
        serial_estimate_s=max(sf.serial_estimate_s for sf in stage_floors),
    )
