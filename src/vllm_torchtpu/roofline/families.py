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
"""Layout-family grouping, concurrency skylines, and reference-layout ranking."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, cast

from .constants import DEFAULT_TOP_K, SLO_VERDICT_RULED_OUT
from .schema import (
    ParetoPoint,
    PointLaunchPayload,
    PoolMeshPlan,
    ServingPlan,
    SolvePayload,
    _point_struct_key,
    effective_slo_verdict,
    format_parallelism,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_REF_AXIS_MAP: Final[dict[str, str]] = {
    "TP": "tp",
    "DP": "dp",
    "EP": "ep",
    "PP": "pp",
    "PCP": "pcp",
    "DCP": "kv_split",
}
_EXPECTED_DISAGG_POOL_PARTS: Final[int] = 2
_INSTANCE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+)\s*[xX]\s+(.+)$")
_TOPOLOGY_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*\d+[xX]\d+[xX]\d+\s+(.+)$"
)

__all__ = [
    "attach_reference_report",
    "build_bound_families_by_concurrency",
    "pareto_skyline_by_concurrency",
    "parse_layout",
    "reference_layout_report",
    "strict_rank",
]


def strict_rank(
    value: float,
    values: Sequence[float],
    *,
    lower_is_better: bool,
) -> int:
    """Return 1 plus the number of entries strictly better than ``value``."""
    if lower_is_better:
        return 1 + sum(1 for other in values if other < value)
    return 1 + sum(1 for other in values if other > value)


@dataclass(frozen=True, slots=True)
class _SkylineFillConfig:
    """Slot limit and TPOT lower-bound selector for skyline filling."""

    max_slots: int
    use_envelope: bool


def _skyline_tpot_floor(point: ParetoPoint, *, use_envelope: bool) -> float:
    """Return active TPOT lower bound used consistently across skyline orderings."""
    return point.envelope_tpot_ms if use_envelope else point.bound_tpot_ms


def _pool_mesh_family(pool: PoolMeshPlan | None) -> tuple[object, ...] | None:
    """Return mesh family key ignoring KV split and identical-CLI MoE dispatch."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
    )


def _pair_family_key(
    point: ParetoPoint,
) -> tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]:
    """Return full (mode, prefill_mesh_family, decode_mesh_family) key."""
    return (
        point.mode,
        _pool_mesh_family(point.prefill_pool),
        _pool_mesh_family(point.decode_pool),
    )


def _decode_group_key(
    point: ParetoPoint,
) -> tuple[
    str,
    int,
    int,
    tuple[object, ...] | None,
    tuple[object, ...] | None,
]:
    """Return diversity group key preserving prefill, decode, and xP:yD families."""
    prefill_instances = (
        point.prefill_pool.instances if point.prefill_pool is not None else 0
    )
    return (
        point.mode,
        prefill_instances,
        point.decode_pool.instances,
        _pool_mesh_family(point.prefill_pool),
        _pool_mesh_family(point.decode_pool),
    )


def _skyline_preference(
    point: ParetoPoint, *, use_envelope: bool = False
) -> tuple[bool, bool, float, float, float]:
    """Order candidate variants within one mesh family by verdict and 3 objectives."""
    return (
        point.slo_met,
        effective_slo_verdict(point) != SLO_VERDICT_RULED_OUT,
        -_skyline_tpot_floor(point, use_envelope=use_envelope),
        point.token_throughput_per_chip,
        point.est_interactivity_tok_s,
    )


def _fill_skyline_slots(
    selected: list[ParetoPoint],
    chosen_keys: set[tuple[object, ...]],
    candidates: Sequence[ParetoPoint],
    key_fn: Callable[[ParetoPoint], tuple[object, ...]],
    config: _SkylineFillConfig,
) -> None:
    """Fill remaining skyline slots up to max_slots without duplicating keys."""
    ordered = sorted(
        (pt for pt in candidates if key_fn(pt) not in chosen_keys),
        key=lambda pt: (
            not pt.slo_met,
            _skyline_tpot_floor(pt, use_envelope=config.use_envelope),
            -pt.token_throughput_per_chip,
            -pt.est_interactivity_tok_s,
        ),
    )
    for point in ordered:
        if len(selected) >= config.max_slots:
            break
        selected.append(point)
        chosen_keys.add(key_fn(point))


def _select_concurrency_skyline(
    points: Sequence[ParetoPoint],
    k: int,
    *,
    use_envelope: bool = False,
) -> tuple[ParetoPoint, ...]:
    """Select up to k 3-objective Pareto skyline points at one concurrency."""
    best_by_pair: dict[tuple[object, ...], ParetoPoint] = {}
    for point in points:
        pair_key = _pair_family_key(point)
        current = best_by_pair.get(pair_key)
        if current is None or _skyline_preference(
            point, use_envelope=use_envelope
        ) > _skyline_preference(current, use_envelope=use_envelope):
            best_by_pair[pair_key] = point

    best_by_group: dict[tuple[object, ...], ParetoPoint] = {}
    for point in best_by_pair.values():
        group_key = _decode_group_key(point)
        current = best_by_group.get(group_key)
        if current is None or _skyline_preference(
            point, use_envelope=use_envelope
        ) > _skyline_preference(current, use_envelope=use_envelope):
            best_by_group[group_key] = point

    group_pool = list(best_by_group.values())
    anchors = (
        min(
            group_pool,
            key=lambda pt: (
                not pt.slo_met,
                _skyline_tpot_floor(pt, use_envelope=use_envelope),
                -pt.token_throughput_per_chip,
                -pt.est_interactivity_tok_s,
            ),
        ),
        max(
            group_pool,
            key=lambda pt: (
                pt.slo_met,
                pt.token_throughput_per_chip,
                -_skyline_tpot_floor(pt, use_envelope=use_envelope),
                pt.est_interactivity_tok_s,
            ),
        ),
        max(
            group_pool,
            key=lambda pt: (
                pt.slo_met,
                pt.est_interactivity_tok_s,
                -_skyline_tpot_floor(pt, use_envelope=use_envelope),
                pt.token_throughput_per_chip,
            ),
        ),
    )

    selected: list[ParetoPoint] = []
    chosen_groups: set[tuple[object, ...]] = set()
    for anchor in anchors:
        if len(selected) >= k:
            break
        group_key = _decode_group_key(anchor)
        if group_key not in chosen_groups:
            selected.append(anchor)
            chosen_groups.add(group_key)

    fill_cfg = _SkylineFillConfig(max_slots=k, use_envelope=use_envelope)
    _fill_skyline_slots(
        selected, chosen_groups, group_pool, _decode_group_key, fill_cfg
    )
    if len(selected) < k:
        chosen_pairs: set[tuple[object, ...]] = {
            _pair_family_key(pt) for pt in selected
        }
        _fill_skyline_slots(
            selected,
            chosen_pairs,
            list(best_by_pair.values()),
            _pair_family_key,
            fill_cfg,
        )

    selected.sort(
        key=lambda pt: (
            -pt.token_throughput_per_chip,
            _skyline_tpot_floor(pt, use_envelope=use_envelope),
            -pt.est_interactivity_tok_s,
        )
    )
    return tuple(selected)


def pareto_skyline_by_concurrency(
    plan: ServingPlan,
    k: int = DEFAULT_TOP_K,
) -> dict[int, tuple[ParetoPoint, ...]]:
    """Select up to k 3-objective heuristic skyline layouts per concurrency."""
    if k <= 0 or not plan.candidates:
        return {}
    by_concurrency: dict[int, list[ParetoPoint]] = {}
    for point in plan.candidates:
        by_concurrency.setdefault(point.concurrency, []).append(point)
    result: dict[int, tuple[ParetoPoint, ...]] = {}
    for concurrency in sorted(by_concurrency):
        conc_points = by_concurrency[concurrency]
        not_ruled_out = [
            pt
            for pt in conc_points
            if effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT
        ]
        pool = not_ruled_out if not_ruled_out else conc_points
        result[concurrency] = _select_concurrency_skyline(
            pool,
            k,
            use_envelope=plan.activation_model.prunes_on_envelope,
        )
    return result


def _bound_family_key(
    point: ParetoPoint,
) -> tuple[str, tuple[object, ...] | None, tuple[object, ...] | None]:
    """Return concurrency-grouping family key preserving disagg chip counts."""
    return (
        point.mode,
        _pool_mesh_family(point.prefill_pool),
        _pool_mesh_family(point.decode_pool),
    )


def _pool_family_label(pool: PoolMeshPlan) -> str:
    """Format mesh topology and non-DCP parallelism label for one pool."""
    topo = f"{pool.instances}x {pool.topology}" if pool.instances > 1 else pool.topology
    pool_map = dict(asdict(pool))
    pool_map["kv_split"] = 1
    return f"{topo} {format_parallelism(pool_map)}"


def _bound_family_label(point: ParetoPoint) -> tuple[str, ...]:
    """Build human-readable multi-part family label tuple for bound tables."""
    d_label = _pool_family_label(point.decode_pool)
    if point.prefill_pool is None:
        return (point.mode, d_label)
    p_label = _pool_family_label(point.prefill_pool)
    return (point.mode, f"P:{p_label}", f"D:{d_label}")


@dataclass(frozen=True, slots=True)
class _BoundFamilySummary:
    """Intermediate sortable family summary for one concurrency level."""

    sort_key: tuple[bool, float, float, float]
    payload: dict[str, object]


def _summarize_bound_family(
    family_label: tuple[str, ...],
    group: Sequence[ParetoPoint],
    launch_fn: Callable[[ParetoPoint], PointLaunchPayload] | None,
    *,
    use_envelope: bool,
) -> _BoundFamilySummary:
    """Aggregate one structural family at a concurrency into bound metadata."""
    active = [pt for pt in group if effective_slo_verdict(pt) != SLO_VERDICT_RULED_OUT]
    any_not_ruled_out = bool(active)
    pool = active if any_not_ruled_out else list(group)
    best_pt = min(
        pool,
        key=lambda pt: (
            pt.envelope_tpot_ms if use_envelope else pt.bound_tpot_ms,
            pt.bound_tpot_ms,
            pt.bound_ttft_ms,
        ),
    )
    sorted_members = sorted(
        pool,
        key=lambda pt: (
            pt.envelope_tpot_ms if use_envelope else pt.bound_tpot_ms,
            pt.bound_tpot_ms,
            -pt.token_throughput_per_chip,
        ),
    )
    min_bound_tpot = min(pt.bound_tpot_ms for pt in pool)
    min_bound_ttft = min(pt.bound_ttft_ms for pt in pool)
    min_env_tpot = min(pt.envelope_tpot_ms for pt in pool)
    min_env_ttft = min(pt.envelope_ttft_ms for pt in pool)
    active_tpot = min_env_tpot if use_envelope else min_bound_tpot
    members: list[dict[str, object]] = []
    for pt in sorted_members:
        entry: dict[str, object] = dict(asdict(pt))
        if launch_fn is not None:
            entry["launch"] = launch_fn(pt)
        members.append(entry)
    payload: dict[str, object] = {
        "rank": 0,
        "family": family_label,
        "min_bound_tpot_ms": min_bound_tpot,
        "min_bound_ttft_ms": min_bound_ttft,
        "min_envelope_tpot_ms": min_env_tpot,
        "min_envelope_ttft_ms": min_env_ttft,
        "binding": best_pt.binding,
        "any_not_ruled_out": any_not_ruled_out,
        "member_count": len(members),
        "members": members,
    }
    return _BoundFamilySummary(
        sort_key=(not any_not_ruled_out, active_tpot, min_bound_tpot, min_bound_ttft),
        payload=payload,
    )


def build_bound_families_by_concurrency(
    plan: ServingPlan,
    launch_fn: Callable[[ParetoPoint], PointLaunchPayload] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Group and rank layout families by active lower bound per concurrency."""
    if not plan.candidates:
        return {}
    use_envelope = plan.activation_model.prunes_on_envelope
    by_concurrency: dict[int, list[ParetoPoint]] = {}
    for point in plan.candidates:
        by_concurrency.setdefault(point.concurrency, []).append(point)
    result: dict[str, list[dict[str, object]]] = {}
    for concurrency in sorted(by_concurrency):
        grouped: dict[tuple[object, ...], list[ParetoPoint]] = {}
        labels: dict[tuple[object, ...], tuple[str, ...]] = {}
        for point in by_concurrency[concurrency]:
            key = _bound_family_key(point)
            grouped.setdefault(key, []).append(point)
            labels.setdefault(key, _bound_family_label(point))
        summaries = [
            _summarize_bound_family(
                labels[key],
                group,
                launch_fn,
                use_envelope=use_envelope,
            )
            for key, group in grouped.items()
        ]
        summaries.sort(key=lambda item: item.sort_key)
        ranked_entries: list[dict[str, object]] = []
        for rank, item in enumerate(summaries, start=1):
            entry = dict(item.payload)
            entry["rank"] = rank
            ranked_entries.append(entry)
        result[str(concurrency)] = ranked_entries
    return result


def _extract_pool_prefixes(pool_text: str) -> tuple[str, int | None]:
    """Strip optional leading Nx instances and topology token from one pool string."""
    remainder = pool_text.strip()
    if not remainder:
        msg = f"empty layout string: {pool_text!r}"
        raise ValueError(msg)
    instances: int | None = None
    inst_match = _INSTANCE_PREFIX_RE.match(remainder)
    if inst_match is not None:
        instances = int(inst_match.group(1))
        if instances < 1:
            msg = f"instance count must be at least 1: {pool_text!r}"
            raise ValueError(msg)
        remainder = inst_match.group(2).strip()
    topo_match = _TOPOLOGY_PREFIX_RE.match(remainder)
    if topo_match is not None:
        remainder = topo_match.group(1).strip()
    return remainder, instances


def _parse_single_pool_layout(pool_text: str) -> dict[str, int]:
    """Parse one pool layout string (e.g. '4x TP8 x EP8') into an axis dictionary."""
    axes_text, instances = _extract_pool_prefixes(pool_text)
    raw_tokens = re.split(r"\s*[xX]\s*", axes_text)
    seen_axes: set[str] = set()
    axis_values: dict[str, int] = {
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "pp": 1,
        "pcp": 1,
        "kv_split": 1,
    }
    if instances is not None:
        axis_values["instances"] = instances
    for raw_token in raw_tokens:
        token = raw_token.strip()
        if not token:
            msg = f"empty axis token in layout: {pool_text!r}"
            raise ValueError(msg)
        match = re.fullmatch(r"([a-zA-Z]+)(\d+)", token)
        if not match:
            msg = f"malformed parallelism axis token: {token!r}"
            raise ValueError(msg)
        axis_name = match.group(1).upper()
        if axis_name not in _REF_AXIS_MAP:
            msg = f"unknown parallelism axis in layout: {token!r}"
            raise ValueError(msg)
        if axis_name in seen_axes:
            msg = f"axis given twice in layout: {token!r}"
            raise ValueError(msg)
        seen_axes.add(axis_name)
        axis_degree = int(match.group(2))
        if axis_degree < 1:
            msg = f"parallelism degree must be at least 1: {token!r}"
            raise ValueError(msg)
        axis_values[_REF_AXIS_MAP[axis_name]] = axis_degree
    return axis_values


def _parse_disagg_layout_parts(
    layout_text: str, pool_parts: Sequence[str]
) -> tuple[str, str]:
    """Extract prefill and decode pool strings from disaggregated layout parts."""
    if len(pool_parts) != _EXPECTED_DISAGG_POOL_PARTS:
        msg = (
            f"disaggregated layout requires two parts separated by '/': {layout_text!r}"
        )
        raise ValueError(msg)
    prefill_part: str | None = None
    decode_part: str | None = None
    for part in pool_parts:
        upper_part = part.upper()
        if upper_part.startswith("P:"):
            if prefill_part is not None:
                msg = f"multiple prefill parts in layout: {part!r}"
                raise ValueError(msg)
            prefill_part = part[2:].strip()
        elif upper_part.startswith("D:"):
            if decode_part is not None:
                msg = f"multiple decode parts in layout: {part!r}"
                raise ValueError(msg)
            decode_part = part[2:].strip()
        else:
            msg = (
                f"unrecognized pool prefix in layout (expected 'P:' or 'D:'): {part!r}"
            )
            raise ValueError(msg)
    if prefill_part is None:
        msg = f"layout needs a prefill part: {layout_text!r}"
        raise ValueError(msg)
    if decode_part is None:
        msg = f"layout needs a decode part: {layout_text!r}"
        raise ValueError(msg)
    return prefill_part, decode_part


def _parse_disagg_layout(
    layout_text: str, stripped_layout: str
) -> dict[str, dict[str, int]]:
    """Parse disaggregated layout text with prefill and decode pools."""
    pool_parts = [part.strip() for part in stripped_layout.split("/")]
    prefill_part, decode_part = _parse_disagg_layout_parts(layout_text, pool_parts)
    return {
        "prefill": _parse_single_pool_layout(prefill_part),
        "decode": _parse_single_pool_layout(decode_part),
    }


def _parse_single_layout(
    layout_text: str, stripped_layout: str
) -> dict[str, dict[str, int]]:
    """Parse single-pool layout text into decode axes."""
    decode_text = stripped_layout
    if decode_text.upper().startswith("D:"):
        decode_text = decode_text[2:].strip()
    elif decode_text.upper().startswith("P:"):
        msg = f"layout needs a decode part: {layout_text!r}"
        raise ValueError(msg)
    return {
        "decode": _parse_single_pool_layout(decode_text),
    }


def parse_layout(layout_text: str) -> dict[str, dict[str, int]]:
    """Parse reference layout specification into decode and optional prefill axes."""
    stripped_layout = layout_text.strip()
    if not stripped_layout:
        msg = f"empty layout text: {layout_text!r}"
        raise ValueError(msg)
    if "/" in stripped_layout:
        return _parse_disagg_layout(layout_text, stripped_layout)
    return _parse_single_layout(layout_text, stripped_layout)


def _pool_matches_axes(pool: PoolMeshPlan, axes: Mapping[str, int]) -> bool:
    """Check if a pool mesh plan matches exact reference axis values."""
    if "instances" in axes and pool.instances != axes["instances"]:
        return False
    return (
        pool.tp == axes["tp"]
        and pool.dp == axes["dp"]
        and pool.ep == axes["ep"]
        and pool.pp == axes["pp"]
        and pool.pcp == axes["pcp"]
        and pool.kv_split == axes["kv_split"]
    )


def _candidate_matches_layout(
    candidate: ParetoPoint, parsed_layout: Mapping[str, Mapping[str, int]]
) -> bool:
    """Check if a candidate matches reference decode and prefill pools."""
    if not _pool_matches_axes(candidate.decode_pool, parsed_layout["decode"]):
        return False
    if "prefill" in parsed_layout:
        if candidate.prefill_pool is None:
            return False
        return _pool_matches_axes(candidate.prefill_pool, parsed_layout["prefill"])
    return candidate.prefill_pool is None


def _best_values_by_non_matching_layout(
    candidates: Sequence[ParetoPoint],
    parsed_layout: Mapping[str, Mapping[str, int]],
    value_getter: Callable[[ParetoPoint], float],
    *,
    lower_is_better: bool,
) -> list[float]:
    """Return best metric value per distinct layout for non-matching candidates."""
    best_by_layout_key: dict[
        tuple[str, tuple[object, ...] | None, tuple[object, ...] | None],
        float,
    ] = {}
    for candidate in candidates:
        if not _candidate_matches_layout(candidate, parsed_layout):
            layout_key = _point_struct_key(candidate)
            candidate_value = value_getter(candidate)
            if (
                layout_key not in best_by_layout_key
                or (
                    lower_is_better and candidate_value < best_by_layout_key[layout_key]
                )
                or (
                    not lower_is_better
                    and candidate_value > best_by_layout_key[layout_key]
                )
            ):
                best_by_layout_key[layout_key] = candidate_value
    return list(best_by_layout_key.values())


def _format_parsed_pool_layout(axes: Mapping[str, int]) -> str:
    """Format parsed pool axes while preserving any explicit Nx prefix."""
    base_layout = format_parallelism(axes)
    if "instances" in axes:
        return f"{axes['instances']}x {base_layout}"
    return base_layout


def _format_candidate_layout(point: ParetoPoint) -> str:
    """Format a candidate's topology, instances, and parallelism axes as text."""
    decode_prefix = (
        f"{point.decode_pool.instances}x " if point.decode_pool.instances > 1 else ""
    )
    decode_text = (
        f"{decode_prefix}{point.decode_pool.topology} "
        f"{format_parallelism(asdict(point.decode_pool))}"
    )
    if point.prefill_pool is None:
        return decode_text
    prefill_prefix = (
        f"{point.prefill_pool.instances}x " if point.prefill_pool.instances > 1 else ""
    )
    prefill_text = (
        f"{prefill_prefix}{point.prefill_pool.topology} "
        f"{format_parallelism(asdict(point.prefill_pool))}"
    )
    return f"P: {prefill_text} / D: {decode_text}"


def _find_skyline_family_peer(
    plan: ServingPlan,
    target_point: ParetoPoint | None,
) -> tuple[bool, str | None]:
    """Return whether target_point's mesh family has a Top-K skyline peer."""
    if target_point is None:
        return False, None
    skyline_by_conc = pareto_skyline_by_concurrency(plan, k=DEFAULT_TOP_K)
    target_family = _pair_family_key(target_point)
    for peer in skyline_by_conc.get(target_point.concurrency, ()):
        if _pair_family_key(peer) == target_family:
            return True, _format_candidate_layout(peer)
    return False, None


def _format_ruled_out_reason(candidate: ParetoPoint, plan: ServingPlan) -> str:
    """Format physical bound vs SLO margin explaining why candidate was ruled out."""
    use_envelope = plan.activation_model.prunes_on_envelope
    active_tpot_ms = (
        candidate.envelope_tpot_ms if use_envelope else candidate.bound_tpot_ms
    )
    active_ttft_ms = (
        candidate.envelope_ttft_ms if use_envelope else candidate.bound_ttft_ms
    )
    violations: list[str] = []
    if (
        plan.scenario.tpot_slo_ms is not None
        and active_tpot_ms > plan.scenario.tpot_slo_ms
    ):
        margin_ms = active_tpot_ms - plan.scenario.tpot_slo_ms
        violations.append(
            f"TPOT lower bound {active_tpot_ms:.3f} ms exceeds TPOT SLO "
            f"{plan.scenario.tpot_slo_ms:.3f} ms (+{margin_ms:.3f} ms)"
        )
    if (
        plan.scenario.ttft_slo_ms is not None
        and active_ttft_ms > plan.scenario.ttft_slo_ms
    ):
        margin_ms = active_ttft_ms - plan.scenario.ttft_slo_ms
        violations.append(
            f"TTFT lower bound {active_ttft_ms:.2f} ms exceeds TTFT SLO "
            f"{plan.scenario.ttft_slo_ms:.2f} ms (+{margin_ms:.2f} ms)"
        )
    if violations:
        return "; ".join(violations)
    return (
        "ruled out by closed-loop capacity ceiling "
        f"(TPOT bound {active_tpot_ms:.3f} ms, TTFT bound {active_ttft_ms:.2f} ms)"
    )


def _point_diagnostics(point: ParetoPoint | None) -> dict[str, object]:
    """Extract uniform scalar bound and memory diagnostic fields for one candidate."""
    if point is None:
        return {
            "slo_verdict": None,
            "bound_tpot_ms": None,
            "bound_ttft_ms": None,
            "floor_throughput_per_chip": None,
            "hbm_used_gb": None,
            "min_decode_seqs": None,
        }
    hbm_used_gb = point.decode_pool.hbm_used_gb
    if point.prefill_pool is not None:
        hbm_used_gb = max(hbm_used_gb, point.prefill_pool.hbm_used_gb)
    return {
        "slo_verdict": effective_slo_verdict(point),
        "bound_tpot_ms": point.bound_tpot_ms,
        "bound_ttft_ms": point.bound_ttft_ms,
        "floor_throughput_per_chip": point.floor_throughput_per_chip,
        "hbm_used_gb": hbm_used_gb,
        "min_decode_seqs": point.min_decode_seqs,
    }


def reference_layout_report(
    plan: ServingPlan,
    top_points: Sequence[ParetoPoint],
    layout_text: str,
    rejected_by: str,
) -> dict[str, object]:
    """Compile reference layout ranking report against same-concurrency peers."""
    parsed_layout = parse_layout(layout_text)
    if "prefill" in parsed_layout:
        normalized_layout = (
            f"P: {_format_parsed_pool_layout(parsed_layout['prefill'])} / "
            f"D: {_format_parsed_pool_layout(parsed_layout['decode'])}"
        )
    else:
        normalized_layout = _format_parsed_pool_layout(parsed_layout["decode"])

    matching_candidates = [
        candidate
        for candidate in plan.candidates
        if _candidate_matches_layout(candidate, parsed_layout)
    ]
    found = len(matching_candidates) > 0
    ruled_out = False
    diag_point: ParetoPoint | None = None

    if found:
        best_candidate = max(
            matching_candidates,
            key=lambda candidate: candidate.token_throughput_per_chip,
        )
        diag_point = best_candidate
        peer_candidates = [
            candidate
            for candidate in plan.candidates
            if candidate.concurrency == best_candidate.concurrency
        ]
        num_candidates = len(
            {_point_struct_key(candidate) for candidate in peer_candidates}
        )
        reference_bound = best_candidate.bound_tpot_ms
        other_bounds = _best_values_by_non_matching_layout(
            peer_candidates,
            parsed_layout,
            lambda candidate: candidate.bound_tpot_ms,
            lower_is_better=True,
        )
        rank_by_tpot_bound = strict_rank(
            reference_bound, other_bounds, lower_is_better=True
        )

        reference_throughput = best_candidate.token_throughput_per_chip
        other_throughputs = _best_values_by_non_matching_layout(
            peer_candidates,
            parsed_layout,
            lambda candidate: candidate.token_throughput_per_chip,
            lower_is_better=False,
        )
        rank_by_estimated_throughput = strict_rank(
            reference_throughput, other_throughputs, lower_is_better=False
        )

        in_top_k = any(
            _candidate_matches_layout(candidate, parsed_layout)
            for candidate in top_points
        )
        candidate_point_dict: Mapping[str, object] | None = asdict(best_candidate)
        final_rejection_reason = ""
    else:
        num_candidates = len(
            {_point_struct_key(candidate) for candidate in plan.candidates}
        )
        rank_by_tpot_bound = None
        rank_by_estimated_throughput = None
        in_top_k = False
        candidate_point_dict = None
        ruled_matches = [
            cand
            for cand in plan.ruled_out_candidates
            if _candidate_matches_layout(cand, parsed_layout)
        ]
        if ruled_matches:
            ruled_out = True
            diag_point = min(
                ruled_matches,
                key=lambda cand: (
                    cand.bound_tpot_ms,
                    cand.bound_ttft_ms,
                    -cand.token_throughput_per_chip,
                ),
            )
            final_rejection_reason = _format_ruled_out_reason(diag_point, plan)
        else:
            final_rejection_reason = rejected_by

    family_in_skyline_k, skyline_family_peer = _find_skyline_family_peer(
        plan, diag_point
    )
    return {
        "layout": normalized_layout,
        "found": found,
        "ruled_out": ruled_out,
        "candidates": num_candidates,
        "rank_by_tpot_bound": rank_by_tpot_bound,
        "rank_by_estimated_throughput": rank_by_estimated_throughput,
        "in_top_k": in_top_k,
        "family_in_skyline_k": family_in_skyline_k,
        "skyline_family_peer": skyline_family_peer,
        "point": candidate_point_dict,
        "rejected_by": final_rejection_reason,
        "rejection_reason": final_rejection_reason,
        **_point_diagnostics(diag_point),
    }


def attach_reference_report(
    payload: SolvePayload,
    plan: ServingPlan,
    reference: Mapping[str, object],
    *,
    launch_fn: Callable[[ParetoPoint], PointLaunchPayload] | None = None,
) -> SolvePayload:
    """Attach a reference-layout report with resolved launch args to a payload."""
    ref_dict = dict(reference)
    ref_launch: PointLaunchPayload | None = None
    if ref_dict.get("found") and launch_fn is not None:
        parsed_ref = parse_layout(str(ref_dict["layout"]))
        matching = [
            candidate
            for candidate in plan.candidates
            if _candidate_matches_layout(candidate, parsed_ref)
        ]
        if matching:
            best_candidate = max(
                matching,
                key=lambda candidate: candidate.token_throughput_per_chip,
            )
            ref_launch = launch_fn(best_candidate)
    ref_dict["launch"] = ref_launch
    updated = dict(payload)
    updated["reference"] = ref_dict
    return cast("SolvePayload", updated)
