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
"""Per-core HBM weight, KV cache, recurrent state, and SMEM page-table sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, TypedDict, Unpack

from .chips import CHIPS, ChipSpec
from .constants import (
    BYTES_PER_PAGE_ENTRY,
    MAX_CTX_TOKENS_FOR_SMALL_PAGE,
    MLA_PAGE_SIZE_TOKENS,
    RPA_LONG_CTX_PAGE_SIZE_TOKENS,
    RPA_MAX_PAGE_TOKENS,
    RPA_MIN_PAGE_TOKENS,
    STEP_FLOOR_CACHE_MAXSIZE,
)
from .geometry import (
    LayerCounts,
    aligned_moe_tp_shard_width,
    count_layer_types,
)
from .modules import (
    Dtypes,
    Parallel,
    dsa_indexer_layer_weight_elems,
    kv_bytes_per_token_per_layer,
)

if TYPE_CHECKING:
    from .geometry import ModelGeometry

_HC_LOWRANK_DIRECTIONS: int = 2


@dataclass(frozen=True, slots=True)
class ParamBreakdown:
    """Unsharded model parameter count breakdown matching HBM weight terms."""

    attention: float
    linear_attention: float
    dense_ffn: float
    moe_routed_ffn: float
    moe_shared_ffn: float
    router: float
    embedding: float
    lm_head: float
    mtp: float
    hc: float
    total_params: float
    active_params_per_token: float


@dataclass(frozen=True, slots=True)
class HbmBreakdown:
    """Per-TensorCore HBM memory breakdown across weights, KV, and state."""

    weights: float
    kv: float
    state: float
    total: float


@dataclass(frozen=True, slots=True)
class SeqKvLayout:
    """Sequence count, context length, and KV packing layout for HBM sizing."""

    seqs_per_dp: int
    context: int = 0
    physical_kv_layout: bool = True


@dataclass(frozen=True, slots=True)
class KvFitInput:
    """Memory capacity constraints used to compute clamped prefix hit rate."""

    b_ws: int
    ctx_full: int
    hbm_limit_b: float
    w_base_b: float
    b_kv_phys_core: float
    offload_b_per_core: float
    prefix_hit_rate: float
    kv_working_set_turns: float
    fixed_kv_per_seq_bytes: float = 0.0


def pp_stage_layer_scale(num_layers: int, pp: int) -> float:
    """Return heaviest pipeline stage layer fraction ceil(L / pp) / L."""
    if pp <= 1 or num_layers <= 0:
        return 1.0
    return float(math.ceil(num_layers / pp)) / float(num_layers)


def _compute_mla_w_attn(
    geom: ModelGeometry,
    dt: Dtypes,
    tp: int,
    pp: int,
    counts: LayerCounts,
) -> float:
    """Compute per-core weight bytes for MLA and DSA attention layers."""
    if geom.kv_lora_rank <= 0:
        msg = "MLA layer requires config field 'kv_lora_rank'"
        raise ValueError(msg)
    if geom.qk_nope_head_dim <= 0:
        msg = "MLA layer requires config field 'qk_nope_head_dim'"
        raise ValueError(msg)
    if geom.qk_rope_head_dim < 0:
        msg = "MLA layer requires config field 'qk_rope_head_dim'"
        raise ValueError(msg)
    kv_lora = geom.kv_lora_rank
    nope_d = geom.qk_nope_head_dim
    rope_d = geom.qk_rope_head_dim
    v_d = geom.v_head_dim or nope_d
    stage_scale = pp_stage_layer_scale(geom.num_layers, pp)
    if geom.q_lora_rank > 0:
        q_term_hidden = float(geom.hidden * geom.q_lora_rank) * stage_scale
        q_term_proj = (
            float(geom.q_lora_rank * geom.num_heads * (nope_d + rope_d))
            * stage_scale
            / tp
        )
    else:
        q_term_hidden = 0.0
        q_term_proj = (
            float(geom.hidden * geom.num_heads * (nope_d + rope_d)) * stage_scale / tp
        )
    w_attn = (
        float(counts.mla)
        * (
            q_term_hidden
            + float(geom.hidden * (kv_lora + rope_d)) * stage_scale
            + q_term_proj
            + float(
                kv_lora * geom.num_heads * (nope_d + v_d)
                + geom.num_heads * v_d * geom.hidden
            )
            * stage_scale
            / tp
        )
        * dt.w_dense
    )
    if counts.dsa > 0:
        n_full_idx = (
            geom.indexer_types.count("full") if geom.indexer_types else counts.dsa
        )
        w_attn += (
            float(n_full_idx)
            * dsa_indexer_layer_weight_elems(geom)
            * dt.w_dense
            * stage_scale
        )
    return w_attn


def compute_w_attn(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    counts: LayerCounts,
) -> float:
    """Compute per-core attention projection and QKV weight bytes."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    q_mult = 2 if geom.attn_output_gate else 1
    kvh = max(1, geom.num_kv_heads // tp) if geom.num_kv_heads > 0 else 0
    w_attn = 0.0
    gqa_hd = geom.global_head_dim if geom.global_head_dim > 0 else geom.head_dim
    stage_scale = pp_stage_layer_scale(geom.num_layers, pp)
    for n_layers, hd in ((counts.gqa, gqa_hd), (counts.swa, geom.head_dim)):
        if n_layers <= 0:
            continue
        w_q_o = (
            float(n_layers)
            * float(
                geom.hidden * geom.num_heads * hd * q_mult
                + geom.num_heads * hd * geom.hidden
            )
            * dt.w_dense
            * stage_scale
            / tp
        )
        w_kv = (
            float(n_layers)
            * float(2 * geom.hidden * kvh * hd)
            * dt.w_dense
            * stage_scale
        )
        w_attn += w_q_o + w_kv
    if counts.mla > 0:
        w_attn += _compute_mla_w_attn(geom, dt, tp, pp, counts)
    return w_attn


def compute_w_rec(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    counts: LayerCounts,
) -> float:
    """Compute per-core recurrent weight bytes for linear attention layers."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_rec = 0.0
    stage_scale = pp_stage_layer_scale(geom.num_layers, pp)
    if counts.gdn > 0:
        key_heads = max(1, geom.linear_k_heads // tp)
        value_heads = max(1, geom.linear_v_heads // tp)
        qkv_rank_width = (
            2 * key_heads * geom.linear_k_dim + value_heads * geom.linear_v_dim
        )
        z_rank_width = value_heads * geom.linear_v_dim
        w_rec += (
            float(counts.gdn)
            * (
                float(
                    geom.hidden * (qkv_rank_width + z_rank_width)
                    + z_rank_width * geom.hidden
                )
                * dt.w_dense
                + float(
                    geom.hidden * 2 * value_heads + qkv_rank_width * geom.conv_kernel
                )
                * dt.w_aux
            )
            * stage_scale
        )
    if counts.kda > 0:
        key_heads = max(1, (geom.linear_k_heads or geom.num_heads) // tp)
        value_heads = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        k_head_dim = geom.linear_k_dim or geom.head_dim
        v_head_dim = geom.linear_v_dim or geom.head_dim
        w_rec += (
            float(counts.kda)
            * float(
                geom.hidden
                * (2 * key_heads * k_head_dim + 2 * value_heads * v_head_dim)
                + value_heads * v_head_dim * geom.hidden
            )
            * dt.w_dense
            * stage_scale
        )
    return w_rec


def _compute_ffn_component_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
) -> tuple[float, float, float, float]:
    """Compute per-core (dense_ffn, moe_routed_ffn, moe_shared_ffn, router) bytes."""
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    is_ep = parallel.moe == "ep" and parallel.ep > 1
    ep_ranks = parallel.ep if is_ep else tp
    w_dense_ffn = 0.0
    w_routed_ffn = 0.0
    w_shared_ffn = 0.0
    w_router = 0.0
    stage_scale = pp_stage_layer_scale(geom.num_layers, pp)
    n_moe_ffn, n_dense_ffn = geom.ffn_layer_counts()
    if n_dense_ffn > 0 and geom.dense_inter > 0:
        w_dense_ffn = (
            float(n_dense_ffn * geom.ffn_num_matrices * geom.hidden * geom.dense_inter)
            * dt.w_dense
            * stage_scale
            / tp
        )
    if n_moe_ffn > 0:
        expert_div = ep_ranks if is_ep else 1
        tp_shard_div = (
            max(1, tp // ep_ranks)
            if (is_ep and ep_ranks < tp)
            else (1 if is_ep else tp)
        )
        expert_width = (
            aligned_moe_tp_shard_width(geom.moe_inter, tp_shard_div)
            if tp_shard_div > 1
            else geom.moe_inter
        )
        expert_hidden = (
            geom.routed_expert_hidden if geom.routed_expert_hidden > 0 else geom.hidden
        )
        w_routed_ffn = (
            float(
                n_moe_ffn
                * geom.num_experts
                * geom.ffn_num_matrices
                * expert_hidden
                * expert_width
            )
            * dt.w_expert
            * stage_scale
            / expert_div
        )
        if geom.routed_expert_hidden > 0:
            w_routed_ffn += (
                float(n_moe_ffn * 2 * geom.hidden * geom.routed_expert_hidden)
                * dt.w_dense
                * stage_scale
                / tp
            )
        if geom.shared_inter > 0:
            shared_rank_width = (
                geom.shared_inter
                if parallel.residual == "sp"
                else max(1, geom.shared_inter // tp)
            )
            w_shared_ffn = (
                float(
                    n_moe_ffn * geom.ffn_num_matrices * geom.hidden * shared_rank_width
                )
                * dt.w_dense
                * stage_scale
            )
        w_router = (
            float(n_moe_ffn * geom.hidden * geom.num_experts) * dt.router * stage_scale
        )
    return w_dense_ffn, w_routed_ffn, w_shared_ffn, w_router


def compute_w_ffn(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
) -> tuple[float, float]:
    """Compute per-core dense and routed MoE FFN weight bytes."""
    w_dense_ffn, w_routed_ffn, w_shared_ffn, w_router = _compute_ffn_component_bytes(
        geom, dt, parallel
    )
    return w_dense_ffn + w_routed_ffn + w_shared_ffn + w_router, w_dense_ffn


def _compute_w_mtp(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    w_attn_data: tuple[float, int],
    w_ffn_terms: tuple[float, float],
) -> float:
    """Compute per-core weight bytes for multi-token prediction layers."""
    if geom.mtp_layers <= 0:
        return 0.0
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    unscale = 1.0 / pp_stage_layer_scale(geom.num_layers, pp)
    w_attn, n_attn_total = w_attn_data
    w_ffn, w_dense_ffn = w_ffn_terms
    mtp_fc = float(2 * geom.hidden * geom.hidden) * dt.w_aux / tp
    if n_attn_total > 0:
        mtp_att = (w_attn / float(n_attn_total)) * unscale
    else:
        draft_layer_type = (
            "dsa"
            if (geom.kv_lora_rank > 0 and geom.index_topk > 0)
            else ("mla" if geom.kv_lora_rank > 0 else "full_attention")
        )
        draft_attn_geom = replace(geom, num_layers=1, layer_types=(draft_layer_type,))
        draft_counts = count_layer_types(draft_attn_geom)
        mtp_att = compute_w_attn(
            draft_attn_geom, dt, replace(parallel, pp=1), draft_counts
        )
    n_moe_ffn, n_dense_ffn = geom.ffn_layer_counts()
    if geom.mtp_inter > 0:
        mtp_ffn = (
            float(geom.ffn_num_matrices * geom.hidden * geom.mtp_inter)
            * dt.w_dense
            / tp
        )
    elif geom.num_experts > 0:
        if n_moe_ffn > 0:
            mtp_ffn = ((w_ffn - w_dense_ffn) / float(n_moe_ffn)) * unscale
        elif geom.moe_inter > 0:
            draft_moe_geom = replace(
                geom,
                num_layers=1,
                first_k_dense_replace=0,
                ffn_kinds=("moe",),
                n_moe_layers=None,
                n_dense_layers=None,
            )
            mtp_ffn, _ = compute_w_ffn(draft_moe_geom, dt, replace(parallel, pp=1))
        elif n_dense_ffn > 0:
            mtp_ffn = (w_dense_ffn / float(n_dense_ffn)) * unscale
        else:
            mtp_ffn = 0.0
    elif n_dense_ffn > 0:
        mtp_ffn = (w_dense_ffn / float(n_dense_ffn)) * unscale
    else:
        mtp_ffn = (w_ffn / float(max(1, geom.num_layers))) * unscale
    return float(geom.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn)


def _effective_dsa_kv_layers(
    geom: ModelGeometry,
    counts: LayerCounts,
) -> tuple[float, float]:
    """Split effective MLA layers into latent-only and full-indexer DSA counts."""
    plain_eff_mla = max(0.0, counts.eff_mla - counts.eff_dsa)
    if counts.dsa > 0 and geom.indexer_types:
        full_frac = float(geom.indexer_types.count("full")) / float(counts.dsa)
        full_dsa = counts.eff_dsa * full_frac
        return plain_eff_mla + (counts.eff_dsa - full_dsa), full_dsa
    return plain_eff_mla, counts.eff_dsa


def _compute_kv_and_state(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    chip: ChipSpec | None = None,
) -> tuple[float, float]:
    """Compute (kv_bytes, state_bytes) active on one core."""
    counts = count_layer_types(geom)
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    if counts.mla > 0:
        if geom.kv_lora_rank <= 0:
            msg = "MLA layer requires config field 'kv_lora_rank'"
            raise ValueError(msg)
        if geom.qk_rope_head_dim < 0:
            msg = "MLA layer requires config field 'qk_rope_head_dim'"
            raise ValueError(msg)
    gqa_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "gqa", physical=layout.physical_kv_layout, chip=chip
    )
    swa_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "swa", physical=layout.physical_kv_layout, chip=chip
    )
    mla_only_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "mla", physical=layout.physical_kv_layout, chip=chip
    )
    dsa_kv_b_per_tok = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "dsa", physical=layout.physical_kv_layout, chip=chip
    )
    stage_scale = pp_stage_layer_scale(geom.num_layers, pp)
    kv_div = parallel.kv_split * parallel.pcp
    kv_gqa = (
        float(layout.seqs_per_dp * layout.context) * counts.eff_gqa * gqa_kv_b_per_tok
    )
    eff_swa_ctx = (
        min(layout.context, geom.sliding_window)
        if geom.sliding_window > 0
        else layout.context
    )
    kv_swa = float(layout.seqs_per_dp * eff_swa_ctx) * counts.eff_swa * swa_kv_b_per_tok
    mla_only_layers, full_dsa_layers = _effective_dsa_kv_layers(geom, counts)
    kv_mla = float(layout.seqs_per_dp * layout.context) * (
        mla_only_layers * mla_only_kv_b_per_tok + full_dsa_layers * dsa_kv_b_per_tok
    )
    kv_bytes = (kv_gqa + kv_swa + kv_mla) * stage_scale / kv_div

    st_bytes = 0.0
    if counts.gdn > 0:
        key_heads = max(1, geom.linear_k_heads // tp)
        value_heads = max(1, geom.linear_v_heads // tp)
        qkv_rank_width = (
            2 * key_heads * geom.linear_k_dim + value_heads * geom.linear_v_dim
        )
        st_bytes += (
            float(layout.seqs_per_dp * counts.gdn)
            * (
                float(value_heads * geom.linear_k_dim * geom.linear_v_dim) * dt.state
                + float(qkv_rank_width * max(0, geom.conv_kernel - 1)) * dt.conv
            )
            * stage_scale
        )
    if counts.kda > 0:
        value_heads = max(1, (geom.linear_v_heads or geom.num_heads) // tp)
        k_head_dim = geom.linear_k_dim or geom.head_dim
        v_head_dim = geom.linear_v_dim or geom.head_dim
        st_bytes += (
            float(
                layout.seqs_per_dp * counts.kda * value_heads * k_head_dim * v_head_dim
            )
            * dt.state
            * stage_scale
        )
    return kv_bytes, st_bytes


def _compute_w_head(
    geom: ModelGeometry,
    dt: Dtypes,
    tp: int,
) -> tuple[float, float]:
    """Compute single-stage per-core (embedding, lm_head) weight bytes."""
    emb = float(geom.vocab * geom.hidden) * dt.lm_head / tp
    lm_head = 0.0 if geom.tie_word_embeddings else emb
    return emb, lm_head


def _compute_w_hc(geom: ModelGeometry, dt: Dtypes, tp: int, pp: int) -> float:
    """Compute per-core hyper-connection weight bytes."""
    if geom.hc_mult <= 1:
        return 0.0
    stage_scale = pp_stage_layer_scale(geom.num_layers, pp)
    if geom.hc_lowrank > 0:
        return (
            float(
                geom.num_layers
                * _HC_LOWRANK_DIRECTIONS
                * geom.hidden
                * geom.hc_lowrank
                * geom.hc_mult
            )
            * dt.w_dense
            * stage_scale
            / float(tp)
        )
    return (
        float(geom.num_layers * (geom.hc_mult**2) * geom.hidden)
        * dt.w_dense
        * stage_scale
        / float(tp)
    )


_UNIT_DTYPES = Dtypes(
    w_dense=1.0,
    w_expert=1.0,
    w_aux=1.0,
    kv=1.0,
    state=1.0,
    conv=1.0,
    router=1.0,
    lm_head=1.0,
)
_UNIT_PARALLEL = Parallel(tp=1, pp=1, ep=1, moe="tp_slice")


def param_counts(geom: ModelGeometry) -> ParamBreakdown:
    """Return unsharded model parameter counts aligned 1:1 with HBM weight bytes.

    Shares the exact per-component element calculations with
    :func:`hbm_breakdown_per_core_bytes`, intentionally omitting normalization
    weights, linear biases, PLE parameters, and KDA auxiliary parameters that
    HBM weight sizing does not model.
    """
    counts = count_layer_types(geom)
    attention = compute_w_attn(geom, _UNIT_DTYPES, _UNIT_PARALLEL, counts)
    linear_attention = compute_w_rec(geom, _UNIT_DTYPES, _UNIT_PARALLEL, counts)
    dense_ffn, moe_routed_ffn, moe_shared_ffn, router = _compute_ffn_component_bytes(
        geom, _UNIT_DTYPES, _UNIT_PARALLEL
    )
    w_ffn = dense_ffn + moe_routed_ffn + moe_shared_ffn + router
    mtp = _compute_w_mtp(
        geom,
        _UNIT_DTYPES,
        _UNIT_PARALLEL,
        (attention, counts.attn_total),
        (w_ffn, dense_ffn),
    )
    embedding, lm_head = _compute_w_head(geom, _UNIT_DTYPES, tp=1)
    hc = _compute_w_hc(geom, _UNIT_DTYPES, tp=1, pp=1)
    total_params = (
        attention
        + linear_attention
        + dense_ffn
        + moe_routed_ffn
        + moe_shared_ffn
        + router
        + embedding
        + lm_head
        + mtp
        + hc
    )
    inactive_routed = 0.0
    if geom.num_experts > 0:
        expert_hidden = (
            geom.routed_expert_hidden if geom.routed_expert_hidden > 0 else geom.hidden
        )
        n_moe_ffn, _ = geom.ffn_layer_counts()
        mtp_moe_layers = (
            geom.mtp_layers
            if (n_moe_ffn > 0 and geom.mtp_layers > 0 and geom.mtp_inter <= 0)
            else 0
        )
        inactive_experts = max(0, geom.num_experts - geom.top_k)
        inactive_routed = float(
            (n_moe_ffn + mtp_moe_layers)
            * inactive_experts
            * geom.ffn_num_matrices
            * expert_hidden
            * geom.moe_inter
        )
    return ParamBreakdown(
        attention=attention,
        linear_attention=linear_attention,
        dense_ffn=dense_ffn,
        moe_routed_ffn=moe_routed_ffn,
        moe_shared_ffn=moe_shared_ffn,
        router=router,
        embedding=embedding,
        lm_head=lm_head,
        mtp=mtp,
        hc=hc,
        total_params=total_params,
        active_params_per_token=max(0.0, total_params - inactive_routed),
    )


@lru_cache(maxsize=STEP_FLOOR_CACHE_MAXSIZE)
def hbm_breakdown_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    chip: ChipSpec | None = None,
) -> HbmBreakdown:
    """Compute per-core static weight, active KV cache, and state byte totals."""
    counts = count_layer_types(geom)
    tp = max(1, parallel.tp)
    pp = max(1, parallel.pp)
    w_attn = compute_w_attn(geom, dt, parallel, counts)
    w_rec = compute_w_rec(geom, dt, parallel, counts)
    w_ffn, w_dense_ffn = compute_w_ffn(geom, dt, parallel)
    w_emb, w_lm_head = _compute_w_head(geom, dt, tp)
    w_mtp = _compute_w_mtp(
        geom,
        dt,
        parallel,
        (w_attn, counts.attn_total),
        (w_ffn, w_dense_ffn),
    )
    # Stage 0 holds embeddings whole; last stage holds untied lm_head + MTP whole.
    w_head_mtp = w_emb + w_lm_head + w_mtp if pp == 1 else max(w_emb, w_lm_head + w_mtp)
    w_hc = _compute_w_hc(geom, dt, tp, pp)
    weights = w_attn + w_rec + w_ffn + w_head_mtp + w_hc
    kv, state = _compute_kv_and_state(geom, dt, parallel, layout, chip=chip)
    return HbmBreakdown(
        weights=weights,
        kv=kv,
        state=state,
        total=weights + kv + state,
    )


def hbm_footprint_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    layout: SeqKvLayout,
    chip: ChipSpec | None = None,
) -> float:
    """Compute total HBM footprint (weights + KV + state) in bytes on one core."""
    return hbm_breakdown_per_core_bytes(geom, dt, parallel, layout, chip=chip).total


def default_rpa_page_size(ctx_tokens: int, *, is_mla: bool = False) -> int:
    """Determine default RPA or MLA page size matching TPU attention backends."""
    if is_mla:
        return MLA_PAGE_SIZE_TOKENS
    if ctx_tokens > MAX_CTX_TOKENS_FOR_SMALL_PAGE:
        return RPA_LONG_CTX_PAGE_SIZE_TOKENS
    pow2 = 1 << (max(1, ctx_tokens) - 1).bit_length()
    return min(
        RPA_MAX_PAGE_TOKENS,
        max(RPA_MIN_PAGE_TOKENS, pow2 // RPA_MIN_PAGE_TOKENS),
    )


def tpu_min_page_size(
    ctx_tokens: int,
    kv_split: int,
    max_num_seqs: int,
    *,
    pcp: int = 1,
    page_table_bytes: int | None = None,
) -> int:
    """Return minimum power-of-2 page size that fits SMEM page tables."""
    if pcp > 1:
        return 1
    budget = (
        CHIPS["tpu7x"].smem_page_table_bytes
        if page_table_bytes is None
        else page_table_bytes
    )
    max_pages_per_req = max(
        1,
        budget // max(1, max_num_seqs) // BYTES_PER_PAGE_ENTRY,
    )
    local_ctx = math.ceil(max(1, ctx_tokens) / max(1, kv_split))
    min_page = math.ceil(local_ctx / max_pages_per_req)
    return 1 << (max(1, min_page) - 1).bit_length()


def smem_max_seqs(
    ctx_tokens: int,
    kv_split: int,
    page_size: int,
    page_table_bytes: int | None = None,
) -> int:
    """Estimate maximum concurrent sequences fitting inside SMEM page tables."""
    budget = (
        CHIPS["tpu7x"].smem_page_table_bytes
        if page_table_bytes is None
        else page_table_bytes
    )
    pages_per_req = max(
        1,
        math.ceil((ctx_tokens / max(1, kv_split)) / max(1, page_size)),
    )
    return budget // pages_per_req // BYTES_PER_PAGE_ENTRY


class _KvPartsOptions(TypedDict, total=False):
    """Optional keyword arguments for :func:`kv_bytes_per_core_parts`."""

    physical: bool
    chip: ChipSpec | None


def kv_bytes_per_core_parts(
    geom: ModelGeometry,
    dt: Dtypes,
    parallel: Parallel,
    ctx_tokens: int,
    **options: Unpack[_KvPartsOptions],
) -> tuple[float, float]:
    """Return (full_context_kv_per_token_bytes, fixed_window_kv_per_seq_bytes)."""
    physical = options.get("physical", True)
    chip = options.get("chip")
    counts = count_layer_types(geom)
    pp = max(1, parallel.pp)
    kv_div = float(parallel.kv_split * parallel.pcp * pp)
    gqa_kv_b = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "gqa", physical=physical, chip=chip
    )
    swa_kv_b = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "swa", physical=physical, chip=chip
    )
    mla_kv_b = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "mla", physical=physical, chip=chip
    )
    dsa_kv_b = kv_bytes_per_token_per_layer(
        geom, dt, parallel, "dsa", physical=physical, chip=chip
    )
    mla_only_layers, full_dsa_layers = _effective_dsa_kv_layers(geom, counts)
    linear_part = (
        counts.eff_gqa * gqa_kv_b
        + mla_only_layers * mla_kv_b
        + full_dsa_layers * dsa_kv_b
    )
    if geom.sliding_window > 0:
        full_kv_per_tok = linear_part / kv_div
        eff_swa_ctx = min(max(0, ctx_tokens), geom.sliding_window)
        fixed_kv_per_seq = (float(eff_swa_ctx) * counts.eff_swa * swa_kv_b) / kv_div
        return full_kv_per_tok, fixed_kv_per_seq
    full_kv_per_tok = (linear_part + counts.eff_swa * swa_kv_b) / kv_div
    return full_kv_per_tok, 0.0


class _KvCapacityOptions(TypedDict, total=False):
    """Optional keyword arguments for :func:`kv_capacity_and_hit_rate`."""

    fixed_kv_per_seq_bytes: float | None
    chip: ChipSpec | None


def kv_capacity_and_hit_rate(
    geom: ModelGeometry,
    dtypes: Dtypes,
    parallel: Parallel,
    b_res: int,
    kv_fit: KvFitInput,
    **options: Unpack[_KvCapacityOptions],
) -> tuple[float, float]:
    """Compute physical KV token capacity and clamped prefix cache hit rate."""
    fixed_kv_per_seq_bytes = options.get("fixed_kv_per_seq_bytes")
    chip = options.get("chip")
    st_b = (
        hbm_footprint_per_core_bytes(
            geom, dtypes, parallel, SeqKvLayout(b_res, 0), chip=chip
        )
        - kv_fit.w_base_b
    )
    fixed_seq_b = max(
        0.0,
        kv_fit.fixed_kv_per_seq_bytes
        if fixed_kv_per_seq_bytes is None
        else fixed_kv_per_seq_bytes,
    )
    fixed_resident_b = float(max(0, b_res)) * fixed_seq_b
    avail_b = (
        max(0.0, kv_fit.hbm_limit_b - kv_fit.w_base_b - st_b - fixed_resident_b)
        + kv_fit.offload_b_per_core
    )
    cap_tok = (
        avail_b / kv_fit.b_kv_phys_core if kv_fit.b_kv_phys_core > 0.0 else math.inf
    )
    w_tok = max(0.0, kv_fit.kv_working_set_turns) * float(kv_fit.b_ws * kv_fit.ctx_full)
    if kv_fit.prefix_hit_rate > 0.0 and w_tok > 0.0 and kv_fit.b_kv_phys_core > 0.0:
        h_eff = kv_fit.prefix_hit_rate * min(1.0, cap_tok / max(1.0, w_tok))
    elif (
        kv_fit.prefix_hit_rate > 0.0
        and kv_fit.b_kv_phys_core <= 0.0
        and fixed_seq_b > 0.0
    ):
        w_seqs = max(0.0, kv_fit.kv_working_set_turns) * float(max(0, kv_fit.b_ws))
        avail_no_fixed_b = (
            max(0.0, kv_fit.hbm_limit_b - kv_fit.w_base_b - st_b)
            + kv_fit.offload_b_per_core
        )
        cap_seqs = avail_no_fixed_b / fixed_seq_b
        h_eff = (
            kv_fit.prefix_hit_rate * min(1.0, cap_seqs / max(1.0, w_seqs))
            if w_seqs > 0.0
            else kv_fit.prefix_hit_rate
        )
    else:
        h_eff = kv_fit.prefix_hit_rate
    return cap_tok, h_eff
