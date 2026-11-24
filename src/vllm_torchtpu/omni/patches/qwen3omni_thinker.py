# Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
"""TPU patches for Qwen3-Omni Thinker encoders and multimodal embedding merge.

1. **Audio encoder (``Qwen3OmniMoeAudioEncoder``)**:
   Evaluates 100-frame window counts with integer division on CPU (avoiding TPU
   float32 reciprocal rounding on exact multiples of 100 frames; see upstream
   ``vllm-omni#9010``), splits clips longer than 128 windows along 128-window
   (``16 * n_window_infer``) boundaries, packs consecutive clips into power-of-2
   window buckets ``(8, 16, 32, 64, 128)``, and compiles one fused TPU graph per
   bucket via ``torch.compile(..., backend="tpu", fullgraph=True, dynamic=False)``.
2. **Vision encoder (``Qwen3Omni_VisionTransformer``)**:
   Packs consecutive image/video items into ViT passes of at most ``16,384``
   patches, splits larger videos along the frame axis, and pads each pass and
   its ``cu_seqlens`` to a static ``(4096, 8192, 16384)``-patch bucket.
3. **Multimodal embedding merge and ``embed_input_ids``**:
   Pads flattened multimodal embeddings to ``inputs_embeds.shape[0]`` before
   delegating to ``vllm_torchtpu``'s static ``_merge_multimodal_embeddings``,
   routes interleaved video+audio merges through the same static path, clears
   stale ``deepstack_input_embeds``, moves ``is_multimodal`` to CPU only when
   ``multimodal_embeddings`` is non-empty, and cuts the deferred TorchTPU graph
   around the merge.

TODO: retire the 0.32.0rc1 ``xfail`` test in ``test_qwen3_omni_components.py`` once
vLLM-Omni is bumped to >= 0.33.0rc1 (vllm-omni#9010), and remove this module
once upstream supports native static bucketing and TPU-friendly multimodal
merging (tracked upstream in vllm-project/vllm-omni#7452 and
vllm-project/vllm-omni#9566).
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_QWEN3_OMNI_ARCHS = frozenset(
    {
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    }
)
# Idempotence marker on wrapped multimodal merge / embed_input_ids functions.
_MM_MERGE_STATIC_PATCHED = "_tpu_mm_merge_static_patched"
# Cap at 128 100-frame windows (1,664 = 13 * 128 post-CNN tokens): exact
# multiple of Pallas FlashAttention's 128-token block size at minimum
# per-window latency before quadratic 2D masked-tile overhead grows at 256+.
_AUDIO_CHUNK_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128)
# Uniform 128-entry int32 buffer across all buckets: negligible H2D/broadcast
# cost (512 B) and keeps a single grouping budget across every bucket size.
_CU_SEQLENS_BLOCK_SIZE = 128
# Reserve 1 leading ``0`` entry plus up to 1 trailing ``padded_total`` sentinel.
_CU_SEQLENS_RESERVED_ENTRIES = 2
# Static ViT patch buckets (1024, 2048, 4096 merged tokens at spatial_merge_size=2).
# The Thinker calls self.visual(pixel_values, grid_thw) directly, bypassing
# MMEncoderManager, which also does not pad patches or cu_seqlens to a static shape.
_VISION_PATCH_BUCKETS: tuple[int, ...] = (4096, 8192, 16384)


class _AudioSlice(NamedTuple):
    """Frame range and chunk/token counts for one ``<= max_bucket`` audio slice."""

    frame_start: int
    frame_len: int
    num_chunks: int
    valid_tokens: int


def _select_bucket(val: int, buckets: tuple[int, ...], align: int) -> int:
    for b in buckets:
        if val <= b:
            return b
    return cdiv(val, align) * align


def _pad_rows(x: torch.Tensor, target_rows: int) -> torch.Tensor:
    pad = target_rows - x.shape[0]
    if pad <= 0:
        return x
    return F.pad(x, (0, 0, 0, pad))


def _pad_cu_seqlens(
    cu_cpu: list[int],
    padded_total: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad ``cu_seqlens`` to fixed ``_CU_SEQLENS_BLOCK_SIZE`` with ``padded_total``.

    Padding with ``padded_total`` keeps trailing segments zero-length for
    ``torch.split`` (CPU/native SDPA) while assigning all FlashAttention
    block-padding positions (``t >= padded_total``) to their own isolated
    segment id in ``TpuMMEncoderAttention``.
    """
    pad_len = _CU_SEQLENS_BLOCK_SIZE - len(cu_cpu)
    if pad_len < 0 or (pad_len == 0 and cu_cpu[-1] < padded_total):
        raise ValueError(
            f"cu_seqlens length ({len(cu_cpu)}) exceeds fixed "
            f"_CU_SEQLENS_BLOCK_SIZE ({_CU_SEQLENS_BLOCK_SIZE})."
        )
    return torch.tensor(
        cu_cpu + [padded_total] * pad_len, dtype=torch.int32, device="cpu"
    ).to(device=device)


def _validate_audio_window_config(
    encoder: Any, window_size: int, max_bucket: int
) -> int:
    """Validate ``n_window_infer`` alignment and return chunks per attention window."""
    if encoder.n_window_infer <= 0 or encoder.n_window_infer % window_size != 0:
        raise ValueError(
            f"n_window_infer ({encoder.n_window_infer}) must be a positive multiple "
            f"of 2 * n_window ({window_size})."
        )
    chunks_per_attn_window = encoder.n_window_infer // window_size
    if max_bucket % chunks_per_attn_window != 0:
        raise ValueError(
            f"max_bucket ({max_bucket}) must be divisible by "
            f"n_window_infer // (2 * n_window) ({chunks_per_attn_window}) "
            "so multi-pass splits preserve attention window boundaries."
        )
    max_item_cu_entries = (
        max_bucket // chunks_per_attn_window + _CU_SEQLENS_RESERVED_ENTRIES
    )
    if max_item_cu_entries > _CU_SEQLENS_BLOCK_SIZE:
        raise ValueError(
            f"A full {max_bucket}-window clip needs up to {max_item_cu_entries} "
            f"cu_seqlens entries (n_window_infer // (2 * n_window)="
            f"{chunks_per_attn_window}), exceeding _CU_SEQLENS_BLOCK_SIZE "
            f"({_CU_SEQLENS_BLOCK_SIZE})."
        )
    return chunks_per_attn_window


def _compute_audio_chunk_metadata(
    encoder: Any, feature_lens: torch.Tensor, window_size: int
) -> tuple[list[int], list[int], list[int], int]:
    """Compute host-side window counts and post-CNN token lengths on CPU.

    Avoids ``vllm-omni==0.32.0rc1``'s TPU float32 ``ceil(feature_lens / 100)``
    reciprocal rounding bug (``vllm-omni#9010``) by evaluating exact integer
    division and ``_get_cnn_output_lengths`` on CPU.
    """
    lens_cpu = feature_lens.detach().cpu().to(dtype=torch.long).reshape(-1)
    if (lens_cpu < 0).any().item():
        raise ValueError(f"feature_lens must be non-negative, got {lens_cpu.tolist()}.")

    chunk_nums_t = torch.div(
        lens_cpu + window_size - 1, window_size, rounding_mode="floor"
    )
    rem = torch.remainder(lens_cpu, window_size)
    tail_frames = torch.where(rem == 0, window_size, rem)
    probe = torch.cat(
        [
            torch.tensor([window_size], dtype=torch.long, device="cpu"),
            tail_frames,
        ]
    )
    cnn_probe = encoder._get_cnn_output_lengths(probe)
    tokens_per_chunk = int(cnn_probe[0].item())
    aftercnn_lens_t = torch.where(
        lens_cpu > 0,
        (chunk_nums_t - 1) * tokens_per_chunk + cnn_probe[1:],
        0,
    )
    return (
        lens_cpu.tolist(),
        chunk_nums_t.tolist(),
        aftercnn_lens_t.tolist(),
        tokens_per_chunk,
    )


def _split_audio_clips(
    feature_lens: list[int],
    chunk_nums: list[int],
    aftercnn_lens: list[int],
    window_size: int,
    max_bucket: int,
    tokens_per_chunk: int,
) -> list[_AudioSlice]:
    """Split mel clips into ``<= max_bucket`` frame ranges without copying tensors."""
    max_frames = max_bucket * window_size
    max_tokens = max_bucket * tokens_per_chunk
    slices: list[_AudioSlice] = []
    col_off = 0
    for clip_len, n_chunks, clip_tokens in zip(
        feature_lens, chunk_nums, aftercnn_lens, strict=True
    ):
        if clip_len > 0:
            full_splits = (n_chunks - 1) // max_bucket
            slices.extend(
                _AudioSlice(
                    frame_start=col_off + i * max_frames,
                    frame_len=max_frames,
                    num_chunks=max_bucket,
                    valid_tokens=max_tokens,
                )
                for i in range(full_splits)
            )
            tail_frame_off = full_splits * max_frames
            slices.append(
                _AudioSlice(
                    frame_start=col_off + tail_frame_off,
                    frame_len=clip_len - tail_frame_off,
                    num_chunks=n_chunks - full_splits * max_bucket,
                    valid_tokens=clip_tokens - full_splits * max_tokens,
                )
            )
        col_off += clip_len
    return slices


def _group_slices(
    slices: list[Any], costs: list[tuple[int, int]], max_bucket: int
) -> list[list[Any]]:
    """Greedily pack slices into groups fitting ``max_bucket`` and ``cu_seqlens``.

    ``costs`` gives each slice's ``(bucket units, cu_seqlens entries)``.
    """
    groups: list[list[Any]] = []
    cur_group: list[Any] = []
    cur_units = 0
    cur_cu_entries = _CU_SEQLENS_RESERVED_ENTRIES
    for item, (units, cu_entries) in zip(slices, costs, strict=True):
        if (
            cur_units + units > max_bucket
            or cur_cu_entries + cu_entries > _CU_SEQLENS_BLOCK_SIZE
        ) and cur_group:
            groups.append(cur_group)
            cur_group = []
            cur_units = 0
            cur_cu_entries = _CU_SEQLENS_RESERVED_ENTRIES
        cur_group.append(item)
        cur_units += units
        cur_cu_entries += cu_entries
    if cur_group:
        groups.append(cur_group)
    return groups


def _build_bucket_inputs(
    feats_u8_np: np.ndarray,
    dtype: torch.dtype,
    group: list[_AudioSlice],
    window_size: int,
    tokens_per_chunk: int,
    window_aftercnn: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], int]:
    """Pack one slice group into a single static bucket tensor and ``cu_seqlens``."""
    mel_bins = feats_u8_np.shape[0]
    elem_size = dtype.itemsize
    row_bytes = window_size * elem_size
    group_chunks = sum(item.num_chunks for item in group)
    bucket_chunks = next(b for b in _AUDIO_CHUNK_BUCKETS if group_chunks <= b)
    # Single-threaded NumPy packing avoids PyTorch's multi-thread OpenMP
    # overhead on small host copies.
    x_np = np.zeros((bucket_chunks, 1, mel_bins, row_bytes), dtype=np.uint8)
    cu_list = [0]
    valid_spans: list[tuple[int, int]] = []
    chunk_off = 0
    for idx, item in enumerate(group):
        full_chunks, tail_frames = divmod(item.frame_len, window_size)
        if full_chunks > 0:
            start_b = item.frame_start * elem_size
            end_b = start_b + full_chunks * row_bytes
            x_np[chunk_off : chunk_off + full_chunks, 0] = (
                feats_u8_np[:, start_b:end_b]
                .reshape(mel_bins, full_chunks, row_bytes)
                .transpose(1, 0, 2)
            )
        if tail_frames > 0:
            tail_start_b = (item.frame_start + full_chunks * window_size) * elem_size
            tail_bytes = tail_frames * elem_size
            x_np[chunk_off + full_chunks, 0, :, :tail_bytes] = feats_u8_np[
                :, tail_start_b : tail_start_b + tail_bytes
            ]

        row_start = chunk_off * tokens_per_chunk
        row_valid_end = row_start + item.valid_tokens
        row_chunk_end = (chunk_off + item.num_chunks) * tokens_per_chunk
        if valid_spans and valid_spans[-1][1] == row_start:
            valid_spans[-1] = (valid_spans[-1][0], row_valid_end)
        else:
            valid_spans.append((row_start, row_valid_end))

        cu_list.extend(
            range(row_start + window_aftercnn, row_valid_end, window_aftercnn)
        )
        cu_list.append(row_valid_end)
        # Fence off dead tail tokens of a partial chunk into their own
        # isolated attention segment before the next packed slice starts.
        if idx + 1 < len(group) and row_valid_end < row_chunk_end:
            cu_list.append(row_chunk_end)
        chunk_off += item.num_chunks

    padded_feature = torch.from_numpy(x_np).view(dtype).to(device=device)
    cu_seqlens = _pad_cu_seqlens(
        cu_list, bucket_chunks * tokens_per_chunk, device=device
    )
    return padded_feature, cu_seqlens, valid_spans, bucket_chunks


def _run_static_bucket_pass(
    encoder: Any,
    padded_feature: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Execute one static-shape Conv2d + Transformer + MLP pass.

    Copied from upstream ``vllm-omni@v0.32.0rc1``
    (``vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py``,
    ``Qwen3OmniMoeAudioEncoder.forward``) with TPU static-shape edits noted inline.
    """
    # TPU: skip conv_chunksize split; callers cap batch size <= max_bucket (128).
    padded_embed = F.gelu(encoder.conv2d1(padded_feature))
    padded_embed = F.gelu(encoder.conv2d2(padded_embed))
    padded_embed = F.gelu(encoder.conv2d3(padded_embed))

    b, c, f, t = padded_embed.size()
    padded_embed, _ = encoder.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )

    positional_embedding = (
        encoder.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(device=padded_embed.device, dtype=padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding

    # TPU: flatten static [b * t, d] instead of dynamic padded_embed[mask].
    hidden_states = padded_embed.reshape(b * t, -1)

    # TPU: pass max_seqlen=None to avoid host sync inside fullgraph torch.compile.
    for encoder_layer in encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens,
            None,
        )

    hidden_states = encoder.ln_post(hidden_states)
    hidden_states, _ = encoder.proj1(hidden_states)
    hidden_states = encoder.act(hidden_states)
    hidden_states, _ = encoder.proj2(hidden_states)

    return hidden_states


def _get_compiled_bucket_fn(
    encoder: Any, bucket_chunks: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Cache one ``torch.compile(backend="tpu")`` callable with a distinct CodeType."""
    cache: dict[int, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = (
        getattr(encoder, "_tpu_compiled_buckets", None)
    )
    if cache is None:
        cache = {}
        encoder._tpu_compiled_buckets = cache
    compiled_fn = cache.get(bucket_chunks)
    if compiled_fn is None:

        def _bucket_forward(x_in: torch.Tensor, cu_in: torch.Tensor) -> torch.Tensor:
            return _run_static_bucket_pass(encoder, x_in, cu_in)

        unique_fn = types.FunctionType(
            _bucket_forward.__code__.replace(
                co_name=f"_audio_bucket_{bucket_chunks}_{id(encoder)}"
            ),
            _bucket_forward.__globals__,
            name=f"_audio_bucket_{bucket_chunks}",
            closure=_bucket_forward.__closure__,
        )
        compiled_fn = torch.compile(
            unique_fn, backend="tpu", fullgraph=True, dynamic=False
        )
        cache[bucket_chunks] = compiled_fn
    return compiled_fn


def _padded_audio_forward(
    self: Any,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``Qwen3OmniMoeAudioEncoder.forward`` over static power-of-2 buckets."""
    # Recomputed on CPU in _compute_audio_chunk_metadata to avoid a TPU D2H sync.
    del aftercnn_lens
    window_size = self.n_window * 2
    max_bucket = _AUDIO_CHUNK_BUCKETS[-1]
    chunks_per_attn_window = _validate_audio_window_config(
        self, window_size, max_bucket
    )

    feature_lens_list, chunk_nums, aftercnn_lens_list, tokens_per_chunk = (
        _compute_audio_chunk_metadata(self, feature_lens, window_size)
    )
    device = self.conv2d1.weight.device
    dtype = self.conv2d1.weight.dtype
    if sum(aftercnn_lens_list) == 0:
        return torch.empty((0, self.proj2.output_size), device=device, dtype=dtype)

    total_frames = sum(feature_lens_list)
    if input_features.ndim != 2 or input_features.shape[1] < total_frames:
        raise ValueError(
            f"Expected 2D input_features with >= {total_frames} frames, "
            f"got shape {tuple(input_features.shape)}."
        )

    # Upstream stages input_audio_features on TPU; moving to host once here lets
    # us pack static bucket tensors on CPU instead of keying TPU graphs on raw
    # mel frame counts.
    # TODO: keep input_audio_features on host before audio encoding to eliminate
    # the D2H round-trip.
    feats_cpu = input_features.to(device="cpu", dtype=dtype)
    feats_u8_np = feats_cpu.detach().contiguous().view(torch.uint8).numpy()
    window_aftercnn = tokens_per_chunk * chunks_per_attn_window
    slices = _split_audio_clips(
        feature_lens_list,
        chunk_nums,
        aftercnn_lens_list,
        window_size,
        max_bucket,
        tokens_per_chunk,
    )
    groups = _group_slices(
        slices,
        [(s.num_chunks, cdiv(s.valid_tokens, window_aftercnn) + 1) for s in slices],
        max_bucket,
    )

    is_tpu = device.type == "tpu"
    use_compiled = is_tpu and not self._tpu_enforce_eager
    outputs: list[torch.Tensor] = []
    for group in groups:
        x, cu_seqlens, valid_spans, bucket_chunks = _build_bucket_inputs(
            feats_u8_np,
            dtype,
            group,
            window_size,
            tokens_per_chunk,
            window_aftercnn,
            device,
        )
        if use_compiled:
            hidden_states = _get_compiled_bucket_fn(self, bucket_chunks)(x, cu_seqlens)
        else:
            hidden_states = _run_static_bucket_pass(self, x, cu_seqlens)
        if is_tpu:
            # Flush each bucket asynchronously so multi-pass uploads overlap
            # with TPU execution in both compiled and eager modes.
            synchronize_tensors(hidden_states, wait=False)

        outputs.extend(hidden_states[start:end] for start, end in valid_spans)

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def _patch_audio_encoder(
    encoder_cls: type[Any], *, enforce_eager: bool = False
) -> bool:
    """Install static-bucketed ``forward`` on ``Qwen3OmniMoeAudioEncoder``."""
    encoder_cls._tpu_enforce_eager = enforce_eager
    if encoder_cls.forward is _padded_audio_forward:
        return False
    encoder_cls.forward = _padded_audio_forward
    return True


def _padded_vit_forward(
    self: Any,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
) -> torch.Tensor:
    """Run ``Qwen3Omni_VisionTransformer.forward`` over static patch buckets.

    Upstream ``prepare_encoder_metadata`` and ``forward_with_encoder_metadata``
    still do all the encoding. Follows the audio forward's steps:
    - Splits each item along the frame axis into pieces that fit the largest
      bucket and the ``cu_seqlens`` budget (exact: each frame is its own
      attention segment), then packs consecutive pieces with ``_group_slices``.
    - Pads each pass's patches, ``grid_thw`` and ``cu_seqlens`` to a
      ``_VISION_PATCH_BUCKETS`` shape and slices the valid tokens afterwards.
    """
    x_dev = x.to(device=self.device, dtype=self.dtype)
    merge_size = self.spatial_merge_size
    merge_unit = merge_size**2
    max_bucket = _VISION_PATCH_BUCKETS[-1]
    max_frames = _CU_SEQLENS_BLOCK_SIZE - _CU_SEQLENS_RESERVED_ENTRIES

    pieces: list[list[int]] = []
    for t, h, w in torch.as_tensor(grid_thw).tolist():
        chunk_t = min(max(1, max_bucket // (h * w)), max_frames)
        pieces.extend([min(chunk_t, t - t0), h, w] for t0 in range(0, t, chunk_t))
    groups = _group_slices(pieces, [(t * h * w, t) for t, h, w in pieces], max_bucket)

    outputs: list[torch.Tensor] = []
    px = 0
    for group in groups:
        valid_tokens = sum(t * h * w for t, h, w in group)
        # The processor's max_pixels keeps one frame <= 16384 patches; align=512
        # is a bounded fallback if a caller overrides max_pixels.
        target_tokens = _select_bucket(
            valid_tokens, _VISION_PATCH_BUCKETS, align=max(512, merge_unit)
        )
        pad_tokens = target_tokens - valid_tokens
        pad_grid = [[pad_tokens // merge_unit, merge_size, merge_size]]
        metadata = self.prepare_encoder_metadata(
            torch.tensor(group + pad_grid * (pad_tokens > 0), dtype=torch.int32)
        )
        valid_frames = sum(t for t, _, _ in group)
        cu_list = metadata["cu_seqlens"].detach().cpu().tolist()[: valid_frames + 1]
        metadata["cu_seqlens"] = _pad_cu_seqlens(cu_list, target_tokens, x_dev.device)
        metadata["max_seqlen"] = None

        merged_all = self.forward_with_encoder_metadata(
            _pad_rows(x_dev[px : px + valid_tokens], target_tokens), metadata
        )
        # Cut the deferred TorchTPU graph before the valid-token slice so the ViT
        # compiles once per bucket shape rather than once per valid_tokens length.
        if merged_all.device.type == "tpu":
            synchronize_tensors(merged_all, wait=False)
        outputs.append(merged_all[: valid_tokens // merge_unit])
        px += valid_tokens

    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def _patch_vision_encoder(vit_cls: type[Any]) -> bool:
    """Install frame chunking and static bucketing on Qwen3Omni_VisionTransformer."""
    orig_forward = vit_cls.forward
    if orig_forward is _padded_vit_forward:
        return False
    vit_cls._tpu_orig_forward = orig_forward
    vit_cls.forward = _padded_vit_forward
    return True


def _patch_mm_merge_and_embed_input_ids(thinker_mod: Any) -> bool:
    """Install static-shape multimodal embedding merge and ``embed_input_ids``."""
    import vllm.model_executor.models.utils as vllm_utils
    from vllm.multimodal.utils import get_mm_embedding_modalities

    from vllm_torchtpu import _patch_vllm_merge_multimodal_embeddings

    patched = False

    # First call installs static_merge_multimodal_embeddings on vllm_utils so
    # base_merge captures the static-indexing implementation; the second call
    # after rebinding vllm_utils._merge_multimodal_embeddings propagates
    # _omni_merge across all already-imported vllm.model_executor.models.* modules.
    _patch_vllm_merge_multimodal_embeddings()
    base_merge = vllm_utils._merge_multimodal_embeddings
    if not getattr(base_merge, _MM_MERGE_STATIC_PATCHED, False):

        def _omni_merge(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_multimodal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Static-shape wrapper around ``static_merge_multimodal_embeddings``.

            Upstream: ``vllm.model_executor.models.utils._merge_multimodal_embeddings``
            and ``vllm_torchtpu``'s ``static_merge_multimodal_embeddings``.
            Stayed the same: flattens ``multimodal_embeddings`` and scatters
            rows into ``inputs_embeds`` where ``is_multimodal`` is ``True``.
            Adapted for TPU: zero-pads flattened embeddings up to
            ``inputs_embeds.shape[0]`` so every merge compiles once per token
            bucket rather than once per media token count, and cuts the deferred
            TorchTPU graph via ``synchronize_tensors(..., wait=False)`` before
            and after ``base_merge``.
            """
            sync_inputs: list[torch.Tensor] = []
            if len(multimodal_embeddings) > 0:
                flat = vllm_utils._flatten_embeddings(multimodal_embeddings)
                if 0 < flat.shape[0] < inputs_embeds.shape[0]:
                    flat = _pad_rows(flat, inputs_embeds.shape[0])
                if flat.device.type == "tpu":
                    flat = flat.contiguous()
                    sync_inputs.append(flat)
                multimodal_embeddings = [flat]
            if is_multimodal is not None:
                is_multimodal = is_multimodal.to(inputs_embeds.device)
                if is_multimodal.device.type == "tpu":
                    sync_inputs.append(is_multimodal)
            if inputs_embeds.device.type == "tpu":
                sync_inputs.append(inputs_embeds)
            # Cut the deferred TorchTPU graph before and after base_merge so
            # the static-padded (num_tokens, D) merge compiles once per token
            # bucket rather than fusing with variable-row-count producers.
            if sync_inputs:
                synchronize_tensors(sync_inputs, wait=False)
            merged = base_merge(inputs_embeds, multimodal_embeddings, is_multimodal)
            if merged.device.type == "tpu":
                synchronize_tensors(merged, wait=False)
            return merged

        setattr(_omni_merge, _MM_MERGE_STATIC_PATCHED, True)
        vllm_utils._merge_multimodal_embeddings = _omni_merge
        _patch_vllm_merge_multimodal_embeddings()
        patched = True

    active_merge = vllm_utils._merge_multimodal_embeddings
    if thinker_mod._merge_multimodal_embeddings is not active_merge:
        thinker_mod._merge_multimodal_embeddings = active_merge
        patched = True

    orig_check = thinker_mod.check_interleaved_audio_video
    if not getattr(orig_check, _MM_MERGE_STATIC_PATCHED, False):

        def _check_interleaved(
            is_video: torch.Tensor,
            is_audio: torch.Tensor,
            num_videos: int,
            num_audios: int,
        ) -> bool:
            """Host-side wrapper around ``check_interleaved_audio_video``.

            Upstream: ``check_interleaved_audio_video`` in
            ``vllm/model_executor/models/qwen2_5_omni_thinker.py``.
            Stayed the same: span-overlap check via ``orig_check`` when both
            video and audio tokens are present (fast-path ``False`` otherwise).
            Adapted for TPU: moves ``is_video`` and ``is_audio`` masks to CPU
            before ``orig_check``'s dynamic ``torch.nonzero`` / span indexing.
            """
            if not num_videos or not num_audios:
                return False
            # Matches upstream GPU path (`input_ids.cpu()` before span check).
            return orig_check(is_video.cpu(), is_audio.cpu(), num_videos, num_audios)

        setattr(_check_interleaved, _MM_MERGE_STATIC_PATCHED, True)
        thinker_mod.check_interleaved_audio_video = _check_interleaved
        patched = True

    if not getattr(
        thinker_mod.merge_interleaved_embeddings, _MM_MERGE_STATIC_PATCHED, False
    ):

        def _merge_interleaved(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: Any,
            is_video: torch.Tensor,
            is_audio: torch.Tensor,
            is_multimodal: torch.Tensor,
        ) -> torch.Tensor:
            """Merge interleaved image/video/audio embeddings using static shapes.

            Upstream: ``merge_interleaved_embeddings`` in
            ``vllm/model_executor/models/qwen2_5_omni_thinker.py`` (imported by
            ``qwen3_omni_moe_thinker.py``).

            Stayed the same:
            - Derives ``is_image = is_multimodal & ~(is_video | is_audio)``.
            - Groups ``multimodal_embeddings`` by ``.modality`` tag
              (``get_mm_embedding_modalities``) into ``("image", "video", "audio")``
              and validates non-zero placeholder counts when a group is empty.

            Adapted for TPU:
            - Replaces dynamic boolean mask assignment
              (``inputs_embeds[modality_mask] = flattened``) with per-modality
              calls to ``thinker_mod._merge_multimodal_embeddings``
              (``_omni_merge``) so each scatter runs at static token-bucket shape.
            """
            is_image = is_multimodal & ~(is_video | is_audio)
            modalities = get_mm_embedding_modalities(multimodal_embeddings)
            grouped: dict[str, list[torch.Tensor]] = {
                "image": [],
                "video": [],
                "audio": [],
            }
            for emb, mod in zip(multimodal_embeddings, modalities):
                if mod is None or mod not in grouped:
                    raise ValueError(
                        f"Unsupported or missing modality {mod!r}. "
                        "Expected one of ('image', 'video', 'audio')."
                    )
                grouped[mod].append(emb)

            for mod_name, modality_mask in (
                ("image", is_image),
                ("video", is_video),
                ("audio", is_audio),
            ):
                modality_embeds = grouped[mod_name]
                if len(modality_embeds) == 0:
                    num_expected = int(modality_mask.sum().item())
                    if num_expected > 0:
                        raise ValueError(
                            f"Attempted to assign 0 {mod_name} tokens to "
                            f"{num_expected} placeholders"
                        )
                else:
                    inputs_embeds = thinker_mod._merge_multimodal_embeddings(
                        inputs_embeds, modality_embeds, modality_mask
                    )
            return inputs_embeds

        setattr(_merge_interleaved, _MM_MERGE_STATIC_PATCHED, True)
        thinker_mod.merge_interleaved_embeddings = _merge_interleaved
        patched = True

    thinker_cls = thinker_mod.Qwen3OmniMoeThinkerForConditionalGeneration
    if not getattr(thinker_cls.embed_input_ids, _MM_MERGE_STATIC_PATCHED, False):
        orig_embed_input_ids = thinker_cls.embed_input_ids

        def _patched_embed_input_ids(
            self: Any,
            input_ids: torch.Tensor,
            multimodal_embeddings: Any = None,
            *,
            is_multimodal: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """TPU wrapper around Thinker ``embed_input_ids``.

            Upstream: ``Qwen3OmniMoeThinkerForConditionalGeneration.embed_input_ids``
            in ``vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py``.

            Stayed the same:
            - Delegates text lookup, interleaved audio-in-video detection,
              DeepStack feature splitting (``embeddings_main`` /
              ``embeddings_multiscale``), and multimodal merge to
              ``orig_embed_input_ids``.

            Adapted for TPU:
            - Clears stale ``deepstack_input_embeds`` before steps that may carry
              no new visual tokens so ``_add_deepstack_embeds`` adds zeros on
              decode/audio steps.
            - Converts ``multimodal_embeddings`` to a mutable ``list`` (upstream
              mutates ``multimodal_embeddings[index] = embeddings_main``) and
              copies ``is_multimodal`` to CPU only when ``multimodal_embeddings``
              is non-empty (for upstream's ``torch.nonzero(is_multimodal)``).
            - Cuts the deferred TorchTPU graph on ``out`` and
              ``deepstack_input_embeds`` after multimodal or DeepStack updates.
            """
            cleared_deepstack = (
                self.use_deepstack and self.deepstack_input_embeds_num_tokens > 0
            )
            if cleared_deepstack:
                self._clear_deepstack_input_embeds(
                    self.deepstack_input_embeds_num_tokens
                )
            if multimodal_embeddings:
                multimodal_embeddings = list(multimodal_embeddings)
                # When has_deepstack_embeddings and not is_interleaved,
                # Qwen3OmniMoeThinkerForConditionalGeneration.embed_input_ids
                # runs `mm_positions = torch.nonzero(is_multimodal, as_tuple=True)[0]`
                # and indexes `is_vision[current_positions] = True`, which
                # requires `is_multimodal` on CPU (matching GPUModelRunner).
                # Decode and text-only steps have empty `multimodal_embeddings`
                # and skip this copy so `is_multimodal` stays on TPU.
                if is_multimodal is not None and is_multimodal.device.type != "cpu":
                    is_multimodal = is_multimodal.cpu()
            out = orig_embed_input_ids(
                self,
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            # Cut the deferred TorchTPU graph after embed_input_ids (including
            # any updated deepstack_input_embeds buffers) so the embedding and
            # deepstack scatter stays isolated at the static token bucket size.
            if (
                (multimodal_embeddings or cleared_deepstack)
                and isinstance(out, torch.Tensor)
                and out.device.type == "tpu"
            ):
                sync_tensors = [out]
                if self.use_deepstack and (
                    self.deepstack_input_embeds_num_tokens > 0 or cleared_deepstack
                ):
                    sync_tensors.extend(self.deepstack_input_embeds)
                synchronize_tensors(sync_tensors, wait=False)
            return out

        setattr(_patched_embed_input_ids, _MM_MERGE_STATIC_PATCHED, True)
        thinker_cls.embed_input_ids = _patched_embed_input_ids
        patched = True

    return patched


def maybe_patch_qwen3_omni_thinker(
    architectures: list[str] | None = None,
    *,
    enforce_eager: bool = False,
) -> bool:
    """Apply TPU patches when a Qwen3-Omni Thinker architecture is loaded."""
    if not architectures or not _QWEN3_OMNI_ARCHS.intersection(architectures):
        return False

    from vllm_omni.model_executor.models.qwen3_omni import (
        qwen3_omni_moe_thinker as thinker_mod,
    )

    patched = _patch_audio_encoder(
        thinker_mod.Qwen3OmniMoeAudioEncoder, enforce_eager=enforce_eager
    )
    patched = _patch_vision_encoder(thinker_mod.Qwen3Omni_VisionTransformer) or patched
    patched = _patch_mm_merge_and_embed_input_ids(thinker_mod) or patched
    if patched:
        logger.info(
            "Applied TPU patch: Qwen3-Omni Thinker encoders and multimodal merge."
        )
    return patched
