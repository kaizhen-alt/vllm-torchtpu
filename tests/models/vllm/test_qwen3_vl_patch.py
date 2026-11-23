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
"""Tests for the Qwen3-VL processor patches."""

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from vllm.config import ModelConfig
from vllm.model_executor.models import qwen3_vl
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import InputProcessingContext

from vllm_torchtpu.models.vllm import qwen3_vl_patch


def test_maybe_patch_qwen3_vl_video_prompt_expands_trained_format(
    monkeypatch: pytest.MonkeyPatch,
):
    # The patch rebinds a class attribute; restore it after the test.
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(
        processor_cls,
        "_expands_only_video_token",
        processor_cls._expands_only_video_token,
    )
    monkeypatch.setattr(
        processor_cls, "_tpu_trained_video_prompt", False, raising=False
    )
    # Only the configs and tokenizer are needed, not the weights.
    path = snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"]
    )
    model_config = ModelConfig(model=path, limit_mm_per_prompt={"image": 0, "video": 1})
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_video_prompt(model_config) is None

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    # 4 frames at 2 fps, no resampling: two temporal patches of 2 frames.
    video = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 2.0,
        "total_num_frames": 4,
        "frames_indices": [0, 1, 2, 3],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    prompt = "<|vision_start|><|video_pad|><|vision_end|>Hi"
    out = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"video": [(video, metadata)]}),
        hf_processor_mm_kwargs={},
    )
    tokenizer = processor.info.get_tokenizer()
    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    text = tokenizer.decode([t for t in out["prompt_token_ids"] if t != video_token_id])
    # Per frame group: timestamp, then its own vision markers; no outer pair.
    assert text == (
        "<0.2 seconds><|vision_start|><|vision_end|>"
        "<1.2 seconds><|vision_start|><|vision_end|>Hi"
    )


def _processor(limits):
    path = snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct", allow_patterns=["*.json", "*.txt"]
    )
    model_config = ModelConfig(model=path, limit_mm_per_prompt=limits)
    return model_config, MULTIMODAL_REGISTRY.create_processor(model_config)


def _assert_same_outputs(got, want, modalities):
    assert got["prompt_token_ids"] == want["prompt_token_ids"]
    for modality in modalities:
        for got_item, want_item in zip(
            got["mm_kwargs"][modality], want["mm_kwargs"][modality], strict=True
        ):
            assert got_item.keys() == want_item.keys()
            for key, elem in want_item.items():
                if isinstance(elem.data, torch.Tensor):
                    assert torch.equal(got_item[key].data, elem.data)
                else:
                    assert got_item[key].data == elem.data


def test_maybe_patch_qwen3_vl_image_hf_text_keeps_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """Images skip the HF dummy prompt; the prompt ids and image fields stay."""
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(processor_cls, "_get_hf_mm_text", processor_cls._get_hf_mm_text)
    monkeypatch.setattr(
        processor_cls, "_tpu_image_hf_text_skipped", False, raising=False
    )
    model_config, processor = _processor({"image": 2, "video": 1})
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        for h, w in ((64, 96), (128, 64))
    ]
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * 2 + "Hi"

    def run():
        mm_items = processor.info.parse_mm_data({"image": images})
        return processor(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})

    want = run()
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_image_hf_text(model_config) is None
    assert processor._get_hf_mm_text({"image": 2}) is None
    assert processor._get_hf_mm_text({"image": 1, "video": 1})
    got = run()
    hf_processor_cls = type(processor.info.get_hf_processor())
    monkeypatch.setattr(hf_processor_cls, "__call__", lambda self, **kw: None)
    assert processor._get_hf_mm_text({"image": 2})  # its own call keeps the text
    _assert_same_outputs(got, want, ["image"])


def test_maybe_patch_qwen3_vl_video_hf_text_keeps_outputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """Videos skip the HF text vLLM drops; the prompt ids and video fields stay."""
    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    monkeypatch.setattr(
        processor_cls,
        "_apply_hf_processor_main",
        processor_cls._apply_hf_processor_main,
    )
    monkeypatch.setattr(
        processor_cls, "_tpu_video_hf_text_skipped", False, raising=False
    )
    monkeypatch.setattr(
        InputProcessingContext,
        "call_hf_processor",
        InputProcessingContext.call_hf_processor,
    )
    model_config, processor = _processor({"image": 1, "video": 1})
    rng = np.random.default_rng(0)
    video = rng.integers(0, 256, (4, 64, 96, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 2.0,
        "total_num_frames": 4,
        "frames_indices": [0, 1, 2, 3],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    image = Image.fromarray(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    prompt = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>Hi"
    )

    def run():
        mm_items = processor.info.parse_mm_data(
            {"image": [image], "video": [(video, metadata)]}
        )
        return processor(prompt, mm_items=mm_items, hf_processor_mm_kwargs={})

    tokenized = []
    tokenizer_cls = type(processor.info.get_tokenizer())
    upstream_tokenize = tokenizer_cls.__call__

    def tokenize(self, text=None, *args, **kwargs):
        tokenized.append(text)
        return upstream_tokenize(self, text, *args, **kwargs)

    monkeypatch.setattr(tokenizer_cls, "__call__", tokenize)
    want = run()
    calls = len(tokenized)
    assert qwen3_vl_patch.maybe_patch_qwen3_vl_video_hf_text(model_config) is None
    got = run()
    assert len(tokenized) - calls == calls - 1  # the video's text is not tokenized
    _assert_same_outputs(got, want, ["image", "video"])
