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
"""Tests for the TPU GateLinear's fp32-accurate tier for fp32 router weights."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_torchtpu.layers.adapter import gate_linear as tpu_gate_linear
from vllm_torchtpu.layers.adapter.gate_linear import TpuGateLinear
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedLinearMethod,
)

BF16, FP32 = torch.bfloat16, torch.float32

# Router shape and top-k as in Beam: hidden 5120, 256 experts, top-8.
ROUTER_EXPERTS, ROUTER_HIDDEN, ROUTER_TOPK = 256, 5120, 8
ROUTER_INIT_STD = 0.02
FP32_ROUTER_MAX_REL_ERROR = 1e-6

# Gates and inputs the tier must leave alone, as (gate kwargs, input dtype,
# non-finite weight). DeepSeek-V4 and Kimi-K3 build bf16-weight gates.
TIER_OFF_CASES = [
    pytest.param(dict(params_dtype=BF16), BF16, False, id="bf16-weight"),
    pytest.param(dict(bias=True), BF16, False, id="bias"),
    pytest.param(dict(out_dtype=BF16), BF16, False, id="bf16-out"),
    pytest.param(dict(), FP32, False, id="fp32-input"),
    pytest.param(dict(), BF16, True, id="non-finite"),
]


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    """A one-rank TP group and a host vLLM config, enough to build a gate.

    The gate is replicated, so it runs no collectives; TPU tests build and
    load it on the host and move it afterwards.
    """
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "_TP", group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_rank", lambda: 0)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


def _router_weight(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(ROUTER_EXPERTS, ROUTER_HIDDEN, generator=generator)
        * ROUTER_INIT_STD
    )


def _router_inputs(num_tokens, *, input_dtype=BF16, non_finite=False):
    """A seeded fp32 router weight and activations."""
    weight = _router_weight()
    if non_finite:
        weight[0, 0] = float("inf")
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(num_tokens, ROUTER_HIDDEN, generator=generator)
    return weight, x.to(BF16).to(input_dtype)


def _gate(*, params_dtype=FP32, out_dtype=FP32, bias=False, quant_config=None):
    """A gate as an fp32-router model builds it (upstream Beam, MiniMax-M2)."""
    return GateLinear(
        ROUTER_HIDDEN, ROUTER_EXPERTS, bias=bias, params_dtype=params_dtype,
        out_dtype=out_dtype, quant_config=quant_config, prefix="mlp.gate",
    )  # fmt: skip


def _load(gate, weight):
    """Loads ``weight`` and runs the post-load hook, as the model loader does."""
    gate.weight.weight_loader(gate.weight, weight.to(gate.weight.dtype))
    if gate.bias is not None:
        generator = torch.Generator().manual_seed(2)
        bias = torch.randn(gate.bias.shape, generator=generator)
        gate.bias.data.copy_(bias.to(gate.bias.dtype))
    gate.quant_method.process_weights_after_loading(gate)
    return gate


def _loaded_gate(weight, **kwargs):
    return _load(_gate(**kwargs), weight)


def _router_error(logits, reference):
    return ((logits.double() - reference).abs().max() / reference.abs().max()).item()


def _topk_flip_rate(logits, reference):
    """Fraction of tokens whose top-k expert set differs from the reference."""
    top = logits.topk(ROUTER_TOPK, -1).indices.sort(-1).values
    ref_top = reference.topk(ROUTER_TOPK, -1).indices.sort(-1).values
    return (top != ref_top).any(-1).float().mean().item()


@pytest.fixture
def bf16x3_calls(monkeypatch):
    """Records each forward that takes the bf16x3 tier."""
    calls = []
    real = tpu_gate_linear._bf16x3_router_logits

    def spy(x, weight_split):
        calls.append(x.shape)
        return real(x, weight_split)

    monkeypatch.setattr(tpu_gate_linear, "_bf16x3_router_logits", spy)
    return calls


def test_fp32_weight_is_split_into_bf16_terms():
    """The [E, H] fp32 weight stays; a [3, E, H] bf16 split sums back to it."""
    weight = _router_weight()
    gate = _loaded_gate(weight)
    assert type(gate) is TpuGateLinear
    torch.testing.assert_close(gate.weight.data, weight, rtol=0, atol=0)
    split = gate._bf16x3_weight
    assert split.shape == (3, ROUTER_EXPERTS, ROUTER_HIDDEN)
    assert split.dtype == BF16
    torch.testing.assert_close(split.double().sum(0), weight.double(), rtol=0, atol=0)
    # Derived from the weight: not in the checkpoint, and no new parameter
    # for expert placement to renumber.
    assert "_bf16x3_weight" not in gate.state_dict()
    assert "_bf16x3_weight" in dict(gate.named_buffers())
    assert [name for name, _ in gate.named_parameters()] == ["weight"]


@torch.no_grad()
def test_fp32_gate_is_fp32_accurate_on_cpu(bf16x3_calls):
    weight, x = _router_inputs(64)
    logits, bias = _loaded_gate(weight)(x)
    assert bf16x3_calls
    assert bias is None and logits.dtype == FP32
    reference = x.double() @ weight.double().T
    assert _router_error(logits, reference) <= FP32_ROUTER_MAX_REL_ERROR


def test_out_dtype_set_later_turns_the_tier_on():
    gate = _gate(out_dtype=None)
    assert not gate.allow_tpu_bf16x3_router_gemm
    gate.set_out_dtype(FP32)
    _load(gate, _router_weight())
    assert gate.allow_tpu_bf16x3_router_gemm
    assert gate._bf16x3_weight is not None


@pytest.mark.parametrize("kwargs, input_dtype, non_finite", TIER_OFF_CASES)
@torch.no_grad()
def test_tier_is_off_for_other_gates_and_inputs(
    kwargs, input_dtype, non_finite, bf16x3_calls
):
    weight, x = _router_inputs(4, input_dtype=input_dtype, non_finite=non_finite)
    gate = _loaded_gate(weight, **kwargs)
    gate(x)
    assert not bf16x3_calls


@torch.no_grad()
def test_tier_is_off_with_the_tpu_linear_method(bf16x3_calls):
    """TPU's linear method, which flips the weight to [H, E], is left alone."""
    weight, x = _router_inputs(4)
    gate = _loaded_gate(weight, quant_config=VllmUnquantizedConfig())
    assert type(gate.quant_method) is VllmUnquantizedLinearMethod
    assert gate.weight.shape == (ROUTER_HIDDEN, ROUTER_EXPERTS)
    gate(x)
    assert not bf16x3_calls


def test_reload_rebuilds_the_split():
    """A reload is split again; non-finite weights turn the tier off and
    finite ones back on."""
    gate = _loaded_gate(_router_weight())
    non_finite = _router_weight()
    non_finite[0, 0] = float("nan")
    _load(gate, non_finite)
    assert not gate.allow_tpu_bf16x3_router_gemm
    assert gate._bf16x3_weight is None
    reloaded = _router_weight().flip(0)
    _load(gate, reloaded)
    assert gate.allow_tpu_bf16x3_router_gemm
    torch.testing.assert_close(
        gate._bf16x3_weight.double().sum(0), reloaded.double(), rtol=0, atol=0
    )


def _tpu_logits(gate, x, compiled):
    gate = gate.to("tpu")
    run = torch.compile(gate, backend="tpu", fullgraph=True) if compiled else gate
    logits, bias = run(x.to("tpu"))
    assert bias is None and logits.dtype == gate.out_dtype
    return logits.cpu().double()


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@torch.no_grad()
def test_fp32_gate_is_fp32_accurate_on_tpu(compiled):
    """The fp32 gate's TPU logits are fp32-accurate and pick the fp32 top-k."""
    weight, x = _router_inputs(1024)
    logits = _tpu_logits(_loaded_gate(weight), x, compiled)
    reference = x.double() @ weight.double().T
    assert _router_error(logits, reference) <= FP32_ROUTER_MAX_REL_ERROR
    assert _topk_flip_rate(logits, reference) == 0


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@torch.no_grad()
def test_stock_fp32_gate_is_not_fp32_accurate_on_tpu(monkeypatch, compiled):
    """Control: upstream's GateLinear loses fp32 precision on TPU (one bf16
    pass), so the accuracy test above is meaningful."""
    monkeypatch.delitem(op_registry_oot, "GateLinear")
    weight, x = _router_inputs(1024)
    stock = _loaded_gate(weight)
    assert type(stock) is GateLinear
    logits = _tpu_logits(stock, x, compiled)
    reference = x.double() @ weight.double().T
    assert _router_error(logits, reference) > FP32_ROUTER_MAX_REL_ERROR
    assert _topk_flip_rate(logits, reference) > 0


@pytest.mark.tpu_test
@pytest.mark.parametrize("kwargs, input_dtype, non_finite", TIER_OFF_CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@torch.no_grad()
def test_other_gates_match_stock_on_tpu(
    monkeypatch, kwargs, input_dtype, non_finite, compiled
):
    """Where the tier is off, the TPU logits are upstream GateLinear's exactly."""
    weight, x = _router_inputs(256, input_dtype=input_dtype, non_finite=non_finite)
    logits = _tpu_logits(_loaded_gate(weight, **kwargs), x, compiled)
    monkeypatch.delitem(op_registry_oot, "GateLinear")
    stock = _loaded_gate(weight, **kwargs)
    assert type(stock) is GateLinear
    assert type(stock.quant_method) is UnquantizedLinearMethod
    stock_logits = _tpu_logits(stock, x, compiled)
    torch.testing.assert_close(logits, stock_logits, rtol=0, atol=0, equal_nan=True)
