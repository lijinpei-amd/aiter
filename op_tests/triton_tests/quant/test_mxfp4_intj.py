# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise MXFP4 wrappers without their Triton bracket launches."""

import pytest
import torch
from intj import make_launcher
from intj.launcher import UnsupportedKernel
from triton.runtime.autotuner import Heuristics
from triton.runtime.jit import JITFunction

from aiter.ops.triton._triton_kernels.quant import fused_mxfp4_quant as _kernels

from aiter.ops.triton.quant.fused_mxfp4_quant import (
    fused_dynamic_mxfp4_quant_moe_sort,
    fused_flatten_mxfp4_quant,
    fused_quant_fp8_sort,
    fused_reduce_act_mul_and_mxfp4_quant,
    fused_reduce_rms_mxfp4_quant,
    fused_rms_mxfp4_quant,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp4_quant, dynamic_nvfp4_quant


@pytest.fixture(autouse=True)
def _reject_triton_brackets(request, monkeypatch):
    if request.node.name.endswith("_stays_on_triton"):
        return

    def reject(*_args, **_kwargs):
        raise AssertionError("Triton bracket launch was used")

    monkeypatch.setattr(JITFunction, "__getitem__", reject)
    monkeypatch.setattr(Heuristics, "__getitem__", reject)


def _input(rows=2, columns=64):
    return torch.ones((rows, columns), dtype=torch.bfloat16, device="cuda")


def test_dynamic_mxfp4_quant_uses_intj():
    packed, scales = dynamic_mxfp4_quant(_input())
    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.cuda.synchronize()


def test_dynamic_nvfp4_quant_uses_intj():
    packed, scales = dynamic_nvfp4_quant(_input())
    assert packed.shape == (2, 32)
    assert scales.shape == (2, 4)
    torch.cuda.synchronize()


def test_fused_flatten_mxfp4_quant_uses_intj():
    packed, scales = fused_flatten_mxfp4_quant(_input().reshape(2, 2, 32))
    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.cuda.synchronize()


def test_fused_rms_mxfp4_quant_stays_on_triton():
    (packed, scales), norm, _, _ = fused_rms_mxfp4_quant(
        _input(), _input(1)[0], 1e-6, output_unquantized_inp1=True, inargs="triton"
    )
    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.testing.assert_close(norm, _input())


def test_fused_reduce_act_mul_mxfp4_quant_stays_on_triton():
    (packed, scales), _ = fused_reduce_act_mul_and_mxfp4_quant(
        _input(columns=128), "silu"
    )
    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.cuda.synchronize()


def test_fused_reduce_rms_mxfp4_quant_stays_on_triton():
    (packed, scales), norm, _, _, _ = fused_reduce_rms_mxfp4_quant(
        _input(), _input(1)[0], 1e-6, output_unquantized_inp1=True, args="triton"
    )
    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.testing.assert_close(norm, _input())


def test_fused_dynamic_mxfp4_quant_moe_sort_uses_intj():
    sorted_ids = torch.arange(32, dtype=torch.int64, device="cuda")
    valid = torch.tensor([32], dtype=torch.int64, device="cuda")
    packed, scales = fused_dynamic_mxfp4_quant_moe_sort(
        _input(32, 128), sorted_ids, valid, token_num=32, topk=1, args="triton"
    )
    assert packed.shape == (32, 64)
    assert scales.shape == (32, 8)
    torch.cuda.synchronize()


def test_fused_quant_fp8_sort_uses_intj():
    sorted_ids = torch.arange(32, dtype=torch.int64, device="cuda")
    valid = torch.tensor([32], dtype=torch.int64, device="cuda")
    quantized, scales = fused_quant_fp8_sort(
        _input(32, 256), sorted_ids, valid, token_num=32
    )
    assert quantized.shape == (32, 256)
    assert scales.shape == (32, 8)
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "name",
    [
        "_fused_rms_mxfp4_quant_kernel",
        "_fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel",
        "_fused_reduce_rms_mxfp4_quant_kernel",
    ],
)
def test_partial_heuristics_are_refused_by_intj(name):
    # These wrappers launch through Triton because their functools.partial
    # heuristics are outside intj's heuristic subset. Convert them if this fails.
    with pytest.raises(UnsupportedKernel, match="must be a lambda or def"):
        make_launcher(getattr(_kernels, name))
