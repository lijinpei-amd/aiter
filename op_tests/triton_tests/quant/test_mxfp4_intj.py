# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise MXFP4 wrappers without their Triton bracket launches."""

import pytest
import torch
from intj import make_launcher
from triton.runtime.autotuner import Heuristics
from triton.runtime.jit import JITFunction

import aiter.ops.triton.quant.fused_mxfp4_quant as _wrapper
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
    if request.node.name.split("[")[0].endswith("_matches_triton_reference"):
        return

    def reject(*_args, **_kwargs):
        raise AssertionError("Triton bracket launch was used")

    monkeypatch.setattr(JITFunction, "__getitem__", reject)
    monkeypatch.setattr(Heuristics, "__getitem__", reject)


def _input(rows=2, columns=64):
    return torch.ones((rows, columns), dtype=torch.bfloat16, device="cuda")


def _raw_bracket_launch(kernel, grid, *args, **kwargs):
    # A plain Triton bracket launch, used as the reference implementation:
    # bypasses `launch_tuned`/intj entirely.
    return kernel[grid](*args, **kwargs)


def _bracket_handle(kernel, grid_arg=None):
    """A launcher stand-in that launches ``kernel[grid](...)`` instead."""
    jit, assigned = kernel, set()
    while not isinstance(jit, JITFunction):
        assigned.update(jit.values)
        jit = jit.fn
    names = [p.name for p in jit.params if p.name not in assigned]

    def launch(_device, _stream, *args):
        if grid_arg:
            grid, args = args[:grid_arg], args[grid_arg:]
        else:
            grid, args = args[0], args[1:]
        kernel[grid](**dict(zip(names, args)))

    return launch


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


def test_fused_rms_mxfp4_quant_matches_triton_reference(monkeypatch):
    def _run():
        return fused_rms_mxfp4_quant(
            _input(), _input(1)[0], 1e-6, output_unquantized_inp1=True, inargs="triton"
        )

    monkeypatch.setattr(
        _wrapper,
        "_fused_rms_mxfp4_quant_launch",
        _bracket_handle(_kernels._fused_rms_mxfp4_quant_kernel, grid_arg=1),
    )
    (packed_t, scales_t), norm_t, _, _ = _run()
    monkeypatch.undo()

    (packed_i, scales_i), norm_i, _, _ = _run()
    assert packed_i.shape == (2, 32)
    assert scales_i.shape == (2, 2)
    torch.testing.assert_close(norm_i, _input())
    torch.testing.assert_close(packed_i, packed_t)
    torch.testing.assert_close(scales_i, scales_t)
    torch.testing.assert_close(norm_i, norm_t)


def test_fused_reduce_act_mul_mxfp4_quant_matches_triton_reference(monkeypatch):
    def _run():
        return fused_reduce_act_mul_and_mxfp4_quant(_input(columns=128), "silu")

    monkeypatch.setattr(_wrapper, "launch_tuned", _raw_bracket_launch)
    packed_t, scales_t = _run()[0]
    monkeypatch.undo()

    (packed_i, scales_i), _ = _run()
    assert packed_i.shape == (2, 32)
    assert scales_i.shape == (2, 2)
    torch.testing.assert_close(packed_i, packed_t)
    torch.testing.assert_close(scales_i, scales_t)


def test_fused_reduce_rms_mxfp4_quant_matches_triton_reference(monkeypatch):
    def _run():
        return fused_reduce_rms_mxfp4_quant(
            _input(), _input(1)[0], 1e-6, output_unquantized_inp1=True, args="triton"
        )

    monkeypatch.setattr(
        _wrapper,
        "_reduce_rms_mxfp4_quant_launch",
        _bracket_handle(_kernels._fused_reduce_rms_mxfp4_quant_kernel),
    )
    (packed_t, scales_t), norm_t, _, _, _ = _run()
    monkeypatch.undo()

    (packed_i, scales_i), norm_i, _, _, _ = _run()
    assert packed_i.shape == (2, 32)
    assert scales_i.shape == (2, 2)
    torch.testing.assert_close(norm_i, _input())
    torch.testing.assert_close(packed_i, packed_t)
    torch.testing.assert_close(scales_i, scales_t)
    torch.testing.assert_close(norm_i, norm_t)


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
def test_partial_heuristics_are_accepted_by_intj(name):
    # These heuristics used to be `functools.partial(even_m_n, ...)`, which
    # intj's make_launcher refuses (not a lambda or single-return def with no
    # free variables). They are now module-level defs reading literal keys;
    # construction must succeed.
    make_launcher(getattr(_kernels, name))


@pytest.mark.parametrize(
    "name",
    [
        "_gluon_fused_rms_mxfp4_quant_kernel",
        "_gluon_fused_reduce_rms_mxfp4_quant_kernel",
    ],
)
def test_gluon_partial_heuristics_are_accepted_by_intj(name):
    # The gfx1250 Gluon kernels got the same module-level defs; the lazy
    # launcher analyzes their heuristics at construction, without a gfx1250 GPU.
    from aiter.ops.triton._gluon_kernels.gfx1250.quant import (
        fused_mxfp4_quant as gluon_kernels,
    )

    make_launcher(getattr(gluon_kernels, name))
