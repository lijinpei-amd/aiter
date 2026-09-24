# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise Sage quant wrappers without their Triton bracket launches."""

import pytest
import torch
from triton.runtime.jit import JITFunction

from aiter.ops.triton.quant import sage_attention_quant_wrappers as sage
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

_SAGE_KERNELS = (
    sage.sage_quant_v_kernel,
    sage._q_smooth_int8_kernel,
    sage._compute_delta_s_kernel,
    sage.sage_quant_v_fp4_colmajor_kernel,
    sage.sage_quant_v_mxfp4_colmajor_kernel,
    sage.sage_quant_kernel,
    sage._rot_q_kernel,
    sage._rot_k_only_kernel,
    sage._rotate_quantize_q_kernel,
    sage._rotate_quantize_k_kernel,
)


@pytest.fixture(autouse=True)
def _reject_sage_bracket_launch(monkeypatch):
    original = JITFunction.__getitem__

    def reject(self, grid):
        if any(self is kernel for kernel in _SAGE_KERNELS):
            raise AssertionError("Sage Triton bracket launch was used")
        return original(self, grid)

    monkeypatch.setattr(JITFunction, "__getitem__", reject)


def _input():
    torch.manual_seed(0)
    return torch.randn((1, 128, 1, 128), dtype=torch.bfloat16, device="cuda")


def test_sage_v_f4f4_pack_uses_intj():
    v = _input()
    packed, descale = sage.sage_quant_v_f4f4(v)
    assert packed.shape == v.shape and packed.dtype == torch.uint8
    torch.testing.assert_close(descale, v.abs().amax(dim=1).float() / 6.0)


@pytest.mark.skipif(arch_info.get_arch() != "gfx950", reason="native FP4 pack requires gfx950")
def test_sage_v_mxfp4_pack_uses_intj():
    raw, scales = sage.pack_v_mxfp4_colmajor_raw(_input().contiguous())
    assert raw.shape == (sage.fp4_v_raw_buffer_size(1, 128, 1),)
    assert scales.shape == (1, 1, 512)


def test_sage_int8_q_smoothing_uses_intj():
    q = _input()
    q_out, delta = sage._apply_int8_q_smoothing(q, q, 32, "bshd", 0.1)
    assert q_out.shape == q.shape
    assert delta.shape == (1, 1, 4, 128)
    assert torch.isfinite(delta).all()


def test_sage_rotation_smoothing_uses_intj():
    q = _input()
    q_rot, k_rot, delta = sage.rotation_smooth_qk(
        q, q, 32, BLOCK_R=32, q_smoothing=True, sm_scale=0.1, layout="bshd"
    )
    assert q_rot.shape == k_rot.shape == q.shape
    assert delta.shape == (1, 1, 4, 128)


def test_sage_rotate_downcast_uses_intj():
    q = _input()
    q_packed, q_scale, k_packed, k_scale, delta = sage.smooth_rotate_downcast_qk(
        q,
        q,
        32,
        hadamard_rotation=True,
        BLOCK_R=32,
        q_smoothing=True,
        sm_scale=0.1,
        layout="bshd",
    )
    assert q_packed.shape == k_packed.shape == (1, 128, 1, 64)
    assert q_scale.shape == k_scale.shape == (1, 128, 1, 4)
    assert delta.shape == (1, 1, 4, 128)


def test_sage_int8_quant_uses_intj():
    q = _input()
    result = sage.sage_quant(
        q, q, q, get_fp8_e4m3_dtype(), 240.0, BLKQ=32, BLKK=64,
        smooth_k=False, q_smoothing=False, layout="bshd",
    )
    assert result[0].dtype == result[2].dtype == torch.int8
    torch.testing.assert_close(result[5], q.abs().amax(dim=1).float() / 240.0)


def test_sage_mxfp4_quant_uses_intj():
    q = _input()
    result = sage.sage_quant_mxfp4(
        q, q, q, get_fp8_e4m3_dtype(), 240.0, 32, 64,
        q_smoothing=False, layout="bshd",
    )
    assert result[4].dtype == get_fp8_e4m3_dtype()
    torch.testing.assert_close(result[5], q.abs().amax(dim=1).float() / 240.0)


def test_sage_mxfp6_quant_uses_intj():
    q = _input()
    result = sage.sage_quant_mxfp6(
        q, q, q, get_fp8_e4m3_dtype(), 240.0, 32, 64,
        q_smoothing=False, layout="bshd",
    )
    assert result[0].shape == result[2].shape == (1, 128, 1, 96)
    torch.testing.assert_close(result[5], q.abs().amax(dim=1).float() / 240.0)
