# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Direct register loads must preserve every mini-K fragment without using LDS."""

import re

import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe._lds import LDSManager
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.test_moe_gemm_gluon_registers import _register_config


@gluon.jit
def _register_load_copy(
    src, dst, tc: gl.constexpr, operand: gl.constexpr, SCALE: gl.constexpr
):
    # Unused staging allocations must disappear from the compiled kernel.
    lds = LDSManager.alloc(tc.func_cfg, tc)
    if SCALE:
        shape: gl.constexpr = tc.scale_lds_shape_slot(operand)
        layout: gl.constexpr = tc.dot_operand_scale_fragment_layout(operand)
        k_dim: gl.constexpr = 1
        width: gl.constexpr = tc.MINI_BLOCK_K // 32
    else:
        shape: gl.constexpr = tc.payload_lds_shape_slot(operand)
        layout: gl.constexpr = tc.dot_operand_fragment_layout(operand)
        k_dim: gl.constexpr = 1 - operand
        width: gl.constexpr = tc.MINI_BLOCK_K // tc.func_cfg.pack_divisor(operand)
    rows = gl.arange(0, shape[0], layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, shape[1], layout=gl.SliceLayout(0, layout))
    offsets = rows[:, None] * shape[1] + cols[None, :]
    if SCALE:
        fragments = lds.buffer_load_scale(operand, False, None, 0, src, offsets, 16)
    else:
        fragments = lds.buffer_load_payload(operand, False, None, 0, src, offsets, 16)
    for mini in gl.static_range(tc.num_k_slots_per_tile()):
        if k_dim == 0:
            fragment_offsets = gl.amd.slice(
                offsets, [width, shape[1]], [mini * width, 0]
            )
        else:
            fragment_offsets = gl.amd.slice(
                offsets, [shape[0], width], [0, mini * width]
            )
        gl.store(dst + fragment_offsets, fragments[mini])


@pytest.mark.parametrize("operand", [0, 1], ids=["a", "b"])
@pytest.mark.parametrize("scale", [False, True], ids=["payload", "scale"])
@pytest.mark.parametrize("mini_k", [128, 256])
def test_register_load_does_not_stage_through_lds(operand, scale, mini_k):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    config = dict(
        _register_config("mxfp4"),
        MINI_BLOCK_K=mini_k,
        B_IN_REG=True,
        B_PRESHUFFLED=True,
        A_SCALE_IN_REG=True,
        B_SCALE_IN_REG=True,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
    )
    tc = host._probe_tuning_config(config, DtypeQuant.MXFP4, DtypeQuant.MXFP4)
    shape = (
        tc.scale_lds_shape_slot(operand)
        if scale
        else tc.payload_lds_shape_slot(operand)
    )
    elements = shape[0] * shape[1]
    src = (torch.arange(elements + 16, device="cuda") % 251).to(torch.uint8)
    dst = torch.empty(elements, dtype=torch.uint8, device="cuda")
    compiled = _register_load_copy[(1,)](
        src, dst, tc, operand, scale, num_warps=tc.num_warps()
    )
    torch.testing.assert_close(dst, src[16:], rtol=0, atol=0)
    assert compiled.metadata.shared == 0
    assert "amdg.buffer_load" in compiled.asm["ttgir"]
    assert re.search(r"\bbuffer_load_", compiled.asm["amdgcn"])
    assert not re.search(r"\bds_(?:read|write)\w*", compiled.asm["amdgcn"])
