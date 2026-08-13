# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon port of the MXFP4 quantisation op used by gemm1's fused epilogue.

This is a line-for-line port of ``_triton_kernels/quant/quant.py::_mxfp4_quant_op``.
It exists only because that function calls ``tl.full``, which in Gluon needs an explicit
layout; every other operation is identical, and it must stay that way -- the contract
of the fused epilogue is that its payload and E8M0 scale are **bit-identical** to what
the standalone ``mxfp4_quant`` launch produces, so ``gemm2`` needs no change.
``test_moe_gemm_a4w4.py::test_gemm1_fused_mxfp4_out`` enforces that byte for byte; if
the Triton op ever changes, that test fails rather than the two drifting silently.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

__all__ = ["mxfp4_quant_gluon"]


@gluon.jit
def mxfp4_quant_gluon(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """fp32 ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` -> (packed E2M1 ``[M, N/2]``, E8M0 ``[M, N/32]``)."""
    EXP_BIAS_FP32: gl.constexpr = 127
    EXP_BIAS_FP4: gl.constexpr = 1
    EBITS_F32: gl.constexpr = 8
    EBITS_FP4: gl.constexpr = 2
    MBITS_F32: gl.constexpr = 23
    MBITS_FP4: gl.constexpr = 1

    max_normal: gl.constexpr = 6
    min_normal: gl.constexpr = 1

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # scale: amax over the 32 emitted columns of the group, rounded up to a power of 2
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    qx = x * quant_scale
    qx = qx.to(tl.uint32, bitcast=True)
    s = qx & 0x80000000
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_exp: gl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: gl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: gl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # The Triton original adds a negative constant to a uint32 tensor; Gluon rejects
    # that mix, so use the two's-complement representative -- uint32 addition is
    # modular, so the result is bit-identical.
    val_to_add: gl.constexpr = (
        ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    ) & 0xFFFFFFFF
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # `tl.full` needs a layout in Gluon; take it from the tensor being merged so the
    # constant lands in the same distribution and no convert_layout is inserted.
    sat_value = gl.full(
        [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE],
        0x7,
        gl.uint8,
        layout=normal_x.type.layout,
    )
    e2m1_value = tl.where(normal_mask, normal_x, sat_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value,
        [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2],
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)
