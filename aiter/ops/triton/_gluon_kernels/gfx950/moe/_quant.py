# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon MXFP4 quantisation for gemm1's fused epilogue.

The E8M0 scale is computed exactly as ``_triton_kernels/quant/quant.py::_mxfp4_quant_op``
computes it. The E2M1 conversion is not: where that op spells out the
sign/saturate/denormal/normal round-to-even ladder in ~300 VALU ops, this uses the CDNA4
``v_cvt_scalef32_pk_fp4_f32`` instruction, which does the whole thing in one. The two
agree byte for byte -- ``test_moe_gemm_a4w4.py::test_gemm1_fused_mxfp4_out`` enforces
the payload and the scale exactly, so if the Triton op ever changes that test fails
rather than the two drifting silently.
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
    LANE_ELEMS: gl.constexpr = 0,
):
    """fp32 ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` -> (packed E2M1 ``[M, N/2]``, E8M0 ``[M, N/32]``)."""

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    # scale: amax over the 32 emitted columns of the group, rounded up to a power of 2.
    #
    # Done on the exponent field rather than through log2/floor/exp2. The `+ 0x200000`
    # rounds the mantissa up into the exponent when its top two bits are set, and the
    # reference then zeroes the mantissa -- so by this point the value IS a power of
    # two and its exponent is exact. Extracting the biased exponent E with one bitfield
    # extract makes the rest integer:
    #
    #     unbiased = log2(2**(E-127)) - 2 = E - 129
    #     bs_e8m0  = unbiased + 127       = E - 2
    #     scale    = 2**-unbiased         = bitcast((256 - E) << 23)
    #
    # The reference clamps `unbiased` to [-127, 127]. The upper bound cannot bind (E is
    # 8 bits, so unbiased <= 126); the lower one binds for E < 2, i.e. a zero or
    # subnormal amax, and becomes a single v_max_i32 instead of a compare/select pair.
    # Bit-identical to the float form, which is what `_HW`-free FlyDSL does too.
    if LANE_ELEMS > 1 and MXFP4_QUANT_BLOCK_SIZE % LANE_ELEMS == 0:
        # Everything -- amax, scale and pack -- in ONE reshape lineage.
        #
        # Two constraints force that. (a) The reduction is split: the inner part runs
        # in fp32 so abs rides along as an |v| source modifier, the outer part in
        # integer so it needs no NaN canonicalisation (`tl.max` on f32 emits a
        # `v_max_f32 x, x, x` in front of every cross-lane step). (b) The E8M0 scale is
        # handed to the pack instruction rather than applied as a multiply, and the
        # instruction's operands come out of `tl.split`, whose result carries a
        # SliceLayout of *this* reshape -- a scale computed in any other lineage cannot
        # broadcast against it.
        #
        # Hence the 5-D shape: [M, NQB, OUTER, LANE_ELEMS//2, 2]. The last axis is the
        # emitted E2M1 pair, the two before it are the in-lane fp32 reduction, and
        # OUTER is the cross-lane integer one.
        OUTER: gl.constexpr = MXFP4_QUANT_BLOCK_SIZE // LANE_ELEMS
        x5 = x.reshape(
            BLOCK_SIZE_M, NUM_QUANT_BLOCKS, OUTER, LANE_ELEMS // 2, 2
        )
        a5 = tl.abs(x5)
        part = tl.max(a5, axis=-1, keep_dims=True)
        part = tl.max(part, axis=3, keep_dims=True)
        amax = tl.max(
            part.to(tl.int32, bitcast=True), axis=2, keep_dims=True
        )
        e_biased = ((amax + 0x200000) >> 23) & 0xFF
        e_biased = tl.maximum(e_biased, 2)
        bs_e8m0 = (e_biased - 2).to(tl.uint8)
        # The *forward* scale 2**(E-129) = bitcast(bs_e8m0 << 23), exactly what FlyDSL
        # feeds the instruction. Broadcast to the full tile so that splitting it yields
        # the same SliceLayout as `evens`/`odds`; `fmul x, 1.0` folds away, so the
        # broadcast costs nothing and the per-element multiply is gone.
        hw_scale = ((e_biased - 2) << 23).to(tl.float32, bitcast=True)
        hw5 = gl.full(
            x5.shape, 1.0, tl.float32, layout=x5.type.layout
        ) * hw_scale
        evens, odds = tl.split(x5)
        scale_e, _scale_o = tl.split(hw5)
        old_vdst = gl.full(evens.shape, 0, tl.int32, layout=evens.type.layout)
        packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp4_f32 $0, $2, $3, $4",
            "=v,0,v,v,v",
            [old_vdst, evens, odds, scale_e],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        return (
            (packed & 0xFF).to(tl.uint8).reshape(
                BLOCK_SIZE_M, BLOCK_SIZE_N // 2
            ),
            bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS),
        )
    else:
        amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        e_biased = ((amax + 0x200000) >> 23) & 0xFF
        e_biased = tl.maximum(e_biased, 2)
        bs_e8m0 = (e_biased - 2).to(tl.uint8)
        quant_scale = ((256 - e_biased) << 23).to(tl.float32, bitcast=True)
        qx = x * quant_scale

        # Hardware pack. `v_cvt_scalef32_pk_fp4_f32 vdst, src0, src1, scale` applies the
        # scale, rounds two f32 to E2M1 and packs them into one byte of vdst -- the whole
        # sign/saturate/denormal/normal round-to-even ladder in a single instruction. It is
        # bit-identical to the software sequence it replaced (verified byte for byte over
        # 2M payload bytes) and worth ~29 us on the tuned 4-wave gemm1. Triton has no op
        # for it -- only the upcast direction, amdgpu.scaled_upcast_fp4 -- so it goes in as
        # inline asm.
        #
        # Two things about the asm are load-bearing:
        #
        #   * `old_vdst` is a real input. The instruction writes only the byte selected by
        #     dst_sel and preserves the other three, so it MUST be declared tied
        #     ("=v,0,v,v") rather than seeded with a `v_mov_b32 $0, 0` inside the asm body.
        #     With $0 write-only, LLVM coalesces it across invocations and hoists a later
        #     mov past an earlier result: that silently zeroed exactly 1 in 4 elements,
        #     uniformly across every magnitude bucket and only on the src0 (low nibble)
        #     side -- a 0.750 ratio in each bucket, which is what gave it away.
        #   * the scale operand is the literal 1.0 here because this fallback pre-scales
        #     `qx` with a multiply. The split path above hands the real E8M0 scale to the
        #     instruction instead and has no multiply; it can only do that because every
        #     tensor there descends from one reshape.
        xr = qx.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2)
        evens, odds = tl.split(xr)
        old_vdst = gl.full(evens.shape, 0, tl.int32, layout=evens.type.layout)
        packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp4_f32 $0, $2, $3, 1.0",
            "=v,0,v,v",
            [old_vdst, evens, odds],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        return (
            (packed & 0xFF).to(tl.uint8).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2),
            bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS),
        )
