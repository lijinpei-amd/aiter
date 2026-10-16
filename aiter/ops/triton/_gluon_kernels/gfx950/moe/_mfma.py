# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The matrix instruction: one MFMA dispatched on the operand pair.

:func:`_dot` issues one MFMA for one K fragment; :func:`_maybe_block_dot` accumulates
a stage's whole-stage K fragments through it. Shared by the pipelined loop in
:mod:`._pipeline` and the fused final drain step in :mod:`.moe_gemm`.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import require_constexpr
from ._types import DotKind

_DK_MFMA: gl.constexpr = gl.constexpr(int(DotKind.MFMA))
_DK_MFMA_SCALED: gl.constexpr = gl.constexpr(int(DotKind.MFMA_SCALED))
_DK_UPCAST_MFMA: gl.constexpr = gl.constexpr(int(DotKind.UPCAST_MFMA))


@gluon.jit
def _dot(
    a,
    a_scale,
    b,
    b_scale,
    acc,
    func_cfg,
    tuning_cfg,
    layout,
    K_PHASE: gl.constexpr = 0,
):
    """The one matrix instruction, dispatched on the operand pair."""
    kind: gl.constexpr = func_cfg.dot_kind()
    a_sel: gl.constexpr = layout.scale_packed_sel(0, K_PHASE)
    b_sel: gl.constexpr = layout.scale_packed_sel(1, K_PHASE)
    any_packed: gl.constexpr = layout.any_scale_packed
    if require_constexpr(kind == _DK_MFMA_SCALED and any_packed):
        if require_constexpr(not func_cfg.has_scale(0)):
            a_scale = gl.full(
                layout.fragment_shape_slot(0, True),
                127,
                gl.uint8,
                gl.amd.cdna4.get_mfma_scale_layout(
                    layout.fragment_layout(0, False),
                    layout.fragment_shape_slot(0, True),
                ),
            )
        if require_constexpr(not func_cfg.has_scale(1)):
            b_scale = gl.full(
                layout.fragment_shape_slot(1, True),
                127,
                gl.uint8,
                gl.amd.cdna4.get_mfma_scale_layout(
                    layout.fragment_layout(1, False),
                    layout.fragment_shape_slot(1, True),
                ),
            )
        # At least one scale fragment arrived as pre-packed dwords; the other, if any,
        # takes the ordinary path through the same instruction.
        out = gl.amd.cdna4.mfma_scaled_packed(
            a=a,
            a_scale=a_scale,
            a_scale_sel=a_sel,
            a_format=func_cfg.mx_format(0),
            b=b,
            b_scale=b_scale,
            b_scale_sel=b_sel,
            b_format=func_cfg.mx_format(1),
            acc=acc,
        )
    elif require_constexpr(kind == _DK_MFMA_SCALED):
        out = gl.amd.cdna4.mfma_scaled(
            a=a,
            a_scale=a_scale,
            a_format=func_cfg.mx_format(0),
            b=b,
            b_scale=b_scale,
            b_format=func_cfg.mx_format(1),
            acc=acc,
        )
    elif require_constexpr(kind == _DK_UPCAST_MFMA):
        out = gl.amd.cdna4.mfma(a, b, acc)
    else:
        gl.static_assert(kind == _DK_MFMA)
        out = gl.amd.cdna4.mfma(a, b, acc)
    return out


_NO_SCALE: gl.constexpr = gl.constexpr(None)


@gluon.jit
def _maybe_block_dot(
    a_frags,
    b_frags,
    acc,
    N_FRAGS: gl.constexpr,
    func_cfg,
    tuning_cfg,
    layout,
    DO_MFMA: gl.constexpr,
    K_PHASE: gl.constexpr = 0,
):
    """Accumulate ``N_FRAGS`` whole-stage K fragments when this stage emits MFMA."""
    if require_constexpr(DO_MFMA):
        for i in gl.static_range(N_FRAGS):
            if require_constexpr(func_cfg.has_scale(0)):
                a_s = a_frags[2 * i + 1]
            else:
                a_s = _NO_SCALE
            if require_constexpr(func_cfg.has_scale(1)):
                b_s = b_frags[2 * i + 1]
            else:
                b_s = _NO_SCALE
            acc = _dot(
                a_frags[2 * i],
                a_s,
                b_frags[2 * i],
                b_s,
                acc,
                func_cfg,
                tuning_cfg,
                layout,
                (K_PHASE + i * tuning_cfg.BLOCK_K // 128) % 2,
            )
    return acc
