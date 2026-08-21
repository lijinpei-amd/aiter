# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950 / CDNA4) MoE grouped GEMM kernels.

Two entry points over **one** ``@gluon.jit`` body:

* :func:`_moe_gluon_gemm1` -- gemm1 + fused activation. Gathers tokens according to the
  routing metadata and writes its output contiguously (dense, expert-sorted
  ``[n_gates, I]``), optionally already quantised to gemm2's operand-A format.
* :func:`_moe_gluon_gemm2` -- gemm2 + router-combine-weight multiply. Reads
  contiguously, multiplies by ``gammas``, writes contiguously.

They differ only in the ``KernelFuncConfig`` instantiated in-kernel. The reduce/combine
step stays in the existing ``moe/reduce.py::reduce_grouped``.

Operand dtypes are independent -- everything derived from one (LDS element type, tile
shape, copy width, ``k_width``, the MFMA shape) is per operand, because the mixed pairs
are real:

===============  =============  ==================================================
 A                B              dot
===============  =============  ==================================================
 MXFP4            MXFP4          mfma_scaled e2m1 x e2m1        (a4w4)
 MXFP8            MXFP8          mfma_scaled e4m3 x e4m3        (a8w8)
 MXFP8 / FP8      MXFP4          mfma_scaled e4m3 x e2m1        (a8w4)
 BF16             BF16           mfma
 BF16             MXFP4          scaled_upcast + mfma -- refused, see gluon_supported
===============  =============  ==================================================

FP8 without a scale still goes through ``mfma_scaled`` with ``a_scale=None``: the
backend folds the synthesized unit scales back into ``V_MFMA_*_F8F6F4`` and only that
path reaches the double-rate K=64/128 pipes. Its per-tensor scale is applied to the raw
accumulator in the epilogue instead.

Memory layout contract (the wrapper hard-asserts it), with ``pack`` = 2 for MXFP4 and
1 otherwise:

* activations  ``(M, K/pack)`` row-major, ``stride(-1) == 1``
* weights      ``(E, K/pack, N)`` with ``stride(-2) == 1``  (K contiguous)
* scales       ``(E, K/32, N)`` strided view, K contiguous, strides carried explicitly
* stage 1: ``N = 2*I``, ``K = H``; stage 2: ``N = H``, ``K = I``
* stage 1 gate/up arrive pre-fused and column-interleaved (even = gate, odd = up)

``N`` and ``K`` are compile-time: it is what makes the pipeline drain, the mini-tile
decomposition and the rotating buffer index fully static. The in-scope models present
only a handful of distinct ``(N, K)`` pairs, so this does not flood the Triton cache the
way a constexpr batch size would.

There is no split-K (``moe_op_gemm_a4w4.py`` already hardcodes ``split_k == 1``), and
fusing the activation and the output quant into the epilogue forecloses turning it back
on -- partial sums cannot be activated, and the MX group amax needs the final value.
The output buffer still carries the leading ``(1, M, N)`` axis that ``reduce_grouped``
indexes.
"""

import os
from typing import NamedTuple

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import field_at as _at
from ._lang import optional as _opt
from ._lang import require_constexpr
from ._lang import unwrap as _v
from ._lang import unwrap_attr as _cv
from ._lds import LDSManager
from ._quant import mxfp4_quant_gluon
from ._types import DotKind, DtypeQuant, TileSched

#: EXPERIMENT -- put the LDS->register reads in the ``mfma`` warp-pipeline region
#: instead of ``mem``. By default a slot's region split is
#:   mfma: the MFMAs                       (16 ops)
#:   mem : ds_read + global->LDS fills     (40 ops)
#: which is badly unbalanced, so the wave group in the mfma region idles at the
#: border. Moving the reads across gives 48 / 8 the other way; whether that is
#: better is an empirical question, hence the flag.
_DS_IN_MFMA: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_DS_IN_MFMA", "0"))
)

#: EXPERIMENT -- spread the global->LDS fills one per slot instead of bunching them on
#: the first row and column of the slot grid. The default schedule gives slot (0,0) two
#: fills and every slot with ``mi != 0 and ni != 0`` none, so at NM=NN=2 one of the four
#: ``mem`` regions is empty yet still costs a barrier. Mirrors the reference kernel in
#: gfx950-gluon-tutorials .../a4w4/v2_mfma32x32x64, where each of the four region pairs
#: moves exactly one tile and commits one group.
_EVEN_FILL: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_EVEN_FILL", "0"))
)

#: EXPERIMENT ONLY -- drop the matrix instructions, passing the accumulator through.
#: Nothing then consumes the LDS reads either, so DCE takes those with it; together with
#: AITER_TRITON_MOE_GLUON_NO_FILL this walks the loop down to its empty skeleton.
#: EXPERIMENT -- how many operands' LDS reads move into the ``mfma`` region:
#: 0 = none (both in ``mem``), 1 = A only, 2 = A and B. Finer than _DS_IN_MFMA,
#: which is all-or-nothing.
_DS_MOVE: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_DS_MOVE", "0"))
)

#: EXPERIMENT -- priority given to the ``mem`` warp-pipeline region (default 1).
#: Setting it to 0 makes both regions equal priority, disabling the ping-pong's
#: asymmetric s_setprio.
_MEM_PRIO: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_MEM_PRIO", "1"))
)

#: EXPERIMENT -- put the global->LDS copies in the ``mfma`` region instead of ``mem``,
#: the mirror of _DS_MOVE. Leaves the LDS reads alone in ``mem``.
_FILL_IN_MFMA: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_FILL_IN_MFMA", "0"))
)

_NO_MFMA: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_NO_MFMA", "0"))
)

_DQ_MXFP4: gl.constexpr = gl.constexpr(int(DtypeQuant.MXFP4))
_TS_XCD_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.XCD_GROUP_M))
_TS_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.GROUP_M))
_DK_MFMA: gl.constexpr = gl.constexpr(int(DotKind.MFMA))
_DK_MFMA_SCALED: gl.constexpr = gl.constexpr(int(DotKind.MFMA_SCALED))
_DK_UPCAST_MFMA: gl.constexpr = gl.constexpr(int(DotKind.UPCAST_MFMA))

__all__ = [
    "MoeKernelConfig",
    "_moe_gluon_gemm1",
    "_moe_gluon_gemm2",
    "moe_gemm_launch_metadata",
]


class MoeKernelConfig(NamedTuple):
    """Every compile-time knob, as **four** leaves.

    A ``@gluon.aggregate`` cannot be a kernel argument, so the config is *constructed
    twice*: once on the host off these same numbers (for the grid tuple), once in-kernel
    (for the layouts). Keeping the arithmetic in ``KernelTuningConfig`` is what stops the
    two from drifting.

    ``func`` and ``tuning`` are whole plain-Python NamedTuples inside a single
    ``gl.constexpr`` rather than 36 separate constexpr fields: Triton's argument
    specializer walks every leaf of a tuple argument on **every launch**, and at decode
    the launch path is the critical path (a T=1 grouped GEMM is a ~30 us kernel behind a
    ~100 us host issue). Four leaves instead of thirty-eight is worth the indirection.
    """

    func: gl.constexpr  # FuncSpec
    tuning: gl.constexpr  # TuningSpec
    N: gl.constexpr
    K: gl.constexpr


def moe_gemm_launch_metadata(grid, kernel, args):
    ret = {}
    cfg = args["cfg"]
    N = _cv(cfg.N)
    K = _cv(cfg.K)
    hist = args["rt"].expt_hist
    n_tokens = float(hist.sum()) if hist is not None else None
    w = args["b"].ptr
    n_w_bytes = (
        (w.numel() * w.element_size() // hist.numel()) * (hist > 0).sum()
        if hist is not None
        else w.numel() * w.element_size()
    )
    ret["name"] = f"{kernel.name} [M={n_tokens}, N={N}, K={K}]"
    if n_tokens is not None:
        ret["flops32"] = 2.0 * n_tokens * N * K
        y = args["res"].ptr
        x = args["a"].ptr
        ret["bytes"] = int(
            n_tokens * x.shape[-1] * x.element_size()
            + n_tokens * y.shape[-1] * y.element_size()
            + n_w_bytes
        )
    return ret


# `cfg` is a flattened tuple argument, so its leaves never show up in
# `specialization.constants` under a usable key; the two entry-point names already
# distinguish the stages in a profile.
_moe_gemm_repr_fields = []


@gluon.jit
def _build_configs(cfg, FuncCfgT: gl.constexpr, TuningCfgT: gl.constexpr):
    f: gl.constexpr = cfg.func
    t: gl.constexpr = cfg.tuning
    func_cfg = FuncCfgT(
        _at(f, 0),
        _at(f, 1),
        _at(f, 2),
        _at(f, 3),
        _at(f, 4),
        _at(f, 5),
        _at(f, 6),
        _at(f, 7),
        _at(f, 8),
        _at(f, 9),
        _at(f, 10),
    )
    tuning_cfg = TuningCfgT(
        func_cfg,
        _at(t, 0),
        _at(t, 1),
        _at(t, 2),
        _at(t, 3),
        _at(t, 4),
        _at(t, 5),
        _at(t, 6),
        _at(t, 7),
        _at(t, 8),
        _at(t, 9),
        _at(t, 10),
        _at(t, 11),
        _at(t, 12),
        _at(t, 13),
        _at(t, 14),
        _at(t, 15),
        _at(t, 16),
        _at(t, 17),
        _at(t, 18),
        _at(t, 19),
        _at(t, 20),
        _at(t, 21),
        _at(t, 22),
        _at(t, 23),
        _at(t, 24),
    )
    return func_cfg, tuning_cfg


@gluon.jit
def _gather_rows(
    rt,
    block_id,
    M_e,
    start_m,
    layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    ROW_OFF: gl.constexpr,
    MINI_M: gl.constexpr,
    HAS_GATHER: gl.constexpr,
):
    """Row indices of one mini-M block, already resolved through the gather table."""
    offs = BLOCK_M * block_id + ROW_OFF + gl.arange(0, MINI_M, layout=layout)
    live = offs < M_e
    if require_constexpr(HAS_GATHER):
        # gather_indx is uint16 when n_gates <= 65535, else int32; it indexes gates, so
        # divide by n_expts_act to get the token row.
        rows = (
            gl.load(rt.gather_indx + start_m + offs, mask=live, other=0)
            // rt.n_expts_act
        )
    else:
        rows = start_m + gl.where(live, offs, 0)
    return rows.to(gl.int32)


@gluon.jit
def _dot(a, a_scale, b, b_scale, acc, func_cfg):
    """The one matrix instruction, dispatched on the operand pair."""
    kind: gl.constexpr = func_cfg.dot_kind()
    if require_constexpr(kind == _DK_MFMA_SCALED):
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


@gluon.jit
def _mini_scale_off(base, step, HAS_SCALE: gl.constexpr):
    """Advance a register-path scale offset, or pass the None sentinel through."""
    if require_constexpr(HAS_SCALE):
        out = base + step
    else:
        out = base
    return out


_NO_SCALE: gl.constexpr = gl.constexpr(None)


@gluon.jit
def _opt_at(t, i: gl.constexpr, PRESENT: gl.constexpr):
    """Index a per-mini-block tuple, or pass the absent-scale sentinel through."""
    if require_constexpr(PRESENT):
        out = t[i]
    else:
        out = _NO_SCALE
    return out


@gluon.jit
def _a_block_frags(lds, buf_idx, mi: gl.constexpr, scale_ptr, scale_offs, func_cfg, tc):
    """Every LDS->register A fragment of one mini-M block, all mini-K, as a flat tuple.

    Two entries per mini-K step -- payload, scale -- so :func:`_pipeline_step` can hold
    the tail of one stage across a pipeline iteration and hand it to :func:`_block_dot`
    one stage later.

    The pair is present even for an operand with no scale: the tuple is loop-carried, so
    Triton asks every element for its ``.type`` and a ``None`` there aborts codegen. An
    absent scale therefore parks the payload's own SSA value in the slot -- a duplicate
    reference costs no register -- and :func:`_block_dot` puts the ``None`` back from the
    same compile-time predicate.
    """
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = func_cfg.has_scale(0)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        a, a_s = lds.load_a_frag(
            buf_idx, mi, i, scale_ptr, _mini_scale_off(scale_offs, i * SK_MINI, HAS)
        )
        if require_constexpr(HAS):
            slot = a_s
        else:
            slot = a
        frags = frags + (a, slot)
    return frags


@gluon.jit
def _b_block_frags(lds, buf_idx, ni: gl.constexpr, scale_ptr, scale_offs, func_cfg, tc):
    """The operand-B mirror of :func:`_a_block_frags`, over one mini-N block."""
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = func_cfg.has_scale(1)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        b, b_s = lds.load_b_frag(
            buf_idx, ni, i, scale_ptr, _mini_scale_off(scale_offs, i * SK_MINI, HAS)
        )
        if require_constexpr(HAS):
            slot = b_s
        else:
            slot = b
        frags = frags + (b, slot)
    return frags


@gluon.jit
def _take_pairs(frags, LO: gl.constexpr, N: gl.constexpr):
    """Mini-K steps ``[LO, LO+N)`` of a one-operand fragment tuple, as a fresh tuple.

    Built element-wise rather than sliced: a loop-carried tuple arrives as a
    ``tl.tuple``, which indexes but does not slice.
    """
    out = ()
    for i in gl.static_range(N):
        out = out + (frags[2 * (LO + i)], frags[2 * (LO + i) + 1])
    return out


@gluon.jit
def _block_dot(a_frags, b_frags, acc, N_MINI: gl.constexpr, func_cfg):
    """The MFMAs of one mini (M, N) block over ``N_MINI`` mini-K steps."""
    if require_constexpr(_NO_MFMA):
        return acc
    for i in gl.static_range(N_MINI):
        if require_constexpr(func_cfg.has_scale(0)):
            a_s = a_frags[2 * i + 1]
        else:
            a_s = _NO_SCALE
        if require_constexpr(func_cfg.has_scale(1)):
            b_s = b_frags[2 * i + 1]
        else:
            b_s = _NO_SCALE
        acc = _dot(a_frags[2 * i], a_s, b_frags[2 * i], b_s, acc, func_cfg)
    return acc


@aggregate
@strip_annotate
class _PipelineConst:
    """The loop-*invariant* half of a K-pipeline step.

    Hoisted once before the loop: the LDS buffers, the HBM scale bases, the tile offsets
    every stage reuses, and the per-stage pointer increments. Split from
    :class:`_PipelineState` so the loop-carried set is exactly the values that have to
    cross the back edge -- anything parked in here by mistake would otherwise be threaded
    through every iteration for nothing.

    Every offset grid is a *tuple*, one entry per mini block: ``num_mini_m()`` for the A
    side, ``num_mini_n()`` for the B side. Each entry is laid out for its own mini tile's
    copy layout, which is what keeps every per-mini-block direct-to-LDS copy as wide and
    as coalesced as the whole-tile copy it replaces.

    ``a_scale_offs``/``b_scale_offs`` serve *both* sides: the direct-to-LDS copy and, for
    the register fallback, the consume-side load. Both walk K by advancing their scalar
    base pointer in :class:`_PipelineState` instead of the offset grid, so this tile grid
    is computed once and never moves.

    The two ``*_scale_stride_k`` are lifted out of the operand tuples rather than kept as
    ``a``/``b``: the non-quantised operand types have no such field at all, so it can only
    be read behind the ``has_scale`` predicate that the caller already evaluates.
    """

    lds: LDSManager
    a_offs: tl.tuple | tuple
    b_offs: tl.tuple | tuple
    a_scale_offs: tl.tuple | tuple | gl.constexpr
    b_scale_offs: tl.tuple | tuple | gl.constexpr
    a_scale_stride_k: gl.tensor | gl.constexpr
    b_scale_stride_k: gl.tensor | gl.constexpr
    a_step: gl.constexpr
    b_step: gl.constexpr
    s_step: gl.constexpr
    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig

    @gluon.constexpr_function
    def __init__(
        self,
        lds,
        a_offs,
        b_offs,
        a_scale_offs,
        b_scale_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
    ):
        self.lds = lds
        self.a_offs = a_offs
        self.b_offs = b_offs
        self.a_scale_offs = _opt(a_scale_offs)
        self.b_scale_offs = _opt(b_scale_offs)
        self.a_scale_stride_k = _opt(a_scale_stride_k)
        self.b_scale_stride_k = _opt(b_scale_stride_k)
        # re-wrapped: the frontend unwraps a constexpr argument before a
        # constexpr_function sees it, so these arrive as plain ints
        self.a_step = gl.constexpr(_v(a_step))
        self.b_step = gl.constexpr(_v(b_step))
        self.s_step = gl.constexpr(_v(s_step))
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg


@aggregate
@strip_annotate
class _PipelineState:
    """The loop-*variant* half: everything one ``BLOCK_K`` stage hands to the next.

    Every address here is in HBM -- the LDS side of the pipeline is entirely inside
    ``_PipelineConst.lds``. The ``*_hbm_ptr`` are the copy *sources*, walked one
    ``BLOCK_K`` per stage; the ``*_scale_read_offs`` are the consume-side offsets used
    by the register fallback, when a scale tile is too small to be written to LDS
    coalesced and is loaded straight from HBM at MFMA time instead.

    Bundled rather than passed as a flat tuple because the two advance at *different*
    points in a step -- the copy side before the fill is issued, the read side one stage
    later when its fragments are consumed -- and keeping them adjacent is what makes it
    obvious that advancing only one of the pair is the bug.

    ``a_frags``/``b_frags`` hold the fragments read one stage ahead of the MFMAs that
    take them, ``num_mini_m()`` (resp. ``num_mini_n()``) blocks of ``num_prefetch_mini()``
    payload/scale pairs. ``acc`` is one ``MINI_BLOCK_M x MINI_BLOCK_N`` accumulator per
    mini block, in ``mi * num_mini_n() + ni`` order: a dot writes a whole tensor, so a
    per-mini-block MFMA needs a per-mini-block accumulator -- there is no way to write
    back into a slice of a single ``BLOCK_M x BLOCK_N`` one.
    """

    a_hbm_ptr: gl.tensor
    b_hbm_ptr: gl.tensor
    a_scale_hbm_ptr: gl.tensor | gl.constexpr
    b_scale_hbm_ptr: gl.tensor | gl.constexpr
    a_scale_read_ptr: gl.tensor | gl.constexpr
    b_scale_read_ptr: gl.tensor | gl.constexpr
    a_frags: tl.tuple | tuple
    b_frags: tl.tuple | tuple
    acc: tl.tuple | tuple

    # constexpr_function, not @gluon.jit: Triton's aggregate metaclass calls __init__
    # itself from outside kernel scope, which a JITFunction refuses, and a bare Python
    # function trips its "Unsupported function referenced" member walk. Every argument
    # is already a Triton value (or a constexpr None for an absent scale) built by the
    # caller inside the kernel, so there is nothing to trace here anyway.
    @gluon.constexpr_function
    def __init__(
        self,
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_ptr,
        b_scale_read_ptr,
        a_frags,
        b_frags,
        acc,
    ):
        self.a_hbm_ptr = a_hbm_ptr
        self.b_hbm_ptr = b_hbm_ptr
        # _opt, not a bare assign: an operand with no scale arrives as raw Python None
        # (a8w8, bf16), which the field annotation rejects -- it has to be constexpr.
        self.a_scale_hbm_ptr = _opt(a_scale_hbm_ptr)
        self.b_scale_hbm_ptr = _opt(b_scale_hbm_ptr)
        self.a_scale_read_ptr = _opt(a_scale_read_ptr)
        self.b_scale_read_ptr = _opt(b_scale_read_ptr)
        self.a_frags = a_frags
        self.b_frags = b_frags
        self.acc = acc


@gluon.constexpr_function
def _even_fill(NM, NN):
    """Whether the even fill distribution applies.

    It needs at least as many slots as fills (``NM * NN >= NM + NN``, i.e. both axes
    split), so that no slot has to carry two. Below that the legacy schedule is already
    the only one available and the two agree.
    """
    NM, NN = _v(NM), _v(NN)
    return _EVEN_FILL and NM * NN >= NM + NN and NM > 1 and NN > 1


@gluon.constexpr_function
def _fill_order(NM, NN, EVEN):
    """The ``NM + NN`` mini-block fills of one stage as ``(is_a, tile)``, in issue order.

    Even: A and B interleaved -- A(0), B(0), A(1), B(1) ... -- so that consecutive slots
    alternate operand and the byte volume per slot stays as level as two tile sizes allow.
    Legacy: A(0), B(0), B(1) .. B(NN-1), A(1) .. A(NM-1), which is what falls out of
    "fill A(mi) at ni == 0, fill B(ni) at mi == 0" walked in slot order.
    """
    NM, NN = _v(NM), _v(NN)
    if _v(EVEN):
        out = []
        for i in range(max(NM, NN)):
            if i < NM:
                out.append((1, i))
            if i < NN:
                out.append((0, i))
        return out
    return [(1, 0)] + [(0, j) for j in range(NN)] + [(1, i) for i in range(1, NM)]


@gluon.constexpr_function
def _fill_group_pos(is_a, i, NM, NN):
    """Position of one mini-block fill's commit group inside its stage."""
    is_a, i = _v(is_a), _v(i)
    order = _fill_order(NM, NN, _even_fill(NM, NN))
    return order.index((1 if is_a else 0, i))


@gluon.constexpr_function
def _slot_fill_pos(mi, ni, NM, NN):
    """Which fill (position in :func:`_fill_order`) slot ``(mi, ni)`` issues, or None.

    Even: one fill per slot, in flat slot order -- exactly the tutorial's layout, where
    each of the four ``mfma``/``mem`` region pairs moves one tile and commits one group.
    Legacy: A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``, which loads slot (0, 0)
    with two fills and leaves every slot off the first row and column with none.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if not _even_fill(NM, NN):
        return None  # legacy issues from the (ni == 0) / (mi == 0) predicates directly
    s = mi * NN + ni
    return s if s < NM + NN else None


@gluon.constexpr_function
def _fill_tile_of(pos, NM, NN, EVEN, want_a):
    """Tile index of fill ``pos`` if it is an A (resp. B) fill, else None.

    Two constexpr calls rather than unpacking a tuple inside the unrolled slot loop,
    where the binding would be a reassignment.
    """
    pos = _v(pos)
    if pos is None or not _v(EVEN):
        return None
    is_a, tile = _fill_order(NM, NN, True)[pos]
    return tile if bool(is_a) == bool(_v(want_a)) else None


@gluon.constexpr_function
def _read_a_tile(mi, ni, NM, NN):
    """Operand-A mini block slot ``(mi, ni)`` reads out of LDS, or None.

    Even: the same one-per-slot assignment the fills use, so a slot's fill and its read
    name the *same* position in :func:`_fill_order`. Its wait then works out to
    ``G - 1 - s + STAGES_BETWEEN * G + s`` -- the ``s`` cancels and every slot waits on
    the same constant, which is exactly the uniform ``wait_group`` the reference kernel
    uses. Legacy: A(mi) at ``ni == 0``, which bunches both reads onto slot (0, 0).
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if _even_fill(NM, NN):
        return _fill_tile_of(_slot_fill_pos(mi, ni, NM, NN), NM, NN, True, True)
    return mi if ni == 0 else None


@gluon.constexpr_function
def _read_b_tile(mi, ni, NM, NN):
    """Operand-B mini block slot ``(mi, ni)`` reads out of LDS, or None."""
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if _even_fill(NM, NN):
        return _fill_tile_of(_slot_fill_pos(mi, ni, NM, NN), NM, NN, True, False)
    return ni if mi == 0 else None


@gluon.constexpr_function
def _fills_before(mi, ni, NM, NN, ANY):
    """Groups this stage has already committed when slot ``(mi, ni)`` is reached."""
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if not _v(ANY):
        return 0
    if _even_fill(NM, NN):
        return min(mi * NN + ni, NM + NN)
    n_a = mi if ni == 0 else mi + 1
    n_b = ni if mi == 0 else NN
    return n_a + n_b


@gluon.constexpr_function
def _slot_wait(mi, ni, NM, NN, STAGES_BETWEEN, ANY_FILL):
    """Outstanding-group count that retires everything slot ``(mi, ni)`` is about to read.

    A group is retired once ``wait_group(n)`` leaves at most ``n`` behind it. Counting
    forward from the target group: the rest of its own stage, then ``STAGES_BETWEEN``
    whole stages, then whatever the current stage has committed so far. The slot reads
    A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``; when it reads both, the later
    group's (smaller) count wins. ``None`` means the slot reads nothing and needs no wait.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    G = NM + NN
    base = _v(STAGES_BETWEEN) * G + _fills_before(mi, ni, NM, NN, ANY_FILL)
    out = None
    ta = _read_a_tile(mi, ni, NM, NN)
    if ta is not None:
        out = G - 1 - _fill_group_pos(True, ta, NM, NN) + base
    tb = _read_b_tile(mi, ni, NM, NN)
    if tb is not None:
        w = G - 1 - _fill_group_pos(False, tb, NM, NN) + base
        out = w if out is None else min(out, w)
    return out


@gluon.jit
def _slot_wait_group(
    lds,
    mi: gl.constexpr,
    ni: gl.constexpr,
    NM: gl.constexpr,
    NN: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    ANY_FILL: gl.constexpr,
):
    """Emit slot ``(mi, ni)``'s ``wait_group``, or nothing if it reads nothing.

    A separate function only so the ``: gl.constexpr`` binding is legal: inside the
    unrolled slot loop it would be a reassignment, and without the annotation the count
    reaches ``wait_group`` as a runtime tensor.
    """
    WAIT: gl.constexpr = _slot_wait(mi, ni, NM, NN, STAGES_BETWEEN, ANY_FILL)
    if require_constexpr(WAIT is not None):
        lds.wait_fill_lds_num_group(WAIT)


@gluon.jit
def _slot_fills(
    pc,
    BUF_LOAD_INX: gl.constexpr,
    mi: gl.constexpr,
    ni: gl.constexpr,
    a_hbm_ptr,
    b_hbm_ptr,
    a_scale_hbm_ptr,
    b_scale_hbm_ptr,
    ADVANCE: gl.constexpr = False,
):
    """The global->LDS copies slot ``(mi, ni)`` owns, each its own commit group.

    ``ADVANCE`` walks the returned HBM pointers on to the next ``BLOCK_K`` stage. It is
    done here, at the last slot, rather than after the slot loop on purpose: the warp
    pipeliner only tolerates a ``wait_group`` as the *first* op after a stage border, and
    a stage-tail ``tt.addptr`` sitting between the last ``mem`` border and the next
    slot's wait is exactly what breaks that.
    """
    func_cfg: gl.constexpr = pc.func_cfg
    NM: gl.constexpr = pc.tuning_cfg.num_mini_m()
    NN: gl.constexpr = pc.tuning_cfg.num_mini_n()
    EVEN: gl.constexpr = _even_fill(NM, NN)
    # Under the even schedule the slot owns at most one fill, named by its position in
    # _fill_order; under the legacy one the (ni == 0) / (mi == 0) predicates below pick.
    POS: gl.constexpr = _slot_fill_pos(mi, ni, NM, NN)
    A_TILE: gl.constexpr = _fill_tile_of(POS, NM, NN, EVEN, True)
    B_TILE: gl.constexpr = _fill_tile_of(POS, NM, NN, EVEN, False)
    if require_constexpr(A_TILE is not None):
        pc.lds.fill_a_lds(
            BUF_LOAD_INX,
            A_TILE,
            a_hbm_ptr,
            pc.a_offs[A_TILE],
            a_scale_hbm_ptr,
            _opt_at(pc.a_scale_offs, A_TILE, func_cfg.a_has_scale()),
        )
        pc.lds.commit_fill_lds()
    if require_constexpr(B_TILE is not None):
        pc.lds.fill_b_lds(
            BUF_LOAD_INX,
            B_TILE,
            b_hbm_ptr,
            pc.b_offs[B_TILE],
            b_scale_hbm_ptr,
            _opt_at(pc.b_scale_offs, B_TILE, func_cfg.b_has_scale()),
        )
        pc.lds.commit_fill_lds()
    if require_constexpr(not EVEN and ni == 0):
        pc.lds.fill_a_lds(
            BUF_LOAD_INX,
            mi,
            a_hbm_ptr,
            pc.a_offs[mi],
            a_scale_hbm_ptr,
            _opt_at(pc.a_scale_offs, mi, func_cfg.a_has_scale()),
        )
        pc.lds.commit_fill_lds()
    if require_constexpr(not EVEN and mi == 0):
        pc.lds.fill_b_lds(
            BUF_LOAD_INX,
            ni,
            b_hbm_ptr,
            pc.b_offs[ni],
            b_scale_hbm_ptr,
            _opt_at(pc.b_scale_offs, ni, func_cfg.b_has_scale()),
        )
        pc.lds.commit_fill_lds()
    if require_constexpr(ADVANCE):
        a_hbm_ptr = a_hbm_ptr + pc.a_step
        b_hbm_ptr = b_hbm_ptr + pc.b_step
        if require_constexpr(func_cfg.a_has_scale()):
            a_scale_hbm_ptr = a_scale_hbm_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(func_cfg.b_has_scale()):
            b_scale_hbm_ptr = b_scale_hbm_ptr + pc.s_step * pc.b_scale_stride_k
    return a_hbm_ptr, b_hbm_ptr, a_scale_hbm_ptr, b_scale_hbm_ptr


@gluon.jit
def _slot_a_read(pc, BUF_READ_INX, mi: gl.constexpr, a_scale_read_ptr):
    return _a_block_frags(
        pc.lds,
        BUF_READ_INX,
        mi,
        a_scale_read_ptr,
        _opt_at(pc.a_scale_offs, mi, pc.func_cfg.a_has_scale()),
        pc.func_cfg,
        pc.tuning_cfg,
    )


@gluon.jit
def _slot_b_read(pc, BUF_READ_INX, ni: gl.constexpr, b_scale_read_ptr):
    return _b_block_frags(
        pc.lds,
        BUF_READ_INX,
        ni,
        b_scale_read_ptr,
        _opt_at(pc.b_scale_offs, ni, pc.func_cfg.b_has_scale()),
        pc.func_cfg,
        pc.tuning_cfg,
    )


@gluon.jit
def _pipeline_step(
    pc,
    st,
    BUF_LOAD_INX: gl.constexpr,
    BUF_READ_INX: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_LDS_LOAD: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
):
    """One ``BLOCK_K`` stage, walked as ``num_mini_m() x num_mini_n()`` MFMA slots.

    A mini-M block's copy and ``ds_read`` are issued at its first slot (``ni == 0``) and
    reused by every later ``ni``; a mini-N block's at ``mi == 0``. So the copies and the
    LDS reads are interleaved with the MFMAs instead of standing in one block ahead of
    them, and each is still a whole mini tile at its own coalesced, fully vectorised copy
    layout.

    Every fill is its own commit group, so each slot waits for exactly the mini block it
    is about to read rather than for the whole stage -- see :func:`_slot_wait`, whose
    ``STAGES_BETWEEN`` is ``NUM_LDS_BUFFER - 2`` in the steady state (one buffer is being
    filled by ``buffer_load ... lds`` while one is being consumed by ``ds_read``).

    Along K the ``ds_read`` always pulls a whole stage, mini-K steps ``[0, NUM_MINI)``.
    The MFMAs run one window earlier, over ``[-PF_MINI, NUM_MINI-PF_MINI)``: the negative
    part is ``st.a_frags``/``st.b_frags``, carried from the previous step, and the tail
    this step reads is carried to the next. ``PF_MINI == NUM_MINI`` makes the MFMAs
    consume nothing they read themselves; ``PF_MINI == 0`` makes them consume only what
    they read.

    ``PF_MINI == NUM_MINI`` is also what makes the ping-pong legal: the slot's MFMAs then
    depend on nothing the slot reads, so they are emitted *first* and the reads and copies
    behind them are pure fill for the shadow. Under ``WARP_PIPELINE`` the two halves are
    additionally handed to ``TritonAMDGPUWarpPipeline`` as an ``mfma``/``mem`` stage pair,
    which is what turns the interleave into an inter-wave ping-pong. The ``wait_group``
    stays outside both regions -- the pass rejects a wait inside one.
    """
    func_cfg: gl.constexpr = pc.func_cfg
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_HAS: gl.constexpr = func_cfg.a_has_scale()
    B_HAS: gl.constexpr = func_cfg.b_has_scale()
    # mfma-before-mem is only legal when the slot's MFMAs read nothing the slot loads
    PING_PONG: gl.constexpr = HEAD_MINI == 0
    # Only inside the `tl.range` body. A `warp_pipeline_stage` region runs from the
    # previous border to its own, so a region opened in the peeled step or the drain
    # would still be open when the K loop starts -- and the pass rejects a loop (and
    # every wait) caught inside one.
    PIPE: gl.constexpr = tc.WARP_PIPELINE and PING_PONG and DO_LDS_LOAD and IN_LOOP

    a_hbm_ptr = st.a_hbm_ptr
    b_hbm_ptr = st.b_hbm_ptr
    a_scale_hbm_ptr = st.a_scale_hbm_ptr
    b_scale_hbm_ptr = st.b_scale_hbm_ptr
    a_scale_read_ptr = st.a_scale_read_ptr
    b_scale_read_ptr = st.b_scale_read_ptr

    # Fragments this step reads, appended mini block by mini block. `a_cur` grows once
    # per mi (at ni == 0) and `b_cur` once per ni (at mi == 0), so block `mi` always
    # starts at pair `mi * NUM_MINI` and is already there by the time any slot needs it.
    a_cur = ()
    b_cur = ()
    a_tail = ()
    b_tail = ()
    acc = ()
    for mi in gl.static_range(NM):
        for ni in gl.static_range(NN):
            if require_constexpr(DO_LDS_LOAD):
                _slot_wait_group(pc.lds, mi, ni, NM, NN, STAGES_BETWEEN, DO_BUFFER_LOAD)

            dot_a = _take_pairs(st.a_frags, mi * PF_MINI, PF_MINI)
            dot_b = _take_pairs(st.b_frags, ni * PF_MINI, PF_MINI)

            if require_constexpr(PING_PONG):
                # MFMA first: it consumes only registers carried from the previous stage,
                # so everything below it is shadow work.
                if require_constexpr(PIPE):
                    with warp_pipeline_stage("mfma", priority=0):
                        slot_acc = _block_dot(
                            dot_a, dot_b, st.acc[mi * NN + ni], PF_MINI, func_cfg
                        )
                        if require_constexpr(_DS_IN_MFMA or _DS_MOVE >= 1):
                            if require_constexpr(
                                _read_a_tile(mi, ni, NM, NN) is not None
                            ):
                                a_cur = a_cur + _slot_a_read(
                                    pc,
                                    BUF_READ_INX,
                                    _read_a_tile(mi, ni, NM, NN),
                                    a_scale_read_ptr,
                                )
                            if require_constexpr(
                                (_DS_IN_MFMA or _DS_MOVE >= 2)
                                and _read_b_tile(mi, ni, NM, NN) is not None
                            ):
                                b_cur = b_cur + _slot_b_read(
                                    pc,
                                    BUF_READ_INX,
                                    _read_b_tile(mi, ni, NM, NN),
                                    b_scale_read_ptr,
                                )
                        if require_constexpr(_FILL_IN_MFMA and DO_BUFFER_LOAD):
                            (
                                a_hbm_ptr,
                                b_hbm_ptr,
                                a_scale_hbm_ptr,
                                b_scale_hbm_ptr,
                            ) = _slot_fills(
                                pc,
                                BUF_LOAD_INX,
                                mi,
                                ni,
                                a_hbm_ptr,
                                b_hbm_ptr,
                                a_scale_hbm_ptr,
                                b_scale_hbm_ptr,
                                ADVANCE=(mi == NM - 1) and (ni == NN - 1),
                            )
                else:
                    slot_acc = _block_dot(
                        dot_a, dot_b, st.acc[mi * NN + ni], PF_MINI, func_cfg
                    )
                acc = acc + (slot_acc,)

                if require_constexpr(PIPE):
                    with warp_pipeline_stage("mem", priority=_MEM_PRIO):
                        if require_constexpr(
                            not (_DS_IN_MFMA or _DS_MOVE >= 1)
                            and _read_a_tile(mi, ni, NM, NN) is not None
                        ):
                            a_cur = a_cur + _slot_a_read(
                                pc,
                                BUF_READ_INX,
                                _read_a_tile(mi, ni, NM, NN),
                                a_scale_read_ptr,
                            )
                        if require_constexpr(
                            not (_DS_IN_MFMA or _DS_MOVE >= 2)
                            and _read_b_tile(mi, ni, NM, NN) is not None
                        ):
                            b_cur = b_cur + _slot_b_read(
                                pc,
                                BUF_READ_INX,
                                _read_b_tile(mi, ni, NM, NN),
                                b_scale_read_ptr,
                            )
                        if require_constexpr(not _FILL_IN_MFMA and DO_BUFFER_LOAD):
                            (
                                a_hbm_ptr,
                                b_hbm_ptr,
                                a_scale_hbm_ptr,
                                b_scale_hbm_ptr,
                            ) = _slot_fills(
                                pc,
                                BUF_LOAD_INX,
                                mi,
                                ni,
                                a_hbm_ptr,
                                b_hbm_ptr,
                                a_scale_hbm_ptr,
                                b_scale_hbm_ptr,
                                ADVANCE=(mi == NM - 1) and (ni == NN - 1),
                            )
                else:
                    if require_constexpr(
                        DO_LDS_LOAD and _read_a_tile(mi, ni, NM, NN) is not None
                    ):
                        a_cur = a_cur + _slot_a_read(
                            pc,
                            BUF_READ_INX,
                            _read_a_tile(mi, ni, NM, NN),
                            a_scale_read_ptr,
                        )
                    if require_constexpr(
                        DO_LDS_LOAD and _read_b_tile(mi, ni, NM, NN) is not None
                    ):
                        b_cur = b_cur + _slot_b_read(
                            pc,
                            BUF_READ_INX,
                            _read_b_tile(mi, ni, NM, NN),
                            b_scale_read_ptr,
                        )
                    if require_constexpr(DO_BUFFER_LOAD):
                        (
                            a_hbm_ptr,
                            b_hbm_ptr,
                            a_scale_hbm_ptr,
                            b_scale_hbm_ptr,
                        ) = _slot_fills(
                            pc,
                            BUF_LOAD_INX,
                            mi,
                            ni,
                            a_hbm_ptr,
                            b_hbm_ptr,
                            a_scale_hbm_ptr,
                            b_scale_hbm_ptr,
                            ADVANCE=(mi == NM - 1) and (ni == NN - 1),
                        )
            else:
                # Part of the window comes from this slot's own read, so the read has to
                # come first and there is no shadow to hand the warp pipeline.
                if require_constexpr(DO_BUFFER_LOAD):
                    (
                        a_hbm_ptr,
                        b_hbm_ptr,
                        a_scale_hbm_ptr,
                        b_scale_hbm_ptr,
                    ) = _slot_fills(
                        pc,
                        BUF_LOAD_INX,
                        mi,
                        ni,
                        a_hbm_ptr,
                        b_hbm_ptr,
                        a_scale_hbm_ptr,
                        b_scale_hbm_ptr,
                        ADVANCE=(mi == NM - 1) and (ni == NN - 1),
                    )
                if require_constexpr(DO_LDS_LOAD):
                    if require_constexpr(_read_a_tile(mi, ni, NM, NN) is not None):
                        a_cur = a_cur + _slot_a_read(
                            pc,
                            BUF_READ_INX,
                            _read_a_tile(mi, ni, NM, NN),
                            a_scale_read_ptr,
                        )
                    if require_constexpr(_read_b_tile(mi, ni, NM, NN) is not None):
                        b_cur = b_cur + _slot_b_read(
                            pc,
                            BUF_READ_INX,
                            _read_b_tile(mi, ni, NM, NN),
                            b_scale_read_ptr,
                        )
                    # carried tail + the head of what this slot just read == one stage
                    acc = acc + (
                        _block_dot(
                            dot_a + _take_pairs(a_cur, mi * NUM_MINI, HEAD_MINI),
                            dot_b + _take_pairs(b_cur, ni * NUM_MINI, HEAD_MINI),
                            st.acc[mi * NN + ni],
                            NUM_MINI,
                            func_cfg,
                        ),
                    )
                else:
                    acc = acc + (
                        _block_dot(
                            dot_a, dot_b, st.acc[mi * NN + ni], PF_MINI, func_cfg
                        ),
                    )

            # Indexed by the tile this slot actually read, not by (mi, ni): under the
            # even schedule slot (0, 1) reads B(0), not B(1), so the legacy `ni == 0` /
            # `mi == 0` predicates would reach past the end of the half-built tuple.
            if require_constexpr(
                DO_LDS_LOAD and _read_a_tile(mi, ni, NM, NN) is not None
            ):
                a_tail = a_tail + _take_pairs(
                    a_cur,
                    _read_a_tile(mi, ni, NM, NN) * NUM_MINI + HEAD_MINI,
                    PF_MINI,
                )
            if require_constexpr(
                DO_LDS_LOAD and _read_b_tile(mi, ni, NM, NN) is not None
            ):
                b_tail = b_tail + _take_pairs(
                    b_cur,
                    _read_b_tile(mi, ni, NM, NN) * NUM_MINI + HEAD_MINI,
                    PF_MINI,
                )

    if require_constexpr(DO_LDS_LOAD):
        if require_constexpr(A_HAS):
            a_scale_read_ptr = a_scale_read_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(B_HAS):
            b_scale_read_ptr = b_scale_read_ptr + pc.s_step * pc.b_scale_stride_k
    else:
        a_tail = st.a_frags
        b_tail = st.b_frags

    return _PipelineState(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_ptr,
        b_scale_read_ptr,
        a_tail,
        b_tail,
        acc,
    )


@gluon.jit
def _epilogue_store(
    acc,
    y_ptr,
    y_stride_m,
    y_stride_n,
    ys_ptr,
    ys_stride_m,
    ys_stride_n,
    bias_ptr,
    block_id,
    pid_n,
    M_e,
    gammas_base,
    x_static_scale,
    func_cfg,
    tuning_cfg,
):
    """bias -> activation -> gammas -> output_quant -> store, per mini (M, N) tile.

    The order is fixed and matches ``_triton_kernels/moe/moe_op_gemm_a4w4.py``; the
    clamp and the multiply happen in fp32 before any cast. ``acc`` arrives already split
    into one tensor per mini tile -- the K pipeline accumulates that way -- so the
    epilogue walks the same tiling and consumes them one at a time, which lets the
    compiler kill each mini accumulator before the next tile's epilogue.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_MBN: gl.constexpr = MBN // ARN
    act: gl.constexpr = func_cfg.act()
    # Hoisted: a `x: gl.constexpr = ...` inside a static_range would be a reassignment
    # on the second unrolled iteration, which Gluon rejects outright.
    out_ty: gl.constexpr = y_ptr.dtype.element_ty
    store_layout: gl.constexpr = tuning_cfg.result_store_layout(
        MBM, OUT_MBN, out_ty.primitive_bitwidth
    )

    NN: gl.constexpr = BN // MBN
    for mi in gl.static_range(BM // MBM):
        for ni in gl.static_range(NN):
            sub = acc[mi * NN + ni]

            if require_constexpr(func_cfg.has_x_static_scale):
                # Per-tensor fp8 activation scale. FP8_E4M3 operands go through the
                # MMA with unit scales, so the tensor scale comes back here -- before
                # bias, matching the Triton kernels exactly.
                sub = sub * x_static_scale

            raw_n0 = pid_n * BN + ni * MBN
            if require_constexpr(func_cfg.has_bias):
                # fp32, expert indexed, over the RAW N axis (before the halving)
                bias = gl.load(bias_ptr + raw_n0 + gl.arange(0, MBN))
                sub = sub + bias[None, :]

            if require_constexpr(func_cfg.has_activation()):
                out = _swiglu(sub, act.alpha, act.limit, ADD_RESIDUAL=act.add_residual)
                gl.static_assert(out.shape[1] == OUT_MBN)
            else:
                gl.static_assert(ARN == 1)
                out = sub

            offs_m = BM * block_id + mi * MBM + gl.arange(0, MBM)
            mask_m = offs_m < M_e
            if require_constexpr(func_cfg.has_gammas):
                g = gl.load(gammas_base + offs_m, mask=mask_m, other=0.0)
                out = out * g[:, None]

            out_n0 = raw_n0 // ARN
            if require_constexpr(func_cfg.output_quant is None):
                val = gl.convert_layout(
                    out.to(out_ty), store_layout, assert_trivial=False
                )
                sm = gl.arange(0, MBM, layout=gl.SliceLayout(1, store_layout))
                sn = gl.arange(0, OUT_MBN, layout=gl.SliceLayout(0, store_layout))
                rows = BM * block_id + mi * MBM + sm
                gl.amd.cdna4.buffer_store(
                    val,
                    y_ptr,
                    rows[:, None] * y_stride_m + (out_n0 + sn)[None, :] * y_stride_n,
                    mask=(rows < M_e)[:, None],
                    cache=tuning_cfg.result_mod,
                )
            else:
                # Fused MXFP4 output quant. _mxfp4_quant_op is the exact op the
                # standalone mxfp4_quant launch uses, and the round-trip through bf16
                # reproduces the bf16 intermediate that launch reads back from HBM, so
                # the emitted payload and E8M0 scale are bit-identical and gemm2 needs
                # no change. The MX group runs along the emitted N axis, so the amax is
                # over 32 emitted columns -- with the transposed MFMA accumulator each
                # lane already owns 4 consecutive N, and the reduction over the
                # remaining 8 lanes is what the reshape below expresses.
                gl.static_assert(func_cfg.output_quant == _DQ_MXFP4)
                gl.static_assert(OUT_MBN % MX_GROUP == 0)
                payload, scale = mxfp4_quant_gluon(
                    out.to(gl.bfloat16).to(gl.float32), OUT_MBN, MBM, MX_GROUP
                )
                # Fresh M ranges per store: reusing one auto-layout `offs_m` across two
                # differently-laid-out stores makes GluonResolveAutoEncodings fail with
                # "conflicting encodings" on the expand_dims.
                pm = BM * block_id + mi * MBM + gl.arange(0, MBM)
                gl.store(
                    y_ptr
                    + pm[:, None] * y_stride_m
                    + (out_n0 // 2 + gl.arange(0, OUT_MBN // 2))[None, :] * y_stride_n,
                    payload,
                    mask=(pm < M_e)[:, None],
                )
                sm = BM * block_id + mi * MBM + gl.arange(0, MBM)
                gl.store(
                    ys_ptr
                    + sm[:, None] * ys_stride_m
                    + (out_n0 // MX_GROUP + gl.arange(0, OUT_MBN // MX_GROUP))[None, :]
                    * ys_stride_n,
                    scale,
                    mask=(sm < M_e)[:, None],
                )


@gluon.jit
def _moe_gemm_body(
    a,  # QuantTokenTensor | NonQuantTokenTensor
    b,  # QuantExpertTensor | NonQuantExpertTensor
    res,  # ResultTensor
    rt,  # RoutingMeta
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    cfg,  # MoeKernelConfig
    FuncCfgT: gl.constexpr,
    TuningCfgT: gl.constexpr,
):
    func_cfg, tuning_cfg = _build_configs(cfg, FuncCfgT, TuningCfgT)
    gl.static_assert(tuning_cfg.validate(cfg.N, cfg.K))

    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    BK: gl.constexpr = tuning_cfg.BLOCK_K
    NB: gl.constexpr = tuning_cfg.NUM_LDS_BUFFER
    PK_A: gl.constexpr = BK // func_cfg.a_pack_divisor()
    PK_B: gl.constexpr = BK // func_cfg.b_pack_divisor()
    SK: gl.constexpr = BK // MX_GROUP
    NUM_K: gl.constexpr = tuning_cfg.num_k_tiles(cfg.K)

    pid = gl.program_id(0)
    if require_constexpr(tuning_cfg.TILE_SCHED == _TS_XCD_GROUP_M):
        # Drop the padded tiles first so the swizzle is a bijection over real work; a
        # grid-persistent mode would forfeit this early return, which is why it is out
        # of scope here.
        unpadded_m = gl.load(rt.expt_offs_sum)
        if pid >= unpadded_m * grid_n:
            return
        pid = remap_xcd(pid, unpadded_m * grid_n, tuning_cfg.NUM_XCDS)
        pid_m, pid_n = pid_grid(pid, unpadded_m, grid_n, tuning_cfg.GROUP_M)
    elif require_constexpr(tuning_cfg.TILE_SCHED == _TS_GROUP_M):
        pid_m, pid_n = pid_grid(pid, grid_m, grid_n, tuning_cfg.GROUP_M)
    else:
        pid_m = pid // grid_n
        pid_n = pid % grid_n

    expt_data = gl.load(rt.expt_block_pid_map + pid_m)
    if expt_data == -1:
        # No work. Nothing has been committed yet, so no async group is left dangling.
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M_e = gl.load(rt.expt_hist + expt_id)
    start_m = gl.load(rt.expt_offs_raw + expt_id)

    # Buffer ops carry a 32-bit offset (2 GB window) and V4-Pro's stacked gemm1 weight is
    # ~8.5 GB, so the expert stride is folded into the scalar base in 64-bit; a single
    # expert is ~22 MB and fits comfortably.
    w_ptr = b.ptr + expt_id.to(gl.int64) * b.stride_e
    if require_constexpr(func_cfg.b_has_scale()):
        ws_ptr = b.scale_ptr + expt_id.to(gl.int64) * b.scale_stride_e
    else:
        ws_ptr: gl.constexpr = None
    y_ptr = res.ptr + start_m.to(gl.int64) * res.stride_m
    if require_constexpr(func_cfg.output_quant is not None):
        ys_ptr = res.scale_ptr + start_m.to(gl.int64) * res.scale_stride_m
    else:
        ys_ptr: gl.constexpr = None
    if require_constexpr(func_cfg.has_bias):
        bias_base = bias_ptr + expt_id.to(gl.int64) * stride_bias_e
    else:
        bias_base: gl.constexpr = None
    if require_constexpr(func_cfg.has_gammas):
        gammas_base = rt.gammas + start_m
    else:
        gammas_base: gl.constexpr = None

    # One offset grid per mini block. Each is built at *its own* mini tile's copy
    # layout, so the split into MINI_BLOCK_M rows / MINI_BLOCK_N columns costs no
    # coalescing and no vector width -- the tile it addresses is a whole LDS allocation,
    # not a strided view of a bigger one.
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NM: gl.constexpr = tuning_cfg.num_mini_m()
    NN: gl.constexpr = tuning_cfg.num_mini_n()
    cl_a: gl.constexpr = tuning_cfg.dot_operand_copy_layout(0)
    cl_b: gl.constexpr = tuning_cfg.dot_operand_copy_layout(1)

    a_offs = ()
    for mi in gl.static_range(NM):
        rows_a = _gather_rows(
            rt,
            block_id,
            M_e,
            start_m,
            gl.SliceLayout(1, cl_a),
            BM,
            mi * MBM,
            MBM,
            func_cfg.has_gather,
        )
        a_offs = a_offs + (
            rows_a[:, None] * a.stride_m
            + gl.arange(0, PK_A, layout=gl.SliceLayout(0, cl_a))[None, :],
        )
    b_offs = ()
    for ni in gl.static_range(NN):
        b_offs = b_offs + (
            gl.arange(0, PK_B, layout=gl.SliceLayout(1, cl_b))[:, None] * b.stride_k
            + (
                pid_n * BN
                + ni * MBN
                + gl.arange(0, MBN, layout=gl.SliceLayout(0, cl_b))
            )[None, :]
            * b.stride_n,
        )

    if require_constexpr(func_cfg.a_has_scale()):
        if require_constexpr(tuning_cfg.scale_via_lds(0)):
            asl: gl.constexpr = tuning_cfg.dot_operand_scale_copy_layout(0)
        else:
            asl: gl.constexpr = tuning_cfg.dot_operand_scale_fragment_layout(0)
        a_scale_offs = ()
        for mi in gl.static_range(NM):
            rows_as = _gather_rows(
                rt,
                block_id,
                M_e,
                start_m,
                gl.SliceLayout(1, asl),
                BM,
                mi * MBM,
                MBM,
                func_cfg.has_gather,
            )
            a_scale_offs = a_scale_offs + (
                rows_as[:, None] * a.scale_stride_m
                + gl.arange(0, SK, layout=gl.SliceLayout(0, asl))[None, :]
                * a.scale_stride_k,
            )
        as_ptr = a.scale_ptr
    else:
        a_scale_offs: gl.constexpr = None
        as_ptr: gl.constexpr = None

    if require_constexpr(func_cfg.b_has_scale()):
        if require_constexpr(tuning_cfg.scale_via_lds(1)):
            bsl: gl.constexpr = tuning_cfg.dot_operand_scale_copy_layout(1)
        else:
            bsl: gl.constexpr = tuning_cfg.dot_operand_scale_fragment_layout(1)
        b_scale_offs = ()
        for ni in gl.static_range(NN):
            b_scale_offs = b_scale_offs + (
                (
                    pid_n * BN
                    + ni * MBN
                    + gl.arange(0, MBN, layout=gl.SliceLayout(1, bsl))
                )[:, None]
                * b.scale_stride_n
                + gl.arange(0, SK, layout=gl.SliceLayout(0, bsl))[None, :]
                * b.scale_stride_k,
            )
    else:
        b_scale_offs: gl.constexpr = None

    lds = LDSManager.alloc(func_cfg, tuning_cfg)

    # per-fill pointer bumps (K is the contiguous axis of every operand)
    a_step: gl.constexpr = PK_A
    b_step: gl.constexpr = PK_B
    s_step: gl.constexpr = SK

    a_hbm_ptr = a.ptr
    b_hbm_ptr = w_ptr
    a_scale_hbm_ptr = as_ptr
    b_scale_hbm_ptr = ws_ptr

    A_HAS: gl.constexpr = func_cfg.a_has_scale()
    B_HAS: gl.constexpr = func_cfg.b_has_scale()
    # commit groups per stage: one per mini-block fill
    G: gl.constexpr = NM + NN

    MAIN: gl.constexpr = NUM_K - NB
    # steps left after the peel, rounded down to a whole number of unrolled bodies
    UNROLLED: gl.constexpr = ((MAIN - 1) // tuning_cfg.K_UNROLL) * tuning_cfg.K_UNROLL
    # one buffer is being filled and one consumed on every step, so the read buffer's
    # groups have NUM_LDS_BUFFER - 2 whole stages behind them
    STAGES_BETWEEN: gl.constexpr = NB - 2

    # The scale strides only exist on a quantised operand, so they can only be read
    # behind the same predicate that guards every other use of them.
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_stride_k = a.scale_stride_k
    else:
        a_scale_stride_k: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_stride_k = b.scale_stride_k
    else:
        b_scale_stride_k: gl.constexpr = None
    pc = _PipelineConst(
        lds,
        a_offs,
        b_offs,
        a_scale_offs,
        b_scale_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
    )

    NUM_MINI: gl.constexpr = tuning_cfg.num_mini_k()
    PF_MINI: gl.constexpr = tuning_cfg.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI

    # Prologue fill: every buffer, no mma. Walks the slot grid rather than A-then-B so
    # the commit groups land in the same order _slot_wait() assumes for a loop stage.
    for i in gl.static_range(NB):
        for mi in gl.static_range(NM):
            for ni in gl.static_range(NN):
                _slot_fills(
                    pc,
                    i,
                    mi,
                    ni,
                    a_hbm_ptr,
                    b_hbm_ptr,
                    a_scale_hbm_ptr,
                    b_scale_hbm_ptr,
                )
        a_hbm_ptr = a_hbm_ptr + a_step
        b_hbm_ptr = b_hbm_ptr + b_step
        if require_constexpr(A_HAS):
            a_scale_hbm_ptr = a_scale_hbm_ptr + s_step * a_scale_stride_k
        if require_constexpr(B_HAS):
            b_scale_hbm_ptr = b_scale_hbm_ptr + s_step * b_scale_stride_k

    # Read-side scale pointers. These exist separately from the fill-side ones because
    # the register fallback (scale tile too small for a coalesced direct-to-LDS write)
    # reads the scale at *consume* time, NUM_LDS_BUFFER stages behind the fill;
    # advancing only one of the two silently re-reads the first K tile's scales for the
    # whole loop, which no compile-time check catches. Both start at the same base and
    # walk K at the same rate, just from different points in the pipeline.
    a_scale_read_ptr = as_ptr
    b_scale_read_ptr = ws_ptr

    # Prologue read: stage 0, whose MFMA window has nothing before it, so only its head
    # is dotted here. Its tail seeds the carried fragments the first loop step consumes.
    # It reads every mini block of buffer 0, so it waits out that buffer's whole group
    # set -- the NB-1 buffers behind it stay in flight.
    lds.wait_fill_lds_num_group((NB - 1) * G)
    a0 = ()
    a_tail = ()
    for mi in gl.static_range(NM):
        a0 = a0 + _a_block_frags(
            lds,
            0,
            mi,
            a_scale_read_ptr,
            _opt_at(a_scale_offs, mi, A_HAS),
            func_cfg,
            tuning_cfg,
        )
        a_tail = a_tail + _take_pairs(a0, mi * NUM_MINI + HEAD_MINI, PF_MINI)
    b0 = ()
    b_tail = ()
    for ni in gl.static_range(NN):
        b0 = b0 + _b_block_frags(
            lds,
            0,
            ni,
            b_scale_read_ptr,
            _opt_at(b_scale_offs, ni, B_HAS),
            func_cfg,
            tuning_cfg,
        )
        b_tail = b_tail + _take_pairs(b0, ni * NUM_MINI + HEAD_MINI, PF_MINI)
    if require_constexpr(A_HAS):
        a_scale_read_ptr = a_scale_read_ptr + s_step * a_scale_stride_k
    if require_constexpr(B_HAS):
        b_scale_read_ptr = b_scale_read_ptr + s_step * b_scale_stride_k

    acc = ()
    for mi in gl.static_range(NM):
        for ni in gl.static_range(NN):
            acc = acc + (
                _block_dot(
                    _take_pairs(a0, mi * NUM_MINI, HEAD_MINI),
                    _take_pairs(b0, ni * NUM_MINI, HEAD_MINI),
                    gl.zeros(
                        [MBM, MBN],
                        dtype=func_cfg.mma_acc_dtype,
                        layout=tuning_cfg.dot_result_fragment_layout(),
                    ),
                    HEAD_MINI,
                    func_cfg,
                ),
            )
    st = _PipelineState(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_ptr,
        b_scale_read_ptr,
        a_tail,
        b_tail,
        acc,
    )

    # The whole main sequence is guarded: MAIN is 0 when NUM_K == NUM_LDS_BUFFER, and
    # then every stage belongs to the drain and there is no step here at all.
    if require_constexpr(MAIN > 0):
        # First step peeled out. Its MFMAs are the ones that consume the accumulator
        # while it is still visibly gl.zeros -- inside the loop that is a phi and the
        # zero is invisible.
        st = _pipeline_step(pc, st, 0, 1 % NB, STAGES_BETWEEN, True, True)

        # steady state over the remaining MAIN-1 steps, body unrolled K_UNROLL times.
        # K_UNROLL % NUM_LDS_BUFFER == 0 is asserted in validate(), so global step
        # 1+_k+i is congruent to 1+i mod NB and the rotating index stays a static
        # offset that constant-folds the wait_group counts.
        for _k in tl.range(0, UNROLLED, tuning_cfg.K_UNROLL):
            for i in gl.static_range(tuning_cfg.K_UNROLL):
                st = _pipeline_step(
                    pc,
                    st,
                    (1 + i) % NB,
                    (2 + i) % NB,
                    STAGES_BETWEEN,
                    True,
                    True,
                    IN_LOOP=True,
                )

        # remainder (0 .. K_UNROLL-1 steps); NUM_K is constexpr so this stays static
        for j in gl.static_range(1 + UNROLLED, MAIN):
            st = _pipeline_step(
                pc, st, j % NB, (j + 1) % NB, STAGES_BETWEEN, True, True
            )

    # drain: the last NB stages are consumed with NO copy at all. This, not masking, is
    # what keeps every load inside K -- getting it wrong reads into the next expert's
    # weights while every correctness case still passes. The final step reads nothing and
    # just consumes the fragments still in registers.
    for i in gl.static_range(NB):
        st = _pipeline_step(
            pc, st, 0, (MAIN + i + 1) % NB, NB - 2 - i, False, i + 1 < NB
        )
    acc = st.acc

    # Hoisted out of the mini-tile loop: one scalar load, not one per tile.
    if require_constexpr(func_cfg.has_x_static_scale):
        x_static_scale = gl.load(x_static_scale_ptr)
    else:
        x_static_scale: gl.constexpr = None
    _epilogue_store(
        acc,
        y_ptr,
        res.stride_m,
        res.stride_n,
        ys_ptr,
        res.scale_stride_m,
        res.scale_stride_n,
        bias_base,
        block_id,
        pid_n,
        M_e,
        gammas_base,
        x_static_scale,
        func_cfg,
        tuning_cfg,
    )


_gemm1_repr = make_kernel_repr("_moe_gluon_gemm1", _moe_gemm_repr_fields)
_gemm2_repr = make_kernel_repr("_moe_gluon_gemm2", _moe_gemm_repr_fields)


@gluon.jit(
    repr=_gemm1_repr,
    launch_metadata=moe_gemm_launch_metadata,
    do_not_specialize=["grid_m", "grid_n"],
)
def _moe_gluon_gemm1(
    a,  # QuantTokenTensor | NonQuantTokenTensor
    b,  # QuantExpertTensor | NonQuantExpertTensor
    res,  # ResultTensor
    rt,  # RoutingMeta
    bias_ptr,  # fp32 (E, N) or None
    stride_bias_e,
    x_static_scale_ptr,  # fp32 scalar or None
    grid_m,  # routing_data.n_blocks(M, block_m) -- NOT cdiv
    grid_n,  # tuning_cfg.grid_N(N)
    cfg,  # MoeKernelConfig
    FuncCfgT: gl.constexpr,
    TuningCfgT: gl.constexpr,
):
    """gemm1: gather + X @ W1 + bias + swiglu, optionally fused MXFP4 output quant.

    Signature, verbatim, for the launch site:
        _moe_gluon_gemm1[(grid_m * grid_n,)](
            a, b, res, rt, bias_ptr, stride_bias_e, x_static_scale_ptr, grid_m,
            grid_n, cfg,
            KernelFuncConfig, KernelTuningConfig,
            num_warps=..., waves_per_eu=...)
    where the NamedTuples flatten to
        a  : dtype_quant, ptr, scale_ptr, num_token, stride_m, scale_stride_m,
             scale_stride_k, hidden_dim, topk, scale_swizzle
        b  : dtype_quant, ptr, scale_ptr, stride_e, stride_k, stride_n, scale_stride_e,
             scale_stride_n, scale_stride_k, num_expert, hidden_dim,
             fused_intermediate_dim, scale_swizzle
        res: dtype_quant, ptr, scale_ptr, stride_m, stride_n, scale_stride_m,
             scale_stride_n, out_dim
        rt : expt_block_pid_map, expt_hist, expt_offs_raw, expt_offs_sum, gather_indx,
             scatter_indx, gammas, n_expts_act
    """
    _moe_gemm_body(
        a,
        b,
        res,
        rt,
        bias_ptr,
        stride_bias_e,
        x_static_scale_ptr,
        grid_m,
        grid_n,
        cfg,
        FuncCfgT,
        TuningCfgT,
    )


@gluon.jit(
    repr=_gemm2_repr,
    launch_metadata=moe_gemm_launch_metadata,
    do_not_specialize=["grid_m", "grid_n"],
)
def _moe_gluon_gemm2(
    a,
    b,
    res,
    rt,
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    cfg,
    FuncCfgT: gl.constexpr,
    TuningCfgT: gl.constexpr,
):
    """gemm2: X @ W2 + bias, multiplied by the router combine weight (gammas).

    Same verbatim signature as :func:`_moe_gluon_gemm1`; only the ``KernelFuncConfig``
    built in-kernel differs (``activation is None``, ``output_quant is None``).
    """
    _moe_gemm_body(
        a,
        b,
        res,
        rt,
        bias_ptr,
        stride_bias_e,
        x_static_scale_ptr,
        grid_m,
        grid_n,
        cfg,
        FuncCfgT,
        TuningCfgT,
    )
