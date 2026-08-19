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

from typing import NamedTuple

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
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
    HAS_GATHER: gl.constexpr,
):
    """Row indices of one M block, already resolved through the gather table."""
    offs = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=layout)
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
def _stage_frags(
    lds,
    buf_idx,
    a_scale_ptr,
    a_scale_offs,
    b_scale_ptr,
    b_scale_offs,
    func_cfg,
    tuning_cfg,
):
    """Every LDS->register fragment of one ``BLOCK_K`` stage, as a flat tuple.

    Split out so :func:`_pipeline_step` can hold the result across a
    pipeline iteration and hand it to :func:`_stage_dot` one stage later.

    The tuple is always four wide per mini-K step even for an operand with no scale: it
    is loop-carried, so Triton asks every element for its ``.type`` and a ``None`` there
    aborts codegen. An absent scale therefore parks the payload's own SSA value in its
    slot -- a duplicate reference costs no register -- and :func:`_stage_dot` puts the
    ``None`` back from the same compile-time predicate.
    """
    NUM_MINI: gl.constexpr = tuning_cfg.num_mini_k()
    SK_MINI: gl.constexpr = tuning_cfg.MINI_BLOCK_K // MX_GROUP
    frags = ()
    for i in gl.static_range(NUM_MINI):
        a, a_s = lds.load_a_frag(
            buf_idx,
            i,
            a_scale_ptr,
            _mini_scale_off(a_scale_offs, i * SK_MINI, func_cfg.has_scale(0)),
        )
        b, b_s = lds.load_b_frag(
            buf_idx,
            i,
            b_scale_ptr,
            _mini_scale_off(b_scale_offs, i * SK_MINI, func_cfg.has_scale(1)),
        )
        if require_constexpr(func_cfg.has_scale(0)):
            a_slot = a_s
        else:
            a_slot = a
        if require_constexpr(func_cfg.has_scale(1)):
            b_slot = b_s
        else:
            b_slot = b
        frags = frags + (a, a_slot, b, b_slot)
    return frags


@gluon.jit
def _take_minis(frags, LO: gl.constexpr, N: gl.constexpr):
    """Mini-K steps ``[LO, LO+N)`` of a fragment tuple, as a fresh tuple.

    Built element-wise rather than sliced: a loop-carried tuple arrives as a
    ``tl.tuple``, which indexes but does not slice.
    """
    out = ()
    for i in gl.static_range(N):
        out = out + (
            frags[4 * (LO + i)],
            frags[4 * (LO + i) + 1],
            frags[4 * (LO + i) + 2],
            frags[4 * (LO + i) + 3],
        )
    return out


@gluon.jit
def _stage_dot(frags, acc, N_MINI: gl.constexpr, func_cfg):
    """The MFMAs for ``N_MINI`` mini-K steps, over fragments already in registers."""
    for i in gl.static_range(N_MINI):
        if require_constexpr(func_cfg.has_scale(0)):
            a_s = frags[4 * i + 1]
        else:
            a_s = _NO_SCALE
        if require_constexpr(func_cfg.has_scale(1)):
            b_s = frags[4 * i + 3]
        else:
            b_s = _NO_SCALE
        acc = _dot(frags[4 * i], a_s, frags[4 * i + 2], b_s, acc, func_cfg)
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

    ``a_scale_offs``/``b_scale_offs`` serve *both* sides: the direct-to-LDS copy and, for
    the register fallback, the consume-side load. Both walk K by advancing their scalar
    base pointer in :class:`_PipelineState` instead of the offset grid, so this tile grid
    is computed once and never moves.

    The two ``*_scale_stride_k`` are lifted out of the operand tuples rather than kept as
    ``a``/``b``: the non-quantised operand types have no such field at all, so it can only
    be read behind the ``has_scale`` predicate that the caller already evaluates.
    """

    lds: LDSManager
    a_offs: gl.tensor
    b_offs: gl.tensor
    a_scale_offs: gl.tensor | gl.constexpr
    b_scale_offs: gl.tensor | gl.constexpr
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
    obvious that advancing only one of the pair is the bug. ``frags`` holds the
    fragments read one stage ahead of the MFMAs that take them.
    """

    a_hbm_ptr: gl.tensor
    b_hbm_ptr: gl.tensor
    a_scale_hbm_ptr: gl.tensor | gl.constexpr
    b_scale_hbm_ptr: gl.tensor | gl.constexpr
    a_scale_read_ptr: gl.tensor | gl.constexpr
    b_scale_read_ptr: gl.tensor | gl.constexpr
    frags: tl.tuple | tuple
    acc: gl.tensor

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
        frags,
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
        self.frags = frags
        self.acc = acc


@gluon.jit
def _pipeline_step(
    pc,
    st,
    BUF_LOAD_INX: gl.constexpr,
    BUF_READ_INX: gl.constexpr,
    WAIT_BUF_NUM: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_LDS_LOAD: gl.constexpr,
):
    """One ``BLOCK_K`` stage: wait, copy, read, mma, commit.

    At any moment one buffer is being filled by ``buffer_load ... lds`` and one is being
    consumed by ``ds_read``, so ``WAIT_BUF_NUM`` is ``NUM_LDS_BUFFER - 2``.

    The ``ds_read`` always pulls a whole stage, mini-K steps ``[0, NUM_MINI)``. The MFMAs
    run one window earlier, over ``[-PF_MINI, NUM_MINI-PF_MINI)``: the negative part is
    ``st.frags``, carried from the previous step, and the tail this step reads is carried
    to the next. ``PF_MINI == NUM_MINI`` makes the MFMAs consume nothing they read
    themselves; ``PF_MINI == 0`` makes them consume only what they read.
    """
    func_cfg: gl.constexpr = pc.func_cfg
    NUM_MINI: gl.constexpr = pc.tuning_cfg.num_mini_k()
    PF_MINI: gl.constexpr = pc.tuning_cfg.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI

    if require_constexpr(DO_LDS_LOAD):
        pc.lds.wait_fill_lds_num_buf(WAIT_BUF_NUM)

    a_hbm_ptr = st.a_hbm_ptr
    b_hbm_ptr = st.b_hbm_ptr
    a_scale_hbm_ptr = st.a_scale_hbm_ptr
    b_scale_hbm_ptr = st.b_scale_hbm_ptr
    a_scale_read_ptr = st.a_scale_read_ptr
    b_scale_read_ptr = st.b_scale_read_ptr

    if require_constexpr(DO_BUFFER_LOAD):
        pc.lds.fill_a_lds(
            BUF_LOAD_INX, a_hbm_ptr, pc.a_offs, a_scale_hbm_ptr, pc.a_scale_offs
        )
        pc.lds.fill_b_lds(
            BUF_LOAD_INX, b_hbm_ptr, pc.b_offs, b_scale_hbm_ptr, pc.b_scale_offs
        )
        a_hbm_ptr = a_hbm_ptr + pc.a_step
        b_hbm_ptr = b_hbm_ptr + pc.b_step
        if require_constexpr(func_cfg.a_has_scale()):
            a_scale_hbm_ptr = a_scale_hbm_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(func_cfg.b_has_scale()):
            b_scale_hbm_ptr = b_scale_hbm_ptr + pc.s_step * pc.b_scale_stride_k

    if require_constexpr(DO_LDS_LOAD):
        fresh = _stage_frags(
            pc.lds,
            BUF_READ_INX,
            a_scale_read_ptr,
            pc.a_scale_offs,
            b_scale_read_ptr,
            pc.b_scale_offs,
            func_cfg,
            pc.tuning_cfg,
        )
        if require_constexpr(func_cfg.a_has_scale()):
            a_scale_read_ptr = a_scale_read_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(func_cfg.b_has_scale()):
            b_scale_read_ptr = b_scale_read_ptr + pc.s_step * pc.b_scale_stride_k
        # carried tail + the head of what this step just read == one whole stage
        dot_frags = _take_minis(st.frags, 0, PF_MINI) + _take_minis(fresh, 0, HEAD_MINI)
        frags = _take_minis(fresh, HEAD_MINI, PF_MINI)
        acc = _stage_dot(dot_frags, st.acc, NUM_MINI, func_cfg)
    else:
        # final stage: nothing left to read, so consume what is still in registers
        frags = st.frags
        acc = _stage_dot(st.frags, st.acc, PF_MINI, func_cfg)

    if require_constexpr(DO_BUFFER_LOAD):
        pc.lds.commit_fill_lds()
    return _PipelineState(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_ptr,
        b_scale_read_ptr,
        frags,
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
    clamp and the multiply happen in fp32 before any cast. The mini-tile loop exists to
    drain accumulator registers early: ``gl.amd.slice`` is a register-only,
    layout-preserving view, so it emits no instruction, but it does let the compiler
    kill the consumed part of the accumulator before the next tile's epilogue.
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

    for mi in gl.static_range(BM // MBM):
        for ni in gl.static_range(BN // MBN):
            sub = gl.amd.slice(acc, [MBM, MBN], [mi * MBM, ni * MBN])

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

    cl_a: gl.constexpr = tuning_cfg.dot_operand_copy_layout(0)
    cl_b: gl.constexpr = tuning_cfg.dot_operand_copy_layout(1)

    rows_a = _gather_rows(
        rt, block_id, M_e, start_m, gl.SliceLayout(1, cl_a), BM, func_cfg.has_gather
    )
    a_offs = (
        rows_a[:, None] * a.stride_m
        + gl.arange(0, PK_A, layout=gl.SliceLayout(0, cl_a))[None, :]
    )
    b_offs = (
        gl.arange(0, PK_B, layout=gl.SliceLayout(1, cl_b))[:, None] * b.stride_k
        + (pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, cl_b)))[None, :]
        * b.stride_n
    )

    if require_constexpr(func_cfg.a_has_scale()):
        if require_constexpr(tuning_cfg.scale_via_lds(0)):
            asl: gl.constexpr = tuning_cfg.dot_operand_scale_copy_layout(0)
        else:
            asl: gl.constexpr = tuning_cfg.dot_operand_scale_fragment_layout(0)
        rows_as = _gather_rows(
            rt, block_id, M_e, start_m, gl.SliceLayout(1, asl), BM, func_cfg.has_gather
        )
        a_scale_offs = (
            rows_as[:, None] * a.scale_stride_m
            + gl.arange(0, SK, layout=gl.SliceLayout(0, asl))[None, :]
            * a.scale_stride_k
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
        b_scale_offs = (pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, bsl)))[
            :, None
        ] * b.scale_stride_n + gl.arange(0, SK, layout=gl.SliceLayout(0, bsl))[
            None, :
        ] * b.scale_stride_k
    else:
        b_scale_offs: gl.constexpr = None

    lds = LDSManager.alloc(func_cfg, tuning_cfg)
    acc = gl.zeros(
        [BM, BN],
        dtype=func_cfg.mma_acc_dtype,
        layout=tuning_cfg.dot_result_fragment_layout(),
    )

    # per-fill pointer bumps (K is the contiguous axis of every operand)
    a_step: gl.constexpr = PK_A
    b_step: gl.constexpr = PK_B
    s_step: gl.constexpr = SK

    a_hbm_ptr = a.ptr
    b_hbm_ptr = w_ptr
    a_scale_hbm_ptr = as_ptr
    b_scale_hbm_ptr = ws_ptr

    # prologue: every buffer filled, no mma
    for i in gl.static_range(NB):
        lds.fill_a_lds(i, a_hbm_ptr, a_offs, a_scale_hbm_ptr, a_scale_offs)
        lds.fill_b_lds(i, b_hbm_ptr, b_offs, b_scale_hbm_ptr, b_scale_offs)
        lds.commit_fill_lds()
        a_hbm_ptr = a_hbm_ptr + a_step
        b_hbm_ptr = b_hbm_ptr + b_step
        if require_constexpr(func_cfg.a_has_scale()):
            a_scale_hbm_ptr = a_scale_hbm_ptr + s_step * a.scale_stride_k
        if require_constexpr(func_cfg.b_has_scale()):
            b_scale_hbm_ptr = b_scale_hbm_ptr + s_step * b.scale_stride_k

    # Read-side scale pointers. These exist separately from the fill-side ones because
    # the register fallback (scale tile too small for a coalesced direct-to-LDS write)
    # reads the scale at *consume* time, NUM_LDS_BUFFER stages behind the fill;
    # advancing only one of the two silently re-reads the first K tile's scales for the
    # whole loop, which no compile-time check catches. Both start at the same base and
    # walk K at the same rate, just from different points in the pipeline.
    a_scale_read_ptr = as_ptr
    b_scale_read_ptr = ws_ptr
    MAIN: gl.constexpr = NUM_K - NB
    # steps left after the peel, rounded down to a whole number of unrolled bodies
    UNROLLED: gl.constexpr = ((MAIN - 1) // tuning_cfg.K_UNROLL) * tuning_cfg.K_UNROLL
    WAIT_BUF_NUM: gl.constexpr = NB - 2

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

    # Prologue read: stage 0, whose MFMA window has nothing before it, so only its head
    # is dotted here. Its tail seeds the carried fragments the first loop step consumes.
    lds.wait_fill_lds_num_buf(WAIT_BUF_NUM)
    frags = _stage_frags(
        lds,
        0,
        a_scale_read_ptr,
        a_scale_offs,
        b_scale_read_ptr,
        b_scale_offs,
        func_cfg,
        tuning_cfg,
    )
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_read_ptr = a_scale_read_ptr + s_step * a_scale_stride_k
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_read_ptr = b_scale_read_ptr + s_step * b_scale_stride_k
    acc = _stage_dot(frags, acc, NUM_MINI - PF_MINI, func_cfg)
    st = _PipelineState(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_ptr,
        b_scale_read_ptr,
        _take_minis(frags, NUM_MINI - PF_MINI, PF_MINI),
        acc,
    )

    # The whole main sequence is guarded: MAIN is 0 when NUM_K == NUM_LDS_BUFFER, and
    # then every stage belongs to the drain and there is no step here at all.
    if require_constexpr(MAIN > 0):
        # First step peeled out. Its MFMAs are the ones that consume the accumulator
        # while it is still visibly gl.zeros -- inside the loop that is a phi and the
        # zero is invisible.
        st = _pipeline_step(pc, st, 0, 1 % NB, WAIT_BUF_NUM, True, True)

        # steady state over the remaining MAIN-1 steps, body unrolled K_UNROLL times.
        # K_UNROLL % NUM_LDS_BUFFER == 0 is asserted in validate(), so global step
        # 1+_k+i is congruent to 1+i mod NB and the rotating index stays a static
        # offset that constant-folds the wait_group counts.
        for _k in tl.range(0, UNROLLED, tuning_cfg.K_UNROLL):
            for i in gl.static_range(tuning_cfg.K_UNROLL):
                st = _pipeline_step(
                    pc, st, (1 + i) % NB, (2 + i) % NB, WAIT_BUF_NUM, True, True
                )

        # remainder (0 .. K_UNROLL-1 steps); NUM_K is constexpr so this stays static
        for j in gl.static_range(1 + UNROLLED, MAIN):
            st = _pipeline_step(pc, st, j % NB, (j + 1) % NB, WAIT_BUF_NUM, True, True)

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
