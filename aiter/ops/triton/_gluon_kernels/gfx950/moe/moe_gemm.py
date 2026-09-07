# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950 / CDNA4) MoE grouped GEMM kernels.

The entry points in :mod:`._entry` share **one** ``@gluon.jit`` body:

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
* stage 1 gate/up arrive pre-fused, in one of two packings along N, selected by
  ``KernelFuncConfig.gate_up_split``:

  - ``False`` (default) -- column-interleaved, even = gate, odd = up
  - ``True``  -- whole halves, gate in ``[0, I)`` and up in ``[I, 2I)``. A mini-N block
    is then one entire operand side, so the activation pairs two accumulator *tiles*
    with identical layouts instead of two registers within a lane's quad, and a warp's
    32 emitted channels are exactly one MX group. See ``_n_start``. The caller permutes
    weights, weight scales and bias with ``activations.py::gate_up_split_perm``; the
    emitted output is bit-identical either way.

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

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._triton_kernels.moe.activations import _swiglu_gate
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._epilogue import _epi_bias_tiles, _epilogue_store
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import WARP_SIZE_CE as WARP_SIZE
from ._lang import optional as _opt
from ._lang import pick_warp_pipeline_stage as pick_stage
from ._lang import require_constexpr
from ._lang import unwrap as _v
from ._lds import LDSManager
from ._offsets import (
    _a_payload_hbm_offsets,
    _a_scale_hbm_offsets,
    _b_payload_hbm_offsets,
    _b_scale_hbm_offsets,
    _n_split_offs,
    _slot_index,
)
from ._schedule import (
    _buffer_load_ops,
    _buffer_load_wait,
    _ds_read_a_tile,
    _ds_read_b_tile,
)
from ._types import DotKind, TileSched

_SG_MFMA: gl.constexpr = gl.constexpr(8)
_SG_DS_READ: gl.constexpr = gl.constexpr(256)
_SG_VMEM: gl.constexpr = gl.constexpr(16)


@gluon.jit
def _sched_hint(MODE: gl.constexpr):
    """Scheduling recipe for one K stage, emitted at the top of its region."""
    if require_constexpr(MODE == 1):
        gl.amd.cdna4.iglp_opt(0)
    elif require_constexpr(MODE == 2):
        gl.amd.cdna4.iglp_opt(1)
    elif require_constexpr(MODE == 3):
        for _ in gl.static_range(4):
            gl.amd.cdna4.sched_group_barrier(_SG_MFMA, 16, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_DS_READ, 6, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_VMEM, 4, 0)
    elif require_constexpr(MODE == 4):
        for _ in gl.static_range(8):
            gl.amd.cdna4.sched_group_barrier(_SG_MFMA, 8, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_DS_READ, 3, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_VMEM, 2, 0)


_TS_XCD_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.XCD_GROUP_M))
_TS_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.GROUP_M))
_DK_MFMA: gl.constexpr = gl.constexpr(int(DotKind.MFMA))
_DK_MFMA_SCALED: gl.constexpr = gl.constexpr(int(DotKind.MFMA_SCALED))
_DK_UPCAST_MFMA: gl.constexpr = gl.constexpr(int(DotKind.UPCAST_MFMA))


@gluon.constexpr_function
def _packed_sel(tuning_cfg, idx, k_phase=0):
    """The byte-selector list for a pre-packed scale operand, or None if it is not."""
    if tuning_cfg.scale_packed_ok(idx) and tuning_cfg.scale_via_lds(idx):
        return tuning_cfg.scale_packed_sel(idx, k_phase)
    return None


@gluon.constexpr_function
def _any_packed(tuning_cfg):
    return (
        _packed_sel(tuning_cfg, 0) is not None or _packed_sel(tuning_cfg, 1) is not None
    )


@gluon.jit
def _dot(a, a_scale, b, b_scale, acc, func_cfg, tuning_cfg, K_PHASE: gl.constexpr = 0):
    """The one matrix instruction, dispatched on the operand pair."""
    kind: gl.constexpr = func_cfg.dot_kind()
    a_sel: gl.constexpr = _packed_sel(tuning_cfg, 0, K_PHASE)
    b_sel: gl.constexpr = _packed_sel(tuning_cfg, 1, K_PHASE)
    any_packed: gl.constexpr = _any_packed(tuning_cfg)
    if require_constexpr(kind == _DK_MFMA_SCALED and any_packed):
        # At least one scale tile arrived as pre-packed dwords; the other, if any,
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


@gluon.jit
def _mini_scale_hbm_offset(base, step, HAS_SCALE: gl.constexpr):
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
def _ds_read_block(
    lds_ptrs,
    DS_READ_IDX,
    tile: gl.constexpr,
    scale_hbm_ptr,
    scale_hbm_offs,
    func_cfg,
    tc,
    operand: gl.constexpr,
    RELAXED: gl.constexpr = False,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    """One operand's fragments for a mini-M/N block, all mini-K, as a flat tuple.

    ``operand`` is 0 for A or 1 for B.

    Two entries per mini-K step -- payload, scale -- so :func:`_pipeline_step` can hold
    the tail of one stage across a pipeline iteration and hand it to :func:`_maybe_block_dot`
    one stage later.

    The pair is present even for an operand with no scale: the tuple is loop-carried, so
    Triton asks every element for its ``.type`` and a ``None`` there aborts codegen. An
    absent scale therefore parks the payload's own SSA value in the slot -- a duplicate
    reference costs no register -- and :func:`_maybe_block_dot` puts the ``None`` back from the
    same compile-time predicate.
    """
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = func_cfg.has_scale(operand)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        if require_constexpr(operand == 0):
            payload, scale = lds_ptrs.ds_read_a_frag(
                DS_READ_IDX,
                tile,
                i,
                scale_hbm_ptr,
                _mini_scale_hbm_offset(scale_hbm_offs, i * SK_MINI, HAS),
                RELAXED,
                READ_PAYLOAD,
                READ_SCALE,
            )
        else:
            payload, scale = lds_ptrs.ds_read_b_frag(
                DS_READ_IDX,
                tile,
                i,
                scale_hbm_ptr,
                _mini_scale_hbm_offset(scale_hbm_offs, i * SK_MINI, HAS),
                RELAXED,
                READ_PAYLOAD,
                READ_SCALE,
            )
        if require_constexpr(not READ_PAYLOAD):
            payload = scale
        if require_constexpr(HAS and READ_SCALE):
            slot = scale
        else:
            slot = payload
        frags = frags + (payload, slot)
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
def _maybe_block_dot(
    a_frags,
    b_frags,
    acc,
    N_MINI: gl.constexpr,
    func_cfg,
    tuning_cfg,
    DO_MFMA: gl.constexpr,
    K_PHASE: gl.constexpr = 0,
):
    """Accumulate ``N_MINI`` mini-K steps when this stage emits MFMA."""
    if require_constexpr(DO_MFMA):
        for i in gl.static_range(N_MINI):
            if require_constexpr(func_cfg.has_scale(0)):
                a_s = a_frags[2 * i + 1]
            else:
                a_s = _NO_SCALE
            if require_constexpr(func_cfg.has_scale(1)):
                b_s = b_frags[2 * i + 1]
            else:
                b_s = _NO_SCALE
            acc = _dot(
                a_frags[2 * i], a_s, b_frags[2 * i], b_s, acc, func_cfg, tuning_cfg, K_PHASE
            )
    return acc


@aggregate
@strip_annotate
class _PipelineConst:
    """Loop-invariant data for a K-pipeline step.

    Hoisted once before the loop: LDS buffers, tile offsets, and per-stage pointer
    increments. Only :class:`_PipelinePointers` and :class:`_PipelineRegFragments`
    cross the back edge.

    Every offset grid is a *tuple*, one entry per mini block: ``num_mini_m()`` for the A
    side, ``num_mini_n()`` for the B side. Each entry is laid out for its own mini tile's
    copy layout, which is what keeps every per-mini-block direct-to-LDS copy as wide and
    as coalesced as the whole-tile copy it replaces.

    ``a_scale_hbm_offs``/``b_scale_hbm_offs`` serve *both* sides: the direct-to-LDS copy and, for
    the register fallback, the consume-side load. Both walk K by advancing their scalar
    base pointer in :class:`_PipelinePointers` instead of the offset grid, so the tile
    grid is computed once and never moves.

    The two ``*_scale_stride_k`` are lifted out of the operand tuples rather than kept as
    ``a``/``b``: the non-quantised operand types have no such field at all, so it can only
    be read behind the ``has_scale`` predicate that the caller already evaluates.
    """

    lds_ptrs: LDSManager
    a_hbm_offs: tl.tuple | tuple
    b_hbm_offs: tl.tuple | tuple
    a_scale_hbm_offs: tl.tuple | tuple | gl.constexpr
    b_scale_hbm_offs: tl.tuple | tuple | gl.constexpr
    a_scale_stride_k: gl.constexpr
    b_scale_stride_k: gl.constexpr
    a_step: gl.constexpr
    b_step: gl.constexpr
    s_step: gl.constexpr
    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig

    @gluon.constexpr_function
    def __init__(
        self,
        lds_ptrs,
        a_hbm_offs,
        b_hbm_offs,
        a_scale_hbm_offs,
        b_scale_hbm_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
    ):
        self.lds_ptrs = lds_ptrs
        self.a_hbm_offs = a_hbm_offs
        self.b_hbm_offs = b_hbm_offs
        self.a_scale_hbm_offs = _opt(a_scale_hbm_offs)
        self.b_scale_hbm_offs = _opt(b_scale_hbm_offs)
        # re-wrapped: the frontend unwraps a constexpr argument before a
        # constexpr_function sees it, so these arrive as plain ints
        self.a_scale_stride_k = gl.constexpr(_v(a_scale_stride_k))
        self.b_scale_stride_k = gl.constexpr(_v(b_scale_stride_k))
        self.a_step = gl.constexpr(_v(a_step))
        self.b_step = gl.constexpr(_v(b_step))
        self.s_step = gl.constexpr(_v(s_step))
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg


@aggregate
@strip_annotate
class _PipelinePointers:
    """HBM addresses carried between K stages, independent of register fragments.

    Every address here is in HBM -- the LDS side of the pipeline is entirely inside
    ``_PipelineConst.lds_ptrs``. Each operand has one scale pointer: its scale loads
    either stage through LDS or go straight to registers. LDS source pointers advance
    after the stage's last buffer load; direct scale pointers advance after the stage's
    register loads. Only the selected path advances a scale pointer, so direct scales
    stay at the consumed K stage while LDS copies run ahead in the ring.
    """

    a_hbm_ptr: gl.tensor
    b_hbm_ptr: gl.tensor
    a_scale_hbm_ptr: gl.tensor | gl.constexpr
    b_scale_hbm_ptr: gl.tensor | gl.constexpr

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
    ):
        self.a_hbm_ptr = a_hbm_ptr
        self.b_hbm_ptr = b_hbm_ptr
        # _opt, not a bare assign: an operand with no scale arrives as raw Python None
        # (a8w8, bf16), which the field annotation rejects -- it has to be constexpr.
        self.a_scale_hbm_ptr = _opt(a_scale_hbm_ptr)
        self.b_scale_hbm_ptr = _opt(b_scale_hbm_ptr)


@aggregate
@strip_annotate
class _PipelineRegFragments:
    """Prefetched A/B payloads and scales, plus one accumulator per (ni, mi) slot.

    Operand tuples are ordered by non-K mini block, then mini-K step. An unscaled
    operand aliases its payload in the scale tuple; _maybe_block_dot ignores that slot.
    """

    a_payload: tl.tuple | tuple
    a_scale: tl.tuple | tuple
    b_payload: tl.tuple | tuple
    b_scale: tl.tuple | tuple
    acc: tl.tuple | tuple

    @gluon.constexpr_function
    def __init__(self, a_payload, a_scale, b_payload, b_scale, acc):
        self.a_payload = a_payload
        self.a_scale = a_scale
        self.b_payload = b_payload
        self.b_scale = b_scale
        self.acc = acc


@gluon.jit
def _take_reg_pairs(payload, scale, LO: gl.constexpr, N: gl.constexpr):
    out = ()
    for i in gl.static_range(N):
        out = out + (payload[LO + i], scale[LO + i])
    return out


@gluon.jit
def _make_reg_fragments(a_frags, b_frags, acc):
    a_payload, a_scale, b_payload, b_scale = (), (), (), ()
    for i in gl.static_range(len(a_frags) // 2):
        a_payload = a_payload + (a_frags[2 * i],)
        a_scale = a_scale + (a_frags[2 * i + 1],)
    for i in gl.static_range(len(b_frags) // 2):
        b_payload = b_payload + (b_frags[2 * i],)
        b_scale = b_scale + (b_frags[2 * i + 1],)
    return _PipelineRegFragments(a_payload, a_scale, b_payload, b_scale, acc)


@gluon.jit
def wait_per_stage(
    pc,
    STAGES_BETWEEN: gl.constexpr,
    WAIT_SLACK: gl.constexpr = 0,
):
    """Wait once before the slot walk for either per-stage commit boundary."""
    if require_constexpr(pc.tuning_cfg.commit_per_stage()):
        # Epilogue copies are newer than every group this pipeline step consumes.
        pc.lds_ptrs.wait_buffer_load_groups(STAGES_BETWEEN + WAIT_SLACK)


@gluon.jit
def wait_per_slot(
    pc,
    mi: gl.constexpr,
    ni: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    WAIT_SLACK: gl.constexpr = 0,
):
    """Wait for this slot's copies in the per-op and per-slot commit modes."""
    tc: gl.constexpr = pc.tuning_cfg
    if require_constexpr(not tc.commit_per_stage()):
        WAIT: gl.constexpr = _buffer_load_wait(tc, mi, ni, STAGES_BETWEEN, DO_BUFFER_LOAD)
        if require_constexpr(WAIT is not None):
            pc.lds_ptrs.wait_buffer_load_groups(WAIT + WAIT_SLACK)
        elif require_constexpr(
            _ds_read_a_tile(mi, ni, tc.num_mini_m(), tc.num_mini_n()) is not None
            or _ds_read_b_tile(mi, ni, tc.num_mini_m(), tc.num_mini_n()) is not None
        ):
            # Synchronous register-to-LDS staging still needs a cooperative fence.
            gl.barrier()


@gluon.constexpr_function
def _slot_advances_hbm_ptrs(mi, ni, NM, NN, KI, KU):
    """Does slot ``(mi, ni)`` own the pointer bump onto the next ``BLOCK_K`` stage?

    The stage's last slot -- and, when ``SOFF_UNROLL`` folded the body's steps into
    ``soffset``, only on the body's last step, which then bumps by ``KU`` at once. With
    the flag off ``KI``/``KU`` are ``0``/``1`` and the last term is vacuous.
    """
    return _v(mi) == _v(NM) - 1 and _v(ni) == _v(NN) - 1 and _v(KI) == _v(KU) - 1


@gluon.jit
def _buffer_load(
    pc,
    hbm_ptrs,
    BUFFER_LOAD_IDX,
    mi: gl.constexpr,
    ni: gl.constexpr,
    KI: gl.constexpr = 0,
    STAGE_MARK: gl.constexpr = True,
    K_PHASE: gl.constexpr = 0,
):
    """Issue this slot's copies and commit at the configured op/slot/stage boundary."""
    func_cfg: gl.constexpr = pc.func_cfg
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    OPS: gl.constexpr = _buffer_load_ops(tc, mi, ni)
    A_TILE: gl.constexpr = OPS[0]
    A_SC: gl.constexpr = OPS[1]
    B_TILE: gl.constexpr = OPS[2]
    B_SC: gl.constexpr = OPS[3]
    MARK_OP: gl.constexpr = tc.commit_per_op()
    MARK_SLOT: gl.constexpr = tc.commit_per_slot()
    MARK_STAGE: gl.constexpr = tc.commit_per_stage()
    # Byte displacement of this step from the body's base pointer. The *_step values are
    # in elements (they are added to a typed pointer), soffset is in bytes, so each is
    # scaled by its operand's storage width. All constexpr, so these fold into a literal
    # and the SGPR holding them is hoisted out of the loop.
    SOFF: gl.constexpr = pc.tuning_cfg.SOFF_UNROLL and KI > 0
    A_SOFF: gl.constexpr = (
        KI * pc.a_step * (func_cfg.operand_elem_ty(0).primitive_bitwidth // 8)
        if SOFF
        else 0
    )
    B_SOFF: gl.constexpr = (
        KI * pc.b_step * (func_cfg.operand_elem_ty(1).primitive_bitwidth // 8)
        if SOFF
        else 0
    )
    # E8M0 scales are byte-sized, so the scale strides are already byte counts.
    A_SSOFF: gl.constexpr = (
        tc.scale_hbm_steps(0, KI, (K_PHASE - KI) % 2) * pc.s_step * pc.a_scale_stride_k
        if SOFF and func_cfg.a_has_scale()
        else 0
    )
    B_SSOFF: gl.constexpr = (
        tc.scale_hbm_steps(1, KI, (K_PHASE - KI) % 2) * pc.s_step * pc.b_scale_stride_k
        if SOFF and func_cfg.b_has_scale()
        else 0
    )
    if require_constexpr(A_TILE is not None):
        pc.lds_ptrs.buffer_load_a_payload(
            BUFFER_LOAD_IDX, A_TILE, hbm_ptrs.a_hbm_ptr, pc.a_hbm_offs[A_TILE], A_SOFF
        )
        if require_constexpr(MARK_OP and tc.payload_via_lds(0)):
            pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(A_SC is not None):
        pc.lds_ptrs.buffer_load_a_scale(
            BUFFER_LOAD_IDX,
            A_SC,
            hbm_ptrs.a_scale_hbm_ptr,
            _opt_at(pc.a_scale_hbm_offs, A_SC, func_cfg.a_has_scale()),
            A_SSOFF,
        )
        if require_constexpr(MARK_OP):
            pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_TILE is not None):
        pc.lds_ptrs.buffer_load_b_payload(
            BUFFER_LOAD_IDX, B_TILE, hbm_ptrs.b_hbm_ptr, pc.b_hbm_offs[B_TILE], B_SOFF
        )
        if require_constexpr(MARK_OP and tc.payload_via_lds(1)):
            pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_SC is not None):
        pc.lds_ptrs.buffer_load_b_scale(
            BUFFER_LOAD_IDX,
            B_SC,
            hbm_ptrs.b_scale_hbm_ptr,
            _opt_at(pc.b_scale_hbm_offs, B_SC, func_cfg.b_has_scale()),
            B_SSOFF,
        )
        if require_constexpr(MARK_OP):
            pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(MARK_SLOT):
        # One group for everything this mini block just issued.
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(
        MARK_STAGE and STAGE_MARK and _slot_index(mi, ni, NM, NN) == NM * NN - 1
    ):
        # One group for the whole block, committed at the last slot.
        #
        # ``STAGE_MARK=False`` lets _pipeline_step_impl close the group after its walk
        # instead, which frees the last slot's loads from being pinned mid-walk. Every
        # OTHER caller must leave it True: the prologue fills NB-1 stages through its own
        # slot walk (see _moe_gemm_body), and if the mark is dropped there those stages
        # are never committed at all. The waits that depend on them then have no group to
        # count, UpdateAsyncWaitCount lowers them with no vmcnt, and the cooperative-fill
        # barriers that follow stop meaning anything -- a WAR race that shows up as
        # nondeterministic output in 7 runs out of 8, with the barrier count unchanged.
        pc.lds_ptrs.commit_buffer_load()


@gluon.jit
def _advance_hbm_ptrs(pc, hbm_ptrs, STEPS: gl.constexpr = 1, K_PHASE: gl.constexpr = 0):
    """Advance payload and LDS-staged scale sources after their buffer loads."""
    a_hbm_ptr = hbm_ptrs.a_hbm_ptr + STEPS * pc.a_step
    b_hbm_ptr = hbm_ptrs.b_hbm_ptr + STEPS * pc.b_step
    a_scale_hbm_ptr = hbm_ptrs.a_scale_hbm_ptr
    b_scale_hbm_ptr = hbm_ptrs.b_scale_hbm_ptr
    if require_constexpr(pc.func_cfg.a_has_scale() and pc.tuning_cfg.scale_via_lds(0)):
        a_scale_hbm_ptr = (
            a_scale_hbm_ptr
            + pc.tuning_cfg.scale_hbm_steps(0, STEPS, K_PHASE)
            * pc.s_step
            * pc.a_scale_stride_k
        )
    if require_constexpr(pc.func_cfg.b_has_scale() and pc.tuning_cfg.scale_via_lds(1)):
        b_scale_hbm_ptr = (
            b_scale_hbm_ptr
            + pc.tuning_cfg.scale_hbm_steps(1, STEPS, K_PHASE)
            * pc.s_step
            * pc.b_scale_stride_k
        )
    return _PipelinePointers(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    )


@gluon.jit
def _advance_scale_hbm_ptrs(pc, hbm_ptrs):
    """Advance only scales loaded straight into registers, once per consumed stage."""
    a_scale_hbm_ptr = hbm_ptrs.a_scale_hbm_ptr
    b_scale_hbm_ptr = hbm_ptrs.b_scale_hbm_ptr
    if require_constexpr(
        pc.func_cfg.a_has_scale() and not pc.tuning_cfg.scale_via_lds(0)
    ):
        a_scale_hbm_ptr = a_scale_hbm_ptr + pc.s_step * pc.a_scale_stride_k
    if require_constexpr(
        pc.func_cfg.b_has_scale() and not pc.tuning_cfg.scale_via_lds(1)
    ):
        b_scale_hbm_ptr = b_scale_hbm_ptr + pc.s_step * pc.b_scale_stride_k
    return _PipelinePointers(
        hbm_ptrs.a_hbm_ptr,
        hbm_ptrs.b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    )


@gluon.jit
def _ds_read_operand(
    pc,
    DS_READ_IDX,
    tile: gl.constexpr,
    scale_hbm_ptr,
    operand: gl.constexpr,
    RELAXED: gl.constexpr = False,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    if require_constexpr(operand == 0):
        scale_hbm_offs = pc.a_scale_hbm_offs
    else:
        scale_hbm_offs = pc.b_scale_hbm_offs
    return _ds_read_block(
        pc.lds_ptrs,
        DS_READ_IDX,
        tile,
        scale_hbm_ptr,
        _opt_at(scale_hbm_offs, tile, pc.func_cfg.has_scale(operand)),
        pc.func_cfg,
        pc.tuning_cfg,
        operand,
        RELAXED,
        READ_PAYLOAD,
        READ_SCALE,
    )


# --- per-slot constexpr binding -----------------------------------------------
# ``_ds_read_a_tile`` / ``_ds_read_b_tile`` are pure functions of the slot
# position, and the slot body wants each of them several times. They cannot be hoisted
# to a local at the top of the slot loop: ``X: gl.constexpr = ...`` there is rejected on
# the second unrolled iteration ("constexpr cannot be reassigned"), and an unannotated
# binding is materialised into a runtime ``tensor`` -- verified, even when the right-hand
# side is already a ``gl.constexpr``. A ``@gluon.jit`` body is a fresh scope, so the two
# helpers below are where the annotation is legal and each value is named once.
# ------------------------------------------------------------------------------
@gluon.jit
def _ds_read(
    pc,
    DS_READ_IDX,
    mi: gl.constexpr,
    ni: gl.constexpr,
    a_scale_hbm_ptr,
    b_scale_hbm_ptr,
    WANT_A: gl.constexpr,
    WANT_B: gl.constexpr,
    WANT_A_SCALE: gl.constexpr,
    WANT_B_SCALE: gl.constexpr,
):
    """The LDS reads slot ``(mi, ni)`` owns, as ``(a_frags, b_frags)`` to append.

    Payload and scale reads can belong to different stage groups. Missing components
    temporarily alias the present component; _merge_ds_read_frags selects the real values.
    """
    NM: gl.constexpr = pc.tuning_cfg.num_mini_m()
    NN: gl.constexpr = pc.tuning_cfg.num_mini_n()
    A_TILE: gl.constexpr = _ds_read_a_tile(mi, ni, NM, NN)
    B_TILE: gl.constexpr = _ds_read_b_tile(mi, ni, NM, NN)
    a_frags = ()
    b_frags = ()
    if require_constexpr(
        (WANT_A or (WANT_A_SCALE and pc.func_cfg.a_has_scale())) and A_TILE is not None
    ):
        a_frags = _ds_read_operand(
            pc,
            DS_READ_IDX,
            A_TILE,
            a_scale_hbm_ptr,
            0,
            True,
            WANT_A,
            WANT_A_SCALE,
        )
    if require_constexpr(
        (WANT_B or (WANT_B_SCALE and pc.func_cfg.b_has_scale())) and B_TILE is not None
    ):
        b_frags = _ds_read_operand(
            pc,
            DS_READ_IDX,
            B_TILE,
            b_scale_hbm_ptr,
            1,
            True,
            WANT_B,
            WANT_B_SCALE,
        )
    return a_frags, b_frags


@gluon.jit
def _merge_ds_read_frags(mem, mfma, tc, operand: gl.constexpr):
    if require_constexpr(len(mem) == 0):
        return mfma
    elif require_constexpr(len(mfma) == 0):
        return mem
    else:
        out = ()
        for i in gl.static_range(tc.num_mini_k()):
            if require_constexpr(tc.ds_read_in_mfma(operand)):
                payload = mfma[2 * i]
            else:
                payload = mem[2 * i]
            if require_constexpr(tc.ds_read_in_mfma(operand, scale=True)):
                scale = mfma[2 * i + 1]
            else:
                scale = mem[2 * i + 1]
            out = out + (payload, scale)
        return out


@gluon.jit
def _ds_read_tails(
    pc,
    a_cur,
    b_cur,
    mi: gl.constexpr,
    ni: gl.constexpr,
    DO_DS_READ: gl.constexpr,
):
    """The carried-fragment tail of what slot ``(mi, ni)`` just read.

    Indexed by the tile the slot actually read, not by ``(mi, ni)``: under the even
    schedule slot (0, 1) reads B(0), not B(1), so the legacy ``ni == 0`` / ``mi == 0``
    predicates would reach past the end of the half-built tuple.
    """
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_TILE: gl.constexpr = _ds_read_a_tile(mi, ni, NM, NN)
    B_TILE: gl.constexpr = _ds_read_b_tile(mi, ni, NM, NN)
    a_tail = ()
    b_tail = ()
    if require_constexpr(DO_DS_READ and A_TILE is not None):
        a_tail = _take_pairs(a_cur, A_TILE * NUM_MINI + HEAD_MINI, PF_MINI)
    if require_constexpr(DO_DS_READ and B_TILE is not None):
        b_tail = _take_pairs(b_cur, B_TILE * NUM_MINI + HEAD_MINI, PF_MINI)
    return a_tail, b_tail


@gluon.jit
def _pipeline_step(
    pc,
    hbm_ptrs,
    regs,
    BUFFER_LOAD_IDX,
    DS_READ_IDX,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_DS_READ: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
    DO_MFMA: gl.constexpr = True,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    WAIT_SLACK: gl.constexpr = 0,
    K_PHASE: gl.constexpr = 0,
):
    """Pick the live implementation or the frozen 2026-09-01 snapshot.

    ``_pipeline_step_impl`` is the one to edit; ``_pipeline_step_frozen`` preserves
    the best measured schedule. ``FROZEN_STEP=1`` runs the snapshot, so a
    refactor can be compared against the known-good schedule without a checkout.
    """
    if require_constexpr(pc.tuning_cfg.FROZEN_STEP):
        out = _pipeline_step_frozen(
            pc,
            hbm_ptrs,
            regs,
            BUFFER_LOAD_IDX,
            DS_READ_IDX,
            STAGES_BETWEEN,
            DO_BUFFER_LOAD,
            DO_DS_READ,
            IN_LOOP,
            DO_MFMA,
            KI,
            KU,
            WAIT_SLACK,
        )
    else:
        out = _pipeline_step_impl(
            pc,
            hbm_ptrs,
            regs,
            BUFFER_LOAD_IDX,
            DS_READ_IDX,
            STAGES_BETWEEN,
            DO_BUFFER_LOAD,
            DO_DS_READ,
            IN_LOOP,
            DO_MFMA,
            KI,
            KU,
            WAIT_SLACK,
            K_PHASE,
        )
    return out


@gluon.jit
def _pipeline_step_impl(
    pc,
    hbm_ptrs,
    regs,
    BUFFER_LOAD_IDX,
    DS_READ_IDX,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_DS_READ: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
    DO_MFMA: gl.constexpr = True,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    WAIT_SLACK: gl.constexpr = 0,
    K_PHASE: gl.constexpr = 0,
):
    func_cfg: gl.constexpr = pc.func_cfg
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    gl.static_assert(
        HEAD_MINI == 0,
        "the K-loop step needs VGPR_PREFETCH_K == BLOCK_K (HEAD_MINI == 0): a split "
        "stage would have to dot the head it reads in the slot that reads it, and the "
        "slot's MFMAs would then depend on the slot's own ds_read",
    )
    gl.static_assert(
        not tc.warp_pipeline_manual(),
        "WarpPipeline.MANUAL is implemented only by _pipeline_step_frozen: run it with "
        "AITER_TRITON_MOE_GLUON_FROZEN_STEP=1, or pick NONE / COMPILER",
    )
    WAR_PIPELINE_COMPILER: gl.constexpr = (
        tc.warp_pipeline_compiler() and DO_DS_READ and IN_LOOP
    )
    STAGE: gl.constexpr = pick_stage(WAR_PIPELINE_COMPILER)

    READ_A_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(0)
    READ_B_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(1)
    READ_A_SCALE_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(0, scale=True)
    READ_B_SCALE_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(1, scale=True)

    KIE: gl.constexpr = KI if tc.SOFF_UNROLL else 0
    KUE: gl.constexpr = KU if tc.SOFF_UNROLL else 1
    FILL_K_PHASE: gl.constexpr = tc.scale_k_phase(K_PHASE + tc.NUM_LDS_BUFFER - 1)
    DOT_K_PHASE: gl.constexpr = tc.scale_k_phase(K_PHASE - 1)

    a_cur = ()
    b_cur = ()
    a_tail = ()
    b_tail = ()
    acc = ()
    # Not under the warp pipeliner: it rejects anything it reads as a barrier or
    # wait inside a stage region, and the two are alternative answers to the same
    # question anyway -- iglp_opt overlaps MFMA with memory inside a wave, the
    # pipeliner does it by ping-ponging two wave groups.
    if require_constexpr(tc.SCHED_MODE != 0 and DO_DS_READ and not WAR_PIPELINE_COMPILER):
        _sched_hint(tc.SCHED_MODE)
    if require_constexpr(DO_DS_READ):
        wait_per_stage(pc, STAGES_BETWEEN, WAIT_SLACK)
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            if require_constexpr(DO_DS_READ):
                wait_per_slot(
                    pc, mi, ni, STAGES_BETWEEN, DO_BUFFER_LOAD, WAIT_SLACK
                )

            if require_constexpr(
                DO_DS_READ and not DO_MFMA and _slot_index(mi, ni, NM, NN) == 0
            ):
                # Bound prologue instruction scheduling to avoid spilling its scalar
                # temporaries across the prefetch. The existing wait and shared-memory
                # barrier provide synchronization; this adds no hardware rendezvous.
                gl.amd.cdna4.sched_barrier(0)

            with STAGE("mem"):
                a_mem, b_mem = _ds_read(
                    pc,
                    DS_READ_IDX,
                    mi,
                    ni,
                    hbm_ptrs.a_scale_hbm_ptr,
                    hbm_ptrs.b_scale_hbm_ptr,
                    DO_DS_READ and not READ_A_IN_MFMA,
                    DO_DS_READ and not READ_B_IN_MFMA,
                    DO_DS_READ and not READ_A_SCALE_IN_MFMA,
                    DO_DS_READ and not READ_B_SCALE_IN_MFMA,
                )
                if require_constexpr(DO_BUFFER_LOAD):
                    _buffer_load(
                        pc,
                        hbm_ptrs,
                        BUFFER_LOAD_IDX,
                        mi,
                        ni,
                        KI=KIE,
                        STAGE_MARK=False,
                        K_PHASE=FILL_K_PHASE,
                    )
                    if require_constexpr(
                        _slot_advances_hbm_ptrs(mi, ni, NM, NN, KIE, KUE)
                    ):
                        # Keep pointer updates inside this region, ahead of its border.
                        hbm_ptrs = _advance_hbm_ptrs(pc, hbm_ptrs, KUE, FILL_K_PHASE)

                if require_constexpr(
                    DO_BUFFER_LOAD
                    and tc.commit_per_stage_warp_pipeline()
                    and mi == NM - 1
                    and ni == NN - 1
                ):
                    # Close after the last memory work, before the region border.
                    pc.lds_ptrs.commit_buffer_load()

            with STAGE("mfma"):
                if require_constexpr(DO_MFMA):
                    dot_a = _take_reg_pairs(
                        regs.a_payload, regs.a_scale, mi * PF_MINI, PF_MINI
                    )
                    dot_b = _take_reg_pairs(
                        regs.b_payload, regs.b_scale, ni * PF_MINI, PF_MINI
                    )
                else:
                    dot_a = ()
                    dot_b = ()

                slot_acc = _maybe_block_dot(
                    dot_a,
                    dot_b,
                    regs.acc[_slot_index(mi, ni, NM, NN)],
                    PF_MINI,
                    func_cfg,
                    tc,
                    DO_MFMA,
                    DOT_K_PHASE,
                )
                a_mfma, b_mfma = _ds_read(
                    pc,
                    DS_READ_IDX,
                    mi,
                    ni,
                    hbm_ptrs.a_scale_hbm_ptr,
                    hbm_ptrs.b_scale_hbm_ptr,
                    DO_DS_READ and READ_A_IN_MFMA,
                    DO_DS_READ and READ_B_IN_MFMA,
                    DO_DS_READ and READ_A_SCALE_IN_MFMA,
                    DO_DS_READ and READ_B_SCALE_IN_MFMA,
                )
                if require_constexpr(DO_DS_READ and mi == NM - 1 and ni == NN - 1):
                    # Keep pointer arithmetic before the border so the next wait is
                    # the first operation between pipeline regions.
                    hbm_ptrs = _advance_scale_hbm_ptrs(pc, hbm_ptrs)
            if require_constexpr(
                DO_BUFFER_LOAD
                and tc.commit_per_stage_whole()
                and mi == NM - 1
                and ni == NN - 1
            ):
                # Bound the post-MFMA commit so the next wait stays outside a region.
                with STAGE("commit"):
                    pc.lds_ptrs.commit_buffer_load()
            a_cur = a_cur + _merge_ds_read_frags(a_mem, a_mfma, tc, 0)
            b_cur = b_cur + _merge_ds_read_frags(b_mem, b_mfma, tc, 1)
            acc = acc + (slot_acc,)

            a_new, b_new = _ds_read_tails(pc, a_cur, b_cur, mi, ni, DO_DS_READ)
            a_tail = a_tail + a_new
            b_tail = b_tail + b_new

    if require_constexpr(not DO_DS_READ):
        a_tail = _take_reg_pairs(regs.a_payload, regs.a_scale, 0, NM * PF_MINI)
        b_tail = _take_reg_pairs(regs.b_payload, regs.b_scale, 0, NN * PF_MINI)

    return hbm_ptrs, _make_reg_fragments(a_tail, b_tail, acc)


@gluon.jit
def _drain_last_fused(
    pc,
    regs,
    bias_tiles,
    x_static_scale,
    func_cfg,
    tuning_cfg,
    K_PHASE: gl.constexpr = 0,
):
    """The final drain step, with the gate half's activation folded into it.

    That step reads nothing, fills nothing and waits on nothing -- it only retires the
    MFMAs whose operands are already in registers -- so every other branch of
    :func:`_pipeline_step_impl` would be dead here and the walk collapses to
    :func:`_maybe_block_dot`. The preconditions are asserted rather than assumed.

    :func:`_slot_index` is N outer, so the visit order is ``(0,0), (1,0), (0,1), (1,1)``
    and every gate-side accumulator is final before the first linear-side MFMA issues.
    The silu for mini-M block ``mi`` therefore lands *between* MFMA clusters and its
    exp2/rcp retire under them.

    Returns the accumulator tuple with each ``ni == 0`` slot replaced by
    ``silu(bias + acc)`` -- the same value :func:`_epilogue_one_tile` would have
    computed, which is why it takes ``GATE_PRE`` to skip recomputing it.
    """
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    act: gl.constexpr = func_cfg.act()
    gl.static_assert(func_cfg.gu_split() and NN == 2)

    acc = ()
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            slot_acc = _maybe_block_dot(
                _take_reg_pairs(regs.a_payload, regs.a_scale, mi * PF_MINI, PF_MINI),
                _take_reg_pairs(regs.b_payload, regs.b_scale, ni * PF_MINI, PF_MINI),
                regs.acc[_slot_index(mi, ni, NM, NN)],
                PF_MINI,
                func_cfg,
                tc,
                True,
                K_PHASE,
            )
            if require_constexpr(ni == 0):
                # Gate side: everything up to and including the reciprocal, issued here
                # so it overlaps the ni == 1 clusters still to come.
                if require_constexpr(func_cfg.has_x_static_scale):
                    slot_acc = slot_acc * x_static_scale
                if require_constexpr(func_cfg.has_bias and func_cfg.epilogue < 2):
                    slot_acc = slot_acc + bias_tiles[0][None, :]
                if require_constexpr(func_cfg.epilogue == 0):
                    slot_acc = _swiglu_gate(
                        slot_acc, act.alpha, act.limit, tc.ACT_FAST_RCP
                    )
            acc = acc + (slot_acc,)
    return acc


@gluon.jit
def _moe_gemm_body(
    a,  # QuantTokenTensor
    b,  # QuantExpertTensor
    res,  # ResultTensor
    rt,  # RoutingMeta
    bias_hbm_ptr,
    stride_bias_e,
    x_static_scale_hbm_ptr,
    grid_m,
    grid_n,
    func_cfg,
    tuning_cfg,
    N: gl.constexpr,
    K: gl.constexpr,
):
    gl.static_assert(tuning_cfg.validate(N, K))

    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    BK: gl.constexpr = tuning_cfg.BLOCK_K
    NB: gl.constexpr = tuning_cfg.NUM_LDS_BUFFER
    PK_A: gl.constexpr = BK // func_cfg.a_pack_divisor()
    PK_B: gl.constexpr = BK // func_cfg.b_pack_divisor()
    SK: gl.constexpr = BK // MX_GROUP
    NUM_K: gl.constexpr = tuning_cfg.num_k_tiles(K)

    # One offset grid per mini block. Each is built at *its own* mini tile's copy
    # layout, so the split into MINI_BLOCK_M rows / MINI_BLOCK_N columns costs no
    # coalescing and no vector width -- the tile it addresses is a whole LDS allocation,
    # not a strided view of a bigger one.
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NM: gl.constexpr = tuning_cfg.num_mini_m()
    NN: gl.constexpr = tuning_cfg.num_mini_n()

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

    # Issued right after the routing scalars it depends on: the gather's table loads are
    # the longest-latency thing in the prologue, so everything below overlaps with them.
    a_hbm_offs = _a_payload_hbm_offsets(
        a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg
    )

    # Buffer ops carry a 32-bit offset (2 GB window) and V4-Pro's stacked gemm1 weight is
    # ~8.5 GB, so the expert stride is folded into the scalar base in 64-bit; a single
    # expert is ~22 MB and fits comfortably.
    b_hbm_ptr = b.ptr + expt_id.to(gl.int64) * b.stride_e
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_hbm_ptr = a.scale_ptr
    else:
        a_scale_hbm_ptr: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_hbm_ptr = b.scale_ptr + expt_id.to(gl.int64) * b.scale_stride_e
    else:
        b_scale_hbm_ptr: gl.constexpr = None

    b_hbm_offs = _b_payload_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg)

    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_hbm_offs = _a_scale_hbm_offsets(
            a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg
        )
    else:
        a_scale_hbm_offs: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_hbm_offs = _b_scale_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg)
    else:
        b_scale_hbm_offs: gl.constexpr = None

    lds_ptrs = LDSManager.alloc(func_cfg, tuning_cfg)

    # per-fill pointer bumps (K is the contiguous axis of every operand)
    a_step: gl.constexpr = PK_A
    # In the 16-column blocked layout a K stage is (BLOCK_K/2 / 16) runs of 256 B,
    # so the stage bump is BLOCK_K * 8 rather than the BLOCK_K/2 of a plain tile.
    b_step: gl.constexpr = PK_B // 16 * 256 if tuning_cfg.B_PRESHUFFLED else PK_B
    s_step: gl.constexpr = SK

    hbm_ptrs = _PipelinePointers(
        a.ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    )
    MAIN: gl.constexpr = NUM_K - NB
    # steps left after the peel, rounded down to a whole number of unrolled bodies
    UNROLLED: gl.constexpr = ((MAIN - 1) // tuning_cfg.K_UNROLL) * tuning_cfg.K_UNROLL
    # one buffer is being filled and one consumed on every step, so the read buffer's
    # groups have NUM_LDS_BUFFER - 2 whole stages behind them
    STAGES_BETWEEN: gl.constexpr = NB - 2

    # The scale strides only exist on a quantised operand, so they can only be read
    # behind the same predicate that guards every other use of them.
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_stride_k: gl.constexpr = a.scale_stride_k
    else:
        a_scale_stride_k: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_stride_k: gl.constexpr = b.scale_stride_k
    else:
        b_scale_stride_k: gl.constexpr = None
    pc = _PipelineConst(
        lds_ptrs,
        a_hbm_offs,
        b_hbm_offs,
        a_scale_hbm_offs,
        b_scale_hbm_offs,
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

    # Prologue fill: all but the *last* buffer, no mma. Walks the slot grid rather than
    # A-then-B so the commit groups land in the same order _buffer_load_wait() assumes for a
    # loop stage.
    #
    # The last buffer is held back deliberately. Filling all NB here would mean stalling
    # on buffer 0 with NB whole stages already in flight and nothing but the wait to do;
    # issuing NB-1 lets the wait drop to (NB-2)*G, and the buffer that was not filled is
    # then issued *after* the first ds_read + MFMA, which is real work for its latency to
    # hide behind. The group order is unchanged -- the held-back stage is still the
    # newest -- so every _buffer_load_wait() count downstream still holds, and by the time the
    # first pipeline step runs the outstanding set is the same (NB-1)*G either way.
    for i in gl.static_range(NB - 1):
        for ni in gl.static_range(NN):
            for mi in gl.static_range(NM):
                if require_constexpr(tuning_cfg.FROZEN_STEP):
                    _buffer_load_frozen(
                        pc,
                        i,
                        mi,
                        ni,
                        hbm_ptrs.a_hbm_ptr,
                        hbm_ptrs.b_hbm_ptr,
                        hbm_ptrs.a_scale_hbm_ptr,
                        hbm_ptrs.b_scale_hbm_ptr,
                    )
                else:
                    _buffer_load(
                        pc, hbm_ptrs, i, mi, ni, K_PHASE=tuning_cfg.scale_k_phase(i)
                    )
                if require_constexpr(mi == NM - 1 and ni == NN - 1):
                    hbm_ptrs = _advance_hbm_ptrs(
                        pc, hbm_ptrs, K_PHASE=tuning_cfg.scale_k_phase(i)
                    )

    # Only the snapshot retains its original whole-buffer prologue wait and fence.
    if require_constexpr(tuning_cfg.FROZEN_STEP):
        lds_ptrs.wait_buffer_load_groups((NB - 2) * (NM + NN))
        _frozen_prologue_fence()
    # The live prologue uses the same slot/stage waits as the steady-state step.
    # The prologue's own step: it fills the buffer held back above and issues the first
    # ds_read, but has nothing carried in to dot yet -- which is exactly a pipeline step
    # with the MFMAs switched off. Reusing _pipeline_step keeps the fill/read/wait
    # interleave and the commit-group order in one place instead of
    # a second hand-rolled copy that has to be kept in step with it.
    #
    # Holding the last fill back is what makes that possible: this step reads one buffer
    # while filling one, with NB-2 whole stages in between -- the same STAGES_BETWEEN the
    # loop runs at, so the slot waits it computes are the steady-state ones.
    #
    # This single MFMA-less step seeds the register fragments. A split stage would
    # need to dot its head here, so HEAD_MINI must be zero.
    gl.static_assert(
        HEAD_MINI == 0,
        "split-stage prologue (VGPR_PREFETCH_K < BLOCK_K) is unsupported",
    )
    acc0 = ()
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            acc0 = acc0 + (
                gl.zeros(
                    [MBM, MBN],
                    dtype=func_cfg.mma_acc_dtype,
                    layout=tuning_cfg.dot_result_fragment_layout(),
                ),
            )
    hbm_ptrs, regs = _pipeline_step(
        pc,
        hbm_ptrs,
        _PipelineRegFragments((), (), (), (), acc0),
        NB - 1,
        0,
        STAGES_BETWEEN,
        True,
        True,
        DO_MFMA=False,
    )
    # The whole main sequence is guarded: MAIN is 0 when NUM_K == NUM_LDS_BUFFER, and
    # then every stage belongs to the drain and there is no step here at all.
    if require_constexpr(MAIN > 0):
        # First step peeled out. Its MFMAs are the ones that consume the accumulator
        # while it is still visibly gl.zeros -- inside the loop that is a phi and the
        # zero is invisible.
        hbm_ptrs, regs = _pipeline_step(
            pc,
            hbm_ptrs,
            regs,
            0,
            1 % NB,
            STAGES_BETWEEN,
            True,
            True,
            K_PHASE=tuning_cfg.scale_k_phase(1),
        )

        # steady state over the remaining MAIN-1 steps, body unrolled K_UNROLL times.
        #
        # The buffer index is what decides whether every ds_read and buffer_load address
        # constant-folds. When K_UNROLL is a multiple of NB the trip lands back on the
        # buffer phase it started on, so global step 1+_k+i is congruent to 1+i mod NB
        # and the index is a literal. Otherwise it has to come from the induction
        # variable, which is what decouples the two knobs -- at a measured ~12%
        # (676 -> 756 us at the tuned shape), because the offsets then live in registers
        # instead of the instruction encoding. So: pay it only when asked for a factor
        # that does not divide.
        #
        # The wait_group counts are the same either way; _buffer_load_wait reads only
        # STAGES_BETWEEN and the slot position, never which buffer.
        for _k in tl.range(0, UNROLLED, tuning_cfg.K_UNROLL):
            for i in gl.static_range(tuning_cfg.K_UNROLL):
                if require_constexpr(tuning_cfg.K_UNROLL % NB == 0):
                    buffer_load_step = (i + 1) % NB
                    ds_read_step = (i + 2) % NB
                else:
                    buffer_load_step = (_k + i + 1) % NB
                    ds_read_step = (_k + i + 2) % NB
                hbm_ptrs, regs = _pipeline_step(
                    pc,
                    hbm_ptrs,
                    regs,
                    buffer_load_step,
                    ds_read_step,
                    STAGES_BETWEEN,
                    True,
                    True,
                    IN_LOOP=True,
                    KI=i,
                    KU=tuning_cfg.K_UNROLL,
                    K_PHASE=tuning_cfg.scale_k_phase(i + 2),
                )

        # remainder (0 .. K_UNROLL-1 steps); NUM_K is constexpr so this stays static
        for j in gl.static_range(1 + UNROLLED, MAIN):
            hbm_ptrs, regs = _pipeline_step(
                pc,
                hbm_ptrs,
                regs,
                j % NB,
                (j + 1) % NB,
                STAGES_BETWEEN,
                True,
                True,
                K_PHASE=tuning_cfg.scale_k_phase(j + 1),
            )

    if require_constexpr(func_cfg.has_bias):
        bias_hbm_base = bias_hbm_ptr + expt_id.to(gl.int64) * stride_bias_e
    else:
        bias_hbm_base: gl.constexpr = None
    if require_constexpr(func_cfg.has_gammas):
        gammas_hbm_ptr = rt.gammas + start_m
    else:
        gammas_hbm_ptr: gl.constexpr = None

    # Stage the epilogue's two vectors in LDS, allocated and filled together here rather
    # than in the prologue: the drain issues no fills of its own, so these copies have its
    # whole run of MFMAs to themselves and are done well before the epilogue's
    # wait_group(0). Keeping the allocation next to its one use also lets the shared-memory
    # allocator see that these buffers never overlap the K-loop's.
    #
    # Both tiles are flat: `.index()` drops the leading axis but keeps the layout's rank,
    # so a [rows, mini] tile cannot be viewed as one mini block. `.slice(start, length)`
    # preserves rank, so a 1-D tile slices cleanly into per-mini-block views.
    #
    # The gammas tile is EPI_T elements, not BLOCK_M: buffer_load_to_shared requires
    # exactly 32 or 128 bits per thread, so the copy has to cover one element per thread
    # of the CTA. BLOCK_M=128 over 256 threads would be half of one, so the tile runs to
    # 256 and the surplus is masked off and never read. Bias is BLOCK_N, which is already
    # a whole number of elements per thread.
    EPI_T: gl.constexpr = tuning_cfg.num_warps() * WARP_SIZE
    EPI_BPT: gl.constexpr = BN // EPI_T
    EPI_LDS: gl.constexpr = (
        EPI_T >= BM and BN % EPI_T == 0 and (EPI_BPT == 1 or EPI_BPT == 4)
    )
    EPI_SH: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    EPI_GC: gl.constexpr = gl.BlockedLayout(
        [1], [WARP_SIZE], [tuning_cfg.num_warps()], [0]
    )
    EPI_BC: gl.constexpr = gl.BlockedLayout(
        [EPI_BPT], [WARP_SIZE], [tuning_cfg.num_warps()], [0]
    )
    if require_constexpr(EPI_LDS and func_cfg.has_gammas):
        gamma_lds_ptr = gl.allocate_shared_memory(gl.float32, [EPI_T], layout=EPI_SH)
    else:
        gamma_lds_ptr: gl.constexpr = None
    if require_constexpr(EPI_LDS and func_cfg.has_bias):
        bias_lds_ptr = gl.allocate_shared_memory(gl.float32, [BN], layout=EPI_SH)
    else:
        bias_lds_ptr: gl.constexpr = None

    # They are committed as one group, which is newer than every pipeline group still in
    # flight. That inflates the outstanding count every drain wait is measured against,
    # so each step is handed WAIT_SLACK=EPI_GROUPS and its wait_group count comes out
    # numerically the same as before.
    EPI_GROUPS: gl.constexpr = (
        1 if EPI_LDS and (func_cfg.has_gammas or func_cfg.has_bias) else 0
    )
    if require_constexpr(EPI_LDS):
        if require_constexpr(func_cfg.has_gammas):
            g_offs = BM * block_id + gl.arange(0, EPI_T, layout=EPI_GC)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                gamma_lds_ptr, gammas_hbm_ptr, g_offs, mask=g_offs < M_e, other=0.0
            )
        if require_constexpr(func_cfg.has_bias):
            epi_b_offs = _n_split_offs(
                pid_n,
                gl.arange(0, BN, layout=EPI_BC),
                N,
                func_cfg,
                tuning_cfg,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bias_lds_ptr, bias_hbm_base, epi_b_offs
            )
        if require_constexpr(func_cfg.has_gammas or func_cfg.has_bias):
            # EPI_LDS is only a layout-feasibility flag, so it can be set with neither
            # vector present -- nothing was issued then, and an empty commit group would
            # still count against every drain wait.
            gl.amd.cdna4.async_copy.commit_group()

    # drain: the last NB stages are consumed with NO copy at all. This, not masking, is
    # what keeps every load inside K -- getting it wrong reads into the next expert's
    # weights while every correctness case still passes. The final step reads nothing and
    # just consumes the fragments still in registers.
    # The live path fuses the final activation; the reference keeps its pinned drain.
    FUSE: gl.constexpr = (
        not tuning_cfg.FROZEN_STEP
        and func_cfg.gu_split()
        and tuning_cfg.num_mini_n() == 2
    )
    DRAIN: gl.constexpr = NB - 1 if FUSE else NB
    for i in gl.static_range(DRAIN):
        hbm_ptrs, regs = _pipeline_step(
            pc,
            hbm_ptrs,
            regs,
            0,
            (MAIN + i + 1) % NB,
            NB - 2 - i,
            False,
            i + 1 < NB,
            WAIT_SLACK=EPI_GROUPS,
            K_PHASE=tuning_cfg.scale_k_phase(MAIN + i + 1),
        )

    # Hoisted out of the mini-tile loop: one scalar load, not one per tile.
    if require_constexpr(func_cfg.has_x_static_scale):
        x_static_scale = gl.load(x_static_scale_hbm_ptr)
    else:
        x_static_scale: gl.constexpr = None
    if require_constexpr(EPI_LDS and (func_cfg.has_gammas or func_cfg.has_bias)):
        # The copies were issued at the top of the drain in their own commit group, so
        # this retires them (and nothing else -- every pipeline group is older). The fill
        # is spread over every warp and each warp reads a whole mini block out of it, so
        # the barrier is load-bearing.
        #
        # Under FUSE this sits one step earlier, before the peeled step rather than after
        # it, because that step now consumes the bias. Every step it still follows was
        # handed WAIT_SLACK=EPI_GROUPS while the group was outstanding, and the peeled
        # step issues no wait, so no count changes.
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()

    if require_constexpr(FUSE):
        acc = _drain_last_fused(
            pc,
            regs,
            _epi_bias_tiles(
                bias_lds_ptr, bias_hbm_base, pid_n, N, func_cfg, tuning_cfg
            ),
            x_static_scale,
            func_cfg,
            tuning_cfg,
            K_PHASE=tuning_cfg.scale_k_phase(NUM_K - 1),
        )
    else:
        acc = regs.acc

    y_hbm_ptr = res.ptr + start_m.to(gl.int64) * res.stride_m
    if require_constexpr(func_cfg.output_quant is not None):
        ys_hbm_ptr = res.scale_ptr + start_m.to(gl.int64) * res.scale_stride_m
    else:
        ys_hbm_ptr: gl.constexpr = None
    _epilogue_store(
        acc,
        y_hbm_ptr,
        res.stride_m,
        res.stride_n,
        ys_hbm_ptr,
        res.scale_stride_m,
        res.scale_stride_n,
        bias_hbm_base,
        block_id,
        pid_n,
        N,
        M_e,
        gammas_hbm_ptr,
        gamma_lds_ptr,
        bias_lds_ptr,
        x_static_scale,
        func_cfg,
        tuning_cfg,
        GATE_PRE=FUSE,
    )


# Imported last, and not at the top: ``_frozen`` is a snapshot of this module's K-loop
# step and calls back into the shared halves of the kernel, so it can only be bound once
# everything it names exists. ``_pipeline_step`` looks the name up at compile time, which
# is well after this line has run.
from ._frozen import (
    _buffer_load_frozen,
    _frozen_prologue_fence,
    _pipeline_step_frozen,
)
