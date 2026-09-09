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

``N`` and ``K`` remain compile-time layout parameters. ``NUM_K`` is a separate
runtime scalar controlling the main loop and remainder. Buffer depths, prologue
predicates and drain predicates are compile-time, independent of that loop bound.

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
from ._epilogue import _epi_bias_tiles, _epilogue_store, _stage_epilogue_inputs
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import optional as _opt
from ._lang import require_constexpr
from ._lang import unwrap as _v
from ._lds import LDSManager
from ._offsets import (
    _a_payload_hbm_offsets,
    _a_scale_hbm_offsets,
    _b_payload_hbm_offsets,
    _b_scale_hbm_offsets,
    _slot_index,
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
    if not tuning_cfg.FROZEN_STEP and tuning_cfg.num_mini_k() > 1:
        return None
    if tuning_cfg.scale_packed_ok(idx) and (
        tuning_cfg.scale_via_lds(idx)
        or (not tuning_cfg.FROZEN_STEP and tuning_cfg.scale_packed_k128(idx))
    ):
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
                a_frags[2 * i],
                a_s,
                b_frags[2 * i],
                b_s,
                acc,
                func_cfg,
                tuning_cfg,
                K_PHASE,
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

    ``a_scale_hbm_offs``/``b_scale_hbm_offs`` serve both direct-to-LDS and register
    loads. Both walk K by advancing their scalar base pointer in
    :class:`_PipelinePointers` instead of the offset grid, so the tile grid is
    computed once and never moves.

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
    either stage through LDS or go straight to registers. The unified queues
    advance each stream at its fill point, so their pointers can address different
    K stages. The explicit frozen step retains its original pointer advances.
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
def _ds_read_operand(
    pc,
    DS_READ_IDX,
    tile: gl.constexpr,
    scale_hbm_ptr,
    operand: gl.constexpr,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    """One operand's fragments for a mini-M/N block, all mini-K, as a flat tuple.

    ``operand`` is 0 for A or 1 for B.

    Two entries per mini-K step -- payload, scale -- let the pipeline carry the
    fragments across an iteration and hand them to :func:`_maybe_block_dot` one
    stage later.

    The pair is present even for an operand with no scale: the tuple is loop-carried, so
    Triton asks every element for its ``.type`` and a ``None`` there aborts codegen. An
    absent scale therefore parks the payload's own SSA value in the slot -- a duplicate
    reference costs no register -- and :func:`_maybe_block_dot` puts the ``None`` back from the
    same compile-time predicate.
    """
    tc: gl.constexpr = pc.tuning_cfg
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = pc.func_cfg.has_scale(operand)
    if require_constexpr(operand == 0):
        scale_hbm_offs = pc.a_scale_hbm_offs
    else:
        scale_hbm_offs = pc.b_scale_hbm_offs
    scale_tile_hbm_offs = _opt_at(scale_hbm_offs, tile, HAS)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        payload, scale = pc.lds_ptrs.ds_read_frag(
            operand,
            DS_READ_IDX,
            tile,
            i,
            scale_hbm_ptr,
            _mini_scale_hbm_offset(scale_tile_hbm_offs, i * SK_MINI, HAS),
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
    MFMAs whose operands are already in registers. The walk therefore needs only
    :func:`_maybe_block_dot` and the activation. The preconditions are asserted.

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
    NUM_K,
):
    gl.static_assert(tuning_cfg.validate(N, K))

    BK: gl.constexpr = tuning_cfg.BLOCK_K
    PK_A: gl.constexpr = BK // func_cfg.a_pack_divisor()
    PK_B: gl.constexpr = BK // func_cfg.b_pack_divisor()
    SK: gl.constexpr = BK // MX_GROUP

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

    hbm_ptrs, buffers, regs = _run_buffered_pipeline(pc, hbm_ptrs, NUM_K)

    epi = _stage_epilogue_inputs(
        bias_hbm_ptr,
        stride_bias_e,
        rt,
        expt_id,
        start_m,
        block_id,
        pid_n,
        N,
        M_e,
        func_cfg,
        tuning_cfg,
    )

    # Shallower streams finish their remaining fills during the drain. The final
    # MFMA is separate and consumes only the fragments already in registers.
    # Fuse its activation where supported by the live step.
    FUSE: gl.constexpr = (
        not tuning_cfg.FROZEN_STEP
        and func_cfg.gu_split()
        and tuning_cfg.num_mini_n() == 2
    )
    regs = _drain_buffered_pipeline(pc, hbm_ptrs, buffers, regs, NUM_K, epi.groups)
    if require_constexpr(not FUSE):
        regs = _PipelineRegFragments(
            regs.a_payload,
            regs.a_scale,
            regs.b_payload,
            regs.b_scale,
            _last_mfma(pc, regs),
        )

    # Hoisted out of the mini-tile loop: one scalar load, not one per tile.
    if require_constexpr(func_cfg.has_x_static_scale):
        x_static_scale = gl.load(x_static_scale_hbm_ptr)
    else:
        x_static_scale: gl.constexpr = None
    if require_constexpr(epi.groups > 0):
        # Retire the epilogue copies and any newer operand groups issued during
        # the drain. The cooperative fill needs a barrier before each warp reads
        # its mini block. Under FUSE this also makes bias available to the final
        # MFMA's activation; that step issues no further copies or waits.
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()

    if require_constexpr(FUSE):
        acc = _drain_last_fused(
            pc,
            regs,
            _epi_bias_tiles(
                epi.bias_lds_ptr, epi.bias_hbm_base, pid_n, N, func_cfg, tuning_cfg
            ),
            x_static_scale,
            func_cfg,
            tuning_cfg,
            K_PHASE=0,
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
        epi.bias_hbm_base,
        block_id,
        pid_n,
        N,
        M_e,
        epi.gammas_hbm_ptr,
        epi.gamma_lds_ptr,
        epi.bias_lds_ptr,
        x_static_scale,
        func_cfg,
        tuning_cfg,
        GATE_PRE=FUSE,
    )


# Imported last: the unified driver calls shared helpers from this module.
# JIT functions resolve these names at compile time, after both modules have loaded.
from ._buffered import _drain_buffered_pipeline, _last_mfma, _run_buffered_pipeline
