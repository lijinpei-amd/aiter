# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One payload/scale pipeline for shared and independent buffer depths.

A step named ``r`` selects the K tile ``r`` and accumulates K tile ``r - 1``.
Each component has baseline live span ``L = (D - 1) * R + 1`` payload steps.
An LDS producer is selected after ``F = L`` steps; a direct-register producer
skips the DS-read stage and is selected after ``F = L - 1`` steps.  Register
rings keep fixed SSA slots throughout each complete unrolled body; only the
finite prologue, remainder, and drain rotate the logical queues.

The expansion follows ``pc.schedule``, the build's :class:`~._schedule.Schedule`:
it says which steps exist, and each emitted step is passed its index there as
``STEP``, from which every per-step decision -- its K phase, copies, selections,
commits and waits -- is read.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import optional as _opt
from ._lang import pick_warp_pipeline_stage as pick_stage
from ._lang import require_constexpr
from ._layout import KernelLayout
from ._lds import LDSManager
from ._mfma import _maybe_block_dot
from ._schedule import Schedule
from ._types import A_PAYLOAD, A_SCALE, B_PAYLOAD, B_SCALE, _slot_index

_A_PAYLOAD: gl.constexpr = gl.constexpr(A_PAYLOAD)
_A_SCALE: gl.constexpr = gl.constexpr(A_SCALE)
_B_PAYLOAD: gl.constexpr = gl.constexpr(B_PAYLOAD)
_B_SCALE: gl.constexpr = gl.constexpr(B_SCALE)
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


@aggregate
@strip_annotate
class _PipelineConst:
    """Loop-invariant data for a K-pipeline step.

    Hoisted once before the loop: LDS buffers, HBM offset grids, and per-step pointer
    increments. Only :class:`_PipelinePointers` and :class:`_PipelineRegFragments`
    cross the back edge.

    Every offset grid is a *tuple*, one entry per non-K slot: ``num_m_slots`` for the A
    side, ``num_n_slots`` for the B side. Each entry is laid out for its own slot's
    copy layout, which is what keeps every per-slot direct-to-LDS copy as wide and
    as coalesced as the whole-tile copy it replaces.

    ``a_scale_hbm_offs``/``b_scale_hbm_offs`` serve both direct-to-LDS and register
    loads. Both walk K by advancing their scalar base pointer in
    :class:`_PipelinePointers` instead of the offset grid, so the offset grid is
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
    layout: KernelLayout
    schedule: Schedule

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
        layout,
        schedule,
    ):
        self.lds_ptrs = lds_ptrs
        self.a_hbm_offs = a_hbm_offs
        self.b_hbm_offs = b_hbm_offs
        self.a_scale_hbm_offs = _opt(a_scale_hbm_offs)
        self.b_scale_hbm_offs = _opt(b_scale_hbm_offs)
        # re-wrapped: the frontend unwraps a constexpr argument before a
        # constexpr_function sees it, so these arrive as plain ints
        self.a_scale_stride_k = gl.constexpr(a_scale_stride_k)
        self.b_scale_stride_k = gl.constexpr(b_scale_stride_k)
        self.a_step = gl.constexpr(a_step)
        self.b_step = gl.constexpr(b_step)
        self.s_step = gl.constexpr(s_step)
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg
        self.layout = layout
        self.schedule = schedule


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

    Operand tuples are ordered by non-K slot, then whole-stage K fragment. An unscaled
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
def _index(
    sched,
    step,
    STEP: gl.constexpr,
    component: gl.constexpr,
    FILL: gl.constexpr,
):
    """LDS ring index the step reads (with ``FILL``, writes) for a component."""
    static: gl.constexpr = sched.static_lds_index(STEP, component, FILL)
    if require_constexpr(static is not None):
        out = static
    else:
        depth: gl.constexpr = sched.ring_depth(component)
        ratio: gl.constexpr = sched.ratio_k(component)
        # fill_span, not live_span: they differ only for a direct-register
        # component, and FILL is only ever set from the LDS branches where the
        # two agree. Spelling it this way means the ring index stays right if a
        # register component is ever routed through here.
        advance: gl.constexpr = sched.fill_span(component) - 1 if FILL else 0
        k_tile = (step + advance) // ratio
        if require_constexpr(sched.register_period == 1):
            # Avoid signed ring arithmetic without disturbing live register queues.
            gl.assume(k_tile >= 0)
        out = k_tile % depth
    return out


@gluon.jit
def _replace_fragments(queue, fragments, OFFSET: gl.constexpr):
    out = ()
    for j in gl.static_range(len(queue)):
        if require_constexpr(
            OFFSET <= j and j < OFFSET + len(fragments)  # noqa: PLR1716
        ):
            out += (fragments[j - OFFSET],)
        else:
            out += (queue[j],)
    return out


@gluon.jit
def _rotate(queue, SHIFT: gl.constexpr):
    out = ()
    for j in gl.static_range(len(queue)):
        out += (queue[(j + SHIFT) % len(queue)],)
    return out


@gluon.jit
def _rotate_buffers(buffers, sched, layout, PHASE: gl.constexpr = 0):
    mk: gl.constexpr = layout.num_k_slots
    if require_constexpr(sched.selects_at_phase(_A_SCALE, PHASE + 1)):
        a_scale = _rotate(buffers[2], layout.num_m_slots * mk * sched.ratio_k(_A_SCALE))
    else:
        a_scale = buffers[2]
    if require_constexpr(sched.selects_at_phase(_B_SCALE, PHASE + 1)):
        b_scale = _rotate(buffers[3], layout.num_n_slots * mk * sched.ratio_k(_B_SCALE))
    else:
        b_scale = buffers[3]
    return (
        _rotate(buffers[0], layout.num_m_slots * mk),
        _rotate(buffers[1], layout.num_n_slots * mk),
        a_scale,
        b_scale,
    )


@gluon.jit
def _init_buffers(pc):
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    a, b, a_scale, b_scale = (), (), (), ()
    if require_constexpr(sched.in_reg(_A_PAYLOAD)):
        for _ in gl.static_range(
            tc.num_buffers(0) * layout.num_m_slots * layout.num_k_slots
        ):
            a += (
                gl.zeros(
                    layout.fragment_shape_slot(0, False),
                    pc.func_cfg.operand_elem_ty(0),
                    layout.fragment_layout(0, False),
                ),
            )
    if require_constexpr(sched.in_reg(_B_PAYLOAD)):
        for _ in gl.static_range(
            tc.num_buffers(1) * layout.num_n_slots * layout.num_k_slots
        ):
            b += (
                gl.zeros(
                    layout.fragment_shape_slot(1, False),
                    pc.func_cfg.operand_elem_ty(1),
                    layout.fragment_layout(1, False),
                ),
            )
    if require_constexpr(sched.present(_A_SCALE) and sched.in_reg(_A_SCALE)):
        for _ in gl.static_range(
            tc.num_buffers(0, True)
            * layout.num_m_slots
            * layout.num_k_slots
            * sched.ratio_k(_A_SCALE)
        ):
            a_scale += (
                gl.zeros(
                    layout.fragment_shape_slot(0, True),
                    layout.scale_fragment_dtype(0),
                    layout.fragment_layout(0, True),
                ),
            )
    if require_constexpr(sched.present(_B_SCALE) and sched.in_reg(_B_SCALE)):
        for _ in gl.static_range(
            tc.num_buffers(1, True)
            * layout.num_n_slots
            * layout.num_k_slots
            * sched.ratio_k(_B_SCALE)
        ):
            b_scale += (
                gl.zeros(
                    layout.fragment_shape_slot(1, True),
                    layout.scale_fragment_dtype(1),
                    layout.fragment_layout(1, True),
                ),
            )
    return a, b, a_scale, b_scale


@gluon.jit
def _fill_slot(
    pc,
    ptrs,
    buffers,
    step,
    STEP: gl.constexpr,
    mi: gl.constexpr,
    ni: gl.constexpr,
    MARK_TILE: gl.constexpr = True,
):
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    nm: gl.constexpr = layout.num_m_slots
    nn: gl.constexpr = layout.num_n_slots
    # One query for what this slot issues this step. The four branches below
    # stay separate because they perform different JIT-visible pointer and
    # tuple operations, but they no longer each restate the gate.
    ops: gl.constexpr = sched.issued(STEP, _slot_index(mi, ni, nm, nn))
    mk: gl.constexpr = layout.num_k_slots
    offset_step: gl.constexpr = sched.ki(STEP) if sched.soff_unroll(STEP) else 0
    a_soff: gl.constexpr = (
        offset_step
        * pc.a_step
        * (pc.func_cfg.operand_elem_ty(0).primitive_bitwidth // 8)
    )
    b_soff: gl.constexpr = (
        offset_step
        * pc.b_step
        * (pc.func_cfg.operand_elem_ty(1).primitive_bitwidth // 8)
    )
    as_soff: gl.constexpr = (
        sched.scale_soff_steps(_A_SCALE, offset_step) * pc.s_step * pc.a_scale_stride_k
        if offset_step and pc.func_cfg.has_scale(0)
        else 0
    )
    bs_soff: gl.constexpr = (
        sched.scale_soff_steps(_B_SCALE, offset_step) * pc.s_step * pc.b_scale_stride_k
        if offset_step and pc.func_cfg.has_scale(1)
        else 0
    )
    a, b, a_scale, b_scale = buffers
    if require_constexpr(ops[0] is not None):
        if require_constexpr(sched.via_lds(_A_PAYLOAD)):
            pc.lds_ptrs.buffer_load_payload(
                0,
                True,
                _index(sched, step, STEP, _A_PAYLOAD, True),
                ops[0],
                ptrs.a_hbm_ptr,
                pc.a_hbm_offs[ops[0]],
                a_soff,
            )
            if require_constexpr(sched.commit_per_copy):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_payload(
                0,
                False,
                None,
                ops[0],
                ptrs.a_hbm_ptr,
                pc.a_hbm_offs[ops[0]],
                a_soff,
            )
            a = _replace_fragments(
                a,
                fragments,
                (sched.register_index(STEP, _A_PAYLOAD, True) * nm + ops[0]) * mk,
            )
    if require_constexpr(ops[1] is not None):
        if require_constexpr(sched.via_lds(_A_SCALE)):
            pc.lds_ptrs.buffer_load_scale(
                0,
                True,
                _index(sched, step, STEP, _A_SCALE, True),
                ops[1],
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
                as_soff,
            )
            if require_constexpr(sched.commit_per_copy):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_scale(
                0,
                False,
                None,
                ops[1],
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
                as_soff,
            )
            a_scale = _replace_fragments(
                a_scale,
                fragments,
                (sched.register_index(STEP, _A_SCALE, True) * nm + ops[1])
                * mk
                * sched.ratio_k(_A_SCALE),
            )
    if require_constexpr(ops[2] is not None):
        if require_constexpr(sched.in_reg(_B_PAYLOAD)):
            fragments = pc.lds_ptrs.buffer_load_payload(
                1,
                False,
                None,
                ops[2],
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
                b_soff,
            )
            b = _replace_fragments(
                b,
                fragments,
                (sched.register_index(STEP, _B_PAYLOAD, True) * nn + ops[2]) * mk,
            )
        else:
            pc.lds_ptrs.buffer_load_payload(
                1,
                True,
                _index(sched, step, STEP, _B_PAYLOAD, True),
                ops[2],
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
                b_soff,
            )
            if require_constexpr(sched.commit_per_copy):
                pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ops[3] is not None):
        if require_constexpr(sched.via_lds(_B_SCALE)):
            pc.lds_ptrs.buffer_load_scale(
                1,
                True,
                _index(sched, step, STEP, _B_SCALE, True),
                ops[3],
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
                bs_soff,
            )
            if require_constexpr(sched.commit_per_copy):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_scale(
                1,
                False,
                None,
                ops[3],
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
                bs_soff,
            )
            b_scale = _replace_fragments(
                b_scale,
                fragments,
                (sched.register_index(STEP, _B_SCALE, True) * nn + ops[3])
                * mk
                * sched.ratio_k(_B_SCALE),
            )
    if require_constexpr(
        sched.commit_per_slot
        or (
            MARK_TILE
            and sched.commit_per_tile
            and mi == layout.num_m_slots - 1
            and ni == layout.num_n_slots - 1
        )
    ):
        pc.lds_ptrs.commit_buffer_load()
    return a, b, a_scale, b_scale


@gluon.jit
def _advance(pc, ptrs, STEP: gl.constexpr):
    sched: gl.constexpr = pc.schedule
    a, b, a_scale, b_scale = (
        ptrs.a_hbm_ptr,
        ptrs.b_hbm_ptr,
        ptrs.a_scale_hbm_ptr,
        ptrs.b_scale_hbm_ptr,
    )
    soff: gl.constexpr = sched.soff_unroll(STEP)
    steps: gl.constexpr = sched.unroll if soff else 1
    if require_constexpr(not soff or sched.ki(STEP) + 1 == sched.unroll):
        if require_constexpr(sched.loads(STEP, _A_PAYLOAD)):
            a += steps * pc.a_step
        if require_constexpr(sched.loads(STEP, _B_PAYLOAD)):
            b += steps * pc.b_step
        if require_constexpr(
            sched.active(STEP, _A_SCALE) and (soff or sched.loads(STEP, _A_SCALE))
        ):
            a_scale += (
                (steps if soff else sched.ratio_k(_A_SCALE))
                * pc.s_step
                * pc.a_scale_stride_k
            )
        if require_constexpr(
            sched.active(STEP, _B_SCALE) and (soff or sched.loads(STEP, _B_SCALE))
        ):
            b_scale += (
                (steps if soff else sched.ratio_k(_B_SCALE))
                * pc.s_step
                * pc.b_scale_stride_k
            )
    return _PipelinePointers(a, b, a_scale, b_scale)


@gluon.jit
def _read_payload(
    pc,
    buffers,
    step,
    STEP: gl.constexpr,
    non_k_slot: gl.constexpr,
    operand: gl.constexpr,
):
    """Read one non-K slot of an operand's payload, from LDS or its register ring."""
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    component: gl.constexpr = (operand, False)
    reg_payload: gl.constexpr = sched.in_reg(component)
    frags = ()
    for k in gl.static_range(layout.num_k_slots):
        payload, _ = pc.lds_ptrs.ds_read_frag(
            operand,
            _index(sched, step, STEP, component, False),
            non_k_slot,
            k,
            READ_PAYLOAD=not reg_payload,
            READ_SCALE=False,
        )
        if require_constexpr(reg_payload):
            payload = buffers[operand][
                (
                    sched.register_index(STEP, component)
                    * (layout.num_m_slots if operand == 0 else layout.num_n_slots)
                    + non_k_slot
                )
                * layout.num_k_slots
                + k
            ]
        frags += (payload,)
    return frags


@gluon.jit
def _read_scale(
    pc,
    buffers,
    step,
    STEP: gl.constexpr,
    non_k_slot: gl.constexpr,
    operand: gl.constexpr,
):
    """Read all fragments shared by one non-K slot of a scale and its R K steps."""
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    component: gl.constexpr = (operand, True)
    if require_constexpr(sched.via_lds(component)):
        fragments = pc.lds_ptrs.ds_read_scale(
            operand,
            _index(sched, step, STEP, component, False),
            non_k_slot,
        )
    else:
        buffer: gl.constexpr = sched.register_index(STEP, component)
        non_k_slots: gl.constexpr = layout.num_lds_slots_non_k(operand)
        k_fragments: gl.constexpr = layout.num_k_slots * sched.ratio_k(component)
        offset: gl.constexpr = (buffer * non_k_slots + non_k_slot) * k_fragments
        slot_nonk: gl.constexpr = layout.slot_nonk(operand)
        ratio: gl.constexpr = layout.scale_slot_nonk(operand) // slot_nonk
        fragments = ()
        for k in gl.static_range(ratio * k_fragments):
            fragments += (buffers[operand + 2][offset + k],)
    return fragments


@gluon.jit
def _read_slot(
    pc,
    buffers,
    step,
    STEP: gl.constexpr,
    mi: gl.constexpr,
    ni: gl.constexpr,
    MFMA: gl.constexpr,
):
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    slot: gl.constexpr = _slot_index(mi, ni, layout.num_m_slots, layout.num_n_slots)
    at: gl.constexpr = sched.selected(slot, _A_PAYLOAD)
    ast: gl.constexpr = sched.selected(slot, _A_SCALE)
    bt: gl.constexpr = sched.selected(slot, _B_PAYLOAD)
    bst: gl.constexpr = sched.selected(slot, _B_SCALE)
    a, b, a_scale, b_scale = (), (), (), ()
    if require_constexpr(at is not None and sched.read_in_mfma(_A_PAYLOAD, at) == MFMA):
        a = _read_payload(pc, buffers, step, STEP, at, 0)
    if require_constexpr(
        ast is not None
        and sched.selects(STEP, _A_SCALE)
        and sched.read_in_mfma(_A_SCALE, ast) == MFMA
    ):
        a_scale = _read_scale(pc, buffers, step, STEP, ast, 0)
    if require_constexpr(bt is not None and sched.read_in_mfma(_B_PAYLOAD, bt) == MFMA):
        b = _read_payload(pc, buffers, step, STEP, bt, 1)
    if require_constexpr(
        bst is not None
        and sched.selects(STEP, _B_SCALE)
        and sched.read_in_mfma(_B_SCALE, bst) == MFMA
    ):
        b_scale = _read_scale(pc, buffers, step, STEP, bst, 1)
    return a, b, a_scale, b_scale


@gluon.jit
def _take_operand_pairs(
    payload,
    scale,
    tc,
    layout,
    operand: gl.constexpr,
    non_k_slot: gl.constexpr,
    PHASE: gl.constexpr,
):
    ratio_k: gl.constexpr = layout.scale_slot_k(operand) // layout.scale_block_k
    out = ()
    for k in gl.static_range(layout.num_k_slots):
        if require_constexpr(tc.func_cfg.has_scale(operand)):
            out += (
                payload[non_k_slot * layout.num_k_slots + k],
                scale[
                    (non_k_slot * ratio_k + PHASE % ratio_k) * layout.num_k_slots + k
                ],
            )
        else:
            out += (
                payload[non_k_slot * layout.num_k_slots + k],
                scale[non_k_slot * layout.num_k_slots + k],
            )
    return out


@gluon.jit
def _step_live(pc, ptrs, buffers, regs, step, STEP: gl.constexpr):
    """Advance payloads and independently phased scale producers/selections.

    ``step`` is the K tile the step selects, a runtime value inside the loops;
    ``STEP`` is the step's index in ``pc.schedule``.
    """
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    PHASE: gl.constexpr = sched.phase(STEP)
    IN_LOOP: gl.constexpr = sched.in_loop(STEP)
    DOT: gl.constexpr = sched.dot(STEP)
    warp: gl.constexpr = tc.warp_pipeline_compiler() and IN_LOOP
    region: gl.constexpr = pick_stage(warp)
    nm: gl.constexpr = layout.num_m_slots
    nn: gl.constexpr = layout.num_n_slots
    mk: gl.constexpr = layout.num_k_slots
    a, b, acc = (), (), ()
    a_scale = () if sched.selects(STEP, _A_SCALE) else regs.a_scale
    b_scale = () if sched.selects(STEP, _B_SCALE) else regs.b_scale
    tile_wait: gl.constexpr = sched.wait(STEP)
    if require_constexpr(tile_wait is not None):
        pc.lds_ptrs.wait_buffer_load_groups(tile_wait)
    if require_constexpr(tc.SCHED_MODE != 0 and not warp):
        _sched_hint(tc.SCHED_MODE)
    for ni in gl.static_range(nn):
        for mi in gl.static_range(nm):
            if require_constexpr(
                sched.wait(STEP, _slot_index(mi, ni, nm, nn)) is not None
            ):
                pc.lds_ptrs.wait_buffer_load_groups(
                    sched.wait(STEP, _slot_index(mi, ni, nm, nn))
                )
            if require_constexpr(not DOT and mi == 0 and ni == 0):
                gl.amd.cdna4.sched_barrier(0)
            with region("mem"):
                am, bm, asm, bsm = _read_slot(pc, buffers, step, STEP, mi, ni, False)
                buffers = _fill_slot(
                    pc, ptrs, buffers, step, STEP, mi, ni, MARK_TILE=False
                )
                if require_constexpr(
                    sched.commit_per_tile
                    and not sched.tile_commit_after_mfma
                    and mi == nm - 1
                    and ni == nn - 1
                ):
                    pc.lds_ptrs.commit_buffer_load()
            with region("mfma"):
                if require_constexpr(DOT):
                    dot_a = _take_operand_pairs(
                        regs.a_payload, regs.a_scale, tc, layout, 0, mi, PHASE - 1
                    )
                    dot_b = _take_operand_pairs(
                        regs.b_payload, regs.b_scale, tc, layout, 1, ni, PHASE - 1
                    )
                else:
                    dot_a, dot_b = (), ()
                value = _maybe_block_dot(
                    dot_a,
                    dot_b,
                    regs.acc[_slot_index(mi, ni, nm, nn)],
                    mk,
                    pc.func_cfg,
                    tc,
                    layout,
                    DOT,
                    (PHASE - 1) * tc.BLOCK_K // 128 % 2,
                )
                af, bf, asf, bsf = _read_slot(pc, buffers, step, STEP, mi, ni, True)
                if require_constexpr(mi == nm - 1 and ni == nn - 1):
                    ptrs = _advance(pc, ptrs, STEP)
            if require_constexpr(
                sched.commit_per_tile
                and sched.tile_commit_after_mfma
                and mi == nm - 1
                and ni == nn - 1
            ):
                with region("commit"):
                    pc.lds_ptrs.commit_buffer_load()
            a += am + af
            b += bm + bf
            a_scale += asm + asf
            b_scale += bsm + bsf
            acc += (value,)
    # The compiler warp pipeline supplies the cross-stage ordering itself and
    # rejects explicit barriers in its staged loop. Non-warp loops need this
    # boundary for every kind of direct-register producer, not only B payloads.
    if require_constexpr(IN_LOOP and sched.has_register_component() and not warp):
        gl.amd.cdna4.sched_barrier(0)
    if require_constexpr(not IN_LOOP):
        buffers = _rotate_buffers(buffers, sched, layout, PHASE)
    if require_constexpr(not pc.func_cfg.has_scale(0)):
        a_scale = a
    if require_constexpr(not pc.func_cfg.has_scale(1)):
        b_scale = b
    return ptrs, buffers, _PipelineRegFragments(a, a_scale, b, b_scale, acc)


_step = _step_live


@gluon.jit
def _run_buffered_pipeline(pc, ptrs, NUM_K):
    """Run the prologue and MAIN iterations; leave the drain to overlap the epilogue."""
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    gl.static_assert(
        not tc.warp_pipeline_manual(),
        "the live pipeline requires WARP_PIPELINE=0 or 1",
    )
    gl.static_assert(
        layout.num_prefetch_k_slots == layout.num_k_slots,
        "the pipeline requires VGPR_PREFETCH_K == BLOCK_K",
    )
    depth: gl.constexpr = sched.depth
    unroll: gl.constexpr = sched.unroll
    peeled: gl.constexpr = sched.peeled
    main = NUM_K - depth
    gl.assume(main >= peeled + unroll)
    remaining = main - peeled
    unroll_end = remaining // unroll * unroll
    buffers = _init_buffers(pc)
    # The relative prologue stage starts at 1 - PIPELINE_DEPTH; each component
    # begins filling when that stage reaches 1 - its own fill span.
    for r in gl.static_range(1 - depth, 0):
        for ni in gl.static_range(layout.num_n_slots):
            for mi in gl.static_range(layout.num_m_slots):
                buffers = _fill_slot(
                    pc,
                    ptrs,
                    buffers,
                    r,
                    sched.prologue_step(r),
                    mi,
                    ni,
                )
        ptrs = _advance(pc, ptrs, sched.prologue_step(r))
        buffers = _rotate_buffers(buffers, sched, layout, r)
    acc = ()
    for ni in gl.static_range(layout.num_n_slots):
        for mi in gl.static_range(layout.num_m_slots):
            acc += (
                gl.zeros(
                    layout.accumulator_fragment_shape_slot,
                    pc.func_cfg.mma_acc_dtype,
                    layout.dot_result_fragment_layout,
                ),
            )
    regs = _PipelineRegFragments((), (), (), (), acc)
    ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, 0, sched.warmup_step())
    for x in gl.static_range(peeled):
        ptrs, buffers, regs = _step(
            pc, ptrs, buffers, regs, x + 1, sched.peeled_step(x)
        )
    for base in tl.range(0, unroll_end, unroll):
        for u in gl.static_range(unroll):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                peeled + base + u + 1,
                sched.loop_step(u),
            )
    if require_constexpr(sched.static_remainder):
        # Static expansion keeps short remainders and their LDS indices constant.
        for u in gl.static_range(unroll - 1):
            if unroll_end + u < remaining:
                ptrs, buffers, regs = _step(
                    pc,
                    ptrs,
                    buffers,
                    regs,
                    peeled + unroll_end + u + 1,
                    sched.remainder_step(u),
                )
    else:
        # A single runtime remainder loop avoids the register pressure of a chain
        # of guarded tuple updates. The ring rotates only in this remainder loop.
        for u in tl.range(0, remaining - unroll_end):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                peeled + unroll_end + u + 1,
                sched.remainder_step(0),
            )
    return ptrs, buffers, regs


@gluon.jit
def _drain_buffered_pipeline(pc, ptrs, buffers, regs, NUM_K):
    # Precondition: NUM_K == layout.num_k_tiles(K). The drain's region selection
    # uses the runtime NUM_K while its absolute scale phase comes from the
    # schedule's constexpr num_k, so the two must name the same strip. Every
    # production launch passes K // BLOCK_K for NUM_K, which satisfies it. It matters
    # only for a scale K-step ratio above one or BLOCK_K == 128, where the phase
    # selects which scale block or which half of a packed word the MFMA consumes.
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    sched: gl.constexpr = pc.schedule
    main = NUM_K - sched.depth
    if require_constexpr(sched.static_drain):
        phase = main % sched.ring_period
        for p in gl.static_range(sched.ring_period):
            if phase == p:
                for j in gl.static_range(sched.depth - 1):
                    ptrs, buffers, regs = _step(
                        pc,
                        ptrs,
                        buffers,
                        regs,
                        main + j + 1,
                        sched.drain_step(p, j),
                    )
    else:
        for j in gl.static_range(sched.depth - 1):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                main + j + 1,
                sched.drain_step(0, j),
            )
    a_scale, b_scale = (), ()
    for mi in gl.static_range(layout.num_m_slots):
        pairs = _take_operand_pairs(
            regs.a_payload, regs.a_scale, tc, layout, 0, mi, sched.num_k - 1
        )
        for k in gl.static_range(layout.num_k_slots):
            a_scale += (pairs[2 * k + 1],)
    for ni in gl.static_range(layout.num_n_slots):
        pairs = _take_operand_pairs(
            regs.b_payload, regs.b_scale, tc, layout, 1, ni, sched.num_k - 1
        )
        for k in gl.static_range(layout.num_k_slots):
            b_scale += (pairs[2 * k + 1],)
    return _PipelineRegFragments(
        regs.a_payload, a_scale, regs.b_payload, b_scale, regs.acc
    )


@gluon.jit
def _last_mfma(pc, regs):
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    acc = ()
    for ni in gl.static_range(layout.num_n_slots):
        for mi in gl.static_range(layout.num_m_slots):
            acc += (
                _maybe_block_dot(
                    _take_reg_pairs(
                        regs.a_payload,
                        regs.a_scale,
                        mi * layout.num_k_slots,
                        layout.num_k_slots,
                    ),
                    _take_reg_pairs(
                        regs.b_payload,
                        regs.b_scale,
                        ni * layout.num_k_slots,
                        layout.num_k_slots,
                    ),
                    regs.acc[
                        _slot_index(
                            mi,
                            ni,
                            layout.num_m_slots,
                            layout.num_n_slots,
                        )
                    ],
                    layout.num_k_slots,
                    pc.func_cfg,
                    tc,
                    layout,
                    True,
                    (pc.schedule.num_k - 1) * tc.BLOCK_K // 128 % 2,
                ),
            )
    return acc
