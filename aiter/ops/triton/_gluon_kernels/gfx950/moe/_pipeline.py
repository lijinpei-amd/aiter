# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One payload/scale pipeline for shared and independent buffer depths.

A step named r reads payload tile r and accumulates tile r - 1. LDS payloads
fill r + NB - 1. Direct-register B fills r + NB - 2, equivalently B[k + NB - 1]
while MFMA consumes B[k], because it has no intervening DS-read stage. A scale
tile spanning R steps fills r + (NB - 1) * R every R steps. The driver peels the
seed and first MFMAs, then uses runtime main-loop bounds. Register rings keep
fixed SSA slots throughout each complete unrolled body; only the finite
prologue, remainder and drain rotate the logical queues.
"""

import math

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import pick_warp_pipeline_stage as pick_stage
from ._lang import require_constexpr
from ._layout import _slot_index, accumulator_fragment_shape_slot
from ._schedule import (
    _active,
    _ds_read_a_tile,
    _ds_read_b_tile,
    _fill_span,
    _ops,
    _pipeline_peeled,
    _read_payload_after_fill,
    _wait,
    pipeline_depth,
    pipeline_unroll,
)


@gluon.constexpr_function
def _validate_pipeline(tc, K):
    """Validate the live component rings and runtime-loop lower bound."""
    tc.validate_buffer_counts()
    num_k = tc.num_k_tiles(K)
    depth = pipeline_depth(tc)
    peeled = _pipeline_peeled(tc)
    unroll = pipeline_unroll(tc)
    assert num_k >= depth + peeled + unroll, (
        f"NUM_K ({num_k}) must be at least NB_MAX ({depth}) + PEELED ({peeled}) "
        f"+ UNROLL ({unroll}) = {depth + peeled + unroll}"
    )
    return True


@gluon.jit
def _index(
    tc,
    step,
    component: gl.constexpr,
    FILL: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
):
    operand: gl.constexpr = component[0]
    is_scale: gl.constexpr = component[1]
    if require_constexpr(is_scale and not tc.func_cfg.has_scale(operand)):
        out = 0
    else:
        depth: gl.constexpr = tc.num_buffers(operand, is_scale)
        ratio: gl.constexpr = tc.scale_ratio_k_step(operand) if is_scale else 1
        advance: gl.constexpr = depth - 1 if FILL else 0
        if require_constexpr(IN_LOOP and pipeline_unroll(tc) % (depth * ratio) == 0):
            out = ((_pipeline_peeled(tc) + KI + 1) // ratio + advance) % depth
        else:
            tile = step // ratio + advance
            if require_constexpr(tc.pipeline_register_period() == 1):
                # Avoid signed ring arithmetic without disturbing live register queues.
                gl.assume(tile >= 0)
            out = tile % depth
    return out


@gluon.jit
def _replace_tile(queue, fragments, OFFSET: gl.constexpr):
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
def _rotate(queue, TILE_COUNT: gl.constexpr):
    out = ()
    for j in gl.static_range(len(queue)):
        out += (queue[(j + TILE_COUNT) % len(queue)],)
    return out


@gluon.jit
def _rotate_buffers(buffers, tc, PHASE: gl.constexpr = 0):
    mk: gl.constexpr = tc.num_k_slots_per_tile()
    if require_constexpr((PHASE + 1) % tc.scale_ratio_k_step(0) == 0):
        a_scale = _rotate(buffers[2], tc.scale_cache_fragments(0))
    else:
        a_scale = buffers[2]
    if require_constexpr((PHASE + 1) % tc.scale_ratio_k_step(1) == 0):
        b_scale = _rotate(buffers[3], tc.scale_cache_fragments(1))
    else:
        b_scale = buffers[3]
    return (
        _rotate(buffers[0], tc.num_m_slots_per_block() * mk),
        _rotate(buffers[1], tc.num_n_slots_per_block() * mk),
        a_scale,
        b_scale,
    )


@gluon.jit
def _init_buffers(pc):
    tc: gl.constexpr = pc.tuning_cfg
    a, b, a_scale, b_scale = (), (), (), ()
    if require_constexpr(not tc.payload_via_lds(0)):
        for _ in gl.static_range(
            tc.num_buffers(0) * tc.num_m_slots_per_block() * tc.num_k_slots_per_tile()
        ):
            a += (
                gl.zeros(
                    tc.payload_fragment_shape_slot(0),
                    pc.func_cfg.operand_elem_ty(0),
                    tc.dot_operand_fragment_layout(0),
                ),
            )
    if require_constexpr(not tc.payload_via_lds(1)):
        for _ in gl.static_range(
            tc.num_buffers(1) * tc.num_n_slots_per_block() * tc.num_k_slots_per_tile()
        ):
            b += (
                gl.zeros(
                    tc.payload_fragment_shape_slot(1),
                    pc.func_cfg.operand_elem_ty(1),
                    tc.dot_operand_fragment_layout(1),
                ),
            )
    if require_constexpr(pc.func_cfg.a_has_scale() and not tc.scale_via_lds(0)):
        for _ in gl.static_range(tc.num_buffers(0, True) * tc.scale_cache_fragments(0)):
            if require_constexpr(tc.mfma_scale_selector(0) is not None):
                a_scale += (
                    gl.zeros(
                        tc.packed_scale_shape(0),
                        gl.int32,
                        tc.packed_scale_frag_layout(0),
                    ),
                )
            else:
                a_scale += (
                    gl.zeros(
                        tc.scale_fragment_shape_slot(0),
                        gl.uint8,
                        tc.dot_operand_scale_fragment_layout(0),
                    ),
                )
    if require_constexpr(pc.func_cfg.b_has_scale() and not tc.scale_via_lds(1)):
        for _ in gl.static_range(tc.num_buffers(1, True) * tc.scale_cache_fragments(1)):
            if require_constexpr(tc.mfma_scale_selector(1) is not None):
                b_scale += (
                    gl.zeros(
                        tc.packed_scale_shape(1),
                        gl.int32,
                        tc.packed_scale_frag_layout(1),
                    ),
                )
            else:
                b_scale += (
                    gl.zeros(
                        tc.scale_fragment_shape_slot(1),
                        gl.uint8,
                        tc.dot_operand_scale_fragment_layout(1),
                    ),
                )
    return a, b, a_scale, b_scale


@gluon.constexpr_function
def _soff_unroll(tc, in_loop):
    return in_loop and tc.SOFF_UNROLL


@gluon.constexpr_function
def _scale_soff_steps(tc, operand, offset_step):
    """Advance from the next scale fill at the beginning of an unrolled body."""
    ratio = tc.scale_ratio_k_step(operand)
    start = _pipeline_peeled(tc) + 1
    return ((start + offset_step) // ratio - (start + ratio - 1) // ratio) * ratio


@gluon.constexpr_function
def _scale_register_index(tc, operand, ki, in_loop, fill=False):
    ratio = tc.scale_ratio_k_step(operand)
    start = _pipeline_peeled(tc) + 1
    advance = (start + ki) // ratio - start // ratio if in_loop else 0
    return (
        advance + (tc.num_buffers(operand, True) - 1 if fill else 0)
    ) % tc.num_buffers(operand, True)


@gluon.jit
def _fill_slot(
    pc,
    ptrs,
    buffers,
    step,
    STAGE: gl.constexpr,
    DRAIN: gl.constexpr,
    mi: gl.constexpr,
    ni: gl.constexpr,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    MARK_STAGE: gl.constexpr = True,
    STATIC_PHASE: gl.constexpr = False,
    A_SCALE_LOAD: gl.constexpr = True,
    B_SCALE_LOAD: gl.constexpr = True,
):
    tc: gl.constexpr = pc.tuning_cfg
    ops: gl.constexpr = _ops(tc, mi, ni)
    mk: gl.constexpr = tc.num_k_slots_per_tile()
    offset_step: gl.constexpr = KI if _soff_unroll(tc, IN_LOOP) else 0
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
        _scale_soff_steps(tc, 0, offset_step) * pc.s_step * pc.a_scale_stride_k
        if offset_step and pc.func_cfg.a_has_scale()
        else 0
    )
    bs_soff: gl.constexpr = (
        _scale_soff_steps(tc, 1, offset_step) * pc.s_step * pc.b_scale_stride_k
        if offset_step and pc.func_cfg.b_has_scale()
        else 0
    )
    a, b, a_scale, b_scale = buffers
    if require_constexpr(ops[0] is not None and _active(tc, (0, False), STAGE, DRAIN)):
        if require_constexpr(tc.payload_via_lds(0)):
            pc.lds_ptrs.buffer_load_payload(
                0,
                True,
                _index(tc, step, (0, False), True, KI, IN_LOOP or STATIC_PHASE),
                ops[0],
                ptrs.a_hbm_ptr,
                pc.a_hbm_offs[ops[0]],
                a_soff,
            )
            if require_constexpr(tc.commit_per_op()):
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
            a = _replace_tile(
                a,
                fragments,
                (
                    ((KI if IN_LOOP else 0) + tc.num_buffers(0) - 1)
                    % tc.num_buffers(0)
                    * tc.num_m_slots_per_block()
                    + ops[0]
                )
                * mk,
            )
    if require_constexpr(
        A_SCALE_LOAD and ops[1] is not None and _active(tc, (0, True), STAGE, DRAIN)
    ):
        if require_constexpr(tc.scale_via_lds(0)):
            pc.lds_ptrs.buffer_load_scale(
                0,
                True,
                _index(tc, step, (0, True), True, KI, IN_LOOP or STATIC_PHASE),
                ops[1],
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
                as_soff,
            )
            if require_constexpr(tc.commit_per_op()):
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
            a_scale = _replace_tile(
                a_scale,
                fragments,
                _scale_register_index(tc, 0, KI, IN_LOOP, True)
                * tc.scale_cache_fragments(0)
                + ops[1] * tc.scale_read_k_slots(0),
            )
    if require_constexpr(ops[2] is not None and _active(tc, (1, False), STAGE, DRAIN)):
        if require_constexpr(not tc.payload_via_lds(1)):
            fragments = pc.lds_ptrs.buffer_load_payload(
                1,
                False,
                None,
                ops[2],
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
                b_soff,
            )
            b = _replace_tile(
                b,
                fragments,
                (
                    ((KI if IN_LOOP else 0) + _fill_span(tc, (1, False)) - 1)
                    % tc.num_buffers(1)
                    * tc.num_n_slots_per_block()
                    + ops[2]
                )
                * mk,
            )
        else:
            pc.lds_ptrs.buffer_load_payload(
                1,
                True,
                _index(tc, step, (1, False), True, KI, IN_LOOP or STATIC_PHASE),
                ops[2],
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
                b_soff,
            )
            if require_constexpr(tc.commit_per_op()):
                pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(
        B_SCALE_LOAD and ops[3] is not None and _active(tc, (1, True), STAGE, DRAIN)
    ):
        if require_constexpr(tc.scale_via_lds(1)):
            pc.lds_ptrs.buffer_load_scale(
                1,
                True,
                _index(tc, step, (1, True), True, KI, IN_LOOP or STATIC_PHASE),
                ops[3],
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
                bs_soff,
            )
            if require_constexpr(tc.commit_per_op()):
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
            b_scale = _replace_tile(
                b_scale,
                fragments,
                _scale_register_index(tc, 1, KI, IN_LOOP, True)
                * tc.scale_cache_fragments(1)
                + ops[3] * tc.scale_read_k_slots(1),
            )
    if require_constexpr(
        tc.commit_per_slot()
        or (
            MARK_STAGE
            and tc.commit_per_stage()
            and mi == tc.num_m_slots_per_block() - 1
            and ni == tc.num_n_slots_per_block() - 1
        )
    ):
        pc.lds_ptrs.commit_buffer_load()
    return a, b, a_scale, b_scale


@gluon.jit
def _advance(
    pc,
    ptrs,
    step,
    STAGE: gl.constexpr,
    DRAIN: gl.constexpr,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    STATIC_PHASE: gl.constexpr = False,
    A_SCALE_LOAD: gl.constexpr = True,
    B_SCALE_LOAD: gl.constexpr = True,
):
    tc: gl.constexpr = pc.tuning_cfg
    a, b, a_scale, b_scale = (
        ptrs.a_hbm_ptr,
        ptrs.b_hbm_ptr,
        ptrs.a_scale_hbm_ptr,
        ptrs.b_scale_hbm_ptr,
    )
    steps: gl.constexpr = pipeline_unroll(tc) if _soff_unroll(tc, IN_LOOP) else 1
    if require_constexpr(
        not _soff_unroll(tc, IN_LOOP) or KI + 1 == pipeline_unroll(tc)
    ):
        if require_constexpr(_active(tc, (0, False), STAGE, DRAIN)):
            a += steps * pc.a_step
        if require_constexpr(_active(tc, (1, False), STAGE, DRAIN)):
            b += steps * pc.b_step
        if require_constexpr(
            _active(tc, (0, True), STAGE, DRAIN)
            and (_soff_unroll(tc, IN_LOOP) or A_SCALE_LOAD)
        ):
            a_scale += (
                (steps if _soff_unroll(tc, IN_LOOP) else tc.scale_ratio_k_step(0))
                * pc.s_step
                * pc.a_scale_stride_k
            )
        if require_constexpr(
            _active(tc, (1, True), STAGE, DRAIN)
            and (_soff_unroll(tc, IN_LOOP) or B_SCALE_LOAD)
        ):
            b_scale += (
                (steps if _soff_unroll(tc, IN_LOOP) else tc.scale_ratio_k_step(1))
                * pc.s_step
                * pc.b_scale_stride_k
            )
    return _PipelinePointers(a, b, a_scale, b_scale)


@gluon.jit
def _read_tile(
    pc,
    buffers,
    step,
    tile: gl.constexpr,
    operand: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
    STATIC_PHASE: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    reg_payload: gl.constexpr = not tc.payload_via_lds(operand)
    frags = ()
    for k in gl.static_range(tc.num_k_slots_per_tile()):
        payload, _ = pc.lds_ptrs.ds_read_frag(
            operand,
            _index(tc, step, (operand, False), False, KI, IN_LOOP or STATIC_PHASE),
            tile,
            k,
            READ_PAYLOAD=not reg_payload,
            READ_SCALE=False,
        )
        if require_constexpr(reg_payload):
            payload = buffers[operand][
                (
                    (KI % tc.num_buffers(operand) if IN_LOOP else 0)
                    * (
                        tc.num_m_slots_per_block()
                        if operand == 0
                        else tc.num_n_slots_per_block()
                    )
                    + tile
                )
                * tc.num_k_slots_per_tile()
                + k
            ]
        frags += (payload,)
    return frags


@gluon.jit
def _read_scale_tile(
    pc,
    buffers,
    step,
    tile: gl.constexpr,
    operand: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
    STATIC_PHASE: gl.constexpr = False,
):
    """Read all fragments shared by one non-K scale tile and its R K steps."""
    tc: gl.constexpr = pc.tuning_cfg
    if require_constexpr(tc.scale_via_lds(operand)):
        fragments = pc.lds_ptrs.ds_read_scale(
            operand,
            _index(tc, step, (operand, True), False, KI, IN_LOOP or STATIC_PHASE),
            tile,
        )
    else:
        offset: gl.constexpr = _scale_register_index(
            tc, operand, KI, IN_LOOP
        ) * tc.scale_cache_fragments(operand) + tile * tc.scale_read_k_slots(operand)
        fragments = ()
        for k in gl.static_range(
            tc.scale_ratio_non_k_slot(operand) * tc.scale_read_k_slots(operand)
        ):
            fragments += (buffers[operand + 2][offset + k],)
    return fragments


@gluon.jit
def _read_slot(
    pc,
    ptrs,
    buffers,
    step,
    mi: gl.constexpr,
    ni: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
    MFMA: gl.constexpr,
    STATIC_PHASE: gl.constexpr = False,
    A_SCALE_LOAD: gl.constexpr = True,
    B_SCALE_LOAD: gl.constexpr = True,
):
    tc: gl.constexpr = pc.tuning_cfg
    at: gl.constexpr = _ds_read_a_tile(
        mi, ni, tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    )
    bt: gl.constexpr = _ds_read_b_tile(
        mi, ni, tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    )
    a, b, a_scale, b_scale = (), (), (), ()
    if require_constexpr(at is not None and _read_payload_after_fill(tc, 0) == MFMA):
        a = _read_tile(
            pc,
            buffers,
            step,
            at,
            0,
            KI,
            IN_LOOP,
            STATIC_PHASE,
        )
    if require_constexpr(
        A_SCALE_LOAD
        and at is not None
        and at % tc.scale_ratio_non_k_slot(0) == 0
        and pc.func_cfg.a_has_scale()
        and tc.ds_read_in_mfma(0, True) == MFMA
    ):
        a_scale = _read_scale_tile(pc, buffers, step, at, 0, KI, IN_LOOP, STATIC_PHASE)
    if require_constexpr(bt is not None and _read_payload_after_fill(tc, 1) == MFMA):
        b = _read_tile(
            pc,
            buffers,
            step,
            bt,
            1,
            KI,
            IN_LOOP,
            STATIC_PHASE,
        )
    if require_constexpr(
        B_SCALE_LOAD
        and bt is not None
        and bt % tc.scale_ratio_non_k_slot(1) == 0
        and pc.func_cfg.b_has_scale()
        and tc.ds_read_in_mfma(1, True) == MFMA
    ):
        b_scale = _read_scale_tile(pc, buffers, step, bt, 1, KI, IN_LOOP, STATIC_PHASE)
    return a, b, a_scale, b_scale


@gluon.jit
def _take_operand_pairs(
    payload, scale, tc, operand: gl.constexpr, tile: gl.constexpr, PHASE: gl.constexpr
):
    out = ()
    for k in gl.static_range(tc.num_k_slots_per_tile()):
        if require_constexpr(tc.func_cfg.has_scale(operand)):
            out += (
                payload[tile * tc.num_k_slots_per_tile() + k],
                scale[
                    tile * tc.scale_read_k_slots(operand)
                    + PHASE % tc.scale_ratio_k_step(operand) * tc.num_k_slots_per_tile()
                    + k
                ],
            )
        else:
            out += (
                payload[tile * tc.num_k_slots_per_tile() + k],
                scale[tile * tc.num_k_slots_per_tile() + k],
            )
    return out


@gluon.jit
def _step_live(
    pc,
    ptrs,
    buffers,
    regs,
    step,
    STAGE: gl.constexpr,
    DRAIN: gl.constexpr = False,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    DOT: gl.constexpr = True,
    EPILOGUE_GROUPS: gl.constexpr = 0,
    STATIC_PHASE: gl.constexpr = False,
    A_SCALE_LOAD: gl.constexpr = True,
    B_SCALE_LOAD: gl.constexpr = True,
    PHASE: gl.constexpr = 0,
):
    """Advance payloads and the scale streams enabled for this read phase."""
    tc: gl.constexpr = pc.tuning_cfg
    warp: gl.constexpr = tc.warp_pipeline_compiler() and IN_LOOP
    region: gl.constexpr = pick_stage(warp)
    nm: gl.constexpr = tc.num_m_slots_per_block()
    nn: gl.constexpr = tc.num_n_slots_per_block()
    mk: gl.constexpr = tc.num_k_slots_per_tile()
    a, b, acc = (), (), ()
    a_scale = () if A_SCALE_LOAD else regs.a_scale
    b_scale = () if B_SCALE_LOAD else regs.b_scale
    if require_constexpr(
        tc.commit_per_stage()
        and _wait(tc, STAGE, None, DRAIN, EPILOGUE_GROUPS, PHASE) is not None
    ):
        pc.lds_ptrs.wait_buffer_load_groups(
            _wait(tc, STAGE, None, DRAIN, EPILOGUE_GROUPS, PHASE)
        )
    if require_constexpr(tc.SCHED_MODE != 0 and not warp):
        _sched_hint(tc.SCHED_MODE)
    for ni in gl.static_range(nn):
        for mi in gl.static_range(nm):
            if require_constexpr(
                not tc.commit_per_stage()
                and _wait(
                    tc,
                    STAGE,
                    _slot_index(mi, ni, nm, nn),
                    DRAIN,
                    EPILOGUE_GROUPS,
                    PHASE,
                )
                is not None
            ):
                pc.lds_ptrs.wait_buffer_load_groups(
                    _wait(
                        tc,
                        STAGE,
                        _slot_index(mi, ni, nm, nn),
                        DRAIN,
                        EPILOGUE_GROUPS,
                        PHASE,
                    )
                )
            if require_constexpr(not DOT and mi == 0 and ni == 0):
                gl.amd.cdna4.sched_barrier(0)
            with region("mem"):
                am, bm, asm, bsm = _read_slot(
                    pc,
                    ptrs,
                    buffers,
                    step,
                    mi,
                    ni,
                    KI,
                    IN_LOOP,
                    False,
                    STATIC_PHASE,
                    A_SCALE_LOAD,
                    B_SCALE_LOAD,
                )
                buffers = _fill_slot(
                    pc,
                    ptrs,
                    buffers,
                    step,
                    STAGE,
                    DRAIN,
                    mi,
                    ni,
                    KI,
                    IN_LOOP,
                    MARK_STAGE=False,
                    STATIC_PHASE=STATIC_PHASE,
                    A_SCALE_LOAD=A_SCALE_LOAD,
                    B_SCALE_LOAD=B_SCALE_LOAD,
                )
                if require_constexpr(
                    tc.commit_per_stage_warp_pipeline()
                    and mi == nm - 1
                    and ni == nn - 1
                ):
                    pc.lds_ptrs.commit_buffer_load()
            with region("mfma"):
                if require_constexpr(DOT):
                    dot_a = _take_operand_pairs(
                        regs.a_payload, regs.a_scale, tc, 0, mi, PHASE - 1
                    )
                    dot_b = _take_operand_pairs(
                        regs.b_payload, regs.b_scale, tc, 1, ni, PHASE - 1
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
                    DOT,
                    (PHASE - 1) * tc.BLOCK_K // 128 % 2,
                )
                af, bf, asf, bsf = _read_slot(
                    pc,
                    ptrs,
                    buffers,
                    step,
                    mi,
                    ni,
                    KI,
                    IN_LOOP,
                    True,
                    STATIC_PHASE,
                    A_SCALE_LOAD,
                    B_SCALE_LOAD,
                )
                if require_constexpr(mi == nm - 1 and ni == nn - 1):
                    ptrs = _advance(
                        pc,
                        ptrs,
                        step,
                        STAGE,
                        DRAIN,
                        KI,
                        IN_LOOP,
                        STATIC_PHASE,
                        A_SCALE_LOAD,
                        B_SCALE_LOAD,
                    )
            if require_constexpr(
                tc.commit_per_stage_whole() and mi == nm - 1 and ni == nn - 1
            ):
                with region("commit"):
                    pc.lds_ptrs.commit_buffer_load()
            a += af if _read_payload_after_fill(tc, 0) else am
            b += bf if _read_payload_after_fill(tc, 1) else bm
            a_scale += asf if tc.ds_read_in_mfma(0, True) else asm
            b_scale += bsf if tc.ds_read_in_mfma(1, True) else bsm
            acc += (value,)
    if require_constexpr(IN_LOOP and not tc.payload_via_lds(1)):
        gl.amd.cdna4.sched_barrier(0)
    if require_constexpr(not IN_LOOP):
        buffers = _rotate_buffers(buffers, tc, PHASE)
    if require_constexpr(not pc.func_cfg.a_has_scale()):
        a_scale = a
    if require_constexpr(not pc.func_cfg.b_has_scale()):
        b_scale = b
    return ptrs, buffers, _PipelineRegFragments(a, a_scale, b, b_scale, acc)


_step = _step_live


@gluon.jit
def _run_buffered_pipeline(pc, ptrs, NUM_K):
    """Run the prologue and MAIN iterations; leave the drain to overlap the epilogue."""
    tc: gl.constexpr = pc.tuning_cfg
    gl.static_assert(
        not tc.warp_pipeline_manual(),
        "the live pipeline requires WARP_PIPELINE=0 or 1",
    )
    gl.static_assert(
        tc.num_prefetch_k_slots() == tc.num_k_slots_per_tile(),
        "the pipeline requires VGPR_PREFETCH_K == BLOCK_K",
    )
    depth: gl.constexpr = pipeline_depth(tc)
    unroll: gl.constexpr = pipeline_unroll(tc)
    peeled: gl.constexpr = _pipeline_peeled(tc)
    main = NUM_K - depth
    gl.assume(main >= peeled + unroll)
    remaining = main - peeled
    unroll_end = remaining // unroll * unroll
    buffers = _init_buffers(pc)
    # r = p - (NB_MAX - 1): active streams fill p - NB_DELTA.
    for r in gl.static_range(1 - depth, 0):
        for ni in gl.static_range(tc.num_n_slots_per_block()):
            for mi in gl.static_range(tc.num_m_slots_per_block()):
                buffers = _fill_slot(
                    pc,
                    ptrs,
                    buffers,
                    r,
                    r,
                    False,
                    mi,
                    ni,
                    A_SCALE_LOAD=r % tc.scale_ratio_k_step(0) == 0,
                    B_SCALE_LOAD=r % tc.scale_ratio_k_step(1) == 0,
                )
        ptrs = _advance(
            pc,
            ptrs,
            r,
            r,
            False,
            A_SCALE_LOAD=r % tc.scale_ratio_k_step(0) == 0,
            B_SCALE_LOAD=r % tc.scale_ratio_k_step(1) == 0,
        )
        buffers = _rotate_buffers(buffers, tc, r)
    acc = ()
    for ni in gl.static_range(tc.num_n_slots_per_block()):
        for mi in gl.static_range(tc.num_m_slots_per_block()):
            acc += (
                gl.zeros(
                    accumulator_fragment_shape_slot(tc),
                    pc.func_cfg.mma_acc_dtype,
                    tc.dot_result_fragment_layout(),
                ),
            )
    regs = _PipelineRegFragments((), (), (), (), acc)
    ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, 0, 0, DOT=False)
    for x in gl.static_range(peeled):
        ptrs, buffers, regs = _step(
            pc,
            ptrs,
            buffers,
            regs,
            x + 1,
            x + 1,
            A_SCALE_LOAD=(x + 1) % tc.scale_ratio_k_step(0) == 0,
            B_SCALE_LOAD=(x + 1) % tc.scale_ratio_k_step(1) == 0,
            PHASE=x + 1,
        )
    for base in tl.range(0, unroll_end, unroll):
        for u in gl.static_range(unroll):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                peeled + base + u + 1,
                None,
                KI=u,
                IN_LOOP=True,
                A_SCALE_LOAD=(peeled + u + 1) % tc.scale_ratio_k_step(0) == 0,
                B_SCALE_LOAD=(peeled + u + 1) % tc.scale_ratio_k_step(1) == 0,
                PHASE=peeled + u + 1,
            )
    if require_constexpr(tc.UNROLL_EPILOGUE):
        # Static expansion keeps short remainders and their LDS indices constant.
        for u in gl.static_range(unroll - 1):
            if unroll_end + u < remaining:
                ptrs, buffers, regs = _step(
                    pc,
                    ptrs,
                    buffers,
                    regs,
                    peeled + unroll_end + u + 1,
                    None,
                    KI=u,
                    STATIC_PHASE=True,
                    A_SCALE_LOAD=(peeled + u + 1) % tc.scale_ratio_k_step(0) == 0,
                    B_SCALE_LOAD=(peeled + u + 1) % tc.scale_ratio_k_step(1) == 0,
                    PHASE=peeled + u + 1,
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
                None,
            )
    return ptrs, buffers, regs


@gluon.jit
def _drain_buffered_pipeline(
    pc, ptrs, buffers, regs, NUM_K, EPILOGUE_GROUPS: gl.constexpr
):
    tc: gl.constexpr = pc.tuning_cfg
    main = NUM_K - pipeline_depth(tc)
    period: gl.constexpr = math.lcm(
        tc.num_buffers(0),
        tc.num_buffers(1),
        (
            tc.num_buffers(0, True) * tc.scale_ratio_k_step(0)
            if pc.func_cfg.a_has_scale()
            else 1
        ),
        (
            tc.num_buffers(1, True) * tc.scale_ratio_k_step(1)
            if pc.func_cfg.b_has_scale()
            else 1
        ),
    )
    # Keep short LDS rings in immediate offsets; live queues and quantized
    # epilogues need the smaller dynamic drain to avoid allocation pressure.
    if require_constexpr(
        period <= 3
        and pipeline_unroll(tc) % period == 0
        and tc.pipeline_register_period() == 1
        and pc.func_cfg.output_quant is None
    ):
        phase = main % period
        for p in gl.static_range(period):
            if phase == p:
                for j in gl.static_range(pipeline_depth(tc) - 1):
                    ptrs, buffers, regs = _step(
                        pc,
                        ptrs,
                        buffers,
                        regs,
                        main + j + 1,
                        j,
                        DRAIN=True,
                        EPILOGUE_GROUPS=EPILOGUE_GROUPS,
                        KI=p + j - _pipeline_peeled(tc),
                        STATIC_PHASE=True,
                        A_SCALE_LOAD=(pc.num_k - pipeline_depth(tc) + j + 1)
                        % tc.scale_ratio_k_step(0)
                        == 0,
                        B_SCALE_LOAD=(pc.num_k - pipeline_depth(tc) + j + 1)
                        % tc.scale_ratio_k_step(1)
                        == 0,
                        PHASE=pc.num_k - pipeline_depth(tc) + j + 1,
                    )
    else:
        for j in gl.static_range(pipeline_depth(tc) - 1):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                main + j + 1,
                j,
                DRAIN=True,
                EPILOGUE_GROUPS=EPILOGUE_GROUPS,
                A_SCALE_LOAD=(pc.num_k - pipeline_depth(tc) + j + 1)
                % tc.scale_ratio_k_step(0)
                == 0,
                B_SCALE_LOAD=(pc.num_k - pipeline_depth(tc) + j + 1)
                % tc.scale_ratio_k_step(1)
                == 0,
                PHASE=pc.num_k - pipeline_depth(tc) + j + 1,
            )
    a_scale, b_scale = (), ()
    for tile in gl.static_range(tc.num_m_slots_per_block()):
        pairs = _take_operand_pairs(
            regs.a_payload, regs.a_scale, tc, 0, tile, pc.num_k - 1
        )
        for k in gl.static_range(tc.num_k_slots_per_tile()):
            a_scale += (pairs[2 * k + 1],)
    for tile in gl.static_range(tc.num_n_slots_per_block()):
        pairs = _take_operand_pairs(
            regs.b_payload, regs.b_scale, tc, 1, tile, pc.num_k - 1
        )
        for k in gl.static_range(tc.num_k_slots_per_tile()):
            b_scale += (pairs[2 * k + 1],)
    return _PipelineRegFragments(
        regs.a_payload, a_scale, regs.b_payload, b_scale, regs.acc
    )


@gluon.jit
def _last_mfma(pc, regs):
    tc: gl.constexpr = pc.tuning_cfg
    acc = ()
    for ni in gl.static_range(tc.num_n_slots_per_block()):
        for mi in gl.static_range(tc.num_m_slots_per_block()):
            acc += (
                _maybe_block_dot(
                    _take_reg_pairs(
                        regs.a_payload,
                        regs.a_scale,
                        mi * tc.num_k_slots_per_tile(),
                        tc.num_k_slots_per_tile(),
                    ),
                    _take_reg_pairs(
                        regs.b_payload,
                        regs.b_scale,
                        ni * tc.num_k_slots_per_tile(),
                        tc.num_k_slots_per_tile(),
                    ),
                    regs.acc[
                        _slot_index(
                            mi,
                            ni,
                            tc.num_m_slots_per_block(),
                            tc.num_n_slots_per_block(),
                        )
                    ],
                    tc.num_k_slots_per_tile(),
                    pc.func_cfg,
                    tc,
                    True,
                    (pc.num_k - 1) * tc.BLOCK_K // 128 % 2,
                ),
            )
    return acc


# JIT resolves these helpers after ``moe_gemm`` has defined them.
from .moe_gemm import (
    _maybe_block_dot,
    _PipelinePointers,
    _PipelineRegFragments,
    _sched_hint,
    _take_reg_pairs,
)
