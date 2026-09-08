# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Independent payload/scale queues for the Gluon MoE K pipeline.

At read stage r, stream i fills stage r + depth[i] - 1 in its usual memory
slot. Register streams follow that same schedule, carrying fragments in SSA
queues instead of copying them into LDS. A depth of one fills before reading.
The legacy pipeline remains the default, including its recorded schedule.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import pick_warp_pipeline_stage as pick_stage
from ._lang import require_constexpr
from ._offsets import _slot_index
from ._schedule import (
    _buffer_load_pos,
    _buffer_load_tile,
    _ds_read_a_tile,
    _ds_read_b_tile,
    _scale_buffer_load_tile,
)


@gluon.constexpr_function
def _ops(tc, mi, ni):
    """Fill ownership, including scales whose destination is a register queue."""
    nm, nn = tc.num_mini_m(), tc.num_mini_n()
    pos = _buffer_load_pos(mi, ni, nm, nn)
    ops = []
    for idx in range(2):
        ops.append(_buffer_load_tile(pos, nm, nn, idx == 0))
        tile = None
        if tc.func_cfg.has_scale(idx):
            tile = _scale_buffer_load_tile(mi, ni, nm, nn, idx == 0, tc.SCALE_FILL_MID)
            if (
                idx == 0
                and tile is not None
                and tc.scale_via_lds(0)
                and tc.scale_shuffled(0)
                and tile % tc.scale_tile_ratio_a() != 0
            ):
                tile = None
        ops.append(tile)
    return tuple(ops)


@gluon.constexpr_function
def _active(tc, kind, stage, num_k):
    if kind % 2 and not tc.func_cfg.has_scale(kind // 2):
        return False
    # None denotes the steady state, where all four fill stages are inside K.
    return (
        stage is None
        or 0 <= stage + tc.num_buffers(kind // 2, bool(kind % 2)) - 1 < num_k
    )


@gluon.constexpr_function
def _async(tc, kind):
    return tc.scale_via_lds(kind // 2) if kind % 2 else tc.payload_via_lds(kind // 2)


@gluon.constexpr_function
def _groups(tc, stage, num_k):
    """Async groups per fill slot; ordinary buffer loads never own a group."""
    schedule, whole = [], ()
    for ni in range(tc.num_mini_n()):
        for mi in range(tc.num_mini_m()):
            ops = tuple(
                (kind, tile)
                for kind, tile in enumerate(_ops(tc, mi, ni))
                if tile is not None
                and _active(tc, kind, stage, num_k)
                and _async(tc, kind)
            )
            if tc.commit_per_op():
                groups = tuple((op,) for op in ops)
            elif tc.commit_per_slot():
                groups = (ops,)
            else:
                whole += ops
                groups = (
                    (whole,)
                    if (mi == tc.num_mini_m() - 1 and ni == tc.num_mini_n() - 1)
                    else ()
                )
            schedule.append(groups)
    return tuple(schedule)


@gluon.constexpr_function
def _wait(tc, stage, num_k, slot):
    """Count groups newer than every copy needed by this read slot.

    Each operand can have a different producer stage. Count the actual warmup
    and drain groups as well as the current stage's already committed prefix.
    For a per-stage scheme, slot=None waits for all tiles at the stage head.
    """
    nm, nn = tc.num_mini_m(), tc.num_mini_n()
    r = tc.pipeline_depth() if stage is None else stage
    required = []
    for ni in range(nn):
        for mi in range(nm):
            if slot is not None and _slot_index(mi, ni, nm, nn) != slot:
                continue
            for idx in range(2):
                tile = (
                    _ds_read_a_tile(mi, ni, nm, nn)
                    if idx == 0
                    else _ds_read_b_tile(mi, ni, nm, nn)
                )
                if tile is None:
                    continue
                if tc.payload_via_lds(idx):
                    required.append((r - tc.num_buffers(idx) + 1, idx * 2, tile))
                if tc.func_cfg.has_scale(idx) and tc.scale_via_lds(idx):
                    owner = tile
                    if idx == 0 and tc.scale_shuffled(0):
                        owner -= owner % tc.scale_tile_ratio_a()
                    required.append(
                        (r - tc.num_buffers(idx, True) + 1, idx * 2 + 1, owner)
                    )
    if not required:
        return None
    timeline = []
    for s in range(r - tc.pipeline_depth() + 1, r + 1):
        schedule = _groups(tc, None if stage is None else s, num_k)
        for pos, groups in enumerate(schedule):
            if s == r and (slot is None or pos >= slot):
                break
            timeline.extend(
                tuple((s, kind, tile) for kind, tile in group) for group in groups
            )
    latest = max(
        i for i, group in enumerate(timeline) if any(op in group for op in required)
    )
    return len(timeline) - latest - 1


@gluon.constexpr_function
def _single_buffer(tc):
    return any(
        tc.num_buffers(i, scale) == 1
        for i in range(2)
        for scale in (False, True)
        if not scale or tc.func_cfg.has_scale(i)
    )


@gluon.jit
def _index(
    tc,
    step,
    kind: gl.constexpr,
    FILL: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
):
    depth: gl.constexpr = tc.num_buffers(kind // 2, kind % 2 != 0)
    advance: gl.constexpr = depth - 1 if FILL else 0
    if require_constexpr(IN_LOOP and tc.pipeline_unroll() % depth == 0):
        out = (KI + 1 + advance) % depth
    else:
        out = (step + advance) % depth
    return out


@gluon.jit
def _phase(tc, step, KI: gl.constexpr, IN_LOOP: gl.constexpr):
    if require_constexpr(IN_LOOP and tc.pipeline_unroll() % 2 == 0):
        phase = (KI + 1) % 2
    else:
        phase = step % 2
    return phase


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
def _rotate_buffers(buffers, tc):
    mk: gl.constexpr = tc.num_mini_k()
    return (
        _rotate(buffers[0], tc.num_mini_n() * mk),
        _rotate(buffers[1], tc.num_mini_m() * mk),
        _rotate(buffers[2], tc.num_mini_n() * mk),
    )


@gluon.jit
def _init_buffers(pc):
    tc: gl.constexpr = pc.tuning_cfg
    b, a_scale, b_scale = (), (), ()
    if require_constexpr(tc.B_IN_REG):
        for _ in gl.static_range(tc.num_buffers(1) * tc.num_mini_n() * tc.num_mini_k()):
            b += (
                gl.zeros(
                    [tc.MINI_BLOCK_K // pc.func_cfg.pack_divisor(1), tc.MINI_BLOCK_N],
                    pc.func_cfg.operand_elem_ty(1),
                    tc.dot_operand_fragment_layout(1),
                ),
            )
    if require_constexpr(pc.func_cfg.a_has_scale() and not tc.scale_via_lds(0)):
        for _ in gl.static_range(
            tc.num_buffers(0, True) * tc.num_mini_m() * tc.num_mini_k()
        ):
            if require_constexpr(tc.scale_packed_k128(0)):
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
                        [tc.MINI_BLOCK_M, tc.MINI_BLOCK_K // 32],
                        gl.uint8,
                        tc.dot_operand_scale_fragment_layout(0),
                    ),
                )
    if require_constexpr(pc.func_cfg.b_has_scale() and not tc.scale_via_lds(1)):
        for _ in gl.static_range(
            tc.num_buffers(1, True) * tc.num_mini_n() * tc.num_mini_k()
        ):
            if require_constexpr(tc.scale_packed_k128(1)):
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
                        [tc.MINI_BLOCK_N, tc.MINI_BLOCK_K // 32],
                        gl.uint8,
                        tc.dot_operand_scale_fragment_layout(1),
                    ),
                )
    return b, a_scale, b_scale


@gluon.jit
def _fill_slot(
    pc,
    ptrs,
    buffers,
    step,
    STAGE: gl.constexpr,
    NUM_K: gl.constexpr,
    mi: gl.constexpr,
    ni: gl.constexpr,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    MARK_STAGE: gl.constexpr = True,
):
    tc: gl.constexpr = pc.tuning_cfg
    ops: gl.constexpr = _ops(tc, mi, ni)
    mk: gl.constexpr = tc.num_mini_k()
    b, a_scale, b_scale = buffers
    phase = _phase(tc, step, KI, IN_LOOP)
    if require_constexpr(ops[0] is not None and _active(tc, 0, STAGE, NUM_K)):
        pc.lds_ptrs.buffer_load_a_payload(
            _index(tc, step, 0, True, KI, IN_LOOP),
            ops[0],
            ptrs.a_hbm_ptr,
            pc.a_hbm_offs[ops[0]],
        )
        if require_constexpr(tc.commit_per_op() and tc.payload_via_lds(0)):
            pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ops[1] is not None and _active(tc, 1, STAGE, NUM_K)):
        if require_constexpr(tc.scale_via_lds(0)):
            pc.lds_ptrs.buffer_load_a_scale(
                _index(tc, step, 1, True, KI, IN_LOOP),
                ops[1],
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
            )
            if require_constexpr(tc.commit_per_op()):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_scale_register(
                0,
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
                K_PHASE=(phase + tc.num_buffers(0, True) - 1) % 2,
            )
            a_scale = _replace_tile(
                a_scale,
                fragments,
                ((tc.num_buffers(0, True) - 1) * tc.num_mini_m() + ops[1]) * mk,
            )
    if require_constexpr(ops[2] is not None and _active(tc, 2, STAGE, NUM_K)):
        if require_constexpr(tc.B_IN_REG):
            fragments = pc.lds_ptrs.buffer_load_b_register(
                ptrs.b_hbm_ptr, pc.b_hbm_offs[ops[2]]
            )
            b = _replace_tile(
                b, fragments, ((tc.num_buffers(1) - 1) * tc.num_mini_n() + ops[2]) * mk
            )
        else:
            pc.lds_ptrs.buffer_load_b_payload(
                _index(tc, step, 2, True, KI, IN_LOOP),
                ops[2],
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
            )
            if require_constexpr(tc.commit_per_op() and tc.payload_via_lds(1)):
                pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ops[3] is not None and _active(tc, 3, STAGE, NUM_K)):
        if require_constexpr(tc.scale_via_lds(1)):
            pc.lds_ptrs.buffer_load_b_scale(
                _index(tc, step, 3, True, KI, IN_LOOP),
                ops[3],
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
            )
            if require_constexpr(tc.commit_per_op()):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_scale_register(
                1,
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
                K_PHASE=(phase + tc.num_buffers(1, True) - 1) % 2,
            )
            b_scale = _replace_tile(
                b_scale,
                fragments,
                ((tc.num_buffers(1, True) - 1) * tc.num_mini_n() + ops[3]) * mk,
            )
    if require_constexpr(
        tc.commit_per_slot()
        or (
            MARK_STAGE
            and tc.commit_per_stage()
            and mi == tc.num_mini_m() - 1
            and ni == tc.num_mini_n() - 1
        )
    ):
        pc.lds_ptrs.commit_buffer_load()
    return b, a_scale, b_scale


@gluon.jit
def _advance(
    pc,
    ptrs,
    step,
    STAGE: gl.constexpr,
    NUM_K: gl.constexpr,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    a, b, a_scale, b_scale = (
        ptrs.a_hbm_ptr,
        ptrs.b_hbm_ptr,
        ptrs.a_scale_hbm_ptr,
        ptrs.b_scale_hbm_ptr,
    )
    phase = _phase(tc, step, KI, IN_LOOP)
    if require_constexpr(_active(tc, 0, STAGE, NUM_K)):
        a += pc.a_step
    if require_constexpr(_active(tc, 2, STAGE, NUM_K)):
        b += pc.b_step
    if require_constexpr(_active(tc, 1, STAGE, NUM_K)):
        if require_constexpr(tc.scale_packed_k128(0)):
            a_scale += (
                ((phase + tc.num_buffers(0, True) - 1) % 2)
                * 2
                * pc.s_step
                * pc.a_scale_stride_k
            )
        else:
            a_scale += pc.s_step * pc.a_scale_stride_k
    if require_constexpr(_active(tc, 3, STAGE, NUM_K)):
        if require_constexpr(tc.scale_packed_k128(1)):
            b_scale += (
                ((phase + tc.num_buffers(1, True) - 1) % 2)
                * 2
                * pc.s_step
                * pc.b_scale_stride_k
            )
        else:
            b_scale += pc.s_step * pc.b_scale_stride_k
    return _PipelinePointers(a, b, a_scale, b_scale)


@gluon.jit
def _read_tile(
    pc,
    ptrs,
    buffers,
    step,
    tile: gl.constexpr,
    operand: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
    PAYLOAD: gl.constexpr,
    SCALE: gl.constexpr,
):
    tc: gl.constexpr = pc.tuning_cfg
    has_scale: gl.constexpr = pc.func_cfg.has_scale(operand)
    reg_payload: gl.constexpr = operand == 1 and tc.B_IN_REG
    reg_scale: gl.constexpr = has_scale and not tc.scale_via_lds(operand)
    phase = _phase(tc, step, KI, IN_LOOP)
    frags = ()
    for k in gl.static_range(tc.num_mini_k()):
        if require_constexpr(operand == 0):
            payload, scale = pc.lds_ptrs.ds_read_a_frag(
                _index(tc, step, 0, False, KI, IN_LOOP),
                tile,
                k,
                ptrs.a_scale_hbm_ptr,
                None,
                RELAXED=False,
                READ_PAYLOAD=PAYLOAD,
                READ_SCALE=SCALE and not reg_scale,
                SCALE_READ_IDX=_index(tc, step, 1, False, KI, IN_LOOP),
            )
        else:
            payload, scale = pc.lds_ptrs.ds_read_b_frag(
                _index(tc, step, 2, False, KI, IN_LOOP),
                tile,
                k,
                ptrs.b_scale_hbm_ptr,
                None,
                RELAXED=False,
                READ_PAYLOAD=PAYLOAD and not reg_payload,
                READ_SCALE=SCALE and not reg_scale,
                SCALE_READ_IDX=_index(tc, step, 3, False, KI, IN_LOOP),
            )
        if require_constexpr(PAYLOAD and reg_payload):
            payload = buffers[0][tile * tc.num_mini_k() + k]
        if require_constexpr(SCALE and reg_scale):
            scale = buffers[operand + 1][tile * tc.num_mini_k() + k]
        if require_constexpr(SCALE and has_scale and tc.scale_packed_k128(operand)):
            # Canonicalize the selected K128 half, so an odd GCD needs neither
            # a doubled unroll nor a dynamic MFMA op_sel.
            scale = (scale.to(gl.uint32) >> (16 * phase)).to(gl.int32)
        if require_constexpr(not PAYLOAD):
            payload = scale
        if require_constexpr(not SCALE or not has_scale):
            scale = payload
        frags += (payload, scale)
    return frags


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
):
    tc: gl.constexpr = pc.tuning_cfg
    at: gl.constexpr = _ds_read_a_tile(mi, ni, tc.num_mini_m(), tc.num_mini_n())
    bt: gl.constexpr = _ds_read_b_tile(mi, ni, tc.num_mini_m(), tc.num_mini_n())
    a, b = (), ()
    if require_constexpr(
        at is not None
        and (
            tc.ds_read_in_mfma(0) == MFMA
            or (pc.func_cfg.a_has_scale() and tc.ds_read_in_mfma(0, True) == MFMA)
        )
    ):
        a = _read_tile(
            pc,
            ptrs,
            buffers,
            step,
            at,
            0,
            KI,
            IN_LOOP,
            tc.ds_read_in_mfma(0) == MFMA,
            tc.ds_read_in_mfma(0, True) == MFMA,
        )
    if require_constexpr(
        bt is not None
        and (
            tc.ds_read_in_mfma(1) == MFMA
            or (pc.func_cfg.b_has_scale() and tc.ds_read_in_mfma(1, True) == MFMA)
        )
    ):
        b = _read_tile(
            pc,
            ptrs,
            buffers,
            step,
            bt,
            1,
            KI,
            IN_LOOP,
            tc.ds_read_in_mfma(1) == MFMA,
            tc.ds_read_in_mfma(1, True) == MFMA,
        )
    return a, b


@gluon.jit
def _step(
    pc,
    ptrs,
    buffers,
    regs,
    step,
    STAGE: gl.constexpr,
    NUM_K: gl.constexpr,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    DOT: gl.constexpr = True,
):
    tc: gl.constexpr = pc.tuning_cfg
    single: gl.constexpr = _single_buffer(tc)
    warp: gl.constexpr = tc.warp_pipeline_compiler() and IN_LOOP and not single
    region: gl.constexpr = pick_stage(warp)
    nm: gl.constexpr = tc.num_mini_m()
    nn: gl.constexpr = tc.num_mini_n()
    mk: gl.constexpr = tc.num_mini_k()
    a, b, acc = (), (), ()
    if require_constexpr(single):
        # A one-buffer stream reuses the tile just read in the previous step.
        # Close that WAR window before filling, then make the new copies visible.
        pc.lds_ptrs.wait_buffer_load_groups(0)
        gl.barrier()
        for ni in gl.static_range(nn):
            for mi in gl.static_range(nm):
                buffers = _fill_slot(
                    pc, ptrs, buffers, step, STAGE, NUM_K, mi, ni, KI, IN_LOOP
                )
        pc.lds_ptrs.wait_buffer_load_groups(0)
        gl.barrier()
    elif require_constexpr(tc.commit_per_stage()):
        pc.lds_ptrs.wait_buffer_load_groups(_wait(tc, STAGE, NUM_K, None))
    if require_constexpr(tc.SCHED_MODE != 0 and not warp):
        _sched_hint(tc.SCHED_MODE)
    for ni in gl.static_range(nn):
        for mi in gl.static_range(nm):
            if require_constexpr(not single and not tc.commit_per_stage()):
                if require_constexpr(
                    _wait(tc, STAGE, NUM_K, _slot_index(mi, ni, nm, nn)) is not None
                ):
                    pc.lds_ptrs.wait_buffer_load_groups(
                        _wait(tc, STAGE, NUM_K, _slot_index(mi, ni, nm, nn))
                    )
                else:
                    gl.barrier()
            with region("mem"):
                am, bm = _read_slot(pc, ptrs, buffers, step, mi, ni, KI, IN_LOOP, False)
                if require_constexpr(not single):
                    buffers = _fill_slot(
                        pc,
                        ptrs,
                        buffers,
                        step,
                        STAGE,
                        NUM_K,
                        mi,
                        ni,
                        KI,
                        IN_LOOP,
                        MARK_STAGE=False,
                    )
                    if require_constexpr(
                        tc.commit_per_stage_warp_pipeline()
                        and mi == nm - 1
                        and ni == nn - 1
                    ):
                        pc.lds_ptrs.commit_buffer_load()
            with region("mfma"):
                if require_constexpr(DOT):
                    dot_a = _take_reg_pairs(regs.a_payload, regs.a_scale, mi * mk, mk)
                    dot_b = _take_reg_pairs(regs.b_payload, regs.b_scale, ni * mk, mk)
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
                    0,
                )
                af, bf = _read_slot(pc, ptrs, buffers, step, mi, ni, KI, IN_LOOP, True)
                if require_constexpr(mi == nm - 1 and ni == nn - 1):
                    # Stage contexts emit a border only on exit. Keep the pointer
                    # updates before it so the next unrolled wait starts a region;
                    # arithmetic between that border and the wait would include the
                    # wait inside a stage, which the compiler pipeliner rejects.
                    ptrs = _advance(pc, ptrs, step, STAGE, NUM_K, KI, IN_LOOP)
            if require_constexpr(
                not single
                and tc.commit_per_stage_whole()
                and mi == nm - 1
                and ni == nn - 1
            ):
                with region("commit"):
                    pc.lds_ptrs.commit_buffer_load()
            a += _merge_ds_read_frags(am, af, tc, 0)
            b += _merge_ds_read_frags(bm, bf, tc, 1)
            acc += (value,)
    return ptrs, _rotate_buffers(buffers, tc), _make_reg_fragments(a, b, acc)


@gluon.jit
def _run_buffered_pipeline(pc, ptrs, NUM_K: gl.constexpr):
    """Return the accumulators and final prefetched stage for the common epilogue."""
    tc: gl.constexpr = pc.tuning_cfg
    gl.static_assert(
        not tc.warp_pipeline_manual(),
        "independent buffers require WARP_PIPELINE=0 or 1",
    )
    gl.static_assert(
        tc.num_prefetch_mini() == tc.num_mini_k(),
        "independent buffers require VGPR_PREFETCH_K == BLOCK_K",
    )
    depth: gl.constexpr = tc.pipeline_depth()
    unroll: gl.constexpr = tc.pipeline_unroll()
    # All fills are in bounds here; only the finite warmup/drain need predicates.
    main: gl.constexpr = max(0, NUM_K - depth + 1)
    warm: gl.constexpr = min(NUM_K, max(1, depth - 1))
    unrolled: gl.constexpr = max(0, (main - warm) // unroll) * unroll
    buffers = _init_buffers(pc)
    for s in gl.static_range(1 - depth, 0):
        for ni in gl.static_range(tc.num_mini_n()):
            for mi in gl.static_range(tc.num_mini_m()):
                buffers = _fill_slot(pc, ptrs, buffers, s, s, NUM_K, mi, ni)
        ptrs = _advance(pc, ptrs, s, s, NUM_K)
        buffers = _rotate_buffers(buffers, tc)
    acc = ()
    for ni in gl.static_range(tc.num_mini_n()):
        for mi in gl.static_range(tc.num_mini_m()):
            acc += (
                gl.zeros(
                    [tc.MINI_BLOCK_M, tc.MINI_BLOCK_N],
                    pc.func_cfg.mma_acc_dtype,
                    tc.dot_result_fragment_layout(),
                ),
            )
    regs = _PipelineRegFragments((), (), (), (), acc)
    ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, 0, 0, NUM_K, DOT=False)
    for s in gl.static_range(1, warm):
        ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, s, s, NUM_K)
    for base in tl.range(0, unrolled, unroll):
        for i in gl.static_range(unroll):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                base + i + warm,
                None,
                NUM_K,
                KI=i + warm - 1,
                IN_LOOP=True,
            )
    for s in gl.static_range(warm + unrolled, NUM_K):
        ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, s, s, NUM_K)
    return regs


# JIT resolves shared helpers at compile time. Import after defining the driver
# so either this module or moe_gemm can be imported first.
from .moe_gemm import (
    _make_reg_fragments,
    _maybe_block_dot,
    _merge_ds_read_frags,
    _PipelinePointers,
    _PipelineRegFragments,
    _sched_hint,
    _take_reg_pairs,
)
