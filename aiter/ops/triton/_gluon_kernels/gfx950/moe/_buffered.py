# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One payload/scale pipeline for shared and independent buffer depths.

A step named r reads tile r, fills r + NB - 1, and accumulates tile r - 1.
The driver peels the seed and first MFMAs, then uses runtime main-loop bounds.
Register rings keep fixed SSA slots throughout each complete unrolled body;
only the finite prologue, remainder and drain rotate the logical queues.
"""

import math

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._buffered_schedule import _active, _ops, _wait
from ._lang import pick_warp_pipeline_stage as pick_stage
from ._lang import require_constexpr
from ._offsets import _slot_index
from ._schedule import _ds_read_a_tile, _ds_read_b_tile


@gluon.jit
def _index(
    tc,
    step,
    kind: gl.constexpr,
    FILL: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
):
    if require_constexpr(kind % 2 and not tc.func_cfg.has_scale(kind // 2)):
        out = 0
    else:
        depth: gl.constexpr = tc.num_buffers(kind // 2, kind % 2 != 0)
        advance: gl.constexpr = depth - 1 if FILL else 0
        if require_constexpr(IN_LOOP and tc.pipeline_unroll() % depth == 0):
            out = (tc.pipeline_peeled() + KI + 1 + advance) % depth
        else:
            tile = step + advance
            if require_constexpr(tc.pipeline_register_period() == 1):
                # Avoid signed ring arithmetic without disturbing live register queues.
                gl.assume(tile >= 0)
            out = tile % depth
    return out


@gluon.jit
def _phase(tc, step, KI: gl.constexpr, IN_LOOP: gl.constexpr):
    if require_constexpr(IN_LOOP and tc.pipeline_unroll() % 2 == 0):
        phase = (tc.pipeline_peeled() + KI + 1) % 2
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


@gluon.constexpr_function
def _soff_unroll(tc, in_loop):
    # Packed K128 pairs need a constant starting half for an encoded soffset.
    # With odd unrolls, advance their pointers per iteration instead.
    return (
        in_loop
        and tc.SOFF_UNROLL
        and (
            tc.pipeline_unroll() % 2 == 0
            or not (tc.scale_packed_k128(0) or tc.scale_packed_k128(1))
        )
    )


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
):
    tc: gl.constexpr = pc.tuning_cfg
    ops: gl.constexpr = _ops(tc, mi, ni)
    mk: gl.constexpr = tc.num_mini_k()
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
        tc.scale_hbm_steps(
            0, offset_step, (tc.pipeline_peeled() + tc.num_buffers(0, True)) % 2
        )
        * pc.s_step
        * pc.a_scale_stride_k
        if offset_step and pc.func_cfg.a_has_scale()
        else 0
    )
    bs_soff: gl.constexpr = (
        tc.scale_hbm_steps(
            1, offset_step, (tc.pipeline_peeled() + tc.num_buffers(1, True)) % 2
        )
        * pc.s_step
        * pc.b_scale_stride_k
        if offset_step and pc.func_cfg.b_has_scale()
        else 0
    )
    b, a_scale, b_scale = buffers
    phase = _phase(tc, step, KI, IN_LOOP or STATIC_PHASE)
    if require_constexpr(ops[0] is not None and _active(tc, 0, STAGE, DRAIN)):
        pc.lds_ptrs.buffer_load_a_payload(
            _index(tc, step, 0, True, KI, IN_LOOP or STATIC_PHASE),
            ops[0],
            ptrs.a_hbm_ptr,
            pc.a_hbm_offs[ops[0]],
            a_soff,
        )
        if require_constexpr(tc.commit_per_op() and tc.payload_via_lds(0)):
            pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ops[1] is not None and _active(tc, 1, STAGE, DRAIN)):
        if require_constexpr(tc.scale_via_lds(0)):
            pc.lds_ptrs.buffer_load_a_scale(
                _index(tc, step, 1, True, KI, IN_LOOP or STATIC_PHASE),
                ops[1],
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
                as_soff,
            )
            if require_constexpr(tc.commit_per_op()):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_scale_register(
                0,
                ptrs.a_scale_hbm_ptr,
                pc.a_scale_hbm_offs[ops[1]],
                as_soff,
                K_PHASE=(phase + tc.num_buffers(0, True) - 1) % 2,
            )
            a_scale = _replace_tile(
                a_scale,
                fragments,
                (
                    ((KI if IN_LOOP else 0) + tc.num_buffers(0, True) - 1)
                    % tc.num_buffers(0, True)
                    * tc.num_mini_m()
                    + ops[1]
                )
                * mk,
            )
    if require_constexpr(ops[2] is not None and _active(tc, 2, STAGE, DRAIN)):
        if require_constexpr(tc.B_IN_REG):
            fragments = pc.lds_ptrs.buffer_load_b_register(
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
                b_soff,
            )
            b = _replace_tile(
                b,
                fragments,
                (
                    ((KI if IN_LOOP else 0) + tc.num_buffers(1) - 1)
                    % tc.num_buffers(1)
                    * tc.num_mini_n()
                    + ops[2]
                )
                * mk,
            )
        else:
            pc.lds_ptrs.buffer_load_b_payload(
                _index(tc, step, 2, True, KI, IN_LOOP or STATIC_PHASE),
                ops[2],
                ptrs.b_hbm_ptr,
                pc.b_hbm_offs[ops[2]],
                b_soff,
            )
            if require_constexpr(tc.commit_per_op() and tc.payload_via_lds(1)):
                pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ops[3] is not None and _active(tc, 3, STAGE, DRAIN)):
        if require_constexpr(tc.scale_via_lds(1)):
            pc.lds_ptrs.buffer_load_b_scale(
                _index(tc, step, 3, True, KI, IN_LOOP or STATIC_PHASE),
                ops[3],
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
                bs_soff,
            )
            if require_constexpr(tc.commit_per_op()):
                pc.lds_ptrs.commit_buffer_load()
        else:
            fragments = pc.lds_ptrs.buffer_load_scale_register(
                1,
                ptrs.b_scale_hbm_ptr,
                pc.b_scale_hbm_offs[ops[3]],
                bs_soff,
                K_PHASE=(phase + tc.num_buffers(1, True) - 1) % 2,
            )
            b_scale = _replace_tile(
                b_scale,
                fragments,
                (
                    ((KI if IN_LOOP else 0) + tc.num_buffers(1, True) - 1)
                    % tc.num_buffers(1, True)
                    * tc.num_mini_n()
                    + ops[3]
                )
                * mk,
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
    DRAIN: gl.constexpr,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    STATIC_PHASE: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    a, b, a_scale, b_scale = (
        ptrs.a_hbm_ptr,
        ptrs.b_hbm_ptr,
        ptrs.a_scale_hbm_ptr,
        ptrs.b_scale_hbm_ptr,
    )
    steps: gl.constexpr = tc.pipeline_unroll() if _soff_unroll(tc, IN_LOOP) else 1
    phase = _phase(tc, step, KI, IN_LOOP or STATIC_PHASE)
    if require_constexpr(_soff_unroll(tc, IN_LOOP)):
        phase = (phase - KI) % 2
    if require_constexpr(
        not _soff_unroll(tc, IN_LOOP) or KI + 1 == tc.pipeline_unroll()
    ):
        if require_constexpr(_active(tc, 0, STAGE, DRAIN)):
            a += steps * pc.a_step
        if require_constexpr(_active(tc, 2, STAGE, DRAIN)):
            b += steps * pc.b_step
        if require_constexpr(_active(tc, 1, STAGE, DRAIN)):
            if require_constexpr(tc.scale_packed_k128(0)):
                a_scale += (
                    ((steps + (phase + tc.num_buffers(0, True) - 1) % 2) // 2 * 2)
                    * pc.s_step
                    * pc.a_scale_stride_k
                )
            else:
                a_scale += steps * pc.s_step * pc.a_scale_stride_k
        if require_constexpr(_active(tc, 3, STAGE, DRAIN)):
            if require_constexpr(tc.scale_packed_k128(1)):
                b_scale += (
                    ((steps + (phase + tc.num_buffers(1, True) - 1) % 2) // 2 * 2)
                    * pc.s_step
                    * pc.b_scale_stride_k
                )
            else:
                b_scale += steps * pc.s_step * pc.b_scale_stride_k
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
    STATIC_PHASE: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    has_scale: gl.constexpr = pc.func_cfg.has_scale(operand)
    reg_payload: gl.constexpr = operand == 1 and tc.B_IN_REG
    reg_scale: gl.constexpr = has_scale and not tc.scale_via_lds(operand)
    phase = _phase(tc, step, KI, IN_LOOP or STATIC_PHASE)
    frags = ()
    for k in gl.static_range(tc.num_mini_k()):
        if require_constexpr(operand == 0):
            payload, scale = pc.lds_ptrs.ds_read_a_frag(
                _index(tc, step, 0, False, KI, IN_LOOP or STATIC_PHASE),
                tile,
                k,
                ptrs.a_scale_hbm_ptr,
                None,
                RELAXED=False,
                READ_PAYLOAD=PAYLOAD,
                READ_SCALE=SCALE and not reg_scale,
                SCALE_READ_IDX=_index(tc, step, 1, False, KI, IN_LOOP or STATIC_PHASE),
            )
        else:
            payload, scale = pc.lds_ptrs.ds_read_b_frag(
                _index(tc, step, 2, False, KI, IN_LOOP or STATIC_PHASE),
                tile,
                k,
                ptrs.b_scale_hbm_ptr,
                None,
                RELAXED=False,
                READ_PAYLOAD=PAYLOAD and not reg_payload,
                READ_SCALE=SCALE and not reg_scale,
                SCALE_READ_IDX=_index(tc, step, 3, False, KI, IN_LOOP or STATIC_PHASE),
            )
        if require_constexpr(PAYLOAD and reg_payload):
            payload = buffers[0][
                ((KI % tc.num_buffers(1) if IN_LOOP else 0) * tc.num_mini_n() + tile)
                * tc.num_mini_k()
                + k
            ]
        if require_constexpr(SCALE and reg_scale):
            scale = buffers[operand + 1][
                (
                    (KI % tc.num_buffers(operand, True) if IN_LOOP else 0)
                    * (tc.num_mini_m() if operand == 0 else tc.num_mini_n())
                    + tile
                )
                * tc.num_mini_k()
                + k
            ]
        if require_constexpr(SCALE and has_scale and tc.scale_packed_k128(operand)):
            # Canonicalize the selected K128 half before the MFMA, including
            # phases reached through a runtime remainder or drain.
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
    STATIC_PHASE: gl.constexpr = False,
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
            STATIC_PHASE,
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
            STATIC_PHASE,
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
    DRAIN: gl.constexpr = False,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    DOT: gl.constexpr = True,
    EPILOGUE_GROUPS: gl.constexpr = 0,
    STATIC_PHASE: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    if require_constexpr(tc.FROZEN_STEP):
        # Preserve the explicit reference/manual scheduling mode within the same
        # driver. Its supported configurations have a single shared LDS depth.
        ptrs, regs = _pipeline_step_frozen(
            pc,
            ptrs,
            regs,
            _index(tc, step, 0, True, KI, IN_LOOP or STATIC_PHASE),
            _index(tc, step, 0, False, KI, IN_LOOP or STATIC_PHASE),
            tc.pipeline_depth() - 2 - (STAGE if DRAIN else 0),
            not DRAIN,
            True,
            IN_LOOP,
            DOT,
            0,
            1,
            EPILOGUE_GROUPS,
        )
    else:
        ptrs, buffers, regs = _step_live(
            pc,
            ptrs,
            buffers,
            regs,
            step,
            STAGE,
            DRAIN,
            KI,
            IN_LOOP,
            DOT,
            EPILOGUE_GROUPS,
            STATIC_PHASE,
        )
    return ptrs, buffers, regs


@gluon.jit
def _step_live(
    pc,
    ptrs,
    buffers,
    regs,
    step,
    STAGE: gl.constexpr,
    DRAIN: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
    DOT: gl.constexpr,
    EPILOGUE_GROUPS: gl.constexpr,
    STATIC_PHASE: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    warp: gl.constexpr = tc.warp_pipeline_compiler() and IN_LOOP
    region: gl.constexpr = pick_stage(warp)
    nm: gl.constexpr = tc.num_mini_m()
    nn: gl.constexpr = tc.num_mini_n()
    mk: gl.constexpr = tc.num_mini_k()
    a, b, acc = (), (), ()
    if require_constexpr(tc.commit_per_stage()):
        if require_constexpr(
            _wait(tc, STAGE, None, DRAIN, EPILOGUE_GROUPS) is not None
        ):
            pc.lds_ptrs.wait_buffer_load_groups(
                _wait(tc, STAGE, None, DRAIN, EPILOGUE_GROUPS)
            )
        else:
            gl.barrier()
    if require_constexpr(tc.SCHED_MODE != 0 and not warp):
        _sched_hint(tc.SCHED_MODE)
    for ni in gl.static_range(nn):
        for mi in gl.static_range(nm):
            if require_constexpr(not tc.commit_per_stage()):
                if require_constexpr(
                    _wait(
                        tc, STAGE, _slot_index(mi, ni, nm, nn), DRAIN, EPILOGUE_GROUPS
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
                        )
                    )
                else:
                    gl.barrier()
            if require_constexpr(not DOT and mi == 0 and ni == 0):
                gl.amd.cdna4.sched_barrier(0)
            with region("mem"):
                am, bm = _read_slot(
                    pc, ptrs, buffers, step, mi, ni, KI, IN_LOOP, False, STATIC_PHASE
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
                af, bf = _read_slot(
                    pc, ptrs, buffers, step, mi, ni, KI, IN_LOOP, True, STATIC_PHASE
                )
                if require_constexpr(mi == nm - 1 and ni == nn - 1):
                    ptrs = _advance(
                        pc, ptrs, step, STAGE, DRAIN, KI, IN_LOOP, STATIC_PHASE
                    )
            if require_constexpr(
                tc.commit_per_stage_whole() and mi == nm - 1 and ni == nn - 1
            ):
                with region("commit"):
                    pc.lds_ptrs.commit_buffer_load()
            a += _merge_ds_read_frags(am, af, tc, 0)
            b += _merge_ds_read_frags(bm, bf, tc, 1)
            acc += (value,)
    if require_constexpr(not IN_LOOP):
        buffers = _rotate_buffers(buffers, tc)
    return ptrs, buffers, _make_reg_fragments(a, b, acc)


@gluon.jit
def _run_buffered_pipeline(pc, ptrs, NUM_K):
    """Run the prologue and MAIN iterations; leave the drain to overlap the epilogue."""
    tc: gl.constexpr = pc.tuning_cfg
    gl.static_assert(
        tc.FROZEN_STEP or not tc.warp_pipeline_manual(),
        "the manual pipeline requires FROZEN_STEP",
    )
    gl.static_assert(
        tc.num_prefetch_mini() == tc.num_mini_k(),
        "the pipeline requires VGPR_PREFETCH_K == BLOCK_K",
    )
    depth: gl.constexpr = tc.pipeline_depth()
    unroll: gl.constexpr = tc.pipeline_unroll()
    peeled: gl.constexpr = tc.pipeline_peeled()
    main = NUM_K - depth
    gl.assume(main >= peeled + unroll)
    remaining = main - peeled
    countdown: gl.constexpr = (
        not pc.func_cfg.a_has_scale()
        and not pc.func_cfg.b_has_scale()
        and tc.pipeline_register_period() == 1
    )
    if require_constexpr(not countdown):
        unroll_end = remaining // unroll * unroll
    buffers = _init_buffers(pc)
    if require_constexpr(tc.FROZEN_STEP):
        ptrs = _prologue_frozen(pc, ptrs)
    else:
        # r = p - (NB_MAX - 1): active streams fill p - NB_DELTA.
        for r in gl.static_range(1 - depth, 0):
            for ni in gl.static_range(tc.num_mini_n()):
                for mi in gl.static_range(tc.num_mini_m()):
                    buffers = _fill_slot(pc, ptrs, buffers, r, r, False, mi, ni)
            ptrs = _advance(pc, ptrs, r, r, False)
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
    ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, 0, 0, DOT=False)
    for x in gl.static_range(peeled):
        ptrs, buffers, regs = _step(pc, ptrs, buffers, regs, x + 1, x + 1)
    # Payload-only LDS bodies benefit from removing the live rounded bound.
    if require_constexpr(not countdown):
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
                )
    else:
        base = 0
        left = remaining
        while left >= unroll:
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
                )
            base += unroll
            left -= unroll
        unroll_end = remaining - left
    if require_constexpr(unroll > 6 and tc.pipeline_register_period() > 1):
        # Long register-ring unrolls need a single remainder loop: a chain of
        # guarded tuple updates constrains allocation in the main body and spills
        # its address vectors. The ring rotates only in this remainder loop.
        for u in tl.range(0, remaining - unroll_end):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                peeled + unroll_end + u + 1,
                None,
            )
    else:
        # Short bodies benefit from static remainder expansion and constant LDS
        # indices; a second loop instead lengthens their register lifetimes.
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
                )
    return ptrs, buffers, regs


@gluon.jit
def _drain_buffered_pipeline(
    pc, ptrs, buffers, regs, NUM_K, EPILOGUE_GROUPS: gl.constexpr
):
    tc: gl.constexpr = pc.tuning_cfg
    main = NUM_K - tc.pipeline_depth()
    period: gl.constexpr = math.lcm(
        tc.num_buffers(0),
        tc.num_buffers(1),
        tc.num_buffers(0, True) if pc.func_cfg.a_has_scale() else 1,
        tc.num_buffers(1, True) if pc.func_cfg.b_has_scale() else 1,
        2 if tc.scale_packed_k128(0) or tc.scale_packed_k128(1) else 1,
    )
    # Keep short LDS rings in immediate offsets; live queues and quantized
    # epilogues need the smaller dynamic drain to avoid allocation pressure.
    if require_constexpr(
        period <= 3
        and tc.pipeline_unroll() % period == 0
        and tc.pipeline_register_period() == 1
        and pc.func_cfg.output_quant is None
    ):
        phase = main % period
        for p in gl.static_range(period):
            if phase == p:
                for j in gl.static_range(tc.pipeline_depth() - 1):
                    ptrs, buffers, regs = _step(
                        pc,
                        ptrs,
                        buffers,
                        regs,
                        main + j + 1,
                        j,
                        DRAIN=True,
                        EPILOGUE_GROUPS=EPILOGUE_GROUPS,
                        KI=p + j - tc.pipeline_peeled(),
                        STATIC_PHASE=True,
                    )
    else:
        for j in gl.static_range(tc.pipeline_depth() - 1):
            ptrs, buffers, regs = _step(
                pc,
                ptrs,
                buffers,
                regs,
                main + j + 1,
                j,
                DRAIN=True,
                EPILOGUE_GROUPS=EPILOGUE_GROUPS,
            )
    return regs


@gluon.jit
def _last_mfma(pc, regs):
    tc: gl.constexpr = pc.tuning_cfg
    acc = ()
    for ni in gl.static_range(tc.num_mini_n()):
        for mi in gl.static_range(tc.num_mini_m()):
            acc += (
                _maybe_block_dot(
                    _take_reg_pairs(
                        regs.a_payload,
                        regs.a_scale,
                        mi * tc.num_mini_k(),
                        tc.num_mini_k(),
                    ),
                    _take_reg_pairs(
                        regs.b_payload,
                        regs.b_scale,
                        ni * tc.num_mini_k(),
                        tc.num_mini_k(),
                    ),
                    regs.acc[_slot_index(mi, ni, tc.num_mini_m(), tc.num_mini_n())],
                    tc.num_mini_k(),
                    pc.func_cfg,
                    tc,
                    True,
                    0,
                ),
            )
    return acc


# JIT resolves these shared helpers after both modules have loaded.
from .moe_gemm import (  # noqa: I001 -- initialize shared helpers before the frozen module
    _make_reg_fragments,
    _maybe_block_dot,
    _merge_ds_read_frags,
    _PipelinePointers,
    _PipelineRegFragments,
    _sched_hint,
    _take_reg_pairs,
)

from ._frozen import _pipeline_step_frozen, _prologue_frozen
