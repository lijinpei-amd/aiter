# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile-time issue history for the unified component-buffer pipeline.

Read stage ``r`` supplies tile ``r`` and fills tile ``r + NB - 1``.  Negative
stages are staggered prologue fills; stage zero seeds the first MFMA operands.
The runtime main-loop length never participates in group-count calculations.
"""

from triton.experimental import gluon

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
    """Payload/scale fills owned by a slot, including register destinations."""
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
def _active(tc, kind, stage, drain=False):
    """Whether a component fills at this compile-time relative stage.

    ``None`` denotes a main-loop stage.  Before the drain, ``stage`` is the
    absolute read stage and gates only staggered prologue startup.  In the
    drain it is the zero-based drain iteration, independent of runtime K.
    """
    if kind % 2 and not tc.func_cfg.has_scale(kind // 2):
        return False
    depth = tc.num_buffers(kind // 2, bool(kind % 2))
    if drain:
        return stage < tc.pipeline_depth() - depth
    return stage is None or stage + depth - 1 >= 0


@gluon.constexpr_function
def _async(tc, kind):
    """Direct-register loads own no LDS async-copy group."""
    return tc.scale_via_lds(kind // 2) if kind % 2 else tc.payload_via_lds(kind // 2)


@gluon.constexpr_function
def _groups(tc, stage, drain=False):
    """Committed groups per slot, including explicit empty slot/stage groups."""
    schedule, whole = [], ()
    for ni in range(tc.num_mini_n()):
        for mi in range(tc.num_mini_m()):
            ops = tuple(
                (kind, tile)
                for kind, tile in enumerate(_ops(tc, mi, ni))
                if tile is not None
                and _active(tc, kind, stage, drain)
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
                    if mi == tc.num_mini_m() - 1 and ni == tc.num_mini_n() - 1
                    else ()
                )
            schedule.append(groups)
    return tuple(schedule)


@gluon.constexpr_function
def _wait(tc, stage, slot, drain=False, epilogue_groups=0):
    """Count groups newer than the youngest producer required by this read.

    For a stage commit, ``slot=None`` waits for all operands at the stage head.
    The drain uses a local origin: its first read is stage one, and epilogue
    groups commit between stages zero and one.  Once a producer is newer than
    those epilogue groups, they no longer contribute to its wait allowance.
    """
    nm, nn = tc.num_mini_m(), tc.num_mini_n()
    r = stage + 1 if drain else tc.pipeline_depth() if stage is None else stage
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
        if drain:
            schedule = _groups(tc, s - 1, True) if s > 0 else _groups(tc, None)
        else:
            schedule = _groups(tc, None if stage is None else s)
        for pos, groups in enumerate(schedule):
            if s == r and (slot is None or pos >= slot):
                break
            timeline.extend(
                tuple((s, kind, tile) for kind, tile in group) for group in groups
            )
        if drain and s == 0:
            timeline.extend(() for _ in range(epilogue_groups))

    latest = max(
        i for i, group in enumerate(timeline) if any(op in group for op in required)
    )
    return len(timeline) - latest - 1


@gluon.constexpr_function
def _pipeline_peeled(tc):
    """Smallest initial main prefix after which every wait is invariant.

    The first MFMA is always peeled to retain its visible zero accumulator.
    With every active depth at least two, read stage ``NB_MAX - 2`` already
    has an entirely full producer history.  Checking the finite prefix also
    handles configurations where shared groups hide some staggered fills.
    """
    # These schemes commit even empty prologue slots/stages.  Their group
    # distances already match the steady state when a component first reads.
    # At depth three or below, the mandatory first peel covers any warmup.
    if not tc.commit_per_op() or tc.pipeline_depth() <= 3:
        return 1
    slots = range(tc.num_mini_m() * tc.num_mini_n())
    steady = tuple(_wait(tc, None, slot) for slot in slots)
    peeled = 1
    for x in range(max(0, tc.pipeline_depth() - 2)):
        if tuple(_wait(tc, x + 1, slot) for slot in slots) != steady:
            peeled = x + 1
    return peeled
