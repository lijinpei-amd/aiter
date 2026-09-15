# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Copy ownership and commit/wait accounting for the Gluon MoE pipeline."""

import math

from triton.experimental import gluon

from ._lang import unwrap as _v
from ._layout import _slot_index
from ._types import WaitCommitScheme


@gluon.constexpr_function
def pipeline_depth(tc):
    """Largest prefetch span, measured in payload steps, of active components."""
    depth = max(tc.buffer_live_span(0), tc.buffer_live_span(1))
    for operand in (0, 1):
        if tc.func_cfg.has_scale(operand):
            depth = max(depth, tc.buffer_live_span(operand, True))
    return depth


@gluon.constexpr_function
def pipeline_unroll(tc):
    """LCM of the requested unroll and every active component ring period.

    Payload rings advance every step. Scale rings advance once per scale K tile,
    so their period in payload steps is their depth times the scale step ratio.
    Including LDS rings keeps every complete body at a fixed ring phase as well as
    satisfying the static-index requirement of register tuples.
    """
    requested = _v(tc.K_UNROLL)
    assert requested >= 1, "K_UNROLL must be at least 1"
    period = requested
    for operand in (0, 1):
        period = math.lcm(period, tc.num_buffers(operand))
        if tc.func_cfg.has_scale(operand):
            scale_depth = tc.num_buffers(operand, True)
            scale_ratio = tc.scale_step_ratio(operand)
            period = math.lcm(period, scale_depth * scale_ratio)
    return period


@gluon.constexpr_function
def _pipeline_peeled(tc):
    """Smallest initial main prefix after which every wait is invariant.

    The first MFMA is always peeled to retain its visible zero accumulator.
    With every active depth at least two, read stage ``NB_MAX - 2`` already
    has an entirely full producer history. Checking the finite prefix also
    handles configurations where shared groups hide some staggered fills.
    """
    # These schemes commit even empty prologue slots/stages. Their group
    # distances already match the steady state when a component first reads.
    # At depth three or below, the mandatory first peel covers any warmup.
    if not tc.commit_per_op() or pipeline_depth(tc) <= 3:
        return 1
    slots = range(tc.num_m_slots_per_block() * tc.num_n_slots_per_block())
    peeled = 1
    for x in range(max(0, pipeline_depth(tc) - 2)):
        steady = tuple(_wait(tc, None, slot, phase=x + 1) for slot in slots)
        if tuple(_wait(tc, x + 1, slot) for slot in slots) != steady:
            peeled = x + 1
    return peeled


@gluon.constexpr_function
def _buffer_load_order(NM, NN):
    """Payload copies as ``(is_a, tile)`` in slot order."""
    NM, NN = _v(NM), _v(NN)
    if NM == 2 and NN == 2:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = []
    for i in range(max(NM, NN)):
        if i < NM:
            out.append((1, i))
        if i < NN:
            out.append((0, i))
    return out


@gluon.constexpr_function
def _payload_buffer_load_slot(is_a, tile, NM, NN):
    return _buffer_load_order(NM, NN).index((int(bool(_v(is_a))), _v(tile)))


@gluon.constexpr_function
def _buffer_load_pos(mi, ni, NM, NN):
    s = _slot_index(mi, ni, NM, NN)
    return s if s < _v(NM) + _v(NN) else None


@gluon.constexpr_function
def _buffer_load_tile(pos, NM, NN, want_a):
    pos = _v(pos)
    if pos is None:
        return None
    is_a, tile = _buffer_load_order(NM, NN)[pos]
    return tile if bool(is_a) == bool(_v(want_a)) else None


@gluon.constexpr_function
def _ds_read_a_tile(mi, ni, NM, NN):
    return _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, True)


@gluon.constexpr_function
def _ds_read_b_tile(mi, ni, NM, NN):
    return _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, False)


@gluon.constexpr_function
def _scale_buffer_load_slot(is_a, tile, NM, NN, SCALE_FILL_MID=False):
    """A scale can be copied in a different slot from its payload."""
    if _v(SCALE_FILL_MID) and _v(NM) == 2 and _v(NN) == 2:
        return 1 + _v(tile)
    return _payload_buffer_load_slot(is_a, tile, NM, NN)


@gluon.constexpr_function
def _scale_buffer_load_tile(mi, ni, NM, NN, want_a, SCALE_FILL_MID=False):
    s = _slot_index(mi, ni, NM, NN)
    for tile in range(_v(NM) if _v(want_a) else _v(NN)):
        if _scale_buffer_load_slot(want_a, tile, NM, NN, SCALE_FILL_MID) == s:
            return tile
    return None


@gluon.constexpr_function
def _ops(tc, mi, ni):
    """Payload/scale fills owned by a slot, including register destinations."""
    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    pos = _buffer_load_pos(mi, ni, nm, nn)
    ops = []
    for idx in range(2):
        ops.append(_buffer_load_tile(pos, nm, nn, idx == 0))
        tile = None
        if tc.func_cfg.has_scale(idx):
            tile = _scale_buffer_load_tile(mi, ni, nm, nn, idx == 0, tc.SCALE_FILL_MID)
            if tile is not None and tile % tc.scale_tile_ratio(idx) != 0:
                tile = None
        ops.append(tile)
    return tuple(ops)


@gluon.constexpr_function
def _fill_span(tc, kind):
    """Number of read stages between startup and the final fill.

    Direct B has ``NB`` physical banks but only an ``NB - 1`` stage producer
    span: while MFMA consumes B[k], HBM fills B[k + NB - 1]. LDS payloads
    retain the extra copy/read stage.
    """
    span = tc.buffer_live_span(kind // 2, bool(kind % 2))
    if kind == 2 and not tc.payload_via_lds(1):
        span -= 1
    return span


@gluon.constexpr_function
def _read_payload_after_fill(tc, operand):
    """Whether this step's payload registers are selected after its HBM fill."""
    return tc.ds_read_in_mfma(operand) or (
        operand == 1
        and not tc.payload_via_lds(operand)
        and tc.num_buffers(operand) == 2
    )


@gluon.constexpr_function
def _active(tc, kind, stage, drain=False):
    """Whether a component fills at this compile-time relative stage.

    ``None`` denotes a main-loop stage. Before the drain, ``stage`` is the
    absolute read stage and gates only staggered prologue startup. In the
    drain it is the zero-based drain iteration, independent of runtime K.
    """
    if kind % 2 and not tc.func_cfg.has_scale(kind // 2):
        return False
    depth = _fill_span(tc, kind)
    if drain:
        return stage < pipeline_depth(tc) - depth
    return stage is None or stage + depth - 1 >= 0


@gluon.constexpr_function
def _async(tc, kind):
    """Direct-register loads own no LDS async-copy group."""
    return tc.scale_via_lds(kind // 2) if kind % 2 else tc.payload_via_lds(kind // 2)


@gluon.constexpr_function
def _groups(tc, stage, drain=False, phase=None):
    """Committed groups at a read phase, including empty slot/stage groups."""
    schedule, whole = [], ()
    phase = stage if phase is None and stage is not None else phase or 0
    for ni in range(tc.num_n_slots_per_block()):
        for mi in range(tc.num_m_slots_per_block()):
            ops = tuple(
                (kind, tile)
                for kind, tile in enumerate(_ops(tc, mi, ni))
                if tile is not None
                and _active(tc, kind, stage, drain)
                and _async(tc, kind)
                and (not kind % 2 or phase % tc.scale_step_ratio(kind // 2) == 0)
            )
            if tc.commit_per_op():
                groups = tuple((op,) for op in ops)
            elif tc.commit_per_slot():
                groups = (ops,)
            else:
                whole += ops
                groups = (
                    (whole,)
                    if mi == tc.num_m_slots_per_block() - 1
                    and ni == tc.num_n_slots_per_block() - 1
                    else ()
                )
            schedule.append(groups)
    return tuple(schedule)


@gluon.constexpr_function
def _wait(tc, stage, slot, drain=False, epilogue_groups=0, phase=None):
    """Count groups newer than the youngest producer required by this read.

    For a stage commit, ``slot=None`` waits for all operands at the stage head.
    The drain uses a local origin: its first read is stage one, and epilogue
    groups commit between stages zero and one. Once a producer is newer than
    those epilogue groups, they no longer contribute to its wait allowance.
    ``phase`` retains the absolute scale cadence when stages use a local origin.
    """
    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    r = stage + 1 if drain else pipeline_depth(tc) if stage is None else stage
    phase = r if phase is None else phase
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
                if (
                    tc.func_cfg.has_scale(idx)
                    and tc.scale_via_lds(idx)
                    and phase % tc.scale_step_ratio(idx) == 0
                    and tile % tc.scale_tile_ratio(idx) == 0
                ):
                    required.append(
                        (
                            r - tc.buffer_live_span(idx, True) + 1,
                            idx * 2 + 1,
                            tile,
                        )
                    )
    if not required:
        return None

    timeline = []
    for s in range(r - pipeline_depth(tc) + 1, r + 1):
        if drain:
            schedule = (
                _groups(tc, s - 1, True, phase + s - r)
                if s > 0
                else _groups(tc, None, phase=phase + s - r)
            )
        else:
            schedule = _groups(tc, None if stage is None else s, phase=phase + s - r)
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
def _buffer_load_ops(tc, mi, ni):
    """A payload/scale and B payload/scale tiles, in issue order; None skips a copy."""
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    pos = _buffer_load_pos(mi, ni, NM, NN)
    ops = []
    for idx in range(2):
        ops.append(_buffer_load_tile(pos, NM, NN, idx == 0))
        scale_tile = None
        if tc.func_cfg.has_scale(idx) and tc.scale_via_lds(idx):
            scale_tile = _scale_buffer_load_tile(
                mi, ni, NM, NN, idx == 0, tc.SCALE_FILL_MID
            )
            if scale_tile is not None and scale_tile % tc.scale_tile_ratio(idx) != 0:
                scale_tile = None
        ops.append(scale_tile)
    return tuple(ops)


@gluon.constexpr_function
def _buffer_load_group_schedule(tc):
    """Committed groups per slot, containing async ``(operand_kind, tile)`` copies.

    Kinds 0/1/2/3 name A payload/A scale/B payload/B scale. Direct-register payload
    and scale loads have no async group. Traffic-only experiments
    retain their nominal groups even when the LDS manager suppresses a copy.
    """
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    scheme = _v(tc.WAIT_COMMIT_SCHEME)
    assert scheme in tuple(int(s) for s in WaitCommitScheme)
    schedule = []
    stage_ops = ()
    for ni in range(NN):
        for mi in range(NM):
            ops = tuple(
                (kind, tile)
                for kind, tile in enumerate(_buffer_load_ops(tc, mi, ni))
                if tile is not None and (kind % 2 or tc.payload_via_lds(kind // 2))
            )
            if scheme == int(WaitCommitScheme.PER_OP):
                groups = tuple((op,) for op in ops)
            elif scheme == int(WaitCommitScheme.PER_SLOT):
                groups = (ops,)  # Empty tail slots still commit a group.
            else:
                stage_ops += ops
                groups = (stage_ops,) if mi == NM - 1 and ni == NN - 1 else ()
            schedule.append(groups)
    return tuple(schedule)


@gluon.constexpr_function
def _buffer_load_groups(tc):
    return tuple(group for slot in _buffer_load_group_schedule(tc) for group in slot)


@gluon.constexpr_function
def _buffer_load_wait(tc, mi, ni, STAGES_BETWEEN, DO_BUFFER_LOAD):
    """Number of newer committed groups after this slot's last required copy."""
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    slot = _slot_index(mi, ni, NM, NN)
    if tc.commit_per_stage():
        return _v(STAGES_BETWEEN) if slot == 0 else None

    schedule = _buffer_load_group_schedule(tc)
    groups = tuple(group for entry in schedule for group in entry)
    required = []
    for idx in range(2):
        tile = _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, idx == 0)
        if tile is None:
            continue
        if tc.payload_via_lds(idx):
            required.append((idx * 2, tile))
        if (
            tc.func_cfg.has_scale(idx)
            and tc.scale_via_lds(idx)
            and tile % tc.scale_tile_ratio(idx) == 0
        ):
            required.append((idx * 2 + 1, tile))
    if not required:
        return None
    latest = max(
        i for i, group in enumerate(groups) if any(op in group for op in required)
    )
    prefix = sum(len(entry) for entry in schedule[:slot]) if _v(DO_BUFFER_LOAD) else 0
    # The suffix comprises the target stage's remaining groups, full newer stages,
    # and the groups already committed while walking this stage's slots.
    return (_v(STAGES_BETWEEN) + 1) * len(groups) - 1 - latest + prefix
