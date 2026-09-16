# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Copy ownership and commit/wait accounting for the Gluon MoE pipeline.

Components use ``(operand, is_scale)`` identities throughout the live schedule.
"""

import math

from triton.experimental import gluon

from ._lang import unwrap as _v
from ._layout import _slot_index
from ._types import WaitCommitScheme

A = 0
B = 1
PAYLOAD = False
SCALE = True

A_PAYLOAD = (A, PAYLOAD)
A_SCALE = (A, SCALE)
B_PAYLOAD = (B, PAYLOAD)
B_SCALE = (B, SCALE)

OPERANDS = (A, B)
# Explicit issue order; PER_OP wait counts depend on this ordering.
COMPONENTS = (A_PAYLOAD, A_SCALE, B_PAYLOAD, B_SCALE)


@gluon.constexpr_function
def pipeline_depth(tc):
    """Largest prefetch span, measured in payload steps, of active components."""
    depth = max(tc.buffer_live_span(A), tc.buffer_live_span(B))
    for operand in OPERANDS:
        if tc.func_cfg.has_scale(operand):
            depth = max(depth, tc.buffer_live_span(operand, True))
    return depth


@gluon.constexpr_function
def pipeline_unroll(tc):
    """LCM of the requested unroll and every active component ring period.

    Payload rings advance every step. Scale rings advance once per scale K tile,
    so their period in payload steps is their depth times the scale K-step ratio.
    Including LDS rings keeps every complete body at a fixed ring phase as well as
    satisfying the static-index requirement of register tuples.
    """
    requested = _v(tc.K_UNROLL)
    assert requested >= 1, "K_UNROLL must be at least 1"
    period = requested
    for operand in OPERANDS:
        period = math.lcm(period, tc.num_buffers(operand))
        if tc.func_cfg.has_scale(operand):
            scale_depth = tc.num_buffers(operand, True)
            scale_ratio = tc.scale_ratio_k_step(operand)
            period = math.lcm(period, scale_depth * scale_ratio)
    return period


@gluon.constexpr_function
def _pipeline_peeled(tc):
    """Smallest initial main prefix after which every wait is invariant.

    The first MFMA is always peeled to retain its visible zero accumulator.
    By read stage ``pipeline_depth(tc) - 2``, every active component has an
    entirely full producer history. Checking the finite prefix also handles
    configurations where shared groups hide some staggered fills.
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
    """Tiles in ``COMPONENTS`` order, including direct-register destinations."""
    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    pos = _buffer_load_pos(mi, ni, nm, nn)
    ops = []
    for operand in OPERANDS:
        ops.append(_buffer_load_tile(pos, nm, nn, operand == A))
        tile = None
        if tc.func_cfg.has_scale(operand):
            tile = _scale_buffer_load_tile(
                mi, ni, nm, nn, operand == A, tc.SCALE_FILL_MID
            )
            if tile is not None and tile % tc.scale_ratio_non_k_slot(operand) != 0:
                tile = None
        ops.append(tile)
    return tuple(ops)


@gluon.constexpr_function
def _fill_span(tc, component):
    """Number of read stages between startup and the final fill.

    Direct B has ``NB`` physical banks but only an ``NB - 1`` stage producer
    span: while MFMA consumes B[k], HBM fills B[k + NB - 1]. LDS payloads
    retain the extra copy/read stage.
    """
    operand, is_scale = component[0], component[1]
    span = tc.buffer_live_span(operand, is_scale)
    if operand == B and not is_scale and not tc.payload_via_lds(B):
        span -= 1
    return span


@gluon.constexpr_function
def _read_payload_after_fill(tc, operand):
    """Whether this step's payload registers are selected after its HBM fill."""
    return tc.ds_read_in_mfma(operand) or (
        operand == B
        and not tc.payload_via_lds(operand)
        and tc.num_buffers(operand) == 2
    )


@gluon.constexpr_function
def _active(tc, component, stage, drain=False):
    """Whether a component fills at this compile-time relative stage.

    ``None`` denotes a main-loop stage. Before the drain, ``stage`` is the
    absolute read stage and gates only staggered prologue startup. In the
    drain it is the zero-based drain iteration, independent of runtime K.
    """
    operand, is_scale = component[0], component[1]
    if is_scale and not tc.func_cfg.has_scale(operand):
        return False
    depth = _fill_span(tc, component)
    if drain:
        return stage < pipeline_depth(tc) - depth
    return stage is None or stage + depth - 1 >= 0


@gluon.constexpr_function
def _async(tc, component):
    """Direct-register loads own no LDS async-copy group."""
    operand, is_scale = component[0], component[1]
    return tc.scale_via_lds(operand) if is_scale else tc.payload_via_lds(operand)


@gluon.constexpr_function
def _loads_at_phase(tc, component, phase):
    """Whether a component issues a load at this K-step phase."""
    operand, is_scale = component[0], component[1]
    return not is_scale or phase % tc.scale_ratio_k_step(operand) == 0


@gluon.constexpr_function
def _groups(tc, stage, drain=False, phase=None):
    """Committed groups at a read phase, including empty slot/stage groups."""
    schedule, whole = [], ()
    phase = stage if phase is None and stage is not None else phase or 0
    for ni in range(tc.num_n_slots_per_block()):
        for mi in range(tc.num_m_slots_per_block()):
            ops = tuple(
                (component, tile)
                for component, tile in zip(COMPONENTS, _ops(tc, mi, ni))
                if tile is not None
                and _active(tc, component, stage, drain)
                and _async(tc, component)
                and _loads_at_phase(tc, component, phase)
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
    ``phase`` preserves the absolute payload-step phase used to apply each scale's
    K-step ratio when stages use a local origin.
    """
    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    r = stage + 1 if drain else pipeline_depth(tc) if stage is None else stage
    phase = r if phase is None else phase
    required = []
    for ni in range(nn):
        for mi in range(nm):
            if slot is not None and _slot_index(mi, ni, nm, nn) != slot:
                continue
            for operand in OPERANDS:
                tile = (
                    _ds_read_a_tile(mi, ni, nm, nn)
                    if operand == A
                    else _ds_read_b_tile(mi, ni, nm, nn)
                )
                if tile is None:
                    continue
                if tc.payload_via_lds(operand):
                    required.append(
                        (r - tc.num_buffers(operand) + 1, (operand, PAYLOAD), tile)
                    )
                if (
                    tc.func_cfg.has_scale(operand)
                    and tc.scale_via_lds(operand)
                    and phase % tc.scale_ratio_k_step(operand) == 0
                    and tile % tc.scale_ratio_non_k_slot(operand) == 0
                ):
                    required.append(
                        (
                            r - tc.buffer_live_span(operand, True) + 1,
                            (operand, SCALE),
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
                tuple((s, component, tile) for component, tile in group)
                for group in groups
            )
        if drain and s == 0:
            timeline.extend(() for _ in range(epilogue_groups))

    latest = max(
        i for i, group in enumerate(timeline) if any(op in group for op in required)
    )
    return len(timeline) - latest - 1


@gluon.constexpr_function
def _buffer_load_ops(tc, mi, ni):
    """Candidate tiles in ``COMPONENTS`` issue order; ``None`` skips a copy."""
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    pos = _buffer_load_pos(mi, ni, NM, NN)
    ops = []
    for operand in OPERANDS:
        ops.append(_buffer_load_tile(pos, NM, NN, operand == A))
        scale_tile = None
        if tc.func_cfg.has_scale(operand) and tc.scale_via_lds(operand):
            scale_tile = _scale_buffer_load_tile(
                mi, ni, NM, NN, operand == A, tc.SCALE_FILL_MID
            )
            if (
                scale_tile is not None
                and scale_tile % tc.scale_ratio_non_k_slot(operand) != 0
            ):
                scale_tile = None
        ops.append(scale_tile)
    return tuple(ops)


@gluon.constexpr_function
def _buffer_load_group_schedule(tc):
    """Committed groups per slot, containing async ``(component, tile)`` copies.

    Direct-register payload and scale loads have no async group. Traffic-only
    experiments retain their nominal groups even when the LDS manager suppresses a
    copy.
    """
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    scheme = _v(tc.WAIT_COMMIT_SCHEME)
    assert scheme in tuple(int(s) for s in WaitCommitScheme)
    schedule = []
    stage_ops = ()
    for ni in range(NN):
        for mi in range(NM):
            ops = tuple(
                (component, tile)
                for component, tile in zip(COMPONENTS, _buffer_load_ops(tc, mi, ni))
                if tile is not None and _async(tc, component)
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
    for operand in OPERANDS:
        tile = _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, operand == A)
        if tile is None:
            continue
        if tc.payload_via_lds(operand):
            required.append(((operand, PAYLOAD), tile))
        if (
            tc.func_cfg.has_scale(operand)
            and tc.scale_via_lds(operand)
            and tile % tc.scale_ratio_non_k_slot(operand) == 0
        ):
            required.append(((operand, SCALE), tile))
    if not required:
        return None
    latest = max(
        i for i, group in enumerate(groups) if any(op in group for op in required)
    )
    prefix = sum(len(entry) for entry in schedule[:slot]) if _v(DO_BUFFER_LOAD) else 0
    # The suffix comprises the target stage's remaining groups, full newer stages,
    # and the groups already committed while walking this stage's slots.
    return (_v(STAGES_BETWEEN) + 1) * len(groups) - 1 - latest + prefix
