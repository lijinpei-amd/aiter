# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Copy ownership and commit/wait accounting for the Gluon MoE pipeline.

Components use ``(operand, is_scale)`` identities throughout the live schedule.
"""

import math

from triton.experimental import gluon

from ._lang import unwrap as _v
from ._layout import _slot_index
from ._types import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    OPERANDS,
    PAYLOAD,
    SCALE,
    A,
    B,
    WaitCommitScheme,
)

__all__ = [
    "A_PAYLOAD",
    "A_SCALE",
    "B_PAYLOAD",
    "B_SCALE",
    "COMPONENTS",
    "OPERANDS",
    "PAYLOAD",
    "SCALE",
    "A",
    "B",
]


@gluon.constexpr_function
def _present(tc, component):
    """Whether a component exists for this operand-format pair."""
    operand, is_scale = component[0], component[1]
    return not _v(is_scale) or tc.func_cfg.has_scale(operand)


@gluon.constexpr_function
def _component_depth(tc, component):
    operand, is_scale = component[0], component[1]
    return tc.num_buffers(operand, is_scale)


@gluon.constexpr_function
def _component_ratio(tc, component):
    operand, is_scale = component[0], component[1]
    return tc.scale_ratio_k_step(operand) if _v(is_scale) else 1


@gluon.constexpr_function
def _live_span(tc, component):
    """Baseline inclusive lifetime ``L(c)`` in payload K steps."""
    return (_component_depth(tc, component) - 1) * _component_ratio(tc, component) + 1


@gluon.constexpr_function
def _via_lds(tc, component):
    """Resolved placement for one payload or scale component."""
    return tc.component_via_lds(component)


@gluon.constexpr_function
def _in_reg(tc, component):
    """Resolved direct-register placement for one present component."""
    return tc.component_in_reg(component)


@gluon.constexpr_function
def pipeline_depth(tc):
    """Largest prefetch span, measured in payload steps, of active components."""
    depth = 0
    for component in COMPONENTS:
        if _present(tc, component):
            depth = max(depth, _live_span(tc, component))
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
    for component in COMPONENTS:
        if _present(tc, component):
            period = math.lcm(
                period,
                _component_depth(tc, component) * _component_ratio(tc, component),
            )
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
def _component_fill_slot(tc, component, tile):
    """Flattened slot that owns one component/non-K-tile producer."""
    operand, is_scale = component[0], component[1]
    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    if _v(is_scale):
        return _scale_buffer_load_slot(
            operand == A, tile, nm, nn, tc.SCALE_FILL_MID
        )
    return _payload_buffer_load_slot(operand == A, tile, nm, nn)


@gluon.constexpr_function
def _component_nominal_read_slot(tc, component, tile):
    """Payload-owned slot at which a component would normally be selected."""
    operand = component[0]
    return _payload_buffer_load_slot(
        operand == A,
        tile,
        tc.num_m_slots_per_block(),
        tc.num_n_slots_per_block(),
    )


@gluon.constexpr_function
def _component_read_slot(tc, component, tile):
    """Legal selection slot after accounting for a same-step register producer."""
    nominal = _component_nominal_read_slot(tc, component, tile)
    if _in_reg(tc, component) and _fill_span(tc, component) == 1:
        producer = _component_fill_slot(tc, component, tile)
        if producer > nominal:
            return producer
    return nominal


@gluon.constexpr_function
def _component_read_tile(tc, component, mi, ni):
    """Non-K tile selected in this slot, or ``None`` when this component is idle."""
    if not _present(tc, component):
        return None
    operand, is_scale = component[0], component[1]
    count = (
        tc.num_m_slots_per_block()
        if operand == A
        else tc.num_n_slots_per_block()
    )
    slot = _slot_index(
        mi,
        ni,
        tc.num_m_slots_per_block(),
        tc.num_n_slots_per_block(),
    )
    for tile in range(count):
        if _v(is_scale) and tile % tc.scale_ratio_non_k_slot(operand) != 0:
            continue
        if _component_read_slot(tc, component, tile) == slot:
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
        if _present(tc, (operand, SCALE)):
            tile = _scale_buffer_load_tile(
                mi, ni, nm, nn, operand == A, tc.SCALE_FILL_MID
            )
            if tile is not None and tile % tc.scale_ratio_non_k_slot(operand) != 0:
                tile = None
        ops.append(tile)
    return tuple(ops)


@gluon.constexpr_function
def _fill_span(tc, component):
    """Actual inclusive producer-to-selection span ``F(c)`` in payload steps."""
    span = _live_span(tc, component)
    if _in_reg(tc, component):
        span -= 1
    return span


@gluon.constexpr_function
def _read_in_mfma(tc, component, tile):
    """Effective read region, including mandatory same-step fill ordering."""
    operand, is_scale = component[0], component[1]
    configured = tc.ds_read_in_mfma(operand, is_scale)
    if not _in_reg(tc, component) or _fill_span(tc, component) != 1:
        return configured
    return configured or _component_read_slot(
        tc, component, tile
    ) == _component_fill_slot(tc, component, tile)


@gluon.constexpr_function
def _active(tc, component, stage, drain=False):
    """Whether a component fills at this compile-time relative stage.

    ``None`` denotes a main-loop stage. Before the drain, ``stage`` is the
    absolute read stage and gates only staggered prologue startup. In the
    drain it is the zero-based drain iteration, independent of runtime K.
    """
    if not _present(tc, component):
        return False
    span = _fill_span(tc, component)
    if drain:
        return stage < pipeline_depth(tc) - span
    return stage is None or stage + span - 1 >= 0


@gluon.constexpr_function
def _producer_phase(tc, component):
    """Payload-step residue on which this component issues its producer load."""
    return (1 - _fill_span(tc, component)) % _component_ratio(tc, component)


@gluon.constexpr_function
def _loads_at_phase(tc, component, phase):
    """Whether a component issues a load at this K-step phase."""
    return _present(tc, component) and (
        phase % _component_ratio(tc, component) == _producer_phase(tc, component)
    )


@gluon.constexpr_function
def _reads_at_phase(tc, component, phase):
    """Whether this step selects a new value for the component."""
    return _present(tc, component) and phase % _component_ratio(tc, component) == 0


@gluon.constexpr_function
def _has_register_component(tc):
    """Whether any present live component uses a direct-register ring."""
    for component in COMPONENTS:
        if _present(tc, component) and _in_reg(tc, component):
            return True
    return False


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
                and _via_lds(tc, component)
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
            for component in COMPONENTS:
                tile = _component_read_tile(tc, component, mi, ni)
                if tile is None:
                    continue
                if _via_lds(tc, component) and _reads_at_phase(
                    tc, component, phase
                ):
                    required.append(
                        (
                            r - _fill_span(tc, component) + 1,
                            component,
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
def _stage_wait(tc, stage, drain=False, epilogue_groups=0, phase=None):
    """Wait count for the stage head, or ``None`` when this scheme waits per slot.

    Returning ``None`` for the inapplicable scheme keeps the emitter to one evaluation
    per site: the caller binds this once and tests it, instead of calling ``_wait``
    again inside the guard it just passed. ``_wait`` is the single most expensive
    constexpr in the kernel, so the duplicate mattered.
    """
    if not tc.commit_per_stage():
        return None
    return _wait(tc, stage, None, drain, epilogue_groups, phase)


@gluon.constexpr_function
def _slot_waits(tc, stage, drain=False, epilogue_groups=0, phase=None):
    """Per-slot wait counts for one step, indexed by ``_slot_index``.

    A whole vector rather than one query per slot: Gluon refuses a ``gl.constexpr``
    bound inside a ``static_range`` body ("constexpr cannot be reassigned"), so the
    emitter cannot hold a per-slot count in a local. Returning the vector lets it bind
    once at step scope and index, which evaluates ``_wait`` exactly once per slot
    instead of twice.
    """
    slots = tc.num_m_slots_per_block() * tc.num_n_slots_per_block()
    if tc.commit_per_stage():
        # Full length, all ``None``: the emitter then has one uniform test per slot
        # rather than a separate "is this scheme per-slot at all" guard.
        return (None,) * slots
    return tuple(
        _wait(tc, stage, slot, drain, epilogue_groups, phase) for slot in range(slots)
    )
