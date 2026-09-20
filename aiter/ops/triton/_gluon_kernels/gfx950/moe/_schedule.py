# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Copy ownership and commit/wait accounting for the Gluon MoE pipeline.

Components use ``(operand, is_scale)`` identities throughout the live schedule.
"""

import math
from typing import NamedTuple

from triton.experimental import gluon

from . import _schedule_cache
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
    "ScheduleSpec",
    "async_pattern_period",
    "ring_restoration_period",
    "schedule_spec",
]


class ScheduleSpec(NamedTuple):
    """Every configuration value the schedule model reads, resolved and hashable.

    The model's whole input surface, in one place. Two uses:

    * it states what the schedule actually depends on. Anything not here cannot
      change the schedule, which is what lets a caller reason about whether a tuning
      change can move the emitted code;
    * it is a sound memo key. The tuning aggregate is not: Triton builds it with
      ``eq_default=False``, so it hashes by identity and a single compile constructs
      hundreds of distinct objects for one value, and three of its fields are
      ``gl.constexpr(list)`` and unhashable outright.

    Built from accessor *results*, never from raw fields, because the placement
    accessors fold in automatic fallbacks and because test doubles mutate their
    inputs after construction.
    """

    nm: int
    nn: int
    #: ``(per_op, per_slot, per_stage)`` -- the three granularities the model branches
    #: on, rather than the four-valued scheme, since the two per-stage schemes differ
    #: only in where the emitter puts the commit.
    commit: tuple
    scale_fill_mid: bool
    k_unroll: int
    present: tuple
    depth: tuple
    ratio_k: tuple
    via_lds: tuple
    ds_read_in_mfma: tuple
    #: Per operand, not per component: only scales share across non-K slots.
    ratio_non_k: tuple


@gluon.constexpr_function
def schedule_spec(tc):
    """Resolve the schedule model's input surface from a tuning configuration."""
    present = tuple(_present(tc, c) for c in COMPONENTS)
    via_lds = tuple(bool(_v(tc.component_via_lds(c))) for c in COMPONENTS)
    for c, lds in zip(COMPONENTS, via_lds):
        assert bool(_v(tc.component_in_reg(c))) is not lds, (
            f"component_in_reg must be the complement of component_via_lds for {c}"
        )
    return ScheduleSpec(
        nm=_v(tc.num_m_slots_per_block()),
        nn=_v(tc.num_n_slots_per_block()),
        commit=(
            bool(_v(tc.commit_per_op())),
            bool(_v(tc.commit_per_slot())),
            bool(_v(tc.commit_per_stage())),
        ),
        scale_fill_mid=bool(_v(tc.SCALE_FILL_MID)),
        k_unroll=_v(tc.K_UNROLL),
        present=present,
        depth=tuple(_v(tc.num_buffers(c[0], c[1])) for c in COMPONENTS),
        ratio_k=tuple(_component_ratio(tc, c) for c in COMPONENTS),
        via_lds=via_lds,
        ds_read_in_mfma=tuple(
            bool(_v(tc.ds_read_in_mfma(c[0], c[1]))) for c in COMPONENTS
        ),
        ratio_non_k=tuple(_v(tc.scale_ratio_non_k_slot(o)) for o in OPERANDS),
    )


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
def ring_restoration_period(tc):
    """Payload steps after which every present component ring is back at slot zero.

    A component's ring period is its depth times its scale K-step ratio, not its live
    span ``(D - 1) * R + 1``: the span says when a value is consumed, the period says
    when the ring repeats.

    Distinct from three neighbours it is easy to conflate with. The requested unroll
    (``K_UNROLL``) is a tuning knob. The effective unroll is this LCM'd with that knob.
    The register-only period (``pipeline_register_period``) counts the same quantity
    over direct-register rings alone, because only those need a static index.
    """
    period = 1
    for component in COMPONENTS:
        if _present(tc, component):
            period = math.lcm(
                period,
                _component_depth(tc, component) * _component_ratio(tc, component),
            )
    return period


@gluon.constexpr_function
def async_pattern_period(tc):
    """Payload steps after which the issued async-copy pattern repeats.

    The LCM of the scale K-step ratios over present LDS-backed components; payload
    components contribute one. Shorter than the ring restoration period, because
    which copies issue depends only on the K-step phase, not on which ring slot they
    land in. Direct-register components do not participate -- they issue no async
    copy -- so this is the period of the commit-group pattern specifically.
    """
    period = 1
    for component in COMPONENTS:
        if _present(tc, component) and _via_lds(tc, component):
            period = math.lcm(period, _component_ratio(tc, component))
    return period


@gluon.constexpr_function
def pipeline_unroll(tc):
    """Requested unroll, widened to a whole number of every component ring period.

    Including LDS rings keeps every complete body at a fixed ring phase as well as
    satisfying the static-index requirement of register tuples.
    """
    requested = _v(tc.K_UNROLL)
    assert requested >= 1, "K_UNROLL must be at least 1"
    return math.lcm(requested, ring_restoration_period(tc))


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
def _payload_fill_slots(NM, NN):
    """Slot that owns each payload tile, as ``(A tiles, B tiles)``.

    The resolved form of :func:`_buffer_load_order`. Ownership is a mapping from a
    component's non-K tile to a slot, not a position lookup into a shared list --
    which is what lets several components own copies in the same slot.

    Two policies. With both axes split, the interleaved order (and its 2x2 special
    case) assigns one payload copy per slot, which is what the tuned kernels were
    measured with and must not move.

    With a degenerate axis there are fewer slots than copies -- ``NM + NN`` copies
    into ``NM * NN`` slots -- so one slot has to own both operands. Fall back to
    axis-major, ``A(t)`` at ``(mi=t, ni=0)`` and ``B(t)`` at ``(mi=0, ni=t)``, which
    is total and injective per component for any ``NM, NN >= 1``. Applying it to a
    split tile would reassign that tile's copies, so it is restricted to the case
    the interleave cannot express.
    """
    NM, NN = _v(NM), _v(NN)
    if NM > 1 and NN > 1:
        order = _buffer_load_order(NM, NN)
        return (
            tuple(order.index((1, tile)) for tile in range(NM)),
            tuple(order.index((0, tile)) for tile in range(NN)),
        )
    return (
        tuple(_slot_index(tile, 0, NM, NN) for tile in range(NM)),
        tuple(_slot_index(0, tile, NM, NN) for tile in range(NN)),
    )


@gluon.constexpr_function
def _component_fill_slots(tc, component):
    """Slot that owns each non-K tile of one component.

    ``SCALE_FILL_MID`` is consumed here and nowhere else: downstream ownership,
    grouping, dependency and emission all read the resolved mapping. Note it is
    guarded to the 2x2 split, so on any other geometry it is inert -- a rectangular
    tile that sets it is not testing anything.
    """
    nm, nn = _v(tc.num_m_slots_per_block()), _v(tc.num_n_slots_per_block())
    a_slots, b_slots = _payload_fill_slots(nm, nn)
    if _v(component[1]) and _v(tc.SCALE_FILL_MID) and nm == 2 and nn == 2:
        return (1, 2)
    return a_slots if _v(component[0]) == A else b_slots


@gluon.constexpr_function
def _fill_slots_valid(tc):
    """Every mapped slot exists, and no component owns two tiles in one slot.

    The second half is what keeps the flat ``_ops()`` transport representable: it
    carries one optional tile per component, so a component may own at most one tile
    per slot. Different components sharing a slot is fine and expected.
    """
    slots = _v(tc.num_m_slots_per_block()) * _v(tc.num_n_slots_per_block())
    for component in COMPONENTS:
        mapping = _component_fill_slots(tc, component)
        expected = (
            _v(tc.num_m_slots_per_block())
            if _v(component[0]) == A
            else _v(tc.num_n_slots_per_block())
        )
        if len(mapping) != expected:
            return False
        if any(slot < 0 or slot >= slots for slot in mapping):
            return False
        if len(set(mapping)) != len(mapping):
            return False
    return True


@gluon.constexpr_function
def _payload_buffer_load_slot(is_a, tile, NM, NN):
    a_slots, b_slots = _payload_fill_slots(NM, NN)
    return (a_slots if bool(_v(is_a)) else b_slots)[_v(tile)]


@gluon.constexpr_function
def _component_fill_slot(tc, component, tile):
    """Flattened slot that owns one component/non-K-tile producer."""
    return _component_fill_slots(tc, component)[_v(tile)]


@gluon.constexpr_function
def _candidate_ops(tc, slot):
    """Tiles this slot owns, in ``COMPONENTS`` order; ``None`` where it owns none.

    The single inversion of the fill mappings, and the only per-slot ownership view.
    Scale tiles covered by a wider earlier load are filtered out here rather than
    being absent from the mapping, so the mappings stay indexed by logical payload
    tile.
    """
    out = []
    for component in COMPONENTS:
        tile = None
        if _present(tc, component):
            ratio = (
                _v(tc.scale_ratio_non_k_slot(_v(component[0])))
                if _v(component[1])
                else 1
            )
            for candidate, owner in enumerate(_component_fill_slots(tc, component)):
                if owner == _v(slot) and candidate % ratio == 0:
                    tile = candidate
                    break
        out.append(tile)
    return tuple(out)


@gluon.constexpr_function
def _component_nominal_read_slot(tc, component, tile):
    """Payload-owned slot at which a component would normally be selected."""
    return _component_fill_slots(tc, (component[0], PAYLOAD))[_v(tile)]


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
    """Tiles in ``COMPONENTS`` order, including direct-register destinations.

    The flat four-element transport across the constexpr-to-JIT boundary -- nested
    tuples do not survive it. A view over :func:`_candidate_ops`, not a second
    ownership model.
    """
    return _candidate_ops(
        tc,
        _slot_index(
            mi, ni, tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
        ),
    )


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
def _read_step(tc, stage, drain):
    """Absolute payload step this wait point reads at.

    Three coordinates arrive here. ``None`` is steady state, where only the phase
    matters, so any step late enough for every component to be active will do and
    ``pipeline_depth`` is the cheapest such. An integer is an absolute startup step.
    In the drain it is the zero-based drain iteration, whose first read is step one.
    """
    if drain:
        return stage + 1
    return pipeline_depth(tc) if stage is None else stage


@gluon.constexpr_function
def _required_producers(tc, read_step, slot, phase):
    """Producer tokens the LDS reads at this point demand.

    ``(producer_step, component, non_k_tile)``. The step is part of the identity
    because the same copy repeats every ring period, so naming the component and tile
    alone would match a different iteration's value.

    Direct-register components are absent by construction: they have no commit group,
    and the compiler tracks their VMEM dependency itself.
    """
    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    out = []
    for ni in range(nn):
        for mi in range(nm):
            if slot is not None and _slot_index(mi, ni, nm, nn) != slot:
                continue
            for component in COMPONENTS:
                tile = _component_read_tile(tc, component, mi, ni)
                if tile is None:
                    continue
                if _via_lds(tc, component) and _reads_at_phase(tc, component, phase):
                    out.append(
                        (read_step - _fill_span(tc, component) + 1, component, tile)
                    )
    return tuple(out)


@gluon.constexpr_function
def _committed_before(tc, stage, read_step, slot, drain, epilogue_groups, phase):
    """Commit groups issued before this wait point, oldest first.

    Empty entries are meaningful: a per-slot or per-stage scheme commits even when a
    slot issued nothing, and an output-epilogue group carries no pipeline producer at
    all. Both occupy a FIFO position, so both shift the distance to a required group.

    The window starts at ``read_step - pipeline_depth + 1``. That is sufficient and
    tight: a required producer sits at ``read_step - F(c) + 1`` with ``F(c) = L(c)``
    for LDS components, and ``L(c) <= pipeline_depth`` by definition, with equality
    reachable. Truncating older history is free regardless, since the returned count
    depends only on the suffix after the newest required group.
    """
    out = []
    for step in range(read_step - pipeline_depth(tc) + 1, read_step + 1):
        step_phase = phase + step - read_step
        if drain:
            schedule = (
                _groups(tc, step - 1, True, step_phase)
                if step > 0
                else _groups(tc, None, phase=step_phase)
            )
        else:
            schedule = _groups(
                tc, None if stage is None else step, phase=step_phase
            )
        for pos, groups in enumerate(schedule):
            # This step's own slots from `slot` onward commit after the wait.
            if step == read_step and (slot is None or pos >= slot):
                break
            out.extend(
                tuple((step, component, tile) for component, tile in group)
                for group in groups
            )
        if drain and step == 0:
            # The epilogue commits between the last main step and the first drain
            # iteration, so it lands here and nowhere else.
            out.extend(() for _ in range(epilogue_groups))
    return tuple(out)


@gluon.constexpr_function
def _wait_count(committed, required):
    """Groups left outstanding after retiring the newest required producer.

    ``wait_group(n)`` blocks until at most ``n`` groups remain outstanding, so with
    ``j`` the newest index holding anything required, ``len - j - 1`` retires through
    ``j`` and no further. One more would leave ``j`` in flight; one fewer would drain
    newer work for nothing.

    Sound even though retirement is measured against the outstanding set rather than
    this timeline: retirement is oldest-first, so outstanding is always a suffix of
    committed. If ``j`` is still in flight the suffix reaches it; if an earlier wait
    already retired it the instruction is a no-op.
    """
    for token in required:
        assert sum(token in group for group in committed) == 1, (
            f"required producer {token} must be committed exactly once before its "
            f"read; found {sum(token in group for group in committed)}"
        )
    latest = max(
        i for i, group in enumerate(committed) if any(t in group for t in required)
    )
    return len(committed) - latest - 1


@gluon.constexpr_function
def _wait(tc, stage, slot, drain=False, epilogue_groups=0, phase=None):
    """Immediate for the ``wait_group`` before this read, or ``None`` for no wait.

    Four independent steps: where we are, what the reads demand, what has committed,
    how far back the newest demand sits.
    """
    read_step = _read_step(tc, stage, drain)
    phase = read_step if phase is None else phase
    # Memoized on the resolved input surface rather than the configuration object.
    # This is the most expensive constexpr in the kernel -- it walks pipeline_depth
    # steps of commit groups -- and the emitter asks for it once per slot per step,
    # which repeats across the unrolled body.
    key = (
        schedule_spec(tc), stage, slot, bool(drain), _v(epilogue_groups), phase
    )
    if key in _schedule_cache.WAIT:
        return _schedule_cache.WAIT[key]
    required = _required_producers(tc, read_step, slot, phase)
    if not required:
        count = None
    else:
        committed = _committed_before(
            tc, stage, read_step, slot, drain, epilogue_groups, phase
        )
        count = _wait_count(committed, required)
    if len(_schedule_cache.WAIT) < _schedule_cache.LIMIT:
        # Subscript assignment only. Any callable reachable from a traced body is
        # rejected by Triton's dependency finder -- including `WAIT.clear` -- so the
        # table stops growing at the bound instead of evicting, and `clear()` is for
        # callers outside tracing.
        _schedule_cache.WAIT[key] = count
    return count


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


@gluon.constexpr_function
def validate_pipeline(tc, K):
    """Validate the live component rings and runtime-loop lower bound."""
    tc.validate_buffer_counts()
    num_k = tc.num_k_tiles(K)
    depth = pipeline_depth(tc)
    peeled = _pipeline_peeled(tc)
    unroll = pipeline_unroll(tc)
    assert num_k >= depth + peeled + unroll, (
        f"NUM_K ({num_k}) must be at least PIPELINE_DEPTH ({depth}) "
        f"+ PEELED ({peeled}) + UNROLL ({unroll}) = {depth + peeled + unroll}"
    )
    return True


@gluon.constexpr_function
def _scale_soff_steps(tc, component, offset_step):
    """Advance from the first scale producer in an unrolled body."""
    ratio = _component_ratio(tc, component)
    start = _pipeline_peeled(tc) + 1
    producer_phase = _producer_phase(tc, component)
    first_producer = (start - producer_phase + ratio - 1) // ratio
    producer = (start + offset_step - producer_phase) // ratio
    return (producer - first_producer) * ratio


@gluon.constexpr_function
def _register_index(tc, component, phase, ki, in_loop, fill=False):
    """Static register-bank index derived from producer-to-selection timing."""
    if not _present(tc, component):
        return 0
    ratio = _component_ratio(tc, component)
    if in_loop:
        origin = _pipeline_peeled(tc) + 1
        current = origin + ki
    else:
        # Finite regions rotate their tuples after each step, so logical bank zero
        # always denotes the value selected at ``phase``.
        origin = phase
        current = phase
    target = current + (_fill_span(tc, component) - 1 if fill else 0)
    return (target // ratio - origin // ratio) % _component_depth(tc, component)
