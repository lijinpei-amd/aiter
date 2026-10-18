# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The pipelined K loop of one build, evaluated once as a :class:`Schedule`.

``Schedule(tuning_cfg, layout, K)`` derives the whole schedule in its constructor and
holds the results in its fields. ``_pipeline.py`` expands the loop from it alone:

* how many prologue, warm-up, peeled, loop, remainder and drain steps there are, and
  each step's K phase, unroll position and region;
* in each step and slot, which non-K slot of each component is copied and which is
  selected, and whether the selection reads in the memory or the MFMA region;
* where the async commits go, and the immediate of every async wait.

The public surface is :class:`Schedule`: its constructor, methods and fields.
Everything else here is internal derivation detail that no other module imports.

Vocabulary, per ``agents/CONCEPTS.md``. The *tile* is the ``BLOCK_M x BLOCK_N x
BLOCK_K`` block; one pipeline *step* advances it by one ``BLOCK_K``. A *slot* is one
of its ``num_m_slots x num_n_slots`` parts, numbered by ``_types._slot_index``. A
*component* is ``(operand, is_scale)``, one of ``_types.COMPONENTS``. A component is
split along its operand's non-K axis only -- A by M slot, B by N slot -- and that
per-operand index is a copy's *non-K slot*.

Each component has a ring of ``D`` buffers and a scale K-step ratio ``R`` (1 for a
payload), so a value lives ``L = (D - 1) * R + 1`` steps. An LDS producer is selected
``F = L`` steps after it is issued; a direct-register producer skips the DS read and
is selected after ``F = L - 1``.

Waits are computed in two passes. The first builds the commit plan: for every emitted
step, the commit groups each slot closes, as sets of ``(component, non-K slot)``
copies. The second replays the plan in execution order for this build's K, stamping
each copy with the step that issued it, and reads every wait immediate off the
replayed commit FIFO (:func:`_replay`).
"""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import maybe_unwrap as _v
from ._lang import strip_annotate
from ._types import COMPONENTS, A, _slot_index

__all__ = ["Schedule"]

# Step kinds, in program order. A step is one emitted `_step` call, or for the
# prologue one fill-only expansion.
_PROLOGUE = 0  # fills only, before anything is selected
_WARMUP = 1  # step 0: selects the first K tile, issues no MFMA
_PEELED = 2  # straight-line main steps whose waits differ from the loop body's
_LOOP = 3  # the runtime loop body
_REMAINDER = 4  # the main steps a whole number of loop bodies leaves over
_DRAIN = 5  # after the main steps; only components with spare depth still fill


@gluon.constexpr_function
def _buffer_load_order(nm, nn):
    """Payload copies as ``(is_a, non-K slot)`` in slot order."""
    if nm == 2 and nn == 2:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = []
    for i in range(max(nm, nn)):
        if i < nm:
            out.append((1, i))
        if i < nn:
            out.append((0, i))
    return out


@gluon.constexpr_function
def _payload_fill_slots(nm, nn):
    """Slot that copies each payload non-K slot, as ``(A mapping, B mapping)``.

    Two policies. With both axes split, the interleaved order (and its 2x2 special
    case) gives each slot one payload copy, which is what the tuned kernels were
    measured with and must not move.

    With a degenerate axis there are fewer slots than copies -- ``nm + nn`` copies
    into ``nm * nn`` slots -- so one slot copies both operands. Fall back to
    axis-major, A's M slot ``t`` at ``(mi=t, ni=0)`` and B's N slot ``t`` at
    ``(mi=0, ni=t)``, which is total and injective per component for any
    ``nm, nn >= 1``. It is restricted to the case the interleave cannot express,
    because applying it to a split tile would move that tile's copies.
    """
    if nm > 1 and nn > 1:
        order = _buffer_load_order(nm, nn)
        return (
            tuple(order.index((1, t)) for t in range(nm)),
            tuple(order.index((0, t)) for t in range(nn)),
        )
    return (
        tuple(_slot_index(t, 0, nm, nn) for t in range(nm)),
        tuple(_slot_index(0, t, nm, nn) for t in range(nn)),
    )


@gluon.constexpr_function
def _first_non_k_slot(mapping, slot, ratio):
    """First non-K slot ``mapping`` places in ``slot``, or ``None``.

    Only every ``ratio``-th non-K slot counts: a scale shared across non-K slots is
    copied and selected once, at its first non-K slot.
    """
    for t, mapped in enumerate(mapping):
        if mapped == slot and t % ratio == 0:
            return t
    return None


@gluon.constexpr_function
def _step_plan(m, kind, stage, phase):
    """What one step copies, commits and selects, as the async FIFO sees it.

    ``stage`` places the step for the staggered start and stop: an absolute step
    before the loop (``None`` inside it, where every component fills), the zero-based
    iteration in the drain. ``phase`` is its K step, which decides the scale copies
    and selections. Returns ``(active, loads, issued, commits, demand)``:

    * ``active`` -- whether each component fills here at all. Prologue filling starts
      at ``1 - F(c)``, so a component with a shorter fill span starts later. The
      drain keeps filling while ``j < depth - F(c)``: a component whose values live
      the full pipeline depth has nothing left to fetch, a shallower one does.
    * ``loads`` -- ``active``, and the K step is the component's producer phase.
    * ``issued`` -- per slot, the non-K slot each component copies, register copies
      included.
    * ``commits`` -- per slot, the groups it closes, of LDS copies only. Per-slot and
      per-tile schemes commit even an empty group: it still occupies a FIFO position.
    * ``demand`` -- per slot, the LDS copies its selections read. Register components
      have no commit group; the compiler tracks their loads itself.
    """
    present, fill_span, ratio_k = m["present"], m["fill_span"], m["ratio_k"]
    if kind == _DRAIN:
        active = tuple(p and stage < m["depth"] - f for p, f in zip(present, fill_span))
    else:
        active = tuple(
            p and (stage is None or stage + f - 1 >= 0)
            for p, f in zip(present, fill_span)
        )
    loads = tuple(
        a and phase % r == (1 - f) % r for a, r, f in zip(active, ratio_k, fill_span)
    )
    issued = tuple(
        tuple(t if t is not None and load else None for t, load in zip(owned, loads))
        for owned in m["owned"]
    )
    per_copy, per_slot = m["commit"][0], m["commit"][1]
    commits, whole = [], ()
    for slot, copies in enumerate(issued):
        ops = tuple(
            (ci, t) for ci, t in enumerate(copies) if t is not None and m["via_lds"][ci]
        )
        if per_copy:
            groups = tuple((op,) for op in ops)
        elif per_slot:
            groups = (ops,)
        else:
            whole += ops
            groups = (whole,) if slot == len(issued) - 1 else ()
        commits.append(groups)
    if kind == _PROLOGUE:
        demand = ()
    else:
        selects = tuple(p and phase % r == 0 for p, r in zip(present, ratio_k))
        demand = tuple(
            tuple(
                (ci, t)
                for ci, t in enumerate(selected)
                if t is not None and m["via_lds"][ci] and selects[ci]
            )
            for selected in m["selected"]
        )
    return active, loads, issued, tuple(commits), demand


@gluon.constexpr_function
def _immediate(position, count, step, need, fill_span):
    """Groups that may stay outstanding once every copy in ``need`` has landed."""
    if not need:
        return None
    latest = -1
    for ci, t in need:
        copy = (step - fill_span[ci] + 1, ci, t)
        assert (
            copy in position
        ), f"copy {copy} is selected at step {step} but was never committed before it"
        latest = max(latest, position[copy])
    return count - latest - 1


@gluon.constexpr_function
def _smaller(a, b):
    """The tighter of two wait immediates, ``None`` meaning no wait."""
    return b if a is None else a if b is None else min(a, b)


@gluon.constexpr_function
def _replay(orders, fill_span, per_tile, slots):
    """Wait immediates along execution orders, as ``{key: (tile, slot waits)}``.

    Each order holds one ``(key, step, commits, demand)`` per executed step, in
    order: ``key`` names the emitted step (``None`` for the epilogue's groups),
    ``step`` is the absolute K step it executes as, ``commits``/``demand`` its plan.
    One emitted step can execute many times -- the loop body does, and a remainder
    step appears in more than one order -- and its single immediate has to hold for
    every execution, so each key keeps its smallest.

    A copy is ``(step, component, non-K slot)``: the same copy repeats every ring
    period, so the component and slot alone would match another step's value. A
    selection at step ``s`` reads the copy issued at ``s - F(c) + 1``.

    ``wait_group(n)`` blocks until at most ``n`` groups are outstanding. With ``j`` the
    newest group holding a required copy, ``count - j - 1`` retires through ``j`` and
    no further: one more leaves ``j`` in flight, one fewer drains newer copies for
    nothing. It holds against the outstanding set as well, which is always a suffix
    of the order because groups retire oldest-first; if an earlier wait already
    retired ``j``, the instruction is a no-op.

    A per-tile scheme waits once at the head of the step, for every slot's reads. The
    others wait before each slot, after the slots before it have committed.
    """
    out = {}
    for order in orders:
        position, count = {}, 0
        for key, step, commits, demand in order:
            measure = key is not None and bool(demand)
            if measure:
                tile, slot_waits = out.get(key, (None, (None,) * slots))
                slot_waits = list(slot_waits)
                if per_tile:
                    need = tuple(copy for copies in demand for copy in copies)
                    now = _immediate(position, count, step, need, fill_span)
                    tile = _smaller(tile, now)
            for slot, groups in enumerate(commits):
                if measure and not per_tile:
                    now = _immediate(position, count, step, demand[slot], fill_span)
                    slot_waits[slot] = _smaller(slot_waits[slot], now)
                for group in groups:
                    for ci, t in group:
                        copy = (step, ci, t)
                        assert copy not in position, f"copy {copy} committed twice"
                        position[copy] = count
                    count += 1
            if measure:
                out[key] = (tile, tuple(slot_waits))
    return out


@aggregate
@strip_annotate
class Schedule:
    """The expanded K loop of one build: its steps, copies, commits and waits.

    Built once as ``Schedule(tuning_cfg, layout, K)``, on the host and in jit code
    alike. A value of the whole build is a public field, read directly. Per-component
    values are ``COMPONENTS``-aligned tuples, read through accessors that take a
    component ``(operand, is_scale)``.

    Every emitted step has an index, ``step``. The steps, in program order:

    * ``prologue_step(r)``, ``r`` in ``[1 - depth, 0)``: fills only.
    * ``warmup_step()``: K step 0, selects without an MFMA.
    * ``peeled_step(x)``, ``x`` in ``[0, peeled)``: K step ``x + 1``.
    * ``loop_step(u)``, ``u`` in ``[0, unroll)``: the runtime loop body.
    * ``remainder_step(u)``: the steps a whole number of loop bodies leaves over --
      ``unroll - 1`` guarded static steps, or with ``static_remainder`` false one
      step in a runtime loop.
    * ``drain_step(p, j)``, ``j`` in ``[0, depth - 1)``: one copy of the drain per
      ring phase ``p`` with ``static_drain``, else one (``p == 0``).

    A config that cannot be scheduled -- a fill ownership that is not a mapping, or a
    K too short for the loop -- still gets its sizing, but no steps. ``_validate``
    rejects it.
    """

    # -- per component -----------------------------------------------------------------
    _present: gl.constexpr
    _via_lds: gl.constexpr
    _ring_depth: gl.constexpr
    _ratio_k: gl.constexpr
    _fill_span: gl.constexpr
    _fill_slots: gl.constexpr
    _read_slots: gl.constexpr
    _read_in_mfma: gl.constexpr
    # -- per slot ----------------------------------------------------------------------
    _owned: gl.constexpr
    _selected: gl.constexpr
    # -- sizing ------------------------------------------------------------------------
    #: K tiles in the strip: K // BLOCK_K.
    num_k: gl.constexpr
    #: Pipeline depth: the largest live span of a present component, in steps.
    depth: gl.constexpr
    #: Steps per loop body: K_UNROLL widened to a whole ring restoration period.
    unroll: gl.constexpr
    #: Main steps emitted straight-line between the warm-up step and the loop.
    peeled: gl.constexpr
    #: Steps after which every present ring is back at slot zero.
    ring_period: gl.constexpr
    #: ``ring_period`` over direct-register rings only; 1 when none rotates.
    register_period: gl.constexpr
    #: Steps after which the issued async-copy pattern repeats.
    async_period: gl.constexpr
    #: Whether copy ownership maps each component's non-K slots onto slots.
    fill_slots_valid: gl.constexpr
    # -- policy ------------------------------------------------------------------------
    #: Commit after every LDS copy.
    commit_per_copy: gl.constexpr
    #: Commit at the end of every slot's fill, even an empty one.
    commit_per_slot: gl.constexpr
    #: Commit once per step, after the last slot's fill, even an empty one.
    commit_per_tile: gl.constexpr
    #: Whether a step's tile commit follows the last slot's MFMAs. Otherwise it closes
    #: the last slot's memory region, inside the warp-pipeline stage. Prologue steps
    #: have no MFMA and commit after their last fill either way.
    tile_commit_after_mfma: gl.constexpr
    _soff_unroll: gl.constexpr
    #: Whether the remainder is ``unroll - 1`` guarded static steps, not a loop.
    static_remainder: gl.constexpr
    #: Whether the drain is emitted once per ring phase, with static indices.
    static_drain: gl.constexpr
    #: Commit groups the epilogue issues between the main steps and the drain.
    epilogue_groups: gl.constexpr
    # -- per step ----------------------------------------------------------------------
    _steps: gl.constexpr
    _active: gl.constexpr
    _loads: gl.constexpr
    _issued: gl.constexpr
    _commits: gl.constexpr
    _waits: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, tuning_cfg, layout, K, epilogue_groups=None):
        """Evaluate the schedule of one build.

        ``epilogue_groups`` is the number of commit groups the epilogue issues between
        the last main step and the drain. It is derived from the build -- one group
        when the epilogue stages bias or gammas through LDS -- and the argument only
        exists for tests of the drain accounting.
        """
        # Field reads here stay `_v`: the schedule tests pass a duck-typed fake with
        # raw fields as the configs and the layout.
        tc = tuning_cfg
        fc = tc.func_cfg
        nm, nn = _v(layout.num_m_slots), _v(layout.num_n_slots)
        slots = nm * nn
        num_k = layout.num_k_tiles(K)

        # -- per component -------------------------------------------------------------
        present = tuple(
            not is_scale or bool(fc.has_scale(op)) for op, is_scale in COMPONENTS
        )
        via_lds = tuple(bool(layout.component_via_lds(c)) for c in COMPONENTS)
        ring_depth = tuple(tc.num_buffers(op, is_scale) for op, is_scale in COMPONENTS)
        ratio_k = tuple(
            layout.scale_slot_k(op) // _v(layout.scale_block_k) if is_scale else 1
            for op, is_scale in COMPONENTS
        )
        ratio_non_k = tuple(
            layout.scale_slot_nonk(op) // layout.slot_nonk(op) if is_scale else 1
            for op, is_scale in COMPONENTS
        )
        live_span = tuple((d - 1) * r + 1 for d, r in zip(ring_depth, ratio_k))
        fill_span = tuple(s if lds else s - 1 for s, lds in zip(live_span, via_lds))

        # -- sizing --------------------------------------------------------------------
        # Pipeline depth: the largest live span. The ring restoration period is when
        # every ring is back at slot zero -- a ring repeats every D * R steps, not
        # every live span. Only register rings need a static index, so their own
        # period decides whether a body can be unrolled; LDS rings take runtime
        # indices. The async period is when the issued copy pattern repeats, which
        # depends only on the scale K phase.
        depth, ring_period, register_period, async_period = 0, 1, 1, 1
        for ci in range(len(COMPONENTS)):
            if present[ci]:
                depth = max(depth, live_span[ci])
                ring_period = math.lcm(ring_period, ring_depth[ci] * ratio_k[ci])
                if via_lds[ci]:
                    async_period = math.lcm(async_period, ratio_k[ci])
                else:
                    register_period = math.lcm(
                        register_period, ring_depth[ci] * ratio_k[ci]
                    )
        k_unroll = _v(tc.K_UNROLL)
        assert k_unroll >= 1, "K_UNROLL must be at least 1"
        # Widened to a whole number of every ring period, so every complete body
        # sits at a fixed ring phase and register tuples get static indices.
        unroll = math.lcm(k_unroll, ring_period)

        # -- copy ownership and selection ----------------------------------------------
        # SCALE_FILL_MID is consumed here and nowhere else, and only on the 2x2 split:
        # on any other geometry it is inert.
        a_slots, b_slots = _payload_fill_slots(nm, nn)
        fill_mid = bool(_v(tc.SCALE_FILL_MID)) and nm == 2 and nn == 2
        fill_slots = tuple(
            (1, 2) if is_scale and fill_mid else (a_slots if op == A else b_slots)
            for op, is_scale in COMPONENTS
        )
        # Every mapped slot exists and no component copies two non-K slots in one
        # slot: a slot carries one optional copy per component. Different components
        # sharing a slot is fine and expected.
        fill_slots_valid = all(
            len(mapping) == (nm if op == A else nn)
            and all(0 <= slot < slots for slot in mapping)
            and len(set(mapping)) == len(mapping)
            for (op, _), mapping in zip(COMPONENTS, fill_slots)
        )
        owned = tuple(
            tuple(
                (
                    _first_non_k_slot(fill_slots[ci], slot, ratio_non_k[ci])
                    if present[ci]
                    else None
                )
                for ci in range(len(COMPONENTS))
            )
            for slot in range(slots)
        )
        # A component is selected where its operand's payload is copied, except that
        # a register producer with F = 1 is selected in the step it is issued, so its
        # selection cannot precede its copy.
        read_slots, read_in_mfma = [], []
        for ci, (op, is_scale) in enumerate(COMPONENTS):
            payload_slots = a_slots if op == A else b_slots
            reads, in_mfma = [], []
            for t in range(len(payload_slots)):
                read = payload_slots[t]
                same_step = not via_lds[ci] and fill_span[ci] == 1
                if same_step and fill_slots_valid and fill_slots[ci][t] > read:
                    read = fill_slots[ci][t]
                reads.append(read)
                # A same-step register selection in its own fill slot has to read in
                # the MFMA region, after the fill.
                configured = bool(tc.ds_read_in_mfma(op, is_scale))
                in_mfma.append(
                    configured
                    or (same_step and fill_slots_valid and read == fill_slots[ci][t])
                )
            read_slots.append(tuple(reads))
            read_in_mfma.append(tuple(in_mfma))
        selected = tuple(
            tuple(
                (
                    _first_non_k_slot(read_slots[ci], slot, ratio_non_k[ci])
                    if present[ci]
                    else None
                )
                for ci in range(len(COMPONENTS))
            )
            for slot in range(slots)
        )

        # -- policy --------------------------------------------------------------------
        commit = (
            bool(tc.commit_per_op()),
            bool(tc.commit_per_slot()),
            bool(tc.commit_per_stage()),
            bool(tc.commit_per_stage_whole()),
        )
        static_remainder = bool(_v(tc.UNROLL_EPILOGUE))
        # Keep short LDS rings in immediate offsets; register queues and quantized
        # epilogues need the smaller runtime-indexed drain to avoid allocation
        # pressure.
        static_drain = (
            ring_period <= 3
            and unroll % ring_period == 0
            and register_period == 1
            and _v(fc.output_quant) is None
        )
        if epilogue_groups is None:
            epilogue_groups = int(
                bool(_v(layout.epilogue_inputs_via_lds))
                and bool(_v(fc.has_gammas) or _v(fc.has_bias))
            )
        model = {
            "present": present,
            "via_lds": via_lds,
            "ratio_k": ratio_k,
            "fill_span": fill_span,
            "depth": depth,
            "owned": owned,
            "selected": selected,
            "commit": commit,
        }

        # -- peeling -------------------------------------------------------------------
        # The loop body has one wait immediate per slot for all its executions, so it
        # takes the smallest. Before the read at step depth - 1, some selections read
        # copies issued while shallower components had not started filling, which
        # can need a smaller immediate than the loop does. Peel those steps instead
        # of letting them tighten every loop iteration. The first main step is always
        # peeled to retain its visible zero accumulator.
        peeled = 1
        if fill_slots_valid:
            settled = 3 * depth + async_period
            warm = [
                (None, r) + _step_plan(model, _PROLOGUE, r, r)[3:]
                for r in range(1 - depth, 0)
            ]
            warm += [
                (s, s) + _step_plan(model, _PEELED if s else _WARMUP, s, s)[3:]
                for s in range(settled + 1)
            ]
            waits = _replay([warm], fill_span, commit[2], slots)
            for s in range(1, depth - 1):
                steady = s + async_period * -(-(2 * depth - s) // async_period)
                if waits.get(s) != waits.get(steady):
                    peeled = s

        self._present = gl.constexpr(present)
        self._via_lds = gl.constexpr(via_lds)
        self._ring_depth = gl.constexpr(ring_depth)
        self._ratio_k = gl.constexpr(ratio_k)
        self._fill_span = gl.constexpr(fill_span)
        self._fill_slots = gl.constexpr(fill_slots)
        self._read_slots = gl.constexpr(tuple(read_slots))
        self._read_in_mfma = gl.constexpr(tuple(read_in_mfma))
        self._owned = gl.constexpr(owned)
        self._selected = gl.constexpr(selected)
        self.num_k = gl.constexpr(num_k)
        self.depth = gl.constexpr(depth)
        self.unroll = gl.constexpr(unroll)
        self.peeled = gl.constexpr(peeled)
        self.ring_period = gl.constexpr(ring_period)
        self.register_period = gl.constexpr(register_period)
        self.async_period = gl.constexpr(async_period)
        self.fill_slots_valid = gl.constexpr(fill_slots_valid)
        self.commit_per_copy = gl.constexpr(commit[0])
        self.commit_per_slot = gl.constexpr(commit[1])
        self.commit_per_tile = gl.constexpr(commit[2])
        self.tile_commit_after_mfma = gl.constexpr(commit[3])
        self._soff_unroll = gl.constexpr(bool(_v(tc.SOFF_UNROLL)))
        self.static_remainder = gl.constexpr(static_remainder)
        self.static_drain = gl.constexpr(static_drain)
        self.epilogue_groups = gl.constexpr(epilogue_groups)

        main = num_k - depth
        if not fill_slots_valid or main < peeled + unroll:
            self._steps = gl.constexpr(None)
            self._active = gl.constexpr(None)
            self._loads = gl.constexpr(None)
            self._issued = gl.constexpr(None)
            self._commits = gl.constexpr(None)
            self._waits = gl.constexpr(None)
            return

        # -- the emitted steps ---------------------------------------------------------
        # (kind, stage, phase, ki, in_loop, static_phase, dot). `ki` is the position in
        # an unrolled body, `static_phase` that a finite region still has a static
        # ring phase.
        drain_phases = ring_period if static_drain else 1
        steps = [(_PROLOGUE, r, r, 0, False, False, False) for r in range(1 - depth, 0)]
        steps.append((_WARMUP, 0, 0, 0, False, False, False))
        steps += [(_PEELED, x + 1, x + 1, 0, False, False, True) for x in range(peeled)]
        steps += [
            (_LOOP, None, peeled + u + 1, u, True, False, True) for u in range(unroll)
        ]
        if static_remainder:
            steps += [
                (_REMAINDER, None, peeled + u + 1, u, False, True, True)
                for u in range(unroll - 1)
            ]
        else:
            # Its phase is never read: _validate requires every scale to advance once
            # per step without a static remainder.
            steps.append((_REMAINDER, None, 0, 0, False, False, True))
        steps += [
            (
                _DRAIN,
                j,
                num_k - depth + j + 1,
                p + j - peeled if static_drain else 0,
                False,
                static_drain,
                True,
            )
            for p in range(drain_phases)
            for j in range(depth - 1)
        ]
        plans = [_step_plan(model, s[0], s[1], s[2]) for s in steps]

        # -- the wait plan -------------------------------------------------------------
        # Replay this K's execution order. A static remainder step past this K's
        # remainder never executes, but is still emitted behind a runtime guard, so a
        # second order runs the longest remainder after the same loop. A drain copy
        # for another ring phase never executes either; it repeats the executed copy.
        warmup = depth - 1
        loop = warmup + 1 + peeled
        remainder = loop + unroll
        drain = len(steps) - drain_phases * (depth - 1)
        remaining = main - peeled
        iters, rem = remaining // unroll, remaining % unroll

        def entry(index, step):
            return (index, step, plans[index][3], plans[index][4])

        head = [entry(i, steps[i][1]) for i in range(warmup + 1 + peeled)]
        head += [
            entry(loop + u, peeled + it * unroll + u + 1)
            for it in range(iters)
            for u in range(unroll)
        ]
        base = peeled + iters * unroll

        def tail(count):
            return [
                entry(remainder + (u if static_remainder else 0), base + u + 1)
                for u in range(count)
            ]

        phase = main % ring_period if static_drain else 0
        executed = head + tail(rem)
        executed.append((None, None, (((),) * epilogue_groups,), ()))
        executed += [
            entry(drain + phase * (depth - 1) + j, main + j + 1)
            for j in range(depth - 1)
        ]
        orders = [executed]
        longest = unroll - 1 if static_remainder else 1
        if rem < longest:
            orders.append(head + tail(longest))
        waits = _replay(orders, fill_span, commit[2], slots)
        for p in range(drain_phases):
            for j in range(depth - 1):
                executed_copy = drain + phase * (depth - 1) + j
                if executed_copy in waits:
                    waits[drain + p * (depth - 1) + j] = waits[executed_copy]
        idle = (None, (None,) * slots)

        self._steps = gl.constexpr(tuple(steps))
        self._active = gl.constexpr(tuple(plan[0] for plan in plans))
        self._loads = gl.constexpr(tuple(plan[1] for plan in plans))
        self._issued = gl.constexpr(tuple(plan[2] for plan in plans))
        self._commits = gl.constexpr(tuple(plan[3] for plan in plans))
        self._waits = gl.constexpr(
            tuple(waits.get(index, idle) for index in range(len(steps)))
        )

    # -- the emitted steps -------------------------------------------------------------

    @gluon.constexpr_function
    def scheduled(self):
        """Whether the steps exist: false for a config ``_validate`` rejects."""
        return self._steps.value is not None

    @gluon.constexpr_function
    def num_steps(self):
        return len(self._steps.value)

    @gluon.constexpr_function
    def prologue_step(self, r):
        """The fill-only step ``r``, for ``r`` in ``[1 - depth, 0)``."""
        depth = self.depth.value
        assert 1 - depth <= r < 0, f"no prologue step {r}"
        return r + depth - 1

    @gluon.constexpr_function
    def warmup_step(self):
        return self.depth.value - 1

    @gluon.constexpr_function
    def peeled_step(self, x):
        """The peeled step for K step ``x + 1``."""
        assert 0 <= x < self.peeled.value, f"no peeled step {x}"
        return self.depth.value + x

    @gluon.constexpr_function
    def loop_step(self, u):
        """Position ``u`` of the loop body."""
        assert 0 <= u < self.unroll.value, f"no loop step {u}"
        return self.depth.value + self.peeled.value + u

    @gluon.constexpr_function
    def remainder_step(self, u):
        """Remainder position ``u``; the runtime remainder loop has one step."""
        unroll = self.unroll.value
        count = unroll - 1 if self.static_remainder.value else 1
        assert 0 <= u < count, f"no remainder step {u}"
        return self.depth.value + self.peeled.value + unroll + u

    @gluon.constexpr_function
    def drain_step(self, p, j):
        """Drain iteration ``j`` of the copy for ring phase ``p``."""
        depth = self.depth.value
        phases = self.ring_period.value if self.static_drain.value else 1
        assert 0 <= p < phases and 0 <= j < depth - 1, f"no drain step {p, j}"
        return self.num_steps() - (phases - p) * (depth - 1) + j

    @gluon.constexpr_function
    def phase(self, step):
        """The step's K phase: its K tile modulo every period the emitter reads."""
        return self._steps.value[step][2]

    @gluon.constexpr_function
    def ki(self, step):
        """Position in an unrolled body, which fixes static ring indices."""
        return self._steps.value[step][3]

    @gluon.constexpr_function
    def in_loop(self, step):
        return self._steps.value[step][4]

    @gluon.constexpr_function
    def static_phase(self, step):
        """Whether a step outside the loop still has a static LDS ring phase."""
        return self._steps.value[step][5]

    @gluon.constexpr_function
    def dot(self, step):
        """Whether the step issues the previous K tile's MFMAs."""
        return self._steps.value[step][6]

    @gluon.constexpr_function
    def is_drain(self, step):
        return self._steps.value[step][0] == _DRAIN

    # -- per step and slot -------------------------------------------------------------

    @gluon.constexpr_function
    def active(self, step, component):
        """Whether the component fills in this step at all."""
        return self._active.value[step][_ci(component)]

    @gluon.constexpr_function
    def loads(self, step, component):
        """Whether the component issues its producer copy in this step."""
        return self._loads.value[step][_ci(component)]

    @gluon.constexpr_function
    def issued(self, step, slot):
        """Non-K slot each component copies in this slot, ``COMPONENTS``-aligned.

        ``None`` where it copies nothing. Register copies are included.
        """
        return self._issued.value[step][slot]

    @gluon.constexpr_function
    def selects(self, step, component):
        """Whether the step selects a new value of the component."""
        return self.selects_at_phase(component, self.phase(step))

    @gluon.constexpr_function
    def selects_at_phase(self, component, phase):
        ci = _ci(component)
        return self._present.value[ci] and phase % self._ratio_k.value[ci] == 0

    @gluon.constexpr_function
    def commits(self, step, slot):
        """Commit groups the slot closes in this step, of ``(component, non-K slot)``.

        This is the commit plan the waits are computed from; the emitter issues the
        same groups through :meth:`commit_per_copy` and its neighbours.
        """
        return tuple(
            tuple((COMPONENTS[ci], t) for ci, t in group)
            for group in self._commits.value[step][slot]
        )

    @gluon.constexpr_function
    def wait(self, step, slot=None):
        """Immediate of the async wait before a read, or ``None`` for no wait.

        ``slot=None`` is the tile wait at the head of the step, which per-tile commit
        schemes use; the others wait before each slot.
        """
        tile, slot_waits = self._waits.value[step]
        return tile if slot is None else slot_waits[slot]

    # -- commits -----------------------------------------------------------------------

    @gluon.constexpr_function
    def epilogue_wait(self):
        """Immediate of the wait that retires the epilogue's copies, or ``None``.

        It also retires every copy the drain issued after them.
        """
        return 0 if self.epilogue_groups.value > 0 else None

    # -- per component -----------------------------------------------------------------

    @gluon.constexpr_function
    def present(self, component):
        """Whether the component exists: a scale only for a scaled operand."""
        return self._present.value[_ci(component)]

    @gluon.constexpr_function
    def via_lds(self, component):
        return self._via_lds.value[_ci(component)]

    @gluon.constexpr_function
    def in_reg(self, component):
        return not self.via_lds(component)

    @gluon.constexpr_function
    def has_register_component(self):
        """Whether any present component uses a direct-register ring."""
        return any(
            p and not lds for p, lds in zip(self._present.value, self._via_lds.value)
        )

    @gluon.constexpr_function
    def ring_depth(self, component):
        """Buffers in the component's ring."""
        return self._ring_depth.value[_ci(component)]

    @gluon.constexpr_function
    def ratio_k(self, component):
        """Steps one value spans: the scale K-step ratio, 1 for a payload."""
        return self._ratio_k.value[_ci(component)]

    @gluon.constexpr_function
    def live_span(self, component):
        """Inclusive lifetime ``L = (D - 1) * R + 1`` in steps."""
        return (self.ring_depth(component) - 1) * self.ratio_k(component) + 1

    @gluon.constexpr_function
    def fill_span(self, component):
        """Inclusive producer-to-selection span ``F``: ``L``, ``L - 1`` in registers."""
        return self._fill_span.value[_ci(component)]

    @gluon.constexpr_function
    def producer_phase(self, component):
        """K-step residue on which the component issues its producer copy."""
        ratio = self.ratio_k(component)
        return (1 - self.fill_span(component)) % ratio

    @gluon.constexpr_function
    def fill_slot(self, component, non_k_slot):
        """Slot that copies one non-K slot of the component."""
        return self._fill_slots.value[_ci(component)][non_k_slot]

    @gluon.constexpr_function
    def read_slot(self, component, non_k_slot):
        """Slot that selects one non-K slot of the component."""
        return self._read_slots.value[_ci(component)][non_k_slot]

    @gluon.constexpr_function
    def read_in_mfma(self, component, non_k_slot):
        """Whether the selection reads in the MFMA region rather than the memory one."""
        return self._read_in_mfma.value[_ci(component)][non_k_slot]

    @gluon.constexpr_function
    def owned(self, slot):
        """Non-K slot each component copies in this slot when it loads, or ``None``."""
        return self._owned.value[slot]

    @gluon.constexpr_function
    def selected(self, slot, component):
        """Non-K slot of the component selected in this slot, or ``None``."""
        return self._selected.value[slot][_ci(component)]

    # -- ring indices ------------------------------------------------------------------

    @gluon.constexpr_function
    def register_index(self, step, component, fill=False):
        """Static register-ring index of the step's selection, or with ``fill`` copy.

        In the loop it is measured from the loop's first step, so the same body
        position always names the same register; a finite region rotates its tuples
        after each step, so there logical index zero is the value selected at its
        own phase.
        """
        if not self.present(component):
            return 0
        ratio = self.ratio_k(component)
        if self.in_loop(step):
            origin = self.peeled.value + 1
            current = origin + self.ki(step)
        else:
            origin = self.phase(step)
            current = origin
        target = current + (self.fill_span(component) - 1 if fill else 0)
        return (target // ratio - origin // ratio) % self.ring_depth(component)

    @gluon.constexpr_function
    def static_lds_index(self, step, component, fill):
        """LDS ring index the step reads (or with ``fill`` writes), if static.

        ``None`` when it depends on the runtime step: outside the loop and the static
        finite regions, or when the body does not cover a whole ring period.
        """
        if not self.present(component):
            return 0
        depth, ratio = self.ring_depth(component), self.ratio_k(component)
        if not (self.in_loop(step) or self.static_phase(step)):
            return None
        if self.unroll.value % (depth * ratio) != 0:
            return None
        advance = self.fill_span(component) - 1 if fill else 0
        return ((self.peeled.value + self.ki(step) + 1 + advance) // ratio) % depth

    @gluon.constexpr_function
    def soff_unroll(self, step):
        """Whether the step's copies offset one base pointer through ``soffset``."""
        return self.in_loop(step) and self._soff_unroll.value

    @gluon.constexpr_function
    def scale_soff_steps(self, component, offset_step):
        """Scale K steps from the body's first scale producer to this body position."""
        ratio = self.ratio_k(component)
        start = self.peeled.value + 1
        producer_phase = self.producer_phase(component)
        first_producer = (start - producer_phase + ratio - 1) // ratio
        producer = (start + offset_step - producer_phase) // ratio
        return (producer - first_producer) * ratio


@gluon.constexpr_function
def _ci(component):
    """Index of ``(operand, is_scale)`` in ``COMPONENTS``.

    A loop rather than ``COMPONENTS.index``: Triton's dependency finder rejects a
    method of a global reachable from a traced body.
    """
    key = (component[0], bool(component[1]))
    for ci, candidate in enumerate(COMPONENTS):
        if candidate == key:
            return ci
    raise AssertionError(f"{key} is not a component")
