# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import math
from enum import IntEnum

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import maybe_unwrap as _v
from ._lang import strip_annotate
from ._types import A, A_PAYLOAD, B_PAYLOAD, COMPONENTS, CompSlot

__all__ = ["Schedule", "Step", "StepKind"]


class StepKind(IntEnum):
    PROLOGUE = 0
    PEELED = 1
    LOOP = 2
    REMAINDER = 3
    DRAIN = 4


@aggregate
@strip_annotate
class Step:
    """One emitted step body: its kind and its place among that kind's bodies. Equal
    by value, so it keys dicts and sets.

    ``index`` counts the kind's bodies from 0 in emission order and never depends on
    ``num_k``: prologue step ``x`` is ``x + depth``, peeled step ``x`` is ``x``, loop
    and remainder step ``u`` is ``u``, and drain step ``j`` of drain phase ``p`` is
    ``p * (depth - 1) + j``.
    """

    #: A ``StepKind``, stored as an int.
    kind: gl.constexpr
    #: The body's place among its kind's emitted bodies, from 0.
    index: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, kind, index):
        assert index >= 0, f"step index {index} is negative"
        self.kind = gl.constexpr(int(kind))
        self.index = gl.constexpr(index)

    @gluon.constexpr_function
    def __eq__(self, other):
        return (
            type(other) is type(self)
            and other.kind.value == self.kind.value
            and other.index.value == self.index.value
        )

    @gluon.constexpr_function
    def __hash__(self):
        return hash((self.kind.value, self.index.value))

    @gluon.constexpr_function
    def __repr__(self):
        # A jit helper that takes a `Step` specializes on this string.
        return f"Step({self.kind.value}, {self.index.value})"


@gluon.constexpr_function
def _payload_buffer_load_order(nm, nn):
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
def _payload_buffer_load_pane(nm, nn, operand_index, slot):
    if nm > 1 and nn > 1:
        order = _payload_buffer_load_order(nm, nn)
        return order.index((1 if operand_index == A else 0, slot))
    return slot


@gluon.constexpr_function
def _by_pane_index(pane_index_of, comp_slots, num_panes):
    slot_of = {}
    for comp_slot, pane_index in pane_index_of.items():
        if comp_slot in comp_slots:
            slot_of.setdefault(
                (pane_index, comp_slot.component.index()), comp_slot.index.value
            )
    return tuple(
        tuple(
            slot_of.get((pane_index, component_index))
            for component_index in range(len(COMPONENTS))
        )
        for pane_index in range(num_panes)
    )


@gluon.constexpr_function
def _pane_index_of(by_pane_index, comp_slot):
    component_index = comp_slot.component.index()
    for pane_index, slots in enumerate(by_pane_index):
        if slots[component_index] == comp_slot.index.value:
            return pane_index
    return None


@gluon.constexpr_function
def _scale_q(scale_slot_span, component):
    if not component.is_scale.value:
        return 1
    return scale_slot_span[component.operand_index.value][0]


@gluon.constexpr_function
def _scale_r(scale_slot_span, component):
    if not component.is_scale.value:
        return 1
    return scale_slot_span[component.operand_index.value][1]


@gluon.constexpr_function
def _step_plan(
    layout,
    component_present,
    scale_slot_span,
    load_to_mfma_distance,
    depth,
    buffer_load_slots,
    ds_read_slots,
    commit,
    kind,
    stage,
    phase,
):
    steps_per_buffer = tuple(_scale_r(scale_slot_span, c) for c in COMPONENTS)
    if kind == StepKind.DRAIN:
        active = tuple(
            p and stage < depth - f
            for p, f in zip(component_present, load_to_mfma_distance)
        )
    else:
        active = tuple(
            p and (stage is None or stage + f >= 0)
            for p, f in zip(component_present, load_to_mfma_distance)
        )
    loads = tuple(
        a and (phase + f) % r == 0
        for a, r, f in zip(active, steps_per_buffer, load_to_mfma_distance)
    )
    issued = tuple(
        tuple(
            slot if slot is not None and load else None
            for slot, load in zip(slots, loads)
        )
        for slots in buffer_load_slots
    )
    per_copy, per_pane = commit[0], commit[1]
    commits, whole = [], ()
    for pane_index, copies in enumerate(issued):
        ops = tuple(
            (component_index, slot)
            for component_index, slot in enumerate(copies)
            if slot is not None
            and layout.component_via_lds(COMPONENTS[component_index])
        )
        # A group closes only when it holds a copy.
        if per_copy:
            groups = tuple((op,) for op in ops)
        elif per_pane:
            groups = (ops,) if ops else ()
        else:
            whole += ops
            groups = (whole,) if pane_index == len(issued) - 1 and whole else ()
        commits.append(groups)
    if kind == StepKind.PROLOGUE and phase < -1:
        # Step `phase` DS-reads K tile `phase + 1`: in the prologue, only step -1.
        demand = ()
    else:
        ds_reads = tuple(
            p and (phase + 1) % r == 0
            for p, r in zip(component_present, steps_per_buffer)
        )
        demand = tuple(
            tuple(
                (component_index, slot)
                for component_index, slot in enumerate(slots)
                if slot is not None
                and layout.component_via_lds(COMPONENTS[component_index])
                and ds_reads[component_index]
            )
            for slots in ds_read_slots
        )
    return active, loads, issued, tuple(commits), demand


@gluon.constexpr_function
def _compute_wait_plan(
    orders, load_to_mfma_distance, per_tile, num_panes, windows=False
):
    # A wait's immediate counts the groups committed after the newest one it needs.
    # With `windows`, each wait also gets its window: the loads it leaves in flight,
    # every copy (LDS or register) issued after that group's last LDS copy, in emission
    # order. That models what LLVM counts into the vmcnt (PIPELINE.md, Wait plan).
    def _wait(position, ends, stream, count, step, need):
        if not need:
            return None
        latest = -1
        for component_index, slot in need:
            # Step `step` DS-reads K tile `step + 1`, copied F steps before its MFMAs.
            copy = (
                step + 1 - load_to_mfma_distance[component_index],
                component_index,
                slot,
            )
            assert (
                copy in position
            ), f"copy {copy} is DS-read at step {step} but was never committed before it"
            latest = max(latest, position[copy])
        if not windows:
            return count - latest - 1
        return count - latest - 1, tuple(sorted(stream[ends[latest] + 1 :]))

    def _smaller(a, b):
        return b if a is None else a if b is None else min(a, b)

    out = {}
    for order in orders:
        # `stream` holds every copy in issue order; `ends[g]` is the stream position
        # of group g's last LDS copy.
        position, ends, stream, count = {}, [], [], 0
        for key, step, issued, commits, demand in order:
            measure = key is not None and bool(demand)
            if measure:
                tile, pane_waits = out.get(key, (None, (None,) * num_panes))
                pane_waits = list(pane_waits)
                if per_tile:
                    need = tuple(copy for copies in demand for copy in copies)
                    now = _wait(position, ends, stream, count, step, need)
                    tile = _smaller(tile, now)
            where = {}
            for pane_index, groups in enumerate(commits):
                if measure and not per_tile:
                    now = _wait(position, ends, stream, count, step, demand[pane_index])
                    pane_waits[pane_index] = _smaller(pane_waits[pane_index], now)
                for component_index, slot in enumerate(issued[pane_index]):
                    if slot is not None:
                        where[(component_index, slot)] = len(stream)
                        stream.append((component_index, slot))
                for group in groups:
                    for component_index, slot in group:
                        copy = (step, component_index, slot)
                        assert copy not in position, f"copy {copy} committed twice"
                        position[copy] = count
                    ends.append(max((where[op] for op in group), default=-1))
                    count += 1
            if measure:
                out[key] = (tile, tuple(pane_waits))
    return out


@aggregate
@strip_annotate
class Schedule:

    _component_present: gl.constexpr
    _num_buffers: gl.constexpr
    _scale_slot_span: gl.constexpr
    _load_to_mfma_distance: gl.constexpr
    _ds_read_in_mfma: gl.constexpr
    _buffer_load_slots: gl.constexpr
    _ds_read_slots: gl.constexpr
    num_k: gl.constexpr
    depth: gl.constexpr
    unroll: gl.constexpr
    peeled: gl.constexpr
    buffer_period: gl.constexpr
    reg_buffer_period: gl.constexpr
    commit_per_copy: gl.constexpr
    commit_per_pane: gl.constexpr
    commit_per_tile: gl.constexpr
    tile_commit_after_mfma: gl.constexpr
    _soff_unroll: gl.constexpr
    static_remainder: gl.constexpr
    static_drain: gl.constexpr
    epilogue_groups: gl.constexpr
    _step_counts: gl.constexpr
    _steps: gl.constexpr
    _active: gl.constexpr
    _loads: gl.constexpr
    _commits: gl.constexpr
    _waits: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, tuning_cfg, layout, K, epilogue_groups=None):
        tc = tuning_cfg
        fc = tc.func_cfg
        nm, nn = _v(layout.num_m_slots), _v(layout.num_n_slots)
        num_panes = nm * nn
        num_k = layout.num_k_tiles(K)

        component_present = tuple(
            not c.is_scale.value or bool(fc.has_scale(c.operand_index.value))
            for c in COMPONENTS
        )
        num_buffers = tuple(
            tc.num_buffers(c.operand_index.value, c.is_scale.value) for c in COMPONENTS
        )
        scale_slot_span = _v(layout.scale_slot_span)
        buffer_live_span = tuple(
            (d - 1) * _scale_r(scale_slot_span, c) + 1
            for d, c in zip(num_buffers, COMPONENTS)
        )
        load_to_mfma_distance = tuple(
            s if layout.component_via_lds(c) else s - 1
            for s, c in zip(buffer_live_span, COMPONENTS)
        )

        depth, buffer_period, reg_buffer_period, copy_period = 0, 1, 1, 1
        for component_index, c in enumerate(COMPONENTS):
            if component_present[component_index]:
                r = _scale_r(scale_slot_span, c)
                depth = max(depth, buffer_live_span[component_index])
                buffer_period = math.lcm(
                    buffer_period, num_buffers[component_index] * r
                )
                copy_period = math.lcm(copy_period, r)
                if not layout.component_via_lds(c):
                    reg_buffer_period = math.lcm(
                        reg_buffer_period, num_buffers[component_index] * r
                    )
        k_unroll = _v(tc.K_UNROLL)
        assert k_unroll >= 1, "K_UNROLL must be at least 1"
        unroll = math.lcm(k_unroll, buffer_period)

        fill_mid = bool(_v(tc.SCALE_FILL_MID)) and nm == 2 and nn == 2
        slots = {
            CompSlot(c, slot)
            for c in COMPONENTS
            for slot in range(nm if c.operand_index.value == A else nn)
        }
        first_slots = {
            comp_slot
            for comp_slot in slots
            if component_present[comp_slot.component.index()]
            and comp_slot.index.value % _scale_q(scale_slot_span, comp_slot.component)
            == 0
        }
        # The pane index that issues each component slot's buffer load.
        buffer_load_pane = {
            CompSlot(c, slot): (
                (1, 2)[slot]
                if fill_mid and c.is_scale.value
                else _payload_buffer_load_pane(nm, nn, c.operand_index.value, slot)
            )
            for c in COMPONENTS
            for slot in range(nm if c.operand_index.value == A else nn)
        }
        # Every mapped pane exists, and no pane copies two slots of one component: a
        # pane carries one optional copy per component. Components may share a pane.
        pane_copies = {
            (comp_slot.component, pane_index)
            for comp_slot, pane_index in buffer_load_pane.items()
        }
        assert len(pane_copies) == len(buffer_load_pane) and all(
            0 <= pane_index < num_panes for _, pane_index in pane_copies
        ), f"buffer-load panes are not a valid mapping for {nm} x {nn} slots"
        # Per pane index, the slot of each component it buffer-loads, or None.
        buffer_load_slots = _by_pane_index(buffer_load_pane, first_slots, num_panes)
        # A slot is DS-read in the pane that buffer-loads its operand's payload slot.
        # Only LDS components have a DS read: the MFMA reads a register ring itself.
        ds_read_pane = {
            CompSlot(c, slot): buffer_load_pane[
                CompSlot(A_PAYLOAD if c.operand_index.value == A else B_PAYLOAD, slot)
            ]
            for c in COMPONENTS
            for slot in range(nm if c.operand_index.value == A else nn)
        }
        ds_read_slots = _by_pane_index(
            ds_read_pane,
            {cs for cs in first_slots if layout.component_via_lds(cs.component)},
            num_panes,
        )

        commit = (
            bool(tc.commit_per_op()),
            bool(tc.commit_per_pane()),
            bool(tc.commit_per_stage()),
            bool(tc.commit_per_stage_whole()),
        )
        static_remainder = bool(_v(tc.UNROLL_EPILOGUE))
        static_drain = (
            buffer_period <= 3
            and reg_buffer_period == 1
            and _v(fc.output_quant) is None
        )
        if epilogue_groups is None:
            epilogue_groups = int(
                bool(_v(layout.epilogue_inputs_via_lds))
                and bool(_v(fc.has_gammas) or _v(fc.has_bias))
            )

        def step_plan(kind, stage, phase):
            return _step_plan(
                layout,
                component_present,
                scale_slot_span,
                load_to_mfma_distance,
                depth,
                buffer_load_slots,
                ds_read_slots,
                commit,
                kind,
                stage,
                phase,
            )

        def reach(x):
            # How far back the waits of step x reach: per wait, the smallest F among
            # the LDS components it DS-reads, whose copy is the newest it needs; the
            # largest over the step's waits, or 0 when none needs a copy.
            needs = [
                [
                    load_to_mfma_distance[component_index]
                    for component_index, slot in enumerate(slots)
                    if slot is not None
                    and (x + 1) % _scale_r(scale_slot_span, COMPONENTS[component_index])
                    == 0
                ]
                for slots in ds_read_slots
            ]
            if commit[2]:
                needs = [[f for fs in needs for f in fs]]
            return max((min(fs) for fs in needs if fs), default=0)

        # Peel through the last early step whose wait differs from its steady twin's,
        # in the immediate or in the loads its window holds (PIPELINE.md, Peeling).
        # From step `first_steady` on every component copies, so a step whose window
        # starts there or later is steady: only steps below `bound` can differ, and
        # their twins sit in the `copy_period` steps after it. `reach(x) <= max(F)`,
        # so only steps below `max(F) - min(F) - 1` can fail.
        peeled = 1
        present = [f for f, p in zip(load_to_mfma_distance, component_present) if p]
        first_steady = -min(present)
        bound = 1
        for x in range(max(present) + first_steady - 1):
            if x + 1 - reach(x) < first_steady:
                bound = x + 1
        if bound > 1:
            early = []
            for x in range(-depth, bound + copy_period):
                kind = StepKind.PEELED if x >= 0 else StepKind.PROLOGUE
                early.append((x, x) + step_plan(kind, x, x)[2:])
            waits = _compute_wait_plan(
                [early], load_to_mfma_distance, commit[2], num_panes, windows=True
            )
            for x in range(bound):
                twin = x + copy_period * -(-(bound - x) // copy_period)
                if waits.get(x) != waits.get(twin):
                    peeled = x + 1

        self._component_present = gl.constexpr(component_present)
        self._num_buffers = gl.constexpr(num_buffers)
        self._scale_slot_span = gl.constexpr(scale_slot_span)
        self._load_to_mfma_distance = gl.constexpr(load_to_mfma_distance)
        self._ds_read_in_mfma = gl.constexpr(
            tuple(
                bool(tc.ds_read_in_mfma(c.operand_index.value, c.is_scale.value))
                for c in COMPONENTS
            )
        )
        self._buffer_load_slots = gl.constexpr(buffer_load_slots)
        self._ds_read_slots = gl.constexpr(ds_read_slots)
        self.num_k = gl.constexpr(num_k)
        self.depth = gl.constexpr(depth)
        self.unroll = gl.constexpr(unroll)
        self.peeled = gl.constexpr(peeled)
        self.buffer_period = gl.constexpr(buffer_period)
        self.reg_buffer_period = gl.constexpr(reg_buffer_period)
        self.commit_per_copy = gl.constexpr(commit[0])
        self.commit_per_pane = gl.constexpr(commit[1])
        self.commit_per_tile = gl.constexpr(commit[2])
        self.tile_commit_after_mfma = gl.constexpr(commit[3])
        self._soff_unroll = gl.constexpr(bool(_v(tc.SOFF_UNROLL)))
        self.static_remainder = gl.constexpr(static_remainder)
        self.static_drain = gl.constexpr(static_drain)
        self.epilogue_groups = gl.constexpr(epilogue_groups)

        drain_phases = buffer_period if static_drain else 1
        # Emitted bodies per kind, in `StepKind` order; none depends on `num_k`.
        step_counts = (
            depth,
            peeled,
            unroll,
            unroll - 1 if static_remainder else 1,
            drain_phases * (depth - 1),
        )
        self._step_counts = gl.constexpr(step_counts)

        main = num_k - depth
        if main < peeled + unroll:
            self._steps = gl.constexpr(None)
            self._active = gl.constexpr(None)
            self._loads = gl.constexpr(None)
            self._commits = gl.constexpr(None)
            self._waits = gl.constexpr(None)
            return

        # One row per emitted body, in `_row` order: (kind, stage, phase, ki).
        steps = [(int(StepKind.PROLOGUE), x, x, 0) for x in range(-depth, 0)]
        steps += [(int(StepKind.PEELED), x, x, 0) for x in range(peeled)]
        steps += [(int(StepKind.LOOP), None, peeled + u, u) for u in range(unroll)]
        if static_remainder:
            steps += [
                (int(StepKind.REMAINDER), None, peeled + u, u)
                for u in range(unroll - 1)
            ]
        else:
            steps.append((int(StepKind.REMAINDER), None, 0, 0))
        steps += [
            (
                int(StepKind.DRAIN),
                j,
                num_k - depth + j,
                p + j - peeled if static_drain else 0,
            )
            for p in range(drain_phases)
            for j in range(depth - 1)
        ]
        assert len(steps) == sum(step_counts)
        plans = [step_plan(s[0], s[1], s[2]) for s in steps]

        loop = depth + peeled
        remainder = loop + unroll
        drain = len(steps) - drain_phases * (depth - 1)
        remaining = main - peeled
        iters, rem = remaining // unroll, remaining % unroll

        def entry(index, step):
            return (index, step) + plans[index][2:]

        head = [entry(i, steps[i][1]) for i in range(loop)]
        head += [
            entry(loop + u, peeled + it * unroll + u)
            for it in range(iters)
            for u in range(unroll)
        ]
        base = peeled + iters * unroll

        def tail(count):
            return [
                entry(remainder + (u if static_remainder else 0), base + u)
                for u in range(count)
            ]

        phase = main % buffer_period if static_drain else 0
        executed = head + tail(rem)
        executed.append((None, None, ((),), (((),) * epilogue_groups,), ()))
        executed += [
            entry(drain + phase * (depth - 1) + j, main + j) for j in range(depth - 1)
        ]
        orders = [executed]
        longest = unroll - 1 if static_remainder else 1
        if rem < longest:
            orders.append(head + tail(longest))
        waits = _compute_wait_plan(orders, load_to_mfma_distance, commit[2], num_panes)
        for p in range(drain_phases):
            for j in range(depth - 1):
                executed_copy = drain + phase * (depth - 1) + j
                if executed_copy in waits:
                    waits[drain + p * (depth - 1) + j] = waits[executed_copy]
        idle = (None, (None,) * num_panes)

        self._steps = gl.constexpr(tuple(steps))
        self._active = gl.constexpr(tuple(plan[0] for plan in plans))
        self._loads = gl.constexpr(tuple(plan[1] for plan in plans))
        self._commits = gl.constexpr(tuple(plan[3] for plan in plans))
        self._waits = gl.constexpr(
            tuple(waits.get(index, idle) for index in range(len(steps)))
        )

    @gluon.constexpr_function
    def prologue_step(self, x):
        depth = self.depth.value
        assert -depth <= x < 0, f"no prologue step {x}"
        return Step(StepKind.PROLOGUE, x + depth)

    @gluon.constexpr_function
    def peeled_step(self, x):
        assert 0 <= x < self.peeled.value, f"no peeled step {x}"
        return Step(StepKind.PEELED, x)

    @gluon.constexpr_function
    def loop_step(self, u):
        assert 0 <= u < self.unroll.value, f"no loop step {u}"
        return Step(StepKind.LOOP, u)

    @gluon.constexpr_function
    def remainder_step(self, u):
        count = self._step_counts.value[StepKind.REMAINDER]
        assert 0 <= u < count, f"no remainder step {u}"
        return Step(StepKind.REMAINDER, u)

    @gluon.constexpr_function
    def drain_step(self, p, j):
        depth = self.depth.value
        phases = self.buffer_period.value if self.static_drain.value else 1
        assert 0 <= p < phases and 0 <= j < depth - 1, f"no drain step {p, j}"
        return Step(StepKind.DRAIN, p * (depth - 1) + j)

    @gluon.constexpr_function
    def _row(self, step):
        # The step's row in the per-step tables, which hold the kinds' bodies in
        # `StepKind` order.
        counts = self._step_counts.value
        kind, index = step.kind.value, step.index.value
        assert index < counts[kind], f"no step {step!r}"
        return sum(counts[:kind]) + index

    @gluon.constexpr_function
    def phase(self, step):
        return self._steps.value[self._row(step)][2]

    @gluon.constexpr_function
    def ki(self, step):
        return self._steps.value[self._row(step)][3]

    @gluon.constexpr_function
    def in_loop(self, step):
        return step.kind.value == StepKind.LOOP

    @gluon.constexpr_function
    def dot(self, step):
        return step.kind.value != StepKind.PROLOGUE

    @gluon.constexpr_function
    def is_drain(self, step):
        return step.kind.value == StepKind.DRAIN

    @gluon.constexpr_function
    def active(self, step, component):
        return self._active.value[self._row(step)][component.index()]

    @gluon.constexpr_function
    def loads(self, step, component):
        return self._loads.value[self._row(step)][component.index()]

    @gluon.constexpr_function
    def issued(self, step, pane_index):
        return tuple(
            slot if load else None
            for slot, load in zip(
                self._buffer_load_slots.value[pane_index],
                self._loads.value[self._row(step)],
            )
        )

    @gluon.constexpr_function
    def ds_reads(self, step, component):
        return self.ds_reads_at_phase(component, self.phase(step))

    @gluon.constexpr_function
    def ds_reads_at_phase(self, component, phase):
        # Step `phase` DS-reads K tile `phase + 1`; a new value starts every R tiles.
        return (
            self.component_present(component)
            and (phase + 1) % self.steps_per_buffer(component) == 0
        )

    @gluon.constexpr_function
    def commits(self, step, pane_index):
        return tuple(
            tuple(
                CompSlot(COMPONENTS[component_index], slot)
                for component_index, slot in group
            )
            for group in self._commits.value[self._row(step)][pane_index]
        )

    @gluon.constexpr_function
    def closes_group(self, step, pane_index):
        # Whether the pane commits a group at this step: not when it copies nothing
        # through LDS, or under the per-stage schemes, before the last pane.
        return bool(self._commits.value[self._row(step)][pane_index])

    @gluon.constexpr_function
    def wait(self, step, pane_index=None):
        tile, pane_waits = self._waits.value[self._row(step)]
        return tile if pane_index is None else pane_waits[pane_index]

    @gluon.constexpr_function
    def epilogue_wait(self):
        return 0 if self.epilogue_groups.value > 0 else None

    @gluon.constexpr_function
    def component_present(self, component):
        return self._component_present.value[component.index()]

    @gluon.constexpr_function
    def num_buffers(self, component):
        return self._num_buffers.value[component.index()]

    @gluon.constexpr_function
    def steps_per_buffer(self, component):
        return _scale_r(self._scale_slot_span.value, component)

    @gluon.constexpr_function
    def load_to_mfma_distance(self, component):
        return self._load_to_mfma_distance.value[component.index()]

    @gluon.constexpr_function
    def buffer_load_pane(self, comp_slot):
        return _pane_index_of(self._buffer_load_slots.value, comp_slot)

    @gluon.constexpr_function
    def ds_read_pane(self, comp_slot):
        return _pane_index_of(self._ds_read_slots.value, comp_slot)

    @gluon.constexpr_function
    def ds_read_in_mfma(self, comp_slot):
        return self._ds_read_in_mfma.value[comp_slot.component.index()]

    @gluon.constexpr_function
    def buffer_load_slots(self, pane_index):
        return self._buffer_load_slots.value[pane_index]

    @gluon.constexpr_function
    def ds_read_slot(self, pane_index, component):
        return self._ds_read_slots.value[pane_index][component.index()]

    @gluon.constexpr_function
    def register_index(self, step, component, buffer_load=False):
        # The register buffer this step's buffer load fills, or the one its MFMA reads.
        # Buffer 0 holds the value of K tile `origin`: the tile this step DS-reads,
        # which `_rotate_buffers` keeps there, or in the loop the tile its first step
        # DS-reads.
        if not self.component_present(component):
            return 0
        r = self.steps_per_buffer(component)
        if self.in_loop(step):
            current = self.peeled.value + self.ki(step)
            origin = self.peeled.value + 1
        else:
            current = self.phase(step)
            origin = current + 1
        target = current + (self.load_to_mfma_distance(component) if buffer_load else 0)
        return (target // r - origin // r) % self.num_buffers(component)

    @gluon.constexpr_function
    def final_register_index(self, component):
        # The register buffer the final MFMA, step `num_k - 1` after the last drain
        # step, reads.
        if not self.component_present(component):
            return 0
        r, last = self.steps_per_buffer(component), self.num_k.value - 1
        return (last // r - (last + 1) // r) % self.num_buffers(component)

    @gluon.constexpr_function
    def static_lds_index(self, step, component, buffer_load):
        if not self.component_present(component):
            return 0
        depth, r = self.num_buffers(component), self.steps_per_buffer(component)
        kind = step.kind.value
        if not (
            kind == StepKind.LOOP
            or (kind == StepKind.REMAINDER and self.static_remainder.value)
            or (kind == StepKind.DRAIN and self.static_drain.value)
        ):
            return None
        if self.unroll.value % (depth * r) != 0:
            return None
        advance = self.load_to_mfma_distance(component) if buffer_load else 1
        return ((self.peeled.value + self.ki(step) + advance) // r) % depth

    @gluon.constexpr_function
    def soff_unroll(self, step):
        return self.in_loop(step) and self._soff_unroll.value

    @gluon.constexpr_function
    def scale_soff_steps(self, component, offset_step):
        r = self.steps_per_buffer(component)
        start = self.peeled.value
        producer_phase = (-self.load_to_mfma_distance(component)) % r
        first_producer = (start - producer_phase + r - 1) // r
        producer = (start + offset_step - producer_phase) // r
        return (producer - first_producer) * r
