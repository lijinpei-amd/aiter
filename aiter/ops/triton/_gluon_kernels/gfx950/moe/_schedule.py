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

__all__ = ["Schedule"]


class _StepKind(IntEnum):
    PROLOGUE = 0
    WARMUP = 1
    PEELED = 2
    LOOP = 3
    REMAINDER = 4
    DRAIN = 5


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
    if kind == _StepKind.DRAIN:
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
        if per_copy:
            groups = tuple((op,) for op in ops)
        elif per_pane:
            groups = (ops,)
        else:
            whole += ops
            groups = (whole,) if pane_index == len(issued) - 1 else ()
        commits.append(groups)
    if kind == _StepKind.PROLOGUE:
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
    return active, loads, tuple(commits), demand


@gluon.constexpr_function
def _replay(orders, load_to_mfma_distance, per_tile, num_panes):
    def _immediate(position, count, step, need, load_to_mfma_distance):
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
        return count - latest - 1

    def _smaller(a, b):
        return b if a is None else a if b is None else min(a, b)

    out = {}
    for order in orders:
        position, count = {}, 0
        for key, step, commits, demand in order:
            measure = key is not None and bool(demand)
            if measure:
                tile, pane_waits = out.get(key, (None, (None,) * num_panes))
                pane_waits = list(pane_waits)
                if per_tile:
                    need = tuple(copy for copies in demand for copy in copies)
                    now = _immediate(position, count, step, need, load_to_mfma_distance)
                    tile = _smaller(tile, now)
            for pane_index, groups in enumerate(commits):
                if measure and not per_tile:
                    now = _immediate(
                        position,
                        count,
                        step,
                        demand[pane_index],
                        load_to_mfma_distance,
                    )
                    pane_waits[pane_index] = _smaller(pane_waits[pane_index], now)
                for group in groups:
                    for component_index, slot in group:
                        copy = (step, component_index, slot)
                        assert copy not in position, f"copy {copy} committed twice"
                        position[copy] = count
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
    buffer_load_panes_valid: gl.constexpr
    commit_per_copy: gl.constexpr
    commit_per_pane: gl.constexpr
    commit_per_tile: gl.constexpr
    tile_commit_after_mfma: gl.constexpr
    _soff_unroll: gl.constexpr
    static_remainder: gl.constexpr
    static_drain: gl.constexpr
    epilogue_groups: gl.constexpr
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

        depth, buffer_period, reg_buffer_period, async_period = 0, 1, 1, 1
        for component_index, c in enumerate(COMPONENTS):
            if component_present[component_index]:
                r = _scale_r(scale_slot_span, c)
                depth = max(depth, buffer_live_span[component_index])
                buffer_period = math.lcm(
                    buffer_period, num_buffers[component_index] * r
                )
                if layout.component_via_lds(c):
                    async_period = math.lcm(async_period, r)
                else:
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
        buffer_load_panes_valid = (
            set(buffer_load_pane) == slots
            and all(
                0 <= pane_index < num_panes for pane_index in buffer_load_pane.values()
            )
            and len(
                {
                    (comp_slot.component, pane_index)
                    for comp_slot, pane_index in buffer_load_pane.items()
                }
            )
            == len(buffer_load_pane)
        )
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
            and unroll % buffer_period == 0
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

        peeled = 1
        if buffer_load_panes_valid:
            settled = 3 * depth + async_period
            warm = [
                (None, x) + step_plan(_StepKind.PROLOGUE, x, x)[2:]
                for x in range(-depth, -1)
            ]
            for x in range(-1, settled):
                kind = _StepKind.PEELED if x >= 0 else _StepKind.WARMUP
                warm.append((x, x) + step_plan(kind, x, x)[2:])
            waits = _replay([warm], load_to_mfma_distance, commit[2], num_panes)
            for x in range(depth - 2):
                steady = x + async_period * -(-(2 * depth - 1 - x) // async_period)
                if waits.get(x) != waits.get(steady):
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
        self.buffer_load_panes_valid = gl.constexpr(buffer_load_panes_valid)
        self.commit_per_copy = gl.constexpr(commit[0])
        self.commit_per_pane = gl.constexpr(commit[1])
        self.commit_per_tile = gl.constexpr(commit[2])
        self.tile_commit_after_mfma = gl.constexpr(commit[3])
        self._soff_unroll = gl.constexpr(bool(_v(tc.SOFF_UNROLL)))
        self.static_remainder = gl.constexpr(static_remainder)
        self.static_drain = gl.constexpr(static_drain)
        self.epilogue_groups = gl.constexpr(epilogue_groups)

        main = num_k - depth
        if not buffer_load_panes_valid or main < peeled + unroll:
            self._steps = gl.constexpr(None)
            self._active = gl.constexpr(None)
            self._loads = gl.constexpr(None)
            self._commits = gl.constexpr(None)
            self._waits = gl.constexpr(None)
            return

        drain_phases = buffer_period if static_drain else 1
        steps = [
            (int(_StepKind.PROLOGUE), x, x, 0, False, False, False)
            for x in range(-depth, -1)
        ]
        steps.append((int(_StepKind.WARMUP), -1, -1, 0, False, False, False))
        steps += [
            (int(_StepKind.PEELED), x, x, 0, False, False, True) for x in range(peeled)
        ]
        steps += [
            (int(_StepKind.LOOP), None, peeled + u, u, True, False, True)
            for u in range(unroll)
        ]
        if static_remainder:
            steps += [
                (int(_StepKind.REMAINDER), None, peeled + u, u, False, True, True)
                for u in range(unroll - 1)
            ]
        else:
            steps.append((int(_StepKind.REMAINDER), None, 0, 0, False, False, True))
        steps += [
            (
                int(_StepKind.DRAIN),
                j,
                num_k - depth + j,
                p + j - peeled if static_drain else 0,
                False,
                static_drain,
                True,
            )
            for p in range(drain_phases)
            for j in range(depth - 1)
        ]
        plans = [step_plan(s[0], s[1], s[2]) for s in steps]

        warmup = depth - 1
        loop = warmup + 1 + peeled
        remainder = loop + unroll
        drain = len(steps) - drain_phases * (depth - 1)
        remaining = main - peeled
        iters, rem = remaining // unroll, remaining % unroll

        def entry(index, step):
            return (index, step, plans[index][2], plans[index][3])

        head = [entry(i, steps[i][1]) for i in range(warmup + 1 + peeled)]
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
        executed.append((None, None, (((),) * epilogue_groups,), ()))
        executed += [
            entry(drain + phase * (depth - 1) + j, main + j) for j in range(depth - 1)
        ]
        orders = [executed]
        longest = unroll - 1 if static_remainder else 1
        if rem < longest:
            orders.append(head + tail(longest))
        waits = _replay(orders, load_to_mfma_distance, commit[2], num_panes)
        for p in range(drain_phases):
            for j in range(depth - 1):
                executed_copy = drain + phase * (depth - 1) + j
                if executed_copy in waits:
                    waits[drain + p * (depth - 1) + j] = waits[executed_copy]
        idle = (None, (None,) * num_panes)

        self._steps = gl.constexpr(tuple(steps))
        self._active = gl.constexpr(tuple(plan[0] for plan in plans))
        self._loads = gl.constexpr(tuple(plan[1] for plan in plans))
        self._commits = gl.constexpr(tuple(plan[2] for plan in plans))
        self._waits = gl.constexpr(
            tuple(waits.get(index, idle) for index in range(len(steps)))
        )

    @gluon.constexpr_function
    def prologue_step(self, x):
        depth = self.depth.value
        assert -depth <= x < -1, f"no prologue step {x}"
        return x + depth

    @gluon.constexpr_function
    def warmup_step(self):
        return self.depth.value - 1

    @gluon.constexpr_function
    def peeled_step(self, x):
        assert 0 <= x < self.peeled.value, f"no peeled step {x}"
        return self.depth.value + x

    @gluon.constexpr_function
    def loop_step(self, u):
        assert 0 <= u < self.unroll.value, f"no loop step {u}"
        return self.depth.value + self.peeled.value + u

    @gluon.constexpr_function
    def remainder_step(self, u):
        unroll = self.unroll.value
        count = unroll - 1 if self.static_remainder.value else 1
        assert 0 <= u < count, f"no remainder step {u}"
        return self.depth.value + self.peeled.value + unroll + u

    @gluon.constexpr_function
    def drain_step(self, p, j):
        depth = self.depth.value
        phases = self.buffer_period.value if self.static_drain.value else 1
        assert 0 <= p < phases and 0 <= j < depth - 1, f"no drain step {p, j}"
        return len(self._steps.value) - (phases - p) * (depth - 1) + j

    @gluon.constexpr_function
    def phase(self, step):
        return self._steps.value[step][2]

    @gluon.constexpr_function
    def ki(self, step):
        return self._steps.value[step][3]

    @gluon.constexpr_function
    def in_loop(self, step):
        return self._steps.value[step][4]

    @gluon.constexpr_function
    def dot(self, step):
        return self._steps.value[step][6]

    @gluon.constexpr_function
    def is_drain(self, step):
        return self._steps.value[step][0] == _StepKind.DRAIN

    @gluon.constexpr_function
    def active(self, step, component):
        return self._active.value[step][component.index()]

    @gluon.constexpr_function
    def loads(self, step, component):
        return self._loads.value[step][component.index()]

    @gluon.constexpr_function
    def issued(self, step, pane_index):
        return tuple(
            slot if load else None
            for slot, load in zip(
                self._buffer_load_slots.value[pane_index], self._loads.value[step]
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
            for group in self._commits.value[step][pane_index]
        )

    @gluon.constexpr_function
    def wait(self, step, pane_index=None):
        tile, pane_waits = self._waits.value[step]
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
        if not (self.in_loop(step) or self._steps.value[step][5]):
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
