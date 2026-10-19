# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import math
from enum import IntEnum

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import maybe_unwrap as _v
from ._lang import strip_annotate
from ._types import COMPONENTS, A

__all__ = ["Schedule"]


class _StepKind(IntEnum):
    PROLOGUE = 0
    WARMUP = 1
    PEELED = 2
    LOOP = 3
    REMAINDER = 4
    DRAIN = 5


@gluon.constexpr_function
def _buffer_load_order(nm, nn):
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
def _payload_buffer_load_slots(nm, nn):
    if nm > 1 and nn > 1:
        order = _buffer_load_order(nm, nn)
        return (
            tuple(order.index((1, t)) for t in range(nm)),
            tuple(order.index((0, t)) for t in range(nn)),
        )
    return tuple(range(nm)), tuple(range(nn))


@gluon.constexpr_function
def _first_non_k_slot(mapping, slot, ratio):
    for t, mapped in enumerate(mapping):
        if mapped == slot and t % ratio == 0:
            return t
    return None


@gluon.constexpr_function
def _step_plan(m, kind, stage, phase):
    component_present, load_to_mfma_distance, steps_per_buffer = (
        m["component_present"],
        m["load_to_mfma_distance"],
        m["steps_per_buffer"],
    )
    if kind == _StepKind.DRAIN:
        active = tuple(
            p and stage < m["depth"] - f
            for p, f in zip(component_present, load_to_mfma_distance)
        )
    else:
        active = tuple(
            p and (stage is None or stage + f - 1 >= 0)
            for p, f in zip(component_present, load_to_mfma_distance)
        )
    loads = tuple(
        a and phase % r == (1 - f) % r
        for a, r, f in zip(active, steps_per_buffer, load_to_mfma_distance)
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
    if kind == _StepKind.PROLOGUE:
        demand = ()
    else:
        ds_reads = tuple(
            p and phase % r == 0 for p, r in zip(component_present, steps_per_buffer)
        )
        demand = tuple(
            tuple(
                (ci, t)
                for ci, t in enumerate(non_k_slots)
                if t is not None and m["via_lds"][ci] and ds_reads[ci]
            )
            for non_k_slots in m["ds_read_non_k_slots"]
        )
    return active, loads, issued, tuple(commits), demand


@gluon.constexpr_function
def _replay(orders, load_to_mfma_distance, per_tile, slots):
    def _immediate(position, count, step, need, load_to_mfma_distance):
        if not need:
            return None
        latest = -1
        for ci, t in need:
            copy = (step - load_to_mfma_distance[ci] + 1, ci, t)
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
                tile, slot_waits = out.get(key, (None, (None,) * slots))
                slot_waits = list(slot_waits)
                if per_tile:
                    need = tuple(copy for copies in demand for copy in copies)
                    now = _immediate(position, count, step, need, load_to_mfma_distance)
                    tile = _smaller(tile, now)
            for slot, groups in enumerate(commits):
                if measure and not per_tile:
                    now = _immediate(
                        position, count, step, demand[slot], load_to_mfma_distance
                    )
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

    _component_present: gl.constexpr
    _num_buffers: gl.constexpr
    _steps_per_buffer: gl.constexpr
    _load_to_mfma_distance: gl.constexpr
    _buffer_load_slots: gl.constexpr
    _ds_read_slots: gl.constexpr
    _ds_read_in_mfma: gl.constexpr
    _owned: gl.constexpr
    _ds_read_non_k_slots: gl.constexpr
    num_k: gl.constexpr
    depth: gl.constexpr
    unroll: gl.constexpr
    peeled: gl.constexpr
    ring_period: gl.constexpr
    register_period: gl.constexpr
    async_period: gl.constexpr
    buffer_load_slots_valid: gl.constexpr
    commit_per_copy: gl.constexpr
    commit_per_slot: gl.constexpr
    commit_per_tile: gl.constexpr
    tile_commit_after_mfma: gl.constexpr
    _soff_unroll: gl.constexpr
    static_remainder: gl.constexpr
    static_drain: gl.constexpr
    epilogue_groups: gl.constexpr
    _steps: gl.constexpr
    _active: gl.constexpr
    _loads: gl.constexpr
    _issued: gl.constexpr
    _commits: gl.constexpr
    _waits: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, tuning_cfg, layout, K, epilogue_groups=None):
        tc = tuning_cfg
        fc = tc.func_cfg
        nm, nn = _v(layout.num_m_slots), _v(layout.num_n_slots)
        slots = nm * nn
        num_k = layout.num_k_tiles(K)

        component_present = tuple(
            not is_scale or bool(fc.has_scale(op)) for op, is_scale in COMPONENTS
        )
        via_lds = tuple(bool(layout.component_via_lds(c)) for c in COMPONENTS)
        num_buffers = tuple(tc.num_buffers(op, is_scale) for op, is_scale in COMPONENTS)
        steps_per_buffer = tuple(
            layout.scale_slot_k(op) // _v(layout.scale_block_k) if is_scale else 1
            for op, is_scale in COMPONENTS
        )
        ratio_non_k = tuple(
            layout.scale_slot_nonk(op) // layout.slot_nonk(op) if is_scale else 1
            for op, is_scale in COMPONENTS
        )
        live_span = tuple(
            (d - 1) * r + 1 for d, r in zip(num_buffers, steps_per_buffer)
        )
        load_to_mfma_distance = tuple(
            s if lds else s - 1 for s, lds in zip(live_span, via_lds)
        )

        depth, ring_period, register_period, async_period = 0, 1, 1, 1
        for ci in range(len(COMPONENTS)):
            if component_present[ci]:
                depth = max(depth, live_span[ci])
                ring_period = math.lcm(
                    ring_period, num_buffers[ci] * steps_per_buffer[ci]
                )
                if via_lds[ci]:
                    async_period = math.lcm(async_period, steps_per_buffer[ci])
                else:
                    register_period = math.lcm(
                        register_period, num_buffers[ci] * steps_per_buffer[ci]
                    )
        k_unroll = _v(tc.K_UNROLL)
        assert k_unroll >= 1, "K_UNROLL must be at least 1"
        unroll = math.lcm(k_unroll, ring_period)

        a_slots, b_slots = _payload_buffer_load_slots(nm, nn)
        fill_mid = bool(_v(tc.SCALE_FILL_MID)) and nm == 2 and nn == 2
        buffer_load_slots = tuple(
            (1, 2) if is_scale and fill_mid else (a_slots if op == A else b_slots)
            for op, is_scale in COMPONENTS
        )
        buffer_load_slots_valid = all(
            len(mapping) == (nm if op == A else nn)
            and all(0 <= slot < slots for slot in mapping)
            and len(set(mapping)) == len(mapping)
            for (op, _), mapping in zip(COMPONENTS, buffer_load_slots)
        )
        owned = tuple(
            tuple(
                (
                    _first_non_k_slot(buffer_load_slots[ci], slot, ratio_non_k[ci])
                    if component_present[ci]
                    else None
                )
                for ci in range(len(COMPONENTS))
            )
            for slot in range(slots)
        )
        ds_read_slots, ds_read_in_mfma = [], []
        for ci, (op, is_scale) in enumerate(COMPONENTS):
            payload_slots = a_slots if op == A else b_slots
            ds_reads, in_mfma = [], []
            for t in range(len(payload_slots)):
                ds_read = payload_slots[t]
                same_step = not via_lds[ci] and load_to_mfma_distance[ci] == 1
                if (
                    same_step
                    and buffer_load_slots_valid
                    and buffer_load_slots[ci][t] > ds_read
                ):
                    ds_read = buffer_load_slots[ci][t]
                ds_reads.append(ds_read)
                configured = bool(tc.ds_read_in_mfma(op, is_scale))
                in_mfma.append(
                    configured
                    or (
                        same_step
                        and buffer_load_slots_valid
                        and ds_read == buffer_load_slots[ci][t]
                    )
                )
            ds_read_slots.append(tuple(ds_reads))
            ds_read_in_mfma.append(tuple(in_mfma))
        ds_read_non_k_slots = tuple(
            tuple(
                (
                    _first_non_k_slot(ds_read_slots[ci], slot, ratio_non_k[ci])
                    if component_present[ci]
                    else None
                )
                for ci in range(len(COMPONENTS))
            )
            for slot in range(slots)
        )

        commit = (
            bool(tc.commit_per_op()),
            bool(tc.commit_per_slot()),
            bool(tc.commit_per_stage()),
            bool(tc.commit_per_stage_whole()),
        )
        static_remainder = bool(_v(tc.UNROLL_EPILOGUE))
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
            "component_present": component_present,
            "via_lds": via_lds,
            "steps_per_buffer": steps_per_buffer,
            "load_to_mfma_distance": load_to_mfma_distance,
            "depth": depth,
            "owned": owned,
            "ds_read_non_k_slots": ds_read_non_k_slots,
            "commit": commit,
        }

        peeled = 1
        if buffer_load_slots_valid:
            settled = 3 * depth + async_period
            warm = [
                (None, r) + _step_plan(model, _StepKind.PROLOGUE, r, r)[3:]
                for r in range(1 - depth, 0)
            ]
            for s in range(settled + 1):
                kind = _StepKind.PEELED if s else _StepKind.WARMUP
                warm.append((s, s) + _step_plan(model, kind, s, s)[3:])
            waits = _replay([warm], load_to_mfma_distance, commit[2], slots)
            for s in range(1, depth - 1):
                steady = s + async_period * -(-(2 * depth - s) // async_period)
                if waits.get(s) != waits.get(steady):
                    peeled = s

        self._component_present = gl.constexpr(component_present)
        self._num_buffers = gl.constexpr(num_buffers)
        self._steps_per_buffer = gl.constexpr(steps_per_buffer)
        self._load_to_mfma_distance = gl.constexpr(load_to_mfma_distance)
        self._buffer_load_slots = gl.constexpr(buffer_load_slots)
        self._ds_read_slots = gl.constexpr(tuple(ds_read_slots))
        self._ds_read_in_mfma = gl.constexpr(tuple(ds_read_in_mfma))
        self._owned = gl.constexpr(owned)
        self._ds_read_non_k_slots = gl.constexpr(ds_read_non_k_slots)
        self.num_k = gl.constexpr(num_k)
        self.depth = gl.constexpr(depth)
        self.unroll = gl.constexpr(unroll)
        self.peeled = gl.constexpr(peeled)
        self.ring_period = gl.constexpr(ring_period)
        self.register_period = gl.constexpr(register_period)
        self.async_period = gl.constexpr(async_period)
        self.buffer_load_slots_valid = gl.constexpr(buffer_load_slots_valid)
        self.commit_per_copy = gl.constexpr(commit[0])
        self.commit_per_slot = gl.constexpr(commit[1])
        self.commit_per_tile = gl.constexpr(commit[2])
        self.tile_commit_after_mfma = gl.constexpr(commit[3])
        self._soff_unroll = gl.constexpr(bool(_v(tc.SOFF_UNROLL)))
        self.static_remainder = gl.constexpr(static_remainder)
        self.static_drain = gl.constexpr(static_drain)
        self.epilogue_groups = gl.constexpr(epilogue_groups)

        main = num_k - depth
        if not buffer_load_slots_valid or main < peeled + unroll:
            self._steps = gl.constexpr(None)
            self._active = gl.constexpr(None)
            self._loads = gl.constexpr(None)
            self._issued = gl.constexpr(None)
            self._commits = gl.constexpr(None)
            self._waits = gl.constexpr(None)
            return

        drain_phases = ring_period if static_drain else 1
        steps = [
            (int(_StepKind.PROLOGUE), r, r, 0, False, False, False)
            for r in range(1 - depth, 0)
        ]
        steps.append((int(_StepKind.WARMUP), 0, 0, 0, False, False, False))
        steps += [
            (int(_StepKind.PEELED), x + 1, x + 1, 0, False, False, True)
            for x in range(peeled)
        ]
        steps += [
            (int(_StepKind.LOOP), None, peeled + u + 1, u, True, False, True)
            for u in range(unroll)
        ]
        if static_remainder:
            steps += [
                (int(_StepKind.REMAINDER), None, peeled + u + 1, u, False, True, True)
                for u in range(unroll - 1)
            ]
        else:
            steps.append((int(_StepKind.REMAINDER), None, 0, 0, False, False, True))
        steps += [
            (
                int(_StepKind.DRAIN),
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
        waits = _replay(orders, load_to_mfma_distance, commit[2], slots)
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

    @gluon.constexpr_function
    def scheduled(self):
        return self._steps.value is not None

    @gluon.constexpr_function
    def num_steps(self):
        return len(self._steps.value)

    @gluon.constexpr_function
    def prologue_step(self, r):
        depth = self.depth.value
        assert 1 - depth <= r < 0, f"no prologue step {r}"
        return r + depth - 1

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
        phases = self.ring_period.value if self.static_drain.value else 1
        assert 0 <= p < phases and 0 <= j < depth - 1, f"no drain step {p, j}"
        return self.num_steps() - (phases - p) * (depth - 1) + j

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
    def static_phase(self, step):
        return self._steps.value[step][5]

    @gluon.constexpr_function
    def dot(self, step):
        return self._steps.value[step][6]

    @gluon.constexpr_function
    def is_drain(self, step):
        return self._steps.value[step][0] == _StepKind.DRAIN

    @gluon.constexpr_function
    def active(self, step, component):
        return self._active.value[step][_ci(component)]

    @gluon.constexpr_function
    def loads(self, step, component):
        return self._loads.value[step][_ci(component)]

    @gluon.constexpr_function
    def issued(self, step, slot):
        return self._issued.value[step][slot]

    @gluon.constexpr_function
    def ds_reads(self, step, component):
        return self.ds_reads_at_phase(component, self.phase(step))

    @gluon.constexpr_function
    def ds_reads_at_phase(self, component, phase):
        ci = _ci(component)
        return (
            self._component_present.value[ci]
            and phase % self._steps_per_buffer.value[ci] == 0
        )

    @gluon.constexpr_function
    def commits(self, step, slot):
        return tuple(
            tuple((COMPONENTS[ci], t) for ci, t in group)
            for group in self._commits.value[step][slot]
        )

    @gluon.constexpr_function
    def wait(self, step, slot=None):
        tile, slot_waits = self._waits.value[step]
        return tile if slot is None else slot_waits[slot]

    @gluon.constexpr_function
    def epilogue_wait(self):
        return 0 if self.epilogue_groups.value > 0 else None

    @gluon.constexpr_function
    def component_present(self, component):
        return self._component_present.value[_ci(component)]

    @gluon.constexpr_function
    def num_buffers(self, component):
        return self._num_buffers.value[_ci(component)]

    @gluon.constexpr_function
    def steps_per_buffer(self, component):
        return self._steps_per_buffer.value[_ci(component)]

    @gluon.constexpr_function
    def live_span(self, component):
        return (self.num_buffers(component) - 1) * self.steps_per_buffer(component) + 1

    @gluon.constexpr_function
    def load_to_mfma_distance(self, component):
        return self._load_to_mfma_distance.value[_ci(component)]

    @gluon.constexpr_function
    def producer_phase(self, component):
        ratio = self.steps_per_buffer(component)
        return (1 - self.load_to_mfma_distance(component)) % ratio

    @gluon.constexpr_function
    def buffer_load_slot(self, component, non_k_slot):
        return self._buffer_load_slots.value[_ci(component)][non_k_slot]

    @gluon.constexpr_function
    def ds_read_slot(self, component, non_k_slot):
        return self._ds_read_slots.value[_ci(component)][non_k_slot]

    @gluon.constexpr_function
    def ds_read_in_mfma(self, component, non_k_slot):
        return self._ds_read_in_mfma.value[_ci(component)][non_k_slot]

    @gluon.constexpr_function
    def owned(self, slot):
        return self._owned.value[slot]

    @gluon.constexpr_function
    def ds_read_non_k_slot(self, slot, component):
        return self._ds_read_non_k_slots.value[slot][_ci(component)]

    @gluon.constexpr_function
    def register_index(self, step, component, buffer_load=False):
        if not self.component_present(component):
            return 0
        ratio = self.steps_per_buffer(component)
        if self.in_loop(step):
            origin = self.peeled.value + 1
            current = origin + self.ki(step)
        else:
            origin = self.phase(step)
            current = origin
        target = current + (
            self.load_to_mfma_distance(component) - 1 if buffer_load else 0
        )
        return (target // ratio - origin // ratio) % self.num_buffers(component)

    @gluon.constexpr_function
    def static_lds_index(self, step, component, buffer_load):
        if not self.component_present(component):
            return 0
        depth, ratio = self.num_buffers(component), self.steps_per_buffer(component)
        if not (self.in_loop(step) or self.static_phase(step)):
            return None
        if self.unroll.value % (depth * ratio) != 0:
            return None
        advance = self.load_to_mfma_distance(component) - 1 if buffer_load else 0
        return ((self.peeled.value + self.ki(step) + 1 + advance) // ratio) % depth

    @gluon.constexpr_function
    def soff_unroll(self, step):
        return self.in_loop(step) and self._soff_unroll.value

    @gluon.constexpr_function
    def scale_soff_steps(self, component, offset_step):
        ratio = self.steps_per_buffer(component)
        start = self.peeled.value + 1
        producer_phase = self.producer_phase(component)
        first_producer = (start - producer_phase + ratio - 1) // ratio
        producer = (start + offset_step - producer_phase) // ratio
        return (producer - first_producer) * ratio


@gluon.constexpr_function
def _ci(component):
    key = (component[0], bool(component[1]))
    for ci, candidate in enumerate(COMPONENTS):
        if candidate == key:
            return ci
    raise AssertionError(f"{key} is not a component")
