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
def _payload_buffer_load_slots(nm, nn):
    if nm > 1 and nn > 1:
        order = _payload_buffer_load_order(nm, nn)
        return (
            tuple(order.index((1, panel)) for panel in range(nm)),
            tuple(order.index((0, panel)) for panel in range(nn)),
        )
    return tuple(range(nm)), tuple(range(nn))


@gluon.constexpr_function
def _by_slot_index(slot_index_of, keys, num_slots):
    panel_of = {}
    for (component_index, panel), slot_index in slot_index_of.items():
        if (component_index, panel) in keys:
            panel_of.setdefault((slot_index, component_index), panel)
    return tuple(
        tuple(
            panel_of.get((slot_index, component_index))
            for component_index in range(len(COMPONENTS))
        )
        for slot_index in range(num_slots)
    )


@gluon.constexpr_function
def _slot_index_of(by_slot_index, component, panel):
    component_index = _component_index(component)
    for slot_index, panels in enumerate(by_slot_index):
        if panels[component_index] == panel:
            return slot_index
    return None


@gluon.constexpr_function
def _steps_per_buffer(scale_steps_per_buffer, component, axis):
    return scale_steps_per_buffer[component[0]][axis] if component[1] else 1


@gluon.constexpr_function
def _step_plan(m, layout, kind, stage, phase):
    component_present, load_to_mfma_distance = (
        m["component_present"],
        m["load_to_mfma_distance"],
    )
    steps_per_buffer = tuple(
        _steps_per_buffer(m["scale_steps_per_buffer"], c, 1) for c in COMPONENTS
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
        tuple(
            panel if panel is not None and load else None
            for panel, load in zip(panels, loads)
        )
        for panels in m["buffer_load_panels"]
    )
    per_copy, per_slot = m["commit"][0], m["commit"][1]
    commits, whole = [], ()
    for slot_index, copies in enumerate(issued):
        ops = tuple(
            (component_index, panel)
            for component_index, panel in enumerate(copies)
            if panel is not None
            and layout.component_via_lds(COMPONENTS[component_index])
        )
        if per_copy:
            groups = tuple((op,) for op in ops)
        elif per_slot:
            groups = (ops,)
        else:
            whole += ops
            groups = (whole,) if slot_index == len(issued) - 1 else ()
        commits.append(groups)
    if kind == _StepKind.PROLOGUE:
        demand = ()
    else:
        ds_reads = tuple(
            p and phase % r == 0 for p, r in zip(component_present, steps_per_buffer)
        )
        demand = tuple(
            tuple(
                (component_index, panel)
                for component_index, panel in enumerate(panels)
                if panel is not None
                and layout.component_via_lds(COMPONENTS[component_index])
                and ds_reads[component_index]
            )
            for panels in m["ds_read_panels"]
        )
    return active, loads, tuple(commits), demand


@gluon.constexpr_function
def _replay(orders, load_to_mfma_distance, per_tile, num_slots):
    def _immediate(position, count, step, need, load_to_mfma_distance):
        if not need:
            return None
        latest = -1
        for component_index, panel in need:
            copy = (
                step - load_to_mfma_distance[component_index] + 1,
                component_index,
                panel,
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
                tile, slot_waits = out.get(key, (None, (None,) * num_slots))
                slot_waits = list(slot_waits)
                if per_tile:
                    need = tuple(copy for copies in demand for copy in copies)
                    now = _immediate(position, count, step, need, load_to_mfma_distance)
                    tile = _smaller(tile, now)
            for slot_index, groups in enumerate(commits):
                if measure and not per_tile:
                    now = _immediate(
                        position, count, step, demand[slot_index], load_to_mfma_distance
                    )
                    slot_waits[slot_index] = _smaller(slot_waits[slot_index], now)
                for group in groups:
                    for component_index, panel in group:
                        copy = (step, component_index, panel)
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
    _scale_steps_per_buffer: gl.constexpr
    _load_to_mfma_distance: gl.constexpr
    _ds_read_in_mfma: gl.constexpr
    _buffer_load_panels: gl.constexpr
    _ds_read_panels: gl.constexpr
    num_k: gl.constexpr
    depth: gl.constexpr
    unroll: gl.constexpr
    peeled: gl.constexpr
    buffer_period: gl.constexpr
    reg_buffer_period: gl.constexpr
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
    _commits: gl.constexpr
    _waits: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, tuning_cfg, layout, K, epilogue_groups=None):
        tc = tuning_cfg
        fc = tc.func_cfg
        nm, nn = _v(layout.num_m_slots), _v(layout.num_n_slots)
        num_slots = nm * nn
        num_k = layout.num_k_tiles(K)

        component_present = tuple(
            not is_scale or bool(fc.has_scale(op)) for op, is_scale in COMPONENTS
        )
        num_buffers = tuple(tc.num_buffers(op, is_scale) for op, is_scale in COMPONENTS)
        scale_steps_per_buffer = _v(layout.scale_steps_per_buffer)
        buffer_live_span = tuple(
            (d - 1) * _steps_per_buffer(scale_steps_per_buffer, c, 1) + 1
            for d, c in zip(num_buffers, COMPONENTS)
        )
        load_to_mfma_distance = tuple(
            s if layout.component_via_lds(c) else s - 1
            for s, c in zip(buffer_live_span, COMPONENTS)
        )

        depth, buffer_period, reg_buffer_period, async_period = 0, 1, 1, 1
        for component_index, c in enumerate(COMPONENTS):
            if component_present[component_index]:
                ratio = _steps_per_buffer(scale_steps_per_buffer, c, 1)
                depth = max(depth, buffer_live_span[component_index])
                buffer_period = math.lcm(
                    buffer_period, num_buffers[component_index] * ratio
                )
                if layout.component_via_lds(c):
                    async_period = math.lcm(async_period, ratio)
                else:
                    reg_buffer_period = math.lcm(
                        reg_buffer_period, num_buffers[component_index] * ratio
                    )
        k_unroll = _v(tc.K_UNROLL)
        assert k_unroll >= 1, "K_UNROLL must be at least 1"
        unroll = math.lcm(k_unroll, buffer_period)

        a_slots, b_slots = _payload_buffer_load_slots(nm, nn)
        fill_mid = bool(_v(tc.SCALE_FILL_MID)) and nm == 2 and nn == 2
        a_scale_slots, b_scale_slots = (
            ((1, 2), (1, 2)) if fill_mid else (a_slots, b_slots)
        )
        panels = {
            (component_index, panel)
            for component_index, (op, _) in enumerate(COMPONENTS)
            for panel in range(nm if op == A else nn)
        }
        non_k_ratio = tuple(
            _steps_per_buffer(scale_steps_per_buffer, c, 0) for c in COMPONENTS
        )
        first_panels = {
            (component_index, panel)
            for component_index, panel in panels
            if component_present[component_index]
            and panel % non_k_ratio[component_index] == 0
        }
        buffer_load_slot = {
            (component_index, panel): slot_index
            for component_index, (op, is_scale) in enumerate(COMPONENTS)
            for panel, slot_index in enumerate(
                (a_scale_slots if op == A else b_scale_slots)
                if is_scale
                else (a_slots if op == A else b_slots)
            )
        }
        buffer_load_slots_valid = (
            set(buffer_load_slot) == panels
            and all(
                0 <= slot_index < num_slots for slot_index in buffer_load_slot.values()
            )
            and len(
                {(key[0], slot_index) for key, slot_index in buffer_load_slot.items()}
            )
            == len(buffer_load_slot)
        )
        buffer_load_panels = _by_slot_index(buffer_load_slot, first_panels, num_slots)
        ds_read_slot, ds_read_in_mfma = {}, {}
        for component_index, (op, is_scale) in enumerate(COMPONENTS):
            payload_slots = a_slots if op == A else b_slots
            for panel in range(len(payload_slots)):
                key = (component_index, panel)
                ds_read = payload_slots[panel]
                same_step = (
                    layout.component_in_reg((op, is_scale))
                    and load_to_mfma_distance[component_index] == 1
                )
                if (
                    same_step
                    and buffer_load_slots_valid
                    and buffer_load_slot[key] > ds_read
                ):
                    ds_read = buffer_load_slot[key]
                ds_read_slot[key] = ds_read
                configured = bool(tc.ds_read_in_mfma(op, is_scale))
                ds_read_in_mfma[key] = configured or (
                    same_step
                    and buffer_load_slots_valid
                    and ds_read == buffer_load_slot[key]
                )
                if same_step and buffer_load_slots_valid:
                    buffer_load = buffer_load_slot[key]
                    assert ds_read > buffer_load or (
                        ds_read == buffer_load and ds_read_in_mfma[key]
                    ), (
                        f"component {(op, is_scale)} is in registers at F == 1, so "
                        f"its DS read of panel {panel} (slot {ds_read}) must follow "
                        f"the same step's buffer load (slot {buffer_load})"
                    )
        ds_read_panels = _by_slot_index(ds_read_slot, first_panels, num_slots)

        commit = (
            bool(tc.commit_per_op()),
            bool(tc.commit_per_slot()),
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
        model = {
            "component_present": component_present,
            "scale_steps_per_buffer": scale_steps_per_buffer,
            "load_to_mfma_distance": load_to_mfma_distance,
            "depth": depth,
            "buffer_load_panels": buffer_load_panels,
            "ds_read_panels": ds_read_panels,
            "commit": commit,
        }

        peeled = 1
        if buffer_load_slots_valid:
            settled = 3 * depth + async_period
            warm = [
                (None, r) + _step_plan(model, layout, _StepKind.PROLOGUE, r, r)[2:]
                for r in range(1 - depth, 0)
            ]
            for s in range(settled + 1):
                kind = _StepKind.PEELED if s else _StepKind.WARMUP
                warm.append((s, s) + _step_plan(model, layout, kind, s, s)[2:])
            waits = _replay([warm], load_to_mfma_distance, commit[2], num_slots)
            for s in range(1, depth - 1):
                steady = s + async_period * -(-(2 * depth - s) // async_period)
                if waits.get(s) != waits.get(steady):
                    peeled = s

        self._component_present = gl.constexpr(component_present)
        self._num_buffers = gl.constexpr(num_buffers)
        self._scale_steps_per_buffer = gl.constexpr(scale_steps_per_buffer)
        self._load_to_mfma_distance = gl.constexpr(load_to_mfma_distance)
        self._ds_read_in_mfma = gl.constexpr(
            tuple(
                tuple(
                    ds_read_in_mfma[(component_index, panel)]
                    for panel in range(nm if op == A else nn)
                )
                for component_index, (op, _) in enumerate(COMPONENTS)
            )
        )
        self._buffer_load_panels = gl.constexpr(buffer_load_panels)
        self._ds_read_panels = gl.constexpr(ds_read_panels)
        self.num_k = gl.constexpr(num_k)
        self.depth = gl.constexpr(depth)
        self.unroll = gl.constexpr(unroll)
        self.peeled = gl.constexpr(peeled)
        self.buffer_period = gl.constexpr(buffer_period)
        self.reg_buffer_period = gl.constexpr(reg_buffer_period)
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
            self._commits = gl.constexpr(None)
            self._waits = gl.constexpr(None)
            return

        drain_phases = buffer_period if static_drain else 1
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
        plans = [_step_plan(model, layout, s[0], s[1], s[2]) for s in steps]

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

        phase = main % buffer_period if static_drain else 0
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
        waits = _replay(orders, load_to_mfma_distance, commit[2], num_slots)
        for p in range(drain_phases):
            for j in range(depth - 1):
                executed_copy = drain + phase * (depth - 1) + j
                if executed_copy in waits:
                    waits[drain + p * (depth - 1) + j] = waits[executed_copy]
        idle = (None, (None,) * num_slots)

        self._steps = gl.constexpr(tuple(steps))
        self._active = gl.constexpr(tuple(plan[0] for plan in plans))
        self._loads = gl.constexpr(tuple(plan[1] for plan in plans))
        self._commits = gl.constexpr(tuple(plan[2] for plan in plans))
        self._waits = gl.constexpr(
            tuple(waits.get(index, idle) for index in range(len(steps)))
        )

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
        return self._active.value[step][_component_index(component)]

    @gluon.constexpr_function
    def loads(self, step, component):
        return self._loads.value[step][_component_index(component)]

    @gluon.constexpr_function
    def issued(self, step, slot_index):
        return tuple(
            panel if load else None
            for panel, load in zip(
                self._buffer_load_panels.value[slot_index], self._loads.value[step]
            )
        )

    @gluon.constexpr_function
    def ds_reads(self, step, component):
        return self.ds_reads_at_phase(component, self.phase(step))

    @gluon.constexpr_function
    def ds_reads_at_phase(self, component, phase):
        return (
            self.component_present(component)
            and phase % self.steps_per_buffer(component) == 0
        )

    @gluon.constexpr_function
    def commits(self, step, slot_index):
        return tuple(
            tuple(
                (COMPONENTS[component_index], panel) for component_index, panel in group
            )
            for group in self._commits.value[step][slot_index]
        )

    @gluon.constexpr_function
    def wait(self, step, slot_index=None):
        tile, slot_waits = self._waits.value[step]
        return tile if slot_index is None else slot_waits[slot_index]

    @gluon.constexpr_function
    def epilogue_wait(self):
        return 0 if self.epilogue_groups.value > 0 else None

    @gluon.constexpr_function
    def component_present(self, component):
        return self._component_present.value[_component_index(component)]

    @gluon.constexpr_function
    def num_buffers(self, component):
        return self._num_buffers.value[_component_index(component)]

    @gluon.constexpr_function
    def steps_per_buffer(self, component):
        return _steps_per_buffer(self._scale_steps_per_buffer.value, component, 1)

    @gluon.constexpr_function
    def load_to_mfma_distance(self, component):
        return self._load_to_mfma_distance.value[_component_index(component)]

    @gluon.constexpr_function
    def buffer_load_slot(self, component, panel):
        return _slot_index_of(self._buffer_load_panels.value, component, panel)

    @gluon.constexpr_function
    def ds_read_slot(self, component, panel):
        return _slot_index_of(self._ds_read_panels.value, component, panel)

    @gluon.constexpr_function
    def ds_read_in_mfma(self, component, panel):
        return self._ds_read_in_mfma.value[_component_index(component)][panel]

    @gluon.constexpr_function
    def buffer_load_panels(self, slot_index):
        return self._buffer_load_panels.value[slot_index]

    @gluon.constexpr_function
    def ds_read_panel(self, slot_index, component):
        return self._ds_read_panels.value[slot_index][_component_index(component)]

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
        if not (self.in_loop(step) or self._steps.value[step][5]):
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
        producer_phase = (1 - self.load_to_mfma_distance(component)) % ratio
        first_producer = (start - producer_phase + ratio - 1) // ratio
        producer = (start + offset_step - producer_phase) // ratio
        return (producer - first_producer) * ratio


@gluon.constexpr_function
def _component_index(component):
    key = (component[0], bool(component[1]))
    for component_index, candidate in enumerate(COMPONENTS):
        if candidate == key:
            return component_index
    raise AssertionError(f"{key} is not a component")
