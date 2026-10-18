# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Execute the unified schedule on CPU with symbolic HBM/LDS/register values.

Only hardware operations and Gluon aggregates are replaced.  The actual driver,
fill guards, pointer advances, register tuple updates, read selection and waits
run as Python, so the oracle observes their issue history rather than reproducing
the pipeline implementation.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe import _pipeline as pipeline
from aiter.ops.triton._gluon_kernels.gfx950.moe import moe_gemm as kernel
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import Schedule
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    OPERANDS,
    PAYLOAD,
    SCALE,
    A,
    WaitCommitScheme,
    _slot_index,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import validate
from op_tests.triton_tests.moe.test_moe_gemm_gluon_scale_layout import _HostView
from op_tests.triton_tests.moe.test_moe_gemm_gluon_wait_commit import _ScheduleConfig


class _Config(_ScheduleConfig):
    """Like its base: one duck-typed func config, tuning config and layout."""

    def __init__(
        self,
        scheme,
        depths,
        register_mask,
        *,
        shape=(2, 2),
        scales=True,
        packed=False,
        read_mask=0,
        middle=False,
        unroll=3,
        unroll_epilogue=True,
        soff=False,
        scale_steps=None,
        scale_non_k_slots=(1, 1),
        compiler_pipeline=False,
    ):
        super().__init__(*shape, scheme, middle)
        self.depths = dict(zip(COMPONENTS, depths, strict=True))
        self.scales = (scales, scales)
        self.payload_async = (
            not bool(register_mask & 8),
            not bool(register_mask & 1),
        )
        self.scale_async = (not bool(register_mask & 2), not bool(register_mask & 4))
        self.ALLOW_B_VIA_LDS = not (register_mask & 1)
        self.packed = packed
        self.scale_steps = scale_steps or ((2, 2) if packed else (1, 1))
        self.scale_non_k_slots = scale_non_k_slots
        self.compiler_pipeline = compiler_pipeline
        self.read_mask = read_mask
        self.K_UNROLL = unroll
        self.UNROLL_EPILOGUE = unroll_epilogue
        self.SOFF_UNROLL = soff
        self.FROZEN_STEP = False
        self.SCHED_MODE = 0
        self.SLOT_M = self.SLOT_N = 1
        self.BLOCK_K = 128 if packed else 256
        self.mma_acc_dtype = None
        self.output_quant = None

    def num_buffers(self, operand, scale=False):
        return self.depths[(operand, bool(scale))]

    def buffer_live_span(self, operand, scale=False):
        return (self.num_buffers(operand, scale) - 1) * (
            self.ratio_k(operand) if scale else 1
        ) + 1

    def active_components(self):
        return tuple(
            component
            for component in COMPONENTS
            if not component[1] or self.has_scale(component[0])
        )

    def sizing(self):
        """``(depth, peeled, unroll)``, which do not depend on K."""
        schedule = Schedule(self, self, self.BLOCK_K)
        return (
            maybe_unwrap(schedule.depth),
            maybe_unwrap(schedule.peeled),
            maybe_unwrap(schedule.unroll),
        )

    def minimum_num_k(self):
        return sum(self.sizing())

    num_k_slots = num_prefetch_k_slots = 1

    def slot_nonk(self, operand):
        return (self.SLOT_M, self.SLOT_N)[operand]

    def scale_slot_nonk(self, operand):
        return self.slot_nonk(operand) * self.scale_non_k_slots[operand]

    def scale_slot_k(self, operand):
        steps = self.scale_steps[operand] if self.has_scale(operand) else 1
        return self.BLOCK_K * steps

    def ratio_k(self, operand):
        """Steps one scale slot spans, ``scale_slot_k / BLOCK_K``."""
        return self.scale_slot_k(operand) // self.BLOCK_K

    def ratio_non_k(self, operand):
        """Non-K slots one scale slot spans, ``scale_slot_nonk / slot_nonk``."""
        return self.scale_slot_nonk(operand) // self.slot_nonk(operand)


    def operand_elem_ty(self, operand):
        return SimpleNamespace(primitive_bitwidth=8)

    def ds_read_in_mfma(self, operand, scale=False):
        return bool(self.read_mask & (1 << (operand + 2 * scale)))

    def warp_pipeline_compiler(self):
        return self.compiler_pipeline

    def warp_pipeline_manual(self):
        return False

    def commit_per_stage_warp_pipeline(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_STAGE_WARP_PIPELINE

    def commit_per_stage_whole(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_STAGE_WHOLE

    dot_result_fragment_layout = None

    @property
    def accumulator_fragment_shape_slot(self):
        return [self.SLOT_M, self.SLOT_N]


@dataclass(frozen=True)
class _Fragment:
    component: tuple[int, bool]
    tile: int
    mini: int
    k: int

    def to(self, dtype):
        return self


def _pointers(a, b, a_scale, b_scale):
    return SimpleNamespace(
        a_hbm_ptr=a, b_hbm_ptr=b, a_scale_hbm_ptr=a_scale, b_scale_hbm_ptr=b_scale
    )


def _fragments(a, a_scale, b, b_scale, acc):
    return SimpleNamespace(
        a_payload=a, a_scale=a_scale, b_payload=b, b_scale=b_scale, acc=acc
    )


class _Machine:
    """Literal copy FIFO and immutable fragment lifetimes, with absolute K tiles."""

    def __init__(self, tc, num_k):
        self.tc, self.num_k = tc, num_k
        self.current_read = None
        self.loads, self.reads, self.mfmas = set(), set(), set()
        self.pending, self.committed, self.outstanding = [], [], []
        self.completed = set()
        self.lds = {}
        self.steps = []
        self.current_in_loop = False
        self.current_drain = False
        self.events = []
        # The latest wait as (immediate, groups committed when it issued), held until
        # the reads it guards are checked against it.
        self.wait = None

    def load(self, component, ring, mini, pointer):
        tc = self.tc
        operand, is_scale = component
        tile = pointer
        ratio = tc.ratio_k(operand) if is_scale else 1
        nonk = tc.ratio_non_k(operand) if is_scale else 1
        mk = tc.num_k_slots
        assert 0 <= tile < self.num_k, "HBM load exceeded the K strip"
        live_span = (tc.depths[component] - 1) * ratio + 1
        fill_span = live_span - int(tc.component_in_reg(component))
        lead = fill_span - 1
        assert tile == self.current_read + lead
        assert tile % ratio == 0 and mini % nonk == 0
        token = component, mini, tile
        assert token not in self.loads, "HBM tile loaded more than once"
        self.loads.add(token)
        self.events.append(
            (
                self.current_read,
                self.current_in_loop,
                self.current_drain,
                "load",
                component,
                tile,
                mini,
            )
        )
        fragments = tuple(
            _Fragment(component, tile + k // mk, mini + mn, k % mk)
            for mn in range(nonk)
            for k in range(mk * ratio)
        )
        if tc.component_via_lds(component):
            key = component, mini, ring
            if key in self.lds:
                old = self.lds[key][0]
                assert (
                    component,
                    mini,
                    old,
                ) in self.reads, "LDS overwritten before its DS read"
            assert ring == tile // ratio % tc.depths[component]
            self.lds[key] = tile, fragments
            self.pending.append(token)
        return fragments

    def buffer_load_payload(
        self, operand, VIA_LDS, ring, mini, pointer, offset, soff=0
    ):
        fragments = self.load((operand, PAYLOAD), ring, mini, pointer + soff)
        if not VIA_LDS:
            return fragments

    def buffer_load_scale(self, operand, VIA_LDS, ring, mini, pointer, offset, soff=0):
        fragments = self.load((operand, SCALE), ring, mini, pointer + soff)
        if not VIA_LDS:
            return fragments

    def commit_buffer_load(self):
        group = tuple(self.pending)
        self.pending.clear()
        self.committed.append(group)
        self.outstanding.append(group)

    def wait_buffer_load_groups(self, count):
        assert self.wait is None, "two waits before one read point"
        self.wait = count, len(self.committed)
        while len(self.outstanding) > count:
            self.completed.update(self.outstanding.pop(0))

    def required(self, slot):
        tc = self.tc
        if (tc.nm, tc.nn) == (2, 2):
            payloads = [
                (B_PAYLOAD, 0),
                (A_PAYLOAD, 0),
                (A_PAYLOAD, 1),
                (B_PAYLOAD, 1),
            ]
        else:
            payloads = []
            for tile in range(max(tc.nm, tc.nn)):
                if tile < tc.nm:
                    payloads.append((A_PAYLOAD, tile))
                if tile < tc.nn:
                    payloads.append((B_PAYLOAD, tile))
        required = set()
        for pos, (payload_component, mini) in enumerate(payloads):
            if slot is not None and slot != pos:
                continue
            operand, _ = payload_component
            for component in ((operand, PAYLOAD), (operand, SCALE)):
                if component in tc.active_components() and tc.component_via_lds(
                    component
                ):
                    if component[1] and (
                        self.current_read % tc.ratio_k(operand)
                        or mini % tc.ratio_non_k(operand)
                    ):
                        continue
                    required.add((component, mini, self.current_read))
        return required

    def check_wait(self, slot):
        """The wait before this read point retires exactly the copies it reads.

        Checked when the slot first reads, against the FIFO as it stood when the wait
        issued: the immediate must leave the newest required group retired and every
        newer one in flight. A per-tile scheme waits once, before slot zero, for the
        whole step (``slot=None``).
        """
        required = self.required(slot)
        if required:
            assert self.wait is not None, (self.current_read, slot, "no wait")
            actual, committed = self.wait
            youngest = max(
                i
                for i, group in enumerate(self.committed[:committed])
                if required.intersection(group)
            )
            assert actual == committed - youngest - 1, (
                self.current_read,
                slot,
                actual,
                youngest,
                self.committed[:committed],
            )
        else:
            assert self.wait is None, (self.current_read, slot, self.wait)
        self.wait = None

    def read(self, component, ring, mini, k):
        token = component, mini, self.current_read
        assert token in self.completed, "DS read issued before its async copy completed"
        actual, fragments = self.lds[component, mini, ring]
        assert actual == self.current_read
        self.reads.add(token)
        return fragments[k]

    def ds_read_frag(
        self, operand, ring, mini, k, *, READ_PAYLOAD, READ_SCALE, SCALE_READ_IDX=None
    ):
        payload = self.read((operand, PAYLOAD), ring, mini, k) if READ_PAYLOAD else None
        scale = (
            self.read((operand, SCALE), SCALE_READ_IDX, mini, k)
            if READ_SCALE and self.tc.has_scale(operand)
            else None
        )
        return payload, scale

    def ds_read_scale(self, operand, ring, mini):
        component = (operand, SCALE)
        token = component, mini, self.current_read
        assert token in self.completed, "Scale DS read issued before its copy completed"
        actual, fragments = self.lds[component, mini, ring]
        assert actual == self.current_read
        self.reads.add(token)
        return fragments

    def dot(self, a, b, accumulator, mk, fc, tc, layout, enabled, phase):
        if not enabled:
            return accumulator
        self.events.append(
            (
                self.current_read,
                self.current_in_loop,
                self.current_drain,
                "mfma",
                None,
                accumulator,
                None,
            )
        )
        assert 0 <= accumulator < self.num_k
        assert phase == accumulator * tc.BLOCK_K // 128 % 2
        for operand, fragments in zip(OPERANDS, (a, b)):
            for k in range(mk):
                payload, scale = fragments[k * 2 : k * 2 + 2]
                assert payload.component == (operand, PAYLOAD) and payload.k == k
                assert payload.tile == accumulator, (
                    "MFMA consumed stale or overwritten registers"
                )
                if tc.has_scale(operand):
                    assert scale == _Fragment(
                        (operand, SCALE), accumulator, payload.mini, k
                    )
                self.reads.add((payload.component, payload.mini, accumulator))
                if tc.has_scale(operand):
                    self.reads.add(
                        (
                            scale.component,
                            scale.mini
                            - scale.mini % tc.ratio_non_k(operand),
                            accumulator - accumulator % tc.ratio_k(operand),
                        )
                    )
        token = a[0].mini, b[0].mini, accumulator
        assert token not in self.mfmas, "K tile accumulated twice"
        self.mfmas.add(token)
        return accumulator + 1


def _execute(tc, num_k, monkeypatch, epilogue_groups=2):
    machine = _Machine(tc, num_k)
    machine.runtime_ranges = []
    schedule = Schedule(tc, tc, num_k * tc.BLOCK_K, epilogue_groups)
    per_tile = tc.commit_per_stage()
    pc = SimpleNamespace(
        tuning_cfg=tc,
        layout=tc,
        func_cfg=tc,
        # Plain field values: the emulated steps key their FIFO on raw K tiles.
        schedule=_HostView(schedule),
        lds_ptrs=machine,
        a_hbm_offs=tuple(range(tc.nm)),
        b_hbm_offs=tuple(range(tc.nn)),
        a_scale_hbm_offs=tuple(range(tc.nm)),
        b_scale_hbm_offs=tuple(range(tc.nn)),
        a_step=1,
        b_step=1,
        s_step=1,
        a_scale_stride_k=1,
        b_scale_stride_k=1,
    )
    step_fn, fill_fn = pipeline._step.fn, pipeline._fill_slot.fn
    read_slot_fn = pipeline._read_slot.fn
    read_payload_fn = pipeline._read_payload.fn
    read_scale_fn = pipeline._read_scale.fn

    # Positional layouts: _step(pc, ptrs, buffers, regs, step, STEP),
    # _fill_slot(pc, ptrs, buffers, step, STEP, mi, ni), _read_slot(pc, buffers, step,
    # STEP, mi, ni, MFMA), _read_payload/_read_scale(pc, buffers, step, STEP,
    # non-K slot, operand).
    def step(*args, **kwargs):
        assert machine.wait is None, "a wait guarded no read"
        machine.current_read = args[4]
        machine.current_in_loop = schedule.in_loop(args[5])
        machine.current_drain = schedule.is_drain(args[5])
        machine.steps.append((args[4], machine.current_in_loop))
        return step_fn(*args, **kwargs)

    def fill(*args, **kwargs):
        machine.current_read = args[3]
        machine.current_in_loop = schedule.in_loop(args[4])
        machine.current_drain = schedule.is_drain(args[4])
        return fill_fn(*args, **kwargs)

    def read_slot(*args, **kwargs):
        if not args[6]:  # the slot's first read, right after its wait
            slot = _slot_index(args[4], args[5], tc.nm, tc.nn)
            if not per_tile:
                machine.check_wait(slot)
            elif slot == 0:
                machine.check_wait(None)
        return read_slot_fn(*args, **kwargs)

    def read_payload(*args, **kwargs):
        component = (args[5], PAYLOAD)
        if tc.component_in_reg(component):
            machine.events.append(
                (
                    machine.current_read,
                    machine.current_in_loop,
                    machine.current_drain,
                    "select",
                    component,
                    machine.current_read,
                    args[4],
                )
            )
        return read_payload_fn(*args, **kwargs)

    def read_scale(*args, **kwargs):
        operand = args[5]
        component = (operand, SCALE)
        if tc.component_in_reg(component):
            ratio = tc.ratio_k(operand)
            machine.events.append(
                (
                    machine.current_read,
                    machine.current_in_loop,
                    machine.current_drain,
                    "select",
                    component,
                    machine.current_read - machine.current_read % ratio,
                    args[4],
                )
            )
        return read_scale_fn(*args, **kwargs)

    def check_assumption(value):
        assert value

    def sched_barrier(mask):
        machine.events.append(
            (
                machine.current_read,
                machine.current_in_loop,
                machine.current_drain,
                "sched_barrier",
                None,
                mask,
                None,
            )
        )

    def runtime_range(*args):
        machine.runtime_ranges.append(args)
        return range(*args)

    def init_buffers(pc):
        return tuple(
            (
                (None,)
                * (
                    tc.depths[component]
                    * (tc.nm if component[0] == A else tc.nn)
                    * tc.num_k_slots
                    * (tc.ratio_k(component[0]) if component[1] else 1)
                )
                if component in tc.active_components()
                and not tc.component_via_lds(component)
                else ()
            )
            for component in (A_PAYLOAD, B_PAYLOAD, A_SCALE, B_SCALE)
        )

    with monkeypatch.context() as patch:
        for name in (
            "_index",
            "_replace_fragments",
            "_rotate",
            "_rotate_buffers",
            "_advance",
            "_read_payload",
            "_read_scale",
            "_read_slot",
            "_step_live",
            "_take_reg_pairs",
            "_take_operand_pairs",
        ):
            patch.setattr(pipeline, name, getattr(pipeline, name).fn)
        patch.setattr(kernel, "_PipelineRegFragments", _fragments)
        patch.setattr(pipeline, "_PipelineRegFragments", _fragments)
        patch.setattr(pipeline, "_PipelinePointers", _pointers)
        patch.setattr(pipeline, "_step", step)
        patch.setattr(pipeline, "_fill_slot", fill)
        patch.setattr(pipeline, "_read_slot", read_slot)
        patch.setattr(pipeline, "_read_payload", read_payload)
        patch.setattr(pipeline, "_read_scale", read_scale)
        patch.setattr(pipeline, "_init_buffers", init_buffers)
        patch.setattr(pipeline, "_maybe_block_dot", machine.dot)
        patch.setattr(
            pipeline, "pick_stage", lambda enabled: lambda name: nullcontext()
        )
        patch.setattr(pipeline.gl, "static_range", range)
        patch.setattr(pipeline.tl, "range", runtime_range)
        patch.setattr(pipeline.gl, "static_assert", lambda value, message: None)
        patch.setattr(pipeline.gl, "assume", check_assumption)
        patch.setattr(pipeline.gl, "zeros", lambda *args, **kwargs: 0)
        patch.setattr(pipeline.gl, "barrier", lambda: None)
        patch.setattr(pipeline.gl.amd.cdna4, "sched_barrier", sched_barrier)
        pointers, buffers, regs = pipeline._run_buffered_pipeline.fn(
            pc, _pointers(0, 0, 0, 0), num_k
        )
        main = num_k - maybe_unwrap(schedule.depth)
        assert regs.acc == (main,) * (tc.nm * tc.nn)
        for _ in range(epilogue_groups):
            machine.commit_buffer_load()
        regs = pipeline._drain_buffered_pipeline.fn(pc, pointers, buffers, regs, num_k)
        assert regs.acc == (num_k - 1,) * (tc.nm * tc.nn)
        assert machine.wait is None, "a wait guarded no read"
        assert pipeline._last_mfma.fn(pc, regs) == (num_k,) * (tc.nm * tc.nn)

    expected = {
        (component, mini, tile)
        for component in tc.active_components()
        for mini in range(
            0,
            tc.nm if component[0] == A else tc.nn,
            tc.ratio_non_k(component[0]) if component[1] else 1,
        )
        for tile in range(
            0,
            num_k,
            tc.ratio_k(component[0]) if component[1] else 1,
        )
    }
    assert machine.loads == expected
    assert machine.reads == expected
    assert machine.mfmas == {
        (mi, ni, tile)
        for mi in range(tc.nm)
        for ni in range(tc.nn)
        for tile in range(num_k)
    }
    assert [read for read, in_loop in machine.steps] == list(range(num_k))
    assert any(in_loop for read, in_loop in machine.steps)
    assert not machine.pending
    return machine


@pytest.mark.parametrize(
    "unroll_epilogue,has_runtime_tail", [(True, False), (False, True)]
)
def test_unroll_epilogue_selects_remainder_loop(
    unroll_epilogue, has_runtime_tail, monkeypatch
):
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        (3, 1, 3, 1),
        0,
        scales=False,
        unroll_epilogue=unroll_epilogue,
    )
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)
    assert any(len(args) == 2 for args in machine.runtime_ranges) is has_runtime_tail


@pytest.mark.parametrize(
    "scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name
)
@pytest.mark.parametrize(
    "depths,register_mask,options",
    [
        ((2, 2, 2, 2), 0, {}),
        ((2, 1, 2, 1), 0, {"scales": False, "unroll": 2}),
        ((3, 3, 3, 3), 0, {}),
        ((3, 1, 3, 1), 0, {"scales": False}),
        ((3, 1, 3, 1), 0, {"scales": False, "unroll": 2}),
        ((3, 3, 3, 3), 1, {}),
        ((3, 3, 3, 3), 8, {}),
        ((3, 3, 3, 3), 2, {"read_mask": 5}),
        ((3, 3, 3, 3), 4, {"read_mask": 10}),
        ((3, 3, 3, 3), 15, {"read_mask": 15}),
        ((5, 5, 2, 2), 0, {"read_mask": 10}),
        ((5, 2, 2, 3), 15, {"middle": True, "soff": True}),
        ((2, 4, 3, 2), 6, {"shape": (4, 2), "read_mask": 5}),
        ((3, 2, 4, 3), 0, {"shape": (2, 4), "read_mask": 10}),
        ((2, 1, 3, 1), 1, {"scales": False}),
        ((3, 2, 2, 3), 6, {"packed": True, "middle": True, "soff": True}),
        ((3, 3, 3, 3), 6, {"packed": True, "soff": True}),
        ((2, 4, 3, 2), 15, {"packed": True}),
        (
            (3, 2, 3, 2),
            0,
            {"packed": True, "scale_steps": (4, 4), "scale_non_k_slots": (2, 2)},
        ),
        (
            (2, 2, 2, 2),
            6,
            {
                "packed": True,
                "scale_steps": (8, 8),
                "scale_non_k_slots": (2, 2),
                "soff": True,
            },
        ),
        (
            (3, 2, 2, 3),
            4,
            {
                "packed": True,
                "scale_steps": (1, 4),
                "scale_non_k_slots": (1, 2),
                "middle": True,
            },
        ),
        (
            (2, 3, 3, 2),
            2,
            {
                "packed": True,
                "scale_steps": (4, 1),
                "scale_non_k_slots": (2, 1),
                "shape": (4, 2),
            },
        ),
    ],
)
def test_emitted_unified_pipeline_all_reachable_remainders(
    scheme,
    depths,
    register_mask,
    options,
    monkeypatch,
):
    tc = _Config(scheme, depths, register_mask, **options)
    _, _, unroll = tc.sizing()
    quantum = math.lcm(*tc.scale_steps)
    first = math.ceil(tc.minimum_num_k() / quantum) * quantum
    period = math.lcm(quantum, unroll)
    for num_k in range(first, first + period, quantum):
        _execute(tc, num_k, monkeypatch)
    # Re-enter the same static register-ring mapping from another runtime loop body.
    _execute(tc, first + 2 * unroll, monkeypatch)


def _sched_barrier_stream(machine):
    return [
        (event_read, in_loop, drain, value)
        for event_read, in_loop, drain, op, component, value, mini in machine.events
        if op == "sched_barrier"
    ]


def _assert_direct_b_stage_barriers(machine):
    in_loop_reads = [read for read, in_loop in machine.steps if in_loop]
    assert _sched_barrier_stream(machine) == [
        (0, False, False, 0),  # Existing seed barrier for DOT=False.
        *((read, True, False, 0) for read in in_loop_reads),
    ]
    return in_loop_reads


@pytest.mark.parametrize(
    "component,register_mask",
    [
        (A_PAYLOAD, 8),
        (A_SCALE, 2),
        (B_PAYLOAD, 1),
        (B_SCALE, 4),
    ],
    ids=["a-payload", "a-scale", "b-payload", "b-scale"],
)
def test_depth_two_direct_component_selects_after_same_step_fill(
    component, register_mask, monkeypatch
):
    depths = [3, 3, 3, 3]
    depths[COMPONENTS.index(component)] = 2
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        tuple(depths),
        register_mask,
        unroll=2,
    )
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)

    selections = [
        (pos, event)
        for pos, event in enumerate(machine.events)
        if event[3] == "select" and event[4] == component
    ]
    assert selections
    for select_pos, selection in selections:
        event_read, _, _, _, _, tile, mini = selection
        matching_loads = [
            pos
            for pos, event in enumerate(machine.events[:select_pos])
            if event[0] == event_read
            and event[3] == "load"
            and event[4:] == (component, tile, mini)
        ]
        assert matching_loads, "F=1 direct selection must follow its same-step fill"


@pytest.mark.parametrize("read_mask", [0, 8], ids=["configured-mem", "configured-mfma"])
def test_middle_direct_b_scale_defers_tile_zero_and_preserves_order(
    read_mask, monkeypatch
):
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        (3, 3, 3, 2),
        4,
        middle=True,
        read_mask=read_mask,
    )
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)

    for event_read in range(num_k):
        events = [
            (pos, event)
            for pos, event in enumerate(machine.events)
            if event[0] == event_read and event[4] == B_SCALE
        ]
        selections = [(pos, event) for pos, event in events if event[3] == "select"]
        assert [event[6] for _, event in selections] == [0, 1]
        tile_zero_load = next(
            pos
            for pos, event in events
            if event[3] == "load" and event[5:] == (event_read, 0)
        )
        assert tile_zero_load < selections[0][0]


@pytest.mark.parametrize("component,register_mask", [(A_SCALE, 2), (B_SCALE, 4)])
@pytest.mark.parametrize("ratio", [2, 4])
def test_direct_scale_producer_phase_and_soffset_stream_match(
    component, register_mask, ratio, monkeypatch
):
    machines = []
    for soff in (False, True):
        tc = _Config(
            WaitCommitScheme.PER_STAGE_WHOLE,
            (2, 2, 2, 2),
            register_mask,
            packed=True,
            scale_steps=(ratio, ratio),
            unroll=2,
            soff=soff,
        )
        minimum = tc.minimum_num_k()
        num_k = math.ceil(minimum / ratio) * ratio
        machines.append(_execute(tc, num_k, monkeypatch))

    load_streams = []
    for machine in machines:
        loads = [
            (event_read, in_loop, drain, tile, mini)
            for event_read, in_loop, drain, op, loaded, tile, mini in machine.events
            if op == "load" and loaded == component
        ]
        assert loads
        assert {event_read % ratio for event_read, *_ in loads} == {1 % ratio}
        load_streams.append(loads)
    assert load_streams[0] == load_streams[1]


@pytest.mark.parametrize("soff", [False, True], ids=["pointer-step", "soffset-unroll"])
def test_direct_b3_fills_k_plus_2_while_mfma_consumes_k(soff, monkeypatch):
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        (3, 3, 3, 3),
        1,  # Direct-register B payload.
        scales=False,
        soff=soff,
    )
    # Exercise one non-loop remainder stage as well as the steady loop and drain.
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)

    in_loop_reads = _assert_direct_b_stage_barriers(machine)
    assert len(in_loop_reads) == tc.sizing()[2] == 3
    for read in in_loop_reads:
        assert [
            op
            for event_read, in_loop, drain, op, component, tile, mini in machine.events
            if in_loop and not drain and event_read == read
        ][-1] == "sched_barrier"
        mfma_tiles = [
            tile
            for event_read, in_loop, drain, op, component, tile, mini in machine.events
            if in_loop and not drain and event_read == read and op == "mfma"
        ]
        assert mfma_tiles == [read - 1] * (tc.nm * tc.nn)

        loads = [
            (component, tile, mini)
            for event_read, in_loop, drain, op, component, tile, mini in machine.events
            if in_loop and not drain and event_read == read and op == "load"
        ]
        assert {
            (tile, mini) for component, tile, mini in loads if component == A_PAYLOAD
        } == {(read + 2, mi) for mi in range(tc.nm)}
        assert {
            (tile, mini) for component, tile, mini in loads if component == B_PAYLOAD
        } == {(read + 1, ni) for ni in range(tc.nn)}

    drain_b_loads = {
        (event_read, tile, mini)
        for event_read, in_loop, drain, op, component, tile, mini in machine.events
        if drain and op == "load" and component == B_PAYLOAD
    }
    assert drain_b_loads == {(num_k - 2, num_k - 1, ni) for ni in range(tc.nn)}


@pytest.mark.parametrize("soff", [False, True], ids=["pointer-step", "soffset-unroll"])
def test_direct_b2_fills_k_plus_1_while_mfma_consumes_k(soff, monkeypatch):
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        (3, 3, 2, 3),
        1,
        scales=False,
        soff=soff,
    )
    # Exercise one non-loop remainder stage as well as the steady loop and drain.
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)

    assert {
        (event_read, tile, mini)
        for event_read, in_loop, drain, op, component, tile, mini in machine.events
        if event_read < 0 and op == "load" and component == B_PAYLOAD
    } == set()
    assert {
        (tile, mini)
        for event_read, in_loop, drain, op, component, tile, mini in machine.events
        if event_read == 0 and op == "load" and component == B_PAYLOAD
    } == {(0, ni) for ni in range(tc.nn)}

    in_loop_reads = _assert_direct_b_stage_barriers(machine)
    assert len(in_loop_reads) == tc.sizing()[2] == 6
    for read in in_loop_reads:
        assert [
            op
            for event_read, in_loop, drain, op, component, tile, mini in machine.events
            if in_loop and not drain and event_read == read
        ][-1] == "sched_barrier"
        mfma_tiles = [
            tile
            for event_read, in_loop, drain, op, component, tile, mini in machine.events
            if in_loop and not drain and event_read == read and op == "mfma"
        ]
        assert mfma_tiles == [read - 1] * (tc.nm * tc.nn)
        assert {
            (tile, mini)
            for event_read, in_loop, drain, op, component, tile, mini in machine.events
            if in_loop
            and not drain
            and event_read == read
            and op == "load"
            and component == B_PAYLOAD
        } == {(read, ni) for ni in range(tc.nn)}

    drain_b_loads = {
        (event_read, tile, mini)
        for event_read, in_loop, drain, op, component, tile, mini in machine.events
        if drain and op == "load" and component == B_PAYLOAD
    }
    assert drain_b_loads == {
        (read, read, ni) for read in range(num_k - 2, num_k) for ni in range(tc.nn)
    }


def test_lds_b_has_only_the_existing_seed_sched_barrier(monkeypatch):
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        (3, 3, 3, 3),
        0,
        scales=False,
    )
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)

    assert any(in_loop for read, in_loop in machine.steps)
    assert _sched_barrier_stream(machine) == [(0, False, False, 0)]


def test_compiler_pipeline_owns_register_stage_ordering(monkeypatch):
    tc = _Config(
        WaitCommitScheme.PER_STAGE_WHOLE,
        (3, 3, 3, 3),
        4,
        compiler_pipeline=True,
    )
    num_k = tc.minimum_num_k() + 1
    machine = _execute(tc, num_k, monkeypatch)

    assert any(in_loop for _, in_loop in machine.steps)
    assert _sched_barrier_stream(machine) == [(0, False, False, 0)]


def _straight_line_waits(schedule, tc, last):
    """Wait immediates of K steps ``0..last`` as if every one were emitted in a line.

    An independent FIFO over the schedule's public commit plan and selections: the
    prologue and warm-up steps as emitted, then one step per K tile with the loop
    body's commit pattern for that K phase -- which is what every step after the
    warm-up issues. Returns ``{step: tile wait}`` for a per-tile scheme, else
    ``{step: per-slot waits}``.
    """
    slots = tc.nm * tc.nn
    per_tile = tc.commit_per_stage()
    depth, peeled, unroll = (
        maybe_unwrap(schedule.depth),
        maybe_unwrap(schedule.peeled),
        maybe_unwrap(schedule.unroll),
    )
    committed, waits = [], {}

    def required(step, slot):
        return {
            (step - schedule.fill_span(c) + 1, c, schedule.selected(slot, c))
            for c in COMPONENTS
            if schedule.selected(slot, c) is not None
            and schedule.via_lds(c)
            and schedule.selects_at_phase(c, step)
        }

    def immediate(need):
        if not need:
            return None
        youngest = max(i for i, group in enumerate(committed) if need & set(group))
        return len(committed) - youngest - 1

    for step in range(1 - depth, last + 1):
        if step < 0:
            index = schedule.prologue_step(step)
        elif step == 0:
            index = schedule.warmup_step()
        else:
            index = schedule.loop_step((step - peeled - 1) % unroll)
        if step >= 0 and per_tile:
            waits[step] = immediate(
                set().union(*(required(step, slot) for slot in range(slots)))
            )
        vector = []
        for slot in range(slots):
            if step >= 0 and not per_tile:
                vector.append(immediate(required(step, slot)))
            for group in schedule.commits(index, slot):
                committed.append(tuple((step, c, t) for c, t in group))
        if step >= 0 and not per_tile:
            waits[step] = tuple(vector)
    return waits


@pytest.mark.parametrize(
    "scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name
)
@pytest.mark.parametrize("register_mask", range(16))
@pytest.mark.parametrize(
    "extra",
    [{}, {"packed": True}, {"packed": True, "scale_steps": (2, 4)},
     {"scale_steps": (4, 1)}],
    ids=["R1", "R2", "R2R4", "R4R1"],
)
def test_peeled_wait_prefix_is_minimal(scheme, register_mask, extra):
    """The peel covers exactly the startup steps whose wait differs from the loop's.

    Compared *at the same K phase*: with a scale K-step ratio above one the phases
    interleave, and pinning the steady reference to one phase is a different, much
    weaker criterion that reports a peel far larger than the schedule needs. The
    emitted wait plan must also be the straight-line one: peeled steps keep their own
    immediates and the loop body carries the steady ones.
    """
    tc = _Config(scheme, (2, 4, 6, 3), register_mask, middle=True, **extra)
    depth, peeled, unroll = tc.sizing()
    schedule = Schedule(tc, tc, (tc.minimum_num_k() + 2 * unroll) * tc.BLOCK_K)
    settled = unroll * -(-(4 * depth) // unroll)
    waits = _straight_line_waits(schedule, tc, settled + 2 * depth + unroll)

    def steady(step):
        return waits[step + settled]

    def emitted(index):
        if tc.commit_per_stage():
            return schedule.wait(index)
        return tuple(schedule.wait(index, slot) for slot in range(tc.nm * tc.nn))

    differs = [x for x in range(depth * 2) if waits[x + 1] != steady(x + 1)]
    assert peeled == max(1, (max(differs) + 1) if differs else 0)
    assert emitted(schedule.warmup_step()) == waits[0]
    for x in range(peeled):
        assert emitted(schedule.peeled_step(x)) == waits[x + 1]
    for u in range(unroll):
        assert emitted(schedule.loop_step(u)) == steady(peeled + u + 1)


@pytest.mark.parametrize("nn", [1, 2, 3, 4])
@pytest.mark.parametrize("nm", [1, 2, 3, 4])
@pytest.mark.parametrize("middle", [False, True])
def test_every_payload_tile_is_filled_exactly_once(nm, nn, middle):
    """Copy ownership must be a total, injective map from non-K slots to slots.

    A step needs `NM + NN` payload copies and has `NM * NN` slots. While a slot
    could copy only one non-K slot, that held only for `NM, NN >= 2`; below it a copy
    was assigned a slot index that does not exist and was silently never issued, so
    the kernel read whichever stale value the ring still held.
    """
    tc = _Config(WaitCommitScheme.PER_OP, (3, 3, 3, 3), 0, shape=(nm, nn),
                 middle=middle)
    schedule = Schedule(tc, tc, tc.BLOCK_K)
    assert maybe_unwrap(schedule.fill_slots_valid), "ownership is not a valid mapping"
    for component in COMPONENTS:
        non_k_slots = nm if component[0] == A else nn
        owners = [
            non_k_slot
            for slot in range(nm * nn)
            for non_k_slot in (schedule.owned(slot)[COMPONENTS.index(component)],)
            if non_k_slot is not None
        ]
        ratio = tc.ratio_non_k(component[0]) if component[1] else 1
        expected = [t for t in range(non_k_slots) if t % ratio == 0]
        assert sorted(owners) == expected, (component, owners, expected)


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)])
def test_scale_fill_mid_moves_only_the_scale_fill_slots(shape):
    """`SCALE_FILL_MID` is a copy-ownership knob and nothing else.

    It must not disturb payload ownership, the selection slots, or the configured
    DS-read region. It is also inert outside the guarded 2x2 split, which is easy to
    forget when reading a config that sets it on a rectangular tile.
    """
    off = _Config(WaitCommitScheme.PER_OP, (3, 3, 3, 3), 0, shape=shape, middle=False)
    on = _Config(WaitCommitScheme.PER_OP, (3, 3, 3, 3), 0, shape=shape, middle=True)
    off_schedule = Schedule(off, off, off.BLOCK_K)
    on_schedule = Schedule(on, on, on.BLOCK_K)
    nm, nn = shape
    for component in (A_PAYLOAD, B_PAYLOAD):
        non_k_slots = range(nm if component[0] == A else nn)
        assert [off_schedule.fill_slot(component, t) for t in non_k_slots] == [
            on_schedule.fill_slot(component, t) for t in non_k_slots
        ], component
    for component in COMPONENTS:
        non_k_slots = range(nm if component[0] == A else nn)
        assert [off_schedule.read_slot(component, t) for t in non_k_slots] == [
            on_schedule.read_slot(component, t) for t in non_k_slots
        ], component
        assert off.ds_read_in_mfma(*component) == on.ds_read_in_mfma(*component)

    scale_off = [off_schedule.fill_slot(A_SCALE, t) for t in range(nm)]
    scale_on = [on_schedule.fill_slot(A_SCALE, t) for t in range(nm)]
    b_off = [off_schedule.fill_slot(B_SCALE, t) for t in range(nn)]
    b_on = [on_schedule.fill_slot(B_SCALE, t) for t in range(nn)]
    if shape == (2, 2):
        # The guarded case: A scale already owns the middle slots, so only B moves.
        assert scale_off == scale_on == [1, 2]
        assert b_off == [0, 3] and b_on == [1, 2]
    else:
        # Outside 2x2 the flag is inert -- a config that sets it here tests nothing.
        assert scale_off == scale_on and b_off == b_on


def test_epilogue_groups_stop_adding_slack_after_a_drain_producer():
    tc = _Config(WaitCommitScheme.PER_OP, (5, 5, 2, 2), 0, scales=False)
    K = (tc.minimum_num_k() + 3) * tc.BLOCK_K
    two, none = Schedule(tc, tc, K, 2), Schedule(tc, tc, K, 0)
    first, second = two.drain_step(0, 0), two.drain_step(0, 1)
    # B's first N slot is selected in slot zero. Its depth-two producer moves past
    # the epilogue commit at drain iteration one; globally adding slack is unsafe.
    assert two.wait(first, 0) == none.wait(first, 0) + 2
    assert two.wait(second, 0) == none.wait(second, 0)


@pytest.mark.parametrize("scale_depths", [(0, 0), (-1, 0), (-7, -3), (1, 1)])
def test_absent_scale_ring_indices_ignore_invalid_unused_depths(scale_depths):
    from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
    from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant
    from aiter.ops.triton.moe import moe_op_gemm_gluon as host
    from op_tests.triton_tests.moe.test_moe_gemm_gluon_registers import (
        _register_config,
    )

    config = _register_config("bf16")
    config.update(
        NUM_LDS_BUFFER=0,
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=2,
        A_SCALE_NUM_BUFFER=scale_depths[0],
        B_SCALE_NUM_BUFFER=scale_depths[1],
    )
    tc = host._launch_spec(
        128,
        512,
        960,
        DtypeQuant.BF16,
        DtypeQuant.BF16,
        False,
        False,
        False,
        True,
        False,
        None,
        None,
        tuple((key, host._hashable(value)) for key, value in config.items()),
    )[-1]
    assert validate(tc, 512, 960)
    schedule = Schedule(tc, KernelLayout(tc), 960)
    assert maybe_unwrap(schedule.depth) == 2
    assert maybe_unwrap(schedule.register_period) == 1
    for component in (A_SCALE, B_SCALE):
        for step in (schedule.warmup_step(), schedule.loop_step(1)):
            for fill in (False, True):
                assert pipeline._index.fn(schedule, 7, step, component, fill) == 0
