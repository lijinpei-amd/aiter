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
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    OPERANDS,
    PAYLOAD,
    SCALE,
    A,
    _pipeline_peeled,
    _wait,
    pipeline_depth,
    pipeline_unroll,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import WaitCommitScheme
from op_tests.triton_tests.moe.test_moe_gemm_gluon_wait_commit import _ScheduleConfig


class _Config(_ScheduleConfig):
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
        scale_tiles=(1, 1),
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
        self.B_IN_REG = bool(register_mask & 1)
        self.packed = packed
        self.scale_steps = scale_steps or ((2, 2) if packed else (1, 1))
        self.scale_tiles = scale_tiles
        self.compiler_pipeline = compiler_pipeline
        self.read_mask = read_mask
        self.K_UNROLL = unroll
        self.UNROLL_EPILOGUE = unroll_epilogue
        self.SOFF_UNROLL = soff
        self.FROZEN_STEP = False
        self.SCHED_MODE = 0
        self.MINI_BLOCK_M = self.MINI_BLOCK_N = 1
        self.BLOCK_K = 128 if packed else 256
        self.mma_acc_dtype = None
        self.output_quant = None

    def num_buffers(self, operand, scale=False):
        return self.depths[(operand, bool(scale))]

    def buffer_live_span(self, operand, scale=False):
        return (self.num_buffers(operand, scale) - 1) * (
            self.scale_ratio_k_step(operand) if scale else 1
        ) + 1

    def active_components(self):
        return tuple(
            component
            for component in COMPONENTS
            if not component[1] or self.has_scale(component[0])
        )

    def component_via_lds(self, component):
        operand, is_scale = component
        return (
            self.scale_via_lds(operand) if is_scale else self.payload_via_lds(operand)
        )

    def component_in_reg(self, component):
        return not self.component_via_lds(component)

    def pipeline_register_period(self):
        return math.lcm(
            *(
                self.depths[component]
                * (self.scale_ratio_k_step(component[0]) if component[1] else 1)
                for component in self.active_components()
                if not self.component_via_lds(component)
            )
        )

    def pipeline_peeled(self):
        if not hasattr(self, "_peeled"):
            self._peeled = _pipeline_peeled(self)
        return self._peeled

    def num_k_slots_per_tile(self):
        return 1

    num_prefetch_k_slots = num_k_slots_per_tile

    def scale_ratio_k_step(self, operand):
        return self.scale_steps[operand] if self.has_scale(operand) else 1

    def scale_ratio_non_k_slot(self, operand):
        return self.scale_tiles[operand]

    def scale_read_k_slots(self, operand):
        return self.num_k_slots_per_tile() * self.scale_ratio_k_step(operand)

    def scale_cache_fragments(self, operand):
        return self.num_lds_slots_per_block_non_k(operand) * self.scale_read_k_slots(
            operand
        )

    def scale_hbm_steps(self, operand, steps, phase):
        ratio = self.scale_ratio_k_step(operand)
        return (steps + phase) // ratio * ratio

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

    def dot_result_fragment_layout(self):
        return None


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

    def load(self, component, ring, mini, pointer):
        tc = self.tc
        operand, is_scale = component
        tile = pointer
        ratio = tc.scale_ratio_k_step(operand) if is_scale else 1
        nonk = tc.scale_ratio_non_k_slot(operand) if is_scale else 1
        mk = tc.num_k_slots_per_tile()
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
                        self.current_read % tc.scale_ratio_k_step(operand)
                        or mini % tc.scale_ratio_non_k_slot(operand)
                    ):
                        continue
                    required.add((component, mini, self.current_read))
        return required

    def check_wait(self, tc, stage, slot, drain=False, epilogue_groups=0, phase=None):
        actual = _wait(tc, stage, slot, drain, epilogue_groups, phase)
        required = self.required(slot)
        if required:
            youngest = max(
                i
                for i, group in enumerate(self.committed)
                if required.intersection(group)
            )
            assert actual == len(self.committed) - youngest - 1, (
                self.current_read,
                slot,
                actual,
                youngest,
                self.committed,
            )
        else:
            assert actual is None
        return actual

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

    def dot(self, a, b, accumulator, mk, fc, tc, enabled, phase):
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
                            - scale.mini % tc.scale_ratio_non_k_slot(operand),
                            accumulator - accumulator % tc.scale_ratio_k_step(operand),
                        )
                    )
        token = a[0].mini, b[0].mini, accumulator
        assert token not in self.mfmas, "K tile accumulated twice"
        self.mfmas.add(token)
        return accumulator + 1


def _execute(tc, num_k, monkeypatch, epilogue_groups=2):
    machine = _Machine(tc, num_k)
    machine.runtime_ranges = []
    peeled = tc.pipeline_peeled()
    pc = SimpleNamespace(
        tuning_cfg=tc,
        func_cfg=tc,
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
        num_k=num_k,
    )
    step_fn, fill_fn = pipeline._step.fn, pipeline._fill_slot.fn
    read_tile_fn = pipeline._read_tile.fn
    read_scale_tile_fn = pipeline._read_scale_tile.fn

    def step(*args, **kwargs):
        machine.current_read = args[4]
        machine.current_in_loop = kwargs.get("IN_LOOP", False)
        machine.current_drain = kwargs.get("DRAIN", False)
        machine.steps.append((args[4], machine.current_in_loop))
        return step_fn(*args, **kwargs)

    def fill(*args, **kwargs):
        machine.current_read = args[3]
        machine.current_in_loop = kwargs.get(
            "IN_LOOP", args[9] if len(args) > 9 else False
        )
        machine.current_drain = kwargs.get("DRAIN", args[5] if len(args) > 5 else False)
        return fill_fn(*args, **kwargs)

    def read_tile(*args, **kwargs):
        component = (args[4], PAYLOAD)
        if tc.component_in_reg(component):
            machine.events.append(
                (
                    machine.current_read,
                    machine.current_in_loop,
                    machine.current_drain,
                    "select",
                    component,
                    machine.current_read,
                    args[3],
                )
            )
        return read_tile_fn(*args, **kwargs)

    def read_scale_tile(*args, **kwargs):
        operand = args[4]
        component = (operand, SCALE)
        if tc.component_in_reg(component):
            ratio = tc.scale_ratio_k_step(operand)
            machine.events.append(
                (
                    machine.current_read,
                    machine.current_in_loop,
                    machine.current_drain,
                    "select",
                    component,
                    machine.current_read - machine.current_read % ratio,
                    args[3],
                )
            )
        return read_scale_tile_fn(*args, **kwargs)

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
                    * (
                        tc.scale_cache_fragments(component[0])
                        if component[1]
                        else (tc.nm if component[0] == A else tc.nn)
                        * tc.num_k_slots_per_tile()
                    )
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
            "_replace_tile",
            "_rotate",
            "_rotate_buffers",
            "_advance",
            "_read_tile",
            "_read_scale_tile",
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
        patch.setattr(pipeline, "_read_tile", read_tile)
        patch.setattr(pipeline, "_read_scale_tile", read_scale_tile)
        patch.setattr(pipeline, "_init_buffers", init_buffers)
        patch.setattr(pipeline, "_pipeline_peeled", lambda _: peeled)
        patch.setattr(pipeline, "_wait", machine.check_wait)
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
        main = num_k - pipeline_depth(tc)
        assert regs.acc == (main,) * (tc.nm * tc.nn)
        for _ in range(epilogue_groups):
            machine.commit_buffer_load()
        regs = pipeline._drain_buffered_pipeline.fn(
            pc, pointers, buffers, regs, num_k, epilogue_groups
        )
        assert regs.acc == (num_k - 1,) * (tc.nm * tc.nn)
        assert pipeline._last_mfma.fn(pc, regs) == (num_k,) * (tc.nm * tc.nn)

    expected = {
        (component, mini, tile)
        for component in tc.active_components()
        for mini in range(
            0,
            tc.nm if component[0] == A else tc.nn,
            tc.scale_ratio_non_k_slot(component[0]) if component[1] else 1,
        )
        for tile in range(
            0,
            num_k,
            tc.scale_ratio_k_step(component[0]) if component[1] else 1,
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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
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
            {"packed": True, "scale_steps": (4, 4), "scale_tiles": (2, 2)},
        ),
        (
            (2, 2, 2, 2),
            6,
            {
                "packed": True,
                "scale_steps": (8, 8),
                "scale_tiles": (2, 2),
                "soff": True,
            },
        ),
        (
            (3, 2, 2, 3),
            4,
            {
                "packed": True,
                "scale_steps": (1, 4),
                "scale_tiles": (1, 2),
                "middle": True,
            },
        ),
        (
            (2, 3, 3, 2),
            2,
            {
                "packed": True,
                "scale_steps": (4, 1),
                "scale_tiles": (2, 1),
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
    minimum = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc)
    quantum = math.lcm(*tc.scale_steps)
    first = math.ceil(minimum / quantum) * quantum
    for num_k in range(first, first + math.lcm(quantum, pipeline_unroll(tc)), quantum):
        _execute(tc, num_k, monkeypatch)
    # Re-enter the same static register-ring mapping from another runtime loop body.
    _execute(tc, first + 2 * pipeline_unroll(tc), monkeypatch)


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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
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
        minimum = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc)
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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
    machine = _execute(tc, num_k, monkeypatch)

    in_loop_reads = _assert_direct_b_stage_barriers(machine)
    assert len(in_loop_reads) == pipeline_unroll(tc) == 3
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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
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
    assert len(in_loop_reads) == pipeline_unroll(tc) == 6
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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
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
    num_k = pipeline_depth(tc) + tc.pipeline_peeled() + pipeline_unroll(tc) + 1
    machine = _execute(tc, num_k, monkeypatch)

    assert any(in_loop for _, in_loop in machine.steps)
    assert _sched_barrier_stream(machine) == [(0, False, False, 0)]


@pytest.mark.parametrize(
    "scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name
)
@pytest.mark.parametrize("register_mask", range(16))
def test_peeled_wait_prefix_is_minimal(scheme, register_mask):
    tc = _Config(scheme, (2, 4, 6, 3), register_mask, middle=True)
    slots = (None,) if tc.commit_per_stage() else range(tc.nm * tc.nn)
    steady = tuple(_wait(tc, None, slot) for slot in slots)
    waits = [
        tuple(_wait(tc, x + 1, slot) for slot in slots)
        for x in range(pipeline_depth(tc) * 2)
    ]
    last_changed = max(
        (x for x, vector in enumerate(waits) if vector != steady), default=-1
    )
    assert tc.pipeline_peeled() == max(1, last_changed + 1)
    assert all(vector == steady for vector in waits[tc.pipeline_peeled() :])


def test_epilogue_groups_stop_adding_slack_after_a_drain_producer():
    tc = _Config(WaitCommitScheme.PER_OP, (5, 5, 2, 2), 0, scales=False)
    # B tile zero is read in slot zero. Its depth-two producer moves past the
    # epilogue commit at drain iteration one; globally adding slack is unsafe.
    assert _wait(tc, 0, 0, True, 2) == _wait(tc, 0, 0, True, 0) + 2
    assert _wait(tc, 1, 0, True, 2) == _wait(tc, 1, 0, True, 0)


@pytest.mark.parametrize("scale_depths", [(0, 0), (-1, 0), (-7, -3), (1, 1)])
def test_absent_scale_ring_indices_ignore_invalid_unused_depths(scale_depths):
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
    assert tc.validate(512, 960)
    assert pipeline_depth(tc) == 2
    assert tc.pipeline_register_period() == 1
    for component in (A_SCALE, B_SCALE):
        for in_loop in (False, True):
            for fill in (False, True):
                assert pipeline._index.fn(tc, 7, component, fill, 1, in_loop) == 0
