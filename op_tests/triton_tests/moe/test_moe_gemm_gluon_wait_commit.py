# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check emitted async groups against an independent CPU completion queue."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe import _lds
from aiter.ops.triton._gluon_kernels.gfx950.moe import _pipeline as buffered
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import Schedule
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    PAYLOAD,
    SCALE,
    A,
    B,
    WaitCommitScheme,
)


@dataclass
class _ScheduleConfig:
    """Small scheduling inputs, independent of MFMA layout/resource constraints.

    One duck-typed object stands in for the func config, the tuning config and the
    ``KernelLayout`` alike: it is passed wherever production passes any of the three.
    """

    nm: int
    nn: int
    WAIT_COMMIT_SCHEME: WaitCommitScheme
    SCALE_FILL_MID: bool
    scales: tuple = (True, True)
    scale_async: tuple = (True, True)
    payload_async: tuple = (True, True)
    a_scale_ratio: int = 1
    SOFF_UNROLL: bool = False
    K_UNROLL: int = 1
    UNROLL_EPILOGUE: bool = True
    BLOCK_K: int = 256
    output_quant = None
    has_gammas = False
    has_bias = False
    token_cache_modifier: str = ""
    token_scale_cache_modifier: str = ""
    expert_cache_modifier: str = ""
    expert_scale_cache_modifier: str = ""

    @property
    def func_cfg(self):
        return self

    def has_scale(self, idx):
        return self.scales[idx]

    @property
    def num_m_slots(self):
        return self.nm

    @property
    def num_n_slots(self):
        return self.nn

    def num_lds_slots_non_k(self, idx):
        return (self.nm, self.nn)[idx]

    def component_via_lds(self, component):
        operand, is_scale = component
        return (self.scale_async if is_scale else self.payload_async)[operand]

    def component_in_reg(self, component):
        return not self.component_via_lds(component)

    def preshuffled(self, idx, is_scale=False):
        return is_scale and idx == 0 and self.a_scale_ratio > 1

    def slot_nonk(self, idx):
        return 32

    def scale_slot_nonk(self, idx):
        return self.slot_nonk(idx) * (self.a_scale_ratio if idx == 0 else 1)

    @property
    def scale_block_k(self):
        return self.BLOCK_K // 32

    def scale_slot_k(self, idx):
        return 8

    def buffer_live_span(self, idx, scale=False):
        return self.num_buffers(idx, scale)

    def num_buffers(self, idx, scale=False):
        return 3

    def commit_per_op(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_OP

    def commit_per_slot(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_SLOT

    def commit_per_stage(self):
        return self.WAIT_COMMIT_SCHEME in (
            WaitCommitScheme.PER_STAGE_WARP_PIPELINE,
            WaitCommitScheme.PER_STAGE_WHOLE,
        )

    def commit_per_stage_warp_pipeline(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_STAGE_WARP_PIPELINE

    def commit_per_stage_whole(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_STAGE_WHOLE

    def ds_read_in_mfma(self, operand, scale=False):
        return False

    epilogue_inputs_via_lds = False

    def num_k_tiles(self, K):
        assert K % self.BLOCK_K == 0
        return K // self.BLOCK_K

    def make_schedule(self, num_k=64, epilogue_groups=0):
        """The build's schedule; the default K is long enough for any test depth."""
        return Schedule(self, self, num_k * self.BLOCK_K, epilogue_groups)


def _config(shape, scheme, middle, traffic):
    tc = _ScheduleConfig(*shape, scheme, middle)
    if traffic == "no-scales":
        tc.scales = (False, False)
    elif traffic == "a-direct-scale":
        tc.scale_async = (False, True)
    elif traffic == "b-direct-scale":
        tc.scale_async = (True, False)
    elif traffic == "direct-scales":
        tc.scale_async = (False, False)
    elif traffic == "shared-a-pair":
        tc.a_scale_ratio = 2
    elif traffic == "shared-a-whole":
        tc.a_scale_ratio = tc.nm
    elif traffic == "a-register-payload":
        tc.scales = (False, False)
        tc.payload_async = (False, True)
    elif traffic == "register-payloads":
        tc.scales = (False, False)
        tc.payload_async = (False, False)
    return tc


def _reference_slots(tc):
    # State the copy order independently of production's slot/group helpers.
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
    copies = [[] for _ in range(tc.nm * tc.nn)]
    reads = [set() for _ in copies]
    for slot, (component, tile) in enumerate(payloads):
        operand, is_scale = component
        assert not is_scale
        copies[slot].append((component, tile))
        if tc.component_via_lds(component):
            reads[slot].add((component, tile))
        scale_component = (operand, SCALE)
        if not (
            tc.has_scale(operand) and tc.component_via_lds(scale_component)
        ):
            continue
        owner = tile - tile % tc.a_scale_ratio if operand == A else tile
        if tile == owner:
            reads[slot].add((scale_component, owner))
            scale_slot = (
                1 + tile if tc.SCALE_FILL_MID and (tc.nm, tc.nn) == (2, 2) else slot
            )
            copies[scale_slot].append((scale_component, tile))
    # Within a slot, components issue in A, A-scale, B, B-scale order.
    component_rank = {
        A_PAYLOAD: 0,
        A_SCALE: 1,
        B_PAYLOAD: 2,
        B_SCALE: 3,
    }
    ordered = [tuple(sorted(slot, key=lambda copy: component_rank[copy[0]])) for slot in copies]
    return ordered, reads


def _reference_groups(tc, copies):
    def is_async(copy):
        component, _ = copy
        return tc.component_via_lds(component)

    asynchronous = [tuple(copy for copy in slot if is_async(copy)) for slot in copies]
    if tc.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_OP:
        return tuple(tuple((copy,) for copy in slot) for slot in asynchronous)
    if tc.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_SLOT:
        return tuple((slot,) for slot in asynchronous)
    return ((),) * (len(copies) - 1) + (
        (tuple(copy for slot in asynchronous for copy in slot),),
    )


class _Descriptor:
    def __init__(self, component, tc):
        self.component = component
        operand, _ = component
        self.ratio = tc.a_scale_ratio if component == A_SCALE else 1
        self.tiles = (tc.nm if operand == A else tc.nn) // self.ratio

    def index(self, index):
        stage, tile = divmod(maybe_unwrap(index), self.tiles)
        return stage, self.component, tile * self.ratio


class _LDSRecorder:
    """Run actual LDS copy guards; replace only the hardware copy operation."""

    def __init__(self, tc):
        self.func_cfg = self.tuning_cfg = self.layout = tc
        self.a_payload_lds_ptr = _Descriptor(A_PAYLOAD, tc)
        self.a_scale_lds_ptr = _Descriptor(A_SCALE, tc)
        self.b_payload_lds_ptr = _Descriptor(B_PAYLOAD, tc)
        self.b_scale_lds_ptr = _Descriptor(B_SCALE, tc)
        self.slot = 0
        self.pending = []
        self.copies = [[] for _ in range(tc.nm * tc.nn)]
        self.groups = [[] for _ in self.copies]

    def buffer_load_payload(self, operand, VIA_LDS, buffer, tile, *args):
        if VIA_LDS:
            _lds.LDSManager.buffer_load_payload.fn(
                self, operand, VIA_LDS, buffer, tile, *args
            )
        else:
            # A direct-register load is issued but owns no LDS async-copy group.
            self.copies[self.slot].append(((operand, PAYLOAD), tile))
            return (0,)

    def buffer_load_scale(self, operand, VIA_LDS, *args, **kwargs):
        if VIA_LDS:
            _lds.LDSManager.buffer_load_scale.fn(
                self, operand, VIA_LDS, *args, **kwargs
            )
        else:
            # A register load has no LDS destination and owns no async-copy group.
            return (0,)

    def copy(self, descriptor, base, offsets, modifier, soffset):
        stage, component, tile = descriptor
        assert stage == 2, "The emitter must index the requested LDS stage."
        self.copies[self.slot].append((component, tile))
        self.pending.append((component, tile))

    def commit_buffer_load(self):
        self.groups[self.slot].append(tuple(self.pending))
        self.pending.clear()


def _steady_step(tc):
    """A schedule and its runtime-remainder step: steady state at K phase zero.

    Every component fills there and nothing is unrolled, so the step's copies are the
    slot ownership itself, with dynamic ring indices.
    """
    tc.UNROLL_EPILOGUE = False
    schedule = tc.make_schedule()
    return schedule, schedule.remainder_step(0)


def _trace_emitter(tc, monkeypatch, defer_stage_commit=False):
    tc.ALLOW_B_VIA_LDS = True
    tc.num_buffers = lambda operand, scale=False: 3
    tc.num_k_slots = 1
    tc.operand_elem_ty = lambda operand: SimpleNamespace(primitive_bitwidth=8)
    sink = _LDSRecorder(tc)
    schedule, step = _steady_step(tc)
    pc = SimpleNamespace(
        lds_ptrs=sink,
        func_cfg=tc,
        tuning_cfg=tc,
        layout=tc,
        schedule=schedule,
        a_hbm_offs=(0,) * tc.nm,
        b_hbm_offs=(0,) * tc.nn,
        a_scale_hbm_offs=(0,) * tc.nm,
        b_scale_hbm_offs=(0,) * tc.nn,
        a_step=0,
        b_step=0,
        s_step=0,
        a_scale_stride_k=0,
        b_scale_stride_k=0,
    )
    pointers = SimpleNamespace(
        a_hbm_ptr=1,
        a_scale_hbm_ptr=2,
        b_hbm_ptr=3,
        b_scale_hbm_ptr=4,
    )
    buffers = (
        (0,) * (3 * tc.nm) if tc.component_in_reg(A_PAYLOAD) else (),
        (0,) * (3 * tc.nn) if tc.component_in_reg(B_PAYLOAD) else (),
        (0,) * (3 * tc.nm)
        if tc.has_scale(A) and tc.component_in_reg(A_SCALE)
        else (),
        (0,) * (3 * tc.nn)
        if tc.has_scale(B) and tc.component_in_reg(B_SCALE)
        else (),
    )

    def check_assumption(value):
        assert value

    with monkeypatch.context() as patch:
        patch.setattr(_lds, "_buffer_load_to_lds", sink.copy)
        patch.setattr(buffered, "_index", buffered._index.fn)
        patch.setattr(buffered, "_replace_fragments", buffered._replace_fragments.fn)
        patch.setattr(buffered.gl, "static_range", range)
        patch.setattr(buffered.gl, "assume", check_assumption)
        for slot in range(tc.nm * tc.nn):
            sink.slot = slot
            buffers = buffered._fill_slot.fn(
                pc,
                pointers,
                buffers,
                0,
                step,
                slot % tc.nm,
                slot // tc.nm,
                MARK_TILE=not defer_stage_commit,
            )
    if defer_stage_commit:
        assert not any(sink.groups)
        sink.commit_buffer_load()
    assert not sink.pending
    return tuple(tuple(slot) for slot in sink.copies), tuple(
        tuple(slot) for slot in sink.groups
    )


@pytest.mark.parametrize("shape", [(2, 2), (4, 2), (2, 4), (4, 4)])
@pytest.mark.parametrize(
    "scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name
)
@pytest.mark.parametrize(
    "middle", [False, True], ids=["paired-scales", "middle-scales"]
)
@pytest.mark.parametrize(
    "traffic",
    [
        "both-scales",
        "no-scales",
        "a-direct-scale",
        "b-direct-scale",
        "direct-scales",
        "shared-a-pair",
        "shared-a-whole",
        "a-register-payload",
        "register-payloads",
    ],
)
def test_emitted_groups_and_waits_retire_required_copies(
    shape, scheme, middle, traffic, monkeypatch
):
    tc = _config(shape, scheme, middle, traffic)
    copies, reads = _reference_slots(tc)
    expected = _reference_groups(tc, copies)
    emitted_copies, emitted_groups = _trace_emitter(tc, monkeypatch)
    assert emitted_copies == tuple(copies)
    assert emitted_groups == expected
    # Tie the schedule's commit plan to the traced emitter, not a parallel model. At
    # a steady-state step every component is active, which is the situation this
    # reference describes.
    schedule, step = _steady_step(tc)
    assert (
        tuple(schedule.commits(step, slot) for slot in range(tc.nm * tc.nn))
        == expected
    )
    for slot, operations in enumerate(copies):
        # Ownership keeps direct-register destinations because the schedule still
        # issues them; only scales are dropped from the async view, since a register
        # scale has no commit group while a register payload still occupies its slot.
        nominal = tuple(
            non_k_slot
            if (
                not component[1]
                or non_k_slot is None
                or schedule.via_lds(component)
            )
            else None
            for component, non_k_slot in zip(COMPONENTS, schedule.owned(slot))
        )
        assert nominal == tuple(
            dict(operations).get(component) for component in COMPONENTS
        )

    # The wait plan is not re-derived here. It is covered end to end by two
    # independent FIFO oracles that model every pipeline region rather than a
    # nominal steady-state step: `_Machine` in
    # test_moe_gemm_gluon_buffered_schedule.py, which drives the real emitter, and
    # `_audit_buffer_schedule` in test_moe_gemm_gluon_config.py. This test's job is
    # the ownership/grouping half of the contract, above.


@pytest.mark.parametrize(
    "scheme",
    [
        WaitCommitScheme.PER_STAGE_WARP_PIPELINE,
        WaitCommitScheme.PER_STAGE_WHOLE,
    ],
)
def test_stage_commit_can_close_after_the_slot_walk(scheme, monkeypatch):
    tc = _config((4, 2), scheme, False, "shared-a-pair")
    copies, _ = _reference_slots(tc)
    _, emitted = _trace_emitter(tc, monkeypatch, defer_stage_commit=True)
    assert emitted == _reference_groups(tc, copies)


@pytest.mark.parametrize(
    "scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name
)
def test_register_only_slots_need_no_cooperative_fence(scheme, monkeypatch):
    from aiter.ops.triton._gluon_kernels.gfx950.moe import _pipeline as pipeline

    waits, barriers = [], []
    tc = _config((4, 2), scheme, False, "register-payloads")
    tc.num_k_slots = 1
    tc.warp_pipeline_compiler = lambda: False
    tc.commit_per_stage_warp_pipeline = lambda: (
        scheme == WaitCommitScheme.PER_STAGE_WARP_PIPELINE
    )
    tc.commit_per_stage_whole = lambda: scheme == WaitCommitScheme.PER_STAGE_WHOLE
    tc.SCHED_MODE = 0
    tc.BLOCK_K = 256
    tc.ds_read_in_mfma = lambda *args: False
    schedule, step = _steady_step(tc)
    pc = SimpleNamespace(
        tuning_cfg=tc,
        layout=tc,
        func_cfg=tc,
        schedule=schedule,
        lds_ptrs=SimpleNamespace(
            wait_buffer_load_groups=waits.append, commit_buffer_load=lambda: None
        ),
    )
    regs = SimpleNamespace(
        a_payload=(), a_scale=(), b_payload=(), b_scale=(), acc=(0,) * 8
    )
    monkeypatch.setattr(pipeline.gl, "barrier", lambda: barriers.append(True))
    monkeypatch.setattr(pipeline.gl, "static_range", range)
    monkeypatch.setattr(pipeline, "_read_slot", lambda *args: ((), (), (), ()))
    monkeypatch.setattr(
        pipeline, "_fill_slot", lambda pc, ptrs, buffers, *args, **kwargs: buffers
    )
    monkeypatch.setattr(pipeline, "_advance", lambda pc, ptrs, *args: ptrs)
    monkeypatch.setattr(pipeline, "_rotate_buffers", lambda buffers, *args: buffers)
    monkeypatch.setattr(pipeline, "_take_operand_pairs", lambda *args: ())
    monkeypatch.setattr(pipeline, "_PipelineRegFragments", lambda *args: args)
    monkeypatch.setattr(pipeline, "_maybe_block_dot", lambda *args: 0)
    pipeline._step_live.fn(pc, None, ((), (), (), ()), regs, 1, step)
    assert not waits
    assert not barriers


@pytest.mark.parametrize(
    "scheme",
    [
        WaitCommitScheme.PER_STAGE_WARP_PIPELINE,
        WaitCommitScheme.PER_STAGE_WHOLE,
    ],
    ids=lambda scheme: scheme.name,
)
@pytest.mark.parametrize("compiler", [False, True], ids=["none", "compiler"])
@pytest.mark.parametrize(
    "dot,in_loop,drain",
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
    ids=["prefetch", "loop", "drain"],
)
def test_pipeline_stage_commit_boundaries(
    scheme, compiler, dot, in_loop, drain, monkeypatch
):
    """Execute the unified step and record ordering at its actual region borders."""
    from aiter.ops.triton._gluon_kernels.gfx950.moe import _pipeline as pipeline

    events = []
    tc = _config((2, 2), scheme, False, "no-scales")
    tc.num_k_slots = 1
    tc.warp_pipeline_compiler = lambda: compiler
    tc.commit_per_stage_warp_pipeline = lambda: (
        scheme == WaitCommitScheme.PER_STAGE_WARP_PIPELINE
    )
    tc.commit_per_stage_whole = lambda: scheme == WaitCommitScheme.PER_STAGE_WHOLE
    tc.SCHED_MODE = 0
    tc.BLOCK_K = 256
    tc.ds_read_in_mfma = lambda *args: False
    schedule = tc.make_schedule(epilogue_groups=2)
    if not dot:
        step = schedule.warmup_step()
    elif in_loop:
        step = schedule.loop_step(0)
    else:
        step = schedule.drain_step(0, 0)
    assert schedule.dot(step) == dot and schedule.in_loop(step) == in_loop
    assert schedule.is_drain(step) == drain
    wait = schedule.wait(step)
    assert wait is not None, "a per-tile step that reads LDS waits once, at its head"
    pc = SimpleNamespace(
        func_cfg=tc,
        tuning_cfg=tc,
        layout=tc,
        schedule=schedule,
        lds_ptrs=SimpleNamespace(
            wait_buffer_load_groups=lambda count: events.append(("wait", count)),
            commit_buffer_load=lambda: events.append(("commit",)),
        ),
    )
    regs = SimpleNamespace(
        a_payload=(), a_scale=(), b_payload=(), b_scale=(), acc=(0,) * 4
    )

    @contextmanager
    def stage(name):
        events.append(("enter", name))
        yield
        events.append(("exit", name))

    def pick_stage(enabled):
        assert bool(maybe_unwrap(enabled)) == (compiler and in_loop)
        return stage

    def buffer_load(pc, ptrs, buffers, *args, **kwargs):
        assert kwargs["MARK_TILE"] is False
        events.append(("load",))
        return buffers

    monkeypatch.setattr(pipeline, "pick_stage", pick_stage)
    monkeypatch.setattr(pipeline.gl, "static_range", range)
    monkeypatch.setattr(pipeline.gl.amd.cdna4, "sched_barrier", lambda mask: None)
    monkeypatch.setattr(pipeline, "_fill_slot", buffer_load)
    monkeypatch.setattr(pipeline, "_read_slot", lambda *args: ((), (), (), ()))
    monkeypatch.setattr(pipeline, "_take_operand_pairs", lambda *args: ())
    monkeypatch.setattr(pipeline, "_PipelineRegFragments", lambda *args: args)
    monkeypatch.setattr(pipeline, "_rotate_buffers", lambda buffers, *args: buffers)
    monkeypatch.setattr(pipeline, "_advance", lambda pc, ptrs, *args: ptrs)
    monkeypatch.setattr(
        pipeline, "_maybe_block_dot", lambda *args: events.append(("dot",))
    )
    pipeline._step_live.fn(pc, None, ((), (), (), ()), regs, 1, step)
    assert [event for event in events if event[0] == "wait"] == [("wait", wait)]
    assert events[0] == ("wait", wait)
    # Stage/slot commit schemes retain empty groups during the drain. The FIFO
    # accounting models those exact commits, including their region placement.
    assert events.count(("commit",)) == 1
    commit = events.index(("commit",))
    assert events[:commit].count(("load",)) == 4
    if scheme == WaitCommitScheme.PER_STAGE_WARP_PIPELINE:
        assert events[commit - 1] == ("load",)
        assert events[commit + 1] == ("exit", "mem")
        assert events[commit + 2] == ("enter", "mfma")
    else:
        assert events[commit - 2] == ("exit", "mfma")
        assert events[commit - 1] == ("enter", "commit")
        assert events[commit + 1] == ("exit", "commit")
        assert commit == len(events) - 2
