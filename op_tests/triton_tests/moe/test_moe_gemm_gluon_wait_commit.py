# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check emitted async groups against an independent CPU completion queue."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe import _lds
from aiter.ops.triton._gluon_kernels.gfx950.moe import moe_gemm as kernel
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import (
    _buffer_load_group_schedule,
    _buffer_load_groups,
    _buffer_load_ops,
    _buffer_load_wait,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import WaitCommitScheme


@dataclass
class _ScheduleConfig:
    """Small scheduling inputs, independent of MFMA layout/resource constraints."""

    nm: int
    nn: int
    WAIT_COMMIT_SCHEME: WaitCommitScheme
    SCALE_FILL_MID: bool
    scales: tuple = (True, True)
    scale_async: tuple = (True, True)
    payload_async: tuple = (True, True)
    a_scale_ratio: int = 1
    SOFF_UNROLL: bool = False
    token_mod: str = ""
    token_scale_mod: str = ""
    expert_mod: str = ""
    expert_scale_mod: str = ""

    @property
    def func_cfg(self):
        return self

    def has_scale(self, idx):
        return self.scales[idx]

    def a_has_scale(self):
        return self.scales[0]

    def b_has_scale(self):
        return self.scales[1]

    def num_mini_m(self):
        return self.nm

    def num_mini_n(self):
        return self.nn

    def num_lds_tiles(self, idx):
        return (self.nm, self.nn)[idx]

    def payload_via_lds(self, idx):
        return self.payload_async[idx]

    def scale_via_lds(self, idx):
        return self.scale_async[idx]

    def scale_shuffled(self, idx):
        return idx == 0 and self.a_scale_ratio > 1

    def scale_tile_ratio_a(self):
        return self.a_scale_ratio

    def commit_per_op(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_OP

    def commit_per_slot(self):
        return self.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_SLOT

    def commit_per_stage(self):
        return self.WAIT_COMMIT_SCHEME in (
            WaitCommitScheme.PER_STAGE_WARP_PIPELINE, WaitCommitScheme.PER_STAGE_WHOLE,
        )


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
    elif traffic == "a-synchronous":
        tc.scales = (False, False)
        tc.payload_async = (False, True)
    elif traffic == "synchronous":
        tc.scales = (False, False)
        tc.payload_async = (False, False)
    return tc


def _reference_slots(tc):
    # State the copy order independently of production's slot/group helpers.
    if (tc.nm, tc.nn) == (2, 2):
        payloads = [(2, 0), (0, 0), (0, 1), (2, 1)]
    else:
        payloads = sorted(
            [(0, tile) for tile in range(tc.nm)]
            + [(2, tile) for tile in range(tc.nn)],
            key=lambda copy: (copy[1], copy[0]),
        )
    copies = [[] for _ in range(tc.nm * tc.nn)]
    reads = [set() for _ in copies]
    for slot, (kind, tile) in enumerate(payloads):
        operand = kind // 2
        copies[slot].append((kind, tile))
        if tc.payload_async[operand]:
            reads[slot].add((kind, tile))
        if not (tc.scales[operand] and tc.scale_async[operand]):
            continue
        owner = tile - tile % tc.a_scale_ratio if operand == 0 else tile
        reads[slot].add((kind + 1, owner))
        if tile == owner:
            scale_slot = 1 + tile if tc.SCALE_FILL_MID and (tc.nm, tc.nn) == (2, 2) else slot
            copies[scale_slot].append((kind + 1, tile))
    # Within a slot the four copy kinds issue in A, A-scale, B, B-scale order.
    return [tuple(sorted(slot)) for slot in copies], reads


def _reference_groups(tc, copies):
    asynchronous = [
        tuple(copy for copy in slot if copy[0] % 2 or tc.payload_async[copy[0] // 2])
        for slot in copies
    ]
    if tc.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_OP:
        return tuple(tuple((copy,) for copy in slot) for slot in asynchronous)
    if tc.WAIT_COMMIT_SCHEME == WaitCommitScheme.PER_SLOT:
        return tuple((slot,) for slot in asynchronous)
    return ((),) * (len(copies) - 1) + ((tuple(copy for slot in asynchronous for copy in slot),),)


class _Descriptor:
    def __init__(self, kind, tc):
        self.kind = kind
        self.ratio = tc.a_scale_ratio if kind == 1 else 1
        self.tiles = (tc.nm if kind < 2 else tc.nn) // self.ratio

    def index(self, index):
        stage, tile = divmod(unwrap(index), self.tiles)
        return stage, self.kind, tile * self.ratio


class _LDSRecorder:
    """Run actual LDS copy guards; replace only the hardware copy operation."""

    def __init__(self, tc):
        self.func_cfg = self.tuning_cfg = tc
        self.a_payload_lds_ptr = _Descriptor(0, tc)
        self.a_scale_lds_ptr = _Descriptor(1, tc)
        self.b_payload_lds_ptr = _Descriptor(2, tc)
        self.b_scale_lds_ptr = _Descriptor(3, tc)
        self.slot = 0
        self.pending = []
        self.copies = [[] for _ in range(tc.nm * tc.nn)]
        self.groups = [[] for _ in self.copies]

    def buffer_load_a_payload(self, *args):
        _lds.LDSManager.buffer_load_a_payload.fn(self, *args)

    def buffer_load_a_scale(self, *args):
        _lds.LDSManager.buffer_load_a_scale.fn(self, *args)

    def buffer_load_b_payload(self, *args):
        _lds.LDSManager.buffer_load_b_payload.fn(self, *args)

    def buffer_load_b_scale(self, *args):
        _lds.LDSManager.buffer_load_b_scale.fn(self, *args)

    def copy(self, descriptor, base, offsets, asynchronous, modifier, soffset):
        stage, kind, tile = descriptor
        assert stage == 2, "The emitter must index the requested LDS stage."
        self.copies[self.slot].append((kind, tile))
        if asynchronous:
            self.pending.append((kind, tile))

    def commit_buffer_load(self):
        self.groups[self.slot].append(tuple(self.pending))
        self.pending.clear()


def _trace_emitter(tc, monkeypatch, defer_stage_commit=False):
    sink = _LDSRecorder(tc)
    pc = SimpleNamespace(
        lds_ptrs=sink, func_cfg=tc, tuning_cfg=tc,
        a_hbm_offs=(0,) * tc.nm, b_hbm_offs=(0,) * tc.nn,
        a_scale_hbm_offs=(0,) * tc.nm, b_scale_hbm_offs=(0,) * tc.nn,
        a_step=0, b_step=0, s_step=0,
        a_scale_stride_k=0, b_scale_stride_k=0,
    )
    pointers = SimpleNamespace(
        a_hbm_ptr=1, a_scale_hbm_ptr=2, b_hbm_ptr=3, b_scale_hbm_ptr=4,
    )
    with monkeypatch.context() as patch:
        patch.setattr(_lds, "_buffer_load_to_lds", sink.copy)
        patch.setattr(kernel, "_opt_at", kernel._opt_at.fn)
        for slot in range(tc.nm * tc.nn):
            sink.slot = slot
            kernel._buffer_load.fn(
                pc, pointers, 2, slot % tc.nm, slot // tc.nm,
                STAGE_MARK=not defer_stage_commit,
            )
    if defer_stage_commit:
        assert not any(sink.groups)
        sink.commit_buffer_load()
    assert not sink.pending
    return tuple(tuple(slot) for slot in sink.copies), tuple(tuple(slot) for slot in sink.groups)


@pytest.mark.parametrize("shape", [(2, 2), (4, 2), (2, 4), (4, 4)])
@pytest.mark.parametrize("scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name)
@pytest.mark.parametrize("middle", [False, True], ids=["paired-scales", "middle-scales"])
@pytest.mark.parametrize(
    "traffic",
    ["both-scales", "no-scales", "a-direct-scale", "b-direct-scale", "direct-scales",
     "shared-a-pair", "shared-a-whole", "a-synchronous", "synchronous"],
)
def test_emitted_groups_and_waits_retire_required_copies(shape, scheme, middle, traffic, monkeypatch):
    tc = _config(shape, scheme, middle, traffic)
    copies, reads = _reference_slots(tc)
    expected = _reference_groups(tc, copies)
    emitted_copies, emitted_groups = _trace_emitter(tc, monkeypatch)
    assert emitted_copies == tuple(copies)
    assert emitted_groups == expected
    assert _buffer_load_group_schedule(tc) == expected
    flat = tuple(group for slot in expected for group in slot)
    assert _buffer_load_groups(tc) == flat
    for slot, operations in enumerate(copies):
        actual = _buffer_load_ops(tc, slot % tc.nm, slot // tc.nm)
        assert actual == tuple(dict(operations).get(kind) for kind in range(4))

    # The oracle is a literal FIFO of committed groups, with unique stage-tagged
    # copy tokens. It never uses production's group-count or wait-count formula.
    for between in (0, 1, 2):
        for filling in (False, True):
            for slack in (0, 1, 2):
                queue = [
                    {(stage, kind, tile) for kind, tile in group}
                    for stage in range(between + 1) for group in flat
                ]
                queue.extend({("epilogue", index)} for index in range(slack))
                committed = list(queue)
                completed = set()
                for slot, required in enumerate(reads):
                    target = {(0, kind, tile) for kind, tile in required}
                    wait = _buffer_load_wait(tc, slot % tc.nm, slot // tc.nm, between, filling)
                    if wait is not None:
                        wait += slack
                        if tc.commit_per_stage():
                            assert slot == 0
                            assert wait == between + slack
                        else:
                            # Any larger count could leave the last required copy
                            # pending; any smaller one unnecessarily drains newer work.
                            newest_required = max(i for i, group in enumerate(committed) if group & target)
                            assert wait == len(committed[newest_required + 1:])
                        while len(queue) > wait:
                            completed.update(queue.pop(0))
                    elif not tc.commit_per_stage():
                        assert not target
                    assert target <= completed, (slot, required, wait, queue)
                    if filling:
                        for group in expected[slot]:
                            issued = {(between + 1, kind, tile) for kind, tile in group}
                            queue.append(issued)
                            committed.append(issued)


@pytest.mark.parametrize("scheme", [
    WaitCommitScheme.PER_STAGE_WARP_PIPELINE, WaitCommitScheme.PER_STAGE_WHOLE,
])
def test_stage_commit_can_close_after_the_slot_walk(scheme, monkeypatch):
    tc = _config((4, 2), scheme, False, "shared-a-pair")
    copies, _ = _reference_slots(tc)
    _, emitted = _trace_emitter(tc, monkeypatch, defer_stage_commit=True)
    assert emitted == _reference_groups(tc, copies)


@pytest.mark.parametrize("scheme", list(WaitCommitScheme), ids=lambda scheme: scheme.name)
def test_synchronous_reads_keep_a_cooperative_fence(scheme, monkeypatch):
    tc = _config((4, 2), scheme, False, "synchronous")
    waits, barriers = [], []
    pc = SimpleNamespace(
        tuning_cfg=tc,
        lds_ptrs=SimpleNamespace(wait_buffer_load_groups=waits.append),
    )
    monkeypatch.setattr(kernel.gl, "barrier", lambda: barriers.append(True))
    kernel.wait_per_stage.fn(pc, 1)
    for ni in range(tc.nn):
        for mi in range(tc.nm):
            kernel.wait_per_slot.fn(pc, mi, ni, 1, True)
    if tc.commit_per_stage():
        assert waits == [1]
        assert not barriers
    else:
        assert not waits
        assert len(barriers) == tc.nm + tc.nn


@pytest.mark.parametrize("scheme", [
    WaitCommitScheme.PER_STAGE_WARP_PIPELINE, WaitCommitScheme.PER_STAGE_WHOLE,
], ids=lambda scheme: scheme.name)
@pytest.mark.parametrize("compiler", [False, True], ids=["none", "compiler"])
@pytest.mark.parametrize("fill,read,dot,in_loop", [
    (True, False, False, False),
    (True, True, False, False),
    (True, True, True, True),
    (False, True, True, False),
], ids=["fill", "prefetch", "loop", "drain"])
def test_pipeline_stage_commit_boundaries(
    scheme, compiler, fill, read, dot, in_loop, monkeypatch
):
    """Execute the live step and record ordering around its actual region borders."""
    events = []
    tc = _config((2, 2), scheme, False, "no-scales")
    tc.num_mini_k = tc.num_prefetch_mini = lambda: 1
    tc.warp_pipeline_compiler = lambda: compiler
    tc.warp_pipeline_manual = lambda: False
    tc.commit_per_stage_warp_pipeline = lambda: scheme == WaitCommitScheme.PER_STAGE_WARP_PIPELINE
    tc.commit_per_stage_whole = lambda: scheme == WaitCommitScheme.PER_STAGE_WHOLE
    tc.ds_read_in_mfma = lambda *args, **kwargs: False
    tc.scale_k_phase = lambda phase: 0
    tc.NUM_LDS_BUFFER = 3
    tc.SCHED_MODE = 0
    pc = SimpleNamespace(
        func_cfg=tc, tuning_cfg=tc,
        lds_ptrs=SimpleNamespace(
            wait_buffer_load_groups=lambda count: events.append(("wait", count)),
            commit_buffer_load=lambda: events.append(("commit",)),
        ),
    )
    pointers = SimpleNamespace(a_scale_hbm_ptr=None, b_scale_hbm_ptr=None)
    regs = SimpleNamespace(a_payload=(), a_scale=(), b_payload=(), b_scale=(), acc=(0,) * 4)

    @contextmanager
    def stage(name):
        events.append(("enter", name))
        yield
        events.append(("exit", name))

    def pick_stage(enabled):
        assert bool(unwrap(enabled)) == (compiler and read and in_loop)
        return stage

    def buffer_load(*args, **kwargs):
        assert kwargs["STAGE_MARK"] is False
        events.append(("load",))

    monkeypatch.setattr(kernel, "pick_stage", pick_stage)
    monkeypatch.setattr(kernel.gl, "static_range", range)
    monkeypatch.setattr(kernel.gl, "static_assert", lambda value, message: None)
    monkeypatch.setattr(kernel.gl.amd.cdna4, "sched_barrier", lambda mask: None)
    monkeypatch.setattr(kernel, "wait_per_stage", kernel.wait_per_stage.fn)
    monkeypatch.setattr(kernel, "wait_per_slot", kernel.wait_per_slot.fn)
    monkeypatch.setattr(kernel, "_buffer_load", buffer_load)
    monkeypatch.setattr(kernel, "_ds_read", lambda *args: ((), ()))
    monkeypatch.setattr(kernel, "_merge_ds_read_frags", lambda *args: ())
    monkeypatch.setattr(kernel, "_ds_read_tails", lambda *args: ((), ()))
    monkeypatch.setattr(kernel, "_take_reg_pairs", lambda *args: ())
    monkeypatch.setattr(kernel, "_make_reg_fragments", lambda *args: args)
    monkeypatch.setattr(kernel, "_advance_hbm_ptrs", lambda pc, ptrs, *args: ptrs)
    monkeypatch.setattr(kernel, "_advance_scale_hbm_ptrs", lambda pc, ptrs: ptrs)
    monkeypatch.setattr(kernel, "_maybe_block_dot", lambda *args: events.append(("dot",)))

    kernel._pipeline_step_impl.fn(
        pc, pointers, regs, 2, 0, 1, fill, read, in_loop, dot, WAIT_SLACK=2,
    )
    assert [event for event in events if event[0] == "wait"] == ([("wait", 3)] if read else [])
    if read:
        assert events[0] == ("wait", 3), "Stage waits must precede the entire slot walk."
    assert events.count(("commit",)) == int(fill)
    if fill:
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
