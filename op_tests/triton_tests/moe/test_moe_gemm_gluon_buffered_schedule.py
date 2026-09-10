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
from aiter.ops.triton._gluon_kernels.gfx950.moe._pipeline import (
    _pipeline_peeled,
    _wait,
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
        soff=False,
        scale_steps=None,
        scale_tiles=(1, 1),
    ):
        super().__init__(*shape, scheme, middle)
        self.depths = depths  # A, A scale, B, B scale.
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
        self.read_mask = read_mask
        self.unroll = unroll
        self.SOFF_UNROLL = soff
        self.FROZEN_STEP = False
        self.SCHED_MODE = 0
        self.MINI_BLOCK_M = self.MINI_BLOCK_N = 1
        self.MINI_BLOCK_K = 128 if packed else 256
        self.BLOCK_K = self.MINI_BLOCK_K * self.num_k_slots_per_tile()
        self.mma_acc_dtype = None
        self.output_quant = None

    def num_buffers(self, operand, scale=False):
        return self.depths[operand * 2 + bool(scale)]

    def pipeline_depth(self):
        return max(
            self.component_span(kind // 2, bool(kind % 2))
            for kind in self.active_kinds()
        )

    def component_span(self, operand, scale=False):
        return (self.num_buffers(operand, scale) - 1) * (
            self.scale_step_ratio(operand) if scale else 1
        ) + 1

    def active_kinds(self):
        return tuple(
            kind for kind in range(4) if not kind % 2 or self.has_scale(kind // 2)
        )

    def is_async(self, kind):
        return (
            self.scale_via_lds(kind // 2)
            if kind % 2
            else self.payload_via_lds(kind // 2)
        )

    def pipeline_register_period(self):
        return math.lcm(
            *(
                self.depths[kind]
                * (self.scale_step_ratio(kind // 2) if kind % 2 else 1)
                for kind in self.active_kinds()
                if not self.is_async(kind)
            )
        )

    def pipeline_unroll(self):
        period = math.lcm(self.pipeline_register_period(), *self.scale_steps)
        return math.ceil(self.unroll / period) * period

    def pipeline_peeled(self):
        if not hasattr(self, "_peeled"):
            self._peeled = _pipeline_peeled(self)
        return self._peeled

    def num_k_slots_per_tile(self):
        return 1 if self.packed else 2

    num_prefetch_k_slots = num_k_slots_per_tile

    def scale_step_ratio(self, operand):
        return self.scale_steps[operand] if self.has_scale(operand) else 1

    def scale_tile_ratio(self, operand):
        return self.scale_tiles[operand]

    def scale_read_k_slots(self, operand):
        return self.num_k_slots_per_tile() * self.scale_step_ratio(operand)

    def scale_cache_fragments(self, operand):
        return self.num_lds_slots_per_block_non_k(operand) * self.scale_read_k_slots(
            operand
        )

    def scale_hbm_steps(self, operand, steps, phase):
        ratio = self.scale_step_ratio(operand)
        return (steps + phase) // ratio * ratio

    def operand_elem_ty(self, operand):
        return SimpleNamespace(primitive_bitwidth=8)

    def ds_read_in_mfma(self, operand, scale=False):
        return bool(self.read_mask & (1 << (operand + 2 * scale)))

    def warp_pipeline_compiler(self):
        return True

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
    kind: int
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

    def load(self, kind, ring, mini, pointer):
        tc = self.tc
        tile = pointer
        ratio = tc.scale_step_ratio(kind // 2) if kind % 2 else 1
        nonk = tc.scale_tile_ratio(kind // 2) if kind % 2 else 1
        mk = tc.num_k_slots_per_tile()
        assert 0 <= tile < self.num_k, "HBM load exceeded the K strip"
        assert tile == self.current_read + (tc.depths[kind] - 1) * ratio
        assert tile % ratio == 0 and mini % nonk == 0
        token = kind, mini, tile
        assert token not in self.loads, "HBM tile loaded more than once"
        self.loads.add(token)
        fragments = tuple(
            _Fragment(kind, tile + k // mk, mini + mn, k % mk)
            for mn in range(nonk)
            for k in range(mk * ratio)
        )
        if tc.is_async(kind):
            key = kind, mini, ring
            if key in self.lds:
                old = self.lds[key][0]
                assert (
                    kind,
                    mini,
                    old,
                ) in self.reads, "LDS overwritten before its DS read"
            assert ring == tile // ratio % tc.depths[kind]
            self.lds[key] = tile, fragments
            self.pending.append(token)
        return fragments

    def buffer_load_payload(
        self, operand, VIA_LDS, ring, mini, pointer, offset, soff=0
    ):
        fragments = self.load(operand * 2, ring, mini, pointer + soff)
        if not VIA_LDS:
            return fragments

    def buffer_load_scale(self, operand, VIA_LDS, ring, mini, pointer, offset, soff=0):
        kind = operand * 2 + 1
        fragments = self.load(kind, ring, mini, pointer + soff)
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
            payloads = [(2, 0), (0, 0), (0, 1), (2, 1)]
        else:
            payloads = sorted(
                [(0, i) for i in range(tc.nm)] + [(2, i) for i in range(tc.nn)],
                key=lambda token: (token[1], token[0]),
            )
        required = set()
        for pos, (kind, mini) in enumerate(payloads):
            if slot is not None and slot != pos:
                continue
            for target in (kind, kind + 1):
                if target in tc.active_kinds() and tc.is_async(target):
                    if target % 2 and (
                        self.current_read % tc.scale_step_ratio(target // 2)
                        or mini % tc.scale_tile_ratio(target // 2)
                    ):
                        continue
                    required.add((target, mini, self.current_read))
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

    def read(self, kind, ring, mini, k):
        token = kind, mini, self.current_read
        assert token in self.completed, "DS read issued before its async copy completed"
        actual, fragments = self.lds[kind, mini, ring]
        assert actual == self.current_read
        self.reads.add(token)
        return fragments[k]

    def ds_read_frag(
        self, operand, ring, mini, k, *, READ_PAYLOAD, READ_SCALE, SCALE_READ_IDX=None
    ):
        payload = self.read(operand * 2, ring, mini, k) if READ_PAYLOAD else None
        scale = (
            self.read(operand * 2 + 1, SCALE_READ_IDX, mini, k)
            if READ_SCALE and self.tc.has_scale(operand)
            else None
        )
        return payload, scale

    def ds_read_scale(self, operand, ring, mini):
        kind = operand * 2 + 1
        token = kind, mini, self.current_read
        assert token in self.completed, "Scale DS read issued before its copy completed"
        actual, fragments = self.lds[kind, mini, ring]
        assert actual == self.current_read
        self.reads.add(token)
        return fragments

    def dot(self, a, b, accumulator, mk, fc, tc, enabled, phase):
        if not enabled:
            return accumulator
        assert 0 <= accumulator < self.num_k
        assert phase == accumulator * tc.BLOCK_K // 128 % 2
        for operand, fragments in enumerate((a, b)):
            for k in range(mk):
                payload, scale = fragments[k * 2 : k * 2 + 2]
                assert payload.kind == operand * 2 and payload.k == k
                assert payload.tile == accumulator, (
                    "MFMA consumed stale or overwritten registers"
                )
                if tc.has_scale(operand):
                    assert scale == _Fragment(
                        operand * 2 + 1, accumulator, payload.mini, k
                    )
                self.reads.add((payload.kind, payload.mini, accumulator))
                if tc.has_scale(operand):
                    self.reads.add(
                        (
                            scale.kind,
                            scale.mini - scale.mini % tc.scale_tile_ratio(operand),
                            accumulator - accumulator % tc.scale_step_ratio(operand),
                        )
                    )
        token = a[0].mini, b[0].mini, accumulator
        assert token not in self.mfmas, "K tile accumulated twice"
        self.mfmas.add(token)
        return accumulator + 1


def _execute(tc, num_k, monkeypatch, epilogue_groups=2):
    machine = _Machine(tc, num_k)
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

    def step(*args, **kwargs):
        machine.current_read = args[4]
        machine.steps.append((args[4], kwargs.get("IN_LOOP", False)))
        return step_fn(*args, **kwargs)

    def fill(*args, **kwargs):
        machine.current_read = args[3]
        return fill_fn(*args, **kwargs)

    def check_assumption(value):
        assert value

    def init_buffers(pc):
        return tuple(
            (
                (None,)
                * (
                    tc.depths[kind]
                    * (
                        tc.scale_cache_fragments(kind // 2)
                        if kind % 2
                        else (tc.nm if kind < 2 else tc.nn) * tc.num_k_slots_per_tile()
                    )
                )
                if kind in tc.active_kinds() and not tc.is_async(kind)
                else ()
            )
            for kind in (0, 2, 1, 3)
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
        patch.setattr(pipeline, "_init_buffers", init_buffers)
        patch.setattr(pipeline, "_pipeline_peeled", lambda _: peeled)
        patch.setattr(pipeline, "_wait", machine.check_wait)
        patch.setattr(pipeline, "_maybe_block_dot", machine.dot)
        patch.setattr(
            pipeline, "pick_stage", lambda enabled: lambda name: nullcontext()
        )
        patch.setattr(pipeline.gl, "static_range", range)
        patch.setattr(pipeline.tl, "range", range)
        patch.setattr(pipeline.gl, "static_assert", lambda value, message: None)
        patch.setattr(pipeline.gl, "assume", check_assumption)
        patch.setattr(pipeline.gl, "zeros", lambda *args, **kwargs: 0)
        patch.setattr(pipeline.gl, "barrier", lambda: None)
        patch.setattr(pipeline.gl.amd.cdna4, "sched_barrier", lambda mask: None)
        pointers, buffers, regs = pipeline._run_buffered_pipeline.fn(
            pc, _pointers(0, 0, 0, 0), num_k
        )
        main = num_k - tc.pipeline_depth()
        assert regs.acc == (main,) * (tc.nm * tc.nn)
        for _ in range(epilogue_groups):
            machine.commit_buffer_load()
        regs = pipeline._drain_buffered_pipeline.fn(
            pc, pointers, buffers, regs, num_k, epilogue_groups
        )
        assert regs.acc == (num_k - 1,) * (tc.nm * tc.nn)
        assert pipeline._last_mfma.fn(pc, regs) == (num_k,) * (tc.nm * tc.nn)

    expected = {
        (kind, mini, tile)
        for kind in tc.active_kinds()
        for mini in range(
            0,
            tc.nm if kind < 2 else tc.nn,
            tc.scale_tile_ratio(kind // 2) if kind % 2 else 1,
        )
        for tile in range(0, num_k, tc.scale_step_ratio(kind // 2) if kind % 2 else 1)
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
    minimum = tc.pipeline_depth() + tc.pipeline_peeled() + tc.pipeline_unroll()
    quantum = math.lcm(*tc.scale_steps)
    first = math.ceil(minimum / quantum) * quantum
    for num_k in range(first, first + math.lcm(quantum, tc.pipeline_unroll()), quantum):
        _execute(tc, num_k, monkeypatch)
    # Re-enter the same static register-ring mapping from another runtime loop body.
    _execute(tc, first + 2 * tc.pipeline_unroll(), monkeypatch)


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
        for x in range(tc.pipeline_depth() * 2)
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
    assert tc.pipeline_depth() == 2
    assert tc.pipeline_register_period() == 1
    for kind in (1, 3):
        for in_loop in (False, True):
            for fill in (False, True):
                assert pipeline._index.fn(tc, 7, kind, fill, 1, in_loop) == 0
