# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU oracles for packed E8M0 bytes and their K128 pipeline consumers."""

import math
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from itertools import product
from types import SimpleNamespace

import pytest
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe import _buffered as buffered
from aiter.ops.triton._gluon_kernels.gfx950.moe import moe_gemm as kernel
from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DtypeQuant,
    EpilogueMode,
)


def _config(dtype=DtypeQuant.MXFP8, fused=True, **overrides):
    fc = KernelFuncConfig(
        dtype,
        dtype,
        dtype,
        dtype,
        gl.float32,
        ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        int(DtypeQuant.BF16),
        False,
        False,
        True,
        False,
        fused,
        int(EpilogueMode.NOP),
    )
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "K_UNROLL": 6,
        "MINI_BLOCK_K": 128,
        "MINI_BLOCK_M": 64,
        "MINI_BLOCK_N": 64,
        "NUM_LDS_BUFFER": 3,
        "mfma_instr_shape": (16, 16, 128),
        "warps_per_cta": (2, 2),
        "tiles_per_warp": (2, 2),
        "k_width": None,
        "transposed": True,
        "WAVES_PER_EU": 1,
        "TILE_SCHED": 0,
        "GROUP_M": 1,
        "NUM_XCDS": 8,
        "token_mod": "",
        "token_scale_mod": "",
        "expert_mod": "",
        "expert_scale_mod": "",
        "result_mod": "",
        "result_scale_mod": "",
        "WARP_PIPELINE": 0,
        "VGPR_PREFETCH_K": 128,
        "A_SCALE_SORTED_SHUFFLED": True,
        "B_SCALE_SHUFFLED": True,
    }
    args.update(overrides)
    return KernelTuningConfig(fc, **args)


def _coordinate(bases, index):
    result = [0, 0]
    for bit, basis in enumerate(bases):
        if index & (1 << bit):
            result = [x ^ y for x, y in zip(result, basis)]
    return result


def _thread_coordinate(layout, reg, lane, warp):
    parts = (
        _coordinate(layout.reg_bases, reg),
        _coordinate(layout.lane_bases, lane),
        _coordinate(layout.warp_bases, warp),
    )
    return tuple(parts[0][axis] ^ parts[1][axis] ^ parts[2][axis] for axis in (0, 1))


def _producer_byte_coordinates(nonk):
    """Independent producer order for one K256 tile, without gate/up interleave.

    Each stripe contains 64 lane dwords. Byte order is two rows sixteen apart,
    then the same rows four scale groups later. A's row numbers are already in
    routing order; B's row numbers remain the supplied weight-row order.
    """
    return [
        (stripe * 32 + row + half_row * 16, group + half_k * 4)
        for stripe in range(nonk // 32)
        for group in range(4)
        for row in range(16)
        for half_k in range(2)
        for half_row in range(2)
    ]


@pytest.mark.parametrize(
    "block_k,dtype", [(128, DtypeQuant.MXFP8), (256, DtypeQuant.MXFP4)]
)
@pytest.mark.parametrize(
    "warps,mini_m,mini_n",
    [((2, 4), 64, 128), ((1, 4), 64, 128), ((4, 1), 128, 64), ((2, 2), 64, 64)],
)
@pytest.mark.parametrize("operand", [0, 1], ids=["A", "B"])
def test_packed_words_and_selectors_match_producer(
    block_k, dtype, warps, mini_m, mini_n, operand
):
    tc = _config(
        dtype,
        BLOCK_K=block_k,
        MINI_BLOCK_K=block_k,
        VGPR_PREFETCH_K=block_k,
        BLOCK_M=mini_m * 2,
        BLOCK_N=mini_n * 2,
        MINI_BLOCK_M=mini_m,
        MINI_BLOCK_N=mini_n,
        warps_per_cta=warps,
    )
    assert tc.validate(4096, 7168)
    assert tc.scale_packed_ok(operand)
    byte_layout = tc.dot_operand_scale_fragment_layout(operand)
    word_layout = tc.packed_scale_frag_layout(operand)
    word_shape = tc.packed_scale_shape(operand)
    producer = _producer_byte_coordinates((mini_m, mini_n)[operand])
    assert word_shape[0] * word_shape[1] * 4 == len(producer)

    # Invert the actual LDS read layout to find each fragment word's physical
    # address, rather than reusing production's scale_dword_delta formula.
    read_bases = tc.packed_scale_read_layout(operand).offset_bases
    physical_word = {
        tuple(_coordinate(read_bases, offset)): offset
        for offset in range(len(producer) // 4)
    }
    assert len(physical_word) == len(producer) // 4
    for phase in (0, 1):
        selectors = tc.scale_packed_sel(operand, phase)
        # Retain the frozen K256 contract, including its nontrivial byte order.
        assert selectors == (
            [0, 2, 1, 3] if block_k == 256 else [2 * phase, 2 * phase + 1]
        )
        reg_count = 1 << len(byte_layout.reg_bases)
        assert reg_count % len(selectors) == 0
        assert 1 << len(word_layout.reg_bases) == reg_count // len(selectors)
        for warp, lane, reg in product(
            range(tc.num_warps()), range(64), range(reg_count)
        ):
            row, group = _thread_coordinate(byte_layout, reg, lane, warp)
            word_coord = _thread_coordinate(
                word_layout, reg // len(selectors), lane, warp
            )
            byte_offset = (
                physical_word[word_coord] * 4 + selectors[reg % len(selectors)]
            )
            expected_group = group + (4 * phase if block_k == 128 else 0)
            assert producer[byte_offset] == (row, expected_group), (
                operand,
                warps,
                warp,
                lane,
                reg,
                phase,
                word_coord,
                byte_offset,
            )


@pytest.mark.parametrize(
    "changes,k,diagnostic",
    [
        ({"A_SCALE_SORTED_SHUFFLED": False}, 7168, "both operands"),
        ({"B_SCALE_SHUFFLED": False}, 7168, "both operands"),
        (
            {"mfma_instr_shape": (32, 32, 64), "tiles_per_warp": (1, 1)},
            7168,
            "MFMA 16x16x128",
        ),
        ({"MINI_BLOCK_K": 64}, 7168, None),
        ({"FROZEN_STEP": True}, 7168, "live pipeline"),
        ({}, 384, "complete K256 scale words"),
    ],
)
def test_invalid_packed_k128_configs_are_rejected(changes, k, diagnostic):
    with pytest.raises(AssertionError, match=diagnostic):
        _config(**changes).validate(4096, k)


class _HostView:
    """Expose constexpr fields as ordinary values when executing Python bodies."""

    def __init__(self, config):
        self.config = config

    def __getattr__(self, name):
        value = unwrap(getattr(self.config, name))
        if callable(value):
            value = cache(value)
        setattr(self, name, value)
        return value


@pytest.mark.parametrize(
    "block_k,dtype", [(128, DtypeQuant.MXFP8), (256, DtypeQuant.MXFP4)]
)
@pytest.mark.parametrize("phase", [0, 1])
def test_dot_helpers_forward_scale_phase(monkeypatch, block_k, dtype, phase):
    tc = _HostView(
        _config(
            dtype,
            BLOCK_K=block_k,
            MINI_BLOCK_K=block_k,
            VGPR_PREFETCH_K=block_k,
        )
    )
    fc = _HostView(tc.func_cfg)
    calls = []

    def mfma(**operands):
        selectors = [0, 2, 1, 3] if block_k == 256 else [2 * phase, 2 * phase + 1]
        assert operands["a_scale_sel"] == operands["b_scale_sel"] == selectors
        assert (operands["a"], operands["a_scale"]) == ("A", "A-scale")
        assert (operands["b"], operands["b_scale"]) == ("B", "B-scale")
        calls.append(operands)
        return operands["acc"] + 1

    monkeypatch.setattr(gl, "static_range", range)
    monkeypatch.setattr(gl.amd.cdna4, "mfma_scaled_packed", mfma)
    monkeypatch.setattr(kernel, "_dot", kernel._dot.fn)
    args = (("A", "A-scale"), ("B", "B-scale"), 13, 1, fc, tc)
    assert kernel._maybe_block_dot.fn((), (), 13, 1, fc, tc, False, phase) == 13
    assert calls == []
    assert kernel._maybe_block_dot.fn(*args, True, phase) == 14
    assert len(calls) == 1


@dataclass
class _Pointers:
    a_hbm_ptr: int = 0
    b_hbm_ptr: int = 0
    a_scale_hbm_ptr: int = 0
    b_scale_hbm_ptr: int = 0


@dataclass(frozen=True)
class _Payload:
    stage: int
    operand: int
    tile: int


class _ScaleWord(int):
    def to(self, _dtype):
        return self

    def __rshift__(self, amount):
        return _ScaleWord(int(self) >> amount)


@dataclass
class _Fragments:
    a_payload: tuple
    a_scale: tuple
    b_payload: tuple
    b_scale: tuple
    acc: tuple


class _CopyRecorder:
    """Scalar HBM/LDS model with distinct values for both halves of a scale word."""

    def __init__(self, tc, b_step):
        self.depths = [tc.num_buffers(kind // 2, bool(kind % 2)) for kind in range(4)]
        self.b_step = b_step
        self.addresses = defaultdict(list)
        self.reads = defaultdict(list)
        self.lds = {}
        self.dot_tiles = defaultdict(list)

    def _record(self, kind, buffer, tile, pointer, offsets, soffset):
        assert offsets == tile
        history = self.addresses[kind, tile]
        stage = len(history)
        expected = (
            (stage // 2) * 256
            if kind % 2
            else stage * (self.b_step if kind == 2 else 128)
        )
        address = unwrap(pointer + soffset)
        assert address == expected, (kind, tile, stage, address, expected)
        history.append(address)
        if kind % 2:
            even = stage // 2 * 2
            value = _ScaleWord((even + 1) | ((even + 2) << 16))
        else:
            value = _Payload(stage, kind // 2, tile)
        if buffer is not None:
            assert buffer == stage % self.depths[kind]
            key = kind, tile, buffer
            if key in self.lds:
                assert stage - self.depths[kind] in self.reads[kind, tile], (
                    "LDS slot overwritten before its read",
                    key,
                    stage,
                )
            self.lds[key] = value
        return value

    def buffer_load_a_payload(self, *args):
        self._record(0, *args)

    def buffer_load_a_scale(self, *args):
        self._record(1, *args)

    def buffer_load_b_payload(self, *args):
        self._record(2, *args)

    def buffer_load_b_scale(self, *args):
        self._record(3, *args)

    def buffer_load_b_register(self, pointer, offsets, soffset):
        return (self._record(2, None, offsets, pointer, offsets, soffset),)

    def buffer_load_scale_register(self, operand, pointer, offsets, soffset, K_PHASE):
        kind = operand * 2 + 1
        assert K_PHASE == len(self.addresses[kind, offsets]) % 2
        return (self._record(kind, None, offsets, pointer, offsets, soffset),)

    def _read(self, kind, tile, slot):
        stage = len(self.reads[kind, tile])
        assert slot == stage % self.depths[kind]
        self.reads[kind, tile].append(stage)
        return self.lds[kind, tile, slot]

    def _read_frag(
        self,
        operand,
        buffer,
        tile,
        k,
        _scale_ptr,
        _offsets,
        RELAXED,
        READ_PAYLOAD,
        READ_SCALE,
        SCALE_READ_IDX,
    ):
        assert k == 0
        payload = self._read(operand * 2, tile, buffer) if READ_PAYLOAD else None
        scale = (
            self._read(operand * 2 + 1, tile, SCALE_READ_IDX) if READ_SCALE else None
        )
        return payload, scale

    def ds_read_a_frag(self, *args, **kwargs):
        return self._read_frag(0, *args, **kwargs)

    def ds_read_b_frag(self, *args, **kwargs):
        return self._read_frag(1, *args, **kwargs)

    def commit_buffer_load(self):
        pass

    def wait_buffer_load_groups(self, _groups):
        pass

    def dot(self, a, b, acc, num_mini, _fc, _tc, enabled, phase):
        if not enabled:
            return acc
        assert num_mini == 1 and phase == 0
        ap, asc = a
        bp, bsc = b
        history = self.dot_tiles[ap.tile, bp.tile]
        stage = len(history)
        assert ap.stage == bp.stage == stage
        assert int(asc) & 0xFFFF == int(bsc) & 0xFFFF == stage + 1
        history.append(stage)
        return acc + 1


def _run_scalar_pipeline(monkeypatch, tc, stages, fused):
    """Execute the production driver, reads, fills, ring updates and final MFMA."""
    fc = _HostView(tc.func_cfg)
    b_step = 128 * 16 if tc.B_IN_REG else 128
    sink = _CopyRecorder(tc, b_step)
    pc = SimpleNamespace(
        tuning_cfg=tc,
        func_cfg=fc,
        lds_ptrs=sink,
        a_hbm_offs=(0, 1),
        b_hbm_offs=(0, 1),
        a_scale_hbm_offs=(0, 1),
        b_scale_hbm_offs=(0, 1),
        a_step=128,
        b_step=b_step,
        s_step=4,
        a_scale_stride_k=32,
        b_scale_stride_k=32,
    )

    def check_assumption(condition, message=""):
        assert condition, message

    with monkeypatch.context() as patch:
        patch.setattr(gl, "static_range", range)
        patch.setattr(gl, "static_assert", check_assumption)
        patch.setattr(gl, "assume", check_assumption)
        patch.setattr(gl, "zeros", lambda *_args, **_kwargs: 0)
        patch.setattr(gl, "barrier", lambda: None)
        patch.setattr(gl.amd.cdna4, "sched_barrier", lambda *_: None)
        patch.setattr(buffered.tl, "range", range)
        patch.setattr(buffered, "pick_stage", lambda _: lambda _name: nullcontext())
        for name in (
            "_index",
            "_phase",
            "_replace_tile",
            "_rotate",
            "_rotate_buffers",
            "_init_buffers",
            "_fill_slot",
            "_advance",
            "_read_tile",
            "_read_slot",
            "_step",
            "_step_live",
            "_make_reg_fragments",
            "_merge_ds_read_frags",
            "_take_reg_pairs",
        ):
            patch.setattr(buffered, name, getattr(buffered, name).fn)
        patch.setattr(buffered, "_PipelinePointers", _Pointers)
        patch.setattr(buffered, "_PipelineRegFragments", _Fragments)
        patch.setattr(kernel, "_PipelineRegFragments", _Fragments)
        patch.setattr(buffered, "_maybe_block_dot", sink.dot)
        patch.setattr(kernel, "_maybe_block_dot", sink.dot)
        patch.setattr(kernel, "_take_reg_pairs", kernel._take_reg_pairs.fn)
        ptrs, queues, regs = buffered._run_buffered_pipeline.fn(pc, _Pointers(), stages)
        final = buffered._drain_buffered_pipeline.fn(pc, ptrs, queues, regs, stages, 0)
        if fused:
            acc = kernel._drain_last_fused.fn(pc, final, None, None, fc, tc)
        else:
            acc = buffered._last_mfma.fn(pc, final)
    assert acc == (stages,) * 4
    assert set(sink.addresses) == set(product(range(4), range(2)))
    assert all(len(history) == stages for history in sink.addresses.values())
    assert all(history == list(range(stages)) for history in sink.dot_tiles.values())
    assert len(sink.dot_tiles) == 4


@pytest.mark.parametrize("counts", [(2, 2, 2, 2), (3, 3, 3, 3), (4, 2, 2, 3)])
@pytest.mark.parametrize("unroll", [1, 3, 4])
@pytest.mark.parametrize("registers", [False, True])
@pytest.mark.parametrize("soff", [False, True], ids=["pointer-per-stage", "soffset"])
@pytest.mark.parametrize("fused", [False, True], ids=["ordinary-drain", "fused-drain"])
def test_pipeline_phases_and_packed_scale_addresses(
    monkeypatch, counts, unroll, registers, soff, fused
):
    tc = _HostView(
        _config(
            fused=fused,
            K_UNROLL=unroll,
            SOFF_UNROLL=soff,
            A_NUM_BUFFER=counts[0],
            B_NUM_BUFFER=counts[1],
            A_SCALE_NUM_BUFFER=counts[2],
            B_SCALE_NUM_BUFFER=counts[3],
            B_IN_REG=registers,
            B_PRESHUFFLED=registers,
            A_SCALE_IN_REG=registers,
            B_SCALE_IN_REG=registers,
        )
    )
    minimum = (tc.min_num_k() + 1) // 2 * 2
    effective_unroll = tc.pipeline_unroll()
    # Complete K256 scale words constrain NUM_K to even values. Visit the minimum,
    # every reachable remainder, and a longer strip with additional ring wraps.
    lengths = {
        minimum + 2 * offset
        for offset in range(effective_unroll // math.gcd(2, effective_unroll))
    }
    lengths.add(minimum + 2 * effective_unroll)
    for stages in sorted(lengths):
        assert tc.validate(4096, stages * 128)
        _run_scalar_pipeline(monkeypatch, tc, stages, fused)
