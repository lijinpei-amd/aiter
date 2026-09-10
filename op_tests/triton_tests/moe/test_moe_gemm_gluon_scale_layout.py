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

from aiter.ops.triton._gluon_kernels.gfx950.moe import _frozen as frozen
from aiter.ops.triton._gluon_kernels.gfx950.moe import _pipeline as buffered
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
        "token_cache_modifier": "",
        "token_scale_cache_modifier": "",
        "expert_cache_modifier": "",
        "expert_scale_cache_modifier": "",
        "result_cache_modifier": "",
        "result_scale_cache_modifier": "",
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


@pytest.mark.parametrize("scale_k", [256, 512, 1024])
@pytest.mark.parametrize("operand", [0, 1], ids=["A", "B"])
def test_wide_packed_scale_load_layout_matches_producer(scale_k, operand):
    tc = _config(
        BLOCK_N=256,
        MINI_BLOCK_N=128,
        warps_per_cta=(1, 4),
        SCALE_MINI_BLOCK_M=128,
        SCALE_MINI_BLOCK_N=256,
        SCALE_MINI_BLOCK_K=scale_k,
    )
    nonk = tc.scale_mini_block_nonk(operand)
    producer = [
        (stripe * 32 + row + half_row * 16, ku * 8 + group + half_k * 4)
        for stripe in range(nonk // 32)
        for ku in range(scale_k // 256)
        for group in range(4)
        for row in range(16)
        for half_k in range(2)
        for half_row in range(2)
    ]
    byte_bases = tc.shuffled_scale_read_layout(operand, True).offset_bases
    assert [
        tuple(_coordinate(byte_bases, offset)) for offset in range(len(producer))
    ] == producer
    word_bases = tc.packed_scale_read_layout(operand, True).offset_bases
    physical_word = {
        tuple(_coordinate(word_bases, offset)): offset
        for offset in range(len(producer) // 4)
    }
    byte_layout = tc.scale_load_layout(operand)
    word_layout = tc.packed_scale_load_layout(operand)
    registers = [list(basis) for basis in byte_layout.reg_bases]
    nonk_bit, k_bit = registers.index([16, 0]), registers.index([0, 4])
    retained = [bit for bit in range(len(registers)) if bit not in (nonk_bit, k_bit)]
    for warp, lane, reg in product(
        range(tc.num_warps()), range(64), range(1 << len(registers))
    ):
        selector = ((reg >> nonk_bit) & 1) + 2 * ((reg >> k_bit) & 1)
        word_reg = sum(((reg >> bit) & 1) << pos for pos, bit in enumerate(retained))
        word_coord = _thread_coordinate(word_layout, word_reg, lane, warp)
        assert producer[physical_word[word_coord] * 4 + selector] == _thread_coordinate(
            byte_layout, reg, lane, warp
        )


@pytest.mark.parametrize(
    "changes,k,diagnostic",
    [
        ({"MINI_BLOCK_K": 64}, 7168, None),
        ({"FROZEN_STEP": True}, 7168, None),
        ({}, 384, "complete"),
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


@pytest.mark.parametrize("phase", [0, 1])
def test_frozen_dot_helper_advances_scale_phase_per_mini(monkeypatch, phase):
    tc = _HostView(
        _config(
            BLOCK_K=256,
            MINI_BLOCK_K=128,
            VGPR_PREFETCH_K=256,
            FROZEN_STEP=True,
        )
    )
    fc = _HostView(tc.func_cfg)
    phases = []

    def dot(_a, _a_scale, _b, _b_scale, acc, _fc, _tc, k_phase):
        phases.append(k_phase)
        return acc + 1

    monkeypatch.setattr(gl, "static_range", range)
    monkeypatch.setattr(frozen, "_dot", dot)
    a = ("A0", "A-scale", "A1", "A-scale")
    b = ("B0", "B-scale", "B1", "B-scale")
    assert frozen._maybe_block_dot.fn(a, b, 13, 2, fc, tc, True, phase) == 15
    assert phases == [phase, 1 - phase]


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
        self.tc = tc
        self.depths = [tc.num_buffers(kind // 2, bool(kind % 2)) for kind in range(4)]
        self.b_step = b_step
        self.addresses = defaultdict(list)
        self.reads = defaultdict(list)
        self.lds = {}
        self.dot_tiles = defaultdict(list)

    def _record(self, kind, buffer, tile, pointer, offsets, soffset):
        assert offsets == tile
        history = self.addresses[kind, tile]
        ratio = self.tc.scale_step_ratio(kind // 2) if kind % 2 else 1
        stage = len(history) * ratio
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
            assert buffer == stage // ratio % self.depths[kind]
            key = kind, tile, buffer
            if key in self.lds:
                assert stage - self.depths[kind] * ratio in self.reads[kind, tile], (
                    "LDS slot overwritten before its read",
                    key,
                    stage,
                )
            self.lds[key] = value
        return value

    def buffer_load_payload(
        self, operand, VIA_LDS, buffer, tile, pointer, offsets, soffset
    ):
        value = self._record(operand * 2, buffer, tile, pointer, offsets, soffset)
        if not VIA_LDS:
            return (value,)

    def buffer_load_scale(
        self, operand, VIA_LDS, buffer, tile, pointer, offsets, soffset
    ):
        kind = operand * 2 + 1
        value = self._record(kind, buffer, tile, pointer, offsets, soffset)
        if not VIA_LDS:
            return (value,) * self.tc.scale_read_k_slots(operand)

    def _read(self, kind, tile, slot):
        ratio = self.tc.scale_step_ratio(kind // 2) if kind % 2 else 1
        stage = len(self.reads[kind, tile]) * ratio
        assert slot == stage // ratio % self.depths[kind]
        self.reads[kind, tile].append(stage)
        return self.lds[kind, tile, slot]

    def ds_read_frag(
        self,
        operand,
        buffer,
        tile,
        k,
        READ_PAYLOAD,
        READ_SCALE,
        SCALE_READ_IDX=None,
    ):
        assert k == 0
        payload = self._read(operand * 2, tile, buffer) if READ_PAYLOAD else None
        scale = (
            self._read(operand * 2 + 1, tile, SCALE_READ_IDX) if READ_SCALE else None
        )
        return payload, scale

    def ds_read_scale(self, operand, slot, tile):
        value = self._read(operand * 2 + 1, tile, slot)
        return (value,) * self.tc.scale_read_k_slots(operand)

    def commit_buffer_load(self):
        pass

    def wait_buffer_load_groups(self, _groups):
        pass

    def dot(self, a, b, acc, num_mini, _fc, _tc, enabled, phase):
        if not enabled:
            return acc
        assert num_mini == 1
        ap, asc = a
        bp, bsc = b
        history = self.dot_tiles[ap.tile, bp.tile]
        stage = len(history)
        assert phase == stage % 2
        assert ap.stage == bp.stage == stage
        assert (
            (int(asc) >> (phase * 16)) & 0xFFFF
            == (int(bsc) >> (phase * 16)) & 0xFFFF
            == stage + 1
        )
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
        num_k=stages,
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
            "_replace_tile",
            "_rotate",
            "_rotate_buffers",
            "_init_buffers",
            "_fill_slot",
            "_advance",
            "_read_tile",
            "_read_scale_tile",
            "_read_slot",
            "_step",
            "_step_live",
            "_take_reg_pairs",
            "_take_operand_pairs",
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
    assert all(
        len(history) == stages // (2 if kind % 2 else 1)
        for (kind, _tile), history in sink.addresses.items()
    )
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
    minimum_tiles = (
        tc.pipeline_depth() + buffered._pipeline_peeled(tc) + tc.pipeline_unroll()
    )
    minimum = (minimum_tiles + 1) // 2 * 2
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
        assert buffered._validate_pipeline(tc, stages * 128)
        _run_scalar_pipeline(monkeypatch, tc, stages, fused)
