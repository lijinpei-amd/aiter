# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU oracles for packed E8M0 bytes and their K128 pipeline consumers."""

import ast
import inspect
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import pytest
from triton.experimental.gluon import language as gl

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
        ({"K_UNROLL": 3}, 7168, "even K_UNROLL"),
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
        return unwrap(getattr(self.config, name))


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


def _source_body(jit_function):
    source = textwrap.dedent(inspect.getsource(jit_function.fn))
    return ast.parse(source).body[0].body


def _has_call(node, names):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in names
        for child in ast.walk(node)
    )


def _scalar_schedule():
    """Execute production's outer loops/call sites, omitting GPU setup/epilogue.

    Keeping the original AST, including all call arguments, catches incorrect
    phase wiring in the seed, peel, loop, remainder, and fused/unfused drain.
    The oracle below independently expects monotonically numbered K stages.
    """
    calls = {"_buffer_load", "_pipeline_step", "_drain_last_fused"}
    selected = []
    for node in _source_body(kernel._moe_gemm_body):
        assignment = isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        )
        if _has_call(node, calls) or (
            assignment and node.target.id in {"MAIN", "UNROLLED", "FUSE", "DRAIN"}
        ):
            selected.append(node)
    assert any(isinstance(node, ast.For) for node in selected)
    return compile(
        ast.fix_missing_locations(ast.Module(selected, [])),
        "<moe scalar schedule>",
        "exec",
    )


def _step_phase_expressions():
    names = {"KIE", "KUE", "FILL_K_PHASE", "DOT_K_PHASE"}
    selected = [
        node
        for node in _source_body(kernel._pipeline_step_impl)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in names
    ]
    assert {node.target.id for node in selected} == names
    return compile(
        ast.fix_missing_locations(ast.Module(selected, [])), "<moe step phases>", "exec"
    )


class _CopyRecorder:
    def __init__(self, buffers):
        self.buffers = buffers
        self.addresses = defaultdict(list)

    def _record(self, kind, buffer, tile, pointer, offsets, soffset):
        assert offsets == 0
        history = self.addresses[kind, tile]
        stage = len(history)
        assert unwrap(buffer) == stage % self.buffers
        expected = (stage // 2) * 256 if kind % 2 else stage * 128
        address = unwrap(pointer + soffset)
        assert address == expected, (kind, tile, stage, address, expected)
        history.append(address)

    def buffer_load_a_payload(self, *args):
        self._record(0, *args)

    def buffer_load_a_scale(self, *args):
        self._record(1, *args)

    def buffer_load_b_payload(self, *args):
        self._record(2, *args)

    def buffer_load_b_scale(self, *args):
        self._record(3, *args)

    def commit_buffer_load(self):
        pass


@pytest.mark.parametrize("buffers", [3, 4])
@pytest.mark.parametrize("unroll", [2, 4, 6])
@pytest.mark.parametrize("stages", [4, 6, 8, 10, 18, 56])
@pytest.mark.parametrize("soff", [False, True], ids=["pointer-per-stage", "soffset"])
@pytest.mark.parametrize("fused", [False, True], ids=["ordinary-drain", "fused-drain"])
def test_pipeline_phases_and_packed_scale_addresses(
    monkeypatch, buffers, unroll, stages, soff, fused
):
    tc = _HostView(
        _config(fused=fused, NUM_LDS_BUFFER=buffers, K_UNROLL=unroll, SOFF_UNROLL=soff)
    )
    assert tc.validate(4096, stages * 128)
    fc = _HostView(tc.func_cfg)
    sink = _CopyRecorder(buffers)
    pc = SimpleNamespace(
        tuning_cfg=tc,
        func_cfg=fc,
        lds_ptrs=sink,
        a_hbm_offs=(0, 0),
        b_hbm_offs=(0, 0),
        a_scale_hbm_offs=(0, 0),
        b_scale_hbm_offs=(0, 0),
        a_step=128,
        b_step=128,
        s_step=4,
        a_scale_stride_k=32,
        b_scale_stride_k=32,
    )
    emit = kernel._buffer_load.fn
    advance = kernel._advance_hbm_ptrs.fn
    phase_code = _step_phase_expressions()
    schedule = _scalar_schedule()
    reads, dots = [], []

    def step(
        pc,
        pointers,
        regs,
        buffer,
        read_buffer,
        between,
        filling,
        reading,
        IN_LOOP=False,
        DO_MFMA=True,
        KI=0,
        KU=1,
        WAIT_SLACK=0,
        K_PHASE=0,
    ):
        del IN_LOOP, between, WAIT_SLACK
        scope = {"gl": gl, "tc": tc, "KI": KI, "KU": KU, "K_PHASE": K_PHASE}
        exec(phase_code, scope)  # noqa: S102 - scalar AST from the kernel under test
        if reading:
            assert read_buffer == len(reads) % buffers
            assert K_PHASE == len(reads) % 2
            reads.append(K_PHASE)
        if DO_MFMA:
            assert scope["DOT_K_PHASE"] == len(dots) % 2
            dots.append(scope["DOT_K_PHASE"])
        if filling:
            for ni, mi in product(range(2), range(2)):
                emit(
                    pc,
                    pointers,
                    buffer,
                    mi,
                    ni,
                    KI=scope["KIE"],
                    K_PHASE=scope["FILL_K_PHASE"],
                )
                if kernel._slot_advances_hbm_ptrs(
                    mi, ni, 2, 2, scope["KIE"], scope["KUE"]
                ):
                    pointers = advance(
                        pc, pointers, scope["KUE"], scope["FILL_K_PHASE"]
                    )
        return pointers, regs

    def final_dot(pc, regs, bias, x_scale, func_cfg, tuning_cfg, K_PHASE=0):
        assert K_PHASE == len(dots) % 2
        dots.append(K_PHASE)
        return regs.acc

    with monkeypatch.context() as patch:
        patch.setattr(kernel, "_opt_at", kernel._opt_at.fn)
        patch.setattr(kernel, "_PipelinePointers", _Pointers)
        patch.setattr(kernel, "_pipeline_step_impl", step)
        scope = {
            "gl": SimpleNamespace(static_range=range, constexpr=gl.constexpr),
            "tl": SimpleNamespace(range=range),
            "require_constexpr": bool,
            "tuning_cfg": tc,
            "func_cfg": fc,
            "pc": pc,
            "hbm_ptrs": _Pointers(),
            "NB": buffers,
            "NUM_K": stages,
            "NM": 2,
            "NN": 2,
            "STAGES_BETWEEN": buffers - 2,
            "acc0": (0, 0, 0, 0),
            "EPI_GROUPS": 0,
            "bias_lds_ptr": None,
            "bias_hbm_base": None,
            "pid_n": 0,
            "N": 4096,
            "x_static_scale": None,
            "_buffer_load": emit,
            "_advance_hbm_ptrs": advance,
            "_pipeline_step": kernel._pipeline_step.fn,
            "_PipelineRegFragments": lambda *args: SimpleNamespace(acc=args[-1]),
            "_drain_last_fused": final_dot,
            "_epi_bias_tiles": lambda *args: None,
        }
        exec(schedule, scope)  # noqa: S102 - scalar AST from the kernel under test
    assert len(reads) == len(dots) == stages
    assert set(sink.addresses) == set(product(range(4), range(2)))
    assert all(len(history) == stages for history in sink.addresses.values())
    assert scope["hbm_ptrs"].a_scale_hbm_ptr == (stages // 2) * 256
    assert scope["hbm_ptrs"].b_scale_hbm_ptr == (stages // 2) * 256
