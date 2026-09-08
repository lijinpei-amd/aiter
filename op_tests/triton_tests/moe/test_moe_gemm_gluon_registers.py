# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Register/LDS pipeline choices must preserve values through buffer wraparound."""

import os
import re
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_op_gemm_gluon import moe_gemm_gluon
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.test_moe_gemm_gluon_dtypes import _quantize
from op_tests.triton_tests.moe.test_moe_gemm_gluon_read_placement import _config


def _register_config(dtype):
    config = _config()
    config.update(A_SCALE_SORTED_SHUFFLED=True, B_SCALE_SHUFFLED=True)
    if dtype == "mxfp8":
        config.update(
            BLOCK_K=128,
            MINI_BLOCK_K=128,
            VGPR_PREFETCH_K=128,
            K_UNROLL=6,
        )
    elif dtype == "bf16":
        config.update(
            BLOCK_K=64,
            MINI_BLOCK_K=64,
            VGPR_PREFETCH_K=64,
            mfma_instr_shape=(32, 32, 16),
            warps_per_cta=(2, 4),
            tiles_per_warp=(1, 1),
            SCALE_FILL_MID=False,
            A_SCALE_SORTED_SHUFFLED=False,
            B_SCALE_SHUFFLED=False,
        )
    return config


def _build_register_case(dtype, m=257, n=512, k=None, experts=33, topk=8):
    # K7168 and E33 are supported by the native sorted-scale producer. Twenty-eight
    # K256 stages exercise a remainder at depth three and repeated ring wraps.
    k = k or (960 if dtype == "bf16" else 7168)
    torch.manual_seed(72)
    route, gather, _ = routing(
        torch.randn((m, experts), device="cuda", dtype=torch.float16), topk
    )
    route.gate_scal = None
    x_exp = torch.randint(-3, 2, (m, k // 32), device="cuda")
    w_exp = torch.randint(-3, 2, (experts, n, k // 32), device="cuda")
    x = (
        torch.randn((m, k), device="cuda")
        * 0.1
        * x_exp.float().exp2().repeat_interleave(32, -1)
    ).bfloat16()
    w = (
        (
            torch.randn((experts, n, k), device="cuda")
            * 0.1
            * w_exp.float().exp2().repeat_interleave(32, -1)
        )
        .bfloat16()
        .transpose(1, 2)
    )
    x, xs, x_ref = _quantize(x, dtype, axis=-1)
    w, ws, w_ref = _quantize(w, dtype, axis=1)
    raw = moe_gemm_torch(x_ref.float(), w_ref.float(), None, route, gather)
    gate, up = raw.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    return SimpleNamespace(
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        route=route,
        gather=gather,
        x=x,
        xs=xs,
        w=w,
        ws=ws,
        raw=raw,
        expected=expected,
        output=torch.empty((1, m * topk, n // 2), device="cuda", dtype=torch.float32),
    )


def _launch_register_case(case, config, *, swiglu=True, split=True, y_scales=None):
    return moe_gemm_gluon(
        case.output,
        case.x,
        case.w,
        case.xs,
        case.ws,
        None,
        None,
        case.route,
        case.gather,
        None,
        case.n,
        case.k,
        swiglu,
        1.0,
        None,
        False,
        config=config,
        gate_up_split=split,
        y_scales=y_scales,
    )


@pytest.fixture(scope="module", params=["mxfp4", "mxfp8", "bf16"])
def register_case(request):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    with pytest.MonkeyPatch.context() as patch:
        for name in os.environ:
            if name.startswith("AITER_TRITON_MOE_GLUON_"):
                patch.delenv(name)
        patch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
        case = _build_register_case(request.param)
        _launch_register_case(case, _register_config(case.dtype))
        # The FP8 MFMA and the dequantised FP32 reference have different accumulation
        # rounding. The tuning variants below must still reproduce the LDS bytes.
        atol = 3e-4 if case.dtype == "mxfp8" else 3e-5
        torch.testing.assert_close(case.output[0], case.expected, rtol=3e-4, atol=atol)
        case.baseline = case.output.clone()
        yield case


def _assert_register_case(case, config):
    for repeat in range(8):
        case.output.fill_(float("nan"))
        _launch_register_case(case, config)
        assert torch.equal(
            case.output.view(torch.int32), case.baseline.view(torch.int32)
        ), (config, repeat)


@pytest.mark.parametrize("mask", range(8), ids=lambda mask: f"register_mask{mask}")
def test_independent_register_operands(register_case, mask):
    """Every register choice works alone and together, including absent BF16 scales."""
    config = _register_config(register_case.dtype)
    config.update(
        B_IN_REG=bool(mask & 1),
        A_SCALE_IN_REG=bool(mask & 2),
        B_SCALE_IN_REG=bool(mask & 4),
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize(
    "depths,mask",
    [
        ((1, 1, 1, 1), 7),
        ((2, 3, 2, 1), 0),
        ((2, 3, 2, 1), 1),
        ((3, 2, 4, 3), 6),
        ((4, 2, 2, 2), 7),
    ],
    ids=["single", "mixed_lds", "mixed_b_register", "mixed_scale_registers", "gcd2"],
)
def test_independent_buffer_depths(register_case, depths, mask):
    config = _register_config(register_case.dtype)
    config.update(
        zip(
            (
                "A_NUM_BUFFER",
                "B_NUM_BUFFER",
                "A_SCALE_NUM_BUFFER",
                "B_SCALE_NUM_BUFFER",
            ),
            depths,
        )
    )
    config.update(
        B_IN_REG=bool(mask & 1),
        A_SCALE_IN_REG=bool(mask & 2),
        B_SCALE_IN_REG=bool(mask & 4),
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize("mask", [2, 4, 6], ids=["a_scale", "b_scale", "both_scales"])
def test_register_scales_without_payload_shuffle(register_case, mask, monkeypatch):
    """A/B register scales do not require a preshuffled or register B payload."""
    if register_case.dtype == "bf16":
        pytest.skip("BF16 has no scale tensors.")
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "0")
    config = _register_config(register_case.dtype)
    config.update(
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        A_SCALE_IN_REG=bool(mask & 2),
        B_SCALE_IN_REG=bool(mask & 4),
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize("read_mask", [0, 5, 10, 15])
def test_register_buffers_with_compiler_pipeline(register_case, read_mask):
    config = _register_config(register_case.dtype)
    config.update(
        WARP_PIPELINE=1,
        DS_READ_IN_MFMA=read_mask,
        B_IN_REG=True,
        A_SCALE_IN_REG=True,
        B_SCALE_IN_REG=True,
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=3,
        A_SCALE_NUM_BUFFER=4,
        B_SCALE_NUM_BUFFER=2,
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_compiler_pipeline_with_gcd_unroll(register_case, depth):
    """Pointer updates must not pull the next unrolled wait into a stage region."""
    config = _register_config(register_case.dtype)
    config.update(
        WARP_PIPELINE=1,
        B_IN_REG=True,
        warps_per_cta=(1, 4),
        A_NUM_BUFFER=depth,
        B_NUM_BUFFER=depth,
        A_SCALE_NUM_BUFFER=depth,
        B_SCALE_NUM_BUFFER=depth,
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize("scheme", [1, 2, 4], ids=["per_op", "per_slot", "warp_stage"])
@pytest.mark.parametrize("mask", [1, 6], ids=["b_register", "scale_registers"])
def test_mixed_buffers_wait_commit(register_case, scheme, mask):
    config = _register_config(register_case.dtype)
    config.update(
        WAIT_COMMIT_SCHEME=scheme,
        B_IN_REG=bool(mask & 1),
        A_SCALE_IN_REG=bool(mask & 2),
        B_SCALE_IN_REG=bool(mask & 4),
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=3,
        A_SCALE_NUM_BUFFER=4,
        B_SCALE_NUM_BUFFER=2,
    )
    _assert_register_case(register_case, config)


def test_register_b_uses_buffer_load_and_less_lds(register_case):
    config = _register_config(register_case.dtype)
    lds_kernel = _launch_register_case(register_case, config)
    config["B_IN_REG"] = True
    reg_kernel = _launch_register_case(register_case, config)
    assert reg_kernel.metadata.shared < lds_kernel.metadata.shared
    loads = [
        line
        for line in reg_kernel.asm["amdgcn"].splitlines()
        if re.search(r"\bbuffer_load_dwordx4\b", line)
    ]
    assert any(" lds" not in line for line in loads), "B must load into registers"
    assert any(" lds" in line for line in loads), "A must still load directly to LDS"


def test_unquantized_gate_up_split_allows_small_warp_extent(register_case):
    """FP32 output needs no warp-local 32-element output-scale reduction."""
    if register_case.dtype == "bf16":
        pytest.skip("The BF16 reference already has a 32-column MFMA extent.")
    config = _register_config(register_case.dtype)
    config.update(
        BLOCK_N=128,
        MINI_BLOCK_N=64,
        tiles_per_warp=(2, 1),
        B_IN_REG=True,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
    )
    _assert_register_case(register_case, config)


def test_k128_register_scale_phase(register_case):
    """An odd GCD must carry the packed scale-word phase across loop trips."""
    if register_case.dtype != "mxfp8":
        pytest.skip("Packed K128 scale pairing requires MXFP8 operands.")
    config = _register_config(register_case.dtype)
    config.update(
        BLOCK_K=128,
        MINI_BLOCK_K=128,
        VGPR_PREFETCH_K=128,
        B_IN_REG=True,
        A_SCALE_IN_REG=True,
        B_SCALE_IN_REG=True,
        A_NUM_BUFFER=3,
        B_NUM_BUFFER=2,
        A_SCALE_NUM_BUFFER=4,
        B_SCALE_NUM_BUFFER=3,
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize("mask", [1, 2, 4, 7], ids=["b", "a_scale", "b_scale", "all"])
def test_register_operands_with_mini_k_split(register_case, mask):
    if register_case.dtype == "bf16":
        pytest.skip("This case covers microscaled K256 stages split into K128 tiles.")
    config = _register_config(register_case.dtype)
    config.update(
        BLOCK_N=128,
        MINI_BLOCK_N=64,
        BLOCK_K=256,
        MINI_BLOCK_K=128,
        VGPR_PREFETCH_K=256,
        warps_per_cta=(2, 2),
        B_IN_REG=bool(mask & 1),
        A_SCALE_IN_REG=bool(mask & 2),
        B_SCALE_IN_REG=bool(mask & 4),
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=2,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=2,
    )
    _assert_register_case(register_case, config)


@pytest.mark.parametrize("swiglu", [False, True], ids=["gemm2", "interleaved_gemm1"])
def test_register_pipeline_without_fused_tail(register_case, swiglu):
    """The ordinary drain also serves GEMM2 and interleaved gate/up GEMM1."""
    case = SimpleNamespace(**vars(register_case))
    expected = case.raw
    if swiglu:
        expected = torch.nn.functional.silu(expected[:, ::2]) * expected[:, 1::2]
    case.output = torch.empty_like(expected).unsqueeze(0)
    config = _register_config(case.dtype)
    config.update(
        B_IN_REG=True,
        A_SCALE_IN_REG=True,
        B_SCALE_IN_REG=True,
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=3,
        A_SCALE_NUM_BUFFER=4,
        B_SCALE_NUM_BUFFER=2,
    )
    first = None
    for repeat in range(8):
        case.output.fill_(float("nan"))
        _launch_register_case(case, config, swiglu=swiglu, split=False)
        if first is None:
            atol = 3e-4 if case.dtype == "mxfp8" else 3e-5
            torch.testing.assert_close(case.output[0], expected, rtol=3e-4, atol=atol)
            first = case.output.clone()
        else:
            assert torch.equal(
                case.output.view(torch.int32), first.view(torch.int32)
            ), repeat


def test_register_pipeline_fused_mxfp4_output(register_case):
    if register_case.dtype != "mxfp4":
        pytest.skip("This case covers the fused a4w4 output quantisation.")
    case = SimpleNamespace(**vars(register_case))
    rows = case.gather.numel()
    case.output = torch.empty((1, rows, case.n // 4), device="cuda", dtype=torch.uint8)
    scales = torch.empty((rows, case.n // 64), device="cuda", dtype=torch.uint8)
    config = _register_config(case.dtype)
    _launch_register_case(case, config, y_scales=scales)
    baseline, baseline_scales = case.output.clone(), scales.clone()
    config.update(
        B_IN_REG=True,
        A_SCALE_IN_REG=True,
        B_SCALE_IN_REG=True,
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=3,
        A_SCALE_NUM_BUFFER=4,
        B_SCALE_NUM_BUFFER=2,
    )
    for repeat in range(8):
        case.output.fill_(0xAA)
        scales.fill_(0xAA)
        _launch_register_case(case, config, y_scales=scales)
        assert torch.equal(case.output, baseline), repeat
        assert torch.equal(scales, baseline_scales), repeat


@pytest.mark.parametrize("dtype", ["mxfp4", "mxfp8", "bf16"])
def test_single_buffer_single_k_stage(dtype, monkeypatch):
    """A single stage fills and consumes its register/LDS slots before draining."""
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
    config = _register_config(dtype)
    case = _build_register_case(dtype, k=config["BLOCK_K"], experts=4, topk=2)
    config.update(
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        B_IN_REG=True,
        A_SCALE_IN_REG=True,
        B_SCALE_IN_REG=True,
        A_NUM_BUFFER=1,
        B_NUM_BUFFER=1,
        A_SCALE_NUM_BUFFER=1,
        B_SCALE_NUM_BUFFER=1,
    )
    _launch_register_case(case, config)
    torch.testing.assert_close(case.output[0], case.expected, rtol=3e-4, atol=3e-5)
    case.baseline = case.output.clone()
    _assert_register_case(case, config)
    # Repeat the one-stage operation with deeper queues: only real K tiles may
    # be loaded during the prologue, even when each stream requests more buffers.
    config.update(
        A_NUM_BUFFER=4,
        B_NUM_BUFFER=3,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=4,
    )
    _assert_register_case(case, config)
