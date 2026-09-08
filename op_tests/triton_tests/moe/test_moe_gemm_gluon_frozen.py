# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime loop bounds and frozen/manual modes of the common component driver."""

import os

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant, WarpPipeline
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_torch
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.test_moe_gemm_gluon_registers import (
    _build_register_case,
    _launch_register_case,
    _register_config,
)


@pytest.fixture(scope="module", params=["mxfp4", "bf16"])
def frozen_case(request):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    with pytest.MonkeyPatch.context() as patch:
        for name in os.environ:
            if name.startswith("AITER_TRITON_MOE_GLUON_"):
                patch.delenv(name)
        patch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
        case = _build_register_case(
            request.param, k=960 if request.param == "bf16" else 7168
        )
        assert case.route.block_m == _register_config(case.dtype)["BLOCK_M"]
        _launch_register_case(case, _register_config(case.dtype))
        torch.testing.assert_close(case.output[0], case.expected, rtol=3e-4, atol=3e-5)
        case.baseline = case.output.clone()
        yield case


@pytest.mark.parametrize("mode", list(WarpPipeline))
def test_frozen_step_pipeline_modes_are_bitwise_deterministic(frozen_case, mode):
    config = dict(
        _register_config(frozen_case.dtype), FROZEN_STEP=True, WARP_PIPELINE=int(mode)
    )
    for repeat in range(8):
        frozen_case.output.fill_(float("nan"))
        _launch_register_case(frozen_case, config)
        assert torch.equal(
            frozen_case.output.view(torch.int32), frozen_case.baseline.view(torch.int32)
        ), (mode, repeat)


@pytest.mark.parametrize("gated", [False, True], ids=["gemm2", "gemm1"])
def test_same_compiled_entry_honors_runtime_num_k(gated, monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
    case = _build_register_case("bf16", m=257, n=512, k=2048, experts=4, topk=2)
    if not gated:
        case.output = torch.empty_like(case.raw).unsqueeze(0)
    config = dict(
        _register_config("bf16"), NUM_LDS_BUFFER=0, A_NUM_BUFFER=2, B_NUM_BUFFER=2
    )
    assert case.route.block_m == config["BLOCK_M"]
    tc = host._probe_tuning_config(config, DtypeQuant.BF16, DtypeQuant.BF16)
    full_num_k = case.k // config["BLOCK_K"]
    short_num_k = tc.min_num_k() + 1
    short_k = short_num_k * config["BLOCK_K"]
    assert short_num_k < full_num_k
    assert tc.validate_pipeline(short_k)
    prefix = moe_gemm_torch(
        case.x[:, :short_k].float(),
        case.w[:, :short_k, :].float(),
        None,
        case.route,
        case.gather,
    )
    if gated:
        gate, up = prefix.chunk(2, dim=-1)
        prefix = torch.nn.functional.silu(gate) * up
    references = {
        full_num_k: case.expected if gated else case.raw,
        short_num_k: prefix,
    }
    captured = {}

    def capture_launch(kernel, grid, args, *_metadata):
        captured.update(kernel=kernel, grid=grid, args=args)

    monkeypatch.setattr(host, "_fast_launch", capture_launch)
    compiled = _launch_register_case(case, config, swiglu=gated, split=gated)
    torch.testing.assert_close(
        case.output[0], references[full_num_k], rtol=3e-4, atol=3e-5
    )
    kernel = captured["kernel"]
    launch_args = list(captured["args"])
    num_k_arg = kernel.arg_names.index("NUM_K")
    cfg_k_arg = kernel.arg_names.index("CFG_K")
    assert host._cval(launch_args[cfg_k_arg]) == case.k
    function = compiled.function
    run = compiled.run
    stream = torch.cuda.current_stream().cuda_stream

    def unexpected_jit(*_args, **_kwargs):
        pytest.fail("changing NUM_K must execute the same compiled entry")

    monkeypatch.setattr(kernel, "run", unexpected_jit)
    previous = {}
    for num_k in (short_num_k, full_num_k, short_num_k, full_num_k):
        case.output.fill_(float("nan"))
        launch_args[num_k_arg] = num_k
        # Keep every layout/shape argument fixed, including CFG_K, A/B strides,
        # and pointers. This invokes the already-loaded binary without JIT entry.
        run(
            captured["grid"],
            1,
            1,
            stream,
            function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *launch_args,
        )
        torch.testing.assert_close(
            case.output[0], references[num_k], rtol=3e-4, atol=3e-5
        )
        if num_k in previous:
            assert torch.equal(case.output.view(torch.int32), previous[num_k])
        previous[num_k] = case.output.view(torch.int32).clone()
        assert compiled.function == function
        assert host._cval(launch_args[cfg_k_arg]) == case.k
