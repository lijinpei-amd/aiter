# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise component handoffs across the MoE K loop and both pipeline regions."""

import os
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    EpilogueMode,
    TuningSpec,
    WaitCommitScheme,
    WarpPipeline,
)
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_op_gemm_gluon import moe_gemm_gluon
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch


def _config(mask=0, pipeline=WarpPipeline.NONE):
    # Twelve K stages exercise two steady-loop trips, the remainder, and the drain.
    # Both scale tiles use direct-to-LDS loads, and both non-K axes split.
    return TuningSpec(
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        K_UNROLL=3,
        MINI_BLOCK_K=256,
        MINI_BLOCK_M=64,
        MINI_BLOCK_N=128,
        NUM_LDS_BUFFER=3,
        mfma_instr_shape=(16, 16, 128),
        warps_per_cta=(1, 4),
        tiles_per_warp=(2, 2),
        k_width=None,
        transposed=True,
        WAVES_PER_EU=0,
        TILE_SCHED=0,
        GROUP_M=1,
        NUM_XCDS=1,
        token_mod="",
        token_scale_mod="",
        expert_mod="",
        expert_scale_mod="",
        result_mod="",
        result_scale_mod="",
        WARP_PIPELINE=int(pipeline),
        VGPR_PREFETCH_K=256,
        WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_STAGE),
        DS_READ_IN_MFMA=mask,
        SCALE_FILL_MID=True,
    )._asdict()


def _launch(
    case,
    out,
    *,
    config,
    split=True,
    epilogue=EpilogueMode.DEFAULT,
    alpha=1.0,
    limit=None,
    residual=False,
):
    return moe_gemm_gluon(
        out,
        case.x,
        case.w,
        case.x_scales,
        case.w_scales,
        case.bias,
        case.gammas,
        case.routing,
        case.gather,
        None,
        case.n,
        case.k,
        True,
        alpha,
        limit,
        residual,
        config=config,
        gate_up_split=split,
        epilogue=epilogue,
    )


def _expected(case, *, split, epilogue, alpha, limit, residual):
    values = case.raw
    if epilogue != EpilogueMode.NOP:
        values = values + case.bias_rows
    if split:
        gate, linear = values.chunk(2, dim=-1)
    else:
        gate, linear = values[:, ::2], values[:, 1::2]
    if epilogue == EpilogueMode.DEFAULT:
        if limit is not None:
            gate = gate.clamp(max=limit)
            linear = linear.clamp(min=-limit, max=limit)
        gate = gate * torch.sigmoid(alpha * gate)
        if residual:
            linear = linear + 1
    expected = gate * linear
    if epilogue != EpilogueMode.NOP:
        expected = expected * case.gammas[:, None]
    return expected


@pytest.fixture(scope="module")
def case():
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    # These tests pass their tuning choices explicitly; a benchmark shell must not
    # turn on a scale or weight shuffle behind the test's reference tensors.
    with pytest.MonkeyPatch.context() as patch:
        for name in os.environ:
            if name.startswith("AITER_TRITON_MOE_GLUON_"):
                patch.delenv(name)
        torch.manual_seed(41)
        m, n, k, experts, topk = 257, 512, 3072, 4, 2
        route, gather, _ = routing(
            torch.randn((m, experts), device="cuda", dtype=torch.float16), topk
        )
        route.gate_scal = None
        assert route.block_m == 128
        assert bool(torch.any(route.expt_hist % 128 != 0)), "exercise partial tiles"

        # Vary scale exponents between K groups, rows and experts so a scale attached
        # to the wrong payload or stage cannot pass by sharing its neighbour's value.
        x_exp = torch.randint(-3, 2, (m, k // 32), device="cuda")
        w_exp = torch.randint(-3, 2, (experts, n, k // 32), device="cuda")
        x = (
            0.1
            * torch.randn((m, k), device="cuda")
            * torch.exp2(x_exp.float()).repeat_interleave(32, dim=-1)
        ).bfloat16()
        w = (
            0.1
            * torch.randn((experts, n, k), device="cuda")
            * torch.exp2(w_exp.float()).repeat_interleave(32, dim=-1)
        ).bfloat16()
        w = w.transpose(1, 2)
        x, x_scales = downcast_to_mxfp(x, torch.uint8, axis=-1)
        w, w_scales = downcast_to_mxfp(w, torch.uint8, axis=1)
        x_ref = upcast_from_mxfp(x, x_scales, torch.bfloat16, axis=-1).float()
        w_ref = upcast_from_mxfp(w, w_scales, torch.bfloat16, axis=1).float()
        bias = torch.randn((experts, n), device="cuda", dtype=torch.float32) * 0.3
        gammas = torch.rand((m * topk,), device="cuda", dtype=torch.float32) + 0.25
        data = SimpleNamespace(
            n=n,
            k=k,
            x=x,
            w=w,
            x_scales=x_scales,
            w_scales=w_scales,
            bias=bias,
            gammas=gammas,
            routing=route,
            gather=gather,
            bias_rows=torch.repeat_interleave(bias, route.expt_hist.long(), dim=0),
            raw=moe_gemm_torch(x_ref, w_ref, None, route, gather),
        )
        baseline = torch.empty(
            (1, m * topk, n // 2), device="cuda", dtype=torch.float32
        )
        _launch(data, baseline, config=_config())
        torch.testing.assert_close(
            baseline[0],
            _expected(
                data,
                split=True,
                epilogue=EpilogueMode.DEFAULT,
                alpha=1.0,
                limit=None,
                residual=False,
            ),
            rtol=2e-4,
            atol=2e-5,
        )
        data.baseline = baseline
        yield data


@pytest.mark.parametrize("mask", range(16), ids=lambda mask: f"mask{mask}")
@pytest.mark.parametrize(
    "pipeline", [WarpPipeline.NONE, WarpPipeline.COMPILER], ids=["none", "compiler"]
)
def test_component_read_placement(case, mask, pipeline):
    config = _config(mask, pipeline)
    _assert_repeated_output(case, config, f"mask={mask}, pipeline={pipeline.name}")


def _assert_repeated_output(case, config, label):
    output = torch.empty_like(case.baseline)
    for repeat in range(16):
        output.fill_(float("nan"))
        _launch(case, output, config=config)
        torch.testing.assert_close(
            output.view(torch.int32),
            case.baseline.view(torch.int32),
            rtol=0,
            atol=0,
            msg=f"{label}, repeat={repeat}",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("SCHED_MODE", 1, id="iglp0"),
        pytest.param("SCHED_MODE", 2, id="iglp1"),
        pytest.param("SCHED_MODE", 3, id="mfma16"),
        pytest.param("SCHED_MODE", 4, id="mfma8"),
        pytest.param("SOFF_UNROLL", True, id="soffset-unroll"),
        pytest.param("SCALE_FILL_MID", False, id="scales-with-payload"),
    ],
)
def test_explicit_pipeline_controls(case, field, value):
    config = _config(mask=15)
    config[field] = value
    _assert_repeated_output(case, config, f"{field}={value}")


@pytest.mark.parametrize(
    "pipeline", [WarpPipeline.NONE, WarpPipeline.COMPILER], ids=["none", "compiler"]
)
def test_rectangular_read_schedule(case, pipeline):
    # A 4x2 walk reads only in slots 0..5. Slots 6 and 7 do not read, so delaying
    # the non-relaxed read until the final slot would leave every actual read relaxed.
    config = _config(mask=15, pipeline=pipeline)
    config["MINI_BLOCK_M"] = 32
    _assert_repeated_output(case, config, f"4x2 mini tiles, pipeline={pipeline.name}")


@pytest.mark.parametrize(
    "scheme",
    [WaitCommitScheme.PER_FILL, WaitCommitScheme.PER_SLOT],
    ids=["per-fill", "per-slot"],
)
@pytest.mark.parametrize(
    "middle", [False, True], ids=["scales-with-payload", "middle-scales"]
)
def test_scale_wait_accounting(case, scheme, middle):
    # A payload and B scales cross into MFMA; their partners stay in the memory
    # region. Moving the scale fills changes which commit group each read needs.
    config = _config(mask=9)
    config["WAIT_COMMIT_SCHEME"] = int(scheme)
    config["SCALE_FILL_MID"] = middle
    _assert_repeated_output(case, config, f"{scheme.name}, SCALE_FILL_MID={middle}")


@pytest.mark.parametrize("split", [False, True], ids=["interleaved", "split"])
@pytest.mark.parametrize("epilogue", list(EpilogueMode), ids=lambda mode: mode.name)
def test_epilogue_modes_have_distinct_arithmetic(case, split, epilogue):
    output = torch.empty_like(case.baseline)
    kwargs = {
        "split": split,
        "epilogue": epilogue,
        "alpha": 1.702,
        "limit": 0.5,
        "residual": True,
    }
    expected = _expected(case, **kwargs)
    first = None
    for repeat in range(8):
        output.fill_(float("nan"))
        _launch(case, output, config=_config(), **kwargs)
        torch.testing.assert_close(output[0], expected, rtol=2e-4, atol=2e-5)
        if first is None:
            first = output.clone()
        else:
            torch.testing.assert_close(
                output.view(torch.int32),
                first.view(torch.int32),
                rtol=0,
                atol=0,
                msg=f"epilogue={epilogue.name}, split={split}, repeat={repeat}",
            )
