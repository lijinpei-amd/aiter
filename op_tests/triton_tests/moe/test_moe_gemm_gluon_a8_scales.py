# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gfx950 MXFP8 scale dwords shared by pairs of K128 payload stages."""

import os
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    DtypeQuant,
    WaitCommitScheme,
    WarpPipeline,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import assert_close


@pytest.fixture(scope="module")
def case():
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    with pytest.MonkeyPatch.context() as patch:
        for name in os.environ:
            if name.startswith("AITER_TRITON_MOE_GLUON_"):
                patch.delenv(name)
        patch.delenv("AITER_TRITON_MOE_DISABLE_GLUON", raising=False)
        host._launch_spec.cache_clear()
        host._get_gluon_config_cached.cache_clear()
        torch.manual_seed(73)
        # NE33/K7168 has a generated C++ scale-sort instance. The routed row count
        # chooses BM128 and includes partial expert tiles. 56 K128 stages exercise
        # paired scale halves in the prologue, unrolled body, remainder and drain.
        m, n, k, experts, topk = 513, 512, 7168, 33, 8
        route, gather, _ = routing(
            torch.randn(m, experts, device="cuda", dtype=torch.float16), topk
        )
        route.gate_scal = None
        assert route.block_m == 128
        assert bool(torch.any(route.expt_hist % 128 != 0))

        # Adjacent K groups and rows get different exponents, so selecting the
        # wrong half-dword or attaching a scale to a neighbouring row is visible.
        x_exp = torch.randint(-3, 2, (m, k // 32), device="cuda")
        w_exp = torch.randint(-3, 2, (experts, n, k // 32), device="cuda")
        x = (
            0.1
            * torch.randn(m, k, device="cuda")
            * torch.exp2(x_exp.float()).repeat_interleave(32, dim=-1)
        ).bfloat16()
        w = (
            0.1
            * torch.randn(experts, n, k, device="cuda")
            * torch.exp2(w_exp.float()).repeat_interleave(32, dim=-1)
        ).bfloat16()
        x, xs = downcast_to_mxfp(x, torch.float8_e4m3fn, axis=-1)
        w, ws = downcast_to_mxfp(w.transpose(1, 2), torch.float8_e4m3fn, axis=1)
        x_ref = upcast_from_mxfp(x, xs, torch.bfloat16, axis=-1).float()
        w_ref = upcast_from_mxfp(w, ws, torch.bfloat16, axis=1).float()
        raw = moe_gemm_torch(x_ref, w_ref, None, route, gather)
        gate, linear = raw.chunk(2, dim=-1)
        expected = {
            True: torch.nn.functional.silu(gate) * linear,
            False: torch.nn.functional.silu(raw[:, ::2]) * raw[:, 1::2],
        }
        cfg = host.get_gluon_config_uncached(
            128, n, k, DtypeQuant.MXFP8, DtypeQuant.MXFP8
        )
        cfg.update(
            BLOCK_N=256,
            BLOCK_K=128,
            MINI_BLOCK_M=64,
            MINI_BLOCK_N=128,
            MINI_BLOCK_K=128,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(1, 4),
            tiles_per_warp=(2, 2),
            NUM_LDS_BUFFER=3,
            K_UNROLL=6,
            VGPR_PREFETCH_K=128,
            A_SCALE_SORTED_SHUFFLED=True,
            B_SCALE_SHUFFLED=True,
        )
        yield SimpleNamespace(
            n=n,
            k=k,
            x=x,
            w=w,
            xs=xs,
            ws=ws,
            route=route,
            gather=gather,
            expected=expected,
            config=cfg,
        )
        host._launch_spec.cache_clear()
        host._get_gluon_config_cached.cache_clear()


def _check(case, monkeypatch, *, config=None, split=True, packed=True, public=False):
    output = torch.empty_like(case.expected[split], dtype=torch.bfloat16).unsqueeze(0)
    seen = []
    original = host._fast_launch

    def capture(kernel, grid, args, *metadata):
        named = dict(zip(kernel.arg_names, args))
        tuning = host._cval(named["CFG_TUNING"])
        shuffled = packed if isinstance(packed, tuple) else (packed, packed)
        assert (tuning.A_SCALE_SORTED_SHUFFLED, tuning.B_SCALE_SHUFFLED) == shuffled
        assert tuning.BLOCK_K == (config or case.config)["BLOCK_K"]
        assert tuning.K_UNROLL == (config or case.config)["K_UNROLL"]
        assert named["a_ptr"] is case.x and named["b_ptr"] is case.w
        assert (named["a_scale_ptr"] is case.xs) == (not shuffled[0])
        assert (named["b_scale_ptr"] is case.ws) == (not shuffled[1])
        seen.append(True)
        return original(kernel, grid, args, *metadata)

    monkeypatch.setattr(host, "_fast_launch", capture)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_GU_SPLIT", str(int(split)))
    first = None
    for _ in range(8):
        output.fill_(float("nan"))
        if public:
            assert config is None
            ok = host.try_gluon_grouped_gemm(
                op_name="packed_mxfp8_scale_test",
                y=output,
                x=case.x,
                w=case.w,
                x_scales=case.xs,
                w_scales=case.ws,
                bias=None,
                gammas=None,
                routing_data=case.route,
                gather_indx=case.gather,
                scatter_indx=None,
                N=case.n,
                K=case.k,
                apply_swiglu=True,
                alpha=1.0,
                limit=None,
                swiglu_add_residual=False,
                split_k=1,
            )
            assert ok, "The public stage1 capability check must select Gluon."
        else:
            host.moe_gemm_gluon(
                output,
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
                True,
                1.0,
                None,
                False,
                config=config,
                gate_up_split=split,
            )
        assert torch.isfinite(output).all()
        if first is None:
            assert_close(
                case.expected[split], output[0], description="packed MXFP8 K128"
            )
            first = output.clone()
        else:
            assert torch.equal(first, output)
    assert len(seen) == 8


@pytest.mark.parametrize("scheme", list(WaitCommitScheme), ids=lambda mode: mode.name)
@pytest.mark.parametrize(
    "pipeline", [WarpPipeline.NONE, WarpPipeline.COMPILER], ids=["none", "compiler"]
)
@pytest.mark.parametrize("soff_unroll", [False, True], ids=["pointers", "soffsets"])
def test_packed_mxfp8_scales_across_pipeline_modes(
    case, scheme, pipeline, soff_unroll, monkeypatch
):
    config = dict(
        case.config,
        WAIT_COMMIT_SCHEME=int(scheme),
        WARP_PIPELINE=int(pipeline),
        SOFF_UNROLL=soff_unroll,
    )
    _check(case, monkeypatch, config=config)


@pytest.mark.parametrize("split", [False, True], ids=["interleaved", "split"])
def test_public_mxfp8_gemm1_prepares_default_packed_scales(case, split, monkeypatch):
    _check(case, monkeypatch, split=split, public=True)


@pytest.mark.parametrize("unavailable", ["a", "b"])
def test_public_mxfp8_gemm1_restores_only_unavailable_raw_scale(
    case, unavailable, monkeypatch
):
    helper = "_sorted_shuffle_a_scales" if unavailable == "a" else "_shuffled_b_scales"
    monkeypatch.setattr(host, helper, lambda *_: None)
    _check(
        case, monkeypatch, packed=(unavailable != "a", unavailable != "b"), public=True
    )


@pytest.mark.parametrize("scheme", list(WaitCommitScheme), ids=lambda mode: mode.name)
@pytest.mark.parametrize(
    "scale_shape,shuffled,registers,depths,split,soff",
    [
        ((128, 256, 256), (True, True), False, (3, 3, 3, 3), True, False),
        ((128, 256, 512), (True, True), False, (2, 3, 2, 3), False, True),
        ((128, 256, 1024), (True, True), True, (2, 2, 2, 2), True, True),
        ((128, 256, 512), (True, False), False, (3, 2, 3, 2), True, True),
        ((128, 256, 512), (False, True), True, (3, 2, 2, 3), False, False),
        ((1, 1, 1), (False, False), False, (3, 3, 3, 3), True, False),
    ],
    ids=["mn", "mnk", "register-k1024", "a-only", "b-only", "raw-ignores"],
)
def test_independent_scale_mini_blocks(
    case, monkeypatch, scheme, scale_shape, shuffled, registers, depths, split, soff
):
    """Cover scale cadence, shared M/N tiles, skipped reads, and raw-scale isolation."""
    config = dict(
        case.config,
        SCALE_MINI_BLOCK_M=scale_shape[0],
        SCALE_MINI_BLOCK_N=scale_shape[1],
        SCALE_MINI_BLOCK_K=scale_shape[2],
        A_SCALE_SORTED_SHUFFLED=shuffled[0],
        B_SCALE_SHUFFLED=shuffled[1],
        A_SCALE_IN_REG=registers,
        B_SCALE_IN_REG=registers,
        A_NUM_BUFFER=depths[0],
        B_NUM_BUFFER=depths[1],
        A_SCALE_NUM_BUFFER=depths[2],
        B_SCALE_NUM_BUFFER=depths[3],
        WAIT_COMMIT_SCHEME=int(scheme),
        SOFF_UNROLL=soff,
    )
    _check(case, monkeypatch, config=config, packed=shuffled, split=split)
