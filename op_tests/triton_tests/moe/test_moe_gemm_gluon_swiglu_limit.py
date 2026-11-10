# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The SwiGLU clamp of the gfx950 Gluon MoE GEMM1.

The Gluon GEMM1 applies the activation directly to the MFMA accumulators. A clamp that
reads an accumulator before the MFMA has written it gives a result that changes from
run to run, so this test runs each case twice and compares the bits. The inputs are
randn with a large K, so most pre-activations are above the limit and the clamp has an
effect. The reference is the Triton kernel on the same inputs.
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_op_gemm_gluon import gluon_supported
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.moe_model_recipes import ACT_RECIPES

_OPS = {
    "a4w4": (moe_gemm_a4w4, torch.uint8, torch.uint8),
    "a8w8": (moe_gemm_a8w8, torch.float8_e4m3fn, torch.float8_e4m3fn),
    "a8w4": (moe_gemm_a8w4, torch.float8_e4m3fn, torch.uint8),
}
_CLAMPED = [a for a in ACT_RECIPES if a.limit is not None]


@pytest.mark.parametrize("act", _CLAMPED, ids=[a.name for a in _CLAMPED])
@pytest.mark.parametrize("op", sorted(_OPS))
def test_gemm1_swiglu_limit(op, act, monkeypatch, device="cuda"):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    m, n, k, n_expts_tot, n_expts_act = 256, 2048, 2048, 8, 2
    wrapper, x_dtype, w_dtype = _OPS[op]
    torch.manual_seed(0)
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    rdata, gindx, _ = routing(logits, n_expts_act)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    w, w_scale = downcast_to_mxfp(w, w_dtype, axis=1)
    x, x_scale = downcast_to_mxfp(x, x_dtype, axis=-1)
    rows = gindx.shape[0]
    ok, why = gluon_supported(
        x=x,
        w=w,
        x_scales=x_scale,
        w_scales=w_scale,
        y=torch.empty((1, rows, n // 2), dtype=torch.bfloat16, device=device),
        bias=None,
        routing_data=rdata,
        swizzle_mx_scale=None,
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=None,
        N=n,
        K=k,
    )
    assert ok, why

    def gemm1(gluon, swiglu=True):
        monkeypatch.setenv("AITER_TRITON_MOE_DISABLE_GLUON", "0" if gluon else "1")
        y = wrapper(
            x,
            w,
            x_scale,
            w_scale,
            routing_data=rdata,
            gather_indx=gindx,
            out_dtype=torch.bfloat16,
            apply_swiglu=swiglu,
            alpha=act.alpha,
            limit=act.limit,
            swiglu_add_residual=act.add_residual,
        )
        return y.reshape(rows, -1).clone()

    pre = gemm1(gluon=False, swiglu=False).float()
    clamped = (pre.abs() > act.limit).float().mean().item()
    assert clamped > 0.5, f"only {clamped:.2f} of the pre-activations reach the limit"

    first, second = gemm1(gluon=True), gemm1(gluon=True)
    n_diff = (first != second).sum().item()
    assert n_diff == 0, f"{n_diff} elements change between two identical launches"
    torch.testing.assert_close(first, gemm1(gluon=False), rtol=2e-2, atol=2e-2)
