# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operand-dtype coverage for the gfx950 Gluon MoE grouped GEMM.

``moe_gemm_gluon`` infers both operand dtypes from the tensors, and the two sides are
independent: a16w4 is bf16 x fp4, a8w4 is fp8 x fp4. Everything derived from a dtype --
LDS element type, tile shape, copy width, ``k_width``, the MFMA instruction shape, and
which matrix instruction the pair maps onto -- is therefore per operand, and a bug
where one operand's dtype is applied to both shows up only in a mixed configuration.

The ``moe_gemm_*`` op suites cover the pairs their own op ships (a4w4, a8w8, a8w4), but
**bf16 x bf16 has no in-tree MoE caller at all**, so it is exercised here by driving the
launcher directly. This file is also where the refusals are pinned: a combination the
kernel does not serve must be reported by ``gluon_supported``, not discovered as a
miscompile.
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_torch
from aiter.ops.triton.moe.moe_op_gemm_gluon import (
    gluon_supported,
    infer_dtype_quant,
    moe_gemm_gluon,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import assert_close

# (id, x dtype tag, w dtype tag)
_PAIRS = [
    ("bf16xbf16", "bf16", "bf16"),
    ("mxfp4xmxfp4", "mxfp4", "mxfp4"),
    ("mxfp8xmxfp8", "mxfp8", "mxfp8"),
    ("mxfp8xmxfp4", "mxfp8", "mxfp4"),
]


def _quantize(t: torch.Tensor, tag: str, axis: int):
    """Returns (stored, scales, dequantised-reference)."""
    if tag == "bf16":
        return t, None, t
    if tag == "mxfp4":
        q, s = downcast_to_mxfp(t, torch.uint8, axis=axis)
    else:
        q, s = downcast_to_mxfp(t, torch.float8_e4m3fn, axis=axis)
    return q, s, upcast_from_mxfp(q, s, torch.bfloat16, axis=axis)


@pytest.mark.parametrize(
    "x_tag, w_tag", [(x, w) for _, x, w in _PAIRS], ids=[i for i, _, _ in _PAIRS]
)
@pytest.mark.parametrize("m, n, k", [(1024, 2048, 2048), (16, 4096, 4096)])
def test_operand_dtype_pairs(x_tag, w_tag, m, n, k, device="cuda"):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    n_expts_tot, n_expts_act = 64, 4
    torch.manual_seed(0)
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    rdata, gindx, _ = routing(logits, n_expts_act)
    rdata.gate_scal = None

    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    # (E, K, N) with stride(-2) == 1, the layout contract every path asserts
    w = torch.randn((n_expts_tot, n, k), device=device, dtype=torch.bfloat16).transpose(
        1, 2
    )
    bias = torch.randn((n_expts_tot, n), device=device, dtype=torch.float32)
    gammas = torch.rand((gindx.shape[0],), device=device, dtype=torch.float32)

    x_q, x_s, x_ref = _quantize(x, x_tag, axis=-1)
    w_q, w_s, w_ref = _quantize(w, w_tag, axis=1)
    assert w_q.stride(-2) == 1

    M = gindx.shape[0]
    y = torch.zeros((1, M, n), device=device, dtype=torch.bfloat16)
    ok, why = gluon_supported(
        x=x_q,
        w=w_q,
        x_scales=x_s,
        w_scales=w_s,
        y=y,
        bias=bias,
        routing_data=rdata,
        swizzle_mx_scale=None,
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=None,
        N=n,
        K=k,
    )
    if not ok:
        pytest.skip(f"not on the Gluon path: {why}")

    moe_gemm_gluon(
        y,
        x_q,
        w_q,
        x_s,
        w_s,
        bias,
        gammas,
        rdata,
        gindx,
        None,
        n,
        k,
        False,
        1.0,
        None,
        False,
    )
    ref = moe_gemm_torch(x_ref, w_ref, bias, rdata, gindx, None, gammas)
    assert_close(ref, y[0], description=f"{x_tag}x{w_tag}")


def test_dtype_inference():
    """The wrapper's whole dtype dispatch hangs off this one function."""
    from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u8 = torch.zeros(4, 4, dtype=torch.uint8, device=dev)
    fp8 = torch.zeros(4, 4, dtype=torch.float8_e4m3fn, device=dev)
    bf = torch.zeros(4, 4, dtype=torch.bfloat16, device=dev)
    assert infer_dtype_quant(u8, u8) == DtypeQuant.MXFP4
    assert infer_dtype_quant(fp8, u8) == DtypeQuant.MXFP8
    assert infer_dtype_quant(fp8, None) == DtypeQuant.FP8_E4M3
    assert infer_dtype_quant(bf, None) == DtypeQuant.BF16
    # packed fp4 without a scale is not a format, and a scaled bf16 tensor is not one
    # of ours either -- both must be refused rather than guessed at.
    assert infer_dtype_quant(u8, None) is None
    assert infer_dtype_quant(bf, u8) is None


def test_bf16_times_mxfp4_is_refused(device="cuda"):
    """bf16 x microscaled needs ``scaled_upcast`` + plain mfma.

    ``amdg.scaled_upcast_fp4`` has no working lowering in this Triton revision (the only
    upstream coverage is parse-only), so the pair must be reported as unsupported and
    fall back to the Triton kernel rather than miscompile.
    """
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    m, n, k, n_expts_tot, n_expts_act = 256, 2048, 2048, 64, 4
    torch.manual_seed(0)
    rdata, gindx, _ = routing(
        torch.randn((m, n_expts_tot), dtype=torch.float16, device=device), n_expts_act
    )
    rdata.gate_scal = None
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n_expts_tot, n, k), device=device, dtype=torch.bfloat16).transpose(
        1, 2
    )
    w_q, w_s = downcast_to_mxfp(w, torch.uint8, axis=1)
    ok, why = gluon_supported(
        x=x,
        w=w_q,
        x_scales=None,
        w_scales=w_s,
        y=torch.zeros((1, gindx.shape[0], n), device=device, dtype=torch.bfloat16),
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
    assert not ok
    assert "scaled_upcast" in why
