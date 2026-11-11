# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fast launch of the gfx950 Gluon MoE GEMM gives the slow path's results.

The fast launch replays a compiled kernel when the launch matches a recorded one. Two
calls with the same launch spec but other tensor dtypes need other compiled kernels,
so the replay must not take the first call's kernel for the second call.
"""

import os

import pytest
import torch

from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

N, K, EXPERTS, ROWS_PER_EXPERT = 2048, 2048, 16, 48


@pytest.fixture
def clean_env(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("AITER_TRITON_MOE_DISABLE_GLUON", raising=False)
    host._FAST_LAUNCH_CACHE.clear()
    yield monkeypatch
    host._FAST_LAUNCH_CACHE.clear()


def _case():
    """MXFP4 operands and a routing with ROWS_PER_EXPERT rows for each expert."""
    rows = EXPERTS * ROWS_PER_EXPERT
    expert = torch.arange(rows, device="cuda") // ROWS_PER_EXPERT
    logits = torch.zeros((rows, EXPERTS), dtype=torch.float16, device="cuda")
    logits[torch.arange(rows, device="cuda"), expert] = 1
    route, gather, _ = routing(logits, 1)
    torch.manual_seed(11)
    x = torch.randn((rows, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((EXPERTS, N, K), dtype=torch.bfloat16, device="cuda")
    x, xs = downcast_to_mxfp(x, torch.uint8, axis=-1)
    w, ws = downcast_to_mxfp(w.transpose(1, 2), torch.uint8, axis=1)
    return x, w, xs, ws, route, gather


def _gemm(case, out_dtype):
    x, w, xs, ws, route, gather = case
    return moe_gemm_a4w4(
        x, w, xs, ws, routing_data=route, gather_indx=gather, out_dtype=out_dtype
    )


@pytest.mark.parametrize(
    "dtypes",
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
    ids=["fp32_then_bf16", "bf16_then_fp32"],
)
def test_fast_launch_keeps_each_output_dtype(dtypes, clean_env):
    case = _case()
    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "0")
    expected = {dtype: _gemm(case, dtype) for dtype in dtypes}
    assert not host._FAST_LAUNCH_CACHE

    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "1")
    for dtype in dtypes:
        actual = _gemm(case, dtype)
        assert actual.dtype == dtype
        assert torch.equal(
            actual, expected[dtype]
        ), f"{dtype} output after a {dtypes[0]} launch differs from the slow path"
    assert host._FAST_LAUNCH_CACHE, "the Gluon kernel did not run"
