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


TOPK_TOKENS, TOPK_EXPERTS = 10, 8


def _topk_case():
    """MXFP4 operands and routing logits for TOPK_TOKENS tokens over TOPK_EXPERTS."""
    torch.manual_seed(13)
    logits = torch.randn(
        (TOPK_TOKENS, TOPK_EXPERTS), dtype=torch.float16, device="cuda"
    )
    x = torch.randn((TOPK_TOKENS, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((TOPK_EXPERTS, N, K), dtype=torch.bfloat16, device="cuda")
    x, xs = downcast_to_mxfp(x, torch.uint8, axis=-1)
    w, ws = downcast_to_mxfp(w.transpose(1, 2), torch.uint8, axis=1)
    return logits, x, w, xs, ws


def _topk_gemm(case, topk):
    logits, x, w, xs, ws = case
    route, gather, _ = routing(logits, topk)
    return moe_gemm_a4w4(
        x,
        w,
        xs,
        ws,
        routing_data=route,
        gather_indx=gather,
        out_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "topks", [(4, 5), (5, 4)], ids=["topk4_then_5", "topk5_then_4"]
)
def test_fast_launch_keeps_each_topk(topks, clean_env):
    """topk is a kernel constexpr that the memoised launch spec does not take.

    With 10 tokens over 8 experts, topk 4 and 5 give the same grid and launch spec,
    so the two launches share one fast-launch entry and only the spec key can tell
    them apart. A replay of the other topk's kernel divides the gather index by the
    wrong topk and reads the wrong token rows.
    """
    case = _topk_case()
    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "0")
    expected = {topk: _topk_gemm(case, topk) for topk in topks}
    assert not host._FAST_LAUNCH_CACHE

    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "1")
    for topk in topks:
        assert torch.equal(
            _topk_gemm(case, topk), expected[topk]
        ), f"topk {topk} after a topk {topks[0]} launch differs from the slow path"
    # One entry: both launches reached the same fast-launch key, which is the
    # collision this test needs. Two entries would mean it no longer tests anything.
    assert len(host._FAST_LAUNCH_CACHE) == 1
