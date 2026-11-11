# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fast launch of the gfx950 Gluon MoE GEMM gives the slow path's results.

The fast launch replays a compiled kernel when the launch matches a recorded one. Two
calls with the same launch spec can need other compiled kernels: other tensor dtypes,
another topk, or a Triton knob that changes the compiled kernel. The replay must not
take the first call's kernel for the second call. Each test calls each case twice, so
it also proves that the second call takes the fast path and gives the same bits.
"""

import os

import pytest
import torch
from triton import knobs

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


@pytest.fixture
def fast_launches(monkeypatch):
    """One entry for each Gluon launch: the kernel that the fast path replayed, or
    None when the launch took the slow path."""
    launches = []
    original = host._fast_launch

    def spy(*args):
        compiled = original(*args)
        launches.append(compiled)
        return compiled

    monkeypatch.setattr(host, "_fast_launch", spy)
    return launches


def _hits(launches):
    return [compiled is not None for compiled in launches]


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
def test_fast_launch_keeps_each_output_dtype(dtypes, clean_env, fast_launches):
    case = _case()
    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "0")
    expected = {dtype: _gemm(case, dtype) for dtype in dtypes}
    assert not host._FAST_LAUNCH_CACHE
    fast_launches.clear()

    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "1")
    for dtype in dtypes:
        for call in ("first", "second"):
            actual = _gemm(case, dtype)
            assert actual.dtype == dtype
            assert torch.equal(
                actual, expected[dtype]
            ), f"{call} {dtype} launch after {dtypes[0]} differs from the slow path"
    # The first launch of a dtype records it, and the second replays it.
    assert _hits(fast_launches) == [False, True, False, True]


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
def test_fast_launch_keeps_each_topk(topks, clean_env, fast_launches):
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
    fast_launches.clear()

    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "1")
    for topk in topks:
        for call in ("first", "second"):
            assert torch.equal(
                _topk_gemm(case, topk), expected[topk]
            ), f"{call} topk {topk} launch after topk {topks[0]} differs from slow path"
    assert _hits(fast_launches) == [False, True, False, True]
    # One entry: both topks reached the same fast-launch key, which is the
    # collision this test needs. Two entries would mean it no longer tests anything.
    assert len(host._FAST_LAUNCH_CACHE) == 1


@pytest.mark.parametrize("knob_first", [False, True], ids=["plain_first", "knob_first"])
@pytest.mark.parametrize(
    "owner, name, value, plain",
    [
        # Any mode is a compile option. The AMD backend runs no pass for this one, so
        # the kernel compiles; fpsan fails in TritonInstrumentFpSanitizer on it.
        (knobs.compilation, "instrumentation_mode", "aiter-test", ""),
        (knobs.runtime, "debug", True, False),
    ],
    ids=["instrumentation_mode", "debug"],
)
def test_fast_launch_keeps_the_compile_knobs(
    owner, name, value, plain, knob_first, clean_env, fast_launches
):
    """JITFunction.run passes these knobs to the compiler, so a kernel compiled with
    one of them on is not the plain kernel.

    The fast path neither replays nor records a launch with one of them on. So only
    the plain launch replays, and it replays the plain kernel.
    """
    case = _case()
    clean_env.setenv("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", "1")
    for setting in (value, plain) if knob_first else (plain, value):
        clean_env.setattr(owner, name, setting)
        _gemm(case, torch.bfloat16)
        _gemm(case, torch.bfloat16)
    plain_hits, knob_hits = [False, True], [False, False]
    assert _hits(fast_launches) == (
        knob_hits + plain_hits if knob_first else plain_hits + knob_hits
    )
    replayed = [compiled for compiled in fast_launches if compiled is not None]
    assert getattr(replayed[0].metadata, name) == plain
