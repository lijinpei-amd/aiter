# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gfx950 Gluon MoE GEMM2 half-height tile gives the same rows as the full tile.

With ``HALF_M_TILE``, an M block that holds at most ``BLOCK_M / 2`` rows runs a
second tile, ``BLOCK_M / 2`` rows high. Its rows must be bit-identical to the rows
that the full tile computes. The routing here gives block_m 128 and experts with 1,
64, 65 and 126 rows, so both tiles run in one launch.
"""

import os

import pytest
import torch

from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

N, K = 512, 2048
#: Rows of each expert: two blocks take the half-height tile (1 and 64 rows), two
#: take the full tile (65 and 126 rows).
ROWS = (1, 64, 65, 126)
_FP8, _FP4 = torch.float8_e4m3fn, torch.uint8
#: op -> (public wrapper, activation storage, weight storage)
_OPS = {
    "a8w8": (moe_gemm_a8w8, _FP8, _FP8),
    "a8w4": (moe_gemm_a8w4, _FP8, _FP4),
    "a4w4": (moe_gemm_a4w4, _FP4, _FP4),
}


def _clear_config_caches():
    for cached in (
        host._launch_spec,
        conf._get_gluon_config_cached,
        conf._gemm2_launch_config,
        conf._gemm2_half_m_config,
    ):
        cached.cache_clear()


@pytest.fixture
def clean_env(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("AITER_TRITON_MOE_DISABLE_GLUON", raising=False)
    _clear_config_caches()
    yield monkeypatch
    _clear_config_caches()


@pytest.fixture
def half_tiles(monkeypatch):
    """The BLOCK_M of the half-height tile of every Gluon launch, or None."""
    seen = []
    original = host._fast_launch

    def capture(kernel, grid, args, *metadata):
        half = dict(zip(kernel.arg_names, args)).get("CFG_TUNING_HALF")
        half = conf._cval(half)
        seen.append(None if half is None else half.BLOCK_M)
        return original(kernel, grid, args, *metadata)

    monkeypatch.setattr(host, "_fast_launch", capture)
    return seen


def _case(op):
    """Routing with block_m 128 and ROWS rows per expert, and MX operands."""
    experts = len(ROWS)
    expert = torch.repeat_interleave(
        torch.arange(experts, device="cuda"),
        torch.tensor(ROWS, device="cuda"),
    )
    logits = torch.zeros((sum(ROWS), experts), dtype=torch.float16, device="cuda")
    logits[torch.arange(sum(ROWS), device="cuda"), expert] = 1
    route, gather, scatter = routing(logits, 1)
    assert route.block_m == 128
    torch.manual_seed(7)
    x = torch.randn((sum(ROWS), K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((experts, N, K), dtype=torch.bfloat16, device="cuda")
    x, xs = downcast_to_mxfp(x, _OPS[op][1], axis=-1)
    w, ws = downcast_to_mxfp(w.transpose(1, 2), _OPS[op][2], axis=1)
    return x, w, xs, ws, route, gather, scatter


def _gemm2(op, case):
    x, w, xs, ws, route, gather, scatter = case
    return _OPS[op][0](
        x, w, xs, ws, routing_data=route, gather_indx=gather, scatter_indx=scatter
    )


@pytest.mark.parametrize("op", list(_OPS))
def test_half_m_tile_rows_match_the_full_tile(op, clean_env, half_tiles):
    case = _case(op)
    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "0")
    full = _gemm2(op, case)
    assert half_tiles == [None]

    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "1")
    _clear_config_caches()
    half = _gemm2(op, case)
    assert half_tiles == [None, 64]
    assert torch.equal(half, full)

    clean_env.setenv("AITER_TRITON_MOE_DISABLE_GLUON", "1")
    triton = _gemm2(op, case)
    torch.testing.assert_close(half.float(), triton.float(), atol=1e-1, rtol=2e-2)


def test_a8w4_family_takes_the_half_m_tile(clean_env, half_tiles):
    _gemm2("a8w4", _case("a8w4"))
    assert half_tiles == [64]
