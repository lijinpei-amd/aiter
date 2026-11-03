# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gfx950 Gluon MoE GEMM1 half-height tile gives the same rows as the full tile.

With ``HALF_M_TILE``, an M block that holds at most ``BLOCK_M / 2`` rows runs a
second tile, ``BLOCK_M / 2`` rows high. Its rows must be bit-identical to the rows
that the full tile computes. The routing here gives block_m 128 (a mean of exactly
64 rows) and experts with 0, 1, 64, 65, 126 and 150 rows, so both tiles run in one
launch, and the second block of the 150-row expert (22 rows) takes the half-height
tile at ``block_id`` 1. a4w4 takes the tile only through the environment; a8w8
has none.
"""

import os

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant, ScaleSwizzle
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, moe_gemm_torch
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp

#: NE 33 and K 3584 have a C++ scale-sort instance, so the A scales are sorted in
#: block_m 128 blocks and the half-height tile reads them at 2 * pid_m.
N, K, EXPERTS = 512, 3584, 33
#: Rows of each expert: a mean of exactly 64, so the routing gives block_m 128.
ROWS = (0, 1, 64, 65, 126, 150) + (64,) * 5 + (63,) * 22
_FP8, _FP4 = torch.float8_e4m3fn, torch.uint8
#: op -> (public wrapper, activation storage, weight storage)
_OPS = {
    "a8w4": (moe_gemm_a8w4, _FP8, _FP4),
    "a8w8": (moe_gemm_a8w8, _FP8, _FP8),
    "a4w4": (moe_gemm_a4w4, _FP4, _FP4),
}


def _clear_config_caches():
    for cached in (
        host._launch_spec,
        conf._get_gluon_config_cached,
        conf._gemm1_family_entry,
        conf._gemm1_half_m_config,
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
def launches(monkeypatch):
    """(HALF_M_TILE, half-height BLOCK_M or None, A scale swizzle) of every launch."""
    seen = []
    original = host._fast_launch

    def capture(kernel, grid, args, *metadata):
        named = dict(zip(kernel.arg_names, args))
        half = conf._cval(named.get("CFG_TUNING_HALF"))
        tuning = conf._cval(named["CFG_TUNING"])
        seen.append(
            (
                tuning.HALF_M_TILE,
                None if half is None else half.BLOCK_M,
                conf._cval(named["A_SCALE_SWIZZLE"]),
            )
        )
        return original(kernel, grid, args, *metadata)

    monkeypatch.setattr(host, "_fast_launch", capture)
    return seen


@pytest.fixture(scope="module")
def cases():
    """op -> routing with block_m 128 and ROWS rows per expert, and MX operands."""
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    expert = torch.repeat_interleave(
        torch.arange(EXPERTS, device="cuda"), torch.tensor(ROWS, device="cuda")
    )
    logits = torch.full(
        (sum(ROWS), EXPERTS), float("-inf"), dtype=torch.float16, device="cuda"
    )
    logits[torch.arange(sum(ROWS), device="cuda"), expert] = 1
    route, gather, _ = routing(logits, 1)
    route.gate_scal = None
    assert route.block_m == 128
    out = {}
    for op, (_, x_dtype, w_dtype) in _OPS.items():
        torch.manual_seed(7)
        x = torch.randn((sum(ROWS), K), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((EXPERTS, N, K), dtype=torch.bfloat16, device="cuda") / 16
        bias = torch.randn((EXPERTS, N), dtype=torch.float32, device="cuda")
        x, xs = downcast_to_mxfp(x, x_dtype, axis=-1)
        w, ws = downcast_to_mxfp(w.transpose(1, 2), w_dtype, axis=1)
        out[op] = (x, w, xs, ws, bias, route, gather)
    return out


def _gemm1(op, case):
    x, w, xs, ws, bias, route, gather = case
    return _OPS[op][0](
        x,
        w,
        xs,
        ws,
        bias=bias,
        routing_data=route,
        gather_indx=gather,
        apply_swiglu=True,
        alpha=1.702,
        limit=7.0,
    )


def _reference(op, case):
    x, w, xs, ws, bias, route, gather = case
    x_ref = upcast_from_mxfp(x, xs, torch.bfloat16, axis=-1).float()
    w_ref = upcast_from_mxfp(w, ws, torch.bfloat16, axis=1).float()
    return moe_gemm_torch(
        x_ref,
        w_ref,
        bias,
        route,
        gather,
        apply_swiglu=True,
        alpha=1.702,
        limit=7.0,
    )


@pytest.mark.parametrize("op", ["a8w4", "a4w4"])
def test_half_m_tile_rows_match_the_full_tile(op, cases, clean_env, launches):
    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "0")
    full = _gemm1(op, cases[op])
    assert [launch[:2] for launch in launches] == [(False, None)]

    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "1")
    _clear_config_caches()
    half = _gemm1(op, cases[op])
    assert launches[1][:2] == (True, 64)
    # Both launches read the same A scales; for MX activations, the sorted ones.
    assert launches[0][2] == launches[1][2] == ScaleSwizzle.SORTED_SHUFFLED
    assert torch.equal(half, full)

    ref = _reference(op, cases[op])
    torch.testing.assert_close(half.float(), ref, atol=0.25, rtol=0.05)


def test_a8w4_takes_the_half_m_tile_by_default(cases, clean_env, launches):
    _gemm1("a8w4", cases["a8w4"])
    assert [launch[:2] for launch in launches] == [(True, 64)]


def test_a8w8_keeps_one_tile(cases, clean_env, launches):
    # Its block_m 64 config has another BLOCK_N and warp split, so no half-height
    # tile fits the block_m 128 launch.
    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "1")
    out = _gemm1("a8w8", cases["a8w8"])
    assert [launch[:2] for launch in launches] == [(False, None)]
    ref = _reference("a8w8", cases["a8w8"])
    torch.testing.assert_close(out.float(), ref, atol=0.25, rtol=0.05)


def _a8w4_gemm1_config(block_m):
    return conf._default_launch_config(
        block_m, N, K, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, True
    )


def test_a8w4_gemm1_family_sets_the_half_m_tile(clean_env):
    # The A8W4-GEMM1 family entry asks for the tile at block_m 128 only.
    entry = conf.get_moe_dispatch("A8W4-GEMM1", conf.get_arch(), "gluon")
    assert {k: v for k, v in entry.items() if v} == {"bm128_any": {"HALF_M_TILE": True}}
    assert _a8w4_gemm1_config(128)["HALF_M_TILE"] is True
    assert _a8w4_gemm1_config(64)["HALF_M_TILE"] is False
    # Not for GEMM2.
    gemm2 = conf._default_launch_config(
        128, N, K, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, False
    )
    assert not gemm2["HALF_M_TILE"]


@pytest.mark.parametrize("value", ["0", "1"])
def test_environment_wins_over_the_gemm1_family(value, clean_env):
    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", value)
    assert _a8w4_gemm1_config(128)["HALF_M_TILE"] is (value == "1")


def test_no_gemm1_family_entry_keeps_one_tile(clean_env, monkeypatch):
    monkeypatch.setattr(conf, "get_moe_dispatch", lambda *_: {})
    assert _a8w4_gemm1_config(128)["HALF_M_TILE"] is False
