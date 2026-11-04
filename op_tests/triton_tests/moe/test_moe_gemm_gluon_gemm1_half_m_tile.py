# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gfx950 Gluon MoE GEMM1 half-height tile gives the same rows as the full tile.

With ``HALF_M_TILE``, an M block that holds at most ``BLOCK_M / 2`` rows runs a
second tile, ``BLOCK_M / 2`` rows high. Its rows must be bit-identical to the rows
that the full tile computes. The routings here give block_m 128, 64 and 32 (a mean of
exactly block_m / 2 rows; experts with 0, 1, block_m / 2, block_m / 2 + 1, block_m - 2
and more than block_m rows), so both tiles run in one launch, and the second block of
the largest expert takes the half-height tile at ``block_id`` 1. The GEMM1 family of
each dtype pair asks for the tile at each of them.
"""

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
from op_tests.triton_tests.moe.moe_test_utils import (
    clean_gluon_tuning_environment,
    clear_gluon_config_caches,
)
from op_tests.triton_tests.utils.mxfp_ref import upcast_from_mxfp

#: NE 33 and K 3584 have a C++ scale-sort instance, so the A scales are sorted in
#: block_m blocks and the half-height tile reads them at 2 * pid_m. N 2048 makes the
#: grid large enough for the family's block_m 64 tile (a small grid keeps the ladder).
N, K, EXPERTS = 2048, 3584, 33
#: block_m -> rows of each expert: a mean of exactly block_m / 2, so the routing gives
#: that block_m.
ROWS = {
    128: (0, 1, 64, 65, 126, 150) + (64,) * 5 + (63,) * 22,
    64: (0, 1, 32, 33, 62, 75) + (32,) * 16 + (31,) * 11,
    32: (0, 1, 16, 17, 30, 40) + (16,) * 19 + (15,) * 8,
}
_FP8, _FP4 = torch.float8_e4m3fn, torch.uint8
#: op -> (public wrapper, activation storage, weight storage, operand dtypes)
_OPS = {
    "a8w4": (moe_gemm_a8w4, _FP8, _FP4, (DtypeQuant.MXFP8, DtypeQuant.MXFP4)),
    "a8w8": (moe_gemm_a8w8, _FP8, _FP8, (DtypeQuant.MXFP8, DtypeQuant.MXFP8)),
    "a4w4": (moe_gemm_a4w4, _FP4, _FP4, (DtypeQuant.MXFP4, DtypeQuant.MXFP4)),
}


@pytest.fixture
def clean_env(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    monkeypatch.delenv("AITER_TRITON_MOE_DISABLE_GLUON", raising=False)
    yield from clean_gluon_tuning_environment(monkeypatch)


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
    """(op, block_m) -> routing with ROWS[block_m] rows per expert, and MX operands."""
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    weights = {}
    for op, (_, _, w_dtype, _) in _OPS.items():
        torch.manual_seed(7)
        w = torch.randn((EXPERTS, N, K), dtype=torch.bfloat16, device="cuda") / 16
        bias = torch.randn((EXPERTS, N), dtype=torch.float32, device="cuda")
        weights[op] = (*downcast_to_mxfp(w.transpose(1, 2), w_dtype, axis=1), bias)
    out = {}
    for block_m, rows in ROWS.items():
        expert = torch.repeat_interleave(
            torch.arange(EXPERTS, device="cuda"), torch.tensor(rows, device="cuda")
        )
        logits = torch.full(
            (sum(rows), EXPERTS), float("-inf"), dtype=torch.float16, device="cuda"
        )
        logits[torch.arange(sum(rows), device="cuda"), expert] = 1
        route, gather, _ = routing(logits, 1)
        route.gate_scal = None
        assert route.block_m == block_m
        for op, (_, x_dtype, _, _) in _OPS.items():
            x = torch.randn((sum(rows), K), dtype=torch.bfloat16, device="cuda")
            x, xs = downcast_to_mxfp(x, x_dtype, axis=-1)
            w, ws, bias = weights[op]
            out[op, block_m] = (x, w, xs, ws, bias, route, gather)
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


@pytest.mark.parametrize("block_m", sorted(ROWS))
@pytest.mark.parametrize("op", sorted(_OPS))
def test_half_m_tile_rows_match_the_full_tile(op, block_m, cases, clean_env, launches):
    case = cases[op, block_m]
    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "0")
    full = _gemm1(op, case)
    assert [launch[:2] for launch in launches] == [(False, None)]

    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "1")
    clear_gluon_config_caches()
    half = _gemm1(op, case)
    assert launches[1][:2] == (True, block_m // 2)
    # Both launches read the same A scales: at block_m 64 and 128, the sorted ones.
    assert launches[0][2] == launches[1][2]
    assert (launches[0][2] == ScaleSwizzle.SORTED_SHUFFLED) == (block_m >= 64)
    assert torch.equal(half, full)

    ref = _reference(op, case)
    torch.testing.assert_close(half.float(), ref, atol=0.25, rtol=0.05)


@pytest.mark.parametrize("block_m", sorted(ROWS))
@pytest.mark.parametrize("op", sorted(_OPS))
def test_every_dtype_takes_the_half_m_tile_by_default(
    op, block_m, cases, clean_env, launches
):
    _gemm1(op, cases[op, block_m])
    assert [launch[:2] for launch in launches] == [(True, block_m // 2)]


def test_a_half_tile_that_does_not_fit_keeps_one_tile(
    cases, clean_env, launches, monkeypatch
):
    # Without its family's block_m 64 tile, the a8w8 block_m 64 config has another
    # BLOCK_N and warp split, so no half-height tile fits the block_m 128 launch.
    table = dict(conf.get_moe_dispatch("A8W8-GEMM1", conf.get_arch(), "gluon"))
    table["bm64_any"] = {}
    original = conf.get_moe_dispatch
    monkeypatch.setattr(
        conf,
        "get_moe_dispatch",
        lambda name, *a: table if name == "A8W8-GEMM1" else original(name, *a),
    )
    out = _gemm1("a8w8", cases["a8w8", 128])
    assert [launch[:2] for launch in launches] == [(False, None)]
    ref = _reference("a8w8", cases["a8w8", 128])
    torch.testing.assert_close(out.float(), ref, atol=0.25, rtol=0.05)


def _gemm1_config(op, block_m, half_block=False):
    return conf._default_launch_config(
        block_m, N, K, *_OPS[op][3], False, True, half_block=half_block
    )


@pytest.mark.parametrize("block_m", [32, 64, 128])
@pytest.mark.parametrize("op", sorted(_OPS))
def test_gemm1_family_sets_the_half_m_tile(op, block_m, clean_env):
    # Each GEMM1 family asks for the tile at block_m 32, 64 and 128, and streams the
    # result of a launch whose experts average at most half an M block.
    family = conf._GEMM1_FAMILIES[_OPS[op][3]]
    entry = conf.get_moe_dispatch(family, conf.get_arch(), "gluon")
    assert entry[f"bm{block_m}_any"]["HALF_M_TILE"] is True
    assert entry[f"bm{block_m}_half"] == {"result_cache_modifier": ".cs"}
    # There is no block_m 8 tile.
    assert "HALF_M_TILE" not in entry["bm16_any"]
    assert _gemm1_config(op, block_m)["HALF_M_TILE"] is True
    assert _gemm1_config(op, block_m)["result_cache_modifier"] == ""
    launch = _gemm1_config(op, block_m, half_block=True)
    assert launch["result_cache_modifier"] == ".cs"
    half = dict(
        conf._gemm1_half_m_config(
            conf._cfg_key(launch), block_m, N, K, *_OPS[op][3], False
        )
    )
    assert half["BLOCK_M"] == block_m // 2 and not half["HALF_M_TILE"]
    for key in ("BLOCK_N", "warps_per_cta", "WAVES_PER_EU") + conf._HALF_M_LAUNCH_KEYS:
        # The launch config may lack a newer launch key; the half tile takes its default.
        default = conf.TuningSpec._field_defaults.get(key)
        assert half[key] == launch.get(key, default), key
    # Not for GEMM2.
    gemm2 = conf._default_launch_config(block_m, N, K, *_OPS[op][3], False, False)
    assert not gemm2["HALF_M_TILE"]


def test_a_launch_that_reads_the_weights_directly_takes_no_half_tile_entry(clean_env):
    # A direct_b launch reads no host copies, so the half tile is the plain block_m / 2
    # config: for a8w8 its BLOCK_N and warps differ, so the launch has one tile.
    launch = conf._default_launch_config(
        128, N, K, *_OPS["a8w8"][3], False, True, direct_b=True, half_block=True
    )
    items = conf._cfg_key(launch)
    args = (items, 128, N, K, *_OPS["a8w8"][3], False)
    assert conf._gemm1_half_m_config(*args) is not None
    assert conf._gemm1_half_m_config(*args, direct_b=True) is None


@pytest.mark.parametrize("value", ["0", "1"])
def test_environment_wins_over_the_gemm1_family(value, clean_env):
    clean_env.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", value)
    assert _gemm1_config("a8w4", 128)["HALF_M_TILE"] is (value == "1")
    clean_env.setenv("AITER_TRITON_MOE_GLUON_RESULT_CACHE_MODIFIER", "")
    clear_gluon_config_caches()
    assert _gemm1_config("a8w4", 128, half_block=True)["result_cache_modifier"] == ""


def test_no_gemm1_family_entry_keeps_one_tile(clean_env, monkeypatch):
    monkeypatch.setattr(conf, "get_moe_dispatch", lambda *_: {})
    assert _gemm1_config("a8w4", 128)["HALF_M_TILE"] is False
