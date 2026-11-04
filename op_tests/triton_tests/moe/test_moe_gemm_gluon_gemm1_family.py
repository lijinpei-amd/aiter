# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the gfx950 Gluon MoE GEMM1 config families (``*-GEMM1``): the
block_m 16/32 tiles, the launch keys and the half-height tile entries."""

import re

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import validate
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch

# The configuration tests' environment and cache reset, autouse here too.
from op_tests.triton_tests.moe.test_moe_gemm_gluon_config import (  # noqa: F401
    _clean_tuning_environment,
    _clear_config_caches,
)


@pytest.mark.parametrize("N,K", [(4096, 4096), (6144, 7168)])
@pytest.mark.parametrize("block_m", [16, 32])
def test_small_block_m_mxfp8_mxfp4_gemm1_reads_b_from_registers(block_m, N, K):
    config = conf._default_launch_config(
        block_m, N, K, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, True
    )
    ladder = conf.get_gluon_config(
        block_m, N, K, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False
    )
    # block_m 32 takes the wide tile at two CTAs per CU; block_m 16 a narrow one.
    wide = block_m == 32
    expected = {
        "BLOCK_N": 256 if wide else 128,
        "BLOCK_K": 256,
        "SLOT_M": block_m,
        "SLOT_N": 128 if wide else 64,
        "warps_per_cta": (1, 4),
        "tiles_per_warp": (2, 2) if wide else (1, 1),
        "WAVES_PER_EU": 2 if wide else ladder["WAVES_PER_EU"],
        "B_SCALE_SHUFFLED": True,
        "B_PRESHUFFLED": True,
        "ALLOW_B_VIA_LDS": False,
        "expert_scale_cache_modifier": "",
    }
    assert {name: config[name] for name in expected} == expected
    tuning = conf._probe_tuning_config(config, DtypeQuant.MXFP8, DtypeQuant.MXFP4)
    assert conf._cval(validate(tuning, N, K))
    layout = KernelLayout(tuning)
    assert conf._cval(conf._validate_selected_pipeline(tuning, layout, K))
    # Two N slots, so a gate/up-split caller can run it too.
    assert config["BLOCK_N"] // config["SLOT_N"] == 2
    # B leaves LDS, so LDS does not limit the CTAs per CU.
    assert (
        conf._probe_lds_bytes(config, DtypeQuant.MXFP8, DtypeQuant.MXFP4)
        <= host.LDS_USABLE_BYTES // 4
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"small_grid": True},
        {"apply_swiglu": False},
        # MXFP8 weights: no A8W8-GEMM1 tile at block_m 16.
        {"dq_b": DtypeQuant.MXFP8, "block_m": 16},
        {"block_m": 64},
        {"N": 2880},
        {"N": 4096, "K": 7168},
        {"N": 4096, "K": 7168, "block_m": 16},
        # Too few K steps for the tile's pipeline.
        {"K": 1536},
        {"K": 1536, "block_m": 16},
        # Graph capture, inference or padded weights: no host copies.
        {"direct_b": True},
    ],
)
def test_small_block_m_mxfp8_mxfp4_gemm1_register_b_is_narrow(overrides):
    args = {
        "block_m": 32,
        "N": 4096,
        "K": 4096,
        "dq_a": DtypeQuant.MXFP8,
        "dq_b": DtypeQuant.MXFP4,
        "small_grid": False,
        "apply_swiglu": True,
    }
    args.update(overrides)
    config = conf._default_launch_config(**args)
    # B through LDS: the GEMM1 rule did not apply (GEMM2 has its own families).
    assert config["ALLOW_B_VIA_LDS"] or not args["apply_swiglu"]
    assert (
        config["expert_scale_cache_modifier"] != "" or args["dq_b"] != DtypeQuant.MXFP4
    )


@pytest.mark.parametrize(
    "dq_a,dq_b,block_m",
    [
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 16),
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 32),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8, 32),
    ],
)
def test_small_block_m_gemm1_families_read_b_from_registers(dq_a, dq_b, block_m):
    # The A4W4-GEMM1 and A8W8-GEMM1 tiles at block_m 16/32 (dsv4-flash T=8..1024).
    config = conf._default_launch_config(block_m, 4096, 4096, dq_a, dq_b, False, True)
    assert not config["ALLOW_B_VIA_LDS"] and config["B_PRESHUFFLED"]
    assert config["B_SCALE_SHUFFLED"] and config["BLOCK_K"] == 256
    assert config["BLOCK_N"] == (128 if block_m == 16 else 256)


@pytest.fixture
def gemm1_tables(monkeypatch):
    """Editable copies of the GEMM1 families, served in place of the JSON files."""
    tables = {
        family: {
            k: dict(v) for k, v in get_moe_dispatch(family, "gfx950", "gluon").items()
        }
        for family in conf._GEMM1_FAMILIES.values()
    }
    monkeypatch.setattr(conf, "get_moe_dispatch", lambda name, *_: tables[name])
    return tables


#: A launch the A8W4-GEMM1 family serves: dsv4-flash T=1024 (block_m 32).
_GEMM1_ARGS = (32, 4096, 4096, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, True)


@pytest.mark.parametrize("family", sorted(conf._GEMM1_FAMILIES.values()))
def test_gemm1_families_hold_tuning_values_only(family):
    table = get_moe_dispatch(family, "gfx950", "gluon")
    # Tile entries bm<block_m>_any and _small, optionally for one N and K, and the
    # half-height tile's bm<block_m>_half and bm<block_m>_half_tile.
    assert table
    names = re.compile(
        r"bm(16|32|64|128)_(any|small|half|half_tile|n[0-9]+_k[0-9]+_(any|small))"
    )
    for name in table:
        assert names.fullmatch(name), name
        assert name.split("_", 1)[0] + "_any" in table, name
    for name, entry in table.items():
        # The block_m 128 tile is Python's, and _half sets launch keys.
        if name == "bm128_any" or name.endswith("_half"):
            assert set(entry) <= set(conf._GEMM1_ENV), name
        # Launch keys come from _any and _half only.
        if name.endswith("_small") or re.search(r"_n[0-9]+_k", name):
            assert not set(entry) & set(conf._GEMM1_ENV), name
    known = set(
        conf.get_gluon_config_uncached(
            32, 4096, 4096, DtypeQuant.MXFP8, DtypeQuant.MXFP4
        )
    )
    assert any(table.values())
    for entry in table.values():
        assert set(entry) <= known - {"BLOCK_M"}
        # Register B reads the preshuffled copy.
        assert entry.get("ALLOW_B_VIA_LDS", True) or entry.get("B_PRESHUFFLED")


def test_gemm1_tuning_is_read_from_its_family(gemm1_tables):
    gemm1_tables["A8W4-GEMM1"]["bm32_any"]["GROUP_M"] = 2
    assert conf._default_launch_config(*_GEMM1_ARGS)["GROUP_M"] == 2


def test_gemm1_without_an_entry_keeps_the_ladder(gemm1_tables):
    del gemm1_tables["A8W4-GEMM1"]["bm32_any"]
    assert conf._default_launch_config(*_GEMM1_ARGS) == conf.get_gluon_config(
        *_GEMM1_ARGS[:5]
    )


def test_gemm1_entry_that_cannot_launch_keeps_the_ladder(gemm1_tables):
    # Register B on plain weights is rejected by the launch, so the ladder stays,
    # with the launch keys of the entry on top.
    gemm1_tables["A8W4-GEMM1"]["bm32_any"]["B_PRESHUFFLED"] = False
    assert conf._default_launch_config(*_GEMM1_ARGS) == dict(
        conf.get_gluon_config(*_GEMM1_ARGS[:5]),
        **conf._gemm1_tuned(*_GEMM1_ARGS[:1], *_GEMM1_ARGS[3:5]),
    )


#: A launch with a half-height tile from its family's _half_tile entry: a8w8
#: glm52-base T=1024 (block_m 64, a mean of 32 rows).
#: A small-grid launch of the A8W4-GEMM1 family: dsv4-flash T=1 (block_m 16).
_GEMM1_SMALL_ARGS = (16, 4096, 4096, DtypeQuant.MXFP8, DtypeQuant.MXFP4, True, True)


def test_gemm1_small_grid_entry_overlays_the_ladder(gemm1_tables):
    ladder = conf.get_gluon_config(*_GEMM1_SMALL_ARGS[:6])
    table = gemm1_tables["A8W4-GEMM1"]
    # Without a _small entry a small grid keeps the ladder, whatever _any holds.
    table.pop("bm16_small", None)
    assert conf._default_launch_config(*_GEMM1_SMALL_ARGS) == ladder
    # A _small entry is the whole tile on top of the ladder; _any does not apply.
    _clear_config_caches()
    table["bm16_small"] = {"GROUP_M": 2}
    assert conf._default_launch_config(*_GEMM1_SMALL_ARGS) == dict(ladder, GROUP_M=2)
    # An entry for the launch's N and K replaces the generic one.
    _clear_config_caches()
    table["bm16_n4096_k4096_small"] = {"GROUP_M": 1}
    assert conf._default_launch_config(*_GEMM1_SMALL_ARGS)["GROUP_M"] == 1
    # An empty one keeps the ladder for that shape.
    _clear_config_caches()
    table["bm16_n4096_k4096_small"] = {}
    assert conf._default_launch_config(*_GEMM1_SMALL_ARGS) == ladder


def test_gemm1_shape_entry_replaces_the_any_entry(gemm1_tables):
    tuned = conf._default_launch_config(*_GEMM1_ARGS)
    table = gemm1_tables["A8W4-GEMM1"]
    # An empty shape entry keeps the ladder for that N and K only.
    _clear_config_caches()
    table["bm32_n4096_k4096_any"] = {}
    ladder = dict(
        conf.get_gluon_config(*_GEMM1_ARGS[:5]),
        **conf._gemm1_tuned(*_GEMM1_ARGS[:1], *_GEMM1_ARGS[3:5]),
    )
    assert conf._default_launch_config(*_GEMM1_ARGS) == ladder
    other = (32, 6144, 7168) + _GEMM1_ARGS[3:]
    assert conf._default_launch_config(*other)["BLOCK_N"] == tuned["BLOCK_N"]
    # A shape entry lists the whole tile: it does not take bm32_any's keys.
    _clear_config_caches()
    table["bm32_n4096_k4096_any"] = dict(table["bm32_any"], GROUP_M=2)
    del table["bm32_n4096_k4096_any"]["HALF_M_TILE"]
    assert conf._default_launch_config(*_GEMM1_ARGS) == dict(tuned, GROUP_M=2)


def test_gemm1_small_grid_entry_respects_environment_and_direct_b(
    gemm1_tables, monkeypatch
):
    ladder = conf.get_gluon_config(*_GEMM1_SMALL_ARGS[:6])
    gemm1_tables["A8W4-GEMM1"]["bm16_small"] = {"GROUP_M": 2, "WAVES_PER_EU": 2}
    # A launch that reads the weights directly keeps the ladder.
    direct = conf._default_launch_config(*_GEMM1_SMALL_ARGS, direct_b=True)
    assert direct == ladder
    # An AITER_TRITON_MOE_GLUON_* variable wins over the entry.
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_WAVES_PER_EU", "1")
    _clear_config_caches()
    config = conf._default_launch_config(*_GEMM1_SMALL_ARGS)
    assert (config["GROUP_M"], config["WAVES_PER_EU"]) == (2, 1)


_GEMM1_HALF_ARGS = (64, 4096, 6144, DtypeQuant.MXFP8, DtypeQuant.MXFP8, False, True)


def _gemm1_half_tile_config():
    launch = conf._default_launch_config(*_GEMM1_HALF_ARGS, half_block=True)
    items = conf._gemm1_half_m_config(conf._cfg_key(launch), *_GEMM1_HALF_ARGS[:6])
    return None if items is None else dict(items)


def test_gemm1_half_tile_is_read_from_its_family(gemm1_tables):
    half = _gemm1_half_tile_config()
    assert half["BLOCK_M"] == 32 and half["BLOCK_N"] == 256
    _clear_config_caches()
    gemm1_tables["A8W8-GEMM1"]["bm64_half_tile"]["GROUP_M"] = 2
    assert _gemm1_half_tile_config() == dict(half, GROUP_M=2)
    # Without the entry, the block_m 32 config has another BLOCK_N: no half tile.
    _clear_config_caches()
    del gemm1_tables["A8W8-GEMM1"]["bm64_half_tile"]
    assert _gemm1_half_tile_config() is None


def test_gemm1_environment_override_wins_over_the_half_tile_entry(monkeypatch):
    half = _gemm1_half_tile_config()
    assert half["K_UNROLL"] == 6
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_K_UNROLL", "3")
    _clear_config_caches()
    assert _gemm1_half_tile_config()["K_UNROLL"] == 3


def test_gemm1_environment_override_wins_over_the_family_entry(monkeypatch):
    tuned = conf._default_launch_config(*_GEMM1_ARGS)
    assert tuned["WAVES_PER_EU"] == 2
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_WAVES_PER_EU", "1")
    conf._get_gluon_config_cached.cache_clear()  # only the ladder cache
    assert conf._default_launch_config(*_GEMM1_ARGS) == dict(tuned, WAVES_PER_EU=1)
