# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the host/device configuration contract of the Gluon MoE GEMM."""

import math
import os

import pytest
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._entry import (
    _moe_gluon_gemm1,
    _moe_gluon_gemm2,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import (
    SchedMode,
    constexpr_fields,
    maybe_unwrap,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
from aiter.ops.triton._gluon_kernels.gfx950.moe._pipeline import (
    _PipelinePointers,
    _PipelineRegFragments,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    ActivationSpec,
    ActKind,
    DSReadOperand,
    DtypeQuant,
    EpilogueMode,
    FuncSpec,
    SchedMode,
    TuningSpec,
    WaitCommitScheme,
    WeightGuHbmLayout,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import validate
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch
from op_tests.triton_tests.moe.gluon_buffer_audit import (
    _audit_buffer_schedule,
    _plain,
    _schedule,
)
from op_tests.triton_tests.moe.test_moe_gemm_gluon_buffered_schedule import (
    _assert_minimal_peel,
)


@pytest.fixture(autouse=True)
def _clean_tuning_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    caches = (
        host._launch_spec,
        conf._get_gluon_config_cached,
        conf._family_launch_config,
        conf._gemm2_half_m_config,
        conf._gemm1_family_entry,
        conf._gemm1_half_tile,
        conf._gemm1_half_m_config,
    )
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


@pytest.fixture
def config():
    cfg = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    cfg.update(
        SLOT_M=64,
        SLOT_N=128,
        mfma_instr_shape=(16, 16, 128),
        warps_per_cta=(2, 2),
        tiles_per_warp=(2, 2),
    )
    return cfg


def _func_spec(epilogue=EpilogueMode.DEFAULT):
    return FuncSpec(
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        gl.float32,
        ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        int(DtypeQuant.MXFP4),
        True,
        False,
        True,
        False,
        int(WeightGuHbmLayout.SPLIT),
        int(epilogue),
    )


def _tuning_spec(config):
    return TuningSpec(*(conf._hashable(v) for v in conf._tuning_args(config)))


def _validate(tc, N, K):
    layout = KernelLayout(tc)
    validate(tc, N, K, layout)
    return conf._validate_selected_pipeline(tc, layout, K)


def _minimum(tc, layout):
    schedule = _schedule(tc, layout)
    return (
        maybe_unwrap(schedule.depth)
        + maybe_unwrap(schedule.peeled)
        + maybe_unwrap(schedule.unroll)
    )


@pytest.mark.parametrize("kernel", [_moe_gluon_gemm1, _moe_gluon_gemm2])
def test_num_k_is_a_runtime_scalar_without_specialization(kernel):
    param = next(param for param in kernel.params if param.name == "NUM_K")
    assert not param.is_constexpr
    assert param.do_not_specialize


@pytest.mark.parametrize("kernel", [_moe_gluon_gemm1, _moe_gluon_gemm2])
def test_a_scale_row_stride_preserves_sub_16_byte_alignment(kernel):
    param = next(param for param in kernel.params if param.name == "a_scale_stride_m")
    assert param.is_constexpr


@pytest.mark.parametrize("with_optional_specs", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("B_PRESHUFFLED", True),
        ("DS_READ_A_PAYLOAD_IN_MFMA", True),
        ("DS_READ_A_SCALE_IN_MFMA", True),
        ("DS_READ_B_PAYLOAD_IN_MFMA", True),
        ("DS_READ_B_SCALE_IN_MFMA", True),
        ("SCHED_MODE", int(SchedMode.MFMA_16)),
        ("SOFF_UNROLL", True),
        ("SCALE_FILL_MID", True),
        ("UNROLL_EPILOGUE", False),
        ("ALLOW_B_VIA_LDS", False),
        ("ALLOW_B_SCALE_VIA_LDS", False),
        ("ALLOW_A_SCALE_VIA_LDS", False),
        ("A_NUM_BUFFER", 4),
        ("B_NUM_BUFFER", 2),
        ("A_SCALE_NUM_BUFFER", 1),
        ("B_SCALE_NUM_BUFFER", 6),
        ("SCALE_SLOT_M", 128),
        ("SCALE_SLOT_N", 256),
        ("SCALE_SLOT_K", 16),
    ],
)
def test_launch_specs_match_kernel_config_field_order(
    config, field, value, with_optional_specs
):
    """Host construction and device argument unpacking must preserve every field."""
    config[field] = value
    func = _func_spec(EpilogueMode.NOP)
    if not with_optional_specs:
        # SPLIT folds to INTERLEAVED without an activation, so keep the spec consistent
        # for the round trip; the fold has its own test below.
        func = func._replace(
            activation=None,
            output_quant=None,
            weight_gu_hbm_layout=int(WeightGuHbmLayout.INTERLEAVED),
        )
    tuning = _tuning_spec(config)
    host_func = KernelFuncConfig(*func)
    host_tuning = KernelTuningConfig(host_func, *tuning)
    # Starred arguments pass through Gluon's tuple lowering before construction.
    # Boxing must keep nested ActivationSpec tuples and absent values intact.
    device_func = KernelFuncConfig(*gl.tuple(constexpr_fields(gl.constexpr(func))))
    device_tuning = KernelTuningConfig(
        device_func, *gl.tuple(constexpr_fields(gl.constexpr(tuning)))
    )
    for aggregate in (host_func, device_func):
        for name, expected in func._asdict().items():
            assert _plain(getattr(aggregate, name)) == _plain(expected), name
    for aggregate in (host_tuning, device_tuning):
        for name, expected in tuning._asdict().items():
            assert _plain(getattr(aggregate, name)) == _plain(expected), name


def test_weight_gu_hbm_layout_folds_to_interleaved_without_activation():
    gated = KernelFuncConfig(*_func_spec())
    assert _plain(gated.weight_gu_hbm_layout) == WeightGuHbmLayout.SPLIT
    ungated = KernelFuncConfig(*_func_spec()._replace(activation=None))
    assert _plain(ungated.weight_gu_hbm_layout) == WeightGuHbmLayout.INTERLEAVED


def test_trailing_fields_and_older_config_dict_keep_defaults(config):
    func = FuncSpec(*_func_spec()[:12])
    assert func.epilogue == EpilogueMode.DEFAULT
    legacy_len = TuningSpec._fields.index("DS_READ_A_PAYLOAD_IN_MFMA")
    old_config = {name: config[name] for name in TuningSpec._fields[:legacy_len]}
    tuning = _tuning_spec(old_config)
    assert tuning == TuningSpec(*tuning[:legacy_len])
    assert tuning[legacy_len:] == tuple(
        TuningSpec._field_defaults[name] for name in TuningSpec._fields[legacy_len:]
    )
    fc = KernelFuncConfig(*func[:12])
    tc = KernelTuningConfig(fc, *tuning[:legacy_len])
    assert _plain(fc.epilogue) == EpilogueMode.DEFAULT
    for name in TuningSpec._fields[legacy_len:]:
        assert _plain(getattr(tc, name)) == getattr(tuning, name)
    assert _plain(_validate(tc, 4096, 7168))


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("token_mod", "token_cache_modifier"),
        ("token_scale_mod", "token_scale_cache_modifier"),
        ("expert_mod", "expert_cache_modifier"),
        ("expert_scale_mod", "expert_scale_cache_modifier"),
        ("result_mod", "result_cache_modifier"),
        ("result_scale_mod", "result_scale_cache_modifier"),
    ],
)
def test_legacy_cache_modifier_config_names(config, legacy, canonical):
    config.pop(canonical)
    config[legacy] = ".cg"
    assert getattr(_tuning_spec(config), canonical) == ".cg"

    config[canonical] = ""
    with pytest.raises(ValueError, match=f"{canonical} and legacy {legacy} disagree"):
        _tuning_spec(config)


@pytest.mark.parametrize(
    "field,canonical_env,legacy_env",
    [
        ("token_cache_modifier", "TOKEN_CACHE_MODIFIER", "TOKEN_MOD"),
        (
            "token_scale_cache_modifier",
            "TOKEN_SCALE_CACHE_MODIFIER",
            "TOKEN_SCALE_MOD",
        ),
        ("expert_cache_modifier", "EXPERT_CACHE_MODIFIER", "EXPERT_MOD"),
        (
            "expert_scale_cache_modifier",
            "EXPERT_SCALE_CACHE_MODIFIER",
            "EXPERT_SCALE_MOD",
        ),
        ("result_cache_modifier", "RESULT_CACHE_MODIFIER", "RESULT_MOD"),
        (
            "result_scale_cache_modifier",
            "RESULT_SCALE_CACHE_MODIFIER",
            "RESULT_SCALE_MOD",
        ),
    ],
)
@pytest.mark.parametrize("legacy", [False, True], ids=["canonical", "legacy"])
def test_cache_modifier_environment_names(
    monkeypatch, field, canonical_env, legacy_env, legacy
):
    suffix = legacy_env if legacy else canonical_env
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + suffix, ".ca")
    config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert config[field] == ".ca"


@pytest.mark.parametrize(
    "legacy,renamed,field",
    [
        ("MINI_BLOCK_M", "SLOT_M", "SLOT_M"),
        ("MINI_BLOCK_N", "SLOT_N", "SLOT_N"),
        ("SCALE_MINI_BLOCK_M", "SCALE_SLOT_M", "SCALE_SLOT_M"),
        ("SCALE_MINI_BLOCK_N", "SCALE_SLOT_N", "SCALE_SLOT_N"),
        ("SCALE_MINI_BLOCK_K", "SCALE_SLOT_K", "SCALE_SLOT_K"),
        ("scale_mini_block_m", "scale_slot_m", "SCALE_SLOT_M"),
        ("scale_mini_block_n", "scale_slot_n", "SCALE_SLOT_N"),
        ("scale_mini_block_k", "scale_slot_k", "SCALE_SLOT_K"),
    ],
)
def test_legacy_slot_config_names(config, legacy, renamed, field):
    config.pop(field, None)
    config[legacy] = 96
    # The legacy scale K counted payload K; its new name counts scale columns.
    expected = 96 // conf._LEGACY_SLOT_DIVISORS.get(field, 1)
    # A raw launch config is renamed before the host reads any slot key from it.
    canonical = conf._canonical_slot_keys(config)
    assert canonical[renamed] == expected and legacy not in canonical
    assert getattr(_tuning_spec(config), field) == expected

    config[field] = 128
    with pytest.raises(ValueError, match="disagree"):
        _tuning_spec(config)


@pytest.mark.parametrize("canonical", list(conf._LEGACY_SLOT_KEYS))
def test_legacy_slot_environment_names(monkeypatch, canonical):
    prefix = "AITER_TRITON_MOE_GLUON_"
    legacy = conf._LEGACY_SLOT_KEYS[canonical]
    monkeypatch.delenv(prefix + canonical, raising=False)
    monkeypatch.setenv(prefix + legacy, "64")
    expected = 64 // conf._LEGACY_SLOT_DIVISORS.get(canonical, 1)
    assert conf._env_slot_int(canonical, 0) == expected
    monkeypatch.setenv(prefix + canonical, "128")
    assert conf._env_slot_int(canonical, 0) == 128


def test_legacy_scale_slot_environment_reaches_the_config(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SCALE_MINI_BLOCK_K", "512")
    config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    # The legacy name counted payload K; SCALE_SLOT_K counts scale columns.
    assert config["SCALE_SLOT_K"] == 16
    assert "SCALE_MINI_BLOCK_K" not in config


@pytest.mark.parametrize(
    "M,n_expts_tot,block_m,expected",
    [
        (2048, 33, 64, 1),
        (2112, 33, 64, 1),
        (2113, 33, 64, 2),
        (8192, 33, 128, 2),
    ],
)
def test_expected_m_tiles_per_expert(M, n_expts_tot, block_m, expected):
    assert conf._expected_m_tiles_per_expert(M, n_expts_tot, block_m) == expected


@pytest.mark.parametrize(
    "dq_a,dq_b,apply_swiglu,M,expected",
    [
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8, True, 2048, True),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8, True, 2113, False),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8, False, 2048, False),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP4, True, 2048, False),
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, True, 2048, False),
    ],
)
def test_streaming_expert_payload_is_scoped_to_one_tile_mxfp8_gemm1(
    dq_a, dq_b, apply_swiglu, M, expected
):
    assert (
        conf._stream_mxfp8_gemm1_expert_payload(
            M, 33, 64, dq_a, dq_b, apply_swiglu
        )
        is expected
    )


def test_t256_mxfp8_gemm1_uses_validated_streaming_geometry():
    config = host._launch_spec(
        block_m=64,
        N=4096,
        K=7168,
        dq_a=DtypeQuant.MXFP8,
        dq_b=DtypeQuant.MXFP8,
        small_grid=False,
        has_bias=False,
        has_gammas=False,
        has_gather=True,
        has_x_static_scale=False,
        act=ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        out_quant=None,
        config_items=None,
        stream_expert_payload=True,
    )[4]
    expected = {
        "BLOCK_M": 64,
        "BLOCK_N": 128,
        "BLOCK_K": 256,
        "SLOT_M": 32,
        "SLOT_N": 64,
        "mfma_instr_shape": (16, 16, 128),
        "warps_per_cta": (2, 2),
        "tiles_per_warp": (1, 2),
        "NUM_LDS_BUFFER": 3,
        "K_UNROLL": 3,
        "VGPR_PREFETCH_K": 256,
        "WAVES_PER_EU": 1,
        "WAIT_COMMIT_SCHEME": int(WaitCommitScheme.PER_STAGE_WHOLE),
        "A_SCALE_SORTED_SHUFFLED": True,
        "B_SCALE_SHUFFLED": True,
        "B_PRESHUFFLED": False,
        "ALLOW_B_VIA_LDS": True,
        "ALLOW_B_SCALE_VIA_LDS": True,
        "ALLOW_A_SCALE_VIA_LDS": True,
        "expert_cache_modifier": ".cg",
    }
    assert {name: config[name] for name in expected} == expected
    tuning = conf._probe_tuning_config(
        config, DtypeQuant.MXFP8, DtypeQuant.MXFP8
    )
    assert conf._cval(validate(tuning, 4096, 7168))
    layout = KernelLayout(tuning)
    assert conf._cval(conf._validate_selected_pipeline(tuning, layout, 7168))
    assert conf._probe_lds_bytes(
        config, DtypeQuant.MXFP8, DtypeQuant.MXFP8
    ) <= host.LDS_USABLE_BYTES

    reused = conf._default_launch_config(
        64,
        4096,
        7168,
        DtypeQuant.MXFP8,
        DtypeQuant.MXFP8,
        False,
        True,
        stream_expert_payload=False,
    )
    # stream_expert_payload selects the geometry, and only the geometry. The weight
    # hint used to be part of what it toggled; it is now unconditional, because it
    # measured as a win at every block size rather than only the streaming one.
    assert reused["expert_cache_modifier"] == ".cg"
    assert (reused["SLOT_N"], reused["warps_per_cta"]) != (
        expected["SLOT_N"],
        expected["warps_per_cta"],
    )


@pytest.mark.parametrize(
    "dq_b,ctas_per_cu", [(DtypeQuant.MXFP8, 2), (DtypeQuant.MXFP4, 3)]
)
def test_small_t_mxfp8_gemm1_fits_several_ctas_per_cu(dq_b, ctas_per_cu):
    config = conf._default_launch_config(
        32, 4096, 7168, DtypeQuant.MXFP8, dq_b, False, True
    )
    expected = {
        "BLOCK_M": 32,
        "BLOCK_N": 64,
        "BLOCK_K": 256,
        "SLOT_M": 32,
        "SLOT_N": 32,
        "warps_per_cta": (2, 2),
        "tiles_per_warp": (1, 1),
        "B_SCALE_SHUFFLED": True,
        "expert_cache_modifier": ".cg",
        "expert_scale_cache_modifier": ".cg",
    }
    assert {name: config[name] for name in expected} == expected
    tuning = conf._probe_tuning_config(config, DtypeQuant.MXFP8, dq_b)
    assert conf._cval(validate(tuning, 4096, 7168))
    # The point of the tile: several CTAs share a CU's LDS.
    assert (
        conf._probe_lds_bytes(config, DtypeQuant.MXFP8, dq_b)
        <= host.LDS_USABLE_BYTES // ctas_per_cu
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
        {"dq_b": DtypeQuant.MXFP8},
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
    # bm<block_m>_any entries, plus bm<block_m>_half and bm<block_m>_half_tile for the
    # half-height tile: GEMM1 has no other grid classes.
    assert table
    assert set(table) <= {
        f"bm{bm}_{cls}"
        for bm in conf._SUPPORTED_BLOCK_M
        for cls in ("any", "half", "half_tile")
    }
    for name in table:
        assert name.rsplit("_", 1)[0].removesuffix("_half") + "_any" in table, name
    # The launch keys: the block_m 128 tile is Python's, and _half sets launch keys.
    for name, entry in table.items():
        if name == "bm128_any" or name.endswith("_half"):
            assert set(entry) <= set(conf._GEMM1_ENV), name
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
    # Register B on plain weights is rejected by the launch, so the ladder stays.
    gemm1_tables["A8W4-GEMM1"]["bm32_any"]["B_PRESHUFFLED"] = False
    assert conf._default_launch_config(*_GEMM1_ARGS) == conf.get_gluon_config(
        *_GEMM1_ARGS[:5]
    )


#: A launch with a half-height tile from its family's _half_tile entry: a8w8
#: dsv4-pro T=4096 (block_m 128, a mean of 64 rows).
_GEMM1_HALF_ARGS = (128, 6144, 7168, DtypeQuant.MXFP8, DtypeQuant.MXFP8, False, True)


def _gemm1_half_tile_config():
    launch = conf._default_launch_config(*_GEMM1_HALF_ARGS, half_block=True)
    items = conf._gemm1_half_m_config(conf._cfg_key(launch), *_GEMM1_HALF_ARGS[:6])
    return None if items is None else dict(items)


def test_gemm1_half_tile_is_read_from_its_family(gemm1_tables):
    half = _gemm1_half_tile_config()
    assert half["BLOCK_M"] == 64 and half["BLOCK_N"] == 256
    for cached in (conf._gemm1_half_tile, conf._gemm1_half_m_config):
        cached.cache_clear()
    gemm1_tables["A8W8-GEMM1"]["bm128_half_tile"]["GROUP_M"] = 2
    assert _gemm1_half_tile_config() == dict(half, GROUP_M=2)
    # Without the entry, the block_m 64 config has another BLOCK_N: no half tile.
    for cached in (conf._gemm1_half_tile, conf._gemm1_half_m_config):
        cached.cache_clear()
    del gemm1_tables["A8W8-GEMM1"]["bm128_half_tile"]
    assert _gemm1_half_tile_config() is None


def test_gemm1_environment_override_wins_over_the_half_tile_entry(monkeypatch):
    half = _gemm1_half_tile_config()
    assert half["K_UNROLL"] == 6
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_K_UNROLL", "3")
    for cached in (
        conf._get_gluon_config_cached,
        conf._family_launch_config,
        conf._gemm1_half_tile,
        conf._gemm1_half_m_config,
    ):
        cached.cache_clear()
    assert _gemm1_half_tile_config()["K_UNROLL"] == 3


def test_gemm1_environment_override_wins_over_the_family_entry(monkeypatch):
    tuned = conf._default_launch_config(*_GEMM1_ARGS)
    assert tuned["WAVES_PER_EU"] == 2
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_WAVES_PER_EU", "1")
    conf._get_gluon_config_cached.cache_clear()  # only the ladder cache
    assert conf._default_launch_config(*_GEMM1_ARGS) == dict(tuned, WAVES_PER_EU=1)


@pytest.mark.parametrize(
    "env,sched_mode", [(None, SchedMode.MFMA_8), ("0", SchedMode.NONE)]
)
def test_large_t_mxfp8_mxfp4_gemm1_interleaves_mfma_with_memory(
    monkeypatch, env, sched_mode
):
    # The environment still wins over the measured default.
    if env is not None:
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SCHED_MODE", env)
    config = conf._default_launch_config(
        128, 4096, 7168, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, True
    )
    expected = {
        "BLOCK_N": 256,
        "warps_per_cta": (1, 4),
        "ALLOW_B_VIA_LDS": False,
        "SCHED_MODE": int(sched_mode),
        "DS_READ_A_PAYLOAD_IN_MFMA": True,
        "GROUP_M": 4,
    }
    assert {name: config[name] for name in expected} == expected
    tuning = conf._probe_tuning_config(config, DtypeQuant.MXFP8, DtypeQuant.MXFP4)
    assert conf._cval(validate(tuning, 4096, 7168))
    layout = KernelLayout(tuning)
    assert conf._cval(conf._validate_selected_pipeline(tuning, layout, 7168))


@pytest.mark.parametrize(
    "overrides",
    [
        {"stream_expert_payload": False},
        {"apply_swiglu": False},
        {"dq_a": DtypeQuant.MXFP4},
        {"dq_b": DtypeQuant.MXFP4},
        {"block_m": 32},
        {"N": 2048},
        {"K": 6912},
        {"small_grid": True},
    ],
)
def test_t256_mxfp8_gemm1_geometry_override_is_narrow(overrides, monkeypatch):
    # Without the GEMM1 families, which tune the neighbouring launches on purpose.
    original = conf.get_moe_dispatch
    monkeypatch.setattr(
        conf,
        "get_moe_dispatch",
        lambda name, *a: (
            {} if name in conf._GEMM1_FAMILIES.values() else original(name, *a)
        ),
    )
    args = {
        "block_m": 64,
        "N": 4096,
        "K": 7168,
        "dq_a": DtypeQuant.MXFP8,
        "dq_b": DtypeQuant.MXFP8,
        "small_grid": False,
        "apply_swiglu": True,
        "stream_expert_payload": True,
    }
    args.update(overrides)
    selected = conf._default_launch_config(**args)
    baseline = conf.get_gluon_config(
        args["block_m"],
        args["N"],
        args["K"],
        args["dq_a"],
        args["dq_b"],
        args["small_grid"],
        args["stream_expert_payload"],
    )
    if overrides == {"block_m": 32}:
        # block_m 32 is the small-token regime and now carries its own measured
        # override, so it does not fall back to the bare ladder. What matters here is
        # that it is not wearing the T=256 geometry.
        assert selected != baseline
        assert (selected["BLOCK_N"], selected["SLOT_N"]) != (128, 64)
        assert selected["BLOCK_M"] == 32
    elif overrides == {"dq_b": DtypeQuant.MXFP4}:
        # MXFP4 weights at block_m 64 also carry their own measured override now. The
        # point of this case is still that it does not inherit the MXFP8 one: that
        # tile is BLOCK_N 128 with SLOT_N 64, this one is 256 with 128.
        assert selected != baseline
        assert (selected["BLOCK_N"], selected["SLOT_N"]) == (256, 128)
    elif overrides == {"apply_swiglu": False}:
        # GEMM2 has its own measured tile (test_gemm2_large_block_m_tile), not this
        # GEMM1 one, which is BLOCK_N 128 with SLOT_N 64.
        assert (selected["BLOCK_N"], selected["SLOT_N"]) == (256, 128)
    else:
        assert selected == baseline


_GEMM2_TILE = {
    "BLOCK_N": 256,
    "SLOT_M": 64,
    "SLOT_N": 128,
    "mfma_instr_shape": (16, 16, 128),
    "warps_per_cta": (1, 4),
    "tiles_per_warp": (2, 2),
    "NUM_LDS_BUFFER": 3,
    "GROUP_M": 8,
    "B_SCALE_SHUFFLED": True,
    "B_PRESHUFFLED": False,
    "ALLOW_B_VIA_LDS": True,
}
#: Register B at block_m 64 and 128: no B ring in LDS pays for the taller warp tile.
_GEMM2_REGISTER_B_TILE = dict(
    _GEMM2_TILE,
    tiles_per_warp=(4, 2),
    WAVES_PER_EU=1,
    WAIT_COMMIT_SCHEME=3,
    B_PRESHUFFLED=True,
    ALLOW_B_VIA_LDS=False,
)


#: Per-dtype additions of the block_m 128 entries: soffset addressing and B scales in
#: registers (a4w4), B scales in registers (a8w4), A-scale reads in the MFMA region
#: (a8w8).
_GEMM2_BM128_EXTRA = {
    (DtypeQuant.MXFP4, DtypeQuant.MXFP4): {
        "SOFF_UNROLL": True,
        "ALLOW_B_SCALE_VIA_LDS": False,
    },
    (DtypeQuant.MXFP8, DtypeQuant.MXFP4): {
        "SOFF_UNROLL": False,
        "ALLOW_B_SCALE_VIA_LDS": False,
    },
    (DtypeQuant.MXFP8, DtypeQuant.MXFP8): {
        "SOFF_UNROLL": False,
        "ALLOW_B_SCALE_VIA_LDS": True,
        "DS_READ_A_SCALE_IN_MFMA": True,
    },
}


@pytest.mark.parametrize(
    "dq_a,dq_b,block_m,block_k",
    [
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 64, 256),
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 128, 256),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP4, 64, 256),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP4, 128, 128),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8, 64, 256),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8, 128, 128),
    ],
)
def test_gemm2_large_block_m_tile(dq_a, dq_b, block_m, block_k):
    N, K = 6144, 2048
    config = conf._default_launch_config(block_m, N, K, dq_a, dq_b, False, False)
    # Both take the register-B tile; block_m 64 holds one M slot and BLOCK_K 256.
    expected = dict(
        _GEMM2_REGISTER_B_TILE,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        K_UNROLL=3 * 256 // block_k,
        VGPR_PREFETCH_K=block_k,
        **(_GEMM2_BM128_EXTRA[(dq_a, dq_b)] if block_m == 128 else {}),
    )
    assert {name: config[name] for name in expected} == expected
    assert _plain(_validate(conf._probe_tuning_config(config, dq_a, dq_b), N, K))


@pytest.mark.parametrize(
    "dq_a,dq_b",
    [
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP4),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8),
    ],
)
@pytest.mark.parametrize("block_m,small_grid", [(16, True), (16, False), (32, False)])
def test_gemm2_small_block_m_reads_preshuffled_b(dq_a, dq_b, block_m, small_grid):
    config = conf._default_launch_config(
        block_m, 7168, 3072, dq_a, dq_b, small_grid, False
    )
    ladder = conf.get_gluon_config(block_m, 7168, 3072, dq_a, dq_b, small_grid)
    tuned = conf._family_tuned(
        conf._GEMM2_FAMILIES[(dq_a, dq_b)], block_m, "small" if small_grid else "any"
    )
    changed = {name for name in config if config[name] != ladder[name]}
    # Only the family entry changes the ladder: GEMM2 does not take GEMM1's `.cg` hint
    # for shuffled MXFP8 scales.
    assert changed == {key for key, value in tuned.items() if ladder[key] != value}
    assert all(config[key] == value for key, value in tuned.items())
    assert config["B_PRESHUFFLED"] and not config["ALLOW_B_VIA_LDS"]


@pytest.mark.parametrize(
    "dq_a,dq_b,block_m,N,K",
    [
        # N or K off the 256 grid that the measured shapes share.
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 128, 2880, 2048),
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 128, 4096, 2880),
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 32, 2880, 2048),
        # Too few K steps for the tile's pipeline.
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 128, 4096, 1024),
        # The ladder's BLOCK_K 128, where shuffled scales double the unroll.
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4, 16, 4096, 1024),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP4, 32, 7168, 1280),
        # No MX activations or weights.
        (DtypeQuant.FP8_E4M3, DtypeQuant.MXFP4, 128, 4096, 2048),
        (DtypeQuant.BF16, DtypeQuant.BF16, 128, 4096, 2048),
    ],
)
def test_gemm2_tile_keeps_the_ladder_elsewhere(dq_a, dq_b, block_m, N, K):
    config = conf._default_launch_config(block_m, N, K, dq_a, dq_b, False, False)
    assert config == conf.get_gluon_config(block_m, N, K, dq_a, dq_b)


def _launchable(config, dq_a, dq_b, N, K):
    """The config checks gluon_supported makes; a failure falls back to Triton."""
    return (
        N % config["BLOCK_N"] == 0
        and K % config["BLOCK_K"] == 0
        and conf._pipeline_error(conf._cfg_key(config), dq_a, dq_b, K) is None
        and conf._probe_lds_bytes(config, dq_a, dq_b) <= host.LDS_USABLE_BYTES
    )


@pytest.mark.parametrize(
    "dq_a,dq_b",
    [
        (DtypeQuant.MXFP4, DtypeQuant.MXFP4),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP4),
        (DtypeQuant.MXFP8, DtypeQuant.MXFP8),
    ],
)
@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
def test_gemm2_config_never_drops_a_launchable_ladder(dq_a, dq_b, block_m):
    for N in (512, 2880, 7168):
        for K in (512, 1024, 1280, 1536, 2048, 2816, 3072):
            for small_grid in (False, True):
                ladder = conf.get_gluon_config(block_m, N, K, dq_a, dq_b, small_grid)
                if not _launchable(ladder, dq_a, dq_b, N, K):
                    continue
                config = conf._default_launch_config(
                    block_m, N, K, dq_a, dq_b, small_grid, False
                )
                assert _launchable(config, dq_a, dq_b, N, K), (N, K, small_grid)


@pytest.fixture
def gemm2_tables(monkeypatch):
    """Editable copies of the GEMM2 families, served in place of the JSON files."""
    tables = {
        family: {
            k: dict(v) for k, v in get_moe_dispatch(family, "gfx950", "gluon").items()
        }
        for family in conf._GEMM2_FAMILIES.values()
    }
    monkeypatch.setattr(conf, "get_moe_dispatch", lambda name, *_: tables[name])
    return tables


@pytest.mark.parametrize("family", sorted(conf._GEMM2_FAMILIES.values()))
def test_gemm2_families_hold_tuning_values_only(family):
    table = get_moe_dispatch(family, "gfx950", "gluon")
    any_keys = {f"bm{bm}_any" for bm in conf._SUPPORTED_BLOCK_M}
    grid_keys = {
        f"bm{bm}_{grid}"
        for bm in conf._SUPPORTED_BLOCK_M
        for grid in ("small", "multi", "half", "half_tile")
    }
    assert any_keys <= set(table) <= any_keys | grid_keys
    known = set(
        conf.get_gluon_config_uncached(
            128, 4096, 2048, DtypeQuant.MXFP4, DtypeQuant.MXFP4
        )
    )
    for entry in table.values():
        assert entry and set(entry) <= known - {"BLOCK_M"}
    # A grid entry lists only what it changes on top of the any entry. A half_tile
    # entry goes on top of the block_m / 2 config, and only a _half launch uses it.
    for key, entry in table.items():
        if key.endswith("_half_tile"):
            block_m = int(key[2:].split("_")[0])
            assert f"bm{block_m}_half" in table, key
            base = table[f"bm{block_m // 2}_any"]
        elif not key.endswith("_any"):
            base = table[key.rsplit("_", 1)[0] + "_any"]
        else:
            continue
        assert not {k for k, v in entry.items() if base.get(k) == v}, key


@pytest.mark.parametrize("dq_a,dq_b", list(conf._GEMM2_FAMILIES))
def test_gemm2_multi_block_streams_the_result_and_caches_the_weights(dq_a, dq_b):
    # Experts with several M blocks (T=16384 in the matrix): the large intermediate is
    # stored non-temporally, so it does not evict the weights the other blocks reuse.
    # MXFP4 weight scales are cached too; MXFP8 ones keep their `.cg` hint.
    args = (128, 6144, 2048, dq_a, dq_b, False, False)
    multi = conf._default_launch_config(*args, multi_block=True)
    plain = conf._default_launch_config(*args)
    modifiers = (
        "result_cache_modifier",
        "expert_cache_modifier",
        "expert_scale_cache_modifier",
    )
    scale = "" if dq_b == DtypeQuant.MXFP4 else ".cg"
    assert [multi[k] for k in modifiers] == [".cs", "", scale]
    # Other launches keep the bm128_any entry's result policy (a8w8 streams it too).
    any_entry = conf._family_tuned(conf._GEMM2_FAMILIES[(dq_a, dq_b)], 128, "any")
    assert plain["result_cache_modifier"] == any_entry.get("result_cache_modifier", "")
    assert plain["expert_cache_modifier"]
    changed = {k for k in multi if multi[k] != plain[k]}
    assert changed <= set(modifiers)


def test_gemm2_grid_entry_overlays_the_any_entry(gemm2_tables):
    # A grid entry lists only the keys it changes on top of bm<block_m>_any.
    gemm2_tables["A4W4-GEMM2"]["bm64_multi"] = {"GROUP_M": 2}
    any_entry = conf._family_tuned("A4W4-GEMM2", 64, "any")
    assert any_entry["GROUP_M"] != 2
    assert conf._family_tuned("A4W4-GEMM2", 64, "multi") == dict(any_entry, GROUP_M=2)


@pytest.mark.parametrize(
    "small_grid,multi_block,half_block,grid",
    [
        (False, False, True, "half"),
        (False, False, False, "any"),
        (False, True, False, "multi"),
        (True, False, True, "small"),
    ],
)
def test_gemm2_grid_class_order(
    gemm2_tables, small_grid, multi_block, half_block, grid
):
    # A small grid wins, then several M blocks per expert, then half an M block.
    for name in ("small", "multi", "half"):
        gemm2_tables["A4W4-GEMM2"][f"bm64_{name}"] = {
            "GROUP_M": {"small": 1, "multi": 2, "half": 16}[name]
        }
    config = conf._default_launch_config(
        64,
        6144,
        2048,
        DtypeQuant.MXFP4,
        DtypeQuant.MXFP4,
        small_grid,
        False,
        multi_block=multi_block,
        half_block=half_block,
    )
    expected = {"small": 1, "multi": 2, "half": 16, "any": 8}[grid]
    assert config["GROUP_M"] == expected


def test_gemm2_tuning_is_read_from_its_family(gemm2_tables):
    gemm2_tables["A8W8-GEMM2"]["bm128_any"]["GROUP_M"] = 2
    config = conf._default_launch_config(
        128, 6144, 2048, DtypeQuant.MXFP8, DtypeQuant.MXFP8, False, False
    )
    assert config["GROUP_M"] == 2


def test_gemm2_register_b_entry_needs_preshuffled_weights(gemm2_tables):
    # The launch rejects register B on plain weights, so such an entry keeps the ladder.
    gemm2_tables["A4W4-GEMM2"]["bm32_any"]["B_PRESHUFFLED"] = False
    args = (32, 7168, 3072, DtypeQuant.MXFP4, DtypeQuant.MXFP4)
    assert conf._default_launch_config(*args, False, False) == conf.get_gluon_config(
        *args
    )


def _assert_one_tile(spec):
    """The launch of ``spec`` (``_launch_spec``'s result) has no half-height tile."""
    assert not spec[4].get("HALF_M_TILE") and spec[6] is None and len(spec[1]) == 4


def _gemm2_spec(block_m=128, act=None, half_block=True):
    """``_launch_spec`` of an a8w4 launch with gather and gammas."""
    return host._launch_spec(
        block_m,
        7168,
        3072,
        DtypeQuant.MXFP8,
        DtypeQuant.MXFP4,
        False,
        False,
        True,
        True,
        False,
        act,
        None,
        None,
        half_block=half_block,
    )


def test_gemm2_half_m_tile_is_the_half_block_m_config():
    # a8w4 block_m 128 runs an M block of at most 64 rows on its block_m 64 config,
    # with the bm128_half_tile entry on top.
    spec = _gemm2_spec()
    *_, config, tuning, half = spec
    assert config["HALF_M_TILE"] and _plain(tuning.HALF_M_TILE)
    tile = get_moe_dispatch("A8W4-GEMM2", "gfx950", "gluon")["bm128_half_tile"]
    expected = dict(
        conf._default_launch_config(
            64, 7168, 3072, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, False
        ),
        HALF_M_TILE=False,
        **{k: conf._hashable(v) for k, v in tile.items()},
    )
    assert len(spec[1]) == 5 and _plain(spec[1][4]) == _tuning_spec(expected)
    assert _plain(half.BLOCK_M) == 64 and not _plain(half.HALF_M_TILE)
    for key in ("WAVES_PER_EU", "B_PRESHUFFLED", "B_SCALE_SHUFFLED"):
        assert _plain(getattr(half, key)) == _plain(getattr(tuning, key)), key


def test_gemm2_wider_half_m_tile_sets_the_n_grid():
    # The half-height tile is twice as wide as the block_m tile, so the N grid is the
    # half-height tile's, and the block_m tile covers its N block in two passes.
    grid_n, *_, tuning, half = _gemm2_spec()
    assert _plain(half.BLOCK_N) == 2 * _plain(tuning.BLOCK_N)
    assert grid_n == 7168 // _plain(half.BLOCK_N)


@pytest.mark.parametrize(
    "act,half_block",
    [
        (None, False),  # the experts average more than half an M block
    ],
)
def test_gemm2_half_m_tile_only_where_the_launch_asks(act, half_block):
    _assert_one_tile(_gemm2_spec(128, act, half_block))


def test_gemm2_half_m_tile_needs_the_entry(gemm2_tables):
    # A _half launch whose entry does not set HALF_M_TILE keeps one tile.
    del gemm2_tables["A8W4-GEMM2"]["bm64_half"]["HALF_M_TILE"]
    _assert_one_tile(_gemm2_spec(64))


def test_gemm1_launch_takes_the_gemm1_half_m_tile():
    # A GEMM1 launch gets the GEMM1 half-height tile (the block_m / 2 config, same
    # BLOCK_N), not the GEMM2 family's bm128_half_tile entry.
    act = ActivationSpec(int(ActKind.SILU), 1.0, None, False)
    *_, config, tuning, half = _gemm2_spec(128, act, True)
    assert config["HALF_M_TILE"] and half is not None
    assert _plain(half.BLOCK_M) == 64
    assert _plain(half.BLOCK_N) == _plain(tuning.BLOCK_N)


@pytest.mark.parametrize(
    "key,value",
    [("BLOCK_N", 384), ("WAVES_PER_EU", 2), ("B_PRESHUFFLED", False)],
)
def test_gemm2_half_m_tile_needs_the_launch_grid_and_operands(gemm2_tables, key, value):
    # The half-height tile shares the launch's waves and weight copies, and its N block
    # is a whole number of the launch's.
    entry = gemm2_tables["A8W4-GEMM2"]["bm128_half_tile"]
    entry[key] = value
    if key == "B_PRESHUFFLED":
        entry["ALLOW_B_VIA_LDS"] = True
    _assert_one_tile(_gemm2_spec())


@pytest.mark.parametrize("env,expected", [(None, 6), ("2", 2)])
def test_gemm2_half_m_tile_keeps_environment_keys(
    gemm2_tables, monkeypatch, env, expected
):
    # The half_tile entry goes on top, but a key the environment sets keeps its value.
    gemm2_tables["A8W4-GEMM2"]["bm128_half_tile"]["K_UNROLL"] = 6
    if env is not None:
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_K_UNROLL", env)
    assert _plain(_gemm2_spec()[6].K_UNROLL) == expected


def test_gemm2_half_m_tile_environment_override(monkeypatch):
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_HALF_M_TILE", "0")
    assert _gemm2_spec()[6] is None


def test_gemm2_group_m_environment_override_is_not_cached_away(monkeypatch):
    args = (128, 6144, 2048, DtypeQuant.MXFP4, DtypeQuant.MXFP4, False, False)
    assert conf._default_launch_config(*args)["GROUP_M"] == 8
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_GROUP_M", "4")
    conf._get_gluon_config_cached.cache_clear()  # only the ladder cache
    assert conf._default_launch_config(*args)["GROUP_M"] == 4


@pytest.mark.parametrize(
    "name, value, key, expected",
    [
        ("WAVES_PER_EU", "2", "WAVES_PER_EU", 2),
        # Set to the ladder's own default: still an explicit request.
        ("WAVES_PER_EU", "0", "WAVES_PER_EU", 0),
        ("WAIT_COMMIT_SCHEME", "1", "WAIT_COMMIT_SCHEME", 1),
        ("ALLOW_B_VIA_LDS", "1", "ALLOW_B_VIA_LDS", True),
        ("RESULT_CACHE_MODIFIER", ".cg", "result_cache_modifier", ".cg"),
        ("RESULT_MOD", ".cg", "result_cache_modifier", ".cg"),  # legacy name
        ("EXPERT_CACHE_MODIFIER", ".cg", "expert_cache_modifier", ".cg"),
    ],
)
def test_gemm2_environment_overrides_win_over_the_family_entry(
    monkeypatch, name, value, key, expected
):
    dq = DtypeQuant.MXFP4
    args = (128, 6144, 2048, dq, dq, False, False)
    tuned = conf._default_launch_config(*args, multi_block=True)
    assert tuned[key] != expected
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + name, value)
    conf._get_gluon_config_cached.cache_clear()  # only the ladder cache
    config = conf._default_launch_config(*args, multi_block=True)
    assert config[key] == expected
    # The rest of the entry stays where the result still launches; else the ladder.
    wanted = dict(tuned, **{key: expected})
    if key == "ALLOW_B_VIA_LDS":
        wanted["B_PRESHUFFLED"] = False  # B from LDS needs no packed copy
    if conf._launch_error(conf._cfg_key(wanted), dq, dq, 6144, 2048) is None:
        assert config == wanted
    else:
        assert config == conf.get_gluon_config(128, 6144, 2048, dq, dq)
    if key != "ALLOW_B_VIA_LDS":
        assert config == wanted  # these keys cannot make the entry fail


def test_gemm2_environment_keys_cover_the_family_entries():
    # Every key a GEMM2 entry sets either has its ladder variables listed, or has none.
    no_variable = {
        "BLOCK_N",
        "BLOCK_K",
        "mfma_instr_shape",
        "warps_per_cta",
        "tiles_per_warp",
        "B_PRESHUFFLED",
        "B_SCALE_SHUFFLED",
    }
    families = {**conf._GEMM1_FAMILIES, **conf._GEMM2_FAMILIES}.values()
    for family in families:
        for entry in get_moe_dispatch(family, "gfx950", "gluon").values():
            assert set(entry) <= set(conf._FAMILY_ENV) | no_variable, family


@pytest.mark.parametrize("suffix", ["EXPERT_CACHE_MODIFIER", "EXPERT_MOD"])
@pytest.mark.parametrize("override", ["", ".ca"])
def test_streaming_expert_payload_respects_environment_override(
    monkeypatch, suffix, override
):
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + suffix, override)
    config = conf.get_gluon_config_uncached(
        64,
        4096,
        7168,
        DtypeQuant.MXFP8,
        DtypeQuant.MXFP8,
        False,
        stream_expert_payload=True,
    )
    assert config["expert_cache_modifier"] == override


@pytest.mark.parametrize("explicit", ["", ".ca"])
def test_streaming_expert_payload_respects_explicit_config(explicit):
    config = conf.get_gluon_config_uncached(
        64, 4096, 7168, DtypeQuant.MXFP8, DtypeQuant.MXFP8
    )
    config["expert_cache_modifier"] = explicit
    selected = host._launch_spec(
        block_m=64,
        N=4096,
        K=7168,
        dq_a=DtypeQuant.MXFP8,
        dq_b=DtypeQuant.MXFP8,
        small_grid=False,
        has_bias=False,
        has_gammas=False,
        has_gather=True,
        has_x_static_scale=False,
        act=ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        out_quant=None,
        config_items=tuple(
            sorted((key, conf._hashable(value)) for key, value in config.items())
        ),
        stream_expert_payload=True,
    )[4]
    assert selected == config


@pytest.mark.parametrize(
    "dq_b,block_m,K,stream,block_k,shuffled,expected",
    [
        # Shuffled MXFP8 B scales, the T>=256 tiles: T=256 streams B, T>=1024 does
        # not.
        (DtypeQuant.MXFP8, 64, 7168, True, 256, True, ".cg"),
        (DtypeQuant.MXFP8, 128, 7168, False, 128, True, ".cg"),
        # An unshuffled MXFP8 B scale row of 4 bytes copied through LDS.
        (DtypeQuant.MXFP8, 16, 1024, False, 128, False, ".cg"),
        (DtypeQuant.MXFP8, 128, 2880, False, 128, False, ".cg"),
        # The T<=64 tile shuffles them too.
        (DtypeQuant.MXFP8, 32, 7168, False, 256, True, ".cg"),
        # Unshuffled at BLOCK_K 256, where the hint measured slower.
        (DtypeQuant.MXFP8, 64, 7168, False, 256, False, ""),
        # MXFP4 weights take the hint at every BLOCK_K.
        (DtypeQuant.MXFP4, 16, 1024, False, 128, False, ".cg"),
    ],
)
def test_mxfp8_weight_scale_hint_cases(
    dq_b, block_m, K, stream, block_k, shuffled, expected
):
    config = conf._default_launch_config(
        block_m, 4096, K, DtypeQuant.MXFP8, dq_b, False, True, stream
    )
    assert (config["BLOCK_K"], config["B_SCALE_SHUFFLED"]) == (block_k, shuffled)
    assert config["expert_scale_cache_modifier"] == expected


@pytest.mark.parametrize(
    "env,expected",
    [
        # A register B scale does not take the hint.
        ({"ALLOW_B_SCALE_VIA_LDS": "0"}, ""),
        ({"EXPERT_SCALE_CACHE_MODIFIER": ""}, ""),
        ({"EXPERT_SCALE_MOD": ".ca"}, ".ca"),
    ],
)
def test_mxfp8_weight_scale_hint_respects_environment(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + name, value)
    config = conf._default_launch_config(
        16, 4096, 1024, DtypeQuant.MXFP8, DtypeQuant.MXFP8, False, True
    )
    assert config["BLOCK_K"] == 128
    assert config["expert_scale_cache_modifier"] == expected


@pytest.mark.parametrize(
    "scheme,groups,stage_head",
    [
        (WaitCommitScheme.PER_OP, 8, False),
        (WaitCommitScheme.PER_PANE, 4, False),
        (WaitCommitScheme.PER_STAGE_WARP_PIPELINE, 1, True),
        (WaitCommitScheme.PER_STAGE_WHOLE, 1, True),
    ],
)
def test_commit_scheme_counts_real_payload_and_scale_groups(
    config, scheme, groups, stage_head
):
    config["WAIT_COMMIT_SCHEME"] = int(scheme)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    layout = KernelLayout(tc)
    assert _plain(layout.component_via_lds(A_SCALE))
    assert _plain(layout.component_via_lds(B_SCALE))
    schedule = _schedule(tc, layout, 64)
    loop = schedule.loop_step(0)
    panes = _plain(maybe_unwrap(layout.num_m_slots)) * _plain(
        maybe_unwrap(layout.num_n_slots)
    )
    assert sum(len(schedule.commits(loop, pane)) for pane in range(panes)) == groups
    # commit_per_stage is the predicate the emitter actually branches on; the
    # wait_at_stage_head alias it used to go through had no production caller.
    assert _plain(tc.commit_per_stage()) is stage_head


@pytest.mark.parametrize("has_scales", [False, True])
def test_pipeline_state_aggregates_round_trip_separately(has_scales):
    # Opaque handles need no GPU or IR builder. Real Triton tensors still exercise
    # the aggregate types used to flatten and rebuild values across device calls.
    pointer_names = (
        "a_hbm_ptr",
        "b_hbm_ptr",
        "a_scale_hbm_ptr",
        "b_scale_hbm_ptr",
    )
    pointer_values = [
        gl.tensor(object(), gl.pointer_type(gl.uint8)) for _ in pointer_names
    ]
    if not has_scales:
        pointer_values[2:] = [None] * 2
    pointers = _PipelinePointers(*pointer_values)
    fragment_names = ("a_payload", "a_scale", "b_payload", "b_scale", "acc")
    fragment_values = [
        gl.tuple([gl.tensor(object(), dtype) for _ in range(2)])
        for dtype in (gl.uint8, gl.uint8, gl.uint8, gl.uint8, gl.float32)
    ]
    fragments = _PipelineRegFragments(*fragment_values)
    states = gl.tuple((pointers, fragments))
    handles = []
    states._flatten_ir(handles)
    expected = [value.handle for value in pointer_values if value is not None]
    expected += [value.handle for values in fragment_values for value in values]
    assert handles == expected
    restored, cursor = states.type._unflatten_ir(handles, 0)
    assert cursor == len(handles)
    assert tuple(name for name, _ in restored[0].type.fields) == pointer_names
    assert tuple(name for name, _ in restored[1].type.fields) == fragment_names
    for name, value in zip(pointer_names, pointer_values):
        actual = getattr(restored[0], name)
        if value is None:
            assert isinstance(actual, gl.constexpr) and actual.value is None
        else:
            assert actual.handle is value.handle
    for name, values in zip(fragment_names, fragment_values):
        assert [value.handle for value in getattr(restored[1], name)] == [
            value.handle for value in values
        ]


@pytest.mark.parametrize("mask", range(16))
def test_payload_and_scale_read_placement_are_independent(config, mask):
    config.update(conf._ds_read_flags(mask))
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    actual = {
        component
        for component, operand, scale in (
            (DSReadOperand.A, 0, False),
            (DSReadOperand.B, 1, False),
            (DSReadOperand.A_SCALE, 0, True),
            (DSReadOperand.B_SCALE, 1, True),
        )
        if _plain(tc.ds_read_in_mfma(operand, scale))
    }
    assert sum(actual) == mask
    assert _plain(_validate(tc, 4096, 7168))


@pytest.mark.parametrize("mask", [0, 5, 15])
def test_legacy_read_placement_config_key(config, mask):
    for name, _ in conf._DS_READ_FIELDS:
        config.pop(name)
    config["DS_READ_IN_MFMA"] = mask
    tuning = _tuning_spec(config)
    actual = sum(
        int(bit) for name, bit in conf._DS_READ_FIELDS if getattr(tuning, name)
    )
    assert actual == mask

    config["DS_READ_A_PAYLOAD_IN_MFMA"] = not bool(mask & DSReadOperand.A)
    with pytest.raises(ValueError, match="legacy DS_READ_IN_MFMA disagree"):
        _tuning_spec(config)


@pytest.mark.parametrize("mask", [-1, 16])
def test_legacy_read_placement_config_key_rejects_unknown_bits(config, mask):
    for name, _ in conf._DS_READ_FIELDS:
        config.pop(name)
    config["DS_READ_IN_MFMA"] = mask
    with pytest.raises(AssertionError, match="unknown operand bits"):
        _tuning_spec(config)


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, 0),
        ({"DS_MOVE": "1"}, 5),
        ({"DS_MOVE": "2"}, 15),
        ({"DS_IN_MFMA": "1", "DS_MOVE": "0"}, 15),
        ({"DS_READ_IN_MFMA": "8", "DS_IN_MFMA": "1", "DS_MOVE": "2"}, 8),
        ({"DS_READ_IN_MFMA": "0", "DS_IN_MFMA": "1"}, 0),
    ],
)
def test_legacy_read_placement_translation(monkeypatch, settings, expected):
    for name, value in settings.items():
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + name, value)
    config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    tuning = _tuning_spec(config)
    actual = sum(
        int(bit) for name, bit in conf._DS_READ_FIELDS if getattr(tuning, name)
    )
    assert actual == expected


@pytest.mark.parametrize("field,bit", conf._DS_READ_FIELDS)
def test_individual_read_placement_environment(monkeypatch, field, bit):
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, "1")
    config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    tuning = _tuning_spec(config)
    actual = sum(
        int(component)
        for name, component in conf._DS_READ_FIELDS
        if getattr(tuning, name)
    )
    assert actual == int(bit)


def test_legacy_schedule_settings_reach_the_tuning_spec(monkeypatch):
    for name, value in {
        "SCHED_MODE": "4",
        "SOFF_UNROLL": "1",
        "SCALE_FILL_MID": "1",
    }.items():
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + name, value)
    config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    tuning = _tuning_spec(config)
    assert (
        tuning.SCHED_MODE,
        tuning.SOFF_UNROLL,
        tuning.SCALE_FILL_MID,
    ) == (SchedMode.MFMA_8, True, True)


def test_unroll_epilogue_environment_control_is_removed(monkeypatch):
    baseline = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_UNROLL_EPILOGUE", "0")
    actual = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert actual == baseline
    assert actual["UNROLL_EPILOGUE"] is True


@pytest.mark.parametrize("field", ["MANUAL_PP", "MINI_BLOCK_K", "FROZEN_STEP"])
def test_removed_controls_do_not_change_tuning(config, monkeypatch, field):
    baseline = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, "1")
    actual = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert actual == baseline
    assert field not in actual
    assert field not in TuningSpec._fields
    assert _tuning_spec({**config, field: True}) == _tuning_spec(config)


@pytest.mark.parametrize("storage_mask", range(8))
def test_register_storage_options_are_independent(config, storage_mask):
    config.update(
        ALLOW_B_VIA_LDS=not (storage_mask & 1),
        ALLOW_B_SCALE_VIA_LDS=not (storage_mask & 2),
        ALLOW_A_SCALE_VIA_LDS=not (storage_mask & 4),
        B_PRESHUFFLED=bool(storage_mask & 1),
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=3,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=3,
        K_UNROLL=1,
    )
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    layout = KernelLayout(tc)
    assert _plain(layout.component_via_lds(A_PAYLOAD))
    assert _plain(layout.component_via_lds(B_PAYLOAD)) is not bool(storage_mask & 1)
    assert _plain(layout.component_via_lds(A_SCALE)) is not bool(storage_mask & 4)
    assert _plain(layout.component_via_lds(B_SCALE)) is not bool(storage_mask & 2)
    expected_via_lds = {
        A_PAYLOAD: True,
        A_SCALE: not bool(storage_mask & 4),
        B_PAYLOAD: not bool(storage_mask & 1),
        B_SCALE: not bool(storage_mask & 2),
    }
    for component in COMPONENTS:
        via_lds = expected_via_lds[component]
        # These tiles all admit the copy, so the knob policy alone decides.
        assert _plain(tc.allow_component_via_lds(component)) is via_lds
        assert _plain(layout.component_via_lds(component)) is via_lds
        assert _plain(layout.component_in_reg(component)) is not via_lds
    # Full-block byte counts make a shared scale fill tile and slot splitting
    # irrelevant to the allocation budget. Register storage removes only its own
    # allocation; every remaining component uses its individual ring depth.
    bm, bn, bk = (config[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    expected = bm * bk // 2 * 2
    if not storage_mask & 1:
        expected += bn * bk // 2 * 3
    if not storage_mask & 2:
        expected += bn * bk // 32 * 3
    if not storage_mask & 4:
        expected += bm * bk // 32 * 2
    assert _plain(maybe_unwrap(layout.lds_bytes)) == expected
    assert maybe_unwrap(_schedule(tc, layout).unroll) == 6
    assert _plain(_validate(tc, 4096, 7168))
    assert _plain(_validate(tc, 4096, bk * _minimum(tc, layout)))
    with pytest.raises(AssertionError, match="NUM_K"):
        conf._validate_selected_pipeline(tc, layout, bk)


@pytest.mark.parametrize(
    "counts,period,expected",
    [
        ((6, 4, 2, 8), 8, 168),
        ((4, 4, 2, 2), 4, 28),
        ((0, 4, 0, 6), 12, 84),
    ],
)
def test_all_component_buffer_counts_determine_lcm_unroll(
    config, counts, period, expected
):
    config.update(zip(conf._COMPONENT_BUFFER_KEYS, counts, strict=True))
    config.update(
        K_UNROLL=7,
        ALLOW_B_VIA_LDS=False,
        B_PRESHUFFLED=True,
        ALLOW_A_SCALE_VIA_LDS=False,
        ALLOW_B_SCALE_VIA_LDS=False,
    )
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    layout = KernelLayout(tc)
    schedule = _schedule(tc, layout)
    assert maybe_unwrap(schedule.reg_buffer_period) == period
    assert maybe_unwrap(schedule.unroll) == expected
    resolved = [count or config["NUM_LDS_BUFFER"] for count in counts]
    assert [
        _plain(tc.num_buffers(operand, scale))
        for scale in (False, True)
        for operand in (0, 1)
    ] == resolved
    assert maybe_unwrap(schedule.depth) == max(resolved)


def test_lds_depths_constrain_unroll(config):
    config["K_UNROLL"] = 7
    fc = KernelFuncConfig(*_func_spec())
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert maybe_unwrap(_schedule(tc, KernelLayout(tc)).unroll) == 21
    config.update(A_NUM_BUFFER=2, B_NUM_BUFFER=4, A_SCALE_NUM_BUFFER=2)
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert maybe_unwrap(_schedule(tc, KernelLayout(tc)).unroll) == 84
    config["ALLOW_A_SCALE_VIA_LDS"] = False
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert maybe_unwrap(_schedule(tc, KernelLayout(tc)).unroll) == 84


def test_small_payload_tiles_use_register_rings(config):
    config.update(
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        SLOT_M=32,
        SLOT_N=32,
        VGPR_PREFETCH_K=32,
        mfma_instr_shape=(16, 16, 16),
        warps_per_cta=(2, 2),
        tiles_per_warp=(1, 1),
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=3,
        K_UNROLL=1,
        WARP_PIPELINE=0,
    )
    fc = KernelFuncConfig(
        *_func_spec()._replace(
            token_dtype_quant=int(DtypeQuant.BF16),
            expert_dtype_quant=int(DtypeQuant.BF16),
            token_online_quant=int(DtypeQuant.BF16),
            expert_online_quant=int(DtypeQuant.BF16),
            output_quant=None,
        )
    )
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    layout = KernelLayout(tc)
    # Tuning allows both payloads through LDS; the tiles are too small for the copy,
    # so the layout falls back to registers on its own.
    assert _plain(tc.allow_component_via_lds(A_PAYLOAD))
    assert _plain(tc.allow_component_via_lds(B_PAYLOAD))
    assert _plain(layout.component_in_reg(A_PAYLOAD))
    assert _plain(layout.component_in_reg(B_PAYLOAD))
    assert not _plain(layout.component_via_lds(A_PAYLOAD))
    assert not _plain(layout.component_via_lds(B_PAYLOAD))
    assert config["ALLOW_B_VIA_LDS"]
    assert not config["B_PRESHUFFLED"]
    assert maybe_unwrap(_schedule(tc, layout).reg_buffer_period) == 6
    assert _plain(maybe_unwrap(layout.lds_bytes)) == 0
    assert _plain(_validate(tc, 64, 512))


@pytest.mark.parametrize("register_b", [False, True])
def test_absent_scales_do_not_participate_in_depth_unroll_or_validation(
    config, register_b
):
    config.update(
        A_NUM_BUFFER=3,
        B_NUM_BUFFER=4,
        A_SCALE_NUM_BUFFER=-1,
        B_SCALE_NUM_BUFFER=1,
        ALLOW_B_VIA_LDS=not register_b,
        B_PRESHUFFLED=register_b,
        ALLOW_A_SCALE_VIA_LDS=False,
        ALLOW_B_SCALE_VIA_LDS=False,
    )
    fc = KernelFuncConfig(
        *_func_spec()._replace(
            token_dtype_quant=int(DtypeQuant.BF16),
            expert_dtype_quant=int(DtypeQuant.BF16),
            token_online_quant=int(DtypeQuant.BF16),
            expert_online_quant=int(DtypeQuant.BF16),
        )
    )
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    layout = KernelLayout(tc)
    # Placement accessors describe physical policy only. The absent scale streams
    # are still gated by the function configuration at every scheduling caller.
    assert _plain(layout.component_in_reg(A_SCALE))
    assert _plain(layout.component_in_reg(B_SCALE))
    schedule = _schedule(tc, layout)
    for pane in range(
        maybe_unwrap(layout.num_m_slots) * maybe_unwrap(layout.num_n_slots)
    ):
        slots = schedule.buffer_load_slots(pane)
        assert slots[A_SCALE.index()] is None
        assert slots[B_SCALE.index()] is None
    assert maybe_unwrap(schedule.depth) == 4
    assert maybe_unwrap(schedule.unroll) == 12
    assert _plain(conf._validate_selected_pipeline(tc, layout, 7168))


@pytest.mark.parametrize("field", conf._COMPONENT_BUFFER_KEYS)
@pytest.mark.parametrize("depth", [-1, 1])
def test_active_component_buffer_counts_below_two_are_rejected(config, field, depth):
    config[field] = depth
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match=field + ".*must be at least 2"):
        validate(tc, 4096, 7168)


@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_inherited_buffer_counts_are_validated_after_resolution(config, depth):
    config["NUM_LDS_BUFFER"] = depth
    fc = KernelFuncConfig(*_func_spec())
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    with pytest.raises(AssertionError, match="A_NUM_BUFFER.*must be at least 2"):
        conf._validate_selected_pipeline(tc, KernelLayout(tc), 7168)
    config.update(dict.fromkeys(conf._COMPONENT_BUFFER_KEYS, 2))
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert _plain(_validate(tc, 4096, 7168))


@pytest.mark.parametrize("unroll", [-1, 0])
@pytest.mark.parametrize("registers", [False, True])
def test_unroll_must_be_positive_even_with_register_rings(config, unroll, registers):
    config.update(K_UNROLL=unroll, ALLOW_A_SCALE_VIA_LDS=not registers)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="K_UNROLL must be at least 1"):
        conf._validate_selected_pipeline(tc, KernelLayout(tc), 7168)


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("unroll", [1, 2, 3, 7])
def test_shared_depth_exact_minimum_and_all_unroll_remainders(config, depth, unroll):
    config.update(NUM_LDS_BUFFER=depth, K_UNROLL=unroll)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    layout = KernelLayout(tc)
    schedule = _schedule(tc, layout)
    assert maybe_unwrap(schedule.peeled) == 1
    effective_unroll = maybe_unwrap(schedule.unroll)
    assert effective_unroll == math.lcm(depth, unroll)
    minimum = depth + 1 + effective_unroll
    for remainder in range(effective_unroll):
        assert _plain(_validate(tc, 4096, (minimum + remainder) * config["BLOCK_K"]))
    with pytest.raises(
        AssertionError, match="NUM_K.*PIPELINE_DEPTH.*PEELED.*UNROLL"
    ):
        conf._validate_selected_pipeline(tc, layout, (minimum - 1) * config["BLOCK_K"])


def test_register_weights_require_preshuffle(config):
    config["ALLOW_B_VIA_LDS"] = False
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(
        AssertionError, match="ALLOW_B_VIA_LDS=False requires B_PRESHUFFLED"
    ):
        validate(tc, 4096, 7168)


def test_independent_register_scales_are_supported(config):
    config.update(ALLOW_A_SCALE_VIA_LDS=False, ALLOW_B_SCALE_VIA_LDS=False)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    layout = KernelLayout(tc)
    assert _plain(layout.component_in_reg(A_SCALE))
    assert _plain(layout.component_in_reg(B_SCALE))
    assert _plain(layout.component_via_lds(A_PAYLOAD))
    assert _plain(layout.component_via_lds(B_PAYLOAD))
    assert _plain(_validate(tc, 4096, 7168))


@pytest.mark.parametrize("prefetch", [0, 128])
def test_partial_register_prefetch_is_rejected(config, prefetch):
    config["VGPR_PREFETCH_K"] = prefetch
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="VGPR_PREFETCH_K must equal BLOCK_K"):
        _validate(tc, 4096, 7168)


def test_component_buffers_still_require_nonempty_k(config):
    config["A_NUM_BUFFER"] = 2
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="NUM_K.*must be at least"):
        conf._validate_selected_pipeline(tc, KernelLayout(tc), 0)


@pytest.mark.parametrize(
    "field", conf._COMPONENT_BUFFER_KEYS + conf._ALLOW_VIA_LDS_KEYS
)
def test_storage_environment_settings_reach_tuning_spec(monkeypatch, field):
    value = 0 if field in conf._ALLOW_VIA_LDS_KEYS else 2
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, str(value))
    config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert getattr(_tuning_spec(config), field) == value


def test_a_scale_buffer_alias_is_accepted(config, monkeypatch):
    config["A_SCALE_NUMB_BUFFER"] = 2
    assert _tuning_spec(config).A_SCALE_NUM_BUFFER == 2
    config["A_SCALE_NUM_BUFFER"] = 4
    with pytest.raises(
        ValueError, match="A_SCALE_NUM_BUFFER and A_SCALE_NUMB_BUFFER disagree"
    ):
        _tuning_spec(config)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_A_SCALE_NUMB_BUFFER", "2")
    env_config = conf.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert _tuning_spec(env_config).A_SCALE_NUM_BUFFER == 2


@pytest.mark.parametrize("independent", [False, True])
def test_shuffled_k128_scales_constrain_lcm_unroll(config, independent):
    config.update(
        BLOCK_K=128,
        VGPR_PREFETCH_K=128,
        K_UNROLL=3,
        A_NUM_BUFFER=3 if independent else 0,
        A_SCALE_SORTED_SHUFFLED=True,
        B_SCALE_SHUFFLED=True,
    )
    fc = KernelFuncConfig(
        *_func_spec()._replace(
            token_dtype_quant=int(DtypeQuant.MXFP8),
            expert_dtype_quant=int(DtypeQuant.MXFP8),
            token_online_quant=int(DtypeQuant.MXFP8),
            expert_online_quant=int(DtypeQuant.MXFP8),
        )
    )
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert maybe_unwrap(_schedule(tc, KernelLayout(tc)).unroll) == 6
    assert _plain(_validate(tc, 4096, 7168))
    assert conf._scale_shuffle_supported(config, 0)
    assert conf._scale_shuffle_supported(config, 1)


def test_runtime_epilogue_rejects_multistep_scale_cadence(config):
    config.update(
        BLOCK_K=128,
        VGPR_PREFETCH_K=128,
        A_SCALE_SORTED_SHUFFLED=True,
        B_SCALE_SHUFFLED=True,
        UNROLL_EPILOGUE=False,
    )
    fc = KernelFuncConfig(
        *_func_spec()._replace(
            token_dtype_quant=int(DtypeQuant.MXFP8),
            expert_dtype_quant=int(DtypeQuant.MXFP8),
            token_online_quant=int(DtypeQuant.MXFP8),
            expert_online_quant=int(DtypeQuant.MXFP8),
        )
    )
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    with pytest.raises(AssertionError, match="UNROLL_EPILOGUE=False"):
        _validate(tc, 4096, 7168)


@pytest.mark.parametrize("output_quant", [None, int(DtypeQuant.MXFP4)])
def test_gate_up_small_warp_extent_only_requires_mx_group_for_quantized_output(
    config, output_quant
):
    """Boxed None must not impose quantization constraints on FP32 tuning tiles."""
    config.update(
        BLOCK_N=128,
        SLOT_N=64,
        warps_per_cta=(1, 4),
        tiles_per_warp=(2, 1),
        B_PRESHUFFLED=True,
        ALLOW_B_VIA_LDS=False,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
    )
    fc = KernelFuncConfig(*_func_spec()._replace(output_quant=output_quant))
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    if output_quant is None:
        assert _plain(_validate(tc, 2048, 7168))
    else:
        with pytest.raises(AssertionError, match="a warp's emitted N extent"):
            validate(tc, 2048, 7168)


@pytest.mark.parametrize("epilogue", list(EpilogueMode))
def test_epilogue_modes_preserve_output_geometry(config, epilogue):
    func = KernelFuncConfig(*_func_spec(epilogue))
    tc = KernelTuningConfig(func, *_tuning_spec(config))
    layout = KernelLayout(tc)
    assert _plain(func.activation_reduction_n()) == 2
    # WeightGuHbmLayout.SPLIT halves across two N slots, so one slot's raw columns map 1:1 onto
    # emitted channels.
    assert _plain(maybe_unwrap(layout.output_n_slot)) == _plain(tc.SLOT_N)
    assert _plain(layout.grid_N(4096)) == 16
    assert _plain(_validate(tc, 4096, 7168))


def test_epilogue_launch_specs_are_cached_separately(config, monkeypatch):
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_NO_EPI", "2")
    assert host._resolve_epilogue(None) == EpilogueMode.NOP
    assert host._resolve_epilogue(EpilogueMode.DEFAULT) == EpilogueMode.DEFAULT
    args = {
        "block_m": 128,
        "N": 4096,
        "K": 7168,
        "dq_a": DtypeQuant.MXFP4,
        "dq_b": DtypeQuant.MXFP4,
        "small_grid": False,
        "has_bias": True,
        "has_gammas": False,
        "has_gather": True,
        "has_x_static_scale": False,
        "act": _func_spec().activation,
        "out_quant": int(DtypeQuant.MXFP4),
        "config_items": tuple(
            sorted((k, conf._hashable(v)) for k, v in config.items())
        ),
        "gate_up_split": True,
    }
    default = host._launch_spec(**args, epilogue=int(EpilogueMode.DEFAULT))
    nop = host._launch_spec(**args, epilogue=int(EpilogueMode.NOP))
    assert default is host._launch_spec(**args, epilogue=int(EpilogueMode.DEFAULT))
    assert default[0] == nop[0]
    assert _plain(default[1][0]).epilogue == EpilogueMode.DEFAULT
    assert _plain(nop[1][0]).epilogue == EpilogueMode.NOP


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("SCHED_MODE", -1, "not a SchedMode"),
        ("SCHED_MODE", 5, "not a SchedMode"),
        ("epilogue", 3, "not an EpilogueMode"),
    ],
)
def test_invalid_controls_are_rejected(config, field, value, error):
    func = _func_spec(value if field == "epilogue" else EpilogueMode.DEFAULT)
    if field != "epilogue":
        config[field] = value
    tc = KernelTuningConfig(KernelFuncConfig(*func), *_tuning_spec(config))
    with pytest.raises(AssertionError, match=error):
        validate(tc, 4096, 7168)


@pytest.mark.parametrize("scheme", list(WaitCommitScheme))
@pytest.mark.parametrize("storage_mask", range(8))
@pytest.mark.parametrize(
    "counts", [(3, 3, 3, 3), (2, 4, 3, 2), (4, 2, 2, 4), (2, 5, 2, 2)]
)
@pytest.mark.parametrize("geometry", ["pair", "middle_shared_a", "unequal_tiles"])
def test_independent_buffer_waits_match_copy_history(
    config, monkeypatch, scheme, storage_mask, counts, geometry
):
    config.update(zip(conf._COMPONENT_BUFFER_KEYS, counts, strict=True))
    config.update(
        WAIT_COMMIT_SCHEME=int(scheme),
        ALLOW_B_VIA_LDS=not (storage_mask & 1),
        ALLOW_B_SCALE_VIA_LDS=not (storage_mask & 2),
        ALLOW_A_SCALE_VIA_LDS=not (storage_mask & 4),
        B_PRESHUFFLED=bool(storage_mask & 1),
    )
    if geometry == "middle_shared_a":
        config.update(
            SCALE_FILL_MID=True, A_SCALE_SORTED_SHUFFLED=True, SCALE_SLOT_M=128
        )
    elif geometry == "unequal_tiles":
        config.update(SLOT_N=64, SCALE_FILL_MID=True)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    layout = KernelLayout(tc)
    minimum = _minimum(tc, layout)
    # Both strips contain a prologue, a runtime main body and a complete drain. The
    # second commits output-epilogue copies before shallow queues finish filling.
    _audit_buffer_schedule(tc, layout, minimum)
    _audit_buffer_schedule(tc, layout, minimum + 3, epilogue_groups=2)
    _assert_minimal_peel(_schedule(tc, layout, minimum), layout)


@pytest.mark.parametrize("scheme", list(WaitCommitScheme))
@pytest.mark.parametrize("register_b", [False, True])
@pytest.mark.parametrize("counts", [(3, 2, 1, 1), (2, 2, 1, 1)])
def test_independent_buffer_schedule_without_scales(config, scheme, register_b, counts):
    config.update(zip(conf._COMPONENT_BUFFER_KEYS, counts, strict=True))
    config.update(
        WAIT_COMMIT_SCHEME=int(scheme),
        ALLOW_B_VIA_LDS=not register_b,
        B_PRESHUFFLED=register_b,
    )
    fc = KernelFuncConfig(
        *_func_spec()._replace(
            token_dtype_quant=int(DtypeQuant.BF16),
            expert_dtype_quant=int(DtypeQuant.BF16),
            token_online_quant=int(DtypeQuant.BF16),
            expert_online_quant=int(DtypeQuant.BF16),
        )
    )
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    layout = KernelLayout(tc)
    minimum = _minimum(tc, layout)
    _audit_buffer_schedule(tc, layout, minimum)
    _audit_buffer_schedule(tc, layout, minimum + 3, epilogue_groups=2)
    _assert_minimal_peel(_schedule(tc, layout, minimum), layout)
