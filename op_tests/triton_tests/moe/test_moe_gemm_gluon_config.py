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
from aiter.ops.triton._gluon_kernels.gfx950.moe._frozen import _validate_frozen
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
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import Schedule
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    SCALE,
    A,
    ActivationSpec,
    ActKind,
    B,
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


@pytest.fixture(autouse=True)
def _clean_tuning_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    host._launch_spec.cache_clear()
    host._get_gluon_config_cached.cache_clear()
    yield
    host._launch_spec.cache_clear()
    host._get_gluon_config_cached.cache_clear()


@pytest.fixture
def config():
    cfg = host.get_gluon_config_uncached(
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
    return TuningSpec(*(host._hashable(v) for v in host._tuning_args(config)))


def _plain(value):
    value = host._cval(value)
    return tuple(value) if isinstance(value, list) else value


def _validate(tc, N, K):
    layout = KernelLayout(tc)
    validate(tc, N, K, layout)
    return host._validate_selected_pipeline(tc, layout, K)


def _schedule(tc, layout, num_k=1, epilogue_groups=0):
    """The build's schedule for ``num_k`` K tiles; one tile still gives its sizing."""
    return Schedule(tc, layout, num_k * _plain(tc.BLOCK_K), epilogue_groups)


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
        ("FROZEN_STEP", True),
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
    config = host.get_gluon_config_uncached(
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
    expected = 96 // host._LEGACY_SLOT_DIVISORS.get(field, 1)
    # A raw launch config is renamed before the host reads any slot key from it.
    canonical = host._canonical_slot_keys(config)
    assert canonical[renamed] == expected and legacy not in canonical
    assert getattr(_tuning_spec(config), field) == expected

    config[field] = 128
    with pytest.raises(ValueError, match="disagree"):
        _tuning_spec(config)


@pytest.mark.parametrize("canonical", list(host._LEGACY_SLOT_KEYS))
def test_legacy_slot_environment_names(monkeypatch, canonical):
    prefix = "AITER_TRITON_MOE_GLUON_"
    legacy = host._LEGACY_SLOT_KEYS[canonical]
    monkeypatch.delenv(prefix + canonical, raising=False)
    monkeypatch.setenv(prefix + legacy, "64")
    expected = 64 // host._LEGACY_SLOT_DIVISORS.get(canonical, 1)
    assert host._env_slot_int(canonical, 0) == expected
    monkeypatch.setenv(prefix + canonical, "128")
    assert host._env_slot_int(canonical, 0) == 128


def test_legacy_scale_slot_environment_reaches_the_config(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SCALE_MINI_BLOCK_K", "512")
    config = host.get_gluon_config_uncached(
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
    assert host._expected_m_tiles_per_expert(M, n_expts_tot, block_m) == expected


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
        host._stream_mxfp8_gemm1_expert_payload(
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
    tuning = host._probe_tuning_config(
        config, DtypeQuant.MXFP8, DtypeQuant.MXFP8
    )
    assert host._cval(validate(tuning, 4096, 7168))
    layout = KernelLayout(tuning)
    assert host._cval(host._validate_selected_pipeline(tuning, layout, 7168))
    assert host._probe_lds_bytes(
        config, DtypeQuant.MXFP8, DtypeQuant.MXFP8
    ) <= host.LDS_USABLE_BYTES

    reused = host._default_launch_config(
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
    config = host._default_launch_config(
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
    tuning = host._probe_tuning_config(config, DtypeQuant.MXFP8, dq_b)
    assert host._cval(validate(tuning, 4096, 7168))
    # The point of the tile: several CTAs share a CU's LDS.
    assert (
        host._probe_lds_bytes(config, DtypeQuant.MXFP8, dq_b)
        <= host.LDS_USABLE_BYTES // ctas_per_cu
    )


@pytest.mark.parametrize(
    "env,sched_mode", [(None, SchedMode.MFMA_8), ("0", SchedMode.NONE)]
)
def test_large_t_mxfp8_mxfp4_gemm1_interleaves_mfma_with_memory(
    monkeypatch, env, sched_mode
):
    # The environment still wins over the measured default.
    if env is not None:
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SCHED_MODE", env)
    config = host._default_launch_config(
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
    tuning = host._probe_tuning_config(config, DtypeQuant.MXFP8, DtypeQuant.MXFP4)
    assert host._cval(validate(tuning, 4096, 7168))
    layout = KernelLayout(tuning)
    assert host._cval(host._validate_selected_pipeline(tuning, layout, 7168))


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
def test_t256_mxfp8_gemm1_geometry_override_is_narrow(overrides):
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
    selected = host._default_launch_config(**args)
    baseline = host.get_gluon_config(
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
    else:
        assert selected == baseline


@pytest.mark.parametrize("suffix", ["EXPERT_CACHE_MODIFIER", "EXPERT_MOD"])
@pytest.mark.parametrize("override", ["", ".ca"])
def test_streaming_expert_payload_respects_environment_override(
    monkeypatch, suffix, override
):
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + suffix, override)
    config = host.get_gluon_config_uncached(
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
    config = host.get_gluon_config_uncached(
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
            sorted((key, host._hashable(value)) for key, value in config.items())
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
    config = host._default_launch_config(
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
    config = host._default_launch_config(
        16, 4096, 1024, DtypeQuant.MXFP8, DtypeQuant.MXFP8, False, True
    )
    assert config["BLOCK_K"] == 128
    assert config["expert_scale_cache_modifier"] == expected


@pytest.mark.parametrize(
    "scheme,groups,stage_head",
    [
        (WaitCommitScheme.PER_OP, 8, False),
        (WaitCommitScheme.PER_PANEL, 4, False),
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
    assert _plain(layout.component_via_lds((0, True)))
    assert _plain(layout.component_via_lds((1, True)))
    schedule = _schedule(tc, layout, 64)
    loop = schedule.loop_step(0)
    panels = _plain(maybe_unwrap(layout.num_m_slots)) * _plain(
        maybe_unwrap(layout.num_n_slots)
    )
    assert sum(len(schedule.commits(loop, panel)) for panel in range(panels)) == groups
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
    config.update(host._ds_read_flags(mask))
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
    for name, _ in host._DS_READ_FIELDS:
        config.pop(name)
    config["DS_READ_IN_MFMA"] = mask
    tuning = _tuning_spec(config)
    actual = sum(
        int(bit) for name, bit in host._DS_READ_FIELDS if getattr(tuning, name)
    )
    assert actual == mask

    config["DS_READ_A_PAYLOAD_IN_MFMA"] = not bool(mask & DSReadOperand.A)
    with pytest.raises(ValueError, match="legacy DS_READ_IN_MFMA disagree"):
        _tuning_spec(config)


@pytest.mark.parametrize("mask", [-1, 16])
def test_legacy_read_placement_config_key_rejects_unknown_bits(config, mask):
    for name, _ in host._DS_READ_FIELDS:
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
    config = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    tuning = _tuning_spec(config)
    actual = sum(
        int(bit) for name, bit in host._DS_READ_FIELDS if getattr(tuning, name)
    )
    assert actual == expected


@pytest.mark.parametrize("field,bit", host._DS_READ_FIELDS)
def test_individual_read_placement_environment(monkeypatch, field, bit):
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, "1")
    config = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    tuning = _tuning_spec(config)
    actual = sum(
        int(component)
        for name, component in host._DS_READ_FIELDS
        if getattr(tuning, name)
    )
    assert actual == int(bit)


def test_legacy_schedule_settings_reach_the_tuning_spec(monkeypatch):
    for name, value in {
        "SCHED_MODE": "4",
        "FROZEN_STEP": "1",
        "SOFF_UNROLL": "1",
        "SCALE_FILL_MID": "1",
    }.items():
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + name, value)
    config = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    tuning = _tuning_spec(config)
    assert (
        tuning.SCHED_MODE,
        tuning.FROZEN_STEP,
        tuning.SOFF_UNROLL,
        tuning.SCALE_FILL_MID,
    ) == (SchedMode.MFMA_8, True, True, True)


def test_unroll_epilogue_environment_control_is_removed(monkeypatch):
    baseline = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_UNROLL_EPILOGUE", "0")
    actual = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert actual == baseline
    assert actual["UNROLL_EPILOGUE"] is True


@pytest.mark.parametrize("field", ["MANUAL_PP", "MINI_BLOCK_K"])
def test_removed_controls_do_not_change_tuning(config, monkeypatch, field):
    baseline = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, "1")
    actual = host.get_gluon_config_uncached(
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
    assert _plain(layout.component_via_lds((0, False)))
    assert _plain(layout.component_via_lds((1, False))) is not bool(storage_mask & 1)
    assert _plain(layout.component_via_lds((0, True))) is not bool(storage_mask & 4)
    assert _plain(layout.component_via_lds((1, True))) is not bool(storage_mask & 2)
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
        host._validate_selected_pipeline(tc, layout, bk)


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
    config.update(zip(host._COMPONENT_BUFFER_KEYS, counts, strict=True))
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

    # A FROZEN_STEP build is validated by the frozen snapshot's own copy.
    config.update(A_NUM_BUFFER=0, B_NUM_BUFFER=0, FROZEN_STEP=True)
    frozen = KernelTuningConfig(fc, *_tuning_spec(config))
    with pytest.raises(AssertionError, match="requires operand 0's payload in LDS"):
        _validate_frozen(frozen, 64, 512)


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
    for panel in range(
        maybe_unwrap(layout.num_m_slots) * maybe_unwrap(layout.num_n_slots)
    ):
        slots = schedule.buffer_load_slots(panel)
        assert slots[COMPONENTS.index(A_SCALE)] is None
        assert slots[COMPONENTS.index(B_SCALE)] is None
    assert maybe_unwrap(schedule.depth) == 4
    assert maybe_unwrap(schedule.unroll) == 12
    assert _plain(host._validate_selected_pipeline(tc, layout, 7168))


@pytest.mark.parametrize("field", host._COMPONENT_BUFFER_KEYS)
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
        host._validate_selected_pipeline(tc, KernelLayout(tc), 7168)
    config.update(dict.fromkeys(host._COMPONENT_BUFFER_KEYS, 2))
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert _plain(_validate(tc, 4096, 7168))


@pytest.mark.parametrize("unroll", [-1, 0])
@pytest.mark.parametrize("registers", [False, True])
def test_unroll_must_be_positive_even_with_register_rings(config, unroll, registers):
    config.update(K_UNROLL=unroll, ALLOW_A_SCALE_VIA_LDS=not registers)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="K_UNROLL must be at least 1"):
        host._validate_selected_pipeline(tc, KernelLayout(tc), 7168)


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
        host._validate_selected_pipeline(tc, layout, (minimum - 1) * config["BLOCK_K"])


@pytest.mark.parametrize(
    "field", host._COMPONENT_BUFFER_KEYS + host._ALLOW_VIA_LDS_KEYS
)
def test_frozen_reference_rejects_new_storage_options(config, field):
    config.update(FROZEN_STEP=True, B_PRESHUFFLED=True)
    config[field] = 0 if field in host._ALLOW_VIA_LDS_KEYS else 1
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="FROZEN_STEP does not support"):
        _validate_frozen(tc, 4096, 7168)


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
        host._validate_selected_pipeline(tc, KernelLayout(tc), 0)


@pytest.mark.parametrize(
    "field", host._COMPONENT_BUFFER_KEYS + host._ALLOW_VIA_LDS_KEYS
)
def test_storage_environment_settings_reach_tuning_spec(monkeypatch, field):
    value = 0 if field in host._ALLOW_VIA_LDS_KEYS else 2
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, str(value))
    config = host.get_gluon_config_uncached(
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
    env_config = host.get_gluon_config_uncached(
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
    assert host._scale_shuffle_supported(config, 0)
    assert host._scale_shuffle_supported(config, 1)


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
            sorted((k, host._hashable(v)) for k, v in config.items())
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


def _executed_step(schedule, num_k, k_step):
    """Index of the emitted step that executes K step ``k_step`` for this K.

    Stated from the loop structure alone: the prologue, the warm-up and peeled
    steps, whole loop bodies, the remainder, then the drain copy for the ring phase
    the main steps end on.
    """
    depth, peeled, unroll = (
        maybe_unwrap(schedule.depth),
        maybe_unwrap(schedule.peeled),
        maybe_unwrap(schedule.unroll),
    )
    main = num_k - depth
    if k_step < 0:
        return schedule.prologue_step(k_step)
    if k_step == 0:
        return schedule.warmup_step()
    if k_step <= peeled:
        return schedule.peeled_step(k_step - 1)
    if k_step > main:
        static_drain = maybe_unwrap(schedule.static_drain)
        phase = main % maybe_unwrap(schedule.buffer_period) if static_drain else 0
        return schedule.drain_step(phase, k_step - main - 1)
    loop_end = peeled + (main - peeled) // unroll * unroll
    if k_step <= loop_end:
        return schedule.loop_step((k_step - peeled - 1) % unroll)
    if maybe_unwrap(schedule.static_remainder):
        return schedule.remainder_step(k_step - loop_end - 1)
    return schedule.remainder_step(0)


def _audit_buffer_schedule(tc, layout, num_k, epilogue_groups=0):
    """Compare the commit and wait plans with a complete chronological simulation.

    The oracle records absolute K tiles and group sequence numbers, and does not use
    the schedule's replay. It verifies direct-register DS reads against the
    independent producer history; the buffered-pipeline oracle separately checks
    exact ring-bank contents and overwrite safety.
    """
    schedule = _schedule(tc, layout, num_k, epilogue_groups)
    nm = _plain(maybe_unwrap(layout.num_m_slots))
    nn = _plain(maybe_unwrap(layout.num_n_slots))
    npanels = nm * nn
    if (nm, nn) == (2, 2):
        payloads = [
            (B_PAYLOAD, 0),
            (A_PAYLOAD, 0),
            (A_PAYLOAD, 1),
            (B_PAYLOAD, 1),
        ]
    else:
        payloads = []
        for tile in range(max(nm, nn)):
            if tile < nm:
                payloads.append((A_PAYLOAD, tile))
            if tile < nn:
                payloads.append((B_PAYLOAD, tile))
    has_scale = {operand: _plain(tc.func_cfg.has_scale(operand)) for operand in (A, B)}
    depths, ratios, via_lds, producer_phases = {}, {}, {}, {}
    load_to_mfma_distances = {}
    for component in COMPONENTS:
        operand, is_scale = component
        depths[component] = _plain(tc.num_buffers(operand, is_scale))
        ratios[component] = (
            _plain(layout.scale_slot_k(operand)) // _plain(layout.scale_block_k)
            if is_scale
            else 1
        )
        via_lds[component] = _plain(layout.component_via_lds(component))
        live_span = (depths[component] - 1) * ratios[component] + 1
        load_to_mfma_distances[component] = live_span - int(not via_lds[component])
        producer_phases[component] = (
            1 - load_to_mfma_distances[component]
        ) % ratios[component]
    non_k_ratios = {
        operand: _plain(layout.scale_slot_nonk(operand))
        // _plain(layout.slot_nonk(operand))
        for operand in (A, B)
    }
    payload_panels = {
        (component, tile): panel
        for panel, (component, tile) in enumerate(payloads)
    }
    buffer_load_panels = [[] for _ in range(npanels)]
    for panel, (component, tile) in enumerate(payloads):
        operand, is_scale = component
        assert not is_scale
        buffer_load_panels[panel].append((component, tile))
        if not has_scale[operand] or tile % non_k_ratios[operand]:
            continue
        scale_slot = (
            tile + 1 if (nm, nn) == (2, 2) and _plain(tc.SCALE_FILL_MID) else panel
        )
        buffer_load_panels[scale_slot].append(((operand, SCALE), tile))
    component_rank = {
        A_PAYLOAD: 0,
        A_SCALE: 1,
        B_PAYLOAD: 2,
        B_SCALE: 3,
    }
    buffer_load_panels = [
        tuple(sorted(ops, key=lambda copy: component_rank[copy[0]]))
        for ops in buffer_load_panels
    ]
    buffer_load_panel_for = {
        copy: panel
        for panel, copies in enumerate(buffer_load_panels)
        for copy in copies
    }
    ds_read_panels = [[] for _ in range(npanels)]
    for copy, buffer_load_panel in buffer_load_panel_for.items():
        component, tile = copy
        nominal = payload_panels[((component[0], False), tile)]
        ds_read_panel = nominal
        if not via_lds[component] and load_to_mfma_distances[component] == 1:
            ds_read_panel = max(ds_read_panel, buffer_load_panel)
        assert all(
            existing[0] != component
            for existing in ds_read_panels[ds_read_panel]
        )
        ds_read_panels[ds_read_panel].append(copy)
    for panel, expected in enumerate(buffer_load_panels):
        actual = schedule.buffer_load_slots(panel)
        assert (
            tuple(
                (component, tile)
                for component, tile in zip(COMPONENTS, actual)
                if tile is not None
            )
            == expected
        )

    committed, group_for_copy, issued, consumed, lds = [], {}, set(), set(), {}
    depth = maybe_unwrap(schedule.depth)
    main = num_k - depth
    assert num_k >= depth + maybe_unwrap(schedule.peeled) + maybe_unwrap(
        schedule.unroll
    )
    scheme = _plain(tc.WAIT_COMMIT_SCHEME)
    per_stage = scheme in (
        WaitCommitScheme.PER_STAGE_WHOLE,
        WaitCommitScheme.PER_STAGE_WARP_PIPELINE,
    )

    def required_copies(stage, panel):
        need = []
        panels = range(npanels) if panel is None else (panel,)
        for ds_read_panel in panels:
            for component, tile in ds_read_panels[ds_read_panel]:
                if stage % ratios[component] == 0:
                    need.append(
                        (stage - load_to_mfma_distances[component] + 1, component, tile)
                    )
        return need

    def check_wait(stage, panel):
        required = [
            token for token in required_copies(stage, panel) if via_lds[token[1]]
        ]
        expected = (
            len(committed) - 1 - max(group_for_copy[token] for token in required)
            if required
            else None
        )
        # One immediate serves every execution of an emitted step, so this checks
        # each loop-body step once per loop iteration.
        assert schedule.wait(_executed_step(schedule, num_k, stage), panel) == expected

    for stage in range(1 - depth, num_k):
        pending_stage = []
        expected_groups = []
        stage_copies = [
            tuple(
                (stage, component, tile)
                for component, tile in ops
                if stage % ratios[component] == producer_phases[component]
                and 0 <= stage + load_to_mfma_distances[component] - 1 < num_k
            )
            for ops in buffer_load_panels
        ]
        for panel, copies in enumerate(stage_copies):
            asynchronous = tuple(token for token in copies if via_lds[token[1]])
            if scheme == WaitCommitScheme.PER_OP:
                groups = tuple((copy,) for copy in asynchronous)
            elif scheme == WaitCommitScheme.PER_PANEL:
                groups = (asynchronous,)
            else:
                pending_stage.extend(asynchronous)
                groups = (tuple(pending_stage),) if panel == npanels - 1 else ()
            expected_groups.append(groups)
        index = _executed_step(schedule, num_k, stage)
        actual_groups = tuple(
            schedule.commits(index, panel) for panel in range(npanels)
        )
        assert actual_groups == tuple(
            tuple(tuple(token[1:] for token in group) for group in groups)
            for groups in expected_groups
        )

        def fill_panel(
            panel, copies_by_panel=stage_copies, groups_by_panel=expected_groups
        ):
            for token in copies_by_panel[panel]:
                producer, component, tile = token
                # The ring buffer is named by the step that consumes the value, so
                # it has to be derived from the producer, not carried in the token.
                target = producer + load_to_mfma_distances[component] - 1
                assert token not in issued, "a K tile was loaded twice"
                issued.add(token)
                if via_lds[component]:
                    ring = target // ratios[component] % depths[component]
                    key = component, tile, ring
                    if key in lds:
                        # The previous occupant of this ring buffer must already
                        # have been read; name it by its producer, as `consumed`
                        # does, not by the step that consumed it.
                        assert (
                            lds[key] - load_to_mfma_distances[component] + 1,
                            component,
                            tile,
                        ) in consumed
                    lds[key] = target
            for group in groups_by_panel[panel]:
                for token in group:
                    group_for_copy[token] = len(committed)
                committed.append(group)

        def ds_read_panel(panel, after_fill, ds_read_stage=stage):
            for component, tile in ds_read_panels[panel]:
                if ds_read_stage % ratios[component]:
                    continue
                same_panel_fill = (
                    not via_lds[component]
                    and load_to_mfma_distances[component] == 1
                    and buffer_load_panel_for[component, tile] == panel
                )
                if same_panel_fill != after_fill:
                    continue
                token = (
                    ds_read_stage - load_to_mfma_distances[component] + 1,
                    component,
                    tile,
                )
                if via_lds[component]:
                    ring = ds_read_stage // ratios[component] % depths[component]
                    assert lds[component, tile, ring] == ds_read_stage
                else:
                    assert token in issued, "register DS-read before its producer"
                consumed.add(token)

        if stage < 0:
            for panel in range(npanels):
                fill_panel(panel)
        if stage >= 0:
            if per_stage:
                check_wait(stage, None)
            for panel in range(npanels):
                if not per_stage:
                    check_wait(stage, panel)
                ds_read_panel(panel, False)
                fill_panel(panel)
                ds_read_panel(panel, True)
        if stage == main:
            committed.extend(() for _ in range(epilogue_groups))
    expected_loads = {
        (target - load_to_mfma_distances[component] + 1, component, tile)
        for ops in buffer_load_panels
        for component, tile in ops
        for target in range(0, num_k, ratios[component])
    }
    assert issued == expected_loads
    assert consumed == expected_loads


@pytest.mark.parametrize("scheme", list(WaitCommitScheme))
@pytest.mark.parametrize("storage_mask", range(8))
@pytest.mark.parametrize(
    "counts", [(3, 3, 3, 3), (2, 4, 3, 2), (4, 2, 2, 4), (2, 5, 2, 2)]
)
@pytest.mark.parametrize("geometry", ["pair", "middle_shared_a", "unequal_tiles"])
def test_independent_buffer_waits_match_copy_history(
    config, monkeypatch, scheme, storage_mask, counts, geometry
):
    config.update(zip(host._COMPONENT_BUFFER_KEYS, counts, strict=True))
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
    # Both strips contain warmup, a runtime main body and a complete drain. The
    # second commits output-epilogue copies before shallow queues finish filling.
    _audit_buffer_schedule(tc, layout, minimum)
    _audit_buffer_schedule(tc, layout, minimum + 3, epilogue_groups=2)


@pytest.mark.parametrize("scheme", list(WaitCommitScheme))
@pytest.mark.parametrize("register_b", [False, True])
@pytest.mark.parametrize("counts", [(3, 2, 1, 1), (2, 2, 1, 1)])
def test_independent_buffer_schedule_without_scales(config, scheme, register_b, counts):
    config.update(zip(host._COMPONENT_BUFFER_KEYS, counts, strict=True))
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
