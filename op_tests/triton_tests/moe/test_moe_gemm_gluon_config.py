# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the host/device configuration contract of the Gluon MoE GEMM."""

import os

import pytest
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._entry import MoeKernelConfig
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DSReadOperand,
    DtypeQuant,
    EpilogueMode,
    FuncSpec,
    SchedMode,
    TuningSpec,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe.moe_gemm import (
    _build_configs,
    _PipelinePointers,
    _PipelineRegFragments,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host


@pytest.fixture(autouse=True)
def _clean_tuning_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    host._launch_spec.cache_clear()
    yield
    host._launch_spec.cache_clear()


@pytest.fixture
def config():
    cfg = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    cfg.update(
        MINI_BLOCK_M=64,
        MINI_BLOCK_N=128,
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
        True,
        int(epilogue),
    )


def _tuning_spec(config):
    return TuningSpec(*(host._hashable(v) for v in host._tuning_args(config)))


def _plain(value):
    value = host._cval(value)
    return tuple(value) if isinstance(value, list) else value


@pytest.mark.parametrize(
    "field,value",
    [
        ("B_PRESHUFFLED", True),
        ("DS_READ_IN_MFMA", int(DSReadOperand.B | DSReadOperand.A_SCALE)),
        ("SCHED_MODE", int(SchedMode.MFMA_16)),
        ("FROZEN_STEP", True),
        ("SOFF_UNROLL", True),
        ("SCALE_FILL_MID", True),
    ],
)
def test_host_and_kernel_config_field_order(config, field, value):
    """The explicit device-side tuple indices must match the host spec exactly."""
    config[field] = value
    func = _func_spec(EpilogueMode.NOP)
    tuning = _tuning_spec(config)
    launch = MoeKernelConfig(
        gl.constexpr(func), gl.constexpr(tuning), gl.constexpr(4096), gl.constexpr(7168)
    )
    host_func = KernelFuncConfig(*func)
    host_tuning = KernelTuningConfig(host_func, *tuning)
    # The builder only constructs constexpr aggregates; its Python body needs no GPU.
    device_func, device_tuning = _build_configs.fn(launch)
    for aggregate in (host_func, device_func):
        for name, expected in func._asdict().items():
            assert _plain(getattr(aggregate, name)) == _plain(expected), name
    for aggregate in (host_tuning, device_tuning):
        for name, expected in tuning._asdict().items():
            assert _plain(getattr(aggregate, name)) == _plain(expected), name


def test_trailing_fields_and_older_config_dict_keep_defaults(config):
    func = FuncSpec(*_func_spec()[:12])
    assert func.epilogue == EpilogueMode.DEFAULT
    old_config = {name: config[name] for name in TuningSpec._fields[:30]}
    tuning = _tuning_spec(old_config)
    assert tuning == TuningSpec(*tuning[:30])
    assert tuning[30:] == (0, 0, False, False, False)
    fc = KernelFuncConfig(*func[:12])
    tc = KernelTuningConfig(fc, *tuning[:30])
    assert _plain(fc.epilogue) == EpilogueMode.DEFAULT
    for name in TuningSpec._fields[30:]:
        assert _plain(getattr(tc, name)) == getattr(tuning, name)
    assert _plain(tc.validate(4096, 7168))


@pytest.mark.parametrize("has_scales", [False, True])
def test_pipeline_state_aggregates_round_trip_separately(has_scales):
    # Opaque handles need no GPU or IR builder. Real Triton tensors still exercise
    # the aggregate types used to flatten and rebuild values across device calls.
    pointer_names = (
        "a_hbm_ptr",
        "b_hbm_ptr",
        "a_scale_hbm_ptr",
        "b_scale_hbm_ptr",
        "a_scale_direct_hbm_ptr",
        "b_scale_direct_hbm_ptr",
    )
    pointer_values = [gl.tensor(object(), gl.pointer_type(gl.uint8)) for _ in range(6)]
    if not has_scales:
        pointer_values[2:] = [None] * 4
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
    config["DS_READ_IN_MFMA"] = mask
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
    assert _plain(tc.validate(4096, 7168))


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
    assert _tuning_spec(config).DS_READ_IN_MFMA == expected


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
    assert _tuning_spec(config)[31:] == (SchedMode.MFMA_8, True, True, True)


@pytest.mark.parametrize("field", ["MANUAL_PP", "B_IN_REG"])
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


@pytest.mark.parametrize("epilogue", list(EpilogueMode))
def test_epilogue_modes_preserve_output_geometry(config, epilogue):
    func = KernelFuncConfig(*_func_spec(epilogue))
    tc = KernelTuningConfig(func, *_tuning_spec(config))
    assert _plain(func.activation_reduction_n()) == 2
    assert _plain(func.mini_n_reduction()) == 1
    assert _plain(tc.grid_N(4096)) == 16
    assert _plain(tc.validate(4096, 7168))


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
    assert _plain(default[1].func).epilogue == EpilogueMode.DEFAULT
    assert _plain(nop[1].func).epilogue == EpilogueMode.NOP


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("DS_READ_IN_MFMA", -1, "unknown operand bits"),
        ("DS_READ_IN_MFMA", 16, "unknown operand bits"),
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
        tc.validate(4096, 7168)
