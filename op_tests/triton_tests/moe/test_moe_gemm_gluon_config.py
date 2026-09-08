# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the host/device configuration contract of the Gluon MoE GEMM."""

import os

import pytest
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe._buffered import (
    _groups as buffered_groups,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._buffered import (
    _ops as buffered_ops,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._buffered import (
    _wait as buffered_wait,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import constexpr_fields
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DSReadOperand,
    DtypeQuant,
    EpilogueMode,
    FuncSpec,
    SchedMode,
    TuningSpec,
    WaitCommitScheme,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe.moe_gemm import (
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


@pytest.mark.parametrize("with_optional_specs", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("B_PRESHUFFLED", True),
        ("DS_READ_IN_MFMA", int(DSReadOperand.B | DSReadOperand.A_SCALE)),
        ("SCHED_MODE", int(SchedMode.MFMA_16)),
        ("FROZEN_STEP", True),
        ("SOFF_UNROLL", True),
        ("SCALE_FILL_MID", True),
        ("B_IN_REG", True),
        ("B_SCALE_IN_REG", True),
        ("A_SCALE_IN_REG", True),
        ("A_NUM_BUFFER", 4),
        ("B_NUM_BUFFER", 2),
        ("A_SCALE_NUM_BUFFER", 1),
        ("B_SCALE_NUM_BUFFER", 6),
    ],
)
def test_launch_specs_match_kernel_config_field_order(
    config, field, value, with_optional_specs
):
    """Host construction and device argument unpacking must preserve every field."""
    config[field] = value
    func = _func_spec(EpilogueMode.NOP)
    if not with_optional_specs:
        func = func._replace(activation=None, output_quant=None)
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


def test_trailing_fields_and_older_config_dict_keep_defaults(config):
    func = FuncSpec(*_func_spec()[:12])
    assert func.epilogue == EpilogueMode.DEFAULT
    old_config = {name: config[name] for name in TuningSpec._fields[:30]}
    tuning = _tuning_spec(old_config)
    assert tuning == TuningSpec(*tuning[:30])
    assert tuning[30:] == (0, 0, False, False, False, False, False, False, 0, 0, 0, 0)
    fc = KernelFuncConfig(*func[:12])
    tc = KernelTuningConfig(fc, *tuning[:30])
    assert _plain(fc.epilogue) == EpilogueMode.DEFAULT
    for name in TuningSpec._fields[30:]:
        assert _plain(getattr(tc, name)) == getattr(tuning, name)
    assert _plain(tc.validate(4096, 7168))


@pytest.mark.parametrize(
    "scheme,groups,stage_head",
    [
        (WaitCommitScheme.PER_OP, 8, False),
        (WaitCommitScheme.PER_SLOT, 4, False),
        (WaitCommitScheme.PER_STAGE_WARP_PIPELINE, 1, True),
        (WaitCommitScheme.PER_STAGE_WHOLE, 1, True),
    ],
)
def test_commit_scheme_counts_real_payload_and_scale_groups(
    config, scheme, groups, stage_head
):
    config["WAIT_COMMIT_SCHEME"] = int(scheme)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    assert _plain(tc.scale_via_lds(0)) and _plain(tc.scale_via_lds(1))
    assert _plain(tc.commit_groups_per_stage()) == groups
    assert _plain(tc.wait_at_stage_head()) is stage_head


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
    assert _tuning_spec(config)[31:35] == (SchedMode.MFMA_8, True, True, True)


@pytest.mark.parametrize("field", ["MANUAL_PP"])
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
        B_IN_REG=bool(storage_mask & 1),
        B_SCALE_IN_REG=bool(storage_mask & 2),
        A_SCALE_IN_REG=bool(storage_mask & 4),
        B_PRESHUFFLED=bool(storage_mask & 1),
        A_NUM_BUFFER=2,
        B_NUM_BUFFER=1,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=1,
    )
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    assert _plain(tc.payload_via_lds(0))
    assert _plain(tc.payload_via_lds(1)) is not bool(storage_mask & 1)
    assert _plain(tc.scale_via_lds(0)) is not bool(storage_mask & 4)
    assert _plain(tc.scale_via_lds(1)) is not bool(storage_mask & 2)
    # Full-block byte counts make a shared scale fill tile and mini-tile splitting
    # irrelevant to the allocation budget. Register storage removes only its own
    # allocation; every remaining component uses its individual ring depth.
    bm, bn, bk = (config[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    expected = bm * bk // 2 * 2
    if not storage_mask & 1:
        expected += bn * bk // 2
    if not storage_mask & 2:
        expected += bn * bk // 32
    if not storage_mask & 4:
        expected += bm * bk // 32 * 2
    assert _plain(tc.lds_bytes()) == expected
    assert _plain(tc.pipeline_unroll()) == 1
    assert _plain(tc.validate(4096, 7168))
    assert _plain(tc.validate(4096, bk))


@pytest.mark.parametrize(
    "counts,expected",
    [((6, 4, 2, 8), 2), ((4, 4, 2, 2), 2), ((0, 4, 0, 6), 1), ((1, 1, 1, 1), 1)],
)
def test_component_buffer_counts_determine_gcd_unroll(config, counts, expected):
    config.update(zip(host._COMPONENT_BUFFER_KEYS, counts, strict=True))
    config["K_UNROLL"] = 7
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    assert _plain(tc.independent_buffers())
    assert _plain(tc.pipeline_unroll()) == expected
    resolved = [count or config["NUM_LDS_BUFFER"] for count in counts]
    assert [
        _plain(tc.num_buffers(operand, scale))
        for scale in (False, True)
        for operand in (0, 1)
    ] == resolved
    assert _plain(tc.pipeline_depth()) == max(resolved)


def test_legacy_unroll_is_preserved_until_independent_storage_is_selected(config):
    config["K_UNROLL"] = 7
    fc = KernelFuncConfig(*_func_spec())
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert not _plain(tc.independent_buffers())
    assert _plain(tc.pipeline_unroll()) == 7
    config["A_SCALE_IN_REG"] = True
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    assert _plain(tc.independent_buffers())
    assert _plain(tc.pipeline_unroll()) == config["NUM_LDS_BUFFER"]


def test_absent_scales_participate_in_gcd_but_not_pipeline_depth(config):
    config.update(
        A_NUM_BUFFER=3, B_NUM_BUFFER=3, A_SCALE_NUM_BUFFER=8, B_SCALE_NUM_BUFFER=10
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
    assert _plain(tc.pipeline_depth()) == 3
    assert _plain(tc.pipeline_unroll()) == 1


@pytest.mark.parametrize("field", host._COMPONENT_BUFFER_KEYS)
def test_negative_component_buffer_counts_are_rejected(config, field):
    config[field] = -1
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match=field + ".*must be positive"):
        tc.validate(4096, 7168)


@pytest.mark.parametrize(
    "field", host._COMPONENT_BUFFER_KEYS + host._REGISTER_STORAGE_KEYS
)
def test_frozen_reference_rejects_new_storage_options(config, field):
    config.update(FROZEN_STEP=True, B_PRESHUFFLED=True)
    config[field] = 1
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="FROZEN_STEP does not support"):
        tc.validate(4096, 7168)


def test_register_weights_require_preshuffle(config):
    config["B_IN_REG"] = True
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(AssertionError, match="B_IN_REG requires B_PRESHUFFLED"):
        tc.validate(4096, 7168)


def test_independent_register_scales_allow_mini_k_slicing(config):
    config.update(MINI_BLOCK_K=128, A_SCALE_IN_REG=True, B_SCALE_IN_REG=True)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    assert _plain(tc.validate(4096, 7168))


def test_independent_buffers_still_require_nonempty_k(config):
    config["A_NUM_BUFFER"] = 1
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    with pytest.raises(
        AssertionError, match="K must contain at least one BLOCK_K tile"
    ):
        tc.validate(4096, 0)


@pytest.mark.parametrize(
    "field", host._COMPONENT_BUFFER_KEYS + host._REGISTER_STORAGE_KEYS
)
def test_storage_environment_settings_reach_tuning_spec(monkeypatch, field):
    value = 1 if field in host._REGISTER_STORAGE_KEYS else 2
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


def test_independent_buffers_allow_odd_gcd_with_packed_k128_scales(config):
    config.update(
        BLOCK_K=128,
        MINI_BLOCK_K=128,
        VGPR_PREFETCH_K=128,
        K_UNROLL=3,
        A_NUM_BUFFER=3,
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
    assert _plain(tc.pipeline_unroll()) == 3
    assert _plain(tc.validate(4096, 7168))
    assert host._scale_shuffle_supported(config, 0)
    assert host._scale_shuffle_supported(config, 1)


@pytest.mark.parametrize("output_quant", [None, int(DtypeQuant.MXFP4)])
def test_gate_up_small_warp_extent_only_requires_mx_group_for_quantized_output(
    config, output_quant
):
    """Boxed None must not impose quantization constraints on FP32 tuning tiles."""
    config.update(
        BLOCK_N=128,
        MINI_BLOCK_N=64,
        warps_per_cta=(1, 4),
        tiles_per_warp=(2, 1),
        B_PRESHUFFLED=True,
        B_IN_REG=True,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
    )
    fc = KernelFuncConfig(*_func_spec()._replace(output_quant=output_quant))
    tc = KernelTuningConfig(fc, *_tuning_spec(config))
    if output_quant is None:
        assert _plain(tc.validate(2048, 7168))
    else:
        with pytest.raises(AssertionError, match="a warp's emitted N extent"):
            tc.validate(2048, 7168)


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
    assert _plain(default[1][0]).epilogue == EpilogueMode.DEFAULT
    assert _plain(nop[1][0]).epilogue == EpilogueMode.NOP


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


def _audit_buffer_schedule(tc, num_k):
    """Compare group waits with a complete chronological copy/read simulation.

    The oracle records absolute K tiles and group sequence numbers. It does not
    use the producer-stage arithmetic or truncated timeline used by _wait.
    Register queues hold symbolic tile numbers, so every read also checks that
    warmup, rotation, and drain deliver exactly the current K tile.
    """
    nm, nn = _plain(tc.num_mini_m()), _plain(tc.num_mini_n())
    nslots = nm * nn
    if (nm, nn) == (2, 2):
        payloads = [(2, 0), (0, 0), (0, 1), (2, 1)]
    else:
        payloads = []
        for tile in range(max(nm, nn)):
            if tile < nm:
                payloads.append((0, tile))
            if tile < nn:
                payloads.append((2, tile))
    depths = [_plain(tc.num_buffers(kind // 2, bool(kind % 2))) for kind in range(4)]
    has_scale = [_plain(tc.func_cfg.has_scale(operand)) for operand in (0, 1)]
    async_kind = [
        (
            _plain(tc.scale_via_lds(kind // 2))
            if kind % 2
            else _plain(tc.payload_via_lds(kind // 2))
        )
        for kind in range(4)
    ]
    shared_a = (
        _plain(tc.scale_tile_ratio_a())
        if async_kind[1] and _plain(tc.scale_shuffled(0))
        else 1
    )
    fill_slots = [[] for _ in range(nslots)]
    for slot, (kind, tile) in enumerate(payloads):
        fill_slots[slot].append((kind, tile))
        if not has_scale[kind // 2] or (kind == 0 and tile % shared_a):
            continue
        scale_slot = (
            tile + 1 if (nm, nn) == (2, 2) and _plain(tc.SCALE_FILL_MID) else slot
        )
        fill_slots[scale_slot].append((kind + 1, tile))
    fill_slots = [tuple(sorted(ops)) for ops in fill_slots]
    for slot, expected in enumerate(fill_slots):
        actual = buffered_ops(tc, slot % nm, slot // nm)
        assert (
            tuple((kind, tile) for kind, tile in enumerate(actual) if tile is not None)
            == expected
        )

    committed, group_for_copy, issued, lds = [], {}, set(), {}
    queues = {
        (kind, tile): [None] * depths[kind]
        for ops in fill_slots
        for kind, tile in ops
        if not async_kind[kind]
    }
    depth = _plain(tc.pipeline_depth())
    single = any(
        depths[kind] == 1 for kind in range(4) if kind % 2 == 0 or has_scale[kind // 2]
    )
    scheme = _plain(tc.WAIT_COMMIT_SCHEME)
    per_stage = scheme in (
        WaitCommitScheme.PER_STAGE_WHOLE,
        WaitCommitScheme.PER_STAGE_WARP_PIPELINE,
    )

    def required_copies(stage, slot):
        need = []
        for pos, (kind, tile) in enumerate(payloads):
            if slot is not None and pos != slot:
                continue
            need.append((kind, tile, stage))
            if has_scale[kind // 2]:
                owner = tile - tile % shared_a if kind == 0 else tile
                need.append((kind + 1, owner, stage))
        return need

    def check_wait(stage, slot):
        required = [
            copy for copy in required_copies(stage, slot) if async_kind[copy[0]]
        ]
        expected = (
            len(committed) - 1 - max(group_for_copy[copy] for copy in required)
            if required
            else None
        )
        assert buffered_wait(tc, stage, num_k, slot) == expected
        if depth - 1 <= stage <= num_k - depth:
            assert buffered_wait(tc, None, num_k, slot) == expected

    for stage in range(1 - depth, num_k):
        pending_stage = []
        expected_groups = []
        stage_copies = [
            tuple(
                (kind, tile, stage + depths[kind] - 1)
                for kind, tile in ops
                if 0 <= stage + depths[kind] - 1 < num_k
            )
            for ops in fill_slots
        ]
        for slot, copies in enumerate(stage_copies):
            asynchronous = tuple(copy for copy in copies if async_kind[copy[0]])
            if scheme == WaitCommitScheme.PER_OP:
                groups = tuple((copy,) for copy in asynchronous)
            elif scheme == WaitCommitScheme.PER_SLOT:
                groups = (asynchronous,)
            else:
                pending_stage.extend(asynchronous)
                groups = (tuple(pending_stage),) if slot == nslots - 1 else ()
            expected_groups.append(groups)
        actual_groups = buffered_groups(tc, stage, num_k)
        assert actual_groups == tuple(
            tuple(tuple(copy[:2] for copy in group) for group in groups)
            for groups in expected_groups
        )

        def fill_slot(
            slot, copies_by_slot=stage_copies, groups_by_slot=expected_groups
        ):
            for kind, tile, target in copies_by_slot[slot]:
                copy = (kind, tile, target)
                assert copy not in issued, "a K tile was loaded twice"
                issued.add(copy)
                if async_kind[kind]:
                    lds[kind, tile, target % depths[kind]] = target
                else:
                    queues[kind, tile][-1] = target
            for group in groups_by_slot[slot]:
                for copy in group:
                    group_for_copy[copy] = len(committed)
                committed.append(group)

        if stage < 0 or single:
            for slot in range(nslots):
                fill_slot(slot)
        if stage >= 0:
            if not single and per_stage:
                check_wait(stage, None)
            for slot in range(nslots):
                if not single and not per_stage:
                    check_wait(stage, slot)
                for kind, tile, target in required_copies(stage, slot):
                    if async_kind[kind]:
                        assert lds[kind, tile, target % depths[kind]] == target
                    else:
                        assert queues[kind, tile][0] == target
                if not single:
                    fill_slot(slot)
        for key, queue in queues.items():
            queues[key] = queue[1:] + queue[:1]
    expected_loads = {
        (kind, tile, target)
        for ops in fill_slots
        for kind, tile in ops
        for target in range(num_k)
    }
    assert issued == expected_loads


@pytest.mark.parametrize("scheme", list(WaitCommitScheme))
@pytest.mark.parametrize("storage_mask", range(8))
@pytest.mark.parametrize(
    "counts", [(3, 3, 3, 3), (2, 4, 3, 2), (4, 2, 2, 4), (1, 3, 2, 1)]
)
@pytest.mark.parametrize("geometry", ["pair", "middle_shared_a", "unequal_tiles"])
def test_independent_buffer_waits_match_copy_history(
    config, monkeypatch, scheme, storage_mask, counts, geometry
):
    from aiter.ops.triton._gluon_kernels.gfx950.moe import _config as config_module

    config.update(zip(host._COMPONENT_BUFFER_KEYS, counts, strict=True))
    config.update(
        WAIT_COMMIT_SCHEME=int(scheme),
        B_IN_REG=bool(storage_mask & 1),
        B_SCALE_IN_REG=bool(storage_mask & 2),
        A_SCALE_IN_REG=bool(storage_mask & 4),
        B_PRESHUFFLED=bool(storage_mask & 1),
    )
    if geometry == "middle_shared_a":
        monkeypatch.setattr(config_module, "_SCALE_MINI_M_ENV", 128)
        config.update(SCALE_FILL_MID=True, A_SCALE_SORTED_SHUFFLED=True)
    elif geometry == "unequal_tiles":
        config.update(MINI_BLOCK_N=64, SCALE_FILL_MID=True)
    tc = KernelTuningConfig(KernelFuncConfig(*_func_spec()), *_tuning_spec(config))
    # One strip shorter than several rings, one with warmup, runtime-loop steady
    # state and drain. The latter also checks the stage=None fast-path wait counts.
    for num_k in (2, 9):
        _audit_buffer_schedule(tc, num_k)


@pytest.mark.parametrize("scheme", list(WaitCommitScheme))
@pytest.mark.parametrize("register_b", [False, True])
@pytest.mark.parametrize("counts", [(3, 2, 1, 1), (1, 1, 1, 1)])
def test_independent_buffer_schedule_without_scales(config, scheme, register_b, counts):
    config.update(zip(host._COMPONENT_BUFFER_KEYS, counts, strict=True))
    config.update(
        WAIT_COMMIT_SCHEME=int(scheme), B_IN_REG=register_b, B_PRESHUFFLED=register_b
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
    for num_k in (1, 2, 9):
        _audit_buffer_schedule(tc, num_k)
