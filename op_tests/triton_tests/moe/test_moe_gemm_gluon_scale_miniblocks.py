# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host contracts for independent shuffled-scale load geometry."""

import math

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from op_tests.triton_tests.moe.test_moe_gemm_gluon_scale_layout import _config


@pytest.mark.parametrize("axis,value", [("M", 128), ("N", 128), ("K", 512)])
def test_scale_mini_block_config_and_environment_aliases(monkeypatch, axis, value):
    field = f"SCALE_MINI_BLOCK_{axis}"
    for name in tuple(host.os.environ):
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    config = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    config[field.lower()] = value
    assert dict(zip(host.TuningSpec._fields, host._tuning_args(config)))[field] == value
    config[field] = value
    assert dict(zip(host.TuningSpec._fields, host._tuning_args(config)))[field] == value
    config[field] = value * 2
    with pytest.raises(ValueError, match="disagree"):
        host._tuning_args(config)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_" + field, str(value))
    from_environment = host.get_gluon_config_uncached(
        128, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4
    )
    assert (
        dict(zip(host.TuningSpec._fields, host._tuning_args(from_environment)))[field]
        == value
    )


@pytest.mark.parametrize(
    "shuffled", [(False, False), (True, False), (False, True), (True, True)]
)
def test_scale_mini_blocks_only_change_the_shuffled_operand(shuffled):
    tc = _config(
        A_SCALE_SORTED_SHUFFLED=shuffled[0],
        B_SCALE_SHUFFLED=shuffled[1],
        SCALE_MINI_BLOCK_M=128,
        SCALE_MINI_BLOCK_N=128,
        SCALE_MINI_BLOCK_K=512,
    )
    for operand in range(2):
        assert tc.scale_mini_block_nonk(operand) == (128 if shuffled[operand] else 64)
        assert tc.scale_mini_block_k(operand) == (512 if shuffled[operand] else 128)
        assert tc.scale_step_ratio(operand) == (4 if shuffled[operand] else 1)
        assert tc.scale_tile_ratio(operand) == (2 if shuffled[operand] else 1)
    assert tc.validate(4096, 7168)


def test_invalid_unused_scale_minis_are_ignored():
    tc = _config(
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        SCALE_MINI_BLOCK_M=-1,
        SCALE_MINI_BLOCK_N=3,
        SCALE_MINI_BLOCK_K=1,
    )
    assert tc.validate(4096, 7168)
    assert tc.scale_mini_block_nonk(0) == tc.scale_mini_block_nonk(1) == 64
    assert tc.scale_mini_block_k(0) == tc.scale_mini_block_k(1) == 128


@pytest.mark.parametrize(
    "field,value",
    [
        ("SCALE_MINI_BLOCK_M", 32),
        ("SCALE_MINI_BLOCK_M", 256),
        ("SCALE_MINI_BLOCK_N", 32),
        ("SCALE_MINI_BLOCK_N", 256),
        ("SCALE_MINI_BLOCK_K", 128),
        ("SCALE_MINI_BLOCK_K", 384),
    ],
)
def test_invalid_shuffled_scale_minis_are_rejected(field, value):
    with pytest.raises(AssertionError):
        _config(**{field: value}).validate(4096, 7168)


@pytest.mark.parametrize(
    "block_k,mini_k,scale_k",
    [(128, 128, 256), (128, 128, 512), (256, 128, 512), (512, 128, 256)],
)
@pytest.mark.parametrize("register_scales", [False, True])
def test_scale_load_cadence_sets_buffer_span_and_unroll(
    block_k, mini_k, scale_k, register_scales
):
    tc = _config(
        BLOCK_K=block_k,
        MINI_BLOCK_K=mini_k,
        SCALE_MINI_BLOCK_K=scale_k,
        VGPR_PREFETCH_K=block_k,
        K_UNROLL=3,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=3,
        A_SCALE_IN_REG=register_scales,
        B_SCALE_IN_REG=register_scales,
    )
    ratio = math.ceil(scale_k / block_k)
    for operand, depth in enumerate((2, 3)):
        assert tc.component_span(operand, True) == (depth - 1) * ratio + 1
        assert tc.scale_load_k_tiles(operand) == max(1, block_k // scale_k)
        assert tc.scale_read_k_slots(operand) == max(block_k, scale_k) // mini_k
    period = math.lcm(2 * ratio, 3 * ratio) if register_scales else ratio
    assert tc.pipeline_unroll() == math.ceil(3 / period) * period
