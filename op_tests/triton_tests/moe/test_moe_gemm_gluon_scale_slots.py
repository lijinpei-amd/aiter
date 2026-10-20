# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host contracts for independent shuffled-scale load geometry."""

import math

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe._frozen import _validate_frozen
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import Schedule
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import Component, DtypeQuant
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import validate
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from op_tests.triton_tests.moe.test_moe_gemm_gluon_scale_layout import _config


@pytest.mark.parametrize("axis,value", [("M", 128), ("N", 128), ("K", 16)])
def test_scale_slot_config_and_environment_aliases(monkeypatch, axis, value):
    field = f"SCALE_SLOT_{axis}"
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
def test_scale_slots_only_change_the_shuffled_operand(shuffled):
    tc = _config(
        A_SCALE_SORTED_SHUFFLED=shuffled[0],
        B_SCALE_SHUFFLED=shuffled[1],
        SCALE_SLOT_M=128,
        SCALE_SLOT_N=128,
        SCALE_SLOT_K=16,
    )
    layout = KernelLayout(tc)
    for operand in range(2):
        assert layout.scale_slot_nonk(operand) == (
            128 if shuffled[operand] else 64
        )
        assert layout.scale_slot_k(operand) == (16 if shuffled[operand] else 4)
    assert validate(tc, 4096, 7168)


@pytest.mark.parametrize("block_k", [128, 256, 512])
def test_shuffled_scale_k_defaults_to_at_least_k256(block_k):
    tc = _config(BLOCK_K=block_k, VGPR_PREFETCH_K=block_k)
    layout = KernelLayout(tc)
    expected = max(8, block_k // 32)
    assert layout.scale_slot_k(0) == layout.scale_slot_k(1) == expected


def test_invalid_unused_scale_slots_are_ignored():
    tc = _config(
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        SCALE_SLOT_M=-1,
        SCALE_SLOT_N=3,
        SCALE_SLOT_K=1,
    )
    assert validate(tc, 4096, 7168)
    layout = KernelLayout(tc)
    assert layout.scale_slot_nonk(0) == layout.scale_slot_nonk(1) == 64
    assert layout.scale_slot_k(0) == layout.scale_slot_k(1) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("SCALE_SLOT_M", 32),
        ("SCALE_SLOT_M", 256),
        ("SCALE_SLOT_N", 32),
        ("SCALE_SLOT_N", 256),
        ("SCALE_SLOT_K", 4),
        ("SCALE_SLOT_K", 12),
    ],
)
def test_invalid_shuffled_scale_slots_are_rejected(field, value):
    with pytest.raises(AssertionError):
        validate(_config(**{field: value}), 4096, 7168)


@pytest.mark.parametrize("K", [7168 + 128, 3584 + 128, 1792 + 128])
def test_shuffled_scale_k_must_divide_total_k(K):
    """One scale load has to cover whole payload K steps *and* tile K exactly.

    The other three `S_K` rules (at least `max(8, BLOCK_K / 32)`, a power of two, a
    multiple of `BLOCK_K / 32`) are shape-local and already covered. This one couples
    the scale geometry to the runtime K extent: a `S_K` that divides `BLOCK_K / 32`
    cleanly but not `K / 32` leaves a short final scale load with no payload steps to
    cover.
    """
    tc = _config(SCALE_SLOT_K=8)
    assert K % 256 != 0, "pick a K the scale extent cannot tile"
    with pytest.raises(
        AssertionError, match="K must contain complete SCALE_SLOT_K tiles"
    ):
        validate(tc, 4096, K)
    # The same configuration is legal as soon as K is a whole number of scale loads.
    validate(tc, 4096, K + 256 - (K % 256))


def test_shuffled_scale_k_cannot_be_narrower_than_payload_k():
    tc = _config(
        BLOCK_K=512,
        SCALE_SLOT_K=8,
        VGPR_PREFETCH_K=512,
    )
    with pytest.raises(AssertionError):
        validate(tc, 4096, 7168)


def test_frozen_shuffled_scales_reject_k_wider_than_256():
    tc = _config(
        BLOCK_K=512,
        SCALE_SLOT_K=16,
        VGPR_PREFETCH_K=512,
        FROZEN_STEP=True,
    )
    with pytest.raises(
        AssertionError, match="BLOCK_K == SCALE_SLOT_K == 256"
    ):
        _validate_frozen(tc, 4096, 7168)


@pytest.mark.parametrize(
    "block_k,scale_k",
    [
        (128, 8),
        (128, 16),
        (256, 8),
        (256, 16),
        (512, 16),
        (512, 32),
    ],
)
@pytest.mark.parametrize("register_scales", [False, True])
def test_scale_k_step_ratio_sets_buffer_span_and_unroll(
    block_k, scale_k, register_scales
):
    tc = _config(
        BLOCK_K=block_k,
        SCALE_SLOT_K=scale_k,
        VGPR_PREFETCH_K=block_k,
        K_UNROLL=3,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=3,
        ALLOW_A_SCALE_VIA_LDS=not register_scales,
        ALLOW_B_SCALE_VIA_LDS=not register_scales,
    )
    layout = KernelLayout(tc)
    schedule = Schedule(tc, layout, block_k)
    ratio = scale_k // (block_k // 32)
    for operand, depth in enumerate((2, 3)):
        live_span = (depth - 1) * ratio + 1
        distance = live_span - 1 if register_scales else live_span
        assert schedule.load_to_mfma_distance(Component(operand, True)) == distance
        assert layout.scale_slot_k(operand) == scale_k
    period = math.lcm(3, 2 * ratio, 3 * ratio)
    assert maybe_unwrap(schedule.unroll) == period
