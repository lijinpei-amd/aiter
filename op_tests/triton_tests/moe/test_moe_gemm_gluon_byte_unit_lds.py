# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The byte-unit LDS layout keeps every ds_read_b128 of a 16-wide MFMA conflict-free."""

import pytest

from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import (
    KernelLayout,
    byte_unit_lds_layout,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant
from aiter.ops.triton.moe import moe_op_gemm_gluon as host

# Triton's ds_read_b128 model: each group of 16 lanes reads 16 x 16 B across the 64
# four-byte banks in one cycle.
_HALF = [[*range(0, 4), *range(12, 16), *range(20, 24), *range(24, 28)]]
_HALF.append([*range(4, 8), *range(8, 12), *range(16, 20), *range(28, 32)])
_GROUPS = _HALF + [[lane + 32 for lane in g] for g in _HALF]

_QUANT = {"bf16": (DtypeQuant.BF16, 2), "mxfp8": (DtypeQuant.MXFP8, 1)}


def _byte_addresses(layout, shape, k_dim, itemsize):
    """``{(non-K row, K byte): LDS byte address}`` of a padded shared layout."""
    bases = [list(b) for b in layout.offset_bases]
    pairs = [list(p) for p in layout.interval_padding_pairs]
    out = {}
    for offset in range(shape[0] * shape[1]):
        coord = [0, 0]
        for bit, base in enumerate(bases):
            if offset >> bit & 1:
                coord = [coord[0] ^ base[0], coord[1] ^ base[1]]
        padded = offset + sum((offset // interval) * pad for interval, pad in pairs)
        for byte in range(itemsize):
            out[(coord[1 - k_dim], coord[k_dim] * itemsize + byte)] = (
                padded * itemsize + byte
            )
    return out


def _worst_read(addresses, rows, k_bytes, width):
    """Largest bank multiplicity over the ds_read_b128 reads of ``width``-wide MFMAs.

    Lane ``l`` reads 16 B of row ``l % width`` at K byte ``16 * (l // width)``; one
    read covers ``64 // width * 16`` bytes of K.
    """
    worst = 1
    for r0 in range(0, rows, width):
        for k0 in range(0, k_bytes, 64 // width * 16):
            for group in _GROUPS:
                banks = []
                for lane in group:
                    start = addresses[(r0 + lane % width, k0 + 16 * (lane // width))]
                    banks += [(start // 4 + word) % 64 for word in range(4)]
                worst = max(worst, max(banks.count(b) for b in set(banks)))
    return worst


@pytest.mark.parametrize(
    "dtype, block_m, expected",
    [
        ("bf16", 16, (False, True)),
        ("bf16", 32, (True, True)),
        ("bf16", 64, (True, True)),
        ("bf16", 128, (True, True)),
        # 8-bit keeps tiles_per_warp >= 2: block_m 32 has (1, 1), 64 has (1, 2).
        ("mxfp8", 32, (False, False)),
        ("mxfp8", 64, (False, True)),
        ("mxfp8", 128, (True, True)),
    ],
)
def test_production_payloads_read_conflict_free(dtype, block_m, expected):
    quant, itemsize = _QUANT[dtype]
    config = host._default_launch_config(
        block_m, 4096, 7168, quant, quant, False, True, True
    )
    layout = KernelLayout(host._probe_tuning_config(config, quant, quant))
    ok = tuple(bool(maybe_unwrap(layout.payload_byte_unit_lds_ok(i))) for i in (0, 1))
    assert ok == expected
    for idx in (0, 1):
        if not ok[idx]:
            continue
        shape = [int(x) for x in maybe_unwrap(layout.lds_shape_slot(idx, False))]
        k_dim = 1 - idx
        lds = maybe_unwrap(layout.lds_layout(idx, False))
        addresses = _byte_addresses(lds, shape, k_dim, itemsize)
        rows, k_bytes = shape[idx], shape[k_dim] * itemsize
        assert _worst_read(addresses, rows, k_bytes, 16) == 1, (dtype, block_m, idx)


@pytest.mark.parametrize(
    "k_bytes, width", [(32, 32), (64, 16), (64, 32), (128, 16), (128, 32)]
)
def test_preshuffled_reads_conflict_free_below_128_bytes(k_bytes, width):
    # A preshuffled slot keeps the 16 x 16 B HBM order at any K extent, so one narrower
    # than the 128 B non-preshuffled unit reads like the wider ones.
    shape = [k_bytes // 2, 64]
    lds = byte_unit_lds_layout(shape, 1, 16, 16, True)
    addresses = _byte_addresses(lds, shape, 0, 2)
    assert _worst_read(addresses, 64, k_bytes, width) == 1


def test_32_wide_reads_are_excluded():
    # The 32-row unit is 2-way for a 32-wide MFMA (lane rows 20-23 share banks with
    # rows 0-3), which is why a non-preshuffled operand needs a 16-wide one.
    shape = [64, 128]
    lds = byte_unit_lds_layout(shape, 0, 16, 32, False)
    addresses = _byte_addresses(lds, shape, 1, 2)
    assert _worst_read(addresses, 64, 256, 16) == 1
    assert _worst_read(addresses, 64, 256, 32) == 2
    config = dict(
        host._default_launch_config(64, 4096, 7168, *[DtypeQuant.BF16] * 2, False, True),
        mfma_instr_shape=(32, 32, 16),
        SLOT_N=128,
    )
    tc = host._probe_tuning_config(config, DtypeQuant.BF16, DtypeQuant.BF16)
    layout = KernelLayout(tc)
    assert not any(maybe_unwrap(layout.payload_byte_unit_lds_ok(i)) for i in (0, 1))
