# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The B payload preshuffle is aiter's 16x16 B order for every storage width."""

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from op_tests.triton_tests.moe.test_moe_gemm_gluon_registers import _register_config

# (storage dtype, quant, logical elements per stored element)
_STORAGE = {
    "bf16": (torch.bfloat16, DtypeQuant.BF16, 1),
    "mxfp8": (torch.uint8, DtypeQuant.MXFP8, 1),
    "mxfp4": (torch.uint8, DtypeQuant.MXFP4, 2),
}


def _hbm_offset(n, k, k_storage, itemsize):
    """Storage-element offset of ``(n, k)`` in the 16 x 16 B order."""
    kb, row_bytes = k * itemsize, k_storage * itemsize
    byte = (n // 16) * (row_bytes * 16) + (kb // 16) * 256 + (n % 16) * 16 + kb % 16
    return byte // itemsize


@pytest.mark.parametrize("dtype", list(_STORAGE))
def test_host_preshuffle_matches_shuffle_weight(dtype):
    storage, quant, pack = _STORAGE[dtype]
    experts, n, k = 2, 64, 512
    torch.manual_seed(0)
    w = torch.randint(0, 255, (experts, k // pack, n), dtype=torch.uint8)
    if storage == torch.bfloat16:
        w = torch.randn((experts, k, n)).bfloat16()
    shuffled = host._preshuffled_b(w, {"BLOCK_K": 128}, n, k, quant)
    expected = shuffle_weight(w.transpose(-1, -2).contiguous(), (16, 16))
    assert torch.equal(
        shuffled.contiguous().view(torch.uint8).flatten(),
        expected.contiguous().view(torch.uint8).flatten(),
    )


@pytest.mark.parametrize(
    "dtype, instr, block_k",
    [
        ("mxfp4", (16, 16, 128), None),
        ("mxfp8", (16, 16, 128), None),
        ("bf16", (32, 32, 16), None),
        ("bf16", (16, 16, 32), None),
        # 64 B of K, narrower than the 128 B unit of a non-preshuffled payload.
        ("bf16", (32, 32, 16), 32),
        ("bf16", (16, 16, 32), 32),
    ],
)
def test_preshuffled_b_lds_keeps_hbm_order(dtype, instr, block_k):
    storage, quant, _ = _STORAGE[dtype]
    config = dict(_register_config(dtype), B_PRESHUFFLED=True, mfma_instr_shape=instr)
    if block_k is not None:
        config["BLOCK_K"] = block_k
    layout = KernelLayout(host._probe_tuning_config(config, quant, quant))
    assert maybe_unwrap(layout.payload_byte_unit_lds_ok(1))
    shape = [int(x) for x in maybe_unwrap(layout.lds_shape_slot(1, False))]
    bases = maybe_unwrap(layout.lds_layout(1, False)).offset_bases
    k_storage, itemsize = shape[0], storage.itemsize
    # With K equal to one K step the HBM preshuffle of the slot is one linear run,
    # so the direct-to-LDS copy is linear exactly when LDS keeps the HBM order.
    for offset in range(shape[0] * shape[1]):
        k = n = 0
        for bit, (bk, bn) in enumerate(bases):
            if offset >> bit & 1:
                k ^= bk
                n ^= bn
        assert _hbm_offset(n, k, k_storage, itemsize) == offset, (offset, k, n)
