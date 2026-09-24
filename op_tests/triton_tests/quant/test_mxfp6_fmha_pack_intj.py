# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the FP6 packers without relying on Triton's bracket launch path."""

import pytest
import torch
from triton.runtime.jit import JITFunction

from aiter.ops.triton.quant.mxfp6_fmha_pack import (
    quantize_fp6_k_lds_order_torch,
    quantize_fp6_k_lds_order_triton,
    quantize_fp6_v_clean_triton,
    quantize_fp6_v_data_scale_triton,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype


@pytest.fixture(autouse=True)
def _reject_triton_bracket_launch(monkeypatch):
    def reject(self, grid):
        raise AssertionError("Triton bracket launch was used")

    monkeypatch.setattr(JITFunction, "__getitem__", reject)


def test_fp6_k_pack_matches_torch_reference():
    torch.manual_seed(0)
    x = torch.randn((1, 128, 1, 128), dtype=torch.bfloat16, device="cuda")
    packed, scales = quantize_fp6_k_lds_order_triton(x, return_raw=True)
    ref_packed, ref_scales = quantize_fp6_k_lds_order_torch(x, return_raw=True)

    # The 4096-byte staging hole is intentionally uninitialized by Triton.
    torch.testing.assert_close(packed[:12288], ref_packed[:12288], atol=0, rtol=0)
    torch.testing.assert_close(packed[16384:17408], ref_packed[16384:17408], atol=0, rtol=0)
    torch.testing.assert_close(scales[:512], ref_scales[:512], atol=0, rtol=0)


def test_fp6_v_pack_zero_scale_in_both_layouts():
    v = torch.zeros((1, 128, 1, 128), dtype=get_fp8_e4m3_dtype(), device="cuda")
    combined = quantize_fp6_v_clean_triton(v, direct_p=True)
    data, scales = quantize_fp6_v_data_scale_triton(v)

    torch.testing.assert_close(combined.flatten()[:12288], torch.zeros_like(data[:12288]))
    torch.testing.assert_close(data[:12288], torch.zeros_like(data[:12288]))
    torch.testing.assert_close(combined.flatten()[12288:12800], torch.full_like(scales, 127))
    torch.testing.assert_close(scales, torch.full_like(scales, 127))
