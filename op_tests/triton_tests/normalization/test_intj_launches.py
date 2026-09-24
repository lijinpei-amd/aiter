# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from triton.runtime.jit import JITFunction

from aiter.ops.triton.normalization.norm import layer_norm
from aiter.ops.triton.normalization.rmsnorm import rms_norm


def test_norms_launch_without_triton_brackets(monkeypatch):
    def reject_triton_launch(*_args, **_kwargs):
        raise AssertionError("normalization used Triton's bracket launcher")

    monkeypatch.setattr(JITFunction, "__getitem__", reject_triton_launch)

    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    weight = torch.randn(128, device="cuda", requires_grad=True)
    bias = torch.randn(128, device="cuda", requires_grad=True)

    y = layer_norm(x, weight, bias)
    torch.testing.assert_close(y, F.layer_norm(x, (128,), weight, bias))
    y.sum().backward()
    assert x.grad is not None and weight.grad is not None and bias.grad is not None

    x = torch.randn(4, 128, device="cuda", requires_grad=True)
    weight = torch.randn(128, device="cuda", requires_grad=True)
    y = rms_norm(x, weight, 1e-5)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weight
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    assert x.grad is not None and weight.grad is not None
