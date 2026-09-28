# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Replaces the eager two-kernel x * gate.sigmoid() (one sigmoid pass writing
a temporary, one multiply pass reading it back) with a single pass
"""

import torch
import triton
from intj import make_launcher

from aiter.ops.triton._triton_kernels.fusions.fused_sigmoid_mul import (
    _fused_sigmoid_mul_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
)
from aiter.ops.triton.utils.device_info import current_device_stream
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["fused_sigmoid_mul"]


def _get_config() -> dict:
    base = f"{AITER_TRITON_CONFIGS_PATH}/{get_arch()}/triton/fusions/fused_sigmoid_mul"
    return dict(load_config_json(f"{base}/DEFAULT.json", required=True)["any"])


_fused_sigmoid_mul_kernel_launch = make_launcher(
    _fused_sigmoid_mul_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def fused_sigmoid_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused elementwise ``out = x * sigmoid(gate)``.

    Args:
        x: any shape, contiguous
        gate: same shape and dtype as x, contiguous.
        out: optional destination

    Returns:
        out if given, else x (in place).

    Constraints:
        x and gate must be contiguous, same shape, same dtype
    """
    _LOGGER.info("FUSED_SIGMOID_MUL: x=%s dtype=%s", tuple(x.shape), x.dtype)

    assert x.is_cuda, "x must be a CUDA tensor"
    assert gate.device == x.device, "x and gate must be on the same device"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"unsupported dtype: {x.dtype}"
    assert x.shape == gate.shape, f"shape mismatch: {x.shape} vs {gate.shape}"
    assert x.dtype == gate.dtype, f"dtype mismatch: {x.dtype} vs {gate.dtype}"
    assert x.is_contiguous(), "x must be contiguous"
    assert gate.is_contiguous(), "gate must be contiguous"

    if out is None:
        out = x
    else:
        assert out.shape == x.shape, f"out shape mismatch: {out.shape} vs {x.shape}"
        assert out.dtype == x.dtype, f"out dtype mismatch: {out.dtype} vs {x.dtype}"
        assert out.device == x.device, "out must be on the same device as x"
        assert out.is_contiguous(), "out must be contiguous"

    N = x.numel()
    if N == 0:
        return out

    config = _get_config()
    BLOCK_SIZE_N = config.pop("BLOCK_SIZE_N")

    dev, stream = current_device_stream()
    _fused_sigmoid_mul_kernel_launch(
        dev,
        stream,
        (triton.cdiv(N, BLOCK_SIZE_N),),
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        gate,
        out,
        N,
        BLOCK_SIZE_N,
        N % BLOCK_SIZE_N != 0,
    )
    return out
