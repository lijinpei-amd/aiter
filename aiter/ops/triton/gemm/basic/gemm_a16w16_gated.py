# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16_gated import (
    _gemm_a16_w16_gated_kernel,
    _get_config,
)
from aiter.ops.triton.utils.device_info import current_device_stream
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a16w16_gated(
    x,
    w,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    activation: str | None = None,
):
    """
    Computes 16 bit gated matrix multiplication Y = X @ W^T with gating mechanism (e.g., SwiGLU).
    Uses first half of W output as gate for second half, producing (M, N//2) output.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed. N must be even.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N//2).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).
        activation (Optional[str]): Activation function applied to gate ("gelu", "gelu_tanh",
            "silu", "silu_exp2", "relu").

    Returns:
        torch.Tensor: Gated output with shape (M, N//2).
    """
    _LOGGER.info("GEMM_A16W16_GATED: x=%s w=%s", tuple(x.shape), tuple(w.shape))

    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    M, K = x.shape
    N, K = w.shape

    assert N % 2 == 0, "Weight shape incompatible with gating (N not divisible by 2)"

    w = w.T

    if y is None:
        y = torch.empty((M, N // 2), dtype=dtype, device=x.device)

    if config is None:
        config, _ = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    dev, stream = current_device_stream()
    _gemm_a16_w16_gated_kernel(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["cache_modifier"],
        _get_activation_from_str(activation) if activation else "",
        activation is not None,
    )

    return y
