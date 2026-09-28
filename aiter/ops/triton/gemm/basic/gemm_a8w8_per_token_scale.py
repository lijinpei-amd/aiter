# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from intj import Constexpr, make_launcher

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_per_token_scale import (
    _gemm_a8w8_per_token_scale_kernel,
    _get_config,
)
from aiter.ops.triton.utils.device_info import current_device_stream

_gemm_splitk_reduce_kernel_launch = make_launcher(
    _gemm_splitk_reduce_kernel,
    extra_annotation={
        "KERNEL_NAME": Constexpr(value="_gemm_a8w8_per_token_scale_reduce_kernel"),
        "activation": Constexpr(value=""),
    },
)


def gemm_a8w8_per_token_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config=None,
):
    """
    Computes 8 bit matrix multiplication Y = X @ W^T using per-token quantization scales.
    Each token (row) in x and each output column in w has independent scale factors.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Per-token scale for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Per-output-channel scale for w with shape (N, 1) or (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    M, K = x.shape
    N, K = w.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w = w.T
    w_scale = w_scale.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config, _ = _get_config(M, N, K)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    grid = (
        (
            config["NUM_KSPLIT"]
            * triton.cdiv(M, config["BLOCK_SIZE_M"])
            * triton.cdiv(N, config["BLOCK_SIZE_N"])
        ),
    )
    dev, stream = current_device_stream()
    _gemm_a8w8_per_token_scale_kernel(
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
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_KSPLIT"],
        config["SPLITK_BLOCK_SIZE"],
        config["cache_modifier"],
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        dev, stream = current_device_stream()
        _gemm_splitk_reduce_kernel_launch(
            dev,
            stream,
            grid_reduce,
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            False,  # ADD_BIAS
            False,  # use_activation
        )

    return y
