# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import warnings

import torch
import triton
from intj import make_launcher

from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.gemm.feed_forward.ff_a16w16_fused_gated import (
    _ff_a16w16_fused_gated,
    _get_config,
)
from aiter.ops.triton.utils.device_info import current_device_stream
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


_ff_a16w16_fused_gated_launch = make_launcher(
    _ff_a16w16_fused_gated,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def ff_a16w16_fused_gated(
    x,
    w_up,
    w_down,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    activation: str | None = None,
):
    """
    Computes a full feed-forward operation with a gated activation (e.g FF with SwiGLU)
    Uses the first half of the output (along the N dim) as a gate for the second half.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - w_up: Up-projection W with shape (N, K).
    - w_down: Down-projection W with shape (N//2, K).
    - dtype: Optional parameter to specify bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, K).
    If this is none, then it's created by this API and returned as output.
    - activation: Optional activation function to apply to the gating activations.
    One of ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu", None)

    Returns:
    - Y: The output matrix with shape (M, K).
    """

    _LOGGER.info(
        "FF_A16W16_FUSED_GATED: x=%s w_up=%s w_down=%s",
        tuple(x.shape),
        tuple(w_up.shape),
        tuple(w_down.shape),
    )

    # Shape checks
    assert (
        x.shape[1] == w_up.shape[1] == w_down.shape[1]
    ), f"Incompatible matrix shapes: x:{x.shape}, w_up:{w_up.shape}, w_down:{w_down.shape}"
    assert (
        w_up.shape[0] == w_down.shape[0] * 2
    ), f"Incompatible matrix shapes: w_up:{w_up.shape}, w_down:{w_down.shape}"

    N, K = w_up.shape
    M = x.shape[0]
    if M > 64:
        warnings.warn(
            "The fused FF kernel is slower than the unfused equivalent for large batch sizes (>64)."
        )

    assert N % 2 == 0, "Weight shape incompatible with gating (N not divisible by 2)"

    w_up = w_up.T

    if y is None:
        y = torch.zeros(
            (M, K), dtype=dtype, device=x.device
        )  # zeros, as this does atomic adds on top

    if config is None:
        config, _ = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    dev, stream = current_device_stream()
    _ff_a16w16_fused_gated_launch(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        w_up,
        w_down,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_up.stride(0),
        w_up.stride(1),
        w_down.stride(0),
        w_down.stride(1),
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
