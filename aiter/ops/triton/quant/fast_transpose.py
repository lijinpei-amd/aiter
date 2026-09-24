# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fast_transpose import _transpose_2d_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = ["fast_transpose_2d"]

_LOGGER = AiterTritonLogger()


@cache
def _intj_transpose_launcher(device: int):
    from intj import make_launcher

    with torch.cuda.device(device):
        return make_launcher(
            _transpose_2d_kernel,
            grid_arg=1,
            bind_device=True,
            options={"num_warps": 1, "waves_per_eu": 2, "num_stages": 2},
        ).bind_device(device)


def fast_transpose_2d(x: torch.Tensor) -> torch.Tensor:
    """Transpose a contiguous 2D tensor using a Triton tiled kernel.

    Returns a contiguous (N, M) tensor from a (M, N) input.
    Works with any dtype including FP8 (e4m3, e5m2, fnuz variants).
    Replaces the ``tensor.t().contiguous()`` pattern which dispatches a
    full ``aten::copy_`` kernel.
    """
    _LOGGER.info("FAST_TRANSPOSE_2D: x=%s", tuple(x.shape))
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, N = x.shape

    out = torch.empty((N, M), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 32
    grid_x = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)

    # num_warps=1: 32×32=1024-element tile is small; benchmarks on MI308X show
    # nw=1/2 tie for best, nw≥4 regresses (nw=16 is 2.5× slower than nw=1).
    with torch.cuda.device(x.device):
        _intj_transpose_launcher(x.get_device())(
            torch.cuda.current_stream().cuda_stream,
            grid_x,
            x,
            out,
            M,
            N,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M,
            BLOCK_N,
        )
    return out
