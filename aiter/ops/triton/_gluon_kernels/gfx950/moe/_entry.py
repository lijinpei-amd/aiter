# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel entry points and launch metadata for the gfx950 Gluon MoE GEMMs."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import constexpr_fields
from ._lang import unwrap_attr as _cv
from ._types import QuantExpertTensor, QuantTokenTensor, ResultTensor, RoutingMeta
from .moe_gemm import _moe_gemm_body

__all__ = [
    "_moe_gluon_gemm1",
    "_moe_gluon_gemm2",
    "moe_gemm_launch_metadata",
]


def moe_gemm_launch_metadata(grid, kernel, args):
    # Reads the flat parameter names of _moe_gluon_gemm1 / _moe_gluon_gemm2 -- the
    # aggregates only exist once the kernel has reassembled them, which is after this
    # runs. Drives profiler display only, not codegen.
    ret = {}
    N = _cv(args["CFG_N"])
    K = _cv(args["CFG_K"])
    hist = args["rt_expt_hist"]
    n_tokens = float(hist.sum()) if hist is not None else None
    w = args["b_ptr"]
    n_w_bytes = (
        (w.numel() * w.element_size() // hist.numel()) * (hist > 0).sum()
        if hist is not None
        else w.numel() * w.element_size()
    )
    ret["name"] = f"{kernel.name} [M={n_tokens}, N={N}, K={K}]"
    if n_tokens is not None:
        ret["flops32"] = 2.0 * n_tokens * N * K
        y = args["res_ptr"]
        x = args["a_ptr"]
        ret["bytes"] = int(
            n_tokens * x.shape[-1] * x.element_size()
            + n_tokens * y.shape[-1] * y.element_size()
            + n_w_bytes
        )
    return ret


@gluon.jit(
    launch_metadata=moe_gemm_launch_metadata,
    do_not_specialize=["grid_m", "grid_n"],
)
def _moe_gluon_gemm1(
    a_ptr,
    a_scale_ptr,
    a_num_token,
    a_stride_m,
    a_scale_stride_m,
    b_ptr,
    b_scale_ptr,
    b_stride_e,
    b_stride_n,
    b_scale_stride_e,
    b_num_expert,
    res_ptr,
    res_scale_ptr,
    res_stride_m,
    res_scale_stride_m,
    res_scale_stride_n,
    rt_expt_block_pid_map,
    rt_expt_hist,
    rt_expt_offs_raw,
    rt_expt_offs_sum,
    rt_gather_indx,
    rt_scatter_indx,
    rt_gammas,
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    A_DTYPE_QUANT: gl.constexpr,
    A_HIDDEN_DIM: gl.constexpr,
    A_TOPK: gl.constexpr,
    A_SCALE_STRIDE_K: gl.constexpr,
    A_SCALE_SWIZZLE: gl.constexpr,
    B_DTYPE_QUANT: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    B_SCALE_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_K: gl.constexpr,
    B_HIDDEN_DIM: gl.constexpr,
    B_FUSED_INTERMEDIATE_DIM: gl.constexpr,
    B_SCALE_SWIZZLE: gl.constexpr,
    RES_DTYPE_QUANT: gl.constexpr,
    RES_STRIDE_N: gl.constexpr,
    RES_OUT_DIM: gl.constexpr,
    RT_N_EXPTS_ACT: gl.constexpr,
    CFG_FUNC: gl.constexpr,
    CFG_TUNING: gl.constexpr,
    CFG_N: gl.constexpr,
    CFG_K: gl.constexpr,
):
    """gemm1: gather + X @ W1 + bias + swiglu, optionally fused MXFP4 output quant."""
    func_cfg = KernelFuncConfig(*constexpr_fields(CFG_FUNC))
    tuning_cfg = KernelTuningConfig(func_cfg, *constexpr_fields(CFG_TUNING))
    _moe_gemm_body(
        QuantTokenTensor(
            A_DTYPE_QUANT,
            a_ptr,
            a_scale_ptr,
            a_num_token,
            a_stride_m,
            a_scale_stride_m,
            A_SCALE_STRIDE_K,
            A_HIDDEN_DIM,
            A_TOPK,
            A_SCALE_SWIZZLE,
        ),
        QuantExpertTensor(
            B_DTYPE_QUANT,
            b_ptr,
            b_scale_ptr,
            b_stride_e,
            B_STRIDE_K,
            b_stride_n,
            b_scale_stride_e,
            B_SCALE_STRIDE_N,
            B_SCALE_STRIDE_K,
            b_num_expert,
            B_HIDDEN_DIM,
            B_FUSED_INTERMEDIATE_DIM,
            B_SCALE_SWIZZLE,
        ),
        ResultTensor(
            RES_DTYPE_QUANT,
            res_ptr,
            res_scale_ptr,
            res_stride_m,
            RES_STRIDE_N,
            res_scale_stride_m,
            res_scale_stride_n,
            RES_OUT_DIM,
        ),
        RoutingMeta(
            rt_expt_block_pid_map,
            rt_expt_hist,
            rt_expt_offs_raw,
            rt_expt_offs_sum,
            rt_gather_indx,
            rt_scatter_indx,
            rt_gammas,
            RT_N_EXPTS_ACT,
        ),
        bias_ptr,
        stride_bias_e,
        x_static_scale_ptr,
        grid_m,
        grid_n,
        func_cfg,
        tuning_cfg,
        CFG_N,
        CFG_K,
    )


@gluon.jit(
    launch_metadata=moe_gemm_launch_metadata,
    do_not_specialize=["grid_m", "grid_n"],
)
def _moe_gluon_gemm2(
    a_ptr,
    a_scale_ptr,
    a_num_token,
    a_stride_m,
    a_scale_stride_m,
    b_ptr,
    b_scale_ptr,
    b_stride_e,
    b_stride_n,
    b_scale_stride_e,
    b_num_expert,
    res_ptr,
    res_scale_ptr,
    res_stride_m,
    res_scale_stride_m,
    res_scale_stride_n,
    rt_expt_block_pid_map,
    rt_expt_hist,
    rt_expt_offs_raw,
    rt_expt_offs_sum,
    rt_gather_indx,
    rt_scatter_indx,
    rt_gammas,
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    A_DTYPE_QUANT: gl.constexpr,
    A_HIDDEN_DIM: gl.constexpr,
    A_TOPK: gl.constexpr,
    A_SCALE_STRIDE_K: gl.constexpr,
    A_SCALE_SWIZZLE: gl.constexpr,
    B_DTYPE_QUANT: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    B_SCALE_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_K: gl.constexpr,
    B_HIDDEN_DIM: gl.constexpr,
    B_FUSED_INTERMEDIATE_DIM: gl.constexpr,
    B_SCALE_SWIZZLE: gl.constexpr,
    RES_DTYPE_QUANT: gl.constexpr,
    RES_STRIDE_N: gl.constexpr,
    RES_OUT_DIM: gl.constexpr,
    RT_N_EXPTS_ACT: gl.constexpr,
    CFG_FUNC: gl.constexpr,
    CFG_TUNING: gl.constexpr,
    CFG_N: gl.constexpr,
    CFG_K: gl.constexpr,
):
    """gemm2: X @ W2 + bias, multiplied by the router combine weight (gammas)."""
    func_cfg = KernelFuncConfig(*constexpr_fields(CFG_FUNC))
    tuning_cfg = KernelTuningConfig(func_cfg, *constexpr_fields(CFG_TUNING))
    _moe_gemm_body(
        QuantTokenTensor(
            A_DTYPE_QUANT,
            a_ptr,
            a_scale_ptr,
            a_num_token,
            a_stride_m,
            a_scale_stride_m,
            A_SCALE_STRIDE_K,
            A_HIDDEN_DIM,
            A_TOPK,
            A_SCALE_SWIZZLE,
        ),
        QuantExpertTensor(
            B_DTYPE_QUANT,
            b_ptr,
            b_scale_ptr,
            b_stride_e,
            B_STRIDE_K,
            b_stride_n,
            b_scale_stride_e,
            B_SCALE_STRIDE_N,
            B_SCALE_STRIDE_K,
            b_num_expert,
            B_HIDDEN_DIM,
            B_FUSED_INTERMEDIATE_DIM,
            B_SCALE_SWIZZLE,
        ),
        ResultTensor(
            RES_DTYPE_QUANT,
            res_ptr,
            res_scale_ptr,
            res_stride_m,
            RES_STRIDE_N,
            res_scale_stride_m,
            res_scale_stride_n,
            RES_OUT_DIM,
        ),
        RoutingMeta(
            rt_expt_block_pid_map,
            rt_expt_hist,
            rt_expt_offs_raw,
            rt_expt_offs_sum,
            rt_gather_indx,
            rt_scatter_indx,
            rt_gammas,
            RT_N_EXPTS_ACT,
        ),
        bias_ptr,
        stride_bias_e,
        x_static_scale_ptr,
        grid_m,
        grid_n,
        func_cfg,
        tuning_cfg,
        CFG_N,
        CFG_K,
    )
