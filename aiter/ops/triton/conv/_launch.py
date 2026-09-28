# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from intj import make_launcher

from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _conv2d_1x1_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_1x1 import (
    _get_config as _get_config_1x1,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3 import (
    _conv2d_3x3_cblocked_kernel,
    _conv2d_3x3_nchw_kernel,
    _conv2d_3x3_nhwc_kernel,
    _get_config_cblocked,
    _get_config_nchw,
    _get_config_nhwc,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_gemm as _get_config_wino_gemm,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_input as _get_config_wino_input,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _get_config_output as _get_config_wino_output,
)
from aiter.ops.triton._triton_kernels.conv.conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_cblocked_input_transform_kernel,
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_output_transform_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _conv2d_general_kernel,
)
from aiter.ops.triton._triton_kernels.conv.conv_general import (
    _get_config as _get_config_general,
)
from aiter.ops.triton._triton_kernels.conv.nchw_to_cblocked import (
    _get_config as _get_config_prepack,
)
from aiter.ops.triton._triton_kernels.conv.nchw_to_cblocked import (
    _nchw_to_cblocked_kernel,
)
from aiter.ops.triton.utils.conv_config_utils import (
    format_prepack_shape_key,
    format_shape_key,
)
from aiter.ops.triton.utils.device_info import current_device_stream


def _kernel_activation(activation):
    """Map the public Conv2D GELU name to its existing tanh approximation."""
    return "gelu_tanh" if activation == "gelu" else activation


def _mn_grid(M_total, K_out, config):
    """Grid for the GEMM-style conv kernels (1x1, 3x3 nhwc/cblocked, general):
    one program per (BLOCK_M tile of M_total) x (BLOCK_N tile of K_out)."""
    return (
        triton.cdiv(M_total, config["BLOCK_M"]) * triton.cdiv(K_out, config["BLOCK_N"]),
    )


def _wino_input_grid(T, C_pad, config):
    """Grid for the Winograd F(4,3) input-transform kernels: one program per
    tile T x (BLOCK_C tile of C_pad)."""
    return (T, triton.cdiv(C_pad, config["BLOCK_C"]))


def _wino_gemm_grid(T, K_out, config):
    """Grid for the Winograd F(4,3) batched GEMM: (BLOCK_M tile of T) x
    (BLOCK_N tile of K_out) program blocks, batched over the 36 tile elements."""
    return (
        triton.cdiv(T, config["BLOCK_M"]) * triton.cdiv(K_out, config["BLOCK_N"]),
        36,
    )


def _wino_output_grid(T, K_out, config):
    """Grid for the Winograd F(4,3) output-transform kernels: one program per
    tile T x (BLOCK_K tile of K_out)."""
    return (T, triton.cdiv(K_out, config["BLOCK_K"]))


_nchw_to_cblocked_kernel_launch = make_launcher(
    _nchw_to_cblocked_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_nchw_to_cblocked(x, x_blocked, N, C, H, W, C_pad, block_c):
    """Launch the fused NCHW-to-NCHWc activation pack."""
    HW = H * W
    shape_key = format_prepack_shape_key(N, C, H, W, block_c)
    config = _get_config_prepack(shape_key=shape_key, M=HW)
    grid = (
        triton.cdiv(HW, config["BLOCK_M"]),
        triton.cdiv(C_pad, config["BLOCK_C"]),
        N,
    )
    dev, stream = current_device_stream()
    _nchw_to_cblocked_kernel_launch(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        x_blocked,
        C,
        HW,
        C_pad,
        block_c,
        config["BLOCK_C"],
        config["BLOCK_M"],
    )


_conv2d_1x1_kernel_launch = make_launcher(
    _conv2d_1x1_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_1x1(
    x,
    w_oihw,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    stride,
    padding,
    activation,
    layout="nchw",
):
    """Launch specialized 1x1 kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    sh, sw = stride
    ph, pw = padding

    w = w_oihw.squeeze(-1).squeeze(-1).contiguous()  # [K_out, C]

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=1,
        S=1,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    config = _get_config_1x1(
        shape_key=shape_key,
        M=M_total,
        variants=(layout,),
    )

    dev, stream = current_device_stream()
    _conv2d_1x1_kernel_launch(
        dev,
        stream,
        _mn_grid(M_total, K_out, config),
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        w,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        sh,
        sw,
        ph,
        pw,
        M_total,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_SIZE_M"],
        bias_fp32 is not None,
        _kernel_activation(activation),
        layout,
    )


_conv2d_3x3_nhwc_kernel_launch = make_launcher(
    _conv2d_3x3_nhwc_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_3x3_nhwc(
    x,
    w_3x3,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch specialized 3x3 NHWC kernel (hardcoded stride_c=1, stride_k=1)."""
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_nhwc(
        shape_key=shape_key,
        M=M_total,
    )

    dev, stream = current_device_stream()
    _conv2d_3x3_nhwc_kernel_launch(
        dev,
        stream,
        _mn_grid(M_total, K_out, config),
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        w_3x3,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_SIZE_M"],
        bias_fp32 is not None,
        _kernel_activation(activation),
    )


_conv2d_3x3_cblocked_kernel_launch = make_launcher(
    _conv2d_3x3_cblocked_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_3x3_cblocked(
    x_blocked,
    w_3x3,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    Cb,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch the 3x3 kernel for a materialized 5-D NCHWc input."""
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_cblocked(
        shape_key=shape_key,
        M=M_total,
    )

    dev, stream = current_device_stream()
    _conv2d_3x3_cblocked_kernel_launch(
        dev,
        stream,
        _mn_grid(M_total, K_out, config),
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x_blocked,
        w_3x3,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        Cb,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_SIZE_M"],
        bias_fp32 is not None,
        _kernel_activation(activation),
    )


_conv2d_3x3_nchw_kernel_launch = make_launcher(
    _conv2d_3x3_nchw_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_3x3_nchw(
    x,
    w_3x3,
    bias,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch the repack-free kernel on a contiguous NCHW activation."""
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    M_total = N * P * Q
    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_nchw(shape_key=shape_key, M=M_total)
    row_aligned = "BLOCK_M" in config and Q % config["BLOCK_M"] == 0

    dev, stream = current_device_stream()
    _conv2d_3x3_nchw_kernel_launch(
        dev,
        stream,
        _mn_grid(M_total, K_out, config),
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        w_3x3,
        bias,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_SIZE_M"],
        bias is not None,
        _kernel_activation(activation),
        row_aligned,
    )


_conv2d_general_kernel_launch = make_launcher(
    _conv2d_general_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_general(
    x,
    w_k,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    R,
    S,
    P,
    Q,
    K_pad,
    stride,
    padding,
    dilation,
    block_k,
    activation,
    layout="nchw",
):
    """Launch general conv kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=R,
        S=S,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_general(
        shape_key=shape_key,
        M=M_total,
        variants=(layout,),
    )

    dev, stream = current_device_stream()
    _conv2d_general_kernel_launch(
        dev,
        stream,
        _mn_grid(M_total, K_out, config),
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        x,
        w_k,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        R,
        S,
        P,
        Q,
        K_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["GROUP_SIZE_M"],
        bias_fp32 is not None,
        _kernel_activation(activation),
        layout,
    )


_winograd_f4x3_input_transform_kernel_launch = make_launcher(
    _winograd_f4x3_input_transform_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


_winograd_f4x3_batched_gemm_kernel_launch = make_launcher(
    _winograd_f4x3_batched_gemm_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


_winograd_f4x3_output_transform_kernel_launch = make_launcher(
    _winograd_f4x3_output_transform_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_winograd_f4x3(
    x,
    U,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    layout="nchw",
):
    """Launch Winograd F(4x4,3x3) pipeline: input transform -> batched GEMM -> output transform."""
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Input transform
    dev, stream = current_device_stream()
    _winograd_f4x3_input_transform_kernel_launch(
        dev,
        stream,
        _wino_input_grid(T, C_pad, input_config),
        input_config.get("num_warps", 4),
        input_config.get("num_stages", 2),
        input_config.get("waves_per_eu", 0),
        input_config.get("matrix_instr_nonkdim", 0),
        input_config.get("kpack", 1),
        x,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        input_config["BLOCK_C"],
        layout,
    )

    # 2. Batched GEMM
    dev, stream = current_device_stream()
    _winograd_f4x3_batched_gemm_kernel_launch(
        dev,
        stream,
        _wino_gemm_grid(T, K_out, gemm_config),
        gemm_config.get("num_warps", 4),
        gemm_config.get("num_stages", 2),
        gemm_config.get("waves_per_eu", 0),
        gemm_config.get("matrix_instr_nonkdim", 0),
        gemm_config.get("kpack", 1),
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        gemm_config["BLOCK_M"],
        gemm_config["BLOCK_N"],
        gemm_config["BLOCK_K"],
        gemm_config["GROUP_SIZE_M"],
    )

    # 3. Output transform
    dev, stream = current_device_stream()
    _winograd_f4x3_output_transform_kernel_launch(
        dev,
        stream,
        _wino_output_grid(T, K_out, output_config),
        output_config.get("num_warps", 4),
        output_config.get("num_stages", 2),
        output_config.get("waves_per_eu", 0),
        output_config.get("matrix_instr_nonkdim", 0),
        output_config.get("kpack", 1),
        M,
        bias_fp32,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        output_config["BLOCK_K"],
        bias_fp32 is not None,
        _kernel_activation(activation),
        layout,
    )


_winograd_f4x3_cblocked_input_transform_kernel_launch = make_launcher(
    _winograd_f4x3_cblocked_input_transform_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def _launch_winograd_f4x3_cblocked(
    x_blocked,
    C_pad_blocked,
    U,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    padding,
    activation,
    block_k,
):
    """Launch Winograd F(4x4,3x3) with a materialized NCHWc input."""
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    Cb = block_k
    input_dtype = x_blocked.dtype
    V = torch.empty((36, T, C_pad), device=x_blocked.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x_blocked.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Cblocked input transform
    dev, stream = current_device_stream()
    _winograd_f4x3_cblocked_input_transform_kernel_launch(
        dev,
        stream,
        _wino_input_grid(T, C_pad, input_config),
        input_config.get("num_warps", 4),
        input_config.get("num_stages", 2),
        input_config.get("waves_per_eu", 0),
        input_config.get("matrix_instr_nonkdim", 0),
        input_config.get("kpack", 1),
        x_blocked,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        Cb,
        input_config["BLOCK_C"],
    )

    dev, stream = current_device_stream()
    _winograd_f4x3_batched_gemm_kernel_launch(
        dev,
        stream,
        _wino_gemm_grid(T, K_out, gemm_config),
        gemm_config.get("num_warps", 4),
        gemm_config.get("num_stages", 2),
        gemm_config.get("waves_per_eu", 0),
        gemm_config.get("matrix_instr_nonkdim", 0),
        gemm_config.get("kpack", 1),
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        gemm_config["BLOCK_M"],
        gemm_config["BLOCK_N"],
        gemm_config["BLOCK_K"],
        gemm_config["GROUP_SIZE_M"],
    )

    dev, stream = current_device_stream()
    _winograd_f4x3_output_transform_kernel_launch(
        dev,
        stream,
        _wino_output_grid(T, K_out, output_config),
        output_config.get("num_warps", 4),
        output_config.get("num_stages", 2),
        output_config.get("waves_per_eu", 0),
        output_config.get("matrix_instr_nonkdim", 0),
        output_config.get("kpack", 1),
        M,
        bias_fp32,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        output_config["BLOCK_K"],
        bias_fp32 is not None,
        _kernel_activation(activation),
        output_config.get("LAYOUT", "nchw"),
    )
