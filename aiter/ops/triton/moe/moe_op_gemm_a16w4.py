# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/_matmul.py

import itertools
import os
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w4 import (
    _moe_gemm_a16w4_triton,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w4 import (
    _moe_gemm_a16w4_gluon_stage1,
    _moe_gemm_a16w4_gluon_stage2,
    _moe_gemm_a16w4_gluon_stage3,
)
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch

# -----------------------------------------------------------------------------
#                    Matrix Multiplication + Outer Gather/Scatter
# -----------------------------------------------------------------------------


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def _env_config_int(prefix: str, name: str, k: int, default: int) -> int:
    value = os.environ.get(f"{prefix}_{name}_K{k}", os.environ.get(f"{prefix}_{name}"))
    if value is None:
        return default
    return int(value)


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 1
    split_k = 1
    block_k = 256
    waves_per_eu = 0
    kpack = 1

    if block_m == 16:
        block_n = 128
        num_warps = 4
        # Decode / tiny-M (block_m=16) on gfx1250, tuned on the MiniMax-M3 A16W4
        # eager decode capture (M<=16, N=6144, K in {6144,3072}). A 512-deep K
        # tile beats the flat 256 on the deep gate/up projection (K=6144)
        # consistently across decode M (~6-15% on that GEMM); the shallow down
        # projection (K=3072) keeps the 256 tile.
        block_k = 512 if k >= 4096 else 256

        grid_m = routing_data.n_blocks(m, block_m)
        grid_n = triton.cdiv(n, block_n)
        grid = grid_m * grid_n * split_k
        while block_n >= 64 and grid < 256:
            block_n = block_n // 2
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k

    elif block_m == 32:
        if n <= 1024:
            block_n = 128
            num_warps = 4
        elif n <= 4096:
            block_n = 256
            num_warps = 8
        else:
            block_n = 512
            num_warps = 8

    else:
        # Large block_m (dense prefill grouped MoE, e.g. block_m == 128).
        # Tuned on gfx1250 for the MiniMax-M3 routed GEMMs (N=6144, K in
        # {6144, 3072}, M ~= routed prefill tokens) via rocprofv3 kernel-trace
        # DURATION sweeps. BLOCK_N=256 with a K-dependent BLOCK_K beats the old
        # flat BLOCK_N=512 / BLOCK_K=256 by ~1.4x end-to-end on the fused path.
        num_warps = 8
        block_n = 256
        if k >= 4096:
            # deep-K projection (gate/up, K == hidden): larger K tile + more
            # waves hides the extra K iterations.
            block_k = 512
            waves_per_eu = 2
            group_m = 4
            kpack = 1
        else:
            # shallow-K projection (down, K == intermediate): square K tile,
            # kpack=2 packs two mxfp4 K-slices per MFMA for better throughput.
            block_k = 256
            waves_per_eu = 0
            group_m = 1
            kpack = 2

    ret = {
        "block_m": block_m,
        "block_n": _env_config_int("AITER_MOE_A16W4_TRITON", "BLOCK_N", k, block_n),
        "block_k": _env_config_int("AITER_MOE_A16W4_TRITON", "BLOCK_K", k, block_k),
        "num_warps": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "NUM_WARPS", k, num_warps
        ),
        "num_stages": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "NUM_STAGES", k, num_stages
        ),
        "group_m": _env_config_int("AITER_MOE_A16W4_TRITON", "GROUP_M", k, group_m),
        "xcd_swizzle": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "XCD_SWIZZLE", k, xcd_swizzle
        ),
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "WAVES_PER_EU", k, waves_per_eu
        ),
        "matrix_instr_nonkdim": _env_config_int(
            "AITER_MOE_A16W4_TRITON", "MATRIX_INSTR_NONKDIM", k, 16
        ),
        "kpack": _env_config_int("AITER_MOE_A16W4_TRITON", "KPACK", k, kpack),
    }
    return ret


def get_kernel_config_gluon(m, n, k, routing_data):
    block_m = routing_data.block_m
    group_m = 4
    xcd_swizzle = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_stages = 2
    split_k = 1
    block_k = 512
    num_buffers = 1
    waves_per_eu = 0

    if block_m == 16:
        block_n = 128
        num_warps = 4
        # Decode (block_m=16): the deep-K gate/up projection (K=6144, 12 K-iters
        # at BLOCK_K=512) hides TDM latency behind compute, so the stage-2
        # LDS-prefetch pipeline (NUM_BUFFERS=2) beats single-buffer by ~4-14%
        # across decode M (biggest at M=1). The shallow-K down projection
        # (K=3072, 6 iters) regresses under stage-2 at M>=16, so it stays
        # single-buffered. Verified bit-identical to NUM_BUFFERS=1 output.
        num_buffers = 2 if k >= 4096 else 1
    elif block_m == 32:
        block_n = 128
        # For block_m > 16 the A tile (block_m x block_k) grows with block_m, so
        # the wide 512-deep K tile blows past the LDS budget on GPUs with a
        # 320 KB shared-memory limit. Halve BLOCK_K to 256 to fit.
        block_k = 256
        num_warps = 4
    else:
        # Large-block prefill (block_m>=64) on gfx1250, tuned for the MiniMax-M3
        # A16W4 routed GEMMs (N=6144, K in {6144, 3072}) via do_bench sweeps of
        # each _moe_gemm_a16w4 launch at a saturated grid (~13.7k CTAs).
        #
        # Winner: a 128-wide N tile with the stage-2 (NUM_BUFFERS=2) LDS-prefetch
        # pipeline. Halving BLOCK_N (256->128) halves the W + scale LDS footprint,
        # which lets the double-buffered X/W tiles fit the 320 KB budget while
        # doubling the grid (grid_n 24->48) to keep the GPU saturated -- ~24%
        # faster on gate/up (K=6144) and ~25% on down (K=3072) vs the old
        # single-buffer BLOCK_N=256 config. (An earlier stage-2 attempt at
        # BLOCK_N=256 regressed -- too wide to double-buffer -- which is why
        # single-buffer was previously preferred.)
        #
        # BLOCK_K stays 256: 512 needs ~404 KB LDS at block_m=128 (> 320 KB, fails
        # to launch) and stage-2 has a correctness floor of BLOCK_K>=256, so 256
        # is the only valid choice. xcd_swizzle is projection-dependent: gate/up
        # (deep K=6144) prefers 2, down (K=3072) prefers 8.
        block_n = 128
        block_k = 256
        num_warps = 4
        num_stages = 2
        waves_per_eu = 0
        group_m = 4
        num_buffers = 2
        xcd_swizzle = 2 if k >= 4096 else 8
        # block_m==128 (dense prefill): the down projection (K<4096) is faster
        # single-buffered (stage-1) with xcd_swizzle=2 -- +8-11% end-to-end at
        # M>=4096 vs the stage-2 / xcd_swizzle=8 default (validated full-layer,
        # interleaved min-of-min, gfx1250 tune 2026-07). This regresses at
        # block_m==64, so it is gated on block_m==128 only.
        if block_m == 128 and k < 4096:
            num_buffers = 1
            xcd_swizzle = 2

    return {
        "block_m": block_m,
        "block_n": _env_config_int("AITER_MOE_A16W4_GLUON", "BLOCK_N", k, block_n),
        "block_k": _env_config_int("AITER_MOE_A16W4_GLUON", "BLOCK_K", k, block_k),
        "num_warps": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "NUM_WARPS", k, num_warps
        ),
        "num_stages": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "NUM_STAGES", k, num_stages
        ),
        "group_m": _env_config_int("AITER_MOE_A16W4_GLUON", "GROUP_M", k, group_m),
        "xcd_swizzle": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "XCD_SWIZZLE", k, xcd_swizzle
        ),
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "WAVES_PER_EU", k, waves_per_eu
        ),
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
        "num_buffers": _env_config_int(
            "AITER_MOE_A16W4_GLUON", "NUM_BUFFERS", k, num_buffers
        ),
    }


def _selected_backend() -> str:
    backend = os.environ.get("AITER_MOE_A16W4_BACKEND")
    if backend is None:
        # Default to "gluon": the tuned Gluon path (see get_kernel_config_gluon)
        # is now the default backend. Set AITER_MOE_A16W4_GLUON=0 to fall back to
        # the "auto" behavior (gfx1250 large-block block_m>=64 prefill GEMMs on
        # Gluon, small block_m decode-style shapes on Triton), or set
        # AITER_MOE_A16W4_BACKEND explicitly to override.
        backend = "auto" if os.environ.get("AITER_MOE_A16W4_GLUON") == "0" else "gluon"
    backend = backend.lower()
    if backend not in {"triton", "gluon", "auto"}:
        raise ValueError(f"unknown AITER_MOE_A16W4_BACKEND={backend!r}")
    return backend


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_a16w4(
    x,
    w,
    x_scales,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    w_scales,
    x_static_scale=None,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    quant_static_scale=None,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    unpadded_N=None,
    unpadded_K=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    assert x_scales is None, "x_scales must be none"
    assert x_static_scale is None, "x_static_scale must be none"
    assert quant_static_scale is None, "quant_static_scale must be none"

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    # compute optimization flags
    backend = _selected_backend()
    # The Gluon path is only numerically correct for compact (non-swizzled) MX
    # scales: GFX1250_SCALE hangs the ROCm loader at first launch and CDNA4_SCALE
    # is unsupported (produces inf under swiglu). So swizzled scales ALWAYS stay
    # on Triton, regardless of the selected backend -- including the "gluon"
    # default. An explicit AITER_MOE_A16W4_BACKEND=gluon with swizzled scales
    # cannot be honored correctly, so fail loudly rather than hang / return inf.
    # "auto" additionally restricts Gluon to gfx1250 large-block (block_m>=64)
    # prefill, where it has been validated.
    if backend == "gluon" and get_arch() == "gfx1250" and swizzle_mx_scale is not None:
        raise ValueError(
            "AITER_MOE_A16W4_BACKEND=gluon cannot honor swizzled MX scales "
            f"(swizzle_mx_scale={swizzle_mx_scale!r}): the Gluon a16w4 kernel "
            "supports only compact e8m0 scales. Use the Triton backend "
            "(AITER_MOE_A16W4_BACKEND=triton or =auto) for swizzled scales."
        )
    use_gluon = (
        get_arch() == "gfx1250"
        and swizzle_mx_scale is None
        and (backend == "gluon" or (backend == "auto" and block_m >= 64))
    )
    config = (
        get_kernel_config_gluon(M, N, K, routing_data)
        if use_gluon
        else get_kernel_config(M, N, K, routing_data)
    )
    if os.environ.get("AITER_MOE_A16W4_DEBUG_PRINT"):
        print(
            f"moe_gemm_a16w4 backend={backend} use_gluon={use_gluon} "
            f"M={M} N={N} K={K} swiglu={int(bool(apply_swiglu))} config={config}",
            flush=True,
        )
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1

    # allocate output memory
    y, y_final = allocate_output(
        x,
        w,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
    )
    stride_bias = None if bias is None else bias.stride(0)

    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    # spmd grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]

    # launch kernel
    if use_gluon:
        w_scales_kernel = w_scales.transpose(1, 2)
        # stage-1 (single buffer) and stage-2 (LDS-prefetch double buffer) are
        # separate named kernels; pick by the resolved buffer count so profiles
        # attribute time to the pipeline that actually ran.
        num_buffers = max(
            1, min(config["num_buffers"], triton.cdiv(K, config["block_k"]))
        )
        if num_buffers == 1:
            gluon_kernel = _moe_gemm_a16w4_gluon_stage1
        elif num_buffers == 2:
            gluon_kernel = _moe_gemm_a16w4_gluon_stage2
        else:
            gluon_kernel = _moe_gemm_a16w4_gluon_stage3
        gluon_kernel[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales_kernel,
            w_scales_kernel.stride(0),
            w_scales_kernel.stride(1),
            w_scales_kernel.stride(2),
            bias,
            stride_bias,
            gammas,
            x.shape[-2],
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            swiglu_add_residual,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            config["group_m"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            NUM_BUFFERS=num_buffers,
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
        )
    else:
        _moe_gemm_a16w4_triton[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            bias,
            stride_bias,
            gammas,
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            swiglu_add_residual,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            config["group_m"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
        )

    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
    )

    return y_final


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def swiglu_torch(a, alpha, limit, add_residual=True):
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    if add_residual:
        out = out_gelu * (a_linear + 1)
    else:
        out = out_gelu * a_linear

    return out


def moe_gemm_torch(
    x,
    w,
    bias,
    routing_data: RoutingData = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    add_residual=True,
):
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    n_expts_act = routing_data.n_expts_act

    # memory offsets
    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]

    # compute
    n_rows = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_swiglu else w.shape[-1]
    y = torch.zeros((n_rows, n_cols), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            gather_indx = gather_indx.to(torch.int32)
            idx = gather_indx[lo:hi] // n_expts_act
        out = torch.matmul(x[idx, :].float(), w[i].float())
        if bias is not None:
            out += bias[i, :]
        if apply_swiglu:
            out = swiglu_torch(out, alpha, limit, add_residual)
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[lo:hi, :] = out
    if scatter_indx is None:
        return y

    # accumulate output from all experts
    scatter_indx = scatter_indx.to(torch.int32)
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows):
        out[i, :] = y[src_idx[i], :].float().sum(0)

    return out
