# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from triton import language as tl

_LOGGER = AiterTritonLogger()
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    """
    Note: this is a Gluon jited function and not meant to be called directly.
    Call gemm_a8w8_blockscale below.

    Computes the 8 bit matmul C = A x B using the block-scale quantization
    approach. A and B tiles are streamed via direct global→LDS buffer loads
    (`async_copy.buffer_load_to_shared`) with NUM_STAGES-deep multi-buffering
    so that loads of the next K tile overlap with the MFMA on the current
    tile. Per-tile scales are loaded directly into MFMA-slice VGPRs (the
    tensors are tiny, so an LDS round-trip would only add latency).

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    # Distributed offset layouts copied verbatim from the Triton-compiled
    # ttgir for this kernel (cache key 6OS5LQDK4...): `#linear` for A and
    # `#linear1` for B. Each lane's per-register stride sequence is chosen so
    # the resulting per-thread write into `shared_a`/`shared_b` lands on
    # consecutive bank lanes -- this is the pairing the AMD lowering for
    # `amdg.buffer_load_to_local` into a `padded_shared` destination requires
    # (a plain BlockedLayout source leaves `unrealized_conversion_cast`s the
    # LLVM translator can't resolve). The bases are baked for BLOCK_M=128 /
    # BLOCK_N=256 / BLOCK_K=128 / NUM_WARPS=4 -- same contract as the
    # padded_shared layouts below.
    linear_a: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    linear_b: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 128]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    # warpsPerCTA = [1, NUM_WARPS] — all warps tile N. This matches the
    # `#mma = #ttg.amd_mfma<{warpsPerCTA = [1, 4], ...}>` pattern that the
    # downstream tt.dot_scaled wants to land on. With BLOCK_M=128,
    # BLOCK_N=256, NUM_WARPS=4, each warp owns a (128, 64) output tile and
    # issues 8x4 = 32 MFMA[16,16,128] instructions — same MFMA count as the
    # prior [2, 2] split, just retiled.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )

    # Padded LDS layout copied verbatim from the Triton-compiled ttgir for
    # this kernel (cache key 6OS5LQDK4...): a 1024-element interval with 32
    # bytes of padding plus a permuted offset_bases swizzle that the bank
    # arbiter favors on CDNA4. Unlike a SwizzledSharedLayout, the K-fast bits
    # come first in identity, then the slow-axis bits are reordered so each
    # 1024-fp8 LDS line spans 8 rows/cols stride-16 — which lines up with the
    # 32-bank read pattern that v_mfma_scaled_* issues. The offset_bases are
    # hard-coded for BLOCK_M=128 / BLOCK_N=256 / BLOCK_K=128 (the only config
    # this kernel ships with); a static_assert below pins that contract.
    gl.static_assert(
        BLOCK_SIZE_M == 128 and BLOCK_SIZE_K == 128 and BLOCK_SIZE_N == 256,
        "shared_a/shared_b padded layouts are baked for "
        "BLOCK_M=128, BLOCK_K=128, BLOCK_N=256",
    )
    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 32]],
        offset_bases=[
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
            [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0],
        ],
        cga_layout=[],
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    shared_b: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 32]],
        offset_bases=[
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
            [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 128],
        ],
        cga_layout=[],
        shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    # 1D layout for the scale prefetch. Each lane writes one fp32 (32-bit
    # direct-to-LDS path on CDNA4); the 4-warp / 64-lane CTA covers 256
    # slots, which over-replicates the BLOCK_SIZE_M / BLOCK_SIZE_N scale
    # vectors -- duplicate writes land at the same LDS address with the
    # same value, so the result is well-defined.
    blocked_scale: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[NUM_WARPS],
        order=[0],
    )
    shared_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    a_scale_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    b_scale_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        # SPLITK_BLOCK_SIZE = gl.cdiv(K, NUM_KSPLIT)
        num_k_iter = gl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Multi-buffered LDS for A, B, and the per-tile scales. The prefetch
        # for stage k+1 lands in bufs_*.index((k + 1) % NUM_STAGES) while the
        # MFMA consumes stage k. Putting scales on the same async path lets
        # the wait_group cover all four loads simultaneously, so the
        # accumulator's `mfma_out * a_scale * b_scale` doesn't stall on a
        # synchronous scale fetch.
        bufs_a = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            [NUM_STAGES, BLOCK_SIZE_M, BLOCK_SIZE_K],
            layout=shared_a,
        )
        bufs_b = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            [NUM_STAGES, BLOCK_SIZE_K, BLOCK_SIZE_N],
            layout=shared_b,
        )
        bufs_as = gl.allocate_shared_memory(
            a_scale_ptr.type.element_ty,
            [NUM_STAGES, BLOCK_SIZE_M],
            layout=shared_scale,
        )
        bufs_bs = gl.allocate_shared_memory(
            b_scale_ptr.type.element_ty,
            [NUM_STAGES, BLOCK_SIZE_N],
            layout=shared_scale,
        )

        offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, linear_a))
        offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, linear_b))
        offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, linear_a)
        )
        offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, linear_b)
        )

        offs_a = offs_am[:, None] * stride_am + (
            pid_k * SPLITK_BLOCK_SIZE + offs_ak[None, :]
        ) * stride_ak
        offs_b = (pid_k * SPLITK_BLOCK_SIZE + offs_bk[:, None]) * stride_bk + offs_bn[
            None, :
        ] * stride_bn

        # Scale offsets in the 1D blocked layout used by the direct-to-LDS
        # loads. B_scale indexes into the N-grouped vector, so a single
        # B_scale element is broadcast across GROUP_N consecutive lanes; the
        # broadcast lanes write the same value to LDS, which is harmless.
        offs_am_scale_blk = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=blocked_scale
        )
        offs_bn_scale_n_blk = (
            pid_n * BLOCK_SIZE_N
            + gl.arange(0, BLOCK_SIZE_N, layout=blocked_scale)
        ) // GROUP_N

        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        offs_a_scale = (
            offs_am_scale_blk * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale = (
            offs_k_scale * stride_bscale_k + offs_bn_scale_n_blk * stride_bscale_n
        )
        offs_ks_step: gl.constexpr = BLOCK_SIZE_K // GROUP_K

        # Prologue: kick off the first global→LDS load (A, B, and both
        # scale vectors) and commit it as a pipeline stage. The mask
        # formulation matches the previous kernel so that K-axis tail
        # handling is unchanged.
        if EVEN_K:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_a.index(0),
                a_ptr,
                offs_a,
                mask=offs_am[:, None] < M,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_b.index(0),
                b_ptr,
                offs_b,
                mask=offs_bn[None, :] < N,
            )
        else:
            k_split_remaining = K - pid_k * num_k_iter * BLOCK_SIZE_K
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_a.index(0),
                a_ptr,
                offs_a,
                mask=(offs_ak[None, :] < k_split_remaining)
                & (offs_am[:, None] < M),
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_b.index(0),
                b_ptr,
                offs_b,
                mask=(offs_bk[:, None] < k_split_remaining)
                & (offs_bn[None, :] < N),
            )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_as.index(0), a_scale_ptr, offs_a_scale
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_bs.index(0), b_scale_ptr, offs_b_scale
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
        acc = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
        )
        zeros = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
        )

        # Main loop: each iteration prefetches stage k+1 (A, B, and both
        # scale vectors) into the same async commit group, waits for stage
        # k, then issues the MFMA for stage k. With NUM_STAGES=2 the
        # in-flight depth never exceeds 2 commits.
        for k in range(num_k_iter - 1):
            # Wait until at most NUM_STAGES-1 commits remain pending so the
            # buffer for stage k is guaranteed populated.
            gl.amd.cdna4.async_copy.wait_group(0)
            offs_a += BLOCK_SIZE_K * stride_ak
            offs_b += BLOCK_SIZE_K * stride_bk
            offs_a_scale += offs_ks_step * stride_ascale_k
            offs_b_scale += offs_ks_step * stride_bscale_k

            buf_idx_next = (k + 1) % NUM_STAGES
            if EVEN_K:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    bufs_a.index(buf_idx_next),
                    a_ptr,
                    offs_a,
                    mask=offs_am[:, None] < M,
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    bufs_b.index(buf_idx_next),
                    b_ptr,
                    offs_b,
                    mask=offs_bn[None, :] < N,
                )
            else:
                k_remaining = K - (pid_k * num_k_iter + k + 1) * BLOCK_SIZE_K
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    bufs_a.index(buf_idx_next),
                    a_ptr,
                    offs_a,
                    mask=(offs_ak[None, :] < k_remaining)
                    & (offs_am[:, None] < M),
                )
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    bufs_b.index(buf_idx_next),
                    b_ptr,
                    offs_b,
                    mask=(offs_bk[:, None] < k_remaining)
                    & (offs_bn[None, :] < N),
                )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_as.index(buf_idx_next), a_scale_ptr, offs_a_scale
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_bs.index(buf_idx_next), b_scale_ptr, offs_b_scale
            )
            gl.amd.cdna4.async_copy.commit_group()

            buf_idx_cur = k % NUM_STAGES

            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                bufs_a.index(buf_idx_cur), dot_a_layout
            )
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                bufs_b.index(buf_idx_cur), dot_b_layout
            )
            cur_a_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(
                bufs_as.index(buf_idx_cur), a_scale_layout
            )
            cur_b_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(
                bufs_bs.index(buf_idx_cur), b_scale_layout
            )

            # tt.dot_scaled with no explicit scale operands (None, None)
            # lowers to v_mfma_scaled_* with the unit-MX-scale fast path.
            # The actual per-tile blockscale (one fp32 per [BLOCK_M,
            # BLOCK_K] / [BLOCK_K, BLOCK_N] tile) is still applied below;
            # microscaling at scale_factor=32 is orthogonal.
            mfma_out = gl.amd.cdna4.mfma_scaled(
                cur_a, None, "e4m3", cur_b, None, "e4m3", zeros
            )
            acc += mfma_out * (cur_a_scale[:, None] * cur_b_scale[None, :])

        # Epilogue: drain the final outstanding commit and consume the last
        # tile.
        gl.amd.cdna4.async_copy.wait_group(0)

        buf_idx_last = (num_k_iter - 1) % NUM_STAGES

        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_a.index(buf_idx_last), dot_a_layout
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_b.index(buf_idx_last), dot_b_layout
        )
        cur_a_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_as.index(buf_idx_last), a_scale_layout
        )
        cur_b_scale = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_bs.index(buf_idx_last), b_scale_layout
        )

        mfma_out = gl.amd.cdna4.mfma_scaled(
            cur_a, None, "e4m3", cur_b, None, "e4m3", zeros
        )
        acc += mfma_out * cur_a_scale[:, None] * cur_b_scale[None, :]

        c = acc.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
        )
        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        c_offs = (
            stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        gl.amd.cdna4.buffer_store(
            stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask
        )


@gluon.jit
def _gemm_a8w8_blockscale_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: gl.constexpr,  # Note: Can be distinct from GEMM block size
    BLOCK_SIZE_N: gl.constexpr,
    ACTUAL_KSPLIT: gl.constexpr,
    MAX_KSPLIT: gl.constexpr,
):

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)

    blocked_read: gl.constexpr = gl.BlockedLayout(  # (MAX_KSPLIT, BLOCK_M, BLOCK_N)
        size_per_thread=[1, 1, 4],
        threads_per_warp=[1, 8, 8],
        warps_per_cta=[1, 4, 1],
        order=[2, 1, 0],
    )

    # blocked_write: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[1, 4], # (BLOCK_M, BLOCK_N)
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[4, 1],
    #     order=[1, 0],
    # )

    offs_m = pid_m * BLOCK_SIZE_M + gl.arange(
        0,
        BLOCK_SIZE_M,  # keep dim 1
        gl.SliceLayout(0, gl.SliceLayout(2, blocked_read)),
    )
    offs_n = pid_n * BLOCK_SIZE_N + gl.arange(
        0,
        BLOCK_SIZE_N,  # keep dim 2
        gl.SliceLayout(0, gl.SliceLayout(1, blocked_read)),
    )
    offs_k = gl.arange(
        0, MAX_KSPLIT, gl.SliceLayout(1, gl.SliceLayout(2, blocked_read))  # keep dim 0
    )
    c_in_offs = (
        (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )
    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c_in_mask = (offs_m[None, :, None] < M) & (offs_n[None, None, :] < N)
        c = gl.amd.cdna4.buffer_load(c_in_ptr, c_in_offs, mask=c_in_mask, cache=".ca")
    else:
        c_in_mask = (
            (offs_m[None, :, None] < M)
            & (offs_n[None, None, :] < N)
            & (offs_k[:, None, None] < ACTUAL_KSPLIT)
        )
        c = gl.amd.cdna4.buffer_load(
            c_in_ptr, c_in_offs, mask=c_in_mask, cache=".ca"
        )  # , other=0.0)
    c = tl.sum(c, 0)

    c = c.to(c_out_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, gl.SliceLayout(1, gl.SliceLayout(0, blocked_read))
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, gl.SliceLayout(0, gl.SliceLayout(0, blocked_read))
    )
    c_out_offs = (offs_cm[:, None] * stride_c_out_m) + (
        offs_cn[None, :] * stride_c_out_n
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna4.buffer_store(
        stored_value=c, ptr=c_out_ptr, offsets=c_out_offs, mask=c_mask
    )


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        if int(dev.split("gfx")[1]) < 950:
            raise ValueError(
                "Gluon implementation is not supported on this device (requires CDNA4)."
            )
        _get_config._config_dict = {}
        fpath = (
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE.json"
        )
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_arch()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    # Config keys should be named M_LEQ_<bound> or "any"
    bounds = []
    for setting in _get_config._config_dict[key].keys():
        potential_block_m = setting.replace("M_LEQ_", "")
        if potential_block_m.isnumeric():
            bounds.append(int(potential_block_m))

    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            config = _get_config._config_dict[key][f"M_LEQ_{bound}"]
            break
        else:
            config = _get_config._config_dict[key]["any"]

    config = (
        config.copy()
    )  # avoid later inplace modification from interacting with cached config

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    return config


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: Scale tensor for X with shape (M, *scale_k).
    - W_scale: Scale tensor for W with shape (**scale_n, *scale_k).

    Returns:
    - Y: The output matrix with shape (M, N).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

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
        config = _get_config(M, N, K)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))

    if config["NUM_KSPLIT"] == 1:
        assert (
            config["GROUP_K"] == config["BLOCK_SIZE_K"]
        ), f"GROUP_K: {config['GROUP_K']} must equal BLOCK_SIZE_K: {config['BLOCK_SIZE_K']} when not using KSPLIT"

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    # NUM_STAGES drives the depth of the LDS multi-buffer used by
    # async_copy.buffer_load_to_shared. The pipeline issues one prefetch ahead
    # of the consuming MFMA, so 2 is the minimum value that overlaps anything.
    num_stages = config.get("num_stages", 2)
    num_stages = max(num_stages, 2)

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a8w8_blockscale_kernel[grid](
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
        NUM_WARPS=config["num_warps"],
        NUM_STAGES=num_stages,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )

        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
            y_pp,
            y,
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
        )

    return y
