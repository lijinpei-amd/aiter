# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check the MXFP4 LDS staging independently of the GEMM pipeline."""

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._epilogue import (
    _epi_block_flush,
    _epi_stage_flush,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import constexpr_fields
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DtypeQuant,
    FuncSpec,
    TuningSpec,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def _epilogue_config(split, num_warps):
    func = FuncSpec(
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        gl.float32,
        ActivationSpec(int(ActKind.SILU), 1.702, 7.0, True),
        int(DtypeQuant.MXFP4),
        False,
        False,
        False,
        False,
        split,
    )
    tuning = TuningSpec(
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        K_UNROLL=1,
        MINI_BLOCK_K=256,
        MINI_BLOCK_M=64,
        MINI_BLOCK_N=128,
        NUM_LDS_BUFFER=2,
        mfma_instr_shape=(16, 16, 128),
        warps_per_cta=(num_warps // 4, 4),
        tiles_per_warp=(2, 2),
        k_width=8,
        transposed=True,
        WAVES_PER_EU=0,
        TILE_SCHED=0,
        GROUP_M=1,
        NUM_XCDS=1,
        token_cache_modifier="",
        token_scale_cache_modifier="",
        expert_cache_modifier="",
        expert_scale_cache_modifier="",
        result_cache_modifier="",
        result_scale_cache_modifier="",
        WARP_PIPELINE=0,
        VGPR_PREFETCH_K=0,
        ACT_FAST_RCP=True,
    )
    return gl.constexpr(func), gl.constexpr(tuning)


@gluon.jit
def _staging_bytes(block_id, pid_n, mi, ni, epoch, func, tuning):
    RFL: gl.constexpr = tuning.dot_result_fragment_layout()
    REDUCTION: gl.constexpr = func.mini_n_reduction()
    OUT_N: gl.constexpr = 128 // REDUCTION
    rows = block_id * 128 + mi * 64 + gl.arange(
        0, 64, layout=gl.SliceLayout(1, RFL)
    )
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, RFL))
    out_n = pid_n * 128 + ni * OUT_N + cols // REDUCTION
    payload = (rows[:, None] * 17 + out_n[None, :] // 2 * 13 + epoch * 29).to(
        gl.uint8
    )
    scales = (rows[:, None] * 7 + out_n[None, :] // 32 * 19 + epoch * 31).to(
        gl.uint8
    )
    # Collapse identical codes to the payload/scale shapes using the accumulator layout.
    return (
        gl.max(payload.reshape(64, OUT_N // 2, REDUCTION * 2), 2).to(gl.uint8),
        gl.max(scales.reshape(64, OUT_N // 32, REDUCTION * 32), 2).to(gl.uint8),
    )


@gluon.jit
def _epilogue_probe(
    payload_ptr,
    scale_ptr,
    M,
    epoch,
    FUNC: gl.constexpr,
    TUNING: gl.constexpr,
    N: gl.constexpr = 512,
):
    func = KernelFuncConfig(*constexpr_fields(FUNC))
    tuning = KernelTuningConfig(func, *constexpr_fields(TUNING))
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    block_id = gl.program_id(0)
    pid_n = gl.program_id(1)
    if func.gu_split():
        payload_smem = gl.allocate_shared_memory(gl.uint8, [128, 64], SH)
        scale_smem = gl.allocate_shared_memory(gl.uint8, [128, 4], SH)
        for mi in gl.static_range(2):
            payload, scales = _staging_bytes(block_id, pid_n, mi, 0, epoch, func, tuning)
            payload_smem.slice(mi * 64, 64).store(payload)
            scale_smem.slice(mi * 64, 64).store(scales)
        _epi_block_flush(
            payload_smem, scale_smem,
            payload_ptr, N // 4, 1, scale_ptr, N // 64, 1,
            block_id, pid_n, M, func, tuning,
        )
    else:
        payload_0 = gl.allocate_shared_memory(gl.uint8, [64, 32], SH)
        payload_1 = gl.allocate_shared_memory(gl.uint8, [64, 32], SH)
        scale_0 = gl.allocate_shared_memory(gl.uint8, [64, 2], SH)
        scale_1 = gl.allocate_shared_memory(gl.uint8, [64, 2], SH)
        for tile in gl.static_range(4):
            if tile > 0:
                _epi_stage_flush(
                    payload_1 if (tile - 1) % 2 else payload_0,
                    scale_1 if (tile - 1) % 2 else scale_0,
                    (tile - 1) // 2, (tile - 1) % 2,
                    payload_ptr, N // 4, 1, scale_ptr, N // 64, 1,
                    block_id, pid_n, M, func, tuning,
                )
            payload, scales = _staging_bytes(
                block_id, pid_n, tile // 2, tile % 2, epoch, func, tuning
            )
            (payload_1 if tile % 2 else payload_0).store(payload)
            (scale_1 if tile % 2 else scale_0).store(scales)
            gl.barrier()
        _epi_stage_flush(
            payload_1, scale_1, 1, 1,
            payload_ptr, N // 4, 1, scale_ptr, N // 64, 1,
            block_id, pid_n, M, func, tuning,
        )


@pytest.mark.parametrize("split", [False, True], ids=["rotate", "split"])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_mxfp4_epilogue_staging(split, num_warps):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")

    rows, columns, valid_rows = 1024, 512, 1021
    func, tuning = _epilogue_config(split, num_warps)
    payload = torch.full((rows, columns // 4), 0xA5, device="cuda", dtype=torch.uint8)
    scales = torch.full((rows, columns // 64), 0xA5, device="cuda", dtype=torch.uint8)
    row = torch.arange(rows)[:, None]
    payload_col = torch.arange(columns // 4)[None, :]
    scale_col = torch.arange(columns // 64)[None, :]
    grid = (triton.cdiv(valid_rows, 128), columns // 256)
    for epoch in range(32):
        ref_payload = (row * 17 + payload_col * 13 + epoch * 29).to(torch.uint8)
        ref_scales = (row * 7 + scale_col * 19 + epoch * 31).to(torch.uint8)
        ref_payload[valid_rows:] = 0xA5
        ref_scales[valid_rows:] = 0xA5
        payload.fill_(0xA5)
        scales.fill_(0xA5)
        kernel = _epilogue_probe[grid](
            payload, scales, valid_rows, epoch,
            func, tuning, num_warps=num_warps,
        )
        actual_payload = payload.cpu()
        actual_scales = scales.cpu()
        assert torch.equal(actual_payload, ref_payload), (
            f"payload differs at {(actual_payload != ref_payload).nonzero()[:16].tolist()}"
        )
        assert torch.equal(actual_scales, ref_scales), (
            f"scales differ at {(actual_scales != ref_scales).nonzero()[:16].tolist()}"
        )

    assert kernel.asm["ttgir"].count("ttg.amdg.syncedViaAsyncWait = true") >= 2
    assert "fence syncscope(\"workgroup\") release" in kernel.asm["llir"]
