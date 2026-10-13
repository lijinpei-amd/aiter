# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# ========================================================================
# AOT Gluon kernels for pa_mqa_logits (Gluon archs only):
#   1. Generate the kernels with the Triton that will load them; it requires
#      zip installed.
#          $ cd ${AOT_DUMP_AITER_ROOT}
#          $ python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py --batch=1 -aot [-p]
#      "-p" means kernel could assume the stride of KVCache is aligned to 16B.
#      If enable it, the stride of KVCache in the AOT_load side must also be aligned to 16B.
#   2. Copy generated paged_mqa_logits_aot_kernel.zip to ${AOT_LOAD_AITER_ROOT}/aiter/ops/triton/configs
#      and unzip it.
#          $ cd ${AOT_LOAD_AITER_ROOT}
#          $ cd aiter/ops/triton/configs && unzip paged_mqa_logits_aot_kernel.zip && cd -
#   3. Set env variable to enable AOT gluon kernel loading
#          $ export AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1
#          $ python3 op_tests/op_benchmarks/triton/bench_deepgemm_attention.py -kv_length=32768 --batch=2 -mtp=1 -p
#      Set AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=0 to disable AOT gluon kernel. It will backward
#      to triton JIT kernel
# Both modes launch through intj. In AOT mode a launch that misses intj's
# cache still compiles the kernel, then loads the prebuilt hsaco/json in place
# of what it compiled. Loading AOT kernels on Triton < 3.4 (the kernels' extra
# dummy pointer argument) is gone: intj needs Triton >= 3.7, and `import aiter`
# raises without it.
# ========================================================================

import math
import os
from functools import partial

import torch
import triton
import triton.language as tl
from intj import NEVER, Aligned, Argument, Assume, PointerRange, make_launcher

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton._triton_kernels.attention.pa_mqa_logits import (
    _deepgemm_fp8_paged_mqa_logits,
    _deepgemm_fp8_paged_mqa_logits_ragged_k,
    _deepgemm_fp8_paged_mqa_logits_stage1,
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
    _deepgemm_fp8_paged_mqa_logits_varctx_schedule,
)
from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version
from aiter.ops.triton.gluon.pa_mqa_logits import (
    _gluon_deepgemm_fp8_paged_mqa_logits,
    _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle,
    _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx,
)
from aiter.ops.triton.utils.config_utils import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.device_info import current_device_stream, get_num_sms
from aiter.utility.triton.triton_metadata_redirect import (
    AOTMetadataContext,
    MetadataRedirectMissing,
    require_metadata_redirect,
)

enable_aot_gluon_pa_mqa_logits = os.environ.get(
    "AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS", "0"
)
enable_aot_gluon_pa_mqa_logits = enable_aot_gluon_pa_mqa_logits == "1"
_GLUON_PA_MQA_LOGITS_ARCHS = ("gfx942", "gfx950", "gfx1250")
enable_gluon_pa_mqa_logits = get_gfx() in _GLUON_PA_MQA_LOGITS_ARCHS
enable_jit_gluon_pa_mqa_logits_kernel = not enable_aot_gluon_pa_mqa_logits


def deepgemm_fp8_paged_mqa_logits_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 64,
    SplitKV: int = 5,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": heads,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }

    grid = (batch_size * next_n * config["SplitKV"],)
    dev, stream = current_device_stream()
    _deepgemm_fp8_paged_mqa_logits_ragged_k(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        config["ChunkQ"],
        config["ChunkK"],
        config["HiddenDim"],
        config.get("SplitKV", 1),
    )


def deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    kv_cache_fp8, kv_cache_scale = (
        kv_cache_fp8[..., :hidden_dim],
        kv_cache_fp8[..., hidden_dim:],
    )
    # Since triton doesn't have the reinterpret_cast, we slice the scale out and view it as float
    kv_cache_scale = kv_cache_scale.view(torch.float32)
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)

    config = {
        "ChunkQ": 32,
        "ChunkK": 64,
        "HiddenDim": hidden_dim,
        "SplitKV": 5,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * config["SplitKV"]),)
    dev, stream = current_device_stream()
    _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        prefix_sum_context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        config["ChunkQ"],
        config["ChunkK"],
        config["HiddenDim"],
        config.get("SplitKV", 1),
    )


def deepgemm_fp8_paged_mqa_logits_stage1(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache_fp8: torch.Tensor,  # dtype = float8 [num_blocks, block_size, 1, D+4]
    weights: torch.Tensor,  # dtype = float32
    out_qk: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    ChunkQ: int = 64,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    num_blocks, block_size, num_kv_heads, packed_dim = kv_cache_fp8.size()
    _, max_blk_len = kv_indices.size()

    assert num_kv_heads == 1
    assert packed_dim == hidden_dim + 4, (
        "The stage1 kernel expects one fp32 scale after each packed FP8 token; "
        f"got q hidden_dim={hidden_dim} and packed KV dim={packed_dim}."
    )

    TileQCount = batch_size * next_n * (heads // ChunkQ)
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU

    packed_kv_cache = kv_cache_fp8.view(num_blocks, -1)
    value_elements = block_size * hidden_dim
    kv_cache_values = packed_kv_cache[:, :value_elements].view(
        num_blocks, block_size, hidden_dim
    )
    kv_cache_scale = packed_kv_cache[:, value_elements:].view(torch.float32)
    kv_cache_fp8 = kv_cache_values.view(dtypes.fp8)

    config = {
        "ChunkQ": ChunkQ,
        "ChunkK": ChunkK,
        "HiddenDim": hidden_dim,
        "SplitKV": SplitKV,
    }
    assert heads % config["ChunkQ"] == 0

    grid = (batch_size * next_n * (heads // config["ChunkQ"] * SplitKV),)
    dev, stream = current_device_stream()
    _deepgemm_fp8_paged_mqa_logits_stage1(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        WavePerEU,
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_fp8.stride(1),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        kv_cache_scale.stride(1),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_qk,
        out_qk.stride(0),
        out_qk.stride(1),
        max_model_len,
        max_blk_len,
        config["ChunkQ"],
        config["ChunkK"],
        config["HiddenDim"],
        block_size,
        config.get("SplitKV", 1),
    )


def _check_gluon_block_size(Preshuffle, KVBlockSize, ChunkK):
    assert get_gfx() in _GLUON_PA_MQA_LOGITS_ARCHS
    if get_gfx() == "gfx1250":
        if Preshuffle:
            assert KVBlockSize > 1 and ChunkK % KVBlockSize == 0, (
                f"gfx1250 preshuffle (TDM block-load) requires KVBlockSize>1 "
                f"and ChunkK % KVBlockSize == 0 (ChunkK = N*KVBlockSize); got "
                f"KVBlockSize={KVBlockSize}, ChunkK={ChunkK}."
            )
        else:
            assert KVBlockSize == 1, (
                f"gfx1250 base kernel requires KVBlockSize==1; got "
                f"KVBlockSize={KVBlockSize}. Use Preshuffle=True for "
                f"KVBlockSize>1 (TDM block-load)."
            )


def _aot_metadata_path(
    ChunkQ,
    ChunkK,
    Preshuffle,
    KVBlockSize,
    HiddenDim,
    is_padded_mode,
    WavePerEU,
    VarCtxOpt,
):
    """The directory holding the prebuilt hsaco/json that AOTMetadataContext
    substitutes for this kernel's compile."""
    padded_str = "T" if is_padded_mode and not Preshuffle else "F"
    preshuffle_suffix = "_preshuffle" if Preshuffle else ""
    varctx_suffix = "_varctx" if VarCtxOpt else ""
    kernel_str = f"paged_mqa_logits{preshuffle_suffix}{varctx_suffix}_{ChunkQ}x{ChunkK}x{HiddenDim}_B{KVBlockSize}P{padded_str}W{WavePerEU}"
    return f"{AITER_TRITON_CONFIGS_PATH}/paged_mqa_logits/aot/{kernel_str}"


def _gluon_kernel(Preshuffle, VarCtxOpt):
    if not Preshuffle:
        return _gluon_deepgemm_fp8_paged_mqa_logits
    if VarCtxOpt:
        return _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx
    return _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle


def _gluon_launch_keywords(Preshuffle, VarCtxOpt, is_padded_mode):
    """The make_launcher keywords that need the GPU (get_gfx) for this mode's
    launcher: the types and specialization facts the hand-written
    `triton.compile` this replaced spelled out (asserted, not measured: the
    caller promises them), and the options. A parameter it gives
    no fact keeps none, except that intj still measures the facts an `Assume`
    leaves out (the KV pointers' and the index pointers' remaining ones)."""
    gfx_version = get_gfx()
    is_gfx1250 = gfx_version == "gfx1250"
    # gfx942 uses the AMD fnuz e4m3 variant (*fp8e4b8); gfx950 and gfx1250 use
    # the OCP/IEEE e4m3 variant (*fp8e4nv), matching utils.types.get_fp8_dtypes.
    fp8_ptr = tl.pointer_type(
        tl.float8e4b8 if gfx_version == "gfx942" else tl.float8e4nv
    )
    i32 = Argument(type=tl.int32, specialize=NEVER)
    i32_aligned = Argument(type=tl.int32, specialize=Assume(Aligned(16)))
    # The plain kernel forms per-token KV addresses from the page table, so
    # a cache past 2 GiB overflows a 32-bit stride product.
    kv_stride = Argument(
        type=tl.int32 if Preshuffle else tl.int64,
        specialize=Assume(Aligned(16)) if is_padded_mode else NEVER,
    )
    kv_facts = Assume(Aligned(16)) if is_padded_mode else NEVER
    annotation = {
        "batch_size": i32,
        "next_n": i32,
        "heads_num": i32_aligned,
        "Q_buffer": Argument(
            type=fp8_ptr, specialize=Assume(Aligned(16), PointerRange(32))
        ),
        "stride_q_batch": i32_aligned,
        "stride_q_next_n": i32_aligned,
        "stride_q_heads": i32_aligned,
        "KV_buffer": Argument(type=fp8_ptr, specialize=kv_facts),
        "stride_k_seq": kv_stride,
        "scale_buffer": Argument(type=tl.pointer_type(tl.float32), specialize=kv_facts),
        "stride_scale_seq": kv_stride,
        "context_len_ptr": Argument(
            type=tl.pointer_type(tl.int32), specialize=Assume(PointerRange(32))
        ),
        "kv_indices": Argument(
            type=tl.pointer_type(tl.int32), specialize=Assume(PointerRange(32))
        ),
        "weights": Argument(
            type=tl.pointer_type(tl.float32),
            specialize=Assume(Aligned(16), PointerRange(32)),
        ),
        "stride_w_batch": i32_aligned,
        # OutLogits_buffer: NO tt.pointer_range 32 -- the output row base
        # offset (row * stride_out_batch) can exceed a 32-bit byte offset
        # for wide dense logits (e.g. max_model_len=1<<20). stride_out_batch
        # is i64 and the gluon kernel advances the base pointer in 64 bit
        # (buffer_store voffset stays int32) to avoid the 2**31 overflow.
        "OutLogits_buffer": Argument(
            type=tl.pointer_type(tl.float32), specialize=NEVER
        ),
        "stride_out_batch": Argument(type=tl.int64, specialize=NEVER),
        "max_model_len": i32,
        "max_block_len": i32,
        "num_block": i32,
    }
    if VarCtxOpt:
        annotation["safe_chunks_per_cta_ptr"] = Argument(
            type=tl.pointer_type(tl.int32), specialize=NEVER
        )
    else:
        annotation["SplitKV"] = i32

    return {
        "extra_annotation": annotation,
        "options": {
            "num_warps": 1 if is_gfx1250 and Preshuffle else 4,
            "num_stages": 2,
        },
    }


# One launcher per (Preshuffle, VarCtxOpt, KV-padded) mode, each stating that
# mode's types and facts (delayed: they depend on the GPU). The kernels read
# the import-time constant _Use_2d_instr_shape_mfma_layout. Callers read the
# returned kernel's hash (the AOT benchmark zips its cache directory).
_GLUON_PAGED_MQA_LOGITS_LAUNCHERS = {
    (Preshuffle, VarCtxOpt, is_padded_mode): make_launcher(
        _gluon_kernel(Preshuffle, VarCtxOpt),
        dynamic_options=("waves_per_eu",),
        return_compiled=True,
        assume_constant_globals=True,
        delayed=partial(_gluon_launch_keywords, Preshuffle, VarCtxOpt, is_padded_mode),
    )
    for Preshuffle, VarCtxOpt in ((False, False), (True, False), (True, True))
    for is_padded_mode in (False, True)
}

if enable_aot_gluon_pa_mqa_logits:
    # Every load of these kernels must be of a prebuilt binary: a compile
    # outside AOTMetadataContext raises, and deepgemm_fp8_paged_mqa_logits
    # retries it inside one.
    require_metadata_redirect(
        _gluon_deepgemm_fp8_paged_mqa_logits.fn.__name__,
        _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle.fn.__name__,
        _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx.fn.__name__,
    )


def deepgemm_fp8_paged_mqa_logits_schedule(
    batch_size,
    next_n,
    context_lens: torch.Tensor,
    max_model_len: int,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    assert batch_size < TotalCuCount * WavePerEU // next_n

    max_chunks = math.ceil(max_model_len / ChunkK)
    schedule_waves_per_eu = 4
    grid = (TotalCuCount * schedule_waves_per_eu, 1, 1)
    TryCount = math.ceil(max_chunks / grid[0])
    align_power_of_2_batch = 1 << (batch_size - 1).bit_length()

    safe_chunks_per_cta = torch.empty(
        (1,),
        device="cuda",
        dtype=torch.int32,
    )
    dev, stream = current_device_stream()
    _deepgemm_fp8_paged_mqa_logits_varctx_schedule(
        dev,
        stream,
        grid,
        batch_size,
        context_lens,
        safe_chunks_per_cta,
        TotalCuCount * WavePerEU // next_n,
        ChunkK,
        align_power_of_2_batch,
        TryCount,
    )
    return safe_chunks_per_cta


def deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # dtype = float8
    kv_cache,
    weights: torch.Tensor,  # dtype = float32
    out_logits: torch.Tensor,  # dtype = float32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 1,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
    VarCtxSchedule: torch.Tensor = None,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_Size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    if get_gfx() == "gfx1250":
        if Preshuffle and hidden_dim <= 128:
            WavePerEU = 4
        else:
            WavePerEU = 1

    TileQCount = batch_size * next_n
    SplitKV = (
        (max(1, TotalCuCount // TileQCount) + 4)
        // 5
        * 5
        * WavePerEU
        * (2 if get_gfx() == "gfx1250" else 1)
    )

    assert ChunkK % KVBlockSize == 0 or KVBlockSize % ChunkK == 0
    assert block_Size == KVBlockSize
    if Preshuffle:
        # The shuffled layout feeds the 16-token MFMA B tile directly, so a page
        # normally holds whole tiles. Page 8 is shuffled in groups of its own
        # length -- `shuffle_weight(kv, layout=(8, 16))` -- and the kernel
        # assembles one tile from two pages. The addressing generalises to any
        # page that divides 16, but 8 is the only such page a caller asks for
        # and the only one swept, so it is the only one admitted.
        assert KVBlockSize % 16 == 0 or KVBlockSize == 8, (
            "Preshuffle needs the KV page to be a multiple of the 16-token MFMA "
            f"tile, or to be 8. Got KVBlockSize={KVBlockSize}."
        )
        assert not (
            KVBlockSize < 16 and get_gfx() == "gfx1250"
        ), f"gfx1250 preshuffle (TDM block-load) needs KVBlockSize>=16; got {KVBlockSize}."

    kv_cache = kv_cache.view(-1, KVBlockSize * index_dim)
    num_block = kv_cache.shape[0]
    kv_cache_fp8, kv_cache_scale = (
        kv_cache[..., : KVBlockSize * hidden_dim],
        kv_cache[..., KVBlockSize * hidden_dim :],
    )
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)
    kv_cache_scale = kv_cache_scale.view(torch.float32)

    if VarCtxSchedule is not None and get_gfx() == "gfx1250":
        import warnings

        warnings.warn(
            "VarCtx schedule is not implemented on gfx1250 yet; ignoring it and "
            "falling back to the non-varctx preshuffle path."
        )
        VarCtxSchedule = None

    VarCtxOpt = VarCtxSchedule is not None
    if VarCtxOpt:
        grid = (TotalCuCount * WavePerEU, 1, 1)
    else:
        grid = (batch_size * next_n * SplitKV, 1, 1)

    if enable_gluon_pa_mqa_logits:
        is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0
        _check_gluon_block_size(Preshuffle, KVBlockSize, ChunkK)
        assert Preshuffle or not VarCtxOpt, "VarCtxSchedule needs Preshuffle=True"
        launch = _GLUON_PAGED_MQA_LOGITS_LAUNCHERS[
            Preshuffle, VarCtxOpt, is_padded_mode
        ]
        args = (
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_fp8,
            kv_cache_fp8.stride(0),
            kv_cache_scale,
            kv_cache_scale.stride(0),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            num_block,
            SplitKV if not VarCtxOpt else VarCtxSchedule,
            heads,  # ChunkQ
            ChunkK,
            hidden_dim,  # HiddenDim
            KVBlockSize,
            get_cdna_version(),
            get_gfx(),  # ARCH
        )
        waves_per_eu = 1 if get_gfx() == "gfx1250" and not Preshuffle else WavePerEU
        dev, stream = current_device_stream()
        try:
            kernel = launch(dev, stream, grid, waves_per_eu, *args)
        except MetadataRedirectMissing:
            # AOT mode, and the launch missed intj's cache: its compile, on
            # this thread, refused to load what it compiled. Launch again with
            # the prebuilt hsaco/json swapped in; the prebuilt kernel must have
            # this signature. A hit loads nothing, so it needs no context.
            with AOTMetadataContext(
                _gluon_kernel(Preshuffle, VarCtxOpt).fn.__name__,
                _aot_metadata_path(
                    heads,
                    ChunkK,
                    Preshuffle,
                    KVBlockSize,
                    hidden_dim,
                    is_padded_mode,
                    WavePerEU,
                    VarCtxOpt,
                ),
            ):
                kernel = launch(dev, stream, grid, waves_per_eu, *args)
    else:
        assert not Preshuffle, "Preshuffle mode is only supported on gluon kernel."
        kv_cache_values = kv_cache_fp8.view(num_block, KVBlockSize, hidden_dim)
        kv_cache_scales = kv_cache_scale.view(num_block, KVBlockSize)
        dev, stream = current_device_stream()
        kernel = _deepgemm_fp8_paged_mqa_logits(
            dev,
            stream,
            grid,
            WavePerEU,
            batch_size,
            next_n,
            heads,
            q_fp8,
            q_fp8.stride(0),
            q_fp8.stride(1),
            q_fp8.stride(2),
            kv_cache_values,
            kv_cache_values.stride(0),
            kv_cache_values.stride(1),
            kv_cache_scales,
            kv_cache_scales.stride(0),
            kv_cache_scales.stride(1),
            context_lens,
            kv_indices,
            weights,
            weights.stride(0),
            out_logits,
            out_logits.stride(0),
            max_model_len,
            max_block_len,
            heads,  # ChunkQ
            ChunkK,
            hidden_dim,  # HiddenDim
            KVBlockSize,
            SplitKV,
        )
    return triton.runtime.cache.get_cache_manager(kernel.hash).key
