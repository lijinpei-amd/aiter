# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import functools
import math

import torch
import triton
from intj import make_launcher

from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_decode_fwd_kernel as triton_mla_decode_fwd_kernel,
)
from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_decode_fwd_reduce_kernel as triton_mla_decode_fwd_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_prefill_fwd_kernel as triton_mla_prefill_fwd_kernel,
)
from aiter.ops.triton.utils.device_info import current_device_stream, get_num_sms

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_decode_fwd_kernel as gluon_mla_decode_fwd_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_decode_fwd_kernel_non_pipelined as gluon_mla_decode_fwd_kernel_non_pipelined,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_decode_fwd_reduce_kernel as gluon_mla_decode_fwd_reduce_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.mla import (
        _mla_prefill_fwd_kernel_non_pipelined as gluon_mla_prefill_fwd_kernel_non_pipelined,
    )
except:  # noqa: E722
    gluon_mla_prefill_fwd_kernel_non_pipelined = None
    gluon_mla_decode_fwd_kernel_non_pipelined = None
    gluon_mla_decode_fwd_kernel = None
    gluon_mla_decode_fwd_reduce_kernel = None

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64


def select_2d_config(
    block_size,
    head_size,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
):
    TILE_SIZE = block_size
    num_stages_2d = 1
    num_warps = 8

    return {
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": 1,
    }


def select_3d_config(
    block_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    q_dtype,
    kv_dtype,
    shuffled_kv_cache,
    BLOCK_M,
):
    attn_num_warps = 2
    reduce_num_warps = 2
    attn_waves_per_eu = 1
    reduce_waves_per_eu = 2
    num_segments = 0
    TILE_SIZE = block_size
    if IS_DEVICE_ARCH_GFX12:
        # If we cannot infer max_seqlen_k during graph capture
        maybe_guess_max_seqlen_k = 128000 if max_seqlen_k == 0 else max_seqlen_k
        if BLOCK_M > 32:
            attn_num_warps = 4
        reduce_num_warps = 4
        attn_waves_per_eu = 1
        reduce_waves_per_eu = 1
        if shuffled_kv_cache and kv_dtype == torch.uint8:
            assert (
                block_size == 128
            ), "Only block_size == 128 is supported for FP4 KV cache"

        occ = attn_waves_per_eu * 4 // attn_num_warps
        MAX_SEGMENTS = max(1, math.ceil(maybe_guess_max_seqlen_k / TILE_SIZE))
        num_segments = max(1, target_num_prgms // 4 * occ // max(1, num_2d_prgms))
        num_segments = min(MAX_SEGMENTS, num_segments)
        num_segments = triton.next_power_of_2(num_segments)

    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    if num_segments == 0:
        num_segments = math.ceil(target_num_prgms / num_2d_prgms) * 2
        num_segments = min(num_segments, MAX_SEGMENTS)
        num_segments = triton.next_power_of_2(num_segments)
        num_segments = min(num_segments, 128)
        MIN_SEGMENTS = max(8, num_segments)
        num_segments = max(num_segments, MIN_SEGMENTS)

        if num_segments == MIN_SEGMENTS:
            reduce_num_warps = 1

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_num_warps,
        "waves_per_eu": attn_waves_per_eu,
        "num_stages": 2 if DEVICE_ARCH in ("gfx1250", "gfx950") else 1,
    }
    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "waves_per_eu": reduce_waves_per_eu,
        "num_stages": 1,
    }
    return attn_config, reduce_config


@functools.cache
def _gluon_mla_prefill_fwd_kernel_non_pipelined_launch(num_warps, num_stages):
    return make_launcher(
        gluon_mla_prefill_fwd_kernel_non_pipelined,
        dynamic_options=(
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
        ),
        options={"num_warps": num_warps, "num_stages": num_stages},
    )


@functools.cache
def _triton_mla_prefill_fwd_kernel_launch(num_warps, num_stages):
    return make_launcher(
        triton_mla_prefill_fwd_kernel,
        dynamic_options=(
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
        ),
        options={"num_warps": num_warps, "num_stages": num_stages},
    )


def mla_prefill_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale,
    kv_descale,
    out_scale=None,
    shuffled_kv_cache: bool = False,
):
    assert causal, "Only causal attention is supported"
    assert (
        not shuffled_kv_cache
    ), "Shuffled kv cache is not supported in mla_prefill_fwd"

    _total_num_tokens, num_query_heads, qk_head_dim = q.shape
    _num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads
    q_dtype = q.dtype
    kv_buffer_dtype = kv_buffer.dtype
    K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8
    QUERY_DTYPE = "fp8" if q_dtype == e4m3_dtype else "bf16"
    KV_CACHE_DTYPE = "fp8" if kv_buffer_dtype == e4m3_dtype else "bf16"

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    # BLOCK_M = 128
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1 or (num_queries_per_kv > BLOCK_M)
    BLOCK_Q = max(BLOCK_Q, 1)
    # When num_queries_per_kv > BLOCK_M the query heads of a single KV head do
    # not fit into one BLOCK_M tile, so we split them across NUM_HEAD_BLOCKS
    # blocks along the head dimension.
    NUM_HEAD_BLOCKS = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    # cu_count = get_num_sms()
    total_num_q_blocks = (q.shape[0] // BLOCK_Q + num_seqs) * NUM_HEAD_BLOCKS
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill
    attn_config = select_2d_config(
        block_size,
        kv_lora_rank,
        max_seqlen_kv,
        num_queries_per_kv,
        num_2d_prgms,
    )

    if IS_DEVICE_ARCH_GFX12:
        dev, stream = current_device_stream()
        _gluon_mla_prefill_fwd_kernel_non_pipelined_launch(
            attn_config.get("num_warps", 4), attn_config.get("num_stages", 2)
        )(
            dev,
            stream,
            (num_kv_heads, total_num_q_blocks),
            attn_config.get("waves_per_eu", 0),
            attn_config.get("matrix_instr_nonkdim", 0),
            attn_config.get("kpack", 1),
            out,
            q,
            kv_buffer,
            block_tables,
            seqused_k,
            softmax_scale,
            q_descale,
            kv_descale,
            out_scale,
            num_query_heads,
            num_kv_heads,
            block_tables.stride(0),
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            kv_lora_rank,
            qk_rope_head_dim,
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            kv_buffer.stride(2),
            kv_buffer.stride(3),
            cu_seqlens_q,
            num_seqs,
            attn_config["TILE_SIZE"],
            BLOCK_Q,
            BLOCK_M,
            WARP_SIZE,
            attn_config["num_warps"],
            attn_config["num_stages"],
            NUM_HEAD_BLOCKS,
            QUERY_DTYPE,
            KV_CACHE_DTYPE,
            K_WIDTH,
            attn_config.get("FP8_MIN", torch.finfo(e4m3_dtype).min),
            attn_config.get("FP8_MAX", torch.finfo(e4m3_dtype).max),
        )
    else:
        dev, stream = current_device_stream()
        _triton_mla_prefill_fwd_kernel_launch(
            attn_config.get("num_warps", 4), attn_config.get("num_stages", 2)
        )(
            dev,
            stream,
            (num_kv_heads, total_num_q_blocks),
            attn_config.get("waves_per_eu", 0),
            attn_config.get("matrix_instr_nonkdim", 0),
            attn_config.get("kpack", 1),
            out,
            q,
            kv_buffer,
            block_tables,
            seqused_k,
            softmax_scale,
            q_descale,
            kv_descale,
            out_scale,
            num_query_heads,
            num_kv_heads,
            block_tables.stride(0),
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            kv_lora_rank,
            qk_rope_head_dim,
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            kv_buffer.stride(2),
            kv_buffer.stride(3),
            cu_seqlens_q,
            num_seqs,
            attn_config["TILE_SIZE"],
            BLOCK_Q,
            BLOCK_M,
            attn_config["num_warps"],
            attn_config["num_stages"],
            NUM_HEAD_BLOCKS,
            attn_config.get("FP8_MIN", torch.finfo(e4m3_dtype).min),
            attn_config.get("FP8_MAX", torch.finfo(e4m3_dtype).max),
        )
    return out


@functools.cache
def _gluon_mla_decode_fwd_kernel_launch(num_warps, num_stages):
    return make_launcher(
        gluon_mla_decode_fwd_kernel,
        dynamic_options=(
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
        ),
        options={"num_warps": num_warps, "num_stages": num_stages},
    )


@functools.cache
def _gluon_mla_decode_fwd_kernel_non_pipelined_launch(num_warps, num_stages):
    return make_launcher(
        gluon_mla_decode_fwd_kernel_non_pipelined,
        dynamic_options=(
            "waves_per_eu",
            "matrix_instr_nonkdim",
            "kpack",
        ),
        options={"num_warps": num_warps, "num_stages": num_stages},
    )


@functools.cache
def _triton_mla_decode_fwd_kernel_launch(num_warps, num_stages, waves_per_eu):
    return make_launcher(
        triton_mla_decode_fwd_kernel,
        dynamic_options=(
            "matrix_instr_nonkdim",
            "kpack",
        ),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
            "waves_per_eu": waves_per_eu,
        },
    )


_triton_mla_decode_fwd_reduce_kernel_launch = make_launcher(
    triton_mla_decode_fwd_reduce_kernel,
    dynamic_options=(
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "kpack",
    ),
)


def mla_decode_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale,
    kv_descale,
    q_scales=None,
    out_scale=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
):
    assert causal, "Only causal attention is supported"
    q_dtype = q.dtype
    kv_buffer_dtype = kv_buffer.dtype
    total_num_tokens, num_query_heads, qk_head_dim = q.shape

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        # A4W4
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        qk_head_dim = qk_head_dim * 2
        QUERY_DTYPE = "nvfp4"
    elif q_dtype == e4m3_dtype:
        QUERY_DTYPE = "fp8"
    else:
        QUERY_DTYPE = "bf16"

    if kv_buffer_dtype == torch.uint8:
        # A8W4 A4W4
        assert IS_DEVICE_ARCH_GFX12, "FP4 KV cache is only supported on GFX12"
        KV_CACHE_DTYPE = "nvfp4"
    elif kv_buffer_dtype == e4m3_dtype:
        KV_CACHE_DTYPE = "fp8"
    else:
        KV_CACHE_DTYPE = "bf16"

    SCALE_K_WIDTH_LORA = 0
    SCALE_K_WIDTH_ROPE = 0
    if shuffled_kv_cache:
        SCALE_K_WIDTH_LORA = 4
        SCALE_K_WIDTH_ROPE = 4
        if kv_buffer_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = kv_buffer.shape
            K_WIDTH = 16
            SCALE_K_LORA = kv_lora_rank // 16
            SCALE_K_ROPE = qk_rope_head_dim // 16
            SCALE_K_WIDTH_LORA = (
                min(16, triton.next_power_of_2(SCALE_K_LORA))
                if SCALE_K_LORA >= 4
                else SCALE_K_LORA
            )
            SCALE_K_WIDTH_ROPE = (
                min(16, triton.next_power_of_2(SCALE_K_ROPE))
                if SCALE_K_ROPE >= 4
                else SCALE_K_ROPE
            )
        else:
            num_blocks, num_kv_heads, block_size, _ = kv_buffer.shape
            K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8
    else:
        num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
        K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8

    num_seqs = len(seqused_k)
    num_tokens_per_seq = total_num_tokens // num_seqs
    num_queries_per_kv = num_query_heads // num_kv_heads

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    MAX_BLOCK_M = 128 if kv_buffer_dtype == e4m3_dtype else 64
    if num_queries_per_kv <= 16:
        BLOCK_M = 16
    else:
        BLOCK_M = min(triton.next_power_of_2(num_queries_per_kv), MAX_BLOCK_M)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1 or (num_queries_per_kv > BLOCK_M)
    BLOCK_Q = max(BLOCK_Q, 1)
    NUM_HEAD_BLOCKS = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = num_tokens_per_seq == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs * NUM_HEAD_BLOCKS
    else:
        total_num_q_blocks = (
            ((num_tokens_per_seq + BLOCK_Q - 1) // BLOCK_Q) * num_seqs * NUM_HEAD_BLOCKS
        )
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill

    attn_config, reduce_config = select_3d_config(
        block_size,
        max_seqlen_kv,
        target_num_prgms,
        num_2d_prgms,
        q_dtype,
        kv_buffer_dtype,
        shuffled_kv_cache,
        BLOCK_M,
    )

    NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
    if NUM_SEGMENTS > 1:
        segm_output = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        segm_output = out
        segm_max = out  # dummy ptr
        segm_expsum = out  # dummy ptr

    if IS_DEVICE_ARCH_GFX12:
        if shuffled_kv_cache:
            dev, stream = current_device_stream()
            _gluon_mla_decode_fwd_kernel_launch(
                attn_config.get("num_warps", 4), attn_config.get("num_stages", 2)
            )(
                dev,
                stream,
                (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS),
                attn_config.get("waves_per_eu", 0),
                attn_config.get("matrix_instr_nonkdim", 0),
                attn_config.get("kpack", 1),
                segm_output,
                segm_max,
                segm_expsum,
                q,
                q_scales,
                kv_buffer,
                block_tables,
                seqused_k,
                softmax_scale,
                q_descale,
                kv_descale,
                out_scale if (out_scale is not None and NUM_SEGMENTS == 1) else None,
                num_query_heads,
                num_kv_heads,
                block_tables.stride(0),
                q.stride(0),
                q.stride(1),
                q_scales.stride(0) if q_scales is not None else 0,
                q_scales.stride(1) if q_scales is not None else 0,
                kv_lora_rank,
                qk_rope_head_dim,
                kv_buffer.stride(0),
                kv_buffer.stride(1),
                kv_buffer.stride(2),
                kv_buffer.stride(3),
                cu_seqlens_q,
                num_tokens_per_seq,
                num_blocks,
                attn_config["TILE_SIZE"],
                BLOCK_Q,
                BLOCK_M,
                attn_config["NUM_SEGMENTS_PER_SEQ"],
                WARP_SIZE,
                attn_config["num_warps"],
                attn_config["num_stages"],
                shuffled_kv_cache,
                ALL_DECODE,
                K_WIDTH,
                SCALE_K_WIDTH_LORA,
                SCALE_K_WIDTH_ROPE,
                QUERY_DTYPE,
                KV_CACHE_DTYPE,
                BLOCK_SCALES_SIZE,
                NUM_HEAD_BLOCKS,
                attn_config.get("FP8_MIN", torch.finfo(e4m3_dtype).min),
                attn_config.get("FP8_MAX", torch.finfo(e4m3_dtype).max),
            )
        else:
            dev, stream = current_device_stream()
            _gluon_mla_decode_fwd_kernel_non_pipelined_launch(
                attn_config.get("num_warps", 4), attn_config.get("num_stages", 2)
            )(
                dev,
                stream,
                (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS),
                attn_config.get("waves_per_eu", 0),
                attn_config.get("matrix_instr_nonkdim", 0),
                attn_config.get("kpack", 1),
                segm_output,
                segm_max,
                segm_expsum,
                q,
                q_scales,
                kv_buffer,
                block_tables,
                seqused_k,
                softmax_scale,
                q_descale,
                kv_descale,
                out_scale if (out_scale is not None and NUM_SEGMENTS == 1) else None,
                num_query_heads,
                num_kv_heads,
                block_tables.stride(0),
                q.stride(0),
                q.stride(1),
                q_scales.stride(0) if q_scales is not None else 0,
                q_scales.stride(1) if q_scales is not None else 0,
                kv_lora_rank,
                qk_rope_head_dim,
                kv_buffer.stride(0),
                kv_buffer.stride(1),
                kv_buffer.stride(2),
                kv_buffer.stride(3),
                cu_seqlens_q,
                num_tokens_per_seq,
                num_blocks,
                attn_config["TILE_SIZE"],
                BLOCK_Q,
                BLOCK_M,
                attn_config["NUM_SEGMENTS_PER_SEQ"],
                WARP_SIZE,
                attn_config["num_warps"],
                attn_config["num_stages"],
                NUM_HEAD_BLOCKS,
                shuffled_kv_cache,
                ALL_DECODE,
                K_WIDTH,
                SCALE_K_WIDTH_LORA,
                SCALE_K_WIDTH_ROPE,
                QUERY_DTYPE,
                KV_CACHE_DTYPE,
                BLOCK_SCALES_SIZE,
            )
    else:
        dev, stream = current_device_stream()
        _triton_mla_decode_fwd_kernel_launch(
            attn_config.get("num_warps", 4),
            attn_config.get("num_stages", 2),
            attn_config.get("waves_per_eu", 0),
        )(
            dev,
            stream,
            (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS),
            attn_config.get("matrix_instr_nonkdim", 0),
            attn_config.get("kpack", 1),
            segm_output,
            segm_max,
            segm_expsum,
            q,
            q_scales,
            kv_buffer,
            block_tables,
            seqused_k,
            softmax_scale,
            q_descale,
            kv_descale,
            num_query_heads,
            num_kv_heads,
            block_tables.stride(0),
            q.stride(0),
            q.stride(1),
            q_scales.stride(0) if q_scales is not None else 0,
            q_scales.stride(1) if q_scales is not None else 0,
            kv_lora_rank,
            qk_rope_head_dim,
            kv_buffer.stride(0),
            kv_buffer.stride(1),
            kv_buffer.stride(2),
            kv_buffer.stride(3),
            cu_seqlens_q,
            num_tokens_per_seq,
            attn_config["TILE_SIZE"],
            BLOCK_Q,
            BLOCK_M,
            attn_config["NUM_SEGMENTS_PER_SEQ"],
            attn_config["num_warps"],
            attn_config["waves_per_eu"],
            attn_config["num_stages"],
            NUM_HEAD_BLOCKS,
            ALL_DECODE,
            shuffled_kv_cache,
            q_dtype == e4m3_dtype,
            kv_buffer_dtype == e4m3_dtype,
        )

    if NUM_SEGMENTS == 1:
        return segm_output
    elif skip_reduce:
        return segm_output, segm_max, segm_expsum

    # Temporarily disable gluon reduce kernel, optimize later
    # if IS_DEVICE_ARCH_GFX12:
    #     _reduce_kernel = gluon_mla_decode_fwd_reduce_kernel
    # else:
    #     _reduce_kernel = triton_mla_decode_fwd_reduce_kernel

    dev, stream = current_device_stream()
    _triton_mla_decode_fwd_reduce_kernel_launch(
        dev,
        stream,
        (total_num_tokens, num_query_heads),
        reduce_config.get("num_warps", 4),
        reduce_config.get("num_stages", 2),
        reduce_config.get("waves_per_eu", 0),
        reduce_config.get("matrix_instr_nonkdim", 0),
        reduce_config.get("kpack", 1),
        out,
        segm_output,
        segm_max,
        segm_expsum,
        seqused_k,
        out_scale,
        num_seqs,
        num_query_heads,
        out.stride(0),
        out.stride(1),
        block_tables.stride(0),
        num_tokens_per_seq,
        total_num_tokens,
        reduce_config["TILE_SIZE"],
        kv_lora_rank,
        cu_seqlens_q,
        BLOCK_Q,
        reduce_config["NUM_SEGMENTS_PER_SEQ"],
        ALL_DECODE,
        reduce_config.get("FP8_MIN", torch.finfo(e4m3_dtype).min),
        reduce_config.get("FP8_MAX", torch.finfo(e4m3_dtype).max),
    )
    return out
