# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_kv_cache import (
    _fused_qk_rope_cat_and_cache_mla_kernel as triton_fused_qk_rope_cat_and_cache_mla_kernel,
)
from aiter.ops.triton._triton_kernels.fusions.fused_kv_cache import (
    _fused_qk_rope_cosine_cache_llama_kernel,
)
from aiter.ops.triton._triton_kernels.fusions.fused_kv_cache import (
    _fused_qk_rope_reshape_and_cache_kernel as triton_fused_qk_rope_reshape_and_cache_kernel,
)

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_kv_cache import (
        _fused_qk_rope_cat_and_cache_mla_kernel as gluon_fused_qk_rope_cat_and_cache_mla_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_kv_cache import (
        _fused_qk_rope_reshape_and_cache_kernel as gluon_fused_qk_rope_reshape_and_cache_kernel,
    )
except:  # noqa: E722
    gluon_fused_qk_rope_cat_and_cache_mla_kernel = None
    gluon_fused_qk_rope_reshape_and_cache_kernel = None

from intj import make_launcher

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.device_info import current_device_stream
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

_triton_cat_and_cache_mla_launch = make_launcher(
    triton_fused_qk_rope_cat_and_cache_mla_kernel, options={"num_warps": 1}
)


@functools.cache
def _gluon_cat_and_cache_mla_launch():
    # The Gluon kernel is None when its gfx1250 module fails to import.
    return make_launcher(
        gluon_fused_qk_rope_cat_and_cache_mla_kernel, options={"num_warps": 1}
    )


_LOGGER = AiterTritonLogger()

DEVICE_ARCH = arch_info.get_arch()

_BLOCK_H_MIN_TOKENS = int(os.environ.get("AITER_FUSED_KV_CACHE_MIN_TOKENS", "128"))


def fused_qk_rope_cat_and_cache_mla_fake_tensor(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    is_neox: bool,
    num_decode_toks_for_zeros: int = 0,
    apply_scale: bool = True,
    q_out: torch.Tensor = None,
    decode_q_pe_out: torch.Tensor = None,
    k_pe_out: torch.Tensor = None,
    q_out_dtype: torch.dtype = None,
    shuffled_kv_cache: bool = False,
    upcast_operand: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, qh, d_nope = q_nope.shape
    _, _, d_pe = q_pe.shape
    bk, kh, dk_nope = k_nope.shape

    if q_out is None:
        q_out = torch.empty(
            (b, qh, d_nope + d_pe),
            dtype=q_out_dtype if q_out_dtype is not None else q_nope.dtype,
            device=q_nope.device,
        )

    if decode_q_pe_out is None:
        decode_q_pe_out = torch.empty(
            (num_decode_toks_for_zeros, qh, d_pe),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

    if k_pe_out is None:
        k_pe_out = torch.empty((bk, kh, d_pe), dtype=k_pe.dtype, device=k_pe.device)

    if num_decode_toks_for_zeros > 0:
        q_nope_zeros_out = torch.empty(
            (num_decode_toks_for_zeros, qh, dk_nope),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    else:
        q_nope_zeros_out = torch.empty(
            (0, qh, dk_nope),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

    return q_out, decode_q_pe_out, k_pe_out, q_nope_zeros_out


@torch_compile_guard(gen_fake=fused_qk_rope_cat_and_cache_mla_fake_tensor)
def fused_qk_rope_cat_and_cache_mla(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    is_neox: bool,
    num_decode_toks_for_zeros: int = 0,
    apply_scale: bool = True,
    q_out: torch.Tensor = None,
    decode_q_pe_out: torch.Tensor = None,
    k_pe_out: torch.Tensor = None,
    q_out_dtype: torch.dtype = None,
    shuffled_kv_cache: bool = False,
    upcast_operand: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform RoPE on q_pe and k_pe and concat q_nope with q_pe and k_nope with k_pe along the last dimension
    the concatenated k_nope and k_pe are copied to kv_cache inplace

    Key parameters:
    - q_nope: Matrix X with shape (B, QH, D1).
    - q_pe: Matrix W with shape (B, QH, D2).
    - k_nope: Matrix X with shape (B_slot, KH, D1).
    - k_pe: Matrix W with shape (B_slot, KH, D2).
    - kv_cache: Matrix W with shape (B_cache, KH, D1 + D2).
    - slot_mapping: Matrix W with shape (B_slot, ).

    B is the number of decode tokens, B_slot is the number of prefill + decode tokens, B_cache is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: The output matrix with shape (B, QH, D1+D2).
    - kv_cache: The output matrix with shape (B_max, KH, D1 + D2) (inplace).
    """
    _LOGGER.info(
        "FUSED_QK_ROPE_CAT_AND_CACHE_MLA: q_nope=%s q_pe=%s k_nope=%s k_pe=%s pos=%s cos=%s sin=%s kv_cache=%s slot_mapping=%s",
        tuple(q_nope.shape),
        tuple(q_pe.shape),
        tuple(k_nope.shape),
        tuple(k_pe.shape),
        tuple(pos.shape),
        tuple(cos.shape),
        tuple(sin.shape),
        tuple(kv_cache.shape),
        tuple(slot_mapping.shape),
    )

    b, qh, d_nope = q_nope.shape
    b2, qh2, d_pe = q_pe.shape
    bk, kh, dk_nope = k_nope.shape
    bk2, kh2, dk2 = k_pe.shape
    kv_cache_dtype = kv_cache.dtype
    d_freq = cos.shape[-1]
    assert kv_cache_dtype in [
        torch.bfloat16,
        e4m3_dtype,
        torch.uint8,
    ], "KV cache dtype must be BF16, FP8 or packed FP4"

    block_size = 1
    SCALE_K_WIDTH_NOPE = 4
    SCALE_K_WIDTH_ROPE = 4
    if kv_cache_dtype == torch.uint8:
        assert shuffled_kv_cache, "shuffle_kv_cache must be True for FP4 KV cache"
        _b_cache, h_cache, block_size, d_cache = kv_cache.shape
        SCALE_K_LORA = d_nope // 16
        SCALE_K_ROPE = d_pe // 16
        SCALE_K_WIDTH_NOPE = (
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
        if shuffled_kv_cache:
            _b_cache, h_cache, block_size, d_cache = kv_cache.shape
        else:
            _b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    # allow bk >= b to support prefill + decode mixed scenario
    assert (
        b == b2 and bk == bk2 and b_slot <= bk and b <= bk
    ), "Q batch dimensions should be identical (b == b2), K batch dimensions should be identical (bk == bk2), slot_mapping should not exceed K batch size (b_slot <= bk), and Q batch should not exceed K batch (b <= bk)"
    assert qh == qh2, "Q head should be identical"
    assert kh == kh2 == h_cache, "K head should be identical"
    assert d_pe == dk2, "D dimension of q_pe and k_pe should be identical"
    assert d_nope == dk_nope, "D dimension of q_nope and k_nope should be identical"
    if kv_cache.dtype == torch.uint8:
        assert (
            (d_nope + d_pe) // 2 + (d_nope + d_pe) // 16
        ) == d_cache, "The D dimension of kv_cache should be (d_nope + d_rope) // 2 + (d_nope + d_rope) // 16 for FP4 KV cache"
    else:
        assert (
            dk_nope + d_pe == d_cache
        ), "D dimension of k_nope and k_pe should be summed up to be the D dimension of kv_cache"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    assert (d_freq == d_pe // 2) or (
        d_freq == d_pe
    ), "cos/sin last dim should be the same or half of the qk last dim"
    assert (
        num_decode_toks_for_zeros >= 0
    ), "num_decode_toks_for_zeros must be non-negative to avoid invalid tensor creation"
    if isinstance(k_scale, torch.Tensor):
        assert k_scale.numel() == 1, "k_scale should be a single-element torch.Tensor"
    reuse_freqs_front_part = d_freq == d_pe // 2

    if q_out is None:
        q_out = torch.empty(
            (b, qh, d_nope + d_pe),
            dtype=q_out_dtype if q_out_dtype is not None else q_nope.dtype,
            device=q_nope.device,
        )
    else:
        b_q_out, qh_q_out, d_q_out = q_out.shape
        assert (
            b == b_q_out and qh == qh_q_out and d_nope + d_pe == d_q_out
        ), "q_out shape mismatch"

    if decode_q_pe_out is None:
        decode_q_pe_out = torch.empty(
            (num_decode_toks_for_zeros, qh, d_pe),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    else:
        b_decode_q_pe_out, qh_decode_q_pe_out, d_decode_q_pe_out = decode_q_pe_out.shape
        assert (
            num_decode_toks_for_zeros == b_decode_q_pe_out
            and qh == qh_decode_q_pe_out
            and d_pe == d_decode_q_pe_out
        ), "decode_q_pe_out shape mismatch"

    if k_pe_out is None:
        k_pe_out = torch.empty((bk, kh, d_pe), dtype=k_pe.dtype, device=k_pe.device)
    else:
        b_k_pe_out, hk_k_pe_out, d_k_pe_out = k_pe_out.shape
        assert (
            bk == b_k_pe_out and kh == hk_k_pe_out and d_pe == d_k_pe_out
        ), "k_pe_out shape mismatch, expected (bk, kh, d_pe)"

    q_nope_zeros_out = torch.empty(
        (num_decode_toks_for_zeros, qh, d_nope),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )

    if shuffled_kv_cache:
        kv_cache_stride_b = kv_cache.stride(0)
        kv_cache_stride_h = kv_cache.stride(1)
        kv_cache_stride_d = kv_cache.stride(3)
    else:
        kv_cache_stride_b = kv_cache.stride(0)
        kv_cache_stride_h = kv_cache.stride(1)
        kv_cache_stride_d = kv_cache.stride(2)

    assert (
        kv_cache_stride_d == 1
    ), "The stride of the last dimension of KV cache must be 1"

    n_pid = b * qh + (b_slot - b) * kh
    grid = (n_pid, 1, 1)
    if DEVICE_ARCH == "gfx1250":
        _kernel = _gluon_cat_and_cache_mla_launch()
    else:
        _kernel = _triton_cat_and_cache_mla_launch

    dev, stream = current_device_stream()
    _kernel(
        dev,
        stream,
        grid,
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        pos,
        cos,
        sin,
        q_out,
        decode_q_pe_out,
        k_pe_out,
        q_nope_zeros_out,
        kv_cache,
        slot_mapping,
        b,
        b_slot,
        num_decode_toks_for_zeros,
        *q_nope.stride(),
        *q_pe.stride(),
        *k_nope.stride(),
        *k_pe.stride(),
        pos.stride(0),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        *decode_q_pe_out.stride(),
        *k_pe_out.stride(),
        *q_nope_zeros_out.stride(),
        kv_cache_stride_b,
        kv_cache_stride_h,
        kv_cache_stride_d,
        k_scale,  # k_scale_ptr
        qh // kh,  # QH_PER_KH
        qh,  # QH
        kh,  # KH
        reuse_freqs_front_part,  # REUSE_FREQS_FRONT_PART
        is_neox,  # IS_NEOX
        d_nope,  # BLOCK_D_nope
        d_pe,  # BLOCK_D_pe
        d_pe // 2,  # BLOCK_D_HALF_pe
        block_size,  # BLOCK_SIZE
        shuffled_kv_cache,  # SHUFFLED_KV_CACHE
        SCALE_K_WIDTH_NOPE,
        SCALE_K_WIDTH_ROPE,
        num_decode_toks_for_zeros > 0,  # OUTPUT_Q_NOPE_ZEROS_AND_Q_PE
        k_scale is not None and apply_scale,  # HAVE_K_SCALE
        upcast_operand,  # UPCAST_OPERAND
    )

    return q_out, decode_q_pe_out, k_pe_out, q_nope_zeros_out


@functools.cache
def _gluon_fused_qk_rope_reshape_and_cache_kernel_launch():
    return make_launcher(
        gluon_fused_qk_rope_reshape_and_cache_kernel, options={"num_warps": 1}
    )


_triton_fused_qk_rope_reshape_and_cache_kernel_launch = make_launcher(
    triton_fused_qk_rope_reshape_and_cache_kernel, options={"num_warps": 1}
)


def fused_qk_rope_reshape_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    output_zeros: bool = True,
    zeros_out: torch.Tensor = None,
    upcast_operand: bool = False,
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v into key_cache and value_cache inplace

    Key parameters:
    - q: shape (T, QH, D).
    - k: shape (T, KH, D).
    - v: shape (T, KH, D).
    - if flash_layout:
    -     key_cache: shape (T_cache, block_size, KH, D).
    -     value_cache: shape (T_cache, block_size, KH, D).
    - else:
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).
    -     value_cache: shape (T_cache, KH, D, block_size).
    - slot_mapping: shape (T_slot, ).

    T is the number of decode tokens, T_cache * block_size is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: same shape as input q.
    - k_out: same shape as input k.
    - key_cache: same shape as input key_cache (inplace).
    - value_cache: same shape as input value_cache (inplace).
    - zeros_out: same shape as input q.
    """
    _LOGGER.info(
        "FUSED_QK_ROPE_RESHAPE_AND_CACHE: q=%s k=%s pos=%s cos=%s sin=%s key_cache=%s value_cache=%s slot_mapping=%s",
        tuple(q.shape),
        tuple(k.shape),
        tuple(pos.shape),
        tuple(cos.shape),
        tuple(sin.shape),
        tuple(key_cache.shape),
        tuple(value_cache.shape),
        tuple(slot_mapping.shape),
    )

    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    kv_cache_dtype = key_cache.dtype
    assert kv_cache_dtype in [
        torch.bfloat16,
        e4m3_dtype,
        torch.uint8,
    ], "KV cache dtype must be BF16, FP8 or packed FP4"

    SCALE_K_WIDTH = 4
    x_cache = 8
    value_shuffle_layout = False
    max_embd_pos = cos.shape[0]
    d_freq = cos.shape[-1]
    if kv_cache_dtype == torch.uint8:
        # always shuffled
        t_cache, kh_cache, block_size, d_cache = key_cache.shape
        t_cache_v, vh_cache, block_size_v, d_cache_v = value_cache.shape
        assert block_size == block_size_v
        SCALE_K = dk // 16
        SCALE_K_WIDTH = (
            min(16, triton.next_power_of_2(SCALE_K)) if SCALE_K >= 4 else SCALE_K
        )
    else:
        if flash_layout:
            t_cache, block_size, kh_cache, dk_cache = key_cache.shape
            t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
            value_shuffle_layout = False
        else:
            t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
            if value_cache.ndim == 5:
                # value_cache shuffle: (num_blocks, num_kv_heads, block_size // x, head_size, x)
                t_cache_v, vh_cache, slot_chunk_v, dv_cache, x_v = value_cache.shape
                value_shuffle_layout = True
                block_size_v = slot_chunk_v * x_v
                assert block_size_v == block_size and x_v == x_cache, (
                    f"value_cache shuffle (T,KH,block_size//x,D,x) must match key: "
                    f"{block_size_v=} {block_size=} {x_v=} {x_cache=}"
                )
            else:
                t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
                value_shuffle_layout = False
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t <= t_slot
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more less that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if kv_cache_dtype == torch.uint8:
        assert d == dk == dv, "D dimension should be identical for q, k, and v"
        assert (
            d_cache == d_cache_v
        ), "D dimension should be identical for key_cache and value_cache"
        assert (
            dk // 2 + dk // 16 == d_cache
        ), "D dimension of key_cache should be (dk // 2 + dk // 16) for FP4 KV cache"
    else:
        if flash_layout:
            assert (
                d == dk == dv == dk_cache == dv_cache
            ), "D dimension should be identical for q, k, and v"
        else:
            assert (
                d == dk == dv == dkx_cache * x_cache == dv_cache
            ), "D dimension should be identical for q, k, and v"
            assert x_cache == triton.next_power_of_2(
                x_cache
            ), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert block_size == triton.next_power_of_2(
        block_size
    ), "block_size should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    if k_out is None:
        k_out = torch.empty((tk, kh, dk), dtype=k.dtype, device=q.device)

    if zeros_out is not None:
        tz, qhz, dz = zeros_out.shape
        assert (
            t == tz and qh == qhz and d == dz
        ), f"q and zeros shape mismatch {q.shape=} {zeros_out.shape=}"
        output_zeros = True
    elif output_zeros:
        zeros_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)
    else:
        zeros_out = None

    if kv_cache_dtype == torch.uint8:
        t_cache, kh_cache, block_size, d_cache = key_cache.shape
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_h = key_cache.stride(1)
        key_cache_stride_d = key_cache.stride(3)
        key_cache_stride_b = key_cache.stride(2)
        key_cache_stride_x = 0
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_h = value_cache.stride(1)
        value_cache_stride_d = value_cache.stride(3)
        value_cache_stride_b = value_cache.stride(2)
        value_cache_stride_slot_chunk = 0
        value_cache_stride_x = 0
        assert (
            key_cache_stride_d == value_cache_stride_d == 1
        ), "The stride of the last dimension of key_cache and value_cache must be 1"
    elif value_shuffle_layout:
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_h = key_cache.stride(1)
        key_cache_stride_d = key_cache.stride(2)
        key_cache_stride_b = key_cache.stride(3)
        key_cache_stride_x = key_cache.stride(4)
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_h = value_cache.stride(1)
        value_cache_stride_d = value_cache.stride(3)
        value_cache_stride_b = 0
        value_cache_stride_slot_chunk = value_cache.stride(2)
        value_cache_stride_x = value_cache.stride(4)
    elif not flash_layout:
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_h = key_cache.stride(1)
        key_cache_stride_d = key_cache.stride(2)
        key_cache_stride_b = key_cache.stride(3)
        key_cache_stride_x = key_cache.stride(4)
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_h = value_cache.stride(1)
        value_cache_stride_d = value_cache.stride(2)
        value_cache_stride_b = value_cache.stride(3)
        value_cache_stride_slot_chunk = 0
        value_cache_stride_x = 0
    else:
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_h = key_cache.stride(2)
        key_cache_stride_d = key_cache.stride(3)
        key_cache_stride_b = key_cache.stride(1)
        key_cache_stride_x = 0
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_h = value_cache.stride(2)
        value_cache_stride_d = value_cache.stride(3)
        value_cache_stride_b = value_cache.stride(1)
        value_cache_stride_slot_chunk = 0
        value_cache_stride_x = 0

    # On gfx1250 the gluon 2D kernel handles all bf16/fp8 cache layouts in
    # one path (BLOCK_T > 1). uint8/NVFP4 still falls back to the Triton
    # kernel (no NVFP4 support in the 2D gluon kernel yet).
    if DEVICE_ARCH == "gfx1250" and kv_cache_dtype != torch.uint8:
        if t < 1024:
            BLOCK_T = 1
        elif t < 2048:
            BLOCK_T = 2
        else:
            BLOCK_T = 16
        n_pid = triton.cdiv(t, BLOCK_T) * qh + triton.cdiv(t_slot - t, BLOCK_T) * kh
        use_gluon = True
    else:
        # 1 q-head per program is 2B/lane; tile heads once the grid is big enough.
        BLOCK_H = 1
        if t >= _BLOCK_H_MIN_TOKENS:
            qh_per_kh = qh // kh
            for cand in (16, 8, 4, 2):
                if qh % cand == 0 and (cand % qh_per_kh == 0 or qh_per_kh % cand == 0):
                    BLOCK_H = cand
                    break
        n_pid = t * (qh // BLOCK_H) + (t_slot - t) * kh
        use_gluon = False
    grid = (n_pid, 1, 1)
    if use_gluon:
        dev, stream = current_device_stream()
        _gluon_fused_qk_rope_reshape_and_cache_kernel_launch()(
            dev,
            stream,
            grid,
            q,
            k,
            v,
            pos,
            cos,
            sin,
            offs,
            key_cache,
            value_cache,
            slot_mapping,
            q_out,
            k_out,
            zeros_out,
            t,
            t_slot,
            max_embd_pos,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            cos.stride(0),
            cos.stride(-1),
            *q_out.stride(),
            *k_out.stride(),
            key_cache_stride_t,
            key_cache_stride_h,
            key_cache_stride_d,
            key_cache_stride_b,
            key_cache_stride_x,
            value_cache_stride_t,
            value_cache_stride_h,
            value_cache_stride_d,
            value_cache_stride_b,
            value_cache_stride_slot_chunk,
            value_cache_stride_x,
            zeros_out.stride(0) if zeros_out is not None else 0,
            zeros_out.stride(1) if zeros_out is not None else 0,
            zeros_out.stride(2) if zeros_out is not None else 0,
            k_scale,
            v_scale,
            qh // kh,
            qh,
            kh,
            reuse_freqs_front_part,
            is_neox,
            d,
            d // 2,
            block_size,
            x_cache if not flash_layout else 0,
            SCALE_K_WIDTH,
            flash_layout,
            value_shuffle_layout,
            offs is not None,
            k_scale is not None and apply_scale,
            v_scale is not None and apply_scale,
            output_zeros,
            upcast_operand,
            BLOCK_T,
        )
    else:
        dev, stream = current_device_stream()
        _triton_fused_qk_rope_reshape_and_cache_kernel_launch(
            dev,
            stream,
            grid,
            q,
            k,
            v,
            pos,
            cos,
            sin,
            offs,
            key_cache,
            value_cache,
            slot_mapping,
            q_out,
            k_out,
            zeros_out,
            t,
            t_slot,
            max_embd_pos,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            cos.stride(0),
            cos.stride(-1),
            *q_out.stride(),
            *k_out.stride(),
            key_cache_stride_t,
            key_cache_stride_h,
            key_cache_stride_d,
            key_cache_stride_b,
            key_cache_stride_x,
            value_cache_stride_t,
            value_cache_stride_h,
            value_cache_stride_d,
            value_cache_stride_b,
            value_cache_stride_slot_chunk,
            value_cache_stride_x,
            zeros_out.stride(0) if zeros_out is not None else 0,
            zeros_out.stride(1) if zeros_out is not None else 0,
            zeros_out.stride(2) if zeros_out is not None else 0,
            k_scale,
            v_scale,
            qh // kh,
            qh,
            kh,
            reuse_freqs_front_part,
            is_neox,
            d,
            d // 2,
            block_size,
            x_cache if not flash_layout else 0,
            SCALE_K_WIDTH,
            flash_layout,
            value_shuffle_layout,
            offs is not None,
            k_scale is not None and apply_scale,
            v_scale is not None and apply_scale,
            output_zeros,
            upcast_operand,
            BLOCK_H,
            max(1, BLOCK_H // (qh // kh)),
        )

    if zeros_out is not None:
        return q_out, k_out, key_cache, value_cache, zeros_out
    return q_out, k_out, key_cache, value_cache


_fused_qk_rope_cosine_cache_llama_kernel_launch = make_launcher(
    _fused_qk_rope_cosine_cache_llama_kernel,
    options={"num_warps": 1},
)


def fused_qk_rope_cosine_cache_llama(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v into key_cache and value_cache inplace

    Key parameters:
    - q: shape (T, QH, D).
    - k: shape (T, KH, D).
    - v: shape (T, KH, D).
    - if flash_layout:
    -     key_cache: shape (T_cache, block_size, KH, D).
    -     value_cache: shape (T_cache, block_size, KH, D).
    - else:
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).
    -     value_cache: shape (T_cache, KH, D, block_size).
    - slot_mapping: shape (T_slot, ).

    T is the number of decode tokens, T_cache * block_size is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: same shape as input q.
    - key_cache: same shape as input key_cache (inplace).
    - value_cache: same shape as input value_cache (inplace).
    """
    _LOGGER.info(
        "FUSED_QK_ROPE_COSINE_CACHE_LLAMA: q=%s k=%s pos=%s cos=%s sin=%s key_cache=%s value_cache=%s slot_mapping=%s",
        tuple(q.shape),
        tuple(k.shape),
        tuple(pos.shape),
        tuple(cos.shape),
        tuple(sin.shape),
        tuple(key_cache.shape),
        tuple(value_cache.shape),
        tuple(slot_mapping.shape),
    )

    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t <= t_slot
    ), f"Number of tokens should be identical for q, k and v. The number of tokens of slot_mapping should be no less than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    n_pid = t * qh + (t_slot - t) * kh
    grid = (n_pid, 1, 1)
    dev, stream = current_device_stream()
    _fused_qk_rope_cosine_cache_llama_kernel_launch(
        dev,
        stream,
        grid,
        q,
        k,
        v,
        pos,
        cos,
        sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        q_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        value_cache.stride(2) if not flash_layout else value_cache.stride(3),
        value_cache.stride(3) if not flash_layout else value_cache.stride(1),
        k_scale,  # k_scale_ptr
        v_scale,  # v_scale_ptr
        qh // kh,  # QH_PER_KH
        qh,  # QH
        kh,  # KH
        reuse_freqs_front_part,  # REUSE_FREQS_FRONT_PART
        is_neox,  # IS_NEOX
        d,  # BLOCK_D_pe
        d // 2,  # BLOCK_D_HALF_pe
        block_size,  # BLOCK_SIZE
        x_cache if not flash_layout else 0,  # X_SIZE
        flash_layout,  # FLASH_LAYOUT
        offs is not None,  # HAVE_POS
        k_scale is not None and apply_scale,  # HAVE_K_SCALE
        v_scale is not None and apply_scale,  # HAVE_V_SCALE
    )
    return q_out, key_cache, value_cache
