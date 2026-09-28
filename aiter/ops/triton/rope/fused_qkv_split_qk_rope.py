import torch
import triton
from intj import make_launcher

from aiter.ops.triton._triton_kernels.rope.fused_qkv_split_qk_rope import (
    _fused_qkv_split_qk_rope_kernel,
)
from aiter.ops.triton.utils.device_info import current_device_stream

_fused_qkv_split_qk_rope_kernel_launch = make_launcher(
    _fused_qkv_split_qk_rope_kernel,
    options={"num_warps": 4, "waves_per_eu": 0},
)


def fused_qkv_split_qk_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    nope_first: bool = False,
):
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    assert qh >= kvh and qh % kvh == 0, "qh must be mutiple of kvh"

    q = torch.empty((qkv.shape[0], qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)

    if cos.shape[-1] == head_dim // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == head_dim // 4:
        have_nope = True
    else:
        have_nope = False

    assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim // (2 if have_nope else 1) == triton.next_power_of_2(
        head_dim // (2 if have_nope else 1)
    ), "head_dim should be power of 2"

    if have_nope:
        BLOCK_D = head_dim // 2
        BLOCK_D_HALF = head_dim // 4
    else:
        BLOCK_D = head_dim
        BLOCK_D_HALF = head_dim // 2

    BLOCK_T = 32
    grid = (triton.cdiv(T, BLOCK_T), qh, 1)

    dev, stream = current_device_stream()
    _fused_qkv_split_qk_rope_kernel_launch(
        dev,
        stream,
        grid,
        qkv,
        cos,
        sin,
        positions,
        offsets,
        q,
        k,
        v,
        T,
        *qkv.stride(),
        cos.stride(0),
        cos.stride(-1),
        *positions.stride(),
        *q.stride(),
        *k.stride(),
        have_nope,  # HAVE_NOPE
        nope_first,  # NOPE_FIRST
        reuse_freqs_front_part,  # REUSE_FREQS_FRONT_PART
        is_neox,  # IS_NEOX
        positions is not None,  # HAVE_POS
        offsets is not None,  # HAVE_OFFS
        qh,  # QH
        kvh,  # KVH
        BLOCK_T,
        BLOCK_D,
        BLOCK_D_HALF,
    )

    return q, k, v
