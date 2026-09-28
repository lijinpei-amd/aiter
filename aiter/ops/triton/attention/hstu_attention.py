# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2024-2026, The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

# @manual=//triton:triton
import triton

from aiter.ops.triton._triton_kernels.attention.hstu_attention import (
    _get_bwd_config,
    _get_fwd_config,
    _hstu_attn_bwd,
    _hstu_attn_fwd,
)

# @manual=//triton:triton
from aiter.ops.triton.utils.common_utils import (
    prev_power_of_2,
    switch_to_contiguous_if_needed,
)
from aiter.ops.triton.utils.device_info import current_device_stream
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: torch.Tensor | None,
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: torch.Tensor | None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    HSTU attention forward pass with SiLU activation: Y = silu(alpha * (Q @ K^T)) @ V.
    Works with jagged tensors (variable-length sequences concatenated along batch dimension).

    Args:
        N (int): Maximum sequence length across all sequences.
        alpha (float): Scale factor applied to Q @ K^T before SiLU activation.
        q (torch.Tensor): Query jagged tensor with shape (total_tokens, num_heads, head_dim).
        k (torch.Tensor): Key jagged tensor with shape (total_tokens, num_heads, head_dim).
        v (torch.Tensor): Value jagged tensor with shape (total_tokens, num_heads, head_dim).
        seq_offsets (torch.Tensor): Sequence boundaries with shape (batch_size + 1,).
            Element i contains cumulative token count up to sequence i.
        causal (bool): Apply causal masking.
        num_targets (Optional[torch.Tensor]): Number of target tokens per sequence for masking.
        max_attn_len (int): Maximum attention span limit. 0 disables limit.
        contextual_seq_len (int): Contextual prefix length. 0 disables contextual masking.
        sort_by_length_indices (Optional[torch.Tensor]): Indices to process sequences in descending length order.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N).

    Returns:
        torch.Tensor: Output jagged tensor with shape (total_tokens, num_heads, head_dim).
    """
    _LOGGER.info(
        "HSTU_ATTENTION_FWD: N=%d alpha=%f q=%s k=%s  v=%s seq_offsets=%s",
        N,
        alpha,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(seq_offsets.shape),
    )
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    DeltaSize = 0
    IS_DELTA_Q = False

    if config is None:
        config = _get_fwd_config(
            AUTOTUNE_Z,
        )

    grid = (triton.cdiv(N, config["BLOCK_M"]), Z * H)

    dev, stream = current_device_stream()
    _hstu_attn_fwd(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        q,
        k,
        v,
        sort_by_length_indices,
        seq_offsets,
        num_targets,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        alpha,
        H,
        N,
        DeltaSize,
        contextual_seq_len,
        max_attn_len,
        causal,
        has_multiple_targets,
        IS_DELTA_Q,
        torch.backends.cuda.matmul.allow_tf32,
        DimQ,
        DimV,
        config["BLOCK_M"],
        config["BLOCK_N"],
        has_contextual_seq_len,
        has_max_attn_len,
        has_sort_by_length_indices,
    )

    return out


def triton_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
    N: int,
    alpha: float,
    max_attn_len: int,
    causal: float,
    contextual_seq_len: int,
    sort_by_length_indices: torch.Tensor | None,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HSTU attention backward pass computing gradients for Q, K, V.

    Args:
        dout (torch.Tensor): Output gradient with shape (total_tokens, num_heads, head_dim).
        q (torch.Tensor): Query jagged tensor with shape (total_tokens, num_heads, head_dim).
        k (torch.Tensor): Key jagged tensor with shape (total_tokens, num_heads, head_dim).
        v (torch.Tensor): Value jagged tensor with shape (total_tokens, num_heads, head_dim).
        dq (torch.Tensor): Pre-allocated query gradient with shape (total_tokens, num_heads, head_dim).
        dk (torch.Tensor): Pre-allocated key gradient with shape (total_tokens, num_heads, head_dim).
        dv (torch.Tensor): Pre-allocated value gradient with shape (total_tokens, num_heads, head_dim).
        seq_offsets (torch.Tensor): Sequence boundaries with shape (batch_size + 1,).
        num_targets (Optional[torch.Tensor]): Number of target tokens per sequence.
        N (int): Maximum sequence length.
        alpha (float): Scale factor for Q @ K^T.
        max_attn_len (int): Maximum attention span limit.
        causal (float): Apply causal masking.
        contextual_seq_len (int): Contextual prefix length.
        sort_by_length_indices (Optional[torch.Tensor]): Indices for length-sorted processing.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_N, SEQUENCE_PARALLEL).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients (dq, dk, dv).
    """
    _LOGGER.info(
        "HSTU_ATTENTION_BKWD: dout=%s  q=%s k=%s  v=%s dq=%s dk=%s  dv=%s",
        dout.shape,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(dq.shape),
        tuple(dk.shape),
        tuple(dv.shape),
    )
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape

    AUTOTUNE_Z = prev_power_of_2(Z)
    if config is None:
        config = _get_bwd_config(
            AUTOTUNE_Z,
        )

    grid = (
        Z * H,
        triton.cdiv(N, config["BLOCK_N"]) if config["SEQUENCE_PARALLEL"] else 1,
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )

    dq.zero_()
    if config["SEQUENCE_PARALLEL"] == 1:
        lock.zero_()

    dev, stream = current_device_stream()
    _hstu_attn_bwd(
        dev,
        stream,
        grid,
        config.get("num_warps", 4),
        config.get("num_stages", 2),
        config.get("waves_per_eu", 0),
        config.get("matrix_instr_nonkdim", 0),
        config.get("kpack", 1),
        q,
        k,
        v,
        sort_by_length_indices,
        seq_offsets,
        num_targets,
        dout,
        dq,
        dk,
        dv,
        lock,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        dout.stride(0),
        dout.stride(1),
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        alpha,
        contextual_seq_len,
        max_attn_len,
        H,
        N,
        causal,
        num_targets is not None,
        contextual_seq_len > 0,
        max_attn_len > 0,
        torch.backends.cuda.matmul.allow_tf32,
        DimQ,
        DimV,
        config["SEQUENCE_PARALLEL"],
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["UNROLL"],
        sort_by_length_indices is not None,
    )

    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: torch.Tensor | None,
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor) -> tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
            )
