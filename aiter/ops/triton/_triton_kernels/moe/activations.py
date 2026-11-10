# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from triton.language.extra.libdevice import fast_dividef

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_clip_repr = make_kernel_repr("moe_activation_clip", ["clip_lower"])
_swiglu_repr = make_kernel_repr("moe_swiglu", ["ADD_RESIDUAL"])
_silu_repr = make_kernel_repr("moe_silu", [])
_silu_grad_repr = make_kernel_repr("moe_silu_grad", [])
_gelu_tanh_repr = make_kernel_repr("moe_gelu_tanh", [])
_gelu_tanh_grad_repr = make_kernel_repr("moe_gelu_tanh_grad", [])
_relu_repr = make_kernel_repr("moe_relu", [])
_relu_grad_repr = make_kernel_repr("moe_relu_grad", [])
_relu_sq_repr = make_kernel_repr("moe_relu_sq", [])
_relu_sq_grad_repr = make_kernel_repr("moe_relu_sq_grad", [])


def swiglu_group2_perm(n: int):
    """Column permutation that turns the (g,l,g,l) packing into (g,g,l,l).

    ``new[c] = old[perm[c]]`` over the N axis. Within each group of four the middle
    two swap: gate_2i, gate_2i+1, up_2i, up_2i+1 comes from old 4i, 4i+2, 4i+1, 4i+3.
    Apply to every N-indexed operand (weights, weight scales, bias) and pass
    ``GROUP=2`` to :func:`_swiglu`; the emitted output order is unchanged.
    """
    import torch

    base = torch.arange(n).view(-1, 4)
    return base[:, [0, 2, 1, 3]].reshape(-1)


def gate_up_split_perm(n: int):
    """Column permutation that turns the (g,l,g,l) packing into halves: [gate | up].

    ``new[c] = old[perm[c]]`` over the N axis, so the first ``n//2`` columns are every
    even (gate) column in order and the last ``n//2`` every odd (up) one -- the layout
    FlyDSL's gemm1 reads. Apply to every N-indexed operand (weights, weight scales,
    bias) and pass ``gate_up_split=True`` to the Gluon gemm1; the emitted output order
    is unchanged, so the result is bit-identical to the interleaved form.

    Unlike :func:`swiglu_group2_perm` this is not a within-quad shuffle: it makes each
    mini-N block a whole operand side, which is what puts a full 32-channel MX group
    inside one wave.
    """
    import torch

    return torch.cat([torch.arange(0, n, 2), torch.arange(1, n, 2)])


@triton.jit(repr=_clip_repr)
def clip(x, limit, clip_lower: tl.constexpr):
    # Do not use inline asm here. LLVM adds no MFMA wait states before an inline asm,
    # so it can read an accumulator before the MFMA writes it.
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu_gate(gelu, alpha, limit, FAST_RCP: tl.constexpr = True):
    """``silu(gelu)`` alone -- the half that owns every transcendental.

    Separate from :func:`_swiglu_combine` so a caller that gets the gate operand before
    the linear one can start the exp2/rcp early; the gate/up-split gemm1 issues this
    between the two mini-N MFMA clusters so it retires under them.
    """
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    denom = 1 + tl.exp2(-1.44269504089 * alpha * gelu)
    if FAST_RCP:
        # Hardware v_rcp_f32 instead of the IEEE divide. A true `/` lowers to
        # v_div_scale_f32 plus a Newton refinement (v_fma / v_div_fmas / v_div_fixup);
        # this leaves just v_rcp_f32 + v_mul_f32. ~1 ulp, well inside the bf16 the
        # result is stored as, and what the FlyDSL port does (rocdl.rcp).
        # tl.fdiv(ieee_rounding=False) does NOT do this -- on the AMD backend it lowers
        # identically to the IEEE form, so the libdevice call is the only route.
        return fast_dividef(gelu, denom)
    return gelu / denom


@triton.jit
def _swiglu_combine(s, linear, limit, ADD_RESIDUAL: tl.constexpr):
    """``s * linear``, with the linear side's clamp. The other half of the activation."""
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    return s * linear


@triton.jit
def _swiglu_pair(
    gelu,
    linear,
    alpha,
    limit,
    ADD_RESIDUAL: tl.constexpr,
    FAST_RCP: tl.constexpr = True,
):
    """The activation itself, on gate/linear that are already separated.

    Split out of :func:`_swiglu` so a caller whose two halves arrive as two whole
    tensors -- the gate/up-split gemm1, where they are two mini-N accumulators with
    identical layouts -- can skip the reshape/split entirely. Bit-identical to the
    tail of :func:`_swiglu`, which now calls this.
    """
    return _swiglu_combine(
        _swiglu_gate(gelu, alpha, limit, FAST_RCP), linear, limit, ADD_RESIDUAL
    )


@triton.jit(repr=_swiglu_repr)
def _swiglu(
    input,
    alpha,
    limit,
    ADD_RESIDUAL: tl.constexpr,
    FAST_RCP: tl.constexpr = True,
    GROUP: tl.constexpr = 1,
):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.

    ``FAST_RCP`` swaps the IEEE divide for the hardware reciprocal; see the comment at
    the use site. On by default, as upstream's divide is; the gfx950 Gluon gemm1 passes
    its tuned value.
    """
    if GROUP == 2:
        # Gate/linear interleaved in PAIRS along N (g,g,l,l) instead of singly
        # (g,l,g,l). The caller's weights must be packed to match -- see
        # ``swiglu_group2_perm``. The point is register placement: with the singly
        # interleaved form a lane's accumulator quad holds g,l,g,l, so the gate values
        # sit at stride 2 and every v_pk_* needs two v_mov_b32 to gather them into an
        # aligned pair. Grouped by 2 the split falls on a register-pair boundary.
        #
        # Index arithmetic for the 4-D view (i, h, e) -> 4i + 2h + e:
        #   split on e  -> a = 4i+2h,      b = 4i+2h+1
        #   split each on h -> gate = 4i, 4i+1   linear = 4i+2, 4i+3
        n4: tl.constexpr = input.shape[1] // 4
        a, b = tl.split(tl.reshape(input, (input.shape[0], n4, 2, 2)))
        g0, l0 = tl.split(a)
        g1, l1 = tl.split(b)
        gelu = tl.reshape(tl.join(g0, g1), (input.shape[0], input.shape[1] // 2))
        linear = tl.reshape(tl.join(l0, l1), (input.shape[0], input.shape[1] // 2))
    else:
        gelu, linear = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
    return _swiglu_pair(gelu, linear, alpha, limit, ADD_RESIDUAL, FAST_RCP)


@triton.jit(repr=_silu_repr)
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit(repr=_silu_grad_repr)
def silu_grad(x):
    sigmoid = tl.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


@triton.jit(repr=_gelu_tanh_repr)
def gelu_tanh(x):
    sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
    coeff: tl.constexpr = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    return 0.5 * x * (1.0 + tl.extra.hip.libdevice.tanh(inner))


@triton.jit(repr=_gelu_tanh_grad_repr)
def gelu_tanh_grad(x):
    sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
    coeff: tl.constexpr = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    tanh_value = tl.extra.hip.libdevice.tanh(inner)
    derivative = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
    return (
        0.5 * (1.0 + tanh_value)
        + 0.5 * x * (1.0 - tanh_value * tanh_value) * derivative
    )


@triton.jit(repr=_relu_repr)
def relu(x):
    return tl.where(x > 0, x, 0.0)


@triton.jit(repr=_relu_grad_repr)
def relu_grad(x):
    return tl.where(x > 0, 1.0, 0.0)


@triton.jit(repr=_relu_sq_repr)
def relu_sq(x):
    value = relu(x)
    return value * value


@triton.jit(repr=_relu_sq_grad_repr)
def relu_sq_grad(x):
    return tl.where(x > 0, 2.0 * x, 0.0)


_glu_fwd_repr = make_kernel_repr(
    "sonicmoe_glu_fwd", ["I", "BLOCK_M", "BLOCK_I", "CONCAT_LAYOUT", "ACT_TYPE"]
)
_glu_bwd_repr = make_kernel_repr(
    "sonicmoe_glu_bwd", ["I", "BLOCK_M", "BLOCK_I", "CONCAT_LAYOUT", "ACT_TYPE"]
)
_pointwise_act_fwd_repr = make_kernel_repr(
    "sonicmoe_pointwise_act_fwd", ["I", "BLOCK_M", "BLOCK_I", "ACT_TYPE"]
)
_pointwise_act_bwd_repr = make_kernel_repr(
    "sonicmoe_pointwise_act_bwd", ["I", "BLOCK_M", "BLOCK_I", "ACT_TYPE"]
)


@triton.jit(repr=_glu_fwd_repr)
def _glu_fwd_kernel(
    h_ptr,
    a_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_a_m,
    stride_a_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    if CONCAT_LAYOUT:
        gate_offs = offs_i
        up_offs = offs_i + I
    else:
        gate_offs = offs_i * 2
        up_offs = offs_i * 2 + 1

    gate = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + gate_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + up_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 0:  # swiglu
        act_gate = silu(gate)
    elif ACT_TYPE == 1:  # geglu (tanh approx)
        act_gate = gelu_tanh(gate)
    elif ACT_TYPE == 2:  # reglu
        act_gate = relu(gate)

    out = act_gate * up

    tl.store(
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_a_m
        + offs_i[None, :].to(tl.int64) * stride_a_i,
        out.to(a_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit(repr=_glu_bwd_repr)
def _glu_bwd_kernel(
    h_ptr,
    dh_ptr,
    da_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_dh_m,
    stride_dh_i,
    stride_da_m,
    stride_da_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    if CONCAT_LAYOUT:
        gate_offs = offs_i
        up_offs = offs_i + I
    else:
        gate_offs = offs_i * 2
        up_offs = offs_i * 2 + 1

    gate = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + gate_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + up_offs[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    da = tl.load(
        da_ptr
        + offs_m[:, None].to(tl.int64) * stride_da_m
        + offs_i[None, :].to(tl.int64) * stride_da_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 0:  # swiglu
        d_up = da * silu(gate)
        d_gate = da * up * silu_grad(gate)
    elif ACT_TYPE == 1:  # geglu (tanh approx)
        d_up = da * gelu_tanh(gate)
        d_gate = da * up * gelu_tanh_grad(gate)
    elif ACT_TYPE == 2:  # reglu
        d_up = da * relu(gate)
        d_gate = da * up * relu_grad(gate)

    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + gate_offs[None, :].to(tl.int64) * stride_dh_i,
        d_gate.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )
    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + up_offs[None, :].to(tl.int64) * stride_dh_i,
        d_up.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit(repr=_pointwise_act_fwd_repr)
def _pointwise_act_fwd_kernel(
    h_ptr,
    a_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_a_m,
    stride_a_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    x = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + offs_i[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 3:  # gelu (tanh approx)
        out = gelu_tanh(x)
    elif ACT_TYPE == 4:  # relu
        out = relu(x)
    elif ACT_TYPE == 5:  # silu
        out = silu(x)
    elif ACT_TYPE == 6:  # relu_sq
        out = relu_sq(x)

    tl.store(
        a_ptr
        + offs_m[:, None].to(tl.int64) * stride_a_m
        + offs_i[None, :].to(tl.int64) * stride_a_i,
        out.to(a_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )


@triton.jit(repr=_pointwise_act_bwd_repr)
def _pointwise_act_bwd_kernel(
    h_ptr,
    dh_ptr,
    da_ptr,
    TK,
    I: tl.constexpr,
    stride_h_m,
    stride_h_i,
    stride_dh_m,
    stride_dh_i,
    stride_da_m,
    stride_da_i,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    m_mask = offs_m < TK
    i_mask = offs_i < I

    x = tl.load(
        h_ptr
        + offs_m[:, None].to(tl.int64) * stride_h_m
        + offs_i[None, :].to(tl.int64) * stride_h_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    da = tl.load(
        da_ptr
        + offs_m[:, None].to(tl.int64) * stride_da_m
        + offs_i[None, :].to(tl.int64) * stride_da_i,
        mask=m_mask[:, None] & i_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if ACT_TYPE == 3:  # gelu (tanh approx)
        dx = da * gelu_tanh_grad(x)
    elif ACT_TYPE == 4:  # relu
        dx = da * relu_grad(x)
    elif ACT_TYPE == 5:  # silu
        dx = da * silu_grad(x)
    elif ACT_TYPE == 6:  # relu_sq
        dx = da * relu_sq_grad(x)

    tl.store(
        dh_ptr
        + offs_m[:, None].to(tl.int64) * stride_dh_m
        + offs_i[None, :].to(tl.int64) * stride_dh_i,
        dx.to(dh_ptr.dtype.element_ty),
        mask=m_mask[:, None] & i_mask[None, :],
    )
