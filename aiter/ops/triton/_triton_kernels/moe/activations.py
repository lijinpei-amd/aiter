import triton
import triton.language as tl
from triton.language.extra.hip import libdevice as hip_libdevice


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


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu_gate(gelu, alpha, limit, FAST_RCP: tl.constexpr = False):
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
        return hip_libdevice.fast_dividef(gelu, denom)
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
def _swiglu_pair(gelu, linear, alpha, limit, ADD_RESIDUAL: tl.constexpr,
                 FAST_RCP: tl.constexpr = False):
    """The activation itself, on gate/linear that are already separated.

    Split out of :func:`_swiglu` so a caller whose two halves arrive as two whole
    tensors -- the gate/up-split gemm1, where they are two mini-N accumulators with
    identical layouts -- can skip the reshape/split entirely. Bit-identical to the
    tail of :func:`_swiglu`, which now calls this.
    """
    return _swiglu_combine(
        _swiglu_gate(gelu, alpha, limit, FAST_RCP), linear, limit, ADD_RESIDUAL
    )


@triton.jit
def _swiglu(input, alpha, limit, ADD_RESIDUAL: tl.constexpr, FAST_RCP: tl.constexpr = False,
            GROUP: tl.constexpr = 1):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.

    ``FAST_RCP`` swaps the IEEE divide for the hardware reciprocal; see the comment at
    the use site. Off by default so no existing caller's numerics move.
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
