# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the gfx950 Gluon MoE grouped GEMMs.

One launcher for every ``moe_gemm_*`` op. ``aiter/ops/triton/moe/__init__.py`` is empty
and there is no per-stage op -- each ``moe_gemm_*`` serves both MoE stages, distinguished
only by its arguments -- so the two-kernel split is expressed as *one* wrapper choosing
between two Gluon entry points, and the operand dtypes are inferred from the tensors
rather than encoded in a separate entry point per op.

Wired into ``moe_op_gemm_a4w4.py``, ``moe_op_gemm_a8w8.py`` and ``moe_op_gemm_a8w4.py``
through :func:`try_gluon_grouped_gemm`.

The arch gate is the ``moe_op_gemm_a8w4.py`` idiom (unconditional import plus an
``get_arch() == "gfx950"`` check), but **not** its hard assert: every combination the
Gluon kernel does not serve falls back to the existing Triton kernel, so no existing
caller silently changes behaviour. :func:`gluon_supported` is the single place that
decides.
"""

from __future__ import annotations

import os
from functools import cache

import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    LDS_USABLE_BYTES,
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DtypeQuant,
    FuncSpec,
    QuantExpertTensor,
    QuantTokenTensor,
    ResultTensor,
    RoutingMeta,
    ScaleSwizzle,
    TileSched,
    TuningSpec,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe.moe_gemm import (
    MoeKernelConfig,
    _moe_gluon_gemm1,
    _moe_gluon_gemm2,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

#: MX group size. Fixed by the OCP microscaling spec for both E2M1 and E4M3; not a knob.
MX_GROUP_SIZE = 32
_SUPPORTED_BLOCK_M = (16, 32, 64, 128)


def _can_overflow_int32(t: torch.Tensor | None, drop_leading: int = 0) -> bool:
    """Largest byte-element offset addressable in the tensor, after optionally dropping
    the leading (expert) axis -- the kernel folds that stride into the 64-bit scalar
    base, so only the remainder has to fit the buffer descriptor's 32-bit window."""
    if t is None:
        return False
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(drop_leading, t.ndim):
        offset += (t.shape[i] - 1) * t.stride(i)
    return offset > max_int32


# --------------------------------------------------------------------------------
# dtype inference
# --------------------------------------------------------------------------------
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)
#: operand dtypes that carry an E8M0 group-32 scale tile alongside the payload
_SCALED = (DtypeQuant.MXFP4, DtypeQuant.MXFP8)
_BF16_DTYPES = (torch.bfloat16, torch.float16)


def infer_dtype_quant(t: torch.Tensor, scales: torch.Tensor | None):
    """Map a stored tensor plus its optional scale tensor onto a :class:`DtypeQuant`.

    The four in-tree MoE ops all describe their operands this way and nothing else
    distinguishes them: MXFP4 is the only one stored as ``uint8`` (two E2M1 per byte),
    fp8 with an E8M0 group-32 scale is MXFP8, fp8 without one is FP8_E4M3 (unit scales,
    which still has to go through ``mfma_scaled`` to reach the double-rate pipe).

    Returns ``None`` for anything outside the supported set, which the caller turns into
    a fallback rather than a failure.
    """
    if t.dtype == torch.uint8:
        return DtypeQuant.MXFP4 if scales is not None else None
    if t.dtype in _FP8_DTYPES:
        return DtypeQuant.MXFP8 if scales is not None else DtypeQuant.FP8_E4M3
    if t.dtype in _BF16_DTYPES:
        return DtypeQuant.BF16 if scales is None else None
    return None


def _mfma_instr(dq_a, dq_b, nonk: int):
    """MFMA instruction shape for an operand pair.

    A bf16 pair is the only one that does *not* land on the f8f6f4 pipe, and that pipe's
    K is a quarter of the scaled one's; getting this wrong does not fail, it silently
    halves throughput.
    """
    both_bf16 = dq_a == DtypeQuant.BF16 and dq_b == DtypeQuant.BF16
    if both_bf16:
        return (32, 32, 16) if nonk == 32 else (16, 16, 32)
    return (32, 32, 64) if nonk == 32 else (16, 16, 128)


def _probe_lds_bytes(cfg: dict, dq_a, dq_b) -> int:
    """LDS footprint of a candidate config, computed by the *aggregate* rather than by a
    second copy of the arithmetic -- the host and the device must not be able to
    disagree about whether a config fits."""
    from triton.experimental.gluon import language as gl

    func = KernelFuncConfig(
        int(dq_a),
        int(dq_b),
        int(dq_a),
        int(dq_b),
        gl.float32,
        None,
        None,
        False,
        False,
        False,
        False,
    )
    return KernelTuningConfig(func, *_tuning_args(cfg)).lds_bytes()


# --------------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------------
@cache
def _get_gluon_config_cached(
    block_m: int, N: int, K: int, dq_a, dq_b, small_grid: bool
) -> tuple:
    return tuple(
        sorted(get_gluon_config_uncached(block_m, N, K, dq_a, dq_b, small_grid).items())
    )


def get_gluon_config(
    block_m: int, N: int, K: int, dq_a, dq_b, small_grid: bool = False
) -> dict:
    """Cached view of :func:`get_gluon_config_uncached`. The decode path issues one of
    these per launch and the launch path *is* the critical path there."""
    return dict(_get_gluon_config_cached(block_m, N, K, dq_a, dq_b, bool(small_grid)))


def get_gluon_config_uncached(
    block_m: int, N: int, K: int, dq_a, dq_b, small_grid: bool = False
) -> dict:
    """Explicit Python ladder keyed on ``(block_m, N, K, dtypes, small_grid)``.

    ``aiter/ops/triton/configs/gfx950/gluon/moe/`` is empty and the three in-tree config
    mechanisms are mutually incompatible, so this starts as a ladder; a JSON resolver
    under ``{arch}/gluon/moe/{dtype}/`` can replace the body without touching callers.

    ``BLOCK_M`` is an input, not a knob: the router picks
    ``max(16, min(next_pow2(M // n_expts_tot), 128))`` and the kernel must serve it, so
    it never appears on the left of this ladder.

    The two regimes really are different kernels in everything but source:

    * ``block_m in (16, 32)`` -- decode / small-batch. Weight streaming dominates and
      there is no MFMA-utilisation story (at ``BLOCK_M=16`` with 384 experts, M
      occupancy inside a block is ~6%), so the levers are a long sequential K stream
      (``BLOCK_K=512``) and few warps (4) so each one owns a wide slice. ``BLOCK_K=512``
      also makes the A-scale tile large enough to be written to LDS coalesced, which
      removes the register-path scale load from the K loop -- and register-path global
      accesses in the loop make every ``wait_group`` conservative, because on CDNA4
      direct-to-LDS completes *in order* with ordinary loads.
    * ``block_m in (64, 128)`` -- prefill. MFMA bound, so the wide N tile and warps
      spread over both axes.

    The tile is then shrunk until it fits LDS. That step is not cosmetic: the base
    numbers are tuned for MXFP4, and an MXFP8 or bf16 operand is two or four times
    wider per element, so the same ``(BLOCK_N, BLOCK_K)`` would overflow the 160 KiB
    cap outright.
    """
    if block_m == 16:
        block_n, block_k, warps, nb = 128, 512, (1, 4), 3
        nonk = 16
        if small_grid:
            # A narrow N tile is the only way to keep every XCD busy when the router
            # hands out only a few M blocks.
            block_n = 64
    elif block_m == 32:
        block_n, block_k, warps, nb = 256, 512, (1, 4), 2
        nonk = 16
    elif block_m == 64:
        block_n, block_k, warps, nb = 256, 256, (2, 4), 2
        nonk = 32
    else:  # 128
        block_n, block_k, warps, nb = 256, 256, (4, 2), 2
        nonk = 32
    instr = _mfma_instr(dq_a, dq_b, nonk)

    # shrink the N tile until it divides N (the kernel has no N tail by construction)
    gran_n = instr[1] * warps[1]
    while block_n > gran_n and N % block_n != 0:
        block_n //= 2
    min_k = max(128, instr[2])
    while block_k > min_k and (K % block_k != 0 or K // block_k < nb):
        block_k //= 2

    def _build(bn, bk, n_buf):
        mini_n = min(bn, gran_n * 2)
        while mini_n % gran_n:
            mini_n += gran_n
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "K_UNROLL": n_buf,
            "MINI_BLOCK_K": bk,
            "MINI_PREFETCH_K": 0,
            "MINI_BLOCK_M": block_m,
            "MINI_BLOCK_N": mini_n,
            "MINI_PRESTORE_MN": 1,
            "NUM_LDS_BUFFER": n_buf,
            "mfma_instr_shape": instr,
            "warps_per_cta": warps,
            "tiles_per_warp": (1, 1),
            # None means "derive from the instruction shape and the operand packing".
            # A literal here would silently pick a different MFMA variant.
            "k_width": None,
            "transposed": True,
            "WAVES_PER_EU": 0,
            "TILE_SCHED": int(TileSched.XCD_GROUP_M),
            "GROUP_M": 4,
            "NUM_XCDS": 8,
            "token_mod": "",
            "token_scale_mod": "",
            "expert_mod": ".cg" if block_m <= 32 else "",
            "expert_scale_mod": ".cg" if block_m <= 32 else "",
            "result_mod": "",
            "result_scale_mod": "",
            "WARP_PIPELINE": False,
        }

    # Which axis to give up first when the tile does not fit.
    #
    # With microscaled operands, keep BLOCK_K >= 256: below that the E8M0 scale tiles
    # drop under the width a coalesced direct-to-LDS write needs, which puts a
    # register-path global load back inside the K loop and makes every wait_group
    # conservative (measured 0.6x of the Triton kernel on MXFP8 prefill). So narrow N
    # first and accept the loss of arithmetic intensity.
    #
    # With no scales at all (bf16 x bf16) that argument does not exist, and narrowing N
    # is pure loss -- it was picking BLOCK_N=64 where BLOCK_N=128 with a shorter K fits
    # just as well. Shorten K first.
    shrink_n_first = dq_a in _SCALED or dq_b in _SCALED

    cfg = _build(block_n, block_k, nb)
    while _probe_lds_bytes(cfg, dq_a, dq_b) > LDS_USABLE_BYTES:
        can_n = block_n > gran_n and N % (block_n // 2) == 0
        can_k = block_k > min_k and K % (block_k // 2) == 0
        first, second = (can_n, can_k) if shrink_n_first else (can_k, can_n)
        if first:
            if shrink_n_first:
                block_n //= 2
            else:
                block_k //= 2
        elif nb > 2:
            nb -= 1
        elif second:
            if shrink_n_first:
                block_k //= 2
            else:
                block_n //= 2
        else:
            break  # gluon_supported's validate() will reject it
        if K // block_k < nb:
            nb = max(2, K // block_k)
        cfg = _build(block_n, block_k, nb)
    return cfg


def _small_grid(routing_data, M, N) -> bool:
    """True when the launch would not fill the machine at the default N tile."""
    grid_m = routing_data.n_blocks(M, routing_data.block_m)
    return grid_m * max(1, N // 128) < 512


def gluon_supported(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor | None,
    w_scales: torch.Tensor | None,
    y: torch.Tensor,
    bias: torch.Tensor | None,
    routing_data,
    swizzle_mx_scale,
    split_k: int,
    x_static_scale,
    quant_static_scale,
    out_quant,
    N: int,
    K: int,
) -> tuple[bool, str]:
    """The capability predicate. Returns ``(ok, reason)``; ``reason`` is logged when
    the call falls back to the Triton kernel."""
    if os.environ.get("AITER_TRITON_MOE_DISABLE_GLUON", "0") != "0":
        return False, "disabled by AITER_TRITON_MOE_DISABLE_GLUON"
    if get_arch() != "gfx950":
        return False, "arch is not gfx950"
    if split_k != 1:
        return False, "split_k > 1 has no Gluon path (the fused epilogue forecloses it)"
    dq_a = infer_dtype_quant(x, x_scales)
    dq_b = infer_dtype_quant(w, w_scales)
    if dq_a is None or dq_b is None:
        return False, f"unsupported operand dtypes ({x.dtype} / {w.dtype})"
    if quant_static_scale is not None:
        return False, "fused fp8 output quant is not implemented on the Gluon path"
    if x_static_scale is not None and dq_a != DtypeQuant.FP8_E4M3:
        return False, "a static activation scale only applies to unscaled fp8 operands"
    if out_quant not in (None, DtypeQuant.MXFP4):
        return False, f"output_quant {out_quant} is not implemented"
    if swizzle_mx_scale is not None:
        # CDNA4_SCALE keeps the direct-to-LDS write coalesced but needs the descriptor
        # reshape/permute of the preshuffled tile; not implemented yet.
        return False, f"scale swizzle {swizzle_mx_scale} is not implemented"
    if routing_data is None or routing_data.expt_data is None:
        return False, "Gluon path needs the Family-A routing metadata"
    if x.stride(-1) != 1:
        return False, "x must be row-major"
    if w.stride(-2) != 1:
        return False, "w must be K-contiguous ((E, K/pack, N) with stride(-2) == 1)"
    if bias is not None and bias.dtype != torch.float32:
        return False, "bias must be fp32"
    # bf16 x microscaled needs scaled_upcast + plain mfma; see moe_gemm._load_operands.
    mixed_bf16 = (dq_a == DtypeQuant.BF16) != (dq_b == DtypeQuant.BF16)
    if mixed_bf16:
        return False, "bf16 x microscaled (scaled_upcast path) is not implemented"

    block_m = routing_data.block_m
    if block_m not in _SUPPORTED_BLOCK_M:
        return False, f"block_m {block_m} outside {_SUPPORTED_BLOCK_M}"

    cfg = get_gluon_config(
        block_m, N, K, dq_a, dq_b, _small_grid(routing_data, y.shape[1], N)
    )
    if N % cfg["BLOCK_N"] != 0:
        return False, f"N {N} % BLOCK_N {cfg['BLOCK_N']} != 0"
    if K % cfg["BLOCK_K"] != 0:
        return False, f"K {K} % BLOCK_K {cfg['BLOCK_K']} != 0"
    if K // cfg["BLOCK_K"] < cfg["NUM_LDS_BUFFER"]:
        return False, "K strip shorter than the pipeline depth"
    if _probe_lds_bytes(cfg, dq_a, dq_b) > LDS_USABLE_BYTES:
        return False, "no tile of this shape fits the LDS budget"
    if (dq_a == DtypeQuant.MXFP4 or dq_b == DtypeQuant.MXFP4) and K % 64 != 0:
        return False, "K must be a multiple of 64 for packed E2M1 + group-32 scales"
    if K % MX_GROUP_SIZE != 0:
        return False, "K must be a multiple of the 32-element MX group"

    # 2 GB buffer window. The expert stride is folded into the 64-bit scalar base, so
    # only the per-expert slice has to fit; everything else is checked whole.
    if _can_overflow_int32(w, drop_leading=1):
        return False, "per-expert weight slice exceeds the 2 GB buffer window"
    if _can_overflow_int32(w_scales, drop_leading=1):
        return False, "per-expert weight-scale slice exceeds the 2 GB buffer window"
    for t, name in ((x, "x"), (x_scales, "x_scales"), (y, "y")):
        if _can_overflow_int32(t):
            return False, f"{name} exceeds the 2 GB buffer window"
    return True, ""


# --------------------------------------------------------------------------------
# launch
# --------------------------------------------------------------------------------
@cache
def _launch_spec(
    block_m,
    N,
    K,
    dq_a,
    dq_b,
    small_grid,
    has_bias,
    has_gammas,
    has_gather,
    has_x_static_scale,
    act,
    out_quant,
    config_items,
):
    """Host-side constexpr work, memoised.

    Building the two aggregates plus the 36 ``gl.constexpr`` wrappers costs ~65 us of
    Python, which at decode is comparable to the whole kernel. Everything here depends
    only on compile-time values, so it is computed once per distinct configuration.
    """
    from triton.experimental.gluon import language as gl

    c = (
        dict(config_items)
        if config_items
        else get_gluon_config(block_m, N, K, dq_a, dq_b, small_grid)
    )
    func_spec = FuncSpec(
        int(dq_a),
        int(dq_b),
        int(dq_a),
        int(dq_b),
        gl.float32,
        act,
        out_quant,
        has_bias,
        has_gammas,
        has_gather,
        has_x_static_scale,
    )
    tuning_spec = TuningSpec(*(_hashable(c[k]) for k in _TUNING_KEYS))
    # Construct the tuning config on the host off exactly the numbers the kernel will
    # use, so the grid math and the tile math cannot drift.
    func_cfg_host = KernelFuncConfig(*func_spec)
    tuning_cfg_host = KernelTuningConfig(func_cfg_host, *tuning_spec)
    grid_n = tuning_cfg_host.grid_N(N)
    grid_n = grid_n.value if hasattr(grid_n, "value") else grid_n
    kcfg = MoeKernelConfig(
        gl.constexpr(func_spec),
        gl.constexpr(tuning_spec),
        gl.constexpr(N),
        gl.constexpr(K),
    )
    num_warps = c["warps_per_cta"][0] * c["warps_per_cta"][1]
    return grid_n, kcfg, num_warps, c["WAVES_PER_EU"], c


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def moe_gemm_gluon(
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias: torch.Tensor | None,
    gammas: torch.Tensor | None,
    routing_data,
    gather_indx: torch.Tensor | None,
    scatter_indx: torch.Tensor | None,
    N: int,
    K: int,
    apply_swiglu: bool,
    alpha: float,
    limit: float | None,
    swiglu_add_residual: bool,
    x_static_scale: torch.Tensor | None = None,
    y_scales: torch.Tensor | None = None,
    config: dict | None = None,
):
    """Launch the Gluon grouped GEMM into ``y`` (shape ``(1, M, N // ARN)``).

    Operand dtypes are inferred from the tensors -- ``uint8`` + scales is MXFP4, fp8 +
    scales is MXFP8, fp8 alone is FP8_E4M3 (unit scales, still on the scaled pipe),
    bf16 alone is BF16 -- so every ``moe_gemm_*`` op calls this one function.
    :func:`gluon_supported` must have said yes for these tensors first.

    ``y_scales`` non-None selects the fused MXFP4 output quant: ``y`` then holds the
    E2M1 payload (``N // ARN // 2`` uint8 columns) and ``y_scales`` the E8M0 exponents,
    bit-identical to what the standalone ``mxfp4_quant`` launch produces.

    Returns the compiled kernel handle so callers (the ISA-assertion test) can inspect
    ``.asm``; the result itself is written into ``y`` / ``y_scales``.
    """
    block_m = routing_data.block_m
    expt_data = routing_data.expt_data
    grid_m = routing_data.n_blocks(y.shape[1], block_m)
    dq_a = infer_dtype_quant(x, x_scales)
    dq_b = infer_dtype_quant(w, w_scales)

    if apply_swiglu:
        act = ActivationSpec(
            kind=int(ActKind.SILU if alpha == 1.0 else ActKind.SWIGLU_OAI),
            alpha=alpha,
            limit=limit,
            add_residual=bool(swiglu_add_residual),
        )
    else:
        act = None
    out_quant = int(DtypeQuant.MXFP4) if y_scales is not None else None

    grid_n, kcfg, num_warps, waves_per_eu, _cfg = _launch_spec(
        block_m,
        N,
        K,
        dq_a,
        dq_b,
        _small_grid(routing_data, y.shape[1], N),
        bias is not None,
        gammas is not None,
        gather_indx is not None,
        x_static_scale is not None,
        act,
        out_quant,
        tuple(sorted((k, _hashable(v)) for k, v in config.items())) if config else None,
    )

    # The Quant* tuples carry the scale pointer and strides unconditionally; when the
    # operand has no scale the kernel never reads them, so a null pointer and zero
    # strides keep one tuple type for every dtype instead of four launch sites.
    a = QuantTokenTensor.make(
        dq_a,
        x,
        x_scales,
        x.shape[0],
        x.stride(0),
        0 if x_scales is None else x_scales.stride(0),
        0 if x_scales is None else x_scales.stride(1),
        K,
        routing_data.n_expts_act,
        ScaleSwizzle.NONE,
    )
    b = QuantExpertTensor.make(
        dq_b,
        w,
        w_scales,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        0 if w_scales is None else w_scales.stride(0),
        0 if w_scales is None else w_scales.stride(2),
        0 if w_scales is None else w_scales.stride(1),
        w.shape[0],
        K,
        N,
        ScaleSwizzle.NONE,
    )
    res = ResultTensor.make(
        out_quant if out_quant is not None else int(DtypeQuant.BF16),
        y,
        y_scales,
        y.stride(1),
        y.stride(2),
        0 if y_scales is None else y_scales.stride(0),
        0 if y_scales is None else y_scales.stride(1),
        N // (2 if apply_swiglu else 1),
    )
    rt = RoutingMeta.make(
        expt_data.block_pid_map,
        expt_data.hist,
        expt_data.token_offs_raw,
        expt_data.token_offs_pad[-1],
        gather_indx,
        scatter_indx,
        gammas,
        routing_data.n_expts_act,
    )

    kernel = _moe_gluon_gemm1 if apply_swiglu else _moe_gluon_gemm2
    return kernel[(grid_m * grid_n,)](
        a,
        b,
        res,
        rt,
        bias,
        0 if bias is None else bias.stride(0),
        x_static_scale,
        grid_m,
        grid_n,
        kcfg,
        KernelFuncConfig,
        KernelTuningConfig,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )


def try_gluon_grouped_gemm(
    *,
    op_name: str,
    y: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor | None,
    w_scales: torch.Tensor | None,
    bias: torch.Tensor | None,
    gammas: torch.Tensor | None,
    routing_data,
    gather_indx: torch.Tensor | None,
    scatter_indx: torch.Tensor | None,
    N: int,
    K: int,
    apply_swiglu: bool,
    alpha: float,
    limit: float | None,
    swiglu_add_residual: bool,
    split_k: int,
    x_static_scale=None,
    quant_static_scale=None,
    swizzle_mx_scale=None,
) -> bool:
    """The one hook every ``moe_gemm_*`` wrapper calls.

    Returns True when the Gluon kernel has written ``y``, False when the caller must
    fall through to its own Triton kernel. Capability-gated and never asserted: an
    unsupported combination is a fallback, not a failure, so no existing caller silently
    changes behaviour.
    """
    ok, why = gluon_supported(
        x=x,
        w=w,
        x_scales=x_scales,
        w_scales=w_scales,
        y=y,
        bias=bias,
        routing_data=routing_data,
        swizzle_mx_scale=swizzle_mx_scale,
        split_k=split_k,
        x_static_scale=x_static_scale,
        quant_static_scale=quant_static_scale,
        out_quant=None,
        N=N,
        K=K,
    )
    if not ok:
        _LOGGER.debug(f"{op_name}: falling back to the Triton kernel: {why}")
        return False
    moe_gemm_gluon(
        y,
        x,
        w,
        x_scales,
        w_scales,
        bias,
        gammas,
        routing_data,
        gather_indx,
        scatter_indx,
        N,
        K,
        apply_swiglu,
        alpha,
        limit,
        swiglu_add_residual,
        x_static_scale=x_static_scale,
    )
    return True


def moe_gemm1_a4w4_mxfp4_out(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias: torch.Tensor | None,
    routing_data,
    gather_indx: torch.Tensor | None,
    gammas: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    limit: float | None = None,
    swiglu_add_residual: bool = False,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gemm1 with the activation **and** the MXFP4 output quant fused into the epilogue.

    Returns ``(y_fp4, y_scales)`` in exactly gemm2's operand-A format: E2M1 payload
    packed 2/byte along the emitted N axis, uint8 E8M0 scales per group of 32. This is
    the whole point of the two-kernel split -- today's flow writes bf16, reads it back
    and runs a third full pass (``mxfp4_quant``, which even upcasts to fp32 first),
    roughly 2.5x the necessary intermediate traffic.

    Raises if the shape is outside the Gluon path; there is no silent fallback here
    because the caller is asking for the fused format specifically.
    """
    M = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    N = w.shape[-1]
    K = x.shape[-1] * 2
    out_n = N // 2  # gated activation halves the emitted width

    y = torch.empty((1, M, out_n // 2), dtype=torch.uint8, device=x.device)
    y_scales = torch.empty(
        (M, out_n // MX_GROUP_SIZE), dtype=torch.uint8, device=x.device
    )
    ok, why = gluon_supported(
        x=x,
        w=w,
        x_scales=x_scales,
        w_scales=w_scales,
        y=y,
        bias=bias,
        routing_data=routing_data,
        swizzle_mx_scale=None,
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=DtypeQuant.MXFP4,
        N=N,
        K=K,
    )
    if not ok:
        raise NotImplementedError(f"fused MXFP4 gemm1 not available: {why}")

    moe_gemm_gluon(
        y,
        x,
        w,
        x_scales,
        w_scales,
        bias,
        gammas,
        routing_data,
        gather_indx,
        None,
        N,
        K,
        True,
        alpha,
        limit,
        swiglu_add_residual,
        y_scales=y_scales,
        config=config,
    )
    return y[0], y_scales


_TUNING_KEYS = (
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "K_UNROLL",
    "MINI_BLOCK_K",
    "MINI_PREFETCH_K",
    "MINI_BLOCK_M",
    "MINI_BLOCK_N",
    "MINI_PRESTORE_MN",
    "NUM_LDS_BUFFER",
    "mfma_instr_shape",
    "warps_per_cta",
    "tiles_per_warp",
    "k_width",
    "transposed",
    "WAVES_PER_EU",
    "TILE_SCHED",
    "GROUP_M",
    "NUM_XCDS",
    "token_mod",
    "token_scale_mod",
    "expert_mod",
    "expert_scale_mod",
    "result_mod",
    "result_scale_mod",
    "WARP_PIPELINE",
)


def _tuning_args(c: dict) -> tuple:
    return tuple(c[k] for k in _TUNING_KEYS)
