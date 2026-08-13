# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the gfx950 Gluon MoE A4W4 grouped GEMM.

``aiter/ops/triton/moe/__init__.py`` is empty and there is no per-stage op --
``moe_gemm_a4w4`` serves both stages, distinguished only by its arguments -- so the
two-kernel split is expressed as *one* wrapper choosing between two Gluon entry points.

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

MXFP4_QUANT_BLOCK_SIZE = 32
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
# tuning
# --------------------------------------------------------------------------------
@cache
def _get_gluon_config_cached(block_m: int, N: int, K: int, small_grid: bool) -> tuple:
    return tuple(sorted(get_gluon_config_uncached(block_m, N, K, small_grid).items()))


def get_gluon_config(block_m: int, N: int, K: int, small_grid: bool = False) -> dict:
    """Cached view of :func:`get_gluon_config_uncached`. The decode path issues one of
    these per launch and the launch path *is* the critical path there."""
    return dict(_get_gluon_config_cached(block_m, N, K, bool(small_grid)))


def get_gluon_config_uncached(
    block_m: int, N: int, K: int, small_grid: bool = False
) -> dict:
    """Explicit Python ladder keyed on ``(block_m, N, K, small_grid)``.

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
    * ``block_m in (64, 128)`` -- prefill. MFMA bound, so the 32x32x64 pipe with a wide
      N tile and warps spread over both axes.
    """
    if block_m == 16:
        block_n, block_k, warps, nb = 128, 512, (1, 4), 3
        instr = (16, 16, 128)
        if small_grid:
            # A narrow N tile is the only way to keep every XCD busy when the router
            # hands out only a few M blocks.
            block_n = 64
    elif block_m == 32:
        block_n, block_k, warps, nb = 256, 512, (1, 4), 2
        instr = (16, 16, 128)
    elif block_m == 64:
        block_n, block_k, warps, nb = 256, 256, (2, 4), 2
        instr = (32, 32, 64)
    else:  # 128
        block_n, block_k, warps, nb = 256, 256, (4, 2), 2
        instr = (32, 32, 64)

    # shrink the N tile until it divides N (the kernel has no N tail by construction)
    gran_n = instr[1] * warps[1]
    while block_n > gran_n and N % block_n != 0:
        block_n //= 2
    while block_k > 128 and (K % block_k != 0 or K // block_k < nb):
        block_k //= 2
    mini_n = min(block_n, gran_n * 2)
    while mini_n % gran_n:
        mini_n += gran_n

    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "K_UNROLL": nb,
        "MINI_BLOCK_K": block_k,
        "MINI_PREFETCH_K": 0,
        "MINI_BLOCK_M": block_m,
        "MINI_BLOCK_N": mini_n,
        "MINI_PRESTORE_MN": 1,
        "NUM_LDS_BUFFER": nb,
        "mfma_instr_shape": instr,
        "warps_per_cta": warps,
        "tiles_per_warp": (1, 1),
        "k_width": 16,
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
    if x_scales is None or w_scales is None:
        return False, "Gluon A4W4 needs microscaled activations and weights"
    if x_static_scale is not None or quant_static_scale is not None:
        return False, "static fp8 scales are not implemented on the Gluon path"
    if out_quant not in (None, DtypeQuant.MXFP4):
        return False, f"output_quant {out_quant} is not implemented"
    if swizzle_mx_scale is not None:
        # CDNA4_SCALE keeps the direct-to-LDS write coalesced but needs the descriptor
        # reshape/permute of the preshuffled tile; not implemented yet.
        return False, f"scale swizzle {swizzle_mx_scale} is not implemented"
    if routing_data is None or routing_data.expt_data is None:
        return False, "Gluon path needs the Family-A routing metadata"
    if x.dtype != torch.uint8 or w.dtype != torch.uint8:
        return False, "expected packed E2M1 payloads in uint8"
    if x.stride(-1) != 1:
        return False, "x must be row-major"
    if w.stride(-2) != 1:
        return False, "w must be K-contiguous ((E, K/2, N) with stride(-2) == 1)"
    if bias is not None and bias.dtype != torch.float32:
        return False, "bias must be fp32"

    block_m = routing_data.block_m
    if block_m not in _SUPPORTED_BLOCK_M:
        return False, f"block_m {block_m} outside {_SUPPORTED_BLOCK_M}"

    cfg = get_gluon_config(block_m, N, K, _small_grid(routing_data, y.shape[1], N))
    if N % cfg["BLOCK_N"] != 0:
        return False, f"N {N} % BLOCK_N {cfg['BLOCK_N']} != 0"
    if K % cfg["BLOCK_K"] != 0:
        return False, f"K {K} % BLOCK_K {cfg['BLOCK_K']} != 0"
    if K // cfg["BLOCK_K"] < cfg["NUM_LDS_BUFFER"]:
        return False, "K strip shorter than the pipeline depth"
    if K % (2 * MXFP4_QUANT_BLOCK_SIZE) != 0:
        return False, "K must be a multiple of 64 for packed E2M1 + group-32 scales"

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
    small_grid,
    has_bias,
    has_gammas,
    has_gather,
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
        else get_gluon_config(block_m, N, K, small_grid)
    )
    func_spec = FuncSpec(
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        int(DtypeQuant.MXFP4),
        gl.float32,
        act,
        out_quant,
        has_bias,
        has_gammas,
        has_gather,
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


def moe_gemm_a4w4_gluon(
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
    y_scales: torch.Tensor | None = None,
    config: dict | None = None,
):
    """Launch the Gluon grouped GEMM into ``y`` (shape ``(1, M, N // ARN)``).

    ``y_scales`` non-None selects the fused MXFP4 output quant: ``y`` then holds the
    E2M1 payload (``N // ARN // 2`` uint8 columns) and ``y_scales`` the E8M0 exponents,
    bit-identical to what the standalone ``mxfp4_quant`` launch produces.

    Returns the compiled kernel handle so callers (the ISA-assertion test) can inspect
    ``.asm``; the result itself is written into ``y`` / ``y_scales``.
    """
    block_m = routing_data.block_m
    expt_data = routing_data.expt_data
    grid_m = routing_data.n_blocks(y.shape[1], block_m)

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
        _small_grid(routing_data, y.shape[1], N),
        bias is not None,
        gammas is not None,
        gather_indx is not None,
        act,
        out_quant,
        tuple(sorted((k, _hashable(v)) for k, v in config.items())) if config else None,
    )

    a = QuantTokenTensor.make(
        DtypeQuant.MXFP4,
        x,
        x_scales,
        x.shape[0],
        x.stride(0),
        x_scales.stride(0),
        x_scales.stride(1),
        K,
        routing_data.n_expts_act,
        ScaleSwizzle.NONE,
    )
    b = QuantExpertTensor.make(
        DtypeQuant.MXFP4,
        w,
        w_scales,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scales.stride(0),
        w_scales.stride(2),
        w_scales.stride(1),
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
        grid_m,
        grid_n,
        kcfg,
        KernelFuncConfig,
        KernelTuningConfig,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )


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
        (M, out_n // MXFP4_QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
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

    moe_gemm_a4w4_gluon(
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
