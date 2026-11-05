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

The launch configs (the ladder, the measured overrides and the host probes) are in
``moe_op_gemm_gluon_config.py``.
"""

from __future__ import annotations

import os
import weakref
from functools import cache

import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._entry import (
    _moe_gluon_gemm1,
    _moe_gluon_gemm2,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    LDS_USABLE_BYTES,
    ActivationSpec,
    ActKind,
    DtypeQuant,
    EpilogueMode,
    FuncSpec,
    QuantExpertTensor,
    QuantTokenTensor,
    ResultTensor,
    RoutingMeta,
    ScaleSwizzle,
    TuningSpec,
    WeightGuHbmLayout,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import (
    validate_buffer_counts,
)
from aiter.ops.triton.moe.moe_op_gemm_gluon_config import (
    _SUPPORTED_BLOCK_M,
    MX_GROUP_SIZE,
    _b_scale_shuffle_fits,
    _canonical_slot_keys,
    _cfg_key,
    _cval,
    _default_launch_config,
    _env_int,
    _gemm1_half_m_config,
    _family_env_keys,
    _gemm2_half_m_config,
    _hashable,
    _pipeline_error,
    _probe_lds_bytes,
    _scale_shuffle_supported,
    _stream_mxfp8_gemm1_expert_payload,
    _tuning_args,
    _validate_launch_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

#: Raw current-stream handle. torch.cuda.current_stream().cuda_stream builds a Stream
#: object first and costs ~3 us, which is a third of the fast launch path.
_current_raw_stream = torch._C._cuda_getCurrentRawStream

_LOGGER = AiterTritonLogger()


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


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)
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


def _resolve_epilogue(epilogue: EpilogueMode | int | None) -> int:
    if epilogue is None:
        epilogue = _env_int("AITER_TRITON_MOE_GLUON_NO_EPI", int(EpilogueMode.DEFAULT))
    return int(EpilogueMode(epilogue))


def _small_grid(routing_data, M, N) -> bool:
    """True when the launch would not fill the machine at the default N tile."""
    grid_m = routing_data.n_blocks(M, routing_data.block_m)
    return grid_m * max(1, N // 128) < 512


def _multi_block(routing_data, M) -> bool:
    """True when the experts average at least two M blocks, so each expert's weights
    are read by several CTAs and the launch writes a large intermediate."""
    return M >= 2 * routing_data.block_m * routing_data.n_expts_tot


def _half_block(routing_data, M) -> bool:
    """True when the experts average at most half an M block, so the experts at or
    below the mean fit a half-height tile (the routing raises block_m at an exact
    mean, see ``routing_block_m``)."""
    return 2 * M <= routing_data.block_m * routing_data.n_expts_tot


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
    w_static_scale=None,
    apply_swiglu: bool = False,
    gate_up_split: bool = False,
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
    if w_static_scale is not None:
        # moe_gemm_a8w8 accepts one and the Triton kernel folds it into the
        # accumulator; the Gluon epilogue has no equivalent, so taking this path would
        # drop a scalar factor and return a quietly wrong answer.
        return False, "a per-tensor weight scale is not implemented on the Gluon path"
    if x_static_scale is not None and dq_a != DtypeQuant.FP8_E4M3:
        return False, "a static activation scale only applies to unscaled fp8 operands"
    if out_quant not in (None, DtypeQuant.MXFP4, DtypeQuant.MXFP8):
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

    cfg = _default_launch_config(
        block_m,
        N,
        K,
        dq_a,
        dq_b,
        _small_grid(routing_data, y.shape[1], N),
        apply_swiglu,
        _stream_mxfp8_gemm1_expert_payload(
            y.shape[1],
            routing_data.n_expts_tot,
            block_m,
            dq_a,
            dq_b,
            apply_swiglu,
        ),
        _multi_block(routing_data, y.shape[1]),
        _reads_b_directly(w, w_scales, N, K, dq_b),
        _half_block(routing_data, y.shape[1]),
    )
    if N % cfg["BLOCK_N"] != 0:
        return False, f"N {N} % BLOCK_N {cfg['BLOCK_N']} != 0"
    if K % cfg["BLOCK_K"] != 0:
        return False, f"K {K} % BLOCK_K {cfg['BLOCK_K']} != 0"
    pipeline_error = _pipeline_error(
        _cfg_key(cfg),
        dq_a,
        dq_b,
        K,
    )
    if pipeline_error is not None:
        return False, pipeline_error
    # Fill ownership no longer requires both axes split: a single-pane step copies both
    # operands, since the schedule maps each component's slots to panes rather
    # than handing each pane one copy. validate_pipeline() asserts the mapping
    # invariant itself as the in-kernel backstop.
    #
    # gate/up split is a separate matter, and a real one: it packs the two operand
    # sides as the two halves of N, so an N slot has to be exactly one side. Refused
    # here rather than in validate() so a tile that cannot carry it falls back instead
    # of failing the compile -- validate() asserts the same thing as the backstop.
    # Until the fill-ownership restriction was lifted this was unreachable, because
    # every tile that fails it also failed the both-axes-split rule.
    if gate_up_split and apply_swiglu:
        n_slots = cfg["BLOCK_N"] // cfg["SLOT_N"]
        if n_slots != 2:
            return False, (
                f"gate_up_split needs exactly two N slots, one per operand side: "
                f"BLOCK_N {cfg['BLOCK_N']} / SLOT_N {cfg['SLOT_N']} "
                f"= {n_slots}"
            )
        if N % 2 or (N // 2) % cfg["SLOT_N"]:
            return False, (
                f"gate_up_split needs each half of N ({N}) to be a whole number of "
                f"SLOT_N ({cfg['SLOT_N']}) blocks"
            )
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
    gate_up_split=False,
    epilogue=int(EpilogueMode.DEFAULT),
    stream_expert_payload=False,
    multi_block=False,
    direct_b=False,
    half_block=False,
):
    """Host-side constexpr work, memoised.

    The host aggregates and launch constants depend only on compile-time values,
    so they are computed once per distinct configuration.
    """
    from triton.experimental.gluon import language as gl

    c = (
        _canonical_slot_keys(dict(config_items))
        if config_items
        else _default_launch_config(
            block_m,
            N,
            K,
            dq_a,
            dq_b,
            small_grid,
            act is not None,
            stream_expert_payload,
            multi_block,
            direct_b,
            half_block,
        )
    )
    assert c["BLOCK_M"] == block_m, (
        f"config BLOCK_M {c['BLOCK_M']} must match routing block_m {block_m}"
    )
    half_items = None
    if half_block and act is not None:
        half_items = _gemm1_half_m_config(
            _cfg_key(c), block_m, N, K, dq_a, dq_b, small_grid
        )
    elif half_block:
        half_items = _gemm2_half_m_config(
            _cfg_key(c),
            block_m,
            N,
            K,
            dq_a,
            dq_b,
            small_grid,
            multi_block,
            _family_env_keys(),
        )
    # The flag tells the kernel that it gets a half-height tile.
    c = dict(c, HALF_M_TILE=half_items is not None)
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
        int(
            WeightGuHbmLayout.SPLIT if gate_up_split else WeightGuHbmLayout.INTERLEAVED
        ),
        epilogue,
    )
    tuning_spec = TuningSpec(*(_hashable(v) for v in _tuning_args(c)))
    # Construct the tuning config on the host off exactly the numbers the kernel will
    # use, so the grid math and the tile math cannot drift.
    func_cfg_host = KernelFuncConfig(*func_spec)
    tuning_cfg_host = KernelTuningConfig(func_cfg_host, *tuning_spec)
    # Storage can still change during scale preparation, which changes the exact
    # minimum K. Validate inherited depths and shape arithmetic now; validate the
    # complete effective configuration once preparation has resolved its layouts.
    validate_buffer_counts(tuning_cfg_host)
    layout_host = KernelLayout(tuning_cfg_host)
    layout_host.num_k_tiles(K)
    grid_n = _cval(layout_host.grid_N(N))
    # Keep each spec in one constexpr so launch specialization sees four leaves.
    constexpr_args = (
        gl.constexpr(func_spec),
        gl.constexpr(tuning_spec),
        gl.constexpr(N),
        gl.constexpr(K),
    )
    half_cfg_host = None
    if half_items is not None:
        half_spec = TuningSpec(*(_hashable(v) for v in _tuning_args(dict(half_items))))
        half_cfg_host = KernelTuningConfig(func_cfg_host, *half_spec)
        validate_buffer_counts(half_cfg_host)
        constexpr_args += (gl.constexpr(half_spec),)
        # A wider half-height tile sets the N grid (GEMM2, see _gemm2_half_m_config;
        # GEMM1's has the same BLOCK_N).
        grid_n = _cval(KernelLayout(half_cfg_host).grid_N(N))
    num_warps = c["warps_per_cta"][0] * c["warps_per_cta"][1]
    return (
        grid_n,
        constexpr_args,
        num_warps,
        c["WAVES_PER_EU"],
        c,
        tuning_cfg_host,
        half_cfg_host,
    )


#: Attribute the padded-row token map is memoised under, on the ``ExptData`` instance.
#: Every routing call builds a fresh ``ExptData``, so an entry's lifetime is exactly one
#: routing: the two GEMMs of a layer share one build and a new routing cannot see a
#: stale map. Hanging it off the object rather than a global dict is what ties the two
#: lifetimes together -- a dict keyed by id() would hand a recycled address the previous
#: routing's permutation.
_A_SORT_MAP_ATTR = "_aiter_gluon_sorted_token_ids"


def _sorted_token_id_map(expt_data, gather_indx, n_expts_act, block_m, n_blocks, n_tok):
    """Padded-row ``(sorted_token_ids, cumsum)``, built without reading the device.

    The C++ shuffle wants each expert's rows to start at ``token_offs_pad[e] *
    block_m``, but ``gather_indx`` is packed at the raw offsets, so the rows have to be
    scattered into padded position. Three steps of that scatter used to need a value
    that only exists on the GPU, and each one drained the queue:

    * the padded block count was ``int(token_offs_pad[-1])``; it is now the caller's
      ``n_blocks``, the same closed-form bound ``RoutingData.n_blocks`` already gives
      the kernel grid;
    * the row count was ``int(token_offs_raw[-1])``; it is now ``gather_indx.shape[0]``,
      which is that same total by construction (one gate per routed row);
    * the row -> expert map was ``repeat_interleave`` over the device histogram, whose
      output length is a device-side sum; it is now a ``searchsorted`` over the offsets,
      whose output is shaped by the host-side row count.

    The bound is only an upper bound -- up to ~2x the exact block count when experts are
    unevenly loaded -- but the exact length still reaches the kernel, as a device value
    in ``cumsum``. ``sort_scales_kernel_impl`` reads that as ``actual_sorted`` and
    zero-fills every chunk past it, so the slack costs stores, not gathers.
    """
    key = (block_m, n_expts_act, n_blocks, n_tok)
    cached = getattr(expt_data, _A_SORT_MAP_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    offs_pad = expt_data.token_offs_pad
    if isinstance(offs_pad, dict):
        offs_pad = (
            offs_pad[block_m] if block_m in offs_pad else next(iter(offs_pad.values()))
        )
    offs_raw = expt_data.token_offs_raw
    dev = offs_raw.device
    n_rows = int(gather_indx.shape[0])
    max_sorted = n_blocks * block_m

    rows = torch.arange(n_rows, device=dev, dtype=offs_raw.dtype)
    # right=True counts offsets <= row, which is what makes an expert with no tokens
    # fall out: its zero-width span shows up as a repeated boundary and the search steps
    # past both copies, so no row is ever assigned to it.
    e_of_row = torch.searchsorted(offs_raw[1:].contiguous(), rows, right=True)
    dst = offs_pad[e_of_row].to(torch.int32) * block_m + (
        rows.to(torch.int32) - offs_raw[e_of_row].to(torch.int32)
    )
    # The gaps between experts keep a sentinel the C++ side maps to row 0 (it tests
    # `sti_val < M`); the epilogue masks those rows out of the result anyway. One slot
    # past the buffer absorbs any row the block bound did not anticipate, so a routing
    # that leaves gates unassigned degrades to a dropped row rather than an OOB scatter.
    sorted_token_ids = torch.full(
        (max_sorted + 1,), n_tok, dtype=torch.int32, device=dev
    )
    sorted_token_ids[dst.clamp_(max=max_sorted).long()] = (
        gather_indx.to(torch.int32) // n_expts_act
    )
    built = (sorted_token_ids[:max_sorted], offs_pad[-1:].to(torch.int32) * block_m)
    try:
        setattr(expt_data, _A_SORT_MAP_ATTR, (key, built))
    except AttributeError:
        pass  # not memoisable; correctness is unaffected
    return built


def _sorted_shuffle_a_scales(x_scales, routing_data, gather_indx, K, cfg):
    """Run the C++ pre-pass that sorts + fragment-permutes the token scales.

    Reuses ``csrc/kernels/mxfp4_moe/moe_aux/moe_sort_scales.cuh`` rather than doing the
    permute in Triton: it is ~19 us at T=4096 and the layout it writes is exactly the
    one ``get_mfma_scale_layout`` wants for a 16x16x128 / (num_warps, 1) /
    tiles_per_warp (2, 1) dot operand, so the GEMM reads it with a contiguous load
    instead of BLOCK_M strided ``ds_read_u8``.

    Returns ``None`` when the config is not one the shuffle was written against, so the
    caller falls back to the raw K-contiguous path.
    """
    if x_scales is None or gather_indx is None:
        return None
    block_m = int(cfg["BLOCK_M"])
    # Mirrors KernelLayout.payload_a_sorted_shuffled_ok: the permutation is carried by
    # the LDS tile's layout. A shuffled scale load covers an integer number of BLOCK_K
    # payload steps; K128 therefore retains each packed K256 group for two steps.
    if (
        not _scale_shuffle_supported(cfg, 0)
        or x_scales.ndim != 2
        or x_scales.shape[1] != K // MX_GROUP_SIZE
        or not x_scales.is_contiguous()
        or block_m % 32 != 0
        # The buffer is one 256 B run per 32-row stripe, so a slot that is a
        # whole number of stripes is a contiguous slice of it; the kernel indexes the
        # slice with a stripe offset. It does not have to be the whole block.
        or int(cfg["SLOT_M"]) % 32 != 0
        or K % 256 != 0
    ):
        return None

    from aiter.ops.moe_mxfp4_aux import (
        MXFP4_MOE_SUPPORTED_SHAPES,
        mxfp4_moe_sort_scales,
    )

    dev = x_scales.device
    expt_data = routing_data.expt_data
    n_expts_act = routing_data.n_expts_act
    # The C++ side is a codegen'd template keyed on (BM, NE, D_HIDDEN), BM in
    # {32, 64, 128}. The module is torch-free, so a missing instance aborts the process
    # instead of raising: check the shape table first and fall back to raw scales.
    n_expts_tot = int(expt_data.hist.numel())
    if block_m not in (32, 64, 128) or not any(
        ne == n_expts_tot and h == K for ne, h, _e, _topk in MXFP4_MOE_SUPPORTED_SHAPES
    ):
        _LOGGER.info(
            "sorted-shuffled A scales unavailable, using raw scales: no aux instance "
            f"for BM={block_m} NE={n_expts_tot} H={K}"
        )
        return None
    # _launch_spec requires the config and routing to use the same BLOCK_M, so this
    # bound sizes the scale buffer in the same block space as the launch grid.
    n_blocks = routing_data.n_blocks(int(gather_indx.shape[0]), block_m)
    max_sorted = n_blocks * block_m
    sorted_token_ids, cumsum = _sorted_token_id_map(
        expt_data, gather_indx, n_expts_act, block_m, n_blocks, int(x_scales.shape[0])
    )

    k_pack = 256 // 128
    c_m1, c_k1 = block_m // 32, (K // 32) // (4 * k_pack)
    out = torch.empty(
        n_blocks * c_m1 * c_k1 * 4 * 16 * 4, dtype=torch.uint8, device=dev
    )
    mxfp4_moe_sort_scales(
        x_scales,
        sorted_token_ids,
        cumsum,
        out,
        n_expts_tot,
        int(n_expts_act),
        int(K),  # D_HIDDEN
        int(block_m),  # MB
        max_sorted,
    )
    return out


#: Kinds of copy made from a caller tensor. The weights are static, so in production
#: the shuffle belongs in the checkpoint-load path; caching keeps the benchmark honest
#: (paid once, not per call) without changing every caller.
_B_SCALE_SHUFFLE_COPY = "cdna4_scale_shuffled"
_B_PRESHUFFLE_COPY = "b_preshuffled"


class _Copy:
    """A cached copy of one caller tensor, and its runtime state.

    The entry is keyed by the tensor object, not by its data_ptr: a freed tensor's
    address is reused, and a stale entry then pairs one expert's scales with another's
    weights. A finalizer removes the entry when the tensor dies. Nothing is stored on
    the tensor itself, so ``torch.save`` and ``copy.deepcopy`` of it see no copy.
    """

    __slots__ = ("finalizer", "key", "out", "ready", "src", "streams")

    def __init__(self, src, slot):
        self.src = weakref.ref(src)
        self.finalizer = weakref.finalize(src, _WEIGHT_COPIES.pop, slot, None)
        self.finalizer.atexit = False

    def fill(self, key, out):
        """Keep ``out``, made on the current stream, and mark it not yet written."""
        self.key, self.out, self.ready, self.streams = key, out, None, set()
        if out.is_cuda:
            device = out.get_device()
            self.streams.add(torch._C._cuda_getCurrentRawStream(device))
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(device))

    def use(self):
        """The copy, for a launch on the current stream.

        The stream waits until the copy is written. The first launch from each other
        stream records that stream on the copy's memory, so freeing the copy (a drop or
        a new version) waits for that stream's launches before the memory is reused.
        """
        out = self.out
        if not out.is_cuda:
            return out
        device = out.get_device()
        stream = torch._C._cuda_getCurrentRawStream(device)
        if stream not in self.streams and not torch.cuda.is_current_stream_capturing():
            self.streams.add(stream)
            out.record_stream(torch.cuda.current_stream(device))
        if self.ready is not None:
            if self.ready.query():
                self.ready = None
            else:
                torch.cuda.current_stream(device).wait_event(self.ready)
        return out


#: (id(tensor), kind) -> _Copy.
_WEIGHT_COPIES: dict = {}


def _cached_copy(src, kind, make, *dims):
    """``make()``, kept for ``src`` until ``src`` changes.

    The key is the version counter, which any in-place write through any view
    increments, the storage address and ``dims``. An inference tensor has no version
    counter: its copy stays until :func:`drop_weight_copies`. A copy made while a CUDA
    graph is captured is not kept.
    """
    version = None if src.is_inference() else src._version
    key = (version, src.data_ptr(), *dims)
    slot = (id(src), kind)
    entry = _WEIGHT_COPIES.get(slot)
    if entry is not None and entry.src() is src and entry.key == key:
        return entry.use()
    out = make()
    if torch.cuda.is_current_stream_capturing():
        return out
    if entry is None or entry.src() is not src:
        entry = _WEIGHT_COPIES[slot] = _Copy(src, slot)
    entry.fill(key, out)
    return out


def drop_weight_copies(*tensors):
    """Drop the reordered copies that the Gluon MoE GEMM keeps for weight tensors.

    The GEMM keeps a preshuffled copy of ``w`` and of ``w_scales``, and makes a new
    copy after a PyTorch in-place update. Call this function with the tensors after a
    write that PyTorch does not count: for example the replay of a CUDA graph that
    writes the weights, a write through ``.data``, or an in-place write to an inference
    tensor. ``None`` entries are skipped.
    """
    for t in tensors:
        if t is None:
            continue
        for kind in (_B_PRESHUFFLE_COPY, _B_SCALE_SHUFFLE_COPY):
            entry = _WEIGHT_COPIES.get((id(t), kind))
            if entry is not None and entry.src() is t:
                entry.finalizer()  # removes the entry


def _k_pack(dq) -> int:
    """``dq_pack_divisor`` for the launch path: its constexpr-function call costs ~1 us."""
    return 2 if dq == DtypeQuant.MXFP4 else 1


def _preshuffled_b(w, cfg, N, K, dq_b):
    """Weights permuted 16-column blocked, or None if the config cannot read them.

    Byte ``(n, k)`` of an expert moves to
        ``(n//16)*(KB*16) + (k//16)*256 + (n%16)*16 + k%16``
    where ``KB`` is the stored K extent in bytes, so a wave's 64 lanes -- lane L
    wanting bytes ``[16L, 16L+16)`` of four 16-column, 16-byte tiles -- cover one
    contiguous 1024 B run. Read through the operand-B fragment layout, the plain
    ``(E, KB, N)`` tensor instead puts consecutive lanes ``stride(-2)`` apart: same L2
    and HBM traffic, 2.8x the L1 accesses, 42% slower. The tile is 16 bytes wide for
    every dtype, so this is the permute ``aiter/ops/shuffle.py::shuffle_weight(w,
    (16, 16))`` applies, and the one FlyDSL's port requires of its caller.

    ``byte_unit_lds_layout`` gives the shared tile the matching permutation, so the
    direct-to-LDS copy stays one linear run and the unit drops from 32 rows to 16 and
    loses its padding. A padded tensor (logical N or K below the stored extent) gets
    no copy.
    """
    if w is None or w.ndim != 3:
        return None
    if int(cfg["BLOCK_K"]) % 32 != 0 or N % 16 != 0 or K % 32 != 0:
        return None
    pack = _k_pack(dq_b)
    if w.shape[-1] != N or w.shape[-2] * pack != K:
        return None
    KB = K // pack * w.element_size()
    if KB % 16 != 0:
        return None
    E = w.shape[0]
    # (E, K, N) -> (E, N, KB bytes) -> blocked (E, N/16, KB/16, 16, 16) -> flat
    return _cached_copy(
        w,
        _B_PRESHUFFLE_COPY,
        lambda: w.transpose(-1, -2)
        .contiguous()
        .view(torch.uint8)
        .reshape(E, N // 16, 16, KB // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, N // 16 * KB * 16)
        .contiguous()
        .view(w.dtype),
        N,
        K,
    )


def _shuffled_b_scales(w_scales, cfg, N, K):
    """CDNA4_SCALE-preshuffled weight scales, or None if the config cannot read them.

    Reuses utils/shuffle.py::shuffle_scale_moe, whose gfx950 permute
    (preshuffle_factor 32, scale_kwidth 8) lands each element in exactly the slot
    ``get_mfma_scale_layout`` assigns for a 16x16x128 operand -- verified base for base
    against the layout, same as the A-side shuffle. Padded scales (logical N or K
    below the stored extent) get no copy.
    """
    if w_scales is None:
        return None
    if (
        not _b_scale_shuffle_fits(cfg)
        or w_scales.shape[-1] % 32 != 0
        or w_scales.shape[-2] % 8 != 0
        or w_scales.shape[-1] != N
        or w_scales.shape[-2] * MX_GROUP_SIZE != K
    ):
        return None
    from aiter.ops.triton.utils.shuffle import _shuffle_scale_tile_gfx950

    # The tile permute itself, not shuffle_scale_moe: that transposes the result
    # back to (E, K*, N/32) for the Triton kernels, and materialising that view
    # would undo the very byte order the shuffle exists to produce. What the kernel
    # reads is the pre-transpose (E, N/32, K) tile, which is already contiguous.
    # The N rows keep their input order, including separate gate/up halves.
    return _cached_copy(
        w_scales,
        _B_SCALE_SHUFFLE_COPY,
        lambda: _shuffle_scale_tile_gfx950(w_scales.transpose(-1, -2), 32, 8),
        N,
        K,
    )


def _reads_b_directly(w, w_scales, N, K, dq_b) -> bool:
    """True when the launch must read ``w`` and ``w_scales`` directly, through its
    ladder, instead of the copies that the family entries read.

    A copy is not valid while a CUDA graph is captured (a replay runs no host check),
    for an inference tensor (no version counter shows its writes) or for a padded
    tensor (logical N or K below the stored extent).
    """
    if torch.cuda.is_current_stream_capturing() or w.is_inference():
        return True
    pack = _k_pack(dq_b)
    *_, k_stored, n_stored = w.shape
    if n_stored != N or k_stored * pack != K:
        return True
    if w_scales is None:
        return False
    *_, k_groups, n_scales = w_scales.shape
    return w_scales.is_inference() or n_scales != N or k_groups * MX_GROUP_SIZE != K


#: Compiled kernel + launch constants, keyed by everything that can change the
#: specialization. Bounded by the number of distinct shapes a process sees.
_FAST_LAUNCH_CACHE: dict = {}


def _hooks_empty(hook) -> bool:
    """True when no launch hook would actually run.

    ``knobs.runtime.launch_{enter,exit}_hook`` default to an empty ``HookChain``, which
    is a live object and therefore truthy; only ``.calls`` says whether anything is
    installed. Older Triton exposed a bare ``None`` here, so both shapes are handled.
    """
    if hook is None:
        return True
    calls = getattr(hook, "calls", None)
    return calls is not None and len(calls) == 0


def _spec_key(vals, out):
    """Append the parts of ``vals`` that can change a launch's specialization.

    Triton re-derives a full specialization on every launch -- ~16 us of the ~41 us
    launch here -- but between two launches of one *already compiled* kernel only two
    things can move: the 16-byte alignment of each pointer (which it bakes in as
    ``tt.divisibility``) and the runtime scalars. Constexprs cannot move, because they
    are pinned by the identity of the memoised ``constexpr_args`` in the cache key.

    Recurses into the aggregates, whose fields are ordinary launch arguments.
    """
    for v in vals:
        if isinstance(v, torch.Tensor):
            out.append(v.data_ptr() % 16 == 0)
        elif isinstance(v, tuple):
            _spec_key(v, out)
        elif isinstance(v, int):  # bool is an int; both are fine as key material
            out.append(v)
    return out


def _fast_launch(kernel, grid_x, args, num_warps, waves_per_eu, constexpr_args):
    """Dispatch a previously compiled kernel directly, or ``None`` to take the slow path.

    ``JITFunction.run`` spends ~33 of its ~41 us re-deriving state that is identical
    across launches of one compiled kernel: binding and specializing the arguments
    (~16 us), re-walking the 26 module globals the kernel closed over to check none was
    rebound (~12 us), and rebuilding the cache key that then hits (~5 us). With the
    kernel in hand, dispatch needs only a grid, a stream and the argument list.

    Correctness rests on the key: a miss falls back to the full path, which recompiles
    or re-specializes as needed and then re-memoises.
    """
    if not _env_int("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", 1):
        return None
    from triton import knobs

    # Profiling hooks and Triton's debug mode both act inside the slow path; if either
    # is live, take it so their behaviour is not silently dropped. An installed-but-empty
    # HookChain is truthy, so emptiness has to be tested through .calls -- the launch
    # hooks are non-None by default and a bare truth test never takes the fast path.
    if not _hooks_empty(knobs.runtime.launch_enter_hook):
        return None
    if not _hooks_empty(knobs.runtime.launch_exit_hook):
        return None
    if knobs.runtime.debug:
        return None

    key = (id(kernel), id(constexpr_args), num_warps, waves_per_eu, grid_x)
    entry = _FAST_LAUNCH_CACHE.get(key)
    if entry is None:
        return None
    compiled, spec, run, fn, packed, _anchor, cpp = entry
    if _spec_key(args, []) != spec:
        return None

    stream = _current_raw_stream(torch.cuda.current_device())
    if cpp is not None:
        # Generated launcher: METH_FASTCALL, straight-line unpack, THPVariable_Unpack
        # instead of a data_ptr() method call per tensor. Same top-level arguments.
        cpp(grid_x, stream, *args)
        return compiled

    run(grid_x, 1, 1, stream, fn, packed, None, None, None, *args)
    return compiled


def _fast_launch_memoise(
    kernel, grid_x, args, num_warps, waves_per_eu, constexpr_args, compiled
):
    """Record a completed slow-path launch so the next identical one can skip it."""
    if compiled is None or not _env_int("AITER_TRITON_MOE_GLUON_FAST_LAUNCH", 1):
        return
    key = (id(kernel), id(constexpr_args), num_warps, waves_per_eu, grid_x)
    cpp = None
    if _env_int("AITER_TRITON_MOE_GLUON_CPP_LAUNCH", 0):
        # Off by default: the first build of a given signature costs ~12 s of ninja.
        # It is cached on disk, so later processes only pay the load.
        try:
            from aiter.ops.triton.utils._triton.cpp_launcher import build_for

            cpp = build_for(compiled)
        except Exception as e:  # noqa: BLE001
            _LOGGER.info(f"C++ launcher unavailable, using triton's: {e}")

    _FAST_LAUNCH_CACHE[key] = (
        compiled,
        _spec_key(args, []),
        compiled.run,
        compiled.function,
        compiled.packed_metadata,
        # Keep constexpr_args alive so its address cannot be recycled by a later
        # allocation and turn a stale entry into a silent hit.
        # (The weight copies avoid the same hazard with a finalizer on their tensor.)
        (kernel, constexpr_args),
        cpp,
    )


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
    gate_up_split: bool | None = None,
    epilogue: EpilogueMode | int | None = None,
    out_scale_sorted_shuffled: bool = False,
):
    """Launch the Gluon grouped GEMM into ``y`` (shape ``(1, M, N // ARN)``).

    ``out_scale_sorted_shuffled``: with a fused MX output quant, write ``y_scales`` in
    the sorted-shuffled layout gemm2 reads (OUT_SCALE_SORTED_SHUFFLED).

    Operand dtypes are inferred from the tensors -- ``uint8`` + scales is MXFP4, fp8 +
    scales is MXFP8, fp8 alone is FP8_E4M3 (unit scales, still on the scaled pipe),
    bf16 alone is BF16 -- so every ``moe_gemm_*`` op calls this one function.
    :func:`gluon_supported` must have said yes for these tensors first.
    An explicit config's ``BLOCK_M`` must match ``routing_data.block_m`` because
    the routing offsets and block map are built for that geometry.
    ``scale_slot_m``, ``scale_slot_n`` and ``scale_slot_k``
    (also accepted in uppercase, and under the legacy ``scale_mini_block_*`` names)
    set the scale slots, the shuffled-scale load extents, independently of the
    payload slots. K counts scale columns, one E8M0 byte per MX group of 32
    payload K (the legacy ``scale_mini_block_k`` counted payload K and is divided by
    32). Zero inherits the payload M/N extent, while shuffled scale K defaults to
    ``max(8, BLOCK_K / 32)``. An explicit scale K must be a power-of-two multiple of
    ``BLOCK_K / 32``, at least that wide, and divide total K / 32. Unshuffled scales
    ignore these settings.

    ``y_scales`` non-None selects the fused MXFP4 output quant: ``y`` then holds the
    E2M1 payload (``N // ARN // 2`` uint8 columns) and ``y_scales`` the E8M0 exponents,
    bit-identical to what the standalone ``mxfp4_quant`` launch produces.

    ``gate_up_split`` selects the gated activation's operand packing along N: the
    default ``False`` is the interleaved (g,l,g,l) form every caller uses today, and
    ``True`` expects gate in ``w[..., :N//2]`` and up in ``w[..., N//2:]`` -- see
    ``activations.py::gate_up_split_perm``, which builds the permutation, and the
    ``moe_gemm.py`` module docstring for why the split form is faster. ``None`` takes
    the ``AITER_TRITON_MOE_GLUON_GU_SPLIT`` default. Inert without ``apply_swiglu``.

    ``epilogue`` selects normal arithmetic or the shape-preserving NOP modes in
    :class:`EpilogueMode`. ``None`` accepts the legacy ``NO_EPI`` environment setting.

    Returns the compiled kernel handle so callers (the ISA-assertion test) can inspect
    ``.asm``; the result itself is written into ``y`` / ``y_scales``.
    """
    if gate_up_split is None:
        gate_up_split = bool(_env_int("AITER_TRITON_MOE_GLUON_GU_SPLIT", 0))
    gate_up_split = bool(gate_up_split) and apply_swiglu
    epilogue = _resolve_epilogue(epilogue)
    block_m = routing_data.block_m
    expt_data = routing_data.expt_data
    grid_m = routing_data.n_blocks(y.shape[1], block_m)
    dq_a = infer_dtype_quant(x, x_scales)
    dq_b = infer_dtype_quant(w, w_scales)
    stream_expert_payload = _stream_mxfp8_gemm1_expert_payload(
        y.shape[1],
        routing_data.n_expts_tot,
        block_m,
        dq_a,
        dq_b,
        apply_swiglu,
    )
    small_grid = _small_grid(routing_data, y.shape[1], N)
    multi_block = _multi_block(routing_data, y.shape[1])
    half_block = _half_block(routing_data, y.shape[1])
    direct_b = _reads_b_directly(w, w_scales, N, K, dq_b)

    if apply_swiglu:
        act = ActivationSpec(
            kind=int(ActKind.SILU if alpha == 1.0 else ActKind.SWIGLU_OAI),
            alpha=alpha,
            limit=limit,
            add_residual=bool(swiglu_add_residual),
        )
    else:
        act = None
    if y_scales is None:
        out_quant = None
    elif y.dtype == torch.float8_e4m3fn:
        out_quant = int(DtypeQuant.MXFP8)
    else:
        out_quant = int(DtypeQuant.MXFP4)

    # Every argument but the config tuple is fixed for this launch, so bind them once.
    # Three of the shuffle branches below rebuild the spec against an amended config,
    # and spelling the full argument list four times is how a new field silently
    # reaches one call site and not the others.
    def _spec(cfg_items):
        return _launch_spec(
            block_m,
            N,
            K,
            dq_a,
            dq_b,
            small_grid,
            bias is not None,
            gammas is not None,
            gather_indx is not None,
            x_static_scale is not None,
            act,
            out_quant,
            cfg_items,
            gate_up_split,
            epilogue,
            stream_expert_payload,
            multi_block,
            direct_b,
            half_block,
        )

    grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc, _half_tc = _spec(
        _cfg_key(config) if config else None
    )

    # Tuning flags request preparation of the public raw scale inputs. Only pass an
    # effective shuffle flag to the kernel when preparation actually succeeded.
    a_scales, a_swizzle = x_scales, ScaleSwizzle.NONE
    a_scale_stride_m = 0 if x_scales is None else x_scales.stride(0)
    a_scale_stride_k = 0 if x_scales is None else x_scales.stride(1)
    if (
        _cfg.get("A_SCALE_SORTED_SHUFFLED", False)
        or _env_int("AITER_TRITON_MOE_GLUON_SORTED_SCALES", 0)
    ) and apply_swiglu:
        shuffled = _sorted_shuffle_a_scales(
            x_scales, routing_data, gather_indx, K, _cfg
        )
        if shuffled is not None:
            a_scales, a_swizzle = shuffled, ScaleSwizzle.SORTED_SHUFFLED
            # The gather and the row stride are baked into the buffer; all the kernel
            # still needs is the packed K stride: eight scales span 256 bytes. A scale
            # load wider than BLOCK_K is retained for its scale-K/BLOCK_K payload steps.
            a_scale_stride_m, a_scale_stride_k = 0, 32
    a_shuffled = a_swizzle == ScaleSwizzle.SORTED_SHUFFLED
    b_scales, b_swizzle = w_scales, ScaleSwizzle.NONE
    b_scale_stride_k = 0 if w_scales is None else w_scales.stride(1)
    if _cfg.get("B_SCALE_SHUFFLED", False) or _env_int(
        "AITER_TRITON_MOE_GLUON_SHUFFLED_W_SCALES", 0
    ):
        shuf_w = _shuffled_b_scales(w_scales, _cfg, N, K)
        if shuf_w is not None:
            b_scales, b_swizzle = shuf_w, ScaleSwizzle.CDNA4_SCALE
            b_scale_stride_k = 32
    b_shuffled = b_swizzle == ScaleSwizzle.CDNA4_SCALE
    if (
        bool(_cfg.get("A_SCALE_SORTED_SHUFFLED", False)) != a_shuffled
        or bool(_cfg.get("B_SCALE_SHUFFLED", False)) != b_shuffled
    ):
        _cfg = dict(
            _cfg, A_SCALE_SORTED_SHUFFLED=a_shuffled, B_SCALE_SHUFFLED=b_shuffled
        )
        grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc, _half_tc = _spec(
            _cfg_key(_cfg)
        )

    if out_scale_sorted_shuffled and not _cfg.get("OUT_SCALE_SORTED_SHUFFLED", False):
        assert y_scales is not None
        _cfg = dict(_cfg, OUT_SCALE_SORTED_SHUFFLED=True)
        grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc, _half_tc = _spec(
            _cfg_key(_cfg)
        )

    # The Quant* tuples carry the scale pointer and strides unconditionally; when the
    # operand has no scale the kernel never reads them, so a null pointer and zero
    # strides keep one tuple type for every dtype instead of four launch sites.
    a = QuantTokenTensor.make(
        dq_a,
        x,
        a_scales,
        x.shape[0],
        x.stride(0),
        a_scale_stride_m,
        a_scale_stride_k,
        K,
        routing_data.n_expts_act,
        a_swizzle,
    )
    w_payload = w
    # A direct-register B tile is read through the operand fragment layout, which is
    # only coalesced on a preshuffled weight, so a config that asks for registers is
    # asking for the repack too -- it is not a separate opt-in. The repack is cached
    # per weight tensor (_cached_copy), so it costs one pass per model, not per launch.
    if _env_int("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", 0) or not _cfg.get(
        "ALLOW_B_VIA_LDS", True
    ):
        shuf_b = _preshuffled_b(w, _cfg, N, K, dq_b)
        if shuf_b is not None:
            w_payload = shuf_b
            if not _cfg.get("B_PRESHUFFLED", False):
                _cfg = dict(_cfg, B_PRESHUFFLED=True)
                grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc, _half_tc = (
                    _spec(_cfg_key(_cfg))
                )
    if not _cfg.get("ALLOW_B_VIA_LDS", True) and not _cfg.get("B_PRESHUFFLED", False):
        raise ValueError("ALLOW_B_VIA_LDS=False requires B_PRESHUFFLED weights")
    _validate_launch_config(_tc, N, K)
    if _half_tc is not None:
        _validate_launch_config(_half_tc, N, K)

    # The preshuffled copy is packed, so its expert stride is not the stride of ``w``.
    # The kernel reads only the expert stride of a preshuffled payload.
    b = QuantExpertTensor.make(
        dq_b,
        w_payload,
        b_scales,
        w_payload.stride(0),
        w.stride(1),
        w.stride(2),
        0 if b_scales is None else b_scales.stride(0),
        0 if b_scales is None else b_scales.stride(b_scales.ndim - 1),
        b_scale_stride_k,
        w.shape[0],
        K,
        N,
        b_swizzle,
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

    # Every aggregate is spelled out. The kernel signature is flat because
    # torch.compile / AOTInductor cannot lift the tensors inside a tuple kernel argument
    # into the FX graph -- see _moe_gluon_gemm1. Field order must track the NamedTuple
    # definitions in _types.py.
    kernel = _moe_gluon_gemm1 if apply_swiglu else _moe_gluon_gemm2
    args = (
        a.ptr,
        a.scale_ptr,
        a.num_token,
        a.stride_m,
        a.scale_stride_m,
        b.ptr,
        b.scale_ptr,
        b.stride_e,
        b.stride_n,
        b.scale_stride_e,
        b.num_expert,
        res.ptr,
        res.scale_ptr,
        res.stride_m,
        res.scale_stride_m,
        res.scale_stride_n,
        rt.expt_block_pid_map,
        rt.expt_hist,
        rt.expt_offs_raw,
        rt.expt_offs_sum,
        rt.gather_indx,
        rt.scatter_indx,
        rt.gammas,
        bias,
        0 if bias is None else bias.stride(0),
        x_static_scale,
        grid_m,
        grid_n,
        K // int(_cfg["BLOCK_K"]),
        a.dtype_quant,
        a.hidden_dim,
        a.topk,
        a.scale_stride_k,
        a.scale_swizzle,
        b.dtype_quant,
        b.stride_k,
        b.scale_stride_n,
        b.scale_stride_k,
        b.hidden_dim,
        b.fused_intermediate_dim,
        b.scale_swizzle,
        res.dtype_quant,
        res.stride_n,
        res.out_dim,
        rt.n_expts_act,
        *constexpr_args,
    )
    grid_x = grid_m * grid_n
    fast = _fast_launch(kernel, grid_x, args, num_warps, waves_per_eu, constexpr_args)
    if fast is not None:
        return fast
    compiled = kernel[(grid_x,)](
        *args,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        **_LLVM_COMPILE_OPTIONS,
    )
    _fast_launch_memoise(
        kernel, grid_x, args, num_warps, waves_per_eu, constexpr_args, compiled
    )
    return compiled


#: Extra LLVM function attributes for the Gluon MoE kernels, comma separated, passed
#: through Triton's ``llvm_fn_attrs`` compile option (no Triton patch needed).
#:
#: The motivating one is ``amdgpu-ieee=false``. ``tl.max`` on f32 lowers to
#: ``llvm.maxnum``, and in IEEE mode the AMDGPU backend has to canonicalise both
#: operands first, so every cross-lane reduction step carries two redundant
#: ``v_max_f32 x, x, x``. With IEEE mode off ``v_max_f32`` already returns the non-NaN
#: operand -- exactly maxnum's semantics -- and the canonicalisation is dropped. It is
#: opt-in because it changes NaN/denormal behaviour for every float op in the kernel,
#: not just the reduction.
#:
#:     AITER_TRITON_MOE_GLUON_LLVM_FN_ATTRS=amdgpu-ieee=false
_LLVM_FN_ATTRS = os.environ.get("AITER_TRITON_MOE_GLUON_LLVM_FN_ATTRS", "")

#: Extra LLVM command line options for the Gluon MoE kernels, comma separated, passed
#: through Triton's ``llvm_cl_opts`` compile option; bare names are set to true. Lets a
#: benchmark run a code generator built from patched LLVM with its opt-in flags, e.g.
#:
#:     AITER_TRITON_MOE_GLUON_LLVM_CL_OPTS=amdgpu-ds-read-agpr,amdgpu-mfma-tied-cd
_LLVM_CL_OPTS = os.environ.get("AITER_TRITON_MOE_GLUON_LLVM_CL_OPTS", "")

# Only the options that are set: a Triton without one of them rejects the keyword.
_LLVM_COMPILE_OPTIONS = {
    key: value
    for key, value in (("llvm_fn_attrs", _LLVM_FN_ATTRS), ("llvm_cl_opts", _LLVM_CL_OPTS))
    if value
}


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
    w_static_scale=None,
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
        w_static_scale=w_static_scale,
        apply_swiglu=apply_swiglu,
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
    gate_up_split: bool | None = None,
    epilogue: EpilogueMode | int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gemm1 with the activation **and** the MXFP4 output quant fused into the epilogue.

    Returns ``(y_fp4, y_scales)`` in exactly gemm2's operand-A format: E2M1 payload
    packed 2/byte along the emitted N axis, uint8 E8M0 scales per group of 32. This is
    the whole point of the two-kernel split -- today's flow writes bf16, reads it back
    and runs a third full pass (``mxfp4_quant``, which even upcasts to fp32 first),
    roughly 2.5x the necessary intermediate traffic.

    ``gate_up_split=True`` expects ``w`` (and ``w_scales`` and ``bias``) permuted so
    gate occupies ``[0, N/2)`` and up ``[N/2, N)`` along N; see
    :func:`gate_up_split_perm`. The returned payload and scales are bit-identical to
    the interleaved form, so the two are a drop-in A/B.

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
        apply_swiglu=True,
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
        gate_up_split=gate_up_split,
        epilogue=epilogue,
    )
    return y[0], y_scales


def moe_gemm1_a8w4_mxfp8_out(
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
    gate_up_split: bool | None = None,
    epilogue: EpilogueMode | int | None = None,
    scale_layout: str = "raw",
) -> tuple[torch.Tensor, torch.Tensor]:
    """gemm1 with the activation **and** an MXFP8 output quant fused into the epilogue:
    the a8w4 counterpart of :func:`moe_gemm1_a4w4_mxfp4_out`. Returns
    ``(y_fp8, y_scales)``: e4m3 ``(M, N/2)`` and uint8 E8M0 ``(M, N/64)`` (one scale per
    32 emitted columns, the power-of-2 amax scheme of ``dynamic_mxfp8_quant``), rows in
    the sorted routing order, i.e. gemm2's operand A for an a8w4 layer.

    ``scale_layout="sorted_shuffled"`` writes the scales the way gemm2 reads them with
    A_SCALE_SORTED_SHUFFLED instead (moe_sort_scales.cuh): rows at their padded routing
    position in ``routing_data.block_m`` blocks, one 256 B run per 32-row stripe and
    K256. The buffer is ``(n_blocks * block_m, N/64)`` bytes, zero where no row lands.

    Raises if the shape is outside the Gluon path.
    """
    assert scale_layout in ("raw", "sorted_shuffled"), scale_layout
    M = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    N = w.shape[-1]
    K = x.shape[-1]
    out_n = N // 2
    y = torch.empty((1, M, out_n), dtype=torch.float8_e4m3fn, device=x.device)
    shuffled = scale_layout == "sorted_shuffled"
    if shuffled:
        assert out_n % 256 == 0, "sorted-shuffled scales pack K256 groups"
        block_m = int(routing_data.block_m)
        rows = routing_data.n_blocks(M, block_m) * block_m
        y_scales = torch.zeros(
            (rows, out_n // MX_GROUP_SIZE), dtype=torch.uint8, device=x.device
        )
    else:
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
        out_quant=DtypeQuant.MXFP8,
        N=N,
        K=K,
        apply_swiglu=True,
    )
    if not ok:
        raise NotImplementedError(f"fused MXFP8 gemm1 not available: {why}")
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
        gate_up_split=gate_up_split,
        epilogue=epilogue,
        out_scale_sorted_shuffled=shuffled,
    )
    return y[0], y_scales
