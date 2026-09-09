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

import math
import os
from functools import cache

import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    LDS_USABLE_BYTES,
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._entry import (
    _moe_gluon_gemm1,
    _moe_gluon_gemm2,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._frozen import (
    _validate_frozen_pipeline,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._pipeline import (
    _validate_pipeline as _validate_live_pipeline,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DSReadOperand,
    DtypeQuant,
    EpilogueMode,
    FuncSpec,
    QuantExpertTensor,
    QuantTokenTensor,
    ResultTensor,
    RoutingMeta,
    ScaleSwizzle,
    SchedMode,
    TileSched,
    TuningSpec,
    WaitCommitScheme,
    WarpPipeline,
    dq_pack_divisor,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

#: Raw current-stream handle. torch.cuda.current_stream().cuda_stream builds a Stream
#: object first and costs ~3 us, which is a third of the fast launch path.
_current_raw_stream = torch._C._cuda_getCurrentRawStream

_LOGGER = AiterTritonLogger()

#: MX group size. Fixed by the OCP microscaling spec for both E2M1 and E4M3; not a knob.
MX_GROUP_SIZE = 32
_SUPPORTED_BLOCK_M = (16, 32, 64, 128)
_COMPONENT_BUFFER_KEYS = (
    "A_NUM_BUFFER",
    "B_NUM_BUFFER",
    "A_SCALE_NUM_BUFFER",
    "B_SCALE_NUM_BUFFER",
)
_REGISTER_STORAGE_KEYS = ("B_IN_REG", "B_SCALE_IN_REG", "A_SCALE_IN_REG")
_DS_READ_FIELDS = (
    ("DS_READ_A_PAYLOAD_IN_MFMA", DSReadOperand.A),
    ("DS_READ_A_SCALE_IN_MFMA", DSReadOperand.A_SCALE),
    ("DS_READ_B_PAYLOAD_IN_MFMA", DSReadOperand.B),
    ("DS_READ_B_SCALE_IN_MFMA", DSReadOperand.B_SCALE),
)


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


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def _cval(v):
    """Take the value out of a ``gl.constexpr`` a config method handed back."""
    return v.value if hasattr(v, "value") else v


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


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None else int(v)


def _env_cache_modifier(name: str, default: str, legacy_name: str) -> str:
    """Cache-modifier override, for A/B-testing L1/L2 policy without editing source."""
    prefix = "AITER_TRITON_MOE_GLUON_"
    return os.environ.get(prefix + name, os.environ.get(prefix + legacy_name, default))


def _ds_read_in_mfma_from_env() -> int:
    """Translate the old whole-operand controls into the component mask.

    An explicit mask takes precedence. Legacy DS_MOVE moved A (payload and scale)
    at 1, both operands at 2; DS_IN_MFMA moved both regardless of DS_MOVE.
    """
    mask = os.environ.get("AITER_TRITON_MOE_GLUON_DS_READ_IN_MFMA")
    if mask is not None:
        return int(mask)
    if _env_int("AITER_TRITON_MOE_GLUON_DS_IN_MFMA", 0):
        return int(DSReadOperand.ALL)
    move = _env_int("AITER_TRITON_MOE_GLUON_DS_MOVE", 0)
    if move >= 2:
        return int(DSReadOperand.ALL)
    if move >= 1:
        return int(DSReadOperand.A | DSReadOperand.A_SCALE)
    return int(DSReadOperand.NONE)


def _ds_read_flags(mask: int) -> dict[str, bool]:
    """Expand the legacy environment mask into independent tuning fields."""
    assert mask >= 0 and not (mask & ~int(DSReadOperand.ALL)), (
        f"DS_READ_IN_MFMA {mask} has unknown operand bits"
    )
    return {name: bool(mask & bit) for name, bit in _DS_READ_FIELDS}


def _ds_read_flags_from_env() -> dict[str, bool]:
    flags = _ds_read_flags(_ds_read_in_mfma_from_env())
    for name, _ in _DS_READ_FIELDS:
        value = os.environ.get("AITER_TRITON_MOE_GLUON_" + name)
        if value is not None:
            flags[name] = bool(int(value))
    return flags


def _resolve_epilogue(epilogue: EpilogueMode | int | None) -> int:
    if epilogue is None:
        epilogue = _env_int("AITER_TRITON_MOE_GLUON_NO_EPI", int(EpilogueMode.DEFAULT))
    return int(EpilogueMode(epilogue))


@cache
def _probe_lds_bytes_cached(cfg_items: tuple, dq_a, dq_b) -> int:
    return _probe_lds_bytes_uncached(dict(cfg_items), dq_a, dq_b)


def _pick_warps(block_m: int, block_n: int, num_warps: int, instr) -> tuple:
    """Split ``num_warps`` over (M, N) so each warp's tile is as square as possible.

    This is the single biggest tuning lever and it is not obvious from the tile sizes.
    A warp's operand traffic is ``m_tiles * k + n_tiles * k`` LDS reads for
    ``m_tiles * n_tiles * k`` MFMAs, so the MFMA-per-read ratio is maximised when the
    warp tile is square and falls off linearly with the aspect ratio. At BLOCK_M=128,
    BLOCK_N=256 the old fixed ``(4, 2)`` gave every warp a 32x128 slab -- 16 MFMAs
    against 20 LDS reads. ``(2, 4)`` makes it 64x64: same 16 MFMAs, 16 reads. Measured
    424.6 -> 362.9 us on H7168-I2048-E32-k8 stage 1, and 355.4 us for the best
    tiles_per_warp variant.

    Splitting N is what the caller's ``gran_n`` is derived from, so a split whose N
    granularity does not divide ``block_n`` is not merely suboptimal -- it makes the
    caller's ``mini_n`` alignment walk step past ``block_n`` forever. The N constraint
    is therefore mandatory and the M constraint is only a preference. ``(num_warps, 1)``
    is the guaranteed-safe floor: ``block_n`` is always a multiple of ``instr[1]``.
    """
    best = None
    fallback = None
    wm = 1
    while wm <= num_warps:
        wn = num_warps // wm
        if wm * wn == num_warps and block_n % (instr[1] * wn) == 0:
            # squareness of the per-warp tile, in log space
            skew = abs(math.log2((block_m / wm) / (block_n / wn)))
            if block_m % (instr[0] * wm) == 0:
                if best is None or skew < best[0]:
                    best = (skew, (wm, wn))
            elif fallback is None or skew < fallback[0]:
                fallback = (skew, (wm, wn))
        wm *= 2
    if best is not None:
        return best[1]
    return fallback[1] if fallback is not None else (num_warps, 1)


def _probe_lds_bytes(cfg: dict, dq_a, dq_b) -> int:
    """Memoised. Constructing the two aggregates costs ~55 us of Python, and this runs
    inside ``gluon_supported`` on *every* launch -- at decode that is the critical path
    and it doubled the host issue time before it was cached."""
    return _probe_lds_bytes_cached(
        tuple(sorted((k, _hashable(v)) for k, v in cfg.items())), dq_a, dq_b
    )


def _probe_lds_bytes_uncached(cfg: dict, dq_a, dq_b) -> int:
    """LDS footprint of a candidate config, computed by the *aggregate* rather than by a
    second copy of the arithmetic -- the host and the device must not be able to
    disagree about whether a config fits."""
    return _probe_tuning_config(cfg, dq_a, dq_b).lds_bytes()


def _probe_tuning_config(cfg: dict, dq_a, dq_b):
    """Build the shape-independent configuration used by capability checks."""
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
    return KernelTuningConfig(func, *_tuning_args(cfg))


def _validate_selected_pipeline(tuning_cfg, K):
    validator = (
        _validate_frozen_pipeline
        if _cval(tuning_cfg.FROZEN_STEP)
        else _validate_live_pipeline
    )
    return validator(tuning_cfg, K)


@cache
def _pipeline_error(cfg_items: tuple, dq_a, dq_b, K):
    """Share the exact device ring and minimum-strip contract with host fallback."""
    try:
        tc = _probe_tuning_config(dict(cfg_items), dq_a, dq_b)
        _validate_selected_pipeline(tc, K)
    except AssertionError as error:
        return str(error)
    return None


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
        # BLOCK_K=512 is not a depth choice, it is forced: the A-scale tile is
        # block_m * BLOCK_K/32 bytes, and direct-to-LDS needs a warp to write one
        # contiguous 64-lane x 4 B run. At BLOCK_K=256 that tile is 128 B, the backend
        # refuses to lower the copy, and the scale falls back to a register load inside
        # the K loop -- measured at +17 us on H=7168 I=2048 E=32 topk=8 T=32, which no
        # amount of the occupancy it buys back (4 CTAs/CU vs 2) makes up for.
        # Two buffers, not three: measured 112 us vs 118 at half the LDS.
        block_n, block_k, num_warps, nb = 128, 512, 4, 3
        nonk = 16
        if small_grid:
            # A narrow N tile is the only way to keep every XCD busy when the router
            # hands out only a few M blocks.
            block_n = 64
    elif block_m == 32:
        block_n, block_k, num_warps, nb = 256, 512, 4, 3
        nonk = 16
    elif block_m == 64:
        block_n, block_k, num_warps, nb = 256, 256, 8, 3
        nonk = 32
    else:  # 128
        block_n, block_k, num_warps, nb = 256, 256, 8, 3
        nonk = 32
    nb = _env_int("AITER_TRITON_MOE_GLUON_NB", nb)
    instr = _mfma_instr(dq_a, dq_b, nonk)
    ds_read_flags = _ds_read_flags_from_env()

    # shrink the N tile until it divides N (the kernel has no N tail by construction).
    # The warp split is derived per candidate BLOCK_N inside _build, so use the
    # coarsest granularity any split could need for the divisibility walk.
    gran_n = instr[1]
    while block_n > gran_n and N % block_n != 0:
        block_n //= 2
    min_k = max(128, instr[2])
    while block_k > min_k and (K % block_k != 0 or K // block_k < 2 * nb + 1):
        block_k //= 2

    def _build(bn, bk, n_buf):
        warps = _pick_warps(block_m, bn, num_warps, instr)
        gran_n = instr[1] * warps[1]
        # Not defensive: if this ever fails the loop below never terminates, and the
        # symptom is a test run that spins at 200% CPU for an hour with no output.
        assert (
            bn % gran_n == 0
        ), f"BLOCK_N {bn} not a multiple of warp N-granularity {gran_n}"
        mini_n = min(bn, gran_n * 2)
        while mini_n % gran_n:
            mini_n += gran_n
        # MINI_BLOCK_M/N now also split the LDS staging area, the global->LDS copy and
        # the accumulator, so a sweep over them is a real pipeline experiment, not just
        # an epilogue one. The override is only honoured when it divides the CTA tile
        # and is a whole number of warp tiles -- validate() would reject anything else.
        gran_m = instr[0] * warps[0]
        mini_m = _env_int("AITER_TRITON_MOE_GLUON_MINI_BLOCK_M", block_m)
        if not (
            0 < mini_m <= block_m and block_m % mini_m == 0 and mini_m % gran_m == 0
        ):
            mini_m = block_m
        mini_n_env = _env_int("AITER_TRITON_MOE_GLUON_MINI_BLOCK_N", mini_n)
        if 0 < mini_n_env <= bn and bn % mini_n_env == 0 and mini_n_env % gran_n == 0:
            mini_n = mini_n_env
        out_instr, out_warps, out_tiles = instr, warps, (1, 1)
        out_bk, out_mini_m, out_mini_n = bk, mini_m, mini_n
        if _env_int("AITER_TRITON_MOE_GLUON_FLY", 0) and block_m == 128:
            # The operand shape moe_sort_scales was written against, and the one the
            # FlyDSL port runs: 4 waves each owning 32 contiguous M rows (two 16-row
            # MFMA tiles). Selecting it here rather than in the tuner keeps the shuffle
            # on/off A/B on one config, which is the only way to price the scale path.
            out_instr = (16, 16, 128)
            # (1, 4) is FlyDSL's split: every wave spans all BM rows (so A is broadcast
            # and belongs in LDS) and owns BN/4 columns (so B is wave-private and
            # belongs in registers). (4, 1) inverts that and makes B the broadcast
            # operand, which is why register-resident B costs 4x the global traffic
            # there. Selectable so the two can be compared on one config.
            wn_sel = _env_int("AITER_TRITON_MOE_GLUON_FLY_WARPS_N", 0)
            if wn_sel == 2:
                # 8 waves, so 2 land on each SIMD -- the only arrangement in which the
                # warp pipeliner's ping-pong can actually alternate two waves on one
                # SIMD. (1, 4) and (4, 1) put one wave per SIMD at occupancy 1, where
                # the pipeliner pays its barriers for nothing.
                out_warps = (2, 4)
                # tiles_per_warp[0] >= 2 keeps the M warp stride at 32, which leaves the
                # scale fragment's non-K +16 step in the *registers* rather than across
                # warps -- that is what lets operand A pack four E8M0 bytes into one i32
                # (scale_packed_ok). At tiles (1, n) the M split is 16 and A falls back
                # to eight ds_read_u8 plus a v_perm per dword.
                out_tiles = (
                    _env_int("AITER_TRITON_MOE_GLUON_FLY_TILES_M", 1),
                    _env_int("AITER_TRITON_MOE_GLUON_FLY_TILES_N", 1),
                )
                if block_m % (out_instr[0] * out_warps[0] * out_tiles[0]) != 0:
                    out_tiles = (1, out_tiles[1])
                if bn % (out_instr[1] * out_warps[1] * out_tiles[1]) != 0:
                    out_tiles = (out_tiles[0], 1)
            elif wn_sel:
                # tiles_per_warp is what decides whether a wave's N columns are
                # contiguous. At (1, 1) the four waves interleave every 16 columns, so
                # the scale dword's non-K +16 step crosses waves and CDNA4_SCALE cannot
                # be read one dword per lane. FlyDSL instead gives each wave two
                # adjacent 32-column stripes -- tiles_per_warp (1, 4) here -- which puts
                # +16 back inside a wave's registers.
                tn = _env_int("AITER_TRITON_MOE_GLUON_FLY_TILES_N", 1)
                if tn > 1 and bn % (out_instr[1] * 4 * tn) != 0:
                    tn = 1
                out_warps, out_tiles = (1, 4), (1, tn)
            else:
                out_warps, out_tiles = (4, 1), (2, 1)
            out_bk = 256
            # Whole-block mini tiles by default: the preshuffled scale tiles are
            # read whole, so sorted_shuffled_ok()/_shuffled_b_scales() reject a
            # subdivided config and both shuffles silently switch off. An explicit
            # override is still honoured -- that is the only way to price warp
            # pipelining, which needs NM/NN > 1 to have anything to overlap.
            out_mini_m = _env_int("AITER_TRITON_MOE_GLUON_MINI_BLOCK_M", block_m)
            out_mini_n = _env_int("AITER_TRITON_MOE_GLUON_MINI_BLOCK_N", bn)
            gm = out_instr[0] * out_warps[0]
            gn = out_instr[1] * out_warps[1] * out_tiles[1]
            if not (
                0 < out_mini_m <= block_m
                and block_m % out_mini_m == 0
                and out_mini_m % gm == 0
            ):
                out_mini_m = block_m
            if not (
                0 < out_mini_n <= bn and bn % out_mini_n == 0 and out_mini_n % gn == 0
            ):
                out_mini_n = bn

        # A 16x16x128 operand that is not preshuffled is staged in LDS as 32-row units
        # (_config.py::byte_unit_lds_layout), which only describes what a warp reads
        # while the warp owns both 16-row MFMA tiles of the unit -- so validate()
        # requires tiles_per_warp >= 2 on that axis. Raise it here wherever the tile
        # divides; where it does not (BLOCK_M = 16), byte_unit_lds_ok() is False anyway
        # and the operand keeps compute_efficient_padded_shared_layout.
        if tuple(out_instr) == (16, 16, 128):
            tm, tn = out_tiles
            gm2 = out_instr[0] * out_warps[0] * 2
            gn2 = out_instr[1] * out_warps[1] * 2
            if tm < 2 and block_m % gm2 == 0 and out_mini_m % gm2 == 0:
                tm = 2
            if tn < 2 and bn % gn2 == 0 and out_mini_n % gn2 == 0:
                tn = 2
            out_tiles = (tm, tn)
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": bn,
            "BLOCK_K": out_bk,
            # LDS indices may vary at runtime. Register rings round this requested
            # factor up to a multiple of their active depths' least common multiple.
            "K_UNROLL": _env_int("AITER_TRITON_MOE_GLUON_K_UNROLL", n_buf),
            "MINI_BLOCK_K": out_bk,
            "MINI_BLOCK_M": out_mini_m,
            "MINI_BLOCK_N": out_mini_n,
            "NUM_LDS_BUFFER": n_buf,
            # Zero inherits NUM_LDS_BUFFER in the common component pipeline.
            "A_NUM_BUFFER": _env_int("AITER_TRITON_MOE_GLUON_A_NUM_BUFFER", 0),
            "B_NUM_BUFFER": _env_int("AITER_TRITON_MOE_GLUON_B_NUM_BUFFER", 0),
            "A_SCALE_NUM_BUFFER": _env_int(
                "AITER_TRITON_MOE_GLUON_A_SCALE_NUM_BUFFER",
                _env_int("AITER_TRITON_MOE_GLUON_A_SCALE_NUMB_BUFFER", 0),
            ),
            "B_SCALE_NUM_BUFFER": _env_int(
                "AITER_TRITON_MOE_GLUON_B_SCALE_NUM_BUFFER", 0
            ),
            "mfma_instr_shape": out_instr,
            "warps_per_cta": out_warps,
            "tiles_per_warp": out_tiles,
            # None means "derive from the instruction shape and the operand packing".
            # A literal here would silently pick a different MFMA variant.
            "k_width": None,
            "transposed": True,
            # 0 lets the backend pick. Worth setting: Triton allocates LDS
            # dynamically, so LLVM sees group_segment_fixed_size 0 and models an
            # occupancy the launch cannot reach -- it then derives VGPRCriticalLimit
            # from that and rejects LDS reads on pressure the hardware does not have.
            "WAVES_PER_EU": _env_int("AITER_TRITON_MOE_GLUON_WAVES_PER_EU", 0),
            # Tile-schedule overrides, for sweeping L2 locality without editing source.
            "TILE_SCHED": _env_int(
                "AITER_TRITON_MOE_GLUON_TILE_SCHED", int(TileSched.XCD_GROUP_M)
            ),
            "GROUP_M": _env_int("AITER_TRITON_MOE_GLUON_GROUP_M", 4),
            "NUM_XCDS": _env_int("AITER_TRITON_MOE_GLUON_NUM_XCDS", 8),
            "token_cache_modifier": _env_cache_modifier(
                "TOKEN_CACHE_MODIFIER", "", "TOKEN_MOD"
            ),
            "token_scale_cache_modifier": _env_cache_modifier(
                "TOKEN_SCALE_CACHE_MODIFIER", "", "TOKEN_SCALE_MOD"
            ),
            # .cg (non-temporal) on the weight payload: at decode every line is read
            # once, so streaming it keeps it from evicting anything that is reused.
            # Dropping it costs 10% more HBM traffic.
            "expert_cache_modifier": _env_cache_modifier(
                "EXPERT_CACHE_MODIFIER",
                ".cg" if block_m <= 32 else "",
                "EXPERT_MOD",
            ),
            # ...but NOT on the weight scales. The scale tensor is (E, K/32, N) with K
            # contiguous, so one 128 B line holds 128 consecutive K-scales for a single
            # n, while a BLOCK_K stage consumes only BLOCK_K/32 of them -- 16 bytes at
            # BLOCK_K=512. That line is needed by 8 consecutive K iterations and has to
            # survive in L2; marking it non-temporal turns all 8 touches into separate
            # HBM fetches. Measured on H7168-I2048-E33-k8 T=32 stage 1: 651 -> 517 MB of
            # HBM reads, L2 hit 16% -> 33%, 122.6 -> 96.9 us.
            "expert_scale_cache_modifier": _env_cache_modifier(
                "EXPERT_SCALE_CACHE_MODIFIER", "", "EXPERT_SCALE_MOD"
            ),
            "result_cache_modifier": _env_cache_modifier(
                "RESULT_CACHE_MODIFIER", "", "RESULT_MOD"
            ),
            "result_scale_cache_modifier": _env_cache_modifier(
                "RESULT_SCALE_CACHE_MODIFIER", "", "RESULT_SCALE_MOD"
            ),
            # Which inter-wave ping-pong the K-loop step lays borders down for, a
            # WarpPipeline: 0 none, 1 hand each slot's MFMAs and its ds_read/copy shadow
            # to TritonAMDGPUWarpPipeline as an mfma/mem stage pair, 2 the hand-emitted
            # rendezvous (FROZEN_STEP=1 only). 1 and 2 need VGPR_PREFETCH_K == BLOCK_K.
            # Off by default. Was a bool, and 1 still means what True did.
            "WARP_PIPELINE": _env_int(
                "AITER_TRITON_MOE_GLUON_WARP_PIPELINE", int(WarpPipeline.NONE)
            ),
            # Read the token scales from the moe_sort_scales pre-pass output instead of
            # the raw (M, K/32) tensor. Host-gated: _sorted_shuffle_a_scales() only
            # returns a buffer when the layout matches, so this stays False otherwise.
            "A_SCALE_SORTED_SHUFFLED": False,
            "B_SCALE_SHUFFLED": False,
            "B_PRESHUFFLED": False,
            "B_IN_REG": bool(_env_int("AITER_TRITON_MOE_GLUON_B_IN_REG", 0)),
            "B_SCALE_IN_REG": bool(
                _env_int("AITER_TRITON_MOE_GLUON_B_SCALE_IN_REG", 0)
            ),
            "A_SCALE_IN_REG": bool(
                _env_int("AITER_TRITON_MOE_GLUON_A_SCALE_IN_REG", 0)
            ),
            "ACT_FAST_RCP": bool(_env_int("AITER_TRITON_MOE_GLUON_ACT_FAST_RCP", 0)),
            # PER_OP (1) commits each async copy; PER_SLOT (2) commits each slot.
            # Both wait before each read slot. PER_STAGE (3) commits the whole K
            # stage and waits once at the read stage's head. Counts and copy ownership
            # come from the same schedule; see WaitCommitScheme.
            "WAIT_COMMIT_SCHEME": _env_int(
                "AITER_TRITON_MOE_GLUON_WAIT_COMMIT_SCHEME",
                int(WaitCommitScheme.PER_OP),
            ),
            **ds_read_flags,
            "SCHED_MODE": _env_int(
                "AITER_TRITON_MOE_GLUON_SCHED_MODE", int(SchedMode.NONE)
            ),
            "FROZEN_STEP": bool(_env_int("AITER_TRITON_MOE_GLUON_FROZEN_STEP", 0)),
            "SOFF_UNROLL": bool(_env_int("AITER_TRITON_MOE_GLUON_SOFF_UNROLL", 0)),
            "SCALE_FILL_MID": bool(
                _env_int("AITER_TRITON_MOE_GLUON_SCALE_FILL_MID", 0)
            ),
            # Read stage N's fragments one stage before their MFMA consumes them. Costs
            # one BLOCK_K tile of live registers and one stage of global prefetch depth
            # (the fill is waited on at stage s+NB-1 rather than s+NB), so it wants
            # NUM_LDS_BUFFER >= 3 to break even on the global side. 0 is off; a nonzero
            # m turns it on and sets the wait to wait_group(NB - m), so 1 is the minimal
            # wait and higher m over-waits. Over-waiting only loses: at NB=3 with relaxed
            # loads, T=4096 stage 1 measured 1005.6 us at m=2 (vmcnt(8), one stage
            # outstanding) against 1090.9 us at m=3, where the wait folds to vmcnt(0) and
            # drains every global load three times per loop body.
            # How much of a BLOCK_K stage is read into registers one step before
            # its MFMAs consume it. BLOCK_K (the default) carries the whole stage
            # so a ds_read and its MFMA sit a stage apart and never meet lgkmcnt;
            # 0 reads and consumes in the same step. Intermediate values only
            # become reachable once MINI_BLOCK_K < BLOCK_K.
            "VGPR_PREFETCH_K": _env_int("AITER_TRITON_MOE_GLUON_VGPR_PREFETCH_K", bk),
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
        elif nb > 3:
            nb -= 1
        elif second:
            if shrink_n_first:
                block_k //= 2
            else:
                block_n //= 2
        else:
            break  # gluon_supported's validate() will reject it
        if K // block_k < nb:
            nb = max(3, K // block_k)
        cfg = _build(block_n, block_k, nb)
    return cfg


def _default_launch_config(block_m, N, K, dq_a, dq_b, small_grid, apply_swiglu):
    """Share automatic stage-specific geometry with the capability check."""
    c = get_gluon_config(block_m, N, K, dq_a, dq_b, small_grid)
    if (
        apply_swiglu
        and dq_a == dq_b == DtypeQuant.MXFP8
        and block_m == 128
        and N % 256 == 0
        and K % 256 == 0
        and K // 128 >= 10
        and not c["FROZEN_STEP"]
    ):
        # FP8 K128 has the same payload byte tiles as FP4 K256. The scale
        # preshuffle still packs K256, with consecutive payload stages selecting
        # alternate halves of each dword. Gate/up ordering remains the caller's.
        c = dict(
            c,
            BLOCK_N=256,
            BLOCK_K=128,
            MINI_BLOCK_M=64,
            MINI_BLOCK_N=128,
            MINI_BLOCK_K=128,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(1, 4),
            tiles_per_warp=(2, 2),
            NUM_LDS_BUFFER=3,
            K_UNROLL=6,
            VGPR_PREFETCH_K=128,
            A_SCALE_SORTED_SHUFFLED=True,
            B_SCALE_SHUFFLED=True,
        )
    return c


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
    w_static_scale=None,
    apply_swiglu: bool = False,
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

    cfg = _default_launch_config(
        block_m,
        N,
        K,
        dq_a,
        dq_b,
        _small_grid(routing_data, y.shape[1], N),
        apply_swiglu,
    )
    if N % cfg["BLOCK_N"] != 0:
        return False, f"N {N} % BLOCK_N {cfg['BLOCK_N']} != 0"
    if K % cfg["BLOCK_K"] != 0:
        return False, f"K {K} % BLOCK_K {cfg['BLOCK_K']} != 0"
    pipeline_error = _pipeline_error(
        tuple(sorted((key, _hashable(value)) for key, value in cfg.items())),
        dq_a,
        dq_b,
        K,
    )
    if pipeline_error is not None:
        return False, pipeline_error
    # The fill schedule hands each slot of the NM x NN walk one mini-block copy, so it
    # needs at least as many slots as copies -- both axes split. Refused here rather
    # than in validate() so a tile that cannot split falls back instead of failing the
    # compile; validate() asserts the same thing as the in-kernel backstop.
    if (
        cfg["BLOCK_M"] // cfg["MINI_BLOCK_M"] < 2
        or cfg["BLOCK_N"] // cfg["MINI_BLOCK_N"] < 2
    ):
        return False, (
            f"the fill schedule needs both axes split: BLOCK_M {cfg['BLOCK_M']} / "
            f"MINI_BLOCK_M {cfg['MINI_BLOCK_M']} and BLOCK_N {cfg['BLOCK_N']} / "
            f"MINI_BLOCK_N {cfg['MINI_BLOCK_N']} must each give at least 2 mini blocks"
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
):
    """Host-side constexpr work, memoised.

    The host aggregates and launch constants depend only on compile-time values,
    so they are computed once per distinct configuration.
    """
    from triton.experimental.gluon import language as gl

    c = (
        dict(config_items)
        if config_items
        else _default_launch_config(
            block_m, N, K, dq_a, dq_b, small_grid, act is not None
        )
    )
    assert c["BLOCK_M"] == block_m, (
        f"config BLOCK_M {c['BLOCK_M']} must match routing block_m {block_m}"
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
        gate_up_split,
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
    tuning_cfg_host.validate_buffer_counts()
    tuning_cfg_host.num_k_tiles(K)
    grid_n = tuning_cfg_host.grid_N(N)
    grid_n = grid_n.value if hasattr(grid_n, "value") else grid_n
    # Keep each spec in one constexpr so launch specialization sees four leaves.
    constexpr_args = (
        gl.constexpr(func_spec),
        gl.constexpr(tuning_spec),
        gl.constexpr(N),
        gl.constexpr(K),
    )
    num_warps = c["warps_per_cta"][0] * c["warps_per_cta"][1]
    return grid_n, constexpr_args, num_warps, c["WAVES_PER_EU"], c, tuning_cfg_host


@cache
def _validate_launch_config(tuning_cfg, N, K):
    """Validate each cached effective layout once, before any kernel launch."""
    tuning_cfg.validate(N, K)
    return _validate_selected_pipeline(tuning_cfg, K)


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


def _scale_shuffle_supported(cfg, operand):
    """Whether this operand can consume the packed K256 scale byte order."""
    block_k = int(cfg["BLOCK_K"])
    if tuple(cfg["mfma_instr_shape"]) != (16, 16, 128) or block_k not in (128, 256):
        return False
    if block_k == 128:
        # Each full dword serves two K128 stages. Both non-K bytes must belong
        # to this wave; the component pipeline tracks the phase across any unroll.
        return (
            int(cfg["MINI_BLOCK_K"]) == 128
            and not cfg.get("FROZEN_STEP", False)
            and int(cfg["tiles_per_warp"][operand]) >= 2
        )
    return True


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
    # Mirrors KernelTuningConfig.sorted_shuffled_ok(): the permutation is carried by the
    # LDS tile's layout. K128 payload stages consume alternate halves of the same
    # K256 scale dwords, so their host-side permutation and allocation are identical.
    if (
        not _scale_shuffle_supported(cfg, 0)
        or x_scales.ndim != 2
        or x_scales.shape[1] != K // MX_GROUP_SIZE
        or not x_scales.is_contiguous()
        or block_m % 32 != 0
        # The buffer is one 256 B run per 32-row stripe, so a mini block that is a
        # whole number of stripes is a contiguous slice of it; the kernel indexes the
        # slice with a stripe offset. It does not have to be the whole block.
        or int(cfg["MINI_BLOCK_M"]) % 32 != 0
        or K % 256 != 0
    ):
        return None

    from aiter.ops.moe_mxfp4_aux import mxfp4_moe_sort_scales

    dev = x_scales.device
    expt_data = routing_data.expt_data
    n_expts_act = routing_data.n_expts_act
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
    try:
        mxfp4_moe_sort_scales(
            x_scales,
            sorted_token_ids,
            cumsum,
            out,
            int(expt_data.hist.numel()),
            int(n_expts_act),
            int(K),  # D_HIDDEN
            int(block_m),  # MB
            max_sorted,
        )
    except RuntimeError as e:
        # The C++ side is a codegen'd template keyed on (BM, NE, D_HIDDEN); a shape
        # outside csrc/.../codegen/gen_instances.py::SHAPES has no instance. Fall back
        # to the raw scale path rather than failing the launch.
        _LOGGER.info(f"sorted-shuffled A scales unavailable, using raw scales: {e}")
        return None
    return out


#: Attribute the shuffled copy is memoised under, on the source tensor itself. The
#: weights are static, so in production the shuffle belongs in the checkpoint-load path;
#: memoising keeps the benchmark honest (paid once, not per call) without changing every
#: caller. It hangs off the tensor rather than a dict keyed by data_ptr because a freed
#: tensor's address is reused, and a stale entry then silently pairs one expert's scales
#: with another's weights -- which is exactly what it did before this was fixed.
_B_SCALE_SHUFFLE_ATTR = "_aiter_gluon_cdna4_scale_shuffled"
#: Memoised 16-column-blocked copy of the weight payload, on the source tensor.
_B_PRESHUFFLE_ATTR = "_aiter_gluon_b_preshuffled"


def _preshuffled_b(w, cfg, N, K, dq_b):
    """Weights permuted 16-column blocked, or None if the config cannot read them.

    Byte ``(n, k)`` of an expert moves to
        ``(n//16)*(KB*16) + (k//16)*256 + (n%16)*16 + k%16``
    where ``KB = K // pack_divisor`` is the stored K extent, so a wave's 64 lanes --
    lane L wanting bytes ``[16L, 16L+16)`` of a 16-column, 128-byte tile -- cover one
    contiguous 1024 B run. Read through the operand-B fragment layout, the plain
    ``(E, KB, N)`` tensor instead puts consecutive lanes ``stride(-2)`` apart: same L2
    and HBM traffic, 2.8x the L1 accesses, 42% slower. This is the same permute
    ``utils/shuffle.py::shuffle_weight(w, (16, 16))`` applies, and the same one FlyDSL's
    port requires of its caller.

    ``byte_unit_lds_layout`` gives the shared tile the matching permutation, so the
    direct-to-LDS copy stays one linear run and the unit drops from 32 rows to 16 and
    loses its padding.
    """
    if w is None or w.ndim != 3:
        return None
    if int(cfg["BLOCK_K"]) % 32 != 0 or N % 16 != 0 or K % 32 != 0:
        return None
    KB = K // dq_pack_divisor(dq_b)
    if KB % 16 != 0:
        return None
    hit = getattr(w, _B_PRESHUFFLE_ATTR, None)
    if hit is not None:
        return hit

    E = w.shape[0]
    # (E, KB, N) -> (E, N, KB) -> blocked (E, N/16, KB/16, 16, 16) -> flat
    out = (
        w.transpose(-1, -2)
        .contiguous()
        .reshape(E, N // 16, 16, KB // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(E, N // 16 * KB * 16)
        .contiguous()
    )
    try:
        setattr(w, _B_PRESHUFFLE_ATTR, out)
    except AttributeError:
        pass
    return out


def _shuffled_b_scales(w_scales, cfg):
    """CDNA4_SCALE-preshuffled weight scales, or None if the config cannot read them.

    Reuses utils/shuffle.py::shuffle_scale_moe, whose gfx950 permute
    (preshuffle_factor 32, scale_kwidth 8) lands each element in exactly the slot
    ``get_mfma_scale_layout`` assigns for a 16x16x128 operand -- verified base for base
    against the layout, same as the A-side shuffle.
    """
    if w_scales is None:
        return None
    if (
        not _scale_shuffle_supported(cfg, 1)
        # Stripe-based like the A shuffle: one 256 B run per 32 columns, so a mini
        # block that is a whole number of stripes is a contiguous slice of the tile.
        or int(cfg["MINI_BLOCK_N"]) % 32 != 0
        or w_scales.shape[-1] % 32 != 0
        or w_scales.shape[-2] % 8 != 0
    ):
        return None
    hit = getattr(w_scales, _B_SCALE_SHUFFLE_ATTR, None)
    if hit is None:
        from aiter.ops.triton.utils.shuffle import _shuffle_scale_tile_gfx950

        # The tile permute itself, not shuffle_scale_moe: that transposes the result
        # back to (E, K*, N/32) for the Triton kernels, and materialising that view
        # would undo the very byte order the shuffle exists to produce. What the kernel
        # reads is the pre-transpose (E, N/32, K) tile, which is already contiguous.
        # The N rows keep their input order, including separate gate/up halves.
        hit = _shuffle_scale_tile_gfx950(w_scales.transpose(-1, -2), 32, 8)
        try:
            setattr(w_scales, _B_SCALE_SHUFFLE_ATTR, hit)
        except AttributeError:
            pass  # not memoisable (e.g. a plain view); correctness is unaffected
    return hit


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
        # Same hazard as _B_SCALE_SHUFFLE_ATTR above, same fix.
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
):
    """Launch the Gluon grouped GEMM into ``y`` (shape ``(1, M, N // ARN)``).

    Operand dtypes are inferred from the tensors -- ``uint8`` + scales is MXFP4, fp8 +
    scales is MXFP8, fp8 alone is FP8_E4M3 (unit scales, still on the scaled pipe),
    bf16 alone is BF16 -- so every ``moe_gemm_*`` op calls this one function.
    :func:`gluon_supported` must have said yes for these tensors first.
    An explicit config's ``BLOCK_M`` must match ``routing_data.block_m`` because
    the routing offsets and block map are built for that geometry.

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
            _small_grid(routing_data, y.shape[1], N),
            bias is not None,
            gammas is not None,
            gather_indx is not None,
            x_static_scale is not None,
            act,
            out_quant,
            cfg_items,
            gate_up_split,
            epilogue,
        )

    grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc = _spec(
        tuple(sorted((k, _hashable(v)) for k, v in config.items())) if config else None
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
            # still needs is the packed K stride: eight scales span 256 bytes.
            # A K128 kernel reuses that scale tile over two payload stages.
            a_scale_stride_m, a_scale_stride_k = 0, 32
    a_shuffled = a_swizzle == ScaleSwizzle.SORTED_SHUFFLED
    b_scales, b_swizzle = w_scales, ScaleSwizzle.NONE
    b_scale_stride_k = 0 if w_scales is None else w_scales.stride(1)
    if _cfg.get("B_SCALE_SHUFFLED", False) or _env_int(
        "AITER_TRITON_MOE_GLUON_SHUFFLED_W_SCALES", 0
    ):
        shuf_w = _shuffled_b_scales(w_scales, _cfg)
        if shuf_w is not None:
            b_scales, b_swizzle = shuf_w, ScaleSwizzle.CDNA4_SCALE
            b_scale_stride_k = 32
    b_shuffled = b_swizzle == ScaleSwizzle.CDNA4_SCALE
    if int(_cfg["BLOCK_K"]) == 128 and not (a_shuffled and b_shuffled):
        # The K128 packed MFMA requires matching packed-scale representations on
        # both operands. An unavailable sorter or a one-sided request uses raw
        # scales on both sides, with their original pointers and strides.
        a_scales, a_swizzle = x_scales, ScaleSwizzle.NONE
        a_scale_stride_m = 0 if x_scales is None else x_scales.stride(0)
        a_scale_stride_k = 0 if x_scales is None else x_scales.stride(1)
        b_scales, b_swizzle = w_scales, ScaleSwizzle.NONE
        b_scale_stride_k = 0 if w_scales is None else w_scales.stride(1)
        a_shuffled = b_shuffled = False
    if (
        bool(_cfg.get("A_SCALE_SORTED_SHUFFLED", False)) != a_shuffled
        or bool(_cfg.get("B_SCALE_SHUFFLED", False)) != b_shuffled
    ):
        _cfg = dict(
            _cfg, A_SCALE_SORTED_SHUFFLED=a_shuffled, B_SCALE_SHUFFLED=b_shuffled
        )
        grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc = _spec(
            tuple(sorted((k, _hashable(v)) for k, v in _cfg.items()))
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
    if _env_int("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", 0):
        shuf_b = _preshuffled_b(w, _cfg, N, K, dq_b)
        if shuf_b is not None:
            w_payload = shuf_b
            _cfg = dict(_cfg, B_PRESHUFFLED=True)
            grid_n, constexpr_args, num_warps, waves_per_eu, _cfg, _tc = _spec(
                tuple(sorted((k, _hashable(v)) for k, v in _cfg.items()))
            )
    if _cfg.get("B_IN_REG", False) and not _cfg.get("B_PRESHUFFLED", False):
        raise ValueError("B_IN_REG requires B_PRESHUFFLED weights")
    _validate_launch_config(_tc, N, K)

    b = QuantExpertTensor.make(
        dq_b,
        w_payload,
        b_scales,
        w.stride(0),
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
    if _LLVM_FN_ATTRS:
        compiled = kernel[(grid_x,)](
            *args,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            llvm_fn_attrs=_LLVM_FN_ATTRS,
        )
    else:
        compiled = kernel[(grid_x,)](
            *args,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
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


_TUNING_KEYS = TuningSpec._fields


def _tuning_args(c: dict) -> tuple:
    """Keep older config dictionaries valid when optional tuning fields are added."""
    aliases = {
        "token_mod": "token_cache_modifier",
        "token_scale_mod": "token_scale_cache_modifier",
        "expert_mod": "expert_cache_modifier",
        "expert_scale_mod": "expert_scale_cache_modifier",
        "result_mod": "result_cache_modifier",
        "result_scale_mod": "result_scale_cache_modifier",
    }
    for legacy, canonical in aliases.items():
        if legacy in c:
            if canonical in c and c[canonical] != c[legacy]:
                raise ValueError(f"{canonical} and legacy {legacy} disagree")
            c = dict(c, **{canonical: c[legacy]})
    if "DS_READ_IN_MFMA" in c:
        legacy_flags = _ds_read_flags(int(c["DS_READ_IN_MFMA"]))
        for name, value in legacy_flags.items():
            if name in c and bool(c[name]) != value:
                raise ValueError(f"{name} and legacy DS_READ_IN_MFMA disagree")
        c = dict(c, **legacy_flags)
    if "A_SCALE_NUMB_BUFFER" in c:
        alias = c["A_SCALE_NUMB_BUFFER"]
        canonical = c.get("A_SCALE_NUM_BUFFER", 0)
        if canonical and canonical != alias:
            raise ValueError("A_SCALE_NUM_BUFFER and A_SCALE_NUMB_BUFFER disagree")
        c = dict(c, A_SCALE_NUM_BUFFER=alias)
    return tuple(
        c[k] if k in c else TuningSpec._field_defaults[k] for k in _TUNING_KEYS
    )
