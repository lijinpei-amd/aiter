# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch configs of the gfx950 Gluon MoE GEMMs: the ladder, the measured overrides
and the host probes that check a config before launch.

``moe_op_gemm_gluon.py`` launches the kernels with these configs. This module does not
import torch.
"""

from __future__ import annotations

import math
import os
from functools import cache

from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import KernelLayout
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    B_SCALE,
    LDS_USABLE_BYTES,
    DSReadOperand,
    DtypeQuant,
    SchedMode,
    TileSched,
    TuningSpec,
    WaitCommitScheme,
    WarpPipeline,
    dq_pack_divisor,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import (
    validate_pipeline as _validate_live_pipeline,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch

#: MX group size. Fixed by the OCP microscaling spec for both E2M1 and E4M3; not a knob.
MX_GROUP_SIZE = 32


_SUPPORTED_BLOCK_M = (16, 32, 64, 128)


_COMPONENT_BUFFER_KEYS = (
    "A_NUM_BUFFER",
    "B_NUM_BUFFER",
    "A_SCALE_NUM_BUFFER",
    "B_SCALE_NUM_BUFFER",
)


_ALLOW_VIA_LDS_KEYS = (
    "ALLOW_B_VIA_LDS",
    "ALLOW_B_SCALE_VIA_LDS",
    "ALLOW_A_SCALE_VIA_LDS",
)


#: Knobs renamed with inverted polarity. Rejected rather than aliased: an old key that
#: was silently dropped would fall back to LDS placement with no error.
_INVERTED_KNOB_RENAMES = {
    "B_IN_REG": "ALLOW_B_VIA_LDS",
    "B_SCALE_IN_REG": "ALLOW_B_SCALE_VIA_LDS",
    "A_SCALE_IN_REG": "ALLOW_A_SCALE_VIA_LDS",
}


_SCALE_SLOT_KEYS = (
    "SCALE_SLOT_M",
    "SCALE_SLOT_N",
    "SCALE_SLOT_K",
)


#: Slot extents under their pre-slot names. Still accepted, as config keys and as
#: ``AITER_TRITON_MOE_GLUON_<name>`` variables; the new name wins in the environment.
_LEGACY_SLOT_KEYS = {
    "SLOT_M": "MINI_BLOCK_M",
    "SLOT_N": "MINI_BLOCK_N",
    "SCALE_SLOT_M": "SCALE_MINI_BLOCK_M",
    "SCALE_SLOT_N": "SCALE_MINI_BLOCK_N",
    "SCALE_SLOT_K": "SCALE_MINI_BLOCK_K",
}


#: Slot keys whose legacy name counted another unit: SCALE_MINI_BLOCK_K counted payload
#: K, SCALE_SLOT_K counts scale columns, one per MX group. A legacy value is divided.
_LEGACY_SLOT_DIVISORS = {"SCALE_SLOT_K": MX_GROUP_SIZE}


_DS_READ_FIELDS = (
    ("DS_READ_A_PAYLOAD_IN_MFMA", DSReadOperand.A),
    ("DS_READ_A_SCALE_IN_MFMA", DSReadOperand.A_SCALE),
    ("DS_READ_B_PAYLOAD_IN_MFMA", DSReadOperand.B),
    ("DS_READ_B_SCALE_IN_MFMA", DSReadOperand.B_SCALE),
)


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def _cfg_key(cfg: dict) -> tuple:
    """A config dict as the hashable, order-independent key every memo here uses."""
    return tuple(sorted((k, _hashable(v)) for k, v in cfg.items()))


def _cval(v):
    """Take the value out of a ``gl.constexpr`` a config method handed back."""
    return v.value if hasattr(v, "value") else v


#: operand dtypes that carry an E8M0 group-32 scale tile alongside the payload
_SCALED = (DtypeQuant.MXFP4, DtypeQuant.MXFP8)


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


def _from_legacy_slot(name: str, value: int) -> int:
    """A value given under ``name``'s legacy key, in the unit ``name`` counts."""
    divisor = _LEGACY_SLOT_DIVISORS.get(name.upper(), 1)
    if value % divisor:
        legacy = _LEGACY_SLOT_KEYS[name.upper()]
        raise ValueError(f"legacy {legacy} {value} is not a multiple of {divisor}")
    return value // divisor


def _env_slot_int(name: str, default: int) -> int:
    """A slot extent from the environment, under its name or its legacy name."""
    prefix = "AITER_TRITON_MOE_GLUON_"
    legacy = _env_int(prefix + _LEGACY_SLOT_KEYS[name], None)
    legacy = default if legacy is None else _from_legacy_slot(name, legacy)
    return _env_int(prefix + name, legacy)


def _canonical_slot_keys(c: dict) -> dict:
    """``c`` with its legacy slot keys renamed, upper- or lowercase.

    Lowercase is accepted for the scale slots only, as it was before the rename. A
    legacy value is converted to its new name's unit (``_LEGACY_SLOT_DIVISORS``). A
    legacy key that disagrees with its new name is rejected.
    """
    renames = list(_LEGACY_SLOT_KEYS.items()) + [
        (key.lower(), _LEGACY_SLOT_KEYS[key].lower()) for key in _SCALE_SLOT_KEYS
    ]
    for name, legacy in renames:
        if legacy in c:
            value = _from_legacy_slot(name, c[legacy])
            if name in c and c[name] != value:
                raise ValueError(f"{name} and legacy {legacy} disagree")
            c = {k: v for k, v in c.items() if k != legacy}
            c[name] = value
    return c


def _slot_override(name: str, default: int, block: int, warp_gran: int) -> int:
    """A SLOT_M/N environment override, ignored unless it tiles the CTA block.

    The slot splits the LDS staging area, the global->LDS copy and the accumulator,
    not just the epilogue, so an override that is not a whole number of warp tiles
    dividing the block would be rejected by ``validate()`` -- fall back rather than
    fail the launch.
    """
    value = _env_slot_int(name, default)
    if 0 < value <= block and block % value == 0 and value % warp_gran == 0:
        return value
    return default


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
    caller's ``slot_n`` alignment walk step past ``block_n`` forever. The N constraint
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
        _cfg_key(cfg), dq_a, dq_b
    )


def _probe_lds_bytes_uncached(cfg: dict, dq_a, dq_b) -> int:
    """LDS footprint of a candidate config, computed by the *aggregate* rather than by a
    second copy of the arithmetic -- the host and the device must not be able to
    disagree about whether a config fits."""
    tc = _probe_tuning_config(cfg, dq_a, dq_b)
    return _cval(KernelLayout(tc).lds_bytes)


@cache
def _probe_b_scale_via_lds(cfg_items: tuple, dq_a, dq_b) -> bool:
    """Whether the layout copies the B scale through LDS, asked of the aggregate for the
    same reason as ``_probe_lds_bytes``. A geometry KernelLayout rejects answers False;
    validate() reports it."""
    try:
        tc = _probe_tuning_config(dict(cfg_items), dq_a, dq_b)
        return bool(_cval(KernelLayout(tc).component_via_lds(B_SCALE)))
    except AssertionError:
        return False


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


def _validate_selected_pipeline(tuning_cfg, layout, K):
    """Validate the pipeline the build runs. ``layout`` is the layout, or None."""
    if layout is None:
        layout = KernelLayout(tuning_cfg)
    return _validate_live_pipeline(tuning_cfg, layout, K)


@cache
def _pipeline_error(cfg_items: tuple, dq_a, dq_b, K):
    """Share the exact device ring and minimum-strip contract with host fallback."""
    try:
        tc = _probe_tuning_config(dict(cfg_items), dq_a, dq_b)
        _validate_selected_pipeline(tc, None, K)
    except AssertionError as error:
        return str(error)
    return None


@cache
def _get_gluon_config_cached(
    block_m: int,
    N: int,
    K: int,
    dq_a,
    dq_b,
    small_grid: bool,
    stream_expert_payload: bool,
) -> tuple:
    return tuple(
        sorted(
            get_gluon_config_uncached(
                block_m,
                N,
                K,
                dq_a,
                dq_b,
                small_grid,
                stream_expert_payload,
            ).items()
        )
    )


def get_gluon_config(
    block_m: int,
    N: int,
    K: int,
    dq_a,
    dq_b,
    small_grid: bool = False,
    stream_expert_payload: bool = False,
) -> dict:
    """Cached view of :func:`get_gluon_config_uncached`. The decode path issues one of
    these per launch and the launch path *is* the critical path there."""
    return dict(
        _get_gluon_config_cached(
            block_m,
            N,
            K,
            dq_a,
            dq_b,
            bool(small_grid),
            bool(stream_expert_payload),
        )
    )


def get_gluon_config_uncached(
    block_m: int,
    N: int,
    K: int,
    dq_a,
    dq_b,
    small_grid: bool = False,
    stream_expert_payload: bool = False,
) -> dict:
    """Explicit Python ladder keyed on shape, dtypes and launch-reuse regime.

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
        # Named apart from the enclosing `gran_n`, which is the coarsest granularity
        # any warp split could need and is what the BLOCK_N shrink walk steps by.
        warp_gran_n = instr[1] * warps[1]
        warp_gran_m = instr[0] * warps[0]
        # Not defensive: if this ever fails the loop below never terminates, and the
        # symptom is a test run that spins at 200% CPU for an hour with no output.
        assert bn % warp_gran_n == 0, (
            f"BLOCK_N {bn} not a multiple of warp N-granularity {warp_gran_n}"
        )
        slot_n = min(bn, warp_gran_n * 2)
        while slot_n % warp_gran_n:
            slot_n += warp_gran_n
        slot_m = _slot_override("SLOT_M", block_m, block_m, warp_gran_m)
        slot_n = _slot_override("SLOT_N", slot_n, bn, warp_gran_n)
        out_instr, out_warps, out_tiles = instr, warps, (1, 1)
        out_bk, out_slot_m, out_slot_n = bk, slot_m, slot_n
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
            # Each payload stage carries one whole BLOCK_K fragment. M/N can still be
            # split into pipeline slots, while shuffled scales have independent extents.
            out_slot_m = _slot_override(
                "SLOT_M", block_m, block_m, out_instr[0] * out_warps[0]
            )
            out_slot_n = _slot_override(
                "SLOT_N",
                bn,
                bn,
                out_instr[1] * out_warps[1] * out_tiles[1],
            )

        # A 16x16x128 operand that is not preshuffled is staged in LDS as 32-row units
        # (_layout.py::byte_unit_lds_layout), which only describes what a warp reads
        # while the warp owns both 16-row MFMA tiles of the unit -- so validate()
        # requires tiles_per_warp >= 2 on that axis. Raise it here wherever the tile
        # divides; where it does not (BLOCK_M = 16), payload_byte_unit_lds_ok() is
        # False anyway and the operand keeps compute_efficient_padded_shared_layout.
        if tuple(out_instr) == (16, 16, 128):
            tm, tn = out_tiles
            gm2 = out_instr[0] * out_warps[0] * 2
            gn2 = out_instr[1] * out_warps[1] * 2
            if tm < 2 and block_m % gm2 == 0 and out_slot_m % gm2 == 0:
                tm = 2
            if tn < 2 and bn % gn2 == 0 and out_slot_n % gn2 == 0:
                tn = 2
            out_tiles = (tm, tn)
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": bn,
            "BLOCK_K": out_bk,
            # The effective factor is the LCM of this request, every active
            # payload/scale buffer period and the shuffled-scale K cadence.
            "K_UNROLL": _env_int("AITER_TRITON_MOE_GLUON_K_UNROLL", n_buf),
            "SLOT_M": out_slot_m,
            "SLOT_N": out_slot_n,
            # Zero inherits the corresponding payload M/N extent. Shuffled scale K
            # counts scale columns and defaults to max(8, BLOCK_K / 32); an override
            # must be at least that wide and an integer multiple of BLOCK_K / 32.
            **{key: _env_slot_int(key, 0) for key in _SCALE_SLOT_KEYS},
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
            # .cg (non-temporal) on the weight payload, unconditionally. The hint used
            # to be gated on block_m <= 32, on the theory that it suits a one-pass
            # payload and that the larger block_m of prefill reads each expert's
            # weights from several M tiles, so retaining them should pay.
            #
            # It does not. The payload is 514 MB on H7168-I2048-E33-k8 -- far past any
            # cache -- so it cannot survive between the M tiles that read it, and
            # retaining it only displaces the tokens and scales that do. Measured on
            # that shape across 10 tuned geometries that lacked the hint, every one
            # improved and none regressed:
            #
            #   block_m 64  (T=256):   MXFP8xMXFP4 -19.2%,  BF16 -15.0%
            #   block_m 128 (T=1024):  MXFP4xMXFP4 -15.6%,  MXFP8xMXFP4 -11.3%,
            #                          MXFP8xMXFP8  -8.2%,  BF16  -6.0%
            #   block_m 128 (T=4096):  MXFP8xMXFP8  -2.7%,  BF16  -2.6%,
            #                          MXFP8xMXFP4  -0.9%,  MXFP4xMXFP4 -0.3%
            #
            # The two geometries that already carried a hint moved -0.07% and +0.02%,
            # which is the drift floor of that harness. bench_out/gluon_bm64_cg_20260921,
            # gluon_larget_20260921, gluon_larget_a4bf16_20260921.
            #
            # stream_expert_payload and block_m no longer select anything here; explicit
            # canonical/legacy environment settings remain authoritative.
            "expert_cache_modifier": _env_cache_modifier(
                "EXPERT_CACHE_MODIFIER", ".cg", "EXPERT_MOD"
            ),
            # ...but NOT on the weight scales. The scale tensor is (E, K/32, N) with K
            # contiguous, so one 128 B line holds 128 consecutive K-scales for a single
            # n, while a BLOCK_K stage consumes only BLOCK_K/32 of them -- 16 bytes at
            # BLOCK_K=512. That line is needed by 8 consecutive K iterations and has to
            # survive in L2; marking it non-temporal turns all 8 touches into separate
            # HBM fetches. Measured on H7168-I2048-E33-k8 T=32 stage 1: 651 -> 517 MB of
            # HBM reads, L2 hit 16% -> 33%, 122.6 -> 96.9 us.
            # That holds for MXFP8 weights, and it is why this stays empty for them.
            # It does not hold for MXFP4 weights: the payload is half the bytes, so the
            # scale stream is twice the share of the weight traffic, and the L2 it
            # occupies costs more than its own reuse returns. Measured on
            # H7168-I2048-E33-k8, 400 cold samples per cell, `.cg` on the scales:
            #
            #   MXFP4 x MXFP4  -21.7% -21.4% -17.3% -1.7% -0.3%  (T=16..4096)
            #   MXFP8 x MXFP4   -4.3%  -4.2% -11.8% -1.0% -0.1%
            #
            # All ten improved. BF16 weights, which have no scale tensor at all, moved
            # -0.04%..+0.40% across the same sweep, which is that harness's floor and
            # the null control for it. MXFP8 weights are left alone because they are
            # genuinely mixed there: +3.1%/+3.2% at T=16/64 against -10.7% at T=256.
            # bench_out/gluon_scale_hint_20260923. Re-measured 2026-10-18 on the
            # current pins: +2.2% +2.4% -11.1% -2.5% -0.3% (T=16..4096),
            # bench_out/gluon_moe_regression/harness/cold_a8w8{nocg,cg}. The two
            # MXFP8 exceptions -- shuffled scales (the T>=256 tiles) and BLOCK_K 128
            # through LDS -- are applied in _default_launch_config.
            "expert_scale_cache_modifier": _env_cache_modifier(
                "EXPERT_SCALE_CACHE_MODIFIER",
                ".cg" if dq_b == DtypeQuant.MXFP4 else "",
                "EXPERT_SCALE_MOD",
            ),
            "result_cache_modifier": _env_cache_modifier(
                "RESULT_CACHE_MODIFIER", "", "RESULT_MOD"
            ),
            "result_scale_cache_modifier": _env_cache_modifier(
                "RESULT_SCALE_CACHE_MODIFIER", "", "RESULT_SCALE_MOD"
            ),
            # Which inter-wave ping-pong the K-loop step lays borders down for, a
            # WarpPipeline: 0 none, 1 hand each pane's MFMAs and its ds_read/copy
            # shadow to TritonAMDGPUWarpPipeline as an mfma/mem stage pair. 1 needs
            # VGPR_PREFETCH_K == BLOCK_K.
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
            "ALLOW_B_VIA_LDS": bool(
                _env_int("AITER_TRITON_MOE_GLUON_ALLOW_B_VIA_LDS", 1)
            ),
            "ALLOW_B_SCALE_VIA_LDS": bool(
                _env_int("AITER_TRITON_MOE_GLUON_ALLOW_B_SCALE_VIA_LDS", 1)
            ),
            "ALLOW_A_SCALE_VIA_LDS": bool(
                _env_int("AITER_TRITON_MOE_GLUON_ALLOW_A_SCALE_VIA_LDS", 1)
            ),
            # SwiGLU's reciprocal via v_rcp_f32 rather than the IEEE divide, which
            # lowers to v_div_scale_f32 plus a Newton refinement. ~1 ulp of relative
            # error against a result stored as BF16, whose own quantum is ~2^-8, so it
            # cannot move the stored value except on a tie; the FlyDSL port this is
            # measured against uses rocdl.rcp for the same reason
            # (_triton_kernels/moe/activations.py:58-66).
            #
            # Default on: measured a win in all 12 cells swept, none regressed --
            # MXFP4 -2.8%/-3.5% at T=1024/4096, MXFP8xMXFP4 -0.9%/-2.5%/-1.8% and
            # MXFP8xMXFP8 -0.6%/-1.8%/-1.9% at T=256/1024/4096, BF16 -0.5%/-0.9%/-0.9%.
            # bench_out/gluon_flydsl_final_20260922/{coldk,coldrcp}. It was previously
            # reachable only through the environment, and only the hand-pinned MXFP4
            # cases set it -- which is why those benchmarked ~3% faster than the ladder.
            "ACT_FAST_RCP": bool(_env_int("AITER_TRITON_MOE_GLUON_ACT_FAST_RCP", 1)),
            # PER_OP (1) commits each async copy; PER_PANE (2) commits each pane.
            # Both wait before each DS-read slot. PER_STAGE (3) commits the whole K
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
            "SOFF_UNROLL": bool(_env_int("AITER_TRITON_MOE_GLUON_SOFF_UNROLL", 0)),
            "SCALE_FILL_MID": bool(
                _env_int("AITER_TRITON_MOE_GLUON_SCALE_FILL_MID", 0)
            ),
            "UNROLL_EPILOGUE": True,
            # Read the complete BLOCK_K payload stage into registers one pipeline step
            # before its MFMAs consume it. The pipeline requires this
            # whole-stage handoff; there is no disabled or partial K-prefetch mode.
            "VGPR_PREFETCH_K": _env_int("AITER_TRITON_MOE_GLUON_VGPR_PREFETCH_K", bk),
        }

    # Which axis to give up first when the tile does not fit.
    #
    # With microscaled operands, keep the payload K tile wide when possible. The
    # shuffled scale slot independently defaults to max(256, BLOCK_K) payload K, so a
    # narrower payload stage retains and reuses a coalesced wider scale load. Narrow N
    # first and accept the loss of arithmetic intensity before shortening that payload
    # stream.
    #
    # With no scales at all (bf16 x bf16) that argument does not exist, and narrowing N
    # is pure loss -- it was picking BLOCK_N=64 where BLOCK_N=128 with a shorter K fits
    # just as well. Shorten K first.
    shrink_n_first = dq_a in _SCALED or dq_b in _SCALED

    def _over_budget(c):
        # KernelLayout asserts its tile geometry, and an env override (SLOT_*,
        # SCALE_SLOT_*, FLY_TILES_*) can break it. Shrinking cannot repair that, so
        # stop here and let gluon_supported's _pipeline_error report it as a fallback.
        try:
            return _probe_lds_bytes(c, dq_a, dq_b) > LDS_USABLE_BYTES
        except AssertionError:
            return False

    cfg = _build(block_n, block_k, nb)
    while _over_budget(cfg):
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


#: Tuned GEMM2 (no activation) families, ``configs/gfx950/gluon/moe/<family>/``.
_GEMM2_FAMILIES = {
    (DtypeQuant.MXFP4, DtypeQuant.MXFP4): "A4W4-GEMM2",
    (DtypeQuant.MXFP8, DtypeQuant.MXFP4): "A8W4-GEMM2",
    (DtypeQuant.MXFP8, DtypeQuant.MXFP8): "A8W8-GEMM2",
}


def _gemm2_tuned(block_m, dq_a, dq_b) -> dict:
    """The ``bm<block_m>_any`` entry of the operands' GEMM2 family, or {}; a copy,
    with JSON lists as the tuples the config uses."""
    family = _GEMM2_FAMILIES.get((dq_a, dq_b))
    if family is None:
        return {}
    entry = get_moe_dispatch(family, get_arch(), "gluon").get(f"bm{block_m}_any", {})
    return {k: tuple(v) if isinstance(v, list) else v for k, v in entry.items()}


@cache
def _gemm2_launch_config(
    ladder_items: tuple, block_m, N, K, dq_a, dq_b, group_m: int | None
) -> tuple:
    """GEMM2 (no activation) config: the ladder with its family's tuned entry on top,
    as sorted config items. ``group_m`` is the GROUP_M environment override, if any.

    The entry applies only where the launch can run the result as is; elsewhere the
    ladder stays. Memoised because ``gluon_supported`` asks for it on every launch.
    """
    c = dict(ladder_items)
    tuned = _gemm2_tuned(block_m, dq_a, dq_b)
    # The entries were measured with N and K multiples of 256 only.
    if not tuned or N % 256 or K % 256:
        return ladder_items
    # An entry that keeps the ladder's BLOCK_K was measured at BLOCK_K 256 only.
    if "BLOCK_K" not in tuned and c["BLOCK_K"] < 256:
        return ladder_items
    new = dict(c, **tuned)
    if group_m is not None and "GROUP_M" in tuned:
        new["GROUP_M"] = group_m
    key = _cfg_key(new)
    # The launch drops a shuffle it cannot prepare, so check that too:
    # gluon_supported must judge the config that runs.
    if (
        (new["B_SCALE_SHUFFLED"] and not _b_scale_shuffle_fits(new))
        or _pipeline_error(key, dq_a, dq_b, K) is not None
        or _probe_lds_bytes_cached(key, dq_a, dq_b) > LDS_USABLE_BYTES
    ):
        return ladder_items
    return key


def _default_launch_config(
    block_m,
    N,
    K,
    dq_a,
    dq_b,
    small_grid,
    apply_swiglu,
    stream_expert_payload=False,
):
    """Share automatic stage-specific geometry with the capability check."""
    ladder = _get_gluon_config_cached(
        block_m,
        N,
        K,
        dq_a,
        dq_b,
        bool(small_grid),
        bool(stream_expert_payload),
    )
    if not apply_swiglu:
        # Every override below needs apply_swiglu, so none applies to GEMM2.
        group_m = os.environ.get("AITER_TRITON_MOE_GLUON_GROUP_M")
        ladder = _gemm2_launch_config(
            ladder, block_m, N, K, dq_a, dq_b, None if group_m is None else int(group_m)
        )
    c = dict(ladder)
    if (
        not small_grid
        and stream_expert_payload
        and apply_swiglu
        and dq_a == dq_b == DtypeQuant.MXFP8
        and block_m == 64
        and N == 4096
        and K == 7168
    ):
        # The balanced T=256, E=33, top-k=8 launch has one M tile per expert.
        # BN128 doubles its useful CTA count relative to BN256 and removes the
        # eight-XCD launch tail; this is the measured staged-B winner for that
        # exact model shape. Keep the override narrow until adjacent shapes are
        # benchmarked with the same cold-cache protocol.
        c = dict(
            c,
            BLOCK_N=128,
            BLOCK_K=256,
            SLOT_M=32,
            SLOT_N=64,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(2, 2),
            tiles_per_warp=(1, 2),
            NUM_LDS_BUFFER=3,
            K_UNROLL=3,
            VGPR_PREFETCH_K=256,
            WAVES_PER_EU=_env_int("AITER_TRITON_MOE_GLUON_WAVES_PER_EU", 1),
            WAIT_COMMIT_SCHEME=_env_int(
                "AITER_TRITON_MOE_GLUON_WAIT_COMMIT_SCHEME",
                int(WaitCommitScheme.PER_STAGE_WHOLE),
            ),
            A_SCALE_SORTED_SHUFFLED=True,
            B_SCALE_SHUFFLED=True,
            B_PRESHUFFLED=False,
        )
    if (
        not small_grid
        and apply_swiglu
        and dq_a == DtypeQuant.MXFP8
        and dq_b in (DtypeQuant.MXFP4, DtypeQuant.MXFP8)
        and block_m == 32
        and N == 4096
        and K == 7168
    ):
        # T<=64 with E=33, top-k=8 touches every expert, so gemm1 streams the whole
        # weight tensor whatever the token count and the kernel is limited by how much
        # of that stream it keeps in flight -- not by MFMA and not by the HBM roof
        # (3.9 TB/s is about half of peak here).
        #
        # The first round, measured cold against a paired FlyDSL control
        # (bench_out/a8_smallt_20260920, 4 rounds x 100 samples):
        #
        #   a8w4 T16  200.4 -> 132.8 us   a8w4 T64  206.5 -> 138.8 us
        #   a8w8 T16  280.1 -> 232.9 us   a8w8 T64  287.6 -> 242.1 us
        #
        # FlyDSL on the same shapes was 132.1/133.6 and 241.5/242.5. The `.cg` hint on
        # the expert stream is most of it and comes from the block_m <= 32 rule above.
        # SLOT_M/N at the warp granularity rather than the whole tile matter too: one
        # slot per axis is refused by gate/up split outright.
        #
        # The tile itself is set by occupancy. The old BLOCK_N 128 tile held one CTA
        # (four waves) per CU: 126 KB of LDS and 296 registers for a8w8, 155 KB and
        # 336 for a8w4 at BLOCK_K 512. FlyDSL keeps two or more. BLOCK_N 64 at
        # BLOCK_K 256 fits two a8w8 CTAs (76 KB, 188 registers) and three a8w4 ones
        # (51 KB, 140 registers). Shuffled B scales make each scale fetch a contiguous
        # 256 B run that streams under `.cg`; the unshuffled MXFP8 rows lost with it.
        # In-process cold A/B against FlyDSL, each arm the mean of four weight
        # placements, us:
        #
        #                                      a8w8 T=16  T=64   a8w4 T=16  T=64
        #   BLOCK_N 128 (was)                  224.8  233.6      140.4  144.6
        #   + shuffled B scale (.cg on both)   200.2  203.6      127.9  126.2
        #   + BLOCK_N 64, BLOCK_K 256          181.1  183.8       99.2  104.8
        #   FlyDSL                             211.0  217.8      130.9  125.1
        #
        # A deeper ring loses: two a8w4 stages at BLOCK_K 512 (164 us), four at
        # BLOCK_K 256 (396 registers, one CTA). BLOCK_K 128 cannot read shuffled
        # scales. bench_out/gluon_moe_regression/a8small/ab/{var4,w4v1,w4v2}_{16,64}.
        c = dict(
            c,
            BLOCK_N=64,
            BLOCK_K=256,
            SLOT_M=32,
            SLOT_N=32,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(2, 2),
            tiles_per_warp=(1, 1),
            NUM_LDS_BUFFER=3,
            K_UNROLL=3,
            VGPR_PREFETCH_K=256,
            B_SCALE_SHUFFLED=True,
        )
    if (
        apply_swiglu
        and dq_a == dq_b == DtypeQuant.MXFP8
        and block_m == 128
        and N % 256 == 0
        and K % 256 == 0
        and K // 128 >= 10
    ):
        # FP8 K128 has the same payload byte tiles as FP4 K256. The scale
        # preshuffle still packs K256, with consecutive payload stages selecting
        # alternate halves of each dword. Gate/up ordering remains the caller's.
        c = dict(
            c,
            BLOCK_N=256,
            BLOCK_K=128,
            SLOT_M=64,
            SLOT_N=128,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(1, 4),
            tiles_per_warp=(2, 2),
            NUM_LDS_BUFFER=3,
            K_UNROLL=6,
            VGPR_PREFETCH_K=128,
            A_SCALE_SORTED_SHUFFLED=True,
            B_SCALE_SHUFFLED=True,
        )
    if (
        apply_swiglu
        and dq_a == DtypeQuant.MXFP8
        and dq_b == DtypeQuant.MXFP4
        and block_m == 128
        and N % 256 == 0
        and K % 256 == 0
        and (K // dq_pack_divisor(dq_b)) % 16 == 0
    ):
        # The N tile that wins here is 256, and it does not fit while B is staged:
        # a three-deep B ring at BLOCK_N 256 needs 205,824 B against 159,744 B of
        # usable LDS. Holding B in registers removes that ring entirely, which is
        # what pays for the wider tile -- 1022.5 -> 948.0 us at T=4096 on
        # H7168-I2048-E33-k8, against 1013.2 for the same BLOCK_N 128 geometry once
        # it has the weight hint (-6.4%). At T=1024 the two are level (275.7 vs
        # 276.5), so this tile is chosen for the prefill end and costs nothing at
        # the other. bench_out/gluon_larget_20260921.
        #
        # Register B needs preshuffled weights, so the launch path repacks W once and
        # caches it on the tensor. That is a second copy of the weights resident for
        # the lifetime of the model -- the reason this is not the default everywhere.
        # The divisibility terms above are exactly _preshuffled_b's preconditions, so
        # a config that asks for registers is always one it can serve.
        #
        # Splitting the slot to 64x128 also gives this shape two N slots, so unlike
        # every other block_m 64/128 cell it does not fall off the Gluon path for a
        # gate/up-split caller.
        #
        # One CTA of four waves holds the CU (480 registers), so the MFMAs and the
        # memory work only overlap within a wave. The MFMA_8 interleave hint and A's
        # LDS reads placed in the MFMA region buy it back. In-process cold A/B,
        # each arm 3 rounds x 25 steady-state dispatches on four weight placements,
        # us:
        #
        #                           T=1024   T=4096
        #   neither (was)            270.6    976.9
        #   SCHED_MODE MFMA_8        252.6    916.1
        #   + A reads in MFMA        251.7    906.5
        #   FlyDSL                   279.6    982.6
        #
        # GROUP_M 8 on top reaches 883.9 at T=4096 but costs T=1024 (258.9), and the
        # ladder cannot tell them apart. The IGLP modes spill, and per-op commits
        # lose 12%. bench_out/gluon_moe_regression/a8small/ab/w4m2_{1024,4096}.
        c = dict(
            c,
            BLOCK_N=256,
            BLOCK_K=256,
            SLOT_M=64,
            SLOT_N=128,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(1, 4),
            tiles_per_warp=(4, 2),
            NUM_LDS_BUFFER=3,
            K_UNROLL=3,
            VGPR_PREFETCH_K=256,
            WAVES_PER_EU=1,
            WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_STAGE_WHOLE),
            SCHED_MODE=_env_int(
                "AITER_TRITON_MOE_GLUON_SCHED_MODE", int(SchedMode.MFMA_8)
            ),
            DS_READ_A_PAYLOAD_IN_MFMA=True,
            A_SCALE_SORTED_SHUFFLED=True,
            B_SCALE_SHUFFLED=True,
            B_PRESHUFFLED=True,
            ALLOW_B_VIA_LDS=False,
        )
    if (
        apply_swiglu
        and dq_b == DtypeQuant.MXFP4
        # MXFP8 activations at block_m 128 are handled above, with B in registers.
        and (block_m == 64 or (block_m == 128 and dq_a == DtypeQuant.MXFP4))
        and N % 256 == 0
        and K % 256 == 0
    ):
        # The automatic geometry for MXFP4 at these block sizes is the 32x32x64 MFMA
        # with one warp tile per slot, which is both much slower than the tuned
        # 16x16x128 shape and -- because the warp split then spans the whole N tile --
        # impossible to cut into the two N slots a gate/up-split caller needs. Those
        # are the same defect: the slot is not splittable *because* the geometry is
        # wrong. Measured on H7168-I2048-E33-k8, tuned vs the automatic geometry with
        # its slot split so it can run at all:
        #
        #   T=1024   199.4 vs 277.1 us      T=4096   604.7 vs 897.8 us
        #
        # bench_out/gluon_larget_a4bf16_20260921. Splitting the slot on the
        # automatic geometry is nearly free on its own (+0.9% / +0.5%, bit-identical
        # output, bench_out/mini_block_split_20260921) -- it simply is not the tile
        # worth running.
        c = dict(
            c,
            BLOCK_N=256,
            BLOCK_K=256,
            SLOT_M=block_m // 2,
            SLOT_N=128,
            mfma_instr_shape=(16, 16, 128),
            warps_per_cta=(1, 4),
            tiles_per_warp=(2, 2),
            NUM_LDS_BUFFER=3,
            K_UNROLL=3,
            VGPR_PREFETCH_K=256,
            WAVES_PER_EU=1,
            WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_STAGE_WHOLE),
            A_SCALE_SORTED_SHUFFLED=True,
            B_SCALE_SHUFFLED=True,
        )
    if (
        apply_swiglu
        and dq_a == dq_b == DtypeQuant.BF16
        and block_m in (64, 128)
        and N % 256 == 0
        and K % (64 if block_m == 128 else 128) == 0
    ):
        # Same defect for BF16, and worse: the automatic tile is BLOCK_N 64 with the
        # warps split (4, 2), so N is exactly one warp granularity wide and no
        # slot split exists at all. The 16x16x32 MFMA beats the 32x32x16 one at both
        # block sizes. Block_m 64 halves BLOCK_N and doubles BLOCK_K; block_m 128
        # keeps BLOCK_N 256 and gives each of four N warps the whole M extent,
        # eight M tiles by two N tiles. Screened against the 32x32x16 tile that
        # was here (2026-10-17, H7168-I2048-E33-k8, 768 MiB flush per launch):
        # T=256 392 vs 428 us, T=1024 662 vs 706 us, T=4096 2049 vs 2228 us.
        c = dict(
            c,
            BLOCK_N=256 if block_m == 128 else 128,
            BLOCK_K=64 if block_m == 128 else 128,
            SLOT_M=block_m if block_m == 128 else 32,
            SLOT_N=128 if block_m == 128 else 64,
            mfma_instr_shape=(16, 16, 32),
            warps_per_cta=(1, 4),
            tiles_per_warp=(8, 2) if block_m == 128 else (1, 1),
            NUM_LDS_BUFFER=3,
            K_UNROLL=3,
            VGPR_PREFETCH_K=64 if block_m == 128 else 128,
            WAVES_PER_EU=1,
            WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_STAGE_WHOLE),
        )
    if (
        apply_swiglu
        and dq_a == dq_b == DtypeQuant.BF16
        and block_m in (16, 32)
        and not small_grid
        and N % 256 == 0
        and K % 128 == 0
    ):
        # The automatic tile has one N slot (SLOT_N == BLOCK_N == 128), so a
        # gate/up-split caller cannot run it. Two 64-column slots split it at the
        # same speed as the tuned BLOCK_M 32 tile the benchmark pinned, and the
        # automatic .cg weight hint is what that pin lacked: T=16 357 vs 463 us,
        # T=64 366 vs 463 us at block_m 16, against the pin (screened as above).
        c = dict(
            c,
            BLOCK_N=128,
            BLOCK_K=128,
            SLOT_M=block_m,
            SLOT_N=64,
            mfma_instr_shape=(16, 16, 32),
            warps_per_cta=(1, 4),
            tiles_per_warp=(1, 1),
            NUM_LDS_BUFFER=3,
            K_UNROLL=3,
            VGPR_PREFETCH_K=128,
        )
    if dq_b == DtypeQuant.MXFP8 and (
        # Not for GEMM2: its shuffled scales at block_m 32 are 1.3-2.5% slower with
        # the hint, and block_m 16 is flat.
        (c["B_SCALE_SHUFFLED"] and apply_swiglu)
        or (
            c["BLOCK_K"] == 128
            and _probe_b_scale_via_lds(_cfg_key(c), dq_a, dq_b)
        )
    ):
        # The MXFP8 weight-scale exceptions to get_gluon_config's empty hint.
        #
        # Shuffled scales: every cell that gains reads CDNA4_SCALE scales; both
        # cells that lose read unshuffled scales at BLOCK_K 256. Cold a8w8 pins on
        # H7168-I2048-E33-k8, `.cg` vs none:
        #
        #   T=256 -11.1%   T=1024 -2.5%   T=4096 -0.3%   (T=16 +2.2%, T=64 +2.4%)
        #
        # T=16/64 were the unshuffled ones; their block_m 32 tile now shuffles too,
        # and with the hint gains (see that override).
        #
        # Those overrides only shuffle when N and K are multiples of 256, where
        # _shuffled_b_scales always succeeds, so the launch never drops the shuffle
        # and keeps the hint.
        #
        # BLOCK_K 128 through LDS: an unshuffled scale row is 4 bytes, and its LDS
        # copy has every warp fetch the same rows (all but one warp's writes are
        # discarded). Without `.cg` those fetches allocate in L1, and L1 stalls on L2
        # rose 50%. a8w8 pinned to BLOCK_K 128, cold, us:
        #
        #          register   LDS     LDS + .cg
        #   T=16   272.1      277.8   269.4
        #   T=64   304.0      298.6   298.0
        #
        # A register B scale with `.cg` was 2.7% slower at T=16, so it gets no hint.
        # bench_out/gluon_moe_regression/harness/cold_{a8w8nocg,a8w8cg,bk128reg,
        # bk128lds,a8w8ldscg}.
        c = dict(
            c,
            expert_scale_cache_modifier=_env_cache_modifier(
                "EXPERT_SCALE_CACHE_MODIFIER", ".cg", "EXPERT_SCALE_MOD"
            ),
        )
    return c


def _expected_m_tiles_per_expert(M: int, n_expts_tot: int, block_m: int) -> int:
    """Host-only estimate of B reuse, without synchronizing on the device histogram.

    ``M`` is the routed-row count (tokens times top-k). Dividing its mean per expert by
    ``block_m`` estimates how many independent M CTAs will traverse each expert's full B
    matrix. The actual count can differ for skewed routing, but consulting ``expt_hist``
    would introduce a device-to-host synchronization on every launch.
    """
    assert M >= 0 and n_expts_tot > 0 and block_m > 0
    rows_per_all_expert_tiles = n_expts_tot * block_m
    return (M + rows_per_all_expert_tiles - 1) // rows_per_all_expert_tiles


def _stream_mxfp8_gemm1_expert_payload(
    M: int, n_expts_tot: int, block_m: int, dq_a, dq_b, apply_swiglu: bool
) -> bool:
    """Use streaming B only for the measured one-pass MXFP8 gemm1 regime.

    Keep this narrower than a ``block_m <= 64`` rule: the same block size can have
    multiple M tiles per expert at a larger routed batch, where B has cross-CTA reuse.
    Other dtype pairs and gemm2 retain their existing policy until separately measured.
    """
    return (
        apply_swiglu
        and dq_a == DtypeQuant.MXFP8
        and dq_b == DtypeQuant.MXFP8
        and _expected_m_tiles_per_expert(M, n_expts_tot, block_m) <= 1
    )


def _scale_shuffle_supported(cfg, operand):
    """Whether this operand can consume the packed K256 scale byte order."""
    block_k = int(cfg["BLOCK_K"])
    configured_scale_k = cfg.get("SCALE_SLOT_K", 0)
    if "scale_slot_k" in cfg:
        configured_scale_k = cfg["scale_slot_k"]
    if not configured_scale_k:
        configured_scale_k = _env_slot_int("SCALE_SLOT_K", 0)
    # Scale K counts scale columns, one per MX group; a K256 unit is 8 of them.
    scale_block_k = block_k // MX_GROUP_SIZE
    scale_k = int(configured_scale_k) or max(8, scale_block_k)
    if (
        tuple(cfg["mfma_instr_shape"]) != (16, 16, 128)
        or block_k < 128
        or block_k % 128 != 0
        or scale_k < max(8, scale_block_k)
        or scale_k % scale_block_k != 0
        or (scale_k & (scale_k - 1)) != 0
    ):
        return False
    if block_k == 128:
        # Preserve the K128 stage's dword-load eligibility: both non-K bytes
        # belong to this wave while the component pipeline retains the next half.
        return int(cfg["tiles_per_warp"][operand]) >= 2
    return True


def _b_scale_shuffle_fits(cfg):
    """The config half of what ``_shuffled_b_scales`` needs to shuffle B scales."""
    # Stripe-based like the A shuffle: one 256 B run per 32 columns, so a slot that
    # is a whole number of stripes is a contiguous slice of the tile.
    return _scale_shuffle_supported(cfg, 1) and int(cfg["SLOT_N"]) % 32 == 0


_TUNING_KEYS = TuningSpec._fields


def _tuning_args(c: dict) -> tuple:
    """Keep older config dictionaries valid when optional tuning fields are added."""
    c = _canonical_slot_keys(c)
    aliases = {
        "token_mod": "token_cache_modifier",
        "token_scale_mod": "token_scale_cache_modifier",
        "expert_mod": "expert_cache_modifier",
        "expert_scale_mod": "expert_scale_cache_modifier",
        "result_mod": "result_cache_modifier",
        "result_scale_mod": "result_scale_cache_modifier",
    }
    for legacy, canonical in _INVERTED_KNOB_RENAMES.items():
        if legacy in c:
            raise ValueError(
                f"{legacy} was renamed to {canonical} with inverted polarity; "
                f"use {canonical}={not c[legacy]}"
            )
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
    for canonical in _SCALE_SLOT_KEYS:
        alias = canonical.lower()
        value = c.get(canonical, 0)
        if alias in c:
            if value and value != c[alias]:
                raise ValueError(f"{canonical} and {alias} disagree")
            value = c[alias]
        if not value:
            value = _env_slot_int(canonical, 0)
        c = dict(c, **{canonical: value})
    return tuple(
        c[k] if k in c else TuningSpec._field_defaults[k] for k in _TUNING_KEYS
    )
