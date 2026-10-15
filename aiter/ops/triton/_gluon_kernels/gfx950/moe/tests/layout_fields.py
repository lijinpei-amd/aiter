# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dump every ``KernelLayout`` field for ~220 tuning configs, for refactor diffing.

The CPU-only layout half of the gate (T0b in tests/gate.sh): the kernel reads its
geometry and layouts from ``KernelLayout``, so an unchanged dump means a refactor kept
every layout value. Seconds, not the minutes T2b/T2d take.

    python tests/layout_fields.py > before.txt
    ...edit...
    python tests/layout_fields.py > after.txt
    diff -U0 -F '^== ' before.txt after.txt && echo "LAYOUT UNCHANGED"

Per config: its name, the full tuning config, then every annotated ``KernelLayout``
field in declaration order as a type-strict, value-based string, and a sha256 of the
aggregate's JIT mangle. A config whose layout raises prints the exception instead.

Configs: every scripts/gluon_moe_sched_dump.py case plus deltas on it; the host
ladder for E33 top-8 gemm1/gemm2 across dtypes and token counts, raw and after scale
shuffle / B preshuffle, both ``weight_gu_hbm_layout`` values; and the configs the
op_tests layout tests build (config, scale layout/scale slots, registers, LDS,
epilogue), copied as literals. Includes unsound geometries and ones that raise.

Builds through ``compute_kernel_layout(tc)`` when ``_layout`` has it, else
``KernelLayout(tc)``; nothing on stdout names the API, so the two diff cleanly.

    --list           config names only
    --only SUBSTR    configs whose name contains SUBSTR
    --coverage FILE  also write line coverage of _layout.py's construction code

CPU only (no launch, no compile), but Triton needs a visible GPU to import.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import hashlib
import os
import re
import sys
from pathlib import Path

# The host ladder reads AITER_TRITON_MOE_GLUON_* knobs; the dump must not.
for _name in list(os.environ):
    if _name.startswith("AITER_TRITON_MOE_GLUON_"):
        del os.environ[_name]

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO))

import aiter

if not Path(aiter.__file__).resolve().is_relative_to(REPO):
    sys.exit(f"aiter resolves to {aiter.__file__}, not under {REPO}")

from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.moe import _layout as L
from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
    KernelFuncConfig,
    KernelTuningConfig,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DtypeQuant,
    EpilogueMode,
    FuncSpec,
    TuningSpec,
    WaitCommitScheme,
    WarpPipeline,
    WeightGuHbmLayout,
    dq_pack_divisor,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host

if hasattr(L, "compute_kernel_layout"):
    API, build_layout = "compute_kernel_layout", L.compute_kernel_layout
else:
    API, build_layout = "KernelLayout", L.KernelLayout

# The sched dump may still import the old name mid-refactor. Alias it for that one
# import only; everything else above was imported without it.
sys.path.insert(0, str(REPO / "scripts"))
_alias = not hasattr(L, "compute_kernel_layout")
if _alias:
    L.compute_kernel_layout = L.KernelLayout
try:
    import gluon_moe_sched_dump as SD
finally:
    if _alias:
        del L.compute_kernel_layout

FIELDS = tuple(L.KernelLayout.__annotations__)
DQ = DtypeQuant

# -- deterministic value serialisation ----------------------------------------------

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def ser(v):
    """Deterministic, type-strict, value-based string for ``v``.

    Strict on purpose: list vs tuple, bool vs int and enum vs int all print
    differently, because KernelLayout fields are mangled into JIT signatures by repr.
    """
    if isinstance(v, gl.constexpr):
        return f"cx({ser(v.value)})"
    if isinstance(v, enum.Enum):
        return f"{type(v).__name__}.{v.name}={v.value!r}"
    if v is None or type(v) in (bool, int, float, str):
        return repr(v)
    if type(v) is list:
        return "[" + ", ".join(ser(x) for x in v) + "]"
    if isinstance(v, tuple):
        if hasattr(v, "_fields"):
            inner = ", ".join(f"{f}={ser(getattr(v, f))}" for f in v._fields)
            return f"{type(v).__name__}({inner})"
        if type(v) is not tuple:
            raise TypeError(f"unexpected tuple subclass {type(v)}")
        return "(" + ", ".join(ser(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, (set, frozenset)):
        return f"{type(v).__name__}{{{', '.join(sorted(ser(x) for x in v))}}}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{ser(k)}: {ser(x)}" for k, x in v.items()) + "}"
    agg = getattr(type(v), "__aggregate_fields__", None)
    if agg is not None:
        inner = ", ".join(f"{f}={ser(getattr(v, f))}" for f in agg)
        return f"{type(v).__name__}({inner})"
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        # All fields, including repr=False ones the generated repr would hide.
        inner = ", ".join(
            f"{f.name}={ser(getattr(v, f.name))}" for f in dataclasses.fields(v)
        )
        return f"{type(v).__name__}({inner})"
    r = repr(v)
    if _ADDRESS.search(r) or "object at" in r:
        if hasattr(v, "__dict__"):
            inner = ", ".join(f"{k}={ser(x)}" for k, x in sorted(vars(v).items()))
            return f"{type(v).__name__}{{{inner}}}"
        raise TypeError(f"no value-based serialisation for {type(v)}: {r}")
    return f"{type(v).__name__}:{r}"


def ser_field(value):
    """A field is always a gl.constexpr (the aggregate enforces it); print the value."""
    if isinstance(value, gl.constexpr):
        return ser(value.value)
    return "RAW:" + ser(value)


def exc_line(exc):
    msg = _ADDRESS.sub(" at 0x?", str(exc)).replace("\n", "\\n")
    return f"{type(exc).__name__}: {msg}"


# -- config builders copied from op_tests --------------------------------------------
#
# Importing those modules drags in pytest fixtures, torch inputs and each other, and
# a refactor edits them; only these literals matter here.


def _tuning_spec(cfg):
    """test_moe_gemm_gluon_config._tuning_spec; also what _launch_spec builds."""
    return TuningSpec(*(host._hashable(v) for v in host._tuning_args(cfg)))


def _config_fixture():
    """test_moe_gemm_gluon_config.config (a pytest fixture)."""
    cfg = host.get_gluon_config_uncached(128, 4096, 7168, DQ.MXFP4, DQ.MXFP4)
    cfg.update(
        SLOT_M=64,
        SLOT_N=128,
        mfma_instr_shape=(16, 16, 128),
        warps_per_cta=(2, 2),
        tiles_per_warp=(2, 2),
    )
    return cfg


def _config_func_spec(epilogue=EpilogueMode.DEFAULT):
    """test_moe_gemm_gluon_config._func_spec."""
    return FuncSpec(
        int(DQ.MXFP4),
        int(DQ.MXFP4),
        int(DQ.MXFP4),
        int(DQ.MXFP4),
        gl.float32,
        ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        int(DQ.MXFP4),
        True,
        False,
        True,
        False,
        int(WeightGuHbmLayout.SPLIT),
        int(epilogue),
    )


def _scale_config(dtype=DQ.MXFP8, fused=True, **overrides):
    """test_moe_gemm_gluon_scale_layout._config (also scale_slots')."""
    fc = KernelFuncConfig(
        dtype,
        dtype,
        dtype,
        dtype,
        gl.float32,
        ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        int(DQ.BF16),
        False,
        False,
        True,
        False,
        fused,
        int(EpilogueMode.NOP),
    )
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "K_UNROLL": 6,
        "SLOT_M": 64,
        "SLOT_N": 64,
        "NUM_LDS_BUFFER": 3,
        "mfma_instr_shape": (16, 16, 128),
        "warps_per_cta": (2, 2),
        "tiles_per_warp": (2, 2),
        "k_width": None,
        "transposed": True,
        "WAVES_PER_EU": 1,
        "TILE_SCHED": 0,
        "GROUP_M": 1,
        "NUM_XCDS": 8,
        "token_cache_modifier": "",
        "token_scale_cache_modifier": "",
        "expert_cache_modifier": "",
        "expert_scale_cache_modifier": "",
        "result_cache_modifier": "",
        "result_scale_cache_modifier": "",
        "WARP_PIPELINE": 0,
        "VGPR_PREFETCH_K": 128,
        "A_SCALE_SORTED_SHUFFLED": True,
        "B_SCALE_SHUFFLED": True,
    }
    args.update(overrides)
    return KernelTuningConfig(fc, **args)


def _register_config(dtype):
    """test_moe_gemm_gluon_registers._register_config on read_placement._config()."""
    config = TuningSpec(
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        K_UNROLL=3,
        SLOT_M=64,
        SLOT_N=128,
        NUM_LDS_BUFFER=3,
        mfma_instr_shape=(16, 16, 128),
        warps_per_cta=(1, 4),
        tiles_per_warp=(2, 2),
        k_width=None,
        transposed=True,
        WAVES_PER_EU=0,
        TILE_SCHED=0,
        GROUP_M=1,
        NUM_XCDS=1,
        token_cache_modifier="",
        token_scale_cache_modifier="",
        expert_cache_modifier="",
        expert_scale_cache_modifier="",
        result_cache_modifier="",
        result_scale_cache_modifier="",
        WARP_PIPELINE=int(WarpPipeline.NONE),
        VGPR_PREFETCH_K=256,
        WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_STAGE_WHOLE),
        DS_READ_A_PAYLOAD_IN_MFMA=False,
        DS_READ_A_SCALE_IN_MFMA=False,
        DS_READ_B_PAYLOAD_IN_MFMA=False,
        DS_READ_B_SCALE_IN_MFMA=False,
        SCALE_FILL_MID=True,
    )._asdict()
    config.update(A_SCALE_SORTED_SHUFFLED=True, B_SCALE_SHUFFLED=True)
    if dtype == "mxfp8":
        config.update(BLOCK_K=128, VGPR_PREFETCH_K=128, K_UNROLL=6)
    elif dtype == "bf16":
        config.update(
            BLOCK_K=64,
            VGPR_PREFETCH_K=64,
            mfma_instr_shape=(32, 32, 16),
            warps_per_cta=(2, 4),
            tiles_per_warp=(1, 1),
            SCALE_FILL_MID=False,
            A_SCALE_SORTED_SHUFFLED=False,
            B_SCALE_SHUFFLED=False,
        )
    return config


def _epilogue_config(split, num_warps):
    """test_moe_gemm_gluon_epilogue._epilogue_config, unwrapped from gl.constexpr."""
    func = FuncSpec(
        int(DQ.MXFP4),
        int(DQ.MXFP4),
        int(DQ.MXFP4),
        int(DQ.MXFP4),
        gl.float32,
        ActivationSpec(int(ActKind.SILU), 1.702, 7.0, True),
        int(DQ.MXFP4),
        False,
        False,
        False,
        False,
        int(WeightGuHbmLayout.SPLIT if split else WeightGuHbmLayout.INTERLEAVED),
    )
    tuning = TuningSpec(
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        K_UNROLL=1,
        SLOT_M=64,
        SLOT_N=128,
        NUM_LDS_BUFFER=2,
        mfma_instr_shape=(16, 16, 128),
        warps_per_cta=(num_warps // 4, 4),
        tiles_per_warp=(2, 2),
        k_width=8,
        transposed=True,
        WAVES_PER_EU=0,
        TILE_SCHED=0,
        GROUP_M=1,
        NUM_XCDS=1,
        token_cache_modifier="",
        token_scale_cache_modifier="",
        expert_cache_modifier="",
        expert_scale_cache_modifier="",
        result_cache_modifier="",
        result_scale_cache_modifier="",
        WARP_PIPELINE=0,
        VGPR_PREFETCH_K=256,
        ACT_FAST_RCP=True,
    )
    return KernelTuningConfig(KernelFuncConfig(*func), *tuning)


# -- config cases --------------------------------------------------------------------
#
# Every case is (name, thunk); the thunk returns a KernelTuningConfig.

SILU = ActivationSpec(int(ActKind.SILU), 1.0, None, False)
E_TOT, TOPK = 33, 8
TOKENS = (16, 64, 128, 256, 1024, 4096)
DTYPES = (
    ("a4w4", DQ.MXFP4, DQ.MXFP4),
    ("a8w4", DQ.MXFP8, DQ.MXFP4),
    ("a8w8", DQ.MXFP8, DQ.MXFP8),
    ("a16w16", DQ.BF16, DQ.BF16),
    ("fp8", DQ.FP8_E4M3, DQ.FP8_E4M3),
)


def tc_from(cfg, func_spec):
    """Exactly what moe_op_gemm_gluon._launch_spec builds from a config dict."""
    return KernelTuningConfig(KernelFuncConfig(*func_spec), *_tuning_spec(cfg))


def func_spec(dq_a, dq_b, act, out_quant, split, *, gather, gammas, epilogue=0):
    return FuncSpec(
        int(dq_a),
        int(dq_b),
        int(dq_a),
        int(dq_b),
        gl.float32,
        act,
        None if out_quant is None else int(out_quant),
        False,
        gammas,
        gather,
        False,
        int(WeightGuHbmLayout.SPLIT if split else WeightGuHbmLayout.INTERLEAVED),
        int(epilogue),
    )


def _router_block_m(tokens):
    # aiter/ops/triton/moe/moe_routing/routing.py: routing()
    next_pow2 = 1 << (max(1, tokens * TOPK // E_TOT) - 1).bit_length()
    return max(16, min(next_pow2, 128))


def _n_blocks(n_rows, block_m):
    # moe_routing/routing.py: RoutingData.n_blocks
    if n_rows <= E_TOT:
        return n_rows
    return -(-max(n_rows - E_TOT + 1, 0) // block_m) + E_TOT - 1


def _effective_cfg(cfg, dq_a, dq_b, N, K, swiglu):
    """The shuffle/preshuffle flags moe_gemm_gluon actually launches with.

    Mirrors the pure preconditions of _sorted_shuffle_a_scales, _shuffled_b_scales and
    _preshuffled_b (gemm1 gathers, so the A-scale pre-pass is reachable there).
    """
    c = dict(cfg)
    scaled = (DQ.MXFP4, DQ.MXFP8)
    a_sh = (
        bool(c.get("A_SCALE_SORTED_SHUFFLED", False))
        and swiglu
        and dq_a in scaled
        and host._scale_shuffle_supported(c, 0)
        and int(c["BLOCK_M"]) % 32 == 0
        and int(c["SLOT_M"]) % 32 == 0
        and K % 256 == 0
    )
    b_sh = (
        bool(c.get("B_SCALE_SHUFFLED", False))
        and dq_b in scaled
        and host._scale_shuffle_supported(c, 1)
        and int(c["SLOT_N"]) % 32 == 0
        and N % 32 == 0
        and (K // 32) % 8 == 0
    )
    c.update(A_SCALE_SORTED_SHUFFLED=a_sh, B_SCALE_SHUFFLED=b_sh)
    if not c.get("ALLOW_B_VIA_LDS", True):
        kb = K // dq_pack_divisor(int(dq_b))
        if int(c["BLOCK_K"]) % 32 == 0 and N % 16 == 0 and K % 32 == 0 and kb % 16 == 0:
            c["B_PRESHUFFLED"] = True
    return c


def _reraise(exc):
    raise exc


def production_cases():
    """The host ladder for this model: E33 top-8, gemm1 H7168 -> 2*I4096, gemm2 back.

    For every dtype and token count: the raw ladder config and, where it differs, the
    effective one after scale-shuffle / B-preshuffle preparation. gemm1 in both
    weight_gu_hbm_layout values, plus the fused MXFP4-output variant for a4w4.
    """
    out = []
    shapes = (("g1", 4096, 7168, True), ("g2", 7168, 2048, False))
    for gemm, N, K, swiglu in shapes:
        for label, dq_a, dq_b in DTYPES:
            for T in TOKENS:
                M = T * TOPK
                bm = _router_block_m(T)
                small = _n_blocks(M, bm) * max(1, N // 128) < 512
                stream = host._stream_mxfp8_gemm1_expert_payload(
                    M, E_TOT, bm, dq_a, dq_b, swiglu
                )
                try:
                    raw = host._default_launch_config(
                        bm, N, K, dq_a, dq_b, small, swiglu, stream
                    )
                    eff = _effective_cfg(raw, dq_a, dq_b, N, K, swiglu)
                except Exception as exc:  # noqa: BLE001 - the failure is the datum
                    name = f"prod/{gemm}-{label}-t{T}-bm{bm}/ladder"
                    out.append((name, lambda e=exc: _reraise(e)))
                    continue
                variants = [("raw", raw)] + ([("eff", eff)] if eff != raw else [])
                funcs = []
                if swiglu:
                    funcs.append(("split", SILU, None, True))
                    funcs.append(("ilv", SILU, None, False))
                    if label == "a4w4":
                        funcs.append(("split-mxfp4out", SILU, DQ.MXFP4, True))
                else:
                    funcs.append(("noact", None, None, False))
                for fname, act, oq, split in funcs:
                    fs = func_spec(
                        dq_a, dq_b, act, oq, split, gather=swiglu, gammas=not swiglu
                    )
                    for vname, cfg in variants:
                        name = f"prod/{gemm}-{label}-t{T}-bm{bm}-{fname}/{vname}"
                        out.append((name, lambda c=cfg, f=fs: tc_from(c, f)))
    # The other shape the scale-prep tests launch (test_moe_gemm_gluon_scale_prep._spec):
    # block_m 128, N 512, K 2560, MXFP8 gemm1 and its untouched-ladder variants.
    for name, dq_a, dq_b, swiglu in (
        ("mxfp8", DQ.MXFP8, DQ.MXFP8, True),
        ("mxfp8-noact", DQ.MXFP8, DQ.MXFP8, False),
        ("a8w4", DQ.MXFP8, DQ.MXFP4, True),
    ):
        cfg = host._default_launch_config(
            128, 512, 2560, dq_a, dq_b, False, swiglu, False
        )
        fs = func_spec(
            dq_a, dq_b, SILU if swiglu else None, None, False, gather=True, gammas=False
        )
        out.append((f"prod/scale_prep-{name}", lambda c=cfg, f=fs: tc_from(c, f)))
    return out


def sched_cases():
    """Every case scripts/gluon_moe_sched_dump.py dumps, built the way it builds them."""
    out = []
    for name, cfg, dq_a, dq_b, _K in SD.cases():
        out.append(
            (
                f"sched/{name}",
                lambda c=cfg, a=dq_a, b=dq_b: host._probe_tuning_config(c, a, b),
            )
        )
    # Deltas on the same benchmarked base (SD._case), for branches the list misses.
    extra = (
        # AITER_TRITON_MOE_GLUON_B_PRESHUFFLED=1 keeps B in LDS on a preshuffled
        # tensor: the 16-row byte-unit LDS order.
        ("bpre-lds", DQ.MXFP4, DQ.MXFP4, {"B_PRESHUFFLED": True}),
        ("bpre-lds-a8w8", DQ.MXFP8, DQ.MXFP8, {"B_PRESHUFFLED": True}),
        # An explicit k_width override (the default value, through the other branch).
        ("kwidth16", DQ.MXFP4, DQ.MXFP4, {"k_width": 16}),
        # transposed=False is asserted against by the result layout: exception path.
        ("transposed0", DQ.MXFP4, DQ.MXFP4, {"transposed": False}),
        # Raw (unshuffled) scales on the base geometry.
        (
            "raw-scales",
            DQ.MXFP4,
            DQ.MXFP4,
            {"A_SCALE_SORTED_SHUFFLED": False, "B_SCALE_SHUFFLED": False},
        ),
        # Shuffled B scale in registers on a wave split that keeps the non-K +16 step
        # across waves: the unpacked, base-swapped register load.
        (
            "bscale-reg-unpacked",
            DQ.MXFP4,
            DQ.MXFP4,
            {
                "ALLOW_B_SCALE_VIA_LDS": False,
                "warps_per_cta": (1, 4),
                "tiles_per_warp": (2, 1),
                "SLOT_N": 64,
                "BLOCK_N": 128,
            },
        ),
        (
            "ascale-reg-unpacked",
            DQ.MXFP4,
            DQ.MXFP4,
            {
                "ALLOW_A_SCALE_VIA_LDS": False,
                "warps_per_cta": (4, 1),
                "tiles_per_warp": (1, 2),
                "SLOT_M": 64,
            },
        ),
        # A scale slot narrower than the payload slot, staged in registers instead of
        # LDS: no scale tiles, but the LDS footprint still exists.
        (
            "ascale-narrow-reg",
            DQ.MXFP4,
            DQ.MXFP4,
            {"SCALE_SLOT_M": 32, "ALLOW_A_SCALE_VIA_LDS": False},
        ),
        ("ascale-narrow-lds", DQ.MXFP4, DQ.MXFP4, {"SCALE_SLOT_M": 32}),
        ("bscale-narrow-lds", DQ.MXFP4, DQ.MXFP4, {"SCALE_SLOT_N": 32}),
        # bf16 x MXFP4 / fp8 x MXFP4 operand mixes (gluon_supported refuses the
        # former; the layout is still built and must keep its value or exception).
        (
            "a16w4",
            DQ.BF16,
            DQ.MXFP4,
            {"A_SCALE_SORTED_SHUFFLED": False, "B_SCALE_SHUFFLED": False},
        ),
        ("fp8w4", DQ.FP8_E4M3, DQ.MXFP4, {}),
        # Unsound lattice: BLOCK_K not a multiple of the MFMA K.
        ("bk64-unsound", DQ.MXFP4, DQ.MXFP4, {"BLOCK_K": 64, "VGPR_PREFETCH_K": 64}),
        # Slot that does not divide the block: unsound, scalar geometry only.
        ("slot48-unsound", DQ.MXFP4, DQ.MXFP4, {"SLOT_M": 48}),
        # Slot that is not a whole number of warp tiles: unsound.
        (
            "slot32-tiles4-unsound",
            DQ.MXFP4,
            DQ.MXFP4,
            {"SLOT_M": 32, "tiles_per_warp": (4, 2)},
        ),
        # Zero-size slot: the scalar geometry itself raises.
        ("slot0-raises", DQ.MXFP4, DQ.MXFP4, {"SLOT_M": 0}),
    )
    for name, dq_a, dq_b, over in extra:
        _, cfg, a, b, _K = SD._case(name, dq_a, dq_b, **over)
        out.append(
            (
                f"sched+/{name}",
                lambda c=cfg, a=a, b=b: host._probe_tuning_config(c, a, b),
            )
        )
    return out


def _bf16_func():
    return _config_func_spec()._replace(
        token_dtype_quant=int(DQ.BF16),
        expert_dtype_quant=int(DQ.BF16),
        token_online_quant=int(DQ.BF16),
        expert_online_quant=int(DQ.BF16),
    )


def config_test_cases():
    """op_tests/triton_tests/moe/test_moe_gemm_gluon_config.py."""
    out = []

    def add(name, updates, func=None):
        def thunk():
            cfg = _config_fixture()
            cfg.update(updates)
            return KernelTuningConfig(
                KernelFuncConfig(*(func or _config_func_spec())), *_tuning_spec(cfg)
            )

        out.append((f"test/config/{name}", thunk))

    add("fixture", {})
    # test_register_storage_options_are_independent
    for mask in range(8):
        add(
            f"storage_mask{mask}",
            {
                "ALLOW_B_VIA_LDS": not (mask & 1),
                "ALLOW_B_SCALE_VIA_LDS": not (mask & 2),
                "ALLOW_A_SCALE_VIA_LDS": not (mask & 4),
                "B_PRESHUFFLED": bool(mask & 1),
                "A_NUM_BUFFER": 2,
                "B_NUM_BUFFER": 3,
                "A_SCALE_NUM_BUFFER": 2,
                "B_SCALE_NUM_BUFFER": 3,
                "K_UNROLL": 1,
            },
        )
    # test_small_payload_tiles_use_register_rings
    add(
        "small_payload_registers",
        {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "SLOT_M": 32,
            "SLOT_N": 32,
            "VGPR_PREFETCH_K": 32,
            "mfma_instr_shape": (16, 16, 16),
            "warps_per_cta": (2, 2),
            "tiles_per_warp": (1, 1),
            "A_NUM_BUFFER": 2,
            "B_NUM_BUFFER": 3,
            "K_UNROLL": 1,
            "WARP_PIPELINE": 0,
        },
        _bf16_func()._replace(output_quant=None),
    )
    # test_gate_up_small_warp_extent_only_requires_mx_group_for_quantized_output
    for oq in (None, int(DQ.MXFP4)):
        add(
            f"gate_up_small_warp_oq{oq}",
            {
                "BLOCK_N": 128,
                "SLOT_N": 64,
                "warps_per_cta": (1, 4),
                "tiles_per_warp": (2, 1),
                "B_PRESHUFFLED": True,
                "ALLOW_B_VIA_LDS": False,
                "A_SCALE_SORTED_SHUFFLED": False,
                "B_SCALE_SHUFFLED": False,
            },
            _config_func_spec()._replace(output_quant=oq),
        )
    # test_independent_buffer_waits_match_copy_history (geometry variants)
    counts = dict(zip(host._COMPONENT_BUFFER_KEYS, (2, 4, 3, 2)))
    add(
        "middle_shared_a",
        dict(
            counts,
            SCALE_FILL_MID=True,
            A_SCALE_SORTED_SHUFFLED=True,
            SCALE_SLOT_M=128,
        ),
    )
    add("unequal_tiles", dict(counts, SLOT_N=64, SCALE_FILL_MID=True))
    # test_independent_buffer_schedule_without_scales
    for reg_b in (False, True):
        add(
            f"no_scales_register_b{int(reg_b)}",
            dict(
                zip(host._COMPONENT_BUFFER_KEYS, (3, 2, 1, 1)),
                ALLOW_B_VIA_LDS=not reg_b,
                B_PRESHUFFLED=reg_b,
            ),
            _bf16_func(),
        )
    # test_epilogue_modes_preserve_output_geometry
    add("epilogue_nop", {}, _config_func_spec(EpilogueMode.NOP))
    # test_partial_register_prefetch_is_rejected
    for prefetch in (0, 128):
        add(f"vgpr_prefetch_k{prefetch}", {"VGPR_PREFETCH_K": prefetch})
    return out


def scale_layout_test_cases():
    """test_moe_gemm_gluon_scale_layout.py and test_moe_gemm_gluon_scale_slots.py.

    Both build through _scale_config (MXFP8, BLOCK_K 128, both scales shuffled,
    gate/up split by default).
    """
    out = []

    def add(name, *args, **kw):
        out.append((f"test/scale/{name}", lambda: _scale_config(*args, **kw)))

    add("default")
    add("interleaved", fused=False)
    add("mxfp4", DQ.MXFP4)
    # test_packed_words_and_selectors_match_producer
    for bk, dt in ((128, DQ.MXFP8), (256, DQ.MXFP4)):
        for warps, slot_m, slot_n in (
            ((2, 4), 64, 128),
            ((1, 4), 64, 128),
            ((4, 1), 128, 64),
            ((2, 2), 64, 64),
        ):
            add(
                f"packed-bk{bk}-w{warps[0]}x{warps[1]}-m{slot_m}-n{slot_n}",
                dt,
                BLOCK_K=bk,
                VGPR_PREFETCH_K=bk,
                BLOCK_M=slot_m * 2,
                BLOCK_N=slot_n * 2,
                SLOT_M=slot_m,
                SLOT_N=slot_n,
                warps_per_cta=warps,
            )
    # test_wide_packed_scale_load_layout_matches_producer
    for sk in (256, 512, 1024):
        add(
            f"wide-sk{sk}",
            BLOCK_N=256,
            SLOT_N=128,
            warps_per_cta=(1, 4),
            SCALE_SLOT_M=128,
            SCALE_SLOT_N=256,
            SCALE_SLOT_K=sk,
        )
    # test_invalid_packed_k128_configs_are_rejected
    add("bk64-unsound", BLOCK_K=64, VGPR_PREFETCH_K=64)
    # test_pipeline_phases_and_packed_scale_addresses (registers=True, depths 4/2/2/3)
    add(
        "registers",
        K_UNROLL=3,
        A_NUM_BUFFER=4,
        B_NUM_BUFFER=2,
        A_SCALE_NUM_BUFFER=2,
        B_SCALE_NUM_BUFFER=3,
        ALLOW_B_VIA_LDS=False,
        B_PRESHUFFLED=True,
        ALLOW_A_SCALE_VIA_LDS=False,
        ALLOW_B_SCALE_VIA_LDS=False,
    )
    # test_scale_slots_only_change_the_shuffled_operand
    for sa, sb in ((False, False), (True, False), (False, True), (True, True)):
        add(
            f"scale-slot-shuffled{int(sa)}{int(sb)}",
            A_SCALE_SORTED_SHUFFLED=sa,
            B_SCALE_SHUFFLED=sb,
            SCALE_SLOT_M=128,
            SCALE_SLOT_N=128,
            SCALE_SLOT_K=512,
        )
    # test_shuffled_scale_k_defaults_to_at_least_k256
    for bk in (128, 256, 512):
        add(f"default-sk-bk{bk}", BLOCK_K=bk, VGPR_PREFETCH_K=bk)
    # test_invalid_unused_scale_slots_are_ignored
    add(
        "unused-scale-slots",
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        SCALE_SLOT_M=-1,
        SCALE_SLOT_N=3,
        SCALE_SLOT_K=1,
    )
    # test_invalid_shuffled_scale_slots_are_rejected (32 is narrower than the slot)
    for field, value in (
        ("SCALE_SLOT_M", 32),
        ("SCALE_SLOT_M", 256),
        ("SCALE_SLOT_N", 32),
        ("SCALE_SLOT_N", 256),
        ("SCALE_SLOT_K", 128),
        ("SCALE_SLOT_K", 384),
    ):
        add(f"invalid-{field}-{value}", **{field: value})
    # test_shuffled_scale_k_cannot_be_narrower_than_payload_k
    add("sk-narrower-than-bk", BLOCK_K=512, SCALE_SLOT_K=256, VGPR_PREFETCH_K=512)
    return out


def register_test_cases():
    """test_moe_gemm_gluon_registers.py, test_moe_gemm_gluon_lds.py (_register_config)."""
    dqs = {"mxfp4": DQ.MXFP4, "mxfp8": DQ.MXFP8, "bf16": DQ.BF16}
    out = []

    def add(name, dtype, updates):
        def thunk():
            cfg = _register_config(dtype)
            cfg.update(updates)
            return host._probe_tuning_config(cfg, dqs[dtype], dqs[dtype])

        out.append((f"test/registers/{name}", thunk))

    for dtype in dqs:
        add(f"base-{dtype}", dtype, {})
    small = {
        "VGPR_PREFETCH_K": 32,
        "BLOCK_K": 32,
        "K_UNROLL": 3,
        "tiles_per_warp": (1, 1),
    }
    small.update(mfma_instr_shape=(16, 16, 16))
    # test_small_a_payload_uses_register_ring
    add(
        "small-a",
        "bf16",
        dict(small, SLOT_M=16, SLOT_N=128, warps_per_cta=(1, 4)),
    )
    # test_small_b_payload_uses_register_ring_without_preshuffle
    add(
        "small-b",
        "bf16",
        dict(
            small,
            BLOCK_N=32,
            SLOT_M=64,
            SLOT_N=16,
            warps_per_cta=(4, 1),
            ALLOW_B_VIA_LDS=True,
            B_PRESHUFFLED=False,
        ),
    )
    # test_small_a_and_b_payloads_share_register_pipeline
    add(
        "small-ab",
        "bf16",
        dict(
            small,
            BLOCK_N=64,
            SLOT_M=32,
            SLOT_N=32,
            warps_per_cta=(2, 2),
            ALLOW_B_VIA_LDS=True,
            B_PRESHUFFLED=False,
        ),
    )
    # test_k128_register_scale_phase
    add(
        "k128-scale-phase",
        "mxfp8",
        {
            "BLOCK_K": 128,
            "VGPR_PREFETCH_K": 128,
            "ALLOW_B_VIA_LDS": False,
            "ALLOW_A_SCALE_VIA_LDS": False,
            "ALLOW_B_SCALE_VIA_LDS": False,
            "A_NUM_BUFFER": 3,
            "B_NUM_BUFFER": 2,
            "A_SCALE_NUM_BUFFER": 4,
            "B_SCALE_NUM_BUFFER": 3,
        },
    )
    # test_scale_and_payload_k_load_units_are_independent
    for dtype in ("mxfp4", "mxfp8"):
        for bk, sk, sh in (
            (256, 256, (False, False)),
            (256, 512, (True, False)),
            (512, 512, (True, True)),
            (512, 1024, (False, True)),
        ):
            add(
                f"kunits-{dtype}-bk{bk}-sk{sk}-sh{int(sh[0])}{int(sh[1])}",
                dtype,
                {
                    "BLOCK_K": bk,
                    "SCALE_SLOT_K": sk,
                    "SCALE_SLOT_M": 128,
                    "SCALE_SLOT_N": 256,
                    "VGPR_PREFETCH_K": bk,
                    "NUM_LDS_BUFFER": 2,
                    "K_UNROLL": 2,
                    "ALLOW_B_VIA_LDS": False,
                    "A_SCALE_SORTED_SHUFFLED": sh[0],
                    "B_SCALE_SHUFFLED": sh[1],
                },
            )
    # test_minimum_k_and_every_reachable_unroll_remainder
    for dtype, schedule in (
        ("mxfp4", "mixed_registers"),
        ("mxfp8", "packed_registers"),
        ("bf16", "mixed_lds"),
    ):
        packed = schedule == "packed_registers"
        upd = {
            "NUM_LDS_BUFFER": 2,
            "K_UNROLL": 3,
            "warps_per_cta": (1, 4),
            "A_SCALE_SORTED_SHUFFLED": packed,
            "B_SCALE_SHUFFLED": packed,
            "A_NUM_BUFFER": 2,
            "B_NUM_BUFFER": 3,
            "A_SCALE_NUM_BUFFER": 4,
            "B_SCALE_NUM_BUFFER": 2,
        }
        if "registers" in schedule:
            upd.update(
                ALLOW_B_VIA_LDS=False,
                ALLOW_A_SCALE_VIA_LDS=False,
                ALLOW_B_SCALE_VIA_LDS=False,
            )
        if schedule == "mixed_lds":
            upd["WAIT_COMMIT_SCHEME"] = int(WaitCommitScheme.PER_OP)
        add(f"minimum-k-{dtype}-{schedule}", dtype, upd)
    # test_moe_gemm_gluon_lds.test_register_load_does_not_stage_through_lds
    for bk in (128, 256):
        add(
            f"lds-all-registers-bk{bk}",
            "mxfp4",
            {
                "BLOCK_K": bk,
                "VGPR_PREFETCH_K": bk,
                "ALLOW_B_VIA_LDS": False,
                "B_PRESHUFFLED": True,
                "ALLOW_A_SCALE_VIA_LDS": False,
                "ALLOW_B_SCALE_VIA_LDS": False,
                "A_SCALE_SORTED_SHUFFLED": False,
                "B_SCALE_SHUFFLED": False,
            },
        )
    return out


def epilogue_test_cases():
    """test_moe_gemm_gluon_epilogue._epilogue_config: k_width=8 override, 4/8 warps."""
    out = []
    for split in (False, True):
        for warps in (4, 8):
            out.append(
                (
                    f"test/epilogue/split{int(split)}-w{warps}",
                    lambda s=split, w=warps: _epilogue_config(s, w),
                )
            )
    return out


def all_cases():
    cases = (
        sched_cases()
        + production_cases()
        + config_test_cases()
        + scale_layout_test_cases()
        + register_test_cases()
        + epilogue_test_cases()
    )
    names = [n for n, _ in cases]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"duplicate config names: {dupes}"
    return cases


# -- coverage (optional) -------------------------------------------------------------


class _LineTracer:
    """Records executed lines of _layout.py while layouts are built."""

    def __init__(self, filename):
        self.filename = filename
        self.lines = set()

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines.add(frame.f_lineno)
        return self._local

    def __call__(self, frame, event, arg):
        if frame.f_code.co_filename == self.filename:
            self.lines.add(frame.f_lineno)
            return self._local
        return None


def write_coverage(path, executed):
    """Per construction function: FULL, or the statements no config reached.

    Construction code is every host-side module-level function plus
    ``KernelLayout.__init__``; the accessors only read fields back.
    """
    src = Path(L.__file__).read_text().splitlines()
    tree = ast.parse("\n".join(src))
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "KernelLayout":
            funcs += [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ]
    with open(path, "w") as f:
        f.write(f"# line coverage of {L.__file__}\n")
        for fn in funcs:
            decos = [ast.unparse(d) for d in fn.decorator_list]
            if any("jit" in d and "constexpr" not in d for d in decos):
                continue  # device code: not executed on the host
            body = {
                stmt.lineno
                for stmt in ast.walk(fn)
                if isinstance(stmt, ast.stmt)
                and stmt is not fn
                and not (
                    isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                )
            }
            missed = sorted(body - executed)
            status = f"MISSED {len(missed)}/{len(body)}" if missed else "FULL"
            f.write(f"{fn.name}: {status}\n")
            f.writelines(f"    {ln}: {src[ln - 1].strip()}\n" for ln in missed)


# -- main ----------------------------------------------------------------------------


def dump(cases, out, tracer=None):
    out(f"# KernelLayout fields ({len(FIELDS)}): {', '.join(FIELDS)}")
    out(f"# configs: {len(cases)}")
    signatures = set()
    n_exc = 0
    for name, thunk in cases:
        out(f"== {name}")
        try:
            tc = thunk()
        except Exception as exc:  # noqa: BLE001 - the failure is the datum
            out(f"  TC-EXC {exc_line(exc)}")
            continue
        sig = ser(tc)
        signatures.add(sig)
        out(f"  cfg {sig}")
        if tracer is not None:
            sys.settrace(tracer)
        try:
            layout = build_layout(tc)
        except Exception as exc:  # noqa: BLE001 - the failure is the datum
            n_exc += 1
            out(f"  EXC {exc_line(exc)}")
            continue
        finally:
            if tracer is not None:
                sys.settrace(None)
        for field in FIELDS:
            out(f"  {field} = {ser_field(getattr(layout, field))}")
        try:
            digest = hashlib.sha256(layout.type.mangle().encode()).hexdigest()
        except Exception as exc:  # noqa: BLE001 - the failure is the datum
            digest = f"EXC {exc_line(exc)}"
        out(f"  #mangle_sha256 = {digest}")
    return signatures, n_exc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="config names only")
    ap.add_argument("--only", help="substring filter on config name")
    ap.add_argument("--coverage", help="write _layout.py line coverage to this file")
    args = ap.parse_args(argv)
    cases = all_cases()
    if args.only:
        cases = [c for c in cases if args.only in c[0]]
    if args.list:
        print("\n".join(n for n, _ in cases))
        return 0
    lines = []
    tracer = _LineTracer(L.__file__) if args.coverage else None
    signatures, n_exc = dump(cases, lines.append, tracer)
    sys.stdout.write("\n".join(lines) + "\n")
    print(
        f"[layout_fields] api={API} layout={L.__file__} configs={len(cases)} "
        f"distinct={len(signatures)} build_exceptions={n_exc} fields={len(FIELDS)}",
        file=sys.stderr,
    )
    if tracer is not None:
        write_coverage(args.coverage, tracer.lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
