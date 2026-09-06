# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950 / CDNA4) MoE grouped GEMM kernels.

Two entry points over **one** ``@gluon.jit`` body:

* :func:`_moe_gluon_gemm1` -- gemm1 + fused activation. Gathers tokens according to the
  routing metadata and writes its output contiguously (dense, expert-sorted
  ``[n_gates, I]``), optionally already quantised to gemm2's operand-A format.
* :func:`_moe_gluon_gemm2` -- gemm2 + router-combine-weight multiply. Reads
  contiguously, multiplies by ``gammas``, writes contiguously.

They differ only in the ``KernelFuncConfig`` instantiated in-kernel. The reduce/combine
step stays in the existing ``moe/reduce.py::reduce_grouped``.

Operand dtypes are independent -- everything derived from one (LDS element type, tile
shape, copy width, ``k_width``, the MFMA shape) is per operand, because the mixed pairs
are real:

===============  =============  ==================================================
 A                B              dot
===============  =============  ==================================================
 MXFP4            MXFP4          mfma_scaled e2m1 x e2m1        (a4w4)
 MXFP8            MXFP8          mfma_scaled e4m3 x e4m3        (a8w8)
 MXFP8 / FP8      MXFP4          mfma_scaled e4m3 x e2m1        (a8w4)
 BF16             BF16           mfma
 BF16             MXFP4          scaled_upcast + mfma -- refused, see gluon_supported
===============  =============  ==================================================

FP8 without a scale still goes through ``mfma_scaled`` with ``a_scale=None``: the
backend folds the synthesized unit scales back into ``V_MFMA_*_F8F6F4`` and only that
path reaches the double-rate K=64/128 pipes. Its per-tensor scale is applied to the raw
accumulator in the epilogue instead.

Memory layout contract (the wrapper hard-asserts it), with ``pack`` = 2 for MXFP4 and
1 otherwise:

* activations  ``(M, K/pack)`` row-major, ``stride(-1) == 1``
* weights      ``(E, K/pack, N)`` with ``stride(-2) == 1``  (K contiguous)
* scales       ``(E, K/32, N)`` strided view, K contiguous, strides carried explicitly
* stage 1: ``N = 2*I``, ``K = H``; stage 2: ``N = H``, ``K = I``
* stage 1 gate/up arrive pre-fused, in one of two packings along N, selected by
  ``KernelFuncConfig.gate_up_split``:

  - ``False`` (default) -- column-interleaved, even = gate, odd = up
  - ``True``  -- whole halves, gate in ``[0, I)`` and up in ``[I, 2I)``. A mini-N block
    is then one entire operand side, so the activation pairs two accumulator *tiles*
    with identical layouts instead of two registers within a lane's quad, and a warp's
    32 emitted channels are exactly one MX group. See ``_n_start``. The caller permutes
    weights, weight scales and bias with ``activations.py::gate_up_split_perm``; the
    emitted output is bit-identical either way.

``N`` and ``K`` are compile-time: it is what makes the pipeline drain, the mini-tile
decomposition and the rotating buffer index fully static. The in-scope models present
only a handful of distinct ``(N, K)`` pairs, so this does not flood the Triton cache the
way a constexpr batch size would.

There is no split-K (``moe_op_gemm_a4w4.py`` already hardcodes ``split_k == 1``), and
fusing the activation and the output quant into the epilogue forecloses turning it back
on -- partial sums cannot be activated, and the MX group amax needs the final value.
The output buffer still carries the leading ``(1, M, N)`` axis that ``reduce_grouped``
indexes.
"""

from typing import NamedTuple

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._triton_kernels.moe.activations import (
    _swiglu,
    _swiglu_combine,
    _swiglu_gate,
    _swiglu_pair,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import WARP_SIZE_CE as WARP_SIZE
from ._lang import field_at as _at
from ._lang import optional as _opt
from ._lang import pick_warp_pipeline_stage as pick_stage
from ._lang import require_constexpr
from ._lang import unwrap as _v
from ._lang import unwrap_attr as _cv
from ._lds import LDSManager
from ._quant import mxfp4_quant_gluon
from ._types import (
    DotKind,
    DtypeQuant,
    QuantExpertTensor,
    QuantTokenTensor,
    ResultTensor,
    RoutingMeta,
    TileSched,
    WaitCommitScheme,
)

_SG_MFMA: gl.constexpr = gl.constexpr(8)
_SG_DS_READ: gl.constexpr = gl.constexpr(256)
_SG_VMEM: gl.constexpr = gl.constexpr(16)


@gluon.jit
def _sched_hint(MODE: gl.constexpr):
    """Scheduling recipe for one K stage, emitted at the top of its region."""
    if require_constexpr(MODE == 1):
        gl.amd.cdna4.iglp_opt(0)
    elif require_constexpr(MODE == 2):
        gl.amd.cdna4.iglp_opt(1)
    elif require_constexpr(MODE == 3):
        for _ in gl.static_range(4):
            gl.amd.cdna4.sched_group_barrier(_SG_MFMA, 16, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_DS_READ, 6, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_VMEM, 4, 0)
    elif require_constexpr(MODE == 4):
        for _ in gl.static_range(8):
            gl.amd.cdna4.sched_group_barrier(_SG_MFMA, 8, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_DS_READ, 3, 0)
            gl.amd.cdna4.sched_group_barrier(_SG_VMEM, 2, 0)


_DQ_MXFP4: gl.constexpr = gl.constexpr(int(DtypeQuant.MXFP4))
_TS_XCD_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.XCD_GROUP_M))
_TS_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.GROUP_M))
_DK_MFMA: gl.constexpr = gl.constexpr(int(DotKind.MFMA))
_DK_MFMA_SCALED: gl.constexpr = gl.constexpr(int(DotKind.MFMA_SCALED))
_DK_UPCAST_MFMA: gl.constexpr = gl.constexpr(int(DotKind.UPCAST_MFMA))

__all__ = [
    "MoeKernelConfig",
    "_moe_gluon_gemm1",
    "_moe_gluon_gemm2",
    "moe_gemm_launch_metadata",
]


class MoeKernelConfig(NamedTuple):
    """Every compile-time knob, as **four** leaves.

    A ``@gluon.aggregate`` cannot be a kernel argument, so the config is *constructed
    twice*: once on the host off these same numbers (for the grid tuple), once in-kernel
    (for the layouts). Keeping the arithmetic in ``KernelTuningConfig`` is what stops the
    two from drifting.

    ``func`` and ``tuning`` are whole plain-Python NamedTuples inside a single
    ``gl.constexpr`` rather than 36 separate constexpr fields: Triton's argument
    specializer walks every leaf of a tuple argument on **every launch**, and at decode
    the launch path is the critical path (a T=1 grouped GEMM is a ~30 us kernel behind a
    ~100 us host issue). Four leaves instead of thirty-eight is worth the indirection.
    """

    func: gl.constexpr  # FuncSpec
    tuning: gl.constexpr  # TuningSpec
    N: gl.constexpr
    K: gl.constexpr


def moe_gemm_launch_metadata(grid, kernel, args):
    # Reads the flat parameter names of _moe_gluon_gemm1 / _moe_gluon_gemm2 -- the
    # aggregates only exist once the kernel has reassembled them, which is after this
    # runs. Drives profiler display only, not codegen.
    ret = {}
    N = _cv(args["CFG_N"])
    K = _cv(args["CFG_K"])
    hist = args["rt_expt_hist"]
    n_tokens = float(hist.sum()) if hist is not None else None
    w = args["b_ptr"]
    n_w_bytes = (
        (w.numel() * w.element_size() // hist.numel()) * (hist > 0).sum()
        if hist is not None
        else w.numel() * w.element_size()
    )
    ret["name"] = f"{kernel.name} [M={n_tokens}, N={N}, K={K}]"
    if n_tokens is not None:
        ret["flops32"] = 2.0 * n_tokens * N * K
        y = args["res_ptr"]
        x = args["a_ptr"]
        ret["bytes"] = int(
            n_tokens * x.shape[-1] * x.element_size()
            + n_tokens * y.shape[-1] * y.element_size()
            + n_w_bytes
        )
    return ret


@gluon.jit
def _build_configs(cfg):
    f: gl.constexpr = cfg.func
    t: gl.constexpr = cfg.tuning
    func_cfg = KernelFuncConfig(
        _at(f, 0),
        _at(f, 1),
        _at(f, 2),
        _at(f, 3),
        _at(f, 4),
        _at(f, 5),
        _at(f, 6),
        _at(f, 7),
        _at(f, 8),
        _at(f, 9),
        _at(f, 10),
        _at(f, 11),
        _at(f, 12),
    )
    tuning_cfg = KernelTuningConfig(
        func_cfg,
        _at(t, 0),
        _at(t, 1),
        _at(t, 2),
        _at(t, 3),
        _at(t, 4),
        _at(t, 5),
        _at(t, 6),
        _at(t, 7),
        _at(t, 8),
        _at(t, 9),
        _at(t, 10),
        _at(t, 11),
        _at(t, 12),
        _at(t, 13),
        _at(t, 14),
        _at(t, 15),
        _at(t, 16),
        _at(t, 17),
        _at(t, 18),
        _at(t, 19),
        _at(t, 20),
        _at(t, 21),
        _at(t, 22),
        _at(t, 23),
        _at(t, 24),
        _at(t, 25),
        _at(t, 26),
        _at(t, 27),
        _at(t, 28),
        _at(t, 29),
        _at(t, 30),
        _at(t, 31),
        _at(t, 32),
        _at(t, 33),
        _at(t, 34),
        _at(t, 35),
        _at(t, 36),
    )
    return func_cfg, tuning_cfg


@gluon.jit
def _gather_rows(
    rt,
    block_id,
    M_e,
    start_m,
    layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    ROW_OFF: gl.constexpr,
    MINI_M: gl.constexpr,
    HAS_GATHER: gl.constexpr,
):
    """Row indices of one mini-M block, already resolved through the gather table.

    TODO: stage the table in LDS instead. One block-M-wide ``buffer_load_to_shared``
    issued next to the routing scalars, committed there, with the ``wait_group`` and the
    LDS read deferred to just before ``pc`` is built, would replace ``num_mini_m()``
    register loads per consumer and hide the fetch behind the whole b-side address
    computation -- today the scheduler parks ``s_waitcnt vmcnt(0)`` ~9 instructions after
    the load, so the latency is fully exposed. It also serves both consumers (payload and
    scale offsets) from one fetch instead of one per layout.

    Attempted and reverted: on gfx950 the direct-to-LDS lowering refuses it for *this*
    table, and the op then survives to LLVM as an unconverted
    ``builtin.unrealized_conversion_cast``. ``canLoadDirectToLDS`` (AMD Utility.cpp) wants
    ``contig * elemBits`` in {32, 128} -- CDNA4 disables 8/16-bit direct-to-LDS -- so a
    uint16 table must pack two entries per lane, and ``getContiguity(ptr, offset)`` then
    demands both a 4-byte-aligned *scalar base* and offset contiguity >= 2. The base is
    ``gather_indx + start_m`` with ``start_m`` a raw prefix sum (odd about half the time),
    and any mask or ``minimum`` clamp on the offsets collapses the contiguity. Measured:
    int32 lowers with a bare, clamped *or* masked range; uint16 lowers only with a bare
    range off a provably aligned base.

    The way in is a 32-bit, block-aligned table -- which ``_sorted_token_id_map`` in
    moe_op_gemm_gluon.py already builds (int32, padded to ``block_m``, ``// n_expts_act``
    applied, memoised on ``expt_data``) for the sorted-scales path. Reading rows from it
    at ``pid_m * block_m`` makes alignment and contiguity trivial and keeps the mask; the
    cost is building it for configs that do not already.
    """
    offs = BLOCK_M * block_id + ROW_OFF + gl.arange(0, MINI_M, layout=layout)
    live = offs < M_e
    if require_constexpr(HAS_GATHER):
        # gather_indx is uint16 when n_gates <= 65535, else int32; it indexes gates, so
        # divide by n_expts_act to get the token row.
        rows = (
            gl.load(rt.gather_indx + start_m + offs, mask=live, other=0)
            // rt.n_expts_act
        )
    else:
        rows = start_m + gl.where(live, offs, 0)
    return rows.to(gl.int32)


@gluon.constexpr_function
def _packed_sel(tuning_cfg, idx):
    """The byte-selector list for a pre-packed scale operand, or None if it is not."""
    if tuning_cfg.scale_packed_ok(idx) and tuning_cfg.scale_via_lds(idx):
        return tuning_cfg.scale_packed_sel(idx)
    return None


@gluon.constexpr_function
def _any_packed(tuning_cfg):
    return (
        _packed_sel(tuning_cfg, 0) is not None or _packed_sel(tuning_cfg, 1) is not None
    )


@gluon.jit
def _dot(a, a_scale, b, b_scale, acc, func_cfg, tuning_cfg):
    """The one matrix instruction, dispatched on the operand pair."""
    kind: gl.constexpr = func_cfg.dot_kind()
    a_sel: gl.constexpr = _packed_sel(tuning_cfg, 0)
    b_sel: gl.constexpr = _packed_sel(tuning_cfg, 1)
    any_packed: gl.constexpr = _any_packed(tuning_cfg)
    if require_constexpr(kind == _DK_MFMA_SCALED and any_packed):
        # At least one scale tile arrived as pre-packed dwords; the other, if any,
        # takes the ordinary path through the same instruction.
        out = gl.amd.cdna4.mfma_scaled_packed(
            a=a,
            a_scale=a_scale,
            a_scale_sel=a_sel,
            a_format=func_cfg.mx_format(0),
            b=b,
            b_scale=b_scale,
            b_scale_sel=b_sel,
            b_format=func_cfg.mx_format(1),
            acc=acc,
        )
    elif require_constexpr(kind == _DK_MFMA_SCALED):
        out = gl.amd.cdna4.mfma_scaled(
            a=a,
            a_scale=a_scale,
            a_format=func_cfg.mx_format(0),
            b=b,
            b_scale=b_scale,
            b_format=func_cfg.mx_format(1),
            acc=acc,
        )
    elif require_constexpr(kind == _DK_UPCAST_MFMA):
        out = gl.amd.cdna4.mfma(a, b, acc)
    else:
        gl.static_assert(kind == _DK_MFMA)
        out = gl.amd.cdna4.mfma(a, b, acc)
    return out


@gluon.jit
def _mini_scale_off(base, step, HAS_SCALE: gl.constexpr):
    """Advance a register-path scale offset, or pass the None sentinel through."""
    if require_constexpr(HAS_SCALE):
        out = base + step
    else:
        out = base
    return out


@gluon.jit
def _n_start(pid_n, ni: gl.constexpr, N, func_cfg, tuning_cfg):
    """Raw-N index where mini-N block ``ni`` of CTA column ``pid_n`` begins.

    The one place the gate/up packing is expressed. Interleaved, the CTA tile is a
    contiguous ``BLOCK_N`` run and mini blocks slice it. Split, a CTA owns
    ``MINI_BLOCK_N`` *emitted* channels and reads each side from its own half of N, so
    the two mini blocks are ``N/2`` apart -- their tiles land at the same emitted
    channels, which is what lets the epilogue pair them elementwise.

    Everything downstream of this addresses a plain global ``n``: the 16-column
    preshuffle in :func:`_blocked_b_offsets`, both scale shuffles and every LDS tile are
    unchanged by the choice.
    """
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * MBN + ni * (N // 2)
    else:
        out = pid_n * BN + ni * MBN
    return out


@gluon.jit
def _n_split_offs(pid_n, i, N, func_cfg, tuning_cfg):
    """:func:`_n_start` for a whole-BLOCK_N index tensor ``i`` in ``[0, BLOCK_N)``.

    Same mapping, expressed elementwise, for the two consumers that address the CTA
    tile as one run rather than per mini block: the ``B_IN_REG`` scale fetch and the
    bias staging copy. Uniform arithmetic on a loop-invariant tensor, so it is hoisted.
    """
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * MBN + (i // MBN) * (N // 2) + i % MBN
    else:
        out = pid_n * tuning_cfg.BLOCK_N + i
    return out


@gluon.jit
def _blocked_b_offsets(
    layout: gl.constexpr,
    PK_B: gl.constexpr,
    n0,
    MBN: gl.constexpr,
    KB,
):
    """Byte offsets of one B mini tile in a 16-column-blocked weight tensor.

    ``utils/shuffle.py::shuffle_weight(w, (16, 16))`` moves byte ``(n, k)`` of an expert
    to ``(n//16)*(KB*16) + (k//16)*256 + (n%16)*16 + k%16``, where ``KB`` is the stored
    (packed) K extent. ``k`` here is the byte *within the stage*, which is what the
    caller's per-stage pointer bump of ``PK_B // 16 * 256`` makes correct.

    Used at two layouts: the copy layout, for the direct-to-LDS staging whose shared
    tile carries the matching ``byte_unit_lds_layout`` permutation, and the operand-B
    fragment layout, for the ``B_IN_REG`` path that skips LDS altogether.
    """
    kk = gl.arange(0, PK_B, layout=gl.SliceLayout(1, layout))[:, None]
    nn = (n0 + gl.arange(0, MBN, layout=gl.SliceLayout(0, layout)))[None, :]
    return (nn // 16) * (KB * 16) + (kk // 16) * 256 + (nn % 16) * 16 + kk % 16


_NO_SCALE: gl.constexpr = gl.constexpr(None)


@gluon.jit
def _a_scale_offsets(a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg):
    """Byte offsets of the A scale tiles, one per mini-M block (``None`` if unscaled).

    Two shapes, picked by ``A_SCALE_SORTED_SHUFFLED``: a flat run into the
    moe_sort_scales pre-pass output, or a gathered ``[MBM, SK]`` grid over the raw
    ``(M, K/32)`` tensor. The caller only sees a tuple it can index per mini block.
    """
    if require_constexpr(not func_cfg.a_has_scale()):
        return _NO_SCALE

    BM: gl.constexpr = tuning_cfg.BLOCK_M
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    NM: gl.constexpr = tuning_cfg.num_mini_m()
    SK: gl.constexpr = tuning_cfg.BLOCK_K // MX_GROUP
    if require_constexpr(tuning_cfg.scale_via_lds(0)):
        asl: gl.constexpr = tuning_cfg.dot_operand_scale_copy_layout(0)
    else:
        asl: gl.constexpr = tuning_cfg.dot_operand_scale_fragment_layout(0)

    if require_constexpr(tuning_cfg.A_SCALE_SORTED_SHUFFLED):
        # moe_sort_scales has already applied the gather and the fragment permute,
        # so there is no table lookup and no per-row stride here: the tile is one
        # contiguous run and `asl` (the fragment layout) already places each lane on
        # the byte it needs. The stage bump is a flat 256 B, which the host encodes
        # as scale_stride_k = 32 against the shared s_step = SK.
        gl.static_assert(
            tuning_cfg.sorted_shuffled_ok(),
            "A_SCALE_SORTED_SHUFFLED needs the layout moe_sort_scales writes: "
            "16x16x128, warps (n, 1), tiles_per_warp (2, 1), BLOCK_K 256, "
            "BLOCK_M == 32 * warps_m, MINI_BLOCK_M == BLOCK_M",
        )
        # The shuffle indexes the *padded* row space -- expert e starts at
        # token_offs_pad[e] whole blocks -- while start_m is the raw offset and is
        # not block-aligned. token_offs_pad[e] + block_id is exactly pid_m, which
        # block_pid_map is built to enumerate, so the chunk index is free here.
        #
        # The staging copy is flat: a stage's tile is one 256 B run per 32-row
        # stripe, and consecutive stripes are one whole K sweep apart. Keeping the
        # copy 1-D is what lets it vectorise -- in the [BM, SK] view a lane's four
        # bytes straddle both axes and nothing can widen the access.
        # One tile per mini-M block. The tile is a whole number of 32-row stripes
        # and stripes are contiguous, so mini block mi is just the run starting
        # mi * (MBM // 32) stripes into this pid_m's chunk.
        # SA is the *fill* tile, which scale_mini_m() may make wider than the
        # payload mini block. Mini blocks sharing a tile get the same base offsets;
        # only the owner (mi % RA == 0) actually issues the copy.
        SA: gl.constexpr = tuning_cfg.scale_flat_shape(0)
        RA: gl.constexpr = tuning_cfg.scale_tile_ratio_a()
        SMM: gl.constexpr = tuning_cfg.scale_mini_m()
        sa = gl.arange(0, SA[0], layout=gl.SliceLayout(1, asl))[:, None]
        ja = gl.arange(0, SA[1], layout=gl.SliceLayout(0, asl))[None, :]
        a_scale_offs = ()
        for mi in gl.static_range(NM):
            a_scale_offs = a_scale_offs + (
                (pid_m * (BM // 32) + (mi // RA) * (SMM // 32) + sa) * K + ja,
            )
    else:
        a_scale_offs = ()
        for mi in gl.static_range(NM):
            rows_as = _gather_rows(
                rt,
                block_id,
                M_e,
                start_m,
                gl.SliceLayout(1, asl),
                BM,
                mi * MBM,
                MBM,
                func_cfg.has_gather,
            )
            a_scale_offs = a_scale_offs + (
                rows_as[:, None] * a.scale_stride_m
                + gl.arange(0, SK, layout=gl.SliceLayout(0, asl))[None, :]
                * a.scale_stride_k,
            )
    return a_scale_offs


@gluon.jit
def _b_scale_offsets(b, pid_n, N, K, func_cfg, tuning_cfg):
    """Byte offsets of the B scale tiles, one per mini-N block (``None`` if unscaled).

    Three shapes: the CDNA4_SCALE preshuffle read element-wise into the fragment
    registers (``B_IN_REG``), the same preshuffle read as a flat run for staging, and
    the raw ``(K/32, N)`` grid. The caller only sees a tuple it can index per mini
    block -- the ``B_IN_REG`` form is one whole-BLOCK_N tile, so that tuple is 1-long.
    """
    if require_constexpr(not func_cfg.b_has_scale()):
        return _NO_SCALE

    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NN: gl.constexpr = tuning_cfg.num_mini_n()
    SK: gl.constexpr = tuning_cfg.BLOCK_K // MX_GROUP
    if require_constexpr(tuning_cfg.scale_via_lds(1)):
        bsl: gl.constexpr = tuning_cfg.dot_operand_scale_copy_layout(1)
    elif require_constexpr(tuning_cfg.scale_shuffled(1)):
        # address-ordered, so the widened load fills registers correctly
        bsl: gl.constexpr = tuning_cfg.shuffled_scale_mem_layout(1)
    else:
        bsl: gl.constexpr = tuning_cfg.dot_operand_scale_fragment_layout(1)

    if require_constexpr(tuning_cfg.B_SCALE_SHUFFLED and tuning_cfg.B_IN_REG):
        # Straight into the scale fragment registers, addressed element-wise so it
        # rides the same two-stage pipeline as B's payload.
        ns = _n_split_offs(
            pid_n,
            gl.arange(0, BN, layout=gl.SliceLayout(1, bsl)),
            N,
            func_cfg,
            tuning_cfg,
        )[:, None]
        ks = gl.arange(0, SK, layout=gl.SliceLayout(0, bsl))[None, :]
        b_scale_offs = (
            (ns // 32) * K
            + (ks % 4) * 64
            + (ns % 16) * 4
            + (ks // 4) * 2
            + (ns % 32) // 16,
        )
    elif require_constexpr(tuning_cfg.B_SCALE_SHUFFLED):
        # utils/shuffle.py::shuffle_scale_moe (CDNA4_SCALE) has already permuted the
        # weight scales into MFMA fragment order: per expert the tile is
        # (N/32, K) bytes, and within a 32-row stripe lane L of a stage reads the
        # dword at stage*256 + L*4. Same shape as the A-side shuffle, but done
        # offline -- the weights are static, so this costs nothing at run time.
        # One tile per mini-N block; stripes are contiguous, so block ni starts
        # ni * (MBN // 32) stripes into this pid_n's run. Same slicing as the A side.
        SB: gl.constexpr = tuning_cfg.scale_flat_shape(1)
        sb = gl.arange(0, SB[0], layout=gl.SliceLayout(1, bsl))[:, None]
        jb = gl.arange(0, SB[1], layout=gl.SliceLayout(0, bsl))[None, :]
        b_scale_offs = ()
        for ni in gl.static_range(NN):
            b_scale_offs = b_scale_offs + (
                (_n_start(pid_n, ni, N, func_cfg, tuning_cfg) // 32 + sb) * K + jb,
            )
    else:
        b_scale_offs = ()
        for ni in gl.static_range(NN):
            b_scale_offs = b_scale_offs + (
                (
                    _n_start(pid_n, ni, N, func_cfg, tuning_cfg)
                    + gl.arange(0, MBN, layout=gl.SliceLayout(1, bsl))
                )[:, None]
                * b.scale_stride_n
                + gl.arange(0, SK, layout=gl.SliceLayout(0, bsl))[None, :]
                * b.scale_stride_k,
            )
    return b_scale_offs


@gluon.jit
def _opt_at(t, i: gl.constexpr, PRESENT: gl.constexpr):
    """Index a per-mini-block tuple, or pass the absent-scale sentinel through."""
    if require_constexpr(PRESENT):
        out = t[i]
    else:
        out = _NO_SCALE
    return out


@gluon.jit
def _a_block_frags(
    lds,
    buf_idx,
    mi: gl.constexpr,
    scale_ptr,
    scale_offs,
    func_cfg,
    tc,
    RELAXED: gl.constexpr = False,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    """Every LDS->register A fragment of one mini-M block, all mini-K, as a flat tuple.

    Two entries per mini-K step -- payload, scale -- so :func:`_pipeline_step` can hold
    the tail of one stage across a pipeline iteration and hand it to :func:`_block_dot`
    one stage later.

    The pair is present even for an operand with no scale: the tuple is loop-carried, so
    Triton asks every element for its ``.type`` and a ``None`` there aborts codegen. An
    absent scale therefore parks the payload's own SSA value in the slot -- a duplicate
    reference costs no register -- and :func:`_block_dot` puts the ``None`` back from the
    same compile-time predicate.
    """
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = func_cfg.has_scale(0)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        a, a_s = lds.load_a_frag(
            buf_idx,
            mi,
            i,
            scale_ptr,
            _mini_scale_off(scale_offs, i * SK_MINI, HAS),
            RELAXED,
            READ_PAYLOAD,
            READ_SCALE,
        )
        if require_constexpr(not READ_PAYLOAD):
            a = a_s
        if require_constexpr(HAS and READ_SCALE):
            slot = a_s
        else:
            slot = a
        frags = frags + (a, slot)
    return frags


@gluon.jit
def _b_block_frags(
    lds,
    buf_idx,
    ni: gl.constexpr,
    scale_ptr,
    scale_offs,
    func_cfg,
    tc,
    b_ptr=None,
    b_frag_offs=None,
    RELAXED: gl.constexpr = False,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    """The operand-B mirror of :func:`_a_block_frags`, over one mini-N block."""
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = func_cfg.has_scale(1)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        b, b_s = lds.load_b_frag(
            buf_idx,
            ni,
            i,
            scale_ptr,
            _mini_scale_off(scale_offs, i * SK_MINI, HAS),
            b_ptr,
            b_frag_offs,
            RELAXED,
            READ_PAYLOAD,
            READ_SCALE,
        )
        if require_constexpr(not READ_PAYLOAD):
            b = b_s
        if require_constexpr(HAS and READ_SCALE):
            slot = b_s
        else:
            slot = b
        frags = frags + (b, slot)
    return frags


@gluon.jit
def _take_pairs(frags, LO: gl.constexpr, N: gl.constexpr):
    """Mini-K steps ``[LO, LO+N)`` of a one-operand fragment tuple, as a fresh tuple.

    Built element-wise rather than sliced: a loop-carried tuple arrives as a
    ``tl.tuple``, which indexes but does not slice.
    """
    out = ()
    for i in gl.static_range(N):
        out = out + (frags[2 * (LO + i)], frags[2 * (LO + i) + 1])
    return out


@gluon.jit
def _maybe_block_dot(
    a_frags,
    b_frags,
    acc,
    N_MINI: gl.constexpr,
    func_cfg,
    tuning_cfg,
    DO_MFMA: gl.constexpr,
):
    """:func:`_block_dot`, or the accumulator untouched when the step emits no MFMA."""
    if require_constexpr(DO_MFMA):
        return _block_dot(a_frags, b_frags, acc, N_MINI, func_cfg, tuning_cfg)
    return acc


@gluon.jit
def _block_dot(a_frags, b_frags, acc, N_MINI: gl.constexpr, func_cfg, tuning_cfg):
    """The MFMAs of one mini (M, N) block over ``N_MINI`` mini-K steps."""
    for i in gl.static_range(N_MINI):
        if require_constexpr(func_cfg.has_scale(0)):
            a_s = a_frags[2 * i + 1]
        else:
            a_s = _NO_SCALE
        if require_constexpr(func_cfg.has_scale(1)):
            b_s = b_frags[2 * i + 1]
        else:
            b_s = _NO_SCALE
        acc = _dot(a_frags[2 * i], a_s, b_frags[2 * i], b_s, acc, func_cfg, tuning_cfg)
    return acc


@aggregate
@strip_annotate
class _PipelineConst:
    """The loop-*invariant* half of a K-pipeline step.

    Hoisted once before the loop: the LDS buffers, the HBM scale bases, the tile offsets
    every stage reuses, and the per-stage pointer increments. Split from
    :class:`_PipelineState` so the loop-carried set is exactly the values that have to
    cross the back edge -- anything parked in here by mistake would otherwise be threaded
    through every iteration for nothing.

    Every offset grid is a *tuple*, one entry per mini block: ``num_mini_m()`` for the A
    side, ``num_mini_n()`` for the B side. Each entry is laid out for its own mini tile's
    copy layout, which is what keeps every per-mini-block direct-to-LDS copy as wide and
    as coalesced as the whole-tile copy it replaces.

    ``a_scale_offs``/``b_scale_offs`` serve *both* sides: the direct-to-LDS copy and, for
    the register fallback, the consume-side load. Both walk K by advancing their scalar
    base pointer in :class:`_PipelineState` instead of the offset grid, so this tile grid
    is computed once and never moves.

    The two ``*_scale_stride_k`` are lifted out of the operand tuples rather than kept as
    ``a``/``b``: the non-quantised operand types have no such field at all, so it can only
    be read behind the ``has_scale`` predicate that the caller already evaluates.
    """

    lds: LDSManager
    a_offs: tl.tuple | tuple
    b_offs: tl.tuple | tuple
    b_frag_offs: tl.tuple | tuple | gl.constexpr
    a_scale_offs: tl.tuple | tuple | gl.constexpr
    b_scale_offs: tl.tuple | tuple | gl.constexpr
    a_scale_stride_k: gl.constexpr
    b_scale_stride_k: gl.constexpr
    a_step: gl.constexpr
    b_step: gl.constexpr
    s_step: gl.constexpr
    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig

    @gluon.constexpr_function
    def __init__(
        self,
        lds,
        a_offs,
        b_offs,
        b_frag_offs,
        a_scale_offs,
        b_scale_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
    ):
        self.lds = lds
        self.a_offs = a_offs
        self.b_offs = b_offs
        self.b_frag_offs = _opt(b_frag_offs)
        self.a_scale_offs = _opt(a_scale_offs)
        self.b_scale_offs = _opt(b_scale_offs)
        # re-wrapped: the frontend unwraps a constexpr argument before a
        # constexpr_function sees it, so these arrive as plain ints
        self.a_scale_stride_k = gl.constexpr(_v(a_scale_stride_k))
        self.b_scale_stride_k = gl.constexpr(_v(b_scale_stride_k))
        self.a_step = gl.constexpr(_v(a_step))
        self.b_step = gl.constexpr(_v(b_step))
        self.s_step = gl.constexpr(_v(s_step))
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg


@aggregate
@strip_annotate
class _PipelineState:
    """The loop-*variant* half: everything one ``BLOCK_K`` stage hands to the next.

    Every address here is in HBM -- the LDS side of the pipeline is entirely inside
    ``_PipelineConst.lds``. The plain ``*_hbm_ptr`` are the copy *sources*, walked one
    ``BLOCK_K`` per stage; the ``*_read_hbm_ptr`` are the consume-side pointers used by
    the register fallback, when a tile is too small to be written to LDS coalesced and
    is loaded straight from HBM at MFMA time instead.

    Bundled rather than passed as a flat tuple because the two advance at *different*
    points in a step -- the copy side before the fill is issued, the read side one stage
    later when its fragments are consumed -- and keeping them adjacent is what makes it
    obvious that advancing only one of the pair is the bug.

    ``a_frags``/``b_frags`` hold the fragments read one stage ahead of the MFMAs that
    take them, ``num_mini_m()`` (resp. ``num_mini_n()``) blocks of ``num_prefetch_mini()``
    payload/scale pairs. ``acc`` is one ``MINI_BLOCK_M x MINI_BLOCK_N`` accumulator per
    mini block, in ``mi * num_mini_n() + ni`` order: a dot writes a whole tensor, so a
    per-mini-block MFMA needs a per-mini-block accumulator -- there is no way to write
    back into a slice of a single ``BLOCK_M x BLOCK_N`` one.
    """

    a_hbm_ptr: gl.tensor
    b_hbm_ptr: gl.tensor
    a_scale_hbm_ptr: gl.tensor | gl.constexpr
    b_scale_hbm_ptr: gl.tensor | gl.constexpr
    a_scale_read_hbm_ptr: gl.tensor | gl.constexpr
    b_scale_read_hbm_ptr: gl.tensor | gl.constexpr
    a_frags: tl.tuple | tuple
    b_frags: tl.tuple | tuple
    b_frags2: tl.tuple | tuple
    acc: tl.tuple | tuple

    # constexpr_function, not @gluon.jit: Triton's aggregate metaclass calls __init__
    # itself from outside kernel scope, which a JITFunction refuses, and a bare Python
    # function trips its "Unsupported function referenced" member walk. Every argument
    # is already a Triton value (or a constexpr None for an absent scale) built by the
    # caller inside the kernel, so there is nothing to trace here anyway.
    @gluon.constexpr_function
    def __init__(
        self,
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_hbm_ptr,
        b_scale_read_hbm_ptr,
        a_frags,
        b_frags,
        b_frags2,
        acc,
    ):
        self.a_hbm_ptr = a_hbm_ptr
        self.b_hbm_ptr = b_hbm_ptr
        # _opt, not a bare assign: an operand with no scale arrives as raw Python None
        # (a8w8, bf16), which the field annotation rejects -- it has to be constexpr.
        self.a_scale_hbm_ptr = _opt(a_scale_hbm_ptr)
        self.b_scale_hbm_ptr = _opt(b_scale_hbm_ptr)
        self.a_scale_read_hbm_ptr = _opt(a_scale_read_hbm_ptr)
        self.b_scale_read_hbm_ptr = _opt(b_scale_read_hbm_ptr)
        self.a_frags = a_frags
        self.b_frags = b_frags
        self.b_frags2 = b_frags2
        self.acc = acc


@gluon.constexpr_function
def _slot_index(mi, ni, NM, NN):
    """Position of slot ``(mi, ni)`` in the traversal: N outer, M inner.

    This one number is the slot's place in three orderings at once -- the visitation
    order of the slot loop, the index of its accumulator in the carried tuple, and
    (under the even schedule) the position of its fill in :func:`_fill_order`. They have
    to agree, which is why they all come from here.

    Down the M axis first, so at NM = NN = 2 the four slots are visited
    ``(0,0), (1,0), (0,1), (1,1)`` -- the region order of the reference kernel in
    gfx950-gluon-tutorials .../a16w16/v8_sliceMN, whose four DOTs are
    ``(A_top, B_left), (A_bot, B_left), (A_top, B_right), (A_bot, B_right)``.

    With NM == 1 or NN == 1 this is the identity on the old row-major order, so only a
    genuinely 2-D mini-block split sees any change.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return ni * NM + mi


@gluon.constexpr_function
def _relax_read(mi, ni, NM, NN):
    """May this slot's LDS reads skip the pre-barrier lgkmcnt drain?

    Keep the last actual read plain so it retires the earlier relaxed reads before
    the next K step reuses their buffers. Larger splits have compute-only tail slots.
    """
    return _slot_index(mi, ni, NM, NN) < _v(NM) + _v(NN) - 1


@gluon.constexpr_function
def _fill_order(NM, NN):
    """The ``NM + NN`` mini-block fills of one stage as ``(is_a, tile)``, in issue order.

    Even at NM = NN = 2: B(0), A(0), A(1), B(1) -- the reference kernel's load order
    ``B_left -> A_top -> A_bot -> B_right``. Paired with the N-outer slot walk of
    :func:`_slot_index` it reproduces that kernel's four regions exactly:

        region 0  DOT(A_top, B_left)   fill B(0)
        region 1  DOT(A_bot, B_left)   fill A(0)
        region 2  DOT(A_top, B_right)  fill A(1)
        region 3  DOT(A_bot, B_right)  fill B(1)

    Otherwise: A and B interleaved -- A(0), B(0), A(1), B(1) ... -- so consecutive slots
    alternate operand and the byte volume per slot stays as level as two tile sizes
    allow. The reference order is only defined for the 2x2 split it was designed for, so
    anything else keeps the interleave.
    """
    NM, NN = _v(NM), _v(NN)
    if NM == 2 and NN == 2:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = []
    for i in range(max(NM, NN)):
        if i < NM:
            out.append((1, i))
        if i < NN:
            out.append((0, i))
    return out


@gluon.constexpr_function
def _fill_group_pos(is_a, i, NM, NN):
    """Position of one mini-block fill's commit group inside its stage."""
    is_a, i = _v(is_a), _v(i)
    order = _fill_order(NM, NN)
    return order.index((1 if is_a else 0, i))


@gluon.constexpr_function
def _slot_fill_pos(mi, ni, NM, NN):
    """Which fill (position in :func:`_fill_order`) slot ``(mi, ni)`` issues, or None.

    One fill per slot, in flat slot order -- exactly the tutorial's layout, where each of
    the four ``mfma``/``mem`` region pairs moves one tile and commits one group. Both
    axes are split (validate() insists), so there are never more fills than slots.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    s = _slot_index(mi, ni, NM, NN)
    return s if s < NM + NN else None


@gluon.constexpr_function
def _fill_tile_of(pos, NM, NN, want_a):
    """Tile index of fill ``pos`` if it is an A (resp. B) fill, else None.

    Two constexpr calls rather than unpacking a tuple inside the unrolled slot loop,
    where the binding would be a reassignment.
    """
    pos = _v(pos)
    if pos is None:
        return None
    is_a, tile = _fill_order(NM, NN)[pos]
    return tile if bool(is_a) == bool(_v(want_a)) else None


@gluon.constexpr_function
def _read_a_tile(mi, ni, NM, NN):
    """Operand-A mini block slot ``(mi, ni)`` reads out of LDS, or None.

    The same one-per-slot assignment the fills use, so a slot's fill and its read name
    the *same* position in :func:`_fill_order`. Its wait then works out to
    ``G - 1 - s + STAGES_BETWEEN * G + s`` -- the ``s`` cancels and every slot waits on
    the same constant, which is exactly the uniform ``wait_group`` the reference kernel
    uses.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _fill_tile_of(_slot_fill_pos(mi, ni, NM, NN), NM, NN, True)


@gluon.constexpr_function
def _read_b_tile(mi, ni, NM, NN):
    """Operand-B mini block slot ``(mi, ni)`` reads out of LDS, or None."""
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _fill_tile_of(_slot_fill_pos(mi, ni, NM, NN), NM, NN, False)


@gluon.constexpr_function
def _scale_fill_slot(is_a, tile, NM, NN, SCALE_FILL_MID=False):
    """Slot index that issues the scale copy for A(tile) / B(tile).

    Default: the same slot as the payload, so a tile's scale and payload share one
    commit group. Under ``SCALE_FILL_MID`` at NM=NN=2 both A(0)/B(0) scales go to
    slot 1 and both A(1)/B(1) scales to slot 2.
    """
    is_a, tile = bool(_v(is_a)), _v(tile)
    NM, NN = _v(NM), _v(NN)
    if _v(SCALE_FILL_MID) and NM == 2 and NN == 2:
        return 1 + tile
    return _fill_group_pos(is_a, tile, NM, NN)


@gluon.constexpr_function
def _slot_scale_fills(mi, ni, NM, NN, want_a, SCALE_FILL_MID=False):
    """Tile whose A (resp. B) scale copy slot ``(mi, ni)`` issues, or None.

    Still one slot per mini block even when several share a scale tile: the commit
    group has to be emitted either way, because _slot_wait counts G = NM + NN groups
    per stage. fill_a_scale_lds drops the redundant *copy* and leaves the group empty.
    """
    s = _slot_index(mi, ni, NM, NN)
    n = NM if _v(want_a) else NN
    for t in range(_v(n)):
        if _scale_fill_slot(_v(want_a), t, NM, NN, SCALE_FILL_MID) == s:
            return t
    return None


@gluon.constexpr_function
def _fills_before(mi, ni, NM, NN, ANY):
    """Groups this stage has already committed when slot ``(mi, ni)`` is reached.

    Counted against the N-outer walk of :func:`_slot_index`: one fill per slot, so the
    walk's position is the count, capped at the ``NM + NN`` a stage has.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if not _v(ANY):
        return 0
    return min(_slot_index(mi, ni, NM, NN), NM + NN)


@gluon.constexpr_function
def _groups_per_stage(NM, NN, SCHEME):
    """Commit groups one K stage emits at commit granularity ``SCHEME``.

    The device-side twin of ``KernelTuningConfig.commit_groups_per_stage()`` -- the same
    three cases, reached from the constexpr helpers, which are handed NM/NN rather than
    the config. ``PER_FILL`` counts the payload fills only and is therefore conservative
    when a scale sits on a slot of its own; the other two are exact.
    """
    NM, NN, SCHEME = _v(NM), _v(NN), _v(SCHEME)
    if SCHEME == int(WaitCommitScheme.PER_SLOT):
        return NM * NN
    if SCHEME == int(WaitCommitScheme.PER_STAGE):
        return 1
    return NM + NN


@gluon.constexpr_function
def _slot_wait(mi, ni, NM, NN, STAGES_BETWEEN, ANY_FILL, SCHEME, SCALE_FILL_MID=False):
    """Outstanding-group count that retires everything slot ``(mi, ni)`` is about to read.

    A group is retired once ``wait_group(n)`` leaves at most ``n`` behind it. Counting
    forward from the target group: the rest of its own stage, then ``STAGES_BETWEEN``
    whole stages, then whatever the current stage has committed so far. The slot reads
    A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``; when it reads both, the later
    group's (smaller) count wins. ``None`` means the slot reads nothing and needs no wait.

    Everything here is counted in groups, so it is only a description of the real stream
    while ``SCHEME`` is the granularity :func:`_slot_fills` actually commits at -- which
    is why the two read one knob. Under ``PER_SLOT`` a slot commits exactly one group, so
    the ordinals ``_fill_group_pos`` / ``_scale_fill_slot`` return (they are slot indices)
    line up with the stream directly.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    SCHEME = _v(SCHEME)
    PER_STAGE = SCHEME == int(WaitCommitScheme.PER_STAGE)
    G = _groups_per_stage(NM, NN, SCHEME)
    if PER_STAGE:
        # One group per stage: the stage's own group is not committed until its last
        # slot, so nothing of it is outstanding here and every fill sits at p == 0.
        # The count is therefore just the whole stages in flight.
        base = _v(STAGES_BETWEEN)
    else:
        base = _v(STAGES_BETWEEN) * G + _fills_before(mi, ni, NM, NN, ANY_FILL)
    out = None
    ta = _read_a_tile(mi, ni, NM, NN)
    if ta is not None:
        # The payload and the scale of the same tile can sit in different commit
        # groups (see _scale_fill_slot); the later of the two is what has to retire.
        p = max(
            _fill_group_pos(True, ta, NM, NN),
            _scale_fill_slot(True, ta, NM, NN, SCALE_FILL_MID),
        )
        out = base if PER_STAGE else G - 1 - p + base
    tb = _read_b_tile(mi, ni, NM, NN)
    if tb is not None:
        p = max(
            _fill_group_pos(False, tb, NM, NN),
            _scale_fill_slot(False, tb, NM, NN, SCALE_FILL_MID),
        )
        w = base if PER_STAGE else G - 1 - p + base
        out = w if out is None else min(out, w)
    return out


@gluon.constexpr_function
def _stage_wait(NM, NN, STAGES_BETWEEN, ANY_FILL, SCHEME, SCALE_FILL_MID=False):
    """Strongest (smallest) wait_group count over all slots of a stage."""
    NM, NN = _v(NM), _v(NN)
    ws = [
        _slot_wait(mi, ni, NM, NN, STAGES_BETWEEN, ANY_FILL, SCHEME, SCALE_FILL_MID)
        for ni in range(NN)
        for mi in range(NM)
    ]
    ws = [w for w in ws if w is not None]
    if not ws:
        return None
    return max(min(ws), 0)


@gluon.jit
def _stage_wait_group(
    lds,
    NM: gl.constexpr,
    NN: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    ANY_FILL: gl.constexpr,
    SCHEME: gl.constexpr,
    WAIT_SLACK: gl.constexpr = 0,
    SCALE_FILL_MID: gl.constexpr = False,
):
    """One wait_group for the whole stage, emitted ahead of the slot walk.

    Selected by ``WaitCommitScheme.PER_FILL`` -- see
    ``KernelTuningConfig.wait_at_stage_head()`` for why only that level may hoist it.
    The count is the strongest over the stage's slots, so it stands in for all of them
    and no slot needs one of its own.

    ``WAIT_SLACK`` is the number of commit groups outstanding that this pipeline did
    not issue -- see :func:`_pipeline_step_impl`. They are newer than everything the
    slots read, so they never retire first; leaving them out of the count would make
    the wait retire one real group too few.
    """
    WAIT: gl.constexpr = _stage_wait(
        NM, NN, STAGES_BETWEEN, ANY_FILL, SCHEME, SCALE_FILL_MID
    )
    if require_constexpr(WAIT is not None):
        lds.wait_fill_lds_num_group(WAIT + WAIT_SLACK)


@gluon.jit
def _slot_wait_group(
    lds,
    mi: gl.constexpr,
    ni: gl.constexpr,
    NM: gl.constexpr,
    NN: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    ANY_FILL: gl.constexpr,
    SCHEME: gl.constexpr,
    WAIT_SLACK: gl.constexpr = 0,
    SCALE_FILL_MID: gl.constexpr = False,
):
    """Emit slot ``(mi, ni)``'s ``wait_group``, or nothing if it reads nothing.

    What the coarser commit levels (``PER_SLOT``, ``PER_STAGE``) use: one group then
    covers several tiles, so an earlier slot's wait does not imply this one's.

    A separate function only so the ``: gl.constexpr`` binding is legal: inside the
    unrolled slot loop it would be a reassignment, and without the annotation the count
    reaches ``wait_group`` as a runtime tensor.

    ``WAIT_SLACK``: see :func:`_stage_wait_group`.
    """
    WAIT: gl.constexpr = _slot_wait(
        mi, ni, NM, NN, STAGES_BETWEEN, ANY_FILL, SCHEME, SCALE_FILL_MID
    )
    if require_constexpr(WAIT is not None):
        lds.wait_fill_lds_num_group(WAIT + WAIT_SLACK)


@gluon.constexpr_function
def _slot_advance(mi, ni, NM, NN, KI, KU):
    """Does slot ``(mi, ni)`` own the pointer bump onto the next ``BLOCK_K`` stage?

    The stage's last slot -- and, when ``SOFF_UNROLL`` folded the body's steps into
    ``soffset``, only on the body's last step, which then bumps by ``KU`` at once. With
    the flag off ``KI``/``KU`` are ``0``/``1`` and the last term is vacuous.
    """
    return _v(mi) == _v(NM) - 1 and _v(ni) == _v(NN) - 1 and _v(KI) == _v(KU) - 1


@gluon.jit
def _slot_fills(
    pc,
    BUFFER_LOAD_INX,
    mi: gl.constexpr,
    ni: gl.constexpr,
    a_hbm_ptr,
    b_hbm_ptr,
    a_scale_hbm_ptr,
    b_scale_hbm_ptr,
    ADVANCE: gl.constexpr = False,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    STAGE_MARK: gl.constexpr = True,
):
    """The global->LDS copies slot ``(mi, ni)`` owns, committed per WAIT_COMMIT_SCHEME.

    ``PER_FILL`` marks after each copy, ``PER_SLOT`` once at the end of the slot,
    ``PER_STAGE`` once at the last slot of the stage. Nothing else about the schedule
    moves: the same copies are issued from the same slots either way, only the
    ``commit_group`` boundaries between them change -- and :func:`_slot_wait` counts
    against exactly those boundaries.

    ``KI``/``KU`` are this step's index within the unrolled body and the unroll factor.
    With ``SOFF_UNROLL`` they turn the per-step pointer bump into a per-body one: every
    copy addresses ``base + KI * step`` through ``soffset`` and ``ADVANCE`` then moves the
    base on by ``KU`` steps at once. With the flag off, ``KU`` is 1 and this is the
    original one-bump-per-step form.

    ``ADVANCE`` walks the returned HBM pointers on to the next ``BLOCK_K`` stage. It is
    done here, at the last slot, rather than after the slot loop on purpose: the warp
    pipeliner only tolerates a ``wait_group`` as the *first* op after a stage border, and
    a stage-tail ``tt.addptr`` sitting between the last ``mem`` border and the next
    slot's wait is exactly what breaks that.
    """
    func_cfg: gl.constexpr = pc.func_cfg
    NM: gl.constexpr = pc.tuning_cfg.num_mini_m()
    NN: gl.constexpr = pc.tuning_cfg.num_mini_n()
    # The slot owns at most one fill, named by its position in _fill_order.
    POS: gl.constexpr = _slot_fill_pos(mi, ni, NM, NN)
    A_TILE: gl.constexpr = _fill_tile_of(POS, NM, NN, True)
    B_TILE: gl.constexpr = _fill_tile_of(POS, NM, NN, False)
    # Scale copies may be placed on a different slot than their payload.
    A_SC: gl.constexpr = _slot_scale_fills(
        mi, ni, NM, NN, True, pc.tuning_cfg.SCALE_FILL_MID
    )
    B_SC: gl.constexpr = _slot_scale_fills(
        mi, ni, NM, NN, False, pc.tuning_cfg.SCALE_FILL_MID
    )
    # Where the commit_group marks go, per WaitCommitScheme. Exactly one of the three
    # is True, and _slot_wait counts groups at the same granularity.
    MARK_FILL: gl.constexpr = pc.tuning_cfg.commit_per_fill()
    MARK_SLOT: gl.constexpr = pc.tuning_cfg.commit_per_slot()
    MARK_STAGE: gl.constexpr = pc.tuning_cfg.commit_per_stage()
    # Byte displacement of this step from the body's base pointer. The *_step values are
    # in elements (they are added to a typed pointer), soffset is in bytes, so each is
    # scaled by its operand's storage width. All constexpr, so these fold into a literal
    # and the SGPR holding them is hoisted out of the loop.
    SOFF: gl.constexpr = pc.tuning_cfg.SOFF_UNROLL and KI > 0
    A_SOFF: gl.constexpr = (
        KI * pc.a_step * (func_cfg.operand_elem_ty(0).primitive_bitwidth // 8)
        if SOFF
        else 0
    )
    B_SOFF: gl.constexpr = (
        KI * pc.b_step * (func_cfg.operand_elem_ty(1).primitive_bitwidth // 8)
        if SOFF
        else 0
    )
    # E8M0 scales are byte-sized, so the scale strides are already byte counts.
    A_SSOFF: gl.constexpr = (
        KI * pc.s_step * pc.a_scale_stride_k if SOFF and func_cfg.a_has_scale() else 0
    )
    B_SSOFF: gl.constexpr = (
        KI * pc.s_step * pc.b_scale_stride_k if SOFF and func_cfg.b_has_scale() else 0
    )
    if require_constexpr(A_TILE is not None):
        pc.lds.fill_a_payload_lds(
            BUFFER_LOAD_INX, A_TILE, a_hbm_ptr, pc.a_offs[A_TILE], A_SOFF
        )
        # Payload and its own scale share one commit group; a scale placed elsewhere
        # gets its own below.
        if require_constexpr(A_SC == A_TILE):
            pc.lds.fill_a_scale_lds(
                BUFFER_LOAD_INX,
                A_TILE,
                a_scale_hbm_ptr,
                _opt_at(pc.a_scale_offs, A_TILE, func_cfg.a_has_scale()),
                A_SSOFF,
            )
        if require_constexpr(MARK_FILL):
            pc.lds.commit_fill_lds()
    if require_constexpr(A_SC is not None and A_SC != A_TILE):
        pc.lds.fill_a_scale_lds(
            BUFFER_LOAD_INX,
            A_SC,
            a_scale_hbm_ptr,
            _opt_at(pc.a_scale_offs, A_SC, func_cfg.a_has_scale()),
            A_SSOFF,
        )
        if require_constexpr(MARK_FILL):
            pc.lds.commit_fill_lds()
    if require_constexpr(B_TILE is not None):
        pc.lds.fill_b_payload_lds(
            BUFFER_LOAD_INX, B_TILE, b_hbm_ptr, pc.b_offs[B_TILE], B_SOFF
        )
        if require_constexpr(B_SC == B_TILE):
            pc.lds.fill_b_scale_lds(
                BUFFER_LOAD_INX,
                B_TILE,
                b_scale_hbm_ptr,
                _opt_at(pc.b_scale_offs, B_TILE, func_cfg.b_has_scale()),
                B_SSOFF,
            )
        if require_constexpr(MARK_FILL):
            pc.lds.commit_fill_lds()
    if require_constexpr(B_SC is not None and B_SC != B_TILE):
        pc.lds.fill_b_scale_lds(
            BUFFER_LOAD_INX,
            B_SC,
            b_scale_hbm_ptr,
            _opt_at(pc.b_scale_offs, B_SC, func_cfg.b_has_scale()),
            B_SSOFF,
        )
        if require_constexpr(MARK_FILL):
            pc.lds.commit_fill_lds()
    if require_constexpr(MARK_SLOT):
        # One group for everything this mini block just issued.
        pc.lds.commit_fill_lds()
    if require_constexpr(MARK_STAGE and STAGE_MARK
                         and _slot_index(mi, ni, NM, NN) == NM * NN - 1):
        # One group for the whole block, committed at the last slot.
        #
        # ``STAGE_MARK=False`` lets _pipeline_step_impl close the group after its walk
        # instead, which frees the last slot's loads from being pinned mid-walk. Every
        # OTHER caller must leave it True: the prologue fills NB-1 stages through its own
        # slot walk (see _moe_gemm_body), and if the mark is dropped there those stages
        # are never committed at all. The waits that depend on them then have no group to
        # count, UpdateAsyncWaitCount lowers them with no vmcnt, and the cooperative-fill
        # barriers that follow stop meaning anything -- a WAR race that shows up as
        # nondeterministic output in 7 runs out of 8, with the barrier count unchanged.
        pc.lds.commit_fill_lds()
    if require_constexpr(ADVANCE):
        # KU is 1 unless SOFF_UNROLL folded the body's steps into soffset, in which
        # case one bump at the last step of the body covers all of them.
        a_hbm_ptr = a_hbm_ptr + KU * pc.a_step
        b_hbm_ptr = b_hbm_ptr + KU * pc.b_step
        if require_constexpr(func_cfg.a_has_scale()):
            a_scale_hbm_ptr = a_scale_hbm_ptr + KU * pc.s_step * pc.a_scale_stride_k
        if require_constexpr(func_cfg.b_has_scale()):
            b_scale_hbm_ptr = b_scale_hbm_ptr + KU * pc.s_step * pc.b_scale_stride_k
    return a_hbm_ptr, b_hbm_ptr, a_scale_hbm_ptr, b_scale_hbm_ptr


@gluon.jit
def _slot_a_read(
    pc,
    DS_READ_INX,
    mi: gl.constexpr,
    a_scale_read_hbm_ptr,
    RELAXED: gl.constexpr = False,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    return _a_block_frags(
        pc.lds,
        DS_READ_INX,
        mi,
        a_scale_read_hbm_ptr,
        _opt_at(pc.a_scale_offs, mi, pc.func_cfg.a_has_scale()),
        pc.func_cfg,
        pc.tuning_cfg,
        RELAXED,
        READ_PAYLOAD,
        READ_SCALE,
    )


@gluon.jit
def _b_rotate(carried, fresh, TWO_STAGE: gl.constexpr):
    """The set the *next* step consumes.

    With B in LDS the read is already NUM_LDS_BUFFER stages ahead of its use, so the
    set read this step is next step's. With B in registers there is no LDS to buffer it,
    so the distance is carried explicitly: this step's read is parked for one step and
    the one parked last step is promoted -- two K-stages of cover, matching FlyDSL's
    kStages = 2. Payload and scale sit in the same tuple and are read together, so they
    cannot drift apart.
    """
    if require_constexpr(TWO_STAGE):
        return carried
    return fresh


@gluon.jit
def _slot_b_read(
    pc,
    DS_READ_INX,
    ni: gl.constexpr,
    b_scale_read_hbm_ptr,
    RELAXED: gl.constexpr = False,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    return _b_block_frags(
        pc.lds,
        DS_READ_INX,
        ni,
        b_scale_read_hbm_ptr,
        _opt_at(pc.b_scale_offs, ni, pc.func_cfg.b_has_scale()),
        pc.func_cfg,
        pc.tuning_cfg,
        # B_IN_REG is unsupported for now (see the static_assert in _moe_gemm_body), so
        # the register-path base pointer is gone and this stays None.
        None,
        _opt_at(pc.b_frag_offs, ni, pc.tuning_cfg.B_IN_REG),
        RELAXED,
        READ_PAYLOAD,
        READ_SCALE,
    )


# --- per-slot constexpr binding -----------------------------------------------
# ``_read_a_tile`` / ``_read_b_tile`` / ``_relax_read`` are pure functions of the slot
# position, and the slot body wants each of them several times. They cannot be hoisted
# to a local at the top of the slot loop: ``X: gl.constexpr = ...`` there is rejected on
# the second unrolled iteration ("constexpr cannot be reassigned"), and an unannotated
# binding is materialised into a runtime ``tensor`` -- verified, even when the right-hand
# side is already a ``gl.constexpr``. A ``@gluon.jit`` body is a fresh scope, so the two
# helpers below are where the annotation is legal and each value is named once.
# ------------------------------------------------------------------------------
@gluon.jit
def _slot_reads(
    pc,
    DS_READ_INX,
    mi: gl.constexpr,
    ni: gl.constexpr,
    a_scale_read_hbm_ptr,
    b_scale_read_hbm_ptr,
    WANT_A: gl.constexpr,
    WANT_B: gl.constexpr,
    WANT_A_SCALE: gl.constexpr,
    WANT_B_SCALE: gl.constexpr,
):
    """The LDS reads slot ``(mi, ni)`` owns, as ``(a_frags, b_frags)`` to append.

    Payload and scale reads can belong to different stage groups. Missing components
    temporarily alias the present component; _merge_read_frags selects the real values.
    """
    NM: gl.constexpr = pc.tuning_cfg.num_mini_m()
    NN: gl.constexpr = pc.tuning_cfg.num_mini_n()
    A_TILE: gl.constexpr = _read_a_tile(mi, ni, NM, NN)
    B_TILE: gl.constexpr = _read_b_tile(mi, ni, NM, NN)
    RELAXED: gl.constexpr = _relax_read(mi, ni, NM, NN)
    a_frags = ()
    b_frags = ()
    if require_constexpr(
        (WANT_A or (WANT_A_SCALE and pc.func_cfg.a_has_scale())) and A_TILE is not None
    ):
        a_frags = _slot_a_read(
            pc, DS_READ_INX, A_TILE, a_scale_read_hbm_ptr, RELAXED, WANT_A, WANT_A_SCALE
        )
    if require_constexpr(
        (WANT_B or (WANT_B_SCALE and pc.func_cfg.b_has_scale())) and B_TILE is not None
    ):
        b_frags = _slot_b_read(
            pc, DS_READ_INX, B_TILE, b_scale_read_hbm_ptr, RELAXED, WANT_B, WANT_B_SCALE
        )
    return a_frags, b_frags


@gluon.jit
def _merge_read_frags(mem, mfma, tc, operand: gl.constexpr):
    if require_constexpr(len(mem) == 0):
        return mfma
    elif require_constexpr(len(mfma) == 0):
        return mem
    else:
        out = ()
        for i in gl.static_range(tc.num_mini_k()):
            if require_constexpr(tc.ds_read_in_mfma(operand)):
                payload = mfma[2 * i]
            else:
                payload = mem[2 * i]
            if require_constexpr(tc.ds_read_in_mfma(operand, scale=True)):
                scale = mfma[2 * i + 1]
            else:
                scale = mem[2 * i + 1]
            out = out + (payload, scale)
        return out


@gluon.jit
def _slot_tails(
    pc,
    a_cur,
    b_cur,
    mi: gl.constexpr,
    ni: gl.constexpr,
    DO_DS_READ: gl.constexpr,
):
    """The carried-fragment tail of what slot ``(mi, ni)`` just read.

    Indexed by the tile the slot actually read, not by ``(mi, ni)``: under the even
    schedule slot (0, 1) reads B(0), not B(1), so the legacy ``ni == 0`` / ``mi == 0``
    predicates would reach past the end of the half-built tuple.
    """
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_TILE: gl.constexpr = _read_a_tile(mi, ni, NM, NN)
    B_TILE: gl.constexpr = _read_b_tile(mi, ni, NM, NN)
    a_tail = ()
    b_tail = ()
    if require_constexpr(DO_DS_READ and A_TILE is not None):
        a_tail = _take_pairs(a_cur, A_TILE * NUM_MINI + HEAD_MINI, PF_MINI)
    if require_constexpr(DO_DS_READ and B_TILE is not None):
        b_tail = _take_pairs(b_cur, B_TILE * NUM_MINI + HEAD_MINI, PF_MINI)
    return a_tail, b_tail


@gluon.jit
def _pipeline_step(
    pc,
    st,
    BUFFER_LOAD_INX,
    DS_READ_INX,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_DS_READ: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
    DO_MFMA: gl.constexpr = True,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    WAIT_SLACK: gl.constexpr = 0,
):
    """Pick the live implementation or the frozen 2026-09-01 snapshot.

    ``_pipeline_step_impl`` is the one to edit; ``_pipeline_step_frozen`` is a verbatim
    copy of it from the best measured kernel. ``FROZEN_STEP=1`` runs the snapshot, so a
    refactor can be compared against the known-good schedule without a checkout.
    """
    if require_constexpr(pc.tuning_cfg.FROZEN_STEP):
        out = _pipeline_step_frozen(
            pc,
            st,
            BUFFER_LOAD_INX,
            DS_READ_INX,
            STAGES_BETWEEN,
            DO_BUFFER_LOAD,
            DO_DS_READ,
            IN_LOOP,
            DO_MFMA,
            KI,
            KU,
            WAIT_SLACK,
        )
    else:
        out = _pipeline_step_impl(
            pc,
            st,
            BUFFER_LOAD_INX,
            DS_READ_INX,
            STAGES_BETWEEN,
            DO_BUFFER_LOAD,
            DO_DS_READ,
            IN_LOOP,
            DO_MFMA,
            KI,
            KU,
            WAIT_SLACK,
        )
    return out


@gluon.jit
def _pipeline_step_impl(
    pc,
    st,
    BUFFER_LOAD_INX,
    DS_READ_INX,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_DS_READ: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
    DO_MFMA: gl.constexpr = True,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    WAIT_SLACK: gl.constexpr = 0,
):
    func_cfg: gl.constexpr = pc.func_cfg
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_HAS: gl.constexpr = func_cfg.a_has_scale()
    B_HAS: gl.constexpr = func_cfg.b_has_scale()
    gl.static_assert(
        HEAD_MINI == 0,
        "the K-loop step needs VGPR_PREFETCH_K == BLOCK_K (HEAD_MINI == 0): a split "
        "stage would have to dot the head it reads in the slot that reads it, and the "
        "slot's MFMAs would then depend on the slot's own ds_read",
    )
    gl.static_assert(
        not tc.warp_pipeline_manual(),
        "WarpPipeline.MANUAL is implemented only by _pipeline_step_frozen: run it with "
        "AITER_TRITON_MOE_GLUON_FROZEN_STEP=1, or pick NONE / COMPILER",
    )
    PIPE: gl.constexpr = tc.warp_pipeline_compiler() and DO_DS_READ and IN_LOOP
    STAGE: gl.constexpr = pick_stage(PIPE)

    READ_A_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(0)
    READ_B_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(1)
    READ_A_SCALE_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(0, scale=True)
    READ_B_SCALE_IN_MFMA: gl.constexpr = tc.ds_read_in_mfma(1, scale=True)

    SCHEME: gl.constexpr = tc.WAIT_COMMIT_SCHEME
    STAGE_WAIT: gl.constexpr = tc.wait_at_stage_head()

    KIE: gl.constexpr = KI if tc.SOFF_UNROLL else 0
    KUE: gl.constexpr = KU if tc.SOFF_UNROLL else 1

    a_hbm_ptr = st.a_hbm_ptr
    b_hbm_ptr = st.b_hbm_ptr
    a_scale_hbm_ptr = st.a_scale_hbm_ptr
    b_scale_hbm_ptr = st.b_scale_hbm_ptr
    a_scale_read_hbm_ptr = st.a_scale_read_hbm_ptr
    b_scale_read_hbm_ptr = st.b_scale_read_hbm_ptr

    a_cur = ()
    b_cur = ()
    a_tail = ()
    b_tail = ()
    acc = ()
    # Not under the warp pipeliner: it rejects anything it reads as a barrier or
    # wait inside a stage region, and the two are alternative answers to the same
    # question anyway -- iglp_opt overlaps MFMA with memory inside a wave, the
    # pipeliner does it by ping-ponging two wave groups.
    if require_constexpr(tc.SCHED_MODE != 0 and DO_DS_READ and not PIPE):
        _sched_hint(tc.SCHED_MODE)
    if require_constexpr(DO_DS_READ and STAGE_WAIT):
        _stage_wait_group(
            pc.lds,
            NM,
            NN,
            STAGES_BETWEEN,
            DO_BUFFER_LOAD,
            SCHEME,
            WAIT_SLACK,
            tc.SCALE_FILL_MID,
        )
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            if require_constexpr(DO_DS_READ and not STAGE_WAIT):
                _slot_wait_group(
                    pc.lds,
                    mi,
                    ni,
                    NM,
                    NN,
                    STAGES_BETWEEN,
                    DO_BUFFER_LOAD,
                    SCHEME,
                    WAIT_SLACK,
                    tc.SCALE_FILL_MID,
                )

            if require_constexpr(DO_MFMA):
                dot_a = _take_pairs(st.a_frags, mi * PF_MINI, PF_MINI)
                dot_b = _take_pairs(st.b_frags, ni * PF_MINI, PF_MINI)
            else:
                dot_a = ()
                dot_b = ()

            with STAGE("mem"):
                a_mem, b_mem = _slot_reads(
                    pc,
                    DS_READ_INX,
                    mi,
                    ni,
                    a_scale_read_hbm_ptr,
                    b_scale_read_hbm_ptr,
                    DO_DS_READ and not READ_A_IN_MFMA,
                    DO_DS_READ and not READ_B_IN_MFMA,
                    DO_DS_READ and not READ_A_SCALE_IN_MFMA,
                    DO_DS_READ and not READ_B_SCALE_IN_MFMA,
                )
                if require_constexpr(DO_BUFFER_LOAD):
                    (
                        a_hbm_ptr,
                        b_hbm_ptr,
                        a_scale_hbm_ptr,
                        b_scale_hbm_ptr,
                    ) = _slot_fills(
                        pc,
                        BUFFER_LOAD_INX,
                        mi,
                        ni,
                        a_hbm_ptr,
                        b_hbm_ptr,
                        a_scale_hbm_ptr,
                        b_scale_hbm_ptr,
                        ADVANCE=_slot_advance(mi, ni, NM, NN, KIE, KUE),
                        KI=KIE,
                        KU=KUE,
                        # the step closes the stage group itself, after the walk
                        STAGE_MARK=False,
                    )
                    if require_constexpr(
                        PIPE and tc.commit_per_stage() and mi == NM - 1 and ni == NN - 1
                    ):
                        # Close the copy group before the border, ahead of the next wait.
                        pc.lds.commit_fill_lds()

            with STAGE("mfma"):
                slot_acc = _maybe_block_dot(
                    dot_a,
                    dot_b,
                    st.acc[_slot_index(mi, ni, NM, NN)],
                    PF_MINI,
                    func_cfg,
                    tc,
                    DO_MFMA,
                )
                a_mfma, b_mfma = _slot_reads(
                    pc,
                    DS_READ_INX,
                    mi,
                    ni,
                    a_scale_read_hbm_ptr,
                    b_scale_read_hbm_ptr,
                    DO_DS_READ and READ_A_IN_MFMA,
                    DO_DS_READ and READ_B_IN_MFMA,
                    DO_DS_READ and READ_A_SCALE_IN_MFMA,
                    DO_DS_READ and READ_B_SCALE_IN_MFMA,
                )
            a_cur = a_cur + _merge_read_frags(a_mem, a_mfma, tc, 0)
            b_cur = b_cur + _merge_read_frags(b_mem, b_mfma, tc, 1)
            acc = acc + (slot_acc,)

            a_new, b_new = _slot_tails(pc, a_cur, b_cur, mi, ni, DO_DS_READ)
            a_tail = a_tail + a_new
            b_tail = b_tail + b_new

    if require_constexpr(tc.commit_per_stage() and DO_BUFFER_LOAD and not PIPE):
        # One group for the whole stage, closed after the walk -- see _slot_fills.
        pc.lds.commit_fill_lds()

    if require_constexpr(DO_DS_READ):
        if require_constexpr(A_HAS):
            a_scale_read_hbm_ptr = (
                a_scale_read_hbm_ptr + pc.s_step * pc.a_scale_stride_k
            )
        if require_constexpr(B_HAS):
            b_scale_read_hbm_ptr = (
                b_scale_read_hbm_ptr + pc.s_step * pc.b_scale_stride_k
            )
    else:
        a_tail = st.a_frags
        b_tail = st.b_frags

    return _PipelineState(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
        a_scale_read_hbm_ptr,
        b_scale_read_hbm_ptr,
        a_tail,
        _b_rotate(st.b_frags2, b_tail, pc.tuning_cfg.B_IN_REG),
        b_tail,
        acc,
    )


@gluon.jit
def _drain_last_fused(pc, st, bias_tiles, x_static_scale, func_cfg, tuning_cfg):
    """The final drain step, with the gate half's activation folded into it.

    That step reads nothing, fills nothing and waits on nothing -- it only retires the
    MFMAs whose operands are already in registers -- so every other branch of
    :func:`_pipeline_step_impl` would be dead here and the walk collapses to
    :func:`_maybe_block_dot`. The preconditions are asserted rather than assumed.

    :func:`_slot_index` is N outer, so the visit order is ``(0,0), (1,0), (0,1), (1,1)``
    and every gate-side accumulator is final before the first linear-side MFMA issues.
    The silu for mini-M block ``mi`` therefore lands *between* MFMA clusters and its
    exp2/rcp retire under them.

    Returns the accumulator tuple with each ``ni == 0`` slot replaced by
    ``silu(bias + acc)`` -- the same value :func:`_epilogue_one_tile` would have
    computed, which is why it takes ``GATE_PRE`` to skip recomputing it.
    """
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    act: gl.constexpr = func_cfg.act()
    gl.static_assert(func_cfg.gu_split() and NN == 2)

    acc = ()
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            slot_acc = _maybe_block_dot(
                _take_pairs(st.a_frags, mi * PF_MINI, PF_MINI),
                _take_pairs(st.b_frags, ni * PF_MINI, PF_MINI),
                st.acc[_slot_index(mi, ni, NM, NN)],
                PF_MINI,
                func_cfg,
                tc,
                True,
            )
            if require_constexpr(ni == 0):
                # Gate side: everything up to and including the reciprocal, issued here
                # so it overlaps the ni == 1 clusters still to come.
                if require_constexpr(func_cfg.has_x_static_scale):
                    slot_acc = slot_acc * x_static_scale
                if require_constexpr(func_cfg.has_bias and func_cfg.epilogue < 2):
                    slot_acc = slot_acc + bias_tiles[0][None, :]
                if require_constexpr(func_cfg.epilogue == 0):
                    slot_acc = _swiglu_gate(
                        slot_acc, act.alpha, act.limit, tc.ACT_FAST_RCP
                    )
            acc = acc + (slot_acc,)
    return acc


@gluon.jit
def _epi_bias_tiles(bias_smem, bias_ptr, pid_n, N, func_cfg, tuning_cfg):
    """Block-level bias, one tensor per mini-N block.

    Invariant in mi, so NN of them cover the block where the per-tile form issued
    NM*NN. From LDS when the staging constraints hold, else straight from global.
    The LDS read carries the accumulator's own N slice layout, which is nameable, so
    this really is a prefetch -- unlike gammas, see _epi_gamma_tiles.
    """
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NN: gl.constexpr = BN // MBN
    RFL: gl.constexpr = tuning_cfg.dot_result_fragment_layout()
    out = ()
    if require_constexpr(func_cfg.has_bias):
        for hn in gl.static_range(NN):
            if require_constexpr(bias_smem is not None):
                out = out + (
                    bias_smem.slice(hn * MBN, MBN).load(gl.SliceLayout(0, RFL)),
                )
            else:
                out = out + (
                    gl.load(
                        bias_ptr
                        + _n_start(pid_n, hn, N, func_cfg, tuning_cfg)
                        + gl.arange(0, MBN)
                    ),
                )
    return out


@gluon.jit
def _epi_gamma_tiles(gamma_smem, gammas_base, block_id, M_e, func_cfg, tuning_cfg):
    """Block-level gammas, one tensor per mini-M block -- global path only.

    When gammas live in LDS this returns empty on purpose: gammas multiplies the
    *post-swiglu* tensor, whose layout is a sliced linear layout that swiglu's N
    reduction produces and no config accessor names, so the read has to happen inside
    _epilogue_one_tile off `out.type.layout`. That is a ds_read of MINI_BLOCK_M
    elements against data that landed long before, not a global load.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    NM: gl.constexpr = BM // MBM
    out = ()
    if require_constexpr(func_cfg.has_gammas and gamma_smem is None):
        for hm in gl.static_range(NM):
            hoffs = BM * block_id + hm * MBM + gl.arange(0, MBM)
            out = out + (gl.load(gammas_base + hoffs, mask=hoffs < M_e, other=0.0),)
    return out


@gluon.constexpr_function
def _amax_lane_elems(MBM, OUT_MBN, func_cfg, tuning_cfg):
    """Inner width of the split MXFP4 amax reduction.

    The amax runs as an fp32 reduce over the inner axis followed by an integer reduce
    over the outer one, so that the fp32 half keeps abs free as an |v| source modifier
    while the integer half needs no NaN canonicalisation (``tl.max`` on f32 emits a
    ``v_max_f32 x, x, x`` in front of every cross-lane step).

    The width has to respect the layout, and this expression is a *proxy* for it, not a
    derivation. What matters is which bits of the reduction axis are register-resident:
    from the TTGIR for the tuned 4-wave config the post-swiglu tensor reshapes to
    ``tensor<64x2x2x16xf32, #linear1>`` with, along the reduction axis, register bases
    {1, 8} and lane bases {2, 4} -- so a lane owns {t, t+1, t+8, t+9} and the inner axis
    must be at least 16 to contain that span. This returns 16 there, which is right, but
    by arithmetic coincidence rather than by reading the layout.

    A mismatched width is not incorrect, only slower, and it can go wrong in two ways:
    too coarse and it cuts across lane bits, adding permlane steps (4 and 8 both measured
    worse, 2756 / 2782 instructions against 2492); too fine and the integer tree loses
    the 3-input v_max3 fusion and pays for explicit copies around the destructive
    permlane swaps (2 removes every canonicalisation yet costs 112 extra v_mov, 2576).

    Under ``gate_up_split`` the layout IS readable, so this stops guessing: the tile is
    one accumulator wide, so along the emitted axis a lane owns exactly the transposed
    MFMA quad -- ``instr_m * instr_n / WARP_SIZE`` consecutive columns, 4 at 16x16 --
    before the first lane base. Splitting there puts the whole cross-lane part of the
    tree in the integer half, which is what this function wants and what the
    interleaved packing cannot give it (its quad holds two gate and two linear values,
    so only 2 of the 4 survive the reduction).
    """
    if _v(func_cfg.gu_split()):
        instr = _v(tuning_cfg.mfma_instr_shape)
        return max(1, min(_v(MX_GROUP), (instr[0] * instr[1]) // _v(WARP_SIZE)))
    per_lane = (_v(MBM) * _v(OUT_MBN)) // (
        _v(tuning_cfg.num_warps()) * _v(WARP_SIZE)
    )
    return min(_v(MX_GROUP), max(1, per_lane))


@gluon.jit
def _epi_stage_flush(
    qp_smem,
    qs_smem,
    mi: gl.constexpr,
    ni: gl.constexpr,
    y_ptr,
    y_stride_m,
    y_stride_n,
    ys_ptr,
    ys_stride_m,
    ys_stride_n,
    block_id,
    pid_n,
    M_e,
    func_cfg,
    tuning_cfg,
):
    """Read one staged MXFP4 mini tile back out of LDS and store it.

    The second half of :func:`_epilogue_one_tile`'s staged store, split out so
    ``rotating epilogue`` can run it one iteration behind the write that produced it.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_MBN: gl.constexpr = MBN // ARN
    PL: gl.constexpr = tuning_cfg.result_store_layout(MBM, OUT_MBN // 2, 8)
    SL: gl.constexpr = tuning_cfg.result_store_layout(MBM, OUT_MBN // MX_GROUP, 8)
    raw_n0 = pid_n * BN + ni * MBN
    n0 = raw_n0 // ARN
    # The caller's fenced barrier has retired every wave's staging writes.
    pv = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_smem, PL)
    sv = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_smem, SL)
    pm = BM * block_id + mi * MBM + gl.arange(0, MBM, layout=gl.SliceLayout(1, PL))
    pn = gl.arange(0, OUT_MBN // 2, layout=gl.SliceLayout(0, PL))
    gl.amd.cdna4.buffer_store(
        pv,
        y_ptr,
        pm[:, None] * y_stride_m + (n0 // 2 + pn)[None, :] * y_stride_n,
        mask=(pm < M_e)[:, None],
    )
    sm = BM * block_id + mi * MBM + gl.arange(0, MBM, layout=gl.SliceLayout(1, SL))
    sn = gl.arange(0, OUT_MBN // MX_GROUP, layout=gl.SliceLayout(0, SL))
    gl.amd.cdna4.buffer_store(
        sv,
        ys_ptr,
        sm[:, None] * ys_stride_m + (n0 // MX_GROUP + sn)[None, :] * ys_stride_n,
        mask=(sm < M_e)[:, None],
    )


@gluon.jit
def _epi_block_flush(
    qp_smem,
    qs_smem,
    y_ptr,
    y_stride_m,
    y_stride_n,
    ys_ptr,
    ys_stride_m,
    ys_stride_n,
    block_id,
    pid_n,
    M_e,
    func_cfg,
    tuning_cfg,
):
    """Read the WHOLE staged MXFP4 block back out of LDS and store it.

    The gate/up-split twin of :func:`_epi_stage_flush`. Split collapses the two mini-N
    blocks into one output tile, so a mini tile already spans the CTA's full emitted
    width and the only axis left to walk is M -- which means every tile can stage into
    its own rows of one block-height buffer and the drain needs a single barrier and a
    single pair of stores rather than one per tile.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_MBN: gl.constexpr = MBN // func_cfg.mini_n_reduction()
    PL: gl.constexpr = tuning_cfg.result_store_layout(BM, OUT_MBN // 2, 8)
    SL: gl.constexpr = tuning_cfg.result_store_layout(BM, OUT_MBN // MX_GROUP, 8)
    n0 = pid_n * (BN // ARN)
    # The staging writes are spread over every warp and each warp reads rows it did not
    # write, so this fence is load-bearing. Only one is needed: nothing reuses the
    # buffer afterwards.
    gl.barrier()
    pv = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_smem, PL)
    sv = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_smem, SL)
    pm = BM * block_id + gl.arange(0, BM, layout=gl.SliceLayout(1, PL))
    pn = gl.arange(0, OUT_MBN // 2, layout=gl.SliceLayout(0, PL))
    gl.amd.cdna4.buffer_store(
        pv,
        y_ptr,
        pm[:, None] * y_stride_m + (n0 // 2 + pn)[None, :] * y_stride_n,
        mask=(pm < M_e)[:, None],
    )
    sm = BM * block_id + gl.arange(0, BM, layout=gl.SliceLayout(1, SL))
    sn = gl.arange(0, OUT_MBN // MX_GROUP, layout=gl.SliceLayout(0, SL))
    gl.amd.cdna4.buffer_store(
        sv,
        ys_ptr,
        sm[:, None] * ys_stride_m + (n0 // MX_GROUP + sn)[None, :] * ys_stride_n,
        mask=(sm < M_e)[:, None],
    )


@gluon.jit
def _epilogue_one_tile(
    sub,
    mi: gl.constexpr,
    ni: gl.constexpr,
    bias_tiles,
    gamma_tiles,
    gamma_smem,
    bias_ptr,
    gammas_base,
    y_ptr,
    y_stride_m,
    y_stride_n,
    ys_ptr,
    ys_stride_m,
    ys_stride_n,
    block_id,
    pid_n,
    M_e,
    x_static_scale,
    func_cfg,
    tuning_cfg,
    qp_smem=None,
    qs_smem=None,
    STAGE_ONLY: gl.constexpr = False,
    lin=None,
    STAGE_ROW: gl.constexpr = 0,
    GATE_PRE: gl.constexpr = False,
):
    """bias -> activation -> gammas -> output_quant -> store, for ONE mini tile.

    Lifted verbatim out of :func:`_epilogue_store` so the fused drain step can call it
    per slot the moment that slot's last MFMA retires. ``sub`` is the finished
    accumulator for tile ``(mi, ni)``.

    Under ``gate_up_split`` a "tile" is a *pair*: ``sub`` is the gate side (mini-N block
    ``ni``) and ``lin`` the linear side (block ``ni + 1``), covering the same emitted
    channels. The activation is then a plain elementwise op between two tensors of
    identical layout -- no reshape, no split, and none of the ``v_mov_b32`` gathers the
    interleaved packing needs -- and the tile is ``MBN`` emitted channels wide rather
    than ``MBN // 2``. ``STAGE_ROW`` is the row this tile occupies in a block-height LDS
    staging buffer, which is what lets the caller flush all of them in one store.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_MBN: gl.constexpr = MBN // func_cfg.mini_n_reduction()
    act: gl.constexpr = func_cfg.act()
    out_ty: gl.constexpr = y_ptr.dtype.element_ty
    store_layout: gl.constexpr = tuning_cfg.result_store_layout(
        MBM, OUT_MBN, out_ty.primitive_bitwidth
    )

    # GATE_PRE: the peeled last K step already applied the static scale, the bias and
    # the silu to `sub`, so those are skipped for the gate side only -- `lin` still
    # needs both, and the result is the same expression either way.
    if require_constexpr(func_cfg.has_x_static_scale):
        # Per-tensor fp8 activation scale. FP8_E4M3 operands go through the
        # MMA with unit scales, so the tensor scale comes back here -- before
        # bias, matching the Triton kernels exactly.
        if require_constexpr(not GATE_PRE):
            sub = sub * x_static_scale
        if require_constexpr(func_cfg.gu_split()):
            lin = lin * x_static_scale

    if require_constexpr(func_cfg.has_bias and func_cfg.epilogue < 2):
        # fp32, expert indexed, over the RAW N axis (before the halving). Split: one
        # tile per side, so each accumulator takes its own.
        if require_constexpr(not GATE_PRE):
            sub = sub + bias_tiles[ni][None, :]
        if require_constexpr(func_cfg.gu_split()):
            lin = lin + bias_tiles[ni + 1][None, :]

    if require_constexpr(func_cfg.gu_split() and func_cfg.epilogue != 0):
        # Ablation: the same [MBM, MBN] x 2 -> [MBM, MBN] reduction, minus exp2/rcp.
        out = sub * lin
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(GATE_PRE):
        out = _swiglu_combine(sub, lin, act.limit, ADD_RESIDUAL=act.add_residual)
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(func_cfg.gu_split()):
        out = _swiglu_pair(
            sub,
            lin,
            act.alpha,
            act.limit,
            ADD_RESIDUAL=act.add_residual,
            FAST_RCP=tuning_cfg.ACT_FAST_RCP,
        )
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(func_cfg.has_activation() and func_cfg.epilogue != 0):
        # Ablation: the same reshape/split reduction _swiglu does, minus exp2/rcp/clip.
        gelu, linear = tl.split(
            tl.reshape(sub, (sub.shape[0], sub.shape[1] // 2, 2))
        )
        out = gelu * linear
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(func_cfg.has_activation()):
        out = _swiglu(
            sub,
            act.alpha,
            act.limit,
            ADD_RESIDUAL=act.add_residual,
            FAST_RCP=tuning_cfg.ACT_FAST_RCP,
            GROUP=3,
        )
        gl.static_assert(out.shape[1] == OUT_MBN)
    else:
        gl.static_assert(ARN == 1)
        out = sub

    offs_m = BM * block_id + mi * MBM + gl.arange(0, MBM)
    mask_m = offs_m < M_e
    if require_constexpr(func_cfg.has_gammas and func_cfg.epilogue < 2):
        if require_constexpr(gamma_smem is not None):
            # Only a ds_read: the global copy that filled this ran before the K
            # loop. `out.type.layout` is the one way to name the post-swiglu
            # layout, so the read has to sit here rather than above the walk.
            g = gamma_smem.slice(mi * MBM, MBM).load(
                gl.SliceLayout(1, out.type.layout)
            )
        else:
            g = gamma_tiles[mi]
        out = out * g[:, None]

    # Emitted-channel base. The CTA tile is BLOCK_N // ARN emitted channels wide in both
    # packings -- what differs is only how many mini blocks that is (2 interleaved, 1
    # split, where ni is always the gate block and OUT_MBN is the whole width).
    out_n0 = pid_n * (tuning_cfg.BLOCK_N // ARN) + ni * OUT_MBN
    if require_constexpr(func_cfg.output_quant is None):
        val = gl.convert_layout(
            out.to(out_ty), store_layout, assert_trivial=False
        )
        sm = gl.arange(0, MBM, layout=gl.SliceLayout(1, store_layout))
        sn = gl.arange(0, OUT_MBN, layout=gl.SliceLayout(0, store_layout))
        rows = BM * block_id + mi * MBM + sm
        gl.amd.cdna4.buffer_store(
            val,
            y_ptr,
            rows[:, None] * y_stride_m + (out_n0 + sn)[None, :] * y_stride_n,
            mask=(rows < M_e)[:, None],
            cache=tuning_cfg.result_mod,
        )
    else:
        # Fused MXFP4 output quant, straight off the fp32 accumulator.
        #
        # The unfused flow this replaces writes gemm1's result to HBM as bf16 and
        # mxfp4_quant reads it back (widening to fp32 on the host), so its input has
        # only 8 mantissa bits. Quantizing the live accumulator instead keeps all 24,
        # which is strictly more accurate but NOT bit-identical to that flow: values
        # near an E2M1 or E8M0 rounding boundary can land on the other side. See
        # test_moe_gemm_a4w4's fused-output test for the reference this is compared
        # against.
        #
        # The MX group runs along the emitted N axis, so the amax is over 32 emitted
        # columns -- with the transposed MFMA accumulator each lane already owns 4
        # consecutive N, and the reduction over the remaining 8 lanes is what the
        # reshape below expresses.
        #
        # Interleaved, a warp owns only 16 of those columns (its 32 raw ones halve), so
        # the group spans two warps and the outer reduction crosses the warp boundary.
        # Split, a warp owns 32 emitted columns -- exactly one group -- and validate()
        # enforces it, so the whole reduction stays inside the wave.
        gl.static_assert(func_cfg.output_quant == _DQ_MXFP4)
        gl.static_assert(OUT_MBN % MX_GROUP == 0)
        # LANE_ELEMS splits the amax: how many of the tile's elements one lane owns,
        # so the inner reduction is in-lane fp32 (free |v| modifiers) and only the
        # outer one goes cross-lane. Getting it wrong is not incorrect, just slower --
        # a mismatched split adds permlane steps instead of removing them.
        LANE_ELEMS: gl.constexpr = _amax_lane_elems(
            MBM, OUT_MBN, func_cfg, tuning_cfg
        )
        payload, scale = mxfp4_quant_gluon(
            out, OUT_MBN, MBM, MX_GROUP, LANE_ELEMS
        )
        # Fresh M ranges per store: reusing one auto-layout `offs_m` across two
        # differently-laid-out stores makes GluonResolveAutoEncodings fail with
        # "conflicting encodings" on the expand_dims.
        if require_constexpr(qp_smem is not None):
            # LDS round-trip: write in the accumulator's layout, read back in the
            # store layout.
            #
            # The MFMA D layout gives each lane 4 consecutive M rows in ONE N column,
            # but MXFP4 packs 2 values per byte along N, so storing straight from the
            # accumulator leaves each lane owning a single byte at a strided address --
            # 48 x global_store_byte with 64-bit address arithmetic each.
            # result_store_layout gives size_per_thread=[1, 16] for a uint8 payload, so
            # after the bounce each lane holds 16 contiguous bytes and the store lowers
            # to 4 x buffer_store_dwordx4.
            #
            # Staging the *packed* payload, not the fp32 tile: the fp32 form makes the
            # LDS write dword-granular too (48 ds_write_b8 -> ds_write_b64), but it is
            # 8x the LDS bytes and needs two more barriers per tile, and it measured
            # 15 us WORSE. Global byte stores are what hurt; LDS byte writes are cheap.
            #
            # There is no LDS->global direct store to use instead: BUFFER_STORE_LDS_DWORD
            # is gated isGFX8GFX9NotGFX940 in LLVM, i.e. it does not exist on gfx950.
            #
            # Two barriers per tile because one buffer serves all NM*NN tiles: one so
            # every lane's write is visible before the reads, one so the reads finish
            # before the next tile overwrites the buffer.
            PL: gl.constexpr = tuning_cfg.result_store_layout(MBM, OUT_MBN // 2, 8)
            SL: gl.constexpr = tuning_cfg.result_store_layout(
                MBM, OUT_MBN // MX_GROUP, 8
            )
            # if/elif/else, not early returns: a `return` from inside a nested
            # `if require_constexpr(...)` here does NOT stop the trace, so the code
            # below it still ran and stored the mini tile to the *unsliced* buffer --
            # which only surfaced as a shape mismatch because the split buffer is
            # block-height. Keep exactly one store path reachable per configuration.
            if require_constexpr(func_cfg.gu_split()):
                # Block-height staging: every mi writes its own MBM rows of one buffer,
                # so the caller flushes all NM tiles with a single barrier and a single
                # store instead of a barrier pair per tile. Split halves the tile count
                # too (each covers both sides), so the drain goes from NM*NN staging
                # round-trips to one.
                qp_smem.slice(STAGE_ROW, MBM).store(payload)
                qs_smem.slice(STAGE_ROW, MBM).store(scale)
            elif require_constexpr(STAGE_ONLY):
                # rotating epilogue: stage and leave it; the caller flushes this buffer one
                # iteration later, after its own barrier.
                qp_smem.store(payload)
                qs_smem.store(scale)
            else:
                gl.barrier()
                qp_smem.store(payload)
                qs_smem.store(scale)
                # Relaxed reads still require cross-wave staging-write visibility.
                gl.barrier()
                payload_v = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_smem, PL)
                scale_v = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_smem, SL)

                pm = BM * block_id + mi * MBM + gl.arange(
                    0, MBM, layout=gl.SliceLayout(1, PL)
                )
                pn = gl.arange(0, OUT_MBN // 2, layout=gl.SliceLayout(0, PL))
                gl.amd.cdna4.buffer_store(
                    payload_v,
                    y_ptr,
                    pm[:, None] * y_stride_m
                    + (out_n0 // 2 + pn)[None, :] * y_stride_n,
                    mask=(pm < M_e)[:, None],
                )
                sm = BM * block_id + mi * MBM + gl.arange(
                    0, MBM, layout=gl.SliceLayout(1, SL)
                )
                sn = gl.arange(0, OUT_MBN // MX_GROUP, layout=gl.SliceLayout(0, SL))
                gl.amd.cdna4.buffer_store(
                    scale_v,
                    ys_ptr,
                    sm[:, None] * ys_stride_m
                    + (out_n0 // MX_GROUP + sn)[None, :] * ys_stride_n,
                    mask=(sm < M_e)[:, None],
                )
        else:
            pm = BM * block_id + mi * MBM + gl.arange(0, MBM)
            gl.store(
                y_ptr
                + pm[:, None] * y_stride_m
                + (out_n0 // 2 + gl.arange(0, OUT_MBN // 2))[None, :] * y_stride_n,
                payload,
                mask=(pm < M_e)[:, None],
            )
            sm = BM * block_id + mi * MBM + gl.arange(0, MBM)
            gl.store(
                ys_ptr
                + sm[:, None] * ys_stride_m
                + (out_n0 // MX_GROUP + gl.arange(0, OUT_MBN // MX_GROUP))[None, :]
                * ys_stride_n,
                scale,
                mask=(sm < M_e)[:, None],
            )


@gluon.jit
def _epilogue_store(
    acc,
    y_ptr,
    y_stride_m,
    y_stride_n,
    ys_ptr,
    ys_stride_m,
    ys_stride_n,
    bias_ptr,
    block_id,
    pid_n,
    N,
    M_e,
    gammas_base,
    gamma_smem,
    bias_smem,
    x_static_scale,
    func_cfg,
    tuning_cfg,
    GATE_PRE: gl.constexpr = False,
):
    """bias -> activation -> gammas -> output_quant -> store, per mini (M, N) tile.

    The order is fixed and matches ``_triton_kernels/moe/moe_op_gemm_a4w4.py``; the
    clamp and the multiply happen in fp32 before any cast. ``acc`` arrives already split
    into one tensor per mini tile -- the K pipeline accumulates that way -- so the
    epilogue walks the same tiling and consumes them one at a time, which lets the
    compiler kill each mini accumulator before the next tile's epilogue.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_MBN: gl.constexpr = MBN // func_cfg.mini_n_reduction()
    act: gl.constexpr = func_cfg.act()
    # Hoisted: a `x: gl.constexpr = ...` inside a static_range would be a reassignment
    # on the second unrolled iteration, which Gluon rejects outright.
    out_ty: gl.constexpr = y_ptr.dtype.element_ty
    store_layout: gl.constexpr = tuning_cfg.result_store_layout(
        MBM, OUT_MBN, out_ty.primitive_bitwidth
    )

    NN: gl.constexpr = BN // MBN
    NM: gl.constexpr = BM // MBM

    # Block-level operand loads. Each is invariant in one of the two
    # walk indices -- bias in mi, gammas in ni -- so the per-tile form below issues
    # NM*NN of them where NN (resp. NM) is enough. Own loop variables: reusing `mi`/`ni`
    # here would shadow the walk's.
    # Both are staged in LDS (filled before the K loop, see _moe_gemm_body);
    # the read then carries the accumulator's own M/N slice layout, so nothing downstream
    # changes. Otherwise the block-level load still comes from global.
    bias_tiles = _epi_bias_tiles(
        bias_smem, bias_ptr, pid_n, N, func_cfg, tuning_cfg
    )
    gamma_tiles = _epi_gamma_tiles(
        gamma_smem, gammas_base, block_id, M_e, func_cfg, tuning_cfg
    )

    # Staging for the MXFP4 payload store, bouncing it through LDS so the global store
    # vectorises -- worth ~16 us on the 4-wave a4w4 gemm1, see _epilogue_one_tile.
    #
    # ONE mini tile's worth, reused across the walk: NM*NN buffers would cost NM*NN
    # times the LDS for no gain, since the tiles are stored one after another anyway.
    # Allocated here so it dominates every use; the pipeline's buffers are dead by this
    # point, so Triton's liveness-based shared allocator overlays this on top of them
    # and the kernel's LDS footprint does not grow at all.
    QSH: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    ROT: gl.constexpr = func_cfg.output_quant is not None and not func_cfg.gu_split()
    # Split stages the whole block, not one mini tile: NM tiles into one buffer, flushed
    # once at the end of the walk. Costs NM x the LDS (still a few KB, overlaid on the
    # dead pipeline buffers) and buys NM-1 barrier pairs and one wide store instead of
    # NM narrow ones.
    QROWS: gl.constexpr = BM if func_cfg.gu_split() else MBM
    if require_constexpr(func_cfg.output_quant is not None):
        qp_smem = gl.allocate_shared_memory(gl.uint8, [QROWS, OUT_MBN // 2], layout=QSH)
        qs_smem = gl.allocate_shared_memory(
            gl.uint8, [QROWS, OUT_MBN // MX_GROUP], layout=QSH
        )
    else:
        qp_smem: gl.constexpr = None
        qs_smem: gl.constexpr = None
    if require_constexpr(ROT):
        # Second bank for rotating epilogue. Same overlay argument as above, so the footprint
        # still does not grow.
        qp_smem2 = gl.allocate_shared_memory(gl.uint8, [MBM, OUT_MBN // 2], layout=QSH)
        qs_smem2 = gl.allocate_shared_memory(
            gl.uint8, [MBM, OUT_MBN // MX_GROUP], layout=QSH
        )
    else:
        qp_smem2: gl.constexpr = None
        qs_smem2: gl.constexpr = None

    if require_constexpr(func_cfg.gu_split()):
        # One pass over mi, pairing the two mini-N blocks: block 0 is the gate side and
        # block 1 the linear side of the *same* emitted channels, so the activation is
        # elementwise between two identically laid out accumulators. Every tile stages
        # into its own MBM rows of the block-height buffer, then one barrier pair and
        # one store flushes the lot.
        for mi in gl.static_range(NM):
            _epilogue_one_tile(
                acc[_slot_index(mi, 0, NM, NN)],
                mi,
                0,
                bias_tiles,
                gamma_tiles,
                gamma_smem,
                bias_ptr,
                gammas_base,
                y_ptr,
                y_stride_m,
                y_stride_n,
                ys_ptr,
                ys_stride_m,
                ys_stride_n,
                block_id,
                pid_n,
                M_e,
                x_static_scale,
                func_cfg,
                tuning_cfg,
                qp_smem,
                qs_smem,
                lin=acc[_slot_index(mi, 1, NM, NN)],
                STAGE_ROW=mi * MBM,
                GATE_PRE=GATE_PRE,
            )
        if require_constexpr(func_cfg.output_quant is not None):
            _epi_block_flush(
                qp_smem, qs_smem,
                y_ptr, y_stride_m, y_stride_n,
                ys_ptr, ys_stride_m, ys_stride_n,
                block_id, pid_n, M_e, func_cfg, tuning_cfg,
            )
    elif require_constexpr(ROT):
        # Software-pipelined staging: flush the PREVIOUS tile, then stage this one,
        # then one barrier. The read therefore consumes data written a full iteration
        # ago with a barrier already between them, and the two banks alternate so the
        # write never targets the buffer being read. One s_barrier per tile instead of
        # two, and no `s_waitcnt lgkmcnt(0)` sitting on the write it depends on.
        # No `mi: gl.constexpr = ...` locals here: Gluon rejects rebinding a constexpr
        # on the second unrolled iteration, so the indices are inlined at each use.
        for k in gl.static_range(NM * NN):
            if require_constexpr(k > 0):
                _epi_stage_flush(
                    qp_smem2 if (k - 1) % 2 else qp_smem,
                    qs_smem2 if (k - 1) % 2 else qs_smem,
                    (k - 1) // NN, (k - 1) % NN,
                    y_ptr, y_stride_m, y_stride_n,
                    ys_ptr, ys_stride_m, ys_stride_n,
                    block_id, pid_n, M_e, func_cfg, tuning_cfg,
                )
            _epilogue_one_tile(
                acc[_slot_index(k // NN, k % NN, NM, NN)],
                k // NN, k % NN,
                bias_tiles, gamma_tiles, gamma_smem, bias_ptr, gammas_base,
                y_ptr, y_stride_m, y_stride_n, ys_ptr, ys_stride_m, ys_stride_n,
                block_id, pid_n, M_e, x_static_scale, func_cfg, tuning_cfg,
                qp_smem2 if k % 2 else qp_smem,
                qs_smem2 if k % 2 else qs_smem,
                STAGE_ONLY=True,
            )
            gl.barrier()
        LK: gl.constexpr = NM * NN - 1
        _epi_stage_flush(
            qp_smem2 if LK % 2 else qp_smem,
            qs_smem2 if LK % 2 else qs_smem,
            LK // NN, LK % NN, y_ptr, y_stride_m, y_stride_n,
            ys_ptr, ys_stride_m, ys_stride_n,
            block_id, pid_n, M_e, func_cfg, tuning_cfg,
        )
    else:
        for mi in gl.static_range(NM):
            for ni in gl.static_range(NN):
                # acc is stored in slot-traversal order, which _slot_index defines; the
                # store loop is free to walk the grid however it likes as long as it
                # indexes through the same map.
                _epilogue_one_tile(
                    acc[_slot_index(mi, ni, NM, NN)],
                    mi,
                    ni,
                    bias_tiles,
                    gamma_tiles,
                    gamma_smem,
                    bias_ptr,
                    gammas_base,
                    y_ptr,
                    y_stride_m,
                    y_stride_n,
                    ys_ptr,
                    ys_stride_m,
                    ys_stride_n,
                    block_id,
                    pid_n,
                    M_e,
                    x_static_scale,
                    func_cfg,
                    tuning_cfg,
                    qp_smem,
                    qs_smem,
                )


@gluon.jit
def _moe_gemm_body(
    a,  # QuantTokenTensor
    b,  # QuantExpertTensor
    res,  # ResultTensor
    rt,  # RoutingMeta
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    cfg,  # MoeKernelConfig
):
    func_cfg, tuning_cfg = _build_configs(cfg)
    gl.static_assert(tuning_cfg.validate(cfg.N, cfg.K))

    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    BK: gl.constexpr = tuning_cfg.BLOCK_K
    NB: gl.constexpr = tuning_cfg.NUM_LDS_BUFFER
    PK_A: gl.constexpr = BK // func_cfg.a_pack_divisor()
    PK_B: gl.constexpr = BK // func_cfg.b_pack_divisor()
    SK: gl.constexpr = BK // MX_GROUP
    NUM_K: gl.constexpr = tuning_cfg.num_k_tiles(cfg.K)

    # One offset grid per mini block. Each is built at *its own* mini tile's copy
    # layout, so the split into MINI_BLOCK_M rows / MINI_BLOCK_N columns costs no
    # coalescing and no vector width -- the tile it addresses is a whole LDS allocation,
    # not a strided view of a bigger one.
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NM: gl.constexpr = tuning_cfg.num_mini_m()
    NN: gl.constexpr = tuning_cfg.num_mini_n()
    cl_a: gl.constexpr = tuning_cfg.dot_operand_copy_layout(0)
    cl_b: gl.constexpr = tuning_cfg.dot_operand_copy_layout(1)

    pid = gl.program_id(0)
    if require_constexpr(tuning_cfg.TILE_SCHED == _TS_XCD_GROUP_M):
        # Drop the padded tiles first so the swizzle is a bijection over real work; a
        # grid-persistent mode would forfeit this early return, which is why it is out
        # of scope here.
        unpadded_m = gl.load(rt.expt_offs_sum)
        if pid >= unpadded_m * grid_n:
            return
        pid = remap_xcd(pid, unpadded_m * grid_n, tuning_cfg.NUM_XCDS)
        pid_m, pid_n = pid_grid(pid, unpadded_m, grid_n, tuning_cfg.GROUP_M)
    elif require_constexpr(tuning_cfg.TILE_SCHED == _TS_GROUP_M):
        pid_m, pid_n = pid_grid(pid, grid_m, grid_n, tuning_cfg.GROUP_M)
    else:
        pid_m = pid // grid_n
        pid_n = pid % grid_n

    expt_data = gl.load(rt.expt_block_pid_map + pid_m)
    if expt_data == -1:
        # No work. Nothing has been committed yet, so no async group is left dangling.
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M_e = gl.load(rt.expt_hist + expt_id)
    start_m = gl.load(rt.expt_offs_raw + expt_id)

    # Issued right after the routing scalars it depends on: the gather's table loads are
    # the longest-latency thing in the prologue, so everything below overlaps with them.
    a_offs = ()
    for mi in gl.static_range(NM):
        rows_a = _gather_rows(
            rt,
            block_id,
            M_e,
            start_m,
            gl.SliceLayout(1, cl_a),
            BM,
            mi * MBM,
            MBM,
            func_cfg.has_gather,
        )
        a_offs = a_offs + (
            rows_a[:, None] * a.stride_m
            + gl.arange(0, PK_A, layout=gl.SliceLayout(0, cl_a))[None, :],
        )

    # Buffer ops carry a 32-bit offset (2 GB window) and V4-Pro's stacked gemm1 weight is
    # ~8.5 GB, so the expert stride is folded into the scalar base in 64-bit; a single
    # expert is ~22 MB and fits comfortably.
    w_ptr = b.ptr + expt_id.to(gl.int64) * b.stride_e
    if require_constexpr(func_cfg.a_has_scale()):
        as_ptr = a.scale_ptr
    else:
        as_ptr: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        ws_ptr = b.scale_ptr + expt_id.to(gl.int64) * b.scale_stride_e
    else:
        ws_ptr: gl.constexpr = None

    # The register-resident-B path was stripped along with b_read_hbm_ptr; it needs
    # reinstating before B_IN_REG can be used again.
    gl.static_assert(
        not tuning_cfg.B_IN_REG,
        "B_IN_REG is unsupported: the register-path B pointer was removed",
    )

    KB: gl.constexpr = cfg.K // func_cfg.b_pack_divisor()
    b_offs = ()
    for ni in gl.static_range(NN):
        if require_constexpr(tuning_cfg.B_PRESHUFFLED):
            b_offs = b_offs + (
                _blocked_b_offsets(
                    cl_b,
                    PK_B,
                    _n_start(pid_n, ni, cfg.N, func_cfg, tuning_cfg),
                    MBN,
                    KB,
                ),
            )
        else:
            b_offs = b_offs + (
                gl.arange(0, PK_B, layout=gl.SliceLayout(1, cl_b))[:, None] * b.stride_k
                + (
                    _n_start(pid_n, ni, cfg.N, func_cfg, tuning_cfg)
                    + gl.arange(0, MBN, layout=gl.SliceLayout(0, cl_b))
                )[None, :]
                * b.stride_n,
            )

    if require_constexpr(tuning_cfg.B_IN_REG and tuning_cfg.B_PRESHUFFLED):
        # Same 16-column-blocked addressing as the copy above, but issued at the MFMA
        # operand layout: lane L of a wave takes bytes [16L, 16L+16) of a 1024 B run, so
        # the warp's whole fetch is one contiguous block. On the plain (E, K/2, N)
        # layout the same fragment layout puts consecutive lanes stride(-2) apart, and
        # the 64 lanes touch 64 separate lines; measured 2.8x the L1 accesses and 42%
        # slower.
        fl_b: gl.constexpr = tuning_cfg.dot_operand_fragment_layout(1)
        b_frag_offs = ()
        for ni in gl.static_range(NN):
            b_frag_offs = b_frag_offs + (
                _blocked_b_offsets(
                    fl_b,
                    PK_B,
                    _n_start(pid_n, ni, cfg.N, func_cfg, tuning_cfg),
                    MBN,
                    KB,
                ),
            )
    elif require_constexpr(tuning_cfg.B_IN_REG):
        # Same addressing as b_offs, but at the MFMA operand layout instead of the
        # copy layout: the load lands directly in the registers the dot consumes.
        fl_b: gl.constexpr = tuning_cfg.dot_operand_fragment_layout(1)
        b_frag_offs = ()
        for ni in gl.static_range(NN):
            b_frag_offs = b_frag_offs + (
                gl.arange(0, PK_B, layout=gl.SliceLayout(1, fl_b))[:, None] * b.stride_k
                + (
                    _n_start(pid_n, ni, cfg.N, func_cfg, tuning_cfg)
                    + gl.arange(0, MBN, layout=gl.SliceLayout(0, fl_b))
                )[None, :]
                * b.stride_n,
            )
    else:
        b_frag_offs: gl.constexpr = None

    a_scale_offs = _a_scale_offsets(
        a, rt, block_id, M_e, start_m, pid_m, cfg.K, func_cfg, tuning_cfg
    )
    b_scale_offs = _b_scale_offsets(b, pid_n, cfg.N, cfg.K, func_cfg, tuning_cfg)

    lds = LDSManager.alloc(func_cfg, tuning_cfg)

    # per-fill pointer bumps (K is the contiguous axis of every operand)
    a_step: gl.constexpr = PK_A
    # In the 16-column blocked layout a K stage is (BLOCK_K/2 / 16) runs of 256 B,
    # so the stage bump is BLOCK_K * 8 rather than the BLOCK_K/2 of a plain tile.
    b_step: gl.constexpr = PK_B // 16 * 256 if tuning_cfg.B_PRESHUFFLED else PK_B
    s_step: gl.constexpr = SK

    a_hbm_ptr = a.ptr
    b_hbm_ptr = w_ptr
    a_scale_hbm_ptr = as_ptr
    b_scale_hbm_ptr = ws_ptr

    A_HAS: gl.constexpr = func_cfg.a_has_scale()
    B_HAS: gl.constexpr = func_cfg.b_has_scale()
    # Commit groups per stage, at whatever granularity WAIT_COMMIT_SCHEME emits them.
    # The prologue wait below counts in these, so it has to move with the scheme: with
    # a per-stage commit the stream is one group per buffer, and (NB-2) * (NM+NN) would
    # leave every prologue fill outstanding instead of retiring buffer 0's.
    G: gl.constexpr = tuning_cfg.commit_groups_per_stage()

    MAIN: gl.constexpr = NUM_K - NB
    # steps left after the peel, rounded down to a whole number of unrolled bodies
    UNROLLED: gl.constexpr = ((MAIN - 1) // tuning_cfg.K_UNROLL) * tuning_cfg.K_UNROLL
    # one buffer is being filled and one consumed on every step, so the read buffer's
    # groups have NUM_LDS_BUFFER - 2 whole stages behind them
    STAGES_BETWEEN: gl.constexpr = NB - 2

    # The scale strides only exist on a quantised operand, so they can only be read
    # behind the same predicate that guards every other use of them.
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_stride_k: gl.constexpr = a.scale_stride_k
    else:
        a_scale_stride_k: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_stride_k: gl.constexpr = b.scale_stride_k
    else:
        b_scale_stride_k: gl.constexpr = None
    pc = _PipelineConst(
        lds,
        a_offs,
        b_offs,
        b_frag_offs,
        a_scale_offs,
        b_scale_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
    )

    NUM_MINI: gl.constexpr = tuning_cfg.num_mini_k()
    PF_MINI: gl.constexpr = tuning_cfg.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI

    # Prologue fill: all but the *last* buffer, no mma. Walks the slot grid rather than
    # A-then-B so the commit groups land in the same order _slot_wait() assumes for a
    # loop stage.
    #
    # The last buffer is held back deliberately. Filling all NB here would mean stalling
    # on buffer 0 with NB whole stages already in flight and nothing but the wait to do;
    # issuing NB-1 lets the wait drop to (NB-2)*G, and the buffer that was not filled is
    # then issued *after* the first ds_read + MFMA, which is real work for its latency to
    # hide behind. The group order is unchanged -- the held-back stage is still the
    # newest -- so every _slot_wait() count downstream still holds, and by the time the
    # first pipeline step runs the outstanding set is the same (NB-1)*G either way.
    for i in gl.static_range(NB - 1):
        for ni in gl.static_range(NN):
            for mi in gl.static_range(NM):
                a_hbm_ptr, b_hbm_ptr, a_scale_hbm_ptr, b_scale_hbm_ptr = _slot_fills(
                    pc,
                    i,
                    mi,
                    ni,
                    a_hbm_ptr,
                    b_hbm_ptr,
                    a_scale_hbm_ptr,
                    b_scale_hbm_ptr,
                    ADVANCE=(mi == NM - 1) and (ni == NN - 1),
                )

    # Read-side scale pointers. These exist separately from the fill-side ones because
    # the register fallback (scale tile too small for a coalesced direct-to-LDS write)
    # reads the scale at *consume* time, NUM_LDS_BUFFER stages behind the fill;
    # advancing only one of the two silently re-reads the first K tile's scales for the
    # whole loop, which no compile-time check catches. Both start at the same base and
    # walk K at the same rate, just from different points in the pipeline.
    a_scale_read_hbm_ptr = as_ptr
    b_scale_read_hbm_ptr = ws_ptr

    # Prologue read: stage 0, whose MFMA window has nothing before it, so only its head
    # is dotted here. Its tail seeds the carried fragments the first loop step consumes.
    # It reads every mini block of buffer 0, so it waits out that buffer's whole group
    # set -- the NB-2 buffers behind it stay in flight, the last one not yet issued.
    lds.wait_fill_lds_num_group((NB - 2) * G)
    if require_constexpr(tuning_cfg.MANUAL_PP):
        gl.amd.cdna4.sched_barrier(0)
        # The prologue fills are cooperative -- every wave writes a slice and then
        # reads the whole tile -- so the wait (a per-wave vmcnt retire) is not enough
        # on its own. Membar used to supply this barrier via its after-async_wait
        # rule; with the async-wait rule off it is ours to place.
        gl.barrier()
        gl.amd.cdna4.sched_barrier(0)
    # The prologue's own step: it fills the buffer held back above and issues the first
    # ds_read, but has nothing carried in to dot yet -- which is exactly a pipeline step
    # with the MFMAs switched off. Reusing _pipeline_step keeps the fill/read/wait
    # interleave, the commit-group order and the MANUAL_PP fences in one place instead of
    # a second hand-rolled copy that has to be kept in step with it.
    #
    # Holding the last fill back is what makes that possible: this step reads one buffer
    # while filling one, with NB-2 whole stages in between -- the same STAGES_BETWEEN the
    # loop runs at, so the slot waits it computes are the steady-state ones.
    #
    # The split-stage / register-resident-B variant of this prologue was removed; the
    # single MFMA-less step below is the only form. It needs HEAD_MINI == 0, because a
    # split stage would have to dot the head of stage 0 here, and it needs B_IN_REG off,
    # because that wants *two* register stages seeded before the first step consumes one
    # and a single step produces one. Both are asserted rather than branched on.
    gl.static_assert(
        HEAD_MINI == 0,
        "split-stage prologue (VGPR_PREFETCH_K < BLOCK_K) is unsupported",
    )
    acc0 = ()
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            acc0 = acc0 + (
                gl.zeros(
                    [MBM, MBN],
                    dtype=func_cfg.mma_acc_dtype,
                    layout=tuning_cfg.dot_result_fragment_layout(),
                ),
            )
    st = _pipeline_step(
        pc,
        _PipelineState(
            a_hbm_ptr,
            b_hbm_ptr,
            a_scale_hbm_ptr,
            b_scale_hbm_ptr,
            a_scale_read_hbm_ptr,
            b_scale_read_hbm_ptr,
            (),
            (),
            (),
            acc0,
        ),
        NB - 1,
        0,
        STAGES_BETWEEN,
        True,
        True,
        DO_MFMA=False,
    )
    # The whole main sequence is guarded: MAIN is 0 when NUM_K == NUM_LDS_BUFFER, and
    # then every stage belongs to the drain and there is no step here at all.
    if require_constexpr(MAIN > 0):
        # First step peeled out. Its MFMAs are the ones that consume the accumulator
        # while it is still visibly gl.zeros -- inside the loop that is a phi and the
        # zero is invisible.
        st = _pipeline_step(pc, st, 0, 1 % NB, STAGES_BETWEEN, True, True)

        # steady state over the remaining MAIN-1 steps, body unrolled K_UNROLL times.
        #
        # The buffer index is what decides whether every ds_read and buffer_load address
        # constant-folds. When K_UNROLL is a multiple of NB the trip lands back on the
        # buffer phase it started on, so global step 1+_k+i is congruent to 1+i mod NB
        # and the index is a literal. Otherwise it has to come from the induction
        # variable, which is what decouples the two knobs -- at a measured ~12%
        # (676 -> 756 us at the tuned shape), because the offsets then live in registers
        # instead of the instruction encoding. So: pay it only when asked for a factor
        # that does not divide.
        #
        # The wait_group counts are the same either way; _slot_wait reads only
        # STAGES_BETWEEN and the slot position, never which buffer.
        for _k in tl.range(0, UNROLLED, tuning_cfg.K_UNROLL):
            for i in gl.static_range(tuning_cfg.K_UNROLL):
                if require_constexpr(tuning_cfg.K_UNROLL % NB == 0):
                    buffer_load_step = (i + 1) % NB
                    ds_read_step = (i + 2) % NB
                else:
                    buffer_load_step = (_k + i + 1) % NB
                    ds_read_step = (_k + i + 2) % NB
                st = _pipeline_step(
                    pc,
                    st,
                    buffer_load_step,
                    ds_read_step,
                    STAGES_BETWEEN,
                    True,
                    True,
                    IN_LOOP=True,
                    KI=i,
                    KU=tuning_cfg.K_UNROLL,
                )

        # remainder (0 .. K_UNROLL-1 steps); NUM_K is constexpr so this stays static
        for j in gl.static_range(1 + UNROLLED, MAIN):
            st = _pipeline_step(
                pc, st, j % NB, (j + 1) % NB, STAGES_BETWEEN, True, True
            )

    if require_constexpr(func_cfg.has_bias):
        bias_base = bias_ptr + expt_id.to(gl.int64) * stride_bias_e
    else:
        bias_base: gl.constexpr = None
    if require_constexpr(func_cfg.has_gammas):
        gammas_base = rt.gammas + start_m
    else:
        gammas_base: gl.constexpr = None

    # Stage the epilogue's two vectors in LDS, allocated and filled together here rather
    # than in the prologue: the drain issues no fills of its own, so these copies have its
    # whole run of MFMAs to themselves and are done well before the epilogue's
    # wait_group(0). Keeping the allocation next to its one use also lets the shared-memory
    # allocator see that these buffers never overlap the K-loop's.
    #
    # Both tiles are flat: `.index()` drops the leading axis but keeps the layout's rank,
    # so a [rows, mini] tile cannot be viewed as one mini block. `.slice(start, length)`
    # preserves rank, so a 1-D tile slices cleanly into per-mini-block views.
    #
    # The gammas tile is EPI_T elements, not BLOCK_M: buffer_load_to_shared requires
    # exactly 32 or 128 bits per thread, so the copy has to cover one element per thread
    # of the CTA. BLOCK_M=128 over 256 threads would be half of one, so the tile runs to
    # 256 and the surplus is masked off and never read. Bias is BLOCK_N, which is already
    # a whole number of elements per thread.
    EPI_T: gl.constexpr = tuning_cfg.num_warps() * WARP_SIZE
    EPI_BPT: gl.constexpr = BN // EPI_T
    EPI_LDS: gl.constexpr = (
        EPI_T >= BM
        and BN % EPI_T == 0
        and (EPI_BPT == 1 or EPI_BPT == 4)
    )
    EPI_SH: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    EPI_GC: gl.constexpr = gl.BlockedLayout(
        [1], [WARP_SIZE], [tuning_cfg.num_warps()], [0]
    )
    EPI_BC: gl.constexpr = gl.BlockedLayout(
        [EPI_BPT], [WARP_SIZE], [tuning_cfg.num_warps()], [0]
    )
    if require_constexpr(EPI_LDS and func_cfg.has_gammas):
        gamma_smem = gl.allocate_shared_memory(gl.float32, [EPI_T], layout=EPI_SH)
    else:
        gamma_smem: gl.constexpr = None
    if require_constexpr(EPI_LDS and func_cfg.has_bias):
        bias_smem = gl.allocate_shared_memory(gl.float32, [BN], layout=EPI_SH)
    else:
        bias_smem: gl.constexpr = None

    # They are committed as one group, which is newer than every pipeline group still in
    # flight. That inflates the outstanding count every drain wait is measured against,
    # so each step is handed WAIT_SLACK=EPI_GROUPS and its wait_group count comes out
    # numerically the same as before.
    EPI_GROUPS: gl.constexpr = (
        1 if EPI_LDS and (func_cfg.has_gammas or func_cfg.has_bias) else 0
    )
    if require_constexpr(EPI_LDS):
        if require_constexpr(func_cfg.has_gammas):
            g_offs = BM * block_id + gl.arange(0, EPI_T, layout=EPI_GC)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                gamma_smem, gammas_base, g_offs, mask=g_offs < M_e, other=0.0
            )
        if require_constexpr(func_cfg.has_bias):
            epi_b_offs = _n_split_offs(
                pid_n,
                gl.arange(0, BN, layout=EPI_BC),
                cfg.N,
                func_cfg,
                tuning_cfg,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bias_smem, bias_base, epi_b_offs
            )
        if require_constexpr(func_cfg.has_gammas or func_cfg.has_bias):
            # EPI_LDS is only a layout-feasibility flag, so it can be set with neither
            # vector present -- nothing was issued then, and an empty commit group would
            # still count against every drain wait.
            gl.amd.cdna4.async_copy.commit_group()

    # drain: the last NB stages are consumed with NO copy at all. This, not masking, is
    # what keeps every load inside K -- getting it wrong reads into the next expert's
    # weights while every correctness case still passes. The final step reads nothing and
    # just consumes the fragments still in registers.
    # The live path fuses the final activation; the reference keeps its pinned drain.
    FUSE: gl.constexpr = (
        not tuning_cfg.FROZEN_STEP
        and func_cfg.gu_split()
        and tuning_cfg.num_mini_n() == 2
    )
    DRAIN: gl.constexpr = NB - 1 if FUSE else NB
    for i in gl.static_range(DRAIN):
        st = _pipeline_step(
            pc,
            st,
            0,
            (MAIN + i + 1) % NB,
            NB - 2 - i,
            False,
            i + 1 < NB,
            WAIT_SLACK=EPI_GROUPS,
        )

    # Hoisted out of the mini-tile loop: one scalar load, not one per tile.
    if require_constexpr(func_cfg.has_x_static_scale):
        x_static_scale = gl.load(x_static_scale_ptr)
    else:
        x_static_scale: gl.constexpr = None
    if require_constexpr(
        EPI_LDS and (func_cfg.has_gammas or func_cfg.has_bias)
    ):
        # The copies were issued at the top of the drain in their own commit group, so
        # this retires them (and nothing else -- every pipeline group is older). The fill
        # is spread over every warp and each warp reads a whole mini block out of it, so
        # the barrier is load-bearing.
        #
        # Under FUSE this sits one step earlier, before the peeled step rather than after
        # it, because that step now consumes the bias. Every step it still follows was
        # handed WAIT_SLACK=EPI_GROUPS while the group was outstanding, and the peeled
        # step issues no wait, so no count changes.
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()

    if require_constexpr(FUSE):
        acc = _drain_last_fused(
            pc,
            st,
            _epi_bias_tiles(bias_smem, bias_base, pid_n, cfg.N, func_cfg, tuning_cfg),
            x_static_scale,
            func_cfg,
            tuning_cfg,
        )
    else:
        acc = st.acc

    y_ptr = res.ptr + start_m.to(gl.int64) * res.stride_m
    if require_constexpr(func_cfg.output_quant is not None):
        ys_ptr = res.scale_ptr + start_m.to(gl.int64) * res.scale_stride_m
    else:
        ys_ptr: gl.constexpr = None
    _epilogue_store(
        acc,
        y_ptr,
        res.stride_m,
        res.stride_n,
        ys_ptr,
        res.scale_stride_m,
        res.scale_stride_n,
        bias_base,
        block_id,
        pid_n,
        cfg.N,
        M_e,
        gammas_base,
        gamma_smem,
        bias_smem,
        x_static_scale,
        func_cfg,
        tuning_cfg,
        GATE_PRE=FUSE,
    )


@gluon.jit(
    launch_metadata=moe_gemm_launch_metadata,
    do_not_specialize=["grid_m", "grid_n"],
)
def _moe_gluon_gemm1(
    a_ptr,
    a_scale_ptr,
    a_num_token,
    a_stride_m,
    a_scale_stride_m,
    b_ptr,
    b_scale_ptr,
    b_stride_e,
    b_stride_n,
    b_scale_stride_e,
    b_num_expert,
    res_ptr,
    res_scale_ptr,
    res_stride_m,
    res_scale_stride_m,
    res_scale_stride_n,
    rt_expt_block_pid_map,
    rt_expt_hist,
    rt_expt_offs_raw,
    rt_expt_offs_sum,
    rt_gather_indx,
    rt_scatter_indx,
    rt_gammas,
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    A_DTYPE_QUANT: gl.constexpr,
    A_HIDDEN_DIM: gl.constexpr,
    A_TOPK: gl.constexpr,
    A_SCALE_STRIDE_K: gl.constexpr,
    A_SCALE_SWIZZLE: gl.constexpr,
    B_DTYPE_QUANT: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    B_SCALE_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_K: gl.constexpr,
    B_HIDDEN_DIM: gl.constexpr,
    B_FUSED_INTERMEDIATE_DIM: gl.constexpr,
    B_SCALE_SWIZZLE: gl.constexpr,
    RES_DTYPE_QUANT: gl.constexpr,
    RES_STRIDE_N: gl.constexpr,
    RES_OUT_DIM: gl.constexpr,
    RT_N_EXPTS_ACT: gl.constexpr,
    CFG_FUNC: gl.constexpr,
    CFG_TUNING: gl.constexpr,
    CFG_N: gl.constexpr,
    CFG_K: gl.constexpr,
):
    """gemm1: gather + X @ W1 + bias + swiglu, optionally fused MXFP4 output quant."""
    _moe_gemm_body(
        QuantTokenTensor(
            A_DTYPE_QUANT,
            a_ptr,
            a_scale_ptr,
            a_num_token,
            a_stride_m,
            a_scale_stride_m,
            A_SCALE_STRIDE_K,
            A_HIDDEN_DIM,
            A_TOPK,
            A_SCALE_SWIZZLE,
        ),
        QuantExpertTensor(
            B_DTYPE_QUANT,
            b_ptr,
            b_scale_ptr,
            b_stride_e,
            B_STRIDE_K,
            b_stride_n,
            b_scale_stride_e,
            B_SCALE_STRIDE_N,
            B_SCALE_STRIDE_K,
            b_num_expert,
            B_HIDDEN_DIM,
            B_FUSED_INTERMEDIATE_DIM,
            B_SCALE_SWIZZLE,
        ),
        ResultTensor(
            RES_DTYPE_QUANT,
            res_ptr,
            res_scale_ptr,
            res_stride_m,
            RES_STRIDE_N,
            res_scale_stride_m,
            res_scale_stride_n,
            RES_OUT_DIM,
        ),
        RoutingMeta(
            rt_expt_block_pid_map,
            rt_expt_hist,
            rt_expt_offs_raw,
            rt_expt_offs_sum,
            rt_gather_indx,
            rt_scatter_indx,
            rt_gammas,
            RT_N_EXPTS_ACT,
        ),
        bias_ptr,
        stride_bias_e,
        x_static_scale_ptr,
        grid_m,
        grid_n,
        MoeKernelConfig(CFG_FUNC, CFG_TUNING, CFG_N, CFG_K),
    )


@gluon.jit(
    launch_metadata=moe_gemm_launch_metadata,
    do_not_specialize=["grid_m", "grid_n"],
)
def _moe_gluon_gemm2(
    a_ptr,
    a_scale_ptr,
    a_num_token,
    a_stride_m,
    a_scale_stride_m,
    b_ptr,
    b_scale_ptr,
    b_stride_e,
    b_stride_n,
    b_scale_stride_e,
    b_num_expert,
    res_ptr,
    res_scale_ptr,
    res_stride_m,
    res_scale_stride_m,
    res_scale_stride_n,
    rt_expt_block_pid_map,
    rt_expt_hist,
    rt_expt_offs_raw,
    rt_expt_offs_sum,
    rt_gather_indx,
    rt_scatter_indx,
    rt_gammas,
    bias_ptr,
    stride_bias_e,
    x_static_scale_ptr,
    grid_m,
    grid_n,
    A_DTYPE_QUANT: gl.constexpr,
    A_HIDDEN_DIM: gl.constexpr,
    A_TOPK: gl.constexpr,
    A_SCALE_STRIDE_K: gl.constexpr,
    A_SCALE_SWIZZLE: gl.constexpr,
    B_DTYPE_QUANT: gl.constexpr,
    B_STRIDE_K: gl.constexpr,
    B_SCALE_STRIDE_N: gl.constexpr,
    B_SCALE_STRIDE_K: gl.constexpr,
    B_HIDDEN_DIM: gl.constexpr,
    B_FUSED_INTERMEDIATE_DIM: gl.constexpr,
    B_SCALE_SWIZZLE: gl.constexpr,
    RES_DTYPE_QUANT: gl.constexpr,
    RES_STRIDE_N: gl.constexpr,
    RES_OUT_DIM: gl.constexpr,
    RT_N_EXPTS_ACT: gl.constexpr,
    CFG_FUNC: gl.constexpr,
    CFG_TUNING: gl.constexpr,
    CFG_N: gl.constexpr,
    CFG_K: gl.constexpr,
):
    """gemm2: X @ W2 + bias, multiplied by the router combine weight (gammas)."""
    _moe_gemm_body(
        QuantTokenTensor(
            A_DTYPE_QUANT,
            a_ptr,
            a_scale_ptr,
            a_num_token,
            a_stride_m,
            a_scale_stride_m,
            A_SCALE_STRIDE_K,
            A_HIDDEN_DIM,
            A_TOPK,
            A_SCALE_SWIZZLE,
        ),
        QuantExpertTensor(
            B_DTYPE_QUANT,
            b_ptr,
            b_scale_ptr,
            b_stride_e,
            B_STRIDE_K,
            b_stride_n,
            b_scale_stride_e,
            B_SCALE_STRIDE_N,
            B_SCALE_STRIDE_K,
            b_num_expert,
            B_HIDDEN_DIM,
            B_FUSED_INTERMEDIATE_DIM,
            B_SCALE_SWIZZLE,
        ),
        ResultTensor(
            RES_DTYPE_QUANT,
            res_ptr,
            res_scale_ptr,
            res_stride_m,
            RES_STRIDE_N,
            res_scale_stride_m,
            res_scale_stride_n,
            RES_OUT_DIM,
        ),
        RoutingMeta(
            rt_expt_block_pid_map,
            rt_expt_hist,
            rt_expt_offs_raw,
            rt_expt_offs_sum,
            rt_gather_indx,
            rt_scatter_indx,
            rt_gammas,
            RT_N_EXPTS_ACT,
        ),
        bias_ptr,
        stride_bias_e,
        x_static_scale_ptr,
        grid_m,
        grid_n,
        MoeKernelConfig(CFG_FUNC, CFG_TUNING, CFG_N, CFG_K),
    )


# Imported last, and not at the top: ``_frozen`` is a snapshot of this module's K-loop
# step and calls back into the shared halves of the kernel, so it can only be bound once
# everything it names exists. ``_pipeline_step`` looks the name up at compile time, which
# is well after this line has run.
from ._frozen import _pipeline_step_frozen  # noqa: E402
