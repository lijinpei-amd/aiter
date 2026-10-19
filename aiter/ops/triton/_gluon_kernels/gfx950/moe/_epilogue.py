# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bias, activation, quantization, and output stores for gfx950 Gluon MoE GEMMs."""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton._triton_kernels.moe.activations import (
    _swiglu,
    _swiglu_combine,
    _swiglu_pair,
)
from aiter.ops.triton.utils.common_utils import strip_annotate

from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import optional as _opt
from ._lang import require_constexpr
from ._quant import mxfp4_quant_gluon
from ._types import DtypeQuant, WeightGuHbmLayout, _panel_index

_DQ_MXFP4: gl.constexpr = gl.constexpr(int(DtypeQuant.MXFP4))
_WEIGHT_GU_SPLIT: gl.constexpr = gl.constexpr(int(WeightGuHbmLayout.SPLIT))


@aggregate
@strip_annotate
class _EpilogueInputs:
    """Optional epilogue vectors, in HBM and, when staged, in LDS."""

    bias_hbm_base: gl.tensor | gl.constexpr
    gammas_hbm_ptr: gl.tensor | gl.constexpr
    gamma_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    bias_lds_ptr: gl.shared_memory_descriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(self, bias_hbm_base, gammas_hbm_ptr, gamma_lds_ptr, bias_lds_ptr):
        self.bias_hbm_base = _opt(bias_hbm_base)
        self.gammas_hbm_ptr = _opt(gammas_hbm_ptr)
        self.gamma_lds_ptr = _opt(gamma_lds_ptr)
        self.bias_lds_ptr = _opt(bias_lds_ptr)


@gluon.jit
def _stage_epilogue_inputs(
    bias_hbm_ptr,
    stride_bias_e,
    rt,
    expt_id,
    start_m,
    block_id,
    pid_n,
    N,
    M_e,
    func_cfg,
    tuning_cfg,
    layout,
    schedule,
):
    """Prepare bias and gammas, staging them in LDS before the pipeline drain."""
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    if require_constexpr(func_cfg.has_bias):
        bias_hbm_base = bias_hbm_ptr + expt_id.to(gl.int64) * stride_bias_e
    else:
        bias_hbm_base: gl.constexpr = None
    if require_constexpr(func_cfg.has_gammas):
        gammas_hbm_ptr = rt.gammas + start_m
    else:
        gammas_hbm_ptr: gl.constexpr = None

    # Stage the epilogue's two vectors in LDS, allocated and filled together here
    # rather than in the prologue: they then have the drain's run of MFMAs to overlap
    # with and are done well before the epilogue's wait_group(0). Keeping the
    # allocation next to its one use also lets the shared-memory allocator see that
    # these buffers never overlap the K-loop's.
    #
    # The drain is not idle, despite what this comment used to claim. Any component
    # whose fill span is shorter than the pipeline depth keeps filling through drain
    # iteration `pipeline_depth - fill_span` (see Schedule.active). It happens to
    # issue nothing in the tuned configuration, where every span equals the depth,
    # but that is a property of that configuration and not of the drain.
    #
    # Both tiles are flat: `.index()` drops the leading axis but keeps the layout's rank,
    # so a [rows, slot] tile cannot be viewed as one slot. `.slice(start, length)`
    # preserves rank, so a 1-D tile slices cleanly into per-slot views.
    #
    # The gammas tile is EPI_T elements, not BLOCK_M: buffer_load_to_shared requires
    # exactly 32 or 128 bits per thread, so the copy has to cover one element per thread
    # of the CTA. BLOCK_M=128 over 256 threads would be half of one, so the tile runs to
    # 256 and the surplus is masked off and never read. Bias is BLOCK_N, which is already
    # a whole number of elements per thread.
    EPI_T: gl.constexpr = layout.epilogue_threads
    EPI_LDS: gl.constexpr = layout.epilogue_inputs_via_lds
    if require_constexpr(EPI_LDS and func_cfg.has_gammas):
        gamma_lds_ptr = gl.allocate_shared_memory(
            gl.float32,
            layout.epilogue_gamma_shape,
            layout=layout.epilogue_input_lds_layout,
        )
    else:
        gamma_lds_ptr: gl.constexpr = None
    if require_constexpr(EPI_LDS and func_cfg.has_bias):
        bias_lds_ptr = gl.allocate_shared_memory(
            gl.float32,
            layout.epilogue_bias_shape,
            layout=layout.epilogue_input_lds_layout,
        )
    else:
        bias_lds_ptr: gl.constexpr = None

    if require_constexpr(EPI_LDS):
        if require_constexpr(func_cfg.has_gammas):
            g_offs = BM * block_id + gl.arange(
                0,
                EPI_T,
                layout=layout.epilogue_gamma_buffer_load_lds_layout,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                gamma_lds_ptr, gammas_hbm_ptr, g_offs, mask=g_offs < M_e, other=0.0
            )
        if require_constexpr(func_cfg.has_bias):
            epi_b_offs = layout.n_split_offs(
                pid_n,
                gl.arange(
                    0,
                    BN,
                    layout=layout.epilogue_bias_buffer_load_lds_layout,
                ),
                N,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bias_lds_ptr, bias_hbm_base, epi_b_offs
            )
        # They are committed as the groups the schedule counts, which its drain
        # waits place at their exact FIFO position: after the last main step, before
        # the first drain step. They therefore lengthen the wait only for reads whose
        # newest copy is older -- in practice the first drain step or two -- and not
        # for the rest. Passing a blanket slack to every drain step, as the frozen
        # path still does via WAIT_SLACK, would over-wait the later ones. There are
        # none with neither vector present: EPI_LDS is only a layout-feasibility
        # flag, and an empty group would still occupy a FIFO position.
        for _ in gl.static_range(schedule.epilogue_groups):
            gl.amd.cdna4.async_copy.commit_group()
    return _EpilogueInputs(bias_hbm_base, gammas_hbm_ptr, gamma_lds_ptr, bias_lds_ptr)


@gluon.jit
def _epi_bias_tiles(bias_lds_ptr, bias_hbm_ptr, pid_n, N, func_cfg, tuning_cfg, layout):
    """Block-level bias, one tensor per N slot.

    Invariant in mi, so NN of them cover the block where the per-tile form issued
    NM*NN. From LDS when the staging constraints hold, else straight from global.
    The LDS read carries the accumulator's own N slice layout, which is nameable, so
    this really is a prefetch -- unlike gammas, see _epi_gamma_tiles.
    """
    SLOT_N: gl.constexpr = tuning_cfg.SLOT_N
    NN: gl.constexpr = layout.num_n_slots
    RFL: gl.constexpr = layout.dot_result_fragment_layout
    out = ()
    if require_constexpr(func_cfg.has_bias):
        for hn in gl.static_range(NN):
            if require_constexpr(bias_lds_ptr is not None):
                out = out + (
                    bias_lds_ptr.slice(hn * SLOT_N, SLOT_N).load(gl.SliceLayout(0, RFL)),
                )
            else:
                out = out + (
                    gl.load(
                        bias_hbm_ptr + layout.n_start(pid_n, hn, N) + gl.arange(0, SLOT_N)
                    ),
                )
    return out


@gluon.jit
def _epi_gamma_tiles(
    gamma_lds_ptr, gammas_hbm_ptr, block_id, M_e, func_cfg, tuning_cfg, layout
):
    """Block-level gammas, one tensor per M slot -- global path only.

    When gammas live in LDS this returns empty on purpose: gammas multiplies the
    *post-swiglu* tensor, whose layout is a sliced linear layout that swiglu's N
    reduction produces and no config accessor names, so the read has to happen inside
    _epilogue_one_tile off `out.type.layout`. That is a ds_read of SLOT_M
    elements against data that landed long before, not a global load.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    SLOT_M: gl.constexpr = tuning_cfg.SLOT_M
    NM: gl.constexpr = layout.num_m_slots
    out = ()
    if require_constexpr(func_cfg.has_gammas and gamma_lds_ptr is None):
        for hm in gl.static_range(NM):
            hoffs = BM * block_id + hm * SLOT_M + gl.arange(0, SLOT_M)
            out = out + (gl.load(gammas_hbm_ptr + hoffs, mask=hoffs < M_e, other=0.0),)
    return out


@gluon.jit
def _epi_stage_flush(
    qp_lds_ptr,
    qs_lds_ptr,
    mi: gl.constexpr,
    ni: gl.constexpr,
    y_hbm_ptr,
    y_stride_m,
    y_stride_n,
    ys_hbm_ptr,
    ys_stride_m,
    ys_stride_n,
    block_id,
    pid_n,
    M_e,
    func_cfg,
    tuning_cfg,
    layout,
):
    """Read one staged MXFP4 panel back out of LDS and store it.

    The second half of :func:`_epilogue_one_tile`'s staged store, split out so
    ``rotating epilogue`` can run it one iteration behind the write that produced it.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    SLOT_M: gl.constexpr = tuning_cfg.SLOT_M
    SLOT_N: gl.constexpr = tuning_cfg.SLOT_N
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_SLOT_N: gl.constexpr = layout.output_n_slot
    P_COLS: gl.constexpr = layout.quant_payload_storage_shape(SLOT_M, OUT_SLOT_N)[1]
    S_COLS: gl.constexpr = layout.quant_scale_shape(SLOT_M, OUT_SLOT_N)[1]
    PL: gl.constexpr = layout.result_store_layout(SLOT_M, P_COLS, 8)
    SL: gl.constexpr = layout.result_store_layout(SLOT_M, S_COLS, 8)
    raw_n0 = pid_n * BN + ni * SLOT_N
    n0 = raw_n0 // ARN
    # The caller's fenced barrier has retired every wave's staging writes.
    pv = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_lds_ptr, PL)
    sv = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_lds_ptr, SL)
    pm = BM * block_id + mi * SLOT_M + gl.arange(0, SLOT_M, layout=gl.SliceLayout(1, PL))
    pn = gl.arange(0, P_COLS, layout=gl.SliceLayout(0, PL))
    gl.amd.cdna4.buffer_store(
        pv,
        y_hbm_ptr,
        pm[:, None] * y_stride_m + (n0 // 2 + pn)[None, :] * y_stride_n,
        mask=(pm < M_e)[:, None],
    )
    sm = BM * block_id + mi * SLOT_M + gl.arange(0, SLOT_M, layout=gl.SliceLayout(1, SL))
    sn = gl.arange(0, S_COLS, layout=gl.SliceLayout(0, SL))
    gl.amd.cdna4.buffer_store(
        sv,
        ys_hbm_ptr,
        sm[:, None] * ys_stride_m + (n0 // MX_GROUP + sn)[None, :] * ys_stride_n,
        mask=(sm < M_e)[:, None],
    )


@gluon.jit
def _epi_block_flush(
    qp_lds_ptr,
    qs_lds_ptr,
    y_hbm_ptr,
    y_stride_m,
    y_stride_n,
    ys_hbm_ptr,
    ys_stride_m,
    ys_stride_n,
    block_id,
    pid_n,
    M_e,
    func_cfg,
    tuning_cfg,
    layout,
):
    """Read the WHOLE staged MXFP4 block back out of LDS and store it.

    The gate/up-split twin of :func:`_epi_stage_flush`. Split collapses the two N
    slots into one output tile, so a slot already spans the CTA's full emitted
    width and the only axis left to walk is M -- which means every tile can stage into
    its own rows of one block-height buffer and the drain needs a single barrier and a
    single pair of stores rather than one per tile.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    OUT_SLOT_N: gl.constexpr = layout.output_n_slot
    P_COLS: gl.constexpr = layout.quant_payload_storage_shape(BM, OUT_SLOT_N)[1]
    S_COLS: gl.constexpr = layout.quant_scale_shape(BM, OUT_SLOT_N)[1]
    PL: gl.constexpr = layout.result_store_layout(BM, P_COLS, 8)
    SL: gl.constexpr = layout.result_store_layout(BM, S_COLS, 8)
    n0 = pid_n * layout.output_block_n
    # The staging writes are spread over every warp and each warp reads rows it did not
    # write, so this fence is load-bearing. Only one is needed: nothing reuses the
    # buffer afterwards.
    gl.barrier()
    pv = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_lds_ptr, PL)
    sv = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_lds_ptr, SL)
    pm = BM * block_id + gl.arange(0, BM, layout=gl.SliceLayout(1, PL))
    pn = gl.arange(0, P_COLS, layout=gl.SliceLayout(0, PL))
    gl.amd.cdna4.buffer_store(
        pv,
        y_hbm_ptr,
        pm[:, None] * y_stride_m + (n0 // 2 + pn)[None, :] * y_stride_n,
        mask=(pm < M_e)[:, None],
    )
    sm = BM * block_id + gl.arange(0, BM, layout=gl.SliceLayout(1, SL))
    sn = gl.arange(0, S_COLS, layout=gl.SliceLayout(0, SL))
    gl.amd.cdna4.buffer_store(
        sv,
        ys_hbm_ptr,
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
    gamma_lds_ptr,
    bias_hbm_ptr,
    gammas_hbm_ptr,
    y_hbm_ptr,
    y_stride_m,
    y_stride_n,
    ys_hbm_ptr,
    ys_stride_m,
    ys_stride_n,
    block_id,
    pid_n,
    M_e,
    x_static_scale,
    func_cfg,
    tuning_cfg,
    layout,
    qp_lds_ptr=None,
    qs_lds_ptr=None,
    STAGE_ONLY: gl.constexpr = False,
    lin=None,
    STAGE_ROW: gl.constexpr = 0,
    GATE_PRE: gl.constexpr = False,
):
    """bias -> activation -> gammas -> output_quant -> store, for ONE panel.

    Lifted verbatim out of :func:`_epilogue_store` so the fused drain step can call it
    per panel the moment that panel's last MFMA retires. ``sub`` is the finished
    accumulator for tile ``(mi, ni)``.

    Under ``WeightGuHbmLayout.SPLIT`` a "tile" is a *pair*: ``sub`` is the gate side
    (N slot ``ni``) and ``lin`` the linear side (slot ``ni + 1``), covering the
    same emitted channels. The activation is then a plain elementwise op between two
    tensors of identical layout -- no reshape, no split, and none of the ``v_mov_b32``
    gathers the interleaved packing needs -- and the tile is ``SLOT_N`` emitted channels
    wide rather than ``SLOT_N // 2``. ``STAGE_ROW`` is the row this tile occupies in a
    block-height LDS staging buffer, which is what lets the caller flush all of them
    in one store.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    SLOT_M: gl.constexpr = tuning_cfg.SLOT_M
    ARN: gl.constexpr = func_cfg.activation_reduction_n()
    OUT_SLOT_N: gl.constexpr = layout.output_n_slot
    P_COLS: gl.constexpr = layout.quant_payload_storage_shape(SLOT_M, OUT_SLOT_N)[1]
    S_COLS: gl.constexpr = layout.quant_scale_shape(SLOT_M, OUT_SLOT_N)[1]
    act: gl.constexpr = func_cfg.activation
    out_ty: gl.constexpr = y_hbm_ptr.dtype.element_ty
    store_layout: gl.constexpr = layout.result_store_layout(
        SLOT_M, OUT_SLOT_N, out_ty.primitive_bitwidth
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
        if require_constexpr(func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
            lin = lin * x_static_scale

    if require_constexpr(func_cfg.has_bias and func_cfg.epilogue < 2):
        # fp32, expert indexed, over the RAW N axis (before the halving). Split: one
        # tile per side, so each accumulator takes its own.
        if require_constexpr(not GATE_PRE):
            sub = sub + bias_tiles[ni][None, :]
        if require_constexpr(func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
            lin = lin + bias_tiles[ni + 1][None, :]

    if require_constexpr(
        func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT and func_cfg.epilogue != 0
    ):
        # Ablation: the same [SLOT_M, SLOT_N] x 2 -> [SLOT_M, SLOT_N] reduction, minus exp2/rcp.
        out = sub * lin
        gl.static_assert(out.shape[1] == OUT_SLOT_N)
    elif require_constexpr(GATE_PRE):
        out = _swiglu_combine(sub, lin, act.limit, ADD_RESIDUAL=act.add_residual)
        gl.static_assert(out.shape[1] == OUT_SLOT_N)
    elif require_constexpr(func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
        out = _swiglu_pair(
            sub,
            lin,
            act.alpha,
            act.limit,
            ADD_RESIDUAL=act.add_residual,
            FAST_RCP=tuning_cfg.ACT_FAST_RCP,
        )
        gl.static_assert(out.shape[1] == OUT_SLOT_N)
    elif require_constexpr(func_cfg.has_activation() and func_cfg.epilogue != 0):
        # Ablation: the same reshape/split reduction _swiglu does, minus exp2/rcp/clip.
        gate, linear = tl.split(
            tl.reshape(sub, (sub.shape[0], sub.shape[1] // 2, 2))
        )
        out = gate * linear
        gl.static_assert(out.shape[1] == OUT_SLOT_N)
    elif require_constexpr(func_cfg.has_activation()):
        out = _swiglu(
            sub,
            act.alpha,
            act.limit,
            ADD_RESIDUAL=act.add_residual,
            FAST_RCP=tuning_cfg.ACT_FAST_RCP,
            GROUP=3,
        )
        gl.static_assert(out.shape[1] == OUT_SLOT_N)
    else:
        gl.static_assert(ARN == 1)
        out = sub

    if require_constexpr(func_cfg.has_gammas and func_cfg.epilogue < 2):
        if require_constexpr(gamma_lds_ptr is not None):
            # Only a ds_read: the global copy that filled this ran before the
            # drain. `out.type.layout` is the one way to name the post-swiglu
            # layout, so the read has to sit here rather than above the walk.
            g = gamma_lds_ptr.slice(mi * SLOT_M, SLOT_M).load(
                gl.SliceLayout(1, out.type.layout)
            )
        else:
            g = gamma_tiles[mi]
        out = out * g[:, None]

    # Emitted-channel base. The CTA tile is BLOCK_N // ARN emitted channels wide in both
    # packings -- what differs is only how many N slots that is (2 interleaved, 1
    # split, where ni is always the gate slot and OUT_SLOT_N is the whole width).
    out_n0 = pid_n * layout.output_block_n + ni * OUT_SLOT_N
    if require_constexpr(func_cfg.output_quant is None):
        val = gl.convert_layout(
            out.to(out_ty), store_layout, assert_trivial=False
        )
        sm = gl.arange(0, SLOT_M, layout=gl.SliceLayout(1, store_layout))
        sn = gl.arange(0, OUT_SLOT_N, layout=gl.SliceLayout(0, store_layout))
        rows = BM * block_id + mi * SLOT_M + sm
        gl.amd.cdna4.buffer_store(
            val,
            y_hbm_ptr,
            rows[:, None] * y_stride_m + (out_n0 + sn)[None, :] * y_stride_n,
            mask=(rows < M_e)[:, None],
            cache=tuning_cfg.result_cache_modifier,
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
        gl.static_assert(OUT_SLOT_N % MX_GROUP == 0)
        # LANE_ELEMS splits the amax: how many of the tile's elements one lane owns,
        # so the inner reduction is in-lane fp32 (free |v| modifiers) and only the
        # outer one goes cross-lane. Getting it wrong is not incorrect, just slower --
        # a mismatched split adds permlane steps instead of removing them.
        LANE_ELEMS: gl.constexpr = layout.quant_amax_lane_elems(SLOT_M, OUT_SLOT_N)
        payload, scale = mxfp4_quant_gluon(
            out, OUT_SLOT_N, SLOT_M, MX_GROUP, LANE_ELEMS
        )
        # Fresh M ranges per store: reusing one auto-layout `offs_m` across two
        # differently-laid-out stores makes GluonResolveAutoEncodings fail with
        # "conflicting encodings" on the expand_dims.
        if require_constexpr(qp_lds_ptr is not None):
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
            PL: gl.constexpr = layout.result_store_layout(SLOT_M, P_COLS, 8)
            SL: gl.constexpr = layout.result_store_layout(SLOT_M, S_COLS, 8)
            # if/elif/else, not early returns: a `return` from inside a nested
            # `if require_constexpr(...)` here does NOT stop the trace, so the code
            # below it still ran and stored the panel to the *unsliced* buffer --
            # which only surfaced as a shape mismatch because the split buffer is
            # block-height. Keep exactly one store path reachable per configuration.
            if require_constexpr(func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
                # Block-height staging: every mi writes its own SLOT_M rows of one buffer,
                # so the caller flushes all NM tiles with a single barrier and a single
                # store instead of a barrier pair per tile. Split halves the tile count
                # too (each covers both sides), so the drain goes from NM*NN staging
                # round-trips to one.
                qp_lds_ptr.slice(STAGE_ROW, SLOT_M).store(payload)
                qs_lds_ptr.slice(STAGE_ROW, SLOT_M).store(scale)
            elif require_constexpr(STAGE_ONLY):
                # rotating epilogue: stage and leave it; the caller flushes this buffer one
                # iteration later, after its own barrier.
                qp_lds_ptr.store(payload)
                qs_lds_ptr.store(scale)
            else:
                gl.barrier()
                qp_lds_ptr.store(payload)
                qs_lds_ptr.store(scale)
                # Relaxed reads still require cross-wave staging-write visibility.
                gl.barrier()
                payload_v = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_lds_ptr, PL)
                scale_v = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_lds_ptr, SL)

                pm = BM * block_id + mi * SLOT_M + gl.arange(
                    0, SLOT_M, layout=gl.SliceLayout(1, PL)
                )
                pn = gl.arange(0, P_COLS, layout=gl.SliceLayout(0, PL))
                gl.amd.cdna4.buffer_store(
                    payload_v,
                    y_hbm_ptr,
                    pm[:, None] * y_stride_m
                    + (out_n0 // 2 + pn)[None, :] * y_stride_n,
                    mask=(pm < M_e)[:, None],
                )
                sm = BM * block_id + mi * SLOT_M + gl.arange(
                    0, SLOT_M, layout=gl.SliceLayout(1, SL)
                )
                sn = gl.arange(0, S_COLS, layout=gl.SliceLayout(0, SL))
                gl.amd.cdna4.buffer_store(
                    scale_v,
                    ys_hbm_ptr,
                    sm[:, None] * ys_stride_m
                    + (out_n0 // MX_GROUP + sn)[None, :] * ys_stride_n,
                    mask=(sm < M_e)[:, None],
                )
        else:
            pm = BM * block_id + mi * SLOT_M + gl.arange(0, SLOT_M)
            gl.store(
                y_hbm_ptr
                + pm[:, None] * y_stride_m
                + (out_n0 // 2 + gl.arange(0, P_COLS))[None, :] * y_stride_n,
                payload,
                mask=(pm < M_e)[:, None],
            )
            sm = BM * block_id + mi * SLOT_M + gl.arange(0, SLOT_M)
            gl.store(
                ys_hbm_ptr
                + sm[:, None] * ys_stride_m
                + (out_n0 // MX_GROUP + gl.arange(0, S_COLS))[None, :]
                * ys_stride_n,
                scale,
                mask=(sm < M_e)[:, None],
            )


@gluon.jit
def _epilogue_store(
    acc,
    y_hbm_ptr,
    y_stride_m,
    y_stride_n,
    ys_hbm_ptr,
    ys_stride_m,
    ys_stride_n,
    bias_hbm_ptr,
    block_id,
    pid_n,
    N,
    M_e,
    gammas_hbm_ptr,
    gamma_lds_ptr,
    bias_lds_ptr,
    x_static_scale,
    func_cfg,
    tuning_cfg,
    layout,
    GATE_PRE: gl.constexpr = False,
):
    """bias -> activation -> gammas -> output_quant -> store, per panel.

    The order is fixed and matches ``_triton_kernels/moe/moe_op_gemm_a4w4.py``; the
    clamp and the multiply happen in fp32 before any cast. ``acc`` arrives already split
    into one tensor per panel -- the K pipeline accumulates that way -- so the
    epilogue walks the same tiling and consumes them one at a time, which lets the
    compiler kill each panel's accumulator before the next panel's epilogue.
    """
    SLOT_M: gl.constexpr = tuning_cfg.SLOT_M
    OUT_SLOT_N: gl.constexpr = layout.output_n_slot

    NN: gl.constexpr = layout.num_n_slots
    NM: gl.constexpr = layout.num_m_slots

    # Block-level operand loads. Each is invariant in one of the two
    # walk indices -- bias in mi, gammas in ni -- so the per-tile form below issues
    # NM*NN of them where NN (resp. NM) is enough. Own loop variables: reusing `mi`/`ni`
    # here would shadow the walk's.
    # Both are staged in LDS (filled before the drain, see _stage_epilogue_inputs);
    # the read then carries the accumulator's own M/N slice layout, so nothing downstream
    # changes. Otherwise the block-level load still comes from global.
    bias_tiles = _epi_bias_tiles(
        bias_lds_ptr, bias_hbm_ptr, pid_n, N, func_cfg, tuning_cfg, layout
    )
    gamma_tiles = _epi_gamma_tiles(
        gamma_lds_ptr, gammas_hbm_ptr, block_id, M_e, func_cfg, tuning_cfg, layout
    )

    # Staging for the MXFP4 payload store, bouncing it through LDS so the global store
    # vectorises -- worth ~16 us on the 4-wave a4w4 gemm1, see _epilogue_one_tile.
    #
    # ONE panel's worth, reused across the walk: NM*NN buffers would cost NM*NN
    # times the LDS for no gain, since the tiles are stored one after another anyway.
    # Allocated here so it dominates every use; the pipeline's buffers are dead by this
    # point, so Triton's liveness-based shared allocator overlays this on top of them
    # and the kernel's LDS footprint does not grow at all.
    QSH: gl.constexpr = layout.quant_staging_layout
    ROT: gl.constexpr = layout.quant_staging_rotates
    # Split stages the whole block, not one panel: NM panels into one buffer, flushed
    # once at the end of the walk. Costs NM x the LDS (still a few KB, overlaid on the
    # dead pipeline buffers) and buys NM-1 barrier pairs and one wide store instead of
    # NM narrow ones.
    QROWS: gl.constexpr = layout.quant_staging_rows
    if require_constexpr(func_cfg.output_quant is not None):
        qp_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_payload_storage_shape(QROWS, OUT_SLOT_N), layout=QSH
        )
        qs_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_scale_shape(QROWS, OUT_SLOT_N), layout=QSH
        )
    else:
        qp_lds_ptr: gl.constexpr = None
        qs_lds_ptr: gl.constexpr = None
    if require_constexpr(ROT):
        # Second bank for rotating epilogue. Same overlay argument as above, so the footprint
        # still does not grow.
        qp2_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_payload_storage_shape(SLOT_M, OUT_SLOT_N), layout=QSH
        )
        qs2_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_scale_shape(SLOT_M, OUT_SLOT_N), layout=QSH
        )
    else:
        qp2_lds_ptr: gl.constexpr = None
        qs2_lds_ptr: gl.constexpr = None

    if require_constexpr(func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
        # One pass over mi, pairing the two N slots: slot 0 is the gate side and
        # slot 1 the linear side of the *same* emitted channels, so the activation is
        # elementwise between two identically laid out accumulators. Every tile stages
        # into its own SLOT_M rows of the block-height buffer, then one barrier pair and
        # one store flushes the lot.
        for mi in gl.static_range(NM):
            _epilogue_one_tile(
                acc[_panel_index(mi, 0, NM, NN)],
                mi,
                0,
                bias_tiles,
                gamma_tiles,
                gamma_lds_ptr,
                bias_hbm_ptr,
                gammas_hbm_ptr,
                y_hbm_ptr,
                y_stride_m,
                y_stride_n,
                ys_hbm_ptr,
                ys_stride_m,
                ys_stride_n,
                block_id,
                pid_n,
                M_e,
                x_static_scale,
                func_cfg,
                tuning_cfg,
                layout,
                qp_lds_ptr,
                qs_lds_ptr,
                lin=acc[_panel_index(mi, 1, NM, NN)],
                STAGE_ROW=mi * SLOT_M,
                GATE_PRE=GATE_PRE,
            )
        if require_constexpr(func_cfg.output_quant is not None):
            _epi_block_flush(
                qp_lds_ptr, qs_lds_ptr,
                y_hbm_ptr, y_stride_m, y_stride_n,
                ys_hbm_ptr, ys_stride_m, ys_stride_n,
                block_id, pid_n, M_e, func_cfg, tuning_cfg, layout,
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
                    qp2_lds_ptr if (k - 1) % 2 else qp_lds_ptr,
                    qs2_lds_ptr if (k - 1) % 2 else qs_lds_ptr,
                    (k - 1) // NN, (k - 1) % NN,
                    y_hbm_ptr, y_stride_m, y_stride_n,
                    ys_hbm_ptr, ys_stride_m, ys_stride_n,
                    block_id, pid_n, M_e, func_cfg, tuning_cfg, layout,
                )
            _epilogue_one_tile(
                acc[_panel_index(k // NN, k % NN, NM, NN)],
                k // NN, k % NN,
                bias_tiles, gamma_tiles, gamma_lds_ptr, bias_hbm_ptr, gammas_hbm_ptr,
                y_hbm_ptr, y_stride_m, y_stride_n, ys_hbm_ptr, ys_stride_m, ys_stride_n,
                block_id, pid_n, M_e, x_static_scale, func_cfg, tuning_cfg,
                layout,
                qp2_lds_ptr if k % 2 else qp_lds_ptr,
                qs2_lds_ptr if k % 2 else qs_lds_ptr,
                STAGE_ONLY=True,
            )
            gl.barrier()
        LK: gl.constexpr = NM * NN - 1
        _epi_stage_flush(
            qp2_lds_ptr if LK % 2 else qp_lds_ptr,
            qs2_lds_ptr if LK % 2 else qs_lds_ptr,
            LK // NN, LK % NN, y_hbm_ptr, y_stride_m, y_stride_n,
            ys_hbm_ptr, ys_stride_m, ys_stride_n,
            block_id, pid_n, M_e, func_cfg, tuning_cfg, layout,
        )
    else:
        for mi in gl.static_range(NM):
            for ni in gl.static_range(NN):
                # acc is stored in panel-traversal order, which _panel_index defines;
                # the store loop is free to walk the grid however it likes as long as
                # it indexes through the same map.
                _epilogue_one_tile(
                    acc[_panel_index(mi, ni, NM, NN)],
                    mi,
                    ni,
                    bias_tiles,
                    gamma_tiles,
                    gamma_lds_ptr,
                    bias_hbm_ptr,
                    gammas_hbm_ptr,
                    y_hbm_ptr,
                    y_stride_m,
                    y_stride_n,
                    ys_hbm_ptr,
                    ys_stride_m,
                    ys_stride_n,
                    block_id,
                    pid_n,
                    M_e,
                    x_static_scale,
                    func_cfg,
                    tuning_cfg,
                    layout,
                    qp_lds_ptr,
                    qs_lds_ptr,
                )
