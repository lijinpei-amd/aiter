# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operation-shape, data-layout, and offset contracts for gfx950 Gluon MoE GEMMs.

Every tile shape, register/shared layout, resource count and resolved LDS/register
placement of a build lives in one :class:`KernelLayout`, evaluated once from the
tuning configuration by its constructor, ``KernelLayout(tuning_cfg)``. The yes/no
knob policy it is built on (``allow_component_via_lds``, ``scale_shuffled``, ...) is
``KernelTuningConfig`` methods. The final placement also depends on whether the slot
admits a direct-to-LDS copy, so it is resolved here. The host (grid, LDS probes) and
the kernel both read that object, so neither re-derives a layout per query and the
two cannot drift. The device-side raw-N mapping and HBM offset grids are
``KernelLayout`` jit methods.

The public surface is :class:`KernelLayout`: its constructor, methods and fields.
Everything else here is internal derivation detail that no other module imports.

Top to bottom: helpers that take explicit arguments (layout builders such as
:func:`byte_unit_lds_layout`, the jit HBM-offset grids), then the
:class:`KernelLayout` aggregate: its fields, the constructor that derives every field
from the tuning configuration in one pass, computing each quantity once, and the
accessors.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import MX_GROUP, WARP_SIZE, require_constexpr, strip_annotate
from ._types import WeightGuHbmLayout

__all__ = ["KernelLayout"]

_NO_SCALE: gl.constexpr = gl.constexpr(None)
_WEIGHT_GU_SPLIT: gl.constexpr = gl.constexpr(int(WeightGuHbmLayout.SPLIT))


@gluon.constexpr_function
def _bases_to_distributed(offset_bases, contiguity, num_warps, warp_size, shape):
    """Turn a ``PaddedSharedLayout``'s offset bases into the matching register layout.

    Mirrors Triton's ``CoalesceAsyncCopy`` partition -- lg2(C) bases to reg, lg2(WS) to
    lane, lg2(NW) to warp, leftovers back to reg -- which is what makes
    ``buffer_load_to_shared`` fold into a single ``buffer_load_dwordx4 ... lds`` instead
    of silently degrading to dword or failing to legalise. Same helper as
    ``gfx950/attention/fp8_mqa_logits.py::_offset_bases_to_blocked``.
    """
    rank = len(shape)
    lg2_c = contiguity.bit_length() - 1
    lg2_nw = num_warps.bit_length() - 1
    lg2_ws = warp_size.bit_length() - 1

    i = 0
    reg = list(offset_bases[i : i + lg2_c])
    i += lg2_c
    lane = list(offset_bases[i : i + lg2_ws])
    i += lg2_ws
    warp = list(offset_bases[i : i + lg2_nw])
    i += lg2_nw
    warp = warp + [[0] * rank] * (lg2_nw - len(warp))
    reg = reg + list(offset_bases[i:])
    return gl.DistributedLinearLayout(
        reg_bases=[list(b) for b in reg],
        lane_bases=[list(b) for b in lane],
        warp_bases=[list(b) for b in warp],
        block_bases=[],
        shape=list(shape),
    )


#: K extent of one LDS unit, in bytes. Fixed, for every dtype: the operand shape in
#: scope is 16x16x128 B, which is 16x16x128 for fp8 and 16x16x256 for packed fp4 (one
#: unit then spans two MFMA K-steps, and the layout is unaffected).
LDS_UNIT_K_BYTES = 128
#: The 16 B a lane moves per access -- the granule the preshuffled HBM order blocks by.
LDS_LANE_BYTES = 16
#: Padding of a non-preshuffled unit: 32 B every 1024 B.
LDS_PAD_INTERVAL_BYTES = 1024
LDS_PAD_BYTES = 32


@gluon.constexpr_function
def _pow2_bases(dim, rank, values):
    """``[[v, 0], ...]`` (or ``[[0, v], ...]``) for each stride in ``values``."""
    return [[v if i == dim else 0 for i in range(rank)] for v in values]


@gluon.constexpr_function
def _ramp(lo, hi):
    """Powers of two ``lo, 2*lo, ... hi/2``; empty when ``hi <= lo``."""
    out = []
    v = lo
    while v < hi:
        out.append(v)
        v *= 2
    return out


@gluon.constexpr_function
def byte_unit_lds_layout(shape, nk_dim, storage_bits, unit_rows, preshuffled):
    """LDS layout of a payload operand, tiled by a fixed (rows x 128 B) unit.

    The slot is read as ``(non-K units, K units, unit_rows, 128 B)`` with the rightmost
    axis contiguous, so the unit -- not the slot -- is what fixes the layout. That
    is the difference from ``compute_efficient_padded_shared_layout``, which sizes its
    non-K unit from the whole slot and therefore hands a 128-row slot a different
    permutation than a 32-row one.

    Two orders, selected by whether the operand is already preshuffled in HBM:

    * not preshuffled -- 32 rows, the measured row permutation ``1, 4, 16, 2, 8``, and
      32 B of padding every 1024 B. A warp's 64 lanes cover exactly one 1024 B run, so
      the padding never splits a direct-to-LDS write.
    * preshuffled -- 16 rows in the byte order ``utils/shuffle.py::shuffle_weight(w,
      (16, 16))`` already writes, i.e. ``(n//16)*(KB*16) + (k//16)*256 + (n%16)*16 +
      k%16``. The copy is then a straight linear run and no padding is needed, because
      the 64 lanes of a fetch already land on 64 consecutive 16 B slots.

    ``shape`` and the returned bases are a storage shape; with the 8-bit storage dtype
    both fp8 and fp4 use, one storage element is one byte.
    """
    shape = [int(x) for x in shape]
    rank = len(shape)
    kd = 1 - nk_dim
    els = lambda nbytes: nbytes * 8 // storage_bits

    U = els(LDS_UNIT_K_BYTES)  # storage elements per 128 B K unit
    V = els(LDS_LANE_BYTES)  # storage elements per 16 B lane access
    rows, k_storage = shape[nk_dim], shape[kd]

    K = lambda vs: _pow2_bases(kd, rank, vs)
    NK = lambda vs: _pow2_bases(nk_dim, rank, vs)

    if preshuffled:
        # 16 B of K, then the 16 rows, then the rest of the 128 B unit.
        bases = K(_ramp(1, V)) + NK(_ramp(1, unit_rows)) + K(_ramp(V, U))
    else:
        assert unit_rows == 32, "the 1, 4, 16, 2, 8 permutation is a 32-row order"
        bases = K(_ramp(1, U)) + NK([1, 4, 16, 2, 8])
    bases = bases + K(_ramp(U, k_storage)) + NK(_ramp(unit_rows, rows))

    if preshuffled:
        # PaddedSharedLayout insists on at least one interval/padding pair, so name an
        # interval wider than the slot: no pad can land inside it. 16 rather than 1 so
        # that even if the allocator rounds the slot up, it stays ds_read_b128 aligned.
        pairs = [[1 << (rows * k_storage).bit_length(), 16]]
    else:
        pairs = [[els(LDS_PAD_INTERVAL_BYTES), els(LDS_PAD_BYTES)]]
    return gl.PaddedSharedLayout(
        interval_padding_pairs=pairs,
        offset_bases=bases,
        cga_layout=[],
        shape=shape,
    )


@gluon.jit
def _blocked_b_hbm_offsets(
    layout: gl.constexpr,
    payload_block_k_storage: gl.constexpr,
    n0,
    n_slot: gl.constexpr,
    k_storage,
):
    """Byte offsets for one B slot in the 16-column-blocked HBM layout.

    ``utils/shuffle.py::shuffle_weight(w, (16, 16))`` maps storage ``(n, k)``
    to ``(n//16)*(k_storage*16) + (k//16)*256 + (n%16)*16 + k%16``. Keeping
    this physical-layout transform beside the layout that consumes it prevents
    the HBM and LDS permutations from drifting apart.
    """
    kk = gl.arange(0, payload_block_k_storage, layout=gl.SliceLayout(1, layout))[
        :, None
    ]
    nn = (n0 + gl.arange(0, n_slot, layout=gl.SliceLayout(0, layout)))[None, :]
    return (nn // 16) * (k_storage * 16) + (kk // 16) * 256 + (nn % 16) * 16 + kk % 16


@gluon.jit
def _shuffled_scale_register_offsets(
    layout: gl.constexpr,
    nonk0,
    nonk: gl.constexpr,
    scale_block_k_storage: gl.constexpr,
    K,
    packed: gl.constexpr = False,
    split_extent: gl.constexpr = 0,
    split_stride=0,
):
    """HBM offsets for CDNA4_SCALE / SORTED_SHUFFLED scale storage.

    Each 32-row stripe stores 256 K values' scales in 256 bytes. Within a
    dword, non-K +16 advances one byte and K +128 advances two bytes. K128
    stages load the complete packed word, matching the LDS representation.
    """
    if require_constexpr(packed):
        rr = gl.arange(0, nonk, layout=gl.SliceLayout(1, layout))[:, None]
        width: gl.constexpr = scale_block_k_storage // 4
        cc = gl.arange(0, width, layout=gl.SliceLayout(0, layout))[None, :]
        word = rr * width + cc
        stripe_dwords: gl.constexpr = scale_block_k_storage * 8
        row = (word // stripe_dwords) * 32
        if require_constexpr(split_extent):
            row = row // split_extent * split_stride + row % split_extent
        offsets = ((nonk0 + row) // 32) * K + (word % stripe_dwords) * 4
    else:
        nn = gl.arange(0, nonk, layout=gl.SliceLayout(1, layout))[:, None]
        if require_constexpr(split_extent):
            nn = nn // split_extent * split_stride + nn % split_extent
        nn += nonk0
        kk = gl.arange(0, scale_block_k_storage, layout=gl.SliceLayout(0, layout))[
            None, :
        ]
        offsets = (
            (nn // 32) * K
            + (kk // 8) * 256
            + (kk % 4) * 64
            + (nn % 16) * 4
            + ((kk % 8) // 4) * 2
            + (nn % 32) // 16
        )
    return offsets


@gluon.jit
def _shuffled_scale_stage_offsets(
    layout: gl.constexpr,
    nonk0,
    stripes: gl.constexpr,
    K,
    scale_slot_k: gl.constexpr = 256,
    split_extent: gl.constexpr = 0,
    split_stride=0,
):
    """Flat HBM offsets for shuffled scales starting at logical non-K row ``nonk0``."""
    stripe = gl.arange(0, stripes, layout=gl.SliceLayout(1, layout))[:, None]
    byte = gl.arange(0, scale_slot_k, layout=gl.SliceLayout(0, layout))[None, :]
    if require_constexpr(not split_extent):
        return (nonk0 // 32 + stripe) * K + byte
    row = stripe * 32
    row = row // split_extent * split_stride + row % split_extent
    return ((nonk0 + row) // 32) * K + byte


@gluon.jit
def _gather_rows(
    rt,
    block_id,
    M_e,
    start_m,
    layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    ROW_OFFSET_SLOT: gl.constexpr,
    M_SLOT: gl.constexpr,
    HAS_GATHER: gl.constexpr,
):
    """Resolve the row indices for one M slot through the gather table.

    TODO: stage the table in LDS instead. One BLOCK_M-wide ``buffer_load_to_shared``
    issued next to the routing scalars, committed there, with the ``wait_group`` and the
    LDS read deferred to just before ``pc`` is built, would replace
    ``num_m_slots`` register loads per consumer and hide the fetch behind
    the whole B-side address computation -- today the scheduler parks
    ``s_waitcnt vmcnt(0)`` ~9 instructions after the load, so the latency is fully
    exposed. It also serves both consumers (payload and scale offsets) from one fetch
    instead of one per layout.

    Attempted and reverted: on gfx950 the direct-to-LDS lowering refuses it for *this*
    table, and the op then survives to LLVM as an unconverted
    ``builtin.unrealized_conversion_cast``. ``canLoadDirectToLDS`` (AMD Utility.cpp)
    wants ``contig * elemBits`` in {32, 128} -- CDNA4 disables 8/16-bit direct-to-LDS --
    so a uint16 table must pack two entries per lane, and
    ``getContiguity(ptr, offset)`` then demands both a 4-byte-aligned *scalar base* and
    offset contiguity >= 2. The base is ``gather_indx + start_m`` with ``start_m`` a raw
    prefix sum (odd about half the time), and any mask or ``minimum`` clamp on the
    offsets collapses the contiguity. Measured: int32 lowers with a bare, clamped *or*
    masked range; uint16 lowers only with a bare range off a provably aligned base.

    The way in is a 32-bit, tile-aligned table -- which ``_sorted_token_id_map`` in
    moe_op_gemm_gluon.py already builds (int32, padded to ``block_m``,
    ``// n_expts_act`` applied, memoised on ``expt_data``) for the sorted-scales path.
    Reading rows from it at ``pid_m * block_m`` makes alignment and contiguity trivial
    and keeps the mask; the cost is building it for configs that do not already.
    """
    offsets_slot = (
        BLOCK_M * block_id + ROW_OFFSET_SLOT + gl.arange(0, M_SLOT, layout=layout)
    )
    live_slot = offsets_slot < M_e
    if require_constexpr(HAS_GATHER):
        # gather_indx is uint16 when n_gates <= 65535, else int32; it indexes gates, so
        # divide by n_expts_act to get the token row.
        rows_slot = (
            gl.load(
                rt.gather_indx + start_m + offsets_slot,
                mask=live_slot,
                other=0,
            )
            // rt.n_expts_act
        )
    else:
        rows_slot = start_m + gl.where(live_slot, offsets_slot, 0)
    return rows_slot.to(gl.int32)


@gluon.constexpr_function
def _n_reduction_slot(func_cfg):
    """Emitted columns per raw column within one N slot.

    The tile-level ratio is always ``KernelFuncConfig.activation_reduction_n``, but
    under ``WeightGuHbmLayout.SPLIT`` the halving happens *across* two slots rather
    than inside each one -- a slot is a whole operand side, so its width maps 1:1 onto
    emitted channels and two of them collapse into one output tile.
    """
    if func_cfg.weight_gu_hbm_layout.value == WeightGuHbmLayout.SPLIT:
        return 1
    return func_cfg.activation_reduction_n()


@gluon.constexpr_function
def _scale_dword_delta(delta, k=256):
    """Where a (non-K, K) step lands, as a coordinate of the i32 tile.

    The tile is [non-K, BLOCK_K/128] and its row-major linearisation *is* the dword
    index, so a step worth ``d`` dwords is the coordinate ``(d // 2, d % 2)``.
    """
    b = [int(x) for x in delta]
    if b == [0, 0]:
        return [0, 0]
    # CDNA4_SCALE writes byte (n//32)*256 + (k%4)*64 + (n%16)*4 + (k//4)*2 +
    # (n%32)//16; dropping the two within-dword terms and dividing by four
    # leaves the dword index below.
    width = k // 128
    d = (b[0] // 32) * (k // 4) + (b[1] // 8) * 64 + (b[1] % 4) * 16 + (b[0] % 16)
    return [d // width, d % width]


# -- the computed layout -------------------------------------------------------------
#
# The aggregate below holds every derived quantity of one build, evaluated once by
# its constructor; kernel code, the schedule model and the host all read the same
# object instead of re-deriving a layout at every query.


@aggregate
@strip_annotate
class KernelLayout:
    """Every tile shape, storage size and register/shared layout of one build.

    Built once as ``KernelLayout(tuning_cfg)``, on the host and in jit code alike,
    and passed explicitly to everything that needs a layout. A value of the whole
    build is a public field, read directly. Per-operand values are stored as
    ``(A, B)`` pairs and read through accessors indexed by operand (0 = A / tokens,
    1 = B / experts), and by ``is_scale`` where the payload and the scale share a
    concept. ``None`` marks a value that does not exist for this build -- a scale
    layout of an unscaled operand, the LDS layout of a register-resident payload.

    Accessors that take a free argument (a K extent, a step, a store extent) are
    evaluated from the stored scalars; there is nothing to precompute for them.
    """

    # Grouped by what they describe: the tile, the operands (payload, scale), the
    # result and the epilogue. Each value keeps its fields together: scalars, then
    # shapes, then layouts, each from HBM through LDS to registers and from the whole
    # tile down to one slot.
    # -- tile --
    _BLOCK_N: gl.constexpr
    _BLOCK_K: gl.constexpr
    _mfma_instr_shape: gl.constexpr
    _weight_gu_hbm_layout: gl.constexpr
    num_warps: gl.constexpr
    #: Payload fragments per BLOCK_K stage.
    num_k_slots: gl.constexpr
    #: Whether the complete BLOCK_K stage is carried across a pipeline step.
    num_prefetch_k_slots: gl.constexpr
    #: M slots per tile. Operand A and the accumulator split by this.
    num_m_slots: gl.constexpr
    #: N slots per tile. Operand B and the accumulator split by this.
    num_n_slots: gl.constexpr
    #: Unswizzled row-major 2-D shared layout: the scale staging tile and the
    #: fused-quant staging buffer.
    identity_lds_layout: gl.constexpr
    # -- operand placement --
    _component_via_lds: gl.constexpr
    # -- operand payload --
    _payload_block_k_storage: gl.constexpr
    _payload_hbm_step: gl.constexpr
    _payload_byte_unit_lds_ok: gl.constexpr
    _payload_lds_shape_tile: gl.constexpr
    _payload_fragment_shape_slot: gl.constexpr
    _payload_hbm_offset_layout: gl.constexpr
    _payload_lds_layout: gl.constexpr
    _payload_fragment_layout: gl.constexpr
    # -- operand scale --
    #: E8M0 elements along K in one pipeline stage.
    scale_block_k_storage: gl.constexpr
    _scale_slot_nonk: gl.constexpr
    _scale_slot_k: gl.constexpr
    _scale_ratio_k_step: gl.constexpr
    _scale_ratio_non_k_slot: gl.constexpr
    _scale_num_slots: gl.constexpr
    _scale_cache_fragments: gl.constexpr
    #: Does the tile match the layout the C++ A-scale shuffle writes?
    sorted_shuffled_ok: gl.constexpr
    _scale_packed_ok: gl.constexpr
    _scale_packed_sel: gl.constexpr
    #: Whether either live MFMA scale operand uses packed dword storage.
    has_mfma_packed_scale: gl.constexpr
    # A shuffled scale is read a whole scale slot at a time and split into per-slot
    # fragments in registers. Its shapes and layouts spell out what, where and how much:
    # `scale_<fragment|lds_view>_<layout|shape>_<slot|scale_slot>`.
    # - `fragment`: registers. `lds_view`: the LDS view a `ds_read` reads from. The two
    #   share a shape, named after the fragment; `lds_shape` is the allocation's.
    # - `slot`: one MFMA step's operand, one non-K slot by one step (packed: by K256,
    #   a dword's K extent). `scale_slot`: what one copy lands, spanning
    #   `scale_ratio_non_k_slot` non-K slots and `scale_slot_k / BLOCK_K` steps.
    #   `_lds._scale_fragments` slices each slot out of a scale slot; when the two
    #   have the same extent, they cover the same tensor.
    # A packed scale (`scale_packed_ok`) never exists as u8 in registers: it is read
    # and carried as i32, a dword's four E8M0 bytes in one register. So its fragment
    # shapes and layouts and its LDS view are the i32 ones, and u8 otherwise; its LDS
    # allocation stays u8. The slot fragment is the shared
    # `fragment_layout(idx, True)`. LDS is never read a slot at a time, so
    # no `lds_view` is per slot.
    _scale_lds_shape_tile: gl.constexpr
    _scale_lds_shape_scale_slot: gl.constexpr
    _scale_lds_shape_slot: gl.constexpr
    _scale_fragment_shape_scale_slot: gl.constexpr
    _scale_fragment_shape_slot: gl.constexpr
    _scale_hbm_offset_layout: gl.constexpr
    _scale_lds_view_layout_scale_slot: gl.constexpr
    _scale_fragment_layout_scale_slot: gl.constexpr
    _scale_fragment_layout: gl.constexpr
    # -- result --
    acc_vgprs_per_lane: gl.constexpr
    #: Shape of one slot's accumulator.
    accumulator_fragment_shape_slot: gl.constexpr
    #: AMDMFMALayout for dot / scaled-dot results in register.
    dot_result_fragment_layout: gl.constexpr
    # -- epilogue --
    #: Emitted columns produced by one N slot.
    output_n_slot: gl.constexpr
    #: Emitted columns produced by the tile's raw BLOCK_N columns.
    output_block_n: gl.constexpr
    epilogue_threads: gl.constexpr
    #: Whether bias/gamma vectors are staged through LDS.
    epilogue_inputs_via_lds: gl.constexpr
    epilogue_input_lds_layout: gl.constexpr
    epilogue_gamma_shape: gl.constexpr
    epilogue_gamma_buffer_load_lds_layout: gl.constexpr
    epilogue_bias_shape: gl.constexpr
    epilogue_bias_buffer_load_lds_layout: gl.constexpr
    quant_staging_rows: gl.constexpr
    #: Whether the fused-quant staging alternates two per-slot LDS banks.
    quant_staging_rotates: gl.constexpr
    # -- resources --
    lds_bytes: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, tuning_cfg):
        """Evaluate every field of one build from its tuning configuration.

        Host and device both construct it from the same configuration, so the host's
        LDS and grid arithmetic and the kernel's tile arithmetic cannot drift. A config
        whose tile lattice is unsound (``ok`` below) still gets its scalar geometry,
        but its layout objects are ``None``; ``_validate.validate_layout`` rejects it.

        One pass in dependency order, not in field order. Each quantity is
        computed once, as a local that later quantities read, and per-operand values
        are ``(A, B)`` tuples. The gates are load-bearing, not an optimisation:
        ``None`` is the documented value of a layout that does not exist for this
        build, and every field is mangled into jit calls, so building a gated value
        unconditionally changes the layout even where its constructor would not raise.
        Some gates also keep a constructor from raising: the raw-scale copy layout at
        BLOCK_K 64 would divide by zero.
        """
        tc = tuning_cfg
        fc = tc.func_cfg
        ops = (0, 1)

        # -- tile and slot geometry -----------------------------------------------
        BM, BN, BK = tc.BLOCK_M.value, tc.BLOCK_N.value, tc.BLOCK_K.value
        SLOT_M, SLOT_N = tc.SLOT_M.value, tc.SLOT_N.value
        instr = tc.mfma_instr_shape.value
        warps = tc.warps_per_cta.value
        tiles_per_warp = tc.tiles_per_warp.value
        is_16x16x128 = list(instr) == [16, 16, 128]
        gu_split = fc.weight_gu_hbm_layout.value == WeightGuHbmLayout.SPLIT
        num_warps = warps[0] * warps[1]
        threads = num_warps * WARP_SIZE
        storage_dtype = tuple(fc.operand_elem_ty(idx) for idx in ops)
        storage_bits = tuple(ty.primitive_bitwidth for ty in storage_dtype)
        has_scale = tuple(fc.has_scale(idx) for idx in ops)
        shuffled = tuple(tc.scale_shuffled(idx) for idx in ops)
        preshuffled = tuple(tc.operand_preshuffled(idx) for idx in ops)

        # Non-K slot extents, which the scale soundness check below reads. Both
        # shuffles (SORTED_SHUFFLED for A, CDNA4_SCALE for B) lay a scale out as one
        # 256 B run per 32-row stripe, so a slot that is a whole number of stripes is a
        # contiguous slice and its scale fragment can use the slot extent. A shuffle
        # always packs K256; a K128 payload stage selects one half, hence the floor on
        # `scale_stage_k`.
        slot_nonk = (SLOT_M, SLOT_N)
        scale_stage_k = max(256, BK)
        scale_slot_nonk = (
            (tc.SCALE_SLOT_M.value or SLOT_M) if shuffled[0] else SLOT_M,
            (tc.SCALE_SLOT_N.value or SLOT_N) if shuffled[1] else SLOT_N,
        )
        scale_slot_k = tuple(
            (tc.SCALE_SLOT_K.value or scale_stage_k) if shuffled[idx] else BK
            for idx in ops
        )

        # Whether the tile lattice is sound enough to build register/shared layouts on:
        # the N/K-independent part of `_validate.validate_layout`, as a predicate. A
        # config that fails it still gets its scalar geometry (the host probes LDS
        # footprints of candidates it has not validated yet) but no layout objects, so
        # the descriptive assertion in `validate_layout` reports it, not a layout
        # constructor.
        ok = not (
            BK % instr[2] or BK % MX_GROUP or BM % SLOT_M or BN % SLOT_N
        ) and not (
            SLOT_M % (instr[0] * warps[0] * tiles_per_warp[0])
            or SLOT_N % (instr[1] * warps[1] * tiles_per_warp[1])
        )
        for idx in ops:
            if ok and shuffled[idx]:
                tile, nonk = (BM, BN)[idx], scale_slot_nonk[idx]
                sk = scale_slot_k[idx]
                ok = (
                    tile >= nonk >= slot_nonk[idx]
                    and not (tile % nonk or nonk % slot_nonk[idx] or nonk % 32)
                    and not (sk < scale_stage_k or sk & (sk - 1) or sk % BK)
                )

        num_k_slots = 1
        num_prefetch_k_slots = 1 if tc.VGPR_PREFETCH_K.value else 0
        num_m_slots = BM // SLOT_M
        num_n_slots = BN // SLOT_N
        num_lds_slots_non_k = (num_m_slots, num_n_slots)
        payload_block_k_storage = tuple(BK // fc.pack_divisor(idx) for idx in ops)
        scale_block_k_storage = BK // MX_GROUP
        payload_fragment_shape_slot = (
            [SLOT_M, payload_block_k_storage[0]],
            [payload_block_k_storage[1], SLOT_N],
        )
        scale_fragment_shape_slot = tuple(
            [nonk, scale_block_k_storage] for nonk in (SLOT_M, SLOT_N)
        )
        accumulator_fragment_shape_slot = [SLOT_M, SLOT_N]
        # One slot, not the whole tile: A splits along M into `num_m_slots` slots of
        # SLOT_M rows, B along N into `num_n_slots` slots of SLOT_N columns.
        # Each slot is allocated, copied and read as an independent unit, so each gets
        # its own efficient padded layout and fully coalesced direct-to-LDS copy.
        # Slicing one tile-wide buffer would give the copy a strided view of a permuted
        # layout. With one K slot per stage, a slot is read whole into one fragment, so
        # it is allocated in the fragment's shape.
        payload_lds_shape_slot = payload_fragment_shape_slot
        scale_lds_shape_slot = tuple(
            [nonk, scale_block_k_storage] for nonk in (SLOT_M, SLOT_N)
        )
        payload_lds_shape_tile = tuple(
            [tc.num_buffers(idx) * num_lds_slots_non_k[idx]]
            + payload_lds_shape_slot[idx]
            for idx in ops
        )
        # 128 bits per lane, in storage elements: 8 for bf16, 16 for fp8 and fp4 (uint8).
        # Per operand, because one value for both would allocate and copy the fp4 weight
        # tile as if it were bf16 in every mixed configuration.
        payload_copy_contiguity = tuple(128 // bits for bits in storage_bits)
        # k_width is the same 128 bits per lane, in storage elements; the tuning field is
        # only an override. Verified against a reference GEMM on both f8f6f4 instruction
        # shapes: k_width 32 has no efficient padded shared layout at all, and 8 with
        # the 16x16x128 shape compiles and runs but computes the wrong answer -- exactly
        # the class of bug no correctness-by-construction argument catches.
        k_width_override = tc.k_width.value
        k_width_for = tuple(
            (
                payload_copy_contiguity[idx]
                if k_width_override is None
                else k_width_override
            )
            for idx in ops
        )
        # A preshuffled B is blocked 16x16 in HBM (see `_blocked_b_hbm_offsets`), so the
        # pointer advances 256 per 16 storage K elements, one whole block.
        payload_hbm_step = tuple(
            step // 16 * 256 if pre else step
            for step, pre in zip(payload_block_k_storage, preshuffled)
        )
        # A preshuffled operand is blocked 16 rows at a time in HBM and the copy has to
        # stay linear, so the unit is that block. A plain one is read two MFMA tiles at
        # a time (hence the `tiles_per_warp >= 2` that `validate_layout` insists on), so
        # the unit is 32 and the row permutation has a 5th base to work with.
        lds_unit_rows = tuple(16 if pre else 32 for pre in preshuffled)
        # `byte_unit_lds_layout` covers only the 16x16x128 B operand shape -- 16x16x128
        # for fp8 and 16x16x256 for fp4, both 8-bit in storage. The 32x32x64 prefill
        # configs, the bf16 ones (16x16x32 / 32x32x16) and any slot too short for a
        # whole unit keep `compute_efficient_padded_shared_layout`. A non-preshuffled
        # 32-row unit also needs the warp to own both of its 16-row MFMA tiles, i.e.
        # `tiles_per_warp >= 2` on that axis. That is not always reachable -- at BLOCK_N
        # 64 with warps (1, 4) the CTA's whole N tile is four 16-column tiles, so no
        # warp can have two -- hence a gate rather than an assert.
        payload_byte_unit_lds_ok = []
        for idx in ops:
            rows = payload_lds_shape_slot[idx][idx]
            k_storage = payload_lds_shape_slot[idx][1 - idx]
            unit_k = LDS_UNIT_K_BYTES * 8 // storage_bits[idx]
            payload_byte_unit_lds_ok.append(
                is_16x16x128
                and storage_bits[idx] == 8
                and bool(preshuffled[idx] or tiles_per_warp[idx] >= 2)
                and rows % lds_unit_rows[idx] == 0
                and k_storage % unit_k == 0
            )
        payload_byte_unit_lds_ok = tuple(payload_byte_unit_lds_ok)

        # -- storage placement ----------------------------------------------------
        # The final LDS-or-register decision per (operand, is_scale): the tuning
        # policy (`KernelTuningConfig.allow_component_via_lds`) allows LDS and the slot
        # admits a direct-to-LDS copy. There is no third placement, so a component not
        # staged through LDS is in registers.
        #
        # A payload slot must give every wavefront one complete 128-bit-per-lane access.
        # One that cannot stays in the register ring; there is deliberately no
        # register-to-LDS fallback.
        #
        # A shuffled scale is fragment-ordered in HBM, so the copy is linear and the LDS
        # tile keeps the same permutation; the read is then a 32-bit ds_read. Staying on
        # LDS also keeps the warp-broadcast B tile a single fetch rather than one per
        # warp. A raw scale slot is bound by CDNA4 direct-to-LDS, which supports only
        # 128 or 32 bits per lane, with each warp writing one contiguous run. That
        # excludes
        #   * slots smaller than 64 lanes * 4 B, which cannot be lowered at all (the
        #     A-scale slot at BLOCK_M == 16 with a short BLOCK_K);
        #   * rows shorter than 8 scales (BLOCK_K < 256): one lane then owns a whole row
        #     and the reg->shared map is more contiguous than the 32-bit access can
        #     cover, which `canLoadDirectToLDS` rejects.
        # Both fall back to a register `buffer_load` straight into the scale fragment
        # layout. That is correct but costs an in-loop register-path global access,
        # which makes every `wait_group` conservative -- so the tuner should prefer a
        # BLOCK_K that keeps the scale on LDS.
        component_via_lds = []
        for idx in ops:
            payload, scale = payload_lds_shape_slot[idx], scale_lds_shape_slot[idx]
            component_via_lds.append(
                (
                    bool(tc.allow_component_via_lds((idx, False)))
                    and (
                        payload[0] * payload[1]
                        >= WARP_SIZE * payload_copy_contiguity[idx] * num_warps
                    ),
                    bool(tc.allow_component_via_lds((idx, True)))
                    and (
                        bool(shuffled[idx])
                        or (
                            scale[0] * scale[1] >= WARP_SIZE * 4
                            and scale[1] % 4 == 0
                            and scale[1] >= 8
                        )
                    ),
                )
            )
        component_via_lds = tuple(component_via_lds)
        # `component_via_lds` is not gated on whether a scale exists (callers gate
        # absent scales with `has_scale`); `scale_via_lds` is.
        payload_via_lds = tuple(placement[0] for placement in component_via_lds)
        scale_via_lds = tuple(
            has_scale[idx] and component_via_lds[idx][1] for idx in ops
        )

        # -- payload layouts ------------------------------------------------------
        dot_result_fragment_layout = None
        payload_fragment_layout = (None, None)
        payload_lds_layout = [None, None]
        dot_operand_buffer_load_lds_layout = [None, None]
        payload_hbm_offset_layout = (None, None)
        if ok:
            # `transposed` is pinned True: it gives each lane 4 consecutive N elements
            # (a 16 B contiguous store, and a lane-local even/odd gate-up pair for the
            # fused activation) instead of 4 strided M rows.
            assert (
                tc.transposed.value
            ), "transposed=True is pinned, see the module docstring"
            dot_result_fragment_layout = gl.amd.AMDMFMALayout(
                version=4,
                instr_shape=instr,
                transposed=True,
                warps_per_cta=warps,
                tiles_per_warp=tiles_per_warp,
            )
            payload_fragment_layout = tuple(
                gl.DotOperandLayout(
                    operand_index=idx,
                    parent=dot_result_fragment_layout,
                    k_width=k_width_for[idx],
                )
                for idx in ops
            )
            for idx in ops:
                if not payload_via_lds[idx]:
                    continue
                shape = payload_lds_shape_slot[idx]
                if payload_byte_unit_lds_ok[idx]:
                    layout = byte_unit_lds_layout(
                        shape,
                        idx,
                        storage_bits[idx],
                        lds_unit_rows[idx],
                        preshuffled[idx],
                    )
                else:
                    # Both operands are K-packed under the memory-layout contract, so
                    # is_k_contig=True and the plain `smem.load(dot_layout)` path
                    # applies -- `load_shared_fp4_repacked` is only needed for an
                    # M/N-packed checkpoint, which none of the four models in scope
                    # produce.
                    layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
                        payload_fragment_layout[idx],
                        shape,
                        storage_dtype[idx],
                        True,
                    )
                    if layout is None:
                        # The helper declines whenever the slot holds a single MFMA tile
                        # along the non-K axis (BLOCK_M == instr_shape[0], i.e. the
                        # whole decode regime): it has no row permutation left to build.
                        # Fall back to a plain identity-mapped padded layout whose
                        # interval is still >= vec * warpSize, which is the condition
                        # `canLoadDirectToLDS` checks, so the copy stays a 128-bit
                        # direct-to-LDS.
                        vec = payload_copy_contiguity[idx]
                        layout = gl.PaddedSharedLayout.with_identity_for(
                            [[WARP_SIZE * vec, vec]],
                            shape,
                            [1, 0] if idx == 0 else [0, 1],
                        )
                payload_lds_layout[idx] = layout
                dot_operand_buffer_load_lds_layout[idx] = _bases_to_distributed(
                    layout.offset_bases,
                    payload_copy_contiguity[idx],
                    num_warps,
                    WARP_SIZE,
                    shape,
                )
            payload_hbm_offset_layout = tuple(
                (
                    dot_operand_buffer_load_lds_layout[idx]
                    if payload_via_lds[idx]
                    else payload_fragment_layout[idx]
                )
                for idx in ops
            )
        payload_lds_layout = tuple(payload_lds_layout)
        dot_operand_buffer_load_lds_layout = tuple(dot_operand_buffer_load_lds_layout)

        # -- scale geometry -------------------------------------------------------
        scale_ratio_k_step = tuple(k // BK for k in scale_slot_k)
        scale_ratio_non_k_slot = tuple(
            scale_slot_nonk[idx] // slot_nonk[idx] for idx in ops
        )
        # A shuffled scale slot narrower than the payload slot (which
        # `validate_layout` rejects) covers no whole slot: it has no tile count, and no
        # LDS footprint where it is staged there.
        scale_slots_ok = tuple(ratio > 0 for ratio in scale_ratio_non_k_slot)
        scale_num_slots = tuple(
            (
                num_lds_slots_non_k[idx] // scale_ratio_non_k_slot[idx]
                if scale_slots_ok[idx]
                else None
            )
            for idx in ops
        )
        scale_read_k_slots = tuple(num_k_slots * ratio for ratio in scale_ratio_k_step)
        scale_cache_fragments = tuple(
            num_lds_slots_non_k[idx] * scale_read_k_slots[idx] for idx in ops
        )
        # A shuffled scale slot is staged in HBM byte order, not fragment order, because
        # direct-to-LDS on gfx9 cannot scatter -- each warp must write coalesced, which
        # a fragment-ordered SharedLinearLayout does not (`canLoadDirectToLDS` rejects
        # it, see TritonAMDGPUToLLVM/Utility.cpp). The bytes therefore sit in (non-K 16,
        # K 4) order while the u8 fragment numbers its registers (K 4, non-K 16);
        # reconciling the two used to cost a v_perm per dword, which is what
        # `scale_packed_ok` avoids by handing the dword to the MFMA whole and naming the
        # byte order in a selector list.
        scale_lds_shape_scale_slot = tuple(
            [scale_slot_nonk[idx] // 32, scale_slot_k[idx]] for idx in ops
        )
        # Staged representation: a shuffled scale fills one flat run per scale slot, a
        # raw one one [slot non-K, BLOCK_K/32] tile per slot.
        scale_lds_shape_tile = []
        for idx in ops:
            if not scale_slots_ok[idx]:
                scale_lds_shape_tile.append(None)
            elif shuffled[idx]:
                scale_lds_shape_tile.append(
                    [tc.num_buffers(idx, True) * scale_num_slots[idx]]
                    + scale_lds_shape_scale_slot[idx]
                )
            else:
                scale_lds_shape_tile.append(
                    [tc.num_buffers(idx, True) * num_lds_slots_non_k[idx]]
                    + scale_lds_shape_slot[idx]
                )
        scale_lds_shape_tile = tuple(scale_lds_shape_tile)
        # ScaleSwizzle.SORTED_SHUFFLED: moe_sort_scales.cuh writes one dword per
        # (chunk, mi, ku, k_lane, n_lane) with
        #     row = chunk*BM + (mi*MN_PACK + im_a)*16 + n_lane
        #     k   = ku*K_PACK*4 + ikxdl*4 + k_lane
        #     byte within the dword = ikxdl*MN_PACK + im_a
        # so n_lane/k_lane index the 64 lanes of a wavefront, im_a/ikxdl the 4 bytes a
        # lane holds, and mi the wave. That is exactly get_mfma_scale_layout's
        # decomposition for a 16x16x128 dot operand with warps_per_cta = (num_warps, 1)
        # and tiles_per_warp = (2, 1) -- verified base-for-base -- so the shuffled
        # buffer is read with a plain contiguous load and no cross-lane movement.
        #
        # The permutation lives in the LDS tile's SharedLinearLayout, which is a
        # property of the byte order the shuffle writes -- not of how warps are split --
        # so the warp arrangement is free. An M slot only has to be a whole number of
        # 32-row stripes: the shuffle writes one 256 B run per stripe, so a slot is a
        # contiguous slice.
        sorted_shuffled_ok = is_16x16x128 and BM % 32 == 0 and SLOT_M % 32 == 0
        scale_fragment_shape_scale_slot = tuple(
            [scale_slot_nonk[idx], scale_slot_k[idx] // MX_GROUP] for idx in ops
        )

        # -- scale layouts --------------------------------------------------------
        # Identity (K-contiguous) staging tile for raw and shuffled scales and both
        # operands alike. For raw E8M0 scales the three LDS goals are not
        # simultaneously satisfiable: with `get_mfma_scale_layout` each lane wants
        # K-scale elements strided by 2 or 4, so the read is `ds_read_u8` regardless,
        # and the identity tile keeps the *write* side (direct-to-LDS) coalesced, which
        # is the side that matters. A shuffled scale slot is a flat run in HBM byte
        # order; its fragment order returns on the read. The fused-quant staging
        # buffer uses the same layout.
        identity_lds_layout = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )
        scale_fragment_layout = [None, None]
        scale_fragment_layout_scale_slot = [None, None]
        dot_operand_scale_buffer_load_lds_layout = [None, None]
        scale_lds_view_layout_scale_slot = [None, None]
        for idx in ops:
            if not (ok and has_scale[idx]):
                continue
            frag = payload_fragment_layout[idx]
            scale_fragment_layout[idx] = gl.amd.cdna4.get_mfma_scale_layout(
                frag, scale_fragment_shape_slot[idx], MX_GROUP
            )
            scale_fragment_layout_scale_slot[idx] = gl.amd.cdna4.get_mfma_scale_layout(
                frag, scale_fragment_shape_scale_slot[idx], MX_GROUP
            )
            if scale_via_lds[idx]:
                # 32 bits per lane. A shuffled scale slot is a flat run, so one warp
                # covers 64 lanes x 4 contiguous bytes of it; a raw row of BLOCK_K/32
                # scales is split 4 per lane.
                lanes_k = WARP_SIZE if shuffled[idx] else scale_block_k_storage // 4
                dot_operand_scale_buffer_load_lds_layout[idx] = gl.BlockedLayout(
                    size_per_thread=[1, 4],
                    threads_per_warp=[WARP_SIZE // lanes_k, lanes_k],
                    warps_per_cta=[num_warps, 1],
                    order=[1, 0],
                )
            # Offset base i is byte 1 << i of one scale slot's flat run, in the
            # CDNA4_SCALE byte order spelled out in `_scale_dword_delta`.
            nonk, k = scale_slot_nonk[idx], scale_slot_k[idx]
            scale_lds_view_layout_scale_slot[idx] = gl.SharedLinearLayout(
                offset_bases=[[16, 0], [0, 4]]  # the 4 bytes of a lane's dword
                + _pow2_bases(0, 2, [1, 2, 4, 8])  # 16 non-K lanes
                + _pow2_bases(1, 2, [1, 2])  # 4 K lanes
                + _pow2_bases(1, 2, _ramp(8, k // MX_GROUP))  # the rest of K
                + _pow2_bases(0, 2, _ramp(32, nonk))  # further 32-row stripes
            )
        scale_fragment_layout = tuple(scale_fragment_layout)
        scale_fragment_layout_scale_slot = tuple(scale_fragment_layout_scale_slot)
        dot_operand_scale_buffer_load_lds_layout = tuple(
            dot_operand_scale_buffer_load_lds_layout
        )
        scale_lds_view_layout_scale_slot = tuple(scale_lds_view_layout_scale_slot)

        # -- packed (i32) scales --------------------------------------------------
        # Every shuffle here puts four scales in a dword, and the matrix instruction can
        # pick one of them with op_sel. So a lane loads the dword whole and hands it to
        # mfma_scaled_packed as an i32, with the byte order named in a selector list --
        # no reinterpret to u8, no register renumbering, no v_perm.
        #
        # CDNA4_SCALE / SORTED_SHUFFLED put non-K +16 and K +4 inside the dword, at byte
        # (nonK) + 2*(K). That fixed pair only works when the fragment happens to hold
        # both steps in registers, so `scale_packed_ok` checks it rather than assuming
        # it: K256 folds K+4 and non-K+16, K128 folds only non-K+16 and selects its K
        # half separately, and each folded step must be register-private. A shuffled
        # operand is a scaled one, so under `ok` its fragment layout exists.
        scale_packed_ok = [False, False]
        for idx in ops:
            if ok and shuffled[idx] and BK in (128, 256):
                regs = [list(b) for b in scale_fragment_layout[idx].reg_bases]
                if BK == 128:
                    scale_packed_ok[idx] = len(regs) >= 1 and regs[0] == [16, 0]
                else:
                    scale_packed_ok[idx] = (
                        len(regs) >= 2 and regs[0] == [0, 4] and regs[1] == [16, 0]
                    )
        scale_packed_ok = tuple(scale_packed_ok)
        # Selectors per K phase. The instruction index counts non-K major and K minor,
        # so its low two bits are the fragment's first two register bases in that order.
        # CDNA4_SCALE numbers the dword's bytes the other way round, hence the
        # transposition; a K128 stage picks the two bytes of its half instead.
        scale_packed_sel = tuple(
            tuple(
                (
                    ([2 * k_phase, 2 * k_phase + 1] if BK == 128 else [0, 2, 1, 3])
                    if scale_packed_ok[idx]
                    else None
                )
                for k_phase in (0, 1)
            )
            for idx in ops
        )
        has_mfma_packed_scale = scale_packed_ok[0] or scale_packed_ok[1]
        packed_scale_fragment_shape_slot = tuple(
            [nonk, scale_stage_k // 128] for nonk in slot_nonk
        )
        packed_scale_fragment_shape_scale_slot = tuple(
            [scale_slot_nonk[idx], scale_slot_k[idx] // 128] for idx in ops
        )
        packed_scale_fragment_layout_slot = [None, None]
        packed_scale_lds_view_layout_scale_slot = [None, None]
        packed_scale_fragment_layout_scale_slot = [None, None]
        for idx in ops:
            if not (ok and has_scale[idx]):
                continue
            frag = gl.amd.cdna4.get_mfma_scale_layout(
                payload_fragment_layout[idx],
                [slot_nonk[idx], scale_stage_k // MX_GROUP],
                MX_GROUP,
            )
            packed_scale_fragment_layout_slot[idx] = gl.DistributedLinearLayout(
                reg_bases=[
                    _scale_dword_delta(b, scale_stage_k) for b in frag.reg_bases[2:]
                ],
                lane_bases=[
                    _scale_dword_delta(b, scale_stage_k) for b in frag.lane_bases
                ],
                warp_bases=[
                    _scale_dword_delta(b, scale_stage_k) for b in frag.warp_bases
                ],
                block_bases=[],
                shape=packed_scale_fragment_shape_slot[idx],
            )
            packed_scale_lds_view_layout_scale_slot[idx] = gl.SharedLinearLayout(
                offset_bases=[
                    _scale_dword_delta(b, scale_slot_k[idx])
                    for b in list(scale_lds_view_layout_scale_slot[idx].offset_bases)[
                        2:
                    ]
                ]
            )
            # The two within-dword axes are identified by value, not position. Wider
            # scale K tiles insert additional K register bases between them.
            frag = scale_fragment_layout_scale_slot[idx]
            k = scale_slot_k[idx]
            within_dword = ([0, 4], [16, 0])
            packed_scale_fragment_layout_scale_slot[idx] = gl.DistributedLinearLayout(
                reg_bases=[
                    _scale_dword_delta(b, k)
                    for b in frag.reg_bases
                    if list(b) not in within_dword
                ],
                lane_bases=[_scale_dword_delta(b, k) for b in frag.lane_bases],
                warp_bases=[_scale_dword_delta(b, k) for b in frag.warp_bases],
                block_bases=[],
                shape=packed_scale_fragment_shape_scale_slot[idx],
            )
        packed_scale_fragment_layout_slot = tuple(packed_scale_fragment_layout_slot)
        packed_scale_lds_view_layout_scale_slot = tuple(
            packed_scale_lds_view_layout_scale_slot
        )
        packed_scale_fragment_layout_scale_slot = tuple(
            packed_scale_fragment_layout_scale_slot
        )

        # The layout one scale slot's HBM offsets are formed in: the copy layout when
        # the scale is staged through LDS, otherwise the register layout the load lands
        # in.
        scale_hbm_offset_layout = [None, None]
        for idx in ops:
            if not (ok and has_scale[idx]):
                continue
            if scale_via_lds[idx]:
                layout = dot_operand_scale_buffer_load_lds_layout[idx]
            elif not shuffled[idx]:
                layout = scale_fragment_layout[idx]
            elif scale_packed_ok[idx]:
                layout = packed_scale_fragment_layout_scale_slot[idx]
            else:
                # get_mfma_scale_layout puts the K-group bit before the non-K bit in
                # reg_bases, so a lane's four bytes land in registers in address order
                # 0, +2, +1, +3. A widened load fills registers in ascending order;
                # issue it with those two bases swapped, then convert with only a
                # lane-local register renumber because the lane and warp bases stay
                # unchanged.
                frag = scale_fragment_layout_scale_slot[idx]
                lead = [[16, 0], [0, 4]]
                rest = [list(b) for b in frag.reg_bases if list(b) not in lead]
                layout = gl.DistributedLinearLayout(
                    reg_bases=lead + rest,
                    lane_bases=[list(b) for b in frag.lane_bases],
                    warp_bases=[list(b) for b in frag.warp_bases],
                    block_bases=[],
                    shape=list(frag.shape),
                )
            scale_hbm_offset_layout[idx] = layout
        scale_hbm_offset_layout = tuple(scale_hbm_offset_layout)

        # A packed scale never exists as u8 in registers, so from here on its
        # fragment shapes and layouts and its LDS view are the i32 ones. The u8 ones
        # were only needed above, to test and derive the packing.
        scale_fragment_shape_slot = tuple(
            (
                packed_scale_fragment_shape_slot[idx]
                if scale_packed_ok[idx]
                else scale_fragment_shape_slot[idx]
            )
            for idx in ops
        )
        scale_fragment_shape_scale_slot = tuple(
            (
                packed_scale_fragment_shape_scale_slot[idx]
                if scale_packed_ok[idx]
                else scale_fragment_shape_scale_slot[idx]
            )
            for idx in ops
        )
        scale_lds_view_layout_scale_slot = tuple(
            (
                packed_scale_lds_view_layout_scale_slot[idx]
                if scale_packed_ok[idx]
                else scale_lds_view_layout_scale_slot[idx]
            )
            for idx in ops
        )
        scale_fragment_layout_scale_slot = tuple(
            (
                packed_scale_fragment_layout_scale_slot[idx]
                if scale_packed_ok[idx]
                else scale_fragment_layout_scale_slot[idx]
            )
            for idx in ops
        )
        scale_fragment_layout = tuple(
            (
                packed_scale_fragment_layout_slot[idx]
                if scale_packed_ok[idx]
                else scale_fragment_layout[idx]
            )
            for idx in ops
        )

        # -- epilogue -------------------------------------------------------------
        output_n_slot = SLOT_N // _n_reduction_slot(fc)
        output_block_n = BN // fc.activation_reduction_n()
        epilogue_threads = threads
        epilogue_bias_per_thread = BN // epilogue_threads
        # No knob controls this placement: the bias/gamma vectors go through LDS
        # whenever the copy is legal for the tile.
        epilogue_inputs_via_lds = (
            epilogue_threads >= BM
            and BN % epilogue_threads == 0
            and epilogue_bias_per_thread in (1, 4)
        )
        epilogue_input_lds_layout = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[0]
        )
        epilogue_gamma_buffer_load_lds_layout = None
        epilogue_bias_buffer_load_lds_layout = None
        if epilogue_inputs_via_lds:
            epilogue_gamma_buffer_load_lds_layout = gl.BlockedLayout(
                [1], [WARP_SIZE], [num_warps], [0]
            )
            epilogue_bias_buffer_load_lds_layout = gl.BlockedLayout(
                [epilogue_bias_per_thread], [WARP_SIZE], [num_warps], [0]
            )
        epilogue_gamma_shape = [epilogue_threads]
        epilogue_bias_shape = [BN]
        quant_staging_rows = BM if gu_split else SLOT_M
        quant_staging_rotates = fc.output_quant.value is not None and not gu_split

        # -- resources ------------------------------------------------------------
        # LDS footprint of the components staged there. A staged scale without a tile
        # count (see `scale_slots_ok`) has no footprint, so neither has the build.
        lds_bytes = None
        if all(scale_slots_ok[idx] or not scale_via_lds[idx] for idx in ops):
            lds_bytes = 0
            for idx in ops:
                if payload_via_lds[idx]:
                    # Ring depth x slots x one slot, in storage elements.
                    tile = payload_lds_shape_tile[idx]
                    lds_bytes += tile[0] * tile[1] * tile[2] * (storage_bits[idx] // 8)
                if scale_via_lds[idx]:
                    tile = scale_lds_shape_tile[idx]
                    lds_bytes += tile[0] * tile[1] * tile[2]
        acc_vgprs_per_lane = BM * BN // threads

        # -- fields ---------------------------------------------------------------
        # Stored last, in declaration order: an aggregate accepts nothing but its
        # annotated fields, so every intermediate above stays a local.

        # tile
        self._BLOCK_N = gl.constexpr(BN)
        self._BLOCK_K = gl.constexpr(BK)
        self._mfma_instr_shape = gl.constexpr(list(instr))
        self._weight_gu_hbm_layout = gl.constexpr(fc.weight_gu_hbm_layout.value)
        self.num_warps = gl.constexpr(num_warps)
        self.num_k_slots = gl.constexpr(num_k_slots)
        self.num_prefetch_k_slots = gl.constexpr(num_prefetch_k_slots)
        self.num_m_slots = gl.constexpr(num_m_slots)
        self.num_n_slots = gl.constexpr(num_n_slots)
        self.identity_lds_layout = gl.constexpr(identity_lds_layout)

        # operand placement
        self._component_via_lds = gl.constexpr(component_via_lds)

        # operand payload
        self._payload_block_k_storage = gl.constexpr(payload_block_k_storage)
        self._payload_hbm_step = gl.constexpr(payload_hbm_step)
        self._payload_byte_unit_lds_ok = gl.constexpr(payload_byte_unit_lds_ok)
        self._payload_lds_shape_tile = gl.constexpr(payload_lds_shape_tile)
        self._payload_fragment_shape_slot = gl.constexpr(payload_fragment_shape_slot)
        self._payload_hbm_offset_layout = gl.constexpr(payload_hbm_offset_layout)
        self._payload_lds_layout = gl.constexpr(payload_lds_layout)
        self._payload_fragment_layout = gl.constexpr(payload_fragment_layout)

        # operand scale
        self.scale_block_k_storage = gl.constexpr(scale_block_k_storage)
        self._scale_slot_nonk = gl.constexpr(scale_slot_nonk)
        self._scale_slot_k = gl.constexpr(scale_slot_k)
        self._scale_ratio_k_step = gl.constexpr(scale_ratio_k_step)
        self._scale_ratio_non_k_slot = gl.constexpr(scale_ratio_non_k_slot)
        self._scale_num_slots = gl.constexpr(scale_num_slots)
        self._scale_cache_fragments = gl.constexpr(scale_cache_fragments)
        self.sorted_shuffled_ok = gl.constexpr(sorted_shuffled_ok)
        self._scale_packed_ok = gl.constexpr(scale_packed_ok)
        self._scale_packed_sel = gl.constexpr(scale_packed_sel)
        self.has_mfma_packed_scale = gl.constexpr(has_mfma_packed_scale)
        self._scale_lds_shape_tile = gl.constexpr(scale_lds_shape_tile)
        self._scale_lds_shape_scale_slot = gl.constexpr(scale_lds_shape_scale_slot)
        self._scale_lds_shape_slot = gl.constexpr(scale_lds_shape_slot)
        self._scale_fragment_shape_scale_slot = gl.constexpr(
            scale_fragment_shape_scale_slot
        )
        self._scale_fragment_shape_slot = gl.constexpr(scale_fragment_shape_slot)
        self._scale_hbm_offset_layout = gl.constexpr(scale_hbm_offset_layout)
        self._scale_lds_view_layout_scale_slot = gl.constexpr(
            scale_lds_view_layout_scale_slot
        )
        self._scale_fragment_layout_scale_slot = gl.constexpr(
            scale_fragment_layout_scale_slot
        )
        self._scale_fragment_layout = gl.constexpr(scale_fragment_layout)

        # result
        self.acc_vgprs_per_lane = gl.constexpr(acc_vgprs_per_lane)
        self.accumulator_fragment_shape_slot = gl.constexpr(
            accumulator_fragment_shape_slot
        )
        self.dot_result_fragment_layout = gl.constexpr(dot_result_fragment_layout)

        # epilogue
        self.output_n_slot = gl.constexpr(output_n_slot)
        self.output_block_n = gl.constexpr(output_block_n)
        self.epilogue_threads = gl.constexpr(epilogue_threads)
        self.epilogue_inputs_via_lds = gl.constexpr(epilogue_inputs_via_lds)
        self.epilogue_input_lds_layout = gl.constexpr(epilogue_input_lds_layout)
        self.epilogue_gamma_shape = gl.constexpr(epilogue_gamma_shape)
        self.epilogue_gamma_buffer_load_lds_layout = gl.constexpr(
            epilogue_gamma_buffer_load_lds_layout
        )
        self.epilogue_bias_shape = gl.constexpr(epilogue_bias_shape)
        self.epilogue_bias_buffer_load_lds_layout = gl.constexpr(
            epilogue_bias_buffer_load_lds_layout
        )
        self.quant_staging_rows = gl.constexpr(quant_staging_rows)
        self.quant_staging_rotates = gl.constexpr(quant_staging_rotates)

        # resources
        self.lds_bytes = gl.constexpr(lds_bytes)

    # -- tile and slot geometry --------------------------------------------------

    @gluon.constexpr_function
    def grid_N(self, N):
        """Number of N tiles. Callable from BOTH host and device.

        Host code gets a plain int, ready for the grid tuple. grid-M comes from the
        routing metadata (``RoutingData.n_blocks``), never from ``cdiv``.
        """
        BN = self._BLOCK_N.value
        assert N % BN == 0, f"N ({N}) must be a multiple of BLOCK_N ({BN})"
        return N // BN

    @gluon.constexpr_function
    def num_k_tiles(self, K):
        BK = self._BLOCK_K.value
        assert K % BK == 0, f"K ({K}) must be a multiple of BLOCK_K ({BK})"
        return K // BK

    @gluon.constexpr_function
    def num_lds_slots_non_k(self, idx):
        """Slots along one operand's non-K axis, each copied and read independently."""
        return (self.num_m_slots.value, self.num_n_slots.value)[idx]

    @gluon.constexpr_function
    def payload_block_k_storage(self, idx):
        """Payload storage elements along K in one pipeline stage."""
        return self._payload_block_k_storage.value[idx]

    @gluon.constexpr_function
    def fragment_shape_slot(self, idx, is_scale):
        """Register shape for one non-K slot and K slot.

        Payload storage elements, or for a scale E8M0 bytes, or i32 dwords with
        both K128 halves when :meth:`scale_packed_ok`.
        """
        if is_scale:
            return list(self._scale_fragment_shape_slot.value[idx])
        return list(self._payload_fragment_shape_slot.value[idx])

    @gluon.constexpr_function
    def lds_shape_slot(self, idx, is_scale):
        """Shared shape of one slot.

        A payload's is a storage shape, its fragment's; a scale's is E8M0,
        ``[slot non-K extent, BLOCK_K/32]``.
        """
        if is_scale:
            return list(self._scale_lds_shape_slot.value[idx])
        return list(self._payload_fragment_shape_slot.value[idx])

    @gluon.constexpr_function
    def lds_shape_tile(self, idx, is_scale):
        """Full LDS shape of the tile, including ring and slot axes.

        A scale's is in its staged representation.
        """
        if is_scale:
            return list(self._scale_lds_shape_tile.value[idx])
        return list(self._payload_lds_shape_tile.value[idx])

    @gluon.constexpr_function
    def payload_hbm_step(self, idx):
        """Base-pointer increment for one payload K stage, in storage elements."""
        return self._payload_hbm_step.value[idx]

    @gluon.constexpr_function
    def payload_byte_unit_lds_ok(self, idx):
        """Does :func:`byte_unit_lds_layout` apply to this operand?"""
        return self._payload_byte_unit_lds_ok.value[idx]

    # -- storage placement --------------------------------------------------------
    # Resolved: tuning's ``allow_component_via_lds`` policy and the automatic register
    # fallback for slots the direct-to-LDS copy cannot take. ``component_in_reg`` is
    # exactly the complement of ``component_via_lds``. Neither is gated on whether a
    # scale exists; callers gate absent scales with ``has_scale``.

    @gluon.constexpr_function
    def component_via_lds(self, component):
        """Whether ``(operand, is_scale)`` is staged through LDS."""
        operand, is_scale = component[0], bool(component[1])
        assert operand in (0, 1), "component operand must be 0 (A) or 1 (B)"
        return self._component_via_lds.value[operand][int(is_scale)]

    @gluon.constexpr_function
    def component_in_reg(self, component):
        """Whether ``(operand, is_scale)`` is loaded directly into registers."""
        return not self.component_via_lds(component)

    # -- payload and scale layouts ------------------------------------------------

    @gluon.constexpr_function
    def fragment_layout(self, idx, is_scale):
        """Register layout of one slot fragment; i32 for a packed scale."""
        if is_scale:
            return self._scale_fragment_layout.value[idx]
        return self._payload_fragment_layout.value[idx]

    @gluon.constexpr_function
    def lds_layout(self, idx, is_scale):
        """Shared layout of a component staged through LDS.

        Padded for a payload; :attr:`identity_lds_layout` for a scale, a K-contiguous
        tile shared by both operands.
        """
        if is_scale:
            return self.identity_lds_layout.value
        return self._payload_lds_layout.value[idx]

    @gluon.constexpr_function
    def hbm_offset_layout(self, idx, is_scale):
        """Layout used to form one payload slot's or scale slot's HBM offsets."""
        if is_scale:
            return self._scale_hbm_offset_layout.value[idx]
        return self._payload_hbm_offset_layout.value[idx]

    # -- scale geometry -----------------------------------------------------------

    @gluon.constexpr_function
    def slot_nonk(self, idx):
        """Non-K extent of the operand's non-K slot, ``SLOT_M`` or ``SLOT_N``.

        The accumulator slot's extent along the same axis.
        """
        return self.accumulator_fragment_shape_slot.value[idx]

    @gluon.constexpr_function
    def scale_slot_nonk(self, idx):
        """Non-K extent of one scale slot; a raw scale's is its payload slot's."""
        return self._scale_slot_nonk.value[idx]

    @gluon.constexpr_function
    def scale_slot_k(self, idx):
        """K extent of one scale slot, in logical payload elements, not E8M0 bytes."""
        return self._scale_slot_k.value[idx]

    @gluon.constexpr_function
    def scale_ratio_k_step(self, idx):
        """Number of payload K steps covered by one scale load."""
        return self._scale_ratio_k_step.value[idx]

    @gluon.constexpr_function
    def scale_ratio_non_k_slot(self, idx):
        """Number of payload non-K slots covered by one scale load."""
        return self._scale_ratio_non_k_slot.value[idx]

    @gluon.constexpr_function
    def scale_num_slots(self, idx):
        return self._scale_num_slots.value[idx]

    @gluon.constexpr_function
    def scale_read_k_slots(self, idx):
        """K slots one scale slot covers: one per K slot of each step it spans."""
        return self.num_k_slots.value * self._scale_ratio_k_step.value[idx]

    @gluon.constexpr_function
    def scale_cache_fragments(self, idx):
        return self._scale_cache_fragments.value[idx]

    @gluon.constexpr_function
    def scale_lds_shape_scale_slot(self, idx):
        """LDS allocation shape of one shuffled scale slot: a 256 B run per stripe."""
        return list(self._scale_lds_shape_scale_slot.value[idx])

    @gluon.constexpr_function
    def scale_fragment_shape_scale_slot(self, idx):
        """Shape of one scale slot, its fragment's and its LDS view's.

        In i32 dwords when :meth:`scale_packed_ok`, else in u8.
        """
        return list(self._scale_fragment_shape_scale_slot.value[idx])

    @gluon.constexpr_function
    def scale_packed_ok(self, idx):
        """Can this operand's scale fragment be fed as one dword per lane?"""
        return self._scale_packed_ok.value[idx]

    @gluon.constexpr_function
    def scale_fragment_dtype(self, idx):
        """Element type of the scale's fragments and LDS view: i32 when packed."""
        return gl.int32 if self._scale_packed_ok.value[idx] else gl.uint8

    @gluon.constexpr_function
    def scale_packed_sel(self, idx, k_phase=0):
        """Byte selectors in MFMA order, for one K128 half or the full K256 word."""
        return self._scale_packed_sel.value[idx][k_phase]

    @gluon.constexpr_function
    def mfma_scale_selector(self, idx, k_phase=0):
        """Packed-scale selector for the live MFMA path, or ``None``."""
        return self.scale_packed_sel(idx, k_phase)

    # -- scale layouts ------------------------------------------------------------

    @gluon.constexpr_function
    def scale_lds_view_layout_scale_slot(self, idx):
        """[non-K, K] view of one scale slot's flat LDS run, in fragment order.

        In i32 dwords when :meth:`scale_packed_ok`, an offset bit per dword bit;
        else in u8.
        """
        return self._scale_lds_view_layout_scale_slot.value[idx]

    @gluon.constexpr_function
    def scale_fragment_layout_scale_slot(self, idx):
        """One whole scale slot's scale, as a single copy or read lands it.

        In i32 dwords when :meth:`scale_packed_ok`, else in u8. Spans
        ``scale_ratio_non_k_slot`` non-K slots and ``scale_slot_k / BLOCK_K``
        steps; each slot fragment is sliced out of it.
        """
        return self._scale_fragment_layout_scale_slot.value[idx]

    # -- epilogue and emitted-output layouts --------------------------------------

    @gluon.constexpr_function
    def result_store_layout(self, extent_m, extent_n, storage_bits):
        """Blocked layout for the masked global store of one result slot.

        ``vec`` is capped by the elements each thread actually owns, not just by the
        128-bit access width. Without that cap a narrow tile is *over-covered*: the
        MXFP4 payload tile is [64, 32] uint8 = 2048 elements, but 256 threads x 16
        would be 4096, so the layout spanned 64x64 and the surplus warp column held a
        replica -- every payload store issued twice, writing the same bytes. With the
        cap it tiles exactly at both 4 and 8 waves.

        Covering *less* than the tile is fine and common (the bf16 path does): the
        layout simply repeats, one register set per repetition. Only over-coverage
        costs anything. It remains unavoidable for the E8M0 scale tile ([64, 2] is 128
        elements against 256 threads), which is why that one is still 2x over.
        """
        num_warps = self.num_warps.value
        ept = max(1, (extent_m * extent_n) // (num_warps * WARP_SIZE))
        vec = max(1, min(128 // storage_bits, extent_n, ept))
        lanes_n = max(1, min(WARP_SIZE, extent_n // vec))
        lanes_m = max(1, WARP_SIZE // lanes_n)
        warps_m = max(1, min(num_warps, extent_m // lanes_m))
        warps_n = max(1, num_warps // warps_m)
        return gl.BlockedLayout(
            size_per_thread=[1, vec],
            threads_per_warp=[lanes_m, lanes_n],
            warps_per_cta=[warps_m, warps_n],
            order=[1, 0],
        )

    @gluon.constexpr_function
    def quant_payload_storage_shape(self, rows, output_n=None):
        """E2M1 storage shape for ``rows`` by emitted-output columns."""
        output_n = self.output_n_slot.value if output_n is None else output_n
        return [rows, output_n // 2]

    @gluon.constexpr_function
    def quant_scale_shape(self, rows, output_n=None):
        """E8M0 shape for ``rows`` by emitted-output columns."""
        output_n = self.output_n_slot.value if output_n is None else output_n
        return [rows, output_n // MX_GROUP]

    @gluon.constexpr_function
    def quant_amax_lane_elems(self, extent_m=None, output_n=None):
        """Inner lane-local width for the split MXFP4 amax reduction.

        The amax is an fp32 in-lane reduction followed by an integer cross-lane
        reduction. The fp32 half keeps ``abs`` as a free source modifier, while the
        integer half avoids the NaN-canonicalizing ``v_max_f32 x, x, x`` otherwise
        emitted before every cross-lane step.

        For the interleaved result this remains a conservative arithmetic proxy for
        the post-activation layout. The tuned four-wave TTGIR reshapes to
        ``tensor<64x2x2x16xf32>`` with reduction-axis register bases ``{1, 8}`` and
        lane bases ``{2, 4}``; returning 16 contains each lane's full span. A smaller
        split adds permlane work, while an overly fine one loses the three-input max
        fusion and adds copies around destructive lane swaps.

        Gate/up-split is derivable directly: one lane owns the transposed MFMA quad,
        ``instr_m * instr_n / WARP_SIZE`` consecutive emitted columns (four for
        16x16), before the first lane base.
        """
        extent_m = (
            self.accumulator_fragment_shape_slot.value[0]
            if extent_m is None
            else extent_m
        )
        output_n = self.output_n_slot.value if output_n is None else output_n
        if self._weight_gu_hbm_layout.value == WeightGuHbmLayout.SPLIT:
            instr = self._mfma_instr_shape.value
            return max(1, min(MX_GROUP, (instr[0] * instr[1]) // WARP_SIZE))
        per_lane = (extent_m * output_n) // (self.num_warps.value * WARP_SIZE)
        return min(MX_GROUP, max(1, per_lane))

    # -- resources ----------------------------------------------------------------

    # -- raw-N mapping and HBM offsets (device) -----------------------------------
    # Jit methods, called on the layout from kernel code. The raw-N mapping reads only
    # layout fields. The offset grids also read knobs the layout does not hold (the
    # gather, preshuffle and scale-shuffle flags), so they take ``func_cfg`` and
    # ``tuning_cfg`` as arguments rather than adding layout fields: each field is
    # mangled into every jit call that carries the layout.

    @gluon.jit
    def n_start(self, pid_n, n_slot_idx: gl.constexpr, N):
        """Raw-N start of slot ``n_slot_idx`` under either ``WeightGuHbmLayout``.

        Interleaved tiles are contiguous. Split tiles cover the same emitted channels but
        read gate and up from separate halves of N.
        """
        # One N slot's raw-N width (SLOT_N) is the accumulator tile's N extent.
        n_slot: gl.constexpr = self.accumulator_fragment_shape_slot[1]
        if require_constexpr(self._weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
            out = pid_n * n_slot + n_slot_idx * (N // 2)
        else:
            out = pid_n * self._BLOCK_N + n_slot_idx * n_slot
        return out

    @gluon.jit
    def n_split_offs(self, pid_n, i, N):
        """Map a whole raw-N CTA index tensor through the ``WeightGuHbmLayout``."""
        n_slot: gl.constexpr = self.accumulator_fragment_shape_slot[1]
        if require_constexpr(self._weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
            out = pid_n * n_slot + (i // n_slot) * (N // 2) + i % n_slot
        else:
            out = pid_n * self._BLOCK_N + i
        return out

    @gluon.jit
    def a_payload_hbm_offsets(
        self, a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg
    ):
        """Gathered A payload offsets, one copy-layout grid per M slot."""
        M_TILE: gl.constexpr = tuning_cfg.BLOCK_M
        M_SLOT: gl.constexpr = tuning_cfg.SLOT_M
        NUM_M_SLOTS: gl.constexpr = self.num_m_slots
        PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = self.payload_block_k_storage(0)
        layout_a_slot: gl.constexpr = self.hbm_offset_layout(0, False)
        a_hbm_offsets_tile = ()
        for m_slot_idx in gl.static_range(NUM_M_SLOTS):
            rows_a_slot = _gather_rows(
                rt,
                block_id,
                M_e,
                start_m,
                gl.SliceLayout(1, layout_a_slot),
                M_TILE,
                m_slot_idx * M_SLOT,
                M_SLOT,
                func_cfg.has_gather,
            )
            a_hbm_offsets_tile = a_hbm_offsets_tile + (
                rows_a_slot[:, None] * a.stride_m
                + gl.arange(
                    0,
                    PAYLOAD_BLOCK_K_STORAGE,
                    layout=gl.SliceLayout(0, layout_a_slot),
                )[None, :],
            )

        return a_hbm_offsets_tile

    @gluon.jit
    def b_payload_hbm_offsets(self, b, pid_n, N, K, func_cfg, tuning_cfg):
        """B payload offsets for plain or preshuffled weights, one grid per N slot."""
        N_SLOT: gl.constexpr = tuning_cfg.SLOT_N
        NUM_N_SLOTS: gl.constexpr = self.num_n_slots
        PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = self.payload_block_k_storage(1)
        layout_b_slot: gl.constexpr = self.hbm_offset_layout(1, False)
        K_STORAGE: gl.constexpr = K // func_cfg.pack_divisor(1)
        b_hbm_offsets_tile = ()
        for n_slot_idx in gl.static_range(NUM_N_SLOTS):
            if require_constexpr(tuning_cfg.B_PRESHUFFLED):
                b_hbm_offsets_tile = b_hbm_offsets_tile + (
                    _blocked_b_hbm_offsets(
                        layout_b_slot,
                        PAYLOAD_BLOCK_K_STORAGE,
                        self.n_start(pid_n, n_slot_idx, N),
                        N_SLOT,
                        K_STORAGE,
                    ),
                )
            else:
                b_hbm_offsets_tile = b_hbm_offsets_tile + (
                    gl.arange(
                        0,
                        PAYLOAD_BLOCK_K_STORAGE,
                        layout=gl.SliceLayout(1, layout_b_slot),
                    )[:, None]
                    * b.stride_k
                    + (
                        self.n_start(pid_n, n_slot_idx, N)
                        + gl.arange(0, N_SLOT, layout=gl.SliceLayout(0, layout_b_slot))
                    )[None, :]
                    * b.stride_n,
                )

        return b_hbm_offsets_tile

    @gluon.jit
    def a_scale_hbm_offsets(
        self, a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg
    ):
        """Byte offsets of the A-scale grids, one per M slot (``None`` if unscaled).

        Two shapes, picked by ``A_SCALE_SORTED_SHUFFLED``: a flat run into the
        moe_sort_scales pre-pass output, or a gathered
        ``[M_SLOT, SCALE_BLOCK_K_STORAGE]`` grid
        over the raw ``(M, K/32)`` tensor. The caller sees a tuple indexed by M slot.
        """
        if require_constexpr(not func_cfg.has_scale(0)):
            return _NO_SCALE

        M_TILE: gl.constexpr = tuning_cfg.BLOCK_M
        M_SLOT: gl.constexpr = tuning_cfg.SLOT_M
        NUM_M_SLOTS: gl.constexpr = self.num_m_slots
        SCALE_BLOCK_K_STORAGE: gl.constexpr = self.scale_block_k_storage
        layout_a_scale_slot: gl.constexpr = self.hbm_offset_layout(0, True)
        SCALE_M: gl.constexpr = self.scale_slot_nonk(0)
        SCALE_K: gl.constexpr = self.scale_slot_k(0)
        SCALE_RATIO: gl.constexpr = self.scale_ratio_non_k_slot(0)

        if require_constexpr(
            tuning_cfg.scale_shuffled(0) and self.component_in_reg((0, True))
        ):
            a_scale_hbm_offsets_tile = ()
            for m_slot_idx in gl.static_range(NUM_M_SLOTS):
                a_scale_hbm_offsets_tile = a_scale_hbm_offsets_tile + (
                    _shuffled_scale_register_offsets(
                        layout_a_scale_slot,
                        pid_m * M_TILE + (m_slot_idx // SCALE_RATIO) * SCALE_M,
                        SCALE_M,
                        SCALE_K // 32,
                        K,
                        self.scale_packed_ok(0),
                    ),
                )
        elif require_constexpr(tuning_cfg.A_SCALE_SORTED_SHUFFLED):
            # moe_sort_scales has already applied the gather and the fragment permute,
            # so there is no table lookup and no per-row stride here: the tile is one
            # contiguous run and `layout_a_scale_slot` (the fragment layout) already
            # places each lane on the byte it needs. A K256 scale group spans 256 B per
            # row stripe; K128 payload stages reuse the same group before advancing the
            # pointer.
            gl.static_assert(
                self.sorted_shuffled_ok,
                "A_SCALE_SORTED_SHUFFLED requires MFMA 16x16x128 "
                "and whole 32-row A stripes",
            )
            # The shuffle indexes the *padded* row space -- expert e starts at
            # token_offs_pad[e] whole blocks -- while start_m is the raw offset and is
            # not block-aligned. token_offs_pad[e] + block_id is exactly pid_m, which
            # block_pid_map is built to enumerate, so the chunk index is free here.
            #
            # The staging copy is flat: a stage's tile is one 256 B run per 32-row
            # stripe, and consecutive stripes are one whole K sweep apart. Keeping the
            # copy 1-D is what lets it vectorise -- in the [M_TILE, SK] view a lane's four
            # bytes straddle both axes and nothing can widen the access.
            # One scale slot per SCALE_RATIO M slots. It is a whole number of 32-row
            # stripes and stripes are contiguous, so it is just the run starting
            # (m_slot_idx // SCALE_RATIO) * (SCALE_M // 32) stripes into this pid_m's
            # chunk. Slots sharing a scale slot get the same base offsets; only the
            # owner actually issues the copy.
            SCALE_FILL_SHAPE: gl.constexpr = self.scale_lds_shape_scale_slot(0)
            a_scale_hbm_offsets_tile = ()
            for m_slot_idx in gl.static_range(NUM_M_SLOTS):
                a_scale_hbm_offsets_tile = a_scale_hbm_offsets_tile + (
                    _shuffled_scale_stage_offsets(
                        layout_a_scale_slot,
                        pid_m * M_TILE + (m_slot_idx // SCALE_RATIO) * SCALE_M,
                        SCALE_FILL_SHAPE[0],
                        K,
                        SCALE_K,
                    ),
                )
        else:
            a_scale_hbm_offsets_tile = ()
            for m_slot_idx in gl.static_range(NUM_M_SLOTS):
                rows_a_scale_slot = _gather_rows(
                    rt,
                    block_id,
                    M_e,
                    start_m,
                    gl.SliceLayout(1, layout_a_scale_slot),
                    M_TILE,
                    m_slot_idx * M_SLOT,
                    M_SLOT,
                    func_cfg.has_gather,
                )
                a_scale_hbm_offsets_tile = a_scale_hbm_offsets_tile + (
                    rows_a_scale_slot[:, None] * a.scale_stride_m
                    + gl.arange(
                        0,
                        SCALE_BLOCK_K_STORAGE,
                        layout=gl.SliceLayout(0, layout_a_scale_slot),
                    )[None, :]
                    * a.scale_stride_k,
                )
        return a_scale_hbm_offsets_tile

    @gluon.jit
    def b_scale_hbm_offsets(self, b, pid_n, N, K, func_cfg, tuning_cfg):
        """Byte offsets of the B-scale grids, one per N slot (``None`` if unscaled).

        Either a flat CDNA4_SCALE preshuffle run for staging or the raw ``(K/32, N)``
        grid. The caller receives a tuple indexed by N slot.
        """
        if require_constexpr(not func_cfg.has_scale(1)):
            return _NO_SCALE

        N_SLOT: gl.constexpr = tuning_cfg.SLOT_N
        NUM_N_SLOTS: gl.constexpr = self.num_n_slots
        SCALE_BLOCK_K_STORAGE: gl.constexpr = self.scale_block_k_storage
        layout_b_scale_slot: gl.constexpr = self.hbm_offset_layout(1, True)
        SCALE_N: gl.constexpr = self.scale_slot_nonk(1)
        SCALE_K: gl.constexpr = self.scale_slot_k(1)
        SCALE_RATIO: gl.constexpr = self.scale_ratio_non_k_slot(1)
        SPLIT_EXTENT: gl.constexpr = (
            N_SLOT
            if func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT and SCALE_RATIO > 1
            else 0
        )

        if require_constexpr(
            tuning_cfg.scale_shuffled(1) and self.component_in_reg((1, True))
        ):
            b_scale_hbm_offsets_tile = ()
            for n_slot_idx in gl.static_range(NUM_N_SLOTS):
                b_scale_hbm_offsets_tile = b_scale_hbm_offsets_tile + (
                    _shuffled_scale_register_offsets(
                        layout_b_scale_slot,
                        self.n_start(pid_n, n_slot_idx // SCALE_RATIO * SCALE_RATIO, N),
                        SCALE_N,
                        SCALE_K // 32,
                        K,
                        self.scale_packed_ok(1),
                        SPLIT_EXTENT,
                        N // 2,
                    ),
                )
        elif require_constexpr(tuning_cfg.B_SCALE_SHUFFLED):
            # utils/shuffle.py::shuffle_scale_moe (CDNA4_SCALE) has already permuted the
            # weight scales into MFMA fragment order: per expert the tile is
            # (N/32, K) bytes, and within a 32-row stripe lane L of a stage reads the
            # dword at stage*256 + L*4. Same shape as the A-side shuffle, but done
            # offline -- the weights are static, so this costs nothing at run time.
            # One scale slot per SCALE_RATIO N slots; stripes are contiguous, so it
            # starts where its first N slot does.
            SCALE_FILL_SHAPE: gl.constexpr = self.scale_lds_shape_scale_slot(1)
            b_scale_hbm_offsets_tile = ()
            for n_slot_idx in gl.static_range(NUM_N_SLOTS):
                b_scale_hbm_offsets_tile = b_scale_hbm_offsets_tile + (
                    _shuffled_scale_stage_offsets(
                        layout_b_scale_slot,
                        self.n_start(pid_n, n_slot_idx // SCALE_RATIO * SCALE_RATIO, N),
                        SCALE_FILL_SHAPE[0],
                        K,
                        SCALE_K,
                        SPLIT_EXTENT,
                        N // 2,
                    ),
                )
        else:
            b_scale_hbm_offsets_tile = ()
            for n_slot_idx in gl.static_range(NUM_N_SLOTS):
                b_scale_hbm_offsets_tile = b_scale_hbm_offsets_tile + (
                    (
                        self.n_start(pid_n, n_slot_idx, N)
                        + gl.arange(
                            0,
                            N_SLOT,
                            layout=gl.SliceLayout(1, layout_b_scale_slot),
                        )
                    )[:, None]
                    * b.scale_stride_n
                    + gl.arange(
                        0,
                        SCALE_BLOCK_K_STORAGE,
                        layout=gl.SliceLayout(0, layout_b_scale_slot),
                    )[None, :]
                    * b.scale_stride_k,
                )
        return b_scale_hbm_offsets_tile
