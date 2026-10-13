# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operation-shape, data-layout, and offset contracts for gfx950 Gluon MoE GEMMs.

Every tile shape, register/shared layout, resource count and resolved LDS/register
placement of a build lives in one :class:`KernelLayout`, evaluated once from the
tuning configuration by :func:`compute_kernel_layout`. The yes/no knob policy it is
built on (``allow_component_via_lds``, ``scale_shuffled``, ...) is
``KernelTuningConfig`` methods. The final placement also depends on whether the tile
admits a direct-to-LDS copy, so it is resolved here. The host (grid, LDS probes) and
the kernel both read that object, so neither re-derives a layout per query and the
two cannot drift.
``_KernelFuncShape`` stays a method-only aggregate inherited by ``KernelFuncConfig``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import MX_GROUP, WARP_SIZE, require_constexpr, strip_annotate
from ._lang import unwrap as _v

__all__ = [
    "LDS_CAP_BYTES",
    "LDS_EPILOGUE_RESERVE_BYTES",
    "LDS_USABLE_BYTES",
    "KernelLayout",
    "_a_payload_hbm_offsets",
    "_a_scale_hbm_offsets",
    "_b_payload_hbm_offsets",
    "_b_scale_hbm_offsets",
    "_blocked_b_hbm_offsets",
    "_n_split_offs",
    "_n_start",
    "_shuffled_scale_register_offsets",
    "_shuffled_scale_stage_offsets",
    "_slot_index",
    "byte_unit_lds_layout",
    "compute_kernel_layout",
]

#: gfx950 LDS capacity, mirrors utils/_triton/arch_info.py::_LDS_CAP_BYTES["gfx950"].
LDS_CAP_BYTES = 163840
#: Headroom the operand buffers must leave for LDS the *compiler* allocates on top of
#: them -- the epilogue's convert_layout from the MFMA layout to the store layout needs
#: scratch that `lds_bytes()` cannot see. Empirical: a bf16 64x256 / BLOCK_K=128 tile
#: whose explicit buffers came to exactly 163840 B was rejected at launch at 166368 B.
#: Kept small on purpose -- the tuned MXFP4 configs sit at ~153 KiB and must not move.
LDS_EPILOGUE_RESERVE_BYTES = 4096
#: What the operand buffers may actually use.
LDS_USABLE_BYTES = LDS_CAP_BYTES - LDS_EPILOGUE_RESERVE_BYTES

_NO_SCALE: gl.constexpr = gl.constexpr(None)


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
def byte_unit_lds_layout(shape, nk_dim, elem_bits, unit_rows, preshuffled):
    """LDS layout of a payload operand, tiled by a fixed (rows x 128 B) unit.

    The tile is read as ``(non-K units, K units, unit_rows, 128 B)`` with the rightmost
    axis contiguous, so the unit -- not the CTA tile -- is what fixes the layout. That
    is the difference from ``compute_efficient_padded_shared_layout``, which sizes its
    non-K unit from the whole tile and therefore hands a 128-row tile a different
    permutation than a 32-row one.

    Two orders, selected by whether the operand is already preshuffled in HBM:

    * not preshuffled -- 32 rows, the measured row permutation ``1, 4, 16, 2, 8``, and
      32 B of padding every 1024 B. A warp's 64 lanes cover exactly one 1024 B run, so
      the padding never splits a direct-to-LDS write.
    * preshuffled -- 16 rows in the byte order ``utils/shuffle.py::shuffle_weight(w,
      (16, 16))`` already writes, i.e. ``(n//16)*(KB*16) + (k//16)*256 + (n%16)*16 +
      k%16``. The copy is then a straight linear run and no padding is needed, because
      the 64 lanes of a fetch already land on 64 consecutive 16 B slots.

    ``shape`` and the returned bases are in *stored* elements; with the 8-bit storage
    both fp8 and packed fp4 use, one element is one byte.
    """
    shape = [int(x) for x in _v(shape)]
    nk_dim = _v(nk_dim)
    elem_bits = _v(elem_bits)
    unit_rows = _v(unit_rows)
    rank = len(shape)
    kd = 1 - nk_dim
    els = lambda nbytes: nbytes * 8 // elem_bits

    U = els(LDS_UNIT_K_BYTES)  # elements per 128 B K unit
    V = els(LDS_LANE_BYTES)  # elements per 16 B lane access
    rows, kelems = shape[nk_dim], shape[kd]

    K = lambda vs: _pow2_bases(kd, rank, vs)
    NK = lambda vs: _pow2_bases(nk_dim, rank, vs)

    if preshuffled:
        # 16 B of K, then the 16 rows, then the rest of the 128 B unit.
        bases = K(_ramp(1, V)) + NK(_ramp(1, unit_rows)) + K(_ramp(V, U))
    else:
        assert unit_rows == 32, "the 1, 4, 16, 2, 8 permutation is a 32-row order"
        bases = K(_ramp(1, U)) + NK([1, 4, 16, 2, 8])
    bases = bases + K(_ramp(U, kelems)) + NK(_ramp(unit_rows, rows))

    if preshuffled:
        # PaddedSharedLayout insists on at least one interval/padding pair, so name an
        # interval wider than the tile: no pad can land inside it. 16 rather than 1 so
        # that even if the allocator rounds the tile up, it stays ds_read_b128 aligned.
        pairs = [[1 << (rows * kelems).bit_length(), 16]]
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
    stored_k,
):
    """Byte offsets for a B tile in the 16-column-blocked HBM layout.

    ``utils/shuffle.py::shuffle_weight(w, (16, 16))`` maps logical ``(n, k)``
    to ``(n//16)*(stored_k*16) + (k//16)*256 + (n%16)*16 + k%16``. Keeping
    this physical-layout transform beside the layout that consumes it prevents
    the HBM and LDS permutations from drifting apart.
    """
    kk = gl.arange(0, payload_block_k_storage, layout=gl.SliceLayout(1, layout))[
        :, None
    ]
    nn = (n0 + gl.arange(0, n_slot, layout=gl.SliceLayout(0, layout)))[None, :]
    return (nn // 16) * (stored_k * 16) + (kk // 16) * 256 + (nn % 16) * 16 + kk % 16


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
    scale_mini_k: gl.constexpr = 256,
    split_extent: gl.constexpr = 0,
    split_stride=0,
):
    """Flat HBM offsets for shuffled scales starting at logical non-K row ``nonk0``."""
    stripe = gl.arange(0, stripes, layout=gl.SliceLayout(1, layout))[:, None]
    byte = gl.arange(0, scale_mini_k, layout=gl.SliceLayout(0, layout))[None, :]
    if require_constexpr(not split_extent):
        return (nonk0 // 32 + stripe) * K + byte
    row = stripe * 32
    row = row // split_extent * split_stride + row % split_extent
    return ((nonk0 + row) // 32) * K + byte


@gluon.jit
def _n_start(pid_n, n_slot_idx: gl.constexpr, N, func_cfg, tuning_cfg):
    """Raw-N start of slot ``n_slot_idx`` under interleaved or split gate/up packing.

    Interleaved tiles are contiguous. Split tiles cover the same emitted channels but
    read gate and up from separate halves of N.
    """
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * tuning_cfg.MINI_BLOCK_N + n_slot_idx * (N // 2)
    else:
        out = pid_n * tuning_cfg.BLOCK_N + n_slot_idx * tuning_cfg.MINI_BLOCK_N
    return out


@gluon.jit
def _n_split_offs(pid_n, i, N, func_cfg, tuning_cfg):
    """Map a whole raw-N CTA index tensor through the gate/up packing."""
    n_slot: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * n_slot + (i // n_slot) * (N // 2) + i % n_slot
    else:
        out = pid_n * tuning_cfg.BLOCK_N + i
    return out


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

    TODO: stage the table in LDS instead. One block-M-wide ``buffer_load_to_shared``
    issued next to the routing scalars, committed there, with the ``wait_group`` and the
    LDS read deferred to just before ``pc`` is built, would replace
    ``num_m_slots_per_block()`` register loads per consumer and hide the fetch behind
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

    The way in is a 32-bit, block-aligned table -- which ``_sorted_token_id_map`` in
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


@gluon.jit
def _a_payload_hbm_offsets(a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg, layout):
    """Gathered A payload offsets, one copy-layout grid per M slot."""
    M_BLOCK: gl.constexpr = tuning_cfg.BLOCK_M
    M_SLOT: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    NUM_M_SLOTS_PER_BLOCK: gl.constexpr = layout.num_m_slots_per_block()
    PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = layout.payload_block_k_storage(0)
    layout_a_slot: gl.constexpr = layout.payload_hbm_offset_layout(0)
    a_hbm_offsets_block = ()
    for m_slot_idx in gl.static_range(NUM_M_SLOTS_PER_BLOCK):
        rows_a_slot = _gather_rows(
            rt,
            block_id,
            M_e,
            start_m,
            gl.SliceLayout(1, layout_a_slot),
            M_BLOCK,
            m_slot_idx * M_SLOT,
            M_SLOT,
            func_cfg.has_gather,
        )
        a_hbm_offsets_block = a_hbm_offsets_block + (
            rows_a_slot[:, None] * a.stride_m
            + gl.arange(
                0,
                PAYLOAD_BLOCK_K_STORAGE,
                layout=gl.SliceLayout(0, layout_a_slot),
            )[None, :],
        )

    return a_hbm_offsets_block


@gluon.jit
def _b_payload_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg, layout):
    """B payload offsets for plain or preshuffled weights, one grid per N slot."""
    N_SLOT: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NUM_N_SLOTS_PER_BLOCK: gl.constexpr = layout.num_n_slots_per_block()
    PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = layout.payload_block_k_storage(1)
    layout_b_slot: gl.constexpr = layout.payload_hbm_offset_layout(1)
    K_STORAGE: gl.constexpr = K // func_cfg.b_pack_divisor()
    b_hbm_offsets_block = ()
    for n_slot_idx in gl.static_range(NUM_N_SLOTS_PER_BLOCK):
        if require_constexpr(tuning_cfg.B_PRESHUFFLED):
            b_hbm_offsets_block = b_hbm_offsets_block + (
                _blocked_b_hbm_offsets(
                    layout_b_slot,
                    PAYLOAD_BLOCK_K_STORAGE,
                    _n_start(pid_n, n_slot_idx, N, func_cfg, tuning_cfg),
                    N_SLOT,
                    K_STORAGE,
                ),
            )
        else:
            b_hbm_offsets_block = b_hbm_offsets_block + (
                gl.arange(
                    0,
                    PAYLOAD_BLOCK_K_STORAGE,
                    layout=gl.SliceLayout(1, layout_b_slot),
                )[:, None]
                * b.stride_k
                + (
                    _n_start(pid_n, n_slot_idx, N, func_cfg, tuning_cfg)
                    + gl.arange(0, N_SLOT, layout=gl.SliceLayout(0, layout_b_slot))
                )[None, :]
                * b.stride_n,
            )

    return b_hbm_offsets_block


@gluon.jit
def _a_scale_hbm_offsets(
    a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg, layout
):
    """Byte offsets of the A-scale grids, one per M slot (``None`` if unscaled).

    Two shapes, picked by ``A_SCALE_SORTED_SHUFFLED``: a flat run into the
    moe_sort_scales pre-pass output, or a gathered
    ``[M_SLOT, SCALE_BLOCK_K_STORAGE]`` grid
    over the raw ``(M, K/32)`` tensor. The caller sees a tuple indexed by M slot.
    """
    if require_constexpr(not func_cfg.a_has_scale()):
        return _NO_SCALE

    M_BLOCK: gl.constexpr = tuning_cfg.BLOCK_M
    M_SLOT: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    NUM_M_SLOTS_PER_BLOCK: gl.constexpr = layout.num_m_slots_per_block()
    SCALE_BLOCK_K_STORAGE: gl.constexpr = layout.scale_block_k_storage()
    layout_a_scale_slot: gl.constexpr = layout.scale_hbm_offset_layout(0)
    SCALE_M: gl.constexpr = layout.scale_mini_block_nonk(0)
    SCALE_K: gl.constexpr = layout.scale_mini_block_k(0)
    SCALE_RATIO: gl.constexpr = layout.scale_ratio_non_k_slot(0)

    if require_constexpr(tuning_cfg.scale_shuffled(0) and layout.scale_in_reg(0)):
        a_scale_hbm_offsets_block = ()
        for m_slot_idx in gl.static_range(NUM_M_SLOTS_PER_BLOCK):
            a_scale_hbm_offsets_block = a_scale_hbm_offsets_block + (
                _shuffled_scale_register_offsets(
                    layout_a_scale_slot,
                    pid_m * M_BLOCK + (m_slot_idx // SCALE_RATIO) * SCALE_M,
                    SCALE_M,
                    SCALE_K // 32,
                    K,
                    layout.scale_packed_ok(0),
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
            layout.sorted_shuffled_ok(),
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
        # copy 1-D is what lets it vectorise -- in the [M_BLOCK, SK] view a lane's four
        # bytes straddle both axes and nothing can widen the access.
        # One tile per M slot. The tile is a whole number of 32-row stripes and
        # stripes are contiguous, so slot m_slot_idx is just the run starting
        # m_slot_idx * (M_SLOT // 32) stripes into this pid_m's chunk.
        # Slots sharing a scale mini block get the same base
        # offsets; only the owner actually issues the copy.
        SCALE_FILL_SHAPE: gl.constexpr = layout.scale_flat_shape(0)
        a_scale_hbm_offsets_block = ()
        for m_slot_idx in gl.static_range(NUM_M_SLOTS_PER_BLOCK):
            a_scale_hbm_offsets_block = a_scale_hbm_offsets_block + (
                _shuffled_scale_stage_offsets(
                    layout_a_scale_slot,
                    pid_m * M_BLOCK + (m_slot_idx // SCALE_RATIO) * SCALE_M,
                    SCALE_FILL_SHAPE[0],
                    K,
                    SCALE_K,
                ),
            )
    else:
        a_scale_hbm_offsets_block = ()
        for m_slot_idx in gl.static_range(NUM_M_SLOTS_PER_BLOCK):
            rows_a_scale_slot = _gather_rows(
                rt,
                block_id,
                M_e,
                start_m,
                gl.SliceLayout(1, layout_a_scale_slot),
                M_BLOCK,
                m_slot_idx * M_SLOT,
                M_SLOT,
                func_cfg.has_gather,
            )
            a_scale_hbm_offsets_block = a_scale_hbm_offsets_block + (
                rows_a_scale_slot[:, None] * a.scale_stride_m
                + gl.arange(
                    0,
                    SCALE_BLOCK_K_STORAGE,
                    layout=gl.SliceLayout(0, layout_a_scale_slot),
                )[None, :]
                * a.scale_stride_k,
            )
    return a_scale_hbm_offsets_block


@gluon.jit
def _b_scale_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg, layout):
    """Byte offsets of the B-scale grids, one per N slot (``None`` if unscaled).

    Either a flat CDNA4_SCALE preshuffle run for staging or the raw ``(K/32, N)``
    grid. The caller receives a tuple indexed by N slot.
    """
    if require_constexpr(not func_cfg.b_has_scale()):
        return _NO_SCALE

    N_SLOT: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NUM_N_SLOTS_PER_BLOCK: gl.constexpr = layout.num_n_slots_per_block()
    SCALE_BLOCK_K_STORAGE: gl.constexpr = layout.scale_block_k_storage()
    layout_b_scale_slot: gl.constexpr = layout.scale_hbm_offset_layout(1)
    SCALE_N: gl.constexpr = layout.scale_mini_block_nonk(1)
    SCALE_K: gl.constexpr = layout.scale_mini_block_k(1)
    SCALE_RATIO: gl.constexpr = layout.scale_ratio_non_k_slot(1)
    SPLIT_EXTENT: gl.constexpr = (
        N_SLOT if func_cfg.gu_split() and SCALE_RATIO > 1 else 0
    )

    if require_constexpr(tuning_cfg.scale_shuffled(1) and layout.scale_in_reg(1)):
        b_scale_hbm_offsets_block = ()
        for n_slot_idx in gl.static_range(NUM_N_SLOTS_PER_BLOCK):
            b_scale_hbm_offsets_block = b_scale_hbm_offsets_block + (
                _shuffled_scale_register_offsets(
                    layout_b_scale_slot,
                    _n_start(
                        pid_n,
                        n_slot_idx // SCALE_RATIO * SCALE_RATIO,
                        N,
                        func_cfg,
                        tuning_cfg,
                    ),
                    SCALE_N,
                    SCALE_K // 32,
                    K,
                    layout.scale_packed_ok(1),
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
        # One tile per N slot; stripes are contiguous, so slot n_slot_idx starts
        # n_slot_idx * (N_SLOT // 32) stripes into this pid_n's run.
        SCALE_FILL_SHAPE: gl.constexpr = layout.scale_flat_shape(1)
        b_scale_hbm_offsets_block = ()
        for n_slot_idx in gl.static_range(NUM_N_SLOTS_PER_BLOCK):
            b_scale_hbm_offsets_block = b_scale_hbm_offsets_block + (
                _shuffled_scale_stage_offsets(
                    layout_b_scale_slot,
                    _n_start(
                        pid_n,
                        n_slot_idx // SCALE_RATIO * SCALE_RATIO,
                        N,
                        func_cfg,
                        tuning_cfg,
                    ),
                    SCALE_FILL_SHAPE[0],
                    K,
                    SCALE_K,
                    SPLIT_EXTENT,
                    N // 2,
                ),
            )
    else:
        b_scale_hbm_offsets_block = ()
        for n_slot_idx in gl.static_range(NUM_N_SLOTS_PER_BLOCK):
            b_scale_hbm_offsets_block = b_scale_hbm_offsets_block + (
                (
                    _n_start(pid_n, n_slot_idx, N, func_cfg, tuning_cfg)
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
    return b_scale_hbm_offsets_block


@gluon.constexpr_function
def _slot_index(
    m_slot_idx,
    n_slot_idx,
    num_m_slots_per_block,
    num_n_slots_per_block,
):
    """Position of one M/N slot pair in the N-outer, M-inner traversal.

    This one number is the slot's place in three orderings at once -- the visitation
    order of the slot loop, the index of its accumulator in the carried tuple, and
    (under the even schedule) the position of its fill in
    :func:`_buffer_load_order`. They have to agree, which is why they all come from
    here.

    Down the M axis first, so with two slots on each axis the four slots are visited
    ``(0,0), (1,0), (0,1), (1,1)`` -- the region order of the reference kernel in
    gfx950-gluon-tutorials .../a16w16/v8_sliceMN, whose four DOTs are
    ``(A_top, B_left), (A_bot, B_left), (A_top, B_right), (A_bot, B_right)``.

    With one slot on either axis this is the identity on the old row-major order, so
    only a genuinely 2-D slot split sees any change.
    """
    m_slot_idx, n_slot_idx, num_m_slots_per_block, num_n_slots_per_block = (
        _v(m_slot_idx),
        _v(n_slot_idx),
        _v(num_m_slots_per_block),
        _v(num_n_slots_per_block),
    )
    return n_slot_idx * num_m_slots_per_block + m_slot_idx


@aggregate
class _KernelFuncShape:
    """Operation-shape accessors inherited by ``KernelFuncConfig``."""

    @gluon.constexpr_function
    def activation_reduction_n(self):
        """Emitted columns per raw column. 2 for a gated activation, 1 otherwise."""
        return 2 if _v(self.activation) is not None else 1

    @gluon.constexpr_function
    def gu_split(self):
        """Is the gated pair laid out as two N halves rather than interleaved?

        Only meaningful with a gated activation -- the flag is inert otherwise, so it
        is folded in here instead of at every use site.
        """
        return bool(_v(self.gate_up_split)) and _v(self.activation) is not None

    @gluon.constexpr_function
    def n_reduction_slot(self):
        """Emitted columns per raw column within one N slot.

        The block-level ratio is always :meth:`activation_reduction_n`, but under
        ``gu_split`` the halving happens *across* two slots rather than inside each
        one -- a slot is a whole operand side, so its width maps 1:1 onto emitted
        channels and two of them collapse into one output tile.
        """
        return 1 if self.gu_split() else self.activation_reduction_n()


# -- per-quantity derivations ---------------------------------------------------------
#
# One function per layout quantity, each a pure function of the tuning configuration.
# Only `compute_kernel_layout` and the other derivations here call them; everything
# else reads the result.


@gluon.constexpr_function
def _calc_num_warps(tc):
    w = _v(tc.warps_per_cta)
    return w[0] * w[1]


@gluon.constexpr_function
def _calc_num_k_slots_per_tile(tc):
    """Payload fragments per BLOCK_K stage."""
    return 1


@gluon.constexpr_function
def _calc_num_prefetch_k_slots(tc):
    """Whether the complete BLOCK_K stage is carried across a pipeline step."""
    return 1 if _v(tc.VGPR_PREFETCH_K) else 0


@gluon.constexpr_function
def _calc_num_m_slots_per_block(tc):
    """M slots per CTA block. Operand A and the accumulator split by this."""
    return _v(tc.BLOCK_M) // _v(tc.MINI_BLOCK_M)


@gluon.constexpr_function
def _calc_num_n_slots_per_block(tc):
    """N slots per CTA block. Operand B and the accumulator split by this."""
    return _v(tc.BLOCK_N) // _v(tc.MINI_BLOCK_N)


@gluon.constexpr_function
def _calc_num_lds_slots_per_block_non_k(tc, idx):
    """Number of independently copied/read non-K slots in one operand block."""
    return (
        _calc_num_m_slots_per_block(tc)
        if _v(idx) == 0
        else _calc_num_n_slots_per_block(tc)
    )


@gluon.constexpr_function
def _calc_payload_block_k_storage(tc, idx):
    """Stored payload elements along K in one pipeline stage."""
    return _v(tc.BLOCK_K) // tc.func_cfg.pack_divisor(idx)


@gluon.constexpr_function
def _calc_scale_block_k_storage(tc):
    """E8M0 elements along K in one pipeline stage."""
    return _v(tc.BLOCK_K) // MX_GROUP


@gluon.constexpr_function
def _calc_payload_fragment_shape_slot(tc, idx):
    """Register payload shape for one non-K slot and K slot."""
    if _v(idx) == 0:
        return [
            _v(tc.MINI_BLOCK_M),
            _v(tc.BLOCK_K) // tc.func_cfg.pack_divisor(0),
        ]
    return [
        _v(tc.BLOCK_K) // tc.func_cfg.pack_divisor(1),
        _v(tc.MINI_BLOCK_N),
    ]


@gluon.constexpr_function
def _calc_scale_fragment_shape_slot(tc, idx):
    """Register E8M0 shape for one non-K slot and K slot."""
    nonk = _v(tc.MINI_BLOCK_M) if _v(idx) == 0 else _v(tc.MINI_BLOCK_N)
    return [nonk, _v(tc.BLOCK_K) // MX_GROUP]


@gluon.constexpr_function
def _calc_accumulator_fragment_shape_slot(tc):
    """Shape of one M-slot by N-slot accumulator tile."""
    return [_v(tc.MINI_BLOCK_M), _v(tc.MINI_BLOCK_N)]


@gluon.constexpr_function
def _calc_payload_lds_shape_slot(tc, idx):
    """Shared shape of one payload slot, in stored (packed) elements.

    This is one slot, not the whole CTA block: A is split along M into
    ``num_m_slots_per_block()`` slots of ``MINI_BLOCK_M`` rows, and B along N into
    ``num_n_slots_per_block()`` slots of ``MINI_BLOCK_N`` columns. Each slot is
    allocated, copied, and read as an independent unit, so each gets its own
    efficient padded layout and fully coalesced direct-to-LDS copy. Slicing one
    block-wide tile would instead give the copy a strided view of a permuted layout.
    """
    if _v(idx) == 0:
        return [
            _v(tc.MINI_BLOCK_M),
            _calc_payload_block_k_storage(tc, 0),
        ]
    return [
        _calc_payload_block_k_storage(tc, 1),
        _v(tc.MINI_BLOCK_N),
    ]


@gluon.constexpr_function
def _calc_scale_lds_shape_slot(tc, idx):
    """E8M0 shape of one operand slot: [slot non-K extent, BLOCK_K/32]."""
    non_k = _v(tc.MINI_BLOCK_M) if _v(idx) == 0 else _v(tc.MINI_BLOCK_N)
    return [non_k, _calc_scale_block_k_storage(tc)]


@gluon.constexpr_function
def _calc_payload_lds_shape_block(tc, idx):
    """Full payload LDS shape for one block, including ring and slot axes."""
    return [
        tc.num_buffers(idx) * _calc_num_lds_slots_per_block_non_k(tc, idx)
    ] + _calc_payload_lds_shape_slot(tc, idx)


@gluon.constexpr_function
def _calc_scale_lds_shape_block(tc, idx):
    """Full scale LDS shape for one block in its staged representation."""
    idx = _v(idx)
    if tc.scale_shuffled(idx):
        num_scale_fill_tiles_block = _calc_num_scale_tiles(tc, idx)
        scale_fill_shape = _calc_scale_flat_shape(tc, idx)
    else:
        num_scale_fill_tiles_block = _calc_num_lds_slots_per_block_non_k(tc, idx)
        scale_fill_shape = _calc_scale_lds_shape_slot(tc, idx)
    return [tc.num_buffers(idx, True) * num_scale_fill_tiles_block] + scale_fill_shape


@gluon.constexpr_function
def _calc_copy_contiguity(tc, idx):
    """Elements per lane for a 128-bit direct-to-LDS payload copy.

    Per operand: bf16 gives 8, fp8 and packed-fp4 (stored as uint8) give 16. Using
    one value for both would allocate and copy the fp4 weight tile as if it were
    bf16 in every mixed configuration.
    """
    return 128 // tc.func_cfg.operand_elem_ty(idx).primitive_bitwidth


# -- storage placement ----------------------------------------------------------------
#
# The final LDS-or-register decision for each component: the tuning policy
# (`KernelTuningConfig.allow_component_via_lds`) and whether the tile admits a
# direct-to-LDS copy. There is no third placement, so a component not staged through
# LDS is in registers and `KernelLayout.component_in_reg` is defined as the complement.


@gluon.constexpr_function
def _calc_payload_lds_copy_ok(tc, idx):
    """Whether the payload tile admits a 128-bit-per-lane direct copy into LDS.

    Payload tiles that cannot give every wavefront one complete access stay in
    the register ring. There is deliberately no register-to-LDS fallback.
    """
    shape = _calc_payload_lds_shape_slot(tc, idx)
    vec = _calc_copy_contiguity(tc, idx)
    return shape[0] * shape[1] >= WARP_SIZE * vec * _calc_num_warps(tc)


@gluon.constexpr_function
def _calc_scale_lds_copy_ok(tc, idx):
    """Whether an E8M0 scale tile can be written to LDS with a coalesced copy.

    CDNA4 direct-to-LDS supports only 128-bit or 32-bit per lane and a warp must
    write one contiguous run, so two tiles are excluded:

    * smaller than ``64 lanes * 4 B`` -- cannot be lowered at all (the A-scale tile
      at ``BLOCK_M == 16`` with a short ``BLOCK_K``);
    * a row shorter than 8 scales (``BLOCK_K < 256``) -- one lane then owns a whole
      row and the reg->shared map is more contiguous than the 32-bit access can
      cover, which ``canLoadDirectToLDS`` rejects.

    Both fall back to a register ``buffer_load`` straight into the scale fragment
    layout. That is correct but costs an in-loop register-path global access, which
    makes every ``wait_group`` conservative -- so the tuner should prefer a BLOCK_K
    that keeps this True.
    """
    if tc.scale_shuffled(idx):
        # Fragment-ordered in HBM, so the copy is linear and the LDS tile keeps the
        # same permutation; the read is then a 32-bit ds_read. Staying on LDS also
        # keeps the warp-broadcast B tile a single fetch rather than one per warp.
        return True
    shape = _calc_scale_lds_shape_slot(tc, idx)
    return shape[0] * shape[1] >= WARP_SIZE * 4 and shape[1] % 4 == 0 and shape[1] >= 8


@gluon.constexpr_function
def _calc_component_via_lds(tc, component):
    """Final placement of ``(operand, is_scale)``: tuning allows LDS and the copy fits.

    Not gated on whether a scale exists; callers gate absent scales with
    ``has_scale``.
    """
    operand, is_scale = _v(component[0]), bool(_v(component[1]))
    if not tc.allow_component_via_lds((operand, is_scale)):
        return False
    if is_scale:
        return _calc_scale_lds_copy_ok(tc, operand)
    return _calc_payload_lds_copy_ok(tc, operand)


@gluon.constexpr_function
def _calc_k_width_for(tc, idx):
    """``DotOperandLayout.k_width`` for one operand, in *stored* elements.

    It is 128 bits per lane per access, i.e. the same quantity as
    :meth:`copy_contiguity`: 16 for fp8 and for packed-fp4-in-uint8, 8 for bf16.
    Verified against a reference GEMM on both f8f6f4 instruction shapes -- k_width
    32 has no efficient padded shared layout at all, and 8 with the 16x16x128 shape
    compiles and runs but computes the wrong answer, which is exactly the class of
    bug no correctness-by-construction argument catches.

    The tuning field is an optional override, not the value itself.
    """
    override = _v(tc.k_width)
    if override is not None:
        return override
    return 128 // tc.func_cfg.operand_elem_ty(idx).primitive_bitwidth


@gluon.constexpr_function
def _calc_dot_result_fragment_layout(tc):
    """AMDMFMALayout for dot / scaled-dot results in register.

    ``transposed`` is pinned True: it gives each lane 4 consecutive N elements (a
    16 B contiguous store, and a lane-local even/odd gate-up pair for the fused
    activation) instead of 4 strided M rows.
    """
    assert _v(tc.transposed), "transposed=True is pinned, see the module docstring"
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=_v(tc.mfma_instr_shape),
        transposed=True,
        warps_per_cta=_v(tc.warps_per_cta),
        tiles_per_warp=_v(tc.tiles_per_warp),
    )


@gluon.constexpr_function
def _calc_dot_operand_fragment_layout(tc, idx):
    return gl.DotOperandLayout(
        operand_index=_v(idx),
        parent=_calc_dot_result_fragment_layout(tc),
        k_width=_calc_k_width_for(tc, idx),
    )


@gluon.constexpr_function
def _calc_lds_unit_rows(tc, idx):
    """Non-K extent of one LDS unit.

    A preshuffled operand is blocked 16 rows at a time in HBM and the copy has to
    stay linear, so the unit is that block. A plain one is read two MFMA tiles at a
    time (hence the ``tiles_per_warp >= 2`` that ``validate_layout`` insists on), so
    the unit is 32 and the row permutation has a 5th base to work with.
    """
    return 16 if tc.operand_preshuffled(idx) else 32


@gluon.constexpr_function
def _calc_byte_unit_lds_ok(tc, idx):
    """Does :func:`byte_unit_lds_layout` apply to this operand?

    Only the 16x16x128 B operand shape is in scope -- that is 16x16x128 for fp8 and
    16x16x256 for packed fp4, both stored 8-bit. The 32x32x64 prefill configs, the
    bf16 ones (16x16x32 / 32x32x16) and any tile too short for a whole unit keep
    ``compute_efficient_padded_shared_layout``.

    A non-preshuffled 32-row unit additionally needs the warp to own both of its
    16-row MFMA tiles, i.e. ``tiles_per_warp >= 2`` on that axis. That is not always
    reachable -- at ``BLOCK_N`` 64 with warps (1, 4) the CTA's whole N tile is four
    16-column tiles, so no warp can have two -- hence a gate rather than an assert.
    """
    idx = _v(idx)
    if list(_v(tc.mfma_instr_shape)) != [16, 16, 128]:
        return False
    elem_bits = tc.func_cfg.operand_elem_ty(idx).primitive_bitwidth
    if elem_bits != 8:
        return False
    if not tc.operand_preshuffled(idx) and _v(tc.tiles_per_warp)[idx] < 2:
        return False
    shape = _calc_payload_lds_shape_slot(tc, idx)
    rows, kelems = shape[idx], shape[1 - idx]
    unit_rows = _calc_lds_unit_rows(tc, idx)
    unit_k = LDS_UNIT_K_BYTES * 8 // elem_bits
    return rows % unit_rows == 0 and kelems % unit_k == 0


@gluon.constexpr_function
def _calc_dot_operand_lds_layout(tc, idx):
    """Padded shared layout for a payload operand.

    Both operands are K-packed under the memory-layout contract, so
    ``is_k_contig=True`` and the plain ``smem.load(dot_layout)`` path applies --
    ``load_shared_fp4_repacked`` is only needed for an M/N-packed checkpoint, which
    none of the four models in scope produce.
    """
    idx = _v(idx)
    shape = _calc_payload_lds_shape_slot(tc, idx)
    if _calc_byte_unit_lds_ok(tc, idx):
        return byte_unit_lds_layout(
            shape,
            idx,
            tc.func_cfg.operand_elem_ty(idx).primitive_bitwidth,
            _calc_lds_unit_rows(tc, idx),
            tc.operand_preshuffled(idx),
        )
    layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        _calc_dot_operand_fragment_layout(tc, idx),
        shape,
        tc.func_cfg.operand_elem_ty(idx),
        True,
    )
    if layout is not None:
        return layout
    # The helper declines whenever the tile holds a single MFMA tile along the
    # non-K axis (BLOCK_M == instr_shape[0], i.e. the whole decode regime): it has
    # no row permutation left to build. Fall back to a plain identity-mapped padded
    # layout whose interval is still >= vec * warpSize, which is the condition
    # `canLoadDirectToLDS` checks, so the copy stays a 128-bit direct-to-LDS.
    vec = _calc_copy_contiguity(tc, idx)
    order = [1, 0] if idx == 0 else [0, 1]
    return gl.PaddedSharedLayout.with_identity_for(
        [[WARP_SIZE * vec, vec]], shape, order
    )


@gluon.constexpr_function
def _calc_dot_operand_buffer_load_lds_layout(tc, idx):
    """Register layout of the global->LDS copy offsets for a payload operand."""
    idx = _v(idx)
    shape = _calc_payload_lds_shape_slot(tc, idx)
    return _bases_to_distributed(
        _calc_dot_operand_lds_layout(tc, idx).offset_bases,
        _calc_copy_contiguity(tc, idx),
        _calc_num_warps(tc),
        WARP_SIZE,
        shape,
    )


@gluon.constexpr_function
def _calc_payload_hbm_offset_layout(tc, idx):
    """Layout used to form one payload tile's global-memory offsets."""
    if _calc_component_via_lds(tc, (idx, False)):
        return _calc_dot_operand_buffer_load_lds_layout(tc, idx)
    return _calc_dot_operand_fragment_layout(tc, idx)


@gluon.constexpr_function
def _calc_dot_operand_scale_fragment_layout(tc, idx):
    idx = _v(idx)
    shape = _calc_scale_lds_shape_slot(tc, idx)
    return gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout(tc, idx), shape, MX_GROUP
    )


# -- ScaleSwizzle.SORTED_SHUFFLED --------------------------------------------
# moe_sort_scales.cuh writes one dword per (chunk, mi, ku, k_lane, n_lane) with
#     row = chunk*BM + (mi*MN_PACK + im_a)*16 + n_lane
#     k   = ku*K_PACK*4 + ikxdl*4 + k_lane
#     byte within the dword = ikxdl*MN_PACK + im_a
# so n_lane/k_lane index the 64 lanes of a wavefront, im_a/ikxdl the 4 bytes a lane
# holds, and mi the wave. That is exactly get_mfma_scale_layout's decomposition for
# a 16x16x128 dot operand with warps_per_cta = (num_warps, 1) and
# tiles_per_warp = (2, 1) -- verified base-for-base -- so the shuffled buffer is
# read with a plain contiguous load and no cross-lane movement.


@gluon.constexpr_function
def _calc_scale_mini_block_nonk(tc, idx):
    """Rows or columns in one scale load; raw scales retain payload geometry."""
    mini = _calc_scale_nonk(tc, idx)
    if not tc.scale_shuffled(idx):
        return mini
    requested = tc.SCALE_MINI_BLOCK_M if _v(idx) == 0 else tc.SCALE_MINI_BLOCK_N
    return _v(requested) or mini


@gluon.constexpr_function
def _calc_scale_mini_block_k(tc, idx):
    """Scale load K extent in logical payload elements, not E8M0 bytes."""
    if not tc.scale_shuffled(idx):
        return _v(tc.BLOCK_K)
    return _v(tc.SCALE_MINI_BLOCK_K) or max(256, _v(tc.BLOCK_K))


@gluon.constexpr_function
def _calc_scale_ratio_k_step(tc, idx):
    """Number of payload K steps covered by one scale load."""
    bk = _v(tc.BLOCK_K)
    return _calc_scale_mini_block_k(tc, idx) // bk


@gluon.constexpr_function
def _calc_scale_ratio_non_k_slot(tc, idx):
    """Number of payload non-K slots covered by one scale load."""
    return _calc_scale_mini_block_nonk(tc, idx) // _calc_scale_nonk(tc, idx)


@gluon.constexpr_function
def _calc_num_scale_tiles(tc, idx):
    return _calc_num_lds_slots_per_block_non_k(
        tc, idx
    ) // _calc_scale_ratio_non_k_slot(tc, idx)


@gluon.constexpr_function
def _calc_scale_read_k_slots(tc, idx):
    return _calc_scale_ratio_k_step(tc, idx)


@gluon.constexpr_function
def _calc_scale_cache_fragments(tc, idx):
    return _calc_num_lds_slots_per_block_non_k(tc, idx) * _calc_scale_read_k_slots(
        tc, idx
    )


@gluon.constexpr_function
def _calc_scale_hbm_offset_layout(tc, idx):
    """Layout used to form one scale tile's global-memory offsets."""
    if _calc_component_via_lds(tc, (idx, True)):
        return _calc_dot_operand_scale_buffer_load_lds_layout(tc, idx)
    if tc.scale_shuffled(idx):
        if _calc_scale_packed_ok(tc, idx):
            return _calc_packed_scale_load_layout(tc, idx)
        # get_mfma_scale_layout puts the K-group bit before the non-K bit in
        # reg_bases, so a lane's four bytes land in registers in address order
        # 0, +2, +1, +3. A widened load fills registers in ascending order; issue
        # it with those two bases swapped, then convert with only a lane-local
        # register renumber because the lane and warp bases stay unchanged.
        frag = _calc_scale_load_layout(tc, idx)
        lead = [[16, 0], [0, 4]]
        rest = [list(b) for b in frag.reg_bases if list(b) not in lead]
        return gl.DistributedLinearLayout(
            reg_bases=lead + rest,
            lane_bases=[list(b) for b in frag.lane_bases],
            warp_bases=[list(b) for b in frag.warp_bases],
            block_bases=[],
            shape=list(frag.shape),
        )
    return _calc_dot_operand_scale_fragment_layout(tc, idx)


@gluon.constexpr_function
def _calc_scale_nonk(tc, idx):
    """Non-K extent of one shuffled scale tile.

    The A-side shuffle (SORTED_SHUFFLED) and CDNA4_SCALE both lay the tile out as
    one 256 B run per 32-row stripe, so a slot that is a whole number of stripes is
    a contiguous slice and the tile can use the slot extent.
    """
    if _v(idx) == 0:
        return _v(tc.MINI_BLOCK_M)
    return _v(tc.MINI_BLOCK_N)


@gluon.constexpr_function
def _calc_scale_flat_shape(tc, idx):
    """LDS staging shape of a shuffled scale tile: one 256 B run per 32-row stripe.

    The tile is staged in HBM byte order rather than fragment order because
    direct-to-LDS on gfx9 cannot scatter -- each warp must write coalesced, which a
    fragment-ordered SharedLinearLayout does not (canLoadDirectToLDS rejects it, see
    TritonAMDGPUToLLVM/Utility.cpp). The bytes therefore sit in (non-K 16, K 4) order
    while the u8 fragment numbers its registers (K 4, non-K 16); reconciling the two
    used to cost a v_perm per dword, which is what ``scale_packed_ok`` avoids by
    handing the dword to the MFMA whole and naming the byte order in a selector list.
    """
    return [
        _calc_scale_mini_block_nonk(tc, idx) // 32,
        _calc_scale_mini_block_k(tc, idx),
    ]


@gluon.constexpr_function
def _calc_sorted_shuffled_ok(tc):
    """Does this config produce the layout the C++ A-scale shuffle was written for?"""
    # The permutation lives in the LDS tile's SharedLinearLayout, which is a
    # property of the byte order the shuffle writes -- not of how warps are split.
    # The warp arrangement is free. The shuffle always packs K256;
    # a K128 payload stage selects one half.
    return (
        list(_v(tc.mfma_instr_shape)) == [16, 16, 128]
        and _v(tc.BLOCK_M) % 32 == 0
        # An M slot only has to be a whole number of 32-row stripes: the shuffle
        # writes one 256 B run per stripe, so a slot is a contiguous slice.
        and _v(tc.MINI_BLOCK_M) % 32 == 0
    )


@gluon.constexpr_function
def _calc_shuffled_scale_read_layout(tc, idx, load=False):
    """The [non-K, K] view of the flat LDS run, in fragment order."""
    nonk = (
        _calc_scale_mini_block_nonk(tc, idx) if _v(load) else _calc_scale_nonk(tc, idx)
    )
    k = _calc_scale_mini_block_k(tc, idx) if _v(load) else max(256, _v(tc.BLOCK_K))
    bases = [[16, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
    for bit in _ramp(8, k // MX_GROUP):
        bases += [[0, bit]]
    stripe = 32
    while stripe < nonk:
        bases = bases + [[stripe, 0]]
        stripe = stripe * 2
    return gl.SharedLinearLayout(offset_bases=bases)


# -- packed (i32) scale operands ------------------------------------------------
# Every shuffle here puts four scales in a dword, and the matrix instruction can
# pick one of them with op_sel. So a lane loads the dword whole and hands it to
# mfma_scaled_packed as an i32, with the byte order named in a selector list --
# no reinterpret to u8, no register renumbering, no v_perm.
#
# CDNA4_SCALE / SORTED_SHUFFLED put non-K +16 and K +4 inside the dword, at byte
# (nonK) + 2*(K). That fixed pair only works when the fragment happens to hold both
# steps in registers, so ``scale_packed_ok`` checks it rather than assuming it.


@gluon.constexpr_function
def _scale_dword_delta(delta, k=256):
    """Where a (non-K, K) step lands, as a coordinate of the i32 tile.

    The tile is [non-K, BLOCK_K/128] and its row-major linearisation *is* the dword
    index, so a step worth ``d`` dwords is the coordinate ``(d // 2, d % 2)``.
    """
    b = [int(x) for x in _v(delta)]
    if b == [0, 0]:
        return [0, 0]
    # CDNA4_SCALE writes byte (n//32)*256 + (k%4)*64 + (n%16)*4 + (k//4)*2 +
    # (n%32)//16; dropping the two within-dword terms and dividing by four
    # leaves the dword index below.
    width = _v(k) // 128
    d = (b[0] // 32) * (_v(k) // 4) + (b[1] // 8) * 64 + (b[1] % 4) * 16 + (b[0] % 16)
    return [d // width, d % width]


@gluon.constexpr_function
def _calc_scale_packed_sel(tc, idx, k_phase=0):
    """Byte selectors in MFMA order, for one K128 half or the full K256 word.

    The instruction index counts non-K major and K minor, so its low two bits are
    the fragment's first two register bases in that order. CDNA4_SCALE numbers the
    dword's bytes the other way round, hence the transposition.
    """
    if not _calc_scale_packed_ok(tc, idx):
        return None
    if _v(tc.BLOCK_K) == 128:
        return [2 * _v(k_phase), 2 * _v(k_phase) + 1]
    return [0, 2, 1, 3]


@gluon.constexpr_function
def _calc_has_mfma_packed_scale(tc):
    """Whether either live MFMA scale operand uses packed dword storage."""
    return _calc_scale_packed_ok(tc, 0) or _calc_scale_packed_ok(tc, 1)


@gluon.constexpr_function
def _calc_scale_packed_ok(tc, idx):
    """Can this operand's scale fragment be fed as one dword per lane?

    K256 folds K+4 and non-K+16; K128 folds only non-K+16 and selects
    its K half separately. Each folded step must be register-private.
    """
    if not tc.scale_shuffled(idx) or _v(tc.BLOCK_K) not in (128, 256):
        return False
    frag = gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout(tc, idx),
        _calc_scale_fragment_shape_slot(tc, idx),
        MX_GROUP,
    )
    regs = [list(b) for b in frag.reg_bases]
    if _v(tc.BLOCK_K) == 128:
        return len(regs) >= 1 and regs[0] == [16, 0]
    if len(regs) < 2:
        return False
    return regs[0] == [0, 4] and regs[1] == [16, 0]


@gluon.constexpr_function
def _calc_packed_scale_shape(tc, idx):
    """Physical i32 storage, including both K128 halves even when one is unused."""
    nonk = _calc_scale_nonk(tc, idx)
    return [nonk, max(256, _v(tc.BLOCK_K)) // 128]


@gluon.constexpr_function
def _calc_packed_scale_frag_layout(tc, idx):
    """The i32 fragment with its within-dword register bases folded in."""
    k = max(256, _v(tc.BLOCK_K))
    frag = gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout(tc, idx),
        [_calc_scale_nonk(tc, idx), k // MX_GROUP],
        MX_GROUP,
    )
    return gl.DistributedLinearLayout(
        reg_bases=[_scale_dword_delta(b, k) for b in frag.reg_bases[2:]],
        lane_bases=[_scale_dword_delta(b, k) for b in frag.lane_bases],
        warp_bases=[_scale_dword_delta(b, k) for b in frag.warp_bases],
        block_bases=[],
        shape=_calc_packed_scale_shape(tc, idx),
    )


@gluon.constexpr_function
def _calc_packed_scale_read_layout(tc, idx, load=False):
    """The i32 view of the flat LDS run: one offset bit per dword-address bit."""
    k = _calc_scale_mini_block_k(tc, idx) if _v(load) else max(256, _v(tc.BLOCK_K))
    bases = list(_calc_shuffled_scale_read_layout(tc, idx, load).offset_bases)[2:]
    return gl.SharedLinearLayout(offset_bases=[_scale_dword_delta(b, k) for b in bases])


@gluon.constexpr_function
def _calc_scale_load_shape(tc, idx):
    return [
        _calc_scale_mini_block_nonk(tc, idx),
        _calc_scale_mini_block_k(tc, idx) // MX_GROUP,
    ]


@gluon.constexpr_function
def _calc_scale_load_layout(tc, idx):
    return gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout(tc, idx),
        _calc_scale_load_shape(tc, idx),
        MX_GROUP,
    )


@gluon.constexpr_function
def _calc_packed_scale_load_shape(tc, idx):
    return [
        _calc_scale_mini_block_nonk(tc, idx),
        _calc_scale_mini_block_k(tc, idx) // 128,
    ]


@gluon.constexpr_function
def _calc_packed_scale_load_layout(tc, idx):
    frag = _calc_scale_load_layout(tc, idx)
    k = _calc_scale_mini_block_k(tc, idx)
    # The two within-dword axes are identified by value, not position.  Wider
    # scale K tiles insert additional K register bases between them.
    within_dword = ([0, 4], [16, 0])
    return gl.DistributedLinearLayout(
        reg_bases=[
            _scale_dword_delta(b, k)
            for b in frag.reg_bases
            if list(b) not in within_dword
        ],
        lane_bases=[_scale_dword_delta(b, k) for b in frag.lane_bases],
        warp_bases=[_scale_dword_delta(b, k) for b in frag.warp_bases],
        block_bases=[],
        shape=_calc_packed_scale_load_shape(tc, idx),
    )


@gluon.constexpr_function
def _calc_dot_operand_scale_lds_layout(tc, idx):
    """Identity (K-contiguous) shared tile for the raw E8M0 scales.

    The three LDS goals are not simultaneously satisfiable for the scales: with
    ``get_mfma_scale_layout`` each lane wants K-scale elements strided by 2 or 4, so
    the read is ``ds_read_u8`` regardless. An identity tile is what keeps the
    *write* side (direct-to-LDS) coalesced, which is the side that matters.
    """
    # Flat staging tile -- plain identity; fragment order returns on the read.
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])


@gluon.constexpr_function
def _calc_dot_operand_scale_buffer_load_lds_layout(tc, idx):
    """32-bit-per-lane blocked layout for the scale direct-to-LDS write."""
    idx = _v(idx)
    shape = _calc_scale_lds_shape_slot(tc, idx)
    if tc.scale_shuffled(idx):
        # Flat run, 4 contiguous bytes per lane.
        return gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[1, WARP_SIZE],
            warps_per_cta=[_calc_num_warps(tc), 1],
            order=[1, 0],
        )
    sk = shape[1]
    lanes_k = sk // 4
    return gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[WARP_SIZE // lanes_k, lanes_k],
        warps_per_cta=[_calc_num_warps(tc), 1],
        order=[1, 0],
    )


@gluon.constexpr_function
def _calc_output_n_slot(tc):
    """Emitted columns produced by one logical N-slot epilogue tile."""
    return _v(tc.MINI_BLOCK_N) // tc.func_cfg.n_reduction_slot()


@gluon.constexpr_function
def _calc_output_block_n(tc):
    """Emitted columns produced by one CTA's raw BLOCK_N tile."""
    return _v(tc.BLOCK_N) // tc.func_cfg.activation_reduction_n()


@gluon.constexpr_function
def _calc_epilogue_threads(tc):
    return _calc_num_warps(tc) * WARP_SIZE


@gluon.constexpr_function
def _calc_epilogue_bias_per_thread(tc):
    return _v(tc.BLOCK_N) // _calc_epilogue_threads(tc)


@gluon.constexpr_function
def _calc_epilogue_inputs_via_lds(tc):
    """Whether bias/gamma vectors admit the supported direct-to-LDS copies.

    No knob controls this placement: the vectors go through LDS whenever the copy is
    legal for the tile.
    """
    threads = _calc_epilogue_threads(tc)
    bpt = _calc_epilogue_bias_per_thread(tc)
    return threads >= _v(tc.BLOCK_M) and _v(tc.BLOCK_N) % threads == 0 and bpt in (1, 4)


@gluon.constexpr_function
def _calc_epilogue_input_lds_layout(tc):
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])


@gluon.constexpr_function
def _calc_epilogue_gamma_buffer_load_lds_layout(tc):
    return gl.BlockedLayout([1], [WARP_SIZE], [_calc_num_warps(tc)], [0])


@gluon.constexpr_function
def _calc_epilogue_bias_buffer_load_lds_layout(tc):
    return gl.BlockedLayout(
        [_calc_epilogue_bias_per_thread(tc)],
        [WARP_SIZE],
        [_calc_num_warps(tc)],
        [0],
    )


@gluon.constexpr_function
def _calc_epilogue_gamma_shape(tc):
    return [_calc_epilogue_threads(tc)]


@gluon.constexpr_function
def _calc_epilogue_bias_shape(tc):
    return [_v(tc.BLOCK_N)]


@gluon.constexpr_function
def _calc_quant_staging_layout(tc):
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])


@gluon.constexpr_function
def _calc_quant_staging_rows(tc):
    return _v(tc.BLOCK_M) if tc.func_cfg.gu_split() else _v(tc.MINI_BLOCK_M)


@gluon.constexpr_function
def _calc_quant_staging_rotates(tc):
    """Whether the fused-quant staging alternates two per-tile LDS banks."""
    fc = tc.func_cfg
    return _v(fc.output_quant) is not None and not fc.gu_split()


@gluon.constexpr_function
def _calc_payload_hbm_step(tc, idx):
    """Stored-element base-pointer increment for one payload K stage."""
    idx = _v(idx)
    step = _calc_payload_block_k_storage(tc, idx)
    if idx == 1 and _v(tc.B_PRESHUFFLED):
        return step // 16 * 256
    return step


@gluon.constexpr_function
def _calc_lds_bytes(tc):
    fc = tc.func_cfg
    total = 0
    for idx in (0, 1):
        shape = _calc_payload_lds_shape_slot(tc, idx)
        width = fc.operand_elem_ty(idx).primitive_bitwidth // 8
        # This shape is one slot; a block holds one non-K slot per split.
        num_slots_block = _calc_num_lds_slots_per_block_non_k(tc, idx)
        if _calc_component_via_lds(tc, (idx, False)):
            total += num_slots_block * shape[0] * shape[1] * width * tc.num_buffers(idx)
        if fc.has_scale(idx) and (
            _v(tc.FROZEN_STEP) or _calc_component_via_lds(tc, (idx, True))
        ):
            s = _calc_scale_lds_shape_block(tc, idx)
            total += s[0] * s[1] * s[2]
    return total


@gluon.constexpr_function
def _calc_acc_vgprs_per_lane(tc):
    return _v(tc.BLOCK_M) * _v(tc.BLOCK_N) // (_calc_num_warps(tc) * WARP_SIZE)


# -- the computed layout -------------------------------------------------------------
#
# Everything above derives one quantity from the tuning configuration on demand. The
# aggregate below holds all of them, evaluated once per build by
# `compute_kernel_layout`; kernel code, the schedule model and the host all read the
# same object instead of re-deriving a layout at every query.


@gluon.constexpr_function
def _calc_geometry_ok(tc):
    """Whether the tile lattice is sound enough to build register/shared layouts on.

    The N/K-independent part of ``_validate.validate_layout``, as a predicate. A config
    that fails it still gets its scalar geometry (the host probes LDS footprints of
    candidates it has not validated yet) but no layout objects, so the descriptive
    assertion in ``validate_layout`` is what reports it rather than a layout
    constructor.
    """
    BM, BN, BK = _v(tc.BLOCK_M), _v(tc.BLOCK_N), _v(tc.BLOCK_K)
    MBM, MBN = _v(tc.MINI_BLOCK_M), _v(tc.MINI_BLOCK_N)
    instr = _v(tc.mfma_instr_shape)
    warps = _v(tc.warps_per_cta)
    tiles = _v(tc.tiles_per_warp)
    if BK % instr[2] or BK % MX_GROUP or BM % MBM or BN % MBN:
        return False
    if MBM % (instr[0] * warps[0] * tiles[0]) or MBN % (instr[1] * warps[1] * tiles[1]):
        return False
    for idx in (0, 1):
        if not tc.scale_shuffled(idx):
            continue
        block = BM if idx == 0 else BN
        nonk = _calc_scale_mini_block_nonk(tc, idx)
        sk = _calc_scale_mini_block_k(tc, idx)
        mini = _calc_scale_nonk(tc, idx)
        if not block >= nonk >= mini or block % nonk or nonk % mini or nonk % 32:
            return False
        if sk < max(256, BK) or sk & (sk - 1) or sk % BK:
            return False
    return True


@aggregate
@strip_annotate
class KernelLayout:
    """Every tile shape, storage size and register/shared layout of one build.

    Built once by :func:`compute_kernel_layout` and passed explicitly to everything
    that needs a layout. Per-operand values are stored as ``(A, B)`` pairs and read
    through accessors indexed by operand (0 = A / tokens, 1 = B / experts). ``None``
    marks a value that does not exist for this build -- a scale layout of an unscaled
    operand, the LDS layout of a register-resident payload.

    Accessors that take a free argument (a K extent, a step, a store extent) are
    evaluated from the stored scalars; there is nothing to precompute for them.
    """

    # -- block and slot geometry --
    _BLOCK_N: gl.constexpr
    _BLOCK_K: gl.constexpr
    _MINI_BLOCK_M: gl.constexpr
    _mfma_instr_shape: gl.constexpr
    _gu_split: gl.constexpr
    _num_warps: gl.constexpr
    _num_k_slots_per_tile: gl.constexpr
    _num_prefetch_k_slots: gl.constexpr
    _num_m_slots_per_block: gl.constexpr
    _num_n_slots_per_block: gl.constexpr
    _num_lds_slots_per_block_non_k: gl.constexpr
    _payload_block_k_storage: gl.constexpr
    _scale_block_k_storage: gl.constexpr
    _payload_fragment_shape_slot: gl.constexpr
    _scale_fragment_shape_slot: gl.constexpr
    _accumulator_fragment_shape_slot: gl.constexpr
    _payload_lds_shape_slot: gl.constexpr
    _scale_lds_shape_slot: gl.constexpr
    _payload_lds_shape_block: gl.constexpr
    _scale_lds_shape_block: gl.constexpr
    _copy_contiguity: gl.constexpr
    _k_width_for: gl.constexpr
    _payload_hbm_step: gl.constexpr
    _lds_unit_rows: gl.constexpr
    _byte_unit_lds_ok: gl.constexpr
    # -- storage placement --
    _component_via_lds: gl.constexpr
    # -- payload layouts --
    _dot_result_fragment_layout: gl.constexpr
    _dot_operand_fragment_layout: gl.constexpr
    _dot_operand_lds_layout: gl.constexpr
    _dot_operand_buffer_load_lds_layout: gl.constexpr
    _payload_hbm_offset_layout: gl.constexpr
    # -- scale geometry --
    _scale_nonk: gl.constexpr
    _scale_mini_block_nonk: gl.constexpr
    _scale_mini_block_k: gl.constexpr
    _scale_ratio_k_step: gl.constexpr
    _scale_ratio_non_k_slot: gl.constexpr
    _num_scale_tiles: gl.constexpr
    _scale_read_k_slots: gl.constexpr
    _scale_cache_fragments: gl.constexpr
    _scale_flat_shape: gl.constexpr
    _sorted_shuffled_ok: gl.constexpr
    _scale_load_shape: gl.constexpr
    _scale_packed_ok: gl.constexpr
    _scale_packed_sel: gl.constexpr
    _has_mfma_packed_scale: gl.constexpr
    _packed_scale_shape: gl.constexpr
    _packed_scale_load_shape: gl.constexpr
    # -- scale layouts --
    _dot_operand_scale_fragment_layout: gl.constexpr
    _dot_operand_scale_lds_layout: gl.constexpr
    _dot_operand_scale_buffer_load_lds_layout: gl.constexpr
    _scale_hbm_offset_layout: gl.constexpr
    _shuffled_scale_read_layout: gl.constexpr
    _scale_load_layout: gl.constexpr
    _packed_scale_frag_layout: gl.constexpr
    _packed_scale_read_layout: gl.constexpr
    _packed_scale_load_layout: gl.constexpr
    # -- epilogue --
    _output_n_slot: gl.constexpr
    _output_block_n: gl.constexpr
    _epilogue_threads: gl.constexpr
    _epilogue_bias_per_thread: gl.constexpr
    _epilogue_input_lds_layout: gl.constexpr
    _epilogue_gamma_buffer_load_lds_layout: gl.constexpr
    _epilogue_bias_buffer_load_lds_layout: gl.constexpr
    _epilogue_gamma_shape: gl.constexpr
    _epilogue_bias_shape: gl.constexpr
    _epilogue_inputs_via_lds: gl.constexpr
    _quant_staging_layout: gl.constexpr
    _quant_staging_rows: gl.constexpr
    _quant_staging_rotates: gl.constexpr
    # -- resources --
    _lds_bytes: gl.constexpr
    _acc_vgprs_per_lane: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, **values):
        for name, value in values.items():
            setattr(self, "_" + name, gl.constexpr(value))

    # -- block and slot geometry --------------------------------------------------

    @gluon.constexpr_function
    def num_warps(self):
        return _v(self._num_warps)

    @gluon.constexpr_function
    def grid_N(self, N):
        """Number of N tiles. Callable from BOTH host and device.

        Host: take ``.value`` off the returned constexpr before putting it in the grid
        tuple. grid-M comes from the routing metadata (``RoutingData.n_blocks``), never
        from ``cdiv``.
        """
        N = _v(N)
        BN = _v(self._BLOCK_N)
        assert N % BN == 0, f"N ({N}) must be a multiple of BLOCK_N ({BN})"
        return N // BN

    @gluon.constexpr_function
    def num_k_tiles(self, K):
        K = _v(K)
        BK = _v(self._BLOCK_K)
        assert K % BK == 0, f"K ({K}) must be a multiple of BLOCK_K ({BK})"
        return K // BK

    @gluon.constexpr_function
    def num_k_slots_per_tile(self):
        """Payload fragments per BLOCK_K stage."""
        return _v(self._num_k_slots_per_tile)

    @gluon.constexpr_function
    def num_prefetch_k_slots(self):
        """Whether the complete BLOCK_K stage is carried across a pipeline step."""
        return _v(self._num_prefetch_k_slots)

    @gluon.constexpr_function
    def num_m_slots_per_block(self):
        """M slots per CTA block. Operand A and the accumulator split by this."""
        return _v(self._num_m_slots_per_block)

    @gluon.constexpr_function
    def num_n_slots_per_block(self):
        """N slots per CTA block. Operand B and the accumulator split by this."""
        return _v(self._num_n_slots_per_block)

    @gluon.constexpr_function
    def num_lds_slots_per_block_non_k(self, idx):
        """Number of independently copied/read non-K slots in one operand block."""
        return _v(self._num_lds_slots_per_block_non_k)[_v(idx)]

    @gluon.constexpr_function
    def payload_block_k_storage(self, idx):
        """Stored payload elements along K in one pipeline stage."""
        return _v(self._payload_block_k_storage)[_v(idx)]

    @gluon.constexpr_function
    def scale_block_k_storage(self):
        """E8M0 elements along K in one pipeline stage."""
        return _v(self._scale_block_k_storage)

    @gluon.constexpr_function
    def payload_fragment_shape_slot(self, idx):
        """Register payload shape for one non-K slot and K slot."""
        return list(_v(self._payload_fragment_shape_slot)[_v(idx)])

    @gluon.constexpr_function
    def scale_fragment_shape_slot(self, idx):
        """Register E8M0 shape for one non-K slot and K slot."""
        return list(_v(self._scale_fragment_shape_slot)[_v(idx)])

    @gluon.constexpr_function
    def accumulator_fragment_shape_slot(self):
        """Shape of one M-slot by N-slot accumulator tile."""
        return list(_v(self._accumulator_fragment_shape_slot))

    @gluon.constexpr_function
    def payload_lds_shape_slot(self, idx):
        """Shared shape of one payload slot, in stored (packed) elements."""
        return list(_v(self._payload_lds_shape_slot)[_v(idx)])

    @gluon.constexpr_function
    def scale_lds_shape_slot(self, idx):
        """E8M0 shape of one operand slot: [slot non-K extent, BLOCK_K/32]."""
        return list(_v(self._scale_lds_shape_slot)[_v(idx)])

    @gluon.constexpr_function
    def payload_lds_shape_block(self, idx):
        """Full payload LDS shape for one block, including ring and slot axes."""
        return list(_v(self._payload_lds_shape_block)[_v(idx)])

    @gluon.constexpr_function
    def scale_lds_shape_block(self, idx):
        """Full scale LDS shape for one block in its staged representation."""
        return list(_v(self._scale_lds_shape_block)[_v(idx)])

    @gluon.constexpr_function
    def copy_contiguity(self, idx):
        """Elements per lane for a 128-bit direct-to-LDS payload copy."""
        return _v(self._copy_contiguity)[_v(idx)]

    @gluon.constexpr_function
    def k_width_for(self, idx):
        """``DotOperandLayout.k_width`` for one operand, in *stored* elements."""
        return _v(self._k_width_for)[_v(idx)]

    @gluon.constexpr_function
    def payload_hbm_step(self, idx):
        """Stored-element base-pointer increment for one payload K stage."""
        return _v(self._payload_hbm_step)[_v(idx)]

    @gluon.constexpr_function
    def lds_unit_rows(self, idx):
        """Non-K extent of one LDS unit."""
        return _v(self._lds_unit_rows)[_v(idx)]

    @gluon.constexpr_function
    def byte_unit_lds_ok(self, idx):
        """Does :func:`byte_unit_lds_layout` apply to this operand?"""
        return _v(self._byte_unit_lds_ok)[_v(idx)]

    # -- storage placement --------------------------------------------------------
    # Resolved: tuning's ``allow_*_via_lds`` policy and the automatic register fallback
    # for tiles the direct-to-LDS copy cannot take. Each ``*_in_reg`` is exactly the
    # complement of its ``*_via_lds``. None of these is gated on whether a scale
    # exists; callers gate absent scales with ``has_scale``.

    @gluon.constexpr_function
    def component_via_lds(self, component):
        """Whether ``(operand, is_scale)`` is staged through LDS."""
        component = _v(component)
        operand, is_scale = _v(component[0]), bool(_v(component[1]))
        assert operand in (0, 1), "component operand must be 0 (A) or 1 (B)"
        return _v(self._component_via_lds)[operand][int(is_scale)]

    @gluon.constexpr_function
    def component_in_reg(self, component):
        """Whether ``(operand, is_scale)`` is loaded directly into registers."""
        return not self.component_via_lds(component)

    @gluon.constexpr_function
    def payload_via_lds(self, idx):
        return self.component_via_lds((idx, False))

    @gluon.constexpr_function
    def payload_in_reg(self, idx):
        return not self.payload_via_lds(idx)

    @gluon.constexpr_function
    def scale_via_lds(self, idx):
        return self.component_via_lds((idx, True))

    @gluon.constexpr_function
    def scale_in_reg(self, idx):
        return not self.scale_via_lds(idx)

    # -- payload layouts ----------------------------------------------------------

    @gluon.constexpr_function
    def dot_result_fragment_layout(self):
        """AMDMFMALayout for dot / scaled-dot results in register."""
        return _v(self._dot_result_fragment_layout)

    @gluon.constexpr_function
    def dot_operand_fragment_layout(self, idx):
        return _v(self._dot_operand_fragment_layout)[_v(idx)]

    @gluon.constexpr_function
    def dot_operand_lds_layout(self, idx):
        """Padded shared layout for a payload operand staged through LDS."""
        return _v(self._dot_operand_lds_layout)[_v(idx)]

    @gluon.constexpr_function
    def dot_operand_buffer_load_lds_layout(self, idx):
        """Register layout of the global->LDS copy offsets for a payload operand."""
        return _v(self._dot_operand_buffer_load_lds_layout)[_v(idx)]

    @gluon.constexpr_function
    def payload_hbm_offset_layout(self, idx):
        """Layout used to form one payload tile's global-memory offsets."""
        return _v(self._payload_hbm_offset_layout)[_v(idx)]

    # -- scale geometry -----------------------------------------------------------

    @gluon.constexpr_function
    def scale_nonk(self, idx):
        """Non-K extent of one shuffled scale tile."""
        return _v(self._scale_nonk)[_v(idx)]

    @gluon.constexpr_function
    def scale_mini_block_nonk(self, idx):
        """Rows or columns in one scale load; raw scales retain payload geometry."""
        return _v(self._scale_mini_block_nonk)[_v(idx)]

    @gluon.constexpr_function
    def scale_mini_block_k(self, idx):
        """Scale load K extent in logical payload elements, not E8M0 bytes."""
        return _v(self._scale_mini_block_k)[_v(idx)]

    @gluon.constexpr_function
    def scale_ratio_k_step(self, idx):
        """Number of payload K steps covered by one scale load."""
        return _v(self._scale_ratio_k_step)[_v(idx)]

    @gluon.constexpr_function
    def scale_ratio_non_k_slot(self, idx):
        """Number of payload non-K slots covered by one scale load."""
        return _v(self._scale_ratio_non_k_slot)[_v(idx)]

    @gluon.constexpr_function
    def num_scale_tiles(self, idx):
        return _v(self._num_scale_tiles)[_v(idx)]

    @gluon.constexpr_function
    def scale_read_k_slots(self, idx):
        return _v(self._scale_read_k_slots)[_v(idx)]

    @gluon.constexpr_function
    def scale_cache_fragments(self, idx):
        return _v(self._scale_cache_fragments)[_v(idx)]

    @gluon.constexpr_function
    def scale_k_phase(self, step):
        """Which K128 half of the packed K256 scale word this stage consumes."""
        return (_v(step) * _v(self._BLOCK_K) // 128) % 2

    @gluon.constexpr_function
    def scale_hbm_steps(self, idx, steps, phase=0):
        """Scale pointer displacement in units of one payload stage's scale stride."""
        ratio = self.scale_ratio_k_step(idx)
        return ((_v(phase) + _v(steps)) // ratio) * ratio

    @gluon.constexpr_function
    def scale_flat_shape(self, idx):
        """LDS staging shape of a shuffled scale tile (one 256 B run per stripe)."""
        return list(_v(self._scale_flat_shape)[_v(idx)])

    @gluon.constexpr_function
    def sorted_shuffled_ok(self):
        """Does the tile match the layout the C++ A-scale shuffle writes?"""
        return _v(self._sorted_shuffled_ok)

    @gluon.constexpr_function
    def scale_load_shape(self, idx):
        return list(_v(self._scale_load_shape)[_v(idx)])

    @gluon.constexpr_function
    def scale_dword_delta(self, delta, k=256):
        """Where a (non-K, K) step lands, as a coordinate of the i32 tile."""
        return _scale_dword_delta(delta, k)

    @gluon.constexpr_function
    def scale_packed_ok(self, idx):
        """Can this operand's scale fragment be fed as one dword per lane?"""
        return _v(self._scale_packed_ok)[_v(idx)]

    @gluon.constexpr_function
    def scale_packed_sel(self, idx, k_phase=0):
        """Byte selectors in MFMA order, for one K128 half or the full K256 word."""
        return _v(self._scale_packed_sel)[_v(idx)][_v(k_phase)]

    @gluon.constexpr_function
    def mfma_scale_selector(self, idx, k_phase=0):
        """Packed-scale selector for the live MFMA path, or ``None``."""
        return self.scale_packed_sel(idx, k_phase)

    @gluon.constexpr_function
    def has_mfma_packed_scale(self):
        """Whether either live MFMA scale operand uses packed dword storage."""
        return _v(self._has_mfma_packed_scale)

    @gluon.constexpr_function
    def packed_scale_shape(self, idx):
        """Physical i32 storage, including both K128 halves even when one is unused."""
        return list(_v(self._packed_scale_shape)[_v(idx)])

    @gluon.constexpr_function
    def packed_scale_load_shape(self, idx):
        return list(_v(self._packed_scale_load_shape)[_v(idx)])

    # -- scale layouts ------------------------------------------------------------

    @gluon.constexpr_function
    def dot_operand_scale_fragment_layout(self, idx):
        return _v(self._dot_operand_scale_fragment_layout)[_v(idx)]

    @gluon.constexpr_function
    def dot_operand_scale_lds_layout(self, idx):
        """Identity (K-contiguous) shared tile for the raw E8M0 scales."""
        return _v(self._dot_operand_scale_lds_layout)[_v(idx)]

    @gluon.constexpr_function
    def dot_operand_scale_buffer_load_lds_layout(self, idx):
        """32-bit-per-lane blocked layout for the scale direct-to-LDS write."""
        return _v(self._dot_operand_scale_buffer_load_lds_layout)[_v(idx)]

    @gluon.constexpr_function
    def scale_hbm_offset_layout(self, idx):
        """Layout used to form one scale tile's global-memory offsets."""
        return _v(self._scale_hbm_offset_layout)[_v(idx)]

    @gluon.constexpr_function
    def shuffled_scale_read_layout(self, idx, load=False):
        """The [non-K, K] view of the flat LDS run, in fragment order."""
        return _v(self._shuffled_scale_read_layout)[_v(idx)][int(bool(_v(load)))]

    @gluon.constexpr_function
    def scale_load_layout(self, idx):
        return _v(self._scale_load_layout)[_v(idx)]

    @gluon.constexpr_function
    def packed_scale_frag_layout(self, idx):
        """The i32 fragment with its within-dword register bases folded in."""
        return _v(self._packed_scale_frag_layout)[_v(idx)]

    @gluon.constexpr_function
    def packed_scale_read_layout(self, idx, load=False):
        """The i32 view of the flat LDS run: one offset bit per dword-address bit."""
        return _v(self._packed_scale_read_layout)[_v(idx)][int(bool(_v(load)))]

    @gluon.constexpr_function
    def packed_scale_load_layout(self, idx):
        return _v(self._packed_scale_load_layout)[_v(idx)]

    # -- epilogue and emitted-output layouts --------------------------------------

    @gluon.constexpr_function
    def result_store_layout(self, extent_m, extent_n, elem_bits):
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
        num_warps = self.num_warps()
        ept = max(1, (_v(extent_m) * _v(extent_n)) // (num_warps * WARP_SIZE))
        vec = max(1, min(128 // _v(elem_bits), _v(extent_n), ept))
        lanes_n = max(1, min(WARP_SIZE, _v(extent_n) // vec))
        lanes_m = max(1, WARP_SIZE // lanes_n)
        warps_m = max(1, min(num_warps, _v(extent_m) // lanes_m))
        warps_n = max(1, num_warps // warps_m)
        return gl.BlockedLayout(
            size_per_thread=[1, vec],
            threads_per_warp=[lanes_m, lanes_n],
            warps_per_cta=[warps_m, warps_n],
            order=[1, 0],
        )

    @gluon.constexpr_function
    def output_n_slot(self):
        """Emitted columns produced by one logical N-slot epilogue tile."""
        return _v(self._output_n_slot)

    @gluon.constexpr_function
    def output_block_n(self):
        """Emitted columns produced by one CTA's raw BLOCK_N tile."""
        return _v(self._output_block_n)

    @gluon.constexpr_function
    def quant_payload_shape(self, rows, output_n=None):
        """Packed E2M1 byte shape for ``rows`` by emitted-output columns."""
        output_n = self.output_n_slot() if _v(output_n) is None else _v(output_n)
        return [_v(rows), output_n // 2]

    @gluon.constexpr_function
    def quant_scale_shape(self, rows, output_n=None):
        """E8M0 shape for ``rows`` by emitted-output columns."""
        output_n = self.output_n_slot() if _v(output_n) is None else _v(output_n)
        return [_v(rows), output_n // MX_GROUP]

    @gluon.constexpr_function
    def epilogue_threads(self):
        return _v(self._epilogue_threads)

    @gluon.constexpr_function
    def epilogue_bias_per_thread(self):
        return _v(self._epilogue_bias_per_thread)

    @gluon.constexpr_function
    def epilogue_input_lds_layout(self):
        return _v(self._epilogue_input_lds_layout)

    @gluon.constexpr_function
    def epilogue_gamma_buffer_load_lds_layout(self):
        return _v(self._epilogue_gamma_buffer_load_lds_layout)

    @gluon.constexpr_function
    def epilogue_bias_buffer_load_lds_layout(self):
        return _v(self._epilogue_bias_buffer_load_lds_layout)

    @gluon.constexpr_function
    def epilogue_gamma_shape(self):
        return list(_v(self._epilogue_gamma_shape))

    @gluon.constexpr_function
    def epilogue_bias_shape(self):
        return list(_v(self._epilogue_bias_shape))

    @gluon.constexpr_function
    def epilogue_inputs_via_lds(self):
        """Whether bias/gamma vectors are staged through LDS."""
        return _v(self._epilogue_inputs_via_lds)

    @gluon.constexpr_function
    def quant_staging_layout(self):
        return _v(self._quant_staging_layout)

    @gluon.constexpr_function
    def quant_staging_rows(self):
        return _v(self._quant_staging_rows)

    @gluon.constexpr_function
    def quant_staging_rotates(self):
        """Whether the fused-quant staging alternates two per-tile LDS banks."""
        return _v(self._quant_staging_rotates)

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
        extent_m = _v(self._MINI_BLOCK_M) if _v(extent_m) is None else _v(extent_m)
        output_n = self.output_n_slot() if _v(output_n) is None else _v(output_n)
        if _v(self._gu_split):
            instr = _v(self._mfma_instr_shape)
            return max(1, min(MX_GROUP, (instr[0] * instr[1]) // WARP_SIZE))
        per_lane = (extent_m * output_n) // (self.num_warps() * WARP_SIZE)
        return min(MX_GROUP, max(1, per_lane))

    # -- resources ----------------------------------------------------------------

    @gluon.constexpr_function
    def lds_bytes(self):
        return _v(self._lds_bytes)

    @gluon.constexpr_function
    def acc_vgprs_per_lane(self):
        return _v(self._acc_vgprs_per_lane)


@gluon.constexpr_function
def compute_kernel_layout(tuning_cfg):
    """Evaluate every layout of one build from its tuning configuration.

    Host and device both call this on the same configuration, so the host's LDS and
    grid arithmetic and the kernel's tile arithmetic cannot drift. A config whose tile
    lattice is unsound (see :func:`_calc_geometry_ok`) still gets its scalar geometry,
    but its layout objects are ``None``; ``_validate.validate_layout`` rejects it.
    """
    tc = tuning_cfg
    fc = tc.func_cfg
    ok = _calc_geometry_ok(tc)
    ops = (0, 1)

    def pair(fn, *args):
        return tuple(fn(tc, idx, *args) for idx in ops)

    def pair_if(cond, fn, *args):
        return tuple(fn(tc, idx, *args) if ok and cond(idx) else None for idx in ops)

    def always(idx):
        return True

    def has_scale(idx):
        return fc.has_scale(idx)

    bools = (False, True)
    via_lds = tuple(
        tuple(_calc_component_via_lds(tc, (idx, is_scale)) for is_scale in bools)
        for idx in ops
    )

    def payload_via_lds(idx):
        return via_lds[idx][0]

    def scale_via_lds(idx):
        return has_scale(idx) and via_lds[idx][1]

    def scale_tiles_ok(idx):
        # A shuffled scale mini block narrower than the payload slot (which
        # ``validate_layout`` rejects) covers no whole slot: it has no tile count, and
        # no LDS footprint where it is staged there.
        return _calc_scale_ratio_non_k_slot(tc, idx) > 0

    def scale_tiles(fn):
        return tuple(fn(tc, idx) if scale_tiles_ok(idx) else None for idx in ops)

    lds_ok = all(
        scale_tiles_ok(idx)
        or not (has_scale(idx) and (_v(tc.FROZEN_STEP) or via_lds[idx][1]))
        for idx in ops
    )
    epilogue_via_lds = _calc_epilogue_inputs_via_lds(tc)
    return KernelLayout(
        BLOCK_N=_v(tc.BLOCK_N),
        BLOCK_K=_v(tc.BLOCK_K),
        MINI_BLOCK_M=_v(tc.MINI_BLOCK_M),
        mfma_instr_shape=list(_v(tc.mfma_instr_shape)),
        gu_split=fc.gu_split(),
        num_warps=_calc_num_warps(tc),
        num_k_slots_per_tile=_calc_num_k_slots_per_tile(tc),
        num_prefetch_k_slots=_calc_num_prefetch_k_slots(tc),
        num_m_slots_per_block=_calc_num_m_slots_per_block(tc),
        num_n_slots_per_block=_calc_num_n_slots_per_block(tc),
        num_lds_slots_per_block_non_k=pair(_calc_num_lds_slots_per_block_non_k),
        payload_block_k_storage=pair(_calc_payload_block_k_storage),
        scale_block_k_storage=_calc_scale_block_k_storage(tc),
        payload_fragment_shape_slot=pair(_calc_payload_fragment_shape_slot),
        scale_fragment_shape_slot=pair(_calc_scale_fragment_shape_slot),
        accumulator_fragment_shape_slot=_calc_accumulator_fragment_shape_slot(tc),
        payload_lds_shape_slot=pair(_calc_payload_lds_shape_slot),
        scale_lds_shape_slot=pair(_calc_scale_lds_shape_slot),
        payload_lds_shape_block=pair(_calc_payload_lds_shape_block),
        scale_lds_shape_block=scale_tiles(_calc_scale_lds_shape_block),
        copy_contiguity=pair(_calc_copy_contiguity),
        k_width_for=pair(_calc_k_width_for),
        payload_hbm_step=pair(_calc_payload_hbm_step),
        lds_unit_rows=pair(_calc_lds_unit_rows),
        byte_unit_lds_ok=pair(_calc_byte_unit_lds_ok),
        component_via_lds=via_lds,
        dot_result_fragment_layout=_calc_dot_result_fragment_layout(tc) if ok else None,
        dot_operand_fragment_layout=pair_if(always, _calc_dot_operand_fragment_layout),
        dot_operand_lds_layout=pair_if(payload_via_lds, _calc_dot_operand_lds_layout),
        dot_operand_buffer_load_lds_layout=pair_if(
            payload_via_lds, _calc_dot_operand_buffer_load_lds_layout
        ),
        payload_hbm_offset_layout=pair_if(always, _calc_payload_hbm_offset_layout),
        scale_nonk=pair(_calc_scale_nonk),
        scale_mini_block_nonk=pair(_calc_scale_mini_block_nonk),
        scale_mini_block_k=pair(_calc_scale_mini_block_k),
        scale_ratio_k_step=pair(_calc_scale_ratio_k_step),
        scale_ratio_non_k_slot=pair(_calc_scale_ratio_non_k_slot),
        num_scale_tiles=scale_tiles(_calc_num_scale_tiles),
        scale_read_k_slots=pair(_calc_scale_read_k_slots),
        scale_cache_fragments=pair(_calc_scale_cache_fragments),
        scale_flat_shape=pair(_calc_scale_flat_shape),
        sorted_shuffled_ok=_calc_sorted_shuffled_ok(tc),
        scale_load_shape=pair(_calc_scale_load_shape),
        scale_packed_ok=pair(_calc_scale_packed_ok) if ok else (False, False),
        scale_packed_sel=tuple(
            tuple(
                _calc_scale_packed_sel(tc, idx, k_phase) if ok else None
                for k_phase in (0, 1)
            )
            for idx in ops
        ),
        has_mfma_packed_scale=_calc_has_mfma_packed_scale(tc) if ok else False,
        packed_scale_shape=pair(_calc_packed_scale_shape),
        packed_scale_load_shape=pair(_calc_packed_scale_load_shape),
        dot_operand_scale_fragment_layout=pair_if(
            has_scale, _calc_dot_operand_scale_fragment_layout
        ),
        dot_operand_scale_lds_layout=pair(_calc_dot_operand_scale_lds_layout),
        dot_operand_scale_buffer_load_lds_layout=pair_if(
            scale_via_lds, _calc_dot_operand_scale_buffer_load_lds_layout
        ),
        scale_hbm_offset_layout=pair_if(has_scale, _calc_scale_hbm_offset_layout),
        shuffled_scale_read_layout=tuple(
            tuple(
                (
                    _calc_shuffled_scale_read_layout(tc, idx, load)
                    if ok and has_scale(idx)
                    else None
                )
                for load in bools
            )
            for idx in ops
        ),
        scale_load_layout=pair_if(has_scale, _calc_scale_load_layout),
        packed_scale_frag_layout=pair_if(has_scale, _calc_packed_scale_frag_layout),
        packed_scale_read_layout=tuple(
            tuple(
                (
                    _calc_packed_scale_read_layout(tc, idx, load)
                    if ok and has_scale(idx)
                    else None
                )
                for load in bools
            )
            for idx in ops
        ),
        packed_scale_load_layout=pair_if(has_scale, _calc_packed_scale_load_layout),
        output_n_slot=_calc_output_n_slot(tc),
        output_block_n=_calc_output_block_n(tc),
        epilogue_threads=_calc_epilogue_threads(tc),
        epilogue_bias_per_thread=_calc_epilogue_bias_per_thread(tc),
        epilogue_input_lds_layout=_calc_epilogue_input_lds_layout(tc),
        epilogue_gamma_buffer_load_lds_layout=(
            _calc_epilogue_gamma_buffer_load_lds_layout(tc)
            if epilogue_via_lds
            else None
        ),
        epilogue_bias_buffer_load_lds_layout=(
            _calc_epilogue_bias_buffer_load_lds_layout(tc) if epilogue_via_lds else None
        ),
        epilogue_gamma_shape=_calc_epilogue_gamma_shape(tc),
        epilogue_bias_shape=_calc_epilogue_bias_shape(tc),
        epilogue_inputs_via_lds=epilogue_via_lds,
        quant_staging_layout=_calc_quant_staging_layout(tc),
        quant_staging_rows=_calc_quant_staging_rows(tc),
        quant_staging_rotates=_calc_quant_staging_rotates(tc),
        lds_bytes=_calc_lds_bytes(tc) if lds_ok else None,
        acc_vgprs_per_lane=_calc_acc_vgprs_per_lane(tc),
    )
