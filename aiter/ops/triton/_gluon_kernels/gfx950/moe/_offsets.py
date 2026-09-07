# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HBM operand offsets and mini-tile indexing for gfx950 Gluon MoE GEMMs."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import require_constexpr
from ._lang import unwrap as _v

_NO_SCALE: gl.constexpr = gl.constexpr(None)


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


@gluon.jit
def _n_start(pid_n, ni: gl.constexpr, N, func_cfg, tuning_cfg):
    """Raw-N index where mini-N block ``ni`` of CTA column ``pid_n`` begins.

    The one place the gate/up packing is expressed. Interleaved, the CTA tile is a
    contiguous ``BLOCK_N`` run and mini blocks slice it. Split, a CTA owns
    ``MINI_BLOCK_N`` *emitted* channels and reads each side from its own half of N, so
    the two mini blocks are ``N/2`` apart -- their tiles land at the same emitted
    channels, which is what lets the epilogue pair them elementwise.

    Everything downstream of this addresses a plain global ``n``: the 16-column
    preshuffle in :func:`_blocked_b_hbm_offsets`, both scale shuffles and every LDS tile are
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

    Same mapping, expressed elementwise, for the bias staging copy that addresses the
    CTA tile as one run. Uniform arithmetic on a loop-invariant tensor is hoisted.
    """
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * MBN + (i // MBN) * (N // 2) + i % MBN
    else:
        out = pid_n * tuning_cfg.BLOCK_N + i
    return out


@gluon.jit
def _blocked_b_hbm_offsets(
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

    The copy layout matches the direct-to-LDS tile's ``byte_unit_lds_layout``
    permutation.
    """
    kk = gl.arange(0, PK_B, layout=gl.SliceLayout(1, layout))[:, None]
    nn = (n0 + gl.arange(0, MBN, layout=gl.SliceLayout(0, layout)))[None, :]
    return (nn // 16) * (KB * 16) + (kk // 16) * 256 + (nn % 16) * 16 + kk % 16


@gluon.jit
def _a_payload_hbm_offsets(a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg):
    """Gathered A payload offsets, one copy-layout grid per mini-M tile."""
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    MBM: gl.constexpr = tuning_cfg.MINI_BLOCK_M
    NM: gl.constexpr = tuning_cfg.num_mini_m()
    PK_A: gl.constexpr = tuning_cfg.BLOCK_K // func_cfg.a_pack_divisor()
    cl_a: gl.constexpr = tuning_cfg.dot_operand_copy_layout(0)
    a_hbm_offs = ()
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
        a_hbm_offs = a_hbm_offs + (
            rows_a[:, None] * a.stride_m
            + gl.arange(0, PK_A, layout=gl.SliceLayout(0, cl_a))[None, :],
        )

    return a_hbm_offs


@gluon.jit
def _b_payload_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg):
    """B payload offsets for plain or preshuffled weights, per mini-N tile."""
    MBN: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    NN: gl.constexpr = tuning_cfg.num_mini_n()
    PK_B: gl.constexpr = tuning_cfg.BLOCK_K // func_cfg.b_pack_divisor()
    cl_b: gl.constexpr = tuning_cfg.dot_operand_copy_layout(1)
    KB: gl.constexpr = K // func_cfg.b_pack_divisor()
    b_hbm_offs = ()
    for ni in gl.static_range(NN):
        if require_constexpr(tuning_cfg.B_PRESHUFFLED):
            b_hbm_offs = b_hbm_offs + (
                _blocked_b_hbm_offsets(
                    cl_b,
                    PK_B,
                    _n_start(pid_n, ni, N, func_cfg, tuning_cfg),
                    MBN,
                    KB,
                ),
            )
        else:
            b_hbm_offs = b_hbm_offs + (
                gl.arange(0, PK_B, layout=gl.SliceLayout(1, cl_b))[:, None] * b.stride_k
                + (
                    _n_start(pid_n, ni, N, func_cfg, tuning_cfg)
                    + gl.arange(0, MBN, layout=gl.SliceLayout(0, cl_b))
                )[None, :]
                * b.stride_n,
            )

    return b_hbm_offs


@gluon.jit
def _a_scale_hbm_offsets(a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg):
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
        # the byte it needs. A K256 scale group spans 256 B per row stripe;
        # K128 payload stages reuse the same group before advancing the pointer.
        gl.static_assert(
            tuning_cfg.sorted_shuffled_ok(),
            "A_SCALE_SORTED_SHUFFLED requires MFMA 16x16x128, BLOCK_K 128 or 256, "
            "and whole 32-row A stripes",
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
        a_scale_hbm_offs = ()
        for mi in gl.static_range(NM):
            a_scale_hbm_offs = a_scale_hbm_offs + (
                (pid_m * (BM // 32) + (mi // RA) * (SMM // 32) + sa) * K + ja,
            )
    else:
        a_scale_hbm_offs = ()
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
            a_scale_hbm_offs = a_scale_hbm_offs + (
                rows_as[:, None] * a.scale_stride_m
                + gl.arange(0, SK, layout=gl.SliceLayout(0, asl))[None, :]
                * a.scale_stride_k,
            )
    return a_scale_hbm_offs


@gluon.jit
def _b_scale_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg):
    """Byte offsets of the B scale tiles, one per mini-N block (``None`` if unscaled).

    Either a flat CDNA4_SCALE preshuffle run for staging or the raw ``(K/32, N)``
    grid. The caller receives a tuple indexed by mini block.
    """
    if require_constexpr(not func_cfg.b_has_scale()):
        return _NO_SCALE

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

    if require_constexpr(tuning_cfg.B_SCALE_SHUFFLED):
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
        b_scale_hbm_offs = ()
        for ni in gl.static_range(NN):
            b_scale_hbm_offs = b_scale_hbm_offs + (
                (_n_start(pid_n, ni, N, func_cfg, tuning_cfg) // 32 + sb) * K + jb,
            )
    else:
        b_scale_hbm_offs = ()
        for ni in gl.static_range(NN):
            b_scale_hbm_offs = b_scale_hbm_offs + (
                (
                    _n_start(pid_n, ni, N, func_cfg, tuning_cfg)
                    + gl.arange(0, MBN, layout=gl.SliceLayout(1, bsl))
                )[:, None]
                * b.scale_stride_n
                + gl.arange(0, SK, layout=gl.SliceLayout(0, bsl))[None, :]
                * b.scale_stride_k,
            )
    return b_scale_hbm_offs


@gluon.constexpr_function
def _slot_index(mi, ni, NM, NN):
    """Position of slot ``(mi, ni)`` in the traversal: N outer, M inner.

    This one number is the slot's place in three orderings at once -- the visitation
    order of the slot loop, the index of its accumulator in the carried tuple, and
    (under the even schedule) the position of its fill in :func:`_buffer_load_order`. They have
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
