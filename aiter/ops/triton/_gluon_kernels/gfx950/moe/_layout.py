# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import MX_GROUP, WARP_SIZE, require_constexpr, strip_annotate
from ._packed_scale import (
    packed_scale_dot_layout,
    packed_scale_dot_shape,
    packed_scale_layout,
    packed_scale_lds_view_layout,
    packed_scale_sel,
)
from ._types import WeightGuHbmLayout

__all__ = ["KernelLayout"]

_NO_SCALE: gl.constexpr = gl.constexpr(None)
_WEIGHT_GU_SPLIT: gl.constexpr = gl.constexpr(int(WeightGuHbmLayout.SPLIT))


@gluon.constexpr_function
def _bases_to_distributed(offset_bases, contiguity, num_warps, warp_size, shape):
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


LDS_UNIT_K_BYTES = 128
LDS_LANE_BYTES = 16
LDS_PAD_INTERVAL_BYTES = 1024
LDS_PAD_BYTES = 32


@gluon.constexpr_function
def _pow2_bases(dim, rank, values):
    return [[v if i == dim else 0 for i in range(rank)] for v in values]


@gluon.constexpr_function
def _ramp(lo, hi):
    out = []
    v = lo
    while v < hi:
        out.append(v)
        v *= 2
    return out


@gluon.constexpr_function
def byte_unit_lds_layout(shape, nk_dim, storage_bits, unit_rows, preshuffled):
    shape = [int(x) for x in shape]
    rank = len(shape)
    kd = 1 - nk_dim
    els = lambda nbytes: nbytes * 8 // storage_bits

    rows, k_storage = shape[nk_dim], shape[kd]

    K = lambda vs: _pow2_bases(kd, rank, vs)
    NK = lambda vs: _pow2_bases(nk_dim, rank, vs)

    if preshuffled:
        V = els(LDS_LANE_BYTES)
        bases = K(_ramp(1, V)) + NK(_ramp(1, unit_rows)) + K(_ramp(V, k_storage))
    else:
        assert unit_rows == 32, "the 1, 4, 16, 2, 8 permutation is a 32-row order"
        U = els(LDS_UNIT_K_BYTES)
        bases = K(_ramp(1, U)) + NK([1, 4, 16, 2, 8]) + K(_ramp(U, k_storage))
    bases = bases + NK(_ramp(unit_rows, rows))

    if preshuffled:
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
    run: gl.constexpr,
):
    kk = gl.arange(0, payload_block_k_storage, layout=gl.SliceLayout(1, layout))[
        :, None
    ]
    nn = (n0 + gl.arange(0, n_slot, layout=gl.SliceLayout(0, layout)))[None, :]
    return (
        (nn // 16) * (k_storage * 16)
        + (kk // run) * (16 * run)
        + (nn % 16) * run
        + kk % run
    )


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
    if require_constexpr(packed):
        # packed_scale_shape: one row per 32-row stripe, its dwords in HBM order.
        stripe = gl.arange(0, nonk // 32, layout=gl.SliceLayout(1, layout))[:, None]
        dword = gl.arange(
            0, scale_block_k_storage * 8, layout=gl.SliceLayout(0, layout)
        )[None, :]
        row = stripe * 32
        if require_constexpr(split_extent):
            row = row // split_extent * split_stride + row % split_extent
        offsets = ((nonk0 + row) // 32) * K + dword * 4
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
    offsets_slot = (
        BLOCK_M * block_id + ROW_OFFSET_SLOT + gl.arange(0, M_SLOT, layout=layout)
    )
    live_slot = offsets_slot < M_e
    if require_constexpr(HAS_GATHER):
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
    if func_cfg.weight_gu_hbm_layout.value == WeightGuHbmLayout.SPLIT:
        return 1
    return func_cfg.activation_reduction_n()


@aggregate
@strip_annotate
class KernelLayout:

    _BLOCK_N: gl.constexpr
    _BLOCK_K: gl.constexpr
    _mfma_instr_shape: gl.constexpr
    _weight_gu_hbm_layout: gl.constexpr
    num_warps: gl.constexpr
    num_k_slots: gl.constexpr
    num_prefetch_k_slots: gl.constexpr
    num_m_slots: gl.constexpr
    num_n_slots: gl.constexpr
    _component_via_lds: gl.constexpr
    _payload_block_k_storage: gl.constexpr
    _payload_hbm_step: gl.constexpr
    _payload_byte_unit_lds_ok: gl.constexpr
    payload_a_sorted_shuffled_ok: gl.constexpr
    _payload_shape_tile: gl.constexpr
    _payload_shape_slot: gl.constexpr
    _payload_hbm_offset_layout: gl.constexpr
    _payload_lds_layout: gl.constexpr
    _payload_fragment_layout: gl.constexpr
    scale_block_k_storage: gl.constexpr
    _scale_slot_nonk: gl.constexpr
    _scale_slot_k: gl.constexpr
    _scale_packed_ok: gl.constexpr
    _scale_packed_sel: gl.constexpr
    any_scale_packed: gl.constexpr
    _scale_shape_tile: gl.constexpr
    _scale_slot_shape_storage: gl.constexpr
    _scale_slot_shape_logical: gl.constexpr
    _scale_fragment_shape_slot: gl.constexpr
    _scale_hbm_offset_layout: gl.constexpr
    _scale_lds_layout: gl.constexpr
    _scale_slot_lds_view_layout_storage: gl.constexpr
    _scale_slot_lds_view_layout_logical: gl.constexpr
    _scale_slot_fragment_layout_storage: gl.constexpr
    _scale_slot_fragment_layout_logical: gl.constexpr
    _scale_fragment_layout_slot: gl.constexpr
    acc_vgprs_per_lane: gl.constexpr
    accumulator_fragment_shape_slot: gl.constexpr
    dot_result_fragment_layout: gl.constexpr
    output_n_slot: gl.constexpr
    output_block_n: gl.constexpr
    epilogue_threads: gl.constexpr
    epilogue_inputs_via_lds: gl.constexpr
    epilogue_input_lds_layout: gl.constexpr
    epilogue_gamma_shape: gl.constexpr
    epilogue_gamma_buffer_load_lds_layout: gl.constexpr
    epilogue_bias_shape: gl.constexpr
    epilogue_bias_buffer_load_lds_layout: gl.constexpr
    quant_staging_rows: gl.constexpr
    quant_staging_rotates: gl.constexpr
    quant_staging_layout: gl.constexpr
    lds_bytes: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, tuning_cfg):
        tc = tuning_cfg
        fc = tc.func_cfg
        ops = (0, 1)

        BM, BN, BK = tc.BLOCK_M.value, tc.BLOCK_N.value, tc.BLOCK_K.value
        SLOT_M, SLOT_N = tc.SLOT_M.value, tc.SLOT_N.value
        instr = tc.mfma_instr_shape.value
        warps = tc.warps_per_cta.value
        tiles_per_warp = tc.tiles_per_warp.value

        scale_preshuffled = tuple(tc.preshuffled(idx, True) for idx in ops)
        scale_stage_k = max(256, BK)
        scale_slot_nonk = (
            (tc.SCALE_SLOT_M.value or SLOT_M) if scale_preshuffled[0] else SLOT_M,
            (tc.SCALE_SLOT_N.value or SLOT_N) if scale_preshuffled[1] else SLOT_N,
        )
        scale_slot_k = tuple(
            (tc.SCALE_SLOT_K.value or scale_stage_k) if scale_preshuffled[idx] else BK
            for idx in ops
        )

        slot_nonk = (SLOT_M, SLOT_N)
        assert (
            BK % instr[2] == 0
        ), f"BLOCK_K {BK} must be a multiple of MFMA K {instr[2]}"
        assert BK % MX_GROUP == 0, f"BLOCK_K {BK} must be a multiple of {MX_GROUP}"
        assert BM % SLOT_M == 0, f"SLOT_M {SLOT_M} must divide BLOCK_M {BM}"
        assert BN % SLOT_N == 0, f"SLOT_N {SLOT_N} must divide BLOCK_N {BN}"
        for idx in ops:
            axis = "MN"[idx]
            slot, gran = slot_nonk[idx], instr[idx] * warps[idx] * tiles_per_warp[idx]
            assert slot % gran == 0, (
                f"SLOT_{axis} {slot} must be a multiple of "
                f"instr[{idx}]*warps[{idx}]*tiles[{idx}] = {gran}"
            )
            if not scale_preshuffled[idx]:
                continue
            tile, nonk, sk = (BM, BN)[idx], scale_slot_nonk[idx], scale_slot_k[idx]
            assert (
                tile >= nonk >= slot
            ), f"BLOCK_{axis} >= SCALE_SLOT_{axis} >= SLOT_{axis} is required"
            assert tile % nonk == 0 and nonk % slot == 0, (
                f"SCALE_SLOT_{axis} must divide BLOCK_{axis} and be a multiple of "
                f"SLOT_{axis}"
            )
            assert (
                nonk % 32 == 0
            ), f"SCALE_SLOT_{axis} must contain whole 32-row stripes"
            assert (
                sk >= scale_stage_k
            ), "SCALE_SLOT_K must be at least 256 and at least BLOCK_K"
            assert (
                sk & (sk - 1) == 0 and sk % BK == 0
            ), "SCALE_SLOT_K must be a power of two and a multiple of BLOCK_K"
        assert (
            tc.transposed.value
        ), "transposed=True is pinned, see the module docstring"

        num_warps = warps[0] * warps[1]
        num_k_slots = 1
        num_prefetch_k_slots = 1 if tc.VGPR_PREFETCH_K.value else 0
        num_m_slots = BM // SLOT_M
        num_n_slots = BN // SLOT_N

        threads = num_warps * WARP_SIZE
        acc_vgprs_per_lane = BM * BN // threads
        accumulator_fragment_shape_slot = [SLOT_M, SLOT_N]
        dot_result_fragment_layout = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=instr,
            transposed=True,
            warps_per_cta=warps,
            tiles_per_warp=tiles_per_warp,
        )

        payload_block_k_storage = tuple(BK // fc.pack_divisor(idx) for idx in ops)
        payload_preshuffled = tuple(tc.preshuffled(idx) for idx in ops)
        payload_hbm_step = tuple(
            step * 16 if pre else step
            for step, pre in zip(payload_block_k_storage, payload_preshuffled)
        )
        payload_shape_slot = (
            [SLOT_M, payload_block_k_storage[0]],
            [payload_block_k_storage[1], SLOT_N],
        )
        num_lds_slots_non_k = (num_m_slots, num_n_slots)
        payload_shape_tile = tuple(
            [tc.num_buffers(idx) * num_lds_slots_non_k[idx]] + payload_shape_slot[idx]
            for idx in ops
        )

        storage_dtype = tuple(fc.operand_elem_ty(idx) for idx in ops)
        storage_bits = tuple(ty.primitive_bitwidth for ty in storage_dtype)
        payload_copy_contiguity = tuple(128 // bits for bits in storage_bits)
        payload_via_lds = tuple(
            bool(tc.allow_component_via_lds((idx, False)))
            and (
                payload_shape_slot[idx][0] * payload_shape_slot[idx][1]
                >= WARP_SIZE * payload_copy_contiguity[idx] * num_warps
            )
            for idx in ops
        )
        is_16x16x128 = list(instr) == [16, 16, 128]
        k_width_override = tc.k_width.value
        k_width_for = tuple(
            (
                payload_copy_contiguity[idx]
                if k_width_override is None
                else k_width_override
            )
            for idx in ops
        )
        lds_unit_rows = tuple(16 if pre else 32 for pre in payload_preshuffled)
        lds_unit_k_bytes = tuple(
            LDS_LANE_BYTES if pre else LDS_UNIT_K_BYTES for pre in payload_preshuffled
        )
        payload_byte_unit_lds_ok = []
        for idx in ops:
            rows = payload_shape_slot[idx][idx]
            k_storage = payload_shape_slot[idx][1 - idx]
            unit_k = lds_unit_k_bytes[idx] * 8 // storage_bits[idx]
            read_fits = k_width_for[idx] * storage_bits[idx] == LDS_LANE_BYTES * 8
            if not payload_preshuffled[idx]:
                read_fits = read_fits and list(instr[:2]) == [16, 16]
                if storage_bits[idx] == 8:
                    read_fits = read_fits and tiles_per_warp[idx] >= 2
            payload_byte_unit_lds_ok.append(
                bool(read_fits)
                and rows % lds_unit_rows[idx] == 0
                and k_storage % unit_k == 0
            )
        payload_byte_unit_lds_ok = tuple(payload_byte_unit_lds_ok)
        payload_a_sorted_shuffled_ok = (
            is_16x16x128 and BM % 32 == 0 and SLOT_M % 32 == 0
        )
        payload_fragment_layout = tuple(
            gl.DotOperandLayout(
                operand_index=idx,
                parent=dot_result_fragment_layout,
                k_width=k_width_for[idx],
            )
            for idx in ops
        )
        payload_lds_layout = [None, None]
        dot_operand_buffer_load_lds_layout = [None, None]
        for idx in ops:
            if not payload_via_lds[idx]:
                continue
            shape = payload_shape_slot[idx]
            if payload_byte_unit_lds_ok[idx]:
                layout = byte_unit_lds_layout(
                    shape,
                    idx,
                    storage_bits[idx],
                    lds_unit_rows[idx],
                    payload_preshuffled[idx],
                )
            else:
                layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
                    payload_fragment_layout[idx],
                    shape,
                    storage_dtype[idx],
                    True,
                )
                if layout is None:
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
        payload_lds_layout = tuple(payload_lds_layout)
        dot_operand_buffer_load_lds_layout = tuple(dot_operand_buffer_load_lds_layout)
        payload_hbm_offset_layout = tuple(
            (
                dot_operand_buffer_load_lds_layout[idx]
                if payload_via_lds[idx]
                else payload_fragment_layout[idx]
            )
            for idx in ops
        )

        scale_block_k_storage = BK // MX_GROUP
        scale_slot_shape_logical = tuple(
            [scale_slot_nonk[idx], scale_slot_k[idx] // MX_GROUP] for idx in ops
        )
        # A shuffled scale is stored as its 32-row stripes in HBM byte order, an
        # unshuffled one in its logical shape.
        scale_slot_shape_storage = tuple(
            (
                [scale_slot_nonk[idx] // 32, scale_slot_k[idx]]
                if scale_preshuffled[idx]
                else scale_slot_shape_logical[idx]
            )
            for idx in ops
        )
        scale_shape_tile = tuple(
            [tc.num_buffers(idx, True) * (BM, BN)[idx] // scale_slot_nonk[idx]]
            + scale_slot_shape_storage[idx]
            for idx in ops
        )

        component_via_lds = []
        for idx in ops:
            scale = scale_slot_shape_storage[idx]
            component_via_lds.append(
                (
                    payload_via_lds[idx],
                    bool(tc.allow_component_via_lds((idx, True)))
                    and (
                        bool(scale_preshuffled[idx])
                        or (
                            scale[0] * scale[1] >= WARP_SIZE * 4
                            and scale[1] % 4 == 0
                            and scale[1] >= 8
                        )
                    ),
                )
            )
        component_via_lds = tuple(component_via_lds)
        has_scale = tuple(fc.has_scale(idx) for idx in ops)
        scale_via_lds = tuple(
            has_scale[idx] and component_via_lds[idx][1] for idx in ops
        )
        scale_lds_layout = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )
        scale_fragment_shape_slot = [
            [nonk, scale_block_k_storage] for nonk in (SLOT_M, SLOT_N)
        ]
        scale_fragment_layout_slot = [None, None]
        scale_slot_fragment_layout_logical = [None, None]
        dot_operand_scale_buffer_load_lds_layout = [None, None]
        scale_slot_lds_view_layout_logical = [None, None]
        for idx in ops:
            if not has_scale[idx]:
                continue
            frag = payload_fragment_layout[idx]
            scale_fragment_layout_slot[idx] = gl.amd.cdna4.get_mfma_scale_layout(
                frag, scale_fragment_shape_slot[idx], MX_GROUP
            )
            scale_slot_fragment_layout_logical[idx] = (
                gl.amd.cdna4.get_mfma_scale_layout(
                    frag, scale_slot_shape_logical[idx], MX_GROUP
                )
            )
            if scale_via_lds[idx]:
                lanes_k = (
                    WARP_SIZE if scale_preshuffled[idx] else scale_block_k_storage // 4
                )
                dot_operand_scale_buffer_load_lds_layout[idx] = gl.BlockedLayout(
                    size_per_thread=[1, 4],
                    threads_per_warp=[WARP_SIZE // lanes_k, lanes_k],
                    warps_per_cta=[num_warps, 1],
                    order=[1, 0],
                )
            nonk, k = scale_slot_nonk[idx], scale_slot_k[idx]
            scale_slot_lds_view_layout_logical[idx] = gl.SharedLinearLayout(
                offset_bases=[[16, 0], [0, 4]]
                + _pow2_bases(0, 2, [1, 2, 4, 8])
                + _pow2_bases(1, 2, [1, 2])
                + _pow2_bases(1, 2, _ramp(8, k // MX_GROUP))
                + _pow2_bases(0, 2, _ramp(32, nonk))
            )
        scale_slot_fragment_layout_logical = tuple(scale_slot_fragment_layout_logical)
        dot_operand_scale_buffer_load_lds_layout = tuple(
            dot_operand_scale_buffer_load_lds_layout
        )
        scale_slot_lds_view_layout_logical = tuple(scale_slot_lds_view_layout_logical)

        # Packed scales (_packed_scale.py): a shuffled scale held as whole i32 units
        # and handed to the MFMA with byte selectors. A slot packs when its scale slot
        # has a packed layout and every K phase of the slot has a selector list.
        scale_packed_sel = []
        for idx in ops:
            frag = payload_fragment_layout[idx]
            sel = (None, None)
            if (
                scale_preshuffled[idx]
                and slot_nonk[idx] % 32 == 0
                and packed_scale_layout(frag, scale_slot_nonk[idx], scale_slot_k[idx])
                is not None
            ):
                sel = tuple(
                    packed_scale_sel(
                        frag, slot_nonk[idx], BK, k_phase * BK % scale_stage_k
                    )
                    for k_phase in (0, 1)
                )
                if None in sel:
                    sel = (None, None)
            scale_packed_sel.append(sel)
        scale_packed_sel = tuple(scale_packed_sel)
        scale_packed_ok = tuple(sel[0] is not None for sel in scale_packed_sel)
        any_scale_packed = scale_packed_ok[0] or scale_packed_ok[1]
        # A packed scale slot is read and held in its storage order, as i32: its
        # storage shape with every four bytes of a row one element. Its slot
        # fragments stay i32.
        scale_slot_fragment_layout_storage = [None, None]
        scale_slot_lds_view_layout_storage = [None, None]
        for idx in ops:
            sel = scale_packed_sel[idx][0]
            if sel is None:
                continue
            frag = payload_fragment_layout[idx]
            nonk, k = scale_slot_nonk[idx], scale_slot_k[idx]
            scale_fragment_shape_slot[idx] = packed_scale_dot_shape(
                slot_nonk[idx], BK, sel
            )
            scale_fragment_layout_slot[idx] = packed_scale_dot_layout(
                frag, slot_nonk[idx], BK, sel
            )
            scale_slot_fragment_layout_storage[idx] = packed_scale_layout(
                frag, nonk, k
            )
            scale_slot_lds_view_layout_storage[idx] = packed_scale_lds_view_layout(
                nonk, k
            )
        scale_fragment_shape_slot = tuple(scale_fragment_shape_slot)
        scale_fragment_layout_slot = tuple(scale_fragment_layout_slot)
        scale_slot_fragment_layout_storage = tuple(scale_slot_fragment_layout_storage)
        scale_slot_lds_view_layout_storage = tuple(scale_slot_lds_view_layout_storage)

        scale_hbm_offset_layout = [None, None]
        for idx in ops:
            if not has_scale[idx]:
                continue
            if scale_via_lds[idx]:
                layout = dot_operand_scale_buffer_load_lds_layout[idx]
            elif not scale_preshuffled[idx]:
                layout = scale_fragment_layout_slot[idx]
            elif scale_packed_ok[idx]:
                layout = scale_slot_fragment_layout_storage[idx]
            else:
                frag = scale_slot_fragment_layout_logical[idx]
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

        output_n_slot = SLOT_N // _n_reduction_slot(fc)
        output_block_n = BN // fc.activation_reduction_n()
        epilogue_threads = threads
        epilogue_bias_per_thread = BN // epilogue_threads
        epilogue_inputs_via_lds = (
            epilogue_threads >= BM
            and BN % epilogue_threads == 0
            and epilogue_bias_per_thread in (1, 4)
        )
        epilogue_input_lds_layout = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[0]
        )
        epilogue_gamma_shape = [epilogue_threads]
        epilogue_bias_shape = [BN]
        epilogue_gamma_buffer_load_lds_layout = None
        epilogue_bias_buffer_load_lds_layout = None
        if epilogue_inputs_via_lds:
            epilogue_gamma_buffer_load_lds_layout = gl.BlockedLayout(
                [1], [WARP_SIZE], [num_warps], [0]
            )
            epilogue_bias_buffer_load_lds_layout = gl.BlockedLayout(
                [epilogue_bias_per_thread], [WARP_SIZE], [num_warps], [0]
            )
        gu_split = fc.weight_gu_hbm_layout.value == WeightGuHbmLayout.SPLIT
        quant_staging_rows = BM if gu_split else SLOT_M
        quant_staging_rotates = fc.output_quant.value is not None and not gu_split
        quant_staging_layout = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )

        lds_bytes = 0
        for idx in ops:
            if payload_via_lds[idx]:
                tile = payload_shape_tile[idx]
                lds_bytes += tile[0] * tile[1] * tile[2] * (storage_bits[idx] // 8)
            if scale_via_lds[idx]:
                tile = scale_shape_tile[idx]
                lds_bytes += tile[0] * tile[1] * tile[2]

        self._BLOCK_N = gl.constexpr(BN)
        self._BLOCK_K = gl.constexpr(BK)
        self._mfma_instr_shape = gl.constexpr(list(instr))
        self._weight_gu_hbm_layout = gl.constexpr(fc.weight_gu_hbm_layout.value)
        self.num_warps = gl.constexpr(num_warps)
        self.num_k_slots = gl.constexpr(num_k_slots)
        self.num_prefetch_k_slots = gl.constexpr(num_prefetch_k_slots)
        self.num_m_slots = gl.constexpr(num_m_slots)
        self.num_n_slots = gl.constexpr(num_n_slots)

        self._component_via_lds = gl.constexpr(component_via_lds)

        self._payload_block_k_storage = gl.constexpr(payload_block_k_storage)
        self._payload_hbm_step = gl.constexpr(payload_hbm_step)
        self._payload_byte_unit_lds_ok = gl.constexpr(payload_byte_unit_lds_ok)
        self.payload_a_sorted_shuffled_ok = gl.constexpr(payload_a_sorted_shuffled_ok)
        self._payload_shape_tile = gl.constexpr(payload_shape_tile)
        self._payload_shape_slot = gl.constexpr(payload_shape_slot)
        self._payload_hbm_offset_layout = gl.constexpr(payload_hbm_offset_layout)
        self._payload_lds_layout = gl.constexpr(payload_lds_layout)
        self._payload_fragment_layout = gl.constexpr(payload_fragment_layout)

        self.scale_block_k_storage = gl.constexpr(scale_block_k_storage)
        self._scale_slot_nonk = gl.constexpr(scale_slot_nonk)
        self._scale_slot_k = gl.constexpr(scale_slot_k)
        self._scale_packed_ok = gl.constexpr(scale_packed_ok)
        self._scale_packed_sel = gl.constexpr(scale_packed_sel)
        self.any_scale_packed = gl.constexpr(any_scale_packed)
        self._scale_shape_tile = gl.constexpr(scale_shape_tile)
        self._scale_slot_shape_storage = gl.constexpr(scale_slot_shape_storage)
        self._scale_slot_shape_logical = gl.constexpr(scale_slot_shape_logical)
        self._scale_fragment_shape_slot = gl.constexpr(scale_fragment_shape_slot)
        self._scale_hbm_offset_layout = gl.constexpr(scale_hbm_offset_layout)
        self._scale_lds_layout = gl.constexpr(scale_lds_layout)
        self._scale_slot_lds_view_layout_storage = gl.constexpr(
            scale_slot_lds_view_layout_storage
        )
        self._scale_slot_lds_view_layout_logical = gl.constexpr(
            scale_slot_lds_view_layout_logical
        )
        self._scale_slot_fragment_layout_storage = gl.constexpr(
            scale_slot_fragment_layout_storage
        )
        self._scale_slot_fragment_layout_logical = gl.constexpr(
            scale_slot_fragment_layout_logical
        )
        self._scale_fragment_layout_slot = gl.constexpr(scale_fragment_layout_slot)

        self.acc_vgprs_per_lane = gl.constexpr(acc_vgprs_per_lane)
        self.accumulator_fragment_shape_slot = gl.constexpr(
            accumulator_fragment_shape_slot
        )
        self.dot_result_fragment_layout = gl.constexpr(dot_result_fragment_layout)

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
        self.quant_staging_layout = gl.constexpr(quant_staging_layout)

        self.lds_bytes = gl.constexpr(lds_bytes)

    @gluon.constexpr_function
    def grid_N(self, N):
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
        return (self.num_m_slots.value, self.num_n_slots.value)[idx]

    @gluon.constexpr_function
    def payload_block_k_storage(self, idx):
        return self._payload_block_k_storage.value[idx]

    @gluon.constexpr_function
    def fragment_shape_slot(self, idx, is_scale):
        if is_scale:
            return list(self._scale_fragment_shape_slot.value[idx])
        return list(self._payload_shape_slot.value[idx])

    @gluon.constexpr_function
    def lds_shape_slot(self, idx, is_scale):
        if is_scale:
            return list(self._scale_slot_shape_storage.value[idx])
        return list(self._payload_shape_slot.value[idx])

    @gluon.constexpr_function
    def shape_tile(self, idx, is_scale):
        if is_scale:
            return list(self._scale_shape_tile.value[idx])
        return list(self._payload_shape_tile.value[idx])

    @gluon.constexpr_function
    def payload_hbm_step(self, idx):
        return self._payload_hbm_step.value[idx]

    @gluon.constexpr_function
    def payload_byte_unit_lds_ok(self, idx):
        return self._payload_byte_unit_lds_ok.value[idx]

    @gluon.constexpr_function
    def component_via_lds(self, component):
        operand, is_scale = component[0], bool(component[1])
        assert operand in (0, 1), "component operand must be 0 (A) or 1 (B)"
        return self._component_via_lds.value[operand][int(is_scale)]

    @gluon.constexpr_function
    def component_in_reg(self, component):
        return not self.component_via_lds(component)

    @gluon.constexpr_function
    def fragment_layout(self, idx, is_scale):
        if is_scale:
            return self._scale_fragment_layout_slot.value[idx]
        return self._payload_fragment_layout.value[idx]

    @gluon.constexpr_function
    def lds_layout(self, idx, is_scale):
        if is_scale:
            return self._scale_lds_layout.value
        return self._payload_lds_layout.value[idx]

    @gluon.constexpr_function
    def hbm_offset_layout(self, idx, is_scale):
        if is_scale:
            return self._scale_hbm_offset_layout.value[idx]
        return self._payload_hbm_offset_layout.value[idx]

    @gluon.constexpr_function
    def slot_nonk(self, idx):
        return self.accumulator_fragment_shape_slot.value[idx]

    @gluon.constexpr_function
    def scale_slot_nonk(self, idx):
        return self._scale_slot_nonk.value[idx]

    @gluon.constexpr_function
    def scale_slot_k(self, idx):
        return self._scale_slot_k.value[idx]

    @gluon.constexpr_function
    def scale_slot_shape_storage(self, idx):
        return list(self._scale_slot_shape_storage.value[idx])

    @gluon.constexpr_function
    def scale_slot_shape_logical(self, idx):
        return list(self._scale_slot_shape_logical.value[idx])

    @gluon.constexpr_function
    def scale_packed_ok(self, idx):
        return self._scale_packed_ok.value[idx]

    @gluon.constexpr_function
    def scale_fragment_dtype(self, idx):
        return gl.int32 if self._scale_packed_ok.value[idx] else gl.uint8

    @gluon.constexpr_function
    def scale_packed_sel(self, idx, k_phase=0):
        return self._scale_packed_sel.value[idx][k_phase]

    @gluon.constexpr_function
    def scale_slot_lds_view_layout_storage(self, idx):
        return self._scale_slot_lds_view_layout_storage.value[idx]

    @gluon.constexpr_function
    def scale_slot_lds_view_layout_logical(self, idx):
        return self._scale_slot_lds_view_layout_logical.value[idx]

    @gluon.constexpr_function
    def scale_slot_fragment_layout_storage(self, idx):
        return self._scale_slot_fragment_layout_storage.value[idx]

    @gluon.constexpr_function
    def scale_slot_fragment_layout_logical(self, idx):
        return self._scale_slot_fragment_layout_logical.value[idx]

    @gluon.constexpr_function
    def result_store_layout(self, extent_m, extent_n, storage_bits):
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
        output_n = self.output_n_slot.value if output_n is None else output_n
        return [rows, output_n // 2]

    @gluon.constexpr_function
    def quant_scale_shape(self, rows, output_n=None):
        output_n = self.output_n_slot.value if output_n is None else output_n
        return [rows, output_n // MX_GROUP]

    @gluon.constexpr_function
    def quant_amax_lane_elems(self, extent_m=None, output_n=None):
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

    @gluon.jit
    def n_start(self, pid_n, n_slot_idx: gl.constexpr, N):
        n_slot: gl.constexpr = self.accumulator_fragment_shape_slot[1]
        if require_constexpr(self._weight_gu_hbm_layout == _WEIGHT_GU_SPLIT):
            out = pid_n * n_slot + n_slot_idx * (N // 2)
        else:
            out = pid_n * self._BLOCK_N + n_slot_idx * n_slot
        return out

    @gluon.jit
    def n_split_offs(self, pid_n, i, N):
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
        N_SLOT: gl.constexpr = tuning_cfg.SLOT_N
        NUM_N_SLOTS: gl.constexpr = self.num_n_slots
        PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = self.payload_block_k_storage(1)
        layout_b_slot: gl.constexpr = self.hbm_offset_layout(1, False)
        K_STORAGE: gl.constexpr = K // func_cfg.pack_divisor(1)
        RUN: gl.constexpr = 128 // func_cfg.operand_elem_ty(1).primitive_bitwidth
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
                        RUN,
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
        if require_constexpr(not func_cfg.has_scale(0)):
            return _NO_SCALE

        M_TILE: gl.constexpr = tuning_cfg.BLOCK_M
        M_SLOT: gl.constexpr = tuning_cfg.SLOT_M
        NUM_M_SLOTS: gl.constexpr = self.num_m_slots
        SCALE_BLOCK_K_STORAGE: gl.constexpr = self.scale_block_k_storage
        layout_a_scale_slot: gl.constexpr = self.hbm_offset_layout(0, True)
        SCALE_M: gl.constexpr = self.scale_slot_nonk(0)
        SCALE_K: gl.constexpr = self.scale_slot_k(0)
        SCALE_RATIO: gl.constexpr = SCALE_M // M_SLOT

        if require_constexpr(
            tuning_cfg.preshuffled(0, True) and self.component_in_reg((0, True))
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
            gl.static_assert(
                self.payload_a_sorted_shuffled_ok,
                "A_SCALE_SORTED_SHUFFLED requires MFMA 16x16x128 "
                "and whole 32-row A stripes",
            )
            SCALE_FILL_SHAPE: gl.constexpr = self.scale_slot_shape_storage(0)
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
        if require_constexpr(not func_cfg.has_scale(1)):
            return _NO_SCALE

        N_SLOT: gl.constexpr = tuning_cfg.SLOT_N
        NUM_N_SLOTS: gl.constexpr = self.num_n_slots
        SCALE_BLOCK_K_STORAGE: gl.constexpr = self.scale_block_k_storage
        layout_b_scale_slot: gl.constexpr = self.hbm_offset_layout(1, True)
        SCALE_N: gl.constexpr = self.scale_slot_nonk(1)
        SCALE_K: gl.constexpr = self.scale_slot_k(1)
        SCALE_RATIO: gl.constexpr = SCALE_N // N_SLOT
        SPLIT_EXTENT: gl.constexpr = (
            N_SLOT
            if func_cfg.weight_gu_hbm_layout == _WEIGHT_GU_SPLIT and SCALE_RATIO > 1
            else 0
        )

        if require_constexpr(
            tuning_cfg.preshuffled(1, True) and self.component_in_reg((1, True))
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
            SCALE_FILL_SHAPE: gl.constexpr = self.scale_slot_shape_storage(1)
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
