# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Frozen gfx950 Gluon MoE kernel: a self-contained snapshot of the best measured build.

``_pipeline_step_frozen`` is a verbatim copy of ``moe_gemm._pipeline_step_impl`` taken
from the best measured kernel (2026-09-02), with every env-var constexpr replaced by its
tuned value so no stray flag can perturb the reference schedule.
``AITER_TRITON_MOE_GLUON_FROZEN_STEP=1`` (``FROZEN_STEP`` in the tuning config) selects
this body at the kernel entry. The complete kernel lives here: offsets, LDS allocation
and access, dot selection, pipeline state, prologue, runtime loop, drain, final MFMA
and epilogue.

The module is self-contained. From this package it imports only the two leaves,
``_lang`` and ``_types``, which hold the language helpers and the enums the host shares
with the kernel. Everything else it needs is a private copy kept here:

* the config rephrasings, as functions over the raw ``func_cfg`` / ``tuning_cfg``
  fields (``_has_scale_frozen``, ``_num_buffers_frozen``, ...);
* the layout subset the body reads, as ``_FrozenLayout``, built by
  ``_compute_frozen_layout``;
* the schedule helpers, the HBM offset math, the epilogue with its MXFP4 quant and
  SwiGLU, and the build validation (``_validate_frozen``,
  ``_validate_frozen_buffer_counts``, ``_validate_frozen_pipeline``).

So no change to ``_config``, ``_layout``, ``_validate``, ``_schedule``, ``_epilogue``,
``_lds``, ``_pipeline``, ``_quant``, ``_schedule_cache`` or ``moe_gemm`` can move the
frozen kernel, and those modules may diverge from it freely. The body reads only the
raw fields of the two config aggregates and of the tensor aggregates passed in; it calls
no method of ``KernelFuncConfig``, ``KernelTuningConfig`` or ``KernelLayout``. Only
``_entry`` (to select the body) and the host wrapper (to validate a FROZEN_STEP build)
import this module.

The acceptance test is byte-identical generated code: the frozen assembly of every
dispatchable cell must not move when this module is edited, nor when anything outside
it is.
"""

import math

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate
from triton.language.core import base_value
from triton.language.extra.hip import libdevice as hip_libdevice

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.common_utils import strip_annotate

from ._lang import MX_GROUP as MX_GROUP_INT
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import WARP_SIZE, require_constexpr
from ._lang import optional as _opt
from ._lang import unwrap as _v
from ._types import (
    A,
    DotKind,
    DtypeQuant,
    EpilogueMode,
    SchedMode,
    TileSched,
    WaitCommitScheme,
    WarpPipeline,
    WeightGuHbmLayout,
    dq_has_scale,
    dq_mx_format,
    dq_pack_divisor,
)

# (operand, is_scale) pairs in _types.COMPONENTS order, kept as tuples here.
COMPONENTS = ((0, False), (0, True), (1, False), (1, True))


# --- vendored config rephrasings ------------------------------------------------
# Private copies of the ``KernelFuncConfig`` / ``KernelTuningConfig`` methods this
# snapshot reads, as functions over the aggregates' raw fields. The config classes may
# rename or reshape their methods freely; only a field rename reaches in here, and it
# fails loudly rather than silently changing the frozen kernel.
# ------------------------------------------------------------------------------
@gluon.constexpr_function
def _dtype_quant_frozen(fc, idx):
    if _v(idx) == 0:
        return _v(fc.token_dtype_quant)
    return _v(fc.expert_dtype_quant)


@gluon.constexpr_function
def _online_quant_frozen(fc, idx):
    if _v(idx) == 0:
        return _v(fc.token_online_quant)
    return _v(fc.expert_online_quant)


@gluon.constexpr_function
def _has_scale_frozen(fc, idx):
    return dq_has_scale(_dtype_quant_frozen(fc, idx))


@gluon.constexpr_function
def _pack_divisor_frozen(fc, idx):
    return dq_pack_divisor(_dtype_quant_frozen(fc, idx))


@gluon.constexpr_function
def _mx_format_frozen(fc, idx):
    return dq_mx_format(_online_quant_frozen(fc, idx))


@gluon.constexpr_function
def _operand_elem_ty_frozen(fc, idx):
    """Storage element type of one operand's payload, in LDS and in HBM.

    MXFP4 is two E2M1 values per byte, so it is stored and loaded as ``uint8`` and
    only ``mfma_scaled``'s ``e2m1`` format string tells the hardware otherwise.
    """
    dq = _dtype_quant_frozen(fc, idx)
    if dq == int(DtypeQuant.BF16):
        return gl.bfloat16
    if dq in (int(DtypeQuant.FP8_E4M3), int(DtypeQuant.MXFP8)):
        return gl.float8e4nv
    return gl.uint8


@gluon.constexpr_function
def _dot_kind_frozen(fc):
    """Which dot the operand pair maps onto: MFMA, MFMA_SCALED or UPCAST_MFMA."""
    a_bf16 = _online_quant_frozen(fc, 0) == int(DtypeQuant.BF16)
    b_bf16 = _online_quant_frozen(fc, 1) == int(DtypeQuant.BF16)
    if a_bf16 and b_bf16:
        return int(DotKind.MFMA)
    if a_bf16 or b_bf16:
        return int(DotKind.UPCAST_MFMA)
    return int(DotKind.MFMA_SCALED)


@gluon.constexpr_function
def _act_frozen(fc):
    return _v(fc.activation)


@gluon.constexpr_function
def _has_activation_frozen(fc):
    return _v(fc.activation) is not None


@gluon.constexpr_function
def _activation_reduction_n_frozen(fc):
    """Emitted columns per raw column. 2 for a gated activation, 1 otherwise."""
    return 2 if _v(fc.activation) is not None else 1


@gluon.constexpr_function
def _gu_split_frozen(fc):
    """Is the gated pair laid out as two N halves rather than interleaved?

    Only meaningful with a gated activation -- the flag is inert otherwise.
    """
    return (
        _v(fc.weight_gu_hbm_layout) == WeightGuHbmLayout.SPLIT
        and _v(fc.activation) is not None
    )


@gluon.constexpr_function
def _n_reduction_slot_frozen(fc):
    """Emitted columns per raw column within one N slot."""
    return 1 if _gu_split_frozen(fc) else _activation_reduction_n_frozen(fc)


@gluon.constexpr_function
def _num_buffers_frozen(tc, operand, scale=False):
    """Ring depth for one payload or scale; zero inherits NUM_LDS_BUFFER."""
    operand = _v(operand)
    assert operand in (0, 1), "operand must be 0 (A) or 1 (B)"
    if _v(scale):
        depth = tc.A_SCALE_NUM_BUFFER if operand == 0 else tc.B_SCALE_NUM_BUFFER
    else:
        depth = tc.A_NUM_BUFFER if operand == 0 else tc.B_NUM_BUFFER
    return _v(depth) or _v(tc.NUM_LDS_BUFFER)


@gluon.constexpr_function
def _allow_payload_via_lds_frozen(tc, idx):
    """``ALLOW_B_VIA_LDS`` for B; A has no knob and is always permitted."""
    return _v(idx) == 0 or bool(_v(tc.ALLOW_B_VIA_LDS))


@gluon.constexpr_function
def _allow_scale_via_lds_frozen(tc, idx):
    """``ALLOW_A_SCALE_VIA_LDS`` / ``ALLOW_B_SCALE_VIA_LDS``."""
    return bool(
        _v(tc.ALLOW_A_SCALE_VIA_LDS if _v(idx) == 0 else tc.ALLOW_B_SCALE_VIA_LDS)
    )


@gluon.constexpr_function
def _allow_component_via_lds_frozen(tc, component):
    """The ``allow_*_via_lds`` policy for an ``(operand, is_scale)`` component."""
    component = _v(component)
    operand, is_scale = _v(component[0]), bool(_v(component[1]))
    assert operand in (0, 1), "component operand must be 0 (A) or 1 (B)"
    return (
        _allow_scale_via_lds_frozen(tc, operand)
        if is_scale
        else _allow_payload_via_lds_frozen(tc, operand)
    )


@gluon.constexpr_function
def _operand_preshuffled_frozen(tc, idx):
    """Is this operand's payload already blocked 16-column in HBM? Never for A."""
    return _v(idx) == 1 and _v(tc.B_PRESHUFFLED)


@gluon.constexpr_function
def _scale_shuffled_frozen(tc, idx):
    """Is this operand's scale tile already in MFMA fragment order in memory?"""
    if not _has_scale_frozen(tc.func_cfg, idx):
        return False
    if _v(idx) == 0:
        return _v(tc.A_SCALE_SORTED_SHUFFLED)
    return _v(tc.B_SCALE_SHUFFLED)


@gluon.constexpr_function
def _warp_pipeline_enabled_frozen(tc):
    """Any ping-pong at all -- what the VGPR_PREFETCH_K precondition keys on."""
    return _v(tc.WARP_PIPELINE) != int(WarpPipeline.NONE)


# --- vendored schedule helpers ------------------------------------------------
# Private copies of the schedule-model helpers this snapshot reads. The live model
# (``_schedule.py``) is refactored freely; these hold the frozen fill order, ring
# depths and unroll fixed. ``test_moe_gemm_gluon_frozen.py`` pins their values.
# ------------------------------------------------------------------------------
@gluon.constexpr_function
def _present_frozen(tc, component):
    operand, is_scale = component[0], component[1]
    return not _v(is_scale) or _has_scale_frozen(tc.func_cfg, operand)


@gluon.constexpr_function
def _component_depth_frozen(tc, component):
    return _num_buffers_frozen(tc, component[0], component[1])


@gluon.constexpr_function
def _component_ratio_frozen(tc, layout, component):
    operand, is_scale = component[0], component[1]
    return layout.scale_ratio_k_step(operand) if _v(is_scale) else 1


@gluon.constexpr_function
def _live_span_frozen(tc, layout, component):
    """Baseline inclusive lifetime ``L(c)`` in payload K steps."""
    return (
        _component_depth_frozen(tc, component) - 1
    ) * _component_ratio_frozen(tc, layout, component) + 1


@gluon.constexpr_function
def pipeline_depth_frozen(tc, layout):
    """Largest prefetch span, measured in payload steps, of active components."""
    depth = 0
    for component in COMPONENTS:
        if _present_frozen(tc, component):
            depth = max(depth, _live_span_frozen(tc, layout, component))
    return depth


@gluon.constexpr_function
def pipeline_unroll_frozen(tc, layout):
    """LCM of the requested unroll and every active component ring period."""
    requested = _v(tc.K_UNROLL)
    assert requested >= 1, "K_UNROLL must be at least 1"
    period = requested
    for component in COMPONENTS:
        if _present_frozen(tc, component):
            period = math.lcm(
                period,
                _component_depth_frozen(tc, component)
                * _component_ratio_frozen(tc, layout, component),
            )
    return period


@gluon.constexpr_function
def _buffer_load_order_frozen(NM, NN):
    """Payload copies as ``(is_a, tile)`` in slot order."""
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
def _buffer_load_tile_frozen(pos, NM, NN, want_a):
    pos = _v(pos)
    if pos is None:
        return None
    is_a, tile = _buffer_load_order_frozen(NM, NN)[pos]
    return tile if bool(is_a) == bool(_v(want_a)) else None


@gluon.constexpr_function
def _payload_fill_slots_frozen(NM, NN):
    """Slot that owns each payload tile, as ``(A tiles, B tiles)``.

    With both axes split, the interleaved order (and its 2x2 special case) assigns one
    payload copy per slot. With a degenerate axis there are fewer slots than copies,
    so fall back to axis-major: ``A(t)`` at ``(mi=t, ni=0)`` and ``B(t)`` at
    ``(mi=0, ni=t)``.
    """
    NM, NN = _v(NM), _v(NN)
    if NM > 1 and NN > 1:
        order = _buffer_load_order_frozen(NM, NN)
        return (
            tuple(order.index((1, tile)) for tile in range(NM)),
            tuple(order.index((0, tile)) for tile in range(NN)),
        )
    return (
        tuple(_slot_index_frozen(tile, 0, NM, NN) for tile in range(NM)),
        tuple(_slot_index_frozen(0, tile, NM, NN) for tile in range(NN)),
    )


@gluon.constexpr_function
def _component_fill_slots_frozen(tc, layout, component):
    """Slot that owns each non-K tile of one component.

    ``SCALE_FILL_MID`` is guarded to the 2x2 split, so on any other geometry it is
    inert.
    """
    nm, nn = _v(layout.num_m_slots_per_block()), _v(layout.num_n_slots_per_block())
    a_slots, b_slots = _payload_fill_slots_frozen(nm, nn)
    if _v(component[1]) and _v(tc.SCALE_FILL_MID) and nm == 2 and nn == 2:
        return (1, 2)
    return a_slots if _v(component[0]) == A else b_slots


@gluon.constexpr_function
def _fill_slots_valid_frozen(tc, layout):
    """Every mapped slot exists, and no component owns two tiles in one slot."""
    slots = _v(layout.num_m_slots_per_block()) * _v(layout.num_n_slots_per_block())
    for component in COMPONENTS:
        mapping = _component_fill_slots_frozen(tc, layout, component)
        expected = (
            _v(layout.num_m_slots_per_block())
            if _v(component[0]) == A
            else _v(layout.num_n_slots_per_block())
        )
        if len(mapping) != expected:
            return False
        if any(slot < 0 or slot >= slots for slot in mapping):
            return False
        if len(set(mapping)) != len(mapping):
            return False
    return True


_TS_XCD_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.XCD_GROUP_M))
_TS_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.GROUP_M))
_DK_MFMA: gl.constexpr = gl.constexpr(int(DotKind.MFMA))
_DK_MFMA_SCALED: gl.constexpr = gl.constexpr(int(DotKind.MFMA_SCALED))
_DK_UPCAST_MFMA: gl.constexpr = gl.constexpr(int(DotKind.UPCAST_MFMA))
_NO_SCALE: gl.constexpr = gl.constexpr(None)


# --- vendored layout ----------------------------------------------------------
# The subset of ``_layout.KernelLayout`` this snapshot reads, as a private aggregate
# (``_FrozenLayout``) and builder (``_compute_frozen_layout``). Each derivation is a
# copy of its ``_layout._calc_*`` counterpart with the config methods replaced by the
# raw-field rephrasings above, and the aggregate holds only the fields the frozen body,
# epilogue and validation read. The live layout may add, drop or reshape anything
# without reaching in here.
# ------------------------------------------------------------------------------

#: gfx950 LDS capacity, mirrors utils/_triton/arch_info.py::_LDS_CAP_BYTES["gfx950"].
_LDS_CAP_BYTES = 163840
#: Headroom the operand buffers must leave for LDS the compiler allocates on top of
#: them (the epilogue's convert_layout scratch).
_LDS_EPILOGUE_RESERVE_BYTES = 4096
#: What the operand buffers may actually use.
_LDS_USABLE_BYTES = _LDS_CAP_BYTES - _LDS_EPILOGUE_RESERVE_BYTES
#: K extent of one LDS unit, in bytes.
_LDS_UNIT_K_BYTES = 128
#: The 16 B a lane moves per access -- the granule the preshuffled HBM order blocks by.
_LDS_LANE_BYTES = 16
#: Padding of a non-preshuffled unit: 32 B every 1024 B.
_LDS_PAD_INTERVAL_BYTES = 1024
_LDS_PAD_BYTES = 32


@gluon.constexpr_function
def _bases_to_distributed_frozen(offset_bases, contiguity, num_warps, warp_size, shape):
    """Turn a ``PaddedSharedLayout``'s offset bases into the matching register layout.

    Mirrors Triton's ``CoalesceAsyncCopy`` partition -- lg2(C) bases to reg, lg2(WS) to
    lane, lg2(NW) to warp, leftovers back to reg -- which is what makes
    ``buffer_load_to_shared`` fold into a single ``buffer_load_dwordx4 ... lds``.
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


@gluon.constexpr_function
def _pow2_bases_frozen(dim, rank, values):
    """``[[v, 0], ...]`` (or ``[[0, v], ...]``) for each stride in ``values``."""
    return [[v if i == dim else 0 for i in range(rank)] for v in values]


@gluon.constexpr_function
def _ramp_frozen(lo, hi):
    """Powers of two ``lo, 2*lo, ... hi/2``; empty when ``hi <= lo``."""
    out = []
    v = lo
    while v < hi:
        out.append(v)
        v *= 2
    return out


@gluon.constexpr_function
def _byte_unit_lds_layout_frozen(shape, nk_dim, elem_bits, unit_rows, preshuffled):
    """LDS layout of a payload operand, tiled by a fixed (rows x 128 B) unit.

    Not preshuffled: 32 rows, the row permutation ``1, 4, 16, 2, 8``, and 32 B of
    padding every 1024 B. Preshuffled: 16 rows in the byte order
    ``utils/shuffle.py::shuffle_weight(w, (16, 16))`` already writes, no padding.
    """
    shape = [int(x) for x in _v(shape)]
    nk_dim = _v(nk_dim)
    elem_bits = _v(elem_bits)
    unit_rows = _v(unit_rows)
    rank = len(shape)
    kd = 1 - nk_dim
    els = lambda nbytes: nbytes * 8 // elem_bits

    U = els(_LDS_UNIT_K_BYTES)  # elements per 128 B K unit
    V = els(_LDS_LANE_BYTES)  # elements per 16 B lane access
    rows, kelems = shape[nk_dim], shape[kd]

    K = lambda vs: _pow2_bases_frozen(kd, rank, vs)
    NK = lambda vs: _pow2_bases_frozen(nk_dim, rank, vs)

    if preshuffled:
        # 16 B of K, then the 16 rows, then the rest of the 128 B unit.
        bases = K(_ramp_frozen(1, V)) + NK(_ramp_frozen(1, unit_rows)) + K(
            _ramp_frozen(V, U)
        )
    else:
        assert unit_rows == 32, "the 1, 4, 16, 2, 8 permutation is a 32-row order"
        bases = K(_ramp_frozen(1, U)) + NK([1, 4, 16, 2, 8])
    bases = bases + K(_ramp_frozen(U, kelems)) + NK(_ramp_frozen(unit_rows, rows))

    if preshuffled:
        # PaddedSharedLayout insists on at least one interval/padding pair, so name an
        # interval wider than the tile: no pad can land inside it.
        pairs = [[1 << (rows * kelems).bit_length(), 16]]
    else:
        pairs = [[els(_LDS_PAD_INTERVAL_BYTES), els(_LDS_PAD_BYTES)]]
    return gl.PaddedSharedLayout(
        interval_padding_pairs=pairs,
        offset_bases=bases,
        cga_layout=[],
        shape=shape,
    )


@gluon.constexpr_function
def _calc_num_warps_frozen(tc):
    w = _v(tc.warps_per_cta)
    return w[0] * w[1]


@gluon.constexpr_function
def _calc_num_k_slots_per_tile_frozen(tc):
    """Payload fragments per BLOCK_K stage."""
    return 1


@gluon.constexpr_function
def _calc_num_prefetch_k_slots_frozen(tc):
    """Whether the complete BLOCK_K stage is carried across a pipeline step."""
    return 1 if _v(tc.VGPR_PREFETCH_K) else 0


@gluon.constexpr_function
def _calc_num_m_slots_per_block_frozen(tc):
    """M slots per CTA block. Operand A and the accumulator split by this."""
    return _v(tc.BLOCK_M) // _v(tc.SLOT_M)


@gluon.constexpr_function
def _calc_num_n_slots_per_block_frozen(tc):
    """N slots per CTA block. Operand B and the accumulator split by this."""
    return _v(tc.BLOCK_N) // _v(tc.SLOT_N)


@gluon.constexpr_function
def _calc_num_lds_slots_per_block_non_k_frozen(tc, idx):
    """Number of independently copied/read non-K slots in one operand block."""
    return (
        _calc_num_m_slots_per_block_frozen(tc)
        if _v(idx) == 0
        else _calc_num_n_slots_per_block_frozen(tc)
    )


@gluon.constexpr_function
def _calc_payload_block_k_storage_frozen(tc, idx):
    """Stored payload elements along K in one pipeline stage."""
    return _v(tc.BLOCK_K) // _pack_divisor_frozen(tc.func_cfg, idx)


@gluon.constexpr_function
def _calc_scale_block_k_storage_frozen(tc):
    """E8M0 elements along K in one pipeline stage."""
    return _v(tc.BLOCK_K) // MX_GROUP_INT


@gluon.constexpr_function
def _calc_scale_fragment_shape_slot_frozen(tc, idx):
    """Register E8M0 shape for one non-K slot and K slot."""
    nonk = _v(tc.SLOT_M) if _v(idx) == 0 else _v(tc.SLOT_N)
    return [nonk, _v(tc.BLOCK_K) // MX_GROUP_INT]


@gluon.constexpr_function
def _calc_payload_lds_shape_slot_frozen(tc, idx):
    """Shared shape of one payload slot, in stored (packed) elements."""
    if _v(idx) == 0:
        return [
            _v(tc.SLOT_M),
            _calc_payload_block_k_storage_frozen(tc, 0),
        ]
    return [
        _calc_payload_block_k_storage_frozen(tc, 1),
        _v(tc.SLOT_N),
    ]


@gluon.constexpr_function
def _calc_scale_lds_shape_slot_frozen(tc, idx):
    """E8M0 shape of one operand slot: [slot non-K extent, BLOCK_K/32]."""
    non_k = _v(tc.SLOT_M) if _v(idx) == 0 else _v(tc.SLOT_N)
    return [non_k, _calc_scale_block_k_storage_frozen(tc)]


@gluon.constexpr_function
def _calc_scale_lds_shape_block_frozen(tc, idx):
    """Full scale LDS shape for one block in its staged representation."""
    idx = _v(idx)
    if _scale_shuffled_frozen(tc, idx):
        num_scale_fill_tiles_block = _calc_num_scale_tiles_frozen(tc, idx)
        scale_fill_shape = _calc_scale_flat_shape_frozen(tc, idx)
    else:
        num_scale_fill_tiles_block = _calc_num_lds_slots_per_block_non_k_frozen(
            tc, idx
        )
        scale_fill_shape = _calc_scale_lds_shape_slot_frozen(tc, idx)
    return [
        _num_buffers_frozen(tc, idx, True) * num_scale_fill_tiles_block
    ] + scale_fill_shape


@gluon.constexpr_function
def _calc_copy_contiguity_frozen(tc, idx):
    """Elements per lane for a 128-bit direct-to-LDS payload copy."""
    return 128 // _operand_elem_ty_frozen(tc.func_cfg, idx).primitive_bitwidth


@gluon.constexpr_function
def _calc_payload_lds_copy_ok_frozen(tc, idx):
    """Whether the payload tile admits a 128-bit-per-lane direct copy into LDS."""
    shape = _calc_payload_lds_shape_slot_frozen(tc, idx)
    vec = _calc_copy_contiguity_frozen(tc, idx)
    return shape[0] * shape[1] >= WARP_SIZE * vec * _calc_num_warps_frozen(tc)


@gluon.constexpr_function
def _calc_scale_lds_copy_ok_frozen(tc, idx):
    """Whether an E8M0 scale tile can be written to LDS with a coalesced copy."""
    if _scale_shuffled_frozen(tc, idx):
        return True
    shape = _calc_scale_lds_shape_slot_frozen(tc, idx)
    return shape[0] * shape[1] >= WARP_SIZE * 4 and shape[1] % 4 == 0 and shape[1] >= 8


@gluon.constexpr_function
def _calc_component_via_lds_frozen(tc, component):
    """Final placement of ``(operand, is_scale)``: tuning allows LDS and the copy fits.

    Not gated on whether a scale exists; callers gate absent scales with
    ``has_scale``.
    """
    operand, is_scale = _v(component[0]), bool(_v(component[1]))
    if not _allow_component_via_lds_frozen(tc, (operand, is_scale)):
        return False
    if is_scale:
        return _calc_scale_lds_copy_ok_frozen(tc, operand)
    return _calc_payload_lds_copy_ok_frozen(tc, operand)


@gluon.constexpr_function
def _calc_pipeline_register_period_frozen(tc):
    """LCM of the active direct-register ring periods, in payload K steps.

    Absent scales have no ring. The frozen driver reads it to decide whether a ring
    index may assume a non-negative tile and whether the remainder loop is unrolled.
    """
    period = 1
    for operand in (0, 1):
        for is_scale in (False, True):
            if is_scale and not _has_scale_frozen(tc.func_cfg, operand):
                continue
            if not _calc_component_via_lds_frozen(tc, (operand, is_scale)):
                ratio = _calc_scale_ratio_k_step_frozen(tc, operand) if is_scale else 1
                period = math.lcm(
                    period, _num_buffers_frozen(tc, operand, is_scale) * ratio
                )
    return period


@gluon.constexpr_function
def _calc_k_width_for_frozen(tc, idx):
    """``DotOperandLayout.k_width`` for one operand, in *stored* elements.

    128 bits per lane per access; the tuning field is an optional override.
    """
    override = _v(tc.k_width)
    if override is not None:
        return override
    return 128 // _operand_elem_ty_frozen(tc.func_cfg, idx).primitive_bitwidth


@gluon.constexpr_function
def _calc_dot_result_fragment_layout_frozen(tc):
    """AMDMFMALayout for dot / scaled-dot results in register; ``transposed`` pinned."""
    assert _v(tc.transposed), "transposed=True is pinned, see the module docstring"
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=_v(tc.mfma_instr_shape),
        transposed=True,
        warps_per_cta=_v(tc.warps_per_cta),
        tiles_per_warp=_v(tc.tiles_per_warp),
    )


@gluon.constexpr_function
def _calc_dot_operand_fragment_layout_frozen(tc, idx):
    return gl.DotOperandLayout(
        operand_index=_v(idx),
        parent=_calc_dot_result_fragment_layout_frozen(tc),
        k_width=_calc_k_width_for_frozen(tc, idx),
    )


@gluon.constexpr_function
def _calc_lds_unit_rows_frozen(tc, idx):
    """Non-K extent of one LDS unit: 16 preshuffled, 32 otherwise."""
    return 16 if _operand_preshuffled_frozen(tc, idx) else 32


@gluon.constexpr_function
def _calc_byte_unit_lds_ok_frozen(tc, idx):
    """Does :func:`_byte_unit_lds_layout_frozen` apply to this operand?

    Only the 16x16x128 B operand shape, 8-bit storage, and -- when not preshuffled --
    ``tiles_per_warp >= 2`` on that axis.
    """
    idx = _v(idx)
    if list(_v(tc.mfma_instr_shape)) != [16, 16, 128]:
        return False
    elem_bits = _operand_elem_ty_frozen(tc.func_cfg, idx).primitive_bitwidth
    if elem_bits != 8:
        return False
    if not _operand_preshuffled_frozen(tc, idx) and _v(tc.tiles_per_warp)[idx] < 2:
        return False
    shape = _calc_payload_lds_shape_slot_frozen(tc, idx)
    rows, kelems = shape[idx], shape[1 - idx]
    unit_rows = _calc_lds_unit_rows_frozen(tc, idx)
    unit_k = _LDS_UNIT_K_BYTES * 8 // elem_bits
    return rows % unit_rows == 0 and kelems % unit_k == 0


@gluon.constexpr_function
def _calc_dot_operand_lds_layout_frozen(tc, idx):
    """Padded shared layout for a payload operand."""
    idx = _v(idx)
    shape = _calc_payload_lds_shape_slot_frozen(tc, idx)
    if _calc_byte_unit_lds_ok_frozen(tc, idx):
        return _byte_unit_lds_layout_frozen(
            shape,
            idx,
            _operand_elem_ty_frozen(tc.func_cfg, idx).primitive_bitwidth,
            _calc_lds_unit_rows_frozen(tc, idx),
            _operand_preshuffled_frozen(tc, idx),
        )
    layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        _calc_dot_operand_fragment_layout_frozen(tc, idx),
        shape,
        _operand_elem_ty_frozen(tc.func_cfg, idx),
        True,
    )
    if layout is not None:
        return layout
    # The helper declines whenever the tile holds a single MFMA tile along the
    # non-K axis. Fall back to a plain identity-mapped padded layout whose interval is
    # still >= vec * warpSize, so the copy stays a 128-bit direct-to-LDS.
    vec = _calc_copy_contiguity_frozen(tc, idx)
    order = [1, 0] if idx == 0 else [0, 1]
    return gl.PaddedSharedLayout.with_identity_for(
        [[WARP_SIZE * vec, vec]], shape, order
    )


@gluon.constexpr_function
def _calc_dot_operand_buffer_load_lds_layout_frozen(tc, idx):
    """Register layout of the global->LDS copy offsets for a payload operand."""
    idx = _v(idx)
    shape = _calc_payload_lds_shape_slot_frozen(tc, idx)
    return _bases_to_distributed_frozen(
        _calc_dot_operand_lds_layout_frozen(tc, idx).offset_bases,
        _calc_copy_contiguity_frozen(tc, idx),
        _calc_num_warps_frozen(tc),
        WARP_SIZE,
        shape,
    )


@gluon.constexpr_function
def _calc_payload_hbm_offset_layout_frozen(tc, idx):
    """Layout used to form one payload tile's global-memory offsets."""
    if _calc_component_via_lds_frozen(tc, (idx, False)):
        return _calc_dot_operand_buffer_load_lds_layout_frozen(tc, idx)
    return _calc_dot_operand_fragment_layout_frozen(tc, idx)


@gluon.constexpr_function
def _calc_dot_operand_scale_fragment_layout_frozen(tc, idx):
    idx = _v(idx)
    shape = _calc_scale_lds_shape_slot_frozen(tc, idx)
    return gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout_frozen(tc, idx), shape, MX_GROUP_INT
    )


@gluon.constexpr_function
def _calc_scale_mini_block_nonk_frozen(tc, idx):
    """Rows or columns in one scale load; raw scales retain payload geometry."""
    mini = _calc_scale_nonk_frozen(tc, idx)
    if not _scale_shuffled_frozen(tc, idx):
        return mini
    requested = tc.SCALE_SLOT_M if _v(idx) == 0 else tc.SCALE_SLOT_N
    return _v(requested) or mini


@gluon.constexpr_function
def _calc_scale_mini_block_k_frozen(tc, idx):
    """Scale load K extent in logical payload elements, not E8M0 bytes."""
    if not _scale_shuffled_frozen(tc, idx):
        return _v(tc.BLOCK_K)
    # SCALE_SLOT_K counts scale columns, one per MX group.
    return _v(tc.SCALE_SLOT_K) * MX_GROUP_INT or max(256, _v(tc.BLOCK_K))


@gluon.constexpr_function
def _calc_scale_ratio_k_step_frozen(tc, idx):
    """Number of payload K steps covered by one scale load."""
    bk = _v(tc.BLOCK_K)
    return _calc_scale_mini_block_k_frozen(tc, idx) // bk


@gluon.constexpr_function
def _calc_scale_ratio_non_k_slot_frozen(tc, idx):
    """Number of payload non-K slots covered by one scale load."""
    return _calc_scale_mini_block_nonk_frozen(tc, idx) // _calc_scale_nonk_frozen(
        tc, idx
    )


@gluon.constexpr_function
def _calc_num_scale_tiles_frozen(tc, idx):
    return _calc_num_lds_slots_per_block_non_k_frozen(
        tc, idx
    ) // _calc_scale_ratio_non_k_slot_frozen(tc, idx)


@gluon.constexpr_function
def _calc_scale_hbm_offset_layout_frozen(tc, idx):
    """Layout used to form one scale tile's global-memory offsets."""
    if _calc_component_via_lds_frozen(tc, (idx, True)):
        return _calc_dot_operand_scale_buffer_load_lds_layout_frozen(tc, idx)
    if _scale_shuffled_frozen(tc, idx):
        if _calc_scale_packed_ok_frozen(tc, idx):
            return _calc_packed_scale_load_layout_frozen(tc, idx)
        # get_mfma_scale_layout puts the K-group bit before the non-K bit in
        # reg_bases, so a lane's four bytes land in registers in address order
        # 0, +2, +1, +3. A widened load fills registers in ascending order; issue
        # it with those two bases swapped, then convert with only a lane-local
        # register renumber because the lane and warp bases stay unchanged.
        frag = _calc_scale_load_layout_frozen(tc, idx)
        lead = [[16, 0], [0, 4]]
        rest = [list(b) for b in frag.reg_bases if list(b) not in lead]
        return gl.DistributedLinearLayout(
            reg_bases=lead + rest,
            lane_bases=[list(b) for b in frag.lane_bases],
            warp_bases=[list(b) for b in frag.warp_bases],
            block_bases=[],
            shape=list(frag.shape),
        )
    return _calc_dot_operand_scale_fragment_layout_frozen(tc, idx)


@gluon.constexpr_function
def _calc_scale_nonk_frozen(tc, idx):
    """Non-K extent of one shuffled scale tile: the slot extent."""
    if _v(idx) == 0:
        return _v(tc.SLOT_M)
    return _v(tc.SLOT_N)


@gluon.constexpr_function
def _calc_scale_flat_shape_frozen(tc, idx):
    """LDS staging shape of a shuffled scale tile: one 256 B run per 32-row stripe."""
    return [
        _calc_scale_mini_block_nonk_frozen(tc, idx) // 32,
        _calc_scale_mini_block_k_frozen(tc, idx),
    ]


@gluon.constexpr_function
def _calc_sorted_shuffled_ok_frozen(tc):
    """Does this config produce the layout the C++ A-scale shuffle was written for?"""
    return (
        list(_v(tc.mfma_instr_shape)) == [16, 16, 128]
        and _v(tc.BLOCK_M) % 32 == 0
        and _v(tc.SLOT_M) % 32 == 0
    )


@gluon.constexpr_function
def _calc_shuffled_scale_read_layout_frozen(tc, idx, load=False):
    """The [non-K, K] view of the flat LDS run, in fragment order."""
    nonk = (
        _calc_scale_mini_block_nonk_frozen(tc, idx)
        if _v(load)
        else _calc_scale_nonk_frozen(tc, idx)
    )
    k = (
        _calc_scale_mini_block_k_frozen(tc, idx)
        if _v(load)
        else max(256, _v(tc.BLOCK_K))
    )
    bases = [[16, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
    for bit in _ramp_frozen(8, k // MX_GROUP_INT):
        bases += [[0, bit]]
    stripe = 32
    while stripe < nonk:
        bases = bases + [[stripe, 0]]
        stripe = stripe * 2
    return gl.SharedLinearLayout(offset_bases=bases)


@gluon.constexpr_function
def _scale_dword_delta_frozen(delta, k=256):
    """Where a (non-K, K) step lands, as a coordinate of the i32 tile."""
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
def _calc_scale_packed_sel_frozen(tc, idx, k_phase=0):
    """Byte selectors in MFMA order, for one K128 half or the full K256 word."""
    if not _calc_scale_packed_ok_frozen(tc, idx):
        return None
    if _v(tc.BLOCK_K) == 128:
        return [2 * _v(k_phase), 2 * _v(k_phase) + 1]
    return [0, 2, 1, 3]


@gluon.constexpr_function
def _calc_scale_packed_ok_frozen(tc, idx):
    """Can this operand's scale fragment be fed as one dword per lane?"""
    if not _scale_shuffled_frozen(tc, idx) or _v(tc.BLOCK_K) not in (128, 256):
        return False
    frag = gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout_frozen(tc, idx),
        _calc_scale_fragment_shape_slot_frozen(tc, idx),
        MX_GROUP_INT,
    )
    regs = [list(b) for b in frag.reg_bases]
    if _v(tc.BLOCK_K) == 128:
        return len(regs) >= 1 and regs[0] == [16, 0]
    if len(regs) < 2:
        return False
    return regs[0] == [0, 4] and regs[1] == [16, 0]


@gluon.constexpr_function
def _calc_packed_scale_shape_frozen(tc, idx):
    """Physical i32 storage, including both K128 halves even when one is unused."""
    nonk = _calc_scale_nonk_frozen(tc, idx)
    return [nonk, max(256, _v(tc.BLOCK_K)) // 128]


@gluon.constexpr_function
def _calc_packed_scale_frag_layout_frozen(tc, idx):
    """The i32 fragment with its within-dword register bases folded in."""
    k = max(256, _v(tc.BLOCK_K))
    frag = gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout_frozen(tc, idx),
        [_calc_scale_nonk_frozen(tc, idx), k // MX_GROUP_INT],
        MX_GROUP_INT,
    )
    return gl.DistributedLinearLayout(
        reg_bases=[_scale_dword_delta_frozen(b, k) for b in frag.reg_bases[2:]],
        lane_bases=[_scale_dword_delta_frozen(b, k) for b in frag.lane_bases],
        warp_bases=[_scale_dword_delta_frozen(b, k) for b in frag.warp_bases],
        block_bases=[],
        shape=_calc_packed_scale_shape_frozen(tc, idx),
    )


@gluon.constexpr_function
def _calc_packed_scale_read_layout_frozen(tc, idx, load=False):
    """The i32 view of the flat LDS run: one offset bit per dword-address bit."""
    k = (
        _calc_scale_mini_block_k_frozen(tc, idx)
        if _v(load)
        else max(256, _v(tc.BLOCK_K))
    )
    bases = list(
        _calc_shuffled_scale_read_layout_frozen(tc, idx, load).offset_bases
    )[2:]
    return gl.SharedLinearLayout(
        offset_bases=[_scale_dword_delta_frozen(b, k) for b in bases]
    )


@gluon.constexpr_function
def _calc_scale_load_shape_frozen(tc, idx):
    return [
        _calc_scale_mini_block_nonk_frozen(tc, idx),
        _calc_scale_mini_block_k_frozen(tc, idx) // MX_GROUP_INT,
    ]


@gluon.constexpr_function
def _calc_scale_load_layout_frozen(tc, idx):
    return gl.amd.cdna4.get_mfma_scale_layout(
        _calc_dot_operand_fragment_layout_frozen(tc, idx),
        _calc_scale_load_shape_frozen(tc, idx),
        MX_GROUP_INT,
    )


@gluon.constexpr_function
def _calc_packed_scale_load_shape_frozen(tc, idx):
    return [
        _calc_scale_mini_block_nonk_frozen(tc, idx),
        _calc_scale_mini_block_k_frozen(tc, idx) // 128,
    ]


@gluon.constexpr_function
def _calc_packed_scale_load_layout_frozen(tc, idx):
    frag = _calc_scale_load_layout_frozen(tc, idx)
    k = _calc_scale_mini_block_k_frozen(tc, idx)
    # The two within-dword axes are identified by value, not position.
    within_dword = ([0, 4], [16, 0])
    return gl.DistributedLinearLayout(
        reg_bases=[
            _scale_dword_delta_frozen(b, k)
            for b in frag.reg_bases
            if list(b) not in within_dword
        ],
        lane_bases=[_scale_dword_delta_frozen(b, k) for b in frag.lane_bases],
        warp_bases=[_scale_dword_delta_frozen(b, k) for b in frag.warp_bases],
        block_bases=[],
        shape=_calc_packed_scale_load_shape_frozen(tc, idx),
    )


@gluon.constexpr_function
def _calc_dot_operand_scale_lds_layout_frozen(tc, idx):
    """Identity (K-contiguous) shared tile for the raw E8M0 scales."""
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])


@gluon.constexpr_function
def _calc_dot_operand_scale_buffer_load_lds_layout_frozen(tc, idx):
    """32-bit-per-lane blocked layout for the scale direct-to-LDS write."""
    idx = _v(idx)
    shape = _calc_scale_lds_shape_slot_frozen(tc, idx)
    if _scale_shuffled_frozen(tc, idx):
        # Flat run, 4 contiguous bytes per lane.
        return gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[1, WARP_SIZE],
            warps_per_cta=[_calc_num_warps_frozen(tc), 1],
            order=[1, 0],
        )
    sk = shape[1]
    lanes_k = sk // 4
    return gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[WARP_SIZE // lanes_k, lanes_k],
        warps_per_cta=[_calc_num_warps_frozen(tc), 1],
        order=[1, 0],
    )


@gluon.constexpr_function
def _calc_output_n_slot_frozen(tc):
    """Emitted columns produced by one logical N-slot epilogue tile."""
    return _v(tc.SLOT_N) // _n_reduction_slot_frozen(tc.func_cfg)


@gluon.constexpr_function
def _calc_output_block_n_frozen(tc):
    """Emitted columns produced by one CTA's raw BLOCK_N tile."""
    return _v(tc.BLOCK_N) // _activation_reduction_n_frozen(tc.func_cfg)


@gluon.constexpr_function
def _calc_epilogue_threads_frozen(tc):
    return _calc_num_warps_frozen(tc) * WARP_SIZE


@gluon.constexpr_function
def _calc_epilogue_bias_per_thread_frozen(tc):
    return _v(tc.BLOCK_N) // _calc_epilogue_threads_frozen(tc)


@gluon.constexpr_function
def _calc_epilogue_inputs_via_lds_frozen(tc):
    """Whether bias/gamma vectors admit the supported direct-to-LDS copies."""
    threads = _calc_epilogue_threads_frozen(tc)
    bpt = _calc_epilogue_bias_per_thread_frozen(tc)
    return threads >= _v(tc.BLOCK_M) and _v(tc.BLOCK_N) % threads == 0 and bpt in (1, 4)


@gluon.constexpr_function
def _calc_epilogue_input_lds_layout_frozen(tc):
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])


@gluon.constexpr_function
def _calc_epilogue_gamma_buffer_load_lds_layout_frozen(tc):
    return gl.BlockedLayout([1], [WARP_SIZE], [_calc_num_warps_frozen(tc)], [0])


@gluon.constexpr_function
def _calc_epilogue_bias_buffer_load_lds_layout_frozen(tc):
    return gl.BlockedLayout(
        [_calc_epilogue_bias_per_thread_frozen(tc)],
        [WARP_SIZE],
        [_calc_num_warps_frozen(tc)],
        [0],
    )


@gluon.constexpr_function
def _calc_epilogue_gamma_shape_frozen(tc):
    return [_calc_epilogue_threads_frozen(tc)]


@gluon.constexpr_function
def _calc_epilogue_bias_shape_frozen(tc):
    return [_v(tc.BLOCK_N)]


@gluon.constexpr_function
def _calc_quant_staging_layout_frozen(tc):
    return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])


@gluon.constexpr_function
def _calc_quant_staging_rows_frozen(tc):
    return _v(tc.BLOCK_M) if _gu_split_frozen(tc.func_cfg) else _v(tc.SLOT_M)


@gluon.constexpr_function
def _calc_quant_staging_rotates_frozen(tc):
    """Whether the fused-quant staging alternates two per-tile LDS banks."""
    fc = tc.func_cfg
    return _v(fc.output_quant) is not None and not _gu_split_frozen(fc)


@gluon.constexpr_function
def _calc_lds_bytes_frozen(tc):
    """Explicit LDS of one build. A frozen build stages every present scale in LDS
    (``_FrozenLDSManager.alloc`` allocates it whatever its placement)."""
    fc = tc.func_cfg
    total = 0
    for idx in (0, 1):
        shape = _calc_payload_lds_shape_slot_frozen(tc, idx)
        width = _operand_elem_ty_frozen(fc, idx).primitive_bitwidth // 8
        # This shape is one slot; a block holds one non-K slot per split.
        num_slots_block = _calc_num_lds_slots_per_block_non_k_frozen(tc, idx)
        if _calc_component_via_lds_frozen(tc, (idx, False)):
            total += (
                num_slots_block
                * shape[0]
                * shape[1]
                * width
                * _num_buffers_frozen(tc, idx)
            )
        if _has_scale_frozen(fc, idx) and (
            _v(tc.FROZEN_STEP) or _calc_component_via_lds_frozen(tc, (idx, True))
        ):
            s = _calc_scale_lds_shape_block_frozen(tc, idx)
            total += s[0] * s[1] * s[2]
    return total


@gluon.constexpr_function
def _calc_acc_vgprs_per_lane_frozen(tc):
    return _v(tc.BLOCK_M) * _v(tc.BLOCK_N) // (_calc_num_warps_frozen(tc) * WARP_SIZE)


@gluon.constexpr_function
def _calc_geometry_ok_frozen(tc):
    """Whether the tile lattice is sound enough to build register/shared layouts on.

    A config that fails it still gets its scalar geometry but no layout objects, so
    the descriptive assertion in ``_validate_frozen_layout`` is what reports it rather
    than a layout constructor.
    """
    BM, BN, BK = _v(tc.BLOCK_M), _v(tc.BLOCK_N), _v(tc.BLOCK_K)
    MBM, MBN = _v(tc.SLOT_M), _v(tc.SLOT_N)
    instr = _v(tc.mfma_instr_shape)
    warps = _v(tc.warps_per_cta)
    tiles = _v(tc.tiles_per_warp)
    if BK % instr[2] or BK % MX_GROUP_INT or BM % MBM or BN % MBN:
        return False
    if MBM % (instr[0] * warps[0] * tiles[0]) or MBN % (instr[1] * warps[1] * tiles[1]):
        return False
    for idx in (0, 1):
        if not _scale_shuffled_frozen(tc, idx):
            continue
        block = BM if idx == 0 else BN
        nonk = _calc_scale_mini_block_nonk_frozen(tc, idx)
        sk = _calc_scale_mini_block_k_frozen(tc, idx)
        mini = _calc_scale_nonk_frozen(tc, idx)
        if not block >= nonk >= mini or block % nonk or nonk % mini or nonk % 32:
            return False
        if sk < max(256, BK) or sk & (sk - 1) or sk % BK:
            return False
    return True


@aggregate
@strip_annotate
class _FrozenLayout:
    """The tile shapes, placements and layouts the frozen kernel reads.

    Built once per build by :func:`_compute_frozen_layout`. Per-operand values are
    ``(A, B)`` pairs read through accessors indexed by operand (0 = A / tokens,
    1 = B / experts). ``None`` marks a value that does not exist for this build.
    Accessors with a free argument are evaluated from the stored scalars.
    """

    # -- block and slot geometry --
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
    _payload_lds_shape_slot: gl.constexpr
    _scale_lds_shape_slot: gl.constexpr
    _byte_unit_lds_ok: gl.constexpr
    # -- storage placement --
    _component_via_lds: gl.constexpr
    _pipeline_register_period: gl.constexpr
    # -- payload layouts --
    _dot_result_fragment_layout: gl.constexpr
    _dot_operand_fragment_layout: gl.constexpr
    _dot_operand_lds_layout: gl.constexpr
    _payload_hbm_offset_layout: gl.constexpr
    # -- scale geometry --
    _scale_nonk: gl.constexpr
    _scale_mini_block_nonk: gl.constexpr
    _scale_mini_block_k: gl.constexpr
    _scale_ratio_k_step: gl.constexpr
    _scale_ratio_non_k_slot: gl.constexpr
    _num_scale_tiles: gl.constexpr
    _scale_flat_shape: gl.constexpr
    _sorted_shuffled_ok: gl.constexpr
    _scale_packed_ok: gl.constexpr
    _scale_packed_sel: gl.constexpr
    _packed_scale_shape: gl.constexpr
    # -- scale layouts --
    _dot_operand_scale_fragment_layout: gl.constexpr
    _dot_operand_scale_lds_layout: gl.constexpr
    _scale_hbm_offset_layout: gl.constexpr
    _shuffled_scale_read_layout: gl.constexpr
    _packed_scale_frag_layout: gl.constexpr
    _packed_scale_read_layout: gl.constexpr
    # -- epilogue --
    _output_n_slot: gl.constexpr
    _output_block_n: gl.constexpr
    _epilogue_threads: gl.constexpr
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
    def payload_lds_shape_slot(self, idx):
        """Shared shape of one payload slot, in stored (packed) elements."""
        return list(_v(self._payload_lds_shape_slot)[_v(idx)])

    @gluon.constexpr_function
    def scale_lds_shape_slot(self, idx):
        """E8M0 shape of one operand slot: [slot non-K extent, BLOCK_K/32]."""
        return list(_v(self._scale_lds_shape_slot)[_v(idx)])

    @gluon.constexpr_function
    def byte_unit_lds_ok(self, idx):
        """Does :func:`_byte_unit_lds_layout_frozen` apply to this operand?"""
        return _v(self._byte_unit_lds_ok)[_v(idx)]

    # -- storage placement --------------------------------------------------------
    # Each ``*_in_reg`` is exactly the complement of its ``*_via_lds``. None of these
    # is gated on whether a scale exists; callers gate absent scales with
    # ``_has_scale_frozen``.

    @gluon.constexpr_function
    def component_via_lds(self, component):
        """Whether ``(operand, is_scale)`` is staged through LDS."""
        component = _v(component)
        operand, is_scale = _v(component[0]), bool(_v(component[1]))
        assert operand in (0, 1), "component operand must be 0 (A) or 1 (B)"
        return _v(self._component_via_lds)[operand][int(is_scale)]

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

    @gluon.constexpr_function
    def pipeline_register_period(self):
        """LCM of the direct-register ring periods, in payload K steps."""
        return _v(self._pipeline_register_period)

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
    def scale_packed_ok(self, idx):
        """Can this operand's scale fragment be fed as one dword per lane?"""
        return _v(self._scale_packed_ok)[_v(idx)]

    @gluon.constexpr_function
    def scale_packed_sel(self, idx, k_phase=0):
        """Byte selectors in MFMA order, for one K128 half or the full K256 word."""
        return _v(self._scale_packed_sel)[_v(idx)][_v(k_phase)]

    @gluon.constexpr_function
    def packed_scale_shape(self, idx):
        """Physical i32 storage, including both K128 halves even when one is unused."""
        return list(_v(self._packed_scale_shape)[_v(idx)])

    # -- scale layouts ------------------------------------------------------------

    @gluon.constexpr_function
    def dot_operand_scale_fragment_layout(self, idx):
        return _v(self._dot_operand_scale_fragment_layout)[_v(idx)]

    @gluon.constexpr_function
    def dot_operand_scale_lds_layout(self, idx):
        """Identity (K-contiguous) shared tile for the raw E8M0 scales."""
        return _v(self._dot_operand_scale_lds_layout)[_v(idx)]

    @gluon.constexpr_function
    def scale_hbm_offset_layout(self, idx):
        """Layout used to form one scale tile's global-memory offsets."""
        return _v(self._scale_hbm_offset_layout)[_v(idx)]

    @gluon.constexpr_function
    def shuffled_scale_read_layout(self, idx):
        """The [non-K, K] view of one slot's flat LDS run, in fragment order."""
        return _v(self._shuffled_scale_read_layout)[_v(idx)]

    @gluon.constexpr_function
    def packed_scale_frag_layout(self, idx):
        """The i32 fragment with its within-dword register bases folded in."""
        return _v(self._packed_scale_frag_layout)[_v(idx)]

    @gluon.constexpr_function
    def packed_scale_read_layout(self, idx):
        """The i32 view of one slot's flat LDS run."""
        return _v(self._packed_scale_read_layout)[_v(idx)]

    # -- epilogue and emitted-output layouts --------------------------------------

    @gluon.constexpr_function
    def result_store_layout(self, extent_m, extent_n, elem_bits):
        """Blocked layout for the masked global store of one result slot.

        ``vec`` is capped by the elements each thread actually owns, so a narrow tile
        is not over-covered (which would issue every store twice).
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
        return [_v(rows), output_n // MX_GROUP_INT]

    @gluon.constexpr_function
    def epilogue_threads(self):
        return _v(self._epilogue_threads)

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

        Gate/up split: one lane owns the transposed MFMA quad, ``instr_m * instr_n /
        WARP_SIZE`` consecutive emitted columns. Interleaved: a conservative
        per-lane element count.
        """
        extent_m = _v(self._MINI_BLOCK_M) if _v(extent_m) is None else _v(extent_m)
        output_n = self.output_n_slot() if _v(output_n) is None else _v(output_n)
        if _v(self._gu_split):
            instr = _v(self._mfma_instr_shape)
            return max(1, min(MX_GROUP_INT, (instr[0] * instr[1]) // WARP_SIZE))
        per_lane = (extent_m * output_n) // (self.num_warps() * WARP_SIZE)
        return min(MX_GROUP_INT, max(1, per_lane))

    # -- resources ----------------------------------------------------------------

    @gluon.constexpr_function
    def lds_bytes(self):
        return _v(self._lds_bytes)

    @gluon.constexpr_function
    def acc_vgprs_per_lane(self):
        return _v(self._acc_vgprs_per_lane)


@gluon.constexpr_function
def _compute_frozen_layout(tuning_cfg):
    """Evaluate every layout the frozen kernel reads, from its tuning configuration.

    The kernel body and the host's frozen validation both call this, so they agree on
    one layout. A config whose tile lattice is unsound (see
    :func:`_calc_geometry_ok_frozen`) still gets its scalar geometry, but its layout
    objects are ``None``; ``_validate_frozen_layout`` rejects it.
    """
    tc = tuning_cfg
    fc = tc.func_cfg
    ok = _calc_geometry_ok_frozen(tc)
    ops = (0, 1)

    def pair(fn, *args):
        return tuple(fn(tc, idx, *args) for idx in ops)

    def pair_if(cond, fn, *args):
        return tuple(fn(tc, idx, *args) if ok and cond(idx) else None for idx in ops)

    def always(idx):
        return True

    def has_scale(idx):
        return _has_scale_frozen(fc, idx)

    bools = (False, True)
    via_lds = tuple(
        tuple(
            _calc_component_via_lds_frozen(tc, (idx, is_scale)) for is_scale in bools
        )
        for idx in ops
    )

    def payload_via_lds(idx):
        return via_lds[idx][0]

    def scale_tiles_ok(idx):
        # A shuffled scale mini block narrower than the payload slot covers no whole
        # slot: it has no tile count, and no LDS footprint where it is staged there.
        return _calc_scale_ratio_non_k_slot_frozen(tc, idx) > 0

    def scale_tiles(fn):
        return tuple(fn(tc, idx) if scale_tiles_ok(idx) else None for idx in ops)

    lds_ok = all(
        scale_tiles_ok(idx)
        or not (has_scale(idx) and (_v(tc.FROZEN_STEP) or via_lds[idx][1]))
        for idx in ops
    )
    epilogue_via_lds = _calc_epilogue_inputs_via_lds_frozen(tc)
    return _FrozenLayout(
        BLOCK_K=_v(tc.BLOCK_K),
        MINI_BLOCK_M=_v(tc.SLOT_M),
        mfma_instr_shape=list(_v(tc.mfma_instr_shape)),
        gu_split=_gu_split_frozen(fc),
        num_warps=_calc_num_warps_frozen(tc),
        num_k_slots_per_tile=_calc_num_k_slots_per_tile_frozen(tc),
        num_prefetch_k_slots=_calc_num_prefetch_k_slots_frozen(tc),
        num_m_slots_per_block=_calc_num_m_slots_per_block_frozen(tc),
        num_n_slots_per_block=_calc_num_n_slots_per_block_frozen(tc),
        num_lds_slots_per_block_non_k=pair(
            _calc_num_lds_slots_per_block_non_k_frozen
        ),
        payload_block_k_storage=pair(_calc_payload_block_k_storage_frozen),
        scale_block_k_storage=_calc_scale_block_k_storage_frozen(tc),
        payload_lds_shape_slot=pair(_calc_payload_lds_shape_slot_frozen),
        scale_lds_shape_slot=pair(_calc_scale_lds_shape_slot_frozen),
        byte_unit_lds_ok=pair(_calc_byte_unit_lds_ok_frozen),
        component_via_lds=via_lds,
        pipeline_register_period=_calc_pipeline_register_period_frozen(tc),
        dot_result_fragment_layout=(
            _calc_dot_result_fragment_layout_frozen(tc) if ok else None
        ),
        dot_operand_fragment_layout=pair_if(
            always, _calc_dot_operand_fragment_layout_frozen
        ),
        dot_operand_lds_layout=pair_if(
            payload_via_lds, _calc_dot_operand_lds_layout_frozen
        ),
        payload_hbm_offset_layout=pair_if(
            always, _calc_payload_hbm_offset_layout_frozen
        ),
        scale_nonk=pair(_calc_scale_nonk_frozen),
        scale_mini_block_nonk=pair(_calc_scale_mini_block_nonk_frozen),
        scale_mini_block_k=pair(_calc_scale_mini_block_k_frozen),
        scale_ratio_k_step=pair(_calc_scale_ratio_k_step_frozen),
        scale_ratio_non_k_slot=pair(_calc_scale_ratio_non_k_slot_frozen),
        num_scale_tiles=scale_tiles(_calc_num_scale_tiles_frozen),
        scale_flat_shape=pair(_calc_scale_flat_shape_frozen),
        sorted_shuffled_ok=_calc_sorted_shuffled_ok_frozen(tc),
        scale_packed_ok=(
            pair(_calc_scale_packed_ok_frozen) if ok else (False, False)
        ),
        scale_packed_sel=tuple(
            tuple(
                _calc_scale_packed_sel_frozen(tc, idx, k_phase) if ok else None
                for k_phase in (0, 1)
            )
            for idx in ops
        ),
        packed_scale_shape=pair(_calc_packed_scale_shape_frozen),
        dot_operand_scale_fragment_layout=pair_if(
            has_scale, _calc_dot_operand_scale_fragment_layout_frozen
        ),
        dot_operand_scale_lds_layout=pair(_calc_dot_operand_scale_lds_layout_frozen),
        scale_hbm_offset_layout=pair_if(
            has_scale, _calc_scale_hbm_offset_layout_frozen
        ),
        shuffled_scale_read_layout=tuple(
            (
                _calc_shuffled_scale_read_layout_frozen(tc, idx)
                if ok and has_scale(idx)
                else None
            )
            for idx in ops
        ),
        packed_scale_frag_layout=pair_if(
            has_scale, _calc_packed_scale_frag_layout_frozen
        ),
        packed_scale_read_layout=tuple(
            (
                _calc_packed_scale_read_layout_frozen(tc, idx)
                if ok and has_scale(idx)
                else None
            )
            for idx in ops
        ),
        output_n_slot=_calc_output_n_slot_frozen(tc),
        output_block_n=_calc_output_block_n_frozen(tc),
        epilogue_threads=_calc_epilogue_threads_frozen(tc),
        epilogue_input_lds_layout=_calc_epilogue_input_lds_layout_frozen(tc),
        epilogue_gamma_buffer_load_lds_layout=(
            _calc_epilogue_gamma_buffer_load_lds_layout_frozen(tc)
            if epilogue_via_lds
            else None
        ),
        epilogue_bias_buffer_load_lds_layout=(
            _calc_epilogue_bias_buffer_load_lds_layout_frozen(tc)
            if epilogue_via_lds
            else None
        ),
        epilogue_gamma_shape=_calc_epilogue_gamma_shape_frozen(tc),
        epilogue_bias_shape=_calc_epilogue_bias_shape_frozen(tc),
        epilogue_inputs_via_lds=epilogue_via_lds,
        quant_staging_layout=_calc_quant_staging_layout_frozen(tc),
        quant_staging_rows=_calc_quant_staging_rows_frozen(tc),
        quant_staging_rotates=_calc_quant_staging_rotates_frozen(tc),
        lds_bytes=_calc_lds_bytes_frozen(tc) if lds_ok else None,
        acc_vgprs_per_lane=_calc_acc_vgprs_per_lane_frozen(tc),
    )


# --- vendored offset helpers --------------------------------------------------
# The slot index and the HBM offset math this snapshot addresses its operands with,
# copied from _layout.py. They read only the ``_FrozenLayout`` this module builds.
#
# a1c975cbf showed why this has to be a copy: it reworked the live shuffled-scale
# address expression and, while the helpers were shared, that live change reached
# straight into the frozen kernel.
# ------------------------------------------------------------------------------

@gluon.constexpr_function
def _slot_index_frozen(
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


@gluon.jit
def _gather_rows_frozen(
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
def _n_start_frozen(pid_n, n_slot_idx: gl.constexpr, N, func_cfg, tuning_cfg):
    """Raw-N start of slot ``n_slot_idx`` under interleaved or split gate/up packing.

    Interleaved tiles are contiguous. Split tiles cover the same emitted channels but
    read gate and up from separate halves of N.
    """
    if require_constexpr(_gu_split_frozen(func_cfg)):
        out = pid_n * tuning_cfg.SLOT_N + n_slot_idx * (N // 2)
    else:
        out = pid_n * tuning_cfg.BLOCK_N + n_slot_idx * tuning_cfg.SLOT_N
    return out


@gluon.jit
def _blocked_b_hbm_offsets_frozen(
    layout: gl.constexpr,
    payload_block_k_storage: gl.constexpr,
    n0,
    n_slot: gl.constexpr,
    stored_k,
    run: gl.constexpr,
):
    """Element offsets for a B tile in the 16-column-blocked HBM layout.

    ``aiter/ops/shuffle.py::shuffle_weight(w, (16, 16))`` maps logical ``(n, k)``
    to ``(n//16)*(stored_k*16) + (k//run)*(16*run) + (n%16)*run + k%run``, with
    ``run`` the elements in 16 bytes. Keeping this physical-layout transform beside
    the layout that consumes it prevents the HBM and LDS permutations from drifting
    apart.
    """
    kk = gl.arange(0, payload_block_k_storage, layout=gl.SliceLayout(1, layout))[
        :, None
    ]
    nn = (n0 + gl.arange(0, n_slot, layout=gl.SliceLayout(0, layout)))[None, :]
    return (
        (nn // 16) * (stored_k * 16)
        + (kk // run) * (16 * run)
        + (nn % 16) * run
        + kk % run
    )


@gluon.jit
def _shuffled_scale_stage_offsets_frozen(
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
def _shuffled_scale_register_offsets_frozen(
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
def _a_payload_hbm_offsets_frozen(
    a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg, layout
):
    """Gathered A payload offsets, one copy-layout grid per M slot."""
    M_BLOCK: gl.constexpr = tuning_cfg.BLOCK_M
    M_SLOT: gl.constexpr = tuning_cfg.SLOT_M
    NUM_M_SLOTS_PER_BLOCK: gl.constexpr = layout.num_m_slots_per_block()
    PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = layout.payload_block_k_storage(0)
    layout_a_slot: gl.constexpr = layout.payload_hbm_offset_layout(0)
    a_hbm_offsets_block = ()
    for m_slot_idx in gl.static_range(NUM_M_SLOTS_PER_BLOCK):
        rows_a_slot = _gather_rows_frozen(
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
def _b_payload_hbm_offsets_frozen(b, pid_n, N, K, func_cfg, tuning_cfg, layout):
    """B payload offsets for plain or preshuffled weights, one grid per N slot."""
    N_SLOT: gl.constexpr = tuning_cfg.SLOT_N
    NUM_N_SLOTS_PER_BLOCK: gl.constexpr = layout.num_n_slots_per_block()
    PAYLOAD_BLOCK_K_STORAGE: gl.constexpr = layout.payload_block_k_storage(1)
    layout_b_slot: gl.constexpr = layout.payload_hbm_offset_layout(1)
    K_STORAGE: gl.constexpr = K // _pack_divisor_frozen(func_cfg, 1)
    RUN: gl.constexpr = 128 // _operand_elem_ty_frozen(func_cfg, 1).primitive_bitwidth
    b_hbm_offsets_block = ()
    for n_slot_idx in gl.static_range(NUM_N_SLOTS_PER_BLOCK):
        if require_constexpr(tuning_cfg.B_PRESHUFFLED):
            b_hbm_offsets_block = b_hbm_offsets_block + (
                _blocked_b_hbm_offsets_frozen(
                    layout_b_slot,
                    PAYLOAD_BLOCK_K_STORAGE,
                    _n_start_frozen(pid_n, n_slot_idx, N, func_cfg, tuning_cfg),
                    N_SLOT,
                    K_STORAGE,
                    RUN,
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
                    _n_start_frozen(pid_n, n_slot_idx, N, func_cfg, tuning_cfg)
                    + gl.arange(0, N_SLOT, layout=gl.SliceLayout(0, layout_b_slot))
                )[None, :]
                * b.stride_n,
            )

    return b_hbm_offsets_block


@gluon.jit
def _a_scale_hbm_offsets_frozen(
    a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg, layout
):
    """Byte offsets of the A-scale grids, one per M slot (``None`` if unscaled).

    Two shapes, picked by ``A_SCALE_SORTED_SHUFFLED``: a flat run into the
    moe_sort_scales pre-pass output, or a gathered
    ``[M_SLOT, SCALE_BLOCK_K_STORAGE]`` grid
    over the raw ``(M, K/32)`` tensor. The caller sees a tuple indexed by M slot.
    """
    if require_constexpr(not _has_scale_frozen(func_cfg, 0)):
        return _NO_SCALE

    M_BLOCK: gl.constexpr = tuning_cfg.BLOCK_M
    M_SLOT: gl.constexpr = tuning_cfg.SLOT_M
    NUM_M_SLOTS_PER_BLOCK: gl.constexpr = layout.num_m_slots_per_block()
    SCALE_BLOCK_K_STORAGE: gl.constexpr = layout.scale_block_k_storage()
    layout_a_scale_slot: gl.constexpr = layout.scale_hbm_offset_layout(0)
    SCALE_M: gl.constexpr = layout.scale_mini_block_nonk(0)
    SCALE_K: gl.constexpr = layout.scale_mini_block_k(0)
    SCALE_RATIO: gl.constexpr = layout.scale_ratio_non_k_slot(0)

    if require_constexpr(_scale_shuffled_frozen(tuning_cfg, 0) and layout.scale_in_reg(0)):
        a_scale_hbm_offsets_block = ()
        for m_slot_idx in gl.static_range(NUM_M_SLOTS_PER_BLOCK):
            a_scale_hbm_offsets_block = a_scale_hbm_offsets_block + (
                _shuffled_scale_register_offsets_frozen(
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
                _shuffled_scale_stage_offsets_frozen(
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
            rows_a_scale_slot = _gather_rows_frozen(
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
def _b_scale_hbm_offsets_frozen(b, pid_n, N, K, func_cfg, tuning_cfg, layout):
    """Byte offsets of the B-scale grids, one per N slot (``None`` if unscaled).

    Either a flat CDNA4_SCALE preshuffle run for staging or the raw ``(K/32, N)``
    grid. The caller receives a tuple indexed by N slot.
    """
    if require_constexpr(not _has_scale_frozen(func_cfg, 1)):
        return _NO_SCALE

    N_SLOT: gl.constexpr = tuning_cfg.SLOT_N
    NUM_N_SLOTS_PER_BLOCK: gl.constexpr = layout.num_n_slots_per_block()
    SCALE_BLOCK_K_STORAGE: gl.constexpr = layout.scale_block_k_storage()
    layout_b_scale_slot: gl.constexpr = layout.scale_hbm_offset_layout(1)
    SCALE_N: gl.constexpr = layout.scale_mini_block_nonk(1)
    SCALE_K: gl.constexpr = layout.scale_mini_block_k(1)
    SCALE_RATIO: gl.constexpr = layout.scale_ratio_non_k_slot(1)
    SPLIT_EXTENT: gl.constexpr = (
        N_SLOT if _gu_split_frozen(func_cfg) and SCALE_RATIO > 1 else 0
    )

    if require_constexpr(_scale_shuffled_frozen(tuning_cfg, 1) and layout.scale_in_reg(1)):
        b_scale_hbm_offsets_block = ()
        for n_slot_idx in gl.static_range(NUM_N_SLOTS_PER_BLOCK):
            b_scale_hbm_offsets_block = b_scale_hbm_offsets_block + (
                _shuffled_scale_register_offsets_frozen(
                    layout_b_scale_slot,
                    _n_start_frozen(
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
                _shuffled_scale_stage_offsets_frozen(
                    layout_b_scale_slot,
                    _n_start_frozen(
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
                    _n_start_frozen(pid_n, n_slot_idx, N, func_cfg, tuning_cfg)
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
def _packed_sel(tuning_cfg, layout, idx, k_phase=0):
    """The frozen K256 selector for an LDS-staged packed scale operand."""
    if layout.scale_packed_ok(idx) and layout.scale_via_lds(idx):
        return layout.scale_packed_sel(idx, k_phase)
    return None


@gluon.constexpr_function
def _any_packed(tuning_cfg, layout):
    return (
        _packed_sel(tuning_cfg, layout, 0) is not None
        or _packed_sel(tuning_cfg, layout, 1) is not None
    )


@gluon.jit
def _dot(
    a,
    a_scale,
    b,
    b_scale,
    acc,
    func_cfg,
    tuning_cfg,
    layout,
    K_PHASE: gl.constexpr = 0,
):
    """The frozen snapshot's matrix-instruction dispatch."""
    kind: gl.constexpr = _dot_kind_frozen(func_cfg)
    a_sel: gl.constexpr = _packed_sel(tuning_cfg, layout, 0, K_PHASE)
    b_sel: gl.constexpr = _packed_sel(tuning_cfg, layout, 1, K_PHASE)
    any_packed: gl.constexpr = _any_packed(tuning_cfg, layout)
    if require_constexpr(kind == _DK_MFMA_SCALED and any_packed):
        out = gl.amd.cdna4.mfma_scaled_packed(
            a=a,
            a_scale=a_scale,
            a_scale_sel=a_sel,
            a_format=_mx_format_frozen(func_cfg, 0),
            b=b,
            b_scale=b_scale,
            b_scale_sel=b_sel,
            b_format=_mx_format_frozen(func_cfg, 1),
            acc=acc,
        )
    elif require_constexpr(kind == _DK_MFMA_SCALED):
        out = gl.amd.cdna4.mfma_scaled(
            a=a,
            a_scale=a_scale,
            a_format=_mx_format_frozen(func_cfg, 0),
            b=b,
            b_scale=b_scale,
            b_format=_mx_format_frozen(func_cfg, 1),
            acc=acc,
        )
    elif require_constexpr(kind == _DK_UPCAST_MFMA):
        out = gl.amd.cdna4.mfma(a, b, acc)
    else:
        gl.static_assert(kind == _DK_MFMA)
        out = gl.amd.cdna4.mfma(a, b, acc)
    return out


@gluon.jit
def _opt_at(t, i: gl.constexpr, PRESENT: gl.constexpr):
    if require_constexpr(PRESENT):
        out = t[i]
    else:
        out = _NO_SCALE
    return out


@gluon.jit
def _take_pairs(frags, LO: gl.constexpr, N: gl.constexpr):
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
    layout,
    DO_MFMA: gl.constexpr,
    K_PHASE: gl.constexpr = 0,
):
    if require_constexpr(DO_MFMA):
        for i in gl.static_range(N_MINI):
            if require_constexpr(_has_scale_frozen(func_cfg, 0)):
                a_s = a_frags[2 * i + 1]
            else:
                a_s = _NO_SCALE
            if require_constexpr(_has_scale_frozen(func_cfg, 1)):
                b_s = b_frags[2 * i + 1]
            else:
                b_s = _NO_SCALE
            acc = _dot(
                a_frags[2 * i],
                a_s,
                b_frags[2 * i],
                b_s,
                acc,
                func_cfg,
                tuning_cfg,
                layout,
                (K_PHASE + i * tuning_cfg.BLOCK_K // 128) % 2,
            )
    return acc


@gluon.jit
def _ds_read(lds_ptr, layout: gl.constexpr):
    return lds_ptr.load(layout)


@gluon.jit
def _buffer_load_to_lds(
    lds_ptr,
    hbm_ptr,
    hbm_offs,
    cache: gl.constexpr,
    SOFF: gl.constexpr = 0,
):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        lds_ptr, hbm_ptr, hbm_offs, cache_modifier=cache, soffset=SOFF
    )


@aggregate
@strip_annotate
class _FrozenLDSManager:
    """The LDS shape and accessors used by the frozen snapshot."""

    func_cfg: base_value
    tuning_cfg: base_value
    layout: base_value
    a_payload_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    a_scale_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    b_payload_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    b_scale_lds_ptr: gl.shared_memory_descriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        func_cfg,
        tuning_cfg,
        layout,
        a_payload_lds_ptr,
        a_scale_lds_ptr,
        b_payload_lds_ptr,
        b_scale_lds_ptr,
    ):
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg
        self.layout = layout
        self.a_payload_lds_ptr = _opt(a_payload_lds_ptr)
        self.a_scale_lds_ptr = _opt(a_scale_lds_ptr)
        self.b_payload_lds_ptr = _opt(b_payload_lds_ptr)
        self.b_scale_lds_ptr = _opt(b_scale_lds_ptr)

    @gluon.jit
    def alloc(func_cfg, tuning_cfg, layout):
        NBA: gl.constexpr = _num_buffers_frozen(tuning_cfg, 0)
        NBB: gl.constexpr = _num_buffers_frozen(tuning_cfg, 1)
        NBAS: gl.constexpr = _num_buffers_frozen(tuning_cfg, 0, True)
        NBBS: gl.constexpr = _num_buffers_frozen(tuning_cfg, 1, True)
        NMA: gl.constexpr = layout.num_lds_slots_per_block_non_k(0)
        NMB: gl.constexpr = layout.num_lds_slots_per_block_non_k(1)
        a_ty: gl.constexpr = _operand_elem_ty_frozen(func_cfg, 0)
        b_ty: gl.constexpr = _operand_elem_ty_frozen(func_cfg, 1)
        a_shape: gl.constexpr = layout.payload_lds_shape_slot(0)
        b_shape: gl.constexpr = layout.payload_lds_shape_slot(1)

        if require_constexpr(layout.payload_via_lds(0)):
            a_payload_lds_ptr = gl.allocate_shared_memory(
                a_ty,
                [NBA * NMA, a_shape[0], a_shape[1]],
                layout=layout.dot_operand_lds_layout(0),
            )
        else:
            a_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(layout.payload_via_lds(1)):
            b_payload_lds_ptr = gl.allocate_shared_memory(
                b_ty,
                [NBB * NMB, b_shape[0], b_shape[1]],
                layout=layout.dot_operand_lds_layout(1),
            )
        else:
            b_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(_has_scale_frozen(func_cfg, 0)):
            as_shape: gl.constexpr = layout.scale_lds_shape_slot(0)
            if require_constexpr(_scale_shuffled_frozen(tuning_cfg, 0)):
                a_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBAS * layout.num_scale_tiles(0)] + layout.scale_flat_shape(0),
                    layout=layout.dot_operand_scale_lds_layout(0),
                )
            else:
                a_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBAS * NMA, as_shape[0], as_shape[1]],
                    layout=layout.dot_operand_scale_lds_layout(0),
                )
        else:
            a_scale_lds_ptr: gl.constexpr = None
        if require_constexpr(_has_scale_frozen(func_cfg, 1)):
            bs_shape: gl.constexpr = layout.scale_lds_shape_slot(1)
            if require_constexpr(_scale_shuffled_frozen(tuning_cfg, 1)):
                b_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBBS * NMB] + layout.scale_flat_shape(1),
                    layout=layout.dot_operand_scale_lds_layout(1),
                )
            else:
                b_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBBS * NMB, bs_shape[0], bs_shape[1]],
                    layout=layout.dot_operand_scale_lds_layout(1),
                )
        else:
            b_scale_lds_ptr: gl.constexpr = None

        return _FrozenLDSManager(
            func_cfg,
            tuning_cfg,
            layout,
            a_payload_lds_ptr,
            a_scale_lds_ptr,
            b_payload_lds_ptr,
            b_scale_lds_ptr,
        )

    @gluon.jit
    def buffer_load_payload(
        self,
        operand: gl.constexpr,
        VIA_LDS: gl.constexpr,
        BUFFER_LOAD_IDX,
        tile: gl.constexpr,
        hbm_ptr,
        hbm_offs,
        SOFF: gl.constexpr = 0,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        gl.static_assert(VIA_LDS)
        if require_constexpr(operand == 0):
            lds_ptr = self.a_payload_lds_ptr
        else:
            lds_ptr = self.b_payload_lds_ptr
        cache: gl.constexpr = (
            cfg.token_cache_modifier if operand == 0 else cfg.expert_cache_modifier
        )
        _buffer_load_to_lds(
            lds_ptr.index(
                BUFFER_LOAD_IDX * layout.num_lds_slots_per_block_non_k(operand) + tile
            ),
            hbm_ptr,
            hbm_offs,
            cache,
            SOFF,
        )

    @gluon.jit
    def buffer_load_scale(
        self,
        operand: gl.constexpr,
        VIA_LDS: gl.constexpr,
        BUFFER_LOAD_IDX,
        tile: gl.constexpr,
        hbm_ptr,
        hbm_offs,
        SOFF: gl.constexpr = 0,
        K_PHASE=0,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        gl.static_assert(VIA_LDS)
        ratio: gl.constexpr = layout.scale_ratio_non_k_slot(operand)
        if require_constexpr(
            _has_scale_frozen(self.func_cfg, operand)
            and layout.scale_via_lds(operand)
            and tile % ratio == 0
        ):
            if require_constexpr(operand == 0):
                lds_ptr = self.a_scale_lds_ptr
            else:
                lds_ptr = self.b_scale_lds_ptr
            cache: gl.constexpr = (
                cfg.token_scale_cache_modifier
                if operand == 0
                else cfg.expert_scale_cache_modifier
            )
            _buffer_load_to_lds(
                lds_ptr.index(
                    BUFFER_LOAD_IDX
                    * (layout.num_lds_slots_per_block_non_k(operand) // ratio)
                    + tile // ratio
                ),
                hbm_ptr,
                hbm_offs,
                cache,
                SOFF,
            )

    @gluon.jit
    def commit_buffer_load(self):
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def wait_buffer_load_groups(self, num_group: gl.constexpr):
        gl.amd.cdna4.async_copy.wait_group(num_group)

    @gluon.jit
    def _payload_slice(
        self,
        lds_ptr,
        DS_READ_IDX,
        n_tiles: gl.constexpr,
        tile: gl.constexpr,
        mini_idx: gl.constexpr,
        k_dim: gl.constexpr,
        pack: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(layout.num_k_slots_per_tile() > 1):
            width: gl.constexpr = cfg.BLOCK_K // pack
            tile_lds_ptr = tile_lds_ptr.slice(mini_idx * width, width, dim=k_dim)
        return tile_lds_ptr

    @gluon.jit
    def _scale_slice(
        self,
        lds_ptr,
        DS_READ_IDX,
        n_tiles: gl.constexpr,
        tile: gl.constexpr,
        mini_idx: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(layout.num_k_slots_per_tile() > 1):
            width: gl.constexpr = cfg.BLOCK_K // MX_GROUP
            tile_lds_ptr = tile_lds_ptr.slice(mini_idx * width, width, dim=1)
        return tile_lds_ptr

    @gluon.jit
    def _a_scale_tile(self, DS_READ_IDX, mi: gl.constexpr):
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        RA: gl.constexpr = layout.scale_ratio_non_k_slot(0)
        tile_lds_ptr = self.a_scale_lds_ptr.index(
            DS_READ_IDX * (layout.num_lds_slots_per_block_non_k(0) // RA) + mi // RA
        )
        if require_constexpr(RA > 1):
            stripes: gl.constexpr = cfg.MINI_BLOCK_M // 32
            tile_lds_ptr = tile_lds_ptr.slice((mi % RA) * stripes, stripes, dim=0)
        return tile_lds_ptr

    @gluon.jit
    def ds_read_frag(
        self,
        operand: gl.constexpr,
        DS_READ_IDX,
        tile: gl.constexpr,
        mini_idx: gl.constexpr,
        READ_PAYLOAD: gl.constexpr = True,
        READ_SCALE: gl.constexpr = True,
        SCALE_READ_IDX=None,
    ):
        layout: gl.constexpr = self.layout
        if require_constexpr(operand == 0):
            payload_lds_ptr = self.a_payload_lds_ptr
            scale_lds_ptr = self.a_scale_lds_ptr
        else:
            payload_lds_ptr = self.b_payload_lds_ptr
            scale_lds_ptr = self.b_scale_lds_ptr
        if require_constexpr(SCALE_READ_IDX is None):
            SCALE_READ_IDX = DS_READ_IDX
        if require_constexpr(not READ_PAYLOAD):
            payload: gl.constexpr = None
        else:
            payload = _ds_read(
                self._payload_slice(
                    payload_lds_ptr,
                    DS_READ_IDX,
                    layout.num_lds_slots_per_block_non_k(operand),
                    tile,
                    mini_idx,
                    1 - operand,
                    _pack_divisor_frozen(self.func_cfg, operand),
                ),
                layout.dot_operand_fragment_layout(operand),
            )
        if require_constexpr(READ_SCALE and _has_scale_frozen(self.func_cfg, operand)):
            gl.static_assert(self.layout.scale_via_lds(operand))
            if require_constexpr(_scale_shuffled_frozen(self.tuning_cfg, operand)):
                if require_constexpr(operand == 0):
                    scale_tile_lds_ptr = self._a_scale_tile(SCALE_READ_IDX, tile)
                else:
                    scale_tile_lds_ptr = scale_lds_ptr.index(
                        SCALE_READ_IDX * layout.num_lds_slots_per_block_non_k(operand)
                        + tile
                    )
            if require_constexpr(layout.scale_packed_ok(operand)):
                scale_val = _ds_read(
                    scale_tile_lds_ptr.reinterpret(
                        gl.int32,
                        layout.packed_scale_shape(operand),
                        layout.packed_scale_read_layout(operand),
                    ),
                    layout.packed_scale_frag_layout(operand),
                )
            elif require_constexpr(_scale_shuffled_frozen(self.tuning_cfg, operand)):
                scale_val = _ds_read(
                    scale_tile_lds_ptr.reinterpret(
                        gl.uint8,
                        layout.scale_lds_shape_slot(operand),
                        layout.shuffled_scale_read_layout(operand),
                    ),
                    layout.dot_operand_scale_fragment_layout(operand),
                )
            else:
                scale_val = _ds_read(
                    self._scale_slice(
                        scale_lds_ptr,
                        SCALE_READ_IDX,
                        layout.num_lds_slots_per_block_non_k(operand),
                        tile,
                        mini_idx,
                    ),
                    layout.dot_operand_scale_fragment_layout(operand),
                )
        else:
            scale_val: gl.constexpr = None
        return payload, scale_val


@aggregate
@strip_annotate
class _PipelineConst:
    lds_ptrs: _FrozenLDSManager
    a_hbm_offs: tl.tuple | tuple
    b_hbm_offs: tl.tuple | tuple
    a_scale_hbm_offs: tl.tuple | tuple | gl.constexpr
    b_scale_hbm_offs: tl.tuple | tuple | gl.constexpr
    a_scale_stride_k: gl.constexpr
    b_scale_stride_k: gl.constexpr
    a_step: gl.constexpr
    b_step: gl.constexpr
    s_step: gl.constexpr
    func_cfg: base_value
    tuning_cfg: base_value
    layout: base_value

    @gluon.constexpr_function
    def __init__(
        self,
        lds_ptrs,
        a_hbm_offs,
        b_hbm_offs,
        a_scale_hbm_offs,
        b_scale_hbm_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
        layout,
    ):
        self.lds_ptrs = lds_ptrs
        self.a_hbm_offs = a_hbm_offs
        self.b_hbm_offs = b_hbm_offs
        self.a_scale_hbm_offs = _opt(a_scale_hbm_offs)
        self.b_scale_hbm_offs = _opt(b_scale_hbm_offs)
        self.a_scale_stride_k = gl.constexpr(_v(a_scale_stride_k))
        self.b_scale_stride_k = gl.constexpr(_v(b_scale_stride_k))
        self.a_step = gl.constexpr(_v(a_step))
        self.b_step = gl.constexpr(_v(b_step))
        self.s_step = gl.constexpr(_v(s_step))
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg
        self.layout = layout


@aggregate
@strip_annotate
class _PipelinePointers:
    a_hbm_ptr: gl.tensor
    b_hbm_ptr: gl.tensor
    a_scale_hbm_ptr: gl.tensor | gl.constexpr
    b_scale_hbm_ptr: gl.tensor | gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_hbm_ptr, b_hbm_ptr, a_scale_hbm_ptr, b_scale_hbm_ptr):
        self.a_hbm_ptr = a_hbm_ptr
        self.b_hbm_ptr = b_hbm_ptr
        self.a_scale_hbm_ptr = _opt(a_scale_hbm_ptr)
        self.b_scale_hbm_ptr = _opt(b_scale_hbm_ptr)


@aggregate
@strip_annotate
class _PipelineRegFragments:
    a_payload: tl.tuple | tuple
    a_scale: tl.tuple | tuple
    b_payload: tl.tuple | tuple
    b_scale: tl.tuple | tuple
    acc: tl.tuple | tuple

    @gluon.constexpr_function
    def __init__(self, a_payload, a_scale, b_payload, b_scale, acc):
        self.a_payload = a_payload
        self.a_scale = a_scale
        self.b_payload = b_payload
        self.b_scale = b_scale
        self.acc = acc


@gluon.jit
def _take_reg_pairs(payload, scale, LO: gl.constexpr, N: gl.constexpr):
    out = ()
    for i in gl.static_range(N):
        out = out + (payload[LO + i], scale[LO + i])
    return out


@gluon.jit
def _make_reg_fragments(a_frags, b_frags, acc):
    a_payload, a_scale, b_payload, b_scale = (), (), (), ()
    for i in gl.static_range(len(a_frags) // 2):
        a_payload = a_payload + (a_frags[2 * i],)
        a_scale = a_scale + (a_frags[2 * i + 1],)
    for i in gl.static_range(len(b_frags) // 2):
        b_payload = b_payload + (b_frags[2 * i],)
        b_scale = b_scale + (b_frags[2 * i + 1],)
    return _PipelineRegFragments(a_payload, a_scale, b_payload, b_scale, acc)


@gluon.jit
def _advance_hbm_ptrs(pc, hbm_ptrs, STEPS: gl.constexpr = 1, K_PHASE: gl.constexpr = 0):
    a_hbm_ptr = hbm_ptrs.a_hbm_ptr + STEPS * pc.a_step
    b_hbm_ptr = hbm_ptrs.b_hbm_ptr + STEPS * pc.b_step
    a_scale_hbm_ptr = hbm_ptrs.a_scale_hbm_ptr
    b_scale_hbm_ptr = hbm_ptrs.b_scale_hbm_ptr
    if require_constexpr(_has_scale_frozen(pc.func_cfg, 0) and pc.layout.scale_via_lds(0)):
        a_scale_hbm_ptr = (
            a_scale_hbm_ptr
            + pc.layout.scale_hbm_steps(0, STEPS, K_PHASE)
            * pc.s_step
            * pc.a_scale_stride_k
        )
    if require_constexpr(_has_scale_frozen(pc.func_cfg, 1) and pc.layout.scale_via_lds(1)):
        b_scale_hbm_ptr = (
            b_scale_hbm_ptr
            + pc.layout.scale_hbm_steps(1, STEPS, K_PHASE)
            * pc.s_step
            * pc.b_scale_stride_k
        )
    return _PipelinePointers(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    )


@gluon.jit
def _frozen_prologue_fence():
    """Preserve the tuned snapshot's cooperative prologue fence and schedule."""
    gl.amd.cdna4.sched_barrier(0)
    gl.barrier()
    gl.amd.cdna4.sched_barrier(0)


@gluon.jit
def _prologue_frozen(pc, hbm_ptrs):
    """Fill NB-1 buffers, then retain the snapshot's whole-buffer wait and fence."""
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    NB: gl.constexpr = tc.NUM_LDS_BUFFER
    NM: gl.constexpr = layout.num_m_slots_per_block()
    NN: gl.constexpr = layout.num_n_slots_per_block()
    for i in gl.static_range(NB - 1):
        for ni in gl.static_range(NN):
            for mi in gl.static_range(NM):
                _buffer_load_frozen(
                    pc,
                    i,
                    mi,
                    ni,
                    hbm_ptrs.a_hbm_ptr,
                    hbm_ptrs.b_hbm_ptr,
                    hbm_ptrs.a_scale_hbm_ptr,
                    hbm_ptrs.b_scale_hbm_ptr,
                )
                if require_constexpr(mi == NM - 1 and ni == NN - 1):
                    hbm_ptrs = _advance_hbm_ptrs(
                        pc, hbm_ptrs, K_PHASE=layout.scale_k_phase(i)
                    )
    pc.lds_ptrs.wait_buffer_load_groups((NB - 2) * (NM + NN))
    _frozen_prologue_fence()
    return hbm_ptrs


# --- frozen placement / wait helpers ------------------------------------------
# Twins of the constexpr helpers _pipeline_step_frozen reaches, with SCALE_FILL_MID
# pinned to 1 (its BEST value). They decide which slot owns which copy and how many
# groups a wait covers, so a stray flag here reshapes the frozen schedule even though
# the step itself reads no env var. Both axes are split in any config that now
# compiles (validate() insists), which is what the frozen EVEN test amounted to.
# ------------------------------------------------------------------------------
@gluon.constexpr_function
def _buffer_load_group_pos_frozen(is_a, i, NM, NN):
    """Position of one mini-block fill's commit group inside its stage."""
    is_a, i = _v(is_a), _v(i)
    order = _buffer_load_order_frozen(NM, NN)
    return order.index((1 if is_a else 0, i))


@gluon.constexpr_function
def _buffer_loads_before_frozen(mi, ni, NM, NN, ANY):
    """Groups this stage has already committed when slot ``(mi, ni)`` is reached.

    Counted against the N-outer walk of :func:`_slot_index`. Legacy issues A(m) at
    ``(m, 0)`` and B(n) at ``(0, n)``, so once the walk has left column 0 every A is
    behind it, and the B of the current column is behind it as soon as ``mi > 0``.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if not _v(ANY):
        return 0
    return min(_slot_index_frozen(mi, ni, NM, NN), NM + NN)


@gluon.constexpr_function
def _ds_read_a_tile_frozen(mi, ni, NM, NN):
    """Operand-A mini block slot ``(mi, ni)`` reads out of LDS, or None.

    Even: the same one-per-slot assignment the fills use, so a slot's fill and its read
    name the *same* position in :func:`_buffer_load_order_frozen`. Its wait then works out to
    ``G - 1 - s + STAGES_BETWEEN * G + s`` -- the ``s`` cancels and every slot waits on
    the same constant, which is exactly the uniform ``wait_group`` the reference kernel
    uses. Legacy: A(mi) at ``ni == 0``, which bunches both reads onto slot (0, 0).
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _buffer_load_tile_frozen(_buffer_load_pos_frozen(mi, ni, NM, NN), NM, NN, True)


@gluon.constexpr_function
def _ds_read_b_tile_frozen(mi, ni, NM, NN):
    """Operand-B mini block slot ``(mi, ni)`` reads out of LDS, or None."""
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _buffer_load_tile_frozen(_buffer_load_pos_frozen(mi, ni, NM, NN), NM, NN, False)


@gluon.constexpr_function
def _scale_buffer_load_slot_frozen(is_a, tile, NM, NN):
    """Slot index that issues the scale copy for A(tile) / B(tile).

    Default: the same slot as the payload, so a tile's scale and payload share one
    commit group. Under ``1`` at NM=NN=2 both A(0)/B(0) scales go to
    slot 1 and both A(1)/B(1) scales to slot 2.
    """
    is_a, tile = bool(_v(is_a)), _v(tile)
    NM, NN = _v(NM), _v(NN)
    if 1 and NM == 2 and NN == 2:
        return 1 + tile
    return _buffer_load_group_pos_frozen(is_a, tile, NM, NN)


@gluon.constexpr_function
def _buffer_load_pos_frozen(mi, ni, NM, NN):
    """Which fill (position in :func:`_buffer_load_order_frozen`) slot ``(mi, ni)`` issues, or None.

    Even: one fill per slot, in flat slot order -- exactly the tutorial's layout, where
    each of the four ``mfma``/``mem`` region pairs moves one tile and commits one group.
    Legacy: A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``, which loads slot (0, 0)
    with two fills and leaves every slot off the first row and column with none.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    s = _slot_index_frozen(mi, ni, NM, NN)
    return s if s < NM + NN else None


@gluon.constexpr_function
def _scale_buffer_load_tile_frozen(mi, ni, NM, NN, want_a):
    """Tile whose A (resp. B) scale copy slot ``(mi, ni)`` issues, or None.

    Still one slot per mini block even when several share a scale tile: the commit
    group has to be emitted either way, because _buffer_load_wait counts G = NM + NN groups
    per stage. buffer_load_scale drops the redundant *copy* and leaves the group empty.
    """
    s = _slot_index_frozen(mi, ni, NM, NN)
    n = NM if _v(want_a) else NN
    for t in range(_v(n)):
        if _scale_buffer_load_slot_frozen(_v(want_a), t, NM, NN) == s:
            return t
    return None


@gluon.constexpr_function
def _buffer_load_wait_frozen(mi, ni, NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD):
    """Outstanding-group count that retires everything slot ``(mi, ni)`` is about to read.

    A group is retired once ``wait_group(n)`` leaves at most ``n`` behind it. Counting
    forward from the target group: the rest of its own stage, then ``STAGES_BETWEEN``
    whole stages, then whatever the current stage has committed so far. The slot reads
    A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``; when it reads both, the later
    group's (smaller) count wins. ``None`` means the slot reads nothing and needs no wait.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    G = NM + NN
    base = _v(STAGES_BETWEEN) * G + _buffer_loads_before_frozen(
        mi, ni, NM, NN, ANY_BUFFER_LOAD
    )
    out = None
    ta = _ds_read_a_tile_frozen(mi, ni, NM, NN)
    if ta is not None:
        # The payload and the scale of the same tile can sit in different commit
        # groups (see _scale_buffer_load_slot); the later of the two is what has to retire.
        p = max(
            _buffer_load_group_pos_frozen(True, ta, NM, NN),
            _scale_buffer_load_slot_frozen(True, ta, NM, NN),
        )
        out = G - 1 - p + base
    tb = _ds_read_b_tile_frozen(mi, ni, NM, NN)
    if tb is not None:
        p = max(
            _buffer_load_group_pos_frozen(False, tb, NM, NN),
            _scale_buffer_load_slot_frozen(False, tb, NM, NN),
        )
        w = G - 1 - p + base
        out = w if out is None else min(out, w)
    return out


@gluon.constexpr_function
def _stage_buffer_load_wait_frozen(NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD):
    """Strongest (smallest) wait_group count over all slots of a stage."""
    NM, NN = _v(NM), _v(NN)
    ws = [
        _buffer_load_wait_frozen(mi, ni, NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD)
        for ni in range(NN)
        for mi in range(NM)
    ]
    ws = [w for w in ws if w is not None]
    if not ws:
        return None
    return max(min(ws), 0)


@gluon.jit
def _stage_buffer_load_wait_group_frozen(
    lds_ptrs,
    mi: gl.constexpr,
    ni: gl.constexpr,
    NM: gl.constexpr,
    NN: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    ANY_BUFFER_LOAD: gl.constexpr,
    WAIT_SLACK: gl.constexpr = 0,
):
    """One wait_group per stage, emitted at its first slot, instead of one per slot."""
    WAIT: gl.constexpr = _stage_buffer_load_wait_frozen(
        NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD
    )
    if require_constexpr(_slot_index_frozen(mi, ni, NM, NN) == 0 and WAIT is not None):
        lds_ptrs.wait_buffer_load_groups(WAIT + WAIT_SLACK)


@gluon.jit
def _buffer_load_frozen(
    pc,
    BUFFER_LOAD_IDX,
    mi: gl.constexpr,
    ni: gl.constexpr,
    a_hbm_ptr,
    b_hbm_ptr,
    a_scale_hbm_ptr,
    b_scale_hbm_ptr,
    ADVANCE: gl.constexpr = False,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
):
    """The global->LDS copies slot ``(mi, ni)`` owns, each its own commit group.

    FROZEN SNAPSHOT -- do not refactor; ``_pipeline._fill_slot`` is the live one.

    Specialised on the flags the ~625 us kernel ran with, so nothing here reads
    os.environ: the legacy per-fill groups (the former ONE_MARK 0 default) and
    SOFF_UNROLL 0 (pointers bump every step, so KI and every *_SOFF are 0).
    EVEN_FILL was 1 and is gone -- both axes are split in any
    config that compiles, so the schedule it selected is the only one.
    ``KI`` is kept in the signature only so the two stay call-compatible.

    ``ADVANCE`` walks the returned HBM pointers on to the next ``BLOCK_K`` stage. It is
    done here, at the last slot, rather than after the slot loop on purpose: the warp
    pipeliner only tolerates a ``wait_group`` as the *first* op after a stage border, and
    a stage-tail ``tt.addptr`` sitting between the last ``mem`` border and the next
    slot's wait is exactly what breaks that.
    """
    func_cfg: gl.constexpr = pc.func_cfg
    NM: gl.constexpr = pc.layout.num_m_slots_per_block()
    NN: gl.constexpr = pc.layout.num_n_slots_per_block()
    # Under the even schedule the slot owns at most one fill, named by its position in
    # _buffer_load_order_frozen; under the legacy one the (ni == 0) / (mi == 0) predicates below pick.
    POS: gl.constexpr = _buffer_load_pos_frozen(mi, ni, NM, NN)
    A_TILE: gl.constexpr = _buffer_load_tile_frozen(POS, NM, NN, True)
    B_TILE: gl.constexpr = _buffer_load_tile_frozen(POS, NM, NN, False)
    # Scale copies may be placed on a different slot than their payload.
    A_SC: gl.constexpr = _scale_buffer_load_tile_frozen(mi, ni, NM, NN, True)
    B_SC: gl.constexpr = _scale_buffer_load_tile_frozen(mi, ni, NM, NN, False)
    # Byte displacement of this step from the body's base pointer. The *_step values are
    # in elements (they are added to a typed pointer), soffset is in bytes, so each is
    # scaled by its operand's storage width. All constexpr, so these fold into a literal
    # and the SGPR holding them is hoisted out of the loop.
    # SOFF_UNROLL = 0: every copy addresses its own base, no soffset.
    if require_constexpr(A_TILE is not None):
        pc.lds_ptrs.buffer_load_payload(
            0, True, BUFFER_LOAD_IDX, A_TILE, a_hbm_ptr, pc.a_hbm_offs[A_TILE], 0
        )
        # Payload and its own scale share one commit group; a scale placed elsewhere
        # gets its own below.
        if require_constexpr(
            A_SC == A_TILE and _has_scale_frozen(func_cfg, 0) and pc.layout.scale_via_lds(0)
        ):
            pc.lds_ptrs.buffer_load_scale(
                0,
                True,
                BUFFER_LOAD_IDX,
                A_TILE,
                a_scale_hbm_ptr,
                _opt_at(pc.a_scale_hbm_offs, A_TILE, _has_scale_frozen(func_cfg, 0)),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(A_SC is not None and A_SC != A_TILE):
        if require_constexpr(_has_scale_frozen(func_cfg, 0) and pc.layout.scale_via_lds(0)):
            pc.lds_ptrs.buffer_load_scale(
                0,
                True,
                BUFFER_LOAD_IDX,
                A_SC,
                a_scale_hbm_ptr,
                _opt_at(pc.a_scale_hbm_offs, A_SC, _has_scale_frozen(func_cfg, 0)),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_TILE is not None):
        pc.lds_ptrs.buffer_load_payload(
            1, True, BUFFER_LOAD_IDX, B_TILE, b_hbm_ptr, pc.b_hbm_offs[B_TILE], 0
        )
        if require_constexpr(
            B_SC == B_TILE and _has_scale_frozen(func_cfg, 1) and pc.layout.scale_via_lds(1)
        ):
            pc.lds_ptrs.buffer_load_scale(
                1,
                True,
                BUFFER_LOAD_IDX,
                B_TILE,
                b_scale_hbm_ptr,
                _opt_at(pc.b_scale_hbm_offs, B_TILE, _has_scale_frozen(func_cfg, 1)),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_SC is not None and B_SC != B_TILE):
        if require_constexpr(_has_scale_frozen(func_cfg, 1) and pc.layout.scale_via_lds(1)):
            pc.lds_ptrs.buffer_load_scale(
                1,
                True,
                BUFFER_LOAD_IDX,
                B_SC,
                b_scale_hbm_ptr,
                _opt_at(pc.b_scale_hbm_offs, B_SC, _has_scale_frozen(func_cfg, 1)),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ADVANCE):
        # One bump per step: the frozen kernel did not fold steps into soffset.
        a_hbm_ptr = a_hbm_ptr + KU * pc.a_step
        b_hbm_ptr = b_hbm_ptr + KU * pc.b_step
        if require_constexpr(_has_scale_frozen(func_cfg, 0) and pc.layout.scale_via_lds(0)):
            a_scale_hbm_ptr = a_scale_hbm_ptr + KU * pc.s_step * pc.a_scale_stride_k
        if require_constexpr(_has_scale_frozen(func_cfg, 1) and pc.layout.scale_via_lds(1)):
            b_scale_hbm_ptr = b_scale_hbm_ptr + KU * pc.s_step * pc.b_scale_stride_k
    return a_hbm_ptr, b_hbm_ptr, a_scale_hbm_ptr, b_scale_hbm_ptr


@gluon.jit
def _mini_scale_hbm_offset_frozen(base, step, HAS_SCALE: gl.constexpr):
    """Advance a direct-register scale offset, or pass the None sentinel through."""
    if require_constexpr(HAS_SCALE):
        out = base + step
    else:
        out = base
    return out


@gluon.jit
def _load_scale_register_frozen(
    pc,
    operand: gl.constexpr,
    scale_hbm_ptr,
    scale_hbm_offs,
):
    """Load a frozen-step scale directly from HBM into its MFMA fragment."""
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    gl.static_assert(_has_scale_frozen(pc.func_cfg, operand))
    gl.static_assert(layout.scale_in_reg(operand))
    cache: gl.constexpr = (
        tc.token_scale_cache_modifier
        if operand == 0
        else tc.expert_scale_cache_modifier
    )
    if require_constexpr(_scale_shuffled_frozen(tc, operand)):
        # One dword per lane instead of four ubytes: issue at the address-ordered
        # permutation so the widened load fills registers correctly, then renumber
        # into the layout the MFMA wants.
        scale = gl.convert_layout(
            gl.amd.cdna4.buffer_load(
                ptr=scale_hbm_ptr,
                offsets=scale_hbm_offs,
                cache=cache,
                contiguity=4,
            ),
            layout.dot_operand_scale_fragment_layout(operand),
        )
    else:
        scale = gl.amd.cdna4.buffer_load(
            ptr=scale_hbm_ptr,
            offsets=scale_hbm_offs,
            cache=cache,
        )
    return scale


@gluon.jit
def _ds_read_operand_frozen(
    pc,
    DS_READ_IDX,
    tile: gl.constexpr,
    scale_hbm_ptr,
    operand: gl.constexpr,
    READ_PAYLOAD: gl.constexpr = True,
    READ_SCALE: gl.constexpr = True,
):
    """Read one frozen-step operand tile and any direct-register scale fragments."""
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    NUM_MINI: gl.constexpr = layout.num_k_slots_per_tile()
    SK_MINI: gl.constexpr = tc.BLOCK_K // MX_GROUP
    HAS: gl.constexpr = _has_scale_frozen(pc.func_cfg, operand)
    SCALE_VIA_LDS: gl.constexpr = HAS and layout.scale_via_lds(operand)
    if require_constexpr(operand == 0):
        scale_hbm_offs = pc.a_scale_hbm_offs
    else:
        scale_hbm_offs = pc.b_scale_hbm_offs
    scale_tile_hbm_offs = _opt_at(scale_hbm_offs, tile, HAS)
    frags = ()
    for i in gl.static_range(NUM_MINI):
        payload, scale = pc.lds_ptrs.ds_read_frag(
            operand,
            DS_READ_IDX,
            tile,
            i,
            READ_PAYLOAD,
            READ_SCALE and SCALE_VIA_LDS,
        )
        if require_constexpr(READ_SCALE and HAS and not SCALE_VIA_LDS):
            scale = _load_scale_register_frozen(
                pc,
                operand,
                scale_hbm_ptr,
                _mini_scale_hbm_offset_frozen(scale_tile_hbm_offs, i * SK_MINI, HAS),
            )
        if require_constexpr(not READ_PAYLOAD):
            payload = scale
        if require_constexpr(HAS and READ_SCALE):
            slot = scale
        else:
            slot = payload
        frags = frags + (payload, slot)
    return frags


@gluon.jit
def _pipeline_step_frozen(
    pc,
    hbm_ptrs,
    regs,
    BUFFER_LOAD_IDX,
    DS_READ_IDX,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_DS_READ: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
    DO_MFMA: gl.constexpr = True,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    WAIT_SLACK: gl.constexpr = 0,
):
    """One ``BLOCK_K`` stage, walked across the M/N slot grid.

    FROZEN SNAPSHOT of the 2026-09-02 best kernel -- do not refactor this copy.
    Verbatim ``_pipeline_step_impl`` with every env-var constexpr replaced by the
    value it holds in the tuned config, so nothing here reads ``os.environ`` and no
    stray flag can perturb the reference schedule. Measured at 4 waves with
    WARP_PIPELINE=1, MANUAL_PP=1, the LDS-staged epilogue that is now the only
    epilogue path, and llc flags -amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd
    -amdgpu-no-sched-revert -misched-pin-critical-res=HWXDL: 630.3 us cold (g4q)
    and 622.3 us saturating (bf16).

    Select with AITER_TRITON_MOE_GLUON_FROZEN_STEP=1 to A/B a refactor of
    ``_pipeline._step_live`` against the known-good schedule. Immediately after a
    re-snapshot the two must compile to *identical* assembly -- that, not perf, is
    the check that the copy is faithful.

    Re-snapshot procedure: copy the live step here with the flag values listed
    below folded in, then diff the ``.amdgcn`` of FROZEN_STEP=0 against
    FROZEN_STEP=1 and require zero differing instruction lines.


    A mini-M block's copy and ``ds_read`` are issued at its first slot (``ni == 0``) and
    reused by every later ``ni``; a mini-N block's at ``mi == 0``. So the copies and the
    LDS reads are interleaved with the MFMAs instead of standing in one block ahead of
    them, and each is still a whole mini tile at its own coalesced, fully vectorised copy
    layout.

    Every fill is its own commit group, so each slot waits for exactly the mini block it
    is about to read rather than for the whole stage -- see :func:`_buffer_load_wait`, whose
    ``STAGES_BETWEEN`` is ``NUM_LDS_BUFFER - 2`` in the steady state (one buffer is being
    filled by ``buffer_load ... lds`` while one is being consumed by ``ds_read``).

    Along K the ``ds_read`` always pulls a whole stage, mini-K steps ``[0, NUM_MINI)``.
    The MFMAs run one window earlier, over ``[-PF_MINI, NUM_MINI-PF_MINI)``: the negative
    part is the payload/scale fragments in ``regs``, carried from the previous step,
    and the tail this step reads is carried to the next. ``PF_MINI == NUM_MINI`` makes
    the MFMAs consume nothing they read themselves; ``PF_MINI == 0`` makes them consume
    only what they read.

    ``PF_MINI == NUM_MINI`` is also what makes the ping-pong legal: the slot's MFMAs then
    depend on nothing the slot reads, so they are emitted *first* and the reads and copies
    behind them are pure fill for the shadow. Under ``WARP_PIPELINE`` the two halves are
    additionally handed to ``TritonAMDGPUWarpPipeline`` as an ``mfma``/``mem`` stage pair,
    which is what turns the interleave into an inter-wave ping-pong. The ``wait_group``
    stays outside both regions -- the pass rejects a wait inside one.
    """

    # --- the flag set this snapshot was specialised on ------------------------------
    # Every branch on these is inlined below, so there is nothing left to read from
    # os.environ and no way for a stray flag to perturb this copy:
    #   DS_IN_MFMA 1  DS_MOVE 0  _FILLS_FIRST 1 (the env flag, not the derived
    #   FILLS_FIRST constexpr -- that was not tc.ALLOW_B_VIA_LDS, inlined False below)
    #   FILL_IN_MFMA 0  LOAD_MFMA_READ 0
    #   MANUAL_PP 1   MEM_PRIO 1  MPP_ALL_FENCED 0  MPP_BARRIER_STRIDE 4
    #   MPP_CLOSE_BARRIER 0  MPP_NO_FENCE 0  MPP_NO_STAGE_BARRIER 0  MPP_PIN_MFMA 0
    #   MPP_PIN_WAIT 1  MPP_SCHED_AFTER_ONLY 0  NO_SYNC_STAGE 1  SCHED_MODE 0
    #   SLOT_SCHED_BARRIER 0  SOFF_UNROLL 0  STAGE_WAIT 1  WARP_PIPELINE 1
    # Six came from BEST (DS_IN_MFMA MANUAL_PP MPP_BARRIER_STRIDE MPP_CLOSE_BARRIER
    # NO_SYNC_STAGE STAGE_WAIT); the rest are each flag's code default.
    # Preserve the legacy ONE_MARK 0 + STAGE_WAIT 1 schedule independently of the
    # live path's commit/wait configuration.
    # What remains conditional depends only on this function's parameters: DO_MFMA,
    # DO_DS_READ, DO_BUFFER_LOAD, the mi/ni slot indices, and pc's func/tuning cfg.
    # --------------------------------------------------------------------------------
    func_cfg: gl.constexpr = pc.func_cfg
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    NM: gl.constexpr = layout.num_m_slots_per_block()
    NN: gl.constexpr = layout.num_n_slots_per_block()
    NUM_MINI: gl.constexpr = layout.num_k_slots_per_tile()
    PF_MINI: gl.constexpr = layout.num_prefetch_k_slots()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_HAS: gl.constexpr = _has_scale_frozen(func_cfg, 0)
    B_HAS: gl.constexpr = _has_scale_frozen(func_cfg, 1)
    # ``DO_MFMA=False`` is the prologue's first step: it fills and reads, but has no
    # carried fragments to dot yet. That is only equivalent to the hand-rolled prologue
    # when the stage carries whole -- with HEAD_MINI > 0 the head of what this step reads
    # would have to be dotted here and is not carried, so skipping the MFMAs would
    # silently drop it. Assert rather than branch: the default VGPR_PREFETCH_K == BLOCK_K
    # gives HEAD_MINI == 0, and a config that lowers it must not reach this quietly.
    gl.static_assert(
        DO_MFMA or HEAD_MINI == 0,
        "DO_MFMA=False needs VGPR_PREFETCH_K == BLOCK_K (HEAD_MINI == 0): with a split "
        "stage the head this step reads is dotted here and never carried",
    )
    # mfma-before-mem is only legal when the slot's MFMAs read nothing the slot loads.
    # The frozen kernel ran VGPR_PREFETCH_K == BLOCK_K, so PING_PONG was always True and
    # every branch on it below is inlined. Assert rather than branch: a config that
    # lowers VGPR_PREFETCH_K must fail here, not silently get a schedule this copy was
    # never measured with.
    gl.static_assert(
        HEAD_MINI == 0,
        "_pipeline_step_frozen is the VGPR_PREFETCH_K == BLOCK_K schedule; "
        "split-stage prefetch is unsupported",
    )
    # The frozen kernel always stages B through LDS.
    # PIPE (the warp-pipeline *pass*) is dead here: it needs `not _MANUAL_PP` and the
    # frozen flags pin _MANUAL_PP = 1. Both `if PIPE` branches below are gone with it.
    # MPP -- the hand-emitted rendezvous -- reduces to DO_DS_READ once _WP,
    # _MANUAL_PP and PING_PONG are all 1.
    MPP: gl.constexpr = DO_DS_READ

    # _SOFF_UNROLL alone decides whether the body's steps are folded into soffset.
    # With it off these collapse to (0, 1) -- bump the pointers on every step, as
    # before -- so callers can pass the real position unconditionally.
    # _SOFF_UNROLL = 0: the pointers advance every step, KI/KU unused.
    KIE: gl.constexpr = 0
    KUE: gl.constexpr = 1

    a_hbm_ptr = hbm_ptrs.a_hbm_ptr
    b_hbm_ptr = hbm_ptrs.b_hbm_ptr
    a_scale_hbm_ptr = hbm_ptrs.a_scale_hbm_ptr
    b_scale_hbm_ptr = hbm_ptrs.b_scale_hbm_ptr

    # Fragments this step reads, appended mini block by mini block. `a_cur` grows once
    # per mi (at ni == 0) and `b_cur` once per ni (at mi == 0), so block `mi` always
    # starts at pair `mi * NUM_MINI` and is already there by the time any slot needs it.
    a_cur = ()
    b_cur = ()
    a_tail = ()
    b_tail = ()
    acc = ()
    # _SCHED_MODE = 0, so no iglp_opt hint: it and the ping-pong are alternative
    # answers to the same question, and the rendezvous below is the one measured.
    # N outer, M inner -- see _slot_index. The slot walk, the accumulator tuple's
    # layout and the fill positions are all that one order.
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            # _MPP_PIN_WAIT = 1: the leading sched_barrier of the rendezvous sits
            # here, before the wait, rather than next to the barrier itself.
            # _MPP_BARRIER_STRIDE = 4, inlined as the % 4 below.
            if require_constexpr(MPP and _slot_index_frozen(mi, ni, NM, NN) % 4 == 0):
                gl.amd.cdna4.sched_barrier(0)
            # _STAGE_WAIT = 1, so the per-slot wait form never applied.
            if require_constexpr(DO_DS_READ):
                _stage_buffer_load_wait_group_frozen(
                    pc.lds_ptrs,
                    mi,
                    ni,
                    NM,
                    NN,
                    STAGES_BETWEEN,
                    DO_BUFFER_LOAD,
                    WAIT_SLACK,
                )

            if require_constexpr(DO_MFMA):
                dot_a = _take_reg_pairs(
                    regs.a_payload, regs.a_scale, mi * PF_MINI, PF_MINI
                )
                dot_b = _take_reg_pairs(
                    regs.b_payload, regs.b_scale, ni * PF_MINI, PF_MINI
                )
            else:
                # Nothing carried in yet; the register fragment tuples are empty.
                dot_a = ()
                dot_b = ()

            if require_constexpr(MPP):
                # Rendezvous entering the mfma region. Two per slot -- one here and
                # one before the fills -- so the count per stage is even and a
                # cond_barrier phase shift holds its relationship instead of flipping
                # every stage.
                #
                # The sched_barrier pair is load-bearing, not decoration: a bare
                # s_barrier carries no memory semantics, so without it LLVM freely
                # hoists a buffer_load...lds above the rendezvous that is supposed to
                # separate it from other waves' reads of that slot -- a WAR race that
                # shows up as nondeterministic wrong results. This is the same
                # bracketing emitClusterBarrier uses.
                if require_constexpr(_slot_index_frozen(mi, ni, NM, NN) % 4 == 0):
                    # _MPP_PIN_WAIT = 1 put the leading sched_barrier before the
                    # wait instead, so only the trailing one is emitted here.
                    # _MPP_NO_FENCE = 0 and _MPP_ALL_FENCED = 0: the stage head is
                    # the one fenced barrier, the rest are bare.
                    if require_constexpr(_slot_index_frozen(mi, ni, NM, NN) == 0):
                        gl.barrier()
                    else:
                        gl.amd.cdna4.bare_barrier()
                    gl.amd.cdna4.sched_barrier(0)
                # Only in the K loop, never in the drain (DO_BUFFER_LOAD is False
                # there). The pair is asymmetric as configured: this setprio(1) has no
                # matching setprio(0), because that one sits behind _MPP_CLOSE_BARRIER,
                # which the tuned config leaves at 0. Every wave raised its priority and
                # none lowered it, so there was no relative priority for the ping-pong
                # to exploit -- and the 8 in the drain additionally split the drain into
                # four scheduling regions of 44/17/21/22 instructions, since s_setprio
                # has unmodelled side effects and MachineScheduler treats it as a region
                # boundary. Dropping them merges those into one region of 157 holding 84
                # MFMA and 51 activation instructions together. Perf is a wash (-1.4 us,
                # CI [-6.4, +3.7]); the merge is the reason to keep it.
                if require_constexpr(DO_BUFFER_LOAD):
                    gl.amd.cdna4.setprio(1)
            # MFMA first: it consumes only registers carried from the previous stage,
            # so everything below it is shadow work. _DS_IN_MFMA = 1 puts this slot's
            # ds_reads in the same region, so they issue under the MFMA cluster's shadow
            # rather than next to the buffer_loads.
            slot_acc = _maybe_block_dot(
                dot_a,
                dot_b,
                regs.acc[_slot_index_frozen(mi, ni, NM, NN)],
                PF_MINI,
                func_cfg,
                tc,
                layout,
                DO_MFMA,
            )
            if require_constexpr(MPP):
                if require_constexpr(
                    _ds_read_a_tile_frozen(mi, ni, NM, NN) is not None
                ):
                    a_cur = a_cur + _ds_read_operand_frozen(
                        pc,
                        DS_READ_IDX,
                        _ds_read_a_tile_frozen(mi, ni, NM, NN),
                        a_scale_hbm_ptr,
                        0,
                    )
                if require_constexpr(
                    _ds_read_b_tile_frozen(mi, ni, NM, NN) is not None
                ):
                    b_cur = b_cur + _ds_read_operand_frozen(
                        pc,
                        DS_READ_IDX,
                        _ds_read_b_tile_frozen(mi, ni, NM, NN),
                        b_scale_hbm_ptr,
                        1,
                    )
            acc = acc + (slot_acc,)

            if require_constexpr(DO_BUFFER_LOAD):
                (
                    a_hbm_ptr,
                    b_hbm_ptr,
                    a_scale_hbm_ptr,
                    b_scale_hbm_ptr,
                ) = _buffer_load_frozen(
                    pc,
                    BUFFER_LOAD_IDX,
                    mi,
                    ni,
                    a_hbm_ptr,
                    b_hbm_ptr,
                    a_scale_hbm_ptr,
                    b_scale_hbm_ptr,
                    ADVANCE=(mi == NM - 1) and (ni == NN - 1) and (KIE == KUE - 1),
                    KI=KIE,
                    KU=KUE,
                )
            # Indexed by the tile this slot actually read, not by (mi, ni): under the
            # even schedule slot (0, 1) reads B(0), not B(1), so the legacy `ni == 0` /
            # `mi == 0` predicates would reach past the end of the half-built tuple.
            if require_constexpr(
                DO_DS_READ and _ds_read_a_tile_frozen(mi, ni, NM, NN) is not None
            ):
                a_tail = a_tail + _take_pairs(
                    a_cur,
                    _ds_read_a_tile_frozen(mi, ni, NM, NN) * NUM_MINI + HEAD_MINI,
                    PF_MINI,
                )
            if require_constexpr(
                DO_DS_READ and _ds_read_b_tile_frozen(mi, ni, NM, NN) is not None
            ):
                b_tail = b_tail + _take_pairs(
                    b_cur,
                    _ds_read_b_tile_frozen(mi, ni, NM, NN) * NUM_MINI + HEAD_MINI,
                    PF_MINI,
                )

            # _SLOT_SCHED_BARRIER = 0: no end-of-slot scheduler fence.

    if require_constexpr(DO_DS_READ):
        if require_constexpr(A_HAS and layout.scale_in_reg(0)):
            a_scale_hbm_ptr = a_scale_hbm_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(B_HAS and layout.scale_in_reg(1)):
            b_scale_hbm_ptr = b_scale_hbm_ptr + pc.s_step * pc.b_scale_stride_k
    else:
        a_tail = _take_reg_pairs(regs.a_payload, regs.a_scale, 0, NM * PF_MINI)
        b_tail = _take_reg_pairs(regs.b_payload, regs.b_scale, 0, NN * PF_MINI)

    return _PipelinePointers(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    ), _make_reg_fragments(a_tail, b_tail, acc)


@gluon.constexpr_function
def _pipeline_peeled_frozen(tc):
    """The frozen schedule always peels its seed and first visible MFMA."""
    return 1


@gluon.jit
def _index_frozen(
    tc,
    layout,
    step,
    kind: gl.constexpr,
    FILL: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
):
    if require_constexpr(kind % 2 and not _has_scale_frozen(tc.func_cfg, kind // 2)):
        out = 0
    else:
        depth: gl.constexpr = _num_buffers_frozen(tc, kind // 2, kind % 2 != 0)
        advance: gl.constexpr = depth - 1 if FILL else 0
        if require_constexpr(
            IN_LOOP and pipeline_unroll_frozen(tc, layout) % depth == 0
        ):
            out = (_pipeline_peeled_frozen(tc) + KI + 1 + advance) % depth
        else:
            tile = step + advance
            if require_constexpr(layout.pipeline_register_period() == 1):
                gl.assume(tile >= 0)
            out = tile % depth
    return out


@gluon.jit
def _init_buffers_frozen(pc):
    """Mirror the pre-extraction driver's inert register queues exactly."""
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    a, b, a_scale, b_scale = (), (), (), ()
    if require_constexpr(layout.payload_in_reg(0)):
        for _ in gl.static_range(
            _num_buffers_frozen(tc, 0)
            * layout.num_m_slots_per_block()
            * layout.num_k_slots_per_tile()
        ):
            a += (
                gl.zeros(
                    [
                        tc.SLOT_M,
                        tc.BLOCK_K // _pack_divisor_frozen(pc.func_cfg, 0),
                    ],
                    _operand_elem_ty_frozen(pc.func_cfg, 0),
                    layout.dot_operand_fragment_layout(0),
                ),
            )
    if require_constexpr(layout.payload_in_reg(1)):
        for _ in gl.static_range(
            _num_buffers_frozen(tc, 1)
            * layout.num_n_slots_per_block()
            * layout.num_k_slots_per_tile()
        ):
            b += (
                gl.zeros(
                    [tc.BLOCK_K // _pack_divisor_frozen(pc.func_cfg, 1), tc.SLOT_N],
                    _operand_elem_ty_frozen(pc.func_cfg, 1),
                    layout.dot_operand_fragment_layout(1),
                ),
            )
    if require_constexpr(_has_scale_frozen(pc.func_cfg, 0) and layout.scale_in_reg(0)):
        for _ in gl.static_range(
            _num_buffers_frozen(tc, 0, True)
            * layout.num_m_slots_per_block()
            * layout.num_k_slots_per_tile()
        ):
            if require_constexpr(layout.scale_ratio_k_step(0) > 1):
                a_scale += (
                    gl.zeros(
                        layout.packed_scale_shape(0),
                        gl.int32,
                        layout.packed_scale_frag_layout(0),
                    ),
                )
            else:
                a_scale += (
                    gl.zeros(
                        [tc.SLOT_M, tc.BLOCK_K // 32],
                        gl.uint8,
                        layout.dot_operand_scale_fragment_layout(0),
                    ),
                )
    if require_constexpr(_has_scale_frozen(pc.func_cfg, 1) and layout.scale_in_reg(1)):
        for _ in gl.static_range(
            _num_buffers_frozen(tc, 1, True)
            * layout.num_n_slots_per_block()
            * layout.num_k_slots_per_tile()
        ):
            if require_constexpr(layout.scale_ratio_k_step(1) > 1):
                b_scale += (
                    gl.zeros(
                        layout.packed_scale_shape(1),
                        gl.int32,
                        layout.packed_scale_frag_layout(1),
                    ),
                )
            else:
                b_scale += (
                    gl.zeros(
                        [tc.SLOT_N, tc.BLOCK_K // 32],
                        gl.uint8,
                        layout.dot_operand_scale_fragment_layout(1),
                    ),
                )
    return a, b, a_scale, b_scale


@gluon.jit
def _step_frozen(
    pc,
    ptrs,
    buffers,
    regs,
    step,
    STAGE: gl.constexpr,
    DRAIN: gl.constexpr = False,
    KI: gl.constexpr = 0,
    IN_LOOP: gl.constexpr = False,
    DOT: gl.constexpr = True,
    EPILOGUE_GROUPS: gl.constexpr = 0,
    STATIC_PHASE: gl.constexpr = False,
):
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    ptrs, regs = _pipeline_step_frozen(
        pc,
        ptrs,
        regs,
        _index_frozen(tc, layout, step, 0, True, KI, IN_LOOP or STATIC_PHASE),
        _index_frozen(tc, layout, step, 0, False, KI, IN_LOOP or STATIC_PHASE),
        pipeline_depth_frozen(tc, layout) - 2 - (STAGE if DRAIN else 0),
        not DRAIN,
        True,
        IN_LOOP,
        DOT,
        0,
        1,
        EPILOGUE_GROUPS,
    )
    return ptrs, buffers, regs


@gluon.jit
def _run_frozen_pipeline(pc, ptrs, NUM_K):
    """Run the frozen prologue and runtime K loop, leaving the drain exposed."""
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    gl.static_assert(
        layout.num_prefetch_k_slots() == layout.num_k_slots_per_tile(),
        "the frozen pipeline requires VGPR_PREFETCH_K == BLOCK_K",
    )
    depth: gl.constexpr = pipeline_depth_frozen(tc, layout)
    unroll: gl.constexpr = pipeline_unroll_frozen(tc, layout)
    peeled: gl.constexpr = _pipeline_peeled_frozen(tc)
    main = NUM_K - depth
    gl.assume(main >= peeled + unroll)
    remaining = main - peeled
    unroll_end = remaining // unroll * unroll
    buffers = _init_buffers_frozen(pc)
    ptrs = _prologue_frozen(pc, ptrs)
    acc = ()
    for ni in gl.static_range(layout.num_n_slots_per_block()):
        for mi in gl.static_range(layout.num_m_slots_per_block()):
            acc += (
                gl.zeros(
                    [tc.SLOT_M, tc.SLOT_N],
                    pc.func_cfg.mma_acc_dtype,
                    layout.dot_result_fragment_layout(),
                ),
            )
    regs = _PipelineRegFragments((), (), (), (), acc)
    ptrs, buffers, regs = _step_frozen(pc, ptrs, buffers, regs, 0, 0, DOT=False)
    for x in gl.static_range(peeled):
        ptrs, buffers, regs = _step_frozen(pc, ptrs, buffers, regs, x + 1, x + 1)
    for base in tl.range(0, unroll_end, unroll):
        for u in gl.static_range(unroll):
            ptrs, buffers, regs = _step_frozen(
                pc,
                ptrs,
                buffers,
                regs,
                peeled + base + u + 1,
                None,
                KI=u,
                IN_LOOP=True,
            )
    if require_constexpr(unroll > 6 and layout.pipeline_register_period() > 1):
        for u in tl.range(0, remaining - unroll_end):
            ptrs, buffers, regs = _step_frozen(
                pc,
                ptrs,
                buffers,
                regs,
                peeled + unroll_end + u + 1,
                None,
            )
    else:
        for u in gl.static_range(unroll - 1):
            if unroll_end + u < remaining:
                ptrs, buffers, regs = _step_frozen(
                    pc,
                    ptrs,
                    buffers,
                    regs,
                    peeled + unroll_end + u + 1,
                    None,
                    KI=u,
                    STATIC_PHASE=True,
                )
    return ptrs, buffers, regs


@gluon.jit
def _drain_frozen_pipeline(
    pc, ptrs, buffers, regs, NUM_K, EPILOGUE_GROUPS: gl.constexpr
):
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    main = NUM_K - pipeline_depth_frozen(tc, layout)
    period: gl.constexpr = math.lcm(
        _num_buffers_frozen(tc, 0),
        _num_buffers_frozen(tc, 1),
        _num_buffers_frozen(tc, 0, True) if _has_scale_frozen(pc.func_cfg, 0) else 1,
        _num_buffers_frozen(tc, 1, True) if _has_scale_frozen(pc.func_cfg, 1) else 1,
        layout.scale_ratio_k_step(0),
        layout.scale_ratio_k_step(1),
    )
    if require_constexpr(
        period <= 3
        and pipeline_unroll_frozen(tc, layout) % period == 0
        and layout.pipeline_register_period() == 1
        and pc.func_cfg.output_quant is None
    ):
        phase = main % period
        for p in gl.static_range(period):
            if phase == p:
                for j in gl.static_range(pipeline_depth_frozen(tc, layout) - 1):
                    ptrs, buffers, regs = _step_frozen(
                        pc,
                        ptrs,
                        buffers,
                        regs,
                        main + j + 1,
                        j,
                        DRAIN=True,
                        EPILOGUE_GROUPS=EPILOGUE_GROUPS,
                        KI=p + j - _pipeline_peeled_frozen(tc),
                        STATIC_PHASE=True,
                    )
    else:
        for j in gl.static_range(pipeline_depth_frozen(tc, layout) - 1):
            ptrs, buffers, regs = _step_frozen(
                pc,
                ptrs,
                buffers,
                regs,
                main + j + 1,
                j,
                DRAIN=True,
                EPILOGUE_GROUPS=EPILOGUE_GROUPS,
            )
    return regs


@gluon.jit
def _last_mfma_frozen(pc, regs):
    tc: gl.constexpr = pc.tuning_cfg
    layout: gl.constexpr = pc.layout
    acc = ()
    for ni in gl.static_range(layout.num_n_slots_per_block()):
        for mi in gl.static_range(layout.num_m_slots_per_block()):
            acc += (
                _maybe_block_dot(
                    _take_reg_pairs(
                        regs.a_payload,
                        regs.a_scale,
                        mi * layout.num_k_slots_per_tile(),
                        layout.num_k_slots_per_tile(),
                    ),
                    _take_reg_pairs(
                        regs.b_payload,
                        regs.b_scale,
                        ni * layout.num_k_slots_per_tile(),
                        layout.num_k_slots_per_tile(),
                    ),
                    regs.acc[
                        _slot_index_frozen(
                            mi,
                            ni,
                            layout.num_m_slots_per_block(),
                            layout.num_n_slots_per_block(),
                        )
                    ],
                    layout.num_k_slots_per_tile(),
                    pc.func_cfg,
                    tc,
                    layout,
                    True,
                    0,
                ),
            )
    return acc


# --- vendored SwiGLU ----------------------------------------------------------
# Copies of the ``_triton_kernels/moe/activations.py`` helpers the frozen epilogue
# applies, kept as ``triton.jit`` exactly as the shared module defines them.
# ------------------------------------------------------------------------------
@triton.jit
def _clip_frozen(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu_gate_frozen(gelu, alpha, limit, FAST_RCP: tl.constexpr = False):
    """``silu(gelu)`` alone -- the half that owns every transcendental."""
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = _clip_frozen(gelu, limit, clip_lower=False)
    denom = 1 + tl.exp2(-1.44269504089 * alpha * gelu)
    if FAST_RCP:
        # Hardware v_rcp_f32 instead of the IEEE divide; ~1 ulp, well inside the bf16
        # the result is stored as. tl.fdiv(ieee_rounding=False) does NOT do this on the
        # AMD backend, so the libdevice call is the only route.
        return hip_libdevice.fast_dividef(gelu, denom)
    return gelu / denom


@triton.jit
def _swiglu_combine_frozen(s, linear, limit, ADD_RESIDUAL: tl.constexpr):
    """``s * linear``, with the linear side's clamp. The other half of the activation."""
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = _clip_frozen(linear, limit, clip_lower=True)
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    return s * linear


@triton.jit
def _swiglu_pair_frozen(
    gelu, linear, alpha, limit, ADD_RESIDUAL: tl.constexpr, FAST_RCP: tl.constexpr = False
):
    """The activation itself, on gate/linear that are already separated."""
    return _swiglu_combine_frozen(
        _swiglu_gate_frozen(gelu, alpha, limit, FAST_RCP), linear, limit, ADD_RESIDUAL
    )


@triton.jit
def _swiglu_frozen(
    input,
    alpha,
    limit,
    ADD_RESIDUAL: tl.constexpr,
    FAST_RCP: tl.constexpr = False,
    GROUP: tl.constexpr = 1,
):
    """SwiGLU on an N-interleaved tile: ``s = silu(gelu)``, then ``s * linear``
    (``s * (linear + 1)`` with ADD_RESIDUAL). ``GROUP == 2`` reads (g,g,l,l) pairs,
    anything else (g,l,g,l).
    """
    if GROUP == 2:
        # Index arithmetic for the 4-D view (i, h, e) -> 4i + 2h + e:
        #   split on e  -> a = 4i+2h,      b = 4i+2h+1
        #   split each on h -> gate = 4i, 4i+1   linear = 4i+2, 4i+3
        n4: tl.constexpr = input.shape[1] // 4
        a, b = tl.split(tl.reshape(input, (input.shape[0], n4, 2, 2)))
        g0, l0 = tl.split(a)
        g1, l1 = tl.split(b)
        gelu = tl.reshape(tl.join(g0, g1), (input.shape[0], input.shape[1] // 2))
        linear = tl.reshape(tl.join(l0, l1), (input.shape[0], input.shape[1] // 2))
    else:
        gelu, linear = tl.split(
            tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2))
        )
    return _swiglu_pair_frozen(gelu, linear, alpha, limit, ADD_RESIDUAL, FAST_RCP)


# --- vendored MXFP4 quant -----------------------------------------------------
# Copy of ``_quant.mxfp4_quant_gluon``, the fused-output quantisation.
# ------------------------------------------------------------------------------
@gluon.jit
def _mxfp4_quant_gluon_frozen(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    LANE_ELEMS: gl.constexpr = 0,
):
    """fp32 ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` -> (packed E2M1 ``[M, N/2]``, E8M0 ``[M, N/32]``).

    The E8M0 scale is computed on the exponent field (``E - 2`` of the rounded-up amax);
    the E2M1 pack is the CDNA4 ``v_cvt_scalef32_pk_fp4_f32`` instruction. See
    ``_quant.py`` for the derivation and the two load-bearing details of the asm.
    """

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)

    if LANE_ELEMS > 1 and MXFP4_QUANT_BLOCK_SIZE % LANE_ELEMS == 0:
        # Everything -- amax, scale and pack -- in ONE reshape lineage: the inner
        # reduction in fp32 (free |v| modifiers), the outer one in integer (no NaN
        # canonicalisation), and the E8M0 scale handed to the pack instruction.
        OUTER: gl.constexpr = MXFP4_QUANT_BLOCK_SIZE // LANE_ELEMS
        x5 = x.reshape(
            BLOCK_SIZE_M, NUM_QUANT_BLOCKS, OUTER, LANE_ELEMS // 2, 2
        )
        a5 = tl.abs(x5)
        part = tl.max(a5, axis=-1, keep_dims=True)
        part = tl.max(part, axis=3, keep_dims=True)
        amax = tl.max(
            part.to(tl.int32, bitcast=True), axis=2, keep_dims=True
        )
        e_biased = ((amax + 0x200000) >> 23) & 0xFF
        e_biased = tl.maximum(e_biased, 2)
        bs_e8m0 = (e_biased - 2).to(tl.uint8)
        # The *forward* scale 2**(E-129) = bitcast(bs_e8m0 << 23), broadcast to the
        # full tile so that splitting it yields the same SliceLayout as evens/odds.
        hw_scale = ((e_biased - 2) << 23).to(tl.float32, bitcast=True)
        hw5 = gl.full(
            x5.shape, 1.0, tl.float32, layout=x5.type.layout
        ) * hw_scale
        evens, odds = tl.split(x5)
        scale_e, _scale_o = tl.split(hw5)
        old_vdst = gl.full(evens.shape, 0, tl.int32, layout=evens.type.layout)
        packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp4_f32 $0, $2, $3, $4",
            "=v,0,v,v,v",
            [old_vdst, evens, odds, scale_e],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        return (
            (packed & 0xFF).to(tl.uint8).reshape(
                BLOCK_SIZE_M, BLOCK_SIZE_N // 2
            ),
            bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS),
        )
    else:
        amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        e_biased = ((amax + 0x200000) >> 23) & 0xFF
        e_biased = tl.maximum(e_biased, 2)
        bs_e8m0 = (e_biased - 2).to(tl.uint8)
        quant_scale = ((256 - e_biased) << 23).to(tl.float32, bitcast=True)
        qx = x * quant_scale

        # Hardware pack. `old_vdst` must be a tied input ("=v,0,v,v"): the instruction
        # writes only the selected byte and preserves the other three.
        xr = qx.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2)
        evens, odds = tl.split(xr)
        old_vdst = gl.full(evens.shape, 0, tl.int32, layout=evens.type.layout)
        packed = tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp4_f32 $0, $2, $3, 1.0",
            "=v,0,v,v",
            [old_vdst, evens, odds],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        return (
            (packed & 0xFF).to(tl.uint8).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2),
            bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS),
        )


# --- vendored epilogue --------------------------------------------------------
# Copy of the ``_epilogue.py`` path the frozen body runs: bias/gamma staging before
# the drain, then bias -> activation -> gammas -> output quant -> store per mini tile.
# ------------------------------------------------------------------------------
_DQ_MXFP4: gl.constexpr = gl.constexpr(int(DtypeQuant.MXFP4))


@gluon.jit
def _n_split_offs_frozen(pid_n, i, N, func_cfg, tuning_cfg):
    """Map a whole raw-N CTA index tensor through the gate/up packing."""
    n_slot: gl.constexpr = tuning_cfg.SLOT_N
    if require_constexpr(_gu_split_frozen(func_cfg)):
        out = pid_n * n_slot + (i // n_slot) * (N // 2) + i % n_slot
    else:
        out = pid_n * tuning_cfg.BLOCK_N + i
    return out


@aggregate
@strip_annotate
class _FrozenEpilogueInputs:
    """Optional epilogue vectors and the async groups pending during the drain."""

    bias_hbm_base: gl.tensor | gl.constexpr
    gammas_hbm_ptr: gl.tensor | gl.constexpr
    gamma_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    bias_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    groups: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self, bias_hbm_base, gammas_hbm_ptr, gamma_lds_ptr, bias_lds_ptr, groups
    ):
        self.bias_hbm_base = _opt(bias_hbm_base)
        self.gammas_hbm_ptr = _opt(gammas_hbm_ptr)
        self.gamma_lds_ptr = _opt(gamma_lds_ptr)
        self.bias_lds_ptr = _opt(bias_lds_ptr)
        self.groups = gl.constexpr(_v(groups))


@gluon.jit
def _stage_epilogue_inputs_frozen(
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
    # with and are done well before the epilogue's wait_group(0).
    #
    # Both tiles are flat: `.index()` drops the leading axis but keeps the layout's rank,
    # so a [rows, mini] tile cannot be viewed as one mini block. `.slice(start, length)`
    # preserves rank, so a 1-D tile slices cleanly into per-mini-block views.
    #
    # The gammas tile is EPI_T elements, not BLOCK_M: buffer_load_to_shared requires
    # exactly 32 or 128 bits per thread, so the copy has to cover one element per thread
    # of the CTA; the surplus is masked off and never read. Bias is BLOCK_N, which is
    # already a whole number of elements per thread.
    EPI_T: gl.constexpr = layout.epilogue_threads()
    EPI_LDS: gl.constexpr = layout.epilogue_inputs_via_lds()
    if require_constexpr(EPI_LDS and func_cfg.has_gammas):
        gamma_lds_ptr = gl.allocate_shared_memory(
            gl.float32,
            layout.epilogue_gamma_shape(),
            layout=layout.epilogue_input_lds_layout(),
        )
    else:
        gamma_lds_ptr: gl.constexpr = None
    if require_constexpr(EPI_LDS and func_cfg.has_bias):
        bias_lds_ptr = gl.allocate_shared_memory(
            gl.float32,
            layout.epilogue_bias_shape(),
            layout=layout.epilogue_input_lds_layout(),
        )
    else:
        bias_lds_ptr: gl.constexpr = None

    # Committed as one group, which the frozen drain covers with its WAIT_SLACK.
    EPI_GROUPS: gl.constexpr = (
        1 if EPI_LDS and (func_cfg.has_gammas or func_cfg.has_bias) else 0
    )
    if require_constexpr(EPI_LDS):
        if require_constexpr(func_cfg.has_gammas):
            g_offs = BM * block_id + gl.arange(
                0,
                EPI_T,
                layout=layout.epilogue_gamma_buffer_load_lds_layout(),
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                gamma_lds_ptr, gammas_hbm_ptr, g_offs, mask=g_offs < M_e, other=0.0
            )
        if require_constexpr(func_cfg.has_bias):
            epi_b_offs = _n_split_offs_frozen(
                pid_n,
                gl.arange(
                    0,
                    BN,
                    layout=layout.epilogue_bias_buffer_load_lds_layout(),
                ),
                N,
                func_cfg,
                tuning_cfg,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bias_lds_ptr, bias_hbm_base, epi_b_offs
            )
        if require_constexpr(func_cfg.has_gammas or func_cfg.has_bias):
            # EPI_LDS is only a layout-feasibility flag, so it can be set with
            # neither vector present. Nothing was issued then, and an empty group
            # would still occupy a FIFO position.
            gl.amd.cdna4.async_copy.commit_group()
    return _FrozenEpilogueInputs(
        bias_hbm_base, gammas_hbm_ptr, gamma_lds_ptr, bias_lds_ptr, EPI_GROUPS
    )


@gluon.jit
def _epi_bias_tiles_frozen(
    bias_lds_ptr, bias_hbm_ptr, pid_n, N, func_cfg, tuning_cfg, layout
):
    """Block-level bias, one tensor per mini-N block, from LDS when staged there."""
    MBN: gl.constexpr = tuning_cfg.SLOT_N
    NN: gl.constexpr = layout.num_n_slots_per_block()
    RFL: gl.constexpr = layout.dot_result_fragment_layout()
    out = ()
    if require_constexpr(func_cfg.has_bias):
        for hn in gl.static_range(NN):
            if require_constexpr(bias_lds_ptr is not None):
                out = out + (
                    bias_lds_ptr.slice(hn * MBN, MBN).load(gl.SliceLayout(0, RFL)),
                )
            else:
                out = out + (
                    gl.load(
                        bias_hbm_ptr
                        + _n_start_frozen(pid_n, hn, N, func_cfg, tuning_cfg)
                        + gl.arange(0, MBN)
                    ),
                )
    return out


@gluon.jit
def _epi_gamma_tiles_frozen(
    gamma_lds_ptr, gammas_hbm_ptr, block_id, M_e, func_cfg, tuning_cfg, layout
):
    """Block-level gammas, one tensor per mini-M block -- global path only.

    When gammas live in LDS this returns empty: the read has to happen inside
    _epilogue_one_tile_frozen off the post-swiglu ``out.type.layout``.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    MBM: gl.constexpr = tuning_cfg.SLOT_M
    NM: gl.constexpr = layout.num_m_slots_per_block()
    out = ()
    if require_constexpr(func_cfg.has_gammas and gamma_lds_ptr is None):
        for hm in gl.static_range(NM):
            hoffs = BM * block_id + hm * MBM + gl.arange(0, MBM)
            out = out + (gl.load(gammas_hbm_ptr + hoffs, mask=hoffs < M_e, other=0.0),)
    return out


@gluon.jit
def _epi_stage_flush_frozen(
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
    """Read one staged MXFP4 mini tile back out of LDS and store it."""
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    BN: gl.constexpr = tuning_cfg.BLOCK_N
    MBM: gl.constexpr = tuning_cfg.SLOT_M
    MBN: gl.constexpr = tuning_cfg.SLOT_N
    ARN: gl.constexpr = _activation_reduction_n_frozen(func_cfg)
    OUT_MBN: gl.constexpr = layout.output_n_slot()
    P_COLS: gl.constexpr = layout.quant_payload_shape(MBM, OUT_MBN)[1]
    S_COLS: gl.constexpr = layout.quant_scale_shape(MBM, OUT_MBN)[1]
    PL: gl.constexpr = layout.result_store_layout(MBM, P_COLS, 8)
    SL: gl.constexpr = layout.result_store_layout(MBM, S_COLS, 8)
    raw_n0 = pid_n * BN + ni * MBN
    n0 = raw_n0 // ARN
    # The caller's fenced barrier has retired every wave's staging writes.
    pv = gl.amd.cdna4.async_copy.load_shared_relaxed(qp_lds_ptr, PL)
    sv = gl.amd.cdna4.async_copy.load_shared_relaxed(qs_lds_ptr, SL)
    pm = BM * block_id + mi * MBM + gl.arange(0, MBM, layout=gl.SliceLayout(1, PL))
    pn = gl.arange(0, P_COLS, layout=gl.SliceLayout(0, PL))
    gl.amd.cdna4.buffer_store(
        pv,
        y_hbm_ptr,
        pm[:, None] * y_stride_m + (n0 // 2 + pn)[None, :] * y_stride_n,
        mask=(pm < M_e)[:, None],
    )
    sm = BM * block_id + mi * MBM + gl.arange(0, MBM, layout=gl.SliceLayout(1, SL))
    sn = gl.arange(0, S_COLS, layout=gl.SliceLayout(0, SL))
    gl.amd.cdna4.buffer_store(
        sv,
        ys_hbm_ptr,
        sm[:, None] * ys_stride_m + (n0 // MX_GROUP + sn)[None, :] * ys_stride_n,
        mask=(sm < M_e)[:, None],
    )


@gluon.jit
def _epi_block_flush_frozen(
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
    """Read the WHOLE staged MXFP4 block back out of LDS and store it (gate/up split)."""
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    OUT_MBN: gl.constexpr = layout.output_n_slot()
    P_COLS: gl.constexpr = layout.quant_payload_shape(BM, OUT_MBN)[1]
    S_COLS: gl.constexpr = layout.quant_scale_shape(BM, OUT_MBN)[1]
    PL: gl.constexpr = layout.result_store_layout(BM, P_COLS, 8)
    SL: gl.constexpr = layout.result_store_layout(BM, S_COLS, 8)
    n0 = pid_n * layout.output_block_n()
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
def _epilogue_one_tile_frozen(
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
    """bias -> activation -> gammas -> output_quant -> store, for ONE mini tile.

    Under ``gate_up_split`` a "tile" is a *pair*: ``sub`` is the gate side (mini-N block
    ``ni``) and ``lin`` the linear side (block ``ni + 1``), covering the same emitted
    channels. ``STAGE_ROW`` is the row this tile occupies in a block-height LDS staging
    buffer.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    MBM: gl.constexpr = tuning_cfg.SLOT_M
    ARN: gl.constexpr = _activation_reduction_n_frozen(func_cfg)
    OUT_MBN: gl.constexpr = layout.output_n_slot()
    P_COLS: gl.constexpr = layout.quant_payload_shape(MBM, OUT_MBN)[1]
    S_COLS: gl.constexpr = layout.quant_scale_shape(MBM, OUT_MBN)[1]
    act: gl.constexpr = _act_frozen(func_cfg)
    out_ty: gl.constexpr = y_hbm_ptr.dtype.element_ty
    store_layout: gl.constexpr = layout.result_store_layout(
        MBM, OUT_MBN, out_ty.primitive_bitwidth
    )

    # GATE_PRE: the peeled last K step already applied the static scale, the bias and
    # the silu to `sub`, so those are skipped for the gate side only.
    if require_constexpr(func_cfg.has_x_static_scale):
        # Per-tensor fp8 activation scale, before bias, matching the Triton kernels.
        if require_constexpr(not GATE_PRE):
            sub = sub * x_static_scale
        if require_constexpr(_gu_split_frozen(func_cfg)):
            lin = lin * x_static_scale

    if require_constexpr(func_cfg.has_bias and func_cfg.epilogue < 2):
        # fp32, expert indexed, over the RAW N axis (before the halving). Split: one
        # tile per side, so each accumulator takes its own.
        if require_constexpr(not GATE_PRE):
            sub = sub + bias_tiles[ni][None, :]
        if require_constexpr(_gu_split_frozen(func_cfg)):
            lin = lin + bias_tiles[ni + 1][None, :]

    if require_constexpr(_gu_split_frozen(func_cfg) and func_cfg.epilogue != 0):
        # Ablation: the same [MBM, MBN] x 2 -> [MBM, MBN] reduction, minus exp2/rcp.
        out = sub * lin
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(GATE_PRE):
        out = _swiglu_combine_frozen(
            sub, lin, act.limit, ADD_RESIDUAL=act.add_residual
        )
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(_gu_split_frozen(func_cfg)):
        out = _swiglu_pair_frozen(
            sub,
            lin,
            act.alpha,
            act.limit,
            ADD_RESIDUAL=act.add_residual,
            FAST_RCP=tuning_cfg.ACT_FAST_RCP,
        )
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(_has_activation_frozen(func_cfg) and func_cfg.epilogue != 0):
        # Ablation: the same reshape/split reduction _swiglu does, minus exp2/rcp/clip.
        gate, linear = tl.split(
            tl.reshape(sub, (sub.shape[0], sub.shape[1] // 2, 2))
        )
        out = gate * linear
        gl.static_assert(out.shape[1] == OUT_MBN)
    elif require_constexpr(_has_activation_frozen(func_cfg)):
        out = _swiglu_frozen(
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

    if require_constexpr(func_cfg.has_gammas and func_cfg.epilogue < 2):
        if require_constexpr(gamma_lds_ptr is not None):
            # Only a ds_read: the global copy that filled this ran before the
            # drain. `out.type.layout` is the one way to name the post-swiglu
            # layout, so the read has to sit here rather than above the walk.
            g = gamma_lds_ptr.slice(mi * MBM, MBM).load(
                gl.SliceLayout(1, out.type.layout)
            )
        else:
            g = gamma_tiles[mi]
        out = out * g[:, None]

    # Emitted-channel base. The CTA tile is BLOCK_N // ARN emitted channels wide in both
    # packings -- what differs is only how many mini blocks that is.
    out_n0 = pid_n * layout.output_block_n() + ni * OUT_MBN
    if require_constexpr(func_cfg.output_quant is None):
        val = gl.convert_layout(
            out.to(out_ty), store_layout, assert_trivial=False
        )
        sm = gl.arange(0, MBM, layout=gl.SliceLayout(1, store_layout))
        sn = gl.arange(0, OUT_MBN, layout=gl.SliceLayout(0, store_layout))
        rows = BM * block_id + mi * MBM + sm
        gl.amd.cdna4.buffer_store(
            val,
            y_hbm_ptr,
            rows[:, None] * y_stride_m + (out_n0 + sn)[None, :] * y_stride_n,
            mask=(rows < M_e)[:, None],
            cache=tuning_cfg.result_cache_modifier,
        )
    else:
        # Fused MXFP4 output quant, straight off the fp32 accumulator. The MX group
        # runs along the emitted N axis, so the amax is over 32 emitted columns.
        gl.static_assert(func_cfg.output_quant == _DQ_MXFP4)
        gl.static_assert(OUT_MBN % MX_GROUP == 0)
        # LANE_ELEMS splits the amax: how many of the tile's elements one lane owns,
        # so the inner reduction is in-lane fp32 and only the outer one cross-lane.
        LANE_ELEMS: gl.constexpr = layout.quant_amax_lane_elems(MBM, OUT_MBN)
        payload, scale = _mxfp4_quant_gluon_frozen(
            out, OUT_MBN, MBM, MX_GROUP, LANE_ELEMS
        )
        # Fresh M ranges per store: reusing one auto-layout `offs_m` across two
        # differently-laid-out stores makes GluonResolveAutoEncodings fail with
        # "conflicting encodings" on the expand_dims.
        if require_constexpr(qp_lds_ptr is not None):
            # LDS round-trip: write in the accumulator's layout, read back in the
            # store layout, so the global store vectorises (4 x buffer_store_dwordx4
            # instead of 48 x global_store_byte).
            PL: gl.constexpr = layout.result_store_layout(MBM, P_COLS, 8)
            SL: gl.constexpr = layout.result_store_layout(MBM, S_COLS, 8)
            # if/elif/else, not early returns: a `return` from inside a nested
            # `if require_constexpr(...)` here does NOT stop the trace. Keep exactly
            # one store path reachable per configuration.
            if require_constexpr(_gu_split_frozen(func_cfg)):
                # Block-height staging: every mi writes its own MBM rows of one buffer,
                # so the caller flushes all NM tiles with a single barrier and store.
                qp_lds_ptr.slice(STAGE_ROW, MBM).store(payload)
                qs_lds_ptr.slice(STAGE_ROW, MBM).store(scale)
            elif require_constexpr(STAGE_ONLY):
                # rotating epilogue: stage and leave it; the caller flushes this buffer
                # one iteration later, after its own barrier.
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

                pm = BM * block_id + mi * MBM + gl.arange(
                    0, MBM, layout=gl.SliceLayout(1, PL)
                )
                pn = gl.arange(0, P_COLS, layout=gl.SliceLayout(0, PL))
                gl.amd.cdna4.buffer_store(
                    payload_v,
                    y_hbm_ptr,
                    pm[:, None] * y_stride_m
                    + (out_n0 // 2 + pn)[None, :] * y_stride_n,
                    mask=(pm < M_e)[:, None],
                )
                sm = BM * block_id + mi * MBM + gl.arange(
                    0, MBM, layout=gl.SliceLayout(1, SL)
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
            pm = BM * block_id + mi * MBM + gl.arange(0, MBM)
            gl.store(
                y_hbm_ptr
                + pm[:, None] * y_stride_m
                + (out_n0 // 2 + gl.arange(0, P_COLS))[None, :] * y_stride_n,
                payload,
                mask=(pm < M_e)[:, None],
            )
            sm = BM * block_id + mi * MBM + gl.arange(0, MBM)
            gl.store(
                ys_hbm_ptr
                + sm[:, None] * ys_stride_m
                + (out_n0 // MX_GROUP + gl.arange(0, S_COLS))[None, :]
                * ys_stride_n,
                scale,
                mask=(sm < M_e)[:, None],
            )


@gluon.jit
def _epilogue_store_frozen(
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
    """bias -> activation -> gammas -> output_quant -> store, per mini (M, N) tile.

    ``acc`` arrives already split into one tensor per mini tile, so the epilogue walks
    the same tiling and consumes them one at a time.
    """
    MBM: gl.constexpr = tuning_cfg.SLOT_M
    OUT_MBN: gl.constexpr = layout.output_n_slot()

    NN: gl.constexpr = layout.num_n_slots_per_block()
    NM: gl.constexpr = layout.num_m_slots_per_block()

    # Block-level operand loads, each invariant in one of the two walk indices -- bias
    # in mi, gammas in ni. Own loop variables: reusing `mi`/`ni` here would shadow the
    # walk's.
    bias_tiles = _epi_bias_tiles_frozen(
        bias_lds_ptr, bias_hbm_ptr, pid_n, N, func_cfg, tuning_cfg, layout
    )
    gamma_tiles = _epi_gamma_tiles_frozen(
        gamma_lds_ptr, gammas_hbm_ptr, block_id, M_e, func_cfg, tuning_cfg, layout
    )

    # Staging for the MXFP4 payload store, bouncing it through LDS so the global store
    # vectorises. ONE mini tile's worth, reused across the walk (overlaid on the dead
    # pipeline buffers by the liveness-based shared allocator).
    QSH: gl.constexpr = layout.quant_staging_layout()
    ROT: gl.constexpr = layout.quant_staging_rotates()
    # Split stages the whole block, not one mini tile: NM tiles into one buffer, flushed
    # once at the end of the walk.
    QROWS: gl.constexpr = layout.quant_staging_rows()
    if require_constexpr(func_cfg.output_quant is not None):
        qp_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_payload_shape(QROWS, OUT_MBN), layout=QSH
        )
        qs_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_scale_shape(QROWS, OUT_MBN), layout=QSH
        )
    else:
        qp_lds_ptr: gl.constexpr = None
        qs_lds_ptr: gl.constexpr = None
    if require_constexpr(ROT):
        # Second bank for rotating epilogue. Same overlay argument as above.
        qp2_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_payload_shape(MBM, OUT_MBN), layout=QSH
        )
        qs2_lds_ptr = gl.allocate_shared_memory(
            gl.uint8, layout.quant_scale_shape(MBM, OUT_MBN), layout=QSH
        )
    else:
        qp2_lds_ptr: gl.constexpr = None
        qs2_lds_ptr: gl.constexpr = None

    if require_constexpr(_gu_split_frozen(func_cfg)):
        # One pass over mi, pairing the two mini-N blocks: block 0 is the gate side and
        # block 1 the linear side of the *same* emitted channels. Every tile stages
        # into its own MBM rows of the block-height buffer, then one flush stores all.
        for mi in gl.static_range(NM):
            _epilogue_one_tile_frozen(
                acc[_slot_index_frozen(mi, 0, NM, NN)],
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
                lin=acc[_slot_index_frozen(mi, 1, NM, NN)],
                STAGE_ROW=mi * MBM,
                GATE_PRE=GATE_PRE,
            )
        if require_constexpr(func_cfg.output_quant is not None):
            _epi_block_flush_frozen(
                qp_lds_ptr, qs_lds_ptr,
                y_hbm_ptr, y_stride_m, y_stride_n,
                ys_hbm_ptr, ys_stride_m, ys_stride_n,
                block_id, pid_n, M_e, func_cfg, tuning_cfg, layout,
            )
    elif require_constexpr(ROT):
        # Software-pipelined staging: flush the PREVIOUS tile, then stage this one,
        # then one barrier; the two banks alternate so the write never targets the
        # buffer being read. No `mi: gl.constexpr = ...` locals here: Gluon rejects
        # rebinding a constexpr on the second unrolled iteration.
        for k in gl.static_range(NM * NN):
            if require_constexpr(k > 0):
                _epi_stage_flush_frozen(
                    qp2_lds_ptr if (k - 1) % 2 else qp_lds_ptr,
                    qs2_lds_ptr if (k - 1) % 2 else qs_lds_ptr,
                    (k - 1) // NN, (k - 1) % NN,
                    y_hbm_ptr, y_stride_m, y_stride_n,
                    ys_hbm_ptr, ys_stride_m, ys_stride_n,
                    block_id, pid_n, M_e, func_cfg, tuning_cfg, layout,
                )
            _epilogue_one_tile_frozen(
                acc[_slot_index_frozen(k // NN, k % NN, NM, NN)],
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
        _epi_stage_flush_frozen(
            qp2_lds_ptr if LK % 2 else qp_lds_ptr,
            qs2_lds_ptr if LK % 2 else qs_lds_ptr,
            LK // NN, LK % NN, y_hbm_ptr, y_stride_m, y_stride_n,
            ys_hbm_ptr, ys_stride_m, ys_stride_n,
            block_id, pid_n, M_e, func_cfg, tuning_cfg, layout,
        )
    else:
        for mi in gl.static_range(NM):
            for ni in gl.static_range(NN):
                # acc is stored in slot-traversal order; the store loop indexes through
                # the same map.
                _epilogue_one_tile_frozen(
                    acc[_slot_index_frozen(mi, ni, NM, NN)],
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


@gluon.jit
def _moe_gemm_body_frozen(
    a,
    b,
    res,
    rt,
    bias_hbm_ptr,
    stride_bias_e,
    x_static_scale_hbm_ptr,
    grid_m,
    grid_n,
    func_cfg,
    tuning_cfg,
    N: gl.constexpr,
    K: gl.constexpr,
    NUM_K,
):
    """The complete frozen kernel body, selected once by the entry point.

    Builds its own ``_FrozenLayout`` from ``tuning_cfg``; nothing of the live layout
    reaches this body.
    """
    layout = _compute_frozen_layout(tuning_cfg)
    gl.static_assert(tuning_cfg.FROZEN_STEP)
    gl.static_assert(_validate_frozen(tuning_cfg, N, K, layout))
    gl.static_assert(_validate_frozen_pipeline(tuning_cfg, K, layout))

    BK: gl.constexpr = tuning_cfg.BLOCK_K
    PK_A: gl.constexpr = BK // _pack_divisor_frozen(func_cfg, 0)
    PK_B: gl.constexpr = BK // _pack_divisor_frozen(func_cfg, 1)
    SK: gl.constexpr = BK // MX_GROUP

    pid = gl.program_id(0)
    if require_constexpr(tuning_cfg.TILE_SCHED == _TS_XCD_GROUP_M):
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
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M_e = gl.load(rt.expt_hist + expt_id)
    start_m = gl.load(rt.expt_offs_raw + expt_id)

    a_hbm_offs = _a_payload_hbm_offsets_frozen(
        a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg, layout
    )
    b_hbm_ptr = b.ptr + expt_id.to(gl.int64) * b.stride_e
    if require_constexpr(_has_scale_frozen(func_cfg, 0)):
        a_scale_hbm_ptr = a.scale_ptr
    else:
        a_scale_hbm_ptr: gl.constexpr = None
    if require_constexpr(_has_scale_frozen(func_cfg, 1)):
        b_scale_hbm_ptr = b.scale_ptr + expt_id.to(gl.int64) * b.scale_stride_e
    else:
        b_scale_hbm_ptr: gl.constexpr = None

    b_hbm_offs = _b_payload_hbm_offsets_frozen(
        b, pid_n, N, K, func_cfg, tuning_cfg, layout
    )
    if require_constexpr(_has_scale_frozen(func_cfg, 0)):
        a_scale_hbm_offs = _a_scale_hbm_offsets_frozen(
            a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg, layout
        )
    else:
        a_scale_hbm_offs: gl.constexpr = None
    if require_constexpr(_has_scale_frozen(func_cfg, 1)):
        b_scale_hbm_offs = _b_scale_hbm_offsets_frozen(
            b, pid_n, N, K, func_cfg, tuning_cfg, layout
        )
    else:
        b_scale_hbm_offs: gl.constexpr = None

    lds_ptrs = _FrozenLDSManager.alloc(func_cfg, tuning_cfg, layout)

    a_step: gl.constexpr = PK_A
    b_step: gl.constexpr = PK_B // 16 * 256 if tuning_cfg.B_PRESHUFFLED else PK_B
    s_step: gl.constexpr = SK
    hbm_ptrs = _PipelinePointers(
        a.ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    )
    if require_constexpr(_has_scale_frozen(func_cfg, 0)):
        a_scale_stride_k: gl.constexpr = a.scale_stride_k
    else:
        a_scale_stride_k: gl.constexpr = None
    if require_constexpr(_has_scale_frozen(func_cfg, 1)):
        b_scale_stride_k: gl.constexpr = b.scale_stride_k
    else:
        b_scale_stride_k: gl.constexpr = None
    pc = _PipelineConst(
        lds_ptrs,
        a_hbm_offs,
        b_hbm_offs,
        a_scale_hbm_offs,
        b_scale_hbm_offs,
        a_scale_stride_k,
        b_scale_stride_k,
        a_step,
        b_step,
        s_step,
        func_cfg,
        tuning_cfg,
        layout,
    )

    hbm_ptrs, buffers, regs = _run_frozen_pipeline(pc, hbm_ptrs, NUM_K)
    epi = _stage_epilogue_inputs_frozen(
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
    )
    regs = _drain_frozen_pipeline(pc, hbm_ptrs, buffers, regs, NUM_K, epi.groups)
    acc = _last_mfma_frozen(pc, regs)

    if require_constexpr(func_cfg.has_x_static_scale):
        x_static_scale = gl.load(x_static_scale_hbm_ptr)
    else:
        x_static_scale: gl.constexpr = None
    if require_constexpr(epi.groups > 0):
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()

    y_hbm_ptr = res.ptr + start_m.to(gl.int64) * res.stride_m
    if require_constexpr(func_cfg.output_quant is not None):
        ys_hbm_ptr = res.scale_ptr + start_m.to(gl.int64) * res.scale_stride_m
    else:
        ys_hbm_ptr: gl.constexpr = None
    _epilogue_store_frozen(
        acc,
        y_hbm_ptr,
        res.stride_m,
        res.stride_n,
        ys_hbm_ptr,
        res.scale_stride_m,
        res.scale_stride_n,
        epi.bias_hbm_base,
        block_id,
        pid_n,
        N,
        M_e,
        epi.gammas_hbm_ptr,
        epi.gamma_lds_ptr,
        epi.bias_lds_ptr,
        x_static_scale,
        func_cfg,
        tuning_cfg,
        layout,
        GATE_PRE=False,
    )


# --- vendored validation ------------------------------------------------------
# Copies of ``_validate.validate`` / ``validate_buffer_counts`` (and the
# ``validate_layout`` / ``validate_resources`` they call) as they applied to a
# FROZEN_STEP build, over the frozen layout, plus the frozen pipeline bound. The kernel
# ``static_assert``s these and the host runs them before launching a FROZEN_STEP build,
# so both reject exactly the same configurations, and the live validation may change
# without moving which frozen builds are accepted.
# ------------------------------------------------------------------------------
@gluon.constexpr_function
def _validate_frozen(tc, N, K, layout=None):
    """Validate one frozen build: the tuning enums, the layout and the resource budgets.

    ``layout`` is ``tc``'s :func:`_compute_frozen_layout` result; a caller that already
    holds it passes it in, anything else gets one computed here.
    """
    if _v(layout) is None:
        layout = _compute_frozen_layout(tc)
    fc = tc.func_cfg
    BK = _v(tc.BLOCK_K)

    assert _v(fc.epilogue) in (
        int(EpilogueMode.DEFAULT),
        int(EpilogueMode.NOP_ACTIVATION),
        int(EpilogueMode.NOP),
    ), f"epilogue {_v(fc.epilogue)} is not an EpilogueMode"
    assert _v(tc.SCHED_MODE) in (
        int(SchedMode.NONE),
        int(SchedMode.IGLP_0),
        int(SchedMode.IGLP_1),
        int(SchedMode.MFMA_16),
        int(SchedMode.MFMA_8),
    ), f"SCHED_MODE {_v(tc.SCHED_MODE)} is not a SchedMode"
    _validate_frozen_layout(tc, layout, N, K)

    if not _v(tc.UNROLL_EPILOGUE):
        for idx in (0, 1):
            assert (
                not _has_scale_frozen(fc, idx) or layout.scale_ratio_k_step(idx) == 1
            ), (
                "UNROLL_EPILOGUE=False requires every active scale to advance "
                "once per payload step"
            )

    # -- warp pipelining (the inter-wave ping-pong) --
    assert _v(tc.WARP_PIPELINE) in (
        int(WarpPipeline.NONE),
        int(WarpPipeline.COMPILER),
        int(WarpPipeline.MANUAL),
    ), (
        f"WARP_PIPELINE {_v(tc.WARP_PIPELINE)} is not a WarpPipeline: "
        f"NONE {int(WarpPipeline.NONE)}, COMPILER {int(WarpPipeline.COMPILER)}, "
        f"MANUAL {int(WarpPipeline.MANUAL)}"
    )
    if _warp_pipeline_enabled_frozen(tc):
        assert layout.num_prefetch_k_slots() == layout.num_k_slots_per_tile(), (
            f"WARP_PIPELINE needs VGPR_PREFETCH_K == BLOCK_K (got "
            f"{_v(tc.VGPR_PREFETCH_K)} vs {BK}): with a partial window the slot's "
            "MFMAs depend on the slot's own ds_read and there is no mem stage to "
            "hide behind them"
        )

    # -- register prefetch depth: the driver carries one whole payload K stage --
    pk = _v(tc.VGPR_PREFETCH_K)
    assert pk == BK, (
        f"VGPR_PREFETCH_K must equal BLOCK_K (got {pk} vs {BK}): "
        "the pipeline carries one whole payload K stage"
    )

    # -- commit-group granularity --
    assert _v(tc.WAIT_COMMIT_SCHEME) in (
        int(WaitCommitScheme.PER_OP),
        int(WaitCommitScheme.PER_PANEL),
        int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE),
        int(WaitCommitScheme.PER_STAGE_WHOLE),
    ), (
        f"WAIT_COMMIT_SCHEME {_v(tc.WAIT_COMMIT_SCHEME)} is not a WaitCommitScheme: "
        f"PER_OP {int(WaitCommitScheme.PER_OP)}, "
        f"PER_PANEL {int(WaitCommitScheme.PER_PANEL)}, "
        f"PER_STAGE_WARP_PIPELINE {int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE)}, "
        f"PER_STAGE_WHOLE {int(WaitCommitScheme.PER_STAGE_WHOLE)}"
    )

    _validate_frozen_resources(tc, layout)
    return True


@gluon.constexpr_function
def _validate_frozen_buffer_counts(tc):
    """Only components present in the operand formats participate."""
    for operand in (0, 1):
        for scale in (False, True):
            if scale and not _has_scale_frozen(tc.func_cfg, operand):
                continue
            depth = _num_buffers_frozen(tc, operand, scale)
            assert depth >= 2, (
                f"{'A' if operand == 0 else 'B'}{'_SCALE' if scale else ''}_NUM_BUFFER "
                f"({depth}) must be at least 2"
            )
    assert _v(tc.K_UNROLL) >= 1, "K_UNROLL must be at least 1"
    return True


@gluon.constexpr_function
def _validate_frozen_layout(tc, layout, N, K):
    """Validate storage placement and every tensor/tile shape contract."""
    fc = tc.func_cfg
    BM, BN, BK = _v(tc.BLOCK_M), _v(tc.BLOCK_N), _v(tc.BLOCK_K)
    N, K = _v(N), _v(K)
    instr = _v(tc.mfma_instr_shape)
    warps = _v(tc.warps_per_cta)
    tiles = _v(tc.tiles_per_warp)

    assert _v(tc.ALLOW_B_VIA_LDS) or _operand_preshuffled_frozen(tc, 1), (
        "ALLOW_B_VIA_LDS=False requires B_PRESHUFFLED"
    )
    has_custom_storage = bool(
        not _v(tc.ALLOW_B_VIA_LDS)
        or not _v(tc.ALLOW_B_SCALE_VIA_LDS)
        or not _v(tc.ALLOW_A_SCALE_VIA_LDS)
        or _v(tc.A_NUM_BUFFER)
        or _v(tc.B_NUM_BUFFER)
        or _v(tc.A_SCALE_NUM_BUFFER)
        or _v(tc.B_SCALE_NUM_BUFFER)
    )
    assert not (_v(tc.FROZEN_STEP) and has_custom_storage), (
        "FROZEN_STEP does not support register storage or per-component buffer counts"
    )
    if _v(tc.FROZEN_STEP):
        for idx in (0, 1):
            assert layout.payload_via_lds(idx), (
                f"FROZEN_STEP requires operand {idx}'s payload in LDS; "
                "use the live pipeline for a direct-register payload"
            )
    _validate_frozen_buffer_counts(tc)

    # -- host preconditions (no N tail, no K tail; only the even case exists) --
    assert N % BN == 0, f"N {N} % BLOCK_N {BN} != 0; the wrapper must fall back"
    assert K % BK == 0, f"K {K} % BLOCK_K {BK} != 0; the wrapper must fall back"

    # -- divisibility lattice --
    assert BK % instr[2] == 0
    assert BK % MX_GROUP_INT == 0

    # -- shuffled scale geometry --
    for idx in (0, 1):
        if not _scale_shuffled_frozen(tc, idx):
            continue
        axis = "M" if idx == 0 else "N"
        block = BM if idx == 0 else BN
        nonk, sk = layout.scale_mini_block_nonk(idx), layout.scale_mini_block_k(idx)
        mini = layout.scale_nonk(idx)
        assert block >= nonk >= mini, (
            f"BLOCK_{axis} >= SCALE_SLOT_{axis} >= SLOT_{axis} is required"
        )
        assert block % nonk == 0 and nonk % mini == 0, (
            f"SCALE_SLOT_{axis} must divide BLOCK_{axis} and be a multiple of SLOT_{axis}"
        )
        assert nonk % 32 == 0, (
            f"SCALE_SLOT_{axis} must contain whole 32-row stripes"
        )
        assert sk >= max(256, BK), (
            "SCALE_SLOT_K must be at least 256 and at least BLOCK_K"
        )
        assert sk & (sk - 1) == 0 and sk % BK == 0, (
            "SCALE_SLOT_K must be a power of two and a multiple of BLOCK_K"
        )
        assert K % sk == 0, "K must contain complete SCALE_SLOT_K tiles"
        if _v(tc.FROZEN_STEP):
            assert BK == 256 and sk == BK and layout.scale_ratio_non_k_slot(idx) == 1, (
                "FROZEN_STEP shuffled scales require BLOCK_K == "
                "SCALE_SLOT_K == 256 and one scale tile per payload slot; "
                "use the live pipeline for independent scale mini blocks"
            )

    # -- the slot is the unit of LDS allocation, of the global->LDS copy, of the
    #    MFMA and of the accumulator, so it must align to the CTA block --
    assert BM % _v(tc.SLOT_M) == 0
    assert BN % _v(tc.SLOT_N) == 0
    assert _fill_slots_valid_frozen(tc, layout), (
        f"fill ownership is not a valid mapping for num_m_slots_per_block "
        f"{layout.num_m_slots_per_block()} x num_n_slots_per_block "
        f"{layout.num_n_slots_per_block()} (SLOT_M {_v(tc.SLOT_M)} vs "
        f"BLOCK_M {BM}, SLOT_N {_v(tc.SLOT_N)} vs BLOCK_N {BN})"
    )
    assert _v(tc.SLOT_M) % (instr[0] * warps[0] * tiles[0]) == 0, (
        f"SLOT_M {_v(tc.SLOT_M)} must be a multiple of "
        f"instr[0]*warps[0]*tiles[0] = {instr[0] * warps[0] * tiles[0]}"
    )
    assert _v(tc.SLOT_N) % (instr[1] * warps[1] * tiles[1]) == 0, (
        f"SLOT_N {_v(tc.SLOT_N)} must be a multiple of "
        f"instr[1]*warps[1]*tiles[1] = {instr[1] * warps[1] * tiles[1]}"
    )

    # -- MFMA shape must fit the tile --
    assert BM % (instr[0] * warps[0] * tiles[0]) == 0
    assert BN % (instr[1] * warps[1] * tiles[1]) == 0

    # -- the LDS unit the byte-tiled layout is built from --
    for idx in (0, 1):
        if layout.byte_unit_lds_ok(idx) and not _operand_preshuffled_frozen(tc, idx):
            assert tiles[idx] >= 2, (
                f"operand {idx} is not preshuffled, so its LDS unit is 32 rows and "
                f"tiles_per_warp[{idx}] must be >= 2 (got {tiles[idx]})"
            )
    if _operand_preshuffled_frozen(tc, 1):
        # The 16-column-blocked global offsets only stay inside the N slot while
        # every extent is a whole number of 16-column, 16-byte blocks.
        pk_b = BK // _pack_divisor_frozen(fc, 1)
        assert BN % 16 == 0 and _v(tc.SLOT_N) % 16 == 0, (
            f"B_PRESHUFFLED needs BLOCK_N ({BN}) and SLOT_N "
            f"({_v(tc.SLOT_N)}) to be multiples of the 16-column block"
        )
        assert pk_b % 16 == 0, (
            f"B_PRESHUFFLED needs BLOCK_K/pack ({pk_b}) to be a multiple of the "
            "16-byte block"
        )

    # -- gate/up split: an N slot must be exactly one operand side --
    if _gu_split_frozen(fc):
        MBN = _v(tc.SLOT_N)
        assert layout.num_n_slots_per_block() == 2 and MBN * 2 == BN, (
            f"gate_up_split needs exactly two N slots, one per side: got "
            f"BLOCK_N {BN} / SLOT_N {MBN} = {layout.num_n_slots_per_block()}"
        )
        assert N % 2 == 0 and (N // 2) % MBN == 0, (
            f"gate_up_split needs each half of N ({N}) to be a whole number of "
            f"SLOT_N ({MBN}) tiles"
        )
        # A warp's emitted N extent has to contain a whole MX group, or the fused
        # quant's amax crosses the warp boundary.
        if _v(fc.output_quant) is not None:
            warp_n = instr[1] * tiles[1]
            assert warp_n % MX_GROUP_INT == 0, (
                f"gate_up_split with a fused MX output quant needs a warp's "
                f"emitted N extent (instr_n {instr[1]} * tiles_per_warp_n "
                f"{tiles[1]} = {warp_n}) to be a multiple of {MX_GROUP_INT}, else the "
                "amax reduction still crosses warps"
            )
            assert MBN % MX_GROUP_INT == 0

    # -- the fused MX output quant groups 32 *emitted* columns, so the raw tile has
    #    to carry 32 * activation_reduction_n of them --
    elif _v(fc.output_quant) is not None:
        arn = _activation_reduction_n_frozen(fc)
        assert BN % (MX_GROUP_INT * arn) == 0, (
            f"BLOCK_N {BN} must be a multiple of {MX_GROUP_INT * arn} in raw "
            f"(pre-halving) terms so every MX group is tile-local"
        )
        assert _v(tc.SLOT_N) % (MX_GROUP_INT * arn) == 0

    return True


@gluon.constexpr_function
def _validate_frozen_resources(tc, layout):
    """Validate layout-dependent LDS and accumulator resource budgets."""
    lds = layout.lds_bytes()
    assert lds <= _LDS_USABLE_BYTES, (
        f"LDS {lds} B > {_LDS_USABLE_BYTES} B usable ({_LDS_CAP_BYTES} cap minus "
        f"{_LDS_EPILOGUE_RESERVE_BYTES} B of epilogue scratch): reduce "
        f"NUM_LDS_BUFFER ({_v(tc.NUM_LDS_BUFFER)}) or the block sizes"
    )
    acc = layout.acc_vgprs_per_lane()
    assert acc <= 256, (
        f"fp32 accumulator alone needs {acc} VGPR/lane; 256 is the architectural "
        f"cap and WAVES_PER_EU=2 needs <= 256 total"
    )
    return True


@gluon.constexpr_function
def _validate_frozen_pipeline(tc, K, layout=None):
    """Validate the frozen driver's rings and runtime-loop lower bound.

    ``layout`` is ``tc``'s :func:`_compute_frozen_layout` result; the host omits it and
    gets one computed here.
    """
    if _v(layout) is None:
        layout = _compute_frozen_layout(tc)
    _validate_frozen_buffer_counts(tc)
    num_k = layout.num_k_tiles(K)
    depth = pipeline_depth_frozen(tc, layout)
    peeled = _pipeline_peeled_frozen(tc)
    unroll = pipeline_unroll_frozen(tc, layout)
    assert num_k >= depth + peeled + unroll, (
        f"NUM_K ({num_k}) must be at least NB_MAX ({depth}) + PEELED ({peeled}) "
        f"+ UNROLL ({unroll}) = {depth + peeled + unroll}"
    )
    return True
