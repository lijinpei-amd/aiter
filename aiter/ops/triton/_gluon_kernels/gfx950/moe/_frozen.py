# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Frozen gfx950 Gluon MoE kernel path.

``_pipeline_step_frozen`` is a verbatim copy of ``moe_gemm._pipeline_step_impl`` taken
from the best measured kernel (2026-09-02), with every env-var constexpr replaced by its
tuned value so no stray flag can perturb the reference schedule. It lives in its own
module for exactly that reason: no scheduling or execution logic here is meant to be
refactored alongside the live step. Mechanical shared-interface renames and import
moves may track the live modules; keeping this body separate still makes behavioral
edits visible in the diff.

``AITER_TRITON_MOE_GLUON_FROZEN_STEP=1`` selects this body at the kernel entry.
The complete performance-sensitive path lives here: LDS allocation and access,
dot selection, pipeline state, prologue, runtime loop, drain, and final MFMA. Shared
offset construction and epilogue primitives remain outside because they contain no
frozen/live policy.

The acceptance test for this extraction is byte-identical generated code against the
pre-extraction frozen builds. That keeps the snapshot's scheduling statements,
including its separate prologue fence, fixed while the live implementation evolves.
"""

import math

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._epilogue import _epilogue_store, _stage_epilogue_inputs
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import optional as _opt
from ._lang import require_constexpr
from ._lang import unwrap as _v
from ._layout import (
    _a_payload_hbm_offsets,
    _a_scale_hbm_offsets,
    _b_payload_hbm_offsets,
    _b_scale_hbm_offsets,
    _slot_index,
)
from ._schedule import _buffer_load_order, _buffer_load_tile
from ._types import DotKind, TileSched

_TS_XCD_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.XCD_GROUP_M))
_TS_GROUP_M: gl.constexpr = gl.constexpr(int(TileSched.GROUP_M))
_DK_MFMA: gl.constexpr = gl.constexpr(int(DotKind.MFMA))
_DK_MFMA_SCALED: gl.constexpr = gl.constexpr(int(DotKind.MFMA_SCALED))
_DK_UPCAST_MFMA: gl.constexpr = gl.constexpr(int(DotKind.UPCAST_MFMA))
_NO_SCALE: gl.constexpr = gl.constexpr(None)


@gluon.constexpr_function
def _packed_sel(tuning_cfg, idx, k_phase=0):
    """The frozen K256 selector for an LDS-staged packed scale operand."""
    if tuning_cfg.scale_packed_ok(idx) and tuning_cfg.scale_via_lds(idx):
        return tuning_cfg.scale_packed_sel(idx, k_phase)
    return None


@gluon.constexpr_function
def _any_packed(tuning_cfg):
    return (
        _packed_sel(tuning_cfg, 0) is not None or _packed_sel(tuning_cfg, 1) is not None
    )


@gluon.jit
def _dot(a, a_scale, b, b_scale, acc, func_cfg, tuning_cfg, K_PHASE: gl.constexpr = 0):
    """The frozen snapshot's matrix-instruction dispatch."""
    kind: gl.constexpr = func_cfg.dot_kind()
    a_sel: gl.constexpr = _packed_sel(tuning_cfg, 0, K_PHASE)
    b_sel: gl.constexpr = _packed_sel(tuning_cfg, 1, K_PHASE)
    any_packed: gl.constexpr = _any_packed(tuning_cfg)
    if require_constexpr(kind == _DK_MFMA_SCALED and any_packed):
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
    DO_MFMA: gl.constexpr,
    K_PHASE: gl.constexpr = 0,
):
    if require_constexpr(DO_MFMA):
        for i in gl.static_range(N_MINI):
            if require_constexpr(func_cfg.has_scale(0)):
                a_s = a_frags[2 * i + 1]
            else:
                a_s = _NO_SCALE
            if require_constexpr(func_cfg.has_scale(1)):
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
                K_PHASE,
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

    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig
    a_payload_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    a_scale_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    b_payload_lds_ptr: gl.shared_memory_descriptor | gl.constexpr
    b_scale_lds_ptr: gl.shared_memory_descriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        func_cfg,
        tuning_cfg,
        a_payload_lds_ptr,
        a_scale_lds_ptr,
        b_payload_lds_ptr,
        b_scale_lds_ptr,
    ):
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg
        self.a_payload_lds_ptr = _opt(a_payload_lds_ptr)
        self.a_scale_lds_ptr = _opt(a_scale_lds_ptr)
        self.b_payload_lds_ptr = _opt(b_payload_lds_ptr)
        self.b_scale_lds_ptr = _opt(b_scale_lds_ptr)

    @gluon.jit
    def alloc(func_cfg, tuning_cfg):
        NBA: gl.constexpr = tuning_cfg.num_buffers(0)
        NBB: gl.constexpr = tuning_cfg.num_buffers(1)
        NBAS: gl.constexpr = tuning_cfg.num_buffers(0, True)
        NBBS: gl.constexpr = tuning_cfg.num_buffers(1, True)
        NMA: gl.constexpr = tuning_cfg.num_lds_slots_per_block_non_k(0)
        NMB: gl.constexpr = tuning_cfg.num_lds_slots_per_block_non_k(1)
        a_ty: gl.constexpr = func_cfg.operand_elem_ty(0)
        b_ty: gl.constexpr = func_cfg.operand_elem_ty(1)
        a_shape: gl.constexpr = tuning_cfg.payload_lds_shape_slot(0)
        b_shape: gl.constexpr = tuning_cfg.payload_lds_shape_slot(1)

        if require_constexpr(tuning_cfg.payload_via_lds(0)):
            a_payload_lds_ptr = gl.allocate_shared_memory(
                a_ty,
                [NBA * NMA, a_shape[0], a_shape[1]],
                layout=tuning_cfg.dot_operand_lds_layout(0),
            )
        else:
            a_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(tuning_cfg.payload_via_lds(1)):
            b_payload_lds_ptr = gl.allocate_shared_memory(
                b_ty,
                [NBB * NMB, b_shape[0], b_shape[1]],
                layout=tuning_cfg.dot_operand_lds_layout(1),
            )
        else:
            b_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(func_cfg.has_scale(0)):
            as_shape: gl.constexpr = tuning_cfg.scale_shape_slot(0)
            if require_constexpr(tuning_cfg.scale_shuffled(0)):
                a_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBAS * tuning_cfg.num_scale_tiles_a()]
                    + tuning_cfg.scale_flat_shape(0),
                    layout=tuning_cfg.dot_operand_scale_lds_layout(0),
                )
            else:
                a_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBAS * NMA, as_shape[0], as_shape[1]],
                    layout=tuning_cfg.dot_operand_scale_lds_layout(0),
                )
        else:
            a_scale_lds_ptr: gl.constexpr = None
        if require_constexpr(func_cfg.has_scale(1)):
            bs_shape: gl.constexpr = tuning_cfg.scale_shape_slot(1)
            if require_constexpr(tuning_cfg.scale_shuffled(1)):
                b_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBBS * NMB] + tuning_cfg.scale_flat_shape(1),
                    layout=tuning_cfg.dot_operand_scale_lds_layout(1),
                )
            else:
                b_scale_lds_ptr = gl.allocate_shared_memory(
                    gl.uint8,
                    [NBBS * NMB, bs_shape[0], bs_shape[1]],
                    layout=tuning_cfg.dot_operand_scale_lds_layout(1),
                )
        else:
            b_scale_lds_ptr: gl.constexpr = None

        return _FrozenLDSManager(
            func_cfg,
            tuning_cfg,
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
        gl.static_assert(VIA_LDS)
        if require_constexpr(operand == 0):
            lds_ptr = self.a_payload_lds_ptr
        else:
            lds_ptr = self.b_payload_lds_ptr
        cache: gl.constexpr = (
            cfg.token_cache_modifier
            if operand == 0
            else cfg.expert_cache_modifier
        )
        _buffer_load_to_lds(
            lds_ptr.index(BUFFER_LOAD_IDX * cfg.num_lds_slots_per_block_non_k(operand) + tile),
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
        gl.static_assert(VIA_LDS)
        ratio: gl.constexpr = (
            cfg.scale_tile_ratio_a() if operand == 0 and cfg.scale_shuffled(0) else 1
        )
        if require_constexpr(
            self.func_cfg.has_scale(operand)
            and cfg.scale_via_lds(operand)
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
                    BUFFER_LOAD_IDX * (cfg.num_lds_slots_per_block_non_k(operand) // ratio)
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
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(cfg.num_k_slots_per_tile() > 1):
            width: gl.constexpr = cfg.MINI_BLOCK_K // pack
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
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(cfg.num_k_slots_per_tile() > 1):
            width: gl.constexpr = cfg.MINI_BLOCK_K // MX_GROUP
            tile_lds_ptr = tile_lds_ptr.slice(mini_idx * width, width, dim=1)
        return tile_lds_ptr

    @gluon.jit
    def _a_scale_tile(self, DS_READ_IDX, mi: gl.constexpr):
        cfg: gl.constexpr = self.tuning_cfg
        RA: gl.constexpr = cfg.scale_tile_ratio_a() if cfg.scale_shuffled(0) else 1
        tile_lds_ptr = self.a_scale_lds_ptr.index(
            DS_READ_IDX * (cfg.num_lds_slots_per_block_non_k(0) // RA) + mi // RA
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
        cfg: gl.constexpr = self.tuning_cfg
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
                    cfg.num_lds_slots_per_block_non_k(operand),
                    tile,
                    mini_idx,
                    1 - operand,
                    self.func_cfg.pack_divisor(operand),
                ),
                cfg.dot_operand_fragment_layout(operand),
            )
        if require_constexpr(READ_SCALE and self.func_cfg.has_scale(operand)):
            gl.static_assert(cfg.scale_via_lds(operand))
            if require_constexpr(cfg.scale_shuffled(operand)):
                if require_constexpr(operand == 0):
                    scale_tile_lds_ptr = self._a_scale_tile(SCALE_READ_IDX, tile)
                else:
                    scale_tile_lds_ptr = scale_lds_ptr.index(
                        SCALE_READ_IDX * cfg.num_lds_slots_per_block_non_k(operand) + tile
                    )
            if require_constexpr(cfg.scale_packed_ok(operand)):
                scale_val = _ds_read(
                    scale_tile_lds_ptr.reinterpret(
                        gl.int32,
                        cfg.packed_scale_shape(operand),
                        cfg.packed_scale_read_layout(operand),
                    ),
                    cfg.packed_scale_frag_layout(operand),
                )
            elif require_constexpr(cfg.scale_shuffled(operand)):
                scale_val = _ds_read(
                    scale_tile_lds_ptr.reinterpret(
                        gl.uint8,
                        cfg.scale_shape_slot(operand),
                        cfg.shuffled_scale_read_layout(operand),
                    ),
                    cfg.dot_operand_scale_fragment_layout(operand),
                )
            else:
                scale_val = _ds_read(
                    self._scale_slice(
                        scale_lds_ptr,
                        SCALE_READ_IDX,
                        cfg.num_lds_slots_per_block_non_k(operand),
                        tile,
                        mini_idx,
                    ),
                    cfg.dot_operand_scale_fragment_layout(operand),
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
    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig

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
    if require_constexpr(pc.func_cfg.a_has_scale() and pc.tuning_cfg.scale_via_lds(0)):
        a_scale_hbm_ptr = (
            a_scale_hbm_ptr
            + pc.tuning_cfg.scale_hbm_steps(0, STEPS, K_PHASE)
            * pc.s_step
            * pc.a_scale_stride_k
        )
    if require_constexpr(pc.func_cfg.b_has_scale() and pc.tuning_cfg.scale_via_lds(1)):
        b_scale_hbm_ptr = (
            b_scale_hbm_ptr
            + pc.tuning_cfg.scale_hbm_steps(1, STEPS, K_PHASE)
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
    NB: gl.constexpr = tc.NUM_LDS_BUFFER
    NM: gl.constexpr = tc.num_m_slots_per_block()
    NN: gl.constexpr = tc.num_n_slots_per_block()
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
                        pc, hbm_ptrs, K_PHASE=tc.scale_k_phase(i)
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
    order = _buffer_load_order(NM, NN)
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
    return min(_slot_index(mi, ni, NM, NN), NM + NN)


@gluon.constexpr_function
def _ds_read_a_tile_frozen(mi, ni, NM, NN):
    """Operand-A mini block slot ``(mi, ni)`` reads out of LDS, or None.

    Even: the same one-per-slot assignment the fills use, so a slot's fill and its read
    name the *same* position in :func:`_buffer_load_order`. Its wait then works out to
    ``G - 1 - s + STAGES_BETWEEN * G + s`` -- the ``s`` cancels and every slot waits on
    the same constant, which is exactly the uniform ``wait_group`` the reference kernel
    uses. Legacy: A(mi) at ``ni == 0``, which bunches both reads onto slot (0, 0).
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _buffer_load_tile(_buffer_load_pos_frozen(mi, ni, NM, NN), NM, NN, True)


@gluon.constexpr_function
def _ds_read_b_tile_frozen(mi, ni, NM, NN):
    """Operand-B mini block slot ``(mi, ni)`` reads out of LDS, or None."""
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _buffer_load_tile(_buffer_load_pos_frozen(mi, ni, NM, NN), NM, NN, False)


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
    """Which fill (position in :func:`_buffer_load_order`) slot ``(mi, ni)`` issues, or None.

    Even: one fill per slot, in flat slot order -- exactly the tutorial's layout, where
    each of the four ``mfma``/``mem`` region pairs moves one tile and commits one group.
    Legacy: A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``, which loads slot (0, 0)
    with two fills and leaves every slot off the first row and column with none.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    s = _slot_index(mi, ni, NM, NN)
    return s if s < NM + NN else None


@gluon.constexpr_function
def _scale_buffer_load_tile_frozen(mi, ni, NM, NN, want_a):
    """Tile whose A (resp. B) scale copy slot ``(mi, ni)`` issues, or None.

    Still one slot per mini block even when several share a scale tile: the commit
    group has to be emitted either way, because _buffer_load_wait counts G = NM + NN groups
    per stage. buffer_load_scale drops the redundant *copy* and leaves the group empty.
    """
    s = _slot_index(mi, ni, NM, NN)
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
    if require_constexpr(_slot_index(mi, ni, NM, NN) == 0 and WAIT is not None):
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
    NM: gl.constexpr = pc.tuning_cfg.num_m_slots_per_block()
    NN: gl.constexpr = pc.tuning_cfg.num_n_slots_per_block()
    # Under the even schedule the slot owns at most one fill, named by its position in
    # _buffer_load_order; under the legacy one the (ni == 0) / (mi == 0) predicates below pick.
    POS: gl.constexpr = _buffer_load_pos_frozen(mi, ni, NM, NN)
    A_TILE: gl.constexpr = _buffer_load_tile(POS, NM, NN, True)
    B_TILE: gl.constexpr = _buffer_load_tile(POS, NM, NN, False)
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
            A_SC == A_TILE and func_cfg.has_scale(0) and pc.tuning_cfg.scale_via_lds(0)
        ):
            pc.lds_ptrs.buffer_load_scale(
                0,
                True,
                BUFFER_LOAD_IDX,
                A_TILE,
                a_scale_hbm_ptr,
                _opt_at(pc.a_scale_hbm_offs, A_TILE, func_cfg.a_has_scale()),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(A_SC is not None and A_SC != A_TILE):
        if require_constexpr(func_cfg.has_scale(0) and pc.tuning_cfg.scale_via_lds(0)):
            pc.lds_ptrs.buffer_load_scale(
                0,
                True,
                BUFFER_LOAD_IDX,
                A_SC,
                a_scale_hbm_ptr,
                _opt_at(pc.a_scale_hbm_offs, A_SC, func_cfg.a_has_scale()),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_TILE is not None):
        pc.lds_ptrs.buffer_load_payload(
            1, True, BUFFER_LOAD_IDX, B_TILE, b_hbm_ptr, pc.b_hbm_offs[B_TILE], 0
        )
        if require_constexpr(
            B_SC == B_TILE and func_cfg.has_scale(1) and pc.tuning_cfg.scale_via_lds(1)
        ):
            pc.lds_ptrs.buffer_load_scale(
                1,
                True,
                BUFFER_LOAD_IDX,
                B_TILE,
                b_scale_hbm_ptr,
                _opt_at(pc.b_scale_hbm_offs, B_TILE, func_cfg.b_has_scale()),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_SC is not None and B_SC != B_TILE):
        if require_constexpr(func_cfg.has_scale(1) and pc.tuning_cfg.scale_via_lds(1)):
            pc.lds_ptrs.buffer_load_scale(
                1,
                True,
                BUFFER_LOAD_IDX,
                B_SC,
                b_scale_hbm_ptr,
                _opt_at(pc.b_scale_hbm_offs, B_SC, func_cfg.b_has_scale()),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ADVANCE):
        # One bump per step: the frozen kernel did not fold steps into soffset.
        a_hbm_ptr = a_hbm_ptr + KU * pc.a_step
        b_hbm_ptr = b_hbm_ptr + KU * pc.b_step
        if require_constexpr(func_cfg.a_has_scale() and pc.tuning_cfg.scale_via_lds(0)):
            a_scale_hbm_ptr = a_scale_hbm_ptr + KU * pc.s_step * pc.a_scale_stride_k
        if require_constexpr(func_cfg.b_has_scale() and pc.tuning_cfg.scale_via_lds(1)):
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
    gl.static_assert(pc.func_cfg.has_scale(operand))
    gl.static_assert(not tc.scale_via_lds(operand))
    cache: gl.constexpr = (
        tc.token_scale_cache_modifier
        if operand == 0
        else tc.expert_scale_cache_modifier
    )
    if require_constexpr(tc.scale_shuffled(operand)):
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
            tc.dot_operand_scale_fragment_layout(operand),
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
    NUM_MINI: gl.constexpr = tc.num_k_slots_per_tile()
    SK_MINI: gl.constexpr = tc.MINI_BLOCK_K // MX_GROUP
    HAS: gl.constexpr = pc.func_cfg.has_scale(operand)
    SCALE_VIA_LDS: gl.constexpr = HAS and tc.scale_via_lds(operand)
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
    #   FILLS_FIRST constexpr -- that was tc.B_IN_REG and is inlined False below)
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
    NM: gl.constexpr = tc.num_m_slots_per_block()
    NN: gl.constexpr = tc.num_n_slots_per_block()
    NUM_MINI: gl.constexpr = tc.num_k_slots_per_tile()
    PF_MINI: gl.constexpr = tc.num_prefetch_k_slots()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_HAS: gl.constexpr = func_cfg.a_has_scale()
    B_HAS: gl.constexpr = func_cfg.b_has_scale()
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
            if require_constexpr(MPP and _slot_index(mi, ni, NM, NN) % 4 == 0):
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
                if require_constexpr(_slot_index(mi, ni, NM, NN) % 4 == 0):
                    # _MPP_PIN_WAIT = 1 put the leading sched_barrier before the
                    # wait instead, so only the trailing one is emitted here.
                    # _MPP_NO_FENCE = 0 and _MPP_ALL_FENCED = 0: the stage head is
                    # the one fenced barrier, the rest are bare.
                    if require_constexpr(_slot_index(mi, ni, NM, NN) == 0):
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
                regs.acc[_slot_index(mi, ni, NM, NN)],
                PF_MINI,
                func_cfg,
                tc,
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
        if require_constexpr(A_HAS and not tc.scale_via_lds(0)):
            a_scale_hbm_ptr = a_scale_hbm_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(B_HAS and not tc.scale_via_lds(1)):
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


@gluon.constexpr_function
def _validate_frozen_pipeline(tc, K):
    """Validate the frozen driver's rings and runtime-loop lower bound."""
    tc.validate_buffer_counts()
    num_k = tc.num_k_tiles(K)
    depth = tc.pipeline_depth()
    peeled = _pipeline_peeled_frozen(tc)
    unroll = tc.pipeline_unroll()
    assert num_k >= depth + peeled + unroll, (
        f"NUM_K ({num_k}) must be at least NB_MAX ({depth}) + PEELED ({peeled}) "
        f"+ UNROLL ({unroll}) = {depth + peeled + unroll}"
    )
    return True


@gluon.jit
def _index_frozen(
    tc,
    step,
    kind: gl.constexpr,
    FILL: gl.constexpr,
    KI: gl.constexpr,
    IN_LOOP: gl.constexpr,
):
    if require_constexpr(kind % 2 and not tc.func_cfg.has_scale(kind // 2)):
        out = 0
    else:
        depth: gl.constexpr = tc.num_buffers(kind // 2, kind % 2 != 0)
        advance: gl.constexpr = depth - 1 if FILL else 0
        if require_constexpr(IN_LOOP and tc.pipeline_unroll() % depth == 0):
            out = (_pipeline_peeled_frozen(tc) + KI + 1 + advance) % depth
        else:
            tile = step + advance
            if require_constexpr(tc.pipeline_register_period() == 1):
                gl.assume(tile >= 0)
            out = tile % depth
    return out


@gluon.jit
def _init_buffers_frozen(pc):
    """Mirror the pre-extraction driver's inert register queues exactly."""
    tc: gl.constexpr = pc.tuning_cfg
    a, b, a_scale, b_scale = (), (), (), ()
    if require_constexpr(not tc.payload_via_lds(0)):
        for _ in gl.static_range(
            tc.num_buffers(0)
            * tc.num_m_slots_per_block()
            * tc.num_k_slots_per_tile()
        ):
            a += (
                gl.zeros(
                    [
                        tc.MINI_BLOCK_M,
                        tc.MINI_BLOCK_K // pc.func_cfg.pack_divisor(0),
                    ],
                    pc.func_cfg.operand_elem_ty(0),
                    tc.dot_operand_fragment_layout(0),
                ),
            )
    if require_constexpr(not tc.payload_via_lds(1)):
        for _ in gl.static_range(
            tc.num_buffers(1)
            * tc.num_n_slots_per_block()
            * tc.num_k_slots_per_tile()
        ):
            b += (
                gl.zeros(
                    [tc.MINI_BLOCK_K // pc.func_cfg.pack_divisor(1), tc.MINI_BLOCK_N],
                    pc.func_cfg.operand_elem_ty(1),
                    tc.dot_operand_fragment_layout(1),
                ),
            )
    if require_constexpr(pc.func_cfg.a_has_scale() and not tc.scale_via_lds(0)):
        for _ in gl.static_range(
            tc.num_buffers(0, True) * tc.num_m_slots_per_block() * tc.num_k_slots_per_tile()
        ):
            if require_constexpr(tc.scale_packed_k128(0)):
                a_scale += (
                    gl.zeros(
                        tc.packed_scale_shape(0),
                        gl.int32,
                        tc.packed_scale_frag_layout(0),
                    ),
                )
            else:
                a_scale += (
                    gl.zeros(
                        [tc.MINI_BLOCK_M, tc.MINI_BLOCK_K // 32],
                        gl.uint8,
                        tc.dot_operand_scale_fragment_layout(0),
                    ),
                )
    if require_constexpr(pc.func_cfg.b_has_scale() and not tc.scale_via_lds(1)):
        for _ in gl.static_range(
            tc.num_buffers(1, True) * tc.num_n_slots_per_block() * tc.num_k_slots_per_tile()
        ):
            if require_constexpr(tc.scale_packed_k128(1)):
                b_scale += (
                    gl.zeros(
                        tc.packed_scale_shape(1),
                        gl.int32,
                        tc.packed_scale_frag_layout(1),
                    ),
                )
            else:
                b_scale += (
                    gl.zeros(
                        [tc.MINI_BLOCK_N, tc.MINI_BLOCK_K // 32],
                        gl.uint8,
                        tc.dot_operand_scale_fragment_layout(1),
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
    ptrs, regs = _pipeline_step_frozen(
        pc,
        ptrs,
        regs,
        _index_frozen(tc, step, 0, True, KI, IN_LOOP or STATIC_PHASE),
        _index_frozen(tc, step, 0, False, KI, IN_LOOP or STATIC_PHASE),
        tc.pipeline_depth() - 2 - (STAGE if DRAIN else 0),
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
    gl.static_assert(
        tc.num_prefetch_k_slots() == tc.num_k_slots_per_tile(),
        "the frozen pipeline requires VGPR_PREFETCH_K == BLOCK_K",
    )
    depth: gl.constexpr = tc.pipeline_depth()
    unroll: gl.constexpr = tc.pipeline_unroll()
    peeled: gl.constexpr = _pipeline_peeled_frozen(tc)
    main = NUM_K - depth
    gl.assume(main >= peeled + unroll)
    remaining = main - peeled
    countdown: gl.constexpr = (
        not pc.func_cfg.a_has_scale()
        and not pc.func_cfg.b_has_scale()
        and tc.pipeline_register_period() == 1
    )
    if require_constexpr(not countdown):
        unroll_end = remaining // unroll * unroll
    buffers = _init_buffers_frozen(pc)
    ptrs = _prologue_frozen(pc, ptrs)
    acc = ()
    for ni in gl.static_range(tc.num_n_slots_per_block()):
        for mi in gl.static_range(tc.num_m_slots_per_block()):
            acc += (
                gl.zeros(
                    [tc.MINI_BLOCK_M, tc.MINI_BLOCK_N],
                    pc.func_cfg.mma_acc_dtype,
                    tc.dot_result_fragment_layout(),
                ),
            )
    regs = _PipelineRegFragments((), (), (), (), acc)
    ptrs, buffers, regs = _step_frozen(pc, ptrs, buffers, regs, 0, 0, DOT=False)
    for x in gl.static_range(peeled):
        ptrs, buffers, regs = _step_frozen(pc, ptrs, buffers, regs, x + 1, x + 1)
    if require_constexpr(not countdown):
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
    else:
        base = 0
        left = remaining
        while left >= unroll:
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
            base += unroll
            left -= unroll
        unroll_end = remaining - left
    if require_constexpr(unroll > 6 and tc.pipeline_register_period() > 1):
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
    main = NUM_K - tc.pipeline_depth()
    period: gl.constexpr = math.lcm(
        tc.num_buffers(0),
        tc.num_buffers(1),
        tc.num_buffers(0, True) if pc.func_cfg.a_has_scale() else 1,
        tc.num_buffers(1, True) if pc.func_cfg.b_has_scale() else 1,
        2 if tc.scale_packed_k128(0) or tc.scale_packed_k128(1) else 1,
    )
    if require_constexpr(
        period <= 3
        and tc.pipeline_unroll() % period == 0
        and tc.pipeline_register_period() == 1
        and pc.func_cfg.output_quant is None
    ):
        phase = main % period
        for p in gl.static_range(period):
            if phase == p:
                for j in gl.static_range(tc.pipeline_depth() - 1):
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
        for j in gl.static_range(tc.pipeline_depth() - 1):
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
    acc = ()
    for ni in gl.static_range(tc.num_n_slots_per_block()):
        for mi in gl.static_range(tc.num_m_slots_per_block()):
            acc += (
                _maybe_block_dot(
                    _take_reg_pairs(
                        regs.a_payload,
                        regs.a_scale,
                        mi * tc.num_k_slots_per_tile(),
                        tc.num_k_slots_per_tile(),
                    ),
                    _take_reg_pairs(
                        regs.b_payload,
                        regs.b_scale,
                        ni * tc.num_k_slots_per_tile(),
                        tc.num_k_slots_per_tile(),
                    ),
                    regs.acc[
                        _slot_index(
                            mi,
                            ni,
                            tc.num_m_slots_per_block(),
                            tc.num_n_slots_per_block(),
                        )
                    ],
                    tc.num_k_slots_per_tile(),
                    pc.func_cfg,
                    tc,
                    True,
                    0,
                ),
            )
    return acc


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
    """The complete frozen kernel body, selected once by the entry point."""
    gl.static_assert(tuning_cfg.FROZEN_STEP)
    gl.static_assert(tuning_cfg.validate(N, K))
    gl.static_assert(_validate_frozen_pipeline(tuning_cfg, K))

    BK: gl.constexpr = tuning_cfg.BLOCK_K
    PK_A: gl.constexpr = BK // func_cfg.a_pack_divisor()
    PK_B: gl.constexpr = BK // func_cfg.b_pack_divisor()
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

    a_hbm_offs = _a_payload_hbm_offsets(
        a, rt, block_id, M_e, start_m, func_cfg, tuning_cfg
    )
    b_hbm_ptr = b.ptr + expt_id.to(gl.int64) * b.stride_e
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_hbm_ptr = a.scale_ptr
    else:
        a_scale_hbm_ptr: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_hbm_ptr = b.scale_ptr + expt_id.to(gl.int64) * b.scale_stride_e
    else:
        b_scale_hbm_ptr: gl.constexpr = None

    b_hbm_offs = _b_payload_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg)
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_hbm_offs = _a_scale_hbm_offsets(
            a, rt, block_id, M_e, start_m, pid_m, K, func_cfg, tuning_cfg
        )
    else:
        a_scale_hbm_offs: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
        b_scale_hbm_offs = _b_scale_hbm_offsets(b, pid_n, N, K, func_cfg, tuning_cfg)
    else:
        b_scale_hbm_offs: gl.constexpr = None

    lds_ptrs = _FrozenLDSManager.alloc(func_cfg, tuning_cfg)

    a_step: gl.constexpr = PK_A
    b_step: gl.constexpr = PK_B // 16 * 256 if tuning_cfg.B_PRESHUFFLED else PK_B
    s_step: gl.constexpr = SK
    hbm_ptrs = _PipelinePointers(
        a.ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    )
    if require_constexpr(func_cfg.a_has_scale()):
        a_scale_stride_k: gl.constexpr = a.scale_stride_k
    else:
        a_scale_stride_k: gl.constexpr = None
    if require_constexpr(func_cfg.b_has_scale()):
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
    )

    hbm_ptrs, buffers, regs = _run_frozen_pipeline(pc, hbm_ptrs, NUM_K)
    epi = _stage_epilogue_inputs(
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
    _epilogue_store(
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
        GATE_PRE=False,
    )
