# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LDS allocation, filling and reading for the gfx950 Gluon MoE GEMMs."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import optional as _opt
from ._lang import require_constexpr

__all__ = ["LDSManager"]


@gluon.jit
def _ds_read(lds_ptr, layout: gl.constexpr):
    """LDS -> register after the pipeline's wait and barrier."""
    return lds_ptr.load(layout)


@gluon.jit
def _buffer_load_to_lds(
    lds_ptr,
    hbm_ptr,
    hbm_offs,
    cache: gl.constexpr,
    SOFF: gl.constexpr = 0,
):
    """Copy one tile directly from global memory to LDS.

    ``SOFF`` is a byte offset from ``hbm_ptr`` that is uniform across the block. It rides in
    the buffer op's ``soffset`` (SGPR) field, so unlike folding it into ``hbm_offs`` it
    costs no VGPRs and no per-lane arithmetic. See ``SOFF_UNROLL`` in ``_config.py``:
    it lets one base pointer serve every step of an unrolled body.
    """
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        lds_ptr, hbm_ptr, hbm_offs, cache_modifier=cache, soffset=SOFF
    )


@gluon.jit
def _scale_fragments(scale, cfg, operand: gl.constexpr):
    """Split one shuffled scale load in registers, retaining whole packed words."""
    nonk: gl.constexpr = cfg.scale_mini_block_nonk(operand)
    sk: gl.constexpr = cfg.scale_mini_block_k(operand)
    mini: gl.constexpr = cfg.MINI_BLOCK_K
    packed: gl.constexpr = cfg.scale_packed_ok(operand)
    fragments = ()
    for n in gl.static_range(cfg.scale_tile_ratio(operand)):
        for k in gl.static_range(sk // mini):
            if require_constexpr(packed):
                if require_constexpr(
                    cfg.scale_tile_ratio(operand) == 1 and sk == max(256, mini)
                ):
                    fragment = scale
                else:
                    # Physical order is stripe, K256 group, lane dword. Splitting
                    # these axes never unpacks a dword or moves it across lanes.
                    words = gl.reshape(scale, [nonk // 32, sk // 256, 64])
                    words = gl.amd.slice(
                        words,
                        [cfg.scale_nonk(operand) // 32, max(256, mini) // 256, 64],
                        [n * cfg.scale_nonk(operand) // 32, k * mini // 256, 0],
                    )
                    fragment = gl.convert_layout(
                        gl.reshape(words, cfg.packed_scale_shape(operand)),
                        cfg.packed_scale_frag_layout(operand),
                    )
            else:
                fragment = gl.amd.slice(
                    scale,
                    cfg.scale_fragment_shape_slot(operand),
                    [n * cfg.scale_nonk(operand), k * mini // 32],
                )
            fragments += (fragment,)
    return fragments


@aggregate
@strip_annotate
class LDSManager:
    """Owns the multi-buffered operand staging area."""

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
        """Static factory -- invoked as ``LDSManager.alloc(...)``, no ``self``.

        The leading axis runs over pipeline buffer *and* mini block
        (``num_m_slots_per_block()`` tiles for A, ``num_n_slots_per_block()`` for B), flattened as
        ``buffer_idx * n_tiles + tile``: a shared allocation takes a layout of the tile's own
        rank, so two separate leading axes are not expressible. The mini block is a real
        allocation rather than a slice of one big tile so that each one keeps the padded
        layout ``compute_efficient_padded_shared_layout`` picked *for its own shape* --
        that is what makes the per-mini-block direct-to-LDS copy as coalesced and as wide
        as the whole-tile copy was.
        """
        a_ty: gl.constexpr = func_cfg.operand_elem_ty(0)
        b_ty: gl.constexpr = func_cfg.operand_elem_ty(1)

        if require_constexpr(tuning_cfg.payload_via_lds(0)):
            a_payload_lds_ptr = gl.allocate_shared_memory(
                a_ty,
                tuning_cfg.payload_lds_shape_block(0),
                layout=tuning_cfg.dot_operand_lds_layout(0),
            )
        else:
            a_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(tuning_cfg.payload_via_lds(1)):
            b_payload_lds_ptr = gl.allocate_shared_memory(
                b_ty,
                tuning_cfg.payload_lds_shape_block(1),
                layout=tuning_cfg.dot_operand_lds_layout(1),
            )
        else:
            b_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(func_cfg.has_scale(0) and tuning_cfg.scale_via_lds(0)):
            a_scale_lds_ptr = gl.allocate_shared_memory(
                gl.uint8,
                tuning_cfg.scale_lds_shape_block(0),
                layout=tuning_cfg.dot_operand_scale_lds_layout(0),
            )
        else:
            a_scale_lds_ptr: gl.constexpr = None
        if require_constexpr(func_cfg.has_scale(1) and tuning_cfg.scale_via_lds(1)):
            b_scale_lds_ptr = gl.allocate_shared_memory(
                gl.uint8,
                tuning_cfg.scale_lds_shape_block(1),
                layout=tuning_cfg.dot_operand_scale_lds_layout(1),
            )
        else:
            b_scale_lds_ptr: gl.constexpr = None

        return LDSManager(
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
        """Load one A (0) or B (1) payload tile into LDS or operand registers.

        The register path returns the complete stage as mini-K fragments; its
        offsets must already use the operand fragment layout. The caller owns
        the register ring, so this path never writes the LDS staging allocation.

        Gathered A rows are clamped modulo the expert's token count. No load
        mask is needed: the output store mask drops the padded rows.
        """
        cfg: gl.constexpr = self.tuning_cfg
        cache: gl.constexpr = (
            cfg.token_cache_modifier if operand == 0 else cfg.expert_cache_modifier
        )
        if require_constexpr(VIA_LDS):
            if require_constexpr(operand == 0):
                lds_ptr = self.a_payload_lds_ptr
            else:
                lds_ptr = self.b_payload_lds_ptr
            _buffer_load_to_lds(
                lds_ptr.index(
                    BUFFER_LOAD_IDX * cfg.num_lds_slots_per_block_non_k(operand) + tile
                ),
                hbm_ptr,
                hbm_offs,
                cache,
                SOFF,
            )
        else:
            payload = gl.amd.cdna4.buffer_load(
                ptr=hbm_ptr, offsets=hbm_offs, cache=cache, soffset=SOFF
            )
            fragments = ()
            for mini in gl.static_range(cfg.num_k_slots_per_tile()):
                if require_constexpr(cfg.num_k_slots_per_tile() > 1):
                    fragment = gl.amd.slice(
                        payload,
                        cfg.payload_fragment_shape_slot(operand),
                        cfg.payload_fragment_offset_slot(operand, mini),
                    )
                else:
                    fragment = payload
                fragments = fragments + (fragment,)
            return fragments

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
        """Load scale mini blocks covering one stage, or one wider K scale tile."""
        cfg: gl.constexpr = self.tuning_cfg
        cache: gl.constexpr = (
            cfg.token_scale_cache_modifier
            if operand == 0
            else cfg.expert_scale_cache_modifier
        )
        ratio: gl.constexpr = cfg.scale_tile_ratio(operand)
        if require_constexpr(VIA_LDS):
            if require_constexpr(
                self.func_cfg.has_scale(operand)
                and cfg.scale_via_lds(operand)
                and tile % ratio == 0
            ):
                if require_constexpr(operand == 0):
                    lds_ptr = self.a_scale_lds_ptr
                else:
                    lds_ptr = self.b_scale_lds_ptr
                for k in gl.static_range(cfg.scale_load_k_tiles(operand)):
                    _buffer_load_to_lds(
                        lds_ptr.index(
                            (
                                BUFFER_LOAD_IDX * cfg.num_scale_tiles(operand)
                                + tile // ratio
                            )
                            * cfg.scale_load_k_tiles(operand)
                            + k
                        ),
                        hbm_ptr,
                        hbm_offs,
                        cache,
                        SOFF + k * cfg.scale_mini_block_k(operand),
                    )
        else:
            gl.static_assert(self.func_cfg.has_scale(operand))
            gl.static_assert(not cfg.scale_via_lds(operand))
            if require_constexpr(cfg.scale_shuffled(operand)):
                chunks = ()
                for k in gl.static_range(cfg.scale_load_k_tiles(operand)):
                    if require_constexpr(cfg.scale_packed_ok(operand)):
                        scale = gl.amd.cdna4.buffer_load(
                            ptr=hbm_ptr.to(gl.pointer_type(gl.int32)),
                            offsets=hbm_offs // 4,
                            cache=cache,
                            soffset=SOFF + k * cfg.scale_mini_block_k(operand),
                        )
                    else:
                        scale = gl.convert_layout(
                            gl.amd.cdna4.buffer_load(
                                ptr=hbm_ptr,
                                offsets=hbm_offs,
                                cache=cache,
                                contiguity=4,
                                soffset=SOFF + k * cfg.scale_mini_block_k(operand),
                            ),
                            cfg.scale_load_layout(operand),
                        )
                    chunks += (_scale_fragments(scale, cfg, operand),)
                fragments = ()
                for n in gl.static_range(ratio):
                    for k in gl.static_range(cfg.scale_load_k_tiles(operand)):
                        for mini in gl.static_range(
                            cfg.scale_mini_block_k(operand) // cfg.MINI_BLOCK_K
                        ):
                            fragments += (
                                chunks[k][
                                    n
                                    * (
                                        cfg.scale_mini_block_k(operand)
                                        // cfg.MINI_BLOCK_K
                                    )
                                    + mini
                                ],
                            )
            else:
                scale = gl.amd.cdna4.buffer_load(
                    ptr=hbm_ptr, offsets=hbm_offs, cache=cache, soffset=SOFF
                )
                fragments = ()
                for mini in gl.static_range(cfg.num_k_slots_per_tile()):
                    if require_constexpr(cfg.num_k_slots_per_tile() > 1):
                        fragment = gl.amd.slice(
                            scale,
                            cfg.scale_fragment_shape_slot(operand),
                            cfg.scale_fragment_offset_slot(mini),
                        )
                    else:
                        fragment = scale
                    fragments += (fragment,)
            return fragments

    @gluon.jit
    def ds_read_scale(self, operand: gl.constexpr, SCALE_READ_IDX, tile: gl.constexpr):
        """Read each scale mini block once and retain all of its consumer fragments."""
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(operand == 0):
            lds_ptr = self.a_scale_lds_ptr
        else:
            lds_ptr = self.b_scale_lds_ptr
        if require_constexpr(cfg.scale_shuffled(operand)):
            chunks = ()
            for k in gl.static_range(cfg.scale_load_k_tiles(operand)):
                scale_ptr = lds_ptr.index(
                    (
                        SCALE_READ_IDX * cfg.num_scale_tiles(operand)
                        + tile // cfg.scale_tile_ratio(operand)
                    )
                    * cfg.scale_load_k_tiles(operand)
                    + k
                )
                if require_constexpr(cfg.scale_packed_ok(operand)):
                    scale = _ds_read(
                        scale_ptr.reinterpret(
                            gl.int32,
                            cfg.packed_scale_load_shape(operand),
                            cfg.packed_scale_read_layout(operand, True),
                        ),
                        cfg.packed_scale_load_layout(operand),
                    )
                else:
                    scale = _ds_read(
                        scale_ptr.reinterpret(
                            gl.uint8,
                            cfg.scale_load_shape(operand),
                            cfg.shuffled_scale_read_layout(operand, True),
                        ),
                        cfg.scale_load_layout(operand),
                    )
                chunks += (_scale_fragments(scale, cfg, operand),)
            fragments = ()
            for n in gl.static_range(cfg.scale_tile_ratio(operand)):
                for k in gl.static_range(cfg.scale_load_k_tiles(operand)):
                    for mini in gl.static_range(
                        cfg.scale_mini_block_k(operand) // cfg.MINI_BLOCK_K
                    ):
                        fragments += (
                            chunks[k][
                                n
                                * (cfg.scale_mini_block_k(operand) // cfg.MINI_BLOCK_K)
                                + mini
                            ],
                        )
        else:
            fragments = ()
            for mini in gl.static_range(cfg.num_k_slots_per_tile()):
                fragments += (
                    _ds_read(
                        self._scale_slice(
                            lds_ptr,
                            SCALE_READ_IDX,
                            cfg.num_scale_tiles(operand),
                            tile,
                            mini,
                            operand,
                        ),
                        cfg.dot_operand_scale_fragment_layout(operand),
                    ),
                )
        return fragments

    @gluon.jit
    def commit_buffer_load(self):
        """Close the current async-copy group at the caller's configured boundary."""
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def wait_buffer_load_groups(self, num_group: gl.constexpr):
        """Block until at most ``num_group`` commit groups remain outstanding."""
        gl.amd.cdna4.async_copy.wait_group(num_group)

    @gluon.jit
    def _payload_slice(
        self,
        lds_ptr,
        DS_READ_IDX,
        n_tiles: gl.constexpr,
        tile: gl.constexpr,
        mini_idx: gl.constexpr,
        operand: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(cfg.num_k_slots_per_tile() > 1):
            k_dim: gl.constexpr = 1 - operand
            width: gl.constexpr = cfg.payload_fragment_shape_slot(operand)[k_dim]
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
        operand: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(cfg.num_k_slots_per_tile() > 1):
            width: gl.constexpr = cfg.scale_fragment_shape_slot(operand)[1]
            tile_lds_ptr = tile_lds_ptr.slice(mini_idx * width, width, dim=1)
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
        """Read one A (0) or B (1) mini-M/N, mini-K fragment from LDS.

        An absent or disabled component returns None.
        """
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(operand == 0):
            payload_lds_ptr = self.a_payload_lds_ptr
        else:
            payload_lds_ptr = self.b_payload_lds_ptr
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
                    operand,
                ),
                cfg.dot_operand_fragment_layout(operand),
            )
        if require_constexpr(READ_SCALE and self.func_cfg.has_scale(operand)):
            gl.static_assert(
                cfg.scale_via_lds(operand),
                "ds_read_frag only reads LDS-staged scales",
            )
            ratio: gl.constexpr = cfg.scale_tile_ratio(operand)
            fragments = self.ds_read_scale(
                operand, SCALE_READ_IDX, tile // ratio * ratio
            )
            scale_val = fragments[
                (tile % ratio) * cfg.scale_read_k_slots(operand) + mini_idx
            ]
        else:
            scale_val: gl.constexpr = None
        return payload, scale_val
