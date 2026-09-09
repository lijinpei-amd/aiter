# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LDS allocation, filling and reading for the gfx950 Gluon MoE GEMMs."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import MX_GROUP_CE as MX_GROUP
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
        (``num_mini_m()`` tiles for A, ``num_mini_n()`` for B), flattened as
        ``buffer_idx * n_tiles + tile``: a shared allocation takes a layout of the tile's own
        rank, so two separate leading axes are not expressible. The mini block is a real
        allocation rather than a slice of one big tile so that each one keeps the padded
        layout ``compute_efficient_padded_shared_layout`` picked *for its own shape* --
        that is what makes the per-mini-block direct-to-LDS copy as coalesced and as wide
        as the whole-tile copy was.
        """
        NBA: gl.constexpr = tuning_cfg.num_buffers(0)
        NBB: gl.constexpr = tuning_cfg.num_buffers(1)
        NBAS: gl.constexpr = tuning_cfg.num_buffers(0, True)
        NBBS: gl.constexpr = tuning_cfg.num_buffers(1, True)
        NMA: gl.constexpr = tuning_cfg.num_lds_tiles(0)
        NMB: gl.constexpr = tuning_cfg.num_lds_tiles(1)
        a_ty: gl.constexpr = func_cfg.operand_elem_ty(0)
        b_ty: gl.constexpr = func_cfg.operand_elem_ty(1)
        a_shape: gl.constexpr = tuning_cfg.lds_shape(0)
        b_shape: gl.constexpr = tuning_cfg.lds_shape(1)

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
        if require_constexpr(
            func_cfg.has_scale(0)
            and (tuning_cfg.FROZEN_STEP or tuning_cfg.scale_via_lds(0))
        ):
            as_shape: gl.constexpr = tuning_cfg.scale_shape(0)
            if require_constexpr(tuning_cfg.scale_shuffled(0)):
                # Flat: direct-to-LDS on gfx9 cannot scatter, so the staging tile has to
                # be written coalesced; the fragment view comes back on the read.
                # A-scale tiles are counted separately from the payload mini blocks:
                # scale_mini_m() may cover several of them so the copy's stripe count
                # matches warps_per_cta and no warp is replicated.
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
        if require_constexpr(
            func_cfg.has_scale(1)
            and (tuning_cfg.FROZEN_STEP or tuning_cfg.scale_via_lds(1))
        ):
            bs_shape: gl.constexpr = tuning_cfg.scale_shape(1)
            if require_constexpr(tuning_cfg.scale_shuffled(1)):
                # Flat: direct-to-LDS on gfx9 cannot scatter, so the staging tile has to
                # be written coalesced; the fragment view comes back on the read.
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
        cache: gl.constexpr = cfg.token_mod if operand == 0 else cfg.expert_mod
        if require_constexpr(VIA_LDS):
            if require_constexpr(operand == 0):
                lds_ptr = self.a_payload_lds_ptr
            else:
                lds_ptr = self.b_payload_lds_ptr
            _buffer_load_to_lds(
                lds_ptr.index(BUFFER_LOAD_IDX * cfg.num_lds_tiles(operand) + tile),
                hbm_ptr,
                hbm_offs,
                cache,
                SOFF,
            )
        else:
            payload = gl.amd.cdna4.buffer_load(
                ptr=hbm_ptr, offsets=hbm_offs, cache=cache, soffset=SOFF
            )
            width: gl.constexpr = cfg.MINI_BLOCK_K // self.func_cfg.pack_divisor(
                operand
            )
            fragments = ()
            for mini in gl.static_range(cfg.num_mini_k()):
                if require_constexpr(cfg.num_mini_k() > 1):
                    if require_constexpr(operand == 0):
                        fragment = gl.amd.slice(
                            payload, [cfg.MINI_BLOCK_M, width], [0, mini * width]
                        )
                    else:
                        fragment = gl.amd.slice(
                            payload, [width, cfg.MINI_BLOCK_N], [mini * width, 0]
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
        """Load one A (0) or B (1) scale tile into LDS or return register fragments.

        Only the first A mini block sharing an LDS scale tile issues its copy.
        Register offsets describe the logical scale grid, including any
        preshuffle. K128 stages return the complete packed K256 word, matching
        LDS; the consumer selects its K half after reading the register ring.
        """
        cfg: gl.constexpr = self.tuning_cfg
        cache: gl.constexpr = (
            cfg.token_scale_mod if operand == 0 else cfg.expert_scale_mod
        )
        if require_constexpr(VIA_LDS):
            ratio: gl.constexpr = (
                cfg.scale_tile_ratio_a()
                if operand == 0 and cfg.scale_shuffled(0)
                else 1
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
                _buffer_load_to_lds(
                    lds_ptr.index(
                        BUFFER_LOAD_IDX * (cfg.num_lds_tiles(operand) // ratio)
                        + tile // ratio
                    ),
                    hbm_ptr,
                    hbm_offs,
                    cache,
                    SOFF,
                )
        else:
            gl.static_assert(self.func_cfg.has_scale(operand))
            gl.static_assert(not cfg.scale_via_lds(operand))
            if require_constexpr(cfg.scale_packed_k128(operand)):
                # Both operands must expose the same logical scale factor to the
                # packed MFMA, even when only one of them bypasses LDS.
                scale = gl.amd.cdna4.buffer_load(
                    ptr=hbm_ptr.to(gl.pointer_type(gl.int32)),
                    offsets=hbm_offs // 4,
                    cache=cache,
                    soffset=SOFF,
                )
            elif require_constexpr(cfg.scale_shuffled(operand)):
                scale = gl.convert_layout(
                    gl.amd.cdna4.buffer_load(
                        ptr=hbm_ptr,
                        offsets=hbm_offs,
                        cache=cache,
                        contiguity=4,
                        soffset=SOFF,
                    ),
                    cfg.dot_operand_scale_fragment_layout(operand),
                )
            else:
                scale = gl.amd.cdna4.buffer_load(
                    ptr=hbm_ptr, offsets=hbm_offs, cache=cache, soffset=SOFF
                )
            width: gl.constexpr = cfg.MINI_BLOCK_K // MX_GROUP
            fragments = ()
            for mini in gl.static_range(cfg.num_mini_k()):
                if require_constexpr(cfg.num_mini_k() > 1):
                    fragment = gl.amd.slice(
                        scale,
                        [cfg.scale_shape(operand)[0], width],
                        [0, mini * width],
                    )
                else:
                    fragment = scale
                fragments = fragments + (fragment,)
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
        k_dim: gl.constexpr,
        pack: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        tile_lds_ptr = lds_ptr.index(DS_READ_IDX * n_tiles + tile)
        if require_constexpr(cfg.num_mini_k() > 1):
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
        if require_constexpr(cfg.num_mini_k() > 1):
            width: gl.constexpr = cfg.MINI_BLOCK_K // MX_GROUP
            tile_lds_ptr = tile_lds_ptr.slice(mini_idx * width, width, dim=1)
        return tile_lds_ptr

    @gluon.jit
    def _a_scale_tile(self, DS_READ_IDX, mi: gl.constexpr):
        """Mini block ``mi``'s slice of its A-scale tile, as a flat uint8 descriptor.

        The tile is one 256 B run per 32-row stripe and stripes are contiguous, so a
        mini block is a contiguous stripe range -- exactly the property that lets the
        fill tile be wider than the payload mini block.
        """
        cfg: gl.constexpr = self.tuning_cfg
        RA: gl.constexpr = cfg.scale_tile_ratio_a() if cfg.scale_shuffled(0) else 1
        tile_lds_ptr = self.a_scale_lds_ptr.index(
            DS_READ_IDX * (cfg.num_lds_tiles(0) // RA) + mi // RA
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
        """Read one A (0) or B (1) mini-M/N, mini-K fragment from LDS.

        An absent or disabled component returns None.
        """
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
                    cfg.num_lds_tiles(operand),
                    tile,
                    mini_idx,
                    1 - operand,
                    self.func_cfg.pack_divisor(operand),
                ),
                cfg.dot_operand_fragment_layout(operand),
            )
        if require_constexpr(READ_SCALE and self.func_cfg.has_scale(operand)):
            gl.static_assert(
                cfg.scale_via_lds(operand),
                "ds_read_frag only reads LDS-staged scales",
            )
            if require_constexpr(cfg.scale_shuffled(operand)):
                if require_constexpr(operand == 0):
                    scale_tile_lds_ptr = self._a_scale_tile(SCALE_READ_IDX, tile)
                else:
                    scale_tile_lds_ptr = scale_lds_ptr.index(
                        SCALE_READ_IDX * cfg.num_lds_tiles(operand) + tile
                    )
            if require_constexpr(
                cfg.scale_packed_ok(operand)
                and not (not cfg.FROZEN_STEP and cfg.num_mini_k() > 1)
            ):
                # Straight to i32: the dword the shuffle assembled is the operand the
                # matrix instruction wants, so there is nothing left to do to it.
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
                        cfg.scale_shape(operand),
                        cfg.shuffled_scale_read_layout(operand),
                    ),
                    cfg.dot_operand_scale_fragment_layout(operand),
                )
                if require_constexpr(not cfg.FROZEN_STEP and cfg.num_mini_k() > 1):
                    scale_val = gl.amd.slice(
                        scale_val,
                        [cfg.scale_shape(operand)[0], cfg.MINI_BLOCK_K // MX_GROUP],
                        [0, mini_idx * (cfg.MINI_BLOCK_K // MX_GROUP)],
                    )
            else:
                scale_val = _ds_read(
                    self._scale_slice(
                        scale_lds_ptr,
                        SCALE_READ_IDX,
                        cfg.num_lds_tiles(operand),
                        tile,
                        mini_idx,
                    ),
                    cfg.dot_operand_scale_fragment_layout(operand),
                )
        else:
            scale_val: gl.constexpr = None
        return payload, scale_val
