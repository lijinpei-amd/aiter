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
    VIA_LDS: gl.constexpr,
    cache: gl.constexpr,
    SOFF: gl.constexpr = 0,
):
    """One tile global->LDS, either direct-to-LDS or through registers.

    ``SOFF`` is a byte offset from ``hbm_ptr`` that is uniform across the block. It rides in
    the buffer op's ``soffset`` (SGPR) field, so unlike folding it into ``hbm_offs`` it
    costs no VGPRs and no per-lane arithmetic. See ``SOFF_UNROLL`` in ``_config.py``:
    it lets one base pointer serve every step of an unrolled body.
    """
    if require_constexpr(VIA_LDS):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            lds_ptr, hbm_ptr, hbm_offs, cache_modifier=cache, soffset=SOFF
        )
    else:
        lds_ptr.store(
            gl.amd.cdna4.buffer_load(
                ptr=hbm_ptr, offsets=hbm_offs, cache=cache, soffset=SOFF
            )
        )


@aggregate
@strip_annotate
class LDSManager:
    """Owns the multi-buffered operand staging area."""

    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig
    a_payload_lds_ptr: gl.shared_memory_descriptor
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
        self.a_payload_lds_ptr = a_payload_lds_ptr
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

        a_payload_lds_ptr = gl.allocate_shared_memory(
            a_ty,
            [NBA * NMA, a_shape[0], a_shape[1]],
            layout=tuning_cfg.dot_operand_lds_layout(0),
        )
        if require_constexpr(not tuning_cfg.B_IN_REG):
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
    def buffer_load_a_payload(
        self,
        BUFFER_LOAD_IDX,
        mi: gl.constexpr,
        a_hbm_ptr,
        a_hbm_offs,
        SOFF: gl.constexpr = 0,
    ):
        """Issue the token copy for mini-M block ``mi`` of a stage.

        The ragged last block of each expert is handled by clamping the gathered row
        index modulo the expert's token count, exactly as the Triton kernel does, so no
        mask is needed here and the store mask alone drops the padded rows.
        """
        cfg: gl.constexpr = self.tuning_cfg
        _buffer_load_to_lds(
            self.a_payload_lds_ptr.index(BUFFER_LOAD_IDX * cfg.num_lds_tiles(0) + mi),
            a_hbm_ptr,
            a_hbm_offs,
            cfg.payload_via_lds(0),
            cfg.token_mod,
            SOFF,
        )

    @gluon.jit
    def buffer_load_a_scale(
        self,
        BUFFER_LOAD_IDX,
        mi: gl.constexpr,
        a_scale_hbm_ptr,
        a_scale_hbm_offs,
        SOFF: gl.constexpr = 0,
    ):
        """Issue the token-scale copy for mini-M block ``mi``, if it owns one."""
        cfg: gl.constexpr = self.tuning_cfg
        RA: gl.constexpr = cfg.scale_tile_ratio_a() if cfg.scale_shuffled(0) else 1
        if require_constexpr(
            self.func_cfg.has_scale(0)
            and cfg.scale_via_lds(0)
            # Only the first mini block of each scale tile issues the copy; the rest
            # read their slice out of it.
            and mi % RA == 0
        ):
            _buffer_load_to_lds(
                self.a_scale_lds_ptr.index(
                    BUFFER_LOAD_IDX * (cfg.num_lds_tiles(0) // RA) + mi // RA
                ),
                a_scale_hbm_ptr,
                a_scale_hbm_offs,
                True,
                cfg.token_scale_mod,
                SOFF,
            )

    @gluon.jit
    def buffer_load_b_payload(
        self,
        BUFFER_LOAD_IDX,
        ni: gl.constexpr,
        b_hbm_ptr,
        b_hbm_offs,
        SOFF: gl.constexpr = 0,
    ):
        """Issue the weight copy for mini-N block ``ni`` of a stage."""
        cfg: gl.constexpr = self.tuning_cfg
        _buffer_load_to_lds(
            self.b_payload_lds_ptr.index(BUFFER_LOAD_IDX * cfg.num_lds_tiles(1) + ni),
            b_hbm_ptr,
            b_hbm_offs,
            cfg.payload_via_lds(1),
            cfg.expert_mod,
            SOFF,
        )

    @gluon.jit
    def buffer_load_b_scale(
        self,
        BUFFER_LOAD_IDX,
        ni: gl.constexpr,
        b_scale_hbm_ptr,
        b_scale_hbm_offs,
        SOFF: gl.constexpr = 0,
    ):
        """Issue the weight-scale copy for mini-N block ``ni``, if it owns one."""
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(self.func_cfg.has_scale(1) and cfg.scale_via_lds(1)):
            _buffer_load_to_lds(
                self.b_scale_lds_ptr.index(BUFFER_LOAD_IDX * cfg.num_lds_tiles(1) + ni),
                b_scale_hbm_ptr,
                b_scale_hbm_offs,
                True,
                cfg.expert_scale_mod,
                SOFF,
            )

    @gluon.jit
    def buffer_load_b_register(self, b_hbm_ptr, b_hbm_offs, SOFF: gl.constexpr = 0):
        """Load a preshuffled B stage directly into its MFMA operand registers.

        The caller issues this at the B payload fill slot. The complete stage is
        loaded there, and the mini-K views only select registers from that load.
        """
        cfg: gl.constexpr = self.tuning_cfg
        gl.static_assert(cfg.B_IN_REG and cfg.B_PRESHUFFLED)
        payload = gl.amd.cdna4.buffer_load(
            ptr=b_hbm_ptr,
            offsets=b_hbm_offs,
            cache=cfg.expert_mod,
            soffset=SOFF,
        )
        width: gl.constexpr = cfg.MINI_BLOCK_K // self.func_cfg.pack_divisor(1)
        fragments = ()
        for mini in gl.static_range(cfg.num_mini_k()):
            if require_constexpr(cfg.num_mini_k() > 1):
                fragment = gl.amd.slice(
                    payload, [width, cfg.MINI_BLOCK_N], [mini * width, 0]
                )
            else:
                fragment = payload
            fragments = fragments + (fragment,)
        return fragments

    @gluon.jit
    def buffer_load_scale_register(
        self,
        operand: gl.constexpr,
        hbm_ptr,
        hbm_offs,
        SOFF: gl.constexpr = 0,
        K_PHASE=0,
    ):
        """Load one scale stage into registers and return its mini-K fragments.

        Offsets describe the logical scale grid, including its preshuffle when
        present. K128 stages return the complete packed K256 word, matching the
        LDS path; the consumer selects its K half after reading the register ring.
        """
        cfg: gl.constexpr = self.tuning_cfg
        gl.static_assert(self.func_cfg.has_scale(operand))
        gl.static_assert(not cfg.scale_via_lds(operand))
        cache: gl.constexpr = (
            cfg.token_scale_mod if operand == 0 else cfg.expert_scale_mod
        )
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
    def ds_read_a_frag(
        self,
        DS_READ_IDX,
        mi: gl.constexpr,
        mini_idx: gl.constexpr,
        a_scale_hbm_ptr,
        a_scale_hbm_offs,
        READ_PAYLOAD: gl.constexpr = True,
        READ_SCALE: gl.constexpr = True,
        SCALE_READ_IDX=None,
    ):
        """One operand-A fragment (plus its scale): mini-M block ``mi``, mini-K ``mini_idx``.

        ``a_scale_val`` is None when the token dtype carries no scale. When the A-scale
        tile is too small to be written to LDS coalesced it is loaded straight from
        global into the scale fragment layout; ``a_scale_hbm_offs`` must then already point
        at this mini-M block and this mini-K step.
        """
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(SCALE_READ_IDX is None):
            SCALE_READ_IDX = DS_READ_IDX
        if require_constexpr(not READ_PAYLOAD):
            a_val: gl.constexpr = None
        else:
            a_val = _ds_read(
                self._payload_slice(
                    self.a_payload_lds_ptr,
                    DS_READ_IDX,
                    cfg.num_lds_tiles(0),
                    mi,
                    mini_idx,
                    1,
                    self.func_cfg.pack_divisor(0),
                ),
                cfg.dot_operand_fragment_layout(0),
            )
        if require_constexpr(READ_SCALE and self.func_cfg.has_scale(0)):
            if require_constexpr(
                cfg.scale_packed_ok(0)
                and cfg.scale_via_lds(0)
                and not (not cfg.FROZEN_STEP and cfg.num_mini_k() > 1)
            ):
                # Straight to i32: the dword the shuffle assembled is the operand the
                # matrix instruction wants, so there is nothing left to do to it.
                a_scale_val = _ds_read(
                    self._a_scale_tile(SCALE_READ_IDX, mi).reinterpret(
                        gl.int32,
                        cfg.packed_scale_shape(0),
                        cfg.packed_scale_read_layout(0),
                    ),
                    cfg.packed_scale_frag_layout(0),
                )
            elif require_constexpr(cfg.scale_shuffled(0) and cfg.scale_via_lds(0)):
                a_scale_val = _ds_read(
                    self._a_scale_tile(SCALE_READ_IDX, mi).reinterpret(
                        gl.uint8,
                        cfg.scale_shape(0),
                        cfg.shuffled_scale_read_layout(0),
                    ),
                    cfg.dot_operand_scale_fragment_layout(0),
                )
                if require_constexpr(not cfg.FROZEN_STEP and cfg.num_mini_k() > 1):
                    a_scale_val = gl.amd.slice(
                        a_scale_val,
                        [cfg.MINI_BLOCK_M, cfg.MINI_BLOCK_K // MX_GROUP],
                        [0, mini_idx * (cfg.MINI_BLOCK_K // MX_GROUP)],
                    )
            elif require_constexpr(cfg.scale_via_lds(0)):
                a_scale_val = _ds_read(
                    self._scale_slice(
                        self.a_scale_lds_ptr,
                        SCALE_READ_IDX,
                        cfg.num_lds_tiles(0),
                        mi,
                        mini_idx,
                    ),
                    cfg.dot_operand_scale_fragment_layout(0),
                )
            else:
                if require_constexpr(cfg.scale_shuffled(0)):
                    # One dword per lane instead of four ubytes: issue at the
                    # address-ordered permutation so the widened load fills registers
                    # correctly, then renumber into the layout the MFMA wants.
                    a_scale_val = gl.convert_layout(
                        gl.amd.cdna4.buffer_load(
                            ptr=a_scale_hbm_ptr,
                            offsets=a_scale_hbm_offs,
                            cache=cfg.token_scale_mod,
                            contiguity=4,
                        ),
                        cfg.dot_operand_scale_fragment_layout(0),
                    )
                else:
                    a_scale_val = gl.amd.cdna4.buffer_load(
                        ptr=a_scale_hbm_ptr,
                        offsets=a_scale_hbm_offs,
                        cache=cfg.token_scale_mod,
                    )
        else:
            a_scale_val: gl.constexpr = None
        return a_val, a_scale_val

    @gluon.jit
    def ds_read_b_frag(
        self,
        DS_READ_IDX,
        ni: gl.constexpr,
        mini_idx: gl.constexpr,
        b_scale_hbm_ptr,
        b_scale_hbm_offs,
        READ_PAYLOAD: gl.constexpr = True,
        READ_SCALE: gl.constexpr = True,
        SCALE_READ_IDX=None,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(SCALE_READ_IDX is None):
            SCALE_READ_IDX = DS_READ_IDX
        if require_constexpr(not READ_PAYLOAD):
            b_val: gl.constexpr = None
        else:
            b_val = _ds_read(
                self._payload_slice(
                    self.b_payload_lds_ptr,
                    DS_READ_IDX,
                    cfg.num_lds_tiles(1),
                    ni,
                    mini_idx,
                    0,
                    self.func_cfg.pack_divisor(1),
                ),
                cfg.dot_operand_fragment_layout(1),
            )
        if require_constexpr(READ_SCALE and self.func_cfg.has_scale(1)):
            if require_constexpr(
                cfg.scale_packed_ok(1)
                and cfg.scale_via_lds(1)
                and not (not cfg.FROZEN_STEP and cfg.num_mini_k() > 1)
            ):
                b_scale_val = _ds_read(
                    self.b_scale_lds_ptr.index(
                        SCALE_READ_IDX * cfg.num_lds_tiles(1) + ni
                    ).reinterpret(
                        gl.int32,
                        cfg.packed_scale_shape(1),
                        cfg.packed_scale_read_layout(1),
                    ),
                    cfg.packed_scale_frag_layout(1),
                )
            elif require_constexpr(cfg.scale_shuffled(1) and cfg.scale_via_lds(1)):
                b_scale_val = _ds_read(
                    self.b_scale_lds_ptr.index(
                        SCALE_READ_IDX * cfg.num_lds_tiles(1) + ni
                    ).reinterpret(
                        gl.uint8,
                        cfg.scale_shape(1),
                        cfg.shuffled_scale_read_layout(1),
                    ),
                    cfg.dot_operand_scale_fragment_layout(1),
                )
                if require_constexpr(not cfg.FROZEN_STEP and cfg.num_mini_k() > 1):
                    b_scale_val = gl.amd.slice(
                        b_scale_val,
                        [cfg.MINI_BLOCK_N, cfg.MINI_BLOCK_K // MX_GROUP],
                        [0, mini_idx * (cfg.MINI_BLOCK_K // MX_GROUP)],
                    )
            elif require_constexpr(cfg.scale_via_lds(1)):
                b_scale_val = _ds_read(
                    self._scale_slice(
                        self.b_scale_lds_ptr,
                        SCALE_READ_IDX,
                        cfg.num_lds_tiles(1),
                        ni,
                        mini_idx,
                    ),
                    cfg.dot_operand_scale_fragment_layout(1),
                )
            else:
                if require_constexpr(cfg.scale_shuffled(1)):
                    # One dword per lane instead of four ubytes: issue at the
                    # address-ordered permutation so the widened load fills registers
                    # correctly, then renumber into the layout the MFMA wants.
                    b_scale_val = gl.convert_layout(
                        gl.amd.cdna4.buffer_load(
                            ptr=b_scale_hbm_ptr,
                            offsets=b_scale_hbm_offs,
                            cache=cfg.expert_scale_mod,
                            contiguity=4,
                        ),
                        cfg.dot_operand_scale_fragment_layout(1),
                    )
                else:
                    b_scale_val = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_hbm_ptr,
                        offsets=b_scale_hbm_offs,
                        cache=cfg.expert_scale_mod,
                    )
        else:
            b_scale_val: gl.constexpr = None
        return b_val, b_scale_val
