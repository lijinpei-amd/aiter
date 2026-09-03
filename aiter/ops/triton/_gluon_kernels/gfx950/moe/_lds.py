# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LDS allocation, filling and reading for the gfx950 Gluon MoE GEMMs."""

import os

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig
from ._lang import MX_GROUP_CE as MX_GROUP
from ._lang import optional as _opt
from ._lang import require_constexpr

__all__ = ["LDSManager"]

#: EXPERIMENT ONLY -- replace every E8M0 scale operand with a constant 1.0 (E8M0 bias
#: 127) instead of reading it. Produces wrong results; it exists to measure the ceiling
#: of removing the scale path (the 16 ds_read_u8 + 8 v_perm_b32 per pipeline step, and
#: whatever of the scale global->LDS copy then dies as unreachable).
_CONST_SCALE: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_CONST_SCALE", "0"))
)

#: EXPERIMENT ONLY -- same, for the payload operands: replace the 16 ds_read_b128 per
#: pipeline step with a constant. Together with _CONST_SCALE this strips the whole
#: LDS->register path, leaving only the global->LDS copies and the MFMAs.
_CONST_AB: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_CONST_AB", "0"))
)

#: EXPERIMENT ONLY -- drop the global->LDS copies while keeping their commit groups, so
#: the wait/barrier structure is untouched and only the vmem traffic disappears. The
#: LDS then holds garbage; this exists to price the global side.
_NO_FILL: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_NO_FILL", "0"))
)

#: EXPERIMENT ONLY -- take the ``load_shared_relaxed`` path for every LDS read, which
#: tags them ``syncedViaAsyncWait`` so the backend drops the ``lgkmcnt`` drain in front
#: of each barrier. RACY for this pipeline: step k reads buffer (k+1)%NB and step k+1
#: refills it, so the drain is exactly what closes that WAR window. Timing only.
_RELAXED_LDS: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_RELAXED_LDS", "0"))
)

#: EXPERIMENT ONLY -- drop just the E8M0 scale copies, keeping the payload ones. The
#: scale tile is 8 bytes per row at a K/32 = 224 B row stride, i.e. far below a cache
#: line, so it is the prime suspect for the L2 request amplification.
_NO_SCALE_FILL: gl.constexpr = gl.constexpr(
    int(os.environ.get("AITER_TRITON_MOE_GLUON_NO_SCALE_FILL", "0"))
)


@gluon.jit
def _shared_load(smem, layout: gl.constexpr, RELAXED: gl.constexpr = False):
    """LDS -> register.

    ``load_shared_relaxed`` tags the load ``ttg.amdg.syncedViaAsyncWait`` so the backend
    skips the waits in front of it. That is only safe while nothing writes the buffer
    back; this pipeline refills the buffer it has just consumed, so the default here is
    the plain load and the relaxed form is opt-in.
    """
    if require_constexpr(RELAXED or _RELAXED_LDS):
        out = gl.amd.cdna4.async_copy.load_shared_relaxed(smem, layout)
    else:
        out = smem.load(layout)
    return out


@gluon.jit
def _async_or_reg_fill(
    smem, ptr, offsets, VIA_LDS: gl.constexpr, cache: gl.constexpr,
    SOFF: gl.constexpr = 0,
):
    """One tile global->LDS, either direct-to-LDS or through registers.

    ``SOFF`` is a byte offset from ``ptr`` that is uniform across the block. It rides in
    the buffer op's ``soffset`` (SGPR) field, so unlike folding it into ``offsets`` it
    costs no VGPRs and no per-lane arithmetic. See ``_SOFF_UNROLL`` in ``moe_gemm.py``:
    it lets one base pointer serve every step of an unrolled body.
    """
    if require_constexpr(VIA_LDS):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem, ptr, offsets, cache_modifier=cache, soffset=SOFF
        )
    else:
        smem.store(
            gl.amd.cdna4.buffer_load(
                ptr=ptr, offsets=offsets, cache=cache, soffset=SOFF
            )
        )


@aggregate
@strip_annotate
class LDSManager:
    """Owns the multi-buffered operand staging area."""

    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig
    token_buf: gl.shared_memory_descriptor
    token_scale_buf: gl.shared_memory_descriptor | gl.constexpr
    weight_buf: gl.shared_memory_descriptor
    weight_scale_buf: gl.shared_memory_descriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        func_cfg,
        tuning_cfg,
        token_buf,
        token_scale_buf,
        weight_buf,
        weight_scale_buf,
    ):
        self.func_cfg = func_cfg
        self.tuning_cfg = tuning_cfg
        self.token_buf = token_buf
        self.token_scale_buf = _opt(token_scale_buf)
        self.weight_buf = weight_buf
        self.weight_scale_buf = _opt(weight_scale_buf)

    @gluon.jit
    def alloc(func_cfg, tuning_cfg):
        """Static factory -- invoked as ``LDSManager.alloc(...)``, no ``self``.

        The leading axis runs over pipeline buffer *and* mini block
        (``num_mini_m()`` tiles for A, ``num_mini_n()`` for B), flattened as
        ``buf * n_tiles + tile``: a shared allocation takes a layout of the tile's own
        rank, so two separate leading axes are not expressible. The mini block is a real
        allocation rather than a slice of one big tile so that each one keeps the padded
        layout ``compute_efficient_padded_shared_layout`` picked *for its own shape* --
        that is what makes the per-mini-block direct-to-LDS copy as coalesced and as wide
        as the whole-tile copy was.
        """
        NB: gl.constexpr = tuning_cfg.NUM_LDS_BUFFER
        NMA: gl.constexpr = tuning_cfg.num_lds_tiles(0)
        NMB: gl.constexpr = tuning_cfg.num_lds_tiles(1)
        a_ty: gl.constexpr = func_cfg.operand_elem_ty(0)
        b_ty: gl.constexpr = func_cfg.operand_elem_ty(1)
        a_shape: gl.constexpr = tuning_cfg.lds_shape(0)
        b_shape: gl.constexpr = tuning_cfg.lds_shape(1)

        token_buf = gl.allocate_shared_memory(
            a_ty,
            [NB * NMA, a_shape[0], a_shape[1]],
            layout=tuning_cfg.dot_operand_lds_layout(0),
        )
        if require_constexpr(tuning_cfg.B_IN_REG):
            # Nothing reads this, but the field is typed and the padded dot-operand
            # layout only legalises against the real tile shape -- so keep the shape and
            # drop the buffering, which is where the NUM_LDS_BUFFER-fold saving is.
            weight_buf = gl.allocate_shared_memory(
                b_ty,
                [1, b_shape[0], b_shape[1]],
                layout=tuning_cfg.dot_operand_lds_layout(1),
            )
        else:
            weight_buf = gl.allocate_shared_memory(
                b_ty,
                [NB * NMB, b_shape[0], b_shape[1]],
                layout=tuning_cfg.dot_operand_lds_layout(1),
            )
        if require_constexpr(func_cfg.has_scale(0)):
            as_shape: gl.constexpr = tuning_cfg.scale_shape(0)
            if require_constexpr(tuning_cfg.scale_shuffled(0)):
                # Flat: direct-to-LDS on gfx9 cannot scatter, so the staging tile has to
                # be written coalesced; the fragment view comes back on the read.
                # A-scale tiles are counted separately from the payload mini blocks:
                # scale_mini_m() may cover several of them so the copy's stripe count
                # matches warps_per_cta and no warp is replicated.
                token_scale_buf = gl.allocate_shared_memory(
                    gl.uint8,
                    [NB * tuning_cfg.num_scale_tiles_a()]
                    + tuning_cfg.scale_flat_shape(0),
                    layout=tuning_cfg.dot_operand_scale_lds_layout(0),
                )
            else:
                token_scale_buf = gl.allocate_shared_memory(
                    gl.uint8,
                    [NB * NMA, as_shape[0], as_shape[1]],
                    layout=tuning_cfg.dot_operand_scale_lds_layout(0),
                )
        else:
            token_scale_buf: gl.constexpr = None
        if require_constexpr(func_cfg.has_scale(1)):
            bs_shape: gl.constexpr = tuning_cfg.scale_shape(1)
            if require_constexpr(tuning_cfg.scale_shuffled(1)):
                # Flat: direct-to-LDS on gfx9 cannot scatter, so the staging tile has to
                # be written coalesced; the fragment view comes back on the read.
                weight_scale_buf = gl.allocate_shared_memory(
                    gl.uint8,
                    [NB * NMB] + tuning_cfg.scale_flat_shape(1),
                    layout=tuning_cfg.dot_operand_scale_lds_layout(1),
                )
            else:
                weight_scale_buf = gl.allocate_shared_memory(
                    gl.uint8,
                    [NB * NMB, bs_shape[0], bs_shape[1]],
                    layout=tuning_cfg.dot_operand_scale_lds_layout(1),
                )
        else:
            weight_scale_buf: gl.constexpr = None

        return LDSManager(
            func_cfg,
            tuning_cfg,
            token_buf,
            token_scale_buf,
            weight_buf,
            weight_scale_buf,
        )

    @gluon.jit
    def fill_a_payload_lds(self, idx, mi: gl.constexpr, a_ptr, a_offs, SOFF: gl.constexpr = 0):
        """Issue the token copy for mini-M block ``mi`` of a stage.

        The ragged last block of each expert is handled by clamping the gathered row
        index modulo the expert's token count, exactly as the Triton kernel does, so no
        mask is needed here and the store mask alone drops the padded rows.
        """
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(not _NO_FILL):
            _async_or_reg_fill(
                self.token_buf.index(idx * cfg.num_lds_tiles(0) + mi),
                a_ptr,
                a_offs,
                cfg.payload_via_lds(0),
                cfg.token_mod,
                SOFF,
            )

    @gluon.jit
    def fill_a_scale_lds(self, idx, mi: gl.constexpr, a_scale_ptr, a_scale_offs, SOFF: gl.constexpr = 0):
        """Issue the token-scale copy for mini-M block ``mi``, if it owns one."""
        cfg: gl.constexpr = self.tuning_cfg
        RA: gl.constexpr = cfg.scale_tile_ratio_a() if cfg.scale_shuffled(0) else 1
        if require_constexpr(
            not _NO_FILL
            and not _NO_SCALE_FILL
            and self.func_cfg.has_scale(0)
            and cfg.scale_via_lds(0)
            # Only the first mini block of each scale tile issues the copy; the rest
            # read their slice out of it.
            and mi % RA == 0
        ):
            _async_or_reg_fill(
                self.token_scale_buf.index(
                    idx * (cfg.num_lds_tiles(0) // RA) + mi // RA
                ),
                a_scale_ptr,
                a_scale_offs,
                True,
                cfg.token_scale_mod,
                SOFF,
            )

    @gluon.jit
    def fill_b_payload_lds(self, idx, ni: gl.constexpr, b_ptr, b_offs, SOFF: gl.constexpr = 0):
        """Issue the weight copy for mini-N block ``ni`` of a stage."""
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(not _NO_FILL and not cfg.B_IN_REG):
            _async_or_reg_fill(
                self.weight_buf.index(idx * cfg.num_lds_tiles(1) + ni),
                b_ptr,
                b_offs,
                cfg.payload_via_lds(1),
                cfg.expert_mod,
                SOFF,
            )

    @gluon.jit
    def fill_b_scale_lds(self, idx, ni: gl.constexpr, b_scale_ptr, b_scale_offs, SOFF: gl.constexpr = 0):
        """Issue the weight-scale copy for mini-N block ``ni``, if it owns one."""
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(
            not _NO_FILL
            and not _NO_SCALE_FILL
            and self.func_cfg.has_scale(1)
            and cfg.scale_via_lds(1)
        ):
            _async_or_reg_fill(
                self.weight_scale_buf.index(idx * cfg.num_lds_tiles(1) + ni),
                b_scale_ptr,
                b_scale_offs,
                True,
                cfg.expert_scale_mod,
                SOFF,
            )

    @gluon.jit
    def commit_fill_lds(self):
        """One commit group per mini-block fill.

        Finer than one group per stage on purpose: a slot only ever reads one mini
        block, so a per-mini-block group lets its ``wait_group`` name exactly that copy
        instead of the whole stage's. That is what makes a per-slot wait -- and with it
        the warp-pipelined mfma/mem interleave -- expressible at all.
        """
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def wait_fill_lds_num_group(self, num_group: gl.constexpr):
        """Block until at most ``num_group`` commit groups remain outstanding."""
        gl.amd.cdna4.async_copy.wait_group(num_group)

    @gluon.jit
    def _payload_slice(
        self,
        buf,
        idx,
        n_tiles: gl.constexpr,
        tile: gl.constexpr,
        mini_idx: gl.constexpr,
        k_dim: gl.constexpr,
        pack: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        sub = buf.index(idx * n_tiles + tile)
        if require_constexpr(cfg.num_mini_k() > 1):
            width: gl.constexpr = cfg.MINI_BLOCK_K // pack
            sub = sub.slice(mini_idx * width, width, dim=k_dim)
        return sub

    @gluon.jit
    def _scale_slice(
        self,
        buf,
        idx,
        n_tiles: gl.constexpr,
        tile: gl.constexpr,
        mini_idx: gl.constexpr,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        sub = buf.index(idx * n_tiles + tile)
        if require_constexpr(cfg.num_mini_k() > 1):
            width: gl.constexpr = cfg.MINI_BLOCK_K // MX_GROUP
            sub = sub.slice(mini_idx * width, width, dim=1)
        return sub

    @gluon.jit
    def _a_scale_tile(self, idx, mi: gl.constexpr):
        """Mini block ``mi``'s slice of its A-scale tile, as a flat uint8 descriptor.

        The tile is one 256 B run per 32-row stripe and stripes are contiguous, so a
        mini block is a contiguous stripe range -- exactly the property that lets the
        fill tile be wider than the payload mini block.
        """
        cfg: gl.constexpr = self.tuning_cfg
        RA: gl.constexpr = cfg.scale_tile_ratio_a() if cfg.scale_shuffled(0) else 1
        sub = self.token_scale_buf.index(idx * (cfg.num_lds_tiles(0) // RA) + mi // RA)
        if require_constexpr(RA > 1):
            stripes: gl.constexpr = cfg.MINI_BLOCK_M // 32
            sub = sub.slice((mi % RA) * stripes, stripes, dim=0)
        return sub

    @gluon.jit
    def load_a_frag(
        self,
        idx,
        mi: gl.constexpr,
        mini_idx: gl.constexpr,
        a_scale_ptr,
        a_scale_offs,
        RELAXED: gl.constexpr = False,
    ):
        """One operand-A fragment (plus its scale): mini-M block ``mi``, mini-K ``mini_idx``.

        ``a_scale_val`` is None when the token dtype carries no scale. When the A-scale
        tile is too small to be written to LDS coalesced it is loaded straight from
        global into the scale fragment layout; ``a_scale_offs`` must then already point
        at this mini-M block and this mini-K step.
        """
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(_CONST_AB):
            a_val = gl.full(
                cfg.lds_shape(0),
                1,
                self.func_cfg.operand_elem_ty(0),
                layout=cfg.dot_operand_fragment_layout(0),
            )
        else:
            a_val = _shared_load(
                self._payload_slice(
                    self.token_buf,
                    idx,
                    cfg.num_lds_tiles(0),
                    mi,
                    mini_idx,
                    1,
                    self.func_cfg.pack_divisor(0),
                ),
                cfg.dot_operand_fragment_layout(0),
                RELAXED,
            )
        if require_constexpr(self.func_cfg.has_scale(0)):
            if require_constexpr(_CONST_SCALE):
                a_scale_val = gl.full(
                    cfg.scale_shape(0),
                    127,
                    gl.uint8,
                    layout=cfg.dot_operand_scale_fragment_layout(0),
                )
            elif require_constexpr(cfg.scale_packed_ok(0) and cfg.scale_via_lds(0)):
                # Straight to i32: the dword the shuffle assembled is the operand the
                # matrix instruction wants, so there is nothing left to do to it.
                a_scale_val = _shared_load(
                    self._a_scale_tile(idx, mi).reinterpret(
                        gl.int32,
                        cfg.packed_scale_shape(0),
                        cfg.packed_scale_read_layout(0),
                    ),
                    cfg.packed_scale_frag_layout(0),
                    RELAXED,
                )
            elif require_constexpr(cfg.scale_shuffled(0) and cfg.scale_via_lds(0)):
                a_scale_val = _shared_load(
                    self._a_scale_tile(idx, mi).reinterpret(
                        gl.uint8,
                        cfg.scale_shape(0),
                        cfg.shuffled_scale_read_layout(0),
                    ),
                    cfg.dot_operand_scale_fragment_layout(0),
                    RELAXED,
                )
            elif require_constexpr(cfg.scale_via_lds(0)):
                a_scale_val = _shared_load(
                    self._scale_slice(
                        self.token_scale_buf, idx, cfg.num_lds_tiles(0), mi, mini_idx
                    ),
                    cfg.dot_operand_scale_fragment_layout(0),
                    RELAXED,
                )
            else:
                if require_constexpr(cfg.scale_shuffled(0)):
                    # One dword per lane instead of four ubytes: issue at the
                    # address-ordered permutation so the widened load fills registers
                    # correctly, then renumber into the layout the MFMA wants.
                    a_scale_val = gl.convert_layout(
                        gl.amd.cdna4.buffer_load(
                            ptr=a_scale_ptr,
                            offsets=a_scale_offs,
                            cache=cfg.token_scale_mod,
                            contiguity=4,
                        ),
                        cfg.dot_operand_scale_fragment_layout(0),
                    )
                else:
                    a_scale_val = gl.amd.cdna4.buffer_load(
                        ptr=a_scale_ptr,
                        offsets=a_scale_offs,
                        cache=cfg.token_scale_mod,
                    )
        else:
            a_scale_val: gl.constexpr = None
        return a_val, a_scale_val

    @gluon.jit
    def load_b_frag(
        self,
        idx,
        ni: gl.constexpr,
        mini_idx: gl.constexpr,
        b_scale_ptr,
        b_scale_offs,
        b_ptr=None,
        b_frag_offs=None,
        RELAXED: gl.constexpr = False,
    ):
        cfg: gl.constexpr = self.tuning_cfg
        if require_constexpr(_CONST_AB):
            b_val = gl.full(
                cfg.lds_shape(1),
                1,
                self.func_cfg.operand_elem_ty(1),
                layout=cfg.dot_operand_fragment_layout(1),
            )
        elif require_constexpr(cfg.B_IN_REG):
            # Global -> VGPR in one step, no LDS round trip. The operand-B fragment
            # layout is K-contiguous and K is the fastest axis of the weight tensor, so
            # each lane's 32 packed elements are 16 contiguous bytes: one
            # buffer_load_dwordx4, which is what FlyDSL's BufferCopy128b emits.
            b_val = gl.amd.cdna4.buffer_load(
                ptr=b_ptr, offsets=b_frag_offs, cache=cfg.expert_mod
            )
        else:
            b_val = _shared_load(
                self._payload_slice(
                    self.weight_buf,
                    idx,
                    cfg.num_lds_tiles(1),
                    ni,
                    mini_idx,
                    0,
                    self.func_cfg.pack_divisor(1),
                ),
                cfg.dot_operand_fragment_layout(1),
                RELAXED,
            )
        if require_constexpr(self.func_cfg.has_scale(1)):
            if require_constexpr(_CONST_SCALE):
                b_scale_val = gl.full(
                    cfg.scale_shape(1),
                    127,
                    gl.uint8,
                    layout=cfg.dot_operand_scale_fragment_layout(1),
                )
            elif require_constexpr(cfg.scale_packed_ok(1) and cfg.scale_via_lds(1)):
                b_scale_val = _shared_load(
                    self.weight_scale_buf.index(
                        idx * cfg.num_lds_tiles(1) + ni
                    ).reinterpret(
                        gl.int32,
                        cfg.packed_scale_shape(1),
                        cfg.packed_scale_read_layout(1),
                    ),
                    cfg.packed_scale_frag_layout(1),
                    RELAXED,
                )
            elif require_constexpr(cfg.scale_shuffled(1) and cfg.scale_via_lds(1)):
                b_scale_val = _shared_load(
                    self.weight_scale_buf.index(
                        idx * cfg.num_lds_tiles(1) + ni
                    ).reinterpret(
                        gl.uint8,
                        cfg.scale_shape(1),
                        cfg.shuffled_scale_read_layout(1),
                    ),
                    cfg.dot_operand_scale_fragment_layout(1),
                    RELAXED,
                )
            elif require_constexpr(cfg.scale_via_lds(1)):
                b_scale_val = _shared_load(
                    self._scale_slice(
                        self.weight_scale_buf, idx, cfg.num_lds_tiles(1), ni, mini_idx
                    ),
                    cfg.dot_operand_scale_fragment_layout(1),
                    RELAXED,
                )
            else:
                if require_constexpr(cfg.scale_shuffled(1)):
                    # One dword per lane instead of four ubytes: issue at the
                    # address-ordered permutation so the widened load fills registers
                    # correctly, then renumber into the layout the MFMA wants.
                    b_scale_val = gl.convert_layout(
                        gl.amd.cdna4.buffer_load(
                            ptr=b_scale_ptr,
                            offsets=b_scale_offs,
                            cache=cfg.expert_scale_mod,
                            contiguity=4,
                        ),
                        cfg.dot_operand_scale_fragment_layout(1),
                    )
                else:
                    b_scale_val = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_ptr,
                        offsets=b_scale_offs,
                        cache=cfg.expert_scale_mod,
                    )
        else:
            b_scale_val: gl.constexpr = None
        return b_val, b_scale_val
