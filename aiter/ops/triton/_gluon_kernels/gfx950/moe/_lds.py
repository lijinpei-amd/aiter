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
from ._layout import KernelLayout
from ._types import A_PAYLOAD, A_SCALE, B_PAYLOAD, B_SCALE

__all__ = ["LDSManager"]

_A_PAYLOAD: gl.constexpr = gl.constexpr(A_PAYLOAD)
_A_SCALE: gl.constexpr = gl.constexpr(A_SCALE)
_B_PAYLOAD: gl.constexpr = gl.constexpr(B_PAYLOAD)
_B_SCALE: gl.constexpr = gl.constexpr(B_SCALE)


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
    """Copy one non-K slot of a component directly from global memory to LDS.

    ``SOFF`` is a byte offset from ``hbm_ptr`` that is uniform across the block. It rides in
    the buffer op's ``soffset`` (SGPR) field, so unlike folding it into ``hbm_offs`` it
    costs no VGPRs and no per-lane arithmetic. See ``SOFF_UNROLL`` in ``_config.py``:
    it lets one base pointer serve every step of an unrolled body.
    """
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        lds_ptr, hbm_ptr, hbm_offs, cache_modifier=cache, soffset=SOFF
    )


@gluon.jit
def _scale_fragments(scale, cfg, layout, operand: gl.constexpr):
    """Split one shuffled scale load in registers, retaining whole packed words."""
    nonk: gl.constexpr = layout.scale_slot_nonk(operand)
    sk: gl.constexpr = layout.scale_slot_k(operand)
    bk: gl.constexpr = cfg.BLOCK_K
    packed: gl.constexpr = layout.scale_packed_ok(operand)
    fragments = ()
    for n in gl.static_range(layout.scale_ratio_non_k_slot(operand)):
        for k in gl.static_range(sk // bk):
            if require_constexpr(packed):
                if require_constexpr(
                    layout.scale_ratio_non_k_slot(operand) == 1 and sk == max(256, bk)
                ):
                    fragment = scale
                else:
                    # Physical order is stripe, K256 group, lane dword. Splitting
                    # these axes never unpacks a dword or moves it across lanes.
                    words = gl.reshape(scale, [nonk // 32, sk // 256, 64])
                    words = gl.amd.slice(
                        words,
                        [layout.slot_nonk(operand) // 32, max(256, bk) // 256, 64],
                        [n * layout.slot_nonk(operand) // 32, k * bk // 256, 0],
                    )
                    fragment = gl.convert_layout(
                        gl.reshape(words, layout.fragment_shape_slot(operand, True)),
                        layout.dot_operand_fragment_layout(operand, True),
                    )
            else:
                fragment = gl.amd.slice(
                    scale,
                    layout.fragment_shape_slot(operand, True),
                    [n * layout.slot_nonk(operand), k * bk // 32],
                )
            fragments += (fragment,)
    return fragments


@aggregate
@strip_annotate
class LDSManager:
    """Owns the multi-buffered operand staging area."""

    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig
    layout: KernelLayout
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
        """Static factory -- invoked as ``LDSManager.alloc(...)``, no ``self``.

        The leading axis runs over pipeline buffer *and* non-K slot
        (``num_m_slots`` for A, ``num_n_slots`` for B), flattened as
        ``buffer_idx * n_slots + non_k_slot``: a shared allocation takes a layout of its
        own rank, so two separate leading axes are not expressible. Each non-K slot is a
        real allocation rather than a slice of one big tile so that each one keeps the
        padded layout ``compute_efficient_padded_shared_layout`` picked *for its own
        shape* -- that is what makes the per-slot direct-to-LDS copy as coalesced and as
        wide as the whole-tile copy was.
        """
        a_ty: gl.constexpr = func_cfg.operand_elem_ty(0)
        b_ty: gl.constexpr = func_cfg.operand_elem_ty(1)

        if require_constexpr(layout.component_via_lds(_A_PAYLOAD)):
            a_payload_lds_ptr = gl.allocate_shared_memory(
                a_ty,
                layout.lds_shape_tile(0, False),
                layout=layout.dot_operand_lds_layout(0, False),
            )
        else:
            a_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(layout.component_via_lds(_B_PAYLOAD)):
            b_payload_lds_ptr = gl.allocate_shared_memory(
                b_ty,
                layout.lds_shape_tile(1, False),
                layout=layout.dot_operand_lds_layout(1, False),
            )
        else:
            b_payload_lds_ptr: gl.constexpr = None
        if require_constexpr(
            func_cfg.has_scale(0) and layout.component_via_lds(_A_SCALE)
        ):
            a_scale_lds_ptr = gl.allocate_shared_memory(
                gl.uint8,
                layout.lds_shape_tile(0, True),
                layout=layout.dot_operand_lds_layout(0, True),
            )
        else:
            a_scale_lds_ptr: gl.constexpr = None
        if require_constexpr(
            func_cfg.has_scale(1) and layout.component_via_lds(_B_SCALE)
        ):
            b_scale_lds_ptr = gl.allocate_shared_memory(
                gl.uint8,
                layout.lds_shape_tile(1, True),
                layout=layout.dot_operand_lds_layout(1, True),
            )
        else:
            b_scale_lds_ptr: gl.constexpr = None

        return LDSManager(
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
        non_k_slot: gl.constexpr,
        hbm_ptr,
        hbm_offs,
        SOFF: gl.constexpr = 0,
    ):
        """Load one non-K slot of the A (0) or B (1) payload into LDS or registers.

        The register path returns the complete BLOCK_K stage as one fragment; its
        offsets must already use the operand fragment layout. The caller owns
        the register ring, so this path never writes the LDS staging allocation.

        Gathered A rows are clamped modulo the expert's token count. No load
        mask is needed: the output store mask drops the padded rows.
        """
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
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
                    BUFFER_LOAD_IDX * layout.num_lds_slots_non_k(operand) + non_k_slot
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
            return (payload,)

    @gluon.jit
    def buffer_load_scale(
        self,
        operand: gl.constexpr,
        VIA_LDS: gl.constexpr,
        BUFFER_LOAD_IDX,
        non_k_slot: gl.constexpr,
        hbm_ptr,
        hbm_offs,
        SOFF: gl.constexpr = 0,
        K_PHASE=0,
    ):
        """Load one non-K slot of a scale, covering one or more whole K steps."""
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        cache: gl.constexpr = (
            cfg.token_scale_cache_modifier
            if operand == 0
            else cfg.expert_scale_cache_modifier
        )
        ratio: gl.constexpr = layout.scale_ratio_non_k_slot(operand)
        if require_constexpr(VIA_LDS):
            if require_constexpr(
                self.func_cfg.has_scale(operand)
                and layout.component_via_lds((operand, True))
                and non_k_slot % ratio == 0
            ):
                if require_constexpr(operand == 0):
                    lds_ptr = self.a_scale_lds_ptr
                else:
                    lds_ptr = self.b_scale_lds_ptr
                _buffer_load_to_lds(
                    lds_ptr.index(
                        BUFFER_LOAD_IDX * layout.num_scale_slots(operand)
                        + non_k_slot // ratio
                    ),
                    hbm_ptr,
                    hbm_offs,
                    cache,
                    SOFF,
                )
        else:
            gl.static_assert(self.func_cfg.has_scale(operand))
            gl.static_assert(layout.component_in_reg((operand, True)))
            if require_constexpr(cfg.scale_shuffled(operand)):
                if require_constexpr(layout.scale_packed_ok(operand)):
                    scale = gl.amd.cdna4.buffer_load(
                        ptr=hbm_ptr.to(gl.pointer_type(gl.int32)),
                        offsets=hbm_offs // 4,
                        cache=cache,
                        soffset=SOFF,
                    )
                else:
                    scale = gl.convert_layout(
                        gl.amd.cdna4.buffer_load(
                            ptr=hbm_ptr,
                            offsets=hbm_offs,
                            cache=cache,
                            contiguity=4,
                            soffset=SOFF,
                        ),
                        layout.scale_fragment_layout_scale_slot(operand),
                    )
                fragments = _scale_fragments(scale, cfg, layout, operand)
            else:
                scale = gl.amd.cdna4.buffer_load(
                    ptr=hbm_ptr, offsets=hbm_offs, cache=cache, soffset=SOFF
                )
                fragments = (scale,)
            return fragments

    @gluon.jit
    def ds_read_scale(
        self, operand: gl.constexpr, SCALE_READ_IDX, non_k_slot: gl.constexpr
    ):
        """Read each scale slot once and retain all of its consumer fragments."""
        cfg: gl.constexpr = self.tuning_cfg
        layout: gl.constexpr = self.layout
        if require_constexpr(operand == 0):
            lds_ptr = self.a_scale_lds_ptr
        else:
            lds_ptr = self.b_scale_lds_ptr
        if require_constexpr(cfg.scale_shuffled(operand)):
            scale_ptr = lds_ptr.index(
                SCALE_READ_IDX * layout.num_scale_slots(operand)
                + non_k_slot // layout.scale_ratio_non_k_slot(operand)
            )
            scale = _ds_read(
                scale_ptr.reinterpret(
                    layout.scale_fragment_dtype(operand),
                    layout.scale_fragment_shape_scale_slot(operand),
                    layout.scale_lds_view_layout_scale_slot(operand),
                ),
                layout.scale_fragment_layout_scale_slot(operand),
            )
            fragments = _scale_fragments(scale, cfg, layout, operand)
        else:
            fragments = (
                _ds_read(
                    lds_ptr.index(
                        SCALE_READ_IDX * layout.num_scale_slots(operand) + non_k_slot
                    ),
                    layout.dot_operand_fragment_layout(operand, True),
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
    def ds_read_frag(
        self,
        operand: gl.constexpr,
        DS_READ_IDX,
        non_k_slot: gl.constexpr,
        k_slot: gl.constexpr,
        READ_PAYLOAD: gl.constexpr = True,
        READ_SCALE: gl.constexpr = True,
        SCALE_READ_IDX=None,
    ):
        """Read K slot ``k_slot`` of one non-K slot of A (0) or B (1) from LDS.

        An absent or disabled component returns None.
        """
        layout: gl.constexpr = self.layout
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
                payload_lds_ptr.index(
                    DS_READ_IDX * layout.num_lds_slots_non_k(operand) + non_k_slot
                ),
                layout.dot_operand_fragment_layout(operand, False),
            )
        if require_constexpr(READ_SCALE and self.func_cfg.has_scale(operand)):
            gl.static_assert(
                layout.component_via_lds((operand, True)),
                "ds_read_frag only reads LDS-staged scales",
            )
            ratio: gl.constexpr = layout.scale_ratio_non_k_slot(operand)
            fragments = self.ds_read_scale(
                operand, SCALE_READ_IDX, non_k_slot // ratio * ratio
            )
            scale_val = fragments[
                (non_k_slot % ratio) * layout.scale_read_k_slots(operand) + k_slot
            ]
        else:
            scale_val: gl.constexpr = None
        return payload, scale_val
