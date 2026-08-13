# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LDS allocation, filling and reading for the gfx950 Gluon MoE GEMMs.

Modelled on ``MQAAsyncKVLoader`` in ``gfx950/attention/fp8_mqa_logits.py``: a static
``alloc`` factory, one ``@gluon.jit`` method per pipeline action, and every layout
derived from the tuning config rather than spelled out at the use site.

Three things about the CDNA4 async-copy contract drive the shape of this API:

* ``wait_group(n)`` counts **committed groups**, not buffers or tensors. The
  abstraction only holds if all loads of a stage are issued back to back followed by
  exactly one ``commit_group()`` -- hence the separate :meth:`commit_fill_lds`. The
  number of loads per stage is dtype dependent (bf16 has no scale tensors, so 2 not 4),
  so it is derived from the func config, never hardcoded.
* There is no counted wait on LDS *stores*; the only synchronisation is the full block
  barrier. So no ``wait_store_res_lds`` exists.
* ``buffer_load_to_shared`` completes in order with ordinary ``load`` / ``store`` /
  ``buffer_load`` / ``buffer_store``, so any register-path global access inside the
  K-loop makes every ``wait_group`` conservative. The one place that still happens is
  the A-scale tile when it is too small to be written coalesced (``BLOCK_M == 16``),
  which is flagged by ``tuning_cfg.scale_via_lds(0)``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._config import KernelFuncConfig, KernelTuningConfig

#: MX group size, as a constexpr global -- @gluon.jit code may only read constexpr
#: globals, and a plain module-level int is rejected outright.
MX_GROUP: gl.constexpr = gl.constexpr(32)

__all__ = ["LDSManager"]


@gluon.jit
def _shared_load(smem, layout: gl.constexpr, RELAXED: gl.constexpr = False):
    """LDS -> register.

    ``load_shared_relaxed`` tags the load ``ttg.amdg.syncedViaAsyncWait`` so the backend
    skips the waits in front of it. That is only safe while nothing writes the buffer
    back; this pipeline refills the buffer it has just consumed, so the default here is
    the plain load and the relaxed form is opt-in.
    """
    if RELAXED:
        out = gl.amd.cdna4.async_copy.load_shared_relaxed(smem, layout)
    else:
        out = smem.load(layout)
    return out


@gluon.jit
def _async_or_reg_fill(smem, ptr, offsets, VIA_LDS: gl.constexpr, cache: gl.constexpr):
    """One tile global->LDS, either direct-to-LDS or through registers."""
    if VIA_LDS:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem, ptr, offsets, cache_modifier=cache
        )
    else:
        smem.store(gl.amd.cdna4.buffer_load(ptr=ptr, offsets=offsets, cache=cache))


@aggregate
@strip_annotate
class LDSManager:
    """Owns the multi-buffered operand staging area.

    Fields annotated ``gl.shared_memory_descriptor`` hold ``gl.constexpr(None)`` when
    the corresponding tensor carries no scale -- an aggregate has fixed fields, and the
    annotation is only a field-name list to Triton.
    """

    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig
    token_buf: gl.shared_memory_descriptor
    token_scale_buf: gl.shared_memory_descriptor
    weight_buf: gl.shared_memory_descriptor
    weight_scale_buf: gl.shared_memory_descriptor

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
        self.token_scale_buf = token_scale_buf
        self.weight_buf = weight_buf
        self.weight_scale_buf = weight_scale_buf

    # ------------------------------------------------------------------ allocation
    @gluon.jit
    def alloc(func_cfg, tuning_cfg):
        """Static factory -- invoked as ``LDSManager.alloc(...)``, no ``self``."""
        NB: gl.constexpr = tuning_cfg.NUM_LDS_BUFFER
        elem_ty: gl.constexpr = func_cfg.operand_elem_ty()
        a_shape: gl.constexpr = tuning_cfg.a_lds_shape()
        b_shape: gl.constexpr = tuning_cfg.b_lds_shape()

        token_buf = gl.allocate_shared_memory(
            elem_ty,
            [NB, a_shape[0], a_shape[1]],
            layout=tuning_cfg.dot_operand_lds_layout(0),
        )
        weight_buf = gl.allocate_shared_memory(
            elem_ty,
            [NB, b_shape[0], b_shape[1]],
            layout=tuning_cfg.dot_operand_lds_layout(1),
        )
        if func_cfg.a_has_scale():
            as_shape: gl.constexpr = tuning_cfg.a_scale_shape()
            token_scale_buf = gl.allocate_shared_memory(
                gl.uint8,
                [NB, as_shape[0], as_shape[1]],
                layout=tuning_cfg.dot_operand_scale_lds_layout(0),
            )
        else:
            token_scale_buf: gl.constexpr = None
        if func_cfg.b_has_scale():
            bs_shape: gl.constexpr = tuning_cfg.b_scale_shape()
            weight_scale_buf = gl.allocate_shared_memory(
                gl.uint8,
                [NB, bs_shape[0], bs_shape[1]],
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

    # ------------------------------------------------------------------ filling
    @gluon.jit
    def fill_a_lds(self, idx, a_ptr, a_offs, a_scale_ptr, a_scale_offs):
        """Issue the token (and token-scale) copies for one pipeline stage.

        The ragged last block of each expert is handled by clamping the gathered row
        index modulo the expert's token count, exactly as the Triton kernel does, so no
        mask is needed here and the store mask alone drops the padded rows.
        """
        cfg: gl.constexpr = self.tuning_cfg
        _async_or_reg_fill(
            self.token_buf.index(idx),
            a_ptr,
            a_offs,
            cfg.payload_via_lds(0),
            cfg.token_mod,
        )
        if self.func_cfg.a_has_scale() and cfg.scale_via_lds(0):
            _async_or_reg_fill(
                self.token_scale_buf.index(idx),
                a_scale_ptr,
                a_scale_offs,
                True,
                cfg.token_scale_mod,
            )

    @gluon.jit
    def fill_b_lds(self, idx, b_ptr, b_offs, b_scale_ptr, b_scale_offs):
        cfg: gl.constexpr = self.tuning_cfg
        _async_or_reg_fill(
            self.weight_buf.index(idx),
            b_ptr,
            b_offs,
            cfg.payload_via_lds(1),
            cfg.expert_mod,
        )
        if self.func_cfg.b_has_scale() and cfg.scale_via_lds(1):
            _async_or_reg_fill(
                self.weight_scale_buf.index(idx),
                b_scale_ptr,
                b_scale_offs,
                True,
                cfg.expert_scale_mod,
            )

    @gluon.jit
    def commit_fill_lds(self):
        """Exactly one commit group per pipeline stage, after BOTH fills."""
        gl.amd.cdna4.async_copy.commit_group()

    @gluon.jit
    def wait_fill_lds_num_buf(self, num_buf: gl.constexpr):
        """Block until at most ``num_buf`` commit groups (== pipeline stages) remain."""
        gl.amd.cdna4.async_copy.wait_group(num_buf)

    # ------------------------------------------------------------------ reading
    @gluon.jit
    def _payload_slice(
        self, buf, idx, mini_idx: gl.constexpr, k_dim: gl.constexpr, pack: gl.constexpr
    ):
        cfg: gl.constexpr = self.tuning_cfg
        NUM_MINI: gl.constexpr = cfg.BLOCK_K // cfg.MINI_BLOCK_K
        sub = buf.index(idx)
        if NUM_MINI > 1:
            width: gl.constexpr = cfg.MINI_BLOCK_K // pack
            sub = sub.slice(mini_idx * width, width, dim=k_dim)
        return sub

    @gluon.jit
    def _scale_slice(self, buf, idx, mini_idx: gl.constexpr):
        cfg: gl.constexpr = self.tuning_cfg
        NUM_MINI: gl.constexpr = cfg.BLOCK_K // cfg.MINI_BLOCK_K
        sub = buf.index(idx)
        if NUM_MINI > 1:
            width: gl.constexpr = cfg.MINI_BLOCK_K // MX_GROUP
            sub = sub.slice(mini_idx * width, width, dim=1)
        return sub

    @gluon.jit
    def load_a_frag(self, idx, mini_idx: gl.constexpr, a_scale_ptr, a_scale_offs):
        """One operand-A fragment (plus its scale) for one mini-K step.

        ``a_scale_val`` is None when the token dtype carries no scale. When the A-scale
        tile is too small to be written to LDS coalesced it is loaded straight from
        global into the scale fragment layout; ``a_scale_offs`` must then already point
        at this mini-K step.
        """
        cfg: gl.constexpr = self.tuning_cfg
        a_val = _shared_load(
            self._payload_slice(
                self.token_buf, idx, mini_idx, 1, self.func_cfg.a_pack_divisor()
            ),
            cfg.dot_operand_fragment_layout(0),
        )
        if self.func_cfg.a_has_scale():
            if cfg.scale_via_lds(0):
                a_scale_val = _shared_load(
                    self._scale_slice(self.token_scale_buf, idx, mini_idx),
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
    def load_b_frag(self, idx, mini_idx: gl.constexpr, b_scale_ptr, b_scale_offs):
        cfg: gl.constexpr = self.tuning_cfg
        b_val = _shared_load(
            self._payload_slice(
                self.weight_buf, idx, mini_idx, 0, self.func_cfg.b_pack_divisor()
            ),
            cfg.dot_operand_fragment_layout(1),
        )
        if self.func_cfg.b_has_scale():
            if cfg.scale_via_lds(1):
                b_scale_val = _shared_load(
                    self._scale_slice(self.weight_scale_buf, idx, mini_idx),
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
