# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Functionality and tuning configuration aggregates for the gfx950 Gluon MoE GEMMs.

Both are ``@gluon.aggregate``s built entirely out of ``gl.constexpr`` scalars, so the
*same* object can be constructed on the host (for the grid tuple) and inside the kernel
(for the layouts) from the same numbers -- an aggregate itself cannot be a launch
argument, only its class can. Keeping the arithmetic in the aggregate is what stops the
host grid math and the device tile math from drifting.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import (
    DotKind,
    DtypeQuant,
    EpilogueMode,
    SchedMode,
    WaitCommitScheme,
    WarpPipeline,
    dq_has_scale,
    dq_mx_format,
    dq_pack_divisor,
    dq_uses_mfma_scaled,
    strip_annotate,
)
from ._types import WeightGuHbmLayout

__all__ = ["KernelFuncConfig", "KernelTuningConfig"]


@aggregate
@strip_annotate
class KernelFuncConfig:
    """What the kernel *computes*. Every field is constexpr; optional behaviour is a
    ``None`` sentinel because an aggregate has fixed fields -- "does not exist for
    gemm2" is not expressible, ``activation is None`` is.
    """

    # token / expert dtype and quant-scheme as stored in HBM
    token_dtype_quant: gl.constexpr
    expert_dtype_quant: gl.constexpr
    # MMA operand-A / operand-B representation. v1: format selection only, no in-kernel
    # requantisation, so these equal the stored formats. Selects mfma vs mfma_scaled.
    token_online_quant: gl.constexpr
    expert_online_quant: gl.constexpr
    mma_acc_dtype: gl.constexpr
    activation: gl.constexpr  # ActivationSpec | None
    output_quant: gl.constexpr  # DtypeQuant | None
    has_bias: gl.constexpr
    has_gammas: gl.constexpr
    has_gather: gl.constexpr
    # per-tensor fp8 activation scale, applied to the raw accumulator before bias --
    # the FP8_E4M3 operand carries unit scales through the MMA, so the whole tensor
    # scale has to come back somewhere and the Triton kernels put it exactly here.
    has_x_static_scale: gl.constexpr
    # WeightGuHbmLayout. Under SPLIT an N slot is one whole side, so the pair for
    # an emitted channel is two *tiles*, not two registers. Folded to INTERLEAVED
    # without a gated activation, where it is inert, so no use site re-checks that.
    weight_gu_hbm_layout: gl.constexpr
    epilogue: gl.constexpr  # EpilogueMode

    @gluon.constexpr_function
    def __init__(
        self,
        token_dtype_quant,
        expert_dtype_quant,
        token_online_quant,
        expert_online_quant,
        mma_acc_dtype,
        activation,
        output_quant,
        has_bias,
        has_gammas,
        has_gather,
        has_x_static_scale,
        weight_gu_hbm_layout=int(WeightGuHbmLayout.INTERLEAVED),
        epilogue=int(EpilogueMode.DEFAULT),
    ):
        self.token_dtype_quant = gl.constexpr(token_dtype_quant)
        self.expert_dtype_quant = gl.constexpr(expert_dtype_quant)
        self.token_online_quant = gl.constexpr(token_online_quant)
        self.expert_online_quant = gl.constexpr(expert_online_quant)
        self.mma_acc_dtype = gl.constexpr(mma_acc_dtype)
        self.activation = gl.constexpr(activation)
        self.output_quant = gl.constexpr(output_quant)
        self.has_bias = gl.constexpr(has_bias)
        self.has_gammas = gl.constexpr(has_gammas)
        self.has_gather = gl.constexpr(has_gather)
        self.has_x_static_scale = gl.constexpr(has_x_static_scale)
        self.weight_gu_hbm_layout = gl.constexpr(
            int(weight_gu_hbm_layout)
            if activation is not None
            else int(WeightGuHbmLayout.INTERLEAVED)
        )
        self.epilogue = gl.constexpr(int(epilogue))

    # -- activation --
    @gluon.constexpr_function
    def has_activation(self):
        return self.activation.value is not None

    @gluon.constexpr_function
    def activation_reduction_n(self):
        """Emitted columns per raw column. 2 for a gated activation, 1 otherwise."""
        return 2 if self.activation.value is not None else 1

    # -- dtype accessors --
    #
    # Everything below is indexed by operand (0 = token / LHS, 1 = expert / RHS). The
    # two operands genuinely differ in the mixed configurations -- a16w4 is bf16 x fp4,
    # a8w4 is fp8 x fp4 -- so nothing here may be keyed on one dtype and applied to
    # both.
    @gluon.constexpr_function
    def dtype_quant(self, idx):
        if idx == 0:
            return self.token_dtype_quant.value
        return self.expert_dtype_quant.value

    @gluon.constexpr_function
    def online_quant(self, idx):
        if idx == 0:
            return self.token_online_quant.value
        return self.expert_online_quant.value

    @gluon.constexpr_function
    def dot_kind(self):
        """Which dot the operand pair maps onto.

        * ``DotKind.MFMA`` -- both operands already bf16; the CDNA3-class pipe is the
          only one that takes them.
        * ``DotKind.MFMA_SCALED`` -- neither operand is bf16. Every fp4/fp8 combination
          goes here, including FP8 x FP8 with both scales ``None``: the backend folds
          the synthesized unit scales back into ``V_MFMA_*_F8F6F4`` and only that path
          reaches the double-rate K=64/128 pipes.
        * ``DotKind.UPCAST_MFMA`` -- exactly one operand is bf16. ``mfma_scaled`` cannot
          mix a bf16 operand with a microscaled one, so the low-precision side is
          expanded with ``gl.amd.cdna4.scaled_upcast`` first and the dot is a plain
          ``mfma``.
        """
        a_bf16 = self.online_quant(0) == int(DtypeQuant.BF16)
        b_bf16 = self.online_quant(1) == int(DtypeQuant.BF16)
        if a_bf16 and b_bf16:
            return int(DotKind.MFMA)
        if a_bf16 or b_bf16:
            return int(DotKind.UPCAST_MFMA)
        return int(DotKind.MFMA_SCALED)

    @gluon.constexpr_function
    def uses_mfma_scaled(self):
        return dq_uses_mfma_scaled(self.token_online_quant, self.expert_online_quant)

    @gluon.constexpr_function
    def mx_format(self, idx):
        return dq_mx_format(self.online_quant(idx))

    @gluon.constexpr_function
    def has_scale(self, idx):
        return dq_has_scale(self.dtype_quant(idx))

    @gluon.constexpr_function
    def pack_divisor(self, idx):
        return dq_pack_divisor(self.dtype_quant(idx))

    @gluon.constexpr_function
    def operand_elem_ty(self, idx):
        """Storage element type of one operand's payload, in LDS and in HBM.

        MXFP4 is two E2M1 values per byte, so it is stored and loaded as ``uint8`` and
        only ``mfma_scaled``'s ``e2m1`` format string tells the hardware otherwise.
        """
        dq = self.dtype_quant(idx)
        if dq == int(DtypeQuant.BF16):
            return gl.bfloat16
        if dq in (int(DtypeQuant.FP8_E4M3), int(DtypeQuant.MXFP8)):
            return gl.float8e4nv
        return gl.uint8


@aggregate
@strip_annotate
class KernelTuningConfig:
    """How the kernel *runs*. Holds the func config so ``KernelLayout(tuning_cfg)``
    can see the operand dtypes (an aggregate may hold another aggregate as a typed
    field). Every derived shape and layout lives in the resulting ``KernelLayout``,
    and so does the resolved LDS/register placement: this config only says whether
    tuning *allows* a component through LDS.

    Methods here only rephrase the config's own fields. Anything derived from shapes
    belongs in ``KernelLayout``, and the build preconditions live in ``_validate.py``.
    """

    func_cfg: KernelFuncConfig
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    K_UNROLL: gl.constexpr
    SLOT_M: gl.constexpr
    SLOT_N: gl.constexpr
    NUM_LDS_BUFFER: gl.constexpr
    mfma_instr_shape: gl.constexpr
    warps_per_cta: gl.constexpr
    tiles_per_warp: gl.constexpr
    k_width: gl.constexpr
    transposed: gl.constexpr
    WAVES_PER_EU: gl.constexpr
    TILE_SCHED: gl.constexpr
    GROUP_M: gl.constexpr
    NUM_XCDS: gl.constexpr
    token_cache_modifier: gl.constexpr
    token_scale_cache_modifier: gl.constexpr
    expert_cache_modifier: gl.constexpr
    expert_scale_cache_modifier: gl.constexpr
    result_cache_modifier: gl.constexpr
    result_scale_cache_modifier: gl.constexpr
    WARP_PIPELINE: gl.constexpr
    VGPR_PREFETCH_K: gl.constexpr
    A_SCALE_SORTED_SHUFFLED: gl.constexpr
    B_SCALE_SHUFFLED: gl.constexpr
    B_PRESHUFFLED: gl.constexpr
    ACT_FAST_RCP: gl.constexpr
    WAIT_COMMIT_SCHEME: gl.constexpr
    DS_READ_A_PAYLOAD_IN_MFMA: gl.constexpr
    DS_READ_A_SCALE_IN_MFMA: gl.constexpr
    DS_READ_B_PAYLOAD_IN_MFMA: gl.constexpr
    DS_READ_B_SCALE_IN_MFMA: gl.constexpr
    SCHED_MODE: gl.constexpr
    FROZEN_STEP: gl.constexpr
    SOFF_UNROLL: gl.constexpr
    SCALE_FILL_MID: gl.constexpr
    ALLOW_B_VIA_LDS: gl.constexpr
    ALLOW_B_SCALE_VIA_LDS: gl.constexpr
    ALLOW_A_SCALE_VIA_LDS: gl.constexpr
    A_NUM_BUFFER: gl.constexpr
    B_NUM_BUFFER: gl.constexpr
    A_SCALE_NUM_BUFFER: gl.constexpr
    B_SCALE_NUM_BUFFER: gl.constexpr
    SCALE_SLOT_M: gl.constexpr
    SCALE_SLOT_N: gl.constexpr
    SCALE_SLOT_K: gl.constexpr
    UNROLL_EPILOGUE: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        func_cfg,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_UNROLL,
        SLOT_M,
        SLOT_N,
        NUM_LDS_BUFFER,
        mfma_instr_shape,
        warps_per_cta,
        tiles_per_warp,
        k_width,
        transposed,
        WAVES_PER_EU,
        TILE_SCHED,
        GROUP_M,
        NUM_XCDS,
        token_cache_modifier,
        token_scale_cache_modifier,
        expert_cache_modifier,
        expert_scale_cache_modifier,
        result_cache_modifier,
        result_scale_cache_modifier,
        WARP_PIPELINE,
        VGPR_PREFETCH_K,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        B_PRESHUFFLED=False,
        ACT_FAST_RCP=False,
        WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_OP),
        DS_READ_A_PAYLOAD_IN_MFMA=False,
        DS_READ_A_SCALE_IN_MFMA=False,
        DS_READ_B_PAYLOAD_IN_MFMA=False,
        DS_READ_B_SCALE_IN_MFMA=False,
        SCHED_MODE=int(SchedMode.NONE),
        FROZEN_STEP=False,
        SOFF_UNROLL=False,
        SCALE_FILL_MID=False,
        ALLOW_B_VIA_LDS=True,
        ALLOW_B_SCALE_VIA_LDS=True,
        ALLOW_A_SCALE_VIA_LDS=True,
        A_NUM_BUFFER=0,
        B_NUM_BUFFER=0,
        A_SCALE_NUM_BUFFER=0,
        B_SCALE_NUM_BUFFER=0,
        SCALE_SLOT_M=0,
        SCALE_SLOT_N=0,
        SCALE_SLOT_K=0,
        UNROLL_EPILOGUE=True,
    ):
        self.func_cfg = func_cfg
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.K_UNROLL = gl.constexpr(K_UNROLL)
        self.SLOT_M = gl.constexpr(SLOT_M)
        self.SLOT_N = gl.constexpr(SLOT_N)
        self.NUM_LDS_BUFFER = gl.constexpr(NUM_LDS_BUFFER)
        self.mfma_instr_shape = gl.constexpr(list(mfma_instr_shape))
        self.warps_per_cta = gl.constexpr(list(warps_per_cta))
        self.tiles_per_warp = gl.constexpr(list(tiles_per_warp))
        self.k_width = gl.constexpr(k_width)
        self.transposed = gl.constexpr(transposed)
        self.WAVES_PER_EU = gl.constexpr(WAVES_PER_EU)
        self.TILE_SCHED = gl.constexpr(TILE_SCHED)
        self.GROUP_M = gl.constexpr(GROUP_M)
        self.NUM_XCDS = gl.constexpr(NUM_XCDS)
        self.token_cache_modifier = gl.constexpr(token_cache_modifier)
        self.token_scale_cache_modifier = gl.constexpr(token_scale_cache_modifier)
        self.expert_cache_modifier = gl.constexpr(expert_cache_modifier)
        self.expert_scale_cache_modifier = gl.constexpr(expert_scale_cache_modifier)
        self.result_cache_modifier = gl.constexpr(result_cache_modifier)
        self.result_scale_cache_modifier = gl.constexpr(result_scale_cache_modifier)
        # int, not bool: WarpPipeline has three values and True still lands on COMPILER.
        self.WARP_PIPELINE = gl.constexpr(int(WARP_PIPELINE))
        self.VGPR_PREFETCH_K = gl.constexpr(int(VGPR_PREFETCH_K))
        self.A_SCALE_SORTED_SHUFFLED = gl.constexpr(bool(A_SCALE_SORTED_SHUFFLED))
        self.B_SCALE_SHUFFLED = gl.constexpr(bool(B_SCALE_SHUFFLED))
        self.B_PRESHUFFLED = gl.constexpr(bool(B_PRESHUFFLED))
        self.ACT_FAST_RCP = gl.constexpr(bool(ACT_FAST_RCP))
        self.WAIT_COMMIT_SCHEME = gl.constexpr(int(WAIT_COMMIT_SCHEME))
        self.DS_READ_A_PAYLOAD_IN_MFMA = gl.constexpr(bool(DS_READ_A_PAYLOAD_IN_MFMA))
        self.DS_READ_A_SCALE_IN_MFMA = gl.constexpr(bool(DS_READ_A_SCALE_IN_MFMA))
        self.DS_READ_B_PAYLOAD_IN_MFMA = gl.constexpr(bool(DS_READ_B_PAYLOAD_IN_MFMA))
        self.DS_READ_B_SCALE_IN_MFMA = gl.constexpr(bool(DS_READ_B_SCALE_IN_MFMA))
        self.SCHED_MODE = gl.constexpr(int(SCHED_MODE))
        self.FROZEN_STEP = gl.constexpr(bool(FROZEN_STEP))
        self.SOFF_UNROLL = gl.constexpr(bool(SOFF_UNROLL))
        self.SCALE_FILL_MID = gl.constexpr(bool(SCALE_FILL_MID))
        self.ALLOW_B_VIA_LDS = gl.constexpr(bool(ALLOW_B_VIA_LDS))
        self.ALLOW_B_SCALE_VIA_LDS = gl.constexpr(bool(ALLOW_B_SCALE_VIA_LDS))
        self.ALLOW_A_SCALE_VIA_LDS = gl.constexpr(bool(ALLOW_A_SCALE_VIA_LDS))
        self.A_NUM_BUFFER = gl.constexpr(int(A_NUM_BUFFER))
        self.B_NUM_BUFFER = gl.constexpr(int(B_NUM_BUFFER))
        self.A_SCALE_NUM_BUFFER = gl.constexpr(int(A_SCALE_NUM_BUFFER))
        self.B_SCALE_NUM_BUFFER = gl.constexpr(int(B_SCALE_NUM_BUFFER))
        self.SCALE_SLOT_M = gl.constexpr(int(SCALE_SLOT_M))
        self.SCALE_SLOT_N = gl.constexpr(int(SCALE_SLOT_N))
        self.SCALE_SLOT_K = gl.constexpr(int(SCALE_SLOT_K))
        self.UNROLL_EPILOGUE = gl.constexpr(bool(UNROLL_EPILOGUE))

    @gluon.constexpr_function
    def num_buffers(self, operand, scale=False):
        """Ring depth for one payload or scale; zero inherits NUM_LDS_BUFFER."""
        assert operand in (0, 1), "operand must be 0 (A) or 1 (B)"
        if scale:
            depth = self.A_SCALE_NUM_BUFFER if operand == 0 else self.B_SCALE_NUM_BUFFER
        else:
            depth = self.A_NUM_BUFFER if operand == 0 else self.B_NUM_BUFFER
        return depth.value or self.NUM_LDS_BUFFER.value

    # -- storage placement policy and format decisions -----------------------------
    # Yes/no policy read from the knobs above, and nothing else: no shapes. Which
    # components *actually* go through LDS also depends on whether the tile admits a
    # direct-to-LDS copy, so the final placement is resolved in _layout.py
    # (``KernelLayout.component_via_lds`` / ``component_in_reg``) from these.

    @gluon.constexpr_function
    def allow_component_via_lds(self, component):
        """Whether tuning permits staging a :class:`Component` through LDS.

        A payload reads ``ALLOW_B_VIA_LDS`` for B; A has no knob and is always
        permitted. A scale reads ``ALLOW_A_SCALE_VIA_LDS`` / ``ALLOW_B_SCALE_VIA_LDS``.
        A permitted component can still land in registers when its tile cannot take
        the copy; see ``KernelLayout.component_via_lds``.
        """
        operand, is_scale = component.operand_index.value, component.is_scale.value
        if is_scale:
            flag = (
                self.ALLOW_A_SCALE_VIA_LDS
                if operand == 0
                else self.ALLOW_B_SCALE_VIA_LDS
            )
            return bool(flag.value)
        return operand == 0 or bool(self.ALLOW_B_VIA_LDS.value)

    @gluon.constexpr_function
    def preshuffled(self, idx, is_scale=False):
        """Is this component already stored in HBM in the order the MFMA reads it?

        A payload is blocked 16-column. Never for A: the token rows are gathered per
        launch from the routing order, so there is no static permutation to bake in.
        For B it is the ``B_PRESHUFFLED`` knob, which the caller honours by handing
        over the permuted weight tensor. A scale is in MFMA fragment order, set by
        ``A_SCALE_SORTED_SHUFFLED`` or ``B_SCALE_SHUFFLED``; never for an unscaled
        operand.
        """
        if not is_scale:
            return idx == 1 and self.B_PRESHUFFLED.value
        if not self.func_cfg.has_scale(idx):
            return False
        if idx == 0:
            return self.A_SCALE_SORTED_SHUFFLED.value
        return self.B_SCALE_SHUFFLED.value

    @gluon.constexpr_function
    def ds_read_in_mfma(self, operand, scale=False):
        """Whether one component's read is assigned to the MFMA region.

        ``operand`` is 0 for A (tokens), 1 for B (experts); ``scale`` selects its
        scale rather than its payload. A component absent from a dtype emits no read.
        """
        assert operand in (0, 1), "operand must be 0 (A) or 1 (B)"
        if operand == 0:
            flag = (
                self.DS_READ_A_SCALE_IN_MFMA
                if scale
                else self.DS_READ_A_PAYLOAD_IN_MFMA
            )
        else:
            flag = (
                self.DS_READ_B_SCALE_IN_MFMA
                if scale
                else self.DS_READ_B_PAYLOAD_IN_MFMA
            )
        return bool(flag.value)

    # -- inter-wave ping-pong (WarpPipeline) ----------------------------------------
    # NONE and COMPILER share the live step body; MANUAL selects the frozen body.
    # See _lang.pick_warp_pipeline_stage.

    @gluon.constexpr_function
    def warp_pipeline_enabled(self):
        """Any ping-pong at all -- what the VGPR_PREFETCH_K precondition keys on."""
        return self.WARP_PIPELINE.value != int(WarpPipeline.NONE)

    @gluon.constexpr_function
    def warp_pipeline_compiler(self):
        """Hand the mfma/mem halves to TritonAMDGPUWarpPipeline."""
        return self.WARP_PIPELINE.value == int(WarpPipeline.COMPILER)

    @gluon.constexpr_function
    def warp_pipeline_manual(self):
        """The hand-emitted rendezvous -- ``_pipeline_step_frozen`` only, for now."""
        return self.WARP_PIPELINE.value == int(WarpPipeline.MANUAL)

    # -- commit-group granularity (WaitCommitScheme) --------------------------------
    # Emission and wait arithmetic use the same copy schedule and group boundaries.

    @gluon.constexpr_function
    def commit_per_op(self):
        return self.WAIT_COMMIT_SCHEME.value == int(WaitCommitScheme.PER_OP)

    @gluon.constexpr_function
    def commit_per_pane(self):
        return self.WAIT_COMMIT_SCHEME.value == int(WaitCommitScheme.PER_PANE)

    @gluon.constexpr_function
    def commit_per_stage(self):
        return self.commit_per_stage_warp_pipeline() or self.commit_per_stage_whole()

    @gluon.constexpr_function
    def commit_per_stage_warp_pipeline(self):
        return self.WAIT_COMMIT_SCHEME.value == int(
            WaitCommitScheme.PER_STAGE_WARP_PIPELINE
        )

    @gluon.constexpr_function
    def commit_per_stage_whole(self):
        return self.WAIT_COMMIT_SCHEME.value == int(WaitCommitScheme.PER_STAGE_WHOLE)
