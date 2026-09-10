# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Functionality and tuning configuration aggregates for the gfx950 Gluon MoE GEMMs.

Both are ``@gluon.aggregate``s built entirely out of ``gl.constexpr`` scalars, so the
*same* object can be constructed on the host (for the grid tuple) and inside the kernel
(for the layouts) from the same numbers -- an aggregate itself cannot be a launch
argument, only its class can. Keeping the arithmetic in the aggregate is what stops the
host grid math and the device tile math from drifting.
"""

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from . import _layout
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
from ._lang import (
    unwrap as _v,
)
from ._layout import _KernelFuncShape, _KernelTuningLayout

__all__ = ["KernelFuncConfig", "KernelTuningConfig"]

# Compatibility aliases for callers that imported these before the extraction.
LDS_CAP_BYTES = _layout.LDS_CAP_BYTES
LDS_EPILOGUE_RESERVE_BYTES = _layout.LDS_EPILOGUE_RESERVE_BYTES
LDS_USABLE_BYTES = _layout.LDS_USABLE_BYTES
byte_unit_lds_layout = _layout.byte_unit_lds_layout
make_scale_swizzle_check = _layout.make_scale_swizzle_check


@aggregate
@strip_annotate
class KernelFuncConfig(_KernelFuncShape):
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
    # Gated-activation operand layout along N. False: interleaved (g,l,g,l), the
    # packing every caller uses today. True: the two sides are whole halves of the raw
    # N axis, gate in [0, N/2) and linear in [N/2, N) -- so a mini-N block is one whole
    # side and the pair for an emitted channel is two *tiles*, not two registers. The
    # caller's weights, weight scales and bias must be permuted to match; see
    # ``activations.py::gate_up_split_perm``.
    gate_up_split: gl.constexpr
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
        gate_up_split=False,
        epilogue=int(EpilogueMode.DEFAULT),
    ):
        self.token_dtype_quant = gl.constexpr(_v(token_dtype_quant))
        self.expert_dtype_quant = gl.constexpr(_v(expert_dtype_quant))
        self.token_online_quant = gl.constexpr(_v(token_online_quant))
        self.expert_online_quant = gl.constexpr(_v(expert_online_quant))
        self.mma_acc_dtype = gl.constexpr(_v(mma_acc_dtype))
        self.activation = gl.constexpr(_v(activation))
        self.output_quant = gl.constexpr(_v(output_quant))
        self.has_bias = gl.constexpr(_v(has_bias))
        self.has_gammas = gl.constexpr(_v(has_gammas))
        self.has_gather = gl.constexpr(_v(has_gather))
        self.has_x_static_scale = gl.constexpr(_v(has_x_static_scale))
        self.gate_up_split = gl.constexpr(bool(_v(gate_up_split)))
        self.epilogue = gl.constexpr(int(_v(epilogue)))

    # -- activation accessors (the spec's `activation` is a NamedTuple in a constexpr,
    #    so unwrap it here rather than at every use site) --
    @gluon.constexpr_function
    def act(self):
        return _v(self.activation)

    @gluon.constexpr_function
    def has_activation(self):
        return _v(self.activation) is not None

    # -- dtype accessors --
    #
    # Everything below is indexed by operand (0 = token / LHS, 1 = expert / RHS). The
    # two operands genuinely differ in the mixed configurations -- a16w4 is bf16 x fp4,
    # a8w4 is fp8 x fp4 -- so nothing here may be keyed on one dtype and applied to
    # both. `a_*` / `b_*` remain as thin aliases for readability at the use sites.
    @gluon.constexpr_function
    def dtype_quant(self, idx):
        if _v(idx) == 0:
            return _v(self.token_dtype_quant)
        return _v(self.expert_dtype_quant)

    @gluon.constexpr_function
    def online_quant(self, idx):
        if _v(idx) == 0:
            return _v(self.token_online_quant)
        return _v(self.expert_online_quant)

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
    def a_format(self):
        return dq_mx_format(self.token_online_quant)

    @gluon.constexpr_function
    def b_format(self):
        return dq_mx_format(self.expert_online_quant)

    @gluon.constexpr_function
    def has_scale(self, idx):
        return dq_has_scale(self.dtype_quant(idx))

    @gluon.constexpr_function
    def a_has_scale(self):
        return dq_has_scale(self.token_dtype_quant)

    @gluon.constexpr_function
    def b_has_scale(self):
        return dq_has_scale(self.expert_dtype_quant)

    @gluon.constexpr_function
    def pack_divisor(self, idx):
        return dq_pack_divisor(self.dtype_quant(idx))

    @gluon.constexpr_function
    def a_pack_divisor(self):
        return dq_pack_divisor(self.token_dtype_quant)

    @gluon.constexpr_function
    def b_pack_divisor(self):
        return dq_pack_divisor(self.expert_dtype_quant)

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
class KernelTuningConfig(_KernelTuningLayout):
    """How the kernel *runs*. Holds the func config so the layout methods can see the
    operand dtypes (an aggregate may hold another aggregate as a typed field).
    """

    func_cfg: KernelFuncConfig
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    K_UNROLL: gl.constexpr
    MINI_BLOCK_K: gl.constexpr
    MINI_BLOCK_M: gl.constexpr
    MINI_BLOCK_N: gl.constexpr
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
    B_IN_REG: gl.constexpr
    B_SCALE_IN_REG: gl.constexpr
    A_SCALE_IN_REG: gl.constexpr
    A_NUM_BUFFER: gl.constexpr
    B_NUM_BUFFER: gl.constexpr
    A_SCALE_NUM_BUFFER: gl.constexpr
    B_SCALE_NUM_BUFFER: gl.constexpr
    SCALE_MINI_BLOCK_M: gl.constexpr
    SCALE_MINI_BLOCK_N: gl.constexpr
    SCALE_MINI_BLOCK_K: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        func_cfg,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_UNROLL,
        MINI_BLOCK_K,
        MINI_BLOCK_M,
        MINI_BLOCK_N,
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
        B_IN_REG=False,
        B_SCALE_IN_REG=False,
        A_SCALE_IN_REG=False,
        A_NUM_BUFFER=0,
        B_NUM_BUFFER=0,
        A_SCALE_NUM_BUFFER=0,
        B_SCALE_NUM_BUFFER=0,
        SCALE_MINI_BLOCK_M=0,
        SCALE_MINI_BLOCK_N=0,
        SCALE_MINI_BLOCK_K=0,
    ):
        self.func_cfg = func_cfg
        self.BLOCK_M = gl.constexpr(_v(BLOCK_M))
        self.BLOCK_N = gl.constexpr(_v(BLOCK_N))
        self.BLOCK_K = gl.constexpr(_v(BLOCK_K))
        self.K_UNROLL = gl.constexpr(_v(K_UNROLL))
        self.MINI_BLOCK_K = gl.constexpr(_v(MINI_BLOCK_K))
        self.MINI_BLOCK_M = gl.constexpr(_v(MINI_BLOCK_M))
        self.MINI_BLOCK_N = gl.constexpr(_v(MINI_BLOCK_N))
        self.NUM_LDS_BUFFER = gl.constexpr(_v(NUM_LDS_BUFFER))
        self.mfma_instr_shape = gl.constexpr(list(_v(mfma_instr_shape)))
        self.warps_per_cta = gl.constexpr(list(_v(warps_per_cta)))
        self.tiles_per_warp = gl.constexpr(list(_v(tiles_per_warp)))
        self.k_width = gl.constexpr(_v(k_width))
        self.transposed = gl.constexpr(_v(transposed))
        self.WAVES_PER_EU = gl.constexpr(_v(WAVES_PER_EU))
        self.TILE_SCHED = gl.constexpr(_v(TILE_SCHED))
        self.GROUP_M = gl.constexpr(_v(GROUP_M))
        self.NUM_XCDS = gl.constexpr(_v(NUM_XCDS))
        self.token_cache_modifier = gl.constexpr(_v(token_cache_modifier))
        self.token_scale_cache_modifier = gl.constexpr(_v(token_scale_cache_modifier))
        self.expert_cache_modifier = gl.constexpr(_v(expert_cache_modifier))
        self.expert_scale_cache_modifier = gl.constexpr(_v(expert_scale_cache_modifier))
        self.result_cache_modifier = gl.constexpr(_v(result_cache_modifier))
        self.result_scale_cache_modifier = gl.constexpr(_v(result_scale_cache_modifier))
        # int, not bool: WarpPipeline has three values and True still lands on COMPILER.
        self.WARP_PIPELINE = gl.constexpr(int(_v(WARP_PIPELINE)))
        self.VGPR_PREFETCH_K = gl.constexpr(int(_v(VGPR_PREFETCH_K)))
        self.A_SCALE_SORTED_SHUFFLED = gl.constexpr(bool(_v(A_SCALE_SORTED_SHUFFLED)))
        self.B_SCALE_SHUFFLED = gl.constexpr(bool(_v(B_SCALE_SHUFFLED)))
        self.B_PRESHUFFLED = gl.constexpr(bool(_v(B_PRESHUFFLED)))
        self.ACT_FAST_RCP = gl.constexpr(bool(_v(ACT_FAST_RCP)))
        self.WAIT_COMMIT_SCHEME = gl.constexpr(int(_v(WAIT_COMMIT_SCHEME)))
        self.DS_READ_A_PAYLOAD_IN_MFMA = gl.constexpr(
            bool(_v(DS_READ_A_PAYLOAD_IN_MFMA))
        )
        self.DS_READ_A_SCALE_IN_MFMA = gl.constexpr(bool(_v(DS_READ_A_SCALE_IN_MFMA)))
        self.DS_READ_B_PAYLOAD_IN_MFMA = gl.constexpr(
            bool(_v(DS_READ_B_PAYLOAD_IN_MFMA))
        )
        self.DS_READ_B_SCALE_IN_MFMA = gl.constexpr(bool(_v(DS_READ_B_SCALE_IN_MFMA)))
        self.SCHED_MODE = gl.constexpr(int(_v(SCHED_MODE)))
        self.FROZEN_STEP = gl.constexpr(bool(_v(FROZEN_STEP)))
        self.SOFF_UNROLL = gl.constexpr(bool(_v(SOFF_UNROLL)))
        self.SCALE_FILL_MID = gl.constexpr(bool(_v(SCALE_FILL_MID)))
        self.B_IN_REG = gl.constexpr(bool(_v(B_IN_REG)))
        self.B_SCALE_IN_REG = gl.constexpr(bool(_v(B_SCALE_IN_REG)))
        self.A_SCALE_IN_REG = gl.constexpr(bool(_v(A_SCALE_IN_REG)))
        self.A_NUM_BUFFER = gl.constexpr(int(_v(A_NUM_BUFFER)))
        self.B_NUM_BUFFER = gl.constexpr(int(_v(B_NUM_BUFFER)))
        self.A_SCALE_NUM_BUFFER = gl.constexpr(int(_v(A_SCALE_NUM_BUFFER)))
        self.B_SCALE_NUM_BUFFER = gl.constexpr(int(_v(B_SCALE_NUM_BUFFER)))
        self.SCALE_MINI_BLOCK_M = gl.constexpr(int(_v(SCALE_MINI_BLOCK_M)))
        self.SCALE_MINI_BLOCK_N = gl.constexpr(int(_v(SCALE_MINI_BLOCK_N)))
        self.SCALE_MINI_BLOCK_K = gl.constexpr(int(_v(SCALE_MINI_BLOCK_K)))

    @gluon.constexpr_function
    def num_buffers(self, operand, scale=False):
        """Ring depth for one payload or scale; zero inherits NUM_LDS_BUFFER."""
        operand = _v(operand)
        assert operand in (0, 1), "operand must be 0 (A) or 1 (B)"
        if _v(scale):
            depth = self.A_SCALE_NUM_BUFFER if operand == 0 else self.B_SCALE_NUM_BUFFER
        else:
            depth = self.A_NUM_BUFFER if operand == 0 else self.B_NUM_BUFFER
        return _v(depth) or _v(self.NUM_LDS_BUFFER)

    @gluon.constexpr_function
    def scale_in_reg(self, operand):
        """Whether tuning explicitly selects direct-register scales for an operand."""
        return bool(
            _v(self.A_SCALE_IN_REG if _v(operand) == 0 else self.B_SCALE_IN_REG)
        )

    @gluon.constexpr_function
    def component_span(self, operand, scale=False):
        """Payload steps covered by one component's prefetch ring."""
        depth = self.num_buffers(operand, scale)
        ratio = self.scale_step_ratio(operand) if _v(scale) else 1
        return (depth - 1) * ratio + 1

    @gluon.constexpr_function
    def pipeline_depth(self):
        """Largest prefetch span, measured in payload steps, of active components."""
        depth = max(self.num_buffers(0), self.num_buffers(1))
        for operand in (0, 1):
            if self.func_cfg.has_scale(operand):
                depth = max(depth, self.component_span(operand, True))
        return depth

    @gluon.constexpr_function
    def pipeline_register_period(self):
        """LCM of the active register rings; absent scales have no ring.

        LDS slots may use runtime indices. Register tuples need static indices in
        the main loop, so an unrolled body must return each ring to its first slot.
        """
        period = 1
        for operand in (0, 1):
            if not self.payload_via_lds(operand):
                period = math.lcm(period, self.num_buffers(operand))
            if self.func_cfg.has_scale(operand) and not self.scale_via_lds(operand):
                period = math.lcm(
                    period,
                    self.num_buffers(operand, True) * self.scale_step_ratio(operand),
                )
        return period

    @gluon.constexpr_function
    def pipeline_unroll(self):
        """Smallest unroll covering complete register rings and scale K tiles."""
        requested = _v(self.K_UNROLL)
        assert requested >= 1, "K_UNROLL must be at least 1"
        period = self.pipeline_register_period()
        for operand in (0, 1):
            period = math.lcm(period, self.scale_step_ratio(operand))
        return (requested + period - 1) // period * period

    @gluon.constexpr_function
    def validate_buffer_counts(self):
        """Only components present in the operand formats participate."""
        for operand in (0, 1):
            for scale in (False, True):
                if scale and not self.func_cfg.has_scale(operand):
                    continue
                depth = self.num_buffers(operand, scale)
                assert depth >= 2, (
                    f"{'A' if operand == 0 else 'B'}{'_SCALE' if scale else ''}_NUM_BUFFER "
                    f"({depth}) must be at least 2"
                )
        assert _v(self.K_UNROLL) >= 1, "K_UNROLL must be at least 1"
        return True

    @gluon.constexpr_function
    def ds_read_in_mfma(self, operand, scale=False):
        """Whether one component's read is assigned to the MFMA region.

        ``operand`` is 0 for A (tokens), 1 for B (experts); ``scale`` selects its
        scale rather than its payload. A component absent from a dtype emits no read.
        """
        operand = _v(operand)
        assert operand in (0, 1), "operand must be 0 (A) or 1 (B)"
        if operand == 0:
            flag = (
                self.DS_READ_A_SCALE_IN_MFMA
                if _v(scale)
                else self.DS_READ_A_PAYLOAD_IN_MFMA
            )
        else:
            flag = (
                self.DS_READ_B_SCALE_IN_MFMA
                if _v(scale)
                else self.DS_READ_B_PAYLOAD_IN_MFMA
            )
        return bool(_v(flag))

    # -- inter-wave ping-pong (WarpPipeline) ----------------------------------------
    # NONE and COMPILER share the live step body; MANUAL selects the frozen body.
    # See _lang.pick_warp_pipeline_stage.

    @gluon.constexpr_function
    def warp_pipeline_enabled(self):
        """Any ping-pong at all -- what the VGPR_PREFETCH_K precondition keys on."""
        return _v(self.WARP_PIPELINE) != int(WarpPipeline.NONE)

    @gluon.constexpr_function
    def warp_pipeline_compiler(self):
        """Hand the mfma/mem halves to TritonAMDGPUWarpPipeline."""
        return _v(self.WARP_PIPELINE) == int(WarpPipeline.COMPILER)

    @gluon.constexpr_function
    def warp_pipeline_manual(self):
        """The hand-emitted rendezvous -- ``_pipeline_step_frozen`` only, for now."""
        return _v(self.WARP_PIPELINE) == int(WarpPipeline.MANUAL)

    # -- commit-group granularity (WaitCommitScheme) --------------------------------
    # Emission and wait arithmetic use the same copy schedule and group boundaries.

    @gluon.constexpr_function
    def commit_per_op(self):
        return _v(self.WAIT_COMMIT_SCHEME) == int(WaitCommitScheme.PER_OP)

    @gluon.constexpr_function
    def commit_per_slot(self):
        return _v(self.WAIT_COMMIT_SCHEME) == int(WaitCommitScheme.PER_SLOT)

    @gluon.constexpr_function
    def commit_per_stage(self):
        return self.commit_per_stage_warp_pipeline() or self.commit_per_stage_whole()

    @gluon.constexpr_function
    def commit_per_stage_warp_pipeline(self):
        return _v(self.WAIT_COMMIT_SCHEME) == int(
            WaitCommitScheme.PER_STAGE_WARP_PIPELINE
        )

    @gluon.constexpr_function
    def commit_per_stage_whole(self):
        return _v(self.WAIT_COMMIT_SCHEME) == int(WaitCommitScheme.PER_STAGE_WHOLE)

    @gluon.constexpr_function
    def wait_at_stage_head(self):
        """Both per-stage modes wait once at the head; other modes wait per read slot."""
        return self.commit_per_stage()

    @gluon.constexpr_function
    def validate(self, N, K):
        """Every precondition the kernel body then assumes. Constexpr asserts, not
        comments -- a violated one is either a miscompile or a silent over-read."""
        fc = self.func_cfg
        BK = _v(self.BLOCK_K)

        assert _v(fc.epilogue) in (
            int(EpilogueMode.DEFAULT),
            int(EpilogueMode.NOP_ACTIVATION),
            int(EpilogueMode.NOP),
        ), f"epilogue {_v(fc.epilogue)} is not an EpilogueMode"
        assert _v(self.SCHED_MODE) in (
            int(SchedMode.NONE),
            int(SchedMode.IGLP_0),
            int(SchedMode.IGLP_1),
            int(SchedMode.MFMA_16),
            int(SchedMode.MFMA_8),
        ), f"SCHED_MODE {_v(self.SCHED_MODE)} is not a SchedMode"
        self.validate_layout(N, K)

        # -- warp pipelining (the inter-wave ping-pong) --
        # A slot's MFMAs are handed to TritonAMDGPUWarpPipeline as the `mfma` stage and
        # its ds_reads plus global->LDS copies as the `mem` stage. That is only a legal
        # split when the MFMAs read nothing the same slot loads, i.e. when the whole
        # BLOCK_K window is carried in registers from the previous stage. The pass also
        # rejects a wait inside a stage region, which is why the per-slot wait_group is
        # emitted before the `mfma` region rather than inside the `mem` one.
        assert _v(self.WARP_PIPELINE) in (
            int(WarpPipeline.NONE),
            int(WarpPipeline.COMPILER),
            int(WarpPipeline.MANUAL),
        ), (
            f"WARP_PIPELINE {_v(self.WARP_PIPELINE)} is not a WarpPipeline: "
            f"NONE {int(WarpPipeline.NONE)}, COMPILER {int(WarpPipeline.COMPILER)}, "
            f"MANUAL {int(WarpPipeline.MANUAL)}"
        )
        if self.warp_pipeline_enabled():
            assert self.num_prefetch_k_slots() == self.num_k_slots_per_tile(), (
                f"WARP_PIPELINE needs VGPR_PREFETCH_K == BLOCK_K (got "
                f"{_v(self.VGPR_PREFETCH_K)} vs {BK}): with a partial window the slot's "
                "MFMAs depend on the slot's own ds_read and there is no mem stage to "
                "hide behind them"
            )

        # -- register prefetch depth --
        # VGPR_PREFETCH_K is how much of a BLOCK_K stage is read into registers one step
        # before its MFMAs consume it. BLOCK_K carries the whole stage, 0 carries none,
        # and the ds_read still moves a full stage either way -- only the register
        # handoff changes, which is why the buffer arithmetic above does not mention it.
        pk = _v(self.VGPR_PREFETCH_K)
        if pk:
            assert pk <= BK, f"VGPR_PREFETCH_K {pk} > BLOCK_K {BK}"
            assert pk >= _v(self.MINI_BLOCK_K), (
                f"VGPR_PREFETCH_K {pk} < MINI_BLOCK_K {_v(self.MINI_BLOCK_K)}: the "
                "handoff is a whole number of mini-K steps"
            )
            assert BK % pk == 0, f"BLOCK_K {BK} % VGPR_PREFETCH_K {pk} != 0"
            assert pk % _v(self.MINI_BLOCK_K) == 0, (
                f"VGPR_PREFETCH_K {pk} % MINI_BLOCK_K {_v(self.MINI_BLOCK_K)} != 0"
            )

        # -- commit-group granularity --
        # Reject unknown schemes before building the shared copy/group schedule.
        assert _v(self.WAIT_COMMIT_SCHEME) in (
            int(WaitCommitScheme.PER_OP),
            int(WaitCommitScheme.PER_SLOT),
            int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE),
            int(WaitCommitScheme.PER_STAGE_WHOLE),
        ), (
            f"WAIT_COMMIT_SCHEME {_v(self.WAIT_COMMIT_SCHEME)} is not a WaitCommitScheme: "
            f"PER_OP {int(WaitCommitScheme.PER_OP)}, "
            f"PER_SLOT {int(WaitCommitScheme.PER_SLOT)}, "
            f"PER_STAGE_WARP_PIPELINE {int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE)}, "
            f"PER_STAGE_WHOLE {int(WaitCommitScheme.PER_STAGE_WHOLE)}"
        )

        self.validate_layout_resources()
        return True
