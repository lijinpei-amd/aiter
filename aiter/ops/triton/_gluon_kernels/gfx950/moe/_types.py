# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dtype / quant-scheme enums and the host->device NamedTuples of the gfx950 Gluon MoE
grouped GEMMs.
"""

from enum import IntEnum, IntFlag
from typing import NamedTuple

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import const as _c
from ._lang import unwrap as _unwrap

__all__ = [
    "ActKind",
    "ActivationSpec",
    "DSReadOperand",
    "DotKind",
    "DtypeQuant",
    "EpilogueMode",
    "FuncSpec",
    "NonQuantExpertTensor",
    "NonQuantTokenTensor",
    "QuantExpertTensor",
    "QuantTokenTensor",
    "ResultTensor",
    "RoutingMeta",
    "ScaleSwizzle",
    "SchedMode",
    "TileSched",
    "TuningSpec",
    "WaitCommitScheme",
    "WarpPipeline",
    "dq_has_scale",
    "dq_mx_format",
    "dq_pack_divisor",
    "dq_uses_mfma_scaled",
]


class DtypeQuant(IntEnum):
    """Tensor payload dtype together with its quantisation scheme."""

    BF16 = 0  # bf16, no scale
    FP8_E4M3 = 1  # fp8 e4m3, unit scales (folded into V_MFMA_*_F8F6F4)
    MXFP4 = 2  # E2M1, 2 per byte packed along K, uint8 E8M0 group 32 along K
    MXFP8 = 3  # E4M3, uint8 E8M0 group 32 along K


class ScaleSwizzle(IntEnum):
    NONE = 0
    # TODO name what the swizzle does, not the arch
    CDNA4_SCALE = 1  # utils/shuffle.py:_shuffle_scale_tile_gfx950 preshuffle
    # csrc/kernels/mxfp4_moe/moe_aux/moe_sort_scales.cuh, run per call rather than
    # offline: the token scales are gathered into routing order *and* permuted into the
    # MFMA scale-fragment order, so a stage's scales are a contiguous slice instead of
    # BLOCK_M rows of 8 bytes at a K/32 stride. Only meaningful for operand A, and only
    # for the layout the shuffle was written against -- see
    # KernelTuningConfig.sorted_shuffled_ok().
    SORTED_SHUFFLED = 2


class ActKind(IntEnum):
    SILU = 0  # alpha == 1.0
    SWIGLU_OAI = 1  # alpha != 1.0


class EpilogueMode(IntEnum):
    """Epilogue arithmetic, including shape-preserving benchmark ablations.

    Both NOP modes keep the gated reduction (gate * up), output quantisation and
    stores, so they preserve the output shape and traffic. NOP_ACTIVATION omits the
    activation function; NOP also omits bias and gammas. Operand dequantisation,
    including a per-tensor activation scale, still applies in every mode.
    """

    DEFAULT = 0
    NOP_ACTIVATION = 1
    NOP = 2


class DSReadOperand(IntFlag):
    """Components whose reads belong to the MFMA region instead of the memory region.

    Payload and scale placement are independent for both operands. Components that
    do not use LDS follow the same placement for their register loads.
    """

    NONE = 0
    A = 1
    B = 2
    A_SCALE = 4
    B_SCALE = 8
    ALL = A | B | A_SCALE | B_SCALE


class SchedMode(IntEnum):
    """Backend scheduling hint for a K stage without compiler warp pipelining."""

    NONE = 0
    IGLP_0 = 1
    IGLP_1 = 2
    MFMA_16 = 3  # 4 x (16 MFMA, 6 LDS reads, 4 VMEM operations)
    MFMA_8 = 4  # 8 x (8 MFMA, 3 LDS reads, 2 VMEM operations)


class DotKind(IntEnum):
    """Which CDNA4 matrix instruction an operand pair maps onto."""

    MFMA = 0  # bf16 x bf16
    MFMA_SCALED = 1  # any fp4/fp8 pair, incl. FP8 x FP8 with unit scales
    UPCAST_MFMA = 2  # bf16 x microscaled: scaled_upcast, then plain mfma


class TileSched(IntEnum):
    LINEAR = 0  # plain row-major (pid_m, pid_n), no swizzle
    GROUP_M = 1  # pid_grid GROUP_M blocking, for L2 reuse of the token tile
    XCD_GROUP_M = 2  # remap_xcd first, then GROUP_M blocking


class WarpPipeline(IntEnum):
    """Which inter-wave ping-pong the K-loop step is built for.

    All modes share the runtime component-pipeline driver. The live step places
    MFMA and memory work in regions selected by ``_lang.pick_warp_pipeline_stage``;
    the explicit frozen step retains the reference's manual rendezvous sequence.

    * ``NONE`` -- no borders. One wave group, the MFMAs and the memory work overlapped
      only by the machine scheduler.
    * ``COMPILER`` -- hand the ``mfma``/``mem`` halves to ``TritonAMDGPUWarpPipeline``,
      which turns the interleave into a two-wave-group ping-pong. Needs
      ``VGPR_PREFETCH_K == BLOCK_K`` (the slot's MFMAs must not read what the slot loads)
      and only applies inside the ``tl.range`` body -- a border opened in the peeled step
      or the drain would still be open when the loop starts, and the pass rejects a loop,
      and every wait, caught inside a stage.
    * ``MANUAL`` -- the hand-emitted rendezvous (``cond_barrier`` / ``setprio`` /
      ``bare_barrier``) instead of the pass. Currently implemented only by
      ``_pipeline_step_frozen``, so it needs ``FROZEN_STEP=1``; the live step asserts
      rather than quietly running unpipelined.
    """

    NONE = 0
    COMPILER = 1
    MANUAL = 2


class WaitCommitScheme(IntEnum):
    """Commit granularity and wait placement for the live HBM-to-LDS pipeline.

    * ``PER_OP`` commits each asynchronous copy separately and waits before each
      slot's LDS reads. Payload and scale copies have separate groups; a shared
      scale tile contributes only its owner's copy. Direct HBM payload and scale
      loads into registers contribute no asynchronous group.
    * ``PER_SLOT`` commits once after each slot, including slots with no copies,
      and waits before each slot's LDS reads.
    * ``PER_STAGE_WARP_PIPELINE`` commits once after the last memory region of
      the K stage, before its final MFMA region.
    * ``PER_STAGE_WHOLE`` commits once after the whole K stage, including MFMA.
      Both per-stage modes wait once at the head of each stage that reads LDS.
      Their commit locations also apply when the compiler warp pipeline is off.

    The shared schedule derives group counts and read dependencies from the same
    copy ownership used by the emitter. The frozen snapshot keeps its own pinned
    commit and wait schedule.
    """

    PER_OP = 1
    PER_SLOT = 2
    # Preserve the whole-stage setting used by existing cold-bench recipes.
    PER_STAGE_WHOLE = 3
    PER_STAGE_WARP_PIPELINE = 4


class FuncSpec(NamedTuple):
    """Plain-Python mirror of :class:`KernelFuncConfig`'s fields, carried as a single
    ``gl.constexpr`` leaf so the launch-time argument specializer walks one item instead
    of ten. Field order must match the aggregate's constructor."""

    token_dtype_quant: int
    expert_dtype_quant: int
    token_online_quant: int
    expert_online_quant: int
    mma_acc_dtype: object
    activation: object  # ActivationSpec | None
    output_quant: int | None
    has_bias: bool
    has_gammas: bool
    has_gather: bool
    has_x_static_scale: bool
    # Trailing, with a default: every construction site is positional, so a new field
    # anywhere else would silently reinterpret the existing ones.
    gate_up_split: bool = False
    epilogue: int = int(EpilogueMode.DEFAULT)


class TuningSpec(NamedTuple):
    """Plain-Python mirror of :class:`KernelTuningConfig`'s fields, carried as a single
    ``gl.constexpr`` leaf so the launch-time argument specializer walks one item.
    Field order must match the aggregate's constructor, skipping its
    leading ``func_cfg`` field -- that one is rebuilt from :class:`FuncSpec`."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    #: Requested unroll factor, rounded up to a multiple of every active register
    #: ring depth. LDS rings can keep runtime indices and impose no extra factor.
    K_UNROLL: int
    MINI_BLOCK_K: int
    MINI_BLOCK_M: int
    MINI_BLOCK_N: int
    NUM_LDS_BUFFER: int
    mfma_instr_shape: tuple
    warps_per_cta: tuple
    tiles_per_warp: tuple
    k_width: int
    transposed: bool
    WAVES_PER_EU: int
    TILE_SCHED: int
    GROUP_M: int
    NUM_XCDS: int
    token_mod: str
    token_scale_mod: str
    expert_mod: str
    expert_scale_mod: str
    result_mod: str
    result_scale_mod: str
    #: :class:`WarpPipeline` -- which ping-pong the K-loop step is built for. Was a
    #: bool; ``True``/``1`` is still ``COMPILER``, so old configs keep their meaning.
    WARP_PIPELINE: int
    VGPR_PREFETCH_K: int
    #: read operand A's scales from the moe_sort_scales pre-pass output (already in
    #: routing order and MFMA fragment order) rather than the raw (M, K/32) tensor
    A_SCALE_SORTED_SHUFFLED: bool = False
    #: read operand B's scales from a CDNA4_SCALE-preshuffled tensor. The weights are
    #: static, so unlike the A-side shuffle this costs nothing at run time.
    B_SCALE_SHUFFLED: bool = False
    #: read operand B from a 16-column-blocked weight tensor -- aiter's
    #: utils/shuffle.py::shuffle_weight(w, (16, 16)). The LDS tile takes the matching
    #: permutation, which drops the staging unit from 32 rows to 16 and removes its
    #: padding. Changes the operand contract, so the caller supplies the permuted tensor.
    B_PRESHUFFLED: bool = False
    #: use the hardware reciprocal in the fused SwiGLU instead of an IEEE divide,
    #: as the FlyDSL port does. ~1 ulp, well inside the bf16 the result is stored as.
    ACT_FAST_RCP: bool = False
    #: :class:`WaitCommitScheme` -- how coarsely the global->LDS copies are committed,
    #: and hence how many groups a ``wait_group`` count has to walk past.
    WAIT_COMMIT_SCHEME: int = int(WaitCommitScheme.PER_OP)
    #: :class:`DSReadOperand` mask; each payload and scale may move independently.
    DS_READ_IN_MFMA: int = int(DSReadOperand.NONE)
    SCHED_MODE: int = int(SchedMode.NONE)
    #: Run the preserved reference step within the common runtime pipeline driver.
    FROZEN_STEP: bool = False
    #: Advance HBM pointers per unrolled body and address its steps through soffset.
    SOFF_UNROLL: bool = False
    #: At a 2x2 mini-tile split, fill scales in the two middle slots.
    SCALE_FILL_MID: bool = False
    #: Load preshuffled B directly into registers at its global-load slot.
    B_IN_REG: bool = False
    #: Scale storage is independent of B payload storage and of the other scale.
    B_SCALE_IN_REG: bool = False
    A_SCALE_IN_REG: bool = False
    #: Per-component ring depths. Zero inherits NUM_LDS_BUFFER; every active
    #: component must resolve to at least two buffers, including register storage.
    A_NUM_BUFFER: int = 0
    B_NUM_BUFFER: int = 0
    A_SCALE_NUM_BUFFER: int = 0
    B_SCALE_NUM_BUFFER: int = 0


class ActivationSpec(NamedTuple):
    """Parameterisation of the fused gemm1 epilogue.

    Mirrors ``_triton_kernels/moe/activations.py::_swiglu`` exactly -- the kernel calls
    that very function, so the numerics cannot drift from the Triton path.
    """

    kind: int  # ActKind
    alpha: float  # 1.0 (GLM, DSv4) | 1.702 (M3)
    limit: float | None  # None (GLM) | 10.0 (DSv4) | 7.0 (M3)
    add_residual: bool  # False | False | True


@gluon.constexpr_function
def dq_has_scale(dq):
    return _unwrap(dq) in (int(DtypeQuant.MXFP4), int(DtypeQuant.MXFP8))


@gluon.constexpr_function
def dq_pack_divisor(dq):
    """Logical elements per stored container along K."""
    return 2 if _unwrap(dq) == int(DtypeQuant.MXFP4) else 1


@gluon.constexpr_function
def dq_mx_format(dq):
    """The ``a_format`` / ``b_format`` string ``mfma_scaled`` wants."""
    dq = _unwrap(dq)
    if dq == int(DtypeQuant.MXFP4):
        return "e2m1"
    if dq in (int(DtypeQuant.MXFP8), int(DtypeQuant.FP8_E4M3)):
        return "e4m3"
    return None


@gluon.constexpr_function
def dq_uses_mfma_scaled(dq_a, dq_b):
    """FP8 x FP8 must still go through ``mfma_scaled`` with ``a_scale=b_scale=None``:
    only that path reaches the double-rate K=64/128 ``V_MFMA_*_F8F6F4`` pipes; plain
    ``mfma`` selects the CDNA3-class K=16/32. Only bf16 x bf16 uses plain ``mfma``.
    """
    return not (
        _unwrap(dq_a) == int(DtypeQuant.BF16) and _unwrap(dq_b) == int(DtypeQuant.BF16)
    )


class NonQuantTokenTensor(NamedTuple):
    """bf16 activations. ``stride_k`` is pinned to 1 by the layout contract."""

    dtype_quant: gl.constexpr
    ptr: gl.tensor
    num_token: gl.tensor  # runtime
    stride_m: gl.tensor
    hidden_dim: gl.constexpr  # logical K extent
    topk: gl.constexpr

    @staticmethod
    def make(dtype_quant, ptr, num_token, stride_m, hidden_dim, topk):
        return NonQuantTokenTensor(
            _c(int(dtype_quant)), ptr, num_token, stride_m, _c(hidden_dim), _c(topk)
        )


class QuantTokenTensor(NamedTuple):
    """MXFP4 / MXFP8 activations plus their E8M0 scales."""

    dtype_quant: gl.constexpr
    ptr: gl.tensor
    scale_ptr: gl.tensor
    num_token: gl.tensor
    stride_m: gl.tensor  # payload elements, i.e. hidden_dim // 2 for MXFP4
    # Preserve row alignment below 16 bytes for direct-to-LDS scale copies.
    scale_stride_m: gl.constexpr
    scale_stride_k: gl.constexpr  # constexpr: folds the stage bump into the address
    hidden_dim: gl.constexpr  # logical extent; packed extent from dtype_quant
    topk: gl.constexpr
    scale_swizzle: gl.constexpr

    @staticmethod
    def make(
        dtype_quant,
        ptr,
        scale_ptr,
        num_token,
        stride_m,
        scale_stride_m,
        scale_stride_k,
        hidden_dim,
        topk,
        scale_swizzle,
    ):
        return QuantTokenTensor(
            _c(int(dtype_quant)),
            ptr,
            scale_ptr,
            num_token,
            stride_m,
            scale_stride_m,
            scale_stride_k,
            _c(hidden_dim),
            _c(topk),
            _c(int(scale_swizzle)),
        )


class NonQuantExpertTensor(NamedTuple):
    """bf16 weights, ``(E, K, N)`` with ``stride(-2) == 1``."""

    dtype_quant: gl.constexpr
    ptr: gl.tensor
    stride_e: gl.tensor
    stride_k: gl.tensor
    stride_n: gl.tensor
    num_expert: gl.tensor
    hidden_dim: gl.constexpr
    fused_intermediate_dim: gl.constexpr  # 2*I for stage 1, I for stage 2

    @staticmethod
    def make(
        dtype_quant, ptr, stride_e, stride_k, stride_n, num_expert, hidden_dim, fused
    ):
        return NonQuantExpertTensor(
            _c(int(dtype_quant)),
            ptr,
            stride_e,
            stride_k,
            stride_n,
            num_expert,
            _c(hidden_dim),
            _c(fused),
        )


class QuantExpertTensor(NamedTuple):
    """MXFP4 / MXFP8 weights, ``(E, K/2, N)`` with ``stride(-2) == 1``, plus scales.

    ``hidden_dim`` is always the *logical* K extent; the stored extent is
    ``hidden_dim // dq_pack_divisor(dtype_quant)``.
    """

    dtype_quant: gl.constexpr
    ptr: gl.tensor
    scale_ptr: gl.tensor
    stride_e: gl.tensor
    stride_k: gl.tensor
    stride_n: gl.tensor
    scale_stride_e: gl.tensor
    scale_stride_n: gl.tensor
    scale_stride_k: gl.constexpr  # constexpr: folds the stage bump into the address
    num_expert: gl.tensor
    hidden_dim: gl.constexpr
    fused_intermediate_dim: gl.constexpr
    scale_swizzle: gl.constexpr

    @staticmethod
    def make(
        dtype_quant,
        ptr,
        scale_ptr,
        stride_e,
        stride_k,
        stride_n,
        scale_stride_e,
        scale_stride_n,
        scale_stride_k,
        num_expert,
        hidden_dim,
        fused,
        scale_swizzle,
    ):
        return QuantExpertTensor(
            _c(int(dtype_quant)),
            ptr,
            scale_ptr,
            stride_e,
            stride_k,
            stride_n,
            scale_stride_e,
            scale_stride_n,
            scale_stride_k,
            num_expert,
            _c(hidden_dim),
            _c(fused),
            _c(int(scale_swizzle)),
        )


class ResultTensor(NamedTuple):
    """Destination of the GEMM.

    ``ptr`` addresses a ``(1, M, N)`` buffer -- ``reduce_grouped`` indexes the leading
    split-k axis even though ``split_k`` is unconditionally 1 here, so the axis must
    not be dropped. ``scale_ptr`` is non-None only for the fused MXFP4 output quant,
    in which case ``ptr`` holds the E2M1 payload (``N/2`` uint8 columns).
    """

    dtype_quant: gl.constexpr
    ptr: gl.tensor
    scale_ptr: gl.tensor
    stride_m: gl.tensor
    stride_n: gl.tensor
    scale_stride_m: gl.tensor
    scale_stride_n: gl.tensor
    out_dim: gl.constexpr  # emitted N (already halved for a gated activation)

    @staticmethod
    def make(
        dtype_quant,
        ptr,
        scale_ptr,
        stride_m,
        stride_n,
        scale_stride_m,
        scale_stride_n,
        out_dim,
    ):
        return ResultTensor(
            _c(int(dtype_quant)),
            ptr,
            scale_ptr,
            stride_m,
            stride_n,
            scale_stride_m,
            scale_stride_n,
            _c(out_dim),
        )


class RoutingMeta(NamedTuple):
    """Family-A routing convention (``RoutingData`` + ``reduce_grouped``)."""

    # packed (block_id << 16) | expt_id per launched pid; -1 means "no work, skip"
    expt_block_pid_map: gl.tensor
    expt_hist: gl.tensor  # int32 [n_expts_tot]
    expt_offs_raw: gl.tensor  # int32 [n_expts_tot + 1]
    expt_offs_sum: gl.tensor  # int32 scalar (token_offs_pad[-1])
    gather_indx: gl.tensor  # uint16 if n_gates <= 65535 else int32
    scatter_indx: gl.tensor  # may be None
    gammas: gl.tensor  # fp32 [n_gates]; may be None
    n_expts_act: gl.constexpr

    @staticmethod
    def make(
        expt_block_pid_map,
        expt_hist,
        expt_offs_raw,
        expt_offs_sum,
        gather_indx,
        scatter_indx,
        gammas,
        n_expts_act,
    ):
        return RoutingMeta(
            expt_block_pid_map,
            expt_hist,
            expt_offs_raw,
            expt_offs_sum,
            gather_indx,
            scatter_indx,
            gammas,
            _c(n_expts_act),
        )
