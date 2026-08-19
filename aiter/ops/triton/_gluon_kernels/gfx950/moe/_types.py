# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dtype / quant-scheme enums and the host->device NamedTuples of the gfx950 Gluon MoE
grouped GEMMs.
"""

from enum import IntEnum
from typing import NamedTuple

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import const as _c
from ._lang import unwrap as _unwrap

__all__ = [
    "ActKind",
    "ActivationSpec",
    "DotKind",
    "DtypeQuant",
    "FuncSpec",
    "NonQuantExpertTensor",
    "NonQuantTokenTensor",
    "QuantExpertTensor",
    "QuantTokenTensor",
    "ResultTensor",
    "RoutingMeta",
    "ScaleSwizzle",
    "TileSched",
    "TuningSpec",
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


class ActKind(IntEnum):
    SILU = 0  # alpha == 1.0
    SWIGLU_OAI = 1  # alpha != 1.0


class DotKind(IntEnum):
    """Which CDNA4 matrix instruction an operand pair maps onto."""

    MFMA = 0  # bf16 x bf16
    MFMA_SCALED = 1  # any fp4/fp8 pair, incl. FP8 x FP8 with unit scales
    UPCAST_MFMA = 2  # bf16 x microscaled: scaled_upcast, then plain mfma


class TileSched(IntEnum):
    LINEAR = 0  # plain row-major (pid_m, pid_n), no swizzle
    GROUP_M = 1  # pid_grid GROUP_M blocking, for L2 reuse of the token tile
    XCD_GROUP_M = 2  # remap_xcd first, then GROUP_M blocking


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


class TuningSpec(NamedTuple):
    """Plain-Python mirror of :class:`KernelTuningConfig`'s fields, carried as a single
    ``gl.constexpr`` leaf so the launch-time argument specializer walks one item instead
    of twenty-six. Field order must match the aggregate's constructor, skipping its
    leading ``func_cfg`` field -- that one is rebuilt from :class:`FuncSpec`."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
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
    WARP_PIPELINE: bool
    VGPR_PREFETCH_K: int


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
    scale_stride_m: gl.tensor
    scale_stride_k: gl.tensor
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
    scale_stride_k: gl.tensor
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
