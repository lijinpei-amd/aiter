# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Functionality and tuning configuration aggregates for the gfx950 Gluon MoE GEMMs.

Both are ``@gluon.aggregate``s built entirely out of ``gl.constexpr`` scalars, so the
*same* object can be constructed on the host (for the grid tuple) and inside the kernel
(for the layouts) from the same numbers -- an aggregate itself cannot be a launch
argument, only its class can. Keeping the arithmetic in the aggregate is what stops the
host grid math and the device tile math from drifting.
"""

import os

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._lang import MX_GROUP, WARP_SIZE
from ._lang import unwrap as _v
from ._schedule import _buffer_load_groups

#: Non-K extent of one A-scale fill tile, when it should differ from MINI_BLOCK_M.
#: 0 = follow MINI_BLOCK_M. Read here rather than inside the constexpr_function: those
#: bodies are traced by the Gluon compiler, which rejects os.environ.get.
_SCALE_MINI_M_ENV = int(
    os.environ.get("AITER_TRITON_MOE_GLUON_SCALE_MINI_BLOCK_M", "0")
)
from ._types import (
    ActivationSpec,
    DotKind,
    DSReadOperand,
    DtypeQuant,
    EpilogueMode,
    ScaleSwizzle,
    SchedMode,
    WaitCommitScheme,
    WarpPipeline,
    dq_has_scale,
    dq_mx_format,
    dq_pack_divisor,
    dq_uses_mfma_scaled,
)

__all__ = ["KernelFuncConfig", "KernelTuningConfig"]

#: gfx950 LDS capacity, mirrors utils/_triton/arch_info.py::_LDS_CAP_BYTES["gfx950"].
LDS_CAP_BYTES = 163840
#: Headroom the operand buffers must leave for LDS the *compiler* allocates on top of
#: them -- the epilogue's convert_layout from the MFMA layout to the store layout needs
#: scratch that `lds_bytes()` cannot see. Empirical: a bf16 64x256 / BLOCK_K=128 tile
#: whose explicit buffers came to exactly 163840 B was rejected at launch at 166368 B.
#: Kept small on purpose -- the tuned MXFP4 configs sit at ~153 KiB and must not move.
LDS_EPILOGUE_RESERVE_BYTES = 4096
#: What the operand buffers may actually use.
LDS_USABLE_BYTES = LDS_CAP_BYTES - LDS_EPILOGUE_RESERVE_BYTES


@gluon.constexpr_function
def _bases_to_distributed(offset_bases, contiguity, num_warps, warp_size, shape):
    """Turn a ``PaddedSharedLayout``'s offset bases into the matching register layout.

    Mirrors Triton's ``CoalesceAsyncCopy`` partition -- lg2(C) bases to reg, lg2(WS) to
    lane, lg2(NW) to warp, leftovers back to reg -- which is what makes
    ``buffer_load_to_shared`` fold into a single ``buffer_load_dwordx4 ... lds`` instead
    of silently degrading to dword or failing to legalise. Same helper as
    ``gfx950/attention/fp8_mqa_logits.py::_offset_bases_to_blocked``.
    """
    rank = len(shape)
    lg2_c = contiguity.bit_length() - 1
    lg2_nw = num_warps.bit_length() - 1
    lg2_ws = warp_size.bit_length() - 1

    i = 0
    reg = list(offset_bases[i : i + lg2_c])
    i += lg2_c
    lane = list(offset_bases[i : i + lg2_ws])
    i += lg2_ws
    warp = list(offset_bases[i : i + lg2_nw])
    i += lg2_nw
    warp = warp + [[0] * rank] * (lg2_nw - len(warp))
    reg = reg + list(offset_bases[i:])
    return gl.DistributedLinearLayout(
        reg_bases=[list(b) for b in reg],
        lane_bases=[list(b) for b in lane],
        warp_bases=[list(b) for b in warp],
        block_bases=[],
        shape=list(shape),
    )


#: K extent of one LDS unit, in bytes. Fixed, for every dtype: the operand shape in
#: scope is 16x16x128 B, which is 16x16x128 for fp8 and 16x16x256 for packed fp4 (one
#: unit then spans two MFMA K-steps, and the layout is unaffected).
LDS_UNIT_K_BYTES = 128
#: The 16 B a lane moves per access -- the granule the preshuffled HBM order blocks by.
LDS_LANE_BYTES = 16
#: Padding of a non-preshuffled unit: 32 B every 1024 B.
LDS_PAD_INTERVAL_BYTES = 1024
LDS_PAD_BYTES = 32


@gluon.constexpr_function
def _pow2_bases(dim, rank, values):
    """``[[v, 0], ...]`` (or ``[[0, v], ...]``) for each stride in ``values``."""
    return [[v if i == dim else 0 for i in range(rank)] for v in values]


@gluon.constexpr_function
def _ramp(lo, hi):
    """Powers of two ``lo, 2*lo, ... hi/2``; empty when ``hi <= lo``."""
    out = []
    v = lo
    while v < hi:
        out.append(v)
        v *= 2
    return out


@gluon.constexpr_function
def byte_unit_lds_layout(shape, nk_dim, elem_bits, unit_rows, preshuffled):
    """LDS layout of a payload operand, tiled by a fixed (rows x 128 B) unit.

    The tile is read as ``(non-K units, K units, unit_rows, 128 B)`` with the rightmost
    axis contiguous, so the unit -- not the CTA tile -- is what fixes the layout. That
    is the difference from ``compute_efficient_padded_shared_layout``, which sizes its
    non-K unit from the whole tile and therefore hands a 128-row tile a different
    permutation than a 32-row one.

    Two orders, selected by whether the operand is already preshuffled in HBM:

    * not preshuffled -- 32 rows, the measured row permutation ``1, 4, 16, 2, 8``, and
      32 B of padding every 1024 B. A warp's 64 lanes cover exactly one 1024 B run, so
      the padding never splits a direct-to-LDS write.
    * preshuffled -- 16 rows in the byte order ``utils/shuffle.py::shuffle_weight(w,
      (16, 16))`` already writes, i.e. ``(n//16)*(KB*16) + (k//16)*256 + (n%16)*16 +
      k%16``. The copy is then a straight linear run and no padding is needed, because
      the 64 lanes of a fetch already land on 64 consecutive 16 B slots.

    ``shape`` and the returned bases are in *stored* elements; with the 8-bit storage
    both fp8 and packed fp4 use, one element is one byte.
    """
    shape = [int(x) for x in _v(shape)]
    nk_dim = _v(nk_dim)
    elem_bits = _v(elem_bits)
    unit_rows = _v(unit_rows)
    rank = len(shape)
    kd = 1 - nk_dim
    els = lambda nbytes: nbytes * 8 // elem_bits  # noqa: E731

    U = els(LDS_UNIT_K_BYTES)  # elements per 128 B K unit
    V = els(LDS_LANE_BYTES)  # elements per 16 B lane access
    rows, kelems = shape[nk_dim], shape[kd]

    K = lambda vs: _pow2_bases(kd, rank, vs)  # noqa: E731
    NK = lambda vs: _pow2_bases(nk_dim, rank, vs)  # noqa: E731

    if preshuffled:
        # 16 B of K, then the 16 rows, then the rest of the 128 B unit.
        bases = K(_ramp(1, V)) + NK(_ramp(1, unit_rows)) + K(_ramp(V, U))
    else:
        assert unit_rows == 32, "the 1, 4, 16, 2, 8 permutation is a 32-row order"
        bases = K(_ramp(1, U)) + NK([1, 4, 16, 2, 8])
    bases = bases + K(_ramp(U, kelems)) + NK(_ramp(unit_rows, rows))

    if preshuffled:
        # PaddedSharedLayout insists on at least one interval/padding pair, so name an
        # interval wider than the tile: no pad can land inside it. 16 rather than 1 so
        # that even if the allocator rounds the tile up, it stays ds_read_b128 aligned.
        pairs = [[1 << (rows * kelems).bit_length(), 16]]
    else:
        pairs = [[els(LDS_PAD_INTERVAL_BYTES), els(LDS_PAD_BYTES)]]
    return gl.PaddedSharedLayout(
        interval_padding_pairs=pairs,
        offset_bases=bases,
        cga_layout=[],
        shape=shape,
    )


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
    def act(self) -> ActivationSpec | None:
        return _v(self.activation)

    @gluon.constexpr_function
    def has_activation(self):
        return _v(self.activation) is not None

    @gluon.constexpr_function
    def activation_reduction_n(self):
        """Emitted columns per raw column. 2 for a gated activation, 1 otherwise."""
        return 2 if _v(self.activation) is not None else 1

    @gluon.constexpr_function
    def gu_split(self):
        """Is the gated pair laid out as two N halves rather than interleaved?

        Only meaningful with a gated activation -- the flag is inert otherwise, so it
        is folded in here instead of at every use site.
        """
        return bool(_v(self.gate_up_split)) and _v(self.activation) is not None

    @gluon.constexpr_function
    def mini_n_reduction(self):
        """Emitted columns per raw column *within one mini-N block*.

        The block-level ratio is always :meth:`activation_reduction_n`, but under
        ``gu_split`` the halving happens *across* two mini blocks rather than inside
        each one -- a mini block is a whole operand side, so its width maps 1:1 onto
        emitted channels and two of them collapse into one output tile.
        """
        return 1 if self.gu_split() else self.activation_reduction_n()

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

    @gluon.constexpr_function
    def num_async_loads_per_stage(self):
        """Loads issued per pipeline stage, i.e. what one ``commit_group`` covers.

        Dtype dependent: bf16 has no scale tensors, so it is 2 not 4. Never hardcode.
        """
        n = 2
        if dq_has_scale(self.token_dtype_quant):
            n += 1
        if dq_has_scale(self.expert_dtype_quant):
            n += 1
        return n


@aggregate
@strip_annotate
class KernelTuningConfig:
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
    token_mod: gl.constexpr
    token_scale_mod: gl.constexpr
    expert_mod: gl.constexpr
    expert_scale_mod: gl.constexpr
    result_mod: gl.constexpr
    result_scale_mod: gl.constexpr
    WARP_PIPELINE: gl.constexpr
    VGPR_PREFETCH_K: gl.constexpr
    A_SCALE_SORTED_SHUFFLED: gl.constexpr
    B_SCALE_SHUFFLED: gl.constexpr
    B_PRESHUFFLED: gl.constexpr
    ACT_FAST_RCP: gl.constexpr
    WAIT_COMMIT_SCHEME: gl.constexpr
    DS_READ_IN_MFMA: gl.constexpr
    SCHED_MODE: gl.constexpr
    FROZEN_STEP: gl.constexpr
    SOFF_UNROLL: gl.constexpr
    SCALE_FILL_MID: gl.constexpr

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
        token_mod,
        token_scale_mod,
        expert_mod,
        expert_scale_mod,
        result_mod,
        result_scale_mod,
        WARP_PIPELINE,
        VGPR_PREFETCH_K,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
        B_PRESHUFFLED=False,
        ACT_FAST_RCP=False,
        WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_OP),
        DS_READ_IN_MFMA=int(DSReadOperand.NONE),
        SCHED_MODE=int(SchedMode.NONE),
        FROZEN_STEP=False,
        SOFF_UNROLL=False,
        SCALE_FILL_MID=False,
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
        self.token_mod = gl.constexpr(_v(token_mod))
        self.token_scale_mod = gl.constexpr(_v(token_scale_mod))
        self.expert_mod = gl.constexpr(_v(expert_mod))
        self.expert_scale_mod = gl.constexpr(_v(expert_scale_mod))
        self.result_mod = gl.constexpr(_v(result_mod))
        self.result_scale_mod = gl.constexpr(_v(result_scale_mod))
        # int, not bool: WarpPipeline has three values and True still lands on COMPILER.
        self.WARP_PIPELINE = gl.constexpr(int(_v(WARP_PIPELINE)))
        self.VGPR_PREFETCH_K = gl.constexpr(int(_v(VGPR_PREFETCH_K)))
        self.A_SCALE_SORTED_SHUFFLED = gl.constexpr(bool(_v(A_SCALE_SORTED_SHUFFLED)))
        self.B_SCALE_SHUFFLED = gl.constexpr(bool(_v(B_SCALE_SHUFFLED)))
        self.B_PRESHUFFLED = gl.constexpr(bool(_v(B_PRESHUFFLED)))
        self.ACT_FAST_RCP = gl.constexpr(bool(_v(ACT_FAST_RCP)))
        self.WAIT_COMMIT_SCHEME = gl.constexpr(int(_v(WAIT_COMMIT_SCHEME)))
        self.DS_READ_IN_MFMA = gl.constexpr(int(_v(DS_READ_IN_MFMA)))
        self.SCHED_MODE = gl.constexpr(int(_v(SCHED_MODE)))
        self.FROZEN_STEP = gl.constexpr(bool(_v(FROZEN_STEP)))
        self.SOFF_UNROLL = gl.constexpr(bool(_v(SOFF_UNROLL)))
        self.SCALE_FILL_MID = gl.constexpr(bool(_v(SCALE_FILL_MID)))

    @gluon.constexpr_function
    def ds_read_in_mfma(self, operand, scale=False):
        """Whether one component's read is assigned to the MFMA region.

        ``operand`` is 0 for A (tokens), 1 for B (experts); ``scale`` selects its
        scale rather than its payload. A component absent from a dtype emits no read.
        """
        operand = _v(operand)
        assert operand in (0, 1), "operand must be 0 (A) or 1 (B)"
        bit = 1 << (operand + (2 if _v(scale) else 0))
        return bool(_v(self.DS_READ_IN_MFMA) & bit)

    @gluon.constexpr_function
    def num_warps(self):
        w = _v(self.warps_per_cta)
        return w[0] * w[1]

    @gluon.constexpr_function
    def grid_N(self, N):
        """Number of N tiles. Callable from BOTH host and device.

        Host: take ``.value`` off the returned constexpr before putting it in the grid
        tuple. grid-M comes from the routing metadata (``RoutingData.n_blocks``), never
        from ``cdiv``.
        """
        N = _v(N)
        BN = _v(self.BLOCK_N)
        assert N % BN == 0, f"N ({N}) must be a multiple of BLOCK_N ({BN})"
        return N // BN

    @gluon.constexpr_function
    def num_k_tiles(self, K):
        K = _v(K)
        BK = _v(self.BLOCK_K)
        assert K % BK == 0, f"K ({K}) must be a multiple of BLOCK_K ({BK})"
        return K // BK

    @gluon.constexpr_function
    def num_mini_k(self):
        """Mini-K steps per BLOCK_K stage."""
        return _v(self.BLOCK_K) // _v(self.MINI_BLOCK_K)

    @gluon.constexpr_function
    def num_prefetch_mini(self):
        """Mini-K steps held in registers across a pipeline step.

        ``num_mini_k()`` means the whole stage is carried (the MFMAs never touch a tile
        read in their own step); 0 means none is. Anything between splits the stage.
        """
        return _v(self.VGPR_PREFETCH_K) // _v(self.MINI_BLOCK_K)

    @gluon.constexpr_function
    def num_mini_m(self):
        """Mini-M row blocks per CTA tile. Operand A and the accumulator split by this."""
        return _v(self.BLOCK_M) // _v(self.MINI_BLOCK_M)

    @gluon.constexpr_function
    def num_mini_n(self):
        """Mini-N column blocks per CTA tile. Operand B and the accumulator split."""
        return _v(self.BLOCK_N) // _v(self.MINI_BLOCK_N)

    # -- inter-wave ping-pong (WarpPipeline) ----------------------------------------
    # The three modes share one step body; these only say which stage borders it lays
    # down. See _lang.pick_warp_pipeline_stage.

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
        return _v(self.WAIT_COMMIT_SCHEME) == int(WaitCommitScheme.PER_STAGE)

    @gluon.constexpr_function
    def commit_groups_per_stage(self):
        """Exact number of groups in the live stage's shared copy schedule."""
        return len(_buffer_load_groups(self))

    @gluon.constexpr_function
    def wait_at_stage_head(self):
        """PER_STAGE waits once at the stage head; other modes wait per read slot."""
        return self.commit_per_stage()

    @gluon.constexpr_function
    def num_lds_tiles(self, idx):
        """How many independently copied/read tiles one operand's stage is split into."""
        return self.num_mini_m() if _v(idx) == 0 else self.num_mini_n()

    @gluon.constexpr_function
    def lds_shape(self, idx):
        """Shared tile shape of one payload operand, in stored (packed) elements.

        This is a *mini* block, not the whole CTA tile: A is split along M into
        ``num_mini_m()`` tiles of ``MINI_BLOCK_M`` rows, B along N into ``num_mini_n()``
        tiles of ``MINI_BLOCK_N`` columns. Each tile is allocated, copied and read as an
        independent unit, so each gets its own efficient padded layout and its own fully
        coalesced direct-to-LDS copy -- slicing one big tile would instead hand the copy
        a strided view of a permuted layout.
        """
        if _v(idx) == 0:
            return [
                _v(self.MINI_BLOCK_M),
                _v(self.BLOCK_K) // self.func_cfg.pack_divisor(0),
            ]
        return [
            _v(self.BLOCK_K) // self.func_cfg.pack_divisor(1),
            _v(self.MINI_BLOCK_N),
        ]

    @gluon.constexpr_function
    def a_lds_shape(self):
        return self.lds_shape(0)

    @gluon.constexpr_function
    def b_lds_shape(self):
        return self.lds_shape(1)

    @gluon.constexpr_function
    def scale_shape(self, idx):
        """E8M0 scale tile of one operand, [mini non-K extent, BLOCK_K/32]."""
        non_k = _v(self.MINI_BLOCK_M) if _v(idx) == 0 else _v(self.MINI_BLOCK_N)
        return [non_k, _v(self.BLOCK_K) // MX_GROUP]

    @gluon.constexpr_function
    def a_scale_shape(self):
        return self.scale_shape(0)

    @gluon.constexpr_function
    def b_scale_shape(self):
        return self.scale_shape(1)

    @gluon.constexpr_function
    def copy_contiguity(self, idx):
        """Elements per lane for a 128-bit direct-to-LDS payload copy.

        Per operand: bf16 gives 8, fp8 and packed-fp4 (stored as uint8) give 16. Using
        one value for both would allocate and copy the fp4 weight tile as if it were
        bf16 in every mixed configuration.
        """
        return 128 // self.func_cfg.operand_elem_ty(idx).primitive_bitwidth

    @gluon.constexpr_function
    def k_width_for(self, idx):
        """``DotOperandLayout.k_width`` for one operand, in *stored* elements.

        It is 128 bits per lane per access, i.e. the same quantity as
        :meth:`copy_contiguity`: 16 for fp8 and for packed-fp4-in-uint8, 8 for bf16.
        Verified against a reference GEMM on both f8f6f4 instruction shapes -- k_width
        32 has no efficient padded shared layout at all, and 8 with the 16x16x128 shape
        compiles and runs but computes the wrong answer, which is exactly the class of
        bug no correctness-by-construction argument catches.

        The tuning field is an optional override, not the value itself.
        """
        override = _v(self.k_width)
        if override is not None:
            return override
        return 128 // self.func_cfg.operand_elem_ty(idx).primitive_bitwidth

    @gluon.constexpr_function
    def scale_via_lds(self, idx):
        """Whether an E8M0 scale tile can be written to LDS with a coalesced copy.

        CDNA4 direct-to-LDS supports only 128-bit or 32-bit per lane and a warp must
        write one contiguous run, so two tiles are excluded:

        * smaller than ``64 lanes * 4 B`` -- cannot be lowered at all (the A-scale tile
          at ``BLOCK_M == 16`` with a short ``BLOCK_K``);
        * a row shorter than 8 scales (``BLOCK_K < 256``) -- one lane then owns a whole
          row and the reg->shared map is more contiguous than the 32-bit access can
          cover, which ``canLoadDirectToLDS`` rejects.

        Both fall back to a register ``buffer_load`` straight into the scale fragment
        layout. That is correct but costs an in-loop register-path global access, which
        makes every ``wait_group`` conservative -- so the tuner should prefer a BLOCK_K
        that keeps this True.
        """
        shape = self.scale_shape(idx)
        if self.scale_shuffled(idx):
            # Fragment-ordered in HBM, so the copy is linear and the LDS tile keeps the
            # same permutation; the read is then a 32-bit ds_read. Staying on LDS also
            # keeps the warp-broadcast B tile a single fetch rather than one per warp.
            return True
        return (
            shape[0] * shape[1] >= WARP_SIZE * 4 and shape[1] % 4 == 0 and shape[1] >= 8
        )

    @gluon.constexpr_function
    def payload_via_lds(self, idx):
        """Same question for the payload operand at 128-bit per lane."""
        shape = self.lds_shape(idx)
        vec = self.copy_contiguity(idx)
        return shape[0] * shape[1] >= WARP_SIZE * vec

    @gluon.constexpr_function
    def dot_result_fragment_layout(self):
        """AMDMFMALayout for dot / scaled-dot results in register.

        ``transposed`` is pinned True: it gives each lane 4 consecutive N elements (a
        16 B contiguous store, and a lane-local even/odd gate-up pair for the fused
        activation) instead of 4 strided M rows.
        """
        assert _v(self.transposed), (
            "transposed=True is pinned, see the module docstring"
        )
        return gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=_v(self.mfma_instr_shape),
            transposed=True,
            warps_per_cta=_v(self.warps_per_cta),
            tiles_per_warp=_v(self.tiles_per_warp),
        )

    @gluon.constexpr_function
    def dot_operand_fragment_layout(self, idx):
        return gl.DotOperandLayout(
            operand_index=_v(idx),
            parent=self.dot_result_fragment_layout(),
            k_width=self.k_width_for(idx),
        )

    @gluon.constexpr_function
    def operand_preshuffled(self, idx):
        """Is this operand's payload already blocked 16-column in HBM?

        Never for A: the token rows are gathered per launch from the routing order, so
        there is no static permutation to bake in. For B it is the ``B_PRESHUFFLED``
        knob, which the caller honours by handing over the permuted weight tensor.
        """
        return _v(idx) == 1 and _v(self.B_PRESHUFFLED)

    @gluon.constexpr_function
    def lds_unit_rows(self, idx):
        """Non-K extent of one LDS unit.

        A preshuffled operand is blocked 16 rows at a time in HBM and the copy has to
        stay linear, so the unit is that block. A plain one is read two MFMA tiles at a
        time (hence the ``tiles_per_warp >= 2`` that :meth:`validate` insists on), so
        the unit is 32 and the row permutation has a 5th base to work with.
        """
        return 16 if self.operand_preshuffled(idx) else 32

    @gluon.constexpr_function
    def byte_unit_lds_ok(self, idx):
        """Does :func:`byte_unit_lds_layout` apply to this operand?

        Only the 16x16x128 B operand shape is in scope -- that is 16x16x128 for fp8 and
        16x16x256 for packed fp4, both stored 8-bit. The 32x32x64 prefill configs, the
        bf16 ones (16x16x32 / 32x32x16) and any tile too short for a whole unit keep
        ``compute_efficient_padded_shared_layout``.

        A non-preshuffled 32-row unit additionally needs the warp to own both of its
        16-row MFMA tiles, i.e. ``tiles_per_warp >= 2`` on that axis. That is not always
        reachable -- at ``BLOCK_N`` 64 with warps (1, 4) the CTA's whole N tile is four
        16-column tiles, so no warp can have two -- hence a gate rather than an assert.
        """
        idx = _v(idx)
        if list(_v(self.mfma_instr_shape)) != [16, 16, 128]:
            return False
        elem_bits = self.func_cfg.operand_elem_ty(idx).primitive_bitwidth
        if elem_bits != 8:
            return False
        if not self.operand_preshuffled(idx) and _v(self.tiles_per_warp)[idx] < 2:
            return False
        shape = self.lds_shape(idx)
        rows, kelems = shape[idx], shape[1 - idx]
        unit_rows = self.lds_unit_rows(idx)
        unit_k = LDS_UNIT_K_BYTES * 8 // elem_bits
        return rows % unit_rows == 0 and kelems % unit_k == 0

    @gluon.constexpr_function
    def dot_operand_lds_layout(self, idx):
        """Padded shared layout for a payload operand.

        Both operands are K-packed under the memory-layout contract, so
        ``is_k_contig=True`` and the plain ``smem.load(dot_layout)`` path applies --
        ``load_shared_fp4_repacked`` is only needed for an M/N-packed checkpoint, which
        none of the four models in scope produce.
        """
        idx = _v(idx)
        shape = self.lds_shape(idx)
        if self.byte_unit_lds_ok(idx):
            return byte_unit_lds_layout(
                shape,
                idx,
                self.func_cfg.operand_elem_ty(idx).primitive_bitwidth,
                self.lds_unit_rows(idx),
                self.operand_preshuffled(idx),
            )
        layout = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            self.dot_operand_fragment_layout(idx),
            shape,
            self.func_cfg.operand_elem_ty(idx),
            True,
        )
        if layout is not None:
            return layout
        # The helper declines whenever the tile holds a single MFMA tile along the
        # non-K axis (BLOCK_M == instr_shape[0], i.e. the whole decode regime): it has
        # no row permutation left to build. Fall back to a plain identity-mapped padded
        # layout whose interval is still >= vec * warpSize, which is the condition
        # `canLoadDirectToLDS` checks, so the copy stays a 128-bit direct-to-LDS.
        vec = self.copy_contiguity(idx)
        order = [1, 0] if idx == 0 else [0, 1]
        return gl.PaddedSharedLayout.with_identity_for(
            [[WARP_SIZE * vec, vec]], shape, order
        )

    @gluon.constexpr_function
    def dot_operand_copy_layout(self, idx):
        """Register layout of the global->LDS copy offsets for a payload operand."""
        idx = _v(idx)
        shape = self.lds_shape(idx)
        return _bases_to_distributed(
            self.dot_operand_lds_layout(idx).offset_bases,
            self.copy_contiguity(idx),
            self.num_warps(),
            WARP_SIZE,
            shape,
        )

    @gluon.constexpr_function
    def dot_operand_scale_fragment_layout(self, idx):
        idx = _v(idx)
        shape = self.a_scale_shape() if idx == 0 else self.b_scale_shape()
        return gl.amd.cdna4.get_mfma_scale_layout(
            self.dot_operand_fragment_layout(idx), shape, MX_GROUP
        )

    # -- ScaleSwizzle.SORTED_SHUFFLED --------------------------------------------
    # moe_sort_scales.cuh writes one dword per (chunk, mi, ku, k_lane, n_lane) with
    #     row = chunk*BM + (mi*MN_PACK + im_a)*16 + n_lane
    #     k   = ku*K_PACK*4 + ikxdl*4 + k_lane
    #     byte within the dword = ikxdl*MN_PACK + im_a
    # so n_lane/k_lane index the 64 lanes of a wavefront, im_a/ikxdl the 4 bytes a lane
    # holds, and mi the wave. That is exactly get_mfma_scale_layout's decomposition for
    # a 16x16x128 dot operand with warps_per_cta = (num_warps, 1) and
    # tiles_per_warp = (2, 1) -- verified base-for-base -- so the shuffled buffer is
    # read with a plain contiguous load and no cross-lane movement.

    @gluon.constexpr_function
    def sorted_shuffled_ok(self):
        """Does this config produce the layout the C++ shuffle was written against?"""
        # The permutation lives in the LDS tile's SharedLinearLayout, which is a
        # property of the byte order the shuffle writes -- not of how warps are split.
        # The warp arrangement is free. The shuffle always packs K256;
        # a K128 payload stage selects one half.
        return (
            list(_v(self.mfma_instr_shape)) == [16, 16, 128]
            and _v(self.BLOCK_K) in (128, 256)
            and _v(self.BLOCK_M) % 32 == 0
            # A mini block only has to be a whole number of 32-row stripes: the shuffle
            # writes one 256 B run per stripe, so a mini block is a contiguous slice.
            and _v(self.MINI_BLOCK_M) % 32 == 0
        )

    @gluon.constexpr_function
    def scale_shuffled(self, idx):
        """Is this operand's scale tile already in MFMA fragment order in memory?"""
        if _v(idx) == 0:
            return _v(self.A_SCALE_SORTED_SHUFFLED)
        return _v(self.B_SCALE_SHUFFLED)

    @gluon.constexpr_function
    def scale_packed_k128(self, idx):
        return _v(self.BLOCK_K) == 128 and self.scale_shuffled(idx)

    @gluon.constexpr_function
    def scale_k_phase(self, step):
        """Which K128 half of the packed K256 scale word this stage consumes."""
        if self.scale_packed_k128(0) or self.scale_packed_k128(1):
            return _v(step) % 2
        return 0

    @gluon.constexpr_function
    def scale_hbm_steps(self, idx, steps, phase=0):
        """Scale pointer displacement in units of one payload stage's scale stride."""
        if self.scale_packed_k128(idx):
            return ((_v(phase) + _v(steps)) // 2) * 2
        return _v(steps)

    @gluon.constexpr_function
    def shuffled_scale_mem_layout(self, idx):
        """The scale fragment layout, reordered so registers ascend with the address.

        ``get_mfma_scale_layout`` puts the K-group bit before the non-K bit in
        ``reg_bases``, so a lane's four bytes land in registers in address order
        0, +2, +1, +3. A widened (contiguity 4) load fills registers ascending, so it
        has to be issued at this permutation and converted afterwards -- a within-lane
        register renumber, no cross-lane traffic, because the lane and warp bases are
        untouched.
        """
        frag = self.dot_operand_scale_fragment_layout(idx)
        lead = [[16, 0], [0, 4]]
        rest = [list(b) for b in frag.reg_bases if list(b) not in lead]
        return gl.DistributedLinearLayout(
            reg_bases=lead + rest,
            lane_bases=[list(b) for b in frag.lane_bases],
            warp_bases=[list(b) for b in frag.warp_bases],
            block_bases=[],
            shape=list(frag.shape),
        )

    @gluon.constexpr_function
    def scale_mini_m(self):
        """Non-K extent of one A-scale *fill* tile -- decoupled from MINI_BLOCK_M.

        The payload wants a small mini block (it is what the mfma cluster consumes), but
        the scale copy wants a tile at least as tall as the copy layout is wide: that
        layout is warps_per_cta=[num_warps, 1] over an axis whose extent is
        ``nonk // 32`` stripes, so a tile of fewer stripes than warps is *replicated*
        across the surplus warps -- a half-empty buffer_load_dword at MINI_BLOCK_M=64
        with 4 waves, quarter-empty with 8. Sizing the scale tile independently lets the
        stripe count match the warp count while the payload keeps its own split.
        """
        env = _SCALE_MINI_M_ENV
        m = _v(self.MINI_BLOCK_M)
        if env and env % m == 0 and _v(self.BLOCK_M) % env == 0:
            return env
        return m

    @gluon.constexpr_function
    def num_scale_tiles_a(self):
        """A-scale fill tiles per stage (<= num_mini_m())."""
        return _v(self.BLOCK_M) // self.scale_mini_m()

    @gluon.constexpr_function
    def scale_tile_ratio_a(self):
        """Mini-M payload blocks sharing one A-scale tile."""
        return self.scale_mini_m() // _v(self.MINI_BLOCK_M)

    @gluon.constexpr_function
    def scale_nonk(self, idx):
        """Non-K extent of one shuffled scale tile.

        The A-side shuffle (SORTED_SHUFFLED) and CDNA4_SCALE both lay the tile out as
        one 256 B run per 32-row stripe, so a mini block that is a whole number of
        stripes is a contiguous slice of it and the tile can be the *mini* extent.
        """
        if _v(idx) == 0:
            return _v(self.MINI_BLOCK_M)
        return _v(self.MINI_BLOCK_N)

    @gluon.constexpr_function
    def scale_flat_shape(self, idx):
        """LDS staging shape of a shuffled scale tile: one 256 B run per 32-row stripe.

        The tile is staged in HBM byte order rather than fragment order because
        direct-to-LDS on gfx9 cannot scatter -- each warp must write coalesced, which a
        fragment-ordered SharedLinearLayout does not (canLoadDirectToLDS rejects it, see
        TritonAMDGPUToLLVM/Utility.cpp). The bytes therefore sit in (non-K 16, K 4) order
        while the u8 fragment numbers its registers (K 4, non-K 16); reconciling the two
        used to cost a v_perm per dword, which is what ``scale_packed_ok`` avoids by
        handing the dword to the MFMA whole and naming the byte order in a selector list.
        """
        nonk = self.scale_mini_m() if _v(idx) == 0 else self.scale_nonk(idx)
        return [nonk // 32, 256]

    @gluon.constexpr_function
    def shuffled_scale_read_layout(self, idx):
        """The [non-K, K] view of the flat LDS run, in fragment order."""
        nonk = self.scale_nonk(idx)
        bases = [[16, 0], [0, 4], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
        stripe = 32
        while stripe < nonk:
            bases = bases + [[stripe, 0]]
            stripe = stripe * 2
        return gl.SharedLinearLayout(offset_bases=bases)

    # -- packed (i32) scale operands ------------------------------------------------
    # Every shuffle here puts four scales in a dword, and the matrix instruction can
    # pick one of them with op_sel. So a lane loads the dword whole and hands it to
    # mfma_scaled_packed as an i32, with the byte order named in a selector list --
    # no reinterpret to u8, no register renumbering, no v_perm.
    #
    # CDNA4_SCALE / SORTED_SHUFFLED put non-K +16 and K +4 inside the dword, at byte
    # (nonK) + 2*(K). That fixed pair only works when the fragment happens to hold both
    # steps in registers, so ``scale_packed_ok`` checks it rather than assuming it.

    @gluon.constexpr_function
    def scale_dword_delta(self, delta):
        """Where a (non-K, K) step lands, as a coordinate of the i32 tile.

        The tile is [non-K, BLOCK_K/128] and its row-major linearisation *is* the dword
        index, so a step worth ``d`` dwords is the coordinate ``(d // 2, d % 2)``.
        """
        b = [int(x) for x in _v(delta)]
        if b == [0, 0]:
            return [0, 0]
        # CDNA4_SCALE writes byte (n//32)*256 + (k%4)*64 + (n%16)*4 + (k//4)*2 +
        # (n%32)//16; dropping the two within-dword terms and dividing by four
        # leaves the dword index below.
        d = (b[0] // 32) * 64 + (b[1] % 4) * 16 + (b[0] % 16)
        return [d // 2, d % 2]

    @gluon.constexpr_function
    def scale_packed_sel(self, idx, k_phase=0):
        """Byte selectors in MFMA order, for one K128 half or the full K256 word.

        The instruction index counts non-K major and K minor, so its low two bits are
        the fragment's first two register bases in that order. CDNA4_SCALE numbers the
        dword's bytes the other way round, hence the transposition.
        """
        if not self.scale_packed_ok(idx):
            return None
        if self.scale_packed_k128(idx):
            return [2 * _v(k_phase), 2 * _v(k_phase) + 1]
        return [0, 2, 1, 3]

    @gluon.constexpr_function
    def scale_packed_ok(self, idx):
        """Can this operand's scale fragment be fed as one dword per lane?

        K256 folds K+4 and non-K+16; K128 folds only non-K+16 and selects
        its K half separately. Each folded step must be register-private.
        """
        if not self.scale_shuffled(idx):
            return False
        regs = [list(b) for b in self.dot_operand_scale_fragment_layout(idx).reg_bases]
        if self.scale_packed_k128(idx):
            return len(regs) >= 1 and regs[0] == [16, 0]
        if len(regs) < 2:
            return False
        return regs[0] == [0, 4] and regs[1] == [16, 0]

    @gluon.constexpr_function
    def packed_scale_shape(self, idx):
        """Physical i32 storage, including both K128 halves even when one is unused."""
        nonk = self.scale_nonk(idx)
        return [nonk, max(256, _v(self.BLOCK_K)) // 128]

    @gluon.constexpr_function
    def packed_scale_frag_layout(self, idx):
        """The i32 fragment with its within-dword register bases folded in."""
        frag = self.dot_operand_scale_fragment_layout(idx)
        folded = 1 if self.scale_packed_k128(idx) else 2
        return gl.DistributedLinearLayout(
            reg_bases=[self.scale_dword_delta(b) for b in frag.reg_bases[folded:]],
            lane_bases=[self.scale_dword_delta(b) for b in frag.lane_bases],
            warp_bases=[self.scale_dword_delta(b) for b in frag.warp_bases],
            block_bases=[],
            shape=self.packed_scale_shape(idx),
        )

    @gluon.constexpr_function
    def packed_scale_read_layout(self, idx):
        """The i32 view of the flat LDS run: one offset bit per dword-address bit."""
        bases = list(self.shuffled_scale_read_layout(idx).offset_bases)[2:]
        return gl.SharedLinearLayout(
            offset_bases=[self.scale_dword_delta(b) for b in bases]
        )

    @gluon.constexpr_function
    def sorted_shuffled_c_k1(self, K):
        """Number of K256 scale groups per row stripe."""
        return _v(K) // 256

    @gluon.constexpr_function
    def sorted_shuffled_chunk_dwords(self, K):
        """Dwords per 128-row chunk; the stride from one chunk to the next."""
        return (_v(self.BLOCK_M) // 32) * self.sorted_shuffled_c_k1(K) * 4 * 16

    @gluon.constexpr_function
    def dot_operand_scale_lds_layout(self, idx):
        """Identity (K-contiguous) shared tile for the raw E8M0 scales.

        The three LDS goals are not simultaneously satisfiable for the scales: with
        ``get_mfma_scale_layout`` each lane wants K-scale elements strided by 2 or 4, so
        the read is ``ds_read_u8`` regardless. An identity tile is what keeps the
        *write* side (direct-to-LDS) coalesced, which is the side that matters.
        """
        # Flat staging tile -- plain identity; fragment order returns on the read.
        return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    @gluon.constexpr_function
    def dot_operand_scale_copy_layout(self, idx):
        """32-bit-per-lane blocked layout for the scale direct-to-LDS write."""
        idx = _v(idx)
        shape = self.a_scale_shape() if idx == 0 else self.b_scale_shape()
        if self.scale_shuffled(idx):
            # Flat run, 4 contiguous bytes per lane.
            return gl.BlockedLayout(
                size_per_thread=[1, 4],
                threads_per_warp=[1, WARP_SIZE],
                warps_per_cta=[self.num_warps(), 1],
                order=[1, 0],
            )
        sk = shape[1]
        lanes_k = sk // 4
        return gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[WARP_SIZE // lanes_k, lanes_k],
            warps_per_cta=[self.num_warps(), 1],
            order=[1, 0],
        )

    @gluon.constexpr_function
    def result_store_layout(self, block_m, block_n, elem_bits):
        """Blocked layout for the (masked) global store of one mini result tile.

        ``vec`` is capped by the elements each thread actually owns, not just by the
        128-bit access width. Without that cap a narrow tile is *over-covered*: the
        MXFP4 payload tile is [64, 32] uint8 = 2048 elements, but 256 threads x 16
        would be 4096, so the layout spanned 64x64 and the surplus warp column held a
        replica -- every payload store issued twice, writing the same bytes. With the
        cap it tiles exactly at both 4 and 8 waves.

        Covering *less* than the tile is fine and common (the bf16 path does): the
        layout simply repeats, one register set per repetition. Only over-coverage
        costs anything. It remains unavoidable for the E8M0 scale tile ([64, 2] is 128
        elements against 256 threads), which is why that one is still 2x over.
        """
        ept = max(1, (_v(block_m) * _v(block_n)) // (self.num_warps() * WARP_SIZE))
        vec = max(1, min(128 // _v(elem_bits), _v(block_n), ept))
        lanes_n = max(1, min(WARP_SIZE, _v(block_n) // vec))
        lanes_m = max(1, WARP_SIZE // lanes_n)
        warps_m = max(1, min(self.num_warps(), _v(block_m) // lanes_m))
        warps_n = max(1, self.num_warps() // warps_m)
        return gl.BlockedLayout(
            size_per_thread=[1, vec],
            threads_per_warp=[lanes_m, lanes_n],
            warps_per_cta=[warps_m, warps_n],
            order=[1, 0],
        )

    @gluon.constexpr_function
    def lds_bytes(self):
        fc = self.func_cfg
        per_stage = 0
        for idx in (0, 1):
            shape = self.lds_shape(idx)
            width = fc.operand_elem_ty(idx).primitive_bitwidth // 8
            # lds_shape() is one mini block; a stage holds num_lds_tiles() of them
            n_tiles = self.num_lds_tiles(idx)
            per_stage += n_tiles * shape[0] * shape[1] * width
            # LDSManager.alloc() allocates the scale buffer whenever the operand has
            # a scale, not only when it is filled by a direct-to-LDS copy, so the
            # budget has to count it the same way -- otherwise the host shrink loop
            # accepts a config whose real footprint is larger than it believes.
            if fc.has_scale(idx):
                s = self.scale_shape(idx)
                scale_k = 8 if self.scale_packed_k128(idx) else s[1]
                per_stage += n_tiles * s[0] * scale_k
        return per_stage * _v(self.NUM_LDS_BUFFER)

    @gluon.constexpr_function
    def acc_vgprs_per_lane(self):
        return _v(self.BLOCK_M) * _v(self.BLOCK_N) // (self.num_warps() * WARP_SIZE)

    @gluon.constexpr_function
    def validate(self, N, K):
        """Every precondition the kernel body then assumes. Constexpr asserts, not
        comments -- a violated one is either a miscompile or a silent over-read."""
        fc = self.func_cfg
        BM, BN, BK = _v(self.BLOCK_M), _v(self.BLOCK_N), _v(self.BLOCK_K)
        N, K = _v(N), _v(K)
        instr = _v(self.mfma_instr_shape)
        warps = _v(self.warps_per_cta)
        tiles = _v(self.tiles_per_warp)

        assert _v(fc.epilogue) in (
            int(EpilogueMode.DEFAULT),
            int(EpilogueMode.NOP_ACTIVATION),
            int(EpilogueMode.NOP),
        ), f"epilogue {_v(fc.epilogue)} is not an EpilogueMode"
        assert _v(self.DS_READ_IN_MFMA) >= 0 and not (
            _v(self.DS_READ_IN_MFMA) & ~int(DSReadOperand.ALL)
        ), f"DS_READ_IN_MFMA {_v(self.DS_READ_IN_MFMA)} has unknown operand bits"
        assert _v(self.SCHED_MODE) in (
            int(SchedMode.NONE),
            int(SchedMode.IGLP_0),
            int(SchedMode.IGLP_1),
            int(SchedMode.MFMA_16),
            int(SchedMode.MFMA_8),
        ), f"SCHED_MODE {_v(self.SCHED_MODE)} is not a SchedMode"

        # -- host preconditions (no N tail, no K tail; only the even case exists) --
        assert N % BN == 0, f"N {N} % BLOCK_N {BN} != 0; the wrapper must fall back"
        assert K % BK == 0, f"K {K} % BLOCK_K {BK} != 0; the wrapper must fall back"

        # -- divisibility lattice --
        assert BK % _v(self.MINI_BLOCK_K) == 0
        assert _v(self.MINI_BLOCK_K) % instr[2] == 0
        assert BK % MX_GROUP == 0
        if _v(self.MINI_BLOCK_K) < BK:
            # The LDS scale path slices per mini-tile; the register fallback advances a
            # flat offset by MINI_BLOCK_K/32 and hands mfma_scaled a fragment still
            # shaped for the whole BLOCK_K. The two disagree, and the only symptom is
            # the backend's "Operands must have the same scale factor" at lowering
            # time, which names neither knob. Refuse the pair here instead.
            for idx in (0, 1):
                assert not self.func_cfg.has_scale(idx) or self.scale_via_lds(idx), (
                    f"MINI_BLOCK_K ({_v(self.MINI_BLOCK_K)}) < BLOCK_K ({BK}) needs "
                    f"operand {idx}'s scale on the LDS path, but its tile is too small "
                    "for a coalesced direct-to-LDS copy; raise BLOCK_K or set "
                    "MINI_BLOCK_K == BLOCK_K"
                )

        # -- the constexpr rotating buffer index only folds if this holds --
        assert _v(self.K_UNROLL) >= 1, "K_UNROLL must be at least 1"
        if self.scale_packed_k128(0) or self.scale_packed_k128(1):
            assert self.scale_packed_k128(0) and self.scale_packed_k128(1), (
                "packed K128 scales require both operands to use packed scale words"
            )
            assert K % 256 == 0, "packed K128 scales require complete K256 scale words"
            assert _v(self.MINI_BLOCK_K) == BK, (
                "packed K128 scales require MINI_BLOCK_K == BLOCK_K"
            )
            assert _v(self.K_UNROLL) % 2 == 0, (
                "packed K128 scales require even K_UNROLL"
            )
            assert not _v(self.FROZEN_STEP), (
                "packed K128 scales require the live pipeline"
            )
            assert list(instr) == [16, 16, 128], (
                "packed K128 scales require MFMA 16x16x128"
            )
            for idx in (0, 1):
                if self.scale_packed_k128(idx):
                    assert fc.pack_divisor(idx) == 1, (
                        "packed K128 scale pairing requires MXFP8 operands"
                    )
                    assert fc.has_scale(idx) and self.scale_packed_ok(idx), (
                        f"operand {idx}'s packed K128 scales require register-private non-K +16"
                    )

        # -- the pipeline must issue exactly one fill per K tile --
        n_k = K // BK
        assert n_k >= _v(self.NUM_LDS_BUFFER), (
            f"K/BLOCK_K ({n_k}) < NUM_LDS_BUFFER ({_v(self.NUM_LDS_BUFFER)}): the "
            f"prologue alone would over-read past the end of the K strip"
        )
        n_fill = _v(self.NUM_LDS_BUFFER) + (n_k - _v(self.NUM_LDS_BUFFER))
        assert n_fill == n_k, "fill count must equal cdiv(K, BLOCK_K)"

        # -- the mini block is the unit of LDS allocation, of the global->LDS copy, of
        #    the MFMA and of the accumulator, so it must align to the CTA tiling;
        #    non-divisible values are illegal, not merely wasteful --
        assert BM % _v(self.MINI_BLOCK_M) == 0
        assert BN % _v(self.MINI_BLOCK_N) == 0
        # The fill schedule gives each slot of the NM x NN walk one mini-block copy, so
        # a stage needs at least as many slots as it has copies: NM * NN >= NM + NN,
        # which for integers means both axes split. The unsplit fallback that used to
        # cover NM == 1 / NN == 1 (fill A(mi) at ni == 0, B(ni) at mi == 0) is gone.
        # gluon_supported() checks this first so such a tile falls back gracefully;
        # reaching here means it was constructed some other way.
        assert self.num_mini_m() > 1 and self.num_mini_n() > 1, (
            f"the fill schedule needs both axes split (got num_mini_m "
            f"{self.num_mini_m()}, num_mini_n {self.num_mini_n()}): lower "
            f"MINI_BLOCK_M ({_v(self.MINI_BLOCK_M)} vs BLOCK_M {BM}) and MINI_BLOCK_N "
            f"({_v(self.MINI_BLOCK_N)} vs BLOCK_N {BN})"
        )
        assert _v(self.MINI_BLOCK_M) % (instr[0] * warps[0] * tiles[0]) == 0, (
            f"MINI_BLOCK_M {_v(self.MINI_BLOCK_M)} must be a multiple of "
            f"instr[0]*warps[0]*tiles[0] = {instr[0] * warps[0] * tiles[0]}"
        )
        assert _v(self.MINI_BLOCK_N) % (instr[1] * warps[1] * tiles[1]) == 0, (
            f"MINI_BLOCK_N {_v(self.MINI_BLOCK_N)} must be a multiple of "
            f"instr[1]*warps[1]*tiles[1] = {instr[1] * warps[1] * tiles[1]}"
        )

        # -- a mini block is also the granule of the global->LDS copy, so once an operand
        #    is actually split it must still give every warp a full 128-bit access.
        #    Below that, _bases_to_distributed runs out of bases for the warp dimension
        #    and pads it with zeros, which makes the surplus warps re-fetch and re-write
        #    a tile another warp already owns: correct, but pure wasted HBM traffic. --
        for idx in (0, 1):
            if self.num_lds_tiles(idx) > 1 and self.payload_via_lds(idx):
                tile = self.lds_shape(idx)
                vec = self.copy_contiguity(idx)
                need = WARP_SIZE * vec * self.num_warps()
                assert tile[0] * tile[1] >= need, (
                    f"operand {idx}'s mini tile {tile} holds {tile[0] * tile[1]} "
                    f"elements, below the {need} one 128-bit access per warp needs; "
                    f"raise MINI_BLOCK_{'M' if idx == 0 else 'N'} or BLOCK_K"
                )

        # -- MFMA shape must fit the tile --
        assert BM % (instr[0] * warps[0] * tiles[0]) == 0
        assert BN % (instr[1] * warps[1] * tiles[1]) == 0

        # -- the LDS unit the byte-tiled layout is built from --
        # A non-preshuffled unit is 32 rows because a warp reads two 16-row MFMA tiles
        # from it; at tiles_per_warp 1 the second half of every unit belongs to another
        # warp, so the 1, 4, 16, 2, 8 permutation would spread one warp's rows over two
        # padding intervals instead of one. byte_unit_lds_ok() gates on exactly this, so
        # the assert can only fire if that gate is ever loosened without this being
        # revisited -- which is the point of stating it here.
        for idx in (0, 1):
            if self.byte_unit_lds_ok(idx) and not self.operand_preshuffled(idx):
                assert tiles[idx] >= 2, (
                    f"operand {idx} is not preshuffled, so its LDS unit is 32 rows and "
                    f"tiles_per_warp[{idx}] must be >= 2 (got {tiles[idx]})"
                )
        if self.operand_preshuffled(1):
            # The 16-column-blocked global offsets only stay inside the mini tile while
            # every extent is a whole number of 16-column, 16-byte blocks.
            pk_b = BK // fc.b_pack_divisor()
            assert BN % 16 == 0 and _v(self.MINI_BLOCK_N) % 16 == 0, (
                f"B_PRESHUFFLED needs BLOCK_N ({BN}) and MINI_BLOCK_N "
                f"({_v(self.MINI_BLOCK_N)}) to be multiples of the 16-column block"
            )
            assert pk_b % 16 == 0, (
                f"B_PRESHUFFLED needs BLOCK_K/pack ({pk_b}) to be a multiple of the "
                "16-byte block"
            )

        # -- scale swizzle --
        # (checked by the caller against the tensor's scale_swizzle field)

        # -- gate/up split: a mini-N block must be exactly one operand side --
        if fc.gu_split():
            MBN = _v(self.MINI_BLOCK_N)
            assert self.num_mini_n() == 2 and MBN * 2 == BN, (
                f"gate_up_split needs exactly two mini-N blocks, one per side: got "
                f"BLOCK_N {BN} / MINI_BLOCK_N {MBN} = {self.num_mini_n()}"
            )
            assert N % 2 == 0 and (N // 2) % MBN == 0, (
                f"gate_up_split needs each half of N ({N}) to be a whole number of "
                f"MINI_BLOCK_N ({MBN}) tiles"
            )
            # The whole point of the layout: a warp's emitted N extent has to contain a
            # whole MX group, or the fused quant's amax crosses the warp boundary. That
            # extent is instr_n * tiles_per_warp_n -- warps tile *above* it, so the
            # warp count is irrelevant here.
            if fc.output_quant is not None:
                warp_n = instr[1] * tiles[1]
                assert warp_n % MX_GROUP == 0, (
                    f"gate_up_split with a fused MX output quant needs a warp's "
                    f"emitted N extent (instr_n {instr[1]} * tiles_per_warp_n "
                    f"{tiles[1]} = {warp_n}) to be a multiple of {MX_GROUP}, else the "
                    "amax reduction still crosses warps"
                )
                assert MBN % MX_GROUP == 0

        # -- the fused MX output quant groups 32 *emitted* columns, so the raw tile has
        #    to carry 32 * activation_reduction_n of them --
        elif fc.output_quant is not None:
            arn = fc.activation_reduction_n()
            assert BN % (MX_GROUP * arn) == 0, (
                f"BLOCK_N {BN} must be a multiple of {MX_GROUP * arn} in raw "
                f"(pre-halving) terms so every MX group is tile-local"
            )
            assert _v(self.MINI_BLOCK_N) % (MX_GROUP * arn) == 0

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
            assert self.num_prefetch_mini() == self.num_mini_k(), (
                f"WARP_PIPELINE needs VGPR_PREFETCH_K == BLOCK_K (got "
                f"{_v(self.VGPR_PREFETCH_K)} vs {BK}): with a partial window the slot's "
                "MFMAs depend on the slot's own ds_read and there is no mem stage to "
                "hide behind them"
            )

        # -- pipeline wait depth --
        # The loop is wait-first / commit-last and always copies into a buffer no live
        # stage occupies, so on every step one buffer is being filled by buffer_load and
        # one is being consumed by ds_read: the wait is NUM_LDS_BUFFER-2, and two
        # buffers leave nothing to overlap with (the wait folds to wait_group(0) and
        # drains every copy on every stage -- measured 1217 us against 1028 us).
        assert _v(self.NUM_LDS_BUFFER) >= 3, (
            f"NUM_LDS_BUFFER {_v(self.NUM_LDS_BUFFER)} < 3: one buffer is being filled "
            "and one consumed on every step, leaving nothing to overlap a copy with"
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
            int(WaitCommitScheme.PER_STAGE),
        ), (
            f"WAIT_COMMIT_SCHEME {_v(self.WAIT_COMMIT_SCHEME)} is not a WaitCommitScheme: "
            f"PER_OP {int(WaitCommitScheme.PER_OP)}, "
            f"PER_SLOT {int(WaitCommitScheme.PER_SLOT)}, "
            f"PER_STAGE {int(WaitCommitScheme.PER_STAGE)}"
        )

        # -- resource budgets --
        lds = self.lds_bytes()
        assert lds <= LDS_USABLE_BYTES, (
            f"LDS {lds} B > {LDS_USABLE_BYTES} B usable ({LDS_CAP_BYTES} cap minus "
            f"{LDS_EPILOGUE_RESERVE_BYTES} B of epilogue scratch): reduce "
            f"NUM_LDS_BUFFER ({_v(self.NUM_LDS_BUFFER)}) or the block sizes"
        )
        acc = self.acc_vgprs_per_lane()
        assert acc <= 256, (
            f"fp32 accumulator alone needs {acc} VGPR/lane; 256 is the architectural "
            f"cap and WAVES_PER_EU=2 needs <= 256 total"
        )
        return True


@gluon.constexpr_function
def make_scale_swizzle_check(scale_swizzle, BLOCK_K):
    """``CDNA4_SCALE`` keeps the direct-to-LDS write coalesced but costs BLOCK_K>=256."""
    if _v(scale_swizzle) == int(ScaleSwizzle.CDNA4_SCALE):
        assert _v(BLOCK_K) >= 256, (
            "CDNA4_SCALE preshuffle needs MX_SCALE_BLOCK_K >= 8, i.e. BLOCK_K >= 256"
        )
    return True
