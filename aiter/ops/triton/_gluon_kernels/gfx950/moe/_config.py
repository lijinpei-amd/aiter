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

from aiter.ops.triton.utils.common_utils import strip_annotate

from ._lang import MX_GROUP, WARP_SIZE
from ._lang import unwrap as _v
from ._types import (
    ActivationSpec,
    DotKind,
    DtypeQuant,
    ScaleSwizzle,
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
    MINI_PREFETCH_K: gl.constexpr
    MINI_BLOCK_M: gl.constexpr
    MINI_BLOCK_N: gl.constexpr
    MINI_PRESTORE_MN: gl.constexpr
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

    @gluon.constexpr_function
    def __init__(
        self,
        func_cfg,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        K_UNROLL,
        MINI_BLOCK_K,
        MINI_PREFETCH_K,
        MINI_BLOCK_M,
        MINI_BLOCK_N,
        MINI_PRESTORE_MN,
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
    ):
        self.func_cfg = func_cfg
        self.BLOCK_M = gl.constexpr(_v(BLOCK_M))
        self.BLOCK_N = gl.constexpr(_v(BLOCK_N))
        self.BLOCK_K = gl.constexpr(_v(BLOCK_K))
        self.K_UNROLL = gl.constexpr(_v(K_UNROLL))
        self.MINI_BLOCK_K = gl.constexpr(_v(MINI_BLOCK_K))
        self.MINI_PREFETCH_K = gl.constexpr(_v(MINI_PREFETCH_K))
        self.MINI_BLOCK_M = gl.constexpr(_v(MINI_BLOCK_M))
        self.MINI_BLOCK_N = gl.constexpr(_v(MINI_BLOCK_N))
        self.MINI_PRESTORE_MN = gl.constexpr(_v(MINI_PRESTORE_MN))
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
        self.WARP_PIPELINE = gl.constexpr(_v(WARP_PIPELINE))

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
    def lds_shape(self, idx):
        """Shared tile shape of one payload operand, in stored (packed) elements."""
        if _v(idx) == 0:
            return [_v(self.BLOCK_M), _v(self.BLOCK_K) // self.func_cfg.pack_divisor(0)]
        return [_v(self.BLOCK_K) // self.func_cfg.pack_divisor(1), _v(self.BLOCK_N)]

    @gluon.constexpr_function
    def a_lds_shape(self):
        return self.lds_shape(0)

    @gluon.constexpr_function
    def b_lds_shape(self):
        return self.lds_shape(1)

    @gluon.constexpr_function
    def scale_shape(self, idx):
        """E8M0 scale tile of one operand, always [non-K extent, BLOCK_K/32]."""
        non_k = _v(self.BLOCK_M) if _v(idx) == 0 else _v(self.BLOCK_N)
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
        assert _v(
            self.transposed
        ), "transposed=True is pinned, see the module docstring"
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
    def dot_operand_lds_layout(self, idx):
        """Padded shared layout for a payload operand.

        Both operands are K-packed under the memory-layout contract, so
        ``is_k_contig=True`` and the plain ``smem.load(dot_layout)`` path applies --
        ``load_shared_fp4_repacked`` is only needed for an M/N-packed checkpoint, which
        none of the four models in scope produce.
        """
        idx = _v(idx)
        shape = self.lds_shape(idx)
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

    @gluon.constexpr_function
    def dot_operand_scale_lds_layout(self, idx):
        """Identity (K-contiguous) shared tile for the raw E8M0 scales.

        The three LDS goals are not simultaneously satisfiable for the scales: with
        ``get_mfma_scale_layout`` each lane wants K-scale elements strided by 2 or 4, so
        the read is ``ds_read_u8`` regardless. An identity tile is what keeps the
        *write* side (direct-to-LDS) coalesced, which is the side that matters.
        """
        return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    @gluon.constexpr_function
    def dot_operand_scale_copy_layout(self, idx):
        """32-bit-per-lane blocked layout for the scale direct-to-LDS write."""
        idx = _v(idx)
        shape = self.a_scale_shape() if idx == 0 else self.b_scale_shape()
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
        """Blocked layout for the (masked) global store of one mini result tile."""
        vec = max(1, min(128 // _v(elem_bits), _v(block_n)))
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
            per_stage += shape[0] * shape[1] * width
            # LDSManager.alloc() allocates the scale buffer whenever the operand has
            # a scale, not only when it is filled by a direct-to-LDS copy, so the
            # budget has to count it the same way -- otherwise the host shrink loop
            # accepts a config whose real footprint is larger than it believes.
            if fc.has_scale(idx):
                s = self.scale_shape(idx)
                per_stage += s[0] * s[1]
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

        # -- host preconditions (no N tail, no K tail; only the even case exists) --
        assert N % BN == 0, f"N {N} % BLOCK_N {BN} != 0; the wrapper must fall back"
        assert K % BK == 0, f"K {K} % BLOCK_K {BK} != 0; the wrapper must fall back"

        # -- divisibility lattice --
        assert BK % _v(self.MINI_BLOCK_K) == 0
        assert _v(self.MINI_BLOCK_K) % instr[2] == 0
        assert _v(self.MINI_PREFETCH_K) < BK // _v(self.MINI_BLOCK_K)
        assert BK % MX_GROUP == 0
        if _v(self.MINI_PREFETCH_K) > 0:
            assert (
                _v(self.K_UNROLL) >= 2
            ), "MINI_PREFETCH_K needs the next stage's index"
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
        assert (
            _v(self.K_UNROLL) % _v(self.NUM_LDS_BUFFER) == 0
        ), "K_UNROLL must be a multiple of NUM_LDS_BUFFER or the unroll buys nothing"

        # -- the pipeline must issue exactly one fill per K tile --
        n_k = K // BK
        assert n_k >= _v(self.NUM_LDS_BUFFER), (
            f"K/BLOCK_K ({n_k}) < NUM_LDS_BUFFER ({_v(self.NUM_LDS_BUFFER)}): the "
            f"prologue alone would over-read past the end of the K strip"
        )
        n_fill = _v(self.NUM_LDS_BUFFER) + (n_k - _v(self.NUM_LDS_BUFFER))
        assert n_fill == n_k, "fill count must equal cdiv(K, BLOCK_K)"

        # -- gl.amd.slice is register-only and layout-preserving, so the mini tile must
        #    align to the CTA tiling; non-divisible values are illegal, not wasteful --
        assert BM % _v(self.MINI_BLOCK_M) == 0
        assert BN % _v(self.MINI_BLOCK_N) == 0
        assert _v(self.MINI_BLOCK_M) % (instr[0] * warps[0] * tiles[0]) == 0, (
            f"MINI_BLOCK_M {_v(self.MINI_BLOCK_M)} must be a multiple of "
            f"instr[0]*warps[0]*tiles[0] = {instr[0] * warps[0] * tiles[0]}"
        )
        assert _v(self.MINI_BLOCK_N) % (instr[1] * warps[1] * tiles[1]) == 0, (
            f"MINI_BLOCK_N {_v(self.MINI_BLOCK_N)} must be a multiple of "
            f"instr[1]*warps[1]*tiles[1] = {instr[1] * warps[1] * tiles[1]}"
        )
        assert _v(self.MINI_PRESTORE_MN) <= (BM // _v(self.MINI_BLOCK_M)) * (
            BN // _v(self.MINI_BLOCK_N)
        )
        assert _v(self.MINI_PRESTORE_MN) == 1, (
            "MINI_PRESTORE_MN only means something when results are staged through LDS. "
            "This epilogue does the MX group amax in registers -- the transposed MFMA "
            "accumulator already gives each lane 4 consecutive N, so the reduction over "
            "the remaining lanes is a reshape, not an LDS round trip -- and stores each "
            "mini tile straight to HBM, so there is nothing to pre-stage."
        )

        # -- MFMA shape must fit the tile --
        assert BM % (instr[0] * warps[0] * tiles[0]) == 0
        assert BN % (instr[1] * warps[1] * tiles[1]) == 0

        # -- scale swizzle --
        # (checked by the caller against the tensor's scale_swizzle field)

        # -- the fused MX output quant groups 32 *emitted* columns, so the raw tile has
        #    to carry 32 * activation_reduction_n of them --
        if fc.output_quant is not None:
            arn = fc.activation_reduction_n()
            assert BN % (MX_GROUP * arn) == 0, (
                f"BLOCK_N {BN} must be a multiple of {MX_GROUP * arn} in raw "
                f"(pre-halving) terms so every MX group is tile-local"
            )
            assert _v(self.MINI_BLOCK_N) % (MX_GROUP * arn) == 0

        # -- warp pipelining --
        # `with gl.amd.warp_pipeline_stage(label, priority)` is the right mechanism, but
        # TritonAMDGPUWarpPipeline rejects any barrier or wait inside a stage region and
        # a non-relaxed `smem.load` emits one. `load_shared_relaxed` would remove it, but
        # it is unsafe here: this pipeline refills the very buffer it has just consumed,
        # and suppressing the wait in front of the LDS read corrupts the result (measured
        # as a 10% RMS error). So the knob stays, and stays off, until the pipeline is
        # restructured to write into a buffer no MFMA is still reading.
        assert not _v(self.WARP_PIPELINE), (
            "WARP_PIPELINE is not usable with a consume-then-refill pipeline: the "
            "wait_group and the LDS reads cannot both live inside a stage region"
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
        assert (
            _v(BLOCK_K) >= 256
        ), "CDNA4_SCALE preshuffle needs MX_SCALE_BLOCK_K >= 8, i.e. BLOCK_K >= 256"
    return True
