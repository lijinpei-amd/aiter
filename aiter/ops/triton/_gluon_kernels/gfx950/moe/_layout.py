# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operation-shape and data-layout contracts for the gfx950 Gluon MoE GEMMs.

The method-only aggregates are inherited by the public configuration aggregates in
``_config.py``. Keeping these methods on aggregates preserves the same host/device
constexpr API while isolating tile shapes, register/shared layouts, storage placement,
and their resource accounting from launch and pipeline policy.
"""

import os

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from ._lang import MX_GROUP, WARP_SIZE, ScaleSwizzle, require_constexpr
from ._lang import unwrap as _v

__all__ = [
    "LDS_CAP_BYTES",
    "LDS_EPILOGUE_RESERVE_BYTES",
    "LDS_USABLE_BYTES",
    "_blocked_b_hbm_offsets",
    "_n_split_offs",
    "_n_start",
    "_shuffled_scale_register_offsets",
    "_shuffled_scale_stage_offsets",
    "accumulator_shape",
    "byte_unit_lds_layout",
    "make_scale_swizzle_check",
]

#: Non-K extent of one A-scale fill tile, when it should differ from MINI_BLOCK_M.
#: 0 = follow MINI_BLOCK_M. Read here rather than inside the constexpr_function: those
#: bodies are traced by the Gluon compiler, which rejects os.environ.get.
_SCALE_MINI_M_ENV = int(
    os.environ.get("AITER_TRITON_MOE_GLUON_SCALE_MINI_BLOCK_M", "0")
)

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
    els = lambda nbytes: nbytes * 8 // elem_bits

    U = els(LDS_UNIT_K_BYTES)  # elements per 128 B K unit
    V = els(LDS_LANE_BYTES)  # elements per 16 B lane access
    rows, kelems = shape[nk_dim], shape[kd]

    K = lambda vs: _pow2_bases(kd, rank, vs)
    NK = lambda vs: _pow2_bases(nk_dim, rank, vs)

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


@gluon.constexpr_function
def accumulator_shape(tuning_cfg):
    """Shape of one mini-M by mini-N accumulator tile.

    This accepts the structural tuning-config interface used by the scalar pipeline
    tests as well as the production aggregate.
    """
    return [_v(tuning_cfg.MINI_BLOCK_M), _v(tuning_cfg.MINI_BLOCK_N)]


@gluon.jit
def _blocked_b_hbm_offsets(
    layout: gl.constexpr,
    packed_k: gl.constexpr,
    n0,
    mini_n: gl.constexpr,
    stored_k,
):
    """Byte offsets for a B tile in the 16-column-blocked HBM layout.

    ``utils/shuffle.py::shuffle_weight(w, (16, 16))`` maps logical ``(n, k)``
    to ``(n//16)*(stored_k*16) + (k//16)*256 + (n%16)*16 + k%16``. Keeping
    this physical-layout transform beside the layout that consumes it prevents
    the HBM and LDS permutations from drifting apart.
    """
    kk = gl.arange(0, packed_k, layout=gl.SliceLayout(1, layout))[:, None]
    nn = (n0 + gl.arange(0, mini_n, layout=gl.SliceLayout(0, layout)))[None, :]
    return (
        (nn // 16) * (stored_k * 16)
        + (kk // 16) * 256
        + (nn % 16) * 16
        + kk % 16
    )


@gluon.jit
def _shuffled_scale_register_offsets(
    layout: gl.constexpr,
    nonk0,
    nonk: gl.constexpr,
    scale_k: gl.constexpr,
    K,
    packed_k128: gl.constexpr = False,
):
    """HBM offsets for CDNA4_SCALE / SORTED_SHUFFLED scale storage.

    Each 32-row stripe stores 256 K values' scales in 256 bytes. Within a
    dword, non-K +16 advances one byte and K +128 advances two bytes. K128
    stages load the complete packed word, matching the LDS representation.
    """
    if require_constexpr(packed_k128):
        rr = gl.arange(0, nonk, layout=gl.SliceLayout(1, layout))[:, None]
        cc = gl.arange(0, 2, layout=gl.SliceLayout(0, layout))[None, :]
        word = rr * 2 + cc
        offsets = (nonk0 // 32 + word // 64) * K + (word % 64) * 4
    else:
        nn = nonk0 + gl.arange(0, nonk, layout=gl.SliceLayout(1, layout))[:, None]
        kk = gl.arange(0, scale_k, layout=gl.SliceLayout(0, layout))[None, :]
        offsets = (
            (nn // 32) * K
            + (kk // 8) * 256
            + (kk % 4) * 64
            + (nn % 16) * 4
            + ((kk % 8) // 4) * 2
            + (nn % 32) // 16
        )
    return offsets


@gluon.jit
def _shuffled_scale_stage_offsets(
    layout: gl.constexpr,
    nonk0,
    stripes: gl.constexpr,
    K,
):
    """Flat HBM offsets for shuffled scales starting at logical non-K row ``nonk0``."""
    stripe = gl.arange(0, stripes, layout=gl.SliceLayout(1, layout))[:, None]
    byte = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))[None, :]
    return (nonk0 // 32 + stripe) * K + byte


@gluon.jit
def _n_start(pid_n, ni: gl.constexpr, N, func_cfg, tuning_cfg):
    """Raw-N start of mini block ``ni`` under interleaved or split gate/up packing.

    Interleaved tiles are contiguous. Split tiles cover the same emitted channels but
    read gate and up from separate halves of N.
    """
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * tuning_cfg.MINI_BLOCK_N + ni * (N // 2)
    else:
        out = pid_n * tuning_cfg.BLOCK_N + ni * tuning_cfg.MINI_BLOCK_N
    return out


@gluon.jit
def _n_split_offs(pid_n, i, N, func_cfg, tuning_cfg):
    """Map a whole raw-N CTA index tensor through the gate/up packing."""
    mini_n: gl.constexpr = tuning_cfg.MINI_BLOCK_N
    if require_constexpr(func_cfg.gu_split()):
        out = pid_n * mini_n + (i // mini_n) * (N // 2) + i % mini_n
    else:
        out = pid_n * tuning_cfg.BLOCK_N + i
    return out


@aggregate
class _KernelFuncShape:
    """Operation-shape accessors inherited by ``KernelFuncConfig``."""

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


@aggregate
class _KernelTuningLayout:
    """Tile-shape and layout accessors inherited by ``KernelTuningConfig``."""

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

    @gluon.constexpr_function
    def num_lds_tiles(self, idx):
        """How many independently copied/read tiles one operand's stage is split into."""
        return self.num_mini_m() if _v(idx) == 0 else self.num_mini_n()

    @gluon.constexpr_function
    def payload_stage_k(self, idx):
        """Stored payload elements along K in one pipeline stage."""
        return _v(self.BLOCK_K) // self.func_cfg.pack_divisor(idx)

    @gluon.constexpr_function
    def scale_stage_k(self):
        """E8M0 elements along K in one pipeline stage."""
        return _v(self.BLOCK_K) // MX_GROUP

    @gluon.constexpr_function
    def payload_fragment_shape(self, idx):
        """Register payload shape for one non-K mini tile and mini-K step."""
        if _v(idx) == 0:
            return [
                _v(self.MINI_BLOCK_M),
                _v(self.MINI_BLOCK_K) // self.func_cfg.pack_divisor(0),
            ]
        return [
            _v(self.MINI_BLOCK_K) // self.func_cfg.pack_divisor(1),
            _v(self.MINI_BLOCK_N),
        ]

    @gluon.constexpr_function
    def scale_fragment_shape(self, idx):
        """Register E8M0 shape for one non-K mini tile and mini-K step."""
        nonk = _v(self.MINI_BLOCK_M) if _v(idx) == 0 else _v(self.MINI_BLOCK_N)
        return [nonk, _v(self.MINI_BLOCK_K) // MX_GROUP]

    @gluon.constexpr_function
    def accumulator_shape(self):
        """Shape of one mini-M by mini-N accumulator tile."""
        return accumulator_shape(self)

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
                self.payload_stage_k(0),
            ]
        return [
            self.payload_stage_k(1),
            _v(self.MINI_BLOCK_N),
        ]

    @gluon.constexpr_function
    def scale_shape(self, idx):
        """E8M0 scale tile of one operand, [mini non-K extent, BLOCK_K/32]."""
        non_k = _v(self.MINI_BLOCK_M) if _v(idx) == 0 else _v(self.MINI_BLOCK_N)
        return [non_k, self.scale_stage_k()]

    @gluon.constexpr_function
    def payload_lds_allocation_shape(self, idx):
        """Full payload LDS allocation, including ring and mini-tile axes."""
        return [self.num_buffers(idx) * self.num_lds_tiles(idx)] + self.lds_shape(idx)

    @gluon.constexpr_function
    def scale_lds_allocation_shape(self, idx):
        """Full scale LDS allocation in its physical staged representation."""
        idx = _v(idx)
        if self.scale_shuffled(idx):
            tiles = self.num_scale_tiles_a() if idx == 0 else self.num_lds_tiles(idx)
            tile_shape = self.scale_flat_shape(idx)
        else:
            tiles = self.num_lds_tiles(idx)
            tile_shape = self.scale_shape(idx)
        return [self.num_buffers(idx, True) * tiles] + tile_shape

    @gluon.constexpr_function
    def payload_fragment_offset(self, idx, mini_k):
        """Offset of one mini-K payload fragment within an LDS stage."""
        idx = _v(idx)
        width = _v(self.MINI_BLOCK_K) // self.func_cfg.pack_divisor(idx)
        if idx == 0:
            return [0, _v(mini_k) * width]
        return [_v(mini_k) * width, 0]

    @gluon.constexpr_function
    def scale_fragment_offset(self, mini_k):
        """Offset of one mini-K E8M0 fragment within a scale stage."""
        width = _v(self.MINI_BLOCK_K) // MX_GROUP
        return [0, _v(mini_k) * width]

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
        if self.scale_in_reg(idx):
            return False
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
        """Whether the payload uses a 128-bit-per-lane direct copy through LDS.

        Payload tiles that cannot give every wavefront one complete access stay in
        the register ring. There is deliberately no register-to-LDS fallback.
        """
        if _v(idx) == 1 and _v(self.B_IN_REG):
            return False
        shape = self.lds_shape(idx)
        vec = self.copy_contiguity(idx)
        return shape[0] * shape[1] >= WARP_SIZE * vec * self.num_warps()

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
    def payload_hbm_offset_layout(self, idx):
        """Layout used to form one payload tile's global-memory offsets."""
        if self.payload_via_lds(idx):
            return self.dot_operand_copy_layout(idx)
        return self.dot_operand_fragment_layout(idx)

    @gluon.constexpr_function
    def dot_operand_scale_fragment_layout(self, idx):
        idx = _v(idx)
        shape = self.scale_shape(idx)
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
    def scale_hbm_offset_layout(self, idx):
        """Layout used to form one scale tile's global-memory offsets."""
        if self.scale_via_lds(idx):
            return self.dot_operand_scale_copy_layout(idx)
        if self.scale_packed_k128(idx):
            return self.packed_scale_frag_layout(idx)
        if self.scale_shuffled(idx):
            return self.shuffled_scale_mem_layout(idx)
        return self.dot_operand_scale_fragment_layout(idx)

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
    def scale_flat_fragment_shape(self, idx):
        """Flat shuffled-scale shape corresponding to one payload mini tile."""
        return [self.scale_nonk(idx) // 32, 256]

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
    def mfma_scale_selector(self, idx, k_phase=0):
        """Packed-scale selector for the live MFMA path, or ``None``."""
        if self.num_mini_k() > 1:
            return None
        if self.scale_packed_ok(idx) and (
            self.scale_via_lds(idx) or self.scale_packed_k128(idx)
        ):
            return self.scale_packed_sel(idx, k_phase)
        return None

    @gluon.constexpr_function
    def has_mfma_packed_scale(self):
        """Whether either live MFMA scale operand uses packed dword storage."""
        return (
            self.mfma_scale_selector(0) is not None
            or self.mfma_scale_selector(1) is not None
        )

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
        shape = self.scale_shape(idx)
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

    # -- epilogue and emitted-output layouts --------------------------------------

    @gluon.constexpr_function
    def output_mini_n(self):
        """Emitted columns produced by one logical mini-N epilogue tile."""
        return _v(self.MINI_BLOCK_N) // self.func_cfg.mini_n_reduction()

    @gluon.constexpr_function
    def output_block_n(self):
        """Emitted columns produced by one CTA's raw BLOCK_N tile."""
        return _v(self.BLOCK_N) // self.func_cfg.activation_reduction_n()

    @gluon.constexpr_function
    def quant_payload_shape(self, rows, output_n=None):
        """Packed E2M1 byte shape for ``rows`` by emitted-output columns."""
        output_n = self.output_mini_n() if _v(output_n) is None else _v(output_n)
        return [_v(rows), output_n // 2]

    @gluon.constexpr_function
    def quant_scale_shape(self, rows, output_n=None):
        """E8M0 shape for ``rows`` by emitted-output columns."""
        output_n = self.output_mini_n() if _v(output_n) is None else _v(output_n)
        return [_v(rows), output_n // MX_GROUP]

    @gluon.constexpr_function
    def epilogue_threads(self):
        return self.num_warps() * WARP_SIZE

    @gluon.constexpr_function
    def epilogue_bias_per_thread(self):
        return _v(self.BLOCK_N) // self.epilogue_threads()

    @gluon.constexpr_function
    def epilogue_inputs_via_lds(self):
        """Whether bias/gamma vectors admit the supported direct-to-LDS copies."""
        threads = self.epilogue_threads()
        bpt = self.epilogue_bias_per_thread()
        return (
            threads >= _v(self.BLOCK_M)
            and _v(self.BLOCK_N) % threads == 0
            and bpt in (1, 4)
        )

    @gluon.constexpr_function
    def epilogue_input_lds_layout(self):
        return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])

    @gluon.constexpr_function
    def epilogue_gamma_copy_layout(self):
        return gl.BlockedLayout([1], [WARP_SIZE], [self.num_warps()], [0])

    @gluon.constexpr_function
    def epilogue_bias_copy_layout(self):
        return gl.BlockedLayout(
            [self.epilogue_bias_per_thread()],
            [WARP_SIZE],
            [self.num_warps()],
            [0],
        )

    @gluon.constexpr_function
    def epilogue_gamma_shape(self):
        return [self.epilogue_threads()]

    @gluon.constexpr_function
    def epilogue_bias_shape(self):
        return [_v(self.BLOCK_N)]

    @gluon.constexpr_function
    def quant_staging_layout(self):
        return gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

    @gluon.constexpr_function
    def quant_staging_rotates(self):
        return _v(self.func_cfg.output_quant) is not None and not self.func_cfg.gu_split()

    @gluon.constexpr_function
    def quant_staging_rows(self):
        return _v(self.BLOCK_M) if self.func_cfg.gu_split() else _v(self.MINI_BLOCK_M)

    @gluon.constexpr_function
    def quant_amax_lane_elems(self, block_m=None, output_n=None):
        """Inner lane-local width for the split MXFP4 amax reduction.

        The amax is an fp32 in-lane reduction followed by an integer cross-lane
        reduction. The fp32 half keeps ``abs`` as a free source modifier, while the
        integer half avoids the NaN-canonicalizing ``v_max_f32 x, x, x`` otherwise
        emitted before every cross-lane step.

        For the interleaved result this remains a conservative arithmetic proxy for
        the post-activation layout. The tuned four-wave TTGIR reshapes to
        ``tensor<64x2x2x16xf32>`` with reduction-axis register bases ``{1, 8}`` and
        lane bases ``{2, 4}``; returning 16 contains each lane's full span. A smaller
        split adds permlane work, while an overly fine one loses the three-input max
        fusion and adds copies around destructive lane swaps.

        Gate/up-split is derivable directly: one lane owns the transposed MFMA quad,
        ``instr_m * instr_n / WARP_SIZE`` consecutive emitted columns (four for
        16x16), before the first lane base.
        """
        block_m = _v(self.MINI_BLOCK_M) if _v(block_m) is None else _v(block_m)
        output_n = self.output_mini_n() if _v(output_n) is None else _v(output_n)
        if self.func_cfg.gu_split():
            instr = _v(self.mfma_instr_shape)
            return max(1, min(MX_GROUP, (instr[0] * instr[1]) // WARP_SIZE))
        per_lane = (block_m * output_n) // (self.num_warps() * WARP_SIZE)
        return min(MX_GROUP, max(1, per_lane))

    @gluon.constexpr_function
    def payload_hbm_step(self, idx):
        """Stored-element base-pointer increment for one payload K stage."""
        idx = _v(idx)
        step = self.payload_stage_k(idx)
        if idx == 1 and _v(self.B_PRESHUFFLED):
            return step // 16 * 256
        return step

    @gluon.constexpr_function
    def lds_bytes(self):
        fc = self.func_cfg
        total = 0
        for idx in (0, 1):
            shape = self.lds_shape(idx)
            width = fc.operand_elem_ty(idx).primitive_bitwidth // 8
            # lds_shape() is one mini block; a stage holds num_lds_tiles() of them
            n_tiles = self.num_lds_tiles(idx)
            if self.payload_via_lds(idx):
                total += n_tiles * shape[0] * shape[1] * width * self.num_buffers(idx)
            if fc.has_scale(idx) and (
                _v(self.FROZEN_STEP) or self.scale_via_lds(idx)
            ):
                s = self.scale_shape(idx)
                scale_k = 8 if self.scale_packed_k128(idx) else s[1]
                total += n_tiles * s[0] * scale_k * self.num_buffers(idx, True)
        return total

    @gluon.constexpr_function
    def acc_vgprs_per_lane(self):
        return _v(self.BLOCK_M) * _v(self.BLOCK_N) // (self.num_warps() * WARP_SIZE)

    @gluon.constexpr_function
    def validate_layout(self, N, K):
        """Validate storage placement and every tensor/tile shape contract."""
        fc = self.func_cfg
        BM, BN, BK = _v(self.BLOCK_M), _v(self.BLOCK_N), _v(self.BLOCK_K)
        N, K = _v(N), _v(K)
        instr = _v(self.mfma_instr_shape)
        warps = _v(self.warps_per_cta)
        tiles = _v(self.tiles_per_warp)

        assert not _v(self.B_IN_REG) or self.operand_preshuffled(
            1
        ), "B_IN_REG requires B_PRESHUFFLED"
        has_custom_storage = bool(
            _v(self.B_IN_REG)
            or _v(self.B_SCALE_IN_REG)
            or _v(self.A_SCALE_IN_REG)
            or _v(self.A_NUM_BUFFER)
            or _v(self.B_NUM_BUFFER)
            or _v(self.A_SCALE_NUM_BUFFER)
            or _v(self.B_SCALE_NUM_BUFFER)
        )
        assert not (_v(self.FROZEN_STEP) and has_custom_storage), (
            "FROZEN_STEP does not support register storage or per-component buffer counts"
        )
        if _v(self.FROZEN_STEP):
            for idx in (0, 1):
                assert self.payload_via_lds(idx), (
                    f"FROZEN_STEP requires operand {idx}'s payload in LDS; "
                    "use the live pipeline for a direct-register payload"
                )
        self.validate_buffer_counts()

        # -- host preconditions (no N tail, no K tail; only the even case exists) --
        assert N % BN == 0, f"N {N} % BLOCK_N {BN} != 0; the wrapper must fall back"
        assert K % BK == 0, f"K {K} % BLOCK_K {BK} != 0; the wrapper must fall back"

        # -- divisibility lattice --
        assert BK % _v(self.MINI_BLOCK_K) == 0
        assert _v(self.MINI_BLOCK_K) % instr[2] == 0
        assert BK % MX_GROUP == 0
        if _v(self.MINI_BLOCK_K) < BK and _v(self.FROZEN_STEP):
            for idx in (0, 1):
                assert not fc.has_scale(idx) or self.scale_via_lds(idx), (
                    f"FROZEN_STEP with MINI_BLOCK_K ({_v(self.MINI_BLOCK_K)}) < "
                    f"BLOCK_K ({BK}) requires operand {idx}'s scales in LDS"
                )

        # -- the constexpr rotating buffer index only folds if this holds --
        assert self.pipeline_unroll() >= 1, "K_UNROLL must be at least 1"
        if self.scale_packed_k128(0) or self.scale_packed_k128(1):
            assert self.scale_packed_k128(0) and self.scale_packed_k128(
                1
            ), "packed K128 scales require both operands to use packed scale words"
            assert K % 256 == 0, "packed K128 scales require complete K256 scale words"
            assert (
                _v(self.MINI_BLOCK_K) == BK
            ), "packed K128 scales require MINI_BLOCK_K == BLOCK_K"
            assert not _v(
                self.FROZEN_STEP
            ), "packed K128 scales require the live pipeline"
            assert list(instr) == [
                16,
                16,
                128,
            ], "packed K128 scales require MFMA 16x16x128"
            for idx in (0, 1):
                if self.scale_packed_k128(idx):
                    assert (
                        fc.pack_divisor(idx) == 1
                    ), "packed K128 scale pairing requires MXFP8 operands"
                    assert fc.has_scale(idx) and self.scale_packed_ok(
                        idx
                    ), f"operand {idx}'s packed K128 scales require register-private non-K +16"

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
            if _v(fc.output_quant) is not None:
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
        elif _v(fc.output_quant) is not None:
            arn = fc.activation_reduction_n()
            assert BN % (MX_GROUP * arn) == 0, (
                f"BLOCK_N {BN} must be a multiple of {MX_GROUP * arn} in raw "
                f"(pre-halving) terms so every MX group is tile-local"
            )
            assert _v(self.MINI_BLOCK_N) % (MX_GROUP * arn) == 0

        return True

    @gluon.constexpr_function
    def validate_layout_resources(self):
        """Validate layout-dependent LDS and accumulator resource budgets."""
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
