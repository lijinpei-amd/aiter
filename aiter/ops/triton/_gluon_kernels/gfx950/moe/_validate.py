# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build-time validation for the gfx950 Gluon MoE GEMMs.

Every precondition the kernel body then assumes, as constexpr asserts rather than
comments -- a violated one is either a miscompile or a silent over-read. The kernel
``static_assert``s these and the host runs the same functions before launch, so both
reject exactly the same configurations.

Kept apart from what it checks: the tuning config only rephrases its fields, the
layout only derives shapes and placement, and the schedule only models the pipeline.
Validation reads all three, so it sits after them in the import order.
"""

from triton.experimental import gluon

from ._lang import (
    MX_GROUP,
    EpilogueMode,
    SchedMode,
    WaitCommitScheme,
    WarpPipeline,
)
from ._lang import unwrap as _v
from ._layout import compute_kernel_layout
from ._schedule import (
    _fill_slots_valid,
    _pipeline_peeled,
    pipeline_depth,
    pipeline_unroll,
)
from ._types import (
    LDS_CAP_BYTES,
    LDS_EPILOGUE_RESERVE_BYTES,
    LDS_USABLE_BYTES,
    WeightGuHbmLayout,
)

__all__ = [
    "validate",
    "validate_buffer_counts",
    "validate_layout",
    "validate_pipeline",
    "validate_resources",
]


@gluon.constexpr_function
def validate(tc, N, K, layout=None):
    """Validate one build: the tuning enums, the layout and the resource budgets.

    ``layout`` is ``tc``'s :func:`compute_kernel_layout` result; a caller that already
    holds it passes it in, anything else gets one computed here. That optionality is
    why it comes last here rather than right after ``tc`` as everywhere else.
    """
    if _v(layout) is None:
        layout = compute_kernel_layout(tc)
    fc = tc.func_cfg
    BK = _v(tc.BLOCK_K)

    assert _v(fc.epilogue) in (
        int(EpilogueMode.DEFAULT),
        int(EpilogueMode.NOP_ACTIVATION),
        int(EpilogueMode.NOP),
    ), f"epilogue {_v(fc.epilogue)} is not an EpilogueMode"
    assert _v(fc.weight_gu_hbm_layout) in (
        int(WeightGuHbmLayout.INTERLEAVED),
        int(WeightGuHbmLayout.SPLIT),
    ), f"{_v(fc.weight_gu_hbm_layout)} is not a WeightGuHbmLayout"
    assert _v(tc.SCHED_MODE) in (
        int(SchedMode.NONE),
        int(SchedMode.IGLP_0),
        int(SchedMode.IGLP_1),
        int(SchedMode.MFMA_16),
        int(SchedMode.MFMA_8),
    ), f"SCHED_MODE {_v(tc.SCHED_MODE)} is not a SchedMode"
    validate_layout(tc, layout, N, K)

    if not _v(tc.UNROLL_EPILOGUE):
        for idx in (0, 1):
            assert not fc.has_scale(idx) or layout.scale_ratio_k_step(idx) == 1, (
                "UNROLL_EPILOGUE=False requires every active scale to advance "
                "once per payload step"
            )

    # -- warp pipelining (the inter-wave ping-pong) --
    # A slot's MFMAs are handed to TritonAMDGPUWarpPipeline as the `mfma` stage and
    # its ds_reads plus global->LDS copies as the `mem` stage. That is only a legal
    # split when the MFMAs read nothing the same slot loads, i.e. when the whole
    # BLOCK_K window is carried in registers from the previous stage. The pass also
    # rejects a wait inside a stage region, which is why the per-slot wait_group is
    # emitted before the `mfma` region rather than inside the `mem` one.
    assert _v(tc.WARP_PIPELINE) in (
        int(WarpPipeline.NONE),
        int(WarpPipeline.COMPILER),
        int(WarpPipeline.MANUAL),
    ), (
        f"WARP_PIPELINE {_v(tc.WARP_PIPELINE)} is not a WarpPipeline: "
        f"NONE {int(WarpPipeline.NONE)}, COMPILER {int(WarpPipeline.COMPILER)}, "
        f"MANUAL {int(WarpPipeline.MANUAL)}"
    )
    if tc.warp_pipeline_enabled():
        assert layout.num_prefetch_k_slots() == layout.num_k_slots_per_tile(), (
            f"WARP_PIPELINE needs VGPR_PREFETCH_K == BLOCK_K (got "
            f"{_v(tc.VGPR_PREFETCH_K)} vs {BK}): with a partial window the slot's "
            "MFMAs depend on the slot's own ds_read and there is no mem stage to "
            "hide behind them"
        )

    # -- register prefetch depth --
    # The live driver carries the complete BLOCK_K stage in registers from the
    # preceding step. There is no partial or disabled handoff after removing the
    # payload K split.
    pk = _v(tc.VGPR_PREFETCH_K)
    assert pk == BK, (
        f"VGPR_PREFETCH_K must equal BLOCK_K (got {pk} vs {BK}): "
        "the pipeline carries one whole payload K stage"
    )

    # -- commit-group granularity --
    # Reject unknown schemes before building the shared copy/group schedule.
    assert _v(tc.WAIT_COMMIT_SCHEME) in (
        int(WaitCommitScheme.PER_OP),
        int(WaitCommitScheme.PER_SLOT),
        int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE),
        int(WaitCommitScheme.PER_STAGE_WHOLE),
    ), (
        f"WAIT_COMMIT_SCHEME {_v(tc.WAIT_COMMIT_SCHEME)} is not a WaitCommitScheme: "
        f"PER_OP {int(WaitCommitScheme.PER_OP)}, "
        f"PER_SLOT {int(WaitCommitScheme.PER_SLOT)}, "
        f"PER_STAGE_WARP_PIPELINE {int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE)}, "
        f"PER_STAGE_WHOLE {int(WaitCommitScheme.PER_STAGE_WHOLE)}"
    )

    validate_resources(tc, layout)
    return True


@gluon.constexpr_function
def validate_buffer_counts(tc):
    """Only components present in the operand formats participate."""
    for operand in (0, 1):
        for scale in (False, True):
            if scale and not tc.func_cfg.has_scale(operand):
                continue
            depth = tc.num_buffers(operand, scale)
            assert depth >= 2, (
                f"{'A' if operand == 0 else 'B'}{'_SCALE' if scale else ''}_NUM_BUFFER "
                f"({depth}) must be at least 2"
            )
    assert _v(tc.K_UNROLL) >= 1, "K_UNROLL must be at least 1"
    return True


@gluon.constexpr_function
def validate_layout(tc, layout, N, K):
    """Validate storage placement and every tensor/tile shape contract."""
    fc = tc.func_cfg
    BM, BN, BK = _v(tc.BLOCK_M), _v(tc.BLOCK_N), _v(tc.BLOCK_K)
    N, K = _v(N), _v(K)
    instr = _v(tc.mfma_instr_shape)
    warps = _v(tc.warps_per_cta)
    tiles = _v(tc.tiles_per_warp)

    assert _v(tc.ALLOW_B_VIA_LDS) or tc.operand_preshuffled(1), (
        "ALLOW_B_VIA_LDS=False requires B_PRESHUFFLED"
    )
    validate_buffer_counts(tc)

    # -- host preconditions (no N tail, no K tail; only the even case exists) --
    assert N % BN == 0, f"N {N} % BLOCK_N {BN} != 0; the wrapper must fall back"
    assert K % BK == 0, f"K {K} % BLOCK_K {BK} != 0; the wrapper must fall back"

    # -- divisibility lattice --
    assert BK % instr[2] == 0
    assert BK % MX_GROUP == 0

    # -- shuffled scale geometry --
    for idx in (0, 1):
        if not tc.scale_shuffled(idx):
            continue
        axis = "M" if idx == 0 else "N"
        block = BM if idx == 0 else BN
        nonk, sk = layout.scale_mini_block_nonk(idx), layout.scale_mini_block_k(idx)
        mini = layout.scale_nonk(idx)
        assert block >= nonk >= mini, (
            f"BLOCK_{axis} >= SCALE_MINI_BLOCK_{axis} >= MINI_BLOCK_{axis} is required"
        )
        assert block % nonk == 0 and nonk % mini == 0, (
            f"SCALE_MINI_BLOCK_{axis} must divide BLOCK_{axis} and be a multiple of MINI_BLOCK_{axis}"
        )
        assert nonk % 32 == 0, (
            f"SCALE_MINI_BLOCK_{axis} must contain whole 32-row stripes"
        )
        assert sk >= max(256, BK), (
            "SCALE_MINI_BLOCK_K must be at least 256 and at least BLOCK_K"
        )
        assert sk & (sk - 1) == 0 and sk % BK == 0, (
            "SCALE_MINI_BLOCK_K must be a power of two and a multiple of BLOCK_K"
        )
        assert K % sk == 0, "K must contain complete SCALE_MINI_BLOCK_K tiles"

    # -- the slot is the unit of LDS allocation, of the global->LDS copy, of the
    #    MFMA and of the accumulator, so it must align to the CTA block;
    #    non-divisible values are illegal, not merely wasteful --
    assert BM % _v(tc.MINI_BLOCK_M) == 0
    assert BN % _v(tc.MINI_BLOCK_N) == 0
    # Fill ownership is a mapping from each component's non-K tile to a slot, so
    # what it has to satisfy is that every mapped slot exists and no component
    # owns two of its own tiles in one slot -- the flat _ops() transport carries
    # one optional tile per component. Different components sharing a slot is
    # fine, which is what lets a single-slot stage own both operands.
    assert _fill_slots_valid(tc, layout), (
        f"fill ownership is not a valid mapping for num_m_slots_per_block "
        f"{layout.num_m_slots_per_block()} x num_n_slots_per_block "
        f"{layout.num_n_slots_per_block()} (MINI_BLOCK_M {_v(tc.MINI_BLOCK_M)} vs "
        f"BLOCK_M {BM}, MINI_BLOCK_N {_v(tc.MINI_BLOCK_N)} vs BLOCK_N {BN})"
    )
    assert _v(tc.MINI_BLOCK_M) % (instr[0] * warps[0] * tiles[0]) == 0, (
        f"MINI_BLOCK_M {_v(tc.MINI_BLOCK_M)} must be a multiple of "
        f"instr[0]*warps[0]*tiles[0] = {instr[0] * warps[0] * tiles[0]}"
    )
    assert _v(tc.MINI_BLOCK_N) % (instr[1] * warps[1] * tiles[1]) == 0, (
        f"MINI_BLOCK_N {_v(tc.MINI_BLOCK_N)} must be a multiple of "
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
        if layout.byte_unit_lds_ok(idx) and not tc.operand_preshuffled(idx):
            assert tiles[idx] >= 2, (
                f"operand {idx} is not preshuffled, so its LDS unit is 32 rows and "
                f"tiles_per_warp[{idx}] must be >= 2 (got {tiles[idx]})"
            )
    if tc.operand_preshuffled(1):
        # The 16-column-blocked global offsets only stay inside the N slot while
        # every extent is a whole number of 16-column, 16-byte blocks.
        pk_b = BK // fc.b_pack_divisor()
        assert BN % 16 == 0 and _v(tc.MINI_BLOCK_N) % 16 == 0, (
            f"B_PRESHUFFLED needs BLOCK_N ({BN}) and MINI_BLOCK_N "
            f"({_v(tc.MINI_BLOCK_N)}) to be multiples of the 16-column block"
        )
        assert pk_b % 16 == 0, (
            f"B_PRESHUFFLED needs BLOCK_K/pack ({pk_b}) to be a multiple of the "
            "16-byte block"
        )

    # -- scale swizzle --
    # (checked by the caller against the tensor's scale_swizzle field)

    # -- WeightGuHbmLayout.SPLIT: an N slot must be exactly one operand side --
    if _v(fc.weight_gu_hbm_layout) == WeightGuHbmLayout.SPLIT:
        MBN = _v(tc.MINI_BLOCK_N)
        assert layout.num_n_slots_per_block() == 2 and MBN * 2 == BN, (
            f"WeightGuHbmLayout.SPLIT needs exactly two N slots, one per side: got "
            f"BLOCK_N {BN} / MINI_BLOCK_N {MBN} = {layout.num_n_slots_per_block()}"
        )
        assert N % 2 == 0 and (N // 2) % MBN == 0, (
            f"WeightGuHbmLayout.SPLIT needs each half of N ({N}) to be a whole "
            f"number of MINI_BLOCK_N ({MBN}) tiles"
        )
        # The whole point of the layout: a warp's emitted N extent has to contain a
        # whole MX group, or the fused quant's amax crosses the warp boundary. That
        # extent is instr_n * tiles_per_warp_n -- warps tile *above* it, so the
        # warp count is irrelevant here.
        if _v(fc.output_quant) is not None:
            warp_n = instr[1] * tiles[1]
            assert warp_n % MX_GROUP == 0, (
                f"WeightGuHbmLayout.SPLIT with a fused MX output quant needs a "
                f"warp's emitted N extent (instr_n {instr[1]} * tiles_per_warp_n "
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
        assert _v(tc.MINI_BLOCK_N) % (MX_GROUP * arn) == 0

    return True


@gluon.constexpr_function
def validate_resources(tc, layout):
    """Validate layout-dependent LDS and accumulator resource budgets."""
    # -- resource budgets --
    lds = layout.lds_bytes()
    assert lds <= LDS_USABLE_BYTES, (
        f"LDS {lds} B > {LDS_USABLE_BYTES} B usable ({LDS_CAP_BYTES} cap minus "
        f"{LDS_EPILOGUE_RESERVE_BYTES} B of epilogue scratch): reduce "
        f"NUM_LDS_BUFFER ({_v(tc.NUM_LDS_BUFFER)}) or the block sizes"
    )
    acc = layout.acc_vgprs_per_lane()
    assert acc <= 256, (
        f"fp32 accumulator alone needs {acc} VGPR/lane; 256 is the architectural "
        f"cap and WAVES_PER_EU=2 needs <= 256 total"
    )
    return True


@gluon.constexpr_function
def validate_pipeline(tc, layout, K):
    """Validate the live component rings and runtime-loop lower bound."""
    validate_buffer_counts(tc)
    num_k = layout.num_k_tiles(K)
    depth = pipeline_depth(tc, layout)
    peeled = _pipeline_peeled(tc, layout)
    unroll = pipeline_unroll(tc, layout)
    assert num_k >= depth + peeled + unroll, (
        f"NUM_K ({num_k}) must be at least PIPELINE_DEPTH ({depth}) "
        f"+ PEELED ({peeled}) + UNROLL ({unroll}) = {depth + peeled + unroll}"
    )
    return True
