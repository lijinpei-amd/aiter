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
from ._layout import KernelLayout
from ._schedule import Schedule
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

    ``layout`` is ``KernelLayout(tc)``; a caller that already holds it passes it in,
    anything else gets one computed here. That optionality is why it comes last here
    rather than right after ``tc`` as everywhere else.
    """
    if layout is None:
        layout = KernelLayout(tc)
    fc = tc.func_cfg
    BK = tc.BLOCK_K.value

    assert fc.epilogue.value in (
        int(EpilogueMode.DEFAULT),
        int(EpilogueMode.NOP_ACTIVATION),
        int(EpilogueMode.NOP),
    ), f"epilogue {fc.epilogue.value} is not an EpilogueMode"
    assert fc.weight_gu_hbm_layout.value in (
        int(WeightGuHbmLayout.INTERLEAVED),
        int(WeightGuHbmLayout.SPLIT),
    ), f"{fc.weight_gu_hbm_layout.value} is not a WeightGuHbmLayout"
    assert tc.SCHED_MODE.value in (
        int(SchedMode.NONE),
        int(SchedMode.IGLP_0),
        int(SchedMode.IGLP_1),
        int(SchedMode.MFMA_16),
        int(SchedMode.MFMA_8),
    ), f"SCHED_MODE {tc.SCHED_MODE.value} is not a SchedMode"
    validate_layout(tc, layout, N, K)

    if not tc.UNROLL_EPILOGUE.value:
        for idx in (0, 1):
            assert not fc.has_scale(idx) or layout.scale_slot_k(idx) == BK, (
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
    assert tc.WARP_PIPELINE.value in (
        int(WarpPipeline.NONE),
        int(WarpPipeline.COMPILER),
        int(WarpPipeline.MANUAL),
    ), (
        f"WARP_PIPELINE {tc.WARP_PIPELINE.value} is not a WarpPipeline: "
        f"NONE {int(WarpPipeline.NONE)}, COMPILER {int(WarpPipeline.COMPILER)}, "
        f"MANUAL {int(WarpPipeline.MANUAL)}"
    )
    if tc.warp_pipeline_enabled():
        assert layout.num_prefetch_k_slots.value == layout.num_k_slots.value, (
            f"WARP_PIPELINE needs VGPR_PREFETCH_K == BLOCK_K (got "
            f"{tc.VGPR_PREFETCH_K.value} vs {BK}): with a partial window the slot's "
            "MFMAs depend on the slot's own ds_read and there is no mem stage to "
            "hide behind them"
        )

    # -- register prefetch depth --
    # The live driver carries the complete BLOCK_K stage in registers from the
    # preceding step. There is no partial or disabled handoff after removing the
    # payload K split.
    pk = tc.VGPR_PREFETCH_K.value
    assert pk == BK, (
        f"VGPR_PREFETCH_K must equal BLOCK_K (got {pk} vs {BK}): "
        "the pipeline carries one whole payload K stage"
    )

    # -- commit-group granularity --
    # Reject unknown schemes before building the shared copy/group schedule.
    assert tc.WAIT_COMMIT_SCHEME.value in (
        int(WaitCommitScheme.PER_OP),
        int(WaitCommitScheme.PER_SLOT),
        int(WaitCommitScheme.PER_STAGE_WARP_PIPELINE),
        int(WaitCommitScheme.PER_STAGE_WHOLE),
    ), (
        f"WAIT_COMMIT_SCHEME {tc.WAIT_COMMIT_SCHEME.value} is not a WaitCommitScheme: "
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
    assert tc.K_UNROLL.value >= 1, "K_UNROLL must be at least 1"
    return True


@gluon.constexpr_function
def validate_layout(tc, layout, N, K):
    """Validate storage placement and every tensor/tile shape contract."""
    fc = tc.func_cfg
    BM, BN, BK = tc.BLOCK_M.value, tc.BLOCK_N.value, tc.BLOCK_K.value
    instr = tc.mfma_instr_shape.value
    warps = tc.warps_per_cta.value
    tiles = tc.tiles_per_warp.value

    assert tc.ALLOW_B_VIA_LDS.value or tc.preshuffled(1), (
        "ALLOW_B_VIA_LDS=False requires B_PRESHUFFLED"
    )
    validate_buffer_counts(tc)

    # -- host preconditions (no N tail, no K tail; only the even case exists) --
    assert N % BN == 0, f"N {N} % BLOCK_N {BN} != 0; the wrapper must fall back"
    assert K % BK == 0, f"K {K} % BLOCK_K {BK} != 0; the wrapper must fall back"

    # -- shuffled scale K extent; KernelLayout asserts the tile and slot geometry --
    for idx in (0, 1):
        if tc.preshuffled(idx, True):
            assert K % layout.scale_slot_k(idx) == 0, (
                "K must contain complete SCALE_SLOT_K tiles"
            )

    # -- MFMA shape must fit the tile --
    assert BM % (instr[0] * warps[0] * tiles[0]) == 0
    assert BN % (instr[1] * warps[1] * tiles[1]) == 0

    if tc.preshuffled(1):
        # The 16-column-blocked global offsets only stay inside the N slot while
        # every extent is a whole number of 16-column, 16-byte blocks.
        pk_b = BK // fc.pack_divisor(1)
        pk_b_bytes = pk_b * fc.operand_elem_ty(1).primitive_bitwidth // 8
        assert BN % 16 == 0 and tc.SLOT_N.value % 16 == 0, (
            f"B_PRESHUFFLED needs BLOCK_N ({BN}) and SLOT_N "
            f"({tc.SLOT_N.value}) to be multiples of the 16-column block"
        )
        assert pk_b_bytes % 16 == 0, (
            f"B_PRESHUFFLED needs BLOCK_K/pack ({pk_b} elements, {pk_b_bytes} B) "
            "to be a multiple of the 16-byte block"
        )

    # -- scale swizzle --
    # (checked by the caller against the tensor's scale_swizzle field)

    # -- WeightGuHbmLayout.SPLIT: an N slot must be exactly one operand side --
    if fc.weight_gu_hbm_layout.value == WeightGuHbmLayout.SPLIT:
        SLOT_N = tc.SLOT_N.value
        assert layout.num_n_slots.value == 2 and SLOT_N * 2 == BN, (
            f"WeightGuHbmLayout.SPLIT needs exactly two N slots, one per side: got "
            f"BLOCK_N {BN} / SLOT_N {SLOT_N} = {layout.num_n_slots.value}"
        )
        assert N % 2 == 0 and (N // 2) % SLOT_N == 0, (
            f"WeightGuHbmLayout.SPLIT needs each half of N ({N}) to be a whole "
            f"number of SLOT_N ({SLOT_N}) tiles"
        )
        # The whole point of the layout: a warp's emitted N extent has to contain a
        # whole MX group, or the fused quant's amax crosses the warp boundary. That
        # extent is instr_n * tiles_per_warp_n -- warps tile *above* it, so the
        # warp count is irrelevant here.
        if fc.output_quant.value is not None:
            warp_n = instr[1] * tiles[1]
            assert warp_n % MX_GROUP == 0, (
                f"WeightGuHbmLayout.SPLIT with a fused MX output quant needs a "
                f"warp's emitted N extent (instr_n {instr[1]} * tiles_per_warp_n "
                f"{tiles[1]} = {warp_n}) to be a multiple of {MX_GROUP}, else the "
                "amax reduction still crosses warps"
            )
            assert SLOT_N % MX_GROUP == 0

    # -- the fused MX output quant groups 32 *emitted* columns, so the raw tile has
    #    to carry 32 * activation_reduction_n of them --
    elif fc.output_quant.value is not None:
        arn = fc.activation_reduction_n()
        assert BN % (MX_GROUP * arn) == 0, (
            f"BLOCK_N {BN} must be a multiple of {MX_GROUP * arn} in raw "
            f"(pre-halving) terms so every MX group is tile-local"
        )
        assert tc.SLOT_N.value % (MX_GROUP * arn) == 0

    return True


@gluon.constexpr_function
def validate_resources(tc, layout):
    """Validate layout-dependent LDS and accumulator resource budgets."""
    # -- resource budgets --
    lds = layout.lds_bytes.value
    assert lds <= LDS_USABLE_BYTES, (
        f"LDS {lds} B > {LDS_USABLE_BYTES} B usable ({LDS_CAP_BYTES} cap minus "
        f"{LDS_EPILOGUE_RESERVE_BYTES} B of epilogue scratch): reduce "
        f"NUM_LDS_BUFFER ({tc.NUM_LDS_BUFFER.value}) or the block sizes"
    )
    acc = layout.acc_vgprs_per_lane.value
    assert acc <= 256, (
        f"fp32 accumulator alone needs {acc} VGPR/lane; 256 is the architectural "
        f"cap and WAVES_PER_EU=2 needs <= 256 total"
    )
    return True


@gluon.constexpr_function
def validate_pipeline(tc, layout, K, schedule=None):
    """Validate the live component rings, copy ownership and runtime-loop lower bound.

    ``schedule`` is ``Schedule(tc, layout, K)``; a caller that already holds it passes
    it in, anything else gets one built here once the ring depths are known to be
    sound.
    """
    validate_buffer_counts(tc)
    if schedule is None:
        schedule = Schedule(tc, layout, K)
    # Copy ownership maps each component's non-K slots onto slots, so every mapped
    # slot has to exist and no component may copy two of its own non-K slots in one
    # slot -- a slot carries one optional copy per component. Different components
    # sharing a slot is fine, which is what lets a single-slot step copy both
    # operands.
    assert schedule.fill_slots_valid.value, (
        f"fill ownership is not a valid mapping for num_m_slots "
        f"{layout.num_m_slots.value} x num_n_slots {layout.num_n_slots.value} "
        f"(SLOT_M {tc.SLOT_M.value} vs BLOCK_M {tc.BLOCK_M.value}, "
        f"SLOT_N {tc.SLOT_N.value} vs BLOCK_N {tc.BLOCK_N.value})"
    )
    num_k = schedule.num_k.value
    depth = schedule.depth.value
    peeled = schedule.peeled.value
    unroll = schedule.unroll.value
    assert num_k >= depth + peeled + unroll, (
        f"NUM_K ({num_k}) must be at least PIPELINE_DEPTH ({depth}) "
        f"+ PEELED ({peeled}) + UNROLL ({unroll}) = {depth + peeled + unroll}"
    )
    return True
