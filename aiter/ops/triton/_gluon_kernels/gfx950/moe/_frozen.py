# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Frozen prologue and K-loop step for the gfx950 Gluon MoE kernel.

``_pipeline_step_frozen`` is a verbatim copy of ``moe_gemm._pipeline_step_impl`` taken
from the best measured kernel (2026-09-02), with every env-var constexpr replaced by its
tuned value so no stray flag can perturb the reference schedule. It lives in its own
module for exactly that reason: nothing here is meant to be refactored alongside the
live step, and keeping the two apart makes an accidental edit visible in the diff.

``AITER_TRITON_MOE_GLUON_FROZEN_STEP=1`` selects it -- see ``_buffered._step``.
The acceptance test for a re-snapshot is *identical assembly* against ``FROZEN_STEP=0``,
which is also what proves the ``*_frozen`` helper twins below are still in sync with
their live counterparts.

The shared pointer and register aggregates are adapted at the step boundary; the
snapshot's scheduling statements, including its separate prologue fence, stay fixed.

The shared ``moe_gemm`` helpers are imported after this module's definitions.
The unified driver also imports these frozen steps, so deferring the shared
imports lets either module initialize first. JIT resolves the helpers later.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import require_constexpr
from ._lang import unwrap as _v
from ._schedule import _buffer_load_order, _buffer_load_tile


@gluon.jit
def _frozen_prologue_fence():
    """Preserve the tuned snapshot's cooperative prologue fence and schedule."""
    gl.amd.cdna4.sched_barrier(0)
    gl.barrier()
    gl.amd.cdna4.sched_barrier(0)


@gluon.jit
def _prologue_frozen(pc, hbm_ptrs):
    """Fill NB-1 buffers, then retain the snapshot's whole-buffer wait and fence."""
    tc: gl.constexpr = pc.tuning_cfg
    NB: gl.constexpr = tc.NUM_LDS_BUFFER
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    for i in gl.static_range(NB - 1):
        for ni in gl.static_range(NN):
            for mi in gl.static_range(NM):
                _buffer_load_frozen(
                    pc,
                    i,
                    mi,
                    ni,
                    hbm_ptrs.a_hbm_ptr,
                    hbm_ptrs.b_hbm_ptr,
                    hbm_ptrs.a_scale_hbm_ptr,
                    hbm_ptrs.b_scale_hbm_ptr,
                )
                if require_constexpr(mi == NM - 1 and ni == NN - 1):
                    hbm_ptrs = _advance_hbm_ptrs(
                        pc, hbm_ptrs, K_PHASE=tc.scale_k_phase(i)
                    )
    pc.lds_ptrs.wait_buffer_load_groups((NB - 2) * (NM + NN))
    _frozen_prologue_fence()
    return hbm_ptrs


# --- frozen placement / wait helpers ------------------------------------------
# Twins of the constexpr helpers _pipeline_step_frozen reaches, with SCALE_FILL_MID
# pinned to 1 (its BEST value). They decide which slot owns which copy and how many
# groups a wait covers, so a stray flag here reshapes the frozen schedule even though
# the step itself reads no env var. Both axes are split in any config that now
# compiles (validate() insists), which is what the frozen EVEN test amounted to.
# ------------------------------------------------------------------------------
@gluon.constexpr_function
def _buffer_load_group_pos_frozen(is_a, i, NM, NN):
    """Position of one mini-block fill's commit group inside its stage."""
    is_a, i = _v(is_a), _v(i)
    order = _buffer_load_order(NM, NN)
    return order.index((1 if is_a else 0, i))


@gluon.constexpr_function
def _buffer_loads_before_frozen(mi, ni, NM, NN, ANY):
    """Groups this stage has already committed when slot ``(mi, ni)`` is reached.

    Counted against the N-outer walk of :func:`_slot_index`. Legacy issues A(m) at
    ``(m, 0)`` and B(n) at ``(0, n)``, so once the walk has left column 0 every A is
    behind it, and the B of the current column is behind it as soon as ``mi > 0``.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    if not _v(ANY):
        return 0
    return min(_slot_index(mi, ni, NM, NN), NM + NN)


@gluon.constexpr_function
def _ds_read_a_tile_frozen(mi, ni, NM, NN):
    """Operand-A mini block slot ``(mi, ni)`` reads out of LDS, or None.

    Even: the same one-per-slot assignment the fills use, so a slot's fill and its read
    name the *same* position in :func:`_buffer_load_order`. Its wait then works out to
    ``G - 1 - s + STAGES_BETWEEN * G + s`` -- the ``s`` cancels and every slot waits on
    the same constant, which is exactly the uniform ``wait_group`` the reference kernel
    uses. Legacy: A(mi) at ``ni == 0``, which bunches both reads onto slot (0, 0).
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _buffer_load_tile(_buffer_load_pos_frozen(mi, ni, NM, NN), NM, NN, True)


@gluon.constexpr_function
def _ds_read_b_tile_frozen(mi, ni, NM, NN):
    """Operand-B mini block slot ``(mi, ni)`` reads out of LDS, or None."""
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    return _buffer_load_tile(_buffer_load_pos_frozen(mi, ni, NM, NN), NM, NN, False)


@gluon.constexpr_function
def _scale_buffer_load_slot_frozen(is_a, tile, NM, NN):
    """Slot index that issues the scale copy for A(tile) / B(tile).

    Default: the same slot as the payload, so a tile's scale and payload share one
    commit group. Under ``1`` at NM=NN=2 both A(0)/B(0) scales go to
    slot 1 and both A(1)/B(1) scales to slot 2.
    """
    is_a, tile = bool(_v(is_a)), _v(tile)
    NM, NN = _v(NM), _v(NN)
    if 1 and NM == 2 and NN == 2:
        return 1 + tile
    return _buffer_load_group_pos_frozen(is_a, tile, NM, NN)


@gluon.constexpr_function
def _buffer_load_pos_frozen(mi, ni, NM, NN):
    """Which fill (position in :func:`_buffer_load_order`) slot ``(mi, ni)`` issues, or None.

    Even: one fill per slot, in flat slot order -- exactly the tutorial's layout, where
    each of the four ``mfma``/``mem`` region pairs moves one tile and commits one group.
    Legacy: A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``, which loads slot (0, 0)
    with two fills and leaves every slot off the first row and column with none.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    s = _slot_index(mi, ni, NM, NN)
    return s if s < NM + NN else None


@gluon.constexpr_function
def _scale_buffer_load_tile_frozen(mi, ni, NM, NN, want_a):
    """Tile whose A (resp. B) scale copy slot ``(mi, ni)`` issues, or None.

    Still one slot per mini block even when several share a scale tile: the commit
    group has to be emitted either way, because _buffer_load_wait counts G = NM + NN groups
    per stage. buffer_load_a_scale drops the redundant *copy* and leaves the group empty.
    """
    s = _slot_index(mi, ni, NM, NN)
    n = NM if _v(want_a) else NN
    for t in range(_v(n)):
        if _scale_buffer_load_slot_frozen(_v(want_a), t, NM, NN) == s:
            return t
    return None


@gluon.constexpr_function
def _buffer_load_wait_frozen(mi, ni, NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD):
    """Outstanding-group count that retires everything slot ``(mi, ni)`` is about to read.

    A group is retired once ``wait_group(n)`` leaves at most ``n`` behind it. Counting
    forward from the target group: the rest of its own stage, then ``STAGES_BETWEEN``
    whole stages, then whatever the current stage has committed so far. The slot reads
    A(mi) when ``ni == 0`` and B(ni) when ``mi == 0``; when it reads both, the later
    group's (smaller) count wins. ``None`` means the slot reads nothing and needs no wait.
    """
    mi, ni, NM, NN = _v(mi), _v(ni), _v(NM), _v(NN)
    G = NM + NN
    base = _v(STAGES_BETWEEN) * G + _buffer_loads_before_frozen(
        mi, ni, NM, NN, ANY_BUFFER_LOAD
    )
    out = None
    ta = _ds_read_a_tile_frozen(mi, ni, NM, NN)
    if ta is not None:
        # The payload and the scale of the same tile can sit in different commit
        # groups (see _scale_buffer_load_slot); the later of the two is what has to retire.
        p = max(
            _buffer_load_group_pos_frozen(True, ta, NM, NN),
            _scale_buffer_load_slot_frozen(True, ta, NM, NN),
        )
        out = G - 1 - p + base
    tb = _ds_read_b_tile_frozen(mi, ni, NM, NN)
    if tb is not None:
        p = max(
            _buffer_load_group_pos_frozen(False, tb, NM, NN),
            _scale_buffer_load_slot_frozen(False, tb, NM, NN),
        )
        w = G - 1 - p + base
        out = w if out is None else min(out, w)
    return out


@gluon.constexpr_function
def _stage_buffer_load_wait_frozen(NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD):
    """Strongest (smallest) wait_group count over all slots of a stage."""
    NM, NN = _v(NM), _v(NN)
    ws = [
        _buffer_load_wait_frozen(mi, ni, NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD)
        for ni in range(NN)
        for mi in range(NM)
    ]
    ws = [w for w in ws if w is not None]
    if not ws:
        return None
    return max(min(ws), 0)


@gluon.jit
def _stage_buffer_load_wait_group_frozen(
    lds_ptrs,
    mi: gl.constexpr,
    ni: gl.constexpr,
    NM: gl.constexpr,
    NN: gl.constexpr,
    STAGES_BETWEEN: gl.constexpr,
    ANY_BUFFER_LOAD: gl.constexpr,
    WAIT_SLACK: gl.constexpr = 0,
):
    """One wait_group per stage, emitted at its first slot, instead of one per slot."""
    WAIT: gl.constexpr = _stage_buffer_load_wait_frozen(
        NM, NN, STAGES_BETWEEN, ANY_BUFFER_LOAD
    )
    if require_constexpr(_slot_index(mi, ni, NM, NN) == 0 and WAIT is not None):
        lds_ptrs.wait_buffer_load_groups(WAIT + WAIT_SLACK)


@gluon.jit
def _buffer_load_frozen(
    pc,
    BUFFER_LOAD_IDX,
    mi: gl.constexpr,
    ni: gl.constexpr,
    a_hbm_ptr,
    b_hbm_ptr,
    a_scale_hbm_ptr,
    b_scale_hbm_ptr,
    ADVANCE: gl.constexpr = False,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
):
    """The global->LDS copies slot ``(mi, ni)`` owns, each its own commit group.

    FROZEN SNAPSHOT -- do not refactor; ``_buffered._fill_slot`` is the live one.

    Specialised on the flags the ~625 us kernel ran with, so nothing here reads
    os.environ: the legacy per-fill groups (the former ONE_MARK 0 default) and
    SOFF_UNROLL 0 (pointers bump every step, so KI and every *_SOFF are 0).
    EVEN_FILL was 1 and is gone -- both axes are split in any
    config that compiles, so the schedule it selected is the only one.
    ``KI`` is kept in the signature only so the two stay call-compatible.

    ``ADVANCE`` walks the returned HBM pointers on to the next ``BLOCK_K`` stage. It is
    done here, at the last slot, rather than after the slot loop on purpose: the warp
    pipeliner only tolerates a ``wait_group`` as the *first* op after a stage border, and
    a stage-tail ``tt.addptr`` sitting between the last ``mem`` border and the next
    slot's wait is exactly what breaks that.
    """
    func_cfg: gl.constexpr = pc.func_cfg
    NM: gl.constexpr = pc.tuning_cfg.num_mini_m()
    NN: gl.constexpr = pc.tuning_cfg.num_mini_n()
    # Under the even schedule the slot owns at most one fill, named by its position in
    # _buffer_load_order; under the legacy one the (ni == 0) / (mi == 0) predicates below pick.
    POS: gl.constexpr = _buffer_load_pos_frozen(mi, ni, NM, NN)
    A_TILE: gl.constexpr = _buffer_load_tile(POS, NM, NN, True)
    B_TILE: gl.constexpr = _buffer_load_tile(POS, NM, NN, False)
    # Scale copies may be placed on a different slot than their payload.
    A_SC: gl.constexpr = _scale_buffer_load_tile_frozen(mi, ni, NM, NN, True)
    B_SC: gl.constexpr = _scale_buffer_load_tile_frozen(mi, ni, NM, NN, False)
    # Byte displacement of this step from the body's base pointer. The *_step values are
    # in elements (they are added to a typed pointer), soffset is in bytes, so each is
    # scaled by its operand's storage width. All constexpr, so these fold into a literal
    # and the SGPR holding them is hoisted out of the loop.
    # SOFF_UNROLL = 0: every copy addresses its own base, no soffset.
    if require_constexpr(A_TILE is not None):
        pc.lds_ptrs.buffer_load_a_payload(
            BUFFER_LOAD_IDX, A_TILE, a_hbm_ptr, pc.a_hbm_offs[A_TILE], 0
        )
        # Payload and its own scale share one commit group; a scale placed elsewhere
        # gets its own below.
        if require_constexpr(A_SC == A_TILE):
            pc.lds_ptrs.buffer_load_a_scale(
                BUFFER_LOAD_IDX,
                A_TILE,
                a_scale_hbm_ptr,
                _opt_at(pc.a_scale_hbm_offs, A_TILE, func_cfg.a_has_scale()),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(A_SC is not None and A_SC != A_TILE):
        pc.lds_ptrs.buffer_load_a_scale(
            BUFFER_LOAD_IDX,
            A_SC,
            a_scale_hbm_ptr,
            _opt_at(pc.a_scale_hbm_offs, A_SC, func_cfg.a_has_scale()),
            0,
        )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_TILE is not None):
        pc.lds_ptrs.buffer_load_b_payload(
            BUFFER_LOAD_IDX, B_TILE, b_hbm_ptr, pc.b_hbm_offs[B_TILE], 0
        )
        if require_constexpr(B_SC == B_TILE):
            pc.lds_ptrs.buffer_load_b_scale(
                BUFFER_LOAD_IDX,
                B_TILE,
                b_scale_hbm_ptr,
                _opt_at(pc.b_scale_hbm_offs, B_TILE, func_cfg.b_has_scale()),
                0,
            )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(B_SC is not None and B_SC != B_TILE):
        pc.lds_ptrs.buffer_load_b_scale(
            BUFFER_LOAD_IDX,
            B_SC,
            b_scale_hbm_ptr,
            _opt_at(pc.b_scale_hbm_offs, B_SC, func_cfg.b_has_scale()),
            0,
        )
        pc.lds_ptrs.commit_buffer_load()
    if require_constexpr(ADVANCE):
        # One bump per step: the frozen kernel did not fold steps into soffset.
        a_hbm_ptr = a_hbm_ptr + KU * pc.a_step
        b_hbm_ptr = b_hbm_ptr + KU * pc.b_step
        if require_constexpr(func_cfg.a_has_scale() and pc.tuning_cfg.scale_via_lds(0)):
            a_scale_hbm_ptr = a_scale_hbm_ptr + KU * pc.s_step * pc.a_scale_stride_k
        if require_constexpr(func_cfg.b_has_scale() and pc.tuning_cfg.scale_via_lds(1)):
            b_scale_hbm_ptr = b_scale_hbm_ptr + KU * pc.s_step * pc.b_scale_stride_k
    return a_hbm_ptr, b_hbm_ptr, a_scale_hbm_ptr, b_scale_hbm_ptr


@gluon.jit
def _pipeline_step_frozen(
    pc,
    hbm_ptrs,
    regs,
    BUFFER_LOAD_IDX,
    DS_READ_IDX,
    STAGES_BETWEEN: gl.constexpr,
    DO_BUFFER_LOAD: gl.constexpr,
    DO_DS_READ: gl.constexpr,
    IN_LOOP: gl.constexpr = False,
    DO_MFMA: gl.constexpr = True,
    KI: gl.constexpr = 0,
    KU: gl.constexpr = 1,
    WAIT_SLACK: gl.constexpr = 0,
):
    """One ``BLOCK_K`` stage, walked as ``num_mini_m() x num_mini_n()`` MFMA slots.

    FROZEN SNAPSHOT of the 2026-09-02 best kernel -- do not refactor this copy.
    Verbatim ``_pipeline_step_impl`` with every env-var constexpr replaced by the
    value it holds in the tuned config, so nothing here reads ``os.environ`` and no
    stray flag can perturb the reference schedule. Measured at 4 waves with
    WARP_PIPELINE=1, MANUAL_PP=1, the LDS-staged epilogue that is now the only
    epilogue path, and llc flags -amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd
    -amdgpu-no-sched-revert -misched-pin-critical-res=HWXDL: 630.3 us cold (g4q)
    and 622.3 us saturating (bf16).

    Select with AITER_TRITON_MOE_GLUON_FROZEN_STEP=1 to A/B a refactor of
    ``_buffered._step_live`` against the known-good schedule. Immediately after a
    re-snapshot the two must compile to *identical* assembly -- that, not perf, is
    the check that the copy is faithful.

    Re-snapshot procedure: copy the live step here with the flag values listed
    below folded in, then diff the ``.amdgcn`` of FROZEN_STEP=0 against
    FROZEN_STEP=1 and require zero differing instruction lines.


    A mini-M block's copy and ``ds_read`` are issued at its first slot (``ni == 0``) and
    reused by every later ``ni``; a mini-N block's at ``mi == 0``. So the copies and the
    LDS reads are interleaved with the MFMAs instead of standing in one block ahead of
    them, and each is still a whole mini tile at its own coalesced, fully vectorised copy
    layout.

    Every fill is its own commit group, so each slot waits for exactly the mini block it
    is about to read rather than for the whole stage -- see :func:`_buffer_load_wait`, whose
    ``STAGES_BETWEEN`` is ``NUM_LDS_BUFFER - 2`` in the steady state (one buffer is being
    filled by ``buffer_load ... lds`` while one is being consumed by ``ds_read``).

    Along K the ``ds_read`` always pulls a whole stage, mini-K steps ``[0, NUM_MINI)``.
    The MFMAs run one window earlier, over ``[-PF_MINI, NUM_MINI-PF_MINI)``: the negative
    part is the payload/scale fragments in ``regs``, carried from the previous step,
    and the tail this step reads is carried to the next. ``PF_MINI == NUM_MINI`` makes
    the MFMAs consume nothing they read themselves; ``PF_MINI == 0`` makes them consume
    only what they read.

    ``PF_MINI == NUM_MINI`` is also what makes the ping-pong legal: the slot's MFMAs then
    depend on nothing the slot reads, so they are emitted *first* and the reads and copies
    behind them are pure fill for the shadow. Under ``WARP_PIPELINE`` the two halves are
    additionally handed to ``TritonAMDGPUWarpPipeline`` as an ``mfma``/``mem`` stage pair,
    which is what turns the interleave into an inter-wave ping-pong. The ``wait_group``
    stays outside both regions -- the pass rejects a wait inside one.
    """

    # --- the flag set this snapshot was specialised on ------------------------------
    # Every branch on these is inlined below, so there is nothing left to read from
    # os.environ and no way for a stray flag to perturb this copy:
    #   DS_IN_MFMA 1  DS_MOVE 0  _FILLS_FIRST 1 (the env flag, not the derived
    #   FILLS_FIRST constexpr -- that was tc.B_IN_REG and is inlined False below)
    #   FILL_IN_MFMA 0  LOAD_MFMA_READ 0
    #   MANUAL_PP 1   MEM_PRIO 1  MPP_ALL_FENCED 0  MPP_BARRIER_STRIDE 4
    #   MPP_CLOSE_BARRIER 0  MPP_NO_FENCE 0  MPP_NO_STAGE_BARRIER 0  MPP_PIN_MFMA 0
    #   MPP_PIN_WAIT 1  MPP_SCHED_AFTER_ONLY 0  NO_SYNC_STAGE 1  SCHED_MODE 0
    #   SLOT_SCHED_BARRIER 0  SOFF_UNROLL 0  STAGE_WAIT 1  WARP_PIPELINE 1
    # Six came from BEST (DS_IN_MFMA MANUAL_PP MPP_BARRIER_STRIDE MPP_CLOSE_BARRIER
    # NO_SYNC_STAGE STAGE_WAIT); the rest are each flag's code default.
    # Preserve the legacy ONE_MARK 0 + STAGE_WAIT 1 schedule independently of the
    # live path's commit/wait configuration.
    # What remains conditional depends only on this function's parameters: DO_MFMA,
    # DO_DS_READ, DO_BUFFER_LOAD, the mi/ni slot indices, and pc's func/tuning cfg.
    # --------------------------------------------------------------------------------
    func_cfg: gl.constexpr = pc.func_cfg
    tc: gl.constexpr = pc.tuning_cfg
    NM: gl.constexpr = tc.num_mini_m()
    NN: gl.constexpr = tc.num_mini_n()
    NUM_MINI: gl.constexpr = tc.num_mini_k()
    PF_MINI: gl.constexpr = tc.num_prefetch_mini()
    HEAD_MINI: gl.constexpr = NUM_MINI - PF_MINI
    A_HAS: gl.constexpr = func_cfg.a_has_scale()
    B_HAS: gl.constexpr = func_cfg.b_has_scale()
    # ``DO_MFMA=False`` is the prologue's first step: it fills and reads, but has no
    # carried fragments to dot yet. That is only equivalent to the hand-rolled prologue
    # when the stage carries whole -- with HEAD_MINI > 0 the head of what this step reads
    # would have to be dotted here and is not carried, so skipping the MFMAs would
    # silently drop it. Assert rather than branch: the default VGPR_PREFETCH_K == BLOCK_K
    # gives HEAD_MINI == 0, and a config that lowers it must not reach this quietly.
    gl.static_assert(
        DO_MFMA or HEAD_MINI == 0,
        "DO_MFMA=False needs VGPR_PREFETCH_K == BLOCK_K (HEAD_MINI == 0): with a split "
        "stage the head this step reads is dotted here and never carried",
    )
    # mfma-before-mem is only legal when the slot's MFMAs read nothing the slot loads.
    # The frozen kernel ran VGPR_PREFETCH_K == BLOCK_K, so PING_PONG was always True and
    # every branch on it below is inlined. Assert rather than branch: a config that
    # lowers VGPR_PREFETCH_K must fail here, not silently get a schedule this copy was
    # never measured with.
    gl.static_assert(
        HEAD_MINI == 0,
        "_pipeline_step_frozen is the VGPR_PREFETCH_K == BLOCK_K schedule; "
        "split-stage prefetch is unsupported",
    )
    # The frozen kernel always stages B through LDS.
    # PIPE (the warp-pipeline *pass*) is dead here: it needs `not _MANUAL_PP` and the
    # frozen flags pin _MANUAL_PP = 1. Both `if PIPE` branches below are gone with it.
    # MPP -- the hand-emitted rendezvous -- reduces to DO_DS_READ once _WP,
    # _MANUAL_PP and PING_PONG are all 1.
    MPP: gl.constexpr = DO_DS_READ

    # _SOFF_UNROLL alone decides whether the body's steps are folded into soffset.
    # With it off these collapse to (0, 1) -- bump the pointers on every step, as
    # before -- so callers can pass the real position unconditionally.
    # _SOFF_UNROLL = 0: the pointers advance every step, KI/KU unused.
    KIE: gl.constexpr = 0
    KUE: gl.constexpr = 1

    a_hbm_ptr = hbm_ptrs.a_hbm_ptr
    b_hbm_ptr = hbm_ptrs.b_hbm_ptr
    a_scale_hbm_ptr = hbm_ptrs.a_scale_hbm_ptr
    b_scale_hbm_ptr = hbm_ptrs.b_scale_hbm_ptr

    # Fragments this step reads, appended mini block by mini block. `a_cur` grows once
    # per mi (at ni == 0) and `b_cur` once per ni (at mi == 0), so block `mi` always
    # starts at pair `mi * NUM_MINI` and is already there by the time any slot needs it.
    a_cur = ()
    b_cur = ()
    a_tail = ()
    b_tail = ()
    acc = ()
    # _SCHED_MODE = 0, so no iglp_opt hint: it and the ping-pong are alternative
    # answers to the same question, and the rendezvous below is the one measured.
    # N outer, M inner -- see _slot_index. The slot walk, the accumulator tuple's
    # layout and the fill positions are all that one order.
    for ni in gl.static_range(NN):
        for mi in gl.static_range(NM):
            # _MPP_PIN_WAIT = 1: the leading sched_barrier of the rendezvous sits
            # here, before the wait, rather than next to the barrier itself.
            # _MPP_BARRIER_STRIDE = 4, inlined as the % 4 below.
            if require_constexpr(MPP and _slot_index(mi, ni, NM, NN) % 4 == 0):
                gl.amd.cdna4.sched_barrier(0)
            # _STAGE_WAIT = 1, so the per-slot wait form never applied.
            if require_constexpr(DO_DS_READ):
                _stage_buffer_load_wait_group_frozen(
                    pc.lds_ptrs,
                    mi,
                    ni,
                    NM,
                    NN,
                    STAGES_BETWEEN,
                    DO_BUFFER_LOAD,
                    WAIT_SLACK,
                )

            if require_constexpr(DO_MFMA):
                dot_a = _take_reg_pairs(
                    regs.a_payload, regs.a_scale, mi * PF_MINI, PF_MINI
                )
                dot_b = _take_reg_pairs(
                    regs.b_payload, regs.b_scale, ni * PF_MINI, PF_MINI
                )
            else:
                # Nothing carried in yet; the register fragment tuples are empty.
                dot_a = ()
                dot_b = ()

            if require_constexpr(MPP):
                # Rendezvous entering the mfma region. Two per slot -- one here and
                # one before the fills -- so the count per stage is even and a
                # cond_barrier phase shift holds its relationship instead of flipping
                # every stage.
                #
                # The sched_barrier pair is load-bearing, not decoration: a bare
                # s_barrier carries no memory semantics, so without it LLVM freely
                # hoists a buffer_load...lds above the rendezvous that is supposed to
                # separate it from other waves' reads of that slot -- a WAR race that
                # shows up as nondeterministic wrong results. This is the same
                # bracketing emitClusterBarrier uses.
                if require_constexpr(_slot_index(mi, ni, NM, NN) % 4 == 0):
                    # _MPP_PIN_WAIT = 1 put the leading sched_barrier before the
                    # wait instead, so only the trailing one is emitted here.
                    # _MPP_NO_FENCE = 0 and _MPP_ALL_FENCED = 0: the stage head is
                    # the one fenced barrier, the rest are bare.
                    if require_constexpr(_slot_index(mi, ni, NM, NN) == 0):
                        gl.barrier()
                    else:
                        gl.amd.cdna4.bare_barrier()
                    gl.amd.cdna4.sched_barrier(0)
                # Only in the K loop, never in the drain (DO_BUFFER_LOAD is False
                # there). The pair is asymmetric as configured: this setprio(1) has no
                # matching setprio(0), because that one sits behind _MPP_CLOSE_BARRIER,
                # which the tuned config leaves at 0. Every wave raised its priority and
                # none lowered it, so there was no relative priority for the ping-pong
                # to exploit -- and the 8 in the drain additionally split the drain into
                # four scheduling regions of 44/17/21/22 instructions, since s_setprio
                # has unmodelled side effects and MachineScheduler treats it as a region
                # boundary. Dropping them merges those into one region of 157 holding 84
                # MFMA and 51 activation instructions together. Perf is a wash (-1.4 us,
                # CI [-6.4, +3.7]); the merge is the reason to keep it.
                if require_constexpr(DO_BUFFER_LOAD):
                    gl.amd.cdna4.setprio(1)
            # MFMA first: it consumes only registers carried from the previous stage,
            # so everything below it is shadow work. _DS_IN_MFMA = 1 puts this slot's
            # ds_reads in the same region, so they issue under the MFMA cluster's shadow
            # rather than next to the buffer_loads.
            slot_acc = _maybe_block_dot(
                dot_a,
                dot_b,
                regs.acc[_slot_index(mi, ni, NM, NN)],
                PF_MINI,
                func_cfg,
                tc,
                DO_MFMA,
            )
            if require_constexpr(MPP):
                if require_constexpr(
                    _ds_read_a_tile_frozen(mi, ni, NM, NN) is not None
                ):
                    a_cur = a_cur + _ds_read_operand(
                        pc,
                        DS_READ_IDX,
                        _ds_read_a_tile_frozen(mi, ni, NM, NN),
                        a_scale_hbm_ptr,
                        0,
                    )
                if require_constexpr(
                    _ds_read_b_tile_frozen(mi, ni, NM, NN) is not None
                ):
                    b_cur = b_cur + _ds_read_operand(
                        pc,
                        DS_READ_IDX,
                        _ds_read_b_tile_frozen(mi, ni, NM, NN),
                        b_scale_hbm_ptr,
                        1,
                    )
            acc = acc + (slot_acc,)

            if require_constexpr(DO_BUFFER_LOAD):
                (
                    a_hbm_ptr,
                    b_hbm_ptr,
                    a_scale_hbm_ptr,
                    b_scale_hbm_ptr,
                ) = _buffer_load_frozen(
                    pc,
                    BUFFER_LOAD_IDX,
                    mi,
                    ni,
                    a_hbm_ptr,
                    b_hbm_ptr,
                    a_scale_hbm_ptr,
                    b_scale_hbm_ptr,
                    ADVANCE=(mi == NM - 1) and (ni == NN - 1) and (KIE == KUE - 1),
                    KI=KIE,
                    KU=KUE,
                )
            # Indexed by the tile this slot actually read, not by (mi, ni): under the
            # even schedule slot (0, 1) reads B(0), not B(1), so the legacy `ni == 0` /
            # `mi == 0` predicates would reach past the end of the half-built tuple.
            if require_constexpr(
                DO_DS_READ and _ds_read_a_tile_frozen(mi, ni, NM, NN) is not None
            ):
                a_tail = a_tail + _take_pairs(
                    a_cur,
                    _ds_read_a_tile_frozen(mi, ni, NM, NN) * NUM_MINI + HEAD_MINI,
                    PF_MINI,
                )
            if require_constexpr(
                DO_DS_READ and _ds_read_b_tile_frozen(mi, ni, NM, NN) is not None
            ):
                b_tail = b_tail + _take_pairs(
                    b_cur,
                    _ds_read_b_tile_frozen(mi, ni, NM, NN) * NUM_MINI + HEAD_MINI,
                    PF_MINI,
                )

            # _SLOT_SCHED_BARRIER = 0: no end-of-slot scheduler fence.

    if require_constexpr(DO_DS_READ):
        if require_constexpr(A_HAS and not tc.scale_via_lds(0)):
            a_scale_hbm_ptr = a_scale_hbm_ptr + pc.s_step * pc.a_scale_stride_k
        if require_constexpr(B_HAS and not tc.scale_via_lds(1)):
            b_scale_hbm_ptr = b_scale_hbm_ptr + pc.s_step * pc.b_scale_stride_k
    else:
        a_tail = _take_reg_pairs(regs.a_payload, regs.a_scale, 0, NM * PF_MINI)
        b_tail = _take_reg_pairs(regs.b_payload, regs.b_scale, 0, NN * PF_MINI)

    return _PipelinePointers(
        a_hbm_ptr,
        b_hbm_ptr,
        a_scale_hbm_ptr,
        b_scale_hbm_ptr,
    ), _make_reg_fragments(a_tail, b_tail, acc)


# The driver imports these steps; defer shared helpers until the steps exist.
from .moe_gemm import (
    _advance_hbm_ptrs,
    _ds_read_operand,
    _make_reg_fragments,
    _maybe_block_dot,
    _opt_at,
    _PipelinePointers,
    _slot_index,
    _take_pairs,
    _take_reg_pairs,
)
