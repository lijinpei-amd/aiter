# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Copy ownership and commit/wait accounting for the Gluon MoE pipeline."""

from triton.experimental import gluon

from ._lang import unwrap as _v
from ._layout import _slot_index
from ._types import WaitCommitScheme


@gluon.constexpr_function
def _buffer_load_order(NM, NN):
    """Payload copies as ``(is_a, tile)`` in slot order."""
    NM, NN = _v(NM), _v(NN)
    if NM == 2 and NN == 2:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = []
    for i in range(max(NM, NN)):
        if i < NM:
            out.append((1, i))
        if i < NN:
            out.append((0, i))
    return out


@gluon.constexpr_function
def _payload_buffer_load_slot(is_a, tile, NM, NN):
    return _buffer_load_order(NM, NN).index((int(bool(_v(is_a))), _v(tile)))


@gluon.constexpr_function
def _buffer_load_pos(mi, ni, NM, NN):
    s = _slot_index(mi, ni, NM, NN)
    return s if s < _v(NM) + _v(NN) else None


@gluon.constexpr_function
def _buffer_load_tile(pos, NM, NN, want_a):
    pos = _v(pos)
    if pos is None:
        return None
    is_a, tile = _buffer_load_order(NM, NN)[pos]
    return tile if bool(is_a) == bool(_v(want_a)) else None


@gluon.constexpr_function
def _ds_read_a_tile(mi, ni, NM, NN):
    return _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, True)


@gluon.constexpr_function
def _ds_read_b_tile(mi, ni, NM, NN):
    return _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, False)


@gluon.constexpr_function
def _scale_buffer_load_slot(is_a, tile, NM, NN, SCALE_FILL_MID=False):
    """A scale can be copied in a different slot from its payload."""
    if _v(SCALE_FILL_MID) and _v(NM) == 2 and _v(NN) == 2:
        return 1 + _v(tile)
    return _payload_buffer_load_slot(is_a, tile, NM, NN)


@gluon.constexpr_function
def _scale_buffer_load_tile(mi, ni, NM, NN, want_a, SCALE_FILL_MID=False):
    s = _slot_index(mi, ni, NM, NN)
    for tile in range(_v(NM) if _v(want_a) else _v(NN)):
        if _scale_buffer_load_slot(want_a, tile, NM, NN, SCALE_FILL_MID) == s:
            return tile
    return None


@gluon.constexpr_function
def _buffer_load_ops(tc, mi, ni):
    """A payload/scale and B payload/scale tiles, in issue order; None skips a copy."""
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    pos = _buffer_load_pos(mi, ni, NM, NN)
    ops = []
    for idx in range(2):
        ops.append(_buffer_load_tile(pos, NM, NN, idx == 0))
        scale_tile = None
        if tc.func_cfg.has_scale(idx) and tc.scale_via_lds(idx):
            scale_tile = _scale_buffer_load_tile(
                mi, ni, NM, NN, idx == 0, tc.SCALE_FILL_MID
            )
            if (
                idx == 0
                and scale_tile is not None
                and tc.scale_shuffled(0)
                and scale_tile % tc.scale_tile_ratio_a() != 0
            ):
                scale_tile = None
        ops.append(scale_tile)
    return tuple(ops)


@gluon.constexpr_function
def _buffer_load_group_schedule(tc):
    """Committed groups per slot, containing async ``(operand_kind, tile)`` copies.

    Kinds 0/1/2/3 name A payload/A scale/B payload/B scale. Direct-register payload
    and scale loads have no async group. Traffic-only experiments
    retain their nominal groups even when the LDS manager suppresses a copy.
    """
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    scheme = _v(tc.WAIT_COMMIT_SCHEME)
    assert scheme in tuple(int(s) for s in WaitCommitScheme)
    schedule = []
    stage_ops = ()
    for ni in range(NN):
        for mi in range(NM):
            ops = tuple(
                (kind, tile)
                for kind, tile in enumerate(_buffer_load_ops(tc, mi, ni))
                if tile is not None and (kind % 2 or tc.payload_via_lds(kind // 2))
            )
            if scheme == int(WaitCommitScheme.PER_OP):
                groups = tuple((op,) for op in ops)
            elif scheme == int(WaitCommitScheme.PER_SLOT):
                groups = (ops,)  # Empty tail slots still commit a group.
            else:
                stage_ops += ops
                groups = (stage_ops,) if mi == NM - 1 and ni == NN - 1 else ()
            schedule.append(groups)
    return tuple(schedule)


@gluon.constexpr_function
def _buffer_load_groups(tc):
    return tuple(group for slot in _buffer_load_group_schedule(tc) for group in slot)


@gluon.constexpr_function
def _buffer_load_wait(tc, mi, ni, STAGES_BETWEEN, DO_BUFFER_LOAD):
    """Number of newer committed groups after this slot's last required copy."""
    NM, NN = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    slot = _slot_index(mi, ni, NM, NN)
    if tc.commit_per_stage():
        return _v(STAGES_BETWEEN) if slot == 0 else None

    schedule = _buffer_load_group_schedule(tc)
    groups = tuple(group for entry in schedule for group in entry)
    required = []
    for idx in range(2):
        tile = _buffer_load_tile(_buffer_load_pos(mi, ni, NM, NN), NM, NN, idx == 0)
        if tile is None:
            continue
        if tc.payload_via_lds(idx):
            required.append((idx * 2, tile))
        if tc.func_cfg.has_scale(idx) and tc.scale_via_lds(idx):
            owner = tile
            if idx == 0 and tc.scale_shuffled(0):
                owner -= owner % tc.scale_tile_ratio_a()
            required.append((idx * 2 + 1, owner))
    if not required:
        return None
    latest = max(
        i for i, group in enumerate(groups) if any(op in group for op in required)
    )
    prefix = sum(len(entry) for entry in schedule[:slot]) if _v(DO_BUFFER_LOAD) else 0
    # The suffix comprises the target stage's remaining groups, full newer stages,
    # and the groups already committed while walking this stage's slots.
    return (_v(STAGES_BETWEEN) + 1) * len(groups) - 1 - latest + prefix
