# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The buffer-schedule audit of the Gluon MoE GEMM pipeline, shared by the config tests.

`_audit_buffer_schedule` replays a build's commit and wait plans against a complete
chronological simulation of its copies. Moved out of test_moe_gemm_gluon_config.py so
that file stays below 2000 lines; the helpers are unchanged.
"""

from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import Schedule
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    A_PAYLOAD,
    A_SCALE,
    B_PAYLOAD,
    B_SCALE,
    COMPONENTS,
    SCALE,
    A,
    B,
    Component,
    CompSlot,
    WaitCommitScheme,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf


def _plain(value):
    value = conf._cval(value)
    return tuple(value) if isinstance(value, list) else value


def _schedule(tc, layout, num_k=1, epilogue_groups=0):
    """The build's schedule for ``num_k`` K tiles; one tile still gives its sizing."""
    return Schedule(tc, layout, num_k * _plain(tc.BLOCK_K), epilogue_groups)


def _executed_step(schedule, num_k, k_step):
    """Index of the emitted step that executes K step ``k_step`` for this K.

    Stated from the loop structure alone: the prologue and peeled steps, whole loop
    bodies, the remainder, then the drain copy for the ring phase the main steps end
    on.
    """
    depth, peeled, unroll = (
        maybe_unwrap(schedule.depth),
        maybe_unwrap(schedule.peeled),
        maybe_unwrap(schedule.unroll),
    )
    main = num_k - depth
    if k_step < 0:
        return schedule.prologue_step(k_step)
    if k_step < peeled:
        return schedule.peeled_step(k_step)
    if k_step >= main:
        static_drain = maybe_unwrap(schedule.static_drain)
        phase = main % maybe_unwrap(schedule.buffer_period) if static_drain else 0
        return schedule.drain_step(phase, k_step - main)
    loop_end = peeled + (main - peeled) // unroll * unroll
    if k_step < loop_end:
        return schedule.loop_step((k_step - peeled) % unroll)
    if maybe_unwrap(schedule.static_remainder):
        return schedule.remainder_step(k_step - loop_end)
    return schedule.remainder_step(0)


def _audit_buffer_schedule(tc, layout, num_k, epilogue_groups=0):
    """Compare the commit and wait plans with a complete chronological simulation.

    The oracle records absolute K tiles and group sequence numbers, and does not use
    the schedule's replay. A register component has no DS read: the oracle checks
    that each MFMA reading one finds its copy already issued; the buffered-pipeline
    oracle separately checks exact ring-bank contents and overwrite safety.
    """
    schedule = _schedule(tc, layout, num_k, epilogue_groups)
    nm = _plain(maybe_unwrap(layout.num_m_slots))
    nn = _plain(maybe_unwrap(layout.num_n_slots))
    npanes = nm * nn
    if (nm, nn) == (2, 2):
        payloads = [
            (B_PAYLOAD, 0),
            (A_PAYLOAD, 0),
            (A_PAYLOAD, 1),
            (B_PAYLOAD, 1),
        ]
    else:
        payloads = []
        for tile in range(max(nm, nn)):
            if tile < nm:
                payloads.append((A_PAYLOAD, tile))
            if tile < nn:
                payloads.append((B_PAYLOAD, tile))
    has_scale = {operand: _plain(tc.func_cfg.has_scale(operand)) for operand in (A, B)}
    depths, rs, via_lds, producer_phases = {}, {}, {}, {}
    load_to_mfma_distances = {}
    for component in COMPONENTS:
        operand, is_scale = component.operand_index.value, component.is_scale.value
        depths[component] = _plain(tc.num_buffers(operand, is_scale))
        rs[component] = (
            _plain(layout.scale_slot_k(operand)) // _plain(layout.scale_block_k)
            if is_scale
            else 1
        )
        via_lds[component] = _plain(layout.component_via_lds(component))
        live_span = (depths[component] - 1) * rs[component] + 1
        load_to_mfma_distances[component] = live_span - int(not via_lds[component])
        producer_phases[component] = -load_to_mfma_distances[component] % rs[component]
    qs = {
        operand: _plain(layout.scale_slot_nonk(operand))
        // _plain(layout.slot_nonk(operand))
        for operand in (A, B)
    }
    payload_panes = {
        (component, tile): pane for pane, (component, tile) in enumerate(payloads)
    }
    buffer_load_panes = [[] for _ in range(npanes)]
    for pane, (component, tile) in enumerate(payloads):
        operand = component.operand_index.value
        assert not component.is_scale.value
        buffer_load_panes[pane].append((component, tile))
        if not has_scale[operand] or tile % qs[operand]:
            continue
        scale_slot = (
            tile + 1 if (nm, nn) == (2, 2) and _plain(tc.SCALE_FILL_MID) else pane
        )
        buffer_load_panes[scale_slot].append((Component(operand, SCALE), tile))
    component_rank = {
        A_PAYLOAD: 0,
        A_SCALE: 1,
        B_PAYLOAD: 2,
        B_SCALE: 3,
    }
    buffer_load_panes = [
        tuple(sorted(ops, key=lambda copy: component_rank[copy[0]]))
        for ops in buffer_load_panes
    ]
    buffer_load_pane_for = {
        copy: pane for pane, copies in enumerate(buffer_load_panes) for copy in copies
    }
    ds_read_panes = [[] for _ in range(npanes)]
    for copy in buffer_load_pane_for:
        component, tile = copy
        if not via_lds[component]:
            continue
        pane = payload_panes[(Component(component.operand_index.value, False), tile)]
        assert all(existing[0] != component for existing in ds_read_panes[pane])
        ds_read_panes[pane].append(copy)
    register_copies = [copy for copy in buffer_load_pane_for if not via_lds[copy[0]]]
    for pane, expected in enumerate(buffer_load_panes):
        actual = schedule.buffer_load_slots(pane)
        assert (
            tuple(
                (component, tile)
                for component, tile in zip(COMPONENTS, actual)
                if tile is not None
            )
            == expected
        )

    committed, group_for_copy, issued, consumed, lds = [], {}, set(), set(), {}
    depth = maybe_unwrap(schedule.depth)
    main = num_k - depth
    assert num_k >= depth + maybe_unwrap(schedule.peeled) + maybe_unwrap(
        schedule.unroll
    )
    scheme = _plain(tc.WAIT_COMMIT_SCHEME)
    per_stage = scheme in (
        WaitCommitScheme.PER_STAGE_WHOLE,
        WaitCommitScheme.PER_STAGE_WARP_PIPELINE,
    )

    def required_copies(stage, pane):
        # Step `stage` DS-reads K tile `stage + 1`, copied F steps before its MFMAs.
        need = []
        panes = range(npanes) if pane is None else (pane,)
        for ds_read_pane in panes:
            for component, tile in ds_read_panes[ds_read_pane]:
                if (stage + 1) % rs[component] == 0:
                    need.append(
                        (stage + 1 - load_to_mfma_distances[component], component, tile)
                    )
        return need

    def check_wait(stage, pane):
        required = [
            token for token in required_copies(stage, pane) if via_lds[token[1]]
        ]
        expected = (
            len(committed) - 1 - max(group_for_copy[token] for token in required)
            if required
            else None
        )
        # One immediate serves every execution of an emitted step, so this checks
        # each loop-body step once per loop iteration.
        assert schedule.wait(_executed_step(schedule, num_k, stage), pane) == expected

    def mfma_reads_registers(stage):
        # The MFMAs of `stage` read K tile `stage`; a register value is consumed by
        # the MFMA of its first tile.
        for component, tile in register_copies:
            if stage % rs[component]:
                continue
            token = (stage - load_to_mfma_distances[component], component, tile)
            assert token in issued, "register MFMA operand read before its producer"
            consumed.add(token)

    for stage in range(-depth, num_k - 1):
        pending_stage = []
        expected_groups = []
        stage_copies = [
            tuple(
                (stage, component, tile)
                for component, tile in ops
                if stage % rs[component] == producer_phases[component]
                and 0 <= stage + load_to_mfma_distances[component] < num_k
            )
            for ops in buffer_load_panes
        ]
        for pane, copies in enumerate(stage_copies):
            asynchronous = tuple(token for token in copies if via_lds[token[1]])
            # A pane or step that copies nothing through LDS commits no group.
            if scheme == WaitCommitScheme.PER_OP:
                groups = tuple((copy,) for copy in asynchronous)
            elif scheme == WaitCommitScheme.PER_PANE:
                groups = (asynchronous,) if asynchronous else ()
            else:
                pending_stage.extend(asynchronous)
                last = pane == npanes - 1
                groups = (tuple(pending_stage),) if last and pending_stage else ()
            expected_groups.append(groups)
        index = _executed_step(schedule, num_k, stage)
        actual_groups = tuple(schedule.commits(index, pane) for pane in range(npanes))
        assert actual_groups == tuple(
            tuple(tuple(CompSlot(*token[1:]) for token in group) for group in groups)
            for groups in expected_groups
        )

        def fill_pane(
            pane, copies_by_pane=stage_copies, groups_by_pane=expected_groups
        ):
            for token in copies_by_pane[pane]:
                producer, component, tile = token
                # The ring buffer is named by the K tile the copy carries, so it has
                # to be derived from the producer, not carried in the token.
                target = producer + load_to_mfma_distances[component]
                assert token not in issued, "a K tile was loaded twice"
                issued.add(token)
                if via_lds[component]:
                    ring = target // rs[component] % depths[component]
                    key = component, tile, ring
                    if key in lds:
                        # The previous occupant of this ring buffer must already
                        # have been read; name it by its producer, as `consumed`
                        # does, not by the step that consumed it.
                        assert (
                            lds[key] - load_to_mfma_distances[component],
                            component,
                            tile,
                        ) in consumed
                    lds[key] = target
            for group in groups_by_pane[pane]:
                for token in group:
                    group_for_copy[token] = len(committed)
                committed.append(group)

        def ds_read_pane(pane, ds_read_tile=stage + 1):
            for component, tile in ds_read_panes[pane]:
                if ds_read_tile % rs[component]:
                    continue
                token = (
                    ds_read_tile - load_to_mfma_distances[component],
                    component,
                    tile,
                )
                ring = ds_read_tile // rs[component] % depths[component]
                assert lds[component, tile, ring] == ds_read_tile
                consumed.add(token)

        if stage < -1:
            for pane in range(npanes):
                fill_pane(pane)
        if stage >= 0:
            mfma_reads_registers(stage)
        if stage >= -1:
            if per_stage:
                check_wait(stage, None)
            for pane in range(npanes):
                if not per_stage:
                    check_wait(stage, pane)
                ds_read_pane(pane)
                fill_pane(pane)
        if stage == main - 1:
            committed.extend(() for _ in range(epilogue_groups))
    mfma_reads_registers(num_k - 1)
    expected_loads = {
        (target - load_to_mfma_distances[component], component, tile)
        for ops in buffer_load_panes
        for component, tile in ops
        for target in range(0, num_k, rs[component])
    }
    assert issued == expected_loads
    assert consumed == expected_loads
