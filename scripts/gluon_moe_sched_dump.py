# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dump every constexpr the gfx950 Gluon MoE emitter consumes, for refactor diffing.

The live pipeline carries no runtime schedule state: `_pipeline.py` reads the build's
`Schedule` at trace time and folds it to constants. So if this dump is unchanged across
a refactor, and the emitter body is unchanged, the generated machine code is unchanged.
That makes an empty diff here a much cheaper -- and much sharper -- per-step gate than
a cold benchmark median, which drifts ~7 us between sessions with no code change.

    python scripts/gluon_moe_sched_dump.py > before.txt
    ...edit...
    python scripts/gluon_moe_sched_dump.py > after.txt
    diff -u before.txt after.txt && echo "SCHEDULE UNCHANGED"

CPU only: no kernel launch, no compile. It does import Triton, which needs a detectable
arch, so a GPU must be visible even though none is used.

Everything is read from the build's `Schedule`, one block per emitted step, so the dump
covers exactly the steps `_pipeline.py` expands.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import DtypeQuant as DQ  # noqa: E402
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import maybe_unwrap  # noqa: E402
from aiter.ops.triton._gluon_kernels.gfx950.moe._layout import (  # noqa: E402
    KernelLayout,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._schedule import (  # noqa: E402
    Schedule,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (  # noqa: E402
    COMPONENTS,
    WaitCommitScheme,
    _slot_index,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._validate import (  # noqa: E402
    validate,
    validate_pipeline,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host  # noqa: E402

CNAME = {c: n for c, n in zip(COMPONENTS, ("A_PAY", "A_SCL", "B_PAY", "B_SCL"))}

#: The benchmarked gemm1 geometry (scripts/gluon_moe_gemm1_best.py BEST), which is the
#: only configuration whose perf is recorded. Every case below is a delta from it.
BASE = dict(
    BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, K_UNROLL=3,
    SLOT_M=64, SLOT_N=128, NUM_LDS_BUFFER=3,
    mfma_instr_shape=(16, 16, 128), warps_per_cta=(1, 4), tiles_per_warp=(2, 2),
    k_width=None, transposed=True, WAVES_PER_EU=0, TILE_SCHED=2, GROUP_M=4, NUM_XCDS=8,
    token_cache_modifier="", token_scale_cache_modifier="",
    expert_cache_modifier="", expert_scale_cache_modifier="",
    result_cache_modifier="", result_scale_cache_modifier="",
    WARP_PIPELINE=1, VGPR_PREFETCH_K=256,
    A_SCALE_SORTED_SHUFFLED=True, B_SCALE_SHUFFLED=True, B_PRESHUFFLED=False,
    SCALE_SLOT_M=128, SCALE_FILL_MID=True,
    WAIT_COMMIT_SCHEME=int(WaitCommitScheme.PER_OP),
)

K_DEFAULT = 7168
N_DEFAULT = 4096


DTYPES = [("a4w4", DQ.MXFP4, DQ.MXFP4), ("a8w4", DQ.MXFP8, DQ.MXFP4),
          ("a8w8", DQ.MXFP8, DQ.MXFP8), ("bf16", DQ.BF16, DQ.BF16)]
#: BLOCK_M the router picks at T in {16, 64, 256, 1024, 4096} for E=33, top-k=8.
BLOCK_MS = [(16, 32), (64, 32), (256, 64), (1024, 128), (4096, 128)]


def _case(name, dq_a, dq_b, K=K_DEFAULT, **over):
    return (name, {**BASE, **over}, dq_a, dq_b, K)


def _production_cases():
    """Every shape the dispatcher actually selects, split to the finest legal geometry.

    The slot extent is pinned to the warp granularity `instr * warps * tiles`, the
    smallest value `validate()` accepts, i.e. the most slots this tile can be cut into. Where
    that still leaves NM == 1 or NN == 1 the case reports REJECTED: those are the
    shapes the fill schedule cannot express today, and the entry is here so the
    diff shows them becoming expressible.
    """
    out = []
    for label, dq_a, dq_b in DTYPES:
        for tokens, block_m in BLOCK_MS:
            seb = (dq_a == dq_b == DQ.MXFP8
                   and (tokens * 8 + 33 * block_m - 1) // (33 * block_m) <= 1)
            cfg = dict(host._default_launch_config(
                block_m, N_DEFAULT, K_DEFAULT, dq_a, dq_b, False, True,
                stream_expert_payload=seb))
            i, w, t = cfg["mfma_instr_shape"], cfg["warps_per_cta"], cfg["tiles_per_warp"]
            cfg["SLOT_M"] = i[0] * w[0] * t[0]
            cfg["SLOT_N"] = i[1] * w[1] * t[1]
            out.append((f"prod-{label}-t{tokens}", cfg, dq_a, dq_b, K_DEFAULT))
    return out


def cases():
    """Cover every axis the schedule model branches on, not every product of them."""
    out = _production_cases()
    # 1. the benchmarked arm under all four commit granularities
    for scheme in WaitCommitScheme:
        out.append(_case(f"tuned-mxfp4-{scheme.name}", DQ.MXFP4, DQ.MXFP4,
                         WAIT_COMMIT_SCHEME=int(scheme)))
    # 2. the SCALE_FILL_MID policy toggle (inert outside 2x2 -- proves it)
    out.append(_case("tuned-mxfp4-fillmid0", DQ.MXFP4, DQ.MXFP4, SCALE_FILL_MID=False))
    # 3. scale K-step ratio R > 1 (one scale load covers several payload stages)
    out.append(_case("mxfp4-bk128-sk8-R2", DQ.MXFP4, DQ.MXFP4, BLOCK_K=128,
                     VGPR_PREFETCH_K=128, SCALE_SLOT_K=8, K_UNROLL=1))
    out.append(_case("mxfp4-bk128-sk16-R4", DQ.MXFP4, DQ.MXFP4, BLOCK_K=128,
                     VGPR_PREFETCH_K=128, SCALE_SLOT_K=16, K_UNROLL=1))
    # 4. direct-register placement, per component and at the D=2 same-step boundary
    out.append(_case("breg", DQ.MXFP4, DQ.MXFP4, ALLOW_B_VIA_LDS=False,
                     B_PRESHUFFLED=True))
    out.append(_case("ascale-reg", DQ.MXFP4, DQ.MXFP4, ALLOW_A_SCALE_VIA_LDS=False))
    out.append(_case("bscale-reg", DQ.MXFP4, DQ.MXFP4, ALLOW_B_SCALE_VIA_LDS=False))
    out.append(_case("bscale-reg-d2", DQ.MXFP4, DQ.MXFP4, ALLOW_B_SCALE_VIA_LDS=False,
                     NUM_LDS_BUFFER=0, A_NUM_BUFFER=3, B_NUM_BUFFER=3,
                     A_SCALE_NUM_BUFFER=3, B_SCALE_NUM_BUFFER=2))
    # 5. unequal ring depths
    out.append(_case("depths-2332", DQ.MXFP4, DQ.MXFP4, NUM_LDS_BUFFER=0,
                     A_NUM_BUFFER=2, B_NUM_BUFFER=3, A_SCALE_NUM_BUFFER=3,
                     B_SCALE_NUM_BUFFER=2))
    # 6. slot geometry: the 2x2 reference order, and a rectangle
    out.append(_case("slots-4x2", DQ.MXFP4, DQ.MXFP4, SLOT_M=32))
    # 7. read placement in the MFMA region rather than the memory region
    out.append(_case("ds-read-mfma", DQ.MXFP4, DQ.MXFP4,
                     DS_READ_A_PAYLOAD_IN_MFMA=True, DS_READ_A_SCALE_IN_MFMA=True,
                     DS_READ_B_PAYLOAD_IN_MFMA=True, DS_READ_B_SCALE_IN_MFMA=True))
    # 8. the soffset-addressed unrolled body
    out.append(_case("soff-unroll", DQ.MXFP4, DQ.MXFP4, SOFF_UNROLL=True))
    return out


def _steps(schedule):
    """Every emitted step, labelled the way `_pipeline.py` expands it.

    Each entry is `(label, step, static_phase)`; only an unrolled remainder or drain
    has a static phase outside the loop.
    """
    depth, unroll = maybe_unwrap(schedule.depth), maybe_unwrap(schedule.unroll)
    out = [
        (f"prologue r={r}", schedule.prologue_step(r), False)
        for r in range(1 - depth, 0)
    ]
    out.append(("warmup", schedule.warmup_step(), False))
    out += [
        (f"peeled x={x}", schedule.peeled_step(x), False)
        for x in range(maybe_unwrap(schedule.peeled))
    ]
    out += [(f"loop u={u}", schedule.loop_step(u), False) for u in range(unroll)]
    static_remainder = maybe_unwrap(schedule.static_remainder)
    count = unroll - 1 if static_remainder else 1
    out += [
        (f"remainder u={u}", schedule.remainder_step(u), static_remainder)
        for u in range(count)
    ]
    static_drain = maybe_unwrap(schedule.static_drain)
    phases = maybe_unwrap(schedule.ring_period) if static_drain else 1
    out += [
        (f"drain p={p} j={j}", schedule.drain_step(p, j), static_drain)
        for p in range(phases)
        for j in range(depth - 1)
    ]
    return out


def dump_case(name, cfg, dq_a, dq_b, K, out):
    try:
        tc = host._probe_tuning_config(cfg, dq_a, dq_b)
        layout = KernelLayout(tc)
        validate(tc, N_DEFAULT, K, layout)
        schedule = Schedule(tc, layout, K)
        validate_pipeline(tc, layout, K, schedule)
    except Exception as exc:  # a rejected config is itself a stable, diffable fact
        out(f"== {name}: REJECTED {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return

    nm, nn = maybe_unwrap(layout.num_m_slots), maybe_unwrap(layout.num_n_slots)
    slots = nm * nn
    present = [c for c in COMPONENTS if schedule.component_present(c)]
    async_period = math.lcm(
        *(schedule.steps_per_buffer(c) for c in present if layout.component_via_lds(c))
    )

    out(f"== {name}")
    out(f"  geom NM={nm} NN={nn} slots={slots} mk={maybe_unwrap(layout.num_k_slots)} "
        f"num_k={maybe_unwrap(schedule.num_k)} scheme={int(tc.WAIT_COMMIT_SCHEME)}")
    out(f"  sizing depth={maybe_unwrap(schedule.depth)} "
        f"unroll={maybe_unwrap(schedule.unroll)} "
        f"peeled={maybe_unwrap(schedule.peeled)} "
        f"ring={maybe_unwrap(schedule.ring_period)} "
        f"regperiod={maybe_unwrap(schedule.register_period)} "
        f"async={async_period} "
        f"hasreg={any(schedule.component_present(c) and layout.component_in_reg(c) for c in COMPONENTS)}")
    out(f"  regions static_remainder={maybe_unwrap(schedule.static_remainder)} "
        f"static_drain={maybe_unwrap(schedule.static_drain)} "
        f"epilogue_groups={maybe_unwrap(schedule.epilogue_groups)} "
        f"epilogue_wait={schedule.epilogue_wait()}")
    out(f"  commit per_copy={maybe_unwrap(schedule.commit_per_copy)} "
        f"per_slot={maybe_unwrap(schedule.commit_per_slot)} "
        f"per_tile={maybe_unwrap(schedule.commit_per_tile)} "
        f"tile_after_mfma={maybe_unwrap(schedule.tile_commit_after_mfma)}")

    for c in COMPONENTS:
        if not schedule.component_present(c):
            out(f"  {CNAME[c]} absent")
            continue
        non_k_slots = nm if c[0] == 0 else nn
        D, R = schedule.num_buffers(c), schedule.steps_per_buffer(c)
        F = schedule.load_to_mfma_distance(c)
        out(f"  {CNAME[c]} D={D} R={R} L={(D - 1) * R + 1} F={F} "
            f"via_lds={bool(layout.component_via_lds(c))} pphase={(1 - F) % R} "
            f"nonk_ratio={layout.scale_slot_nonk(c[0]) // layout.slot_nonk(c[0])} "
            f"buffer_load_slots={tuple(schedule.buffer_load_slot(c, t) for t in range(non_k_slots))} "
            f"ds_read_slots={tuple(schedule.ds_read_slot(c, t) for t in range(non_k_slots))} "
            f"ds_read_in_mfma={tuple(schedule.ds_read_in_mfma(c, t) for t in range(non_k_slots))}")

    # ownership and DS reads, per slot
    for ni in range(nn):
        for mi in range(nm):
            s = _slot_index(mi, ni, nm, nn)
            ds_read = tuple(schedule.ds_read_non_k_slot(s, c) for c in COMPONENTS)
            out(f"  slot{s}(mi={mi},ni={ni}) owned={schedule.owned(s)} "
                f"ds_read_non_k_slots={ds_read}")

    # every emitted step: what it copies, commits and waits on, and its ring indices
    for label, step, static_phase in _steps(schedule):
        out(f"  step {step} {label}: phase={schedule.phase(step)} "
            f"ki={schedule.ki(step)} in_loop={schedule.in_loop(step)} "
            f"static_phase={static_phase} dot={schedule.dot(step)} "
            f"soff={schedule.soff_unroll(step)}")
        act = tuple(int(schedule.active(step, c)) for c in COMPONENTS)
        loads = tuple(int(schedule.loads(step, c)) for c in COMPONENTS)
        ds_reads = tuple(int(schedule.ds_reads(step, c)) for c in COMPONENTS)
        out(f"    active={act} loads={loads} ds_reads={ds_reads}")
        out(f"    issued={tuple(schedule.issued(step, s) for s in range(slots))}")
        commits = tuple(
            tuple(tuple((CNAME[c], t) for c, t in g) for g in schedule.commits(step, s))
            for s in range(slots)
        )
        out(f"    commits={commits}")
        out(f"    wait tile={schedule.wait(step)} "
            f"slots={[schedule.wait(step, s) for s in range(slots)]}")
        for c in present:
            out(f"    {CNAME[c]} reg={schedule.register_index(step, c)}/"
                f"{schedule.register_index(step, c, True)} "
                f"lds={schedule.static_lds_index(step, c, False)}/"
                f"{schedule.static_lds_index(step, c, True)}")

    # the soffset stream
    for c in present:
        if c[1]:
            unroll = maybe_unwrap(schedule.unroll)
            soff = [schedule.scale_soff_steps(c, o) for o in range(unroll)]
            out(f"  soff {CNAME[c]} {soff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter on case name")
    args = ap.parse_args()
    lines = []
    for name, cfg, dq_a, dq_b, K in cases():
        if args.only and args.only not in name:
            continue
        dump_case(name, cfg, dq_a, dq_b, K, lines.append)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
