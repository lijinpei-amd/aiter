# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dump every constexpr the gfx950 Gluon MoE emitter consumes, for refactor diffing.

The live pipeline carries no runtime schedule state: `_pipeline.py` reads the schedule
model at trace time and folds it to constants. So if this dump is unchanged across a
refactor, and the emitter body is unchanged, the generated machine code is unchanged.
That makes an empty diff here a much cheaper -- and much sharper -- per-step gate than
a cold benchmark median, which drifts ~7 us between sessions with no code change.

    python scripts/gluon_moe_sched_dump.py > before.txt
    ...edit...
    python scripts/gluon_moe_sched_dump.py > after.txt
    diff -u before.txt after.txt && echo "SCHEDULE UNCHANGED"

CPU only: no kernel launch, no compile. It does import Triton, which needs a detectable
arch, so a GPU must be visible even though none is used.

`_wait()` is undefined for fill-only prologue stages (nothing has been read yet, so the
demand is unsatisfiable) and raises there; those cells print `unreachable` rather than
aborting the dump.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aiter.ops.triton._gluon_kernels.gfx950.moe import _pipeline as P  # noqa: E402
from aiter.ops.triton._gluon_kernels.gfx950.moe import _schedule as S  # noqa: E402
from aiter.ops.triton._gluon_kernels.gfx950.moe._lang import DtypeQuant as DQ  # noqa: E402
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (  # noqa: E402
    COMPONENTS,
    WaitCommitScheme,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host  # noqa: E402

CNAME = {c: n for c, n in zip(COMPONENTS, ("A_PAY", "A_SCL", "B_PAY", "B_SCL"))}

#: The benchmarked gemm1 geometry (scripts/gluon_moe_gemm1_best.py BEST), which is the
#: only configuration whose perf is recorded. Every case below is a delta from it.
BASE = dict(
    BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, K_UNROLL=3,
    MINI_BLOCK_M=64, MINI_BLOCK_N=128, NUM_LDS_BUFFER=3,
    mfma_instr_shape=(16, 16, 128), warps_per_cta=(1, 4), tiles_per_warp=(2, 2),
    k_width=None, transposed=True, WAVES_PER_EU=0, TILE_SCHED=2, GROUP_M=4, NUM_XCDS=8,
    token_cache_modifier="", token_scale_cache_modifier="",
    expert_cache_modifier="", expert_scale_cache_modifier="",
    result_cache_modifier="", result_scale_cache_modifier="",
    WARP_PIPELINE=1, VGPR_PREFETCH_K=256,
    A_SCALE_SORTED_SHUFFLED=True, B_SCALE_SHUFFLED=True, B_PRESHUFFLED=False,
    SCALE_MINI_BLOCK_M=128, SCALE_FILL_MID=True,
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

    MINI_BLOCK is pinned to the warp granularity `instr * warps * tiles`, the smallest
    value `validate()` accepts, i.e. the most slots this tile can be cut into. Where
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
            cfg["MINI_BLOCK_M"] = i[0] * w[0] * t[0]
            cfg["MINI_BLOCK_N"] = i[1] * w[1] * t[1]
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
    out.append(_case("mxfp4-bk128-sk256-R2", DQ.MXFP4, DQ.MXFP4, BLOCK_K=128,
                     VGPR_PREFETCH_K=128, SCALE_MINI_BLOCK_K=256, K_UNROLL=1))
    out.append(_case("mxfp4-bk128-sk512-R4", DQ.MXFP4, DQ.MXFP4, BLOCK_K=128,
                     VGPR_PREFETCH_K=128, SCALE_MINI_BLOCK_K=512, K_UNROLL=1))
    # 4. direct-register placement, per component and at the D=2 same-step boundary
    out.append(_case("breg", DQ.MXFP4, DQ.MXFP4, B_IN_REG=True, B_PRESHUFFLED=True))
    out.append(_case("ascale-reg", DQ.MXFP4, DQ.MXFP4, A_SCALE_IN_REG=True))
    out.append(_case("bscale-reg", DQ.MXFP4, DQ.MXFP4, B_SCALE_IN_REG=True))
    out.append(_case("bscale-reg-d2", DQ.MXFP4, DQ.MXFP4, B_SCALE_IN_REG=True,
                     NUM_LDS_BUFFER=0, A_NUM_BUFFER=3, B_NUM_BUFFER=3,
                     A_SCALE_NUM_BUFFER=3, B_SCALE_NUM_BUFFER=2))
    # 5. unequal ring depths
    out.append(_case("depths-2332", DQ.MXFP4, DQ.MXFP4, NUM_LDS_BUFFER=0,
                     A_NUM_BUFFER=2, B_NUM_BUFFER=3, A_SCALE_NUM_BUFFER=3,
                     B_SCALE_NUM_BUFFER=2))
    # 6. slot geometry: the 2x2 reference order, and a rectangle
    out.append(_case("slots-4x2", DQ.MXFP4, DQ.MXFP4, MINI_BLOCK_M=32))
    # 7. read placement in the MFMA region rather than the memory region
    out.append(_case("ds-read-mfma", DQ.MXFP4, DQ.MXFP4,
                     DS_READ_A_PAYLOAD_IN_MFMA=True, DS_READ_A_SCALE_IN_MFMA=True,
                     DS_READ_B_PAYLOAD_IN_MFMA=True, DS_READ_B_SCALE_IN_MFMA=True))
    # 8. the soffset-addressed unrolled body
    out.append(_case("soff-unroll", DQ.MXFP4, DQ.MXFP4, SOFF_UNROLL=True))
    return out


def _wait(tc, *a, **kw):
    """`_wait` is partial: undefined on fill-only prologue stages (see module docstring)."""
    try:
        return S._wait(tc, *a, **kw)
    except ValueError:
        return "unreachable"


def dump_case(name, cfg, dq_a, dq_b, K, out):
    try:
        tc = host._probe_tuning_config(cfg, dq_a, dq_b)
        tc.validate(N_DEFAULT, K)
    except Exception as exc:  # a rejected config is itself a stable, diffable fact
        out(f"== {name}: REJECTED {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        return

    nm, nn = tc.num_m_slots_per_block(), tc.num_n_slots_per_block()
    slots = nm * nn
    depth = S.pipeline_depth(tc)
    unroll = S.pipeline_unroll(tc)
    peeled = S._pipeline_peeled(tc)
    num_k = tc.num_k_tiles(K)

    out(f"== {name}")
    out(f"  geom NM={nm} NN={nn} slots={slots} mk={tc.num_k_slots_per_tile()} "
        f"num_k={num_k} scheme={int(tc.WAIT_COMMIT_SCHEME)}")
    out(f"  sizing depth={depth} unroll={unroll} peeled={peeled} "
        f"regperiod={tc.pipeline_register_period()} hasreg={S._has_register_component(tc)}")
    out(f"  order {S._buffer_load_order(nm, nn)}")

    for c in COMPONENTS:
        if not S._present(tc, c):
            out(f"  {CNAME[c]} absent")
            continue
        out(f"  {CNAME[c]} D={S._component_depth(tc, c)} R={S._component_ratio(tc, c)} "
            f"L={S._live_span(tc, c)} F={S._fill_span(tc, c)} "
            f"via_lds={S._via_lds(tc, c)} pphase={S._producer_phase(tc, c)} "
            f"nonk_ratio={tc.scale_ratio_non_k_slot(c[0])}")

    # ownership and read binding, per slot
    for ni in range(nn):
        for mi in range(nm):
            s = S._slot_index(mi, ni, nm, nn)
            ops = S._ops(tc, mi, ni)
            reads = [S._component_read_tile(tc, c, mi, ni) for c in COMPONENTS]
            mfma = [S._read_in_mfma(tc, c, t) if t is not None else None
                    for c, t in zip(COMPONENTS, reads)]
            out(f"  slot{s}(mi={mi},ni={ni}) ops={ops} read={reads} in_mfma={mfma}")
    for c in COMPONENTS:
        if not S._present(tc, c):
            continue
        n_tiles = nm if c[0] == 0 else nn
        fill = tuple(S._component_fill_slot(tc, c, t) for t in range(n_tiles))
        rd = tuple(S._component_read_slot(tc, c, t) for t in range(n_tiles))
        out(f"  {CNAME[c]} fill_slots={fill} read_slots={rd}")

    # activity and committed groups across every reachable region
    for stage in range(1 - depth, depth + 2):
        act = tuple(int(S._active(tc, c, stage)) for c in COMPONENTS)
        out(f"  active stage={stage} {act} groups={S._groups(tc, stage)}")
    out(f"  active steady {tuple(int(S._active(tc, c, None)) for c in COMPONENTS)} "
        f"groups={S._groups(tc, None)}")
    for j in range(depth - 1):
        act = tuple(int(S._active(tc, c, j, True)) for c in COMPONENTS)
        out(f"  active drain j={j} {act} groups={S._groups(tc, j, True)}")

    # the emitted wait immediates, which are the perf artifact
    for stage in range(0, depth + 2):
        w = [_wait(tc, stage, s) for s in range(slots)]
        out(f"  wait startup stage={stage} head={_wait(tc, stage, None)} slots={w}")
    for phase in range(peeled + 1, peeled + 1 + unroll):
        w = [_wait(tc, None, s, phase=phase) for s in range(slots)]
        out(f"  wait steady phase={phase} head={_wait(tc, None, None, phase=phase)} slots={w}")
    for epi in (0, 1):
        for j in range(depth - 1):
            ph = num_k - depth + j + 1
            w = [_wait(tc, j, s, True, epi, ph) for s in range(slots)]
            out(f"  wait drain j={j} epi={epi} phase={ph} "
                f"head={_wait(tc, j, None, True, epi, ph)} slots={w}")

    # register-ring indices and the soffset stream
    for c in COMPONENTS:
        if not S._present(tc, c):
            continue
        for fill in (False, True):
            row = [P._register_index(tc, c, ph, 0, False, fill) for ph in range(unroll + 2)]
            loop = [P._register_index(tc, c, 0, ki, True, fill) for ki in range(unroll)]
            out(f"  regidx {CNAME[c]} fill={int(fill)} finite={row} in_loop={loop}")
        if c[1]:
            soff = [P._scale_soff_steps(tc, c, o) for o in range(unroll)]
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
