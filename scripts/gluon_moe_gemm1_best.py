# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Re-run the tuned gfx950 Gluon MoE GEMM1 configuration: perf and correctness.

The host launcher translates this script's environment variables into kernel tuning
configuration. ``BEST`` records the tuned recipe, including ``WAIT_COMMIT_SCHEME``
and ``TRITON_MEMBAR_DEDUP_BARE``, so measurements can be reproduced. The frozen
reference owns its fixed rendezvous schedule.

Perf is measured the way the reported numbers were: ``rocprofv3 --kernel-trace`` over a
saturating back-to-back launch loop, median of the dispatch durations with the warmup
dispatches dropped. The warmup is long (40) on purpose: this kernel takes ~30 dispatches
to reach steady state, and a short one leaves that ramp in the sample. Wall-clock timing around a sync would fold in the host launch path and the
``sort_scales`` kernel that runs between GEMM1 dispatches, which is why this re-execs
itself under the profiler rather than calling ``torch.cuda.Event``.

    python scripts/gluon_moe_gemm1_best.py                  # both geometries, perf + tests
    python scripts/gluon_moe_gemm1_best.py --warps 8 --perf
    python scripts/gluon_moe_gemm1_best.py --correctness --gpu 4

Requires the local ``TRITON_MEMBAR_DEDUP_BARE`` patch to Triton's Membar.cpp; without it
that flag is ignored and the kernel keeps a duplicate barrier per pipeline slot. The
script checks for it and warns rather than silently reporting a slower number.
"""

import argparse
import csv
import glob
import os
import shutil
import statistics as s
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Shape the tuned config was measured on: DeepSeek-V4-ish stage 1, saturating.
HIDDEN, INTER, N_EXPERTS, TOPK = 7168, 2048, 33, 8

TEST_FILE = "op_tests/triton_tests/moe/test_moe_gemm_a4w4.py"

#: The tuned flag set. Everything here is a deviation from the default that was measured
#: to matter; the ones worth calling out:
#:   SHUFFLED_W_SCALES -- the CDNA4_SCALE weight-scale preshuffle.
#:   WAIT_COMMIT_SCHEME -- the copy-commit and wait granularity for the live step.
#: The two occupancy knobs live in PER_WARPS below rather than here, because they are
#: only safe at 8 waves; see the comment there.
BEST = {
    "AITER_MOE_NUM_EXPERT_ACTIVATED": "33",
    "AITER_TRITON_MOE_GLUON_FLY": "1",
    "AITER_TRITON_MOE_GLUON_FLY_TILES_M": "2",
    "AITER_TRITON_MOE_GLUON_FLY_TILES_N": "2",
    "AITER_TRITON_MOE_GLUON_MINI_BLOCK_M": "64",
    "AITER_TRITON_MOE_GLUON_MINI_BLOCK_N": "128",
    "AITER_TRITON_MOE_GLUON_SCALE_MINI_BLOCK_M": "128",
    "AITER_TRITON_MOE_GLUON_SORTED_SCALES": "1",
    "AITER_TRITON_MOE_GLUON_SHUFFLED_W_SCALES": "1",
    "AITER_TRITON_MOE_GLUON_SCALE_FILL_MID": "1",
    # WarpPipeline: 0 none, 1 the TritonAMDGPUWarpPipeline pass, 2 the hand-emitted
    # rendezvous (which only _pipeline_step_frozen implements, so it needs FROZEN_STEP=1
    # -- the live step refuses it rather than running unpipelined).
    "AITER_TRITON_MOE_GLUON_WARP_PIPELINE": "0",
    "AITER_TRITON_MOE_GLUON_DS_IN_MFMA": "1",
    "AITER_TRITON_MOE_GLUON_ACT_FAST_RCP": "1",
    # WaitCommitScheme.PER_OP: commit each async payload/scale copy and wait per
    # read slot. The cold-bench impl arm overrides this with PER_STAGE (3); the
    # frozen arm keeps its private snapshot's commit/wait schedule.
    "AITER_TRITON_MOE_GLUON_WAIT_COMMIT_SCHEME": "1",
    "TRITON_MEMBAR_DEDUP_BARE": "1",
}

#: 8 waves is warps (2, 4); 4 waves is warps (1, 4). Same tile shape either way.
WARPS_N = {8: "2", 4: "1"}

#: Locally built llc carrying the "amdgpu-ds-read-agpr" support (an out-of-tree
#: change to AMDGPUPrepareAGPRAlloc / SIRegisterInfo). Triton's external-llc hook
#: shells out to this instead of its own linked LLVM, so the rest of the toolchain
#: is untouched. Override with TRITON_HIP_DS_AGPR_LLC.
PATCHED_LLC = os.environ.get(
    "TRITON_HIP_DS_AGPR_LLC",
    str(Path.home() / "development/llvm-project/build/bin/llc"),
)

#: Per-geometry additions: two occupancy knobs, both safe only at 8 waves.
#:   TRITON_HIP_AGPR_ALLOC=0 -- upstream Triton env, sets the amdgpu-agpr-alloc fn attr.
#:     Puts the whole accumulator in VGPRs, 245/0 instead of 208/80, and is worth ~6 us:
#:     673.9 -> 667.4 us mean over four interleaved pairs, with the two ranges not
#:     overlapping.
#:   WAVES_PER_EU=2 -- measured neutral at 8 waves and kept deliberately. It does reach
#:     the kernel ("amdgpu-waves-per-eu"="2,2" in the gemm1 IR) but moves no register
#:     count, because 208+80=288 registers already pin occupancy at one wave per SIMD.
#: Neither may be applied at 4 waves: that kernel needs 349 VGPR + 93 AGPR = 442, so
#: either cap forces ~370 spills and takes it from ~677 us to ~4000 us. WAVES_PER_EU is
#: the worse of the two there -- it alone still cost 4200 us with AGPR_ALLOC left unset.
#: Both geometries tie each MFMA's D operand to its C operand
#: ("-amdgpu-mfma-tied-cd"), forcing the accumulate chain to update one register in
#: place. On 8 waves that takes MFMAs with D==C from 55/224 to 208/224 and drops
#: vgpr_count 245 -> 201.
#:
#: 4 waves additionally forces LDS reads that feed only MFMA A/B into AGPRs
#: ("-amdgpu-ds-read-agpr"): 102 of 156 ds_reads move to AGPRs with zero
#: v_accvgpr_read. Measured perf-neutral on its own. That one is inert on 8 waves,
#: because AGPR_ALLOC=0 there reserves AGPR0 and the pass bails out.
#:
#: Both need the patched llc; without it the flags are simply not passed.
PER_WARPS = {
    8: {
        "TRITON_HIP_AGPR_ALLOC": "0",
        "AITER_TRITON_MOE_GLUON_WAVES_PER_EU": "2",
        "TRITON_HIP_EXTERNAL_LLC": PATCHED_LLC,
        "TRITON_HIP_EXTERNAL_LLC_FLAGS": "-amdgpu-mfma-tied-cd",
    },
    4: {
        "TRITON_HIP_EXTERNAL_LLC": PATCHED_LLC,
        "TRITON_HIP_EXTERNAL_LLC_FLAGS": ("-amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd"),
    },
}


def build_env(warps, cache_dir, gpu, overrides=None):
    """The tuned set, with every other knob of ours cleared so a dirty shell can't leak in."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            ("AITER_TRITON_MOE_GLUON_", "TRITON_HIP_", "TRITON_MEMBAR_")
        )
    }
    env.update(BEST)
    env.update(PER_WARPS[warps])
    env["AITER_TRITON_MOE_GLUON_FLY_WARPS_N"] = WARPS_N[warps]
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    if gpu is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
    # Applied last so --set wins over BEST: that is the point of it.
    env.update(overrides or {})
    return env


def check_ds_agpr_llc():
    """The patched llc is outside this repo, so verify it exists and has the flag."""
    llc = Path(PATCHED_LLC)
    if not llc.exists():
        return "missing"
    try:
        out = subprocess.run(
            [str(llc), "--help-hidden"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unusable"
    return "ok" if "amdgpu-ds-read-agpr" in out else "unpatched"


def check_membar_patch():
    """The DEDUP_BARE flag lives in a Triton source patch, not in this repo."""
    src = os.environ.get("TRITON_SRC_DIR")
    candidates = [Path(src) if src else None, Path.home() / "development/triton"]
    for root in candidates:
        membar = root / "lib/Analysis/Membar.cpp" if root else None
        if membar and membar.exists():
            return "TRITON_MEMBAR_DEDUP_BARE" in membar.read_text()
    return None  # could not tell


# --------------------------------------------------------------------------- workload


def run_workload(tokens, reps, warmup):
    """Saturating GEMM1: one kernel, launched back-to-back, nothing else queued."""
    sys.path.insert(0, str(REPO))
    import torch

    from aiter.ops.triton.moe.moe_op_gemm_gluon import moe_gemm_gluon
    from op_tests.op_benchmarks.triton.bench_moe_gemm_gluon import (
        _OPS,
        _build,
        _explicit_recipe,
    )

    recipe = _explicit_recipe(HIDDEN, INTER, N_EXPERTS, TOPK)
    _wrapper, xd, wd = _OPS["a4w4"]
    sh = recipe.gemm_shape(1, tokens)
    n, k = sh.n, sh.k
    rdata, gindx, _sindx, x, xs, w, ws, bias, gammas = _build(
        tokens,
        n,
        k,
        sh.n_expts_tot,
        sh.n_expts_act,
        "cuda",
        xd,
        wd,
        n_active=N_EXPERTS,
    )
    if os.environ.get("AITER_TRITON_MOE_GLUON_GU_SPLIT", "0") != "0":
        # The split packing reads gate from [0, N/2) and up from [N/2, N), so the arm is
        # only meaningful on permuted weights. Done once, outside the timed loop, exactly
        # as a real caller would store them -- see activations.py::gate_up_split_perm.
        from aiter.ops.triton._triton_kernels.moe.activations import (
            gate_up_split_perm,
        )

        perm = gate_up_split_perm(n).to(w.device)

        def _n_perm(t):
            # Keep the K-contiguous (stride(-2) == 1) layout the wrapper requires; a
            # plain .contiguous() after the gather would make N contiguous instead.
            return t[..., perm].transpose(-1, -2).contiguous().transpose(-1, -2)

        w, ws = _n_perm(w), _n_perm(ws)
        if bias is not None:
            bias = bias[..., perm].contiguous()

    y = torch.empty((1, gindx.shape[0], n // 2), dtype=torch.bfloat16, device="cuda")

    def call():
        moe_gemm_gluon(
            y,
            x,
            w,
            xs,
            ws,
            bias,
            gammas,
            rdata,
            gindx,
            None,
            n,
            k,
            True,
            recipe.swiglu.alpha,
            recipe.swiglu.limit,
            False,
        )

    # Long, and deliberately so. The kernel needs ~30 dispatches to reach steady state:
    # block medians over one run fall 743 -> 716 -> 671 -> 660 before flattening near 656.
    # A short warmup leaves that ramp inside the measured window, which both inflates the
    # number and makes it jump run to run depending on how much ramp landed in the sample.
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    print(
        f"saturating: T={tokens} M={gindx.shape[0]} N={n} K={k} "
        f"warmup={warmup} reps={reps}",
        flush=True,
    )
    for _ in range(reps):  # no sync inside: the queue stays full
        call()
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------- perf


def parse_trace(trace_dir, skip):
    files = glob.glob(str(Path(trace_dir) / "**" / "*kernel_trace.csv"), recursive=True)
    if not files:
        return None
    rows = []
    with open(files[0]) as fh:
        for rec in csv.DictReader(fh):
            if "moe_gluon_gemm1" in rec["Kernel_Name"]:
                rows.append((int(rec["Start_Timestamp"]), int(rec["End_Timestamp"])))
    rows.sort()
    # skip is the warmup count: drop exactly the dispatches the workload already
    # labelled as warmup, so the measured window is steady state only.
    if len(rows) <= skip + 1:
        return None
    rows = rows[skip:]
    dur = [(e - s) / 1000 for s, e in rows]
    gap = [(rows[i + 1][0] - rows[i][1]) / 1000 for i in range(len(rows) - 1)]
    med, medgap = s.median(dur), s.median(gap)
    return {
        # Median, not mean: the tail of this distribution is long and one-sided (a
        # descheduled dispatch can be 60+ us slow, nothing is ever fast), so the mean
        # tracks how unlucky the run was rather than how fast the kernel is.
        "median": med,
        "mean": s.mean(dur),
        "min": min(dur),
        "max": max(dur),
        "n": len(dur),
        # Not idle time: sort_scales runs between GEMM1 dispatches on the same stream.
        "duty": 100 * med / (med + medgap),
    }


def read_regs(cache_dir):
    import re

    hits = glob.glob(
        str(Path(cache_dir) / "**" / "_moe_gluon_gemm1.amdgcn"), recursive=True
    )
    if not hits:
        return ""
    text = Path(hits[0]).read_text()

    def field(name):
        m = re.search(rf"\.{name}:\s*(\d+)", text)
        return m.group(1) if m else "?"

    return (
        f"vgpr={field('vgpr_count')} agpr={field('agpr_count')} "
        f"sgpr={field('sgpr_count')} spill={field('vgpr_spill_count')}"
    )


def run_perf(warps, args, workdir):
    rocprofv3 = Path(sys.executable).parent / "rocprofv3"
    if not rocprofv3.exists():
        found = shutil.which("rocprofv3")
        if not found:
            print("  rocprofv3 not found -- skipping perf", file=sys.stderr)
            return None
        rocprofv3 = Path(found)

    cache_dir = workdir / f"cache_w{warps}"
    trace_dir = workdir / f"trace_w{warps}"
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(trace_dir, ignore_errors=True)
    trace_dir.mkdir(parents=True)
    env = build_env(warps, cache_dir, args.gpu, args.overrides)

    cmd = [
        str(rocprofv3),
        "--kernel-trace",
        "-d",
        str(trace_dir),
        "-o",
        "kt",
        "--output-format",
        "csv",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--_workload",
        "--tokens",
        str(args.tokens),
        "--reps",
        str(args.reps),
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    stats = parse_trace(trace_dir, args.warmup)
    if stats is None:
        print(f"  {warps}-wave perf: FAILED (no dispatches traced)")
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        print("\n".join("    " + ln for ln in tail), file=sys.stderr)
        return None
    print(
        f"  {warps}-wave perf: median={stats['median']:6.1f} us  mean={stats['mean']:6.1f}"
        f"  min={stats['min']:6.1f}  max={stats['max']:6.1f}  n={stats['n']}  "
        f"duty={stats['duty']:.1f}%  {read_regs(cache_dir)}"
    )
    return stats


# ------------------------------------------------------------------------ correctness


def run_correctness(warps, args, workdir):
    cache_dir = workdir / f"corr_cache_w{warps}"
    shutil.rmtree(cache_dir, ignore_errors=True)
    env = build_env(warps, cache_dir, args.gpu, args.overrides)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--color=no",
        "-p",
        "no:cacheprovider",
        TEST_FILE,
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=args.timeout,
        check=False,
    )
    last = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary = last[-1] if last else "no output"
    print(f"  {warps}-wave tests: {summary}")
    if proc.returncode != 0:
        for line in proc.stdout.splitlines():
            if line.startswith("FAILED"):
                print("    " + line)
    return proc.returncode == 0


# ------------------------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="gluon_moe_gemm1_best",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--warps", choices=["8", "4", "both"], default="both")
    p.add_argument("--perf", action="store_true", help="perf only (default: both)")
    p.add_argument(
        "--correctness", action="store_true", help="tests only (default: both)"
    )
    p.add_argument("--gpu", type=int, default=None, help="HIP_VISIBLE_DEVICES")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument(
        "--warmup",
        type=int,
        default=40,
        help="dispatches to run and then discard; must clear the ~30-dispatch ramp",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="perf runs per geometry; spread is ~+-6 us, so >1 is worth it",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_env",
        help="override or add one env var on top of BEST; repeatable. "
        "e.g. --set AITER_TRITON_MOE_GLUON_WAVES_PER_EU=2 --set TRITON_HIP_AGPR_ALLOC=0",
    )
    p.add_argument("--timeout", type=int, default=3000)
    p.add_argument("--_workload", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    args.overrides = {}
    for item in args.set_env:
        if "=" not in item:
            p.error(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        args.overrides[key] = value

    if args._workload:  # re-exec'd inside rocprofv3
        run_workload(args.tokens, args.reps, args.warmup)
        return 0

    do_perf = args.perf or not args.correctness
    do_corr = args.correctness or not args.perf
    geometries = [8, 4] if args.warps == "both" else [int(args.warps)]

    patched = check_membar_patch()
    if patched is False:
        print(
            "WARNING: Triton lacks the TRITON_MEMBAR_DEDUP_BARE patch; that flag is "
            "ignored and perf will be slower than the recorded number.",
            file=sys.stderr,
        )
    elif patched is None:
        print(
            "NOTE: could not locate the Triton source to verify the "
            "TRITON_MEMBAR_DEDUP_BARE patch (set TRITON_SRC_DIR).",
            file=sys.stderr,
        )

    if any("TRITON_HIP_EXTERNAL_LLC" in v for g in geometries for v in PER_WARPS[g]):
        state = check_ds_agpr_llc()
        if state != "ok":
            print(
                f"WARNING: patched llc for -amdgpu-ds-read-agpr is {state} at "
                f"{PATCHED_LLC}; the 4-wave numbers below will not include it.",
                file=sys.stderr,
            )

    print(f"shape H={HIDDEN} I={INTER} E={N_EXPERTS} topk={TOPK} T={args.tokens}")
    if args.overrides:
        print("  overrides: " + " ".join(f"{k}={v}" for k, v in args.overrides.items()))
    ok = True
    with tempfile.TemporaryDirectory(prefix="gluon_gemm1_best_") as tmp:
        workdir = Path(tmp)
        for warps in geometries:
            if do_perf:
                for _ in range(args.repeat):
                    if run_perf(warps, args, workdir) is None:
                        ok = False
            if do_corr:
                ok &= run_correctness(warps, args, workdir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
