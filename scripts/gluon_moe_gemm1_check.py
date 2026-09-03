# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Re-qualify the gfx950 Gluon MoE GEMM1 kernel after a refactor: perf + correctness.

One command, fixed protocol, fixed baseline. Run it before a refactor to confirm the
baseline still reproduces on today's machine, and after each round to see what moved.

    python scripts/gluon_moe_gemm1_check.py                    # both, 5 rounds
    python scripts/gluon_moe_gemm1_check.py --perf --rounds 3
    python scripts/gluon_moe_gemm1_check.py --correctness

Three things this script exists to prevent, all of which have already cost a wrong
conclusion on this kernel:

1. **The WARP_PIPELINE trap.** ``cold_bench.py`` builds its env from
   ``gluon_moe_gemm1_best.BEST``, which is stale at ``WARP_PIPELINE=0``. That is worth
   ~14 us at 4 waves, and forgetting it once produced a bogus "+24 us behind FlyDSL,
   parity lost" reading. Every Gluon arm here is forced to ``WARP_PIPELINE=1`` plus the
   four out-of-tree llc flags, and the script prints what it used.

2. **Cross-harness comparison.** Saturating, cold-flush and warm-no-flush disagree by
   tens of us on the same machine code. This script only ever uses the cold harness at
   warmup 40 / reps 100, median of the last 100 dispatches, and the recorded baseline
   was taken exactly that way.

3. **Machine drift.** Absolute numbers have moved ~7 us between sessions with no code
   change. FlyDSL is therefore run in the same interleave as an in-session control, and
   the headline numbers are the *paired* Gluon-minus-FlyDSL deltas, which are drift
   robust. Absolute medians are reported too, but the deltas are the verdict.

The strongest refactor gate is not perf at all -- it is the two determinism shas. A
pure refactor must reproduce them bit for bit; if a sha moves, the change altered
numerics and the perf numbers are meaningless until that is explained.
"""

import argparse
import csv
import glob
import os
import re
import statistics as s
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

MOE_RERUN = Path("/raid/jinpli/workspace/home01/jinpli/moe_rerun")
COLD_BENCH = MOE_RERUN / "cold_bench.py"
TEST_FILE = "op_tests/triton_tests/moe/test_moe_gemm_a4w4.py"

#: Suite shape. 2026-09-03: 9 cases began SKIPPING ("the fill schedule needs both axes
#: split", BLOCK_M 16/32) when commit-group granularity moved into the tuning config and
#: the legacy non-EVEN fill path went away. They are skips, not failures -- the shapes
#: fall off the Gluon path and the test declines to run -- which is exactly the kind of
#: silent coverage loss a "473 passed" string match would have reported as a hard FAIL
#: while telling you nothing.
SUITE_TOTAL = 473
SUITE_SKIPPED = 9

#: Out-of-tree llc flags. See gluon_moe_gemm1_run_best.py for what each buys.
LLC_FLAGS = ("-amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd -amdgpu-no-sched-revert "
             "-misched-pin-critical-res=HWXDL")

#: Forced on top of BEST for every Gluon arm. BEST ships WARP_PIPELINE=0; see (1) above.
GLUON_OVERRIDES = {
    "AITER_TRITON_MOE_GLUON_WARP_PIPELINE": "1",
    "TRITON_HIP_EXTERNAL_LLC_FLAGS": LLC_FLAGS,
}

WARPS = 4

#: Perf arms, interleaved in this order every round. `flydsl no-act` drops only the
#: transcendentals (exp2 + rcp) and keeps the reduction shape and store byte count, so
#: no MFMA and no store is DCE'd -- the delta is the swiglu and nothing else.
#:
#: **The Gluon contract is the FROZEN step, not impl.** ``_pipeline_step_frozen`` holds
#: the best measured schedule, and it is what must not regress while
#: ``_pipeline_step_impl`` is being refactored. Frozen is NOT insulated from that work:
#: it shares the epilogue, ``_slot_a_read``/``_slot_b_read``, ``_maybe_block_dot``, the
#: LDS helpers, the tuning config and the whole kernel body, so any change to those
#: shows up here. Only the step itself is pinned.
#:
#: ``impl act`` rides along as a progress arm: during the refactor it says how far the
#: live path is from the contract. It is not a regression gate -- it is expected to move.
ARMS = [
    ("frozen act",    "g4q", {"AITER_TRITON_MOE_GLUON_FROZEN_STEP": "1"}),
    ("frozen no-act", "g4q", {"AITER_TRITON_MOE_GLUON_FROZEN_STEP": "1",
                              "AITER_TRITON_MOE_GLUON_NO_EPI": "1"}),
    ("impl act",      "g4q", {}),
    ("flydsl act",    "fly", {}),
    ("flydsl no-act", "fly", {"AITER_FLYDSL_NO_ACT": "1"}),
]
KERNEL_PAT = {"g8": "_moe_gluon_gemm1", "g4": "_moe_gluon_gemm1", "g4q": "_moe_gluon_gemm1",
              # g4x varies apply_swiglu; without the gated activation the launcher
              # emits _moe_gluon_gemm2 (ARN=1) from the same source, so match both.
              "g4x": "_moe_gluon_gemm",
              "fly": "gemm1_a4w4_port"}

#: Recorded 2026-09-02. **Pooled over four independent 5-round runs** (20 paired
#: samples), not one, because this comparison was measured four times that day on
#: byte-identical Gluon machine code and the per-run estimates of
#: "frozen - flydsl, act" came out +6.6, +8.8, +13.5 and +11.2. A gate anchored to any
#: single one of those would false-alarm on the next. Absolute figures are the median of
#: the per-run medians.
#:
#: The noise is asymmetric and it is worth knowing which way: across those runs the
#: `flydsl act` median spans 2.6 us while `frozen act` spans 8.5 us and its worst round
#: pair spans 21 us. The Gluon kernel is markedly less reproducible cold than FlyDSL,
#: so a Gluon-side "improvement" of <5 us in one run is not evidence of anything.
BASELINE_ABS = {
    "frozen act": 634.0,
    "frozen no-act": 620.6,
    "impl act": 685.6,
    "flydsl act": 623.2,
    "flydsl no-act": 624.3,
}
#: (label, minuend, subtrahend, pooled delta, envelope of the per-run 95% CIs).
#: The envelope -- not a single run's CI -- is what the overlap test compares against,
#: so the gate tolerates the observed between-run spread and fires only on a real move.
#: The first two are the contract. "impl - frozen" is progress, not a gate: it is
#: expected to shrink toward 0 as the refactor restores the manual schedule.
BASELINE_PAIRED = [
    ("activation cost, frozen",  "frozen act",    "frozen no-act",  +12.5, (+3.9, +22.4)),
    ("activation cost, flydsl",  "flydsl act",    "flydsl no-act",   -1.2, (-8.3, +8.1)),
    ("frozen - flydsl, act",     "frozen act",    "flydsl act",     +10.0, (-0.3, +18.0)),
    ("frozen - flydsl, no-act",  "frozen no-act", "flydsl no-act",   -3.6, (-13.6, +8.1)),
    ("impl - frozen, act",       "impl act",      "frozen act",     +52.1, (+43.4, +62.8)),
]

#: Bit-exactness gates. The bf16 sha has been stable across every variant of this
#: campaign, including the whole epilogue rewrite; the mxfp4 one since 2026-09-02.
EXPECTED_SHA = {
    "bf16": "59d442f663991659",
    "mxfp4": "0cfa77fb04450594",
}

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


# ------------------------------------------------------------------------------ env


def gluon_env(cache_dir, gpu, extra=None):
    """BEST + the forced overrides + whatever the arm adds."""
    import gluon_moe_gemm1_best as B

    env = B.build_env(WARPS, cache_dir, gpu, {**GLUON_OVERRIDES, **(extra or {})})
    return env


def cold_bench_env(target, gpu, extra):
    """cold_bench takes Gluon overrides through the newline-separated OV variable."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OV", "AITER_FLYDSL_NO_ACT")}
    if target == "fly":
        env.update(extra)
    else:
        env["OV"] = "\n".join(f"{k}={v}"
                              for k, v in {**GLUON_OVERRIDES, **extra}.items())
    return env


# ----------------------------------------------------------------------------- perf


def median_of_trace(target, tag, job):
    """Median of the last 100 dispatch durations, in us, from the kernel trace."""
    pat = KERNEL_PAT[target]
    for f in glob.glob(str(job / f"cold_{target}{tag}" / "**" / "*kernel_trace.csv"),
                       recursive=True):
        rows = [r for r in csv.DictReader(open(f)) if pat in r["Kernel_Name"]]
        if not rows:
            continue
        st = next(k for k in rows[0] if "Start" in k)
        en = next(k for k in rows[0] if "End" in k)
        rows.sort(key=lambda r: int(r[st]))
        return s.median(sorted((int(r[en]) - int(r[st])) / 1000 for r in rows[-100:]))
    return None


def run_perf(rounds, gpu, job):
    got = {name: [] for name, _t, _e in ARMS}
    for rnd in range(rounds):
        print(f"round {rnd + 1}", flush=True)
        for i, (name, target, extra) in enumerate(ARMS):
            # One retry. A transient dispatch failure used to drop the arm, leaving the
            # arms with unequal sample counts, which makes every paired delta that
            # touches it report "insufficient data" -- one blip wasted the whole run.
            med = None
            for attempt in range(2):
                tag = f"_chk{i}r{rnd}" + ("b" if attempt else "")
                proc = subprocess.run(
                    [sys.executable, str(COLD_BENCH), "--target", target,
                     "--warmup", "40", "--reps", "100", "--gpu", str(gpu), "--tag", tag],
                    env=cold_bench_env(target, gpu, extra),
                    capture_output=True, text=True, timeout=5400, check=False)
                if proc.returncode == 0:
                    med = median_of_trace(target, tag, job)
                if med is not None:
                    break
                print(f"  {name:14s} attempt {attempt + 1} failed"
                      f"{'' if attempt else ', retrying'}", flush=True)
                if attempt:
                    print(f"    {proc.stdout[-400:]}\n    {proc.stderr[-400:]}")
            if med is None:
                continue
            got[name].append(med)
            print(f"  {name:14s} {med:7.1f}", flush=True)

    print()
    print("absolute medians (informational -- the machine drifts ~7 us between sessions)")
    for name, _t, _e in ARMS:
        v = got[name]
        if not v:
            continue
        base = BASELINE_ABS[name]
        print(f"  {name:14s} median {s.median(v):6.1f}  mean {s.mean(v):6.1f}  "
              f"sd {s.pstdev(v):4.1f}   baseline {base:6.1f}  "
              f"({s.median(v) - base:+5.1f})   {[round(x, 1) for x in v]}")

    print()
    print("paired deltas -- these are the verdict")
    verdict = PASS
    for label, a, b, base_d, base_ci in BASELINE_PAIRED:
        va, vb = got[a], got[b]
        if len(va) != len(vb) or not va:
            print(f"  {label:26s} insufficient data")
            verdict = WARN
            continue
        d = [x - y for x, y in zip(va, vb)]
        m = s.mean(d)
        # One round gives a point estimate with no interval; treat it as maximally
        # uncertain rather than pretending the CI is zero-width.
        ci = 1.96 * s.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("inf")
        moved = m - base_d
        # Flag only when the new interval clears the recorded one -- this kernel's
        # deltas have a ~4 us half-width, so anything inside that is not a signal.
        overlap = (m - ci) <= base_ci[1] and (m + ci) >= base_ci[0]
        mark = "  " if overlap else ("!!" if abs(moved) > 0 else "  ")
        if not overlap:
            verdict = WARN
        print(f"  {mark}{label:26s} {m:+6.1f}  95% CI [{m - ci:+6.1f}, {m + ci:+6.1f}]"
              f"   baseline {base_d:+5.1f} [{base_ci[0]:+.1f}, {base_ci[1]:+.1f}]"
              f"   moved {moved:+5.1f}")
    print(f"\nperf: {verdict}"
          + ("" if verdict == PASS
             else "  (a '!!' line means the new CI does not overlap the recorded one)"))

    # Arm 0's cache, i.e. the FROZEN g4q build -- not the bf16 one run_best reports
    # (388/132/89), which has a different epilogue and different pressure. Spills must
    # stay 0: this kernel goes from ~630 us to ~4000 us the moment it spills.
    regs = read_regs(job / "coldcache_g4q_chk0r0")
    if regs:
        print(f"registers (frozen, mxfp4 build): {regs}"
              f"   (baseline vgpr=384 agpr=128 sgpr=95 spill=0)")
    return verdict != FAIL


def read_regs(cache_dir):
    hits = glob.glob(str(Path(cache_dir) / "**" / "_moe_gluon_gemm1.amdgcn"),
                     recursive=True)
    if not hits:
        return ""
    text = Path(hits[0]).read_text()

    def field(name):
        m = re.search(rf"\.{name}:\s*(\d+)", text)
        return m.group(1) if m else "?"

    return (f"vgpr={field('vgpr_count')} agpr={field('agpr_count')} "
            f"sgpr={field('sgpr_count')} spill={field('vgpr_spill_count')}")


# ------------------------------------------------------------------- build identity

#: Compiled-build fingerprints, checked at compile time. This catches a codegen change
#: instantly and unambiguously, instead of leaving it to be argued about afterwards from
#: cross-run medians -- which is exactly what happened when the no-act binary changed at
#: the WaitCommitScheme refactor (2340 -> 2321 instrs) and looked like a +5 us
#: regression. A paired A/B of the two binaries then measured -4.0 us, CI [-8.5, +0.6]:
#: the new build is if anything faster, and the apparent regression was cross-run drift.
#:
#: The no-act build is listed BECAUSE it was the one previously missing. The act build
#: was verified byte-identical every round while the no-act build silently changed, so
#: "the frozen contract is untouched" was being asserted on incomplete evidence.
#:
#: A CHANGED line is not automatically a bug -- but it must be explained, not absorbed.
#: (label, probe, FROZEN_STEP, NO_EPI, instrs, regs, out-sha, asm-md5)
BUILD_REFS = [
    ("g4q  frozen act",    "q", "1", None, 2549, "vgpr=384 agpr=128 sgpr=95 spill=0",
     "0cfa77fb04450594", "7f2e1712d857"),
    ("g4q  frozen no-act", "q", "1", "1",  2321, "vgpr=384 agpr=128 sgpr=95 spill=0",
     "e7417a39061bafc5", "68d330b516bd"),
    ("g4q  impl act",      "q", "0", None, 2757, "vgpr=388 agpr=132 sgpr=106 spill=0",
     "0cfa77fb04450594", "81ec5684b4f8"),
    ("bf16 frozen act",    "b", "1", None, 2190, "vgpr=384 agpr=128 sgpr=100 spill=0",
     "59d442f663991659", "7c2fb1fd33d6"),
    ("bf16 impl act",      "b", "0", None, 2386, "vgpr=388 agpr=132 sgpr=106 spill=0",
     "59d442f663991659", "957ca25a510a"),
]


def check_builds(gpu, job, learn=False):
    """Compile every build we benchmark and compare against the recorded fingerprint."""
    import hashlib
    import shutil

    ok = True
    print("build identity")
    for label, kind, fz, no_epi, n_ref, r_ref, sha_ref, md5_ref in BUILD_REFS:
        cache = job / ("bld_" + label.replace(" ", "_"))
        shutil.rmtree(cache, ignore_errors=True)
        extra = {"AITER_TRITON_MOE_GLUON_FROZEN_STEP": fz}
        if no_epi:
            extra["AITER_TRITON_MOE_GLUON_NO_EPI"] = no_epi
        if kind == "q":
            cmd = [sys.executable, str(MOE_RERUN / "corr_probe_q.py"), "0"]
        else:
            cmd = [sys.executable, str(MOE_RERUN / "corr_probe.py"),
                   "--mode", "determinism", "--reps", "0"]
        proc = subprocess.run(cmd, env=gluon_env(cache, gpu, extra), cwd=str(REPO),
                              capture_output=True, text=True, timeout=3600, check=False)
        hits = glob.glob(str(cache / "**" / "_moe_gluon_gemm1.amdgcn"), recursive=True)
        if not hits:
            print(f"  {label:20s} COMPILE FAILED")
            print("   ", proc.stderr[-400:])
            ok = False
            continue
        lines = [ln.rstrip() for ln in Path(hits[0]).read_text().splitlines()
                 if ln.startswith("\t") and not ln.lstrip().startswith(".")]
        md5 = hashlib.md5("\n".join(lines).encode()).hexdigest()[:12]
        m = re.search(r"sha=([0-9a-f]{16})", proc.stdout)
        sha = m.group(1) if m else "?"
        regs = read_regs(cache)
        bad = [w for w, got, exp in (("instrs", len(lines), n_ref),
                                     ("regs", regs, r_ref),
                                     ("sha", sha, sha_ref),
                                     ("md5", md5, md5_ref))
               if exp is not None and got != exp]
        if learn:
            print(f"  {label:20s} instrs={len(lines)} {regs} sha={sha} md5={md5}")
            continue
        print(f"  {label:20s} {'OK     ' if not bad else 'CHANGED'} "
              f"instrs={len(lines)} {regs} sha={sha} md5={md5}"
              + (f"   <- {', '.join(bad)} differ" if bad else ""))
        if bad:
            ok = False
    if not learn:
        print(f"\nbuilds: {'PASS' if ok else 'CHANGED'}")
    return ok


# ---------------------------------------------------------------------- correctness


def run_one(label, cmd, gpu, job, cache, expect, reps_note=""):
    """Run a correctness command under the tuned Gluon env; report PASS/FAIL."""
    proc = subprocess.run(cmd, env=gluon_env(job / cache, gpu), cwd=str(REPO),
                          capture_output=True, text=True, timeout=7200, check=False)
    out = proc.stdout
    ok = expect(out) and proc.returncode == 0
    tail = [ln for ln in out.strip().splitlines() if ln.strip()]
    summary = tail[-1] if tail else "no output"
    print(f"  {label:22s} {'PASS' if ok else 'FAIL'}  {summary}{reps_note}")
    if not ok:
        print("    --- stdout tail ---")
        for ln in tail[-15:]:
            print("    " + ln)
        if proc.stderr.strip():
            print("    --- stderr tail ---")
            for ln in proc.stderr.strip().splitlines()[-10:]:
                print("    " + ln)
    return ok


def sha_of(out, key):
    m = re.search(r"sha=([0-9a-f]{16})", out)
    return m.group(1) if m else None


def run_correctness(gpu, job, quick=False):
    reps = 5 if quick else 30
    cases = 1 if quick else 3
    ok = True

    print("correctness")
    if not quick:
        # Pass on "no failures", not on a literal count, but flag any change in the
        # skip count: a shape dropping off the Gluon path shows up as a SKIP, not a
        # failure, so a hard-coded "473 passed" both false-alarms and hides that.
        def suite_ok(out):
            m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", out)
            if not m or "failed" in out or "error" in out.lower():
                return False
            passed, skipped = int(m.group(1)), int(m.group(2) or 0)
            if passed + skipped != SUITE_TOTAL or skipped != SUITE_SKIPPED:
                print(f"    NOTE: {passed} passed / {skipped} skipped, recorded "
                      f"{SUITE_TOTAL - SUITE_SKIPPED}/{SUITE_SKIPPED}. A new skip means "
                      f"a shape left the Gluon path -- run pytest -rs for the reason.")
            return True

        ok &= run_one(
            f"suite ({SUITE_TOTAL} cases)",
            [sys.executable, "-m", "pytest", "-q", "--color=no",
             "-p", "no:cacheprovider", TEST_FILE],
            gpu, job, "chk_suite", suite_ok)

    ok &= run_one(
        "33-expert ref probe",
        [sys.executable, str(MOE_RERUN / "ref_probe.py"), str(cases)],
        gpu, job, "chk_ref",
        lambda o: re.search(r"(\d+)/\1 passed", o) is not None)

    def check_sha(kind):
        def f(out):
            got = sha_of(out, kind)
            if got != EXPECTED_SHA[kind]:
                print(f"    SHA MISMATCH {kind}: got {got}, "
                      f"expected {EXPECTED_SHA[kind]} -- numerics changed")
                return False
            return "-> PASS" in out
        return f

    ok &= run_one(
        "determinism bf16",
        [sys.executable, str(MOE_RERUN / "corr_probe.py"),
         "--mode", "determinism", "--reps", str(reps)],
        gpu, job, "chk_det", check_sha("bf16"),
        f"   (sha must be {EXPECTED_SHA['bf16']})")

    ok &= run_one(
        "determinism mxfp4-out",
        [sys.executable, str(MOE_RERUN / "corr_probe_q.py"), str(reps)],
        gpu, job, "chk_detq", check_sha("mxfp4"),
        f"   (sha must be {EXPECTED_SHA['mxfp4']})")

    print(f"\ncorrectness: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--learn-builds", action="store_true",
                    help="print current build fingerprints instead of checking them")
    ap.add_argument("--quick", action="store_true",
                    help="skip the 473-case suite and shorten the probes (smoke test)")
    a = ap.parse_args()

    if "CLAUDE_JOB_DIR" not in os.environ:
        print("CLAUDE_JOB_DIR must be set -- cold_bench.py writes its traces there",
              file=sys.stderr)
        return 2
    job = Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp"
    job.mkdir(parents=True, exist_ok=True)
    if not COLD_BENCH.exists():
        print(f"missing {COLD_BENCH}", file=sys.stderr)
        return 2

    print(f"gpu          {a.gpu}")
    print(f"overrides    " + " ".join(f"{k.split('GLUON_')[-1].split('_LLC')[0]}={v}"
                                      for k, v in GLUON_OVERRIDES.items()
                                      if "LLC" not in k))
    print(f"llc flags    {LLC_FLAGS}")
    print(f"protocol     cold, 768 MB flush, warmup 40, reps 100, median of last 100")
    print()

    do_perf = a.perf or not a.correctness
    do_corr = a.correctness or not a.perf
    ok = True
    if do_corr:
        ok &= check_builds(a.gpu, job, learn=a.learn_builds)
        print()
        if not a.learn_builds:
            ok &= run_correctness(a.gpu, job, a.quick)
            print()
    if do_perf:
        ok &= run_perf(1 if a.quick else a.rounds, a.gpu, job)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
