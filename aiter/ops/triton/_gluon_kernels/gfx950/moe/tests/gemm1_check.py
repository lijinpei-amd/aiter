# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness half of ``scripts/gluon_moe_gemm1_check.py``, for tests/gate.sh.

Vendored copy that runs the probes in ``probes/`` instead of out-of-tree copies. The
perf half (``--perf``, which needs the out-of-tree ``cold_bench.py``) is not here;
``tests/cold_bench.sh`` is the performance check.

    python tests/gemm1_check.py --correctness --quick --gpu 1 --job $OUT/tmp

The strongest refactor gate is not perf at all -- it is the two determinism shas. A
pure refactor must reproduce them bit for bit; if a sha moves, the change altered
numerics and the perf numbers are meaningless until that is explained.
"""

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "scripts"))

PROBES = HERE / "probes"

TEST_FILE = "op_tests/triton_tests/moe/test_moe_gemm_a4w4.py"

#: Suite shape. 2026-09-03: 9 cases began SKIPPING ("the fill schedule needs both axes
#: split", BLOCK_M 16/32) when commit-group granularity moved into the tuning config and
#: the legacy non-EVEN fill path went away. They are skips, not failures -- the shapes
#: fall off the Gluon path and the test declines to run -- which is exactly the kind of
#: silent coverage loss a "473 passed" string match would have reported as a hard FAIL
#: while telling you nothing.
#:
#: 2026-09-03 (later): 482, up from 473, when
#: test_gemm1_gate_up_split_matches_interleaved added 9 cases -- the gate/up-split
#: layout against the interleaved one, byte for byte.
SUITE_TOTAL = 482
SUITE_SKIPPED = 9

#: Out-of-tree llc flags. See gluon_moe_gemm1_run_best.py for what each buys.
LLC_FLAGS = (
    "-amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd -amdgpu-no-sched-revert "
    "-misched-pin-critical-res=HWXDL"
)

#: Forced on top of BEST for every Gluon arm. BEST ships WARP_PIPELINE=0, which is
#: stale and worth ~14 us at 4 waves.
GLUON_OVERRIDES = {
    "AITER_TRITON_MOE_GLUON_WARP_PIPELINE": "1",
    "TRITON_HIP_EXTERNAL_LLC_FLAGS": LLC_FLAGS,
}

WARPS = 4

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


def read_regs(cache_dir):
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
    (
        "g4q  frozen act",
        "q",
        "1",
        None,
        2549,
        "vgpr=384 agpr=128 sgpr=95 spill=0",
        "0cfa77fb04450594",
        "7f2e1712d857",
    ),
    (
        "g4q  frozen no-act",
        "q",
        "1",
        "1",
        2321,
        "vgpr=384 agpr=128 sgpr=95 spill=0",
        "e7417a39061bafc5",
        "68d330b516bd",
    ),
    (
        "g4q  impl act",
        "q",
        "0",
        None,
        2757,
        "vgpr=388 agpr=132 sgpr=106 spill=0",
        "0cfa77fb04450594",
        "81ec5684b4f8",
    ),
    (
        "bf16 frozen act",
        "b",
        "1",
        None,
        2190,
        "vgpr=384 agpr=128 sgpr=100 spill=0",
        "59d442f663991659",
        "7c2fb1fd33d6",
    ),
    (
        "bf16 impl act",
        "b",
        "0",
        None,
        2386,
        "vgpr=388 agpr=132 sgpr=106 spill=0",
        "59d442f663991659",
        "957ca25a510a",
    ),
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
            cmd = [sys.executable, str(PROBES / "corr_probe_q.py"), "0"]
        else:
            cmd = [
                sys.executable,
                str(PROBES / "corr_probe.py"),
                "--mode",
                "determinism",
                "--reps",
                "0",
            ]
        proc = subprocess.run(
            cmd,
            env=gluon_env(cache, gpu, extra),
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        hits = glob.glob(str(cache / "**" / "_moe_gluon_gemm1.amdgcn"), recursive=True)
        if not hits:
            print(f"  {label:20s} COMPILE FAILED")
            print("   ", proc.stderr[-400:])
            ok = False
            continue
        lines = [
            ln.rstrip()
            for ln in Path(hits[0]).read_text().splitlines()
            if ln.startswith("\t") and not ln.lstrip().startswith(".")
        ]
        md5 = hashlib.md5("\n".join(lines).encode()).hexdigest()[:12]
        m = re.search(r"sha=([0-9a-f]{16})", proc.stdout)
        sha = m.group(1) if m else "?"
        regs = read_regs(cache)
        bad = [
            w
            for w, got, exp in (
                ("instrs", len(lines), n_ref),
                ("regs", regs, r_ref),
                ("sha", sha, sha_ref),
                ("md5", md5, md5_ref),
            )
            if exp is not None and got != exp
        ]
        if learn:
            print(f"  {label:20s} instrs={len(lines)} {regs} sha={sha} md5={md5}")
            continue
        print(
            f"  {label:20s} {'OK     ' if not bad else 'CHANGED'} "
            f"instrs={len(lines)} {regs} sha={sha} md5={md5}"
            + (f"   <- {', '.join(bad)} differ" if bad else "")
        )
        if bad:
            ok = False
    if not learn:
        print(f"\nbuilds: {'PASS' if ok else 'CHANGED'}")
    return ok


# ---------------------------------------------------------------------- correctness


def run_one(label, cmd, gpu, job, cache, expect, reps_note=""):
    """Run a correctness command under the tuned Gluon env; report PASS/FAIL."""
    proc = subprocess.run(
        cmd,
        env=gluon_env(job / cache, gpu),
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
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
                print(
                    f"    NOTE: {passed} passed / {skipped} skipped, recorded "
                    f"{SUITE_TOTAL - SUITE_SKIPPED}/{SUITE_SKIPPED}. A new skip means "
                    f"a shape left the Gluon path -- run pytest -rs for the reason."
                )
            return True

        ok &= run_one(
            f"suite ({SUITE_TOTAL} cases)",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                "-p",
                "no:cacheprovider",
                TEST_FILE,
            ],
            gpu,
            job,
            "chk_suite",
            suite_ok,
        )

    ok &= run_one(
        "33-expert ref probe",
        [sys.executable, str(PROBES / "ref_probe.py"), str(cases)],
        gpu,
        job,
        "chk_ref",
        lambda o: re.search(r"(\d+)/\1 passed", o) is not None,
    )

    def check_sha(kind):
        def f(out):
            got = sha_of(out, kind)
            if got != EXPECTED_SHA[kind]:
                print(
                    f"    SHA MISMATCH {kind}: got {got}, "
                    f"expected {EXPECTED_SHA[kind]} -- numerics changed"
                )
                return False
            return "-> PASS" in out

        return f

    ok &= run_one(
        "determinism bf16",
        [
            sys.executable,
            str(PROBES / "corr_probe.py"),
            "--mode",
            "determinism",
            "--reps",
            str(reps),
        ],
        gpu,
        job,
        "chk_det",
        check_sha("bf16"),
        f"   (sha must be {EXPECTED_SHA['bf16']})",
    )

    ok &= run_one(
        "determinism mxfp4-out",
        [sys.executable, str(PROBES / "corr_probe_q.py"), str(reps)],
        gpu,
        job,
        "chk_detq",
        check_sha("mxfp4"),
        f"   (sha must be {EXPECTED_SHA['mxfp4']})",
    )

    print(f"\ncorrectness: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--correctness",
        action="store_true",
        help="accepted for compatibility; correctness is all this runs",
    )
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument(
        "--job",
        type=Path,
        required=True,
        help="scratch dir for Triton caches (tests/gate.sh passes $OUT/tmp)",
    )
    ap.add_argument(
        "--learn-builds",
        action="store_true",
        help="print current build fingerprints instead of checking them",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="skip the 473-case suite and shorten the probes (smoke test)",
    )
    a = ap.parse_args()

    job = a.job
    job.mkdir(parents=True, exist_ok=True)

    print(f"gpu          {a.gpu}")
    print(
        "overrides    "
        + " ".join(
            f"{k.split('GLUON_')[-1].split('_LLC')[0]}={v}"
            for k, v in GLUON_OVERRIDES.items()
            if "LLC" not in k
        )
    )
    print(f"llc flags    {LLC_FLAGS}")
    print()

    ok = check_builds(a.gpu, job, learn=a.learn_builds)
    print()
    if not a.learn_builds:
        ok &= run_correctness(a.gpu, job, a.quick)
        print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
