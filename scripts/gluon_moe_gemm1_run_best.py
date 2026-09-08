# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the current best gfx950 Gluon MoE GEMM1 kernel (4 waves, ~625 us).

``gluon_moe_gemm1_best.py`` holds the tuned *env*, but its ``BEST`` predates two
flags that are worth ~42 us together, and it says nothing about the compiler. This
script pins the whole thing -- env, llc binary and llc flags -- in one place, and
prints the provenance it used so a number can always be traced back to a build.

    python scripts/gluon_moe_gemm1_run_best.py                 # perf + tests
    python scripts/gluon_moe_gemm1_run_best.py --perf --rounds 3
    python scripts/gluon_moe_gemm1_run_best.py --ab AITER_TRITON_MOE_GLUON_FROZEN_STEP

``--ab KEY`` interleaves KEY=0 against KEY=1 round by round and reports paired
deltas. Use it while refactoring ``_buffered._step_live``: ``FROZEN_STEP=1`` runs
``_pipeline_step_frozen``, the verbatim snapshot of the step this number was measured
with, so a refactor is compared against the known-good schedule rather than against a
remembered figure. Sequential sweeps have disagreed in sign with interleaved ones
repeatedly on this kernel, which is why --ab interleaves rather than running one arm
after the other.

Requires the patched llc (out-of-tree ``-amdgpu-ds-read-agpr`` /
``-amdgpu-mfma-tied-cd`` / ``-misched-pin-critical-res``) and the Triton
``TRITON_MEMBAR_DEDUP_BARE`` patch; both are checked and reported, not assumed.
"""

import argparse
import os
import re
import statistics as s
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEST_SCRIPT = REPO / "scripts" / "gluon_moe_gemm1_best.py"

#: On top of gluon_moe_gemm1_best.py's BEST. Measured, not cosmetic: BEST ships
#: WARP_PIPELINE=0, and the manual ping-pong is worth ~9 us at 4 waves.
#:
#: The old EPI_HOIST flag is gone -- its level 2 is now the only epilogue path, so it
#: needs no override. It was worth ~33 us: the per-mini-tile masked gammas load lowered
#: to predicated control flow with an `s_waitcnt vmcnt(0)`, and vmcnt is not selective,
#: so each one also drained the previous tile's stores. 13 of them carried 14.9% of the
#: kernel's stall. Staging bias/gammas in LDS before the K loop removed all of them.
OVERRIDES = {
    "AITER_TRITON_MOE_GLUON_WARP_PIPELINE": "1",
}

#: Four llc flags, all out-of-tree. ds-read-agpr + mfma-tied-cd ~18 us,
#: no-sched-revert + pin-critical-res ~9-13 us. The pin is only live under the default
#: GCN scheduler -- with -amdgpu-sched-strategy=coexec installed it is inert, which is
#: why an earlier measurement wrongly called it a no-op.
LLC_FLAGS = (
    "-amdgpu-ds-read-agpr "
    "-amdgpu-mfma-tied-cd "
    "-amdgpu-no-sched-revert "
    "-misched-pin-critical-res=HWXDL"
)

WARPS = 4


def provenance(llc):
    """Everything that decides which machine code comes out, printed once."""
    print(f"llc          {llc}")
    if llc.exists():
        mtime = datetime.fromtimestamp(llc.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"             built {mtime}")
        try:
            h = subprocess.run(
                [str(llc), "--help-hidden"], capture_output=True, text=True,
                timeout=120, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            h = ""
        missing = [f for f in ("amdgpu-ds-read-agpr", "amdgpu-mfma-tied-cd",
                               "misched-pin-critical-res", "amdgpu-no-sched-revert")
                   if f not in h]
        print("             out-of-tree flags: " +
              ("all present" if not missing else f"MISSING {missing}"))
        if missing:
            print("             -> the build will fail rather than degrade quietly",
                  file=sys.stderr)
    else:
        print("             MISSING -- build llvm-project first", file=sys.stderr)
    for name, path, needle in (
        ("triton membar", Path.home() / "development/triton/lib/Analysis/Membar.cpp",
         "TRITON_MEMBAR_DEDUP_BARE"),
    ):
        ok = path.exists() and needle in path.read_text()
        print(f"{name:12s} {'patched' if ok else 'NOT PATCHED (perf will be worse)'}")
    print(f"llc flags    {LLC_FLAGS}")
    print(f"overrides    " + " ".join(f"{k.split('GLUON_')[-1]}={v}"
                                      for k, v in OVERRIDES.items()))


def sets(extra=None):
    out = []
    for k, v in {**OVERRIDES, **(extra or {})}.items():
        out += ["--set", f"{k}={v}"]
    return out + ["--set", f"TRITON_HIP_EXTERNAL_LLC_FLAGS={LLC_FLAGS}"]


def run(args, extra=None, quiet=False):
    cmd = [sys.executable, str(BEST_SCRIPT), "--warps", str(WARPS)] + args + sets(extra)
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       timeout=7200, check=False)
    if not quiet:
        for line in p.stdout.splitlines():
            if "perf:" in line or "tests:" in line:
                print("  " + line.strip())
    return p


def medians(p):
    return [float(m) for m in re.findall(r"median=\s*([0-9.]+)", p.stdout)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=2, help="perf runs per invocation")
    ap.add_argument("--ab", metavar="ENVKEY",
                    help="interleave ENVKEY=0 against ENVKEY=1 and report paired deltas")
    a = ap.parse_args()

    llc = Path(os.environ.get(
        "TRITON_HIP_DS_AGPR_LLC",
        str(Path.home() / "development/llvm-project/build/bin/llc")))
    provenance(llc)
    print()

    gpu = ["--gpu", str(a.gpu)]
    if a.ab:
        arms = {f"{a.ab.split('GLUON_')[-1]}=0": {a.ab: "0"},
                f"{a.ab.split('GLUON_')[-1]}=1": {a.ab: "1"}}
        got = {k: [] for k in arms}
        for r in range(a.rounds):
            print(f"round {r + 1}")
            for name, ov in arms.items():
                p = run(["--perf", "--repeat", str(a.repeat)] + gpu, ov, quiet=True)
                m = medians(p)
                got[name] += m
                print(f"  {name:16s} " + " ".join(f"{x:7.1f}" for x in m))
        base = got[list(arms)[0]]
        print()
        for name, v in got.items():
            d = [x - y for x, y in zip(v, base)]
            print(f"  {name:16s} median {s.median(v):6.1f} mean {s.mean(v):6.1f} "
                  f"sd {s.pstdev(v):4.1f}  vs first {s.median(v) - s.median(base):+6.1f}"
                  f"  better {sum(1 for x in d if x < 0)}/{len(d)}")
        return 0

    do_perf = a.perf or not a.correctness
    do_corr = a.correctness or not a.perf
    if do_perf:
        for r in range(a.rounds):
            run(["--perf", "--repeat", str(a.repeat)] + gpu)
    if do_corr:
        run(["--correctness"] + gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
