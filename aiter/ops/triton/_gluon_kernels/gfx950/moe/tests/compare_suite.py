"""Compare two run_suite.sh outputs: totals and the per-test failure set.

    python compare_suite.py <before> <after>

Each side is a run_suite.sh label (`base` for $OUT/suite_base, $OUT as in _env.sh)
or, if it contains a `/` or names a directory, a suite directory: e.g.
`compare_suite.py base new`, or from the moe folder
`compare_suite.py tests/reference/suite_1d3431e50 new`.

Exit 0 only if the after-run has no failure/error the before-run did not have and
the passed/skipped counts are equal. A side that does not exist, holds no *.log or
*.txt, or has one that does not end in a pytest summary line fails the comparison.
"""

import os
import re
import sys
from pathlib import Path

OUT = os.environ.get(
    "OUT", Path(__file__).resolve().parents[7] / "bench_out/gluon_moe_regression"
)
KINDS = ("passed", "failed", "skipped", "error")


def load(arg):
    d = Path(arg) if "/" in arg or Path(arg).is_dir() else Path(OUT) / f"suite_{arg}"
    logs = sorted([*d.glob("*.log"), *d.glob("*.txt")])
    if not logs:
        sys.exit(f"SUITE FAIL: no *.log or *.txt in {d}")
    totals, bad, unfinished = dict.fromkeys(KINDS, 0), set(), []
    for log in logs:
        text = log.read_text(errors="replace")
        for line in text.splitlines():
            m = re.match(r"(FAILED|ERROR) (\S+)", line)
            if m:
                bad.add(m.group(2))
        last = text.strip().splitlines()[-1] if text.strip() else ""
        found = re.findall(r"(\d+) (passed|failed|skipped|error)", last)
        if not found:
            unfinished.append(log.name)
        for n, k in found:
            totals[k] += int(n)
    if unfinished:
        sys.exit(f"SUITE FAIL: no pytest summary ends {d}/{', '.join(unfinished)}")
    return d, totals, bad


if len(sys.argv) != 3:
    sys.exit(__doc__)
(da, ta, fa), (db, tb, fb) = load(sys.argv[1]), load(sys.argv[2])
print("before", ta, da)
print("after ", tb, db)
new, fixed = sorted(fb - fa), sorted(fa - fb)
for t in new:
    print("NEW FAILURE", t)
for t in fixed:
    print("now passing", t)
ok = (
    not new
    and ta["passed"] == tb["passed"] - len(fixed)
    and ta["skipped"] == tb["skipped"]
)
print("SUITE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
