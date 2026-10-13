"""Compare two run_suite.sh outputs: totals and the per-test failure set.

    python compare_suite.py <suite_dir_before> <suite_dir_after>

Exit 0 only if the after-run has no failure/error the before-run did not have and
the passed/skipped counts are equal.
"""

import re
import sys
from pathlib import Path


def load(d):
    totals, bad = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}, set()
    for log in sorted([*Path(d).glob("*.log"), *Path(d).glob("*.txt")]):
        text = log.read_text(errors="replace")
        for line in text.splitlines():
            m = re.match(r"(FAILED|ERROR) (\S+)", line)
            if m:
                bad.add(m.group(2))
        last = text.strip().splitlines()[-1] if text.strip() else ""
        for n, k in re.findall(r"(\d+) (passed|failed|skipped|error)", last):
            totals[k] += int(n)
    return totals, bad


(ta, fa), (tb, fb) = load(sys.argv[1]), load(sys.argv[2])
print("before", ta)
print("after ", tb)
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
