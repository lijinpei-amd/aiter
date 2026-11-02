"""Compare two run_suite.sh outputs: totals, the per-test failure set and the tests.

    python compare_suite.py <before> <after>

Each side is a run_suite.sh label (`base` for $OUT/suite_base, $OUT as in _env.sh)
or, if it contains a `/` or names a directory, a suite directory: e.g.
`compare_suite.py base new`, or from the moe folder
`compare_suite.py tests/reference/suite_1d3431e50 new`.

Exit 0 only if the after-run has no failure/error the before-run did not have and
the passed/skipped counts match. A side that does not exist, holds no *.log or
*.txt, or has one that does not end in a pytest summary line fails the comparison.

When both sides have units.json with node ids, the test inventories are compared
too: a before-test missing after fails, and so does a broken after-unit. Tests added
after the before-run count only with their outcome from the after-run's
outcomes.json: an added pass or skip raises the expected count, an added failure
is a new failure. Without outcomes.json, added tests fail the comparison. When both
sides have outcomes.json, a before-pass that is now skipped fails as well.
"""

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

OUT = os.environ.get(
    "OUT", Path(__file__).resolve().parents[7] / "bench_out/gluon_moe_regression"
)
KINDS = ("passed", "failed", "skipped", "error")


@dataclass
class Run:
    path: Path
    totals: dict
    bad: set  # failed or errored node ids
    tests: set | None  # node ids from units.json; None if unknown
    outcomes: dict | None  # node id -> outcome from outcomes.json; None if absent
    broken: list  # units.json units that did not run cleanly


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
    tests, broken = None, []
    if (d / "units.json").is_file():
        units = json.loads((d / "units.json").read_text())["units"]
        if all(u.get("ids") is not None for u in units):
            tests = {nid for u in units for nid in u["ids"]}
        broken = [u["name"] for u in units if u.get("reason")]
    outcomes = None
    if (d / "outcomes.json").is_file():
        outcomes = json.loads((d / "outcomes.json").read_text())
    return Run(d, totals, bad, tests, outcomes, broken)


def compare(a, b):
    """(ok, report lines) for before-run a and after-run b."""
    lines = [f"before {a.totals} {a.path}", f"after  {b.totals} {b.path}"]
    new, fixed = sorted(b.bad - a.bad), sorted(a.bad - b.bad)
    lines += [f"NEW FAILURE {t}" for t in new]
    lines += [f"now passing {t}" for t in fixed]
    problems = [f"BROKEN UNIT {name}" for name in b.broken]
    passed, skipped = a.totals["passed"] + len(fixed), a.totals["skipped"]
    if b.outcomes is not None:
        tally = Counter(b.outcomes.values())
        if any(tally[k] != b.totals[k] for k in KINDS):
            problems.append(f"outcomes.json {dict(tally)} disagrees with the logs")
    if a.tests is not None and b.tests is not None:
        problems += [f"MISSING TEST {t}" for t in sorted(a.tests - b.tests)]
        added = b.tests - a.tests
        if added and b.outcomes is None:
            problems.append(f"{len(added)} added tests and no outcomes.json")
        elif added:
            unknown = sorted(t for t in added if t not in b.outcomes)
            problems += [f"NO OUTCOME {t}" for t in unknown]
            got = Counter(b.outcomes[t] for t in added if t in b.outcomes)
            passed, skipped = passed + got["passed"], skipped + got["skipped"]
            lines.append(f"added {len(added)} tests: {dict(got)}")
        if a.outcomes is not None and b.outcomes is not None:
            problems += [
                f"NOW {b.outcomes.get(t, 'missing').upper()} {t}"
                for t in sorted(a.tests & b.tests)
                if a.outcomes.get(t) == "passed" and b.outcomes.get(t) != "passed"
            ]
    lines += problems
    ok = (
        not new
        and not problems
        and b.totals["passed"] == passed
        and b.totals["skipped"] == skipped
    )
    if not ok and not new and not problems:
        lines.append(f"expected {passed} passed and {skipped} skipped after")
    return ok, lines


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    ok, lines = compare(load(argv[0]), load(argv[1]))
    print("\n".join(lines))
    print("SUITE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
