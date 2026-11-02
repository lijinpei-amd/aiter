# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the regression suite comparison (tests/compare_suite.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "aiter/ops/triton/_gluon_kernels/gfx950/moe/tests/compare_suite.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_suite", _SCRIPT)
compare_suite = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_suite)

_FILE = "op_tests/triton_tests/moe/test_x.py"
_BASE = {f"{_FILE}::test_{i}": "passed" for i in range(4)} | {
    f"{_FILE}::test_old_fail": "failed",
    f"{_FILE}::test_skip": "skipped",
}


def _run(root, name, outcomes, manifest=True, reason="", summary=True):
    """A run_suite.py output directory with one unit holding ``outcomes``."""
    d = root / name
    d.mkdir()
    counts = {k: list(outcomes.values()).count(k) for k in compare_suite.KINDS}
    log = [f"FAILED {t} - assert" for t, k in outcomes.items() if k == "failed"]
    log += [f"ERROR {t} - boom" for t, k in outcomes.items() if k == "error"]
    if summary:
        log.append(", ".join(f"{n} {k}" for k, n in counts.items() if n) + " in 1.0s")
    (d / "test_x.log").write_text("\n".join(log) + "\n")
    unit = {"name": "test_x", "ids": sorted(outcomes), "reason": reason}
    (d / "units.json").write_text(json.dumps({"units": [unit]}))
    if manifest:
        (d / "outcomes.json").write_text(json.dumps(outcomes))
    return compare_suite.load(str(d))


def _check(tmp_path, after, manifest=True, before=_BASE, before_manifest=False):
    a = _run(tmp_path, "before", before, manifest=before_manifest)
    ok, lines = compare_suite.compare(a, _run(tmp_path, "after", after, manifest))
    return ok, "\n".join(lines)


def test_same_run_passes(tmp_path):
    assert _check(tmp_path, dict(_BASE))[0]


def test_added_passing_and_skipped_tests_pass(tmp_path):
    after = _BASE | {f"{_FILE}::test_new": "passed", f"{_FILE}::test_s": "skipped"}
    ok, report = _check(tmp_path, after)
    assert ok, report


def test_added_tests_without_outcomes_fail(tmp_path):
    ok, report = _check(tmp_path, _BASE | {f"{_FILE}::test_new": "passed"}, False)
    assert not ok and "no outcomes.json" in report


def test_added_failing_test_fails(tmp_path):
    ok, report = _check(tmp_path, _BASE | {f"{_FILE}::test_new": "failed"})
    assert not ok and f"NEW FAILURE {_FILE}::test_new" in report


def test_old_failure_now_passing_passes(tmp_path):
    ok, report = _check(tmp_path, _BASE | {f"{_FILE}::test_old_fail": "passed"})
    assert ok and "now passing" in report


def test_missing_old_test_fails_even_when_counts_match(tmp_path):
    after = {t: k for t, k in _BASE.items() if t != f"{_FILE}::test_0"}
    after[f"{_FILE}::test_new"] = "passed"
    ok, report = _check(tmp_path, after)
    assert not ok and f"MISSING TEST {_FILE}::test_0" in report


@pytest.mark.parametrize("before_manifest", [False, True])
def test_old_pass_now_skipped_fails(tmp_path, before_manifest):
    after = _BASE | {f"{_FILE}::test_1": "skipped"}
    ok, report = _check(tmp_path, after, before_manifest=before_manifest)
    assert not ok
    # Named when both runs have outcomes; otherwise the pass count catches it.
    assert (f"NOW SKIPPED {_FILE}::test_1" in report) == before_manifest


def test_broken_unit_fails(tmp_path):
    a = _run(tmp_path, "before", _BASE, manifest=False)
    b = _run(tmp_path, "after", dict(_BASE), reason="timed out after 1800s")
    ok, report = compare_suite.compare(a, b)
    assert not ok and "BROKEN UNIT test_x" in "\n".join(report)


def test_unfinished_log_fails(tmp_path):
    with pytest.raises(SystemExit, match="no pytest summary"):
        _run(tmp_path, "after", dict(_BASE), summary=False)


def test_outcomes_that_disagree_with_the_logs_fail(tmp_path):
    a = _run(tmp_path, "before", _BASE, manifest=False)
    b = _run(tmp_path, "after", _BASE | {f"{_FILE}::test_new": "passed"})
    (b.path / "outcomes.json").write_text(json.dumps(_BASE))
    ok, report = compare_suite.compare(a, compare_suite.load(str(b.path)))
    assert not ok and "disagrees with the logs" in "\n".join(report)


def test_legacy_runs_without_units_json_compare_counts_only(tmp_path):
    a = _run(tmp_path, "before", _BASE, manifest=False)
    b = _run(tmp_path, "after", _BASE | {f"{_FILE}::test_new": "passed"})
    (a.path / "units.json").unlink()
    ok, report = compare_suite.compare(compare_suite.load(str(a.path)), b)
    assert not ok and "expected 4 passed" in "\n".join(report)
