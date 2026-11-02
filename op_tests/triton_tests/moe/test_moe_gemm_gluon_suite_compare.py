# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the regression suite comparison (tests/compare_suite.py)."""

import contextlib
import importlib.util
import json
from pathlib import Path

import pytest

_TESTS = (
    Path(__file__).resolve().parents[3]
    / "aiter/ops/triton/_gluon_kernels/gfx950/moe/tests"
)


def _load(name):
    """A script of the suite tooling, loaded by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_suite = _load("compare_suite")
run_suite = _load("run_suite")

_FILE = "op_tests/triton_tests/moe/test_x.py"
_BASE = {f"{_FILE}::test_{i}": "passed" for i in range(4)} | {
    f"{_FILE}::test_old_fail": "failed",
    f"{_FILE}::test_skip": "skipped",
}


def _run(root, name, outcomes, manifest=True, reason="", summary=True, ids=None):
    """A run_suite.py output directory with one unit holding ``outcomes``.

    ``ids`` overrides the unit's node ids in units.json (``"none"`` writes null), and
    ``manifest`` may be a dict written as outcomes.json instead of ``outcomes``."""
    d = root / name
    d.mkdir()
    counts = {k: list(outcomes.values()).count(k) for k in compare_suite.KINDS}
    log = [f"FAILED {t} - assert" for t, k in outcomes.items() if k == "failed"]
    log += [f"ERROR {t} - boom" for t, k in outcomes.items() if k == "error"]
    if summary:
        log.append(", ".join(f"{n} {k}" for k, n in counts.items() if n) + " in 1.0s")
    (d / "test_x.log").write_text("\n".join(log) + "\n")
    unit_ids = None if ids == "none" else sorted(outcomes) if ids is None else ids
    unit = {"name": "test_x", "ids": unit_ids, "reason": reason}
    (d / "units.json").write_text(json.dumps({"units": [unit]}))
    if manifest:
        written = manifest if isinstance(manifest, dict) else outcomes
        (d / "outcomes.json").write_text(json.dumps(written))
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


def test_unit_without_node_ids_fails(tmp_path):
    # A whole-file unit without canonical ids must not switch off the inventory check:
    # here it hides that test_old is gone.
    a = _run(tmp_path, "before", {f"{_FILE}::test_old": "passed"} | _BASE)
    after = {f"{_FILE}::test_new": "passed"} | _BASE
    b = _run(tmp_path, "after", after, ids="none")
    ok, report = compare_suite.compare(a, b)
    assert not ok and "NO NODE IDS" in "\n".join(report)


def test_outcomes_outside_the_inventory_fail(tmp_path):
    # units.json keeps test_old, outcomes.json swaps it for test_new: totals match.
    before = {f"{_FILE}::test_old": "passed"} | _BASE
    manifest = {f"{_FILE}::test_new": "passed"} | _BASE
    a = _run(tmp_path, "before", before, manifest=False)
    b = _run(tmp_path, "after", before, manifest=manifest)
    ok, report = compare_suite.compare(a, b)
    assert not ok and "outside units.json" in "\n".join(report)


def test_duplicate_node_ids_fail(tmp_path):
    ids = sorted(_BASE) + [f"{_FILE}::test_0"]
    a = _run(tmp_path, "before", _BASE, manifest=False)
    b = _run(tmp_path, "after", dict(_BASE), ids=ids)
    ok, report = compare_suite.compare(a, b)
    assert not ok and f"DUPLICATE TEST {_FILE}::test_0" in "\n".join(report)


def test_unknown_outcome_kind_fails(tmp_path):
    a = _run(tmp_path, "before", _BASE, manifest=False)
    b = _run(
        tmp_path, "after", dict(_BASE), manifest=_BASE | {f"{_FILE}::test_0": "ok"}
    )
    ok, report = compare_suite.compare(a, b)
    assert not ok and "UNKNOWN OUTCOME 'ok'" in "\n".join(report)


def test_outcome_failures_must_match_the_logs(tmp_path):
    # Same counts, but outcomes.json names a different failing test than the logs.
    manifest = _BASE | {
        f"{_FILE}::test_old_fail": "passed",
        f"{_FILE}::test_0": "failed",
    }
    a = _run(tmp_path, "before", _BASE, manifest=False)
    b = _run(tmp_path, "after", dict(_BASE), manifest=manifest)
    ok, report = compare_suite.compare(a, b)
    assert not ok and "failed/error tests" in "\n".join(report)


def test_incomplete_baseline_inventory_fails(tmp_path):
    a = _run(tmp_path, "before", _BASE, manifest=False, ids="none")
    ok, report = compare_suite.compare(a, _run(tmp_path, "after", dict(_BASE)))
    assert not ok and "NO NODE IDS" in "\n".join(report)


def test_node_ids_with_spaces_are_parsed_from_the_logs(tmp_path):
    spaced = {f"{_FILE}::test_p[a b]": "failed"} | _BASE
    a = _run(tmp_path, "before", spaced)
    ok, report = compare_suite.compare(a, _run(tmp_path, "after", dict(spaced)))
    assert ok, report


@pytest.mark.parametrize("collect_ok", [True, False])
def test_whole_file_unit_needs_its_node_ids(tmp_path, monkeypatch, collect_ok):
    """A unit whose planning collection failed runs whole; it counts only once a
    later collection gives the node ids that its JUnit outcomes cover."""
    ids = [f"{_FILE}::test_a", f"{_FILE}::test_b"]

    class _Proc:
        def wait(self, timeout=None):
            return 0

    @contextlib.contextmanager
    def fake_spawn(cmd, gpu, stdout=None, **_):
        xml = next(a.split("=", 1)[1] for a in cmd if a.startswith("--junitxml="))
        cases = "".join(
            f'<testcase classname="{run_suite.junit_name(n)[0]}" '
            f'name="{run_suite.junit_name(n)[1]}"/>'
            for n in ids
        )
        Path(xml).write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
        stdout.write("2 passed in 1.00s\n")
        stdout.flush()
        yield _Proc()

    monkeypatch.setattr(run_suite, "spawn", fake_spawn)
    monkeypatch.setattr(
        run_suite, "collect", lambda *_: (ids if collect_ok else None, "")
    )
    unit = run_suite.Unit("test_x", _FILE, [_FILE], None, 1.0)
    run_suite.run(unit, "0", tmp_path)
    if collect_ok:
        assert unit.reason == "" and unit.ids == ids
        assert unit.outcomes == dict.fromkeys(ids, "passed")
    else:
        assert "could not be collected" in unit.reason
        assert unit.outcomes == {_FILE: "error"}
