"""Independently recompute statistics and re-audit every cold profiler trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text())


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (
        actual,
        expected,
    )


def trace_rows(run: Path):
    return sorted(
        (
            row
            for path in (run / "trace").rglob("*kernel_trace.csv")
            for row in csv.DictReader(path.open())
        ),
        key=lambda row: int(row["Start_Timestamp"]),
    )


def audit_trace(run: Path, pattern: str, warmups: int, samples: int):
    rows = trace_rows(run)
    total = warmups + samples
    indices = [index for index, row in enumerate(rows) if pattern in row["Kernel_Name"]]
    assert len(indices) >= total
    cold = indices[-total:]
    section = rows[cold[0] - 1 : cold[-1] + 1]
    assert len(section) == 2 * total
    assert len({rows[index]["Kernel_Name"] for index in cold}) == 1
    for ordinal, index in enumerate(cold):
        fill, gemm = rows[index - 1 : index + 1]
        assert index == cold[0] + 2 * ordinal
        assert "FillFunctor<unsigned char>" in fill["Kernel_Name"]
        assert tuple(int(fill[f"Grid_Size_{axis}"]) for axis in "XYZ") == (
            50331648,
            1,
            1,
        )
        assert pattern in gemm["Kernel_Name"]
    for first, second in pairwise(section):
        assert all(
            first[key] == second[key] for key in ("Agent_Id", "Queue_Id", "Stream_Id")
        )
        assert int(second["Dispatch_Id"]) == int(first["Dispatch_Id"]) + 1
        assert int(first["End_Timestamp"]) - int(second["Start_Timestamp"]) <= 1000
    measured = cold[-samples:]
    return [
        (int(rows[index]["End_Timestamp"]) - int(rows[index]["Start_Timestamp"]))
        / 1000
        for index in measured
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    out = ROOT / args.label
    plan = read(out / "plan.json")
    assert plan["mode"] == "cold"
    assert read(out / "failures.json") == []
    provenance = read(out / "provenance.json")
    validation_root = ROOT / plan["validation_label"]
    assert read(validation_root / "provenance.json") == provenance
    assert read(validation_root / "failures.json") == []

    records = read(out / "records.json")
    summary = read(out / "summary.json")
    csv_samples = list(csv.DictReader((out / "samples.csv").open()))
    expected_processes = len(plan["runs"])
    assert len(records) == expected_processes
    assert len(csv_samples) == expected_processes * plan["samples"]
    record_map = {
        (item["round"], item["case"], item["tokens"]): item for item in records
    }
    assert len(record_map) == expected_processes

    raw_groups = defaultdict(list)
    audited_pairs = 0
    audited_measured = 0
    trace_files = 0
    for item in plan["runs"]:
        key = (item["round"], item["case"], item["tokens"])
        record = record_map[key]
        run = out / f"r{item['round']:02d}_{item['case']}_t{item['tokens']}"
        worker = read(run / "worker.json")
        values = audit_trace(
            run,
            worker["kernel_pattern"],
            plan["warmups"],
            plan["samples"],
        )
        trace_files += len(list((run / "trace").rglob("*kernel_trace.csv")))
        audited_pairs += plan["warmups"] + plan["samples"]
        audited_measured += len(values)
        close(statistics.mean(values), record["mean_us"])
        close(statistics.median(values), record["median_us"])
        close(nearest_rank(values, 0.99), record["p99_us"])
        from_csv = [
            float(row["duration_us"])
            for row in csv_samples
            if int(row["round"]) == item["round"]
            and row["case"] == item["case"]
            and int(row["tokens"]) == item["tokens"]
        ]
        assert from_csv == values
        raw_groups[(record["precision"], item["tokens"], item["case"])].extend(
            values
        )

    expected_cells = {
        (spec["precision"], tokens, case)
        for case, spec in plan["cases"].items()
        for tokens in {item["tokens"] for item in plan["runs"]}
    }
    assert set(raw_groups) == expected_cells
    assert len(summary) == len(expected_cells)
    recomputed = {}
    for (precision, tokens, case), values in sorted(raw_groups.items()):
        assert len(values) == plan["rounds"] * plan["samples"]
        item = summary[f"{precision}_t{tokens}_{case}"]
        result = {
            "n": len(values),
            "mean_us": statistics.mean(values),
            "median_us": statistics.median(values),
            "p99_us": nearest_rank(values, 0.99),
        }
        assert item["n"] == result["n"]
        for field in ("mean_us", "median_us", "p99_us"):
            close(item[field], result[field])
        recomputed[f"{precision}_t{tokens}_{case}"] = result

    validated_cells = 0
    for case in plan["cases"]:
        for tokens in sorted({item["tokens"] for item in plan["runs"]}):
            run = validation_root / f"{case}_t{tokens}"
            assert read(run / "complete.json") == {
                "pass": True,
                "provenance": provenance,
            }
            worker = read(run / "worker.json")
            assert worker["mode"] == "validate" and worker["pass"]
            assert worker["validation"]["pass"]
            assert (
                worker["validation"]["cold_exact_replays"]
                >= plan["validation_replays"]
            )
            validated_cells += 1

    result = {
        "pass": True,
        "label": args.label,
        "p99_definition": "nearest rank: sorted[ceil(0.99*n)-1]",
        "expected_processes": expected_processes,
        "audited_processes": len(record_map),
        "audited_trace_csv_files": trace_files,
        "audited_fill_gemm_pairs": audited_pairs,
        "audited_measured_samples": audited_measured,
        "validated_cells": validated_cells,
        "samples_per_cell": plan["rounds"] * plan["samples"],
        "recomputed_summary": recomputed,
    }
    (out / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
