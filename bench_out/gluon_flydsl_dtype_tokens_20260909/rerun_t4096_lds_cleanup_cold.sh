#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
ORIGINAL_ARGV=("$@")

PYTHON_BIN=${PYTHON_BIN:-"$(dirname -- "$REPO_ROOT")/venv/01/bin/python"}
GPU_ID=4
RUN_LABEL="lds_cleanup_$(date +%Y%m%d_%H%M%S)"
BASELINE="cold_tuned_t4096"
MAX_REGRESSION_PCT=1.0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: rerun_t4096_lds_cleanup_cold.sh [options]

Rerun the T=4096 Gluon/FlyDSL cold benchmark, audit every rocprof trace,
and compare Gluon mean, median, and p99 latency with the saved pre-cleanup
default-LDS baseline.

Options:
  --gpu ID                    Physical GPU index (default: 4)
  --label NAME                Output label prefix (default: timestamped)
  --baseline LABEL_OR_PATH    Baseline directory (default: cold_tuned_t4096)
  --max-regression-pct VALUE  Failure threshold for each Gluon statistic
                              (default: 1.0)
  --dry-run                   Print commands without running them
  -h, --help                  Show this help

Environment:
  PYTHON_BIN                  Python executable for the benchmark environment

The benchmark uses four alternating Gluon/FlyDSL rounds, 40 cold warmups and
100 measured samples per process, a 768 MiB cache flush before each GEMM, and
16 exact validation replays. Routing, quantization, scale sorting, transforms,
allocation, compilation, and launch preparation stay outside the timed loop.
EOF
}

while (($#)); do
    case "$1" in
        --gpu)
            GPU_ID=$2
            shift 2
            ;;
        --label)
            RUN_LABEL=$2
            shift 2
            ;;
        --baseline)
            BASELINE=$2
            shift 2
            ;;
        --max-regression-pct)
            MAX_REGRESSION_PCT=$2
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi

if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "--label must be a single safe path component: $RUN_LABEL" >&2
    exit 2
fi

HARNESS="$SCRIPT_DIR/run_bench.py"
AUDITOR="$SCRIPT_DIR/audit_results.py"
if [[ ! -f "$HARNESS" || ! -f "$AUDITOR" ]]; then
    echo "Benchmark harness is incomplete under $SCRIPT_DIR" >&2
    exit 2
fi

if [[ "$BASELINE" = /* ]]; then
    BASELINE_DIR=$BASELINE
else
    BASELINE_DIR="$SCRIPT_DIR/$BASELINE"
fi
VALIDATION_LABEL="${RUN_LABEL}_validation"
COLD_LABEL="${RUN_LABEL}_cold"
VALIDATION_DIR="$SCRIPT_DIR/$VALIDATION_LABEL"
COLD_DIR="$SCRIPT_DIR/$COLD_LABEL"
if [[ -e "$VALIDATION_DIR" || -e "$COLD_DIR" ]]; then
    echo "Output label already exists; choose a fresh --label: $RUN_LABEL" >&2
    exit 2
fi

"$PYTHON_BIN" - "$SCRIPT_DIR" "$BASELINE_DIR" "$GPU_ID" "$MAX_REGRESSION_PCT" <<'PY'
import json
import math
import sys
from pathlib import Path

script_dir = Path(sys.argv[1]).resolve()
baseline_dir = Path(sys.argv[2]).resolve()
try:
    gpu = int(sys.argv[3])
except ValueError as error:
    raise SystemExit(f"--gpu must be an integer: {sys.argv[3]}") from error
try:
    threshold = float(sys.argv[4])
except ValueError as error:
    raise SystemExit(
        f"--max-regression-pct must be numeric: {sys.argv[4]}"
    ) from error
if not math.isfinite(threshold) or threshold < 0:
    raise SystemExit("--max-regression-pct must be finite and non-negative")


def read(name):
    path = baseline_dir / name
    if not path.is_file():
        raise SystemExit(f"Required baseline artifact is missing: {path}")
    return json.loads(path.read_text())


expected_cases = [
    "gluon_a4w4",
    "flydsl_a4w4",
    "gluon_a8w4",
    "flydsl_a8w4",
    "gluon_a8w8",
    "flydsl_a8w8",
    "gluon_a16w16",
    "flydsl_a16w16",
]
plan = read("plan.json")
expected_protocol = {
    "mode": "cold",
    "rounds": 4,
    "warmups": 40,
    "samples": 100,
    "validation_replays": 16,
    "flush_MiB": 768,
}
for field, expected in expected_protocol.items():
    if plan.get(field) != expected:
        raise SystemExit(
            f"Baseline protocol mismatch for {field}: {plan.get(field)!r} != {expected!r}"
        )
if plan.get("gpu") != gpu:
    raise SystemExit(
        f"GPU {gpu} does not match baseline physical GPU {plan.get('gpu')}; "
        "select a matching baseline with --baseline"
    )
if list(plan.get("cases", {})) != expected_cases:
    raise SystemExit("Baseline does not contain the expected ordered dtype/backend pairs")
if {run["tokens"] for run in plan.get("runs", [])} != {4096}:
    raise SystemExit("Baseline is not a T=4096-only run")
if len(plan["runs"]) != 32:
    raise SystemExit(f"Baseline has {len(plan['runs'])} processes instead of 32")
if read("failures.json") != []:
    raise SystemExit("Baseline contains benchmark failures")

audit = read("independent_audit.json")
expected_audit = {
    "pass": True,
    "expected_processes": 32,
    "audited_processes": 32,
    "audited_trace_csv_files": 32,
    "audited_fill_gemm_pairs": 4480,
    "audited_measured_samples": 3200,
    "validated_cells": 8,
    "samples_per_cell": 400,
}
for field, expected in expected_audit.items():
    if audit.get(field) != expected:
        raise SystemExit(
            f"Baseline audit mismatch for {field}: {audit.get(field)!r} != {expected!r}"
        )

summary = read("summary.json")
expected_summary = {
    f"{precision}_t4096_{backend}_{precision}"
    for precision in ("a4w4", "a8w4", "a8w8", "a16w16")
    for backend in ("gluon", "flydsl")
}
if set(summary) != expected_summary:
    raise SystemExit("Baseline summary cells differ from the expected eight cases")
for key, row in summary.items():
    if row.get("n") != 400:
        raise SystemExit(f"Baseline cell {key} does not contain 400 samples")
    for field in ("mean_us", "median_us", "p99_us"):
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"Invalid baseline {field} for {key}: {value!r}")

sys.path.insert(0, str(script_dir))
import run_bench

baseline_provenance = read("provenance.json")
current_provenance = run_bench.provenance()
baseline_harness = dict(baseline_provenance["harness"])
current_harness = dict(current_provenance["harness"])
# HEAD renamed tuning fields while retaining host-side aliases. The worker-only
# compatibility update normalizes those aliases before checking the captured launch;
# it does not change case construction, validation, tracing, or the timed loop.
worker_path = str(script_dir / "worker.py")
expected_baseline_worker_sha256 = (
    "29567854fb1a881d41c61c00ce243943673acaa786608a0fa8e960359f3335bd"
)
expected_compat_worker_sha256 = (
    "ae09c32ba13f24e80ebe9121eb2db19f7bf2431766d7db2ad889265ab75b680c"
)
baseline_worker_sha256 = baseline_harness.pop(worker_path, None)
current_worker_sha256 = current_harness.pop(worker_path, None)
if baseline_worker_sha256 != expected_baseline_worker_sha256:
    raise SystemExit(
        "Baseline worker.py differs from the exact archived benchmark version"
    )
if current_worker_sha256 != expected_compat_worker_sha256:
    raise SystemExit(
        "Current worker.py differs from the exact audited compatibility version"
    )
if baseline_harness != current_harness:
    raise SystemExit("Current benchmark harness (excluding worker.py) differs from baseline")
for field in ("llc_sha256", "libtriton_sha256"):
    if baseline_provenance[field] != current_provenance[field]:
        raise SystemExit(f"Current {field} differs from the baseline")

current_resolved = {
    f"{case}_t4096": run_bench.load_cases(4096)[case] for case in expected_cases
}
if plan["resolved_cases"] != current_resolved:
    raise SystemExit("Current resolved tuning configurations differ from the baseline")

baseline_inventory = read("gpu_inventory.json")
current_inventory = run_bench.gpu_inventory()
if gpu not in current_inventory:
    raise SystemExit(f"Physical GPU {gpu} is unavailable")
old_gpu = baseline_inventory[str(gpu)]
new_gpu = current_inventory[gpu]
for field in ("bdf", "uuid", "hip_ordinal"):
    if old_gpu[field] != new_gpu[field]:
        raise SystemExit(
            f"GPU identity mismatch for {field}: {new_gpu[field]!r} != {old_gpu[field]!r}"
        )

print(
    f"Preflight passed: physical GPU {gpu}, HIP ordinal {new_gpu['hip_ordinal']}, "
    f"BDF {new_gpu['bdf']}, UUID {new_gpu['uuid']}"
)
PY

CASES=(
    gluon_a4w4 flydsl_a4w4
    gluon_a8w4 flydsl_a8w4
    gluon_a8w8 flydsl_a8w8
    gluon_a16w16 flydsl_a16w16
)

VALIDATE_CMD=(
    "$PYTHON_BIN" "$HARNESS"
    --mode validate
    --label "$VALIDATION_LABEL"
    --gpu "$GPU_ID"
    --cases "${CASES[@]}"
    --tokens 4096
    --rounds 1
    --warmups 40
    --samples 100
    --validation-replays 16
)
COLD_CMD=(
    "$PYTHON_BIN" "$HARNESS"
    --mode cold
    --label "$COLD_LABEL"
    --validation-label "$VALIDATION_LABEL"
    --cases "${CASES[@]}"
    --tokens 4096
    --gpu "$GPU_ID"
    --rounds 4
    --warmups 40
    --samples 100
    --validation-replays 16
)
AUDIT_CMD=("$PYTHON_BIN" "$AUDITOR" --label "$COLD_LABEL")

print_command() {
    printf ' %q' "$@"
    printf '\n'
}

cd -- "$REPO_ROOT"

if ((DRY_RUN)); then
    echo "Validation command:"
    print_command "${VALIDATE_CMD[@]}"
    echo "Cold benchmark command:"
    print_command "${COLD_CMD[@]}"
    echo "Trace audit command:"
    print_command "${AUDIT_CMD[@]}"
    echo "Comparison: $COLD_DIR versus $BASELINE_DIR"
    exit 0
fi

echo "Validating all T=4096 kernels on physical GPU $GPU_ID..."
"${VALIDATE_CMD[@]}"

"$PYTHON_BIN" - "$BASELINE_DIR" "$VALIDATION_DIR" <<'PY'
import json
import sys
from pathlib import Path

baseline_dir = Path(sys.argv[1]).resolve()
validation_dir = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(validation_dir.parent))

import run_bench


def read(path):
    return json.loads(path.read_text())


def canonicalize_gluon_tuning(kernel):
    """Compare legacy and current tuning spellings by their launch semantics."""
    tuning = kernel["effective_tuning"]
    aliases = {
        "token_mod": "token_cache_modifier",
        "token_scale_mod": "token_scale_cache_modifier",
        "expert_mod": "expert_cache_modifier",
        "expert_scale_mod": "expert_scale_cache_modifier",
        "result_mod": "result_cache_modifier",
        "result_scale_mod": "result_scale_cache_modifier",
    }
    for legacy, canonical in aliases.items():
        if legacy in tuning:
            tuning[canonical] = tuning.pop(legacy)
    if "DS_READ_IN_MFMA" in tuning:
        mask = int(tuning.pop("DS_READ_IN_MFMA"))
        tuning.update(
            DS_READ_A_PAYLOAD_IN_MFMA=bool(mask & 1),
            DS_READ_B_PAYLOAD_IN_MFMA=bool(mask & 2),
            DS_READ_A_SCALE_IN_MFMA=bool(mask & 4),
            DS_READ_B_SCALE_IN_MFMA=bool(mask & 8),
        )


if read(validation_dir / "failures.json") != []:
    raise SystemExit("Current validation contains failures")

baseline_summary = read(baseline_dir / "summary.json")
validation_records = read(validation_dir / "validation_records.json")
if len(validation_records) != 8:
    raise SystemExit(f"Expected 8 validation records, found {len(validation_records)}")

for precision in ("a4w4", "a8w4", "a8w8", "a16w16"):
    for backend in ("gluon", "flydsl"):
        case = f"{backend}_{precision}"
        old = read(baseline_dir / f"r01_{case}_t4096" / "worker.json")
        new = read(validation_dir / f"{case}_t4096" / "worker.json")
        if not new["pass"] or not new["validation"]["pass"]:
            raise SystemExit(f"Validation failed for {case}")
        if new["validation"]["cold_exact_replays"] < 16:
            raise SystemExit(f"Insufficient exact cold replays for {case}")
        for field in ("device", "versions"):
            if new[field] != old[field]:
                raise SystemExit(f"{field} differs from baseline for {case}")
        old_kernel = run_bench.stable_kernel(old["kernel"])
        new_kernel = run_bench.stable_kernel(new["kernel"])
        if backend == "gluon":
            # The full ELF includes source-location metadata, so a source cleanup can
            # change its hash while leaving the executable text byte-identical.
            old_kernel.pop("binary_sha256", None)
            new_kernel.pop("binary_sha256", None)
            canonicalize_gluon_tuning(old_kernel)
            canonicalize_gluon_tuning(new_kernel)
        if new_kernel != old_kernel:
            raise SystemExit(f"Effective kernel differs from baseline for {case}")
        summary_key = f"{precision}_t4096_{case}"
        if baseline_summary[summary_key]["device"] != new["device"]:
            raise SystemExit(f"Summary device differs from validation for {case}")

print("Validation compatibility check passed for all eight kernels")
PY

echo "Running paired cold benchmark..."
"${COLD_CMD[@]}"

echo "Auditing profiler traces..."
"${AUDIT_CMD[@]}"

"$PYTHON_BIN" - "$SCRIPT_PATH" "$REPO_ROOT" "$BASELINE_DIR" "$COLD_DIR" \
    "${ORIGINAL_ARGV[@]}" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

script = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
baseline = Path(sys.argv[3]).resolve()
current = Path(sys.argv[4]).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = {
    "wrapper": str(script),
    "wrapper_sha256": sha(script),
    "argv": [str(script), *sys.argv[5:]],
    "cwd": str(repo),
    "baseline": str(baseline),
    "current": str(current),
    "baseline_artifacts": {
        name: sha(baseline / name)
        for name in (
            "plan.json",
            "summary.json",
            "provenance.json",
            "independent_audit.json",
        )
    },
}
target = current / "rerun_manifest.json"
with tempfile.NamedTemporaryFile("w", dir=current, delete=False) as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
    temporary = Path(handle.name)
os.replace(temporary, target)
print(f"Wrote {target}")
PY

"$PYTHON_BIN" - "$BASELINE_DIR" "$COLD_DIR" "$MAX_REGRESSION_PCT" <<'PY'
import json
import math
import os
import sys
import tempfile
from pathlib import Path

baseline_dir = Path(sys.argv[1]).resolve()
current_dir = Path(sys.argv[2]).resolve()
threshold = float(sys.argv[3])


def read(path):
    return json.loads(path.read_text())


def write_atomic(path, text):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


baseline_plan = read(baseline_dir / "plan.json")
current_plan = read(current_dir / "plan.json")
for field in ("rounds", "warmups", "samples", "flush_MiB", "gpu"):
    assert baseline_plan[field] == current_plan[field], (
        "benchmark protocol differs",
        field,
        baseline_plan[field],
        current_plan[field],
    )
assert baseline_plan["resolved_cases"] == current_plan["resolved_cases"], (
    "resolved tuning configurations differ"
)
assert read(current_dir / "failures.json") == []
audit = read(current_dir / "independent_audit.json")
expected_audit = {
    "pass": True,
    "expected_processes": 32,
    "audited_processes": 32,
    "audited_trace_csv_files": 32,
    "audited_fill_gemm_pairs": 4480,
    "audited_measured_samples": 3200,
    "validated_cells": 8,
    "samples_per_cell": 400,
}
for field, expected in expected_audit.items():
    assert audit.get(field) == expected, (field, audit.get(field), expected)

baseline = read(baseline_dir / "summary.json")
current = read(current_dir / "summary.json")
expected_summary = {
    f"{precision}_t4096_{backend}_{precision}"
    for precision in ("a4w4", "a8w4", "a8w8", "a16w16")
    for backend in ("gluon", "flydsl")
}
assert set(current) == expected_summary
assert set(baseline) == expected_summary
for key, row in current.items():
    assert row["n"] == 400, (key, row["n"])
    assert all(
        math.isfinite(row[field]) and row[field] > 0
        for field in ("mean_us", "median_us", "p99_us")
    ), key
rows = []
for precision, display in (
    ("a4w4", "A4W4"),
    ("a8w4", "A8W4"),
    ("a8w8", "A8W8"),
    ("a16w16", "BF16"),
):
    def item(summary, backend):
        return summary[f"{precision}_t4096_{backend}_{precision}"]

    old = item(baseline, "gluon")
    new = item(current, "gluon")
    control_old = item(baseline, "flydsl")
    control_new = item(current, "flydsl")
    row = {"dtype": display, "samples": new["n"]}
    for field in ("mean_us", "median_us", "p99_us"):
        name = field.removesuffix("_us")
        row[f"baseline_{name}_us"] = old[field]
        row[f"current_{name}_us"] = new[field]
        row[f"{name}_delta_pct"] = (new[field] / old[field] - 1) * 100
        row[f"normalized_{name}_delta_pct"] = (
            (new[field] / control_new[field])
            / (old[field] / control_old[field])
            - 1
        ) * 100
    row["flydsl_mean_delta_pct"] = (
        control_new["mean_us"] / control_old["mean_us"] - 1
    ) * 100

    baseline_worker = read(
        baseline_dir / f"r01_gluon_{precision}_t4096" / "worker.json"
    )
    current_worker = read(
        current_dir / f"r01_gluon_{precision}_t4096" / "worker.json"
    )
    row["baseline_text_sha256"] = baseline_worker["kernel"]["text_sha256"]
    row["current_text_sha256"] = current_worker["kernel"]["text_sha256"]
    row["text_identical"] = (
        row["baseline_text_sha256"] == row["current_text_sha256"]
    )
    rows.append(row)

regressions = [
    {"dtype": row["dtype"], "metric": metric, "delta_pct": row[f"{metric}_delta_pct"]}
    for row in rows
    for metric in ("mean", "median", "p99")
    if row[f"{metric}_delta_pct"] > threshold
]

result = {
    "baseline": str(baseline_dir),
    "current": str(current_dir),
    "protocol": {
        "gpu": current_plan["gpu"],
        "rounds": current_plan["rounds"],
        "warmups_per_round": current_plan["warmups"],
        "samples_per_round": current_plan["samples"],
        "samples_per_cell": current_plan["rounds"] * current_plan["samples"],
        "flush_MiB": current_plan["flush_MiB"],
        "p99": current_plan["p99_definition"],
    },
    "max_regression_pct": threshold,
    "rows": rows,
    "regressions": regressions,
    "pass": not regressions,
}
write_atomic(
    current_dir / "regression_vs_pre_cleanup.json",
    json.dumps(result, indent=2) + "\n",
)

lines = [
    "# LDS cleanup cold performance regression check",
    "",
    f"Baseline: `{baseline_dir}`  ",
    f"Current: `{current_dir}`  ",
    f"Allowed slowdown per statistic: `{threshold:.3f}%`",
    "",
    "| Dtype | Baseline mean | Current mean | Delta mean | Baseline median | Current median | Delta median | Baseline p99 | Current p99 | Delta p99 | FlyDSL mean drift | G/F normalized mean delta | Text identical |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
]
for row in rows:
    lines.append(
        f"| {row['dtype']} | {row['baseline_mean_us']:.3f} | "
        f"{row['current_mean_us']:.3f} | {row['mean_delta_pct']:+.3f}% | "
        f"{row['baseline_median_us']:.3f} | {row['current_median_us']:.3f} | "
        f"{row['median_delta_pct']:+.3f}% | {row['baseline_p99_us']:.3f} | "
        f"{row['current_p99_us']:.3f} | {row['p99_delta_pct']:+.3f}% | "
        f"{row['flydsl_mean_delta_pct']:+.3f}% | "
        f"{row['normalized_mean_delta_pct']:+.3f}% | "
        f"{'yes' if row['text_identical'] else 'no'} |"
    )
lines += [
    "",
    "Negative deltas are faster. P99 uses the nearest-rank definition from the benchmark harness.",
    "",
    f"Result: **{'PASS' if not regressions else 'FAIL'}**",
]
if regressions:
    lines += ["", "Regressions above the configured threshold:"]
    for regression in regressions:
        lines.append(
            f"- {regression['dtype']} {regression['metric']}: "
            f"{regression['delta_pct']:+.3f}%"
        )

report = "\n".join(lines) + "\n"
write_atomic(current_dir / "regression_vs_pre_cleanup.md", report)
print(report)
print(f"Artifacts: {current_dir}")

if regressions:
    raise SystemExit(1)
PY
