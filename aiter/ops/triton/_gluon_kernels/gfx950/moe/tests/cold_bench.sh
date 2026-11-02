#!/bin/bash
# Cold Gluon-vs-FlyDSL benchmark, full 4 dtype x 5 token matrix (~1 h on an idle GPU).
#
#   cold_bench.sh <label> <physical_gpu> [--tolerant] [run_bench args...]
#
# Extra args go to both harness passes, e.g. a quick subset:
#   cold_bench.sh smoke 5 --tolerant --precisions a4w4 --tokens 16 --rounds 1
# <physical_gpu> is the amd-smi index; the harness maps it to the HIP ordinal.
# --tolerant ignores <=4 MiB idle driver contexts that other users' multi-GPU jobs
# open on every device (otherwise the harness waits forever); real use still blocks.
# Runs the vendored harness in tests/harness/. Results (and the val_<label> validation
# run and the lock file) go to $HARNESS_OUT (default $OUT/harness):
#   $HARNESS_OUT/cold_<label>/{report.md,summary.json}
# Each finished process keeps its results and gzipped trace, not its kernel copies
# (BENCH_KEEP_ARTIFACTS=1 keeps them): about 15 MB per label, not 370 MB. The compile
# cache is $HARNESS_OUT/cache_<label>, deleted once both passes succeed; a failed run
# keeps it, so rerunning the same label resumes without recompiling.
# Compare two labels with
#   compare_cold.py <label_before> <label_after>
set -eu
source "$(dirname "$0")/_env.sh"
LABEL=${1:?label}; GPU=${2:?physical gpu}; shift 2
RUN=$HARNESS/run_bench.py
export HARNESS_CACHE=$HARNESS_OUT/cache_$LABEL
[[ "${1:-}" == "--tolerant" ]] && { RUN=$TESTS_DIR/run_bench_tolerant.py; shift; }
"$PY" "$RUN" --mode validate --label "val_$LABEL" --gpu "$GPU" --keep-going "$@"
"$PY" "$RUN" --mode cold --label "cold_$LABEL" --validation-label "val_$LABEL" \
  --gpu "$GPU" --keep-going "$@"
rm -rf "$HARNESS_CACHE"
sed -n 3p "$HARNESS_OUT/cold_$LABEL/report.md"
