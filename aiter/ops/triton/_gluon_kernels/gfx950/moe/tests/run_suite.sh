#!/bin/bash
# Whole gluon MoE pytest suite, split into balanced units and run longest first
# from one queue on JOBS_PER_GPU workers per GPU; run_suite.py does the
# scheduling and lists the knobs. On HIP 4-7 it takes ~5 min cold, ~1.5 min
# warm, bounded by its slowest single test; one process per file took 34 min.
#
#   run_suite.sh <label> <hip_ordinal>...       e.g. run_suite.sh base 4 5 6 7
#
# The HIP ordinals have no default: each gets JOBS_PER_GPU (8) pytest workers, too
# many for a GPU another user's jobs hold.
#
# Logs go to $OUT/suite_<label>/<file>[.NN].log, one per unit; units.json says
# where and how long each ran. Compare two runs with
#   compare_suite.py <before> <after>     (labels, or suite directories)
set -u
(($# >= 2)) || { echo "usage: run_suite.sh <label> <hip_ordinal>..." >&2; exit 2; }
source "$(dirname "$0")/_env.sh"
LABEL=$1; shift
cd "$REPO"
exec "$PY" "$TESTS_DIR/run_suite.py" --out "$OUT/suite_$LABEL" --gpus "$*"
