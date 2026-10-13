#!/bin/bash
# Whole gluon MoE pytest suite, one process per test file, spread over GPUs.
# A single-process run exceeds 2 h; per file in parallel it is ~25 min.
#
#   run_suite.sh <label> [hip_ordinals...]      e.g. run_suite.sh base 1 2 4 6
#
# Logs go to $OUT/suite_<label>/<file>.log. Compare two runs with
#   compare_suite.py $OUT/suite_<before> $OUT/suite_<after>
set -u
source "$(dirname "$0")/_env.sh"
LABEL=${1:?label}; shift
GPUS=(${@:-1 2 3 4})
D=$OUT/suite_$LABEL; mkdir -p "$D"
cd "$REPO"
files=(op_tests/triton_tests/moe/test_moe_gemm_gluon*.py)
for g in "${!GPUS[@]}"; do
  (
    for ((i = g; i < ${#files[@]}; i += ${#GPUS[@]})); do
      f=${files[$i]}
      HIP_VISIBLE_DEVICES=${GPUS[$g]} timeout 7200 "$PY" -m pytest -q -p no:cacheprovider \
        -rfE "$f" > "$D/$(basename "$f" .py).log" 2>&1
    done
  ) &
done
wait
for l in "$D"/*.log; do printf '%-50s %s\n' "$(basename "$l")" "$(tail -1 "$l")"; done
