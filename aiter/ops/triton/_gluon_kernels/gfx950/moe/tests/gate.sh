#!/bin/bash
# Codegen / correctness gate for a behaviour-preserving change.
#
#   gate.sh <label> [--fast | --full]
#
# Writes $OUT/{sched,layout_fields,build,asm}_<label>.txt and compares them with the
# *_baseline.txt files in $OUT, falling back to tests/reference/gate/ (recorded at
# commit 1d3431e50, T0b's when the KernelLayout fields went public; valid only for
# the same toolchain). To record fresh baselines, run the label `baseline` on the
# unmodified tree, in the same checkout or in a git worktree with aiter/jit/*.so
# symlinked from a built tree (without them kernels silently differ; _env.sh warns).
#
#   T0  schedule tables (CPU only, ~16 s)               byte-identical
#   T0b KernelLayout fields, 224 configs (CPU, ~9 s)    byte-identical
#   T1  symbolic emitter + config tests (~1.5 min)      all pass
#   T2a frozen pipeline tests (~1 min cold, ~10 s warm) all pass
#   T2b tuned gemm1 builds + 2 determinism shas         --full; identity SAME, both PASS
#   T2d assembly md5 of all 20 dispatchable cells       --full; byte-identical (~1 min)
#
#   --fast  T0 + T0b only, side by side: ~16 s. Run it between edit steps, the
#           default (T0-T2a, ~3 min cold) after each round, and --full (~5 min)
#           together with run_suite.sh once at the end (agents/REGRESSION_TESTS.md).
#
# T0b runs tests/layout_fields.py; T2b runs tests/gemm1_check.py (probes in
# tests/probes/); T2d runs tests/asm_hash/asm_hash.py. All are vendored, nothing
# outside the repo is read.
#
# GPU / ASM_GPU are HIP ordinals (HIP_VISIBLE_DEVICES), not amd-smi indices.
set -u
source "$(dirname "$0")/_env.sh"
GPU=${GPU:-1}; ASM_GPU=${ASM_GPU:-3}
LABEL=${1:?label}; MODE=${2:-}
case $MODE in
  "" | --fast | --full) ;;
  *) echo "usage: gate.sh <label> [--fast | --full]" >&2; exit 2 ;;
esac
cd "$REPO"
fail=0
echo "### gate: $LABEL${MODE:+ $MODE}"

base() {  # baseline file: $OUT's if recorded, else the committed reference
  if [[ -f $OUT/$1 ]]; then echo "$OUT/$1"; else echo "$TESTS_DIR/reference/gate/$1"; fi
}
same() {  # same <name> <baseline> <new>; hunks are labelled with their "== case"
  if [[ ! -f $2 ]]; then echo "$1  (no baseline)"
  elif diff -q "$2" "$3" >/dev/null; then echo "$1  SAME"
  else  # the hunk count, then the first 20 diff lines: a few hunks of all of them
    echo "$1  *** DIFFERS ***  $(diff -U0 "$2" "$3" | grep -c '^@@') hunks:" \
      "diff -U0 -F '^== ' $2 $3"
    diff -U0 -F '^== ' "$2" "$3" | sed 1,2d | head -20 | cut -c1-200; fail=1
  fi
}

# T0b runs beside T0: both are CPU-only and mostly Python import time.
HIP_VISIBLE_DEVICES=$GPU "$PY" "$TESTS_DIR/layout_fields.py" \
  > "$OUT/layout_fields_$LABEL.txt" 2> "$OUT/layout_fields_$LABEL.err" &
t0b=$!
HIP_VISIBLE_DEVICES=$GPU "$PY" scripts/gluon_moe_sched_dump.py 2>/dev/null > "$OUT/sched_$LABEL.txt"
same "T0 schedule       " "$(base sched_baseline.txt)" "$OUT/sched_$LABEL.txt"
if wait $t0b; then
  same "T0b layout fields " "$(base layout_fields_baseline.txt)" "$OUT/layout_fields_$LABEL.txt"
else
  echo "T0b layout fields   CRASHED: $(tail -1 "$OUT/layout_fields_$LABEL.err")"; fail=1
fi

if [[ "$MODE" != "--fast" ]]; then
  out=$(HIP_VISIBLE_DEVICES=$GPU timeout 1800 "$PY" -m pytest -q -p no:cacheprovider \
    op_tests/triton_tests/moe/test_moe_gemm_gluon_buffered_schedule.py \
    op_tests/triton_tests/moe/test_moe_gemm_gluon_wait_commit.py \
    op_tests/triton_tests/moe/test_moe_gemm_gluon_config.py \
    op_tests/triton_tests/moe/test_moe_gemm_gluon_scale_miniblocks.py \
    -k "not all_reachable_remainders and not independent_buffer" 2>&1 | tail -1)
  echo "T1 symbolic        $out"
  [[ "$out" == *failed* || "$out" == *error* ]] && fail=1

  out=$(HIP_VISIBLE_DEVICES=$GPU timeout 1800 "$PY" -m pytest -q -p no:cacheprovider \
    op_tests/triton_tests/moe/test_moe_gemm_gluon_frozen.py 2>&1 | tail -1)
  echo "T2a frozen         $out"
  [[ "$out" == *failed* || "$out" == *error* ]] && fail=1
fi

if [[ "$MODE" == "--full" ]]; then
  timeout 3000 "$PY" "$TESTS_DIR/gemm1_check.py" --correctness --quick --gpu "$GPU" \
    --job "$OUT/tmp" > "$OUT/build_$LABEL.txt" 2>&1
  grep -E "impl act|determinism|ref probe" "$OUT/build_$LABEL.txt" | sed 's/^/T2b /'
  # The "CHANGED" marker compares against stale refs inside the script; identity is
  # judged against our own baseline instead.
  grep -E "impl act" "$OUT/build_$LABEL.txt" | sed 's/CHANGED//;s/<-.*//' > "$OUT/build_id_$LABEL.txt"
  same "T2b build identity" "$(base build_id_baseline.txt)" "$OUT/build_id_$LABEL.txt"
  grep -q "determinism bf16.*PASS" "$OUT/build_$LABEL.txt" || { echo "T2c bf16 sha      FAIL"; fail=1; }
  grep -q "determinism mxfp4-out.*PASS" "$OUT/build_$LABEL.txt" || { echo "T2c mxfp4 sha     FAIL"; fail=1; }

  HIP_VISIBLE_DEVICES=$ASM_GPU timeout 2700 "$PY" "$TESTS_DIR/asm_hash/asm_hash.py" \
    2>/dev/null > "$OUT/asm_$LABEL.txt"
  same "T2d codegen x20   " "$(base asm_baseline.txt)" "$OUT/asm_$LABEL.txt"
  grep -q FAILED "$OUT/asm_$LABEL.txt" && { echo "T2d some kernels FAILED to compile"; fail=1; }
fi
echo "### verdict: $([[ $fail == 0 ]] && echo PASS || echo FAIL)"
exit $fail
