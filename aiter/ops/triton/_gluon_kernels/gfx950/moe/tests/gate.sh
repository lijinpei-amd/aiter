#!/bin/bash
# Codegen / correctness gate for a behaviour-preserving change.
#
#   gate.sh <label> [--full]
#
# Writes $OUT/{sched,build,asm}_<label>.txt and compares them with the
# *_baseline.txt files in $OUT, falling back to tests/reference/gate/ (recorded at
# commit 1d3431e50; valid only for the same toolchain). To record fresh baselines,
# run the label `baseline` on the unmodified tree, in the SAME checkout (a git
# worktree is not a valid control: it lacks aiter/jit/*.so and silently compiles
# different kernels).
#
#   T0  schedule tables (CPU only, ~20 s)           byte-identical
#   T1  symbolic emitter + config tests             all pass
#   T2a frozen pipeline tests                       all pass
#   T2b tuned gemm1 builds + 2 determinism shas     --full; identity SAME, both PASS
#   T2d assembly md5 of all 20 dispatchable cells   --full; byte-identical (~12 min)
#
# T2b runs tests/gemm1_check.py (probes in tests/probes/); T2d runs
# tests/asm_hash/asm_hash.py. Both are vendored, nothing outside the repo is read.
#
# GPU / ASM_GPU are HIP ordinals (HIP_VISIBLE_DEVICES), not amd-smi indices.
set -u
source "$(dirname "$0")/_env.sh"
GPU=${GPU:-1}; ASM_GPU=${ASM_GPU:-3}
LABEL=${1:?label}; FULL=${2:-}
cd "$REPO"
fail=0
echo "### gate: $LABEL"

base() {  # baseline file: $OUT's if recorded, else the committed reference
  if [[ -f $OUT/$1 ]]; then echo "$OUT/$1"; else echo "$TESTS_DIR/reference/gate/$1"; fi
}
same() {  # same <name> <baseline> <new>
  if [[ ! -f $2 ]]; then echo "$1  (no baseline)"
  elif diff -q "$2" "$3" >/dev/null; then echo "$1  SAME"
  else echo "$1  *** DIFFERS ***"; diff "$2" "$3" | head -20; fail=1; fi
}

HIP_VISIBLE_DEVICES=$GPU "$PY" scripts/gluon_moe_sched_dump.py 2>/dev/null > "$OUT/sched_$LABEL.txt"
same "T0 schedule       " "$(base sched_baseline.txt)" "$OUT/sched_$LABEL.txt"

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

if [[ "$FULL" == "--full" ]]; then
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
