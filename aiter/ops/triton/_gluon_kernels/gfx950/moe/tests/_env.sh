# Shared environment for the regression scripts in this directory. Sourced, not run.
# Override any of these from the caller's environment.
TESTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$TESTS_DIR/../../../../../../.." && pwd)}
# The venv and the patched llc are toolchain, not test inputs. The defaults assume
# this machine's layout: venv/01 and llvm-project/ next to the aiter checkout.
PY=${PY:-$(dirname "$REPO")/venv/01/bin/python}
export ROCM_PATH=${ROCM_PATH:-$("$PY" -c 'import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))' 2>/dev/null)}
export TRITON_HIP_DS_AGPR_LLC=${TRITON_HIP_DS_AGPR_LLC:-$(dirname "$REPO")/llvm-project/build/bin/llc}
# Keep the vendored sources free of __pycache__.
export PYTHONDONTWRITEBYTECODE=1
# Cold-bench harness (vendored in tests/harness/).
HARNESS=${HARNESS:-$TESTS_DIR/harness}
# Where baselines and run outputs go (untracked).
OUT=${OUT:-$REPO/bench_out/gluon_moe_regression}
# Cold-bench labels, compile cache and lock file.
export HARNESS_OUT=${HARNESS_OUT:-$OUT/harness}
mkdir -p "$OUT" "$HARNESS_OUT"
