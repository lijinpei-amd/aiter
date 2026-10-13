"""Reference check at the exact benchmarked configuration.

The suite's Case list has (m=4096, n=4096, k=7168, topk=8) but only with
n_expts_tot=256; the benchmark and the ATT capture use n_expts_tot=33. Expert count
drives the ragged-block tail, so run _run_case at 33 directly, several times, since
the MANUAL_PP race is nondeterministic.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))
from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import _run_case

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CASES = [
    # gather+scatter on, gammas on, swiglu on, fused quant off/on -- the benchmark
    # calls moe_gemm_gluon with apply_swiglu=True and gather indices.
    {
        "m": 4096,
        "n": 4096,
        "k": 7168,
        "do_gather": True,
        "do_scatter": False,
        "has_y_gammas": True,
        "apply_swiglu": True,
        "fused_quant": False,
        "n_expts_tot": 33,
        "n_expts_act": 8,
        "hbm_swizzling": True,
    },
    {
        "m": 4096,
        "n": 4096,
        "k": 7168,
        "do_gather": True,
        "do_scatter": False,
        "has_y_gammas": True,
        "apply_swiglu": True,
        "fused_quant": True,
        "n_expts_tot": 33,
        "n_expts_act": 8,
        "hbm_swizzling": True,
    },
    {
        "m": 4096,
        "n": 4096,
        "k": 7168,
        "do_gather": True,
        "do_scatter": True,
        "has_y_gammas": True,
        "apply_swiglu": True,
        "fused_quant": False,
        "n_expts_tot": 33,
        "n_expts_act": 8,
        "hbm_swizzling": True,
    },
]
fails = 0
for i in range(REPS):
    for j, c in enumerate(CASES):
        try:
            _run_case(**c)
            print(f"  rep {i} case {j}: PASS", flush=True)
        except Exception:  # noqa: BLE001 -- count every failure, keep going
            fails += 1
            print(f"  rep {i} case {j}: FAIL", flush=True)
            traceback.print_exc()
print(
    f"\n{REPS*len(CASES)-fails}/{REPS*len(CASES)} passed"
    f" -> {'PASS' if not fails else 'FAIL'}"
)
sys.exit(1 if fails else 0)
