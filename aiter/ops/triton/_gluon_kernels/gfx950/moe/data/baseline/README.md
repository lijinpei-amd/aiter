# Regression baselines

Recorded on 2026-11-02 at commit `96a384028`, using the venv rebuilt on 2026-10-30 (torch 2.15
nightly, ROCm 7.14) and the patched llc. See `agents/REGRESSION_TESTS.md` for how
each check runs. These are valid as the "before" side only for a codegen-neutral
descendant on the same machine and toolchain. Otherwise record a fresh baseline.

| Dir | Check | Use |
|---|---|---|
| `gate/` | `tests/gate.sh` T0, T0b, T2b, T2d. T0b is in parts, `layout_fields_baseline.partNN.txt`, that `gate.sh` joins in order | Used automatically when `$OUT` has no `*_baseline.txt` |
| `suite_base/` | `tests/run_suite.sh`: 2625 passed / 20 failed / 22 skipped, one `.txt` per unit (FAILED/ERROR lines + pytest summary) | `python tests/compare_suite.py data/baseline/suite_base <label>` |
| `cold_hv_new_phys5/` | `tests/cold_bench.sh hv_new 5 --tolerant`, physical GPU 5, 160/160 processes | `python tests/compare_cold.py data/baseline/cold_hv_new_phys5 <label>`, on physical GPU 5 only |

When recorded, the gate files were byte-identical to the previous `$OUT` baselines,
and the suite matched the previous base exactly. T0b was re-recorded on 2026-11-02 for
the GEMM2 launch config: only the `prod/g2-*` and `prod/scale_prep-mxfp8-noact` cases
changed.
