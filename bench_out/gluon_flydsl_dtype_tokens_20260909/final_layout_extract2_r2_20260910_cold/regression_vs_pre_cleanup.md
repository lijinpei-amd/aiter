# LDS cleanup cold performance regression check

Baseline: `/raid/jinpli/workspace/home01/jinpli/development/aiter/bench_out/gluon_flydsl_dtype_tokens_20260909/cold_tuned_t4096`

Current: `/raid/jinpli/workspace/home01/jinpli/development/aiter/bench_out/gluon_flydsl_dtype_tokens_20260909/final_layout_extract2_r2_20260910_cold`
Allowed slowdown per statistic: `1.000%`

| Dtype | Baseline mean | Current mean | Delta mean | Baseline median | Current median | Delta median | Baseline p99 | Current p99 | Delta p99 | FlyDSL mean drift | G/F normalized mean delta | Text identical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| A4W4 | 612.268 | 611.317 | -0.155% | 612.004 | 609.284 | -0.444% | 646.764 | 640.884 | -0.909% | -0.375% | +0.221% | yes |
| A8W4 | 1034.925 | 1033.745 | -0.114% | 1032.768 | 1031.887 | -0.085% | 1087.687 | 1103.048 | +1.412% | -0.093% | -0.021% | yes |
| A8W8 | 1182.781 | 1182.634 | -0.012% | 1181.849 | 1181.488 | -0.031% | 1233.007 | 1230.528 | -0.201% | -0.001% | -0.011% | yes |
| BF16 | 2282.290 | 2280.967 | -0.058% | 2280.796 | 2279.193 | -0.070% | 2328.655 | 2329.734 | +0.046% | -0.077% | +0.019% | yes |

Negative deltas are faster. P99 uses the nearest-rank definition from the benchmark harness.

Result: **FAIL for the configured raw-statistic threshold**

Regressions above the configured threshold:
- A8W4 p99: +1.412%

Interpretation: this is a transparent threshold result, not a demonstrated
code-induced regression. A8W4 mean and median were flat/faster, paired FlyDSL
mean drifted by a similar amount, and the current and baseline executable text
is byte-identical.
