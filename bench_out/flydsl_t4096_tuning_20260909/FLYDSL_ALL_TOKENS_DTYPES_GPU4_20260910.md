# FlyDSL all-token/all-dtype cold benchmark, including A4W4 TM128

- Date: 2026-09-10
- Repository HEAD: `591175e3a7f77edd8077737ab8b83ed51f54a62a`
- Device for every result in this report: physical GPU 4, AMD Instinct MI350X,
  PCI BDF `0000:86:00.0`, HIP-visible ordinal 5
- Shape constants: H/K=7168, raw N=4096, intermediate=2048, E=33, top-k=8

No number from a different physical GPU is used in a comparison below.

## Protocol

Each cell contains 400 measured kernel durations pooled from four serialized
processes. Each process used 40 cold warmups and 100 measured samples. Every
sample was exactly one 768 MiB same-stream fill followed immediately by one
prepared GEMM1 dispatch. Routing, input quantization, A-scale sorting,
weight/scale transforms, allocation, compilation, and launch preparation were
outside the measured loop. P99 is nearest-rank
`sorted[ceil(0.99*n)-1]`.

## Correctness-qualified token-aware FlyDSL matrix

All 20 cells passed the independent reference and 16 poisoned-output exact cold
replays before timing.

| Dtype | Tokens | Configuration | Mean (us) | Median (us) | P99 (us) |
|---|---:|---|---:|---:|---:|
| A4W4 | 16 | TM32 x TN256 x TK256, NT | 140.628 | 140.121 | 147.321 |
| A4W4 | 64 | TM32 x TN256 x TK256, NT | 139.261 | 140.921 | 147.041 |
| A4W4 | 256 | TM32 x TN256 x TK256, NT | 174.419 | 175.941 | 182.081 |
| A4W4 | 1024 | TM64 x TN256 x TK256, cached, XCD0 | 236.999 | 237.102 | 248.762 |
| A4W4 | 4096 | TM64 x TN256 x TK256, cached, XCD4 | 683.504 | 682.885 | 710.284 |
| A8W4 | 16 | TM32 x TN128 x TK256, WPE4 | 128.749 | 129.741 | 136.681 |
| A8W4 | 64 | TM32 x TN128 x TK256, WPE4 | 130.516 | 132.701 | 139.401 |
| A8W4 | 256 | TM64 x TN128 x TK256, WPE4 | 141.691 | 142.441 | 148.521 |
| A8W4 | 1024 | TM128 x TN128 x TK256, WPE4 | 319.343 | 318.182 | 340.562 |
| A8W4 | 4096 | TM128 x TN256 x TK256, WPE4 | 954.275 | 952.166 | 1008.766 |
| A8W8 | 16 | TM32 x TN128 x TK256, WPE2 | 238.126 | 240.242 | 247.282 |
| A8W8 | 64 | TM32 x TN128 x TK256, WPE2 | 242.960 | 242.822 | 248.682 |
| A8W8 | 256 | TM64 x TN128 x TK256, WPE2 | 255.774 | 257.902 | 264.002 |
| A8W8 | 1024 | TM128 x TN128 x TK256, WPE2 | 394.957 | 394.343 | 408.563 |
| A8W8 | 4096 | TM128 x TN128 x TK256, WPE2 | 1301.612 | 1299.809 | 1354.729 |
| BF16 | 16 | TM32 x TN128 x TK128, WPE1 | 433.769 | 433.963 | 443.683 |
| BF16 | 64 | TM16 x TN128 x TK128, WPE1 | 439.852 | 439.583 | 451.443 |
| BF16 | 256 | TM64 x TN128 x TK128, WPE1 | 515.871 | 515.663 | 531.323 |
| BF16 | 1024 | TM64 x TN128 x TK128, WPE1 | 1045.558 | 1042.807 | 1084.486 |
| BF16 | 4096 | TM128 x TN128 x TK128, WPE1 | 3448.982 | 3446.882 | 3481.662 |

The independent audit passed all 80 profiler processes, 80 trace CSVs, 11,200
fill/GEMM pairs, and 8,000 measured samples. Every dispatch was on physical GPU
4 / BDF `0000:86:00.0`; trace durations independently reproduced every stored
mean, median, and p99.

## Unified same-GPU comparison with current Gluon

The Gluon source cells below are also physical GPU 4 / BDF `0000:86:00.0`:

- T=16/64/256/1024:
  `../gluon_flydsl_dtype_tokens_20260909/layout_refactor_small_20260909_cold/`
- T=4096:
  `../gluon_flydsl_dtype_tokens_20260909/layout_refactor_20260909_r3_cold/`

The FlyDSL result in each row is one complete configuration selected by the
lowest observed median among the available candidates; its mean and p99 come
from that same run, rather than taking an independent minimum for each
statistic. For A4W4 the candidates are the token-aware selection and fixed
TM128; for the other dtypes only the token-aware result was measured.
"Token-aware" means the per-token-count tuned/selected FlyDSL configuration
(TM32, TM64, or TM128 as appropriate), while "fixed TM128" forces TM128 at
every token count.

The logical inputs match, but the Gluon and new FlyDSL measurements were
separate jobs rather than contemporaneous interleaved pairs. Small-token cells
show several-percent time drift between rounds, so use these as current-status
comparisons rather than precise causal deltas. `G/F` is `100 * (Gluon/FlyDSL -
1)`; positive means FlyDSL is faster and negative means Gluon is faster.

| Dtype | Tokens | Gluon mean | Gluon median | Gluon p99 | Selected FlyDSL configuration | FlyDSL mean | FlyDSL median | FlyDSL p99 | G/F mean | G/F median | G/F p99 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| A4W4 | 16 | 134.345 | 136.021 | 142.241 | Token-aware: TM32 x TN256 x TK256, NT (qualified) | 140.628 | 140.121 | 147.321 | -4.47% | -2.93% | -3.45% |
| A4W4 | 64 | 136.304 | 138.201 | 143.481 | Token-aware: TM32 x TN256 x TK256, NT (qualified) | 139.261 | 140.921 | 147.041 | -2.12% | -1.93% | -2.42% |
| A4W4 | 256 | 139.689 | 139.761 | 144.161 | Token-aware: TM32 x TN256 x TK256, NT (qualified) | 174.419 | 175.941 | 182.081 | -19.91% | -20.56% | -20.83% |
| A4W4 | 1024 | 238.856 | 238.462 | 251.961 | Fixed TM128 x TN256 x TK256, cached, XCD0 (diagnostic only) [^tm128] | 238.008 | 237.022 | 255.802 | +0.36% | +0.61% | -1.50% |
| A4W4 | 4096 | 611.884 | 610.585 | 646.685 | Fixed TM128 x TN256 x TK256, cached, XCD0 (diagnostic only) [^tm128] | 608.695 | 607.944 | 649.284 | +0.52% | +0.43% | -0.40% |
| A8W4 | 16 | 200.540 | 200.402 | 204.722 | Token-aware: TM32 x TN128 x TK256, WPE4 (qualified) | 128.749 | 129.741 | 136.681 | +55.76% | +54.46% | +49.78% |
| A8W4 | 64 | 206.074 | 206.002 | 209.962 | Token-aware: TM32 x TN128 x TK256, WPE4 (qualified) | 130.516 | 132.701 | 139.401 | +57.89% | +55.24% | +50.62% |
| A8W4 | 256 | 195.837 | 195.681 | 201.962 | Token-aware: TM64 x TN128 x TK256, WPE4 (qualified) | 141.691 | 142.441 | 148.521 | +38.21% | +37.38% | +35.98% |
| A8W4 | 1024 | 315.150 | 315.022 | 328.162 | Token-aware: TM128 x TN128 x TK256, WPE4 (qualified) | 319.343 | 318.182 | 340.562 | -1.31% | -0.99% | -3.64% |
| A8W4 | 4096 | 1031.342 | 1029.608 | 1086.648 | Token-aware: TM128 x TN256 x TK256, WPE4 (qualified) | 954.275 | 952.166 | 1008.766 | +8.08% | +8.13% | +7.72% |
| A8W8 | 16 | 275.055 | 272.882 | 287.522 | Token-aware: TM32 x TN128 x TK256, WPE2 (qualified) | 238.126 | 240.242 | 247.282 | +15.51% | +13.59% | +16.27% |
| A8W8 | 64 | 278.073 | 277.862 | 283.562 | Token-aware: TM32 x TN128 x TK256, WPE2 (qualified) | 242.960 | 242.822 | 248.682 | +14.45% | +14.43% | +14.03% |
| A8W8 | 256 | 297.041 | 298.982 | 312.162 | Token-aware: TM64 x TN128 x TK256, WPE2 (qualified) | 255.774 | 257.902 | 264.002 | +16.13% | +15.93% | +18.24% |
| A8W8 | 1024 | 400.633 | 400.843 | 408.843 | Token-aware: TM128 x TN128 x TK256, WPE2 (qualified) | 394.957 | 394.343 | 408.563 | +1.44% | +1.65% | +0.07% |
| A8W8 | 4096 | 1180.138 | 1177.369 | 1227.930 | Token-aware: TM128 x TN128 x TK256, WPE2 (qualified) | 1301.612 | 1299.809 | 1354.729 | -9.33% | -9.42% | -9.36% |
| BF16 | 16 | 442.713 | 440.423 | 451.083 | Token-aware: TM32 x TN128 x TK128, WPE1 (qualified) | 433.769 | 433.963 | 443.683 | +2.06% | +1.49% | +1.67% |
| BF16 | 64 | 442.208 | 441.843 | 452.524 | Token-aware: TM16 x TN128 x TK128, WPE1 (qualified) | 439.852 | 439.583 | 451.443 | +0.54% | +0.51% | +0.24% |
| BF16 | 256 | 500.466 | 500.124 | 514.164 | Token-aware: TM64 x TN128 x TK128, WPE1 (qualified) | 515.871 | 515.663 | 531.323 | -2.99% | -3.01% | -3.23% |
| BF16 | 1024 | 715.895 | 714.825 | 740.965 | Token-aware: TM64 x TN128 x TK128, WPE1 (qualified) | 1045.558 | 1042.807 | 1084.486 | -31.53% | -31.45% | -31.68% |
| BF16 | 4096 | 2278.394 | 2277.097 | 2329.098 | Token-aware: TM128 x TN128 x TK128, WPE1 (qualified) | 3448.982 | 3446.882 | 3481.662 | -33.94% | -33.94% | -33.10% |

[^tm128]: Selected by the lowest-median rule, but not correctness-qualified:
    retained exact-replay failures occurred at T=1024 replay 9 and T=4096
    replays 1, 185, and 210. See [TM128 correctness status](#tm128-correctness-status).

## Fixed-TM128 A4W4 diagnostic sweep

The fixed configuration is TM128 x TN256 x TK256, cached, separated gate/up,
XCD0 at every token count.

| Tokens | Mean (us) | Median (us) | P99 (us) | Mean vs token-aware | Median vs token-aware | P99 vs token-aware |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 190.581 | 190.121 | 199.561 | +35.52% | +35.68% | +35.46% |
| 64 | 189.635 | 189.022 | 200.081 | +36.17% | +34.13% | +36.07% |
| 256 | 191.973 | 191.301 | 203.001 | +10.06% | +8.73% | +11.49% |
| 1024 | 238.008 | 237.022 | 255.802 | +0.43% | -0.03% | +2.83% |
| 4096 | 608.695 | 607.944 | 649.284 | -10.95% | -10.97% | -8.59% |

Positive percentages mean fixed TM128 is slower; negative percentages mean it
is faster. Both suites used the same physical GPU, but they were separate
serialized jobs rather than an interleaved pair, so very small differences
should not be overinterpreted. The separate alternating GPU-4 T=4096 run also
measured TM128 at 606.904 us median and TM64/XCD4 at 685.485 us median, which
corroborates the large T=4096 difference.

The TM128 timing audit passed all 20 profiler processes, 20 trace CSVs, 2,800
fill/GEMM pairs, and 2,000 measured samples.

## TM128 correctness status

TM128 is not correctness-qualified. In the first all-cell validation sweep it
passed T=16, 64, and 256, but failed exact replay equality at:

- T=1024: replay 9
- T=4096: replay 1

A separate retry happened to pass 16 replays at all five token counts and was
used to gate the diagnostic timing run. That pass does not supersede the
retained failures. The same T=4096 binary has also failed longer validations at
replays 185 and 210. TM128 must therefore remain a diagnostic performance row,
not the production/tuned FlyDSL baseline.

## Artifacts

- Production matrix: `cold_flydsl_all_tokens_dtypes_gpu4_20260910/`
- Production summary: `cold_flydsl_all_tokens_dtypes_gpu4_20260910/summary.json`
- Production independent audit:
  `cold_flydsl_all_tokens_dtypes_gpu4_20260910/independent_audit.json`
- Production validation: `validation_flydsl_all_tokens_dtypes_gpu4_20260910/`
- TM128 timing: `cold_flydsl_a4_tm128_all_tokens_gpu4_20260910/`
- TM128 summary: `cold_flydsl_a4_tm128_all_tokens_gpu4_20260910/summary.json`
- TM128 independent audit:
  `cold_flydsl_a4_tm128_all_tokens_gpu4_20260910/independent_audit.json`
- TM128 validation with retained T=1024/T=4096 failures:
  `validation_flydsl_all_tokens_dtypes_tm128_gpu4_20260910/`
- TM128 short-pass retry used for timing:
  `validation_flydsl_a4_tm128_all_tokens_gpu4_retry1_20260910/`
