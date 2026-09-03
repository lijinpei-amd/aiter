# FlyDSL mxmoe a4w4 GEMM2: v1 vs v2

`v1` = `flydsl_mxmoe_g2_a4w4_*` ([mxfp4_gemm2.py](aiter/ops/flydsl/kernels/mxfp4_gemm2.py), 2026-07-09 `b2dd7703d`) &nbsp;&nbsp; `v2` = `flydsl_moe2_layout_*` ([mxmoe_gemm_v2.py](aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py), 2026-08-04 `db24351ee`).

Both families run behind the same mxmoe GEMM1 (`flydsl_mxmoe_g1_a4w4_*`), so `us_g2` isolates the stage2 kernel and `us_e2e` covers sort + GEMM1 + GEMM2. Every candidate is gated on `cosine_diff_compare <= 0.1` against a torch reference before it is timed. `speedup = v1 / v2`, so **> 1 means v2 is faster**.

## Headline

- shapes attempted: **424**
- shapes measured: **300**
- head-to-head (both families ran): **286**
- v2-only (no v1 kernel exists): **14**
- blocked (neither family could run): **124** — see Coverage below
- geomean speedup, GEMM2 only: **1.074x**
- geomean speedup, end-to-end: **1.033x**
- v2 wins 222, v1 wins 37, tie (within 2%) 27 (GEMM2 only)

## Why the GEMM2 win shrinks end-to-end

There is **no v2 GEMM1** — `flydsl_moe2_layout_*` is the only v2 family, and commit `db24351ee` touched no gemm1 file. Both paths run the same `flydsl_mxmoe_g1_a4w4_*`, so GEMM1 is a shared constant here, not a comparison axis. It is still the larger half of the runtime:

| tokens | median share of e2e that is sort + GEMM1 |
|---|---|
| 1-4 | 78% |
| 8-64 | 70% |
| 128-1024 | 66% |
| 2048-8192 | 57% |
| 16384-32768 | 48% |

So stage2 controls only ~22% of end-to-end time at small batch and ~52% at the largest. That is the whole reason a 1.07x GEMM2 win lands as ~1.03x end-to-end — and it says the next real win is in the sort + GEMM1 half, not in further stage2 tuning.

Best GEMM1 variant per shape (chosen jointly with stage2, by e2e): `16x256x256_f16in_nt` x158, `128x256x256` x73, `64x256x256` x38, `32x256x256_nt` x27, `32x256x256` x4.

## Per model

| model | shapes | h2h | geomean g2 | geomean e2e | v2 wins | v1 wins | tie |
|---|---|---|---|---|---|---|---|
| dsv3_fp4 | 95 | 62 | 1.125x | 1.050x | 53 | 6 | 3 |
| glm5_fp4 | 64 | 64 | 1.080x | 1.045x | 50 | 9 | 5 |
| gptoss_fp4 | 24 | 0 | - | - | 0 | 0 | 0 |
| kimik2_fp4 | 80 | 80 | 1.037x | 1.016x | 58 | 13 | 9 |
| kimik3_a4w4 | 17 | 0 | - | - | 0 | 0 | 0 |
| minimax_m25_fp4 | 64 | 48 | 1.084x | 1.032x | 37 | 6 | 5 |
| minimax_m3_fp4 | 32 | 0 | - | - | 0 | 0 | 0 |
| qwen3_5_397b_fp4 | 48 | 32 | 1.043x | 1.025x | 24 | 3 | 5 |

## Per shape

### dsv3_fp4 — model_dim=7168, inter_dim=256, E=257, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.11 | 3.40 | **0.917x** | 25.22 | 25.50 | **0.989x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 3.75 | 4.07 | **0.920x** | 29.51 | 29.56 | **0.998x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 5.72 | 5.74 | **0.995x** | 34.74 | 34.51 | **1.007x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 12.89 | 11.95 | **1.079x** | 45.26 | 42.05 | **1.076x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 19.81 | 18.85 | **1.051x** | 64.94 | 63.21 | **1.027x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 30.80 | 28.88 | **1.067x** | 103.24 | 101.21 | **1.020x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 39.99 | 38.14 | **1.048x** | 126.11 | 125.61 | **1.004x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 128 | 46.54 | 43.90 | **1.060x** | 149.93 | 144.08 | **1.041x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 256 | 51.34 | 48.18 | **1.066x** | 176.80 | 169.20 | **1.045x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 512 | 71.40 | 66.74 | **1.070x** | 212.68 | 206.97 | **1.028x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_sbm32` | 14/14 · 160/160 |
| 1024 | 127.67 | 110.95 | **1.151x** | 298.88 | 272.62 | **1.096x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 199.72 | 189.11 | **1.056x** | 396.54 | 370.49 | **1.070x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_nt_sbm128` | 14/14 · 160/160 |
| 4096 | 345.93 | 315.69 | **1.096x** | 633.27 | 620.38 | **1.021x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 613.75 | 596.98 | **1.028x** | 1030.71 | 1054.49 | **0.977x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1136.59 | 1155.96 | **0.983x** | 1790.61 | 1802.40 | **0.994x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 2195.18 | 2249.15 | **0.976x** | 3249.62 | 3389.68 | **0.959x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |

### dsv3_fp4 — model_dim=7168, inter_dim=512, E=257, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.45 | 3.84 | **0.896x** | 29.11 | 29.24 | **0.995x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 6.17 | 5.14 | **1.201x** | 33.35 | 33.89 | **0.984x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 12.29 | 12.19 | **1.008x** | 43.50 | 42.91 | **1.014x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 22.53 | 21.92 | **1.028x** | 71.77 | 70.32 | **1.021x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 16 | 38.03 | 36.50 | **1.042x** | 119.65 | 119.17 | **1.004x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_sbm32` | 14/14 · 160/160 |
| 32 | 63.63 | 59.17 | **1.075x** | 195.43 | 188.75 | **1.035x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 80.38 | 76.60 | **1.049x** | 248.53 | 241.99 | **1.027x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 91.41 | 87.29 | **1.047x** | 288.51 | 276.09 | **1.045x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 99.55 | 94.30 | **1.056x** | 313.64 | 305.28 | **1.027x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 106.58 | 103.37 | **1.031x** | 338.88 | 336.39 | **1.007x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 151.31 | 136.42 | **1.109x** | 417.22 | 405.78 | **1.028x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 255.85 | 245.94 | **1.040x** | 574.61 | 549.60 | **1.046x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_persist_sbm32` | 14/14 · 160/160 |
| 4096 | 432.82 | 411.03 | **1.053x** | 905.73 | 892.17 | **1.015x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 738.93 | 698.14 | **1.058x** | 1365.05 | 1313.36 | **1.039x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1294.64 | 1334.36 | **0.970x** | 2284.45 | 2336.57 | **0.978x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 2492.28 | 2692.15 | **0.926x** | 4337.46 | 4391.80 | **0.988x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### dsv3_fp4 — model_dim=7168, inter_dim=2048, E=32, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.11 | 11.34 | **1.156x** | 43.29 | 40.91 | **1.058x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 2 | 22.71 | 17.29 | **1.314x** | 60.53 | 54.16 | **1.118x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 4 | 34.69 | 27.25 | **1.273x** | 104.95 | 96.10 | **1.092x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 42.06 | 35.90 | **1.171x** | 126.00 | 119.80 | **1.052x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 16 | 50.44 | 41.52 | **1.215x** | 149.30 | 138.25 | **1.080x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 32 | 52.64 | 42.71 | **1.233x** | 153.97 | 144.09 | **1.069x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 64 | 56.46 | 43.08 | **1.311x** | 166.93 | 153.96 | **1.084x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 128 | 69.35 | 46.50 | **1.491x** | 198.59 | 174.70 | **1.137x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_sbm64` | 14/14 · 160/160 |
| 256 | 97.29 | 66.88 | **1.455x** | 237.20 | 196.55 | **1.207x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_sbm128` | 14/14 · 160/160 |
| 512 | 131.33 | 99.58 | **1.319x** | 299.73 | 263.65 | **1.137x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_atomic_sbm128` | 14/14 · 160/160 |
| 1024 | 217.94 | 173.46 | **1.256x** | 457.91 | 398.64 | **1.149x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 2048 | 348.73 | 298.74 | **1.167x** | 741.10 | 681.92 | **1.087x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 16384 | 2262.23 | 2057.80 | **1.099x** | 4771.97 | 4639.34 | **1.029x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 4586.48 | 4127.52 | **1.111x** | 9497.33 | 8974.41 | **1.058x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### dsv3_fp4 — model_dim=7168, inter_dim=2048, E=33, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.74 | 11.61 | **1.183x** | 44.85 | 41.52 | **1.080x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 2 | 23.78 | 18.53 | **1.284x** | 66.14 | 60.58 | **1.092x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 4 | 36.18 | 29.56 | **1.224x** | 107.69 | 100.00 | **1.077x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 49.55 | 39.22 | **1.263x** | 141.25 | 130.62 | **1.081x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_sbm32` | 14/14 · 160/160 |
| 16 | 53.45 | 45.96 | **1.163x** | 169.90 | 159.32 | **1.066x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 53.83 | 45.65 | **1.179x** | 172.80 | 161.18 | **1.072x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 64 | 58.19 | 47.21 | **1.232x** | 189.65 | 178.65 | **1.062x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_sbm32` | 14/14 · 160/160 |
| 128 | 72.54 | 50.85 | **1.427x** | 211.77 | 190.79 | **1.110x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_atomic_sbm64` | 14/14 · 160/160 |
| 256 | 92.05 | 66.73 | **1.379x** | 243.76 | 210.43 | **1.158x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_nt_sbm128` | 14/14 · 160/160 |
| 512 | 129.72 | 101.72 | **1.275x** | 295.87 | 265.69 | **1.114x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_sbm128` | 14/14 · 160/160 |
| 1024 | 203.47 | 166.13 | **1.225x** | 438.21 | 398.65 | **1.099x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 2048 | 342.67 | 296.81 | **1.155x** | 705.51 | 666.39 | **1.059x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 4096 | 610.57 | 538.17 | **1.135x** | 1265.03 | 1208.23 | **1.047x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 8192 | 1169.64 | 1017.84 | **1.149x** | 2395.55 | 2351.66 | **1.019x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 16384 | 2293.38 | 2070.37 | **1.108x** | 4715.04 | 4576.53 | **1.030x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 4481.76 | 4042.23 | **1.109x** | 9432.10 | 8894.92 | **1.060x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### glm5_fp4 — model_dim=6144, inter_dim=256, E=257, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.10 | 3.27 | **0.950x** | 21.72 | 21.77 | **0.998x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 3.51 | 3.87 | **0.907x** | 26.47 | 26.21 | **1.010x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 4.78 | 5.00 | **0.956x** | 29.96 | 30.00 | **0.999x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 10.93 | 9.99 | **1.093x** | 38.72 | 37.03 | **1.046x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 17.32 | 15.41 | **1.124x** | 54.59 | 53.16 | **1.027x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 25.35 | 23.75 | **1.067x** | 86.39 | 85.51 | **1.010x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 35.80 | 33.09 | **1.082x** | 112.96 | 110.13 | **1.026x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 38.69 | 36.83 | **1.050x** | 124.11 | 122.64 | **1.012x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_sbm64` | 14/14 · 160/160 |
| 256 | 42.73 | 41.42 | **1.032x** | 146.65 | 145.96 | **1.005x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_sbm64` | 14/14 · 160/160 |
| 512 | 56.86 | 57.56 | **0.988x** | 183.06 | 177.65 | **1.030x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_persist_sbm16` | 14/14 · 160/160 |
| 1024 | 96.64 | 95.72 | **1.010x** | 251.60 | 236.34 | **1.065x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 2048 | 161.63 | 162.90 | **0.992x** | 347.44 | 319.24 | **1.088x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_nt_sbm128` | 14/14 · 160/160 |
| 4096 | 296.51 | 280.96 | **1.055x** | 562.02 | 525.03 | **1.071x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 539.56 | 493.67 | **1.093x** | 906.46 | 894.73 | **1.013x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 955.64 | 985.24 | **0.970x** | 1558.88 | 1562.06 | **0.998x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 1858.54 | 1946.41 | **0.955x** | 2794.84 | 2887.77 | **0.968x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |

### glm5_fp4 — model_dim=6144, inter_dim=512, E=257, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.60 | 3.87 | **0.930x** | 26.06 | 26.43 | **0.986x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 4.87 | 5.03 | **0.969x** | 30.22 | 30.05 | **1.006x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 11.47 | 10.74 | **1.067x** | 39.34 | 38.51 | **1.022x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 20.62 | 19.66 | **1.049x** | 77.43 | 76.13 | **1.017x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 16 | 34.57 | 32.56 | **1.062x** | 107.19 | 105.89 | **1.012x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 53.40 | 49.86 | **1.071x** | 164.54 | 158.89 | **1.036x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 72.75 | 68.93 | **1.056x** | 224.83 | 220.19 | **1.021x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 128 | 79.93 | 75.85 | **1.054x** | 248.20 | 243.21 | **1.021x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 85.21 | 80.24 | **1.062x** | 278.53 | 268.69 | **1.037x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 90.06 | 87.15 | **1.033x** | 291.43 | 285.60 | **1.020x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 128.12 | 116.64 | **1.098x** | 366.87 | 353.06 | **1.039x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 218.13 | 215.72 | **1.011x** | 507.04 | 471.98 | **1.074x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` | 14/14 · 160/160 |
| 4096 | 357.91 | 350.95 | **1.020x** | 777.39 | 782.96 | **0.993x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 628.34 | 596.83 | **1.053x** | 1169.88 | 1139.66 | **1.026x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1141.83 | 1181.58 | **0.966x** | 1984.30 | 2025.99 | **0.979x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 2120.04 | 2224.24 | **0.953x** | 3660.72 | 3748.94 | **0.977x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### glm5_fp4 — model_dim=6144, inter_dim=1024, E=257, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.78 | 5.33 | **1.084x** | 30.84 | 30.39 | **1.015x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 12.35 | 11.79 | **1.048x** | 40.37 | 39.64 | **1.018x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 4 | 22.05 | 21.29 | **1.036x** | 79.39 | 79.01 | **1.005x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 38.80 | 36.24 | **1.071x** | 117.74 | 115.28 | **1.021x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 16 | 68.67 | 63.92 | **1.074x** | 214.24 | 208.65 | **1.027x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 110.02 | 103.14 | **1.067x** | 333.07 | 324.98 | **1.025x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 64 | 144.66 | 135.28 | **1.069x** | 435.12 | 429.36 | **1.013x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 159.25 | 153.28 | **1.039x** | 472.47 | 465.47 | **1.015x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 167.11 | 157.59 | **1.060x** | 504.20 | 497.78 | **1.013x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 512 | 179.09 | 169.77 | **1.055x** | 520.26 | 512.37 | **1.015x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 234.68 | 193.98 | **1.210x** | 645.01 | 615.25 | **1.048x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_atomic_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 326.59 | 291.63 | **1.120x** | 817.50 | 752.87 | **1.086x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_nt_sbm128` | 14/14 · 160/160 |
| 4096 | 530.84 | 487.67 | **1.089x** | 1202.89 | 1083.72 | **1.110x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_persist_sbm64` | 14/14 · 160/160 |
| 8192 | 936.79 | 812.66 | **1.153x** | 1841.33 | 1731.37 | **1.063x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 16384 | 1592.56 | 1441.93 | **1.105x** | 3082.87 | 2908.28 | **1.060x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 3018.37 | 2721.82 | **1.109x** | 5852.26 | 5485.36 | **1.067x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### glm5_fp4 — model_dim=6144, inter_dim=2048, E=257, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.76 | 11.52 | **1.194x** | 42.32 | 39.25 | **1.078x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 2 | 25.34 | 22.05 | **1.149x** | 83.35 | 79.76 | **1.045x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 4 | 48.52 | 39.99 | **1.213x** | 144.18 | 136.59 | **1.056x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 87.74 | 73.73 | **1.190x** | 245.57 | 233.90 | **1.050x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 149.89 | 129.52 | **1.157x** | 404.06 | 295.67 | **1.367x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 229.24 | 196.06 | **1.169x** | 633.79 | 466.44 | **1.359x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 64 | 292.24 | 255.84 | **1.142x** | 843.78 | 808.07 | **1.044x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 128 | 328.38 | 291.34 | **1.127x** | 947.81 | 908.46 | **1.043x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 256 | 335.72 | 295.48 | **1.136x** | 988.73 | 950.07 | **1.041x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 512 | 363.04 | 305.88 | **1.187x** | 1033.55 | 989.81 | **1.044x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 419.70 | 324.82 | **1.292x** | 1187.14 | 1097.78 | **1.081x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 593.71 | 440.44 | **1.348x** | 1447.63 | 1264.19 | **1.145x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_nt_sbm128` | 14/14 · 160/160 |
| 4096 | 935.78 | 713.60 | **1.311x** | 2051.54 | 1834.88 | **1.118x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_atomic_persist_sbm64` | 14/14 · 160/160 |
| 8192 | 1450.38 | 1182.52 | **1.226x** | 3167.79 | 2831.25 | **1.119x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_sbm128` | 14/14 · 160/160 |
| 16384 | 2448.44 | 2091.57 | **1.171x** | 5287.79 | 4862.96 | **1.087x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 4578.93 | 3922.30 | **1.167x** | 9859.71 | 8992.99 | **1.096x** | `flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### kimik2_fp4 — model_dim=7168, inter_dim=256, E=384, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.09 | 3.29 | **0.937x** | 24.70 | 21.84 | **1.131x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 3.62 | 3.96 | **0.915x** | 29.25 | 29.12 | **1.005x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 5.57 | 5.26 | **1.060x** | 33.08 | 32.87 | **1.006x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 11.39 | 10.45 | **1.090x** | 41.10 | 39.74 | **1.034x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 19.37 | 18.23 | **1.063x** | 61.61 | 61.67 | **0.999x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 32 | 32.41 | 31.01 | **1.045x** | 105.07 | 104.73 | **1.003x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 64 | 51.85 | 49.83 | **1.040x** | 167.81 | 162.25 | **1.034x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 66.61 | 66.42 | **1.003x** | 204.95 | 197.62 | **1.037x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_persist_nt_sbm16` | 14/14 · 160/160 |
| 256 | 74.01 | 70.51 | **1.050x** | 227.09 | 221.81 | **1.024x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 512 | 82.70 | 79.42 | **1.041x** | 253.48 | 247.64 | **1.024x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 118.62 | 113.57 | **1.044x** | 304.37 | 297.38 | **1.024x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 213.96 | 188.19 | **1.137x** | 429.25 | 404.53 | **1.061x** | `flydsl_mxmoe_g2_a4w4_64x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 336.17 | 313.11 | **1.074x** | 581.87 | 564.20 | **1.031x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 593.47 | 550.12 | **1.079x** | 1031.21 | 1012.29 | **1.019x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1015.56 | 1052.87 | **0.965x** | 1706.06 | 1745.63 | **0.977x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 2030.15 | 2143.13 | **0.947x** | 3151.05 | 3224.22 | **0.977x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |

### kimik2_fp4 — model_dim=7168, inter_dim=256, E=385, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.02 | 3.27 | **0.922x** | 25.68 | 25.93 | **0.990x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 3.64 | 4.08 | **0.890x** | 29.48 | 29.52 | **0.999x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 6.20 | 5.68 | **1.091x** | 34.46 | 34.58 | **0.997x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 12.44 | 11.31 | **1.101x** | 43.22 | 42.21 | **1.024x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 21.81 | 20.82 | **1.048x** | 69.71 | 68.68 | **1.015x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 34.42 | 32.41 | **1.062x** | 109.16 | 107.16 | **1.019x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 64 | 55.45 | 50.98 | **1.088x** | 175.14 | 167.96 | **1.043x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 66.75 | 63.54 | **1.051x** | 205.89 | 203.01 | **1.014x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 74.41 | 72.60 | **1.025x** | 231.81 | 226.03 | **1.026x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 512 | 88.83 | 80.84 | **1.099x** | 271.57 | 263.67 | **1.030x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 131.56 | 125.38 | **1.049x** | 318.88 | 312.81 | **1.019x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 220.45 | 199.02 | **1.108x** | 446.38 | 428.50 | **1.042x** | `flydsl_mxmoe_g2_a4w4_64x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 341.55 | 336.55 | **1.015x** | 603.49 | 603.98 | **0.999x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_sbm128` | 14/14 · 160/160 |
| 8192 | 630.28 | 606.60 | **1.039x** | 1082.95 | 1084.86 | **0.998x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1123.80 | 1142.20 | **0.984x** | 1926.18 | 1937.60 | **0.994x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 2154.60 | 2273.73 | **0.948x** | 3325.84 | 3498.87 | **0.951x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |

### kimik2_fp4 — model_dim=7168, inter_dim=512, E=384, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.90 | 4.52 | **0.864x** | 29.28 | 29.81 | **0.982x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 6.07 | 5.75 | **1.056x** | 33.12 | 33.40 | **0.991x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 12.01 | 11.70 | **1.026x** | 42.04 | 42.19 | **0.997x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 8 | 22.33 | 21.64 | **1.032x** | 70.57 | 70.53 | **1.001x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 16 | 37.03 | 35.63 | **1.039x** | 115.91 | 116.26 | **0.997x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 32 | 68.35 | 64.04 | **1.067x** | 225.79 | 221.27 | **1.020x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 100.66 | 95.08 | **1.059x** | 312.85 | 305.54 | **1.024x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 128 | 124.70 | 118.53 | **1.052x** | 392.04 | 381.95 | **1.026x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 137.27 | 135.73 | **1.011x** | 430.61 | 419.09 | **1.028x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_persist_nt_sbm16` | 14/14 · 160/160 |
| 512 | 146.57 | 139.00 | **1.054x** | 458.21 | 451.12 | **1.016x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 160.42 | 154.62 | **1.038x** | 491.34 | 482.19 | **1.019x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 261.89 | 246.31 | **1.063x** | 636.97 | 622.47 | **1.023x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 414.87 | 397.31 | **1.044x** | 845.08 | 825.76 | **1.023x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` | 14/14 · 160/160 |
| 8192 | 720.24 | 666.87 | **1.080x** | 1502.90 | 1462.41 | **1.028x** | `flydsl_mxmoe_g2_a4w4_64x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1151.85 | 1213.33 | **0.949x** | 2131.98 | 2197.07 | **0.970x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 2273.40 | 2349.37 | **0.968x** | 3965.66 | 4080.62 | **0.972x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### kimik2_fp4 — model_dim=7168, inter_dim=512, E=385, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.09 | 4.43 | **0.924x** | 30.03 | 30.17 | **0.995x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 6.56 | 5.81 | **1.129x** | 34.79 | 34.70 | **1.003x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 13.27 | 12.76 | **1.039x** | 44.56 | 44.40 | **1.004x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 23.23 | 22.92 | **1.014x** | 89.83 | 89.48 | **1.004x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 16 | 41.83 | 40.33 | **1.037x** | 132.86 | 132.46 | **1.003x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 32 | 71.64 | 67.89 | **1.055x** | 231.27 | 227.41 | **1.017x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 64 | 108.21 | 105.80 | **1.023x** | 332.74 | 329.64 | **1.009x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 131.77 | 128.84 | **1.023x** | 404.94 | 396.81 | **1.020x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 139.96 | 139.18 | **1.006x** | 432.47 | 426.82 | **1.013x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 152.32 | 149.98 | **1.016x** | 470.59 | 470.68 | **1.000x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 170.81 | 167.49 | **1.020x** | 501.74 | 495.76 | **1.012x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 281.52 | 264.38 | **1.065x** | 655.67 | 638.56 | **1.027x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 439.05 | 433.08 | **1.014x** | 882.45 | 871.78 | **1.012x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` | 14/14 · 160/160 |
| 8192 | 731.28 | 714.26 | **1.024x** | 1573.17 | 1533.22 | **1.026x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1342.29 | 1386.93 | **0.968x** | 2404.54 | 2434.89 | **0.988x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 2460.02 | 2569.44 | **0.957x** | 4269.91 | 4434.11 | **0.963x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### kimik2_fp4 — model_dim=7168, inter_dim=1024, E=385, topk=9

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.75 | 6.24 | **1.081x** | 34.47 | 34.14 | **1.010x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 13.30 | 12.89 | **1.031x** | 45.57 | 44.62 | **1.021x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 4 | 24.73 | 23.90 | **1.035x** | 92.75 | 91.14 | **1.018x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 47.90 | 44.96 | **1.065x** | 165.23 | 162.95 | **1.014x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 16 | 87.71 | 82.72 | **1.060x** | 270.31 | 264.19 | **1.023x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 136.85 | 128.62 | **1.064x** | 431.35 | 421.24 | **1.024x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 64 | 206.35 | 194.62 | **1.060x** | 646.11 | 627.32 | **1.030x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 257.34 | 244.13 | **1.054x** | 812.52 | 790.08 | **1.028x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 275.45 | 262.66 | **1.049x** | 843.08 | 845.53 | **0.997x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 278.49 | 272.92 | **1.020x** | 901.18 | 902.04 | **0.999x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 336.17 | 292.02 | **1.151x** | 974.46 | 951.18 | **1.024x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 454.83 | 343.47 | **1.324x** | 1173.82 | 1080.28 | **1.087x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 630.94 | 574.68 | **1.098x** | 1435.61 | 1357.81 | **1.057x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_nt_sbm128` | 14/14 · 160/160 |
| 8192 | 1081.47 | 1009.54 | **1.071x** | 2444.66 | 2181.37 | **1.121x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 1880.56 | 1709.02 | **1.100x** | 3771.25 | 3599.63 | **1.048x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 3507.42 | 3227.47 | **1.087x** | 6829.09 | 6555.16 | **1.042x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### minimax_m25_fp4 — model_dim=3072, inter_dim=256, E=256, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.63 | 3.64 | **0.997x** | 15.45 | 15.58 | **0.992x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 3.62 | 3.88 | **0.933x** | 16.27 | 18.24 | **0.892x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 4.23 | 4.49 | **0.943x** | 19.72 | 19.91 | **0.990x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 4.76 | 5.22 | **0.912x** | 21.98 | 22.33 | **0.984x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 16 | 9.15 | 8.58 | **1.066x** | 31.44 | 30.19 | **1.041x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 32 | 13.95 | 12.40 | **1.124x** | 48.06 | 47.15 | **1.019x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 18.51 | 16.96 | **1.091x** | 60.05 | 59.40 | **1.011x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 20.86 | 19.26 | **1.083x** | 68.00 | 66.78 | **1.018x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 256 | 22.87 | 20.19 | **1.133x** | 71.89 | 71.31 | **1.008x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 32.70 | 27.79 | **1.177x** | 97.77 | 93.23 | **1.049x** | `flydsl_mxmoe_g2_a4w4_64x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 49.85 | 40.77 | **1.223x** | 123.31 | 119.46 | **1.032x** | `flydsl_mxmoe_g2_a4w4_64x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_sbm64` | 14/14 · 160/160 |
| 2048 | 77.53 | 65.80 | **1.178x** | 172.36 | 158.86 | **1.085x** | `flydsl_mxmoe_g2_a4w4_64x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x256_reduce_sbm128` | 14/14 · 160/160 |
| 4096 | 129.21 | 134.26 | **0.962x** | 250.45 | 243.22 | **1.030x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 248.45 | 234.73 | **1.058x** | 415.80 | 416.40 | **0.999x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 451.93 | 451.77 | **1.000x** | 754.02 | 739.04 | **1.020x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 866.25 | 895.77 | **0.967x** | 1351.78 | 1381.23 | **0.979x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` | 14/14 · 160/160 |

### minimax_m25_fp4 — model_dim=3072, inter_dim=384, E=256, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 4 | — | 5.12 | — | — | 23.50 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 0/14 · 2/80 |
| 8 | — | 7.25 | — | — | 27.64 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 0/14 · 20/80 |
| 16 | — | 12.41 | — | — | 44.90 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 0/14 · 19/80 |
| 32 | — | 19.00 | — | — | 61.54 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` | 0/14 · 14/80 |
| 64 | — | 24.09 | — | — | 83.14 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_nt_sbm32` | 0/14 · 17/80 |
| 128 | — | 28.78 | — | — | 125.99 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_sbm64` | 0/14 · 15/80 |
| 256 | — | 30.34 | — | — | 109.48 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_sbm64` | 0/14 · 17/80 |
| 512 | — | 37.50 | — | — | 138.31 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_sbm64` | 0/14 · 25/80 |
| 1024 | — | 48.46 | — | — | 153.14 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_sbm64` | 0/14 · 24/80 |
| 2048 | — | 86.74 | — | — | 201.13 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 0/14 · 78/80 |
| 4096 | — | 156.02 | — | — | 307.55 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 0/14 · 76/80 |
| 8192 | — | 272.64 | — | — | 509.49 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 0/14 · 75/80 |
| 16384 | — | 528.90 | — | — | 961.67 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 0/14 · 74/80 |
| 32768 | — | 1034.05 | — | — | 1810.77 | — | `nan` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` | 0/14 · 74/80 |

### minimax_m25_fp4 — model_dim=3072, inter_dim=768, E=256, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.54 | 4.14 | **1.095x** | 18.45 | 18.38 | **1.004x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 4.61 | 4.79 | **0.963x** | 21.11 | 21.00 | **1.005x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 7.89 | 7.76 | **1.016x** | 28.34 | 27.42 | **1.033x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 14.75 | 14.01 | **1.052x** | 49.11 | 48.06 | **1.022x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 23.78 | 23.14 | **1.028x** | 76.18 | 75.10 | **1.014x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 32 | 36.85 | 34.78 | **1.060x** | 114.67 | 112.98 | **1.015x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32` | 14/14 · 160/160 |
| 64 | 52.56 | 48.98 | **1.073x** | 163.13 | 159.84 | **1.021x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 128 | 60.24 | 54.28 | **1.110x** | 181.76 | 173.01 | **1.051x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 62.54 | 60.11 | **1.040x** | 193.35 | 183.87 | **1.052x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 512 | 67.91 | 61.65 | **1.101x** | 207.79 | 202.58 | **1.026x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 94.21 | 76.84 | **1.226x** | 268.20 | 246.16 | **1.089x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 132.59 | 114.21 | **1.161x** | 350.12 | 310.09 | **1.129x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_atomic_sbm64` | 14/14 · 160/160 |
| 4096 | 211.11 | 195.59 | **1.079x** | 458.25 | 448.35 | **1.022x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 8192 | 346.75 | 326.43 | **1.062x** | 757.36 | 741.17 | **1.022x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 16384 | 600.06 | 595.78 | **1.007x** | 1313.03 | 1312.08 | **1.001x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 1143.40 | 1148.34 | **0.996x** | 2286.53 | 2299.34 | **0.994x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### minimax_m25_fp4 — model_dim=3072, inter_dim=1536, E=256, topk=8

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.96 | 4.74 | **1.258x** | 22.73 | 21.33 | **1.065x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 9.62 | 8.59 | **1.120x** | 30.43 | 28.77 | **1.058x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 4 | 16.78 | 14.35 | **1.169x** | 50.35 | 48.42 | **1.040x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 27.46 | 25.03 | **1.097x** | 84.12 | 81.89 | **1.027x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 16 | 47.84 | 45.62 | **1.049x** | 150.28 | 143.74 | **1.046x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 75.77 | 72.08 | **1.051x** | 231.69 | 226.02 | **1.025x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 64 | 104.55 | 99.72 | **1.048x** | 316.40 | 309.78 | **1.021x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 128 | 119.00 | 111.85 | **1.064x** | 354.75 | 347.14 | **1.022x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_atomic_nt_sbm32` | 14/14 · 160/160 |
| 256 | 123.29 | 117.77 | **1.047x** | 370.92 | 336.28 | **1.103x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 512 | 126.72 | 117.07 | **1.082x** | 387.23 | 364.57 | **1.062x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 173.48 | 132.39 | **1.310x** | 482.83 | 445.79 | **1.083x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_atomic_nt_sbm64` | 14/14 · 160/160 |
| 2048 | 247.76 | 181.32 | **1.366x** | 590.76 | 527.16 | **1.121x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_nt_sbm128` | 14/14 · 160/160 |
| 4096 | 333.95 | 276.91 | **1.206x** | 780.08 | 730.74 | **1.067x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_sbm128` | 14/14 · 160/160 |
| 8192 | 511.27 | 452.03 | **1.131x** | 1189.77 | 1116.10 | **1.066x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 16384 | 877.07 | 777.18 | **1.129x** | 2006.43 | 1924.80 | **1.042x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |
| 32768 | 1680.78 | 1439.81 | **1.167x** | 3744.48 | 3524.33 | **1.062x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128` | 14/14 · 160/160 |

### qwen3_5_397b_fp4 — model_dim=4096, inter_dim=256, E=512, topk=10

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.16 | 3.22 | **0.980x** | 16.38 | 16.57 | **0.989x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 3.43 | 3.76 | **0.912x** | 21.99 | 21.59 | **1.019x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 4.28 | 4.48 | **0.955x** | 23.97 | 24.10 | **0.995x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x128_atomic_sbm16` | 14/14 · 160/160 |
| 8 | 8.15 | 8.04 | **1.014x** | 31.22 | 30.37 | **1.028x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 15.76 | 14.42 | **1.093x** | 57.06 | 56.44 | **1.011x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 32 | 25.76 | 23.77 | **1.083x** | 81.73 | 79.87 | **1.023x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 36.24 | 34.66 | **1.046x** | 116.98 | 115.59 | **1.012x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 128 | 47.80 | 45.82 | **1.043x** | 155.80 | 151.62 | **1.028x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16` | 14/14 · 160/160 |
| 256 | 56.74 | 53.55 | **1.060x** | 177.07 | 168.44 | **1.051x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 62.49 | 60.78 | **1.028x** | 197.80 | 192.46 | **1.028x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 1024 | 82.93 | 81.69 | **1.015x** | 235.10 | 229.48 | **1.024x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 150.64 | 135.56 | **1.111x** | 334.43 | 311.41 | **1.074x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_reduce_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 238.56 | 232.25 | **1.027x** | 459.12 | 431.20 | **1.065x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x128_reduce_persist_sbm32` | 14/14 · 160/160 |
| 8192 | 429.97 | 391.19 | **1.099x** | 854.50 | 783.54 | **1.091x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 820.06 | 776.84 | **1.056x** | 1339.32 | 1339.19 | **1.000x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x128_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 1615.86 | 1497.67 | **1.079x** | 2499.48 | 2336.27 | **1.070x** | `flydsl_mxmoe_g2_a4w4_128x256x256_f4out` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_reduce_persist_sbm128` | 14/14 · 160/160 |

### qwen3_5_397b_fp4 — model_dim=4096, inter_dim=512, E=512, topk=10

| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.15 | 4.26 | **0.974x** | 21.49 | 22.14 | **0.971x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x256x256_atomic_sbm16` | 14/14 · 160/160 |
| 2 | 5.50 | 4.97 | **1.106x** | 26.04 | 25.14 | **1.036x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_sbm16` | 14/14 · 160/160 |
| 4 | 9.24 | 8.93 | **1.034x** | 32.65 | 32.44 | **1.006x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 8 | 16.41 | 15.72 | **1.044x** | 58.48 | 58.62 | **0.998x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 16 | 28.44 | 27.19 | **1.046x** | 96.06 | 94.81 | **1.013x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 32 | 50.70 | 46.93 | **1.080x** | 158.41 | 153.21 | **1.034x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 64 | 77.79 | 71.28 | **1.091x** | 238.57 | 230.35 | **1.036x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 128 | 97.21 | 90.90 | **1.069x** | 303.30 | 297.05 | **1.021x** | `flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 256 | 107.06 | 100.84 | **1.062x** | 333.72 | 325.50 | **1.025x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 512 | 111.86 | 107.44 | **1.041x** | 354.38 | 345.84 | **1.025x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16` | 14/14 · 160/160 |
| 1024 | 121.93 | 117.09 | **1.041x** | 370.08 | 365.70 | **1.012x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic_nt` | `flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32` | 14/14 · 160/160 |
| 2048 | 186.91 | 173.41 | **1.078x** | 480.95 | 467.50 | **1.029x** | `flydsl_mxmoe_g2_a4w4_32x256x256_atomic` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x128_atomic_nt_sbm64` | 14/14 · 160/160 |
| 4096 | 309.71 | 304.70 | **1.016x** | 660.20 | 632.15 | **1.044x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_nt_sbm128` | 14/14 · 160/160 |
| 8192 | 520.38 | 481.21 | **1.081x** | 1095.05 | 1071.20 | **1.022x** | `flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 16384 | 916.16 | 889.92 | **1.030x** | 1703.13 | 1642.51 | **1.037x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |
| 32768 | 1718.08 | 1705.72 | **1.007x** | 3060.70 | 3063.42 | **0.999x** | `flydsl_mxmoe_g2_a4w4_128x256x256` | `flydsl_moe2_layout_afp4_wfp4_bf16_t64x256x256_reduce_sbm64` | 14/14 · 160/160 |

## Where the shipped tuned CSVs disagree with this measurement

Of the 286 head-to-head shapes, the tuned CSVs name a v1-or-v2 GEMM2 for some; **55** of those name the family this benchmark measured as slower (GEMM2-only). Stage1 choice and tuner vintage differ, so treat this as a list worth re-tuning, not a bug list.

| model | shape | token | CSV picks | measured faster | g2 speedup (v1/v2) |
|---|---|---|---|---|---|
| glm5_fp4 | 6144x512x257x9 | 1 | v2 | **v1** | 0.930x |
| glm5_fp4 | 6144x512x257x9 | 4 | v1 | **v2** | 1.067x |
| kimik2_fp4 | 7168x256x385x9 | 4 | v1 | **v2** | 1.091x |
| kimik2_fp4 | 7168x256x385x9 | 8 | v1 | **v2** | 1.101x |
| kimik2_fp4 | 7168x256x385x9 | 16 | v1 | **v2** | 1.048x |
| kimik2_fp4 | 7168x256x385x9 | 32 | v1 | **v2** | 1.062x |
| kimik2_fp4 | 7168x256x385x9 | 64 | v1 | **v2** | 1.088x |
| kimik2_fp4 | 7168x256x385x9 | 128 | v1 | **v2** | 1.051x |
| kimik2_fp4 | 7168x256x385x9 | 256 | v1 | **v2** | 1.025x |
| kimik2_fp4 | 7168x256x385x9 | 512 | v1 | **v2** | 1.099x |
| kimik2_fp4 | 7168x256x385x9 | 1024 | v1 | **v2** | 1.049x |
| kimik2_fp4 | 7168x256x385x9 | 2048 | v1 | **v2** | 1.108x |
| kimik2_fp4 | 7168x256x385x9 | 4096 | v1 | **v2** | 1.015x |
| kimik2_fp4 | 7168x256x385x9 | 8192 | v1 | **v2** | 1.039x |
| kimik2_fp4 | 7168x512x384x8 | 2 | v1 | **v2** | 1.056x |
| kimik2_fp4 | 7168x512x384x8 | 4 | v1 | **v2** | 1.026x |
| kimik2_fp4 | 7168x512x384x8 | 8 | v1 | **v2** | 1.032x |
| kimik2_fp4 | 7168x512x384x8 | 16 | v1 | **v2** | 1.039x |
| kimik2_fp4 | 7168x512x384x8 | 32 | v1 | **v2** | 1.067x |
| kimik2_fp4 | 7168x512x384x8 | 64 | v1 | **v2** | 1.059x |
| kimik2_fp4 | 7168x512x384x8 | 128 | v1 | **v2** | 1.052x |
| kimik2_fp4 | 7168x512x384x8 | 256 | v1 | **v2** | 1.011x |
| kimik2_fp4 | 7168x512x385x9 | 2 | v1 | **v2** | 1.129x |
| kimik2_fp4 | 7168x512x385x9 | 4 | v1 | **v2** | 1.039x |
| kimik2_fp4 | 7168x512x385x9 | 8 | v1 | **v2** | 1.014x |
| kimik2_fp4 | 7168x512x385x9 | 16 | v1 | **v2** | 1.037x |
| kimik2_fp4 | 7168x512x385x9 | 32 | v1 | **v2** | 1.055x |
| kimik2_fp4 | 7168x512x385x9 | 64 | v1 | **v2** | 1.023x |
| kimik2_fp4 | 7168x512x385x9 | 128 | v1 | **v2** | 1.023x |
| kimik2_fp4 | 7168x512x385x9 | 256 | v1 | **v2** | 1.006x |
| kimik2_fp4 | 7168x512x385x9 | 512 | v1 | **v2** | 1.016x |
| kimik2_fp4 | 7168x512x385x9 | 1024 | v1 | **v2** | 1.020x |
| kimik2_fp4 | 7168x512x385x9 | 2048 | v1 | **v2** | 1.065x |
| kimik2_fp4 | 7168x512x385x9 | 4096 | v1 | **v2** | 1.014x |
| kimik2_fp4 | 7168x512x385x9 | 8192 | v1 | **v2** | 1.024x |
| kimik2_fp4 | 7168x1024x385x9 | 1 | v1 | **v2** | 1.081x |
| kimik2_fp4 | 7168x1024x385x9 | 2 | v1 | **v2** | 1.031x |
| kimik2_fp4 | 7168x1024x385x9 | 4 | v1 | **v2** | 1.035x |
| kimik2_fp4 | 7168x1024x385x9 | 8 | v1 | **v2** | 1.065x |
| kimik2_fp4 | 7168x1024x385x9 | 16 | v1 | **v2** | 1.060x |

_(15 more rows in `summary.csv`.)_

## Measurement noise

One family was re-measured end-to-end in an independent process. Same kernels, same tuning, different run:

| family | token | v1 g2 drift | v2 g2 drift | speedup run 1 | speedup run 2 |
|---|---|---|---|---|---|
| 6144x512x257x9 | 1 | +13.7% | +10.7% | 0.930x | 0.956x |
| 6144x512x257x9 | 8 | -1.5% | +2.0% | 1.049x | 1.013x |
| 6144x512x257x9 | 64 | -0.7% | -1.7% | 1.056x | 1.066x |
| 6144x512x257x9 | 512 | +2.6% | +3.4% | 1.033x | 1.025x |
| 6144x512x257x9 | 4096 | +1.8% | -0.9% | 1.020x | 1.047x |

Worst single-point drift **13.7%**, worst speedup swing **0.036x**. The drift is concentrated at **token=1**, where the kernel runs for only a few µs and launch overhead dominates. Read per-shape speedups within roughly ±3% as noise (worse at token≤2); the geomeans over hundreds of shapes are far tighter than any single row.

## Coverage and caveats

- **124 shapes could not be measured at all** — neither family ran. `moe_sorting(output_aux=True)` dispatches to an AOT-codegen'd aux kernel keyed on `(NE, TOPK[, H])`, and these combinations are absent from `SHAPES` in [gen_instances.py:19](csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py#L19), so every candidate fails with `no codegen'd instance for shape key 'aux_sort3s_NE<ne>_TOPK<topk>_MB<mb>'`. This is a shared-prerequisite gap, not a v1-vs-v2 difference.

| model | NE | topk | H | shapes blocked |
|---|---|---|---|---|
| dsv3_fp4 | 64 | 8 | 7168 | 1 |
| dsv3_fp4 | 256 | 8 | 7168 | 32 |
| gptoss_fp4 | 128 | 4 | 3072 | 24 |
| kimik3_a4w4 | 896 | 16 | 3584 | 17 |
| minimax_m25_fp4 | 256 | 8 | 3072 | 2 |
| minimax_m3_fp4 | 129 | 5 | 6144 | 32 |
| qwen3_5_397b_fp4 | 513 | 11 | 4096 | 16 |

  To unblock: add those `(NE, H, D_INTER, TOPK)` tuples to `SHAPES` and rebuild. Note that on this machine a rebuild of `module_moe_mxfp4_aux` alone is not enough — `torch.ops.aiter.mxfp4_moe_sort` is already registered when `aiter` is imported, so `compile_ops` short-circuits and never rebuilds; the prebuilt `module_aiter_core.so` has to be rebuilt too.
- **14 shapes have no working v1 kernel** (3072x384). v1's tile is fixed at `<BM>x256x256`, so `_assert_supported` rejects `inter_dim % 256 != 0` ([mxfp4_gemm2_kernels.py:82](aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L82)); v2 tunes `BK ∈ {128, 256}`.
- **73 shapes** are declared Swiglu/Situv2 but measured against a silu reference: the mxmoe GEMM1 calls `_silu_mul_batch` unconditionally ([mxfp4_gemm1.py:704](aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L704)). Timing is unaffected; only the correctness gate's reference changed.
- **Absolute µs here do not match the shipped CSVs at large token counts.** For glm5 `6144x512`, re-running the exact `flydsl_mxmoe_g1/g2` pair the CSV names reproduces it within ~7% at token 2-4, but is ~35% slower at token 16384/32768 (1984 µs vs the CSV's 1440 µs). That is not 8-GPU contention: re-measuring token=16384 alone on an otherwise idle machine gives 1922 µs, within 3% of the sweep. The CSV rows were tuned on an older build, so this is a cross-vintage difference in the absolute level. **It does not affect the v1-vs-v2 conclusion** — both families are measured in the same process, on the same data, back to back.
- **The search budgets are not equal**: v1 exposes ~14 candidates per shape, v2 ~160 (v1's BN/BK are fixed; v2 sweeps `tile_n x tile_k x epilog x nt x persist`). That is a real property of the two designs, not a harness artifact, but v2 does get more shots at a good config. The `cands` column shows ok/total per family.
- Under the atomic epilog the stage2-only timing loop re-accumulates into the same output buffer, so those iterations are numerically meaningless. The work per iteration is identical, so `us_g2` is still a valid timing.
- 19562 of 69856 candidate runs failed (rejected by the cosine gate, unsupported-variant errors, or timeouts). Breakdown in `summary.csv` / the raw CSVs.

Raw per-candidate data: `bench_out/flydsl_mxmoe_v1_v2/raw_*.csv`, per-shape rollup: `bench_out/flydsl_mxmoe_v1_v2/summary.csv`.
