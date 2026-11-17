# GEMM2: WT and grouped M4 + WT selection at T >= 1024

Measured 2026-11-17 on gfx950. AITER source commit: `a62dfc291449074236b2ea7d83e4e5f9d0f5671f`.

The tracked records are [per-case measurements](gemm2_wt_group_m4_20261117/summary.json), [selected policies and full tuning specifications](gemm2_wt_group_m4_20261117/selection.json), and [source, compiler, and verification provenance](gemm2_wt_group_m4_20261117/provenance.json). These JSON files contain historical benchmark data. Paths inside them identify the local sealed archive; they are not report dependencies.

Applied confirmed policies to all 36 benchmark cases: **5 grouped M4 + WT, 3 WT, and 28 baseline**.

The two candidates are `.wt` alone and `TILE_SCHED=1`, `GROUP_M=4`, `.wt` together. Each starts from the same accepted baseline and changes only those fields after the full launch specification is selected. The half-height tile receives the same changes when present. The A4W4 dsv4-flash T1024 baseline already uses grouped M4; selecting WT preserves that schedule.

A candidate must reduce total Gluon latency by at least 2% in at least 8/9 rounds in both screening and a fresh-process confirmation. When both candidates pass, choose grouped M4 only if it also has a confirmed 2% benefit over WT alone. Otherwise choose WT. Cases that fail screening retain baseline.

This report records the selected policies from the local benchmark comparison runner. The frozen selection file is benchmark evidence. Production dispatch is unchanged.

## Current gaps after applying the selections

Positive means Gluon has higher latency. Each gap is the median of nine paired ratios: `100 * ((Gluon GEMM + reduction) / FlyDSL stage2 total - 1)`. Preparation and FlyDSL output reset are excluded. FlyDSL scatter reduction is included for the two configurations that use it. `†` marks a same-buffer matched fixture; other cells retain the original backend fixtures. `*` marks an invalid FlyDSL numerical comparison.

| Precision | Model | T=1024 | T=4096 | T=16384 |
| --- | --- | ---: | ---: | ---: |
| A4W4 | dsv4-flash | +6.55%† | +3.87% | -3.58% |
| A4W4 | dsv4-pro | -4.64% | +4.98% | +1.22% |
| A4W4 | glm52-base | +16.75%† | +5.47% | -6.54% |
| A4W4 | minimax-m3 | +4.77% | +13.82% | +0.17% |
| A8W4 | dsv4-flash | +5.89%† | +3.58% | +2.11%† |
| A8W4 | dsv4-pro | +2.19% | +2.24% | -1.30% |
| A8W4 | glm52-base | +8.73%† | +1.91% | +5.92% |
| A8W4 | minimax-m3 | +4.87% | -0.06% | +2.33% |
| A8W8 | dsv4-flash | +3.01% | +3.28% | -0.37% |
| A8W8 | dsv4-pro | -3.96%* | -21.79%* | -12.92%* |
| A8W8 | glm52-base | +10.04% | +4.18% | -8.29% |
| A8W8 | minimax-m3 | +8.64% | +6.66% | -5.43% |

## Candidate effects and applied policies

Gain is `100 * (1 - candidate total / baseline total)`, measured in the same round. Positive means lower Gluon latency. Screening and confirmation have nine rounds each. Both candidates use fixed buffers. Confirmation cells list WT / grouped M4 + WT; a dash means no confirmation was needed because neither candidate passed screening.

| Case | WT screen gain | M4 + WT screen gain | Confirmation gains | Applied policy |
| --- | ---: | ---: | ---: | --- |
| A4W4 dsv4-flash T=1024 | +4.82% | +5.05% | +5.12% / +4.99% | .wt |
| A4W4 dsv4-flash T=4096 | +1.61% | +1.63% | — | baseline |
| A4W4 dsv4-flash T=16384 | -2.72% | -15.86% | — | baseline |
| A4W4 dsv4-pro T=1024 | +0.24% | +0.64% | +2.67% / +3.12% | baseline |
| A4W4 dsv4-pro T=4096 | -0.17% | -2.48% | — | baseline |
| A4W4 dsv4-pro T=16384 | -0.58% | -3.82% | — | baseline |
| A4W4 glm52-base T=1024 | -0.10% | +0.56% | +6.13% / +7.29% | baseline |
| A4W4 glm52-base T=4096 | +1.75% | +0.99% | — | baseline |
| A4W4 glm52-base T=16384 | -17.00% | -20.18% | — | baseline |
| A4W4 minimax-m3 T=1024 | +1.55% | +2.45% | +1.49% / +2.33% | M4 + .wt |
| A4W4 minimax-m3 T=4096 | -0.09% | +0.18% | +2.39% / +2.46% | baseline |
| A4W4 minimax-m3 T=16384 | -1.76% | -9.48% | — | baseline |
| A8W4 dsv4-flash T=1024 | +5.05% | +8.80% | +4.11% / +7.98% | M4 + .wt |
| A8W4 dsv4-flash T=4096 | +0.07% | +0.32% | — | baseline |
| A8W4 dsv4-flash T=16384 | -0.00% | -11.00% | — | baseline |
| A8W4 dsv4-pro T=1024 | -0.96% | +0.73% | — | baseline |
| A8W4 dsv4-pro T=4096 | -0.66% | -1.77% | — | baseline |
| A8W4 dsv4-pro T=16384 | -0.46% | -1.55% | — | baseline |
| A8W4 glm52-base T=1024 | +4.58% | +6.14% | +5.37% / +7.57% | .wt |
| A8W4 glm52-base T=4096 | +0.40% | +1.43% | — | baseline |
| A8W4 glm52-base T=16384 | -2.30% | -7.96% | — | baseline |
| A8W4 minimax-m3 T=1024 | +1.35% | +2.31% | +1.01% / +2.33% | M4 + .wt |
| A8W4 minimax-m3 T=4096 | +1.12% | -0.29% | — | baseline |
| A8W4 minimax-m3 T=16384 | -0.91% | -4.33% | — | baseline |
| A8W8 dsv4-flash T=1024 | +0.72% | +3.12% | +2.41% / +3.67% | M4 + .wt |
| A8W8 dsv4-flash T=4096 | +0.63% | +1.43% | — | baseline |
| A8W8 dsv4-flash T=16384 | -0.97% | -6.68% | — | baseline |
| A8W8 dsv4-pro T=1024 | -0.08% | +0.77% | — | baseline |
| A8W8 dsv4-pro T=4096 | -1.92% | -1.73% | — | baseline |
| A8W8 dsv4-pro T=16384 | -0.57% | -7.10% | — | baseline |
| A8W8 glm52-base T=1024 | +2.32% | +2.86% | +2.77% / +3.00% | .wt |
| A8W8 glm52-base T=4096 | -0.46% | +3.71% | -1.21% / +1.75% | baseline |
| A8W8 glm52-base T=16384 | -1.46% | -11.28% | — | baseline |
| A8W8 minimax-m3 T=1024 | +1.58% | +2.73% | +1.48% / +2.34% | M4 + .wt |
| A8W8 minimax-m3 T=4096 | -0.91% | +1.39% | — | baseline |
| A8W8 minimax-m3 T=16384 | -0.63% | -14.31% | — | baseline |

## Applied component times

All times are microseconds. Each row uses the same round whose selected-policy total gap is the median of the nine paired ratios. Exact tuning, baseline times, and paired changes are in the tracked JSON records.

| Case | Policy | Gluon GEMM | Gluon reduce | Gluon total | FlyDSL GEMM | FlyDSL reduce | FlyDSL total | Total gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A4W4 dsv4-flash T=1024 | .wt | 226.934 | 10.829 | 237.762 | 223.140 | 0.000 | 223.140 | +6.55% |
| A4W4 dsv4-flash T=4096 | baseline | 357.642 | 41.814 | 399.456 | 384.565 | 0.000 | 384.565 | +3.87% |
| A4W4 dsv4-flash T=16384 | baseline | 843.483 | 186.651 | 1030.134 | 1068.434 | 0.000 | 1068.434 | -3.58% |
| A4W4 dsv4-pro T=1024 | baseline | 823.415 | 24.412 | 847.827 | 889.039 | 0.000 | 889.039 | -4.64% |
| A4W4 dsv4-pro T=4096 | baseline | 948.114 | 86.583 | 1034.697 | 985.626 | 0.000 | 985.626 | +4.98% |
| A4W4 dsv4-pro T=16384 | baseline | 2109.453 | 319.991 | 2429.445 | 2400.275 | 0.000 | 2400.275 | +1.22% |
| A4W4 glm52-base T=1024 | baseline | 378.021 | 24.189 | 402.210 | 344.516 | 0.000 | 344.516 | +16.75% |
| A4W4 glm52-base T=4096 | baseline | 545.633 | 80.810 | 626.443 | 593.981 | 0.000 | 593.981 | +5.47% |
| A4W4 glm52-base T=16384 | baseline | 1533.763 | 369.712 | 1903.475 | 2036.624 | 0.000 | 2036.624 | -6.54% |
| A4W4 minimax-m3 T=1024 | M4 + .wt | 254.610 | 11.633 | 266.242 | 254.119 | 0.000 | 254.119 | +4.77% |
| A4W4 minimax-m3 T=4096 | baseline | 394.535 | 42.254 | 436.789 | 383.754 | 0.000 | 383.754 | +13.82% |
| A4W4 minimax-m3 T=16384 | baseline | 1110.049 | 192.601 | 1302.649 | 1300.404 | 0.000 | 1300.404 | +0.17% |
| A8W4 dsv4-flash T=1024 | M4 + .wt | 231.529 | 11.233 | 242.762 | 229.256 | 0.000 | 229.256 | +5.89% |
| A8W4 dsv4-flash T=4096 | baseline | 433.291 | 41.201 | 474.493 | 458.079 | 0.000 | 458.079 | +3.58% |
| A8W4 dsv4-flash T=16384 | baseline | 1027.818 | 187.422 | 1215.240 | 1190.083 | 0.000 | 1190.083 | +2.11% |
| A8W4 dsv4-pro T=1024 | baseline | 891.588 | 24.888 | 916.475 | 896.826 | 0.000 | 896.826 | +2.19% |
| A8W4 dsv4-pro T=4096 | baseline | 1017.027 | 88.488 | 1105.515 | 1081.320 | 0.000 | 1081.320 | +2.24% |
| A8W4 dsv4-pro T=16384 | baseline | 2654.794 | 321.859 | 2976.653 | 3015.953 | 0.000 | 3015.953 | -1.30% |
| A8W4 glm52-base T=1024 | .wt | 356.659 | 20.441 | 377.100 | 346.821 | 0.000 | 346.821 | +8.73% |
| A8W4 glm52-base T=4096 | baseline | 669.672 | 79.614 | 749.287 | 735.273 | 0.000 | 735.273 | +1.91% |
| A8W4 glm52-base T=16384 | baseline | 2104.070 | 370.800 | 2474.870 | 2336.445 | 0.000 | 2336.445 | +5.92% |
| A8W4 minimax-m3 T=1024 | M4 + .wt | 253.176 | 12.210 | 265.386 | 253.064 | 0.000 | 253.064 | +4.87% |
| A8W4 minimax-m3 T=4096 | baseline | 475.786 | 43.292 | 519.078 | 475.650 | 43.721 | 519.371 | -0.06% |
| A8W4 minimax-m3 T=16384 | baseline | 1451.016 | 194.375 | 1645.392 | 1607.994 | 0.000 | 1607.994 | +2.33% |
| A8W8 dsv4-flash T=1024 | M4 + .wt | 426.835 | 11.900 | 438.735 | 425.917 | 0.000 | 425.917 | +3.01% |
| A8W8 dsv4-flash T=4096 | baseline | 582.520 | 48.957 | 631.477 | 611.398 | 0.000 | 611.398 | +3.28% |
| A8W8 dsv4-flash T=16384 | baseline | 1354.384 | 190.276 | 1544.661 | 1362.828 | 187.536 | 1550.364 | -0.37% |
| A8W8 dsv4-pro T=1024* | baseline | 1668.376 | 24.286 | 1692.661 | 1762.431 | 0.000 | 1762.431 | -3.96% |
| A8W8 dsv4-pro T=4096* | baseline | 1822.595 | 87.673 | 1910.269 | 2442.354 | 0.000 | 2442.354 | -21.79% |
| A8W8 dsv4-pro T=16384* | baseline | 3433.639 | 348.646 | 3782.285 | 4343.567 | 0.000 | 4343.567 | -12.92% |
| A8W8 glm52-base T=1024 | .wt | 649.592 | 20.113 | 669.705 | 608.594 | 0.000 | 608.594 | +10.04% |
| A8W8 glm52-base T=4096 | baseline | 860.897 | 92.852 | 953.750 | 915.519 | 0.000 | 915.519 | +4.18% |
| A8W8 glm52-base T=16384 | baseline | 2481.798 | 369.693 | 2851.491 | 3109.222 | 0.000 | 3109.222 | -8.29% |
| A8W8 minimax-m3 T=1024 | M4 + .wt | 479.072 | 12.433 | 491.505 | 452.430 | 0.000 | 452.430 | +8.64% |
| A8W8 minimax-m3 T=4096 | baseline | 657.931 | 50.945 | 708.876 | 664.632 | 0.000 | 664.632 | +6.66% |
| A8W8 minimax-m3 T=16384 | baseline | 1792.995 | 194.870 | 1987.865 | 2102.044 | 0.000 | 2102.044 | -5.43% |

The initial three-implementation applied measurements are retained in the per-case `applied_r1` directories. The failed HIP3 final attempt is retained in its flagged lock log; it measured no cases. The original HIP3 confirmation batch had a load warning and was replaced by clean HIP5 confirmation. The final table uses clean HIP5 comparisons; changed selections receive a new final comparison.

## Controls and validation

1. Screening holds an exclusive lock on HIP3, physical GPU1, PCI `0000:16:00.0`. Accepted confirmation and final comparison hold an exclusive lock on HIP5, PCI `0000:86:00.0`, after external load made HIP3 unavailable. Both backends stay on the same GPU within every pair. Screening gains and final gaps were measured on different physical devices; clean confirmation gains and final gaps use HIP5. The runner checks idle state before each case. Accepted batch lock logs contain no load warnings.
2. Screening and confirmation rotate and reverse baseline, WT, grouped M4 + WT, and FlyDSL invocation order. The final comparison alternates selected Gluon and FlyDSL, matching the earlier report. The initial three-implementation applied run is retained as a timing diagnostic. Each sample uses warm eager `run_perftest(num_iters=20,num_warmup=3)` with 19 retained events per kernel. No graphs or cache flushes.
3. All Gluon candidates and applied selections have bit-exact finite outputs against their baseline. The 31 native baselines also match the public precision wrapper. The five matched baselines share FlyDSL activation/scale/weight buffers, shuffled layouts, and no stage2 bias, and meet the saved numerical tolerance.
4. Native Gluon fixtures include bias and gammas; native FlyDSL fixtures omit stage2 bias. Their shapes and routing activation counts align, but payloads and epilogues differ. These are the same fixtures as the earlier 36-case comparison.
5. AITER source and compiler hashes are checked against the earlier accepted state. Existing sealed reports remain unchanged. This is a per-case benchmark selection for the balanced-routing, warm-eager fixture.

The three A8W8 dsv4-pro FlyDSL V1 rows retain the known 32-bit expert-weight offset error above 4 GiB. All 384 experts are active. Their gaps are timing diagnostics and do not establish a valid speedup. The V1 kernel uses a 32-bit expert-weight byte offset. These shapes require offsets above 4 GiB, so some expert weights are read from the wrong addresses.

## Evidence and reproduction limits

The two measurement and selection records are byte-identical copies from the sealed local archive. The archive contains 1,436 files. Its manifest SHA-256 is `4348790c5a71f1403f9495864277371be92f406b4269c1a863d94da1fab26b3c`. The published report removes links to local files. The provenance record retains the original report hash, source hashes, compiler identities, and exclusions.

The recorded validation covers 36 screening cases, 12 clean confirmation cases, 36 final comparisons, 756 accepted paired rounds, 216 bit-exact Gluon variant checks, 36 measured FlyDSL GEMM binary bindings, and 35 store-policy codegen pairs. The source and compiler did not drift during those measurements.

The measurements use a modified Triton checkout at `b5d2959a7397d1f56eda229eb870d72f88763455` and an assertion-enabled LLVM 24.0.0git build. Their hashes are in the provenance record. A clone with stock compilers does not establish the same kernel identities.

The local runner, trace captures, compiled binaries, and caches remain in the ignored archive. This commit publishes the results and policies. It does not provide a standalone reproduction harness. A rerun requires the recorded compiler environment and fixture adapters. Keep the original source commit in the archive provenance after this report is committed.
