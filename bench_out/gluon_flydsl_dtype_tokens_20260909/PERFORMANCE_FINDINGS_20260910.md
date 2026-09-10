# gfx950 Gluon MoE performance findings

- Date: 2026-09-10
- Refactored source commit: `f42ad9715406816ea626572f4a3eb12a33052aab`
- Comparison device: physical GPU 4, AMD Instinct MI350X, PCI BDF
  `0000:86:00.0`
- Shape: H/K=7168, raw N=4096, intermediate=2048, E=33, top-k=8

Numbers from different physical GPUs are not compared in any conclusion below.
The cold protocol uses four serialized processes per cell, 40 cold warmups and
100 measured samples per process. Every sample is one 768 MiB same-stream fill
followed immediately by one prepared GEMM1 dispatch. Routing, quantization,
scale sorting, weight/scale transforms, allocation, compilation, and launch
preparation are outside the measured loop. P99 is nearest-rank.

## Refactor regression conclusion

The layout/op-shape extraction has no demonstrated correctness, deterministic,
code-generation, or central-latency regression:

- All 20 production cells (four dtypes by five token counts) passed the
  independent reference and 16 poisoned-output exact replays per cell.
- All 20 current Gluon cells have byte-identical executable `.text` and the
  same VGPR, SGPR, spill, dynamic-LDS, and fixed-LDS metadata as the completed
  pre-cleanup baseline.
- At T=4096 every Gluon mean and median was flat or faster than baseline. The
  only raw 1% threshold miss was A8W4 p99 at +1.412%; its mean was -0.114%, its
  median was -0.085%, the paired FlyDSL mean moved -0.093%, and executable text
  was identical. This is not evidence of a code-induced regression.
- The approximately 610 us A4W4 path remains intact: 611.317 / 609.284 /
  640.884 us mean/median/p99 in the final matrix.

See the [final T=4096 report](final_layout_extract2_r2_20260910_cold/report.md)
and [regression check](final_layout_extract2_r2_20260910_cold/regression_vs_pre_cleanup.md).
The completed [small-token report](layout_refactor_small_20260909_cold/report.md)
uses the same physical GPU; subsequent exact-source validation established
identical executable text and resource metadata for all corresponding cells.

## Final T=4096 same-GPU status

These are correctness-qualified production configurations. The FlyDSL A4W4
row is the safe token-aware TM64 kernel, not the faster but nondeterministic
fixed-TM128 diagnostic.

| Dtype | Gluon mean | Gluon median | Gluon p99 | FlyDSL mean | FlyDSL median | FlyDSL p99 | Median winner |
|---|---:|---:|---:|---:|---:|---:|---|
| A4W4 | 611.317 | 609.284 | 640.884 | 684.157 | 683.904 | 704.765 | Gluon, 1.122x |
| A8W4 | 1033.745 | 1031.887 | 1103.048 | 954.049 | 952.506 | 1020.847 | FlyDSL, 1.083x |
| A8W8 | 1182.634 | 1181.488 | 1230.528 | 1303.422 | 1301.448 | 1355.808 | Gluon, 1.102x |
| BF16 | 2280.967 | 2279.193 | 2329.734 | 3450.505 | 3449.041 | 3480.861 | Gluon, 1.513x |

All values are microseconds. The full all-token/all-dtype comparison, with one
complete FlyDSL configuration selected per row by lowest median, is in the
[unified matrix](../flydsl_t4096_tuning_20260909/FLYDSL_ALL_TOKENS_DTYPES_GPU4_20260910.md).

## A4W4 TM128 finding

The remembered approximately 620 us FlyDSL kernel is real. A fresh fixed-TM128
rerun measured 608.379 / 606.904 / 655.204 us mean/median/p99, while the safe
TM64/XCD4 control measured 685.480 / 685.485 / 707.485 us. However, the exact
TM128 binary reproduced intermittent poisoned-output failures, including
replays 1, 185, and 210 in retained runs. TM128 is therefore a useful
performance target but is not correctness-qualified and must not replace the
safe baseline.

At A4W4 T=1024 and T=4096, the unified table shows fixed TM128 when it is the
lowest-median measured candidate, but labels those rows diagnostic-only. The
safe token-aware configurations are TM64/TN256/TK256 with XCD0 at T=1024 and
XCD4 at T=4096. See the [TM128 recheck](../flydsl_t4096_tuning_20260909/A4W4_TM128_RECHECK.md).

## A8W4 small-token root cause

The dominant proven cause of the T=16/T=64 gap is Gluon's cached B-payload
policy. Holding geometry and instruction structure fixed and changing only 72
direct-to-LDS B loads to NT reduced mean latency:

| Tokens | Cached B | NT B | Improvement |
|---:|---:|---:|---:|
| 16 | 200.667 | 153.673 | 46.994 us / 23.42% |
| 64 | 205.923 | 159.997 | 45.925 us / 22.30% |

A BN128/depth3/WPE2 NT configuration then reached 146.859 us at T=16 and
156.209 us at T=64. It still trailed pooled FlyDSL controls by 15.052 us and
22.577 us respectively. The remaining gap is consistent with FlyDSL's more
specialized direct-B register schedule plus smaller occupancy/resource effects.
Recommended work, in order, is to enable the NT B-payload policy for the small
A8W4 regime, adopt or retune the BN128/WPE2 geometry, and prototype a direct-B
register path before broader micro-optimizations.

See the [A8W4 finalist and root-cause analysis](../a8w4_nt_finalist_20260910/FINALIST_ANALYSIS.md).

## Reproducibility files

`rerun_t4096_lds_cleanup_cold.sh` pins the archived/current compatibility-worker
hashes, and `worker.py` contains the corresponding benchmark compatibility
updates. Raw profiler traces, caches, HSACOs, JSON/CSV datasets, and the stopped
redundant partial small-token rerun are intentionally not part of the findings
commit.
