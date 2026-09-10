# Cold Gluon versus FlyDSL GEMM1 results

Status: **complete**; 32/32 processes recorded; 0 failures.

## Scope and protocol

The logical GEMM1 shape is H/K=7168, intermediate=2048 (raw N=4096), E=33, top-k=8, with `SiLU(gate) * up` and BF16 output except for quantized A4W4 output.

Each process uses 40 cold warmups and 100 measured samples. Cold results use 4 alternating-order rounds and pool all 400 device-duration samples per cell. Every sample is exactly a 768 MiB streaming fill followed by one prepared GEMM1 dispatch on the same stream; the trace parser rejects extra kernels.

Routing, input quantization, A-scale sorting, weight/scale shuffle or preshuffle, output allocation, JIT compilation, and launch-argument preparation are setup work outside the measured loop. P99 is nearest-rank: `sorted[ceil(0.99*n)-1]`.

## Environment

| Item | Value |
|---|---|
| GPU | AMD Instinct MI350X |
| Architecture | `gfx950:sramecc+:xnack-` |
| Physical / HIP index | 4 / 5 |
| PCI BDF / UUID | `0000:86:00.0` / `170075a0-0000-1000-80d5-3df13e2c9fe7` |
| Commit / branch | `591175e3a7f77edd8077737ab8b83ed51f54a62a` / `gluon-moe-gfx950-perf-rerun-2026-08-18` |
| Tracked tree | modified |
| Python | `3.14.4 (main, Jun 18 2026, 14:25:02) [GCC 15.2.0]` |
| Torch / HIP / Triton | `2.15.0.dev20260816+rocm7.14` / `7.14.60850` / `3.8.0` |

## Median winner matrix

A cell shows the lower-latency backend and its speedup over the other backend.

| Precision | 4096 |
|---|---:|
| BF16 | Gluon 1.513x |
| A4W4 | Gluon 1.122x |
| A8W4 | FlyDSL 1.083x |
| A8W8 | Gluon 1.102x |

## Backend results

| Precision | Tokens | Backend | Samples | Mean (us) | Median (us) | P99 (us) |
|---|---:|---|---:|---:|---:|---:|
| BF16 | 4096 | flydsl | 400 | 3450.505 | 3449.041 | 3480.861 |
| BF16 | 4096 | gluon | 400 | 2280.967 | 2279.193 | 2329.734 |
| A4W4 | 4096 | flydsl | 400 | 684.157 | 683.904 | 704.765 |
| A4W4 | 4096 | gluon | 400 | 611.317 | 609.284 | 640.884 |
| A8W4 | 4096 | flydsl | 400 | 954.049 | 952.506 | 1020.847 |
| A8W4 | 4096 | gluon | 400 | 1033.745 | 1031.887 | 1103.048 |
| A8W8 | 4096 | flydsl | 400 | 1303.422 | 1301.448 | 1355.808 |
| A8W8 | 4096 | gluon | 400 | 1182.634 | 1181.488 | 1230.528 |

## Pairwise comparison

Positive delta and G/F ratio above 1 mean FlyDSL is faster.

| Precision | Tokens | Statistic | Gluon (us) | FlyDSL (us) | Delta G-F (us) | G/F ratio | Winner |
|---|---:|---|---:|---:|---:|---:|---|
| BF16 | 4096 | mean | 2280.967 | 3450.505 | -1169.538 | 0.6611x | Gluon |
| BF16 | 4096 | median | 2279.193 | 3449.041 | -1169.847 | 0.6608x | Gluon |
| BF16 | 4096 | p99 | 2329.734 | 3480.861 | -1151.127 | 0.6693x | Gluon |
| A4W4 | 4096 | mean | 611.317 | 684.157 | -72.839 | 0.8935x | Gluon |
| A4W4 | 4096 | median | 609.284 | 683.904 | -74.620 | 0.8909x | Gluon |
| A4W4 | 4096 | p99 | 640.884 | 704.765 | -63.881 | 0.9094x | Gluon |
| A8W4 | 4096 | mean | 1033.745 | 954.049 | +79.697 | 1.0835x | FlyDSL |
| A8W4 | 4096 | median | 1031.887 | 952.506 | +79.381 | 1.0833x | FlyDSL |
| A8W4 | 4096 | p99 | 1103.048 | 1020.847 | +82.201 | 1.0805x | FlyDSL |
| A8W8 | 4096 | mean | 1182.634 | 1303.422 | -120.788 | 0.9073x | Gluon |
| A8W8 | 4096 | median | 1181.488 | 1301.448 | -119.960 | 0.9078x | Gluon |
| A8W8 | 4096 | p99 | 1230.528 | 1355.808 | -125.280 | 0.9076x | Gluon |

## Effective configurations and routing padding

Active rows include per-expert padding; extra rows are not useful token/top-k rows.

| Precision | Tokens | Backend | Tile | Key flags | Useful rows | Active rows | Extra | Padding | Blocks | Output layout / policy |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| BF16 | 4096 | flydsl | `TM128xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | sorted / none |
| BF16 | 4096 | gluon | `BM128xBN256xBK64` | MFMA=[32, 32, 16], warps=[2, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |
| A4W4 | 4096 | flydsl | `TM64xTN256xTK256` | wpe=None, b_nt=0, nt=False, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 528 | sorted / round_up |
| A4W4 | 4096 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4], NT=0 | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / even |
| A8W4 | 4096 | flydsl | `TM128xTN256xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | token_slot / none |
| A8W4 | 4096 | gluon | `BM128xBN128xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |
| A8W8 | 4096 | flydsl | `TM128xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=interleave, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | token_slot / none |
| A8W8 | 4096 | gluon | `BM128xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |

## Validation

Cold measurements are gated by the local
`final_layout_extract2_r2_20260910_validation/` artifact. The timed kernel,
full logical inputs, and output hashes must match validation.

| Precision | Backend | Passed cells | Replays/cell | Worst max normalized error | Worst RMS normalized error |
|---|---|---:|---:|---:|---:|
| BF16 | flydsl | 1/1 | 16 | 0.00389475 | 0.000859655 |
| BF16 | gluon | 1/1 | 16 | 0.00389475 | 0.000859655 |
| A4W4 | flydsl | 1/1 | 16 | 0 | 0 |
| A4W4 | gluon | 1/1 | 16 | 0 | 0 |
| A8W4 | flydsl | 1/1 | 16 | 0.00390383 | 0.000784778 |
| A8W4 | gluon | 1/1 | 16 | 0.00390383 | 0.000784778 |
| A8W8 | flydsl | 1/1 | 16 | 0.00396757 | 0.000859907 |
| A8W8 | gluon | 1/1 | 16 | 0.00396757 | 0.000859907 |

## Interpretation notes

A4W4 native output quantization uses round-to-even scales in Gluon and round-up scales in FlyDSL; each backend is checked against the matching independent reference. Gluon emits expert-sorted rows, while FlyDSL may emit token-slot or sorted native layouts; layout conversion is excluded because only the prepared GEMM1 is measured.

## Artifacts

The local result directory also contains `plan.json`, `provenance.json`,
`gpu_inventory.json`, `summary.json`, `comparison.json`, `records.json`,
`samples.csv`, `samples.json`, `failures.json`, and
`independent_audit.json`. Those raw benchmark artifacts are intentionally not
part of the findings commit.
