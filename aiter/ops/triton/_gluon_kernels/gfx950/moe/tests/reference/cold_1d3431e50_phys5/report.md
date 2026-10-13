# Cold Gluon versus FlyDSL GEMM1 results

Status: **complete**; 160/160 processes recorded; 0 failures.

## Scope and protocol

The logical GEMM1 shape is H/K=7168, intermediate=2048 (raw N=4096), E=33, top-k=8, with `SiLU(gate) * up` and BF16 output except for quantized A4W4 output.

Each process uses 40 cold warmups and 100 measured samples. Cold results use 4 alternating-order rounds and pool all 400 device-duration samples per cell. Every sample is exactly a 768 MiB streaming fill followed by one prepared GEMM1 dispatch on the same stream; the trace parser rejects extra kernels.

Routing, input quantization, A-scale sorting, weight/scale shuffle or preshuffle, output allocation, JIT compilation, and launch-argument preparation are setup work outside the measured loop. P99 is nearest-rank: `sorted[ceil(0.99*n)-1]`.

## Environment

| Item | Value |
|---|---|
| GPU | AMD Instinct MI350X |
| Architecture | `gfx950:sramecc+:xnack-` |
| Physical / HIP index | 5 / 7 |
| PCI BDF / UUID | `0000:96:00.0` / `d90075a0-0000-1000-80f1-c568fe678897` |
| Commit / branch | `1d3431e50fcce5428510228529de441894ce9aeb` / `gluon-moe-gfx950-perf-rerun-2026-08-18` |
| Tracked tree | clean |
| Python | `3.14.4 (main, Jun 18 2026, 14:25:02) [GCC 15.2.0]` |
| Torch / HIP / Triton | `2.15.0.dev20260816+rocm7.14` / `7.14.60850` / `3.8.0` |

## Median winner matrix

A cell shows the lower-latency backend and its speedup over the other backend.

| Precision | 16 | 64 | 256 | 1024 | 4096 |
|---|---:|---:|---:|---:|---:|
| BF16 | FlyDSL 1.099x | FlyDSL 1.101x | Gluon 1.162x | Gluon 1.570x | Gluon 1.557x |
| A4W4 | Gluon 1.245x | Gluon 1.231x | Gluon 1.457x | Gluon 1.191x | Gluon 1.117x |
| A8W4 | Gluon 1.008x | FlyDSL 1.021x | Gluon 1.052x | Gluon 1.033x | Gluon 1.003x |
| A8W8 | FlyDSL 1.034x | FlyDSL 1.054x | Gluon 1.009x | Gluon 1.058x | Gluon 1.007x |

## Backend results

| Precision | Tokens | Backend | Samples | Mean (us) | Median (us) | P99 (us) |
|---|---:|---|---:|---:|---:|---:|
| BF16 | 16 | flydsl | 400 | 420.046 | 419.623 | 433.603 |
| BF16 | 16 | gluon | 400 | 461.099 | 460.983 | 469.404 |
| BF16 | 64 | flydsl | 400 | 423.034 | 422.203 | 433.203 |
| BF16 | 64 | gluon | 400 | 464.718 | 464.684 | 472.764 |
| BF16 | 256 | flydsl | 400 | 504.540 | 504.644 | 516.764 |
| BF16 | 256 | gluon | 400 | 434.324 | 434.163 | 441.643 |
| BF16 | 1024 | flydsl | 400 | 1073.228 | 1072.488 | 1100.328 |
| BF16 | 1024 | gluon | 400 | 683.729 | 683.125 | 698.525 |
| BF16 | 4096 | flydsl | 400 | 3468.879 | 3468.167 | 3502.587 |
| BF16 | 4096 | gluon | 400 | 2228.868 | 2227.197 | 2285.737 |
| A4W4 | 16 | flydsl | 400 | 140.149 | 140.061 | 143.721 |
| A4W4 | 16 | gluon | 400 | 112.752 | 112.480 | 116.360 |
| A4W4 | 64 | flydsl | 400 | 140.804 | 140.681 | 144.241 |
| A4W4 | 64 | gluon | 400 | 114.520 | 114.241 | 118.400 |
| A4W4 | 256 | flydsl | 400 | 179.127 | 179.001 | 183.521 |
| A4W4 | 256 | gluon | 400 | 122.965 | 122.841 | 127.001 |
| A4W4 | 1024 | flydsl | 400 | 237.669 | 237.502 | 244.441 |
| A4W4 | 1024 | gluon | 400 | 199.777 | 199.482 | 208.562 |
| A4W4 | 4096 | flydsl | 400 | 698.872 | 698.265 | 726.886 |
| A4W4 | 4096 | gluon | 400 | 625.705 | 625.124 | 667.245 |
| A8W4 | 16 | flydsl | 400 | 127.373 | 127.141 | 130.801 |
| A8W4 | 16 | gluon | 400 | 126.366 | 126.101 | 129.481 |
| A8W4 | 64 | flydsl | 400 | 129.958 | 129.801 | 132.881 |
| A8W4 | 64 | gluon | 400 | 132.664 | 132.521 | 136.001 |
| A8W4 | 256 | flydsl | 400 | 146.202 | 146.041 | 149.762 |
| A8W4 | 256 | gluon | 400 | 139.038 | 138.861 | 143.401 |
| A8W4 | 1024 | flydsl | 400 | 280.443 | 279.002 | 299.963 |
| A8W4 | 1024 | gluon | 400 | 270.580 | 270.102 | 283.642 |
| A8W4 | 4096 | flydsl | 400 | 975.936 | 974.027 | 1030.807 |
| A8W4 | 4096 | gluon | 400 | 972.601 | 970.687 | 1030.687 |
| A8W8 | 16 | flydsl | 400 | 227.181 | 226.962 | 230.922 |
| A8W8 | 16 | gluon | 400 | 234.735 | 234.621 | 239.562 |
| A8W8 | 64 | flydsl | 400 | 230.499 | 230.402 | 237.482 |
| A8W8 | 64 | gluon | 400 | 242.832 | 242.781 | 246.802 |
| A8W8 | 256 | flydsl | 400 | 256.220 | 256.202 | 260.762 |
| A8W8 | 256 | gluon | 400 | 254.046 | 254.041 | 257.722 |
| A8W8 | 1024 | flydsl | 400 | 382.435 | 382.203 | 392.323 |
| A8W8 | 1024 | gluon | 400 | 361.664 | 361.402 | 370.523 |
| A8W8 | 4096 | flydsl | 400 | 1159.873 | 1159.309 | 1217.289 |
| A8W8 | 4096 | gluon | 400 | 1151.842 | 1151.148 | 1210.809 |

## Pairwise comparison

Positive delta and G/F ratio above 1 mean FlyDSL is faster.

| Precision | Tokens | Statistic | Gluon (us) | FlyDSL (us) | Delta G-F (us) | G/F ratio | Winner |
|---|---:|---|---:|---:|---:|---:|---|
| BF16 | 16 | mean | 461.099 | 420.046 | +41.053 | 1.0977x | FlyDSL |
| BF16 | 16 | median | 460.983 | 419.623 | +41.360 | 1.0986x | FlyDSL |
| BF16 | 16 | p99 | 469.404 | 433.603 | +35.801 | 1.0826x | FlyDSL |
| BF16 | 64 | mean | 464.718 | 423.034 | +41.684 | 1.0985x | FlyDSL |
| BF16 | 64 | median | 464.684 | 422.203 | +42.481 | 1.1006x | FlyDSL |
| BF16 | 64 | p99 | 472.764 | 433.203 | +39.561 | 1.0913x | FlyDSL |
| BF16 | 256 | mean | 434.324 | 504.540 | -70.216 | 0.8608x | Gluon |
| BF16 | 256 | median | 434.163 | 504.644 | -70.481 | 0.8603x | Gluon |
| BF16 | 256 | p99 | 441.643 | 516.764 | -75.121 | 0.8546x | Gluon |
| BF16 | 1024 | mean | 683.729 | 1073.228 | -389.499 | 0.6371x | Gluon |
| BF16 | 1024 | median | 683.125 | 1072.488 | -389.363 | 0.6370x | Gluon |
| BF16 | 1024 | p99 | 698.525 | 1100.328 | -401.803 | 0.6348x | Gluon |
| BF16 | 4096 | mean | 2228.868 | 3468.879 | -1240.011 | 0.6425x | Gluon |
| BF16 | 4096 | median | 2227.197 | 3468.167 | -1240.970 | 0.6422x | Gluon |
| BF16 | 4096 | p99 | 2285.737 | 3502.587 | -1216.850 | 0.6526x | Gluon |
| A4W4 | 16 | mean | 112.752 | 140.149 | -27.397 | 0.8045x | Gluon |
| A4W4 | 16 | median | 112.480 | 140.061 | -27.581 | 0.8031x | Gluon |
| A4W4 | 16 | p99 | 116.360 | 143.721 | -27.361 | 0.8096x | Gluon |
| A4W4 | 64 | mean | 114.520 | 140.804 | -26.284 | 0.8133x | Gluon |
| A4W4 | 64 | median | 114.241 | 140.681 | -26.440 | 0.8121x | Gluon |
| A4W4 | 64 | p99 | 118.400 | 144.241 | -25.841 | 0.8208x | Gluon |
| A4W4 | 256 | mean | 122.965 | 179.127 | -56.162 | 0.6865x | Gluon |
| A4W4 | 256 | median | 122.841 | 179.001 | -56.160 | 0.6863x | Gluon |
| A4W4 | 256 | p99 | 127.001 | 183.521 | -56.520 | 0.6920x | Gluon |
| A4W4 | 1024 | mean | 199.777 | 237.669 | -37.892 | 0.8406x | Gluon |
| A4W4 | 1024 | median | 199.482 | 237.502 | -38.020 | 0.8399x | Gluon |
| A4W4 | 1024 | p99 | 208.562 | 244.441 | -35.879 | 0.8532x | Gluon |
| A4W4 | 4096 | mean | 625.705 | 698.872 | -73.167 | 0.8953x | Gluon |
| A4W4 | 4096 | median | 625.124 | 698.265 | -73.141 | 0.8953x | Gluon |
| A4W4 | 4096 | p99 | 667.245 | 726.886 | -59.641 | 0.9179x | Gluon |
| A8W4 | 16 | mean | 126.366 | 127.373 | -1.007 | 0.9921x | Gluon |
| A8W4 | 16 | median | 126.101 | 127.141 | -1.040 | 0.9918x | Gluon |
| A8W4 | 16 | p99 | 129.481 | 130.801 | -1.320 | 0.9899x | Gluon |
| A8W4 | 64 | mean | 132.664 | 129.958 | +2.706 | 1.0208x | FlyDSL |
| A8W4 | 64 | median | 132.521 | 129.801 | +2.720 | 1.0210x | FlyDSL |
| A8W4 | 64 | p99 | 136.001 | 132.881 | +3.120 | 1.0235x | FlyDSL |
| A8W4 | 256 | mean | 139.038 | 146.202 | -7.165 | 0.9510x | Gluon |
| A8W4 | 256 | median | 138.861 | 146.041 | -7.180 | 0.9508x | Gluon |
| A8W4 | 256 | p99 | 143.401 | 149.762 | -6.361 | 0.9575x | Gluon |
| A8W4 | 1024 | mean | 270.580 | 280.443 | -9.863 | 0.9648x | Gluon |
| A8W4 | 1024 | median | 270.102 | 279.002 | -8.901 | 0.9681x | Gluon |
| A8W4 | 1024 | p99 | 283.642 | 299.963 | -16.321 | 0.9456x | Gluon |
| A8W4 | 4096 | mean | 972.601 | 975.936 | -3.335 | 0.9966x | Gluon |
| A8W4 | 4096 | median | 970.687 | 974.027 | -3.340 | 0.9966x | Gluon |
| A8W4 | 4096 | p99 | 1030.687 | 1030.807 | -0.120 | 0.9999x | Gluon |
| A8W8 | 16 | mean | 234.735 | 227.181 | +7.554 | 1.0333x | FlyDSL |
| A8W8 | 16 | median | 234.621 | 226.962 | +7.660 | 1.0337x | FlyDSL |
| A8W8 | 16 | p99 | 239.562 | 230.922 | +8.640 | 1.0374x | FlyDSL |
| A8W8 | 64 | mean | 242.832 | 230.499 | +12.333 | 1.0535x | FlyDSL |
| A8W8 | 64 | median | 242.781 | 230.402 | +12.380 | 1.0537x | FlyDSL |
| A8W8 | 64 | p99 | 246.802 | 237.482 | +9.320 | 1.0392x | FlyDSL |
| A8W8 | 256 | mean | 254.046 | 256.220 | -2.174 | 0.9915x | Gluon |
| A8W8 | 256 | median | 254.041 | 256.202 | -2.161 | 0.9916x | Gluon |
| A8W8 | 256 | p99 | 257.722 | 260.762 | -3.040 | 0.9883x | Gluon |
| A8W8 | 1024 | mean | 361.664 | 382.435 | -20.772 | 0.9457x | Gluon |
| A8W8 | 1024 | median | 361.402 | 382.203 | -20.801 | 0.9456x | Gluon |
| A8W8 | 1024 | p99 | 370.523 | 392.323 | -21.800 | 0.9444x | Gluon |
| A8W8 | 4096 | mean | 1151.842 | 1159.873 | -8.031 | 0.9931x | Gluon |
| A8W8 | 4096 | median | 1151.148 | 1159.309 | -8.161 | 0.9930x | Gluon |
| A8W8 | 4096 | p99 | 1210.809 | 1217.289 | -6.480 | 0.9947x | Gluon |

## Effective configurations and routing padding

Active rows include per-expert padding; extra rows are not useful token/top-k rows.

| Precision | Tokens | Backend | Tile | Key flags | Useful rows | Active rows | Extra | Padding | Blocks | Output layout / policy |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| BF16 | 16 | flydsl | `TM16xTN128xTK128` | wpe=2, b_nt=2, nt=None, gate=separated, async=True | 128 | 528 | 400 | 312.50% | 33 | sorted / none |
| BF16 | 16 | gluon | `BM32xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| BF16 | 64 | flydsl | `TM16xTN128xTK256` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 512 | 528 | 16 | 3.12% | 33 | sorted / none |
| BF16 | 64 | gluon | `BM32xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| BF16 | 256 | flydsl | `TM64xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 2048 | 2112 | 64 | 3.12% | 33 | sorted / none |
| BF16 | 256 | gluon | `BM64xBN256xBK64` | MFMA=[32, 32, 16], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| BF16 | 1024 | flydsl | `TM64xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 132 | sorted / none |
| BF16 | 1024 | gluon | `BM128xBN256xBK64` | MFMA=[32, 32, 16], warps=[2, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| BF16 | 4096 | flydsl | `TM128xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | sorted / none |
| BF16 | 4096 | gluon | `BM128xBN256xBK64` | MFMA=[32, 32, 16], warps=[2, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |
| A4W4 | 16 | flydsl | `TM32xTN256xTK256` | wpe=None, b_nt=2, nt=True, gate=separated, async=True | 128 | 1056 | 928 | 725.00% | 33 | sorted / round_up |
| A4W4 | 16 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 128 | 2112 | 1984 | 1550.00% | 33 | dense expert-sorted rows / even |
| A4W4 | 64 | flydsl | `TM32xTN256xTK256` | wpe=None, b_nt=2, nt=True, gate=separated, async=True | 512 | 1056 | 544 | 106.25% | 33 | sorted / round_up |
| A4W4 | 64 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 512 | 2112 | 1600 | 312.50% | 33 | dense expert-sorted rows / even |
| A4W4 | 256 | flydsl | `TM32xTN256xTK256` | wpe=None, b_nt=2, nt=True, gate=separated, async=True | 2048 | 2112 | 64 | 3.12% | 66 | sorted / round_up |
| A4W4 | 256 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / even |
| A4W4 | 1024 | flydsl | `TM64xTN256xTK256` | wpe=None, b_nt=0, nt=False, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 132 | sorted / round_up |
| A4W4 | 1024 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / even |
| A4W4 | 4096 | flydsl | `TM64xTN256xTK256` | wpe=None, b_nt=0, nt=False, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 528 | sorted / round_up |
| A4W4 | 4096 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / even |
| A8W4 | 16 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 128 | 1056 | 928 | 725.00% | 33 | token_slot / none |
| A8W4 | 16 | gluon | `BM32xBN128xBK512` | MFMA=[16, 16, 128], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| A8W4 | 64 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 512 | 1056 | 544 | 106.25% | 33 | token_slot / none |
| A8W4 | 64 | gluon | `BM32xBN128xBK512` | MFMA=[16, 16, 128], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| A8W4 | 256 | flydsl | `TM64xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=False | 2048 | 2112 | 64 | 3.12% | 33 | token_slot / none |
| A8W4 | 256 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| A8W4 | 1024 | flydsl | `TM128xTN256xTK256` | wpe=1, b_nt=2, nt=None, gate=interleave, async=True | 8192 | 8448 | 256 | 3.12% | 66 | token_slot / none |
| A8W4 | 1024 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| A8W4 | 4096 | flydsl | `TM128xTN256xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | token_slot / none |
| A8W4 | 4096 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |
| A8W8 | 16 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 128 | 1056 | 928 | 725.00% | 33 | token_slot / none |
| A8W8 | 16 | gluon | `BM32xBN128xBK256` | MFMA=[16, 16, 128], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| A8W8 | 64 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 512 | 1056 | 544 | 106.25% | 33 | token_slot / none |
| A8W8 | 64 | gluon | `BM32xBN128xBK256` | MFMA=[16, 16, 128], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| A8W8 | 256 | flydsl | `TM64xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=False | 2048 | 2112 | 64 | 3.12% | 33 | token_slot / none |
| A8W8 | 256 | gluon | `BM64xBN128xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| A8W8 | 1024 | flydsl | `TM128xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 66 | token_slot / none |
| A8W8 | 1024 | gluon | `BM128xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| A8W8 | 4096 | flydsl | `TM128xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | token_slot / none |
| A8W8 | 4096 | gluon | `BM128xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |

## Validation

Cold measurements are gated by [`val_ref_819123dbc`](../val_ref_819123dbc/). The timed kernel, full logical inputs, and output hashes must match validation.

| Precision | Backend | Passed cells | Replays/cell | Worst max normalized error | Worst RMS normalized error |
|---|---|---:|---:|---:|---:|
| BF16 | flydsl | 5/5 | 16 | 0.00389475 | 0.000860886 |
| BF16 | gluon | 5/5 | 16 | 0.00389475 | 0.000860885 |
| A4W4 | flydsl | 5/5 | 16 | 0 | 0 |
| A4W4 | gluon | 5/5 | 16 | 0 | 0 |
| A8W4 | flydsl | 5/5 | 16 | 0.00390383 | 0.000785945 |
| A8W4 | gluon | 5/5 | 16 | 0.00390383 | 0.000785945 |
| A8W8 | flydsl | 5/5 | 16 | 0.0039697 | 0.000860354 |
| A8W8 | gluon | 5/5 | 16 | 0.0039697 | 0.000860354 |

## Interpretation notes

A4W4 native output quantization uses round-to-even scales in Gluon and round-up scales in FlyDSL; each backend is checked against the matching independent reference. Gluon emits expert-sorted rows, while FlyDSL may emit token-slot or sorted native layouts; layout conversion is excluded because only the prepared GEMM1 is measured.

## Artifacts

[`plan.json`](plan.json) · [`provenance.json`](provenance.json) · [`gpu_inventory.json`](gpu_inventory.json) · [`summary.json`](summary.json) · [`comparison.json`](comparison.json) · [`records.json`](records.json) · [`samples.csv`](samples.csv) · [`samples.json`](samples.json) · [`failures.json`](failures.json) · [`independent_audit.json`](independent_audit.json)
