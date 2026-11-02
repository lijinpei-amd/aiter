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
| Commit / branch | `407004a6c097561a9d992749329d7b09cb314886` / `gluon-moe-gfx950-perf-rerun-2026-08-18` |
| Tracked tree | modified |
| Python | `3.14.4 (main, Jun 18 2026, 14:25:02) [GCC 15.2.0]` |
| Torch / HIP / Triton | `2.15.0.dev20261005+rocm7.14` / `7.14.60850` / `3.8.0` |

## Median winner matrix

A cell shows the lower-latency backend and its speedup over the other backend.

| Precision | 16 | 64 | 256 | 1024 | 4096 |
|---|---:|---:|---:|---:|---:|
| BF16 | Gluon 1.187x | Gluon 1.147x | Gluon 1.227x | Gluon 1.683x | Gluon 1.666x |
| A4W4 | Gluon 1.239x | Gluon 1.181x | Gluon 1.452x | Gluon 1.173x | Gluon 1.117x |
| A8W4 | Gluon 1.325x | Gluon 1.271x | Gluon 1.005x | Gluon 1.093x | Gluon 1.089x |
| A8W8 | Gluon 1.248x | Gluon 1.243x | Gluon 1.135x | Gluon 1.081x | Gluon 1.009x |

## Backend results

| Precision | Tokens | Backend | Samples | Mean (us) | Median (us) | P99 (us) |
|---|---:|---|---:|---:|---:|---:|
| BF16 | 16 | flydsl | 400 | 426.901 | 422.303 | 438.122 |
| BF16 | 16 | gluon | 400 | 355.710 | 355.783 | 363.843 |
| BF16 | 64 | flydsl | 400 | 417.138 | 417.202 | 426.402 |
| BF16 | 64 | gluon | 400 | 363.945 | 363.663 | 372.202 |
| BF16 | 256 | flydsl | 400 | 497.617 | 494.283 | 521.043 |
| BF16 | 256 | gluon | 400 | 403.059 | 402.762 | 410.562 |
| BF16 | 1024 | flydsl | 400 | 1084.099 | 1084.008 | 1114.009 |
| BF16 | 1024 | gluon | 400 | 644.287 | 643.945 | 664.485 |
| BF16 | 4096 | flydsl | 400 | 3474.849 | 3472.883 | 3512.906 |
| BF16 | 4096 | gluon | 400 | 2085.066 | 2085.135 | 2120.735 |
| A4W4 | 16 | flydsl | 400 | 138.835 | 139.620 | 144.161 |
| A4W4 | 16 | gluon | 400 | 112.929 | 112.681 | 117.441 |
| A4W4 | 64 | flydsl | 400 | 134.849 | 134.741 | 140.761 |
| A4W4 | 64 | gluon | 400 | 114.395 | 114.121 | 118.681 |
| A4W4 | 256 | flydsl | 400 | 172.063 | 178.321 | 182.841 |
| A4W4 | 256 | gluon | 400 | 123.004 | 122.801 | 128.081 |
| A4W4 | 1024 | flydsl | 400 | 236.413 | 236.382 | 245.682 |
| A4W4 | 1024 | gluon | 400 | 204.038 | 201.521 | 211.641 |
| A4W4 | 4096 | flydsl | 400 | 706.564 | 705.466 | 734.606 |
| A4W4 | 4096 | gluon | 400 | 633.296 | 631.364 | 669.604 |
| A8W4 | 16 | flydsl | 400 | 127.243 | 127.001 | 131.321 |
| A8W4 | 16 | gluon | 400 | 96.155 | 95.841 | 100.080 |
| A8W4 | 64 | flydsl | 400 | 128.189 | 129.441 | 134.041 |
| A8W4 | 64 | gluon | 400 | 102.174 | 101.840 | 106.121 |
| A8W4 | 256 | flydsl | 400 | 140.099 | 139.561 | 148.921 |
| A8W4 | 256 | gluon | 400 | 139.161 | 138.900 | 143.681 |
| A8W4 | 1024 | flydsl | 400 | 282.873 | 281.482 | 304.803 |
| A8W4 | 1024 | gluon | 400 | 259.564 | 257.642 | 278.682 |
| A8W4 | 4096 | flydsl | 400 | 997.945 | 994.567 | 1065.406 |
| A8W4 | 4096 | gluon | 400 | 911.911 | 912.966 | 957.207 |
| A8W8 | 16 | flydsl | 400 | 224.254 | 225.641 | 230.561 |
| A8W8 | 16 | gluon | 400 | 181.096 | 180.801 | 186.962 |
| A8W8 | 64 | flydsl | 400 | 230.142 | 229.762 | 235.762 |
| A8W8 | 64 | gluon | 400 | 185.018 | 184.862 | 190.482 |
| A8W8 | 256 | flydsl | 400 | 255.720 | 255.762 | 261.322 |
| A8W8 | 256 | gluon | 400 | 225.560 | 225.322 | 230.362 |
| A8W8 | 1024 | flydsl | 400 | 383.065 | 383.142 | 395.283 |
| A8W8 | 1024 | gluon | 400 | 354.622 | 354.502 | 364.562 |
| A8W8 | 4096 | flydsl | 400 | 1171.610 | 1169.988 | 1229.009 |
| A8W8 | 4096 | gluon | 400 | 1161.170 | 1160.087 | 1216.449 |

## Pairwise comparison

Positive delta and G/F ratio above 1 mean FlyDSL is faster.

| Precision | Tokens | Statistic | Gluon (us) | FlyDSL (us) | Delta G-F (us) | G/F ratio | Winner |
|---|---:|---|---:|---:|---:|---:|---|
| BF16 | 16 | mean | 355.710 | 426.901 | -71.190 | 0.8332x | Gluon |
| BF16 | 16 | median | 355.783 | 422.303 | -66.520 | 0.8425x | Gluon |
| BF16 | 16 | p99 | 363.843 | 438.122 | -74.279 | 0.8305x | Gluon |
| BF16 | 64 | mean | 363.945 | 417.138 | -53.192 | 0.8725x | Gluon |
| BF16 | 64 | median | 363.663 | 417.202 | -53.539 | 0.8717x | Gluon |
| BF16 | 64 | p99 | 372.202 | 426.402 | -54.200 | 0.8729x | Gluon |
| BF16 | 256 | mean | 403.059 | 497.617 | -94.558 | 0.8100x | Gluon |
| BF16 | 256 | median | 402.762 | 494.283 | -91.521 | 0.8148x | Gluon |
| BF16 | 256 | p99 | 410.562 | 521.043 | -110.481 | 0.7880x | Gluon |
| BF16 | 1024 | mean | 644.287 | 1084.099 | -439.811 | 0.5943x | Gluon |
| BF16 | 1024 | median | 643.945 | 1084.008 | -440.063 | 0.5940x | Gluon |
| BF16 | 1024 | p99 | 664.485 | 1114.009 | -449.524 | 0.5965x | Gluon |
| BF16 | 4096 | mean | 2085.066 | 3474.849 | -1389.782 | 0.6000x | Gluon |
| BF16 | 4096 | median | 2085.135 | 3472.883 | -1387.748 | 0.6004x | Gluon |
| BF16 | 4096 | p99 | 2120.735 | 3512.906 | -1392.171 | 0.6037x | Gluon |
| A4W4 | 16 | mean | 112.929 | 138.835 | -25.907 | 0.8134x | Gluon |
| A4W4 | 16 | median | 112.681 | 139.620 | -26.939 | 0.8071x | Gluon |
| A4W4 | 16 | p99 | 117.441 | 144.161 | -26.720 | 0.8147x | Gluon |
| A4W4 | 64 | mean | 114.395 | 134.849 | -20.454 | 0.8483x | Gluon |
| A4W4 | 64 | median | 114.121 | 134.741 | -20.620 | 0.8470x | Gluon |
| A4W4 | 64 | p99 | 118.681 | 140.761 | -22.080 | 0.8431x | Gluon |
| A4W4 | 256 | mean | 123.004 | 172.063 | -49.059 | 0.7149x | Gluon |
| A4W4 | 256 | median | 122.801 | 178.321 | -55.520 | 0.6887x | Gluon |
| A4W4 | 256 | p99 | 128.081 | 182.841 | -54.760 | 0.7005x | Gluon |
| A4W4 | 1024 | mean | 204.038 | 236.413 | -32.376 | 0.8631x | Gluon |
| A4W4 | 1024 | median | 201.521 | 236.382 | -34.861 | 0.8525x | Gluon |
| A4W4 | 1024 | p99 | 211.641 | 245.682 | -34.041 | 0.8614x | Gluon |
| A4W4 | 4096 | mean | 633.296 | 706.564 | -73.268 | 0.8963x | Gluon |
| A4W4 | 4096 | median | 631.364 | 705.466 | -74.101 | 0.8950x | Gluon |
| A4W4 | 4096 | p99 | 669.604 | 734.606 | -65.002 | 0.9115x | Gluon |
| A8W4 | 16 | mean | 96.155 | 127.243 | -31.088 | 0.7557x | Gluon |
| A8W4 | 16 | median | 95.841 | 127.001 | -31.160 | 0.7546x | Gluon |
| A8W4 | 16 | p99 | 100.080 | 131.321 | -31.241 | 0.7621x | Gluon |
| A8W4 | 64 | mean | 102.174 | 128.189 | -26.015 | 0.7971x | Gluon |
| A8W4 | 64 | median | 101.840 | 129.441 | -27.601 | 0.7868x | Gluon |
| A8W4 | 64 | p99 | 106.121 | 134.041 | -27.920 | 0.7917x | Gluon |
| A8W4 | 256 | mean | 139.161 | 140.099 | -0.938 | 0.9933x | Gluon |
| A8W4 | 256 | median | 138.900 | 139.561 | -0.661 | 0.9953x | Gluon |
| A8W4 | 256 | p99 | 143.681 | 148.921 | -5.240 | 0.9648x | Gluon |
| A8W4 | 1024 | mean | 259.564 | 282.873 | -23.309 | 0.9176x | Gluon |
| A8W4 | 1024 | median | 257.642 | 281.482 | -23.840 | 0.9153x | Gluon |
| A8W4 | 1024 | p99 | 278.682 | 304.803 | -26.121 | 0.9143x | Gluon |
| A8W4 | 4096 | mean | 911.911 | 997.945 | -86.034 | 0.9138x | Gluon |
| A8W4 | 4096 | median | 912.966 | 994.567 | -81.601 | 0.9180x | Gluon |
| A8W4 | 4096 | p99 | 957.207 | 1065.406 | -108.199 | 0.8984x | Gluon |
| A8W8 | 16 | mean | 181.096 | 224.254 | -43.158 | 0.8075x | Gluon |
| A8W8 | 16 | median | 180.801 | 225.641 | -44.840 | 0.8013x | Gluon |
| A8W8 | 16 | p99 | 186.962 | 230.561 | -43.599 | 0.8109x | Gluon |
| A8W8 | 64 | mean | 185.018 | 230.142 | -45.124 | 0.8039x | Gluon |
| A8W8 | 64 | median | 184.862 | 229.762 | -44.900 | 0.8046x | Gluon |
| A8W8 | 64 | p99 | 190.482 | 235.762 | -45.280 | 0.8079x | Gluon |
| A8W8 | 256 | mean | 225.560 | 255.720 | -30.160 | 0.8821x | Gluon |
| A8W8 | 256 | median | 225.322 | 255.762 | -30.440 | 0.8810x | Gluon |
| A8W8 | 256 | p99 | 230.362 | 261.322 | -30.960 | 0.8815x | Gluon |
| A8W8 | 1024 | mean | 354.622 | 383.065 | -28.443 | 0.9257x | Gluon |
| A8W8 | 1024 | median | 354.502 | 383.142 | -28.640 | 0.9252x | Gluon |
| A8W8 | 1024 | p99 | 364.562 | 395.283 | -30.721 | 0.9223x | Gluon |
| A8W8 | 4096 | mean | 1161.170 | 1171.610 | -10.440 | 0.9911x | Gluon |
| A8W8 | 4096 | median | 1160.087 | 1169.988 | -9.901 | 0.9915x | Gluon |
| A8W8 | 4096 | p99 | 1216.449 | 1229.009 | -12.560 | 0.9898x | Gluon |

## Effective configurations and routing padding

Active rows include per-expert padding; extra rows are not useful token/top-k rows.

| Precision | Tokens | Backend | Tile | Key flags | Useful rows | Active rows | Extra | Padding | Blocks | Output layout / policy |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| BF16 | 16 | flydsl | `TM16xTN128xTK128` | wpe=2, b_nt=2, nt=None, gate=separated, async=True | 128 | 528 | 400 | 312.50% | 33 | sorted / none |
| BF16 | 16 | gluon | `BM32xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| BF16 | 64 | flydsl | `TM16xTN128xTK256` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 512 | 528 | 16 | 3.12% | 33 | sorted / none |
| BF16 | 64 | gluon | `BM32xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| BF16 | 256 | flydsl | `TM64xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 2048 | 2112 | 64 | 3.12% | 33 | sorted / none |
| BF16 | 256 | gluon | `BM64xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| BF16 | 1024 | flydsl | `TM64xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 132 | sorted / none |
| BF16 | 1024 | gluon | `BM128xBN256xBK64` | MFMA=[16, 16, 32], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| BF16 | 4096 | flydsl | `TM128xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | sorted / none |
| BF16 | 4096 | gluon | `BM128xBN256xBK64` | MFMA=[16, 16, 32], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |
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
| A8W4 | 16 | gluon | `BM32xBN64xBK256` | B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| A8W4 | 64 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 512 | 1056 | 544 | 106.25% | 33 | token_slot / none |
| A8W4 | 64 | gluon | `BM32xBN64xBK256` | B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| A8W4 | 256 | flydsl | `TM64xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=False | 2048 | 2112 | 64 | 3.12% | 33 | token_slot / none |
| A8W4 | 256 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| A8W4 | 1024 | flydsl | `TM128xTN256xTK256` | wpe=1, b_nt=2, nt=None, gate=interleave, async=True | 8192 | 8448 | 256 | 3.12% | 66 | token_slot / none |
| A8W4 | 1024 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| A8W4 | 4096 | flydsl | `TM128xTN256xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | token_slot / none |
| A8W4 | 4096 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |
| A8W8 | 16 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 128 | 1056 | 928 | 725.00% | 33 | token_slot / none |
| A8W8 | 16 | gluon | `BM32xBN64xBK256` | B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| A8W8 | 64 | flydsl | `TM32xTN64xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 512 | 1056 | 544 | 106.25% | 33 | token_slot / none |
| A8W8 | 64 | gluon | `BM32xBN64xBK256` | B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| A8W8 | 256 | flydsl | `TM64xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=False | 2048 | 2112 | 64 | 3.12% | 33 | token_slot / none |
| A8W8 | 256 | gluon | `BM64xBN128xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| A8W8 | 1024 | flydsl | `TM128xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 66 | token_slot / none |
| A8W8 | 1024 | gluon | `BM128xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| A8W8 | 4096 | flydsl | `TM128xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=separated, async=True | 32768 | 33792 | 1024 | 3.12% | 264 | token_slot / none |
| A8W8 | 4096 | gluon | `BM128xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 32768 | 33792 | 1024 | 3.12% | 264 | dense expert-sorted rows / none |

## Validation

Cold measurements are gated by [`val_hv_new`](../val_hv_new/). The timed kernel, full logical inputs, and output hashes must match validation.

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
