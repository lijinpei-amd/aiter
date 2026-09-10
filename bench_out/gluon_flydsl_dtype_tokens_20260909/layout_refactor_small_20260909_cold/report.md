# Cold Gluon versus FlyDSL GEMM1 results

Status: **complete**; 128/128 processes recorded; 0 failures.

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
| Commit / branch | `371c2b5bc7e9408bae21eb20e30e8352461e2f65` / `gluon-moe-gfx950-perf-rerun-2026-08-18` |
| Tracked tree | modified |
| Python | `3.14.4 (main, Jun 18 2026, 14:25:02) [GCC 15.2.0]` |
| Torch / HIP / Triton | `2.15.0.dev20260816+rocm7.14` / `7.14.60850` / `3.8.0` |

## Median winner matrix

A cell shows the lower-latency backend and its speedup over the other backend.

| Precision | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|
| BF16 | FlyDSL 1.006x | Gluon 1.002x | Gluon 1.021x | Gluon 1.469x |
| A4W4 | FlyDSL 1.008x | FlyDSL 1.006x | Gluon 1.146x | FlyDSL 1.027x |
| A8W4 | FlyDSL 1.669x | FlyDSL 1.590x | FlyDSL 1.416x | Gluon 1.011x |
| A8W8 | FlyDSL 1.187x | FlyDSL 1.166x | FlyDSL 1.208x | FlyDSL 1.013x |

## Backend results

| Precision | Tokens | Backend | Samples | Mean (us) | Median (us) | P99 (us) |
|---|---:|---|---:|---:|---:|---:|
| BF16 | 16 | flydsl | 400 | 437.658 | 437.583 | 447.884 |
| BF16 | 16 | gluon | 400 | 442.713 | 440.423 | 451.083 |
| BF16 | 64 | flydsl | 400 | 442.773 | 442.903 | 453.123 |
| BF16 | 64 | gluon | 400 | 442.208 | 441.843 | 452.524 |
| BF16 | 256 | flydsl | 400 | 510.198 | 510.444 | 522.324 |
| BF16 | 256 | gluon | 400 | 500.466 | 500.124 | 514.164 |
| BF16 | 1024 | flydsl | 400 | 1052.179 | 1050.108 | 1083.568 |
| BF16 | 1024 | gluon | 400 | 715.895 | 714.825 | 740.965 |
| A4W4 | 16 | flydsl | 400 | 133.558 | 134.881 | 141.601 |
| A4W4 | 16 | gluon | 400 | 134.345 | 136.021 | 142.241 |
| A4W4 | 64 | flydsl | 400 | 136.257 | 137.441 | 143.281 |
| A4W4 | 64 | gluon | 400 | 136.304 | 138.201 | 143.481 |
| A4W4 | 256 | flydsl | 400 | 161.382 | 160.142 | 177.921 |
| A4W4 | 256 | gluon | 400 | 139.689 | 139.761 | 144.161 |
| A4W4 | 1024 | flydsl | 400 | 233.244 | 232.262 | 250.522 |
| A4W4 | 1024 | gluon | 400 | 238.856 | 238.462 | 251.961 |
| A8W4 | 16 | flydsl | 400 | 122.001 | 120.061 | 134.601 |
| A8W4 | 16 | gluon | 400 | 200.540 | 200.401 | 204.722 |
| A8W4 | 64 | flydsl | 400 | 127.665 | 129.521 | 138.201 |
| A8W4 | 64 | gluon | 400 | 206.074 | 206.002 | 209.962 |
| A8W4 | 256 | flydsl | 400 | 138.412 | 138.161 | 143.121 |
| A8W4 | 256 | gluon | 400 | 195.837 | 195.681 | 201.962 |
| A8W4 | 1024 | flydsl | 400 | 319.266 | 318.483 | 342.122 |
| A8W4 | 1024 | gluon | 400 | 315.150 | 315.022 | 328.162 |
| A8W8 | 16 | flydsl | 400 | 231.785 | 229.822 | 244.642 |
| A8W8 | 16 | gluon | 400 | 275.055 | 272.882 | 287.522 |
| A8W8 | 64 | flydsl | 400 | 236.876 | 238.322 | 246.841 |
| A8W8 | 64 | gluon | 400 | 278.073 | 277.862 | 283.562 |
| A8W8 | 256 | flydsl | 400 | 249.248 | 247.462 | 261.041 |
| A8W8 | 256 | gluon | 400 | 297.041 | 298.982 | 312.162 |
| A8W8 | 1024 | flydsl | 400 | 395.707 | 395.603 | 407.243 |
| A8W8 | 1024 | gluon | 400 | 400.633 | 400.843 | 408.843 |

## Pairwise comparison

Positive delta and G/F ratio above 1 mean FlyDSL is faster.

| Precision | Tokens | Statistic | Gluon (us) | FlyDSL (us) | Delta G-F (us) | G/F ratio | Winner |
|---|---:|---|---:|---:|---:|---:|---|
| BF16 | 16 | mean | 442.713 | 437.658 | +5.054 | 1.0115x | FlyDSL |
| BF16 | 16 | median | 440.423 | 437.583 | +2.840 | 1.0065x | FlyDSL |
| BF16 | 16 | p99 | 451.083 | 447.884 | +3.199 | 1.0071x | FlyDSL |
| BF16 | 64 | mean | 442.208 | 442.773 | -0.564 | 0.9987x | Gluon |
| BF16 | 64 | median | 441.843 | 442.903 | -1.060 | 0.9976x | Gluon |
| BF16 | 64 | p99 | 452.524 | 453.123 | -0.599 | 0.9987x | Gluon |
| BF16 | 256 | mean | 500.466 | 510.198 | -9.733 | 0.9809x | Gluon |
| BF16 | 256 | median | 500.124 | 510.444 | -10.320 | 0.9798x | Gluon |
| BF16 | 256 | p99 | 514.164 | 522.324 | -8.160 | 0.9844x | Gluon |
| BF16 | 1024 | mean | 715.895 | 1052.179 | -336.284 | 0.6804x | Gluon |
| BF16 | 1024 | median | 714.825 | 1050.108 | -335.282 | 0.6807x | Gluon |
| BF16 | 1024 | p99 | 740.965 | 1083.568 | -342.603 | 0.6838x | Gluon |
| A4W4 | 16 | mean | 134.345 | 133.558 | +0.786 | 1.0059x | FlyDSL |
| A4W4 | 16 | median | 136.021 | 134.881 | +1.140 | 1.0085x | FlyDSL |
| A4W4 | 16 | p99 | 142.241 | 141.601 | +0.640 | 1.0045x | FlyDSL |
| A4W4 | 64 | mean | 136.304 | 136.257 | +0.047 | 1.0003x | FlyDSL |
| A4W4 | 64 | median | 138.201 | 137.441 | +0.760 | 1.0055x | FlyDSL |
| A4W4 | 64 | p99 | 143.481 | 143.281 | +0.200 | 1.0014x | FlyDSL |
| A4W4 | 256 | mean | 139.689 | 161.382 | -21.693 | 0.8656x | Gluon |
| A4W4 | 256 | median | 139.761 | 160.142 | -20.381 | 0.8727x | Gluon |
| A4W4 | 256 | p99 | 144.161 | 177.921 | -33.760 | 0.8103x | Gluon |
| A4W4 | 1024 | mean | 238.856 | 233.244 | +5.613 | 1.0241x | FlyDSL |
| A4W4 | 1024 | median | 238.462 | 232.262 | +6.200 | 1.0267x | FlyDSL |
| A4W4 | 1024 | p99 | 251.961 | 250.522 | +1.439 | 1.0057x | FlyDSL |
| A8W4 | 16 | mean | 200.540 | 122.001 | +78.540 | 1.6438x | FlyDSL |
| A8W4 | 16 | median | 200.401 | 120.061 | +80.341 | 1.6692x | FlyDSL |
| A8W4 | 16 | p99 | 204.722 | 134.601 | +70.121 | 1.5210x | FlyDSL |
| A8W4 | 64 | mean | 206.074 | 127.665 | +78.409 | 1.6142x | FlyDSL |
| A8W4 | 64 | median | 206.002 | 129.521 | +76.481 | 1.5905x | FlyDSL |
| A8W4 | 64 | p99 | 209.962 | 138.201 | +71.761 | 1.5193x | FlyDSL |
| A8W4 | 256 | mean | 195.837 | 138.412 | +57.425 | 1.4149x | FlyDSL |
| A8W4 | 256 | median | 195.681 | 138.161 | +57.520 | 1.4163x | FlyDSL |
| A8W4 | 256 | p99 | 201.962 | 143.121 | +58.841 | 1.4111x | FlyDSL |
| A8W4 | 1024 | mean | 315.150 | 319.266 | -4.116 | 0.9871x | Gluon |
| A8W4 | 1024 | median | 315.022 | 318.483 | -3.461 | 0.9891x | Gluon |
| A8W4 | 1024 | p99 | 328.162 | 342.122 | -13.960 | 0.9592x | Gluon |
| A8W8 | 16 | mean | 275.055 | 231.785 | +43.270 | 1.1867x | FlyDSL |
| A8W8 | 16 | median | 272.882 | 229.822 | +43.060 | 1.1874x | FlyDSL |
| A8W8 | 16 | p99 | 287.522 | 244.642 | +42.880 | 1.1753x | FlyDSL |
| A8W8 | 64 | mean | 278.073 | 236.876 | +41.197 | 1.1739x | FlyDSL |
| A8W8 | 64 | median | 277.862 | 238.322 | +39.540 | 1.1659x | FlyDSL |
| A8W8 | 64 | p99 | 283.562 | 246.841 | +36.721 | 1.1488x | FlyDSL |
| A8W8 | 256 | mean | 297.041 | 249.248 | +47.792 | 1.1917x | FlyDSL |
| A8W8 | 256 | median | 298.982 | 247.462 | +51.520 | 1.2082x | FlyDSL |
| A8W8 | 256 | p99 | 312.162 | 261.041 | +51.121 | 1.1958x | FlyDSL |
| A8W8 | 1024 | mean | 400.633 | 395.707 | +4.926 | 1.0124x | FlyDSL |
| A8W8 | 1024 | median | 400.843 | 395.603 | +5.240 | 1.0132x | FlyDSL |
| A8W8 | 1024 | p99 | 408.843 | 407.243 | +1.600 | 1.0039x | FlyDSL |

## Effective configurations and routing padding

Active rows include per-expert padding; extra rows are not useful token/top-k rows.

| Precision | Tokens | Backend | Tile | Key flags | Useful rows | Active rows | Extra | Padding | Blocks | Output layout / policy |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| BF16 | 16 | flydsl | `TM32xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 128 | 1056 | 928 | 725.00% | 33 | sorted / none |
| BF16 | 16 | gluon | `BM32xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| BF16 | 64 | flydsl | `TM16xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 512 | 528 | 16 | 3.12% | 33 | sorted / none |
| BF16 | 64 | gluon | `BM32xBN128xBK128` | MFMA=[16, 16, 32], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| BF16 | 256 | flydsl | `TM64xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 2048 | 2112 | 64 | 3.12% | 33 | sorted / none |
| BF16 | 256 | gluon | `BM64xBN256xBK64` | MFMA=[32, 32, 16], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| BF16 | 1024 | flydsl | `TM64xTN128xTK128` | wpe=1, b_nt=2, nt=None, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 132 | sorted / none |
| BF16 | 1024 | gluon | `BM128xBN256xBK64` | MFMA=[32, 32, 16], warps=[2, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| A4W4 | 16 | flydsl | `TM32xTN256xTK256` | wpe=None, b_nt=2, nt=True, gate=separated, async=True | 128 | 1056 | 928 | 725.00% | 33 | sorted / round_up |
| A4W4 | 16 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4], NT=1 | 128 | 2112 | 1984 | 1550.00% | 33 | dense expert-sorted rows / even |
| A4W4 | 64 | flydsl | `TM32xTN256xTK256` | wpe=None, b_nt=2, nt=True, gate=separated, async=True | 512 | 1056 | 544 | 106.25% | 33 | sorted / round_up |
| A4W4 | 64 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4], NT=1 | 512 | 2112 | 1600 | 312.50% | 33 | dense expert-sorted rows / even |
| A4W4 | 256 | flydsl | `TM32xTN256xTK256` | wpe=None, b_nt=2, nt=True, gate=separated, async=True | 2048 | 2112 | 64 | 3.12% | 66 | sorted / round_up |
| A4W4 | 256 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, B-preshuffle, MFMA=[16, 16, 128], warps=[1, 4], NT=1 | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / even |
| A4W4 | 1024 | flydsl | `TM64xTN256xTK256` | wpe=None, b_nt=0, nt=False, gate=separated, async=True | 8192 | 8448 | 256 | 3.12% | 132 | sorted / round_up |
| A4W4 | 1024 | gluon | `BM128xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4], NT=0 | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / even |
| A8W4 | 16 | flydsl | `TM32xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 128 | 1056 | 928 | 725.00% | 33 | token_slot / none |
| A8W4 | 16 | gluon | `BM32xBN256xBK256` | MFMA=[16, 16, 128], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| A8W4 | 64 | flydsl | `TM32xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 512 | 1056 | 544 | 106.25% | 33 | token_slot / none |
| A8W4 | 64 | gluon | `BM32xBN256xBK256` | MFMA=[16, 16, 128], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| A8W4 | 256 | flydsl | `TM64xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 2048 | 2112 | 64 | 3.12% | 33 | token_slot / none |
| A8W4 | 256 | gluon | `BM64xBN256xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| A8W4 | 1024 | flydsl | `TM128xTN128xTK256` | wpe=4, b_nt=2, nt=None, gate=interleave, async=True | 8192 | 8448 | 256 | 3.12% | 66 | token_slot / none |
| A8W4 | 1024 | gluon | `BM128xBN128xBK256` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[2, 2] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |
| A8W8 | 16 | flydsl | `TM32xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=interleave, async=True | 128 | 1056 | 928 | 725.00% | 33 | token_slot / none |
| A8W8 | 16 | gluon | `BM32xBN128xBK256` | MFMA=[16, 16, 128], warps=[1, 4] | 128 | 1056 | 928 | 725.00% | 33 | dense expert-sorted rows / none |
| A8W8 | 64 | flydsl | `TM32xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=interleave, async=True | 512 | 1056 | 544 | 106.25% | 33 | token_slot / none |
| A8W8 | 64 | gluon | `BM32xBN128xBK256` | MFMA=[16, 16, 128], warps=[1, 4] | 512 | 1056 | 544 | 106.25% | 33 | dense expert-sorted rows / none |
| A8W8 | 256 | flydsl | `TM64xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=interleave, async=True | 2048 | 2112 | 64 | 3.12% | 33 | token_slot / none |
| A8W8 | 256 | gluon | `BM64xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 2048 | 2112 | 64 | 3.12% | 33 | dense expert-sorted rows / none |
| A8W8 | 1024 | flydsl | `TM128xTN128xTK256` | wpe=2, b_nt=2, nt=None, gate=interleave, async=True | 8192 | 8448 | 256 | 3.12% | 66 | token_slot / none |
| A8W8 | 1024 | gluon | `BM128xBN256xBK128` | A-scale-shuffle, B-scale-shuffle, MFMA=[16, 16, 128], warps=[1, 4] | 8192 | 8448 | 256 | 3.12% | 66 | dense expert-sorted rows / none |

## Validation

Cold measurements are gated by the local
`layout_refactor_small_20260909_validation/` artifact. The timed kernel, full
logical inputs, and output hashes must match validation.

| Precision | Backend | Passed cells | Replays/cell | Worst max normalized error | Worst RMS normalized error |
|---|---|---:|---:|---:|---:|
| BF16 | flydsl | 4/4 | 16 | 0.00389342 | 0.000860886 |
| BF16 | gluon | 4/4 | 16 | 0.00389342 | 0.000860885 |
| A4W4 | flydsl | 4/4 | 16 | 0 | 0 |
| A4W4 | gluon | 4/4 | 16 | 0 | 0 |
| A8W4 | flydsl | 4/4 | 16 | 0.00390193 | 0.000785945 |
| A8W4 | gluon | 4/4 | 16 | 0.00390193 | 0.000785945 |
| A8W8 | flydsl | 4/4 | 16 | 0.0039697 | 0.000860354 |
| A8W8 | gluon | 4/4 | 16 | 0.0039697 | 0.000860354 |

## Interpretation notes

A4W4 native output quantization uses round-to-even scales in Gluon and round-up scales in FlyDSL; each backend is checked against the matching independent reference. Gluon emits expert-sorted rows, while FlyDSL may emit token-slot or sorted native layouts; layout conversion is excluded because only the prepared GEMM1 is measured.

## Artifacts

The local result directory also contains `plan.json`, `provenance.json`,
`gpu_inventory.json`, `summary.json`, `comparison.json`, `records.json`,
`samples.csv`, `samples.json`, `failures.json`, and
`independent_audit.json`. Those raw benchmark artifacts are intentionally not
part of the findings commit.
