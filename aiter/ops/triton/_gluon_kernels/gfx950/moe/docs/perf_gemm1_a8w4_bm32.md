# gfx950 MoE GEMM1: MXFP8 x MXFP4 at block_m 16 and 32

Measured 2026-11-04 on MI355X (gfx950), HIP device 3. Base: commit `b38920e09`.

## Workload

- Model dsv4-flash: hidden 4096, intermediate 2048, 256 experts, top-k 6.
- GEMM1 (gate/up projection with the SwiGLU epilogue): N = 4096, K = 4096.
- MXFP8 activations, MXFP4 weights, BF16 output.
- T = 1024 tokens, even routing: every expert gets 24 rows. The router picks
  block_m 32, so each expert has one M block.

Each expert's weights are read once, by one M block. The kernel reads 2.15 GB of
weight payload and 0.13 GB of weight scales per call. It is limited by the weight
stream, not by the MFMAs.

## Method

- Warm timing: `aiter.test_common.run_perftest(fn, num_iters=20, num_warmup=3)`. This
  is the sum of the kernel times of one call.
- One process times all the configurations, in alternating order, for 3 rounds.
- FlyDSL: `../aiter` at `dac620634`, stage-1 call only, same routing.
- HBM bytes and L2 hit rate: rocprofv3 PMC counters on HIP 3, all kernels back to
  back on the idle GPU. HBM bytes are the `FETCH_SIZE` terms
  (`TCC_EA0_RDREQ_{32B,64B,128B}_sum`); L2 hit is `TCC_HIT / (TCC_HIT + TCC_MISS)`.
  Each value is the median of the last 10 dispatches, after at least 3 warm ones.

## Cause of the gap

The gap was the configuration, not the kernel code.

No GEMM1 override matched this shape, so the launch used the generic ladder for
block_m 32: BLOCK_N 128, BLOCK_K 512, B through LDS, unshuffled B scales with `.cg`.
That tile has two problems:

1. **Excess HBM reads from unshuffled B scales with `.cg`.** The tile reads
   2.93 GB from HBM per call, 1.26x the ideal 2.33 GB. Each of two changes removes
   the excess: shuffled B scales (still with `.cg`), or the same unshuffled scales
   without `.cg`. So the excess needs both the unshuffled layout and the `.cg`
   policy. On CDNA4, `.cg` on a load sets SC0 and NT. The counters show the excess
   bytes but not the mechanism in L2. One explanation that fits, not verified: a K
   step reads 16 of the 128 scale bytes in a line, and with NT the line leaves L2
   before the next K step reads it.
2. **One CTA per CU.** The tile uses 156 KiB of LDS. With few rows per expert, one
   CTA cannot keep enough weight bytes in flight.

PMC counters, HIP 3:

| kernel | HBM read per call | vs ideal | L2 hit | L1 to L2 read requests |
|---|--:|--:|--:|--:|
| F0, ladder tile (below) | 2.931 GB | 1.258x | 31.8% | 33.4 M |
| F0 without `.cg` on the B scales | 2.309 GB | 0.991x | 46.1% | 33.4 M |
| F1, shuffled B scales with `.cg` | 2.307 GB | 0.990x | 31.0% | 26.0 M |
| F6, new tile | 2.310 GB | 0.991x | 20.6% | 22.6 M |
| FlyDSL stage 1, BF16 out | 2.428 GB | 1.042x | 24.7% | 25.0 M |

In one process, 4 rounds: F0 621.3 µs, F0 without `.cg` 535.4 µs (-13.8%), F1
517.4 µs (-16.7%), F6 430.8 µs.

Factor steps, one change for each step. µs, mean of 3 rounds, output bit-identical
in each step:

| step | change | µs | delta |
|---|---|--:|--:|
| F0 | ladder tile (production before this change) | 613.8 | |
| F1 | shuffled B scales | 515.5 | -16.0% |
| F2 | BLOCK_K 256: 78 KiB of LDS, two CTAs per CU | 481.1 | -6.7% |
| F3 | B in registers (preshuffled weights) | 462.4 | -3.9% |
| F4 | no `.cg` on the shuffled B scales | 450.1 | -2.7% |
| F5 | BLOCK_N 256, two N slots of 128 (260 VGPRs: one CTA per CU) | 452.1 | +0.4% |
| F6 | WAVES_PER_EU 2 (190 VGPRs: two CTAs per CU) | 430.2 | -4.8% |

F6 is the new tile. On F6, `.cg` on the B scales costs +5.3%, and B through LDS
(132 KiB of LDS, one CTA per CU) costs +7.6%. These knobs on top of F6 change the
time by less than 2%: SLOT_N 256, per-stage commits, GROUP_M 1, `SCHED_MODE` 4,
A reads in the MFMA region, four buffers.

## Result

Production path (`gluon_perftest.py`, `STAGE=1`, `N_ACTIVE=max`), four rounds, a fresh
process for each arm, base and new in alternating order, µs:

| cell | base | new | new/base |
|---|--:|--:|--:|
| a8w4 dsv4-flash T=1024 | 617.9 | 431.3 | 0.698 |
| a8w4 dsv4-pro T=1024 | 2083.1 | 1644.3 | 0.789 |

FlyDSL stage 1 in the same rounds: 454.0 µs with BF16 output
(`t32x128x256_w4_gui`, its best), 435.8 µs with fused FP8 output (`w3_gui_fp8`).
Gluon/FlyDSL with the same BF16 output is 0.950. The same 4-round A/B with the tiles
read from the JSON family (same effective config): base 612.7 µs, new 431.1 µs,
FlyDSL best BF16 451.4 µs (`w3_gui`), Gluon/FlyDSL 0.955.

The Gluon number includes a `_reduce_grouped` kernel of 7.0-7.5 µs. The benchmark
passes `scatter_indx` to GEMM1, so the wrapper sums the top-k rows after the GEMM.
The FlyDSL stage-1 time has no such kernel.

Both kernels now read about the ideal HBM bytes (Gluon 0.99x, FlyDSL 1.04x). Both
run 4 waves per CTA: FlyDSL with 144 VGPRs and 33 KB of LDS, Gluon with 190 VGPRs
and 34 KB of LDS.

## Cost

B in registers reads a preshuffled copy of the weights. The launch keeps this copy
for the weight tensor, so the GEMM1 weights use twice the memory. The block_m 128
override for MXFP8 x MXFP4 already keeps the same copy, so a model that also runs
prefill at block_m 128 has no additional memory cost.

## block_m 16

T = 8 and T = 32 in the model matrix route to block_m 16. The ladder tile there is
BLOCK_N 128, BLOCK_K 512, B through 130 KiB of LDS: also one CTA per CU, with
unshuffled `.cg` B scales. The same change helps. Only a few experts are active, so
the narrow tile is better there: BLOCK_N 128, two N slots of 64, B in registers
(136 VGPRs, three CTAs per CU). It is 1-2% faster than the block_m 32 tile at
block_m 16. µs, mean of 2 rounds in one process, HIP 3, `N_ACTIVE=max`:

| cell | ladder | new | delta |
|---|--:|--:|--:|
| a8w4 dsv4-flash T=32 | 385.1 | 300.2 | -22.0% |
| a8w4 glm52-base T=32 | 756.7 | 586.2 | -22.5% |
| a8w4 dsv4-pro T=8 | 249.7 | 194.6 | -22.1% |
| a8w4 minimax-m3 T=8 | 145.0 | 115.0 | -20.7% |

## Model matrix

Production path, GEMM1 of the 72 model cells (`gluon_perftest.py`, `STAGE=1`), one
pass with even routing and one with random routing, base and new in separate
processes, HIP 3. The two entries change the launch config of exactly the 10 cells
below. The other 62 cells and all GEMM2 launches keep their config, and their output
is bit-identical. µs:

| cell | even base | even new | even new/base | random base | random new | random new/base |
|---|--:|--:|--:|--:|--:|--:|
| a8w4 dsv4-flash T=8 | 95.5 | 79.9 | 0.836 | 91.7 | 75.8 | 0.827 |
| a8w4 dsv4-flash T=32 | 380.3 | 299.4 | 0.787 | 277.5 | 213.9 | 0.771 |
| a8w4 dsv4-flash T=1024 | 613.6 | 434.0 | 0.707 | 643.0 | 448.8 | 0.698 |
| a8w4 dsv4-pro T=8 | 248.8 | 200.4 | 0.805 | 243.5 | 196.4 | 0.807 |
| a8w4 dsv4-pro T=32 | 986.4 | 780.3 | 0.791 | 790.5 | 612.2 | 0.774 |
| a8w4 dsv4-pro T=1024 | 2098.0 | 1601.6 | 0.763 | 2122.6 | 1656.9 | 0.781 |
| a8w4 glm52-base T=8 | 191.7 | 151.7 | 0.791 | 179.9 | 141.1 | 0.784 |
| a8w4 glm52-base T=32 | 758.1 | 584.6 | 0.771 | 475.8 | 362.9 | 0.763 |
| a8w4 minimax-m3 T=8 | 143.3 | 115.9 | 0.808 | 135.5 | 108.0 | 0.797 |
| a8w4 minimax-m3 T=32 | 576.2 | 391.8 | 0.680 | 403.6 | 303.6 | 0.752 |

One pass of the full matrix in one process moves some unchanged cells by up to
18% (17 cells above +2% or below -2% in the two passes). Re-timed alone, one cell per
process, in 4 base/new pairs in alternating order, the median new/base ratio of each
is 0.980-1.005. One T=1 cell read 1.032 in its first 4 pairs and 1.002 in 8 more; its
kernel is the same binary as at the base.

## Scope

The tiles are the `A8W4-GEMM1` family,
`aiter/ops/triton/configs/gfx950/gluon/moe/a8w4_gemm1/DEFAULT.json`: `bm16_any` and
`bm32_any`. `_default_launch_config` applies an entry to GEMM1 with MXFP8 activations
and MXFP4 weights when the grid is not small (T = 1 keeps the ladder), when the
launch can read the host's weight copies (not in graph capture, not for inference or
padded tensors), and not at N 4096 with K 7168: that shape keeps its own tiles. The
family code keeps the ladder where N or K is not a multiple of 256 or the entry
cannot launch (for example K below 7 x 256: the pipeline needs 7 K steps).
