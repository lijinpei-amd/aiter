# gfx950 Gluon MoE: performance status

Where the Gluon a4w4 MoE grouped GEMM stands against the in-tree Triton kernel and the
tuned CK/FlyDSL kernels, what was fixed to get there, and what is still open.

Measurements are kernel times from `rocprofv3` on H=7168, I=2048, E=33, topk=8, with
routing aligned across harnesses. Full tables, methodology and caveats:
[gluon_moe_gfx950_perf.md](gluon_moe_gfx950_perf.md).

## Status

Gluon against the best of {Triton, tuned FlyDSL/CK} at each point. **Bold** = faster.
TFLOP/s counts *useful* FLOPs (`2 * T*topk * N * K`), not the padded MFMA work; GB/s is
measured HBM traffic (`TCC_MISS_sum * 128 B`), against ~8 TB/s of peak.

**Decode**

| T | stage | Gluon µs | TFLOP/s | GB/s | best other | µs | TFLOP/s | GB/s |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1 | **27.3** | 17 | 4,579 | Triton | 32.0 | 15 | 3,906 |
| 1 | 2 | 16.0 | 15 | 3,925 | Triton *(tie)* | 16.0 | 15 | 3,925 |
| 8 | 1 | **99.0** | 38 | 5,209 | FlyDSL | 109.5 | 34 | 4,743 |
| 8 | 2 | 52.1 | 36 | 4,979 | Triton | **50.7** | 37 | 5,116 |
| 32 | 1 | 100.9 | 149 | 5,122 | FlyDSL | **98.5** | 153 | 5,278 |
| 32 | 2 | **52.7** | 143 | 4,981 | Triton | 55.5 | 135 | 4,730 |

**Prefill**

| T | stage | Gluon µs | TFLOP/s | GB/s | best other | µs | TFLOP/s | GB/s |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1024 | 1 | 291.2 | 1,652 | 2,087 | FlyDSL | **223.4** | 2,153 | 3,183 |
| 1024 | 2 | **157.2** | 1,530 | 2,665 | FlyDSL | 180.9 | 1,330 | 4,916 |
| 4096 | 1 | 942.3 | 2,042 | 1,610 | FlyDSL | **651.4** | 2,954 | 2,423 |
| 4096 | 2 | **523.9** | 1,836 | 2,067 | FlyDSL | 526.1 | 1,829 | 2,880 |

- **Decode is bandwidth-bound** -- everything runs at 3.9-5.3 TB/s and 15-153 TFLOP/s, so
  the ranking tracks HBM traffic and little else. Gluon is ahead at 3 of 6 points, tied at
  1, behind by 2.4-2.8% at the other 2, and its traffic is within 1% of compulsory
  everywhere, matching Triton and FlyDSL.
- **Prefill is compute-bound** -- 1,530-2,954 TFLOP/s at only 1.6-4.9 TB/s. Gluon leads
  stage 2 at both points and trails FlyDSL on stage 1 by 1.30x (T=1024) and 1.45x (T=4096).
  That stage-1 deficit is the one real gap; against Triton, Gluon leads everywhere at
  prefill by 1.18-1.42x.
- FlyDSL's higher GB/s at prefill is not an advantage -- it moves *more* bytes for the same
  math (711 vs 608 MB at T=1024 stage 1) and is still faster, which is what makes prefill a
  scheduling problem rather than a memory one.

## Fixed this round

| change | effect |
|---|---|
| `expert_scale_mod`: drop `.cg` from weight-scale loads | decode gemm1 130.0 -> 100.9 us; HBM 651 -> 517 MB; L2 hit 16% -> 33% |
| `NUM_LDS_BUFFER` 3 -> 2 at `block_m=16` | decode gemm2 57.8 -> 51.2 us; LDS 120,720 -> 80,464 B (1 -> 2 CTA/CU) |
| warp split derived from the tile instead of hardcoded | 424.6 -> 362.9 us at `BLOCK_M=128` stage 1 |
| `_pick_warps` N-alignment guard | fixes a hang: an unaligned split made the `mini_n` walk never terminate |

The scale-load fix is the significant one. The scale tensor is `(E, K/32, N)` with K
contiguous, so a 128 B line holds 128 consecutive K-scales while a `BLOCK_K=512` stage
consumes 16 of them -- the line must survive in L2 across 8 K-iterations, and `.cg` was
turning each touch into its own HBM fetch. `.cg` on the weight *payload* is correct and
stays; removing it costs 10%.

## Open, ranked

1. **Prefill stage-1 data-wait stalls.** Gluon's waves wait on data 1.56x (T=1024) to
   1.97x (T=4096) as many cycles as FlyDSL's and issue on 15.7-17.5% of cycles against
   18.5-25.9%. Two `BLOCK_K=256` stages in flight cannot cover the latency and there is no
   LDS left at `BLOCK_N=256` to go deeper. Needs more K in flight per byte of LDS, which is
   a pipeline restructure, not another ladder entry. Worth ~1.4x on prefill stage 1.
2. **Fused fp4 epilogue is slower than emitting bf16** at every T >= 8 (320.7 vs 291.2 us
   at T=1024) despite moving less traffic. Fusing the output quant should be close to free;
   ours costs more than the write it saves. Self-contained, in the epilogue.
3. **Decode host launch overhead** -- ~90 us of host time against a ~27 us kernel, so
   wall-clock at decode is launch-bound and hides every kernel gain above. Not a kernel
   problem; the launch path is.
4. **Gammas reloaded per N-mini-tile** (~4x redundant epilogue traffic). Found in review,
   never fixed.
5. **a16w4 (bf16 x MXFP4) blocked upstream** -- `amdg.scaled_upcast_fp4` has no working
   lowering in this Triton revision; the pair is refused rather than miscompiled.

## Ruled out -- do not re-investigate

Each of these was measured and rejected; the numbers are in the perf doc.

| hypothesis | verdict |
|---|---|
| tile -> XCD mapping (`TILE_SCHED`, `GROUP_M`) | < 0.5% across LINEAR / GROUP_M / XCD_GROUP_M and GROUP_M 1/4/8/16 |
| low occupancy at decode | 4 CTAs/CU is 16% *slower* than 1; 8 CTAs/CU is 2x slower |
| low occupancy at prefill | 3-4 CTAs/CU is 2.2-2.5x slower (needs `BK=128`, which doubles per-stage overhead) |
| barrier count (rotate refill off the consumed buffer) | halves barriers but costs a buffer of depth; worse at equal depth |
| M-padding at prefill | 1.031x, and FlyDSL at `tile_m=64` pays the same 1.031x |
| output format (bf16 vs fused fp4) | making it like-for-like *widens* the prefill gap |
| relaxing the 64-lane direct-to-LDS scale threshold | backend cannot lower a 32-lane copy; `BLOCK_K=512` at `block_m=16` is forced |
| copying Triton's cache modifiers | Triton passes the same `.cg` to its scale loads and is fine; the modifier is honoured differently on `buffer_load_to_shared` than on a register `tl.load` |

## Reproducing

```bash
python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py \
  --op a4w4 --shape 7168,2048,33,8 --tokens 8 32 1024 4096 --n-active 33
```

Take kernel times from a `rocprofv3 --kernel-trace` of that command, not the benchmark's
wall-clock columns (see open item 3). `--n-active` is required for any comparison against
`op_tests/test_moe_2stage.py`: without it the two harnesses activate different numbers of
experts and are not measuring the same work.
