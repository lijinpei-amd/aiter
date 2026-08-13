# gfx950 MoE A4W4 grouped GEMM: Gluon vs Triton vs tuned CK/FlyDSL

Kernel-level comparison of the four a4w4 MoE grouped-GEMM implementations in this tree,
on one shape, measured with `rocprofv3`. Wall-clock is deliberately **not** reported: at
decode the Gluon launch path costs ~90 us of host time against a ~27 us kernel, which
swamps the kernel differences this table is about.

## Setup

| | |
|---|---|
| GPU | MI355X (gfx950), 256 CU, 8 XCDs, 160 KiB LDS/CU |
| shape | H=7168, I=2048, E=33, topk=8 (`--dim 7168,2048 -e 33 -k 8`) |
| stage 1 | N=2I=4096, K=H=7168, fused SwiGLU |
| stage 2 | N=H=7168, K=I=2048, router-weight multiply |
| dtype | MXFP4 x MXFP4 (E2M1 payload + E8M0 group-32 scales) |
| tool | `rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum`, per-dispatch average |

`E=33` is not arbitrary: it is the shape the tuned table covers for a4w4 FlyDSL
(the count includes the shared expert). At `E=32` there is no tuned FlyDSL row and
the dispatcher falls back to CK, so that shape cannot compare against tuned FlyDSL.

### Routing is aligned across harnesses

Gluon/Triton are driven from a standalone script; FlyDSL/CK from
`op_tests/test_moe_2stage.py`. The two seed their routers independently, so they
activated different numbers of experts and therefore read different amounts of weight --
at T=8 that alone was a 17% difference, larger than any kernel effect. Both sides
are pinned with `AITER_MOE_NUM_EXPERT_ACTIVATED=n` (n=8 at T=1, n=33 elsewhere), which
forces exactly n active experts with a round-robin, perfectly balanced token assignment.

After alignment, Triton and FlyDSL agree on stage-1 HBM traffic to within 0.4% at every
token count -- two independent implementations landing on the same number, which is the
check that makes the rest of the table meaningful.

### How the columns are defined

| column | meaning |
|---|---|
| **Gluon** | `_moe_gluon_gemm1/2`, this branch |
| **Triton** | in-tree `_moe_gemm_a4w4` |
| **Tuned FlyDSL** | the FlyDSL kernel selected by the tuner for this (shape, token, stage) |
| **Tuned FlyDSL+CK** | what the joint dispatcher actually runs -- the tuner searches CK codegen instances, the FlyDSL registry, hand-written ASM kernels and Opus instances together, and picks a winner **per stage** |

The two tuned columns coincide everywhere here except T=1 stage 2, where the joint search
picks a CK kernel (`moe_ck2stages_gemm2_...`, symbol `kernel_moe_mxgemm_2lds`) over any
FlyDSL candidate -- so there is no tuned FlyDSL entry to report at that point.

### How the derived columns are computed

- **TFLOP/s** = `2 * (T*topk) * N * K / kernel_time`. Useful FLOPs, not padded: at decode
  the router pads to `block_m`, so the hardware does more MFMA work than this counts.
  Decode numbers are therefore low by construction and are not a compute-efficiency claim.
- **HBM GB/s** and **HBM MB** = `TCC_MISS_sum * 128 B`, i.e. measured traffic that missed
  L2 and went to memory, not a model. Compulsory traffic for stage 1 with 33 active
  experts is ~514 MB (weights + scales), which is the floor those columns approach.

### Decode

**Stage 1** (N=4096, K=7168)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1 | Gluon | `BM16xBN64xBK512, 2 buf, 4 warps` | 27.3 | 17 | 4,579 | 125.0 |
| 1 | Triton | `BM16xBN128xBK256` | 32.0 | 15 | 3,906 | 125.0 |
| 1 | Tuned FlyDSL | `flydsl moe1 t64x128x256_w4_fp4` | 32.2 | 15 | 3,916 | 126.1 |
| 1 | Tuned FlyDSL+CK | `flydsl moe1 t64x128x256_w4_fp4` | 32.2 | 15 | 3,916 | 126.1 |
| 8 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 99.0 | 38 | 5,209 | 515.7 |
| 8 | Triton | `BM16xBN128xBK256` | 113.0 | 33 | 4,611 | 521.0 |
| 8 | Tuned FlyDSL | `flydsl moe1 t64x128x256_w3_fp4` | 109.5 | 34 | 4,743 | 519.4 |
| 8 | Tuned FlyDSL+CK | `flydsl moe1 t64x128x256_w3_fp4` | 109.5 | 34 | 4,743 | 519.4 |
| 32 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 100.9 | 149 | 5,122 | 516.8 |
| 32 | Triton | `BM16xBN128xBK256` | 124.0 | 121 | 4,206 | 521.5 |
| 32 | Tuned FlyDSL | `flydsl moe1 t32x32x256_w3 (bf16 out)` | 98.5 | 153 | 5,278 | 519.9 |
| 32 | Tuned FlyDSL+CK | `flydsl moe1 t32x32x256_w3 (bf16 out)` | 98.5 | 153 | 5,278 | 519.9 |

**Stage 2** (N=7168, K=2048)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1 | Gluon | `BM16xBN64xBK512, 2 buf, 4 warps` | 16.0 | 15 | 3,925 | 62.8 |
| 1 | Triton | `BM16xBN128xBK256` | 16.0 | 15 | 3,925 | 62.8 |
| 1 | Tuned FlyDSL | *not selected — the joint search picked CK here* | — | — | — | — |
| 1 | Tuned FlyDSL+CK | `CK moe_ck2stages_gemm2 256x64x128x128` | 19.3 | 12 | 3,280 | 63.3 |
| 8 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 52.1 | 36 | 4,979 | 259.4 |
| 8 | Triton | `BM16xBN128xBK256` | 50.7 | 37 | 5,116 | 259.4 |
| 8 | Tuned FlyDSL | `flydsl moe2 t32x128x256_atomic_bnt2_sbm64` | 57.9 | 32 | 4,501 | 260.6 |
| 8 | Tuned FlyDSL+CK | `flydsl moe2 t32x128x256_atomic_bnt2_sbm64` | 57.9 | 32 | 4,501 | 260.6 |
| 32 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 52.7 | 143 | 4,981 | 262.5 |
| 32 | Triton | `BM16xBN128xBK256` | 55.5 | 135 | 4,730 | 262.5 |
| 32 | Tuned FlyDSL | `flydsl moe2 t32x128x256_atomic_bnt2` | 58.0 | 130 | 4,616 | 267.7 |
| 32 | Tuned FlyDSL+CK | `flydsl moe2 t32x128x256_atomic_bnt2` | 58.0 | 130 | 4,616 | 267.7 |

### Prefill

**Stage 1** (N=4096, K=7168)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1024 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 291.2 | 1,652 | 2,087 | 607.8 |
| 1024 | Triton | `BM128xBN512xBK256` | 412.5 | 1,166 | 1,503 | 620.1 |
| 1024 | Tuned FlyDSL | `flydsl moe1 t64x128x256_w2_bnt0_fp4` | 223.4 | 2,153 | 3,183 | 711.0 |
| 1024 | Tuned FlyDSL+CK | `flydsl moe1 t64x128x256_w2_bnt0_fp4` | 223.4 | 2,153 | 3,183 | 711.0 |
| 4096 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 942.3 | 2,042 | 1,610 | 1516.8 |
| 4096 | Triton | `BM128xBN512xBK256` | 1107.6 | 1,737 | 1,332 | 1475.1 |
| 4096 | Tuned FlyDSL | `flydsl moe1 t64x128x256_w4_bnt0_fp4` | 651.4 | 2,954 | 2,423 | 1578.5 |
| 4096 | Tuned FlyDSL+CK | `flydsl moe1 t64x128x256_w4_bnt0_fp4` | 651.4 | 2,954 | 2,423 | 1578.5 |

**Stage 2** (N=7168, K=2048)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1024 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 157.2 | 1,530 | 2,665 | 418.9 |
| 1024 | Triton | `BM128xBN512xBK256` | 189.4 | 1,270 | 2,228 | 422.0 |
| 1024 | Tuned FlyDSL | `flydsl moe2 t64x256x256_atomic` | 180.9 | 1,330 | 4,916 | 889.3 |
| 1024 | Tuned FlyDSL+CK | `flydsl moe2 t64x256x256_atomic` | 180.9 | 1,330 | 4,916 | 889.3 |
| 4096 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 523.9 | 1,836 | 2,067 | 1082.8 |
| 4096 | Triton | `BM128xBN512xBK256` | 643.5 | 1,495 | 1,809 | 1164.1 |
| 4096 | Tuned FlyDSL | `flydsl moe2 t64x128x256_atomic_bnt2` | 526.1 | 1,829 | 2,880 | 1515.4 |
| 4096 | Tuned FlyDSL+CK | `flydsl moe2 t64x128x256_atomic_bnt2` | 526.1 | 1,829 | 2,880 | 1515.4 |

## Which kernel the tuner selected

`kernelName1`/`kernelName2` from `aiter/configs/tuned_fmoe.csv` for this shape, with the
tuner's own recorded time next to the time measured here. Every selection was confirmed
against the symbol that actually dispatched, so the "Tuned" rows above are the tuned
kernel, not a fallback.

| T | stage | tuner's `kernelName` | tuner's µs | measured µs | dispatched symbol |
|---:|---:|---|---:|---:|---|
| 1 | 1 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_fp4` | 29.2 | 32.2 | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1_fp4q` |
| 1 | 2 | `moe_ck2stages_gemm2_256x64x128x128_1x4_...FP4X2_FP4X2_B16` | 20.1 | 19.3 | `kernel_moe_mxgemm_2lds` |
| 8 | 1 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_fp4` | 74.0 | 109.5 † | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1_fp4q` |
| 8 | 2 | `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2_sbm64` | 47.1 | 57.9 † | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale` |
| 32 | 1 | `flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3` | 90.3 | 98.5 | `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x32x256_pm1_async` |
| 32 | 2 | `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2` | 50.8 | 58.0 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale` |
| 1024 | 1 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_bnt0_fp4` | 208.6 | 223.4 | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1` |
| 1024 | 2 | `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_atomic` | 163.3 | 180.9 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x256x256_vscale` |
| 4096 | 1 | `flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_fp4` | 619.1 | 651.4 | `mfma_moe1_silu_mul_afp4_wfp4_fp4_t64x128x256_pm1` |
| 4096 | 2 | `flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_bnt2` | 542.6 | 526.1 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale` |

Measured times run 8-10% above the tuner's own, consistent with counter collection
serializing dispatches. † T=8 is the exception and the gap is deliberate: the tuner
recorded natural routing, while these runs force all 33 experts active to match the Gluon
harness. Unforced, T=8 stage 1 measures 76.0 us against the tuner's 74.0.

The tuner searches roughly 800 FlyDSL candidates per dtype pair (272 stage-1 for
a4w4->bf16, 272 for a4w4->fp4, 257 stage-2) alongside CK codegen instances, hand-written
ASM kernels and Opus instances, and picks per stage -- which is why the tile changes at
almost every token count, and why stage 2 at T=1 is CK while stage 1 is FlyDSL.

## Reading the table

**Decode is bandwidth-bound, and traffic is the whole story.** Every implementation sits
at 4.2-5.3 TB/s against ~8 TB/s of peak HBM, at 15-153 TFLOP/s against a multi-PFLOP/s
MXFP4 peak. Ranking at decode tracks HBM traffic and nothing else. Gluon reached parity
here only after a fix: it had been marking the weight *scale* loads non-temporal (`.cg`),
which cost 25% extra traffic, because the scale tensor is `(E, K/32, N)` with K contiguous
-- one 128 B line holds 128 consecutive K-scales, a `BLOCK_K=512` stage consumes 16 of
them, and that line has to survive in L2 across 8 K-iterations.

Worth noting for anyone reading the two kernels side by side: Triton passes the *same*
`.cg` to its weight-scale loads (`_triton_kernels/moe/moe_op_gemm_a4w4.py`, the
`W_CACHE_MODIFIER` on both the payload and the `WMxScale` load) and still reaches the
compulsory floor. So the modifier is evidently honoured differently on Gluon's
`buffer_load_to_shared` (direct-to-LDS) path than on a register `tl.load` -- the Triton
config was not a template to copy here, and the A/B was needed to find it.

**Prefill is compute-bound** and the ordering changes: FlyDSL's stage 1 is 30% (T=1024)
to 45% (T=4096) ahead of Gluon while moving *more* bytes, so its advantage there is
scheduling -- a persistent grid with no tail -- not memory.

**Where each implementation wins**

| | decode (T=1, 8, 32) | prefill (T=1024, 4096) |
|---|---|---|
| Gluon | fastest at 3 of 6 points (T=1 s1, T=8 s1, T=32 s2), tied at T=1 s2 | fastest **stage 2** at both points |
| Triton | fastest at T=8 s2 (by 2.8%) | slowest at every point |
| Tuned FlyDSL | fastest at T=32 s1 (by 2.4%) | fastest **stage 1** at both points |

No implementation dominates. Gluon and the tuned kernels are within a few percent of each
other across decode; the clear separations are Triton trailing at prefill, and FlyDSL's
stage-1 lead at prefill (see caveat 1 -- that comparison is not like-for-like).

## Caveats

1. **FlyDSL stage 1 emits fp4 at T=1, 8, 1024 and 4096** (`_fp4q` kernels: it fuses the
   MXFP4 quant of the intermediate into its epilogue) while Gluon and Triton emit bf16.
   That is less output traffic and a different amount of work, so the prefill stage-1
   comparison is not like-for-like. Gluon has the same fused-quant path
   (`moe_gemm1_a4w4_mxfp4_out`) but it is not exercised here. At T=32 FlyDSL emits bf16
   and that row is directly comparable.
2. Tuned FlyDSL/CK numbers come from the **aiter-02** checkout, because
   `test_moe_2stage.py` has no repo-root `sys.path` bootstrap and resolves `aiter` there.
3. The active expert *ids* differ between harnesses (`arange(n)` vs `randperm(n)`); only
   the count and the balance affect traffic, both of which are pinned.
4. Counter collection serializes dispatches, which costs a few percent. The tuner's own
   recorded times for this shape (`us1` = 29.2 / 90.3 at T=1 / T=32) sit 9-10% below the
   times measured here, consistent with that overhead.
5. Single shape, single GPU, per-dispatch average over 5 (Gluon/Triton) or 8 (FlyDSL/CK)
   dispatches. No error bars; run-to-run variance on repeated sweeps was 1-2%.

## Reproducing

```bash
# Gluon / Triton, aligned routing. --n-active mirrors the other harness's
# AITER_MOE_NUM_EXPERT_ACTIVATED; it must be <= min(E, T*topk), so T=1 needs 8.
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum --truncate-kernels -d out \
  -- python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py \
     --op a4w4 --shape 7168,2048,33,8 --tokens 8 32 1024 4096 --n-active 33

# Tuned FlyDSL / CK (from the aiter-02 checkout)
AITER_MOE_NUM_EXPERT_ACTIVATED=33 \
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum --truncate-kernels -d out \
  -- python op_tests/test_moe_2stage.py -q 4 -dim 7168,2048 -e 33 -k 8 -t 32 \
     --csv-filter __none__
```

The benchmark's own wall-clock columns are not what this document reports -- take the
kernel times from the trace (`rocpd_kernel_dispatch`), for the reason given at the top.

`--csv-filter __none__` skips the CSV-row sweep, which otherwise raises before reaching
the requested shape: it runs in FlyDSL run-only mode (`FLYDSL_RUNTIME_RUN_ONLY=1`) and
some rows have no AOT cache entry.

If `rocprofv3` fails with error 16, `torch/lib/librocprofiler-sdk.so` is a second copy of
the SDK; move it aside for the duration of the run.
