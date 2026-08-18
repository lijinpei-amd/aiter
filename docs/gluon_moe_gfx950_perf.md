# gfx950 MoE A4W4 grouped GEMM: Gluon vs Triton vs tuned FlyDSL

Kernel-level comparison of the a4w4 MoE grouped-GEMM implementations in this tree, on one
shape, measured with `rocprofv3`. Wall-clock is deliberately **not** reported: at decode
the Gluon launch path costs ~90 us of host time against a ~25 us kernel, which swamps the
kernel differences this table is about. (For scale: the harness reports 110 us wall for a
25.4 us T=1 stage-1 kernel, and every decode point sits on the same ~100-120 us floor
regardless of shape.)

## Setup

| | |
|---|---|
| GPU | MI355X (gfx950), 256 CU, 8 XCDs, 160 KiB LDS/CU |
| shape | H=7168, I=2048, E=33, topk=8 (`--dim 7168,2048 -e 33 -k 8`) |
| stage 1 | N=2I=4096, K=H=7168, fused SwiGLU |
| stage 2 | N=H=7168, K=I=2048, router-weight multiply |
| dtype | MXFP4 x MXFP4 (E2M1 payload + E8M0 group-32 scales) |
| tool | `rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum`, per-dispatch average |

`E=33` is 32 routed experts plus the shared expert. Note that **no** tuned row shipped for
this shape in any `*_tuned_fmoe.csv`; the tuned column here comes from a tuning run done
for this document (see "Which kernel the tuner selected"), and E=33 also had to be added
to the mxfp4 aux codegen table before it would run at all.

### Routing is aligned across harnesses

Gluon/Triton are driven from a standalone script; FlyDSL/CK from
`op_tests/test_moe_2stage.py`. The two seed their routers independently, so they
activated different numbers of experts and therefore read different amounts of weight --
at T=8 that alone was a 17% difference, larger than any kernel effect. Both sides
are pinned with `AITER_MOE_NUM_EXPERT_ACTIVATED=n` (n=8 at T=1, n=33 elsewhere), which
forces exactly n active experts with a round-robin, perfectly balanced token assignment.

After alignment, Gluon, Triton and FlyDSL agree on stage-1 HBM traffic to within 1% at
every decode point (516-521 MB at T=8 and T=32) -- three independent implementations
landing on the same number, which is the check that makes the rest of the table
meaningful. At prefill they diverge by up to 9% because the tuned kernels choose different
tiling and split-K strategies; there the traffic column is a result, not a control.

### How the columns are defined

| column | meaning |
|---|---|
| **Gluon** | `_moe_gluon_gemm1/2`, this branch |
| **Triton** | in-tree `_moe_gemm_a4w4` |
| **Tuned FlyDSL** | the FlyDSL kernel selected by `gemm_moe_tune.py --mxfp4-flydsl` for this (shape, token), re-tuned for this document |

**The tuned column is a FlyDSL-only search.** `--mxfp4-flydsl` tunes stage 1 and stage 2 as
a coupled `(g1, g2)` unit over the FlyDSL mxfp4 registry; it does not consider CK codegen
instances, hand-written ASM or Opus. An earlier revision of this document reported a
*joint* FlyDSL+CK search, which at T=1 stage 2 preferred a CK kernel
(`moe_ck2stages_gemm2_...`, symbol `kernel_moe_mxgemm_2lds`). That joint search has not
been re-run here, so the tuned rows below are a lower bound on what the full dispatcher
could pick, not the dispatcher's own answer.

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
| 1 | Gluon | `BM16xBN64xBK512, 2 buf, 4 warps` | 25.4 | 18.5 | 4,927 | 125.0 |
| 1 | Triton | `BM16xBN128xBK256` | 31.7 | 14.8 | 3,950 | 125.0 |
| 1 | Tuned FlyDSL | `g1_a4w4_16x256x256_f16in_nt` | 27.2 | 17.3 | 4,611 | 125.4 |
| 8 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 95.6 | 39.3 | 5,395 | 515.7 |
| 8 | Triton | `BM16xBN128xBK256` | 107.4 | 35.0 | 4,852 | 520.9 |
| 8 | Tuned FlyDSL | `g1_a4w4_16x256x256_f16in_nt` | 109.4 | 34.4 | 4,727 | 517.1 |
| 32 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 96.5 | 155.8 | 5,357 | 516.8 |
| 32 | Triton | `BM16xBN128xBK256` | 119.3 | 126.0 | 4,372 | 521.4 |
| 32 | Tuned FlyDSL | `g1_a4w4_16x256x256_f16in_nt` | 108.8 | 138.2 | 4,780 | 520.0 |

**Stage 2** (N=7168, K=2048)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1 | Gluon | `BM16xBN64xBK512, 2 buf, 4 warps` | 16.4 | 14.3 | 3,829 | 62.8 |
| 1 | Triton | `BM16xBN128xBK256` | 16.5 | 14.2 | 3,799 | 62.8 |
| 1 | Tuned FlyDSL | `t16x256x256_atomic_nt_sbm16` | 13.8 | 17.0 | 4,550 | 62.8 |
| 8 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 49.2 | 38.2 | 5,267 | 259.3 |
| 8 | Triton | `BM16xBN128xBK256` | 51.6 | 36.4 | 5,025 | 259.3 |
| 8 | Tuned FlyDSL | `t16x128x256_atomic_sbm16` | 46.5 | 40.4 | 5,590 | 260.1 |
| 32 | Gluon | `BM16xBN128xBK512, 2 buf, 4 warps` | 51.4 | 146.1 | 5,097 | 262.2 |
| 32 | Triton | `BM16xBN128xBK256` | 57.6 | 130.4 | 4,550 | 262.2 |
| 32 | Tuned FlyDSL | `t16x256x128_atomic_persist_nt_sbm16` | 49.8 | 151.0 | 5,416 | 269.6 |

### Prefill

**Stage 1** (N=4096, K=7168)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1024 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 281.8 | 1,707 | 2,157 | 607.9 |
| 1024 | Triton | `BM128xBN512xBK256` | 394.8 | 1,218 | 1,572 | 620.6 |
| 1024 | Tuned FlyDSL | `g1_a4w4_128x256x256` | 218.3 | 2,204 | 3,027 | 660.8 |
| 4096 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 936.3 | 2,055 | 1,621 | 1517.6 |
| 4096 | Triton | `BM128xBN512xBK256` | 1068.1 | 1,802 | 1,380 | 1473.9 |
| 4096 | Tuned FlyDSL | `g1_a4w4_128x256x256` | 574.6 | 3,349 | 2,562 | 1472.3 |
| 16384 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 3488.8 | 2,206 | 1,660 | 5791.7 |
| 16384 | Triton | `BM128xBN512xBK256` | 3790.0 | 2,031 | 1,429 | 5415.6 |
| 16384 | Tuned FlyDSL | `g1_a4w4_128x256x256` | 2159.5 | 3,564 | 2,468 | 5328.6 |

**Stage 2** (N=7168, K=2048)

| T | impl | kernel / tile | µs | TFLOP/s | HBM GB/s | HBM MB |
|---:|---|---|---:|---:|---:|---:|
| 1024 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 153.9 | 1,563 | 2,584 | 397.6 |
| 1024 | Triton | `BM128xBN512xBK256` | 183.7 | 1,310 | 2,228 | 409.2 |
| 1024 | Tuned FlyDSL | `t128x128x256_atomic_sbm128` | 157.1 | 1,531 | 4,846 | 761.5 |
| 4096 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 530.9 | 1,812 | 2,057 | 1091.9 |
| 4096 | Triton | `BM128xBN512xBK256` | 634.8 | 1,516 | 1,830 | 1161.9 |
| 4096 | Tuned FlyDSL | `t128x256x128_reduce_sbm128` (split-K 8) | 393.8 | 2,443 | 6,111 | 2406.8 |
| 16384 | Gluon | `BM128xBN256xBK256, 2 buf, 8 warps` | 1981.3 | 1,942 | 2,202 | 4362.8 |
| 16384 | Triton | `BM128xBN512xBK256` | 2257.5 | 1,705 | 2,021 | 4563.1 |
| 16384 | Tuned FlyDSL | `t128x256x128_reduce_sbm128` (split-K 8) | 1568.7 | 2,453 | 6,386 | 10018.1 |

**Two-stage totals**

| T | Gluon µs | Triton µs | Tuned FlyDSL µs | tuned vs Gluon |
|---:|---:|---:|---:|---:|
| 1 | 41.8 | 48.2 | 41.0 | 1.02x |
| 8 | 144.8 | 159.0 | 155.9 | 0.93x |
| 32 | 147.9 | 176.9 | 158.6 | 0.93x |
| 1024 | 435.7 | 578.5 | 375.4 | 1.16x |
| 4096 | 1467.2 | 1702.9 | 968.4 | **1.52x** |
| 16384 | 5470.1 | 6047.5 | 3728.2 | **1.47x** |

## Which kernel the tuner selected

There is **no tuned row for this shape** in `aiter/configs/tuned_fmoe.csv` or any
`model_configs/*_tuned_fmoe.csv`. Left alone, the dispatcher takes the untuned fallback,
so the tuned column above required running the tuner first:

```bash
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl \
  -i untuned.csv -o tuned.csv --mp 8 --shape_grouped --all
```

`--mxfp4-flydsl` searches `(g1, g2)` as a coupled unit, so it reports one combined `us`
per shape and leaves `us2`/`tflops`/`bw` zero. Winners, with the coupled time the tuner
recorded and the per-stage times measured here:

| T | winning g1 | winning g2 | tuner µs (g1+g2) | measured µs (g1+g2) |
|---:|---|---|---:|---:|
| 1 | `g1_a4w4_16x256x256_f16in_nt` | `t16x256x256_atomic_nt_sbm16` | 42.0 | 41.0 |
| 8 | `g1_a4w4_16x256x256_f16in_nt` | `t16x128x256_atomic_sbm16` | 127.6 | 155.9 |
| 32 | `g1_a4w4_16x256x256_f16in_nt` | `t16x256x128_atomic_persist_nt_sbm16` | 160.3 | 158.6 |
| 1024 | `g1_a4w4_128x256x256` | `t128x128x256_atomic_sbm128` | 394.8 | 375.4 |
| 4096 | `g1_a4w4_128x256x256` | `t128x256x128_reduce_sbm128` | 1187.8 | 968.4 |
| 16384 | `g1_a4w4_128x256x256` | `t128x256x128_reduce_sbm128` | 4542.9 | 3728.2 |

The tile choice splits cleanly at the decode/prefill boundary: `16x256x256` wins all three
decode points, `128x256x256` all three prefill points. Selections were confirmed against
the symbols that actually dispatched (`gemm1_a4w4_port_h7168_i2048_ne33_bm16_iq_sep` /
`..._bm128_cached_sep`), so the tuned rows are the tuned kernel, not a fallback.

The tuner's coupled time and the sum of the two measured kernel times are not the same
quantity -- the tuner times the whole fused call including the sorting and quant aux
kernels, while the measured column sums only the two GEMM dispatches. The tuner's figure
is higher at T=8/32 (aux dominates at decode) and lower at prefill (counter collection is
not serializing its dispatches). Do not read the difference as run-to-run noise.

Errors were 0.0118-0.0128 cosine diff at every point, well inside the tuner's 0.1 gate.
21 of the searched candidates were rejected, all of them CUDA OOM on the `32x256x256` g1
variant at T=16384 -- no correctness rejections.


## Like-for-like: fp4 intermediate

> **Stale for the tuned column.** This section was measured against the *previous* tuned
> selection, whose stage-1 kernels carried an explicit `_fp4q` marker. The re-tuned
> winners above (`gemm1_a4w4_port_..._sep`) carry no such marker and their stage-2
> partners are all `..._bf16_...`, but the intermediate dtype has **not** been confirmed
> at the IR level. Until it is, treat the tuned prefill stage-1 numbers as possibly not
> like-for-like against bf16-emitting Gluon. The Gluon and Triton columns below stand.

The tuned FlyDSL stage-1 kernel fuses the MXFP4 quant of the intermediate into its
epilogue (`_fp4q`) at every token count except T=32, so the tables above compare it
against a bf16-emitting Gluon and Triton. Gluon has the same fused path
(`moe_gemm1_a4w4_mxfp4_out`); the Triton a4w4 kernel has none -- only a8w4 has
`out_mx_quant`, and that emits fp8 -- so its equivalent is gemm1(bf16) followed by the
standalone `mxfp4_quant` launch, and both kernels are charged to it.

| T | Gluon fused | Gluon bf16 | Triton gemm1 + quant | Tuned FlyDSL fused |
|---:|---:|---:|---:|---:|
| 1 | **26.7** | 27.3 | 32.0 + 3.3 = 35.3 | 32.2 |
| 8 | **103.8** | 99.0 | 117.4 + 3.1 = 120.5 | 109.5 |
| 32 | **104.7** | 100.9 | 126.8 + 3.1 = 129.9 | (emits bf16) |
| 1024 | 320.7 | 291.2 | 410.4 + 18.9 = 429.3 | **223.4** |
| 4096 | 1051.3 | 942.3 | 1120.6 + 74.3 = 1194.9 | **651.4** |

HBM moved, same runs:

| T | Gluon fused | Gluon bf16 | Triton gemm1 + quant |
|---:|---:|---:|---:|
| 1024 | 584.8 MB | 607.8 MB | 696.8 MB |
| 4096 | 1420.8 MB | 1516.8 MB | 1783.3 MB |

Two things fall out.

**Against Triton the fused path wins everywhere** -- 1.34x at T=1024, 1.14x at T=4096 --
and moves 16-20% less memory, which is the case the fused epilogue was written for.

**Against FlyDSL it does not close the prefill gap; it widens it** (1.30x -> 1.44x at
T=1024, 1.45x -> 1.61x at T=4096). Gluon's fused quant is *slower than emitting bf16* at
every T >= 8 despite moving less traffic -- 320.7 vs 291.2 us at T=1024 -- so the epilogue
quant costs more than the write traffic it saves. That is a defect in our epilogue, not a
property of fusing, and it is the first thing to look at before reading anything else into
the prefill numbers.

## Why prefill is slower than FlyDSL

Three plausible explanations were measured and ruled out.

**Not occupancy.** Gluon uses 105,952 B of LDS at prefill -- 1 workgroup/CU, 8 waves --
against Triton's 65,536 (2/CU) and FlyDSL's 55,424 (2/CU, also 8 waves). But raising
occupancy makes it worse, because the only way to free that much LDS is a shorter
`BLOCK_K`, and that multiplies per-stage barrier and wait overhead:

| config | LDS | CTA/CU | T=1024 us | T=4096 us |
|---|---:|---:|---:|---:|
| BN256 BK256 nb2 (shipping) | 104,448 | 1 | 281.0 | 1034.1 |
| BN128 BK256 nb2 | 69,632 | 2 | **273.6** | 1013.0 |
| BN256 BK128 nb2 | 52,224 | 3 | 622.2 | 1986.0 |
| BN128 BK128 nb2 | 34,816 | 4 | 701.8 | 2357.6 |
| BN128 BK256 nb3 | 104,448 | 1 | 331.3 | 1193.9 |
| BN256 BK256 nb2, tiles/warp (1,2) | 104,448 | 1 | 282.1 | **982.6** |

Best available is 2-5%, not 30-60%.

**Not M-padding.** `block_m` is 128 at prefill while experts hold ~248 tokens, which looks
like 1.5x wasted work from `RoutingData.n_blocks`. It is not: that function returns a
worst-case grid bound, and the surplus blocks are marked -1 in `block_pid_map` and exit
immediately. Only 66 of 96 blocks execute at T=1024 and 264 of 288 at T=4096 -- **1.031x**
padded work, and FlyDSL at `tile_m=64` pays exactly the same 1.031x.

**Not the output format**, as the section above shows.

**It is data-wait stalls.** The prefill stall profile is the mirror image of decode, where
Gluon was issue-bound:

| | issue | wait-data | wait-issue | `SQ_WAIT_ANY` |
|---|---:|---:|---:|---:|
| FlyDSL T=1024 | 18.5% | 30.2% | 51.3% | 57,886,326 |
| Gluon T=1024 | 15.7% | **37.8%** | 46.5% | **90,181,943** |
| FlyDSL T=4096 | **25.9%** | 28.6% | 45.5% | 157,322,620 |
| Gluon T=4096 | 17.5% | **36.2%** | 46.3% | **310,240,888** |

Gluon's waves wait on data 1.56x (T=1024) and 1.97x (T=4096) as many cycles as FlyDSL's,
and issue an instruction on only 15.7-17.5% of cycles against FlyDSL's 18.5-25.9%. Both
run 8 waves/CU at T=1024, so this is not a wave-count difference -- it is that 8 waves with
two `BLOCK_K=256` stages in flight (512 K-elements) cannot cover the latency, and at
`BLOCK_N=256` there is no LDS left to go deeper. FlyDSL reaches the same wave count as
2-3 workgroups of 4 waves with its own prefetch, and at T=4096 gets to 3 CTAs/CU.

**Advanced Thread Trace confirms it, and names the instruction.** ATT captures
(`rocprofv3 --att --att-target-cu 1 --kernel-iteration-range 1-1`) attribute stall cycles
per instruction on one CU for one dispatch. Share of that CU's total stall cycles:

| | `s_barrier` | `s_waitcnt` | `ds_read` | stall/latency |
|---|---:|---:|---:|---:|
| Gluon T=1024 s1 | 26-33% | -- | 10-14% | 83.2% |
| FlyDSL T=1024 s1 | 8-11% | 22-46% | ~1% | 83.7% |
| Gluon T=4096 s1 | 26-33% | -- | 10-14% | 81.7% |
| FlyDSL T=4096 s1 | 8-11% | 22-46% | ~1% | 78.1% |

Both are ~80% stalled; they stall on different things. Gluon waits at barriers and on LDS
reads, FlyDSL waits on memory counters -- which is what you want, because `s_waitcnt` on a
prefetched load can overlap with another workgroup's compute while `s_barrier` cannot.
The wall-clock cycle span on the traced CU follows: 1,596,480 vs 1,174,420 clocks for
stage 1 at T=4096 (gfx clock 1.90-2.12 GHz, read from the trace's own
`realtime.json` gfx/realtime clock pairs).

The mechanism is the LDS budget already noted above. Gluon's 105,952 B and 512-thread
workgroup mean **one CTA per CU**, so every one of its 9 `s_barrier`s stalls all 8 resident
waves simultaneously with nothing else to run. The tuned kernels are 256-thread workgroups
at 55,424 B, giving two CTAs per CU (three for stage 2 at T=4096), so one workgroup's
barrier is covered by the other's work.

*(The ATT captures were taken against the previous tuned selection, not the re-tuned
kernels in the tables above. The Gluon side -- which is what the argument rests on -- is
unaffected.)*

Closing this needs more K in flight per byte of LDS -- the same restructuring the decode
investigation pointed at -- not another tile from the existing ladder.

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

**Prefill is compute-bound** and the ordering inverts: tuned FlyDSL stage 1 is 29%
(T=1024), 63% (T=4096) and 62% (T=16384) ahead of Gluon in TFLOP/s. At T=4096 it does this
on *less* HBM traffic than Gluon (1472 vs 1518 MB), so it is a straight arithmetic-
throughput win, not a memory one -- consistent with the barrier-stall finding below.

**The tuned stage-2 prefill win is bought with bandwidth, not saved bytes.** At T=4096
the `reduce_sbm128` winner moves 2407 MB against Gluon's 1092 -- 2.2x the traffic, 2.3x at
T=16384 -- and is still 26% faster because it sustains 6.1-6.4 TB/s where Gluon sustains
2.1. That is a split-K-8 kernel trading footprint for parallelism, and it runs close to
HBM peak. On a bandwidth-contended system, or co-resident with other work, that extra
traffic is a real cost this isolated benchmark does not charge it for.

**Where each implementation wins**

| | decode (T=1, 8, 32) | prefill (T=1024, 4096, 16384) |
|---|---|---|
| Gluon | fastest at 3 of 6 points (all three stage 1) | fastest **stage 2** at T=1024 only |
| Triton | never fastest | slowest at every point |
| Tuned FlyDSL | fastest at all three stage-2 points | fastest **stage 1** at all three; fastest stage 2 at T=4096, 16384 |

The split is clean and it is the opposite of what the previous revision of this document
reported. **Gluon owns decode** by ~7% on the two-stage total (its stage-1 lead of 13-14%
outweighs a 3-5% stage-2 deficit); **tuned FlyDSL owns prefill** by 1.16x at T=1024 rising
to 1.52x at T=4096 and 1.47x at T=16384. Triton trails everywhere.

## Caveats

1. **The tuned stage-1 intermediate dtype is unverified.** See the banner on the
   "Like-for-like" section. If the re-tuned `..._sep` kernels still emit fp4, the tuned
   prefill stage-1 rows are not like-for-like against bf16-emitting Gluon.
2. **The tuned column is a FlyDSL-only search**, not the joint FlyDSL+CK+ASM+Opus search
   the dispatcher would run. See "How the columns are defined".
3. The active expert *ids* differ between harnesses (`arange(n)` vs `randperm(n)`); only
   the count and the balance affect traffic, both of which are pinned.
4. Counter collection serializes dispatches, which costs a few percent.
5. Single shape, single GPU, per-dispatch average over 20 dispatches (25 recorded, first
   fifth dropped as warmup). No error bars; repeated sweeps of the same configuration
   varied 2-4% -- e.g. Gluon T=4096 stage 1 measured 903.6, 936.3 and 942.3 us across
   three runs, so differences below ~5% in these tables should not be ranked.
6. All numbers in this document were measured in **this** checkout on one machine, unlike
   an earlier revision whose tuned column was borrowed from a separate `aiter-02` tree.

## Reproducing

```bash
# Gluon / Triton, aligned routing. --n-active mirrors the other harness's
# AITER_MOE_NUM_EXPERT_ACTIVATED; it must be <= min(E, T*topk), so T=1 needs 8.
# PYTHONPATH points at the rocprof shim; see below for why.
PYTHONPATH=op_tests/op_benchmarks/triton/rocprof_shim \
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum --truncate-kernels -d out \
  -- python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py \
     --op a4w4 --shape 7168,2048,33,8 --tokens 8 32 1024 4096 --n-active 33

# Tuned FlyDSL. AITER_CONFIG_FMOE points the dispatcher at the tuned CSV; without it
# this shape has no tuned row and silently takes the untuned fallback.
AITER_MOE_NUM_EXPERT_ACTIVATED=33 AITER_CONFIG_FMOE=tuned.csv \
PYTHONPATH=op_tests/op_benchmarks/triton/rocprof_shim \
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum --truncate-kernels -d out \
  -- python op_tests/test_moe_2stage.py -q 4 -dim 7168,2048 -e 33 -k 8 -t 32 \
     --csv-filter __none__ --kernel
```

### Environment prerequisites for the FlyDSL side

Three traps, all of which fail far from their cause. Set these before any run that
JIT-compiles a FlyDSL kernel:

```bash
export PATH="$ROCM_SDK/lib/llvm/bin:$ROCM_SDK/bin:$PATH"
export FLYDSL_GPU_ARCH=gfx950
```

1. **`rocm_agent_enumerator` must be on PATH.** `flydsl/runtime/device.py::_arch_from_hardware`
   shells out to it, swallows every exception, and returns a hard-coded `"gfx942"` on
   failure. These kernels emit a 16-byte `rocdl.raw.ptr.buffer.load.lds`; gfx942 has no
   128-bit LDS DMA, and LLVM does not diagnose it -- the DAG legalizer hits
   `ExpandIntegerOperand` on the `ptr addrspace(8)` rsrc and dies with
   `LLVM ERROR: Do not know how to expand this operator's operand!`, naming neither the
   target nor the intrinsic. `FLYDSL_GPU_ARCH=gfx950` pins it regardless. To confirm a
   suspected case: `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=d` then `grep 'chip =' d/*/00_origin.mlir`.
2. **The toolchain's `ld.lld` must come first on PATH.** A distro `/usr/bin/ld.lld` from a
   different LLVM major (22 vs the toolchain's 23) makes MLIR's `gpu-module-to-binary`
   fail with `lld invocation failed` and nothing else.
3. **The shape needs an mxfp4 aux instance.** `SHAPES` in
   `csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py` is a hard-coded list; a shape
   missing from it fails at runtime with
   `no codegen'd instance for shape key 'aux_sortzi_NE33_TOPK8_MB16_H7168'`. E=33 was added
   there for this document. After editing it, delete `aiter/jit/build/module_moe_mxfp4_aux`
   and the `.so` to force a real regeneration -- the JIT will otherwise relink stale blobs.

Under `rocprofv3`, a FlyDSL compile abort additionally deadlocks in the profiler's chained
signal handler rather than exiting, so the run hangs until its timeout instead of failing.

The benchmark's own wall-clock columns are not what this document reports -- take the
kernel times from the trace (`rocpd_kernel_dispatch`), for the reason given at the top.

`--csv-filter __none__` skips the CSV-row sweep, which otherwise raises before reaching
the requested shape: it runs in FlyDSL run-only mode (`FLYDSL_RUNTIME_RUN_ONLY=1`) and
some rows have no AOT cache entry.

If `rocprofv3` fails with error 16, `torch/lib/librocprofiler-sdk.so` is a second copy of
the SDK; move it aside for the duration of the run.

If instead it dies with a SIGSEGV in `llvm::DenseMapBase<>::LookupBucketFor<>()` before
any kernel runs, the crash is in `import triton`, not in anything being measured: the
profiler `LD_PRELOAD`s a rocprofiler-sdk that drags in `libLLVM.so.23`, whose global LLVM
symbols the statically-LLVM-linked `libtriton.so` then binds to in its own static
initialisers (`_GLOBAL__sub_I_PassBuilder.cpp`, `llvm::DebugCounter`). The shim in
`op_tests/op_benchmarks/triton/rocprof_shim/` pre-imports Triton with `RTLD_DEEPBIND`
when -- and only when -- a rocprofiler library is preloaded, which is why the command
above sets `PYTHONPATH`. Prepend the same `PYTHONPATH=...` to any other profiled run that
imports `aiter` or `triton`.
