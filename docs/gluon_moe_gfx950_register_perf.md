# gfx950 MoE register/buffer performance checks

With operand preparation outside every benchmark loop, the original four-wave
GEMM1 averages **615.233 us**, and the same recipe with B preshuffled averages
**607.761 us**. These are pooled arithmetic means of 600 cold dispatches per
case, measured on MI350X GPU 4 on 2026-09-08.

| Case | Pooled mean (us) | Median of six round medians (us) |
| --- | ---: | ---: |
| Original | 615.233 | 613.354 |
| Original + B preshuffled | 607.761 | 607.754 |

The pooled-mean reduction is 7.472 us (1.215%). Preshuffling improves every
paired round. The mean difference between paired round medians is -6.707 us,
with a 95% Student-t interval of [-10.342, -3.071] us across six pairs. The mean
paired percentage change is -1.091%; it uses round medians rather than pooled
samples and therefore differs from the pooled-mean percentage.

The workload is T4096/N4096/K7168, 33 experts, top-k 8, seed-0 balanced routing,
MXFP4 inputs and MXFP4 output. Gathered M is 32768. It applies split gate/up
SiLU with alpha 1, no limit, bias, or gammas, and fast reciprocal activation.
The launch has 4,608 CTAs of 256 threads, or four waves per CTA.

**B preshuffling is the only configuration difference in this pair.** Both
cases have `B_IN_REG=False` and stage B through `buffer_load` to LDS
(`buffer_load ... lds`). Both scale-register flags are false. Thus this pair
measures the existing B layout option, not the new B-register path.

A-scale sorting, B-scale shuffling, and any B preshuffling happen during setup.
The benchmark retains the sorted A-scale tensor for its fixed inputs/routing;
no sorter runs in correctness-replay, warmup, or measurement loops. The repeated
GPU sequence is `768 MiB cache flush -> GEMM`, on the same queue and stream.
The sorted scales are subject to the flush alongside the other operands.
Reported rocprofv3 durations include GEMM only; preprocessing, cache flushing,
and host overhead are excluded. This setup cache is benchmark-specific and
does not introduce library caching of changing activation scales.

The shared recipe uses BM128/BN256/BK256, mini-M64/mini-N128/mini-K256,
MFMA(16,16,128), warps(1,4), tiles(2,2), full-stage K prefetching, and shuffled
A/B scales with scale mini-M128. It retains legacy three-buffer/unroll-3
scheduling, `WARP_PIPELINE=0`, `WAIT_COMMIT_SCHEME=3`, `DS_READ_IN_MFMA=15`,
`SCALE_FILL_MID=True`, tile schedule 2, group-M 4, eight XCDs, and waves-per-EU 1.
The B cache modifier remains the original empty setting in both cases.

PyTorch was `2.15.0.dev20260816+rocm7.14`, and Triton was `3.8.0`. Compilation
used the original locally patched external LLVM `llc`, with these flags:

```text
-amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd -amdgpu-no-sched-revert
-misched-pin-critical-res=HWXDL -amdgpu-force-dynamic-lds-size=161216
-disable-post-ra -amdgpu-sched-strategy=coexec -amdgpu-coexec-no-regpressure
```

The forced LDS value is a compiler occupancy estimate. Actual shared memory
is 161216 bytes for Original and 158208 bytes with B preshuffled. Both have zero
VGPR spills/private memory; their SGPR spill counts are 10 and 8 respectively.
The compiler/library fingerprints and generated artifacts match the preceding
comparison. Original also matches the archived ~615 us kernel's allocated ELF
sections and full launch metadata, excluding the cache hash.

The preceding archived protocol ran `flush -> A-scale sort -> GEMM` inside
each loop. It reproduced the historical live/frozen reference at 612.155 and
613.765 us, respectively, over six interleaved rounds. Moving sorting outside
the loop changes A-scale cache state; it does not subtract sorter time from
an earlier GEMM-only measurement.

Under that preceding protocol, a three-round comparison transferred the
earlier FP32-workload tuning knobs to this exact quantized-output workload:

| Complete recipe | Median across rounds (us) | Mean paired latency change |
| --- | ---: | ---: |
| Original bookend control | 608.975 | baseline |
| Original + B preshuffled | 601.884 | -1.48% |
| B registers, A/B/AS/BS buffers 4/2/3/3 | 821.806 | +34.80% |
| Independent LDS, A/B/AS/BS buffers 3/3/3/3 | 663.765 | +8.75% |

Those transferred recipes regress here. The B-register recipe also uses `.cg`
and read mask 0; the independent LDS recipe uses `.cg`, warps(2,2), compiler
warp pipelining, and wait scheme 4. These bundled changes prevent attributing
the result solely to register storage or independent buffer counts. All arms
retain the original compiler flags, including its forced LDS estimate; an
alternative compiler policy was not tested. Default kernel recipes remain
unchanged. The [shipped tuning recipes](../scripts/gluon_moe_register_tuned/README.md)
describe the separate T1024/N2048 FP32-output workload and its measured wins.

All 15 transferred-recipe comparison processes reproduce both packed output
and output-scale bytes, with identical seeded logical inputs and functional
settings. All 240 cold determinism replays passed. The final setup-only sorting
comparison passed another 192 varying-poison cold replays across 12 processes,
with the same reference bytes and identical sorted-A fingerprints. Each process
has 40 cold warmups and 100 measured dispatches. Independent audits recompute
the timings and verify process order, idle checks, input/configuration identity,
and compiled artifacts. No samples or rounds were discarded.

The final protocol has 27 apparent flush/GEMM timestamp overlaps, at most
455 ns, within the existing 1 us tolerance; dispatch order remains consecutive
on one queue/stream. The earlier live/frozen reference has two adjacent
timestamp overlaps above that tolerance (2.853 and 2.110 us), so its strict
timestamp check is recorded separately as failed. These timestamps alone do
not establish the cause. Earlier feature validation passed 505 CPU checks and
108 GPU regression tests across a4w4/a8w8/a16w16, with 12 expected dtype skips.
The latest exact-workload runs cover MXFP4 and do not rerun the other dtypes.

Local evidence remains in the measurement checkout under
`bench_out/gluon_cold_615_20260908/`: `reference_final/`, `register_final/`, and
`presorted_a_final/` contain summaries and independent audits; the latter also
contains all 1,200 final samples. The parent directory's `presorted_a_report.md`
records the full protocol. Earlier three-dtype evidence is in
`bench_out/register_tuning_20260907/`.
