# A4W4 T=4096 FlyDSL TM64 versus TM128 cold rerun

- Date: 2026-09-10
- Repository HEAD: `591175e3a7f77edd8077737ab8b83ed51f54a62a`
- GPU: physical GPU 4, AMD Instinct MI350X, `gfx950:sramecc+:xnack-`
- Shape: T=4096, H/K=7168, raw N=4096, intermediate=2048,
  E=33, top-k=8

## Result

| FlyDSL configuration | Samples | Mean (us) | Median (us) | P99 (us) | Correctness status |
|---|---:|---:|---:|---:|---|
| TM128 x TN256 x TK256, cached, separated, XCD0 | 400 | 608.379 | 606.904 | 655.204 | Short validation passed; extended replay failed |
| TM64 x TN256 x TK256, cached, separated, XCD4 | 400 | 685.480 | 685.485 | 707.485 | Short validation passed; retained safe baseline |

TM128 reduced mean, median, and p99 latency relative to TM64 by
11.25%, 11.46%, and 7.39%, respectively. The median difference was
78.581 us. Equivalently, TM128 was 1.129x as fast by median latency.

The rerun agrees with the previous TM128 recheck: the new mean/median/p99
differed by +0.19%, +0.46%, and -0.19%, respectively.

## Protocol

The two cases ran in alternating AB/BA order for four rounds. Each process used
40 cold warmups and 100 measured samples. Every measured dispatch was
immediately preceded on the same stream by one 768 MiB streaming fill.

Routing, input quantization, A-scale sorting, weight/scale transforms, output
allocation, JIT compilation, and launch-argument preparation were excluded from
the timed loop. Statistics pool all 400 samples per case; p99 is nearest-rank
`sorted[ceil(0.99*n)-1]`.

The independent trace/statistics audit passed:

- 8/8 profiler processes
- 8 kernel-trace CSV files
- 1,120 verified fill/GEMM pairs
- 800 measured samples
- both pooled summaries independently reproduced exactly

## Correctness

Both kernels passed the independent reference and 16 poisoned-output cold exact
replays before timing. Both produced the same combined output SHA256:
`9767953a104f63db88b2e4fe234fe6eaf412bad868dc5cb9d5ae6481ff8e5323`.

The separate 512-request TM128 validation failed exact equality at replay 185.
Its 50,328-byte HSACO SHA256 is
`8e4ecc8008d1150167bad26c6b2fb0421f0025d84c4d256dd6b3136efe392854`,
the same binary as the earlier intermittent failures. TM128 remains valid as
performance evidence but is correctness-disqualified for production selection.

## Artifacts

- Generated report: `report.md`
- Pooled statistics: `summary.json`
- Raw samples: `samples.csv` and `samples.json`
- Independent audit: `independent_audit.json`
- Short validation: `../validation_a4_tm64_tm128_rerun_20260910/`
- Extended TM128 failure:
  `../validation_a4_tm128_long_rerun_20260910/a4_ded_m128_recheck_t4096/run.log`

## Commands

```bash
/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python \
  bench_out/flydsl_t4096_tuning_20260909/run_bench.py \
  --mode validate \
  --label validation_a4_tm64_tm128_rerun_20260910 \
  --tokens 4096 \
  --cases a4_ded_m64_xcd4 a4_ded_m128_recheck \
  --gpu 4 \
  --validation-replays 16

/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python \
  bench_out/flydsl_t4096_tuning_20260909/run_bench.py \
  --mode cold \
  --label cold_a4_tm64_tm128_rerun_20260910 \
  --validation-label validation_a4_tm64_tm128_rerun_20260910 \
  --tokens 4096 \
  --cases a4_ded_m64_xcd4 a4_ded_m128_recheck \
  --gpu 4 \
  --rounds 4 \
  --warmups 40 \
  --samples 100 \
  --validation-replays 16
```
