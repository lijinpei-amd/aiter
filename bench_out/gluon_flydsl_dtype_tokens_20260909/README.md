# Cold Gluon versus FlyDSL GEMM1 dtype/token sweep

This harness compares one prepared Gluon or FlyDSL MoE GEMM1 dispatch for
A4W4, A8W4, A8W8, and BF16 at `T in {16, 64, 256, 1024, 4096}`. The logical
shape is H/K=7168, intermediate=2048 (raw GEMM N=4096), E=33, top-k=8, with
`SiLU(gate) * up`, no bias, and no routing-weight multiply.

Routing, input quantization, A-scale sorting, weight and scale transforms,
output allocation, compilation, and launch-argument preparation all happen
during setup. Gluon replays captured `_fast_launch` arguments and FlyDSL
replays the captured compiled callable and prepared arguments. Each cold
iteration is exactly a 768 MiB streaming fill followed by one GEMM1 dispatch
on the same stream. The rocprofv3 trace parser rejects an extra dispatch,
checks the physical GPU, and excludes the fill duration from the statistics.

Cold comparison runs use four alternating-order rounds, 40 cold warmups, and
100 measured samples per process. The aggregate mean, median, and nearest-rank
p99 pool all 400 device-duration samples per backend/dtype/token cell. P99 is
`sorted(samples)[ceil(0.99*n)-1]`.

The selected M tiles are:

| Precision | Backend | T16 | T64 | T256 | T1024 | T4096 |
|---|---|---:|---:|---:|---:|---:|
| A4W4 | Gluon | 64 | 64 | 64 | 128 | 128 |
| A4W4 | FlyDSL | 32 | 32 | 32 | 64 | 64 |
| A8W4 | Gluon | 32 | 32 | 64 | 128 | 128 |
| A8W4 | FlyDSL | 32 | 32 | 64 | 128 | 128 |
| A8W8 | Gluon | 32 | 32 | 64 | 128 | 128 |
| A8W8 | FlyDSL | 32 | 32 | 64 | 128 | 128 |
| BF16 | Gluon | 32 | 32 | 64 | 128 | 128 |
| BF16 | FlyDSL | 32 | 16 | 64 | 64 | 128 |

`cases.json` records every other tuning field and its source configuration.
`plan.json` stores the fully resolved configuration for every token count.
The harness hashes the full logical input payload, effective kernel artifacts,
source files, compiler, and libtriton. A cold run requires a complete validation
set with exactly matching provenance, inputs, kernel identity, and output hash.

Inspect the final plan without using the GPU:

```sh
/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python \
  bench_out/gluon_flydsl_dtype_tokens_20260909/run_bench.py \
  --mode cold --label planned --validation-label validation_final \
  --cases gluon_a4w4 flydsl_a4w4 gluon_a8w4 flydsl_a8w4 \
          gluon_a8w8 flydsl_a8w8 gluon_a16w16 flydsl_a16w16 \
  --tokens 16 64 256 1024 4096 --gpu 4 \
  --rounds 4 --warmups 40 --samples 100 --dry-run
```

Validate the full matrix:

```sh
/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python \
  bench_out/gluon_flydsl_dtype_tokens_20260909/run_bench.py \
  --mode validate --label validation_final --gpu 4 \
  --cases gluon_a4w4 flydsl_a4w4 gluon_a8w4 flydsl_a8w4 \
          gluon_a8w8 flydsl_a8w8 gluon_a16w16 flydsl_a16w16 \
  --tokens 16 64 256 1024 4096 --validation-replays 16
```

Run the final cold comparison:

```sh
/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python \
  bench_out/gluon_flydsl_dtype_tokens_20260909/run_bench.py \
  --mode cold --label cold_final --validation-label validation_final \
  --cases gluon_a4w4 flydsl_a4w4 gluon_a8w4 flydsl_a8w4 \
          gluon_a8w8 flydsl_a8w8 gluon_a16w16 flydsl_a16w16 \
  --tokens 16 64 256 1024 4096 --gpu 4 \
  --rounds 4 --warmups 40 --samples 100 --validation-replays 16
```

The result directory contains the human-readable `report.md`, pooled raw
samples, per-process records, comparison JSON, profiler traces, worker output,
telemetry, the exact plan, and complete provenance hashes. Existing evidence
directories are never modified.
