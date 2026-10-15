# gfx950 MoE GEMM1: Gluon vs FlyDSL cold performance

Measured 2026-09-24. MI350X (gfx950).

Gluon is this package on commit `b44546ef1` plus the `KernelLayout` refactor. The
refactor moved the precomputed layouts from methods on `KernelTuningConfig` into
`_layout.py::KernelLayout`. FlyDSL is the per-shape retuned stage-1 kernels.

## Workload

- Shape: GEMM1 of a MoE layer, K = hidden = 7168, intermediate = 2048 (raw N = 4096),
  E = 33 experts, top-k = 8.
- Epilogue: `SiLU(gate) * up`, with gate/up split. Output is BF16, except A4W4, which
  writes a fused MXFP4 output.
- dtype labels (activation x weight):

  | label | activation | weight |
  |---|---|---|
  | `a4w4` | MXFP4 | MXFP4 |
  | `a8w4` | MXFP8 | MXFP4 |
  | `a8w8` | MXFP8 | MXFP8 |
  | `a16w16` | BF16 | BF16 |

- T is the number of tokens: 16, 64, 256, 1024 or 4096.

## Benchmark method: cold, trace-timed

Harness: `bench_out/gluon_flydsl_final_20260924/`, files `run_bench.py` (driver) and
`worker.py` (one process per case and T). `bench_out/` is untracked, at the repo root.

Measurement loop, one process per (backend, dtype, T):

1. **Setup (excluded from timing).** Routing, input quantization, A-scale sorting,
   weight and scale shuffle/preshuffle, output allocation, JIT compile, and
   launch-argument preparation all run once, before the loop.
2. **Cache flush before every dispatch.** Each iteration is exactly
   `flush.zero_()` on a 768 MiB buffer, then one prepared GEMM1 dispatch, on the same
   stream. Nothing else runs inside the loop.
3. **Warmup discarded.** 40 cold warmup iterations, then 100 measured ones. The early
   warmups run up to 25% slower than steady state, which is why they are dropped.
4. **High duty cycle.** Dispatches are queued asynchronously with no host sync inside
   the loop. Gluon uses the cached `_fast_launch` path, about 8 µs of host time
   against a ≥100 µs fill, so the host never starves the GPU.
5. **Timing.** Device durations come from the rocprofv3 kernel trace, not host
   timers.
6. **Pooling.** 4 rounds alternate the order of the Gluon and FlyDSL cases, and each
   cell pools 4 × 100 = 400 samples. P99 is nearest-rank.

Checks the harness enforces on every process:

- The traced window holds exactly `2 × 140` kernels, alternating fill and GEMM, with
  consecutive dispatch IDs on one queue. Any extra kernel rejects the run.
- The fill/GEMM timestamp overlap must be ≤ 1 µs.
- The profiled GPU's PCI BDF must equal the requested GPU.
- Before each process starts, the harness waits until the GPU shows no utilization,
  no foreign processes and no allocations.

Correctness gate before timing (`--mode validate`): every case is checked against a
torch reference. It then runs 16 cold replays into a poisoned output buffer, and each
replay must be exactly equal.

Duty cycle, measured from the traces of this run (160 processes): the GPU was busy
**≥ 99.86%** of the measured window in every process, median 99.93%. The largest idle
gap was ≤ 64 µs. That gap is the single `synchronize()` between the warmup loop and
the measured loop.

### Configurations

- **Gluon:** what the production config ladder in `moe_op_gemm_gluon.py` emits for
  the tuned `BLOCK_M`, with the committed cache policies. BF16 at T=16/64 resolves to
  one N slot, so those two cells use the tuned pinned config instead. Source:
  `gluon_config_source.json`, `cases.json`.
- **FlyDSL:** the per-shape winners from `bench_out/flydsl_retune_20260921`. Two of
  those winners are excluded because they fail cold exact replays:
  - A4W4 TM128 at T=4096.
  - A8W4 T=256 with asynchronous A copy. The same tile with synchronous copy is
    reported instead; it is 0.5% slower.

Resolved tiles (Gluon: `BLOCK_M/N/K`, `SLOT_M/N`; FlyDSL: `tile_m/n/k`):

| dtype | T | Gluon | FlyDSL |
|---|---:|---|---|
| a4w4 | 16, 64, 256 | 64/256/256, slot 32/128 | 32/256/256 |
| a4w4 | 1024, 4096 | 128/256/256, slot 64/128 | 64/256/256 |
| a8w4 | 16, 64 | 32/128/512, slot 16/64 | 32/64/256 |
| a8w4 | 256 | 64/256/256, slot 32/128 | 64/128/256 (sync copy) |
| a8w4 | 1024, 4096 | 128/256/256, slot 64/128 | 128/256/256 |
| a8w8 | 16, 64 | 32/128/256, slot 16/64 | 32/64/256 |
| a8w8 | 256 | 64/128/256, slot 32/64 | 64/128/256 |
| a8w8 | 1024, 4096 | 128/256/128, slot 64/128 | 128/128/256 |
| a16w16 | 16, 64 | 32/128/128, slot 16/64 | 16/128/128 (T16), 16/128/256 (T64) |
| a16w16 | 256 | 64/256/64, slot 32/128 | 64/128/128 |
| a16w16 | 1024, 4096 | 128/256/64, slot 64/128 | 64/128/128 (T1024), 128/128/128 (T4096) |

### Reproduce

The GPU indices below are physical (amd-smi). The harness maps them to HIP ordinals
itself; on this machine they differ.

```bash
cd bench_out/gluon_flydsl_final_20260924
export ROCM_PATH=<venv>/lib/python3.14/site-packages/_rocm_sdk_devel
P=<venv>/bin/python
$P run_bench.py --mode validate --label val_<name> --gpu 0 --keep-going
$P run_bench.py --mode cold --label cold_<name> --validation-label val_<name> --gpu 0 --keep-going
```

Full matrix: about 12 min to validate and about 40 min cold, 160 processes. Results
land in `cold_<name>/report.md`, `summary.json` and `samples.csv`.

## Results

Label `cold_klayout`, physical GPU 0 (`0000:06:00.0`). 160/160 processes recorded,
0 failures, n = 400 per cell. Times are in µs. Gluon/FlyDSL below 1 means Gluon is
faster. "tie" means within ±0.5%.

| dtype | T | Gluon median | Gluon p99 | FlyDSL median | FlyDSL p99 | Gluon/FlyDSL | Faster |
|---|---:|---:|---:|---:|---:|---:|---|
| a4w4 | 16 | 112.1 | 119.9 | 140.3 | 145.7 | 0.799 | Gluon |
| a4w4 | 64 | 113.7 | 120.0 | 140.5 | 145.6 | 0.810 | Gluon |
| a4w4 | 256 | 122.8 | 129.2 | 178.7 | 185.0 | 0.687 | Gluon |
| a4w4 | 1024 | 195.6 | 204.4 | 237.1 | 251.6 | 0.825 | Gluon |
| a4w4 | 4096 | 604.6 | 649.1 | 690.6 | 721.8 | 0.875 | Gluon |
| a8w4 | 16 | 126.2 | 132.9 | 127.9 | 133.3 | 0.987 | Gluon |
| a8w4 | 64 | 132.5 | 139.0 | 130.0 | 136.0 | 1.019 | FlyDSL |
| a8w4 | 256 | 138.2 | 145.7 | 145.7 | 151.4 | 0.949 | Gluon |
| a8w4 | 1024 | 268.6 | 281.5 | 272.2 | 291.7 | 0.987 | Gluon |
| a8w4 | 4096 | 930.3 | 961.0 | 944.7 | 989.6 | 0.985 | Gluon |
| a8w8 | 16 | 233.2 | 240.5 | 226.3 | 233.5 | 1.030 | FlyDSL |
| a8w8 | 64 | 240.9 | 248.4 | 230.5 | 239.2 | 1.045 | FlyDSL |
| a8w8 | 256 | 253.1 | 259.7 | 253.9 | 260.9 | 0.997 | tie |
| a8w8 | 1024 | 356.2 | 365.6 | 377.6 | 387.6 | 0.943 | Gluon |
| a8w8 | 4096 | 1122.1 | 1174.8 | 1130.6 | 1186.4 | 0.992 | Gluon |
| a16w16 | 16 | 454.6 | 468.5 | 417.8 | 429.0 | 1.088 | FlyDSL |
| a16w16 | 64 | 458.3 | 471.3 | 423.6 | 431.7 | 1.082 | FlyDSL |
| a16w16 | 256 | 436.1 | 448.8 | 505.1 | 516.7 | 0.863 | Gluon |
| a16w16 | 1024 | 674.1 | 689.9 | 1048.8 | 1078.1 | 0.643 | Gluon |
| a16w16 | 4096 | 2176.9 | 2231.4 | 3456.6 | 3492.7 | 0.630 | Gluon |

**Gluon is faster in 14 cells, tied in 1, and slower in 5.** The five slower cells are
BF16 and MXFP8 x MXFP8 at T ≤ 64, plus MXFP8 x MXFP4 at T = 64. That is the
`block_m` 16/32 regime.

- **Largest Gluon wins:** BF16 at T ≥ 1024 (1.56–1.59x) and A4W4 at T = 256 (1.46x).
- **Largest Gluon loss:** BF16 at T ≤ 64 (1.08–1.09x slower).

## KernelLayout refactor: no regression

Two checks, one on the generated code and one on timing.

**Code.** The refactor is a pure move:
- The assembly of all 20 dispatchable Gluon kernels is byte-identical before and
  after (`scripts/gluon_moe_asm_hash.py`).
- Both determinism shas are unchanged: bf16 `59d442f663991659`, mxfp4-out
  `0cfa77fb04450594`.
- The schedule tables are identical.
- The full gluon test suite gives the same results as HEAD.

**Timing.** The same harness was run on the same GPU on commit `75647754b`, before the
refactor. That is label `cold1`, from session cd590ab1. Two changes sit in between:
`b44546ef1`, a dead-code deletion, and the refactor, whose assembly identity is
verified above. Median µs:

| dtype | T | Gluon `cold1` | Gluon `cold_klayout` | Δ | FlyDSL `cold1` | FlyDSL `cold_klayout` | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| a4w4 | 16 | 112.1 | 112.1 | −0.02% | 140.3 | 140.3 | −0.03% |
| a4w4 | 64 | 114.3 | 113.7 | −0.52% | 140.9 | 140.5 | −0.33% |
| a4w4 | 256 | 122.8 | 122.8 | −0.00% | 178.5 | 178.7 | +0.11% |
| a4w4 | 1024 | 195.9 | 195.6 | −0.15% | 236.8 | 237.1 | +0.16% |
| a4w4 | 4096 | 607.4 | 604.6 | −0.46% | 683.7 | 690.6 | +1.02% |
| a8w4 | 16 | 126.2 | 126.2 | +0.02% | 127.2 | 127.9 | +0.52% |
| a8w4 | 64 | 132.1 | 132.5 | +0.30% | 129.7 | 130.0 | +0.28% |
| a8w4 | 256 | 138.2 | 138.2 | −0.04% | 145.6 | 145.7 | +0.01% |
| a8w4 | 1024 | 267.8 | 268.6 | +0.27% | 272.4 | 272.2 | −0.06% |
| a8w4 | 4096 | 929.0 | 930.3 | +0.13% | 944.1 | 944.7 | +0.06% |
| a8w8 | 16 | 233.0 | 233.2 | +0.07% | 226.7 | 226.3 | −0.16% |
| a8w8 | 64 | 240.9 | 240.9 | −0.03% | 229.4 | 230.5 | +0.46% |
| a8w8 | 256 | 252.8 | 253.1 | +0.10% | 253.8 | 253.9 | +0.01% |
| a8w8 | 1024 | 355.8 | 356.2 | +0.12% | 377.6 | 377.6 | −0.01% |
| a8w8 | 4096 | 1123.2 | 1122.1 | −0.11% | 1132.6 | 1130.6 | −0.18% |
| a16w16 | 16 | 454.2 | 454.6 | +0.08% | 417.7 | 417.8 | +0.03% |
| a16w16 | 64 | 458.6 | 458.3 | −0.07% | 417.5 | 423.6 | +1.47% |
| a16w16 | 256 | 434.9 | 436.1 | +0.27% | 496.5 | 505.1 | +1.73% |
| a16w16 | 1024 | 674.1 | 674.1 | +0.01% | 1046.3 | 1048.8 | +0.24% |
| a16w16 | 4096 | 2175.9 | 2176.9 | +0.05% | 3457.7 | 3456.6 | −0.03% |

Gluon moves −0.52% to +0.30%, inside FlyDSL's −0.33% to +1.73% drift. FlyDSL code did
not change between the two runs, so its movement is the noise floor.

The refactor does have one cost: the cold first-call JIT compile is 5–11% slower
(0.13–0.34 s per kernel). Kernel runtime and host launch cost are unchanged.

## Caveats

- **Only compare runs of the same harness.** `bench_out/gluon_flydsl_dtype_tokens_20260909`
  uses the same cold protocol, but its `cases.json` pins 2026-09-09 configs for both
  backends. It reads, for example, 143.6 µs for A4W4 at T=16 instead of 112.1 µs.
  Absolute numbers from the two harnesses are not comparable.
- Medians drift up to about 1–2% between sessions on byte-identical code, larger for
  FlyDSL BF16. Treat single-run differences under about 1% as noise.
- The FlyDSL numbers are its best *correct* configs. Counting the replay-failing A4W4
  TM128 kernel would bring A4W4 at T=4096 to parity; see
  `bench_out/gluon_flydsl_ceiling_20260924`.

## Rerun after the final config/layout split (2026-09-25)

The code was then reorganised so that knob-controlled decisions live on
`KernelTuningConfig` and only derived shapes and layouts live in `KernelLayout`
(`klayout_v3`). Assembly stayed byte-identical and both determinism shas held.

The same harness was rerun on the same GPU as label `cold_klayout_v3`: 160/160
processes, 0 failures.

- **Gluon vs `cold1`:** −0.39% to +0.35% across all 20 cells.
- **FlyDSL control (unchanged code):** −0.31% to +1.40%.
- **Gluon vs FlyDSL:** Gluon faster in 14, tied in 1, slower in 5, the same split as
  above.
- **Per-cell medians:** within 0.7% of the `cold_klayout` table above. For example,
  a4w4 T=16 is 112.0 vs FlyDSL 140.4, and a16w16 T=4096 is 2174.1 vs FlyDSL 3456.2.

## Rerun on physical GPU 5 (2026-10-13)

Same code (`klayout_v3`, assembly identical) and same harness, run as label
`cold_klayout_v3_g5`. 160/160 processes, 0 failures. GPU 0 was occupied, so this run
used GPU 5.

The harness was run through `~/klayout_job/run_bench_tolerant.py`. It treats as idle
the ≤4 MiB idle driver contexts that other users' jobs on GPUs 0–3 open on every
device. Any GPU use, mapped PID, or larger allocation still blocks a launch.

Results:

- **Gluon vs FlyDSL:** Gluon faster in 15 cells and slower in 5. The 5 are a8w4 T=64,
  a8w8 T≤64 and bf16 T≤64. a8w8 T=256 moved from a tie to 0.991.
- **Both backends vs `cold1` (GPU 0):** Gluon −0.16% to +4.82%, FlyDSL −0.17% to
  +3.75%. Both run up to about 3–5% slower at large T on this GPU.
- **Cause:** a property of the device and the load on the node, not of the code. The
  identical assembly measured −0.39% to +0.35% on GPU 0 (`cold_klayout_v3`).
- **Rule:** compare absolute numbers only within one physical GPU.
