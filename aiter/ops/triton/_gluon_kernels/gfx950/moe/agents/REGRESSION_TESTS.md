# Regression tests

Run these after any change in this folder. The scripts are in `tests/`. Everything
they need is in the repo: run them from a clone plus the venv, with no `bench_out/`
or home-directory files. See [Vendored dependencies](#vendored-dependencies).

Which checks a change needs:

- **Pure refactor** (code must not change): all three below. The codegen gate is the
  real proof; the cold bench only confirms it.
- **Intentional codegen or perf change**: the correctness suite plus the cold bench.
  The assembly and sha comparisons will differ by design, so explain the differences
  instead of gating on them.

## Rules that apply to every check

- **Use the same checkout for the baseline and the change.** Do not use a git
  worktree as the control. A fresh worktree has no built `aiter/jit/*.so`, so the
  sorted-scale pre-pass silently falls back and 11 of the 20 cells compile different
  kernels. To get a baseline: `git stash` (or check out the base commit), run the
  baseline, then restore your change.
- **GPU numbering.** HIP ordinals (`HIP_VISIBLE_DEVICES`, the `GPU=` variables of
  `gate.sh` and `run_suite.sh`) are not amd-smi indices on this machine. HIP 7 is
  physical GPU 5. The cold bench takes the physical (amd-smi) index and maps it to
  HIP itself. Check `amd-smi process` before using a GPU, and never put a side job on
  the GPU a benchmark is using.
- **Environment.** Defaults are set in `tests/_env.sh`. Override `PY`, `ROCM_PATH`,
  `TRITON_HIP_DS_AGPR_LLC` (the patched llc), `HARNESS` or `OUT` from the
  environment. The `PY` and llc defaults assume `venv/01/` and `llvm-project/` next
  to the checkout. Outputs and baselines go to `$OUT` (default
  `bench_out/gluon_moe_regression/`, untracked); nothing is written into `tests/`.
- **A fresh clone needs the built `aiter/jit/*.so`.** Build them, or symlink them from
  a built tree (`ln -s <built>/aiter/jit/*.so aiter/jit/`). Without them the
  sorted-scale pre-pass silently falls back and the asm hashes change.

## 1. Codegen and determinism gate: `tests/gate.sh`

```bash
GPU=1 ASM_GPU=3 tests/gate.sh baseline --full   # on the unmodified tree, once
GPU=1 ASM_GPU=3 tests/gate.sh <label> --full    # after the change
```

The last line must be `### verdict: PASS`. Without `--full`, only T0–T2a run (about
2 minutes); that is useful while iterating.

| Check | What it covers | Must be |
|---|---|---|
| T0 | `scripts/gluon_moe_sched_dump.py`: schedule tables of 35 configs, CPU only | byte-identical |
| T1 | Symbolic emitter and config tests: `buffered_schedule`, `wait_commit`, `config`, `scale_miniblocks` | all pass (856 at commit `819123dbc`) |
| T2a | `test_moe_gemm_gluon_frozen.py` | all pass (18) |
| T2b | Two tuned gemm1 builds (`tests/gemm1_check.py --correctness --quick`) | build identity SAME |
| T2c | Bit-identical output over 5 runs, both epilogues | sha `59d442f663991659` (bf16) and `0cfa77fb04450594` (mxfp4-out) |
| T2d | `tests/asm_hash/asm_hash.py`: md5 of the assembly of all 20 dispatchable cells (4 dtypes × 5 token counts), compile only, about 12 min | byte-identical |

Notes:

- T2d is the decisive check for a pure refactor. Byte-identical assembly rules out
  any runtime perf change. The cold bench cannot resolve differences much below
  about 1%, but the hash catches any change at all.
- T2b prints `CHANGED ... <- instrs, regs, md5 differ`. That compares against stale
  reference values inside the script, so ignore it. The gate judges build identity
  against its own baseline.
- T2d's 20 cells are the 2026-09-09 dtype/token definitions, frozen in
  `tests/asm_hash/cells.json`. Its hashes are comparable with older baselines from
  `scripts/gluon_moe_asm_hash.py`.

## 2. Correctness suite: `tests/run_suite.sh` and `tests/compare_suite.py`

```bash
tests/run_suite.sh base 1 2 4 6        # baseline tree; arguments are HIP ordinals
tests/run_suite.sh new 1 2 4 6         # changed tree
python tests/compare_suite.py bench_out/gluon_moe_regression/suite_base \
                              bench_out/gluon_moe_regression/suite_new
```

- **What it runs:** every `op_tests/triton_tests/moe/test_moe_gemm_gluon*.py` (14
  files), one pytest process per file, spread round-robin over the given GPUs.
- **Run time:** about 25 minutes on 4 GPUs. A single-process run takes over 2 hours
  and hits the timeout.
- **Pass criterion:** `compare_suite.py` prints `SUITE PASS`, meaning no test that
  passes on the base fails or errors after the change.
- **Expected numbers:** 1811 passed / 58 failed / 22 skipped at commit `819123dbc`.
- **The 58 failures already fail at that commit and are tracked separately:**
  - 38 in `buffered_schedule`;
  - 16 in `scale_prep`;
  - 4 in `a8_scales`.

  The criterion is the same failure set, compared test by test, not zero failures.

## 3. Cold performance benchmark: `tests/cold_bench.sh` and `tests/compare_cold.py`

```bash
tests/cold_bench.sh before 0           # baseline tree; 0 = physical (amd-smi) GPU index
tests/cold_bench.sh after 0            # changed tree, SAME physical GPU
python tests/compare_cold.py before after
```

Runs land in `$OUT/harness/` (`val_<label>/`, `cold_<label>/`, the compile cache).
Extra arguments go to the harness, e.g. a 2-process smoke run:
`tests/cold_bench.sh smoke 6 --tolerant --precisions a4w4 --tokens 16 --rounds 1`.
`compare_cold.py` also accepts the path of any run directory holding a
`summary.json`.

Add `--tolerant` to `cold_bench.sh` when other users' jobs on other GPUs leave small
idle driver contexts on yours. Without it, the harness waits forever; see
`tests/run_bench_tolerant.py`. A real tenant on the same GPU still blocks the run.

**Harness.** `tests/harness/` (`run_bench.py`, `worker.py`), vendored from
`bench_out/gluon_flydsl_final_20260924/`.

- **Matrix:** GEMM1 with K=7168, N=4096 (intermediate size 2048), 33 experts, top-k 8,
  SiLU(gate)·up with gate/up split. Four dtypes (a4w4, a8w4, a8w8, a16w16) times five
  token counts (16, 64, 256, 1024, 4096).
- **Gluon config:** what the production ladder emits for each cell.
- **FlyDSL config:** the per-shape tuned winners, restricted to configs that pass
  correctness.
- **Run time:** about 1 hour on an idle GPU, 160 processes.

**Cold method:**

1. Setup work is excluded from timing: routing, quantization, scale sorting,
   shuffling, allocation, compilation and launch preparation all run before the loop.
2. Each loop iteration is a 768 MiB `flush.zero_()` followed by one prepared GEMM1, on
   the same stream. Nothing else runs inside the loop.
3. The first 40 warmup iterations are discarded, then 100 are measured. Early
   warmups run up to 25% slow.
4. Launches are queued asynchronously with no host sync inside the loop, so the GPU
   stays busy at least 99.86% of the measured window.
5. Durations come from the rocprofv3 kernel trace. 4 rounds alternate the backend
   order and are pooled into 400 samples per cell.

**Checks the harness enforces:**

- The traced window holds exactly the alternating fill/GEMM kernels, with
  consecutive dispatch IDs on one queue.
- Fill/GEMM timestamps overlap by at most 1 µs.
- The profiled GPU's PCI address matches the requested GPU.
- The GPU must be idle before each process starts.
- Before timing, every case must match a torch reference and pass 16 exact cold
  replays into a poisoned output buffer.

**Reading the result:**

- Compare labels only when they ran on the same physical GPU. Absolute medians
  differ by up to 3–5% between GPUs for identical code.
- FlyDSL code does not change with a Gluon change, so FlyDSL's movement measures the
  session drift. Judge Gluon by the "net" column of `compare_cold.py`.
- Treat net changes within about ±1% as noise.
- Reference numbers are in `docs/perf_gemm1_gluon_vs_flydsl.md`. On this machine,
  the original harness's label `cold1` on physical GPU 0 is the pre-KernelLayout
  baseline.
- Known failure: the trace auditor sometimes rejects one FlyDSL process for timestamp
  overlap. Resuming does not help, because the stored trace is re-parsed. Delete that
  process's `rNN_<case>_t<T>/` directory and `failures.json`, then rerun the same
  label.
- Do not benchmark with the asm-hash cells. They pin 2026-09-09 configs, for
  example 143.6 µs instead of 112 µs for a4w4 at T=16.

## Reference results (commit `1d3431e50`, 2026-10-13)

Stored in `tests/reference/`. Use them as the "before" side when the base commit is
`1d3431e50` or a codegen-neutral descendant, on the same machine and toolchain.
Otherwise record a fresh baseline.

| Check | Reference | Result |
|---|---|---|
| Gate | `reference/gate/{sched,build_id,asm}_baseline.txt`, used automatically by `gate.sh` when `$OUT` has none | verdict PASS. T1 856 passed, T2a 18 passed, determinism shas `59d442f663991659` / `0cfa77fb04450594` 5/5 |
| Suite | `reference/suite_1d3431e50/*.txt`, the FAILED lines and summary per file | 1811 passed / 58 failed / 22 skipped. Compare with `compare_suite.py tests/reference/suite_1d3431e50 $OUT/suite_<label>` |
| Cold bench | `reference/cold_1d3431e50_phys5/`, `summary.json` and `report.md`. `summary.json` is the run as recorded, with the pre-`dcef897ca` knob names (`B_IN_REG`, ...); `summary_allow_via_lds.json` is the same run with them renamed to `ALLOW_*_VIA_LDS` and inverted | physical GPU 5 (BDF `0000:96:00.0`), `--tolerant`, 160/160 processes, 0 failures. Compare with `compare_cold.py tests/reference/cold_1d3431e50_phys5 <label>`, on physical GPU 5 only |

Run conditions for the cold bench: physical GPUs 0–3 were running other users' jobs
throughout. The same code measured on the same GPU on 2026-10-13 (`cold_klayout_v3_g5`)
agrees within −0.32%..+0.23% for Gluon and −0.56%..+0.59% for FlyDSL.

Cold medians and p99 in µs. Gluon/FlyDSL below 1 means Gluon is faster. Gluon is
faster in 14 cells, tied in 1 (a8w4 T=4096 at 0.997; ties are within ±0.5%) and slower in 5.

| dtype | T | Gluon median | Gluon p99 | FlyDSL median | FlyDSL p99 | Gluon/FlyDSL |
|---|---:|---:|---:|---:|---:|---:|
| a4w4 | 16 | 112.5 | 116.4 | 140.1 | 143.7 | 0.803 |
| a4w4 | 64 | 114.2 | 118.4 | 140.7 | 144.2 | 0.812 |
| a4w4 | 256 | 122.8 | 127.0 | 179.0 | 183.5 | 0.686 |
| a4w4 | 1024 | 199.5 | 208.6 | 237.5 | 244.4 | 0.840 |
| a4w4 | 4096 | 625.1 | 667.2 | 698.3 | 726.9 | 0.895 |
| a8w4 | 16 | 126.1 | 129.5 | 127.1 | 130.8 | 0.992 |
| a8w4 | 64 | 132.5 | 136.0 | 129.8 | 132.9 | 1.021 |
| a8w4 | 256 | 138.9 | 143.4 | 146.0 | 149.8 | 0.951 |
| a8w4 | 1024 | 270.1 | 283.6 | 279.0 | 300.0 | 0.968 |
| a8w4 | 4096 | 970.7 | 1030.7 | 974.0 | 1030.8 | 0.997 |
| a8w8 | 16 | 234.6 | 239.6 | 227.0 | 230.9 | 1.034 |
| a8w8 | 64 | 242.8 | 246.8 | 230.4 | 237.5 | 1.054 |
| a8w8 | 256 | 254.0 | 257.7 | 256.2 | 260.8 | 0.992 |
| a8w8 | 1024 | 361.4 | 370.5 | 382.2 | 392.3 | 0.946 |
| a8w8 | 4096 | 1151.1 | 1210.8 | 1159.3 | 1217.3 | 0.993 |
| a16w16 | 16 | 461.0 | 469.4 | 419.6 | 433.6 | 1.099 |
| a16w16 | 64 | 464.7 | 472.8 | 422.2 | 433.2 | 1.101 |
| a16w16 | 256 | 434.2 | 441.6 | 504.6 | 516.8 | 0.860 |
| a16w16 | 1024 | 683.1 | 698.5 | 1072.5 | 1100.3 | 0.637 |
| a16w16 | 4096 | 2227.2 | 2285.7 | 3468.2 | 3502.6 | 0.642 |

## Vendored dependencies

Copied, not moved; the originals are untouched. Edits are limited to paths
(`Path(__file__)`-relative), lint fixes, and the cuts listed here.

| In `tests/` | Origin | Changes |
|---|---|---|
| `harness/run_bench.py`, `worker.py`, `bench_common.py`, `case_config.py`, `fly_build.py`, `cases.json`, `base_cases.json` | `bench_out/gluon_flydsl_final_20260924/` | Runs, cache and lock go to `$HARNESS_OUT`. `cases.json` points at the local `base_cases.json`. The hashed-source list names the vendored files. `libtriton.so` is located through the imported `triton` package. |
| `harness/bench_utils.py` | `bench_out/gluon_scale_ptr_unify_20260907/a4/bench_utils.py` | Only the functions the harness calls. `PY` is the running interpreter, the llc comes from `TRITON_HIP_DS_AGPR_LLC`, and the inert `CLAUDE_JOB_DIR` is no longer set. |
| `harness/legacy_a4_common.py`, `harness/legacy_fly_build.py` | `bench_out/gluon_flydsl_small_tokens_20260908/{bench_common,fly_build}.py` | Renamed. Lint-only edits. |
| `harness/legacy_dense_common.py` | `bench_out/gluon_flydsl_a8a16_20260907/bench_common.py` | Renamed. Lint-only edits. |
| `asm_hash/asm_hash.py`, `asm_hash/cells.json` | `scripts/gluon_moe_asm_hash.py`; cells resolved from `bench_out/gluon_flydsl_dtype_tokens_20260909/cases.json` | Reads the frozen cells. Shares `harness/bench_common.py`: the 0909 and final harnesses had byte-identical copies. |
| `gemm1_check.py` | `scripts/gluon_moe_gemm1_check.py` | Correctness path only. The perf path needed `~/moe_rerun/cold_bench.py`. Takes `--job` instead of `CLAUDE_JOB_DIR`. |
| `probes/ref_probe.py`, `corr_probe.py`, `corr_probe_q.py` | `~/moe_rerun/` | Repo path from `__file__`. Lint-only edits. |

Outside the repo and not vendored (toolchain): the venv (`PY`, which provides
torch, Triton, FlyDSL, ROCm and the `amd-smi`/`rocm-smi`/`rocprofv3` tools), the
patched llc (`TRITON_HIP_DS_AGPR_LLC`), and the built `aiter/jit/*.so`.
