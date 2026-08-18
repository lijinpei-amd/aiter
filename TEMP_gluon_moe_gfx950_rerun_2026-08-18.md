# TEMP: gfx950 Gluon MoE kernel perf rerun, 2026-08-18

Scratch record of a measurement rerun. Not intended to live in the tree -- fold the
numbers into `docs/gluon_moe_gfx950_perf.md` / `docs/gluon_moe_gfx950_status.md` and
delete this file.

Shape: a4w4 (MXFP4 x MXFP4, per_1x32), H=7168, I=2048, E=33, topk=8.
Routing pinned with `--n-active` so the active-expert count matches
`AITER_MOE_NUM_EXPERT_ACTIVATED` in `op_tests/test_moe_2stage.py`: 33 everywhere except
T=1, where the harness constraint `n_active <= min(E, T*topk)` forces 8.

Stage 1 = swiglu gemm (N=4096, K=7168), stage 2 = down-proj (N=7168, K=2048).

## Method

Wall-clock from the benchmark harness is `torch.cuda.Event` around the launch and
includes host launch overhead, which pins every decode point to a ~100-120 us floor.
Kernel time therefore comes from rocprofv3:

```
PYTHONPATH=op_tests/op_benchmarks/triton/rocprof_shim \
rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum -d <out> -- <harness cmd>
```

From the rocpd sqlite DB: `kernels.duration` per dispatch, grouped by
`(name, grid_x)` across the whole run (not by consecutive rows -- the Gluon harness
interleaves its kernel with the in-tree Triton one), first fifth dropped as warmup,
mean of the rest. HBM bytes are `TCC_MISS_sum * 128 B` over the same tail.

- `TF/s`  = 2*M*N*K / t, with M = T*topk.
- `GB/s`  = measured HBM bytes / t, i.e. achieved bandwidth, not modelled traffic.

The `PYTHONPATH` shim is required: without it rocprofv3 SIGSEGVs in
`llvm::DenseMapBase<>::LookupBucketFor<>()` because its preloaded `libLLVM.so.23`
interposes libtriton's statically linked LLVM. See the note in
`docs/gluon_moe_gfx950_perf.md`.

Also note `rocprofv3` and `python` are not reliably on PATH in non-interactive shells
here; invoke both by absolute path out of the venv.

## Kernel time, Gluon vs in-tree Triton

`_moe_gluon_gemm1` / `_moe_gluon_gemm2` vs `_moe_gemm_a4w4`.

| T | st | gluon us | gl TF/s | gl GB/s | gl HBM MB | triton us | tr TF/s | tr GB/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 25.4 | 18.5 | 4927 | 125 | 31.7 | 14.8 | 3950 |
| 1 | 2 | 16.4 | 14.3 | 3829 | 63 | 16.5 | 14.2 | 3799 |
| 8 | 1 | 95.6 | 39.3 | 5395 | 516 | 107.4 | 35.0 | 4852 |
| 8 | 2 | 49.2 | 38.2 | 5267 | 259 | 51.6 | 36.4 | 5025 |
| 32 | 1 | 96.5 | 155.8 | 5357 | 517 | 119.3 | 126.0 | 4372 |
| 32 | 2 | 51.4 | 146.1 | 5097 | 262 | 57.6 | 130.4 | 4550 |
| 1024 | 1 | 281.8 | 1707.2 | 2157 | 608 | 394.8 | 1218.3 | 1572 |
| 1024 | 2 | 153.9 | 1562.7 | 2584 | 398 | 183.7 | 1309.6 | 2228 |
| 4096 | 1 | 936.3 | 2055.1 | 1621 | 1518 | 1068.1 | 1801.5 | 1380 |
| 4096 | 2 | 530.9 | 1812.0 | 2057 | 1092 | 634.8 | 1515.6 | 1830 |
| 16384 | 1 | 3488.8 | 2206.1 | 1660 | 5792 | 3790.0 | 2030.8 | 1429 |
| 16384 | 2 | 1981.3 | 1942.3 | 2202 | 4363 | 2257.5 | 1704.6 | 2021 |

Gluon leads Triton at every point by kernel time, decode included. Decode is
bandwidth-saturated (5.0-5.4 TB/s achieved); prefill is compute-bound and peaks at
2206 TF/s for stage 1 at T=16384.

Reproduces the earlier measurements within run variance (T=1024 st1 281.8 vs 278.5,
T=4096 st1 936.3 vs 903.6).

## Harness wall clock, for contrast

Same run, `bench_moe_gemm_gluon.py` reported numbers. Included only to show the
launch-overhead floor -- do not quote these as kernel performance.

| T | st | gluon us | triton us | speedup |
|---:|---|---:|---:|---:|
| 1 | 1 | 110.3 | 81.0 | 0.73x |
| 1 | 2 | 101.6 | 79.1 | 0.78x |
| 8 | 1 | 120.1 | 113.0 | 0.94x |
| 8 | 2 | 108.3 | 82.9 | 0.77x |
| 32 | 1 | 110.7 | 135.7 | 1.23x |
| 32 | 2 | 107.0 | 83.8 | 0.78x |
| 1024 | 1 | 320.8 | 411.7 | 1.28x |
| 1024 | 2 | 187.2 | 260.1 | 1.39x |
| 4096 | 1 | 969.3 | 1093.9 | 1.13x |
| 4096 | 2 | 662.9 | 773.9 | 1.17x |
| 16384 | 1 | 3685.1 | 3941.5 | 1.07x |
| 16384 | 2 | 2523.7 | 2772.5 | 1.10x |

The sub-1x decode entries are host overhead, not kernel behaviour.

## Compiled kernel identity

Verified by md5-matching the loaded code objects from the ATT captures against the
Triton cache. Stage 1 is
`~/.triton/cache/YLERAMHYFA3PPD5PBOJCSOL2CGMIYTYSBMFCRIJIZYZ7UJNKASPA/`, stage 2 is
`A5XYPZC56SYK6PGRUJYGQQFXKC3MWFVKDB2WYZ26CX5RMTBXLKEA/`; T=1024 and T=4096 run
identical binaries.

Stage 1: `num_warps=8`, `num_stages=2`, `shared=105952 B`, vgpr=136, agpr=8, sgpr=75,
no spills, 64 mfma / 132 ds_read / 9 s_barrier.

512-thread workgroup = 8 waves, and 105952 B of the 160 KB LDS budget, so exactly one
CTA is resident per CU. The tuned asm kernels are `wg=256` with `lds=55424` (stage 1),
so they get two CTAs per CU and can cover one workgroup's barrier with the other's
work. That is the mechanism behind the ATT finding that Gluon spends 26-33% of stall
cycles on `s_barrier` versus 8-11% for tuned.

## Tuned FlyDSL/CK: not measured, path is broken

Could not be re-measured. `op_tests/test_moe_2stage.py -q 4 -dim 7168,2048 -e 33 -k 8`
fails during kernel compilation at every token count (1, 8, 32, 1024, 4096, 16384),
on plain runs with no profiler attached:

```
ExpandIntegerOperand Op #2: t204: ch = llvm.amdgcn.raw.ptr.buffer.load.lds<
  (dereferenceable load (s128) ... addrspace 8), (dereferenceable store (s8192) ... addrspace 3)>
LLVM ERROR: Do not know how to expand this operator's operand!
```

The dispatcher picks a `flydsl_moe1_afp4_wfp4_bf16_*` stage-1 kernel for every shape
and it fails SelectionDAG legalization on the global->LDS async copy. Unchanged by
`AITER_FLYDSL_FORCE=0`, `AITER_BYPASS_TUNE_CONFIG=1` (dispatches with empty kernel
names and still aborts), or `-p f`. Under rocprofv3 the abort additionally deadlocks
in the chained signal handler instead of exiting.

Contributing factor: the merged `/tmp/aiter_configs/tuned_fmoe.csv` has no tuned row
for H7168/I2048/E33/k8, so this shape always takes the untuned fallback, which now
resolves to FlyDSL. Earlier the same tree dispatched the asm kernels for these shapes
with no source change in between.

Tuned reference numbers below are from the earlier rocprofv3 session, same method,
same day -- NOT from this rerun. T=16384 was never measured for tuned.

| T | st | tuned us | HBM MB | TF/s | GB/s | kernel |
|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | 28.5 | 126 | 16.5 | 4421 | `mfma_moe1_..._fp4q_sort_async` |
| 1 | 2 | 14.2 | 63 | 16.5 | 4437 | `kernel_moe_mxgemm_2lds` |
| 8 | 1 | 111.1 | 519 | 33.8 | 4671 | `mfma_moe1_..._fp4q` |
| 8 | 2 | 47.2 | 261 | 39.8 | 5530 | `mfma_moe2_..._t32x128x256_fp4o` |
| 32 | 1 | 95.1 | 520 | 158.1 | 5468 | `mfma_moe1_..._bf16_t32x32x256` |
| 32 | 2 | 47.5 | 268 | 158.2 | 5642 | `mfma_moe2_..._t32x128x256_fp4o` |
| 1024 | 1 | 216.6 | 718 | 2220.9 | 3315 | `mfma_moe1_..._fp4q` |
| 1024 | 2 | 173.5 | 895 | 1386.3 | 5159 | `mfma_moe2_..._t64x256x256_fp4o` |
| 4096 | 1 | 696.9 | 1570 | 2761.0 | 2253 | `mfma_moe1_..._fp4q` |
| 4096 | 2 | 563.5 | 1516 | 1707.3 | 2690 | `mfma_moe2_..._t64x128x256_fp4o` |

Every tuned stage-1 kernel except T=32 is an `_fp4q_` fp4-intermediate variant, which
writes fp4 where Gluon writes bf16. Those TF/s are flattered and are not
apples-to-apples; only the T=32 stage-1 row (tuned 158.1 vs Gluon 155.8) is a
like-for-like comparison.
