# Tuned gfx950 register/buffer recipes

These recipes were tuned on MI350X for GEMM1 with 1,024 input tokens, N=2,048,
K=7,168, 33 experts, top-k=8, preshuffled B, split gate/up SiLU, and FP32 output.
Each JSON file contains complete `config` dictionaries named `b_in_reg` and
`b_in_lds`. Both use the independent buffer pipeline.

The [four-wave MXFP4-output follow-up](../../docs/gluon_moe_gfx950_register_perf.md)
records the separate T4096/N4096 benchmark near 615 us. These transferred recipes
regress on that workload; its faster measured option keeps the original LDS
pipeline and preshuffles B. The table below applies to the FP32-output workload
described above.

| Input dtype | Legacy baseline | Tuned B in registers | Tuned B in LDS | Fastest recipe |
| --- | ---: | ---: | ---: | --- |
| a4w4 / MXFP4 | 145.201 us | 139.741 us | 136.521 us | `b_in_lds` |
| a8w8 / MXFP8 | 256.992 us | 228.941 us | 219.642 us | `b_in_lds` |
| a16w16 / BF16 | 500.864 us | 421.983 us | 470.983 us | `b_in_reg` |

Results are medians of five round medians on GPU 7. Each round brackets the
candidates with legacy baseline measurements; the baseline column is the median
of the five baseline bookend means. Each measurement uses 40 warmups and 100
timed dispatches. A 768 MiB flush precedes each wrapper call; MX scale sorting
runs between the flush and GEMM. `rocprofv3` timings include only GEMM1.

All final recipes matched the legacy output bits through 16 cold replays per
round and the final timed output. All have zero VGPR spills and zero private
memory. The scale tensors stay in LDS for the selected MX recipes; independent
register-scale options were also screened and remain covered by regression tests.

The B-register recipes use the following effective controls. Buffer counts are
listed as A, B, A scale, B scale; zero-valued fields in a JSON recipe inherit
`NUM_LDS_BUFFER=3`. The effective unroll is the GCD of all four counts.

| Dtype | BLOCK_K | MFMA | Warps | Buffer counts | GCD | WARP_PIPELINE | WAIT_COMMIT_SCHEME | DS_READ_IN_MFMA |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| MXFP4 | 256 | 16,16,128 | 1,4 | 4,2,3,3 | 1 | 0 | 3 | 0 |
| MXFP8 | 128 | 16,16,128 | 1,4 | 3,3,3,3 | 3 | 1 | 3 | 0 |
| BF16 | 64 | 16,16,32 | 1,4 | 3,3,3,3 | 3 | 1 | 3 | 1 |

All three use `expert_mod=".cg"`, BLOCK_M=128, BLOCK_N=256,
MINI_BLOCK_M=64, MINI_BLOCK_N=128, and full-stage K prefetching. The complete
JSON dictionaries also pin scale layout, instruction tiling, and other controls.

## Reproduce or extend the search

Use a Python environment containing PyTorch, Triton, pytest, and `rocprofv3`:

```sh
python scripts/gluon_moe_register_tune.py \
  --dtype mxfp4 --gpu 7 \
  --candidates scripts/gluon_moe_register_tuned/mxfp4.json \
  --output bench_out/mxfp4_register_recheck \
  --warmup 40 --reps 100 --determinism 16 --rounds 5
```

Use `mxfp8` or `bf16` for the other files/dtypes. The output directory must be new.
The runner prepares B in its preshuffled layout, validates each candidate before
timing it, alternates candidate order across rounds, and writes raw profiler
traces, full configurations, resources, source hashes, and per-dispatch samples.
For a new search, pass a JSON list of `{"name": "candidate", "config": {...}}`;
those dictionaries override the dtype's regression baseline. `--env KEY=VALUE`
sets an import-time environment override for the whole batch, including its
baseline. Shorter warmups/sample counts are suitable for screening.

To use a saved recipe through `moe_gemm_gluon`, select its `config` dictionary and
pass it with `gate_up_split=True`. Set
`AITER_TRITON_MOE_GLUON_B_PRESHUFFLED=1` so the wrapper prepares/caches the B
layout, as the runner does. The dictionary's `B_PRESHUFFLED=True` setting instead
means the caller has already physically permuted the weights. These are opt-in
recipes for the measured workload; retune for a different shape or output mode.

Full screening results, final measurements, and independent trace/binary audits
are retained in `bench_out/register_tuning_20260907/` in the tuning checkout.
