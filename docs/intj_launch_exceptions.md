# intj launch migration exceptions

The migration targets Triton and Gluon `JITFunction` launches through
`intj.compat.launch` or a bound `make_launcher`. This branch needs the sibling
intj callable-grid checkout
installed in the environment or on `PYTHONPATH`; Aiter does not yet publish an
intj dependency. The source imports with Python 3.10 and Triton 3.7.1, and all
changed Python files parse under Python 3.10. GPU smoke tests below ran with
Triton 3.7.1 and 3.8.0 on gfx942. Selected chunk-delta tests and benchmarks
also ran on gfx950 with Triton 3.8.0; other gfx950/gfx1250-only paths remain
untested.

These are the 37 remaining tracked Triton-style bracket launches. The three
FlashKDA Triton wrappers use native intj on cache hits; their first call still
uses Triton to select an autotuned config. Most other sites need a runtime
capability or launch behavior intj cannot preserve.
The direct Iris kernels also have unsupported `tl.tensor` annotations; the fused
Iris path remains pending Iris-enabled multi-GPU validation. Line numbers refer
to this branch.

| Site | Target | Reason |
| --- | --- | --- |
| `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py:320` | `_k1_fast` | Its Gluon JIT reads 12 global layout constants that intj refuses. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:94` | `self._kernel` | Explicit bypass uses Triton for comparison tests. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:111` | `self._kernel` | An unhashable argument cannot be guarded by the wrapper cache. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:157` | `self._kernel` | First-call capture uses Triton's pre-run hook to select the autotuned config; warmed Triton FlashKDA calls use native intj. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:985` | `_prepare_fast` | First-call autotuning uses Triton; warmed calls use bound `make_launcher(grid_arg=2)`. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1062` | `_segment_fast` | First-call autotuning uses Triton; warmed calls use bound `make_launcher(grid_arg=2)` with the selected `BW`. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1077` | `_g2.k2_ab_fused_fast` | Its Gluon JIT takes layout objects as constexprs, which intj cannot decode. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1122` | `_seg_scan_fast` | First-call capture uses Triton; warmed calls use bound `make_launcher(grid_arg=2)`. |
| `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py:403` | `torch.library.wrap_triton(_rotary_kernel)` | `torch.library.wrap_triton` is required for `torch.compile` behavior. |
| `aiter/ops/triton/attention/mha.py:720` | `_attn_fwd` | intj cannot resolve its inline `torch.Tensor` parameter annotations, starting with `q_ptr`. |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py:272` | `bwd_kernel_causal` | intj refuses its `used_global_vals` reference to `tl_DROPOUT_USE_PYTORCH`. |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py:329` | `bwd_kernel_noncausal` | intj refuses its `used_global_vals` reference to `tl_DROPOUT_USE_PYTORCH`. |
| `aiter/ops/triton/attention/pa_decode_sparse.py:653` | `_sparse_mla_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/pa_decode_sparse.py:729` | `_sparse_mla_reduce_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:557` | `kernel` | A precompiled `triton.compile` result, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:595` | `kernel` | A precompiled Gluon/AOT kernel, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:627` | `_deepgemm_fp8_paged_mqa_logits` | Caller reads the returned kernel's `hash` for Triton's cache key. |
| `aiter/ops/triton/attention/pod_attention.py:199` | `pod_persistent` | Caller prints the returned kernel's `n_regs` and `n_spills`. |
| `aiter/ops/triton/attention/sparse_mla.py:607` | `_sparse_mla_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/sparse_mla.py:685` | `_sparse_mla_reduce_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/unified_attention.py:704` | `_unified_attention_kernel_2d_gfx1250` | intj refuses its `used_global_vals` reference to `_MAX_PROPAGATE_NAN_ALL`. |
| `aiter/ops/triton/comms/all_gather.py:226` | `_all_gather_kernel` | intj cannot resolve `heap_bases: tl.tensor`; its nested JIT helper also calls Iris `put`. |
| `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py:368` | `fused_pipeline_kernel` | Nested JIT helpers call Iris `load` and `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/reduce_scatter.py:236` | `_reduce_scatter_kernel` | intj cannot resolve `heap_bases: tl.tensor`; its nested JIT helper also calls Iris `load`. |
| `aiter/ops/triton/fusions/attn_res.py:458` | `attnres_fwd_kernel` | Custom cache stores the returned `CompiledKernel` and later calls `cached.run`. |
| `aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py:367` | `_PRESHUFFLE_KERNEL_MAP[kernel_type]` | gfx1250 preshuffle can use `num_ctas > 1` (CGA multicast); intj refuses it. |
| `aiter/ops/triton/gluon/pa_decode_gluon.py:4359` | `paged_attention_kernel` | Both possible targets annotate `softmax_scale: float`, which intj cannot resolve; `paged_attention_decode_sliding_window_head_1` also reads `MFMA` and `VMEM_LOAD` globals. |
| `aiter/ops/triton/gluon/pa_decode_gluon.py:4428` | `paged_attention_kernel` | Can select `paged_attention_decode_v2_gluon_large_block_dot_kernel`, which references `COMPUTE` and `VMEM_LOAD` globals that intj refuses. |
| `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:604` | `_moe_gemm_a4w4_prefill` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a8w4.py:710` | `_moe_gemm_a8w4_prefill_gluon` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py:591` | `kernel` | Can select `paged_attention_decode_v2_gluon_large_block_dot_kernel`, which references `COMPUTE` and `VMEM_LOAD` globals that intj refuses. |
| `csrc/cpp_itfs/utils.py:515` | `self.triton_kernel` | `HsacoKernel._call` preserves the returned kernel alongside its separate HSACO path. |
| `op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py:37` | `_rmsnorm_constexpr_kernel` | Test checks Triton launch behavior under `torch.compile`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5828` | `empty_kernel` | Test asserts the returned `CompiledKernel.metadata.hash`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5866` | `kernel` | Explicit precompiled `triton.compile` result, not a `JITFunction`. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:656` | `_causal_conv1d_update_kernel_sglang` | `**pdl_kwargs` can include `launch_pdl=True`, which intj cannot launch. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:1199` | `_causal_conv1d_update_kernel_vllm` | `launch_pdl=_is_arch_support_pdl()` can request PDL. |

Two cached `CompiledKernel.run` calls also bypass bracket syntax:
`aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:129` (Gluon wrappers) and
`aiter/ops/triton/fusions/attn_res.py:426`. They are part of the custom caches
described above. The 144 FlyDSL `.launch(grid=..., block=..., stream=...)` sites
in 77 files are excluded: they compile FlyDSL kernels with a different launch
ABI, not Triton/Gluon `JITFunction`s.

For a local Triton 3.7.1 gfx942 check, with the intj checkout from this task:

```sh
AITER_TRITON_ONLY=1 PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" \
  /tmp/intj-compat-py312/bin/python -m pytest -q \
  'op_tests/triton_tests/test_softmax.py::test_softmax[1-128-fp32]' \
  'op_tests/triton_tests/test_topk.py::test_topk[dtype0-True-2-128256-1]' \
  'op_tests/triton_tests/test_cross_entropy.py::test_cross_entropy_forward_basic[dtype0-1-128]' \
  'op_tests/triton_tests/attention/test_pa_decode_sparse.py::test_pa_decode_sparse_vs_reference[False-False-True-136-512-16-1]' \
  'op_tests/triton_tests/test_gmm.py::test_gmm[False-trhsF-10-2-3-4]'
```

All five selected tests passed with Triton 3.7.1. Equivalent selected tests
passed with `/tmp/gb2/bin/python` (Triton 3.8.0), and an autotuned sparse
prefill smoke case matched its Torch reference on both versions.

The `test_fast_launch.py` and `test_flash_kda.py` suites in
`op_tests/triton_tests/chunk_delta_attn/` passed on gfx942 with both Triton
3.7.1 and 3.8.0: 86 passed, 3 skipped for each version. On gfx950, six
focused tests passed on the converted revision, including the native cache
hit, converted solve launch, and segmented Gluon K2 route; three matching
tests passed on the baseline. The full suite has not run on gfx950.

## gfx950 chunk-delta benchmark

On an MI350X (`gfx950`), compare isolated Aiter snapshots `28ba24c0a6`
(baseline) and `f48301958` (converted) with intj `8619f455`. The environment
was Python 3.14.4, Torch 2.15.0.dev20260816+rocm7.14, HIP 7.14.60850, and
Triton 3.8.0. intj's automatic tensor access resolved to `CPYTHON` because
its verified fast Torch layout does not cover Torch 2.15. Each result is the
median of five alternating baseline/converted runs on GPU 0, with 300 ms of
warmup and 500 ms in `triton.testing.do_bench`. Negative change means faster.

| Path and shape (B×T×H×K×V) | Baseline ms | Converted ms | Change |
| --- | ---: | ---: | ---: |
| FlashKDA 1×512×12×128×128, normal gfx950 Gluon route | 0.0432 | 0.0432 | 0.00% |
| FlashKDA 1×512×12×128×128, forced Triton route | 0.0467 | 0.0468 | +0.21% |
| FlashKDA 1×16384×12×128×128, normal gfx950 Gluon route | 0.5442 | 0.5441 | −0.02% |
| FlashKDA 1×16384×12×128×128, forced Triton route | 0.7869 | 0.7820 | −0.62% |
| Default pipeline 2×4096×16×64×64 | 2.1290 | 2.0919 | −1.74% |

FlashKDA uses `bench_flash_kda.py` with `AITER_FDA_USE_GLUON=1` or `0`;
the default pipeline uses `bench_chunk_delta_attn.py` with
`AITER_FDA_ENABLE=0`. Both scripts live in `op_tests/op_benchmarks/triton/`
and were run with `AITER_TRITON_ONLY=1`, `HIP_VISIBLE_DEVICES=0`,
`--warmup-ms 300 --rep-ms 500`, and the table's `--shape` values. GPU-event
times include host-induced gaps between eager launches, but exclude initial
JIT compilation and autotuning.

For the forced-Triton 512-token case, 100 warmed K1 prepare launches per
host-timed block took 12.5658 µs/call on the baseline and 11.8462 µs/call
with intj (−5.73%). This excludes GPU completion, allocations, JIT, and
autotuning; two processes per revision each contributed nine blocks. A
same-process A/B changing only the default pipeline's solve dispatch measured
2.1197 ms with Triton and 2.0917 ms with intj (−1.32%) over five alternating
rounds. Its output and `Akk` were bitwise equal. The FlashKDA pipeline
differences are within the observed run-to-run variation.
