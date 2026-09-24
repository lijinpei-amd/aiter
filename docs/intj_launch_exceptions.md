# intj launch migration exceptions

The migration targets Triton and Gluon `JITFunction` launches through
`intj.compat.launch`. This branch needs the sibling intj callable-grid checkout
installed in the environment or on `PYTHONPATH`; Aiter does not yet publish an
intj dependency. The source imports with Python 3.10 and Triton 3.7.1, and all
changed Python files parse under Python 3.10. GPU smoke tests below ran with
Triton 3.7.1 and 3.8.0 on gfx942. The gfx950/gfx1250-only paths remain untested
on this machine.

These are the 23 remaining tracked Triton-style bracket launches. Each needs a
runtime capability or return value that `intj.compat.launch` does not provide.
Line numbers refer to this branch and should be refreshed after edits.

| Site | Exact blocker |
| --- | --- |
| `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py:320` | `_k1_fast` is a custom `_FastLaunch` wrapper with a compiled-kernel cache; the enclosing function returns its `CompiledKernel`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:557` | `kernel` comes from `triton.compile`, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:595` | Same precompiled Gluon/AOT kernel path. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:627` | The caller reads `kernel.hash` to return Triton's cache key. |
| `aiter/ops/triton/attention/pod_attention.py:199` | The caller prints the returned kernel's `n_regs` and `n_spills`. |
| `csrc/cpp_itfs/utils.py:515` | `HsacoKernel._call` preserves the launched kernel's return value alongside its separate HSACO path. |
| `op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py:37` | The test checks Triton launch behavior under `torch.compile`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5828` | The test asserts `CompiledKernel.metadata.hash` after the launch. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5866` | `kernel` is an explicit `triton.compile` result, not a `JITFunction`. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:656` | `**pdl_kwargs` can contain `launch_pdl=True`, which intj cannot launch. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:1199` | Explicit `launch_pdl=_is_arch_support_pdl()` can request PDL. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:89` | `_FastLaunch` bypass preserves the wrapped kernel and its returned `CompiledKernel`. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:106` | Unhashable-argument bypass preserves the wrapped kernel and its returned `CompiledKernel`. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:148` | Cache capture needs `JITFunction.pre_run_hooks` and the returned `CompiledKernel`. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:985` | `_prepare_fast` is a `_FastLaunch` wrapper, not a `JITFunction`. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1062` | `_segment_fast` is a `_FastLaunch` wrapper with autotuning and its own cache. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1077` | `_g2.k2_ab_fused_fast` is a `_FastLaunch` wrapper. |
| `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1122` | `_seg_scan_fast` is a `_FastLaunch` wrapper with autotuning and its own cache. |
| `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py:403` | `torch.library.wrap_triton` is required for its `torch.compile` behavior. |
| `aiter/ops/triton/fusions/attn_res.py:458` | Its custom cache stores the returned `CompiledKernel` and later calls `cached.run`. |
| `aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py:367` | gfx1250 preshuffle can use `num_ctas > 1` (CGA multicast); intj's raw launch ABI refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:604` | The selected config can use `num_ctas > 1`; intj's raw launch ABI refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a8w4.py:710` | The selected config can use `num_ctas > 1`; intj's raw launch ABI refuses it. |

Two cached `CompiledKernel.run` calls also bypass bracket syntax:
`aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py:120` and
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
