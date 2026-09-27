# intj launch migration exceptions

The migration targets Triton and Gluon `JITFunction` launches through a
bound `make_launcher`: 238 call sites call a lazily built handle directly
(see [Direct handles](#direct-handles)); 134 still go through
`intj.compat.launch` and 47 through the tree-local `launch_tuned` bridge. This
branch needs the sibling
intj checkout (`develop` at or after `b4e9f1f`, which added autotune and
heuristics support to `make_launcher`) installed in the environment or on
`PYTHONPATH`; Aiter does not yet publish an intj dependency. The source imports with Python 3.10 and Triton 3.7.1, and all
changed Python files parse under Python 3.10. GPU smoke tests below ran with
Triton 3.7.1 and 3.8.0 on gfx942. Selected chunk-delta tests and benchmarks
also ran on gfx950 with Triton 3.8.0; other gfx950/gfx1250-only paths remain
untested.

These are the 30 remaining tracked Triton-style bracket launches. FlashKDA has
none: all five of its kernels (Triton K1, K2 and segment scan, Gluon K1 and
K2) launch through cached, device-bound `make_launcher` handles in
`flash_kda.py`, the autotuned K1/K2 included (see below). Most remaining sites
need a runtime capability or launch behavior intj cannot preserve.
The direct Iris kernels also have unsupported `tl.tensor` annotations; the fused
Iris path remains pending Iris-enabled multi-GPU validation. Line numbers refer
to this branch.

| Site | Target | Reason |
| --- | --- | --- |
| `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py:409` | `torch.library.wrap_triton(_rotary_kernel)` | `torch.library.wrap_triton` is required for `torch.compile` behavior. |
| `aiter/ops/triton/attention/mha.py:720` | `_attn_fwd` | intj cannot resolve its inline `torch.Tensor` parameter annotations, starting with `q_ptr`. |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py:278` | `bwd_kernel_causal` | intj refuses its `used_global_vals` reference to `tl_DROPOUT_USE_PYTORCH`. |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py:335` | `bwd_kernel_noncausal` | intj refuses its `used_global_vals` reference to `tl_DROPOUT_USE_PYTORCH`. |
| `aiter/ops/triton/attention/pa_decode_sparse.py:653` | `_sparse_mla_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/pa_decode_sparse.py:729` | `_sparse_mla_reduce_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:572` | `kernel` | A precompiled `triton.compile` result, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:610` | `kernel` | A precompiled Gluon/AOT kernel, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:642` | `_deepgemm_fp8_paged_mqa_logits` | Caller reads the returned kernel's `hash` for Triton's cache key. |
| `aiter/ops/triton/attention/pod_attention.py:199` | `pod_persistent` | Caller prints the returned kernel's `n_regs` and `n_spills`. |
| `aiter/ops/triton/attention/sparse_mla.py:607` | `_sparse_mla_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/sparse_mla.py:685` | `_sparse_mla_reduce_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/unified_attention.py:704` | `_unified_attention_kernel_2d_gfx1250` | intj refuses its `used_global_vals` reference to `_MAX_PROPAGATE_NAN_ALL`. |
| `aiter/ops/triton/comms/all_gather.py:226` | `_all_gather_kernel` | intj cannot resolve `heap_bases: tl.tensor`; its nested JIT helper also calls Iris `put`. |
| `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py:368` | `fused_pipeline_kernel` | Nested JIT helpers call Iris `load` and `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/reduce_scatter.py:236` | `_reduce_scatter_kernel` | intj cannot resolve `heap_bases: tl.tensor`; its nested JIT helper also calls Iris `load`. |
| `aiter/ops/triton/fusions/attn_res.py:458` | `attnres_fwd_kernel` | Custom cache stores the returned `CompiledKernel` and later calls `cached.run`. |
| `aiter/ops/triton/fusions/attn_res.py:543` | `attnres_fwd_kernel` | `_run_sequence` passes `res`, a tuple of residual tensors; intj raises `TypeError: unsupported argument 'res' of type tuple` (tuple arguments are unsupported, with or without `ATTN_RES_TRITON_AUTOTUNE`). |
| `aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py:384` | `_PRESHUFFLE_KERNEL_MAP[kernel_type]` | gfx1250 preshuffle can use `num_ctas > 1` (CGA multicast); intj refuses it. |
| `aiter/ops/triton/gluon/pa_decode_gluon.py:4369` | `paged_attention_kernel` | Both possible targets annotate `softmax_scale: float`, which intj cannot resolve; `paged_attention_decode_sliding_window_head_1` also reads `MFMA` and `VMEM_LOAD` globals. |
| `aiter/ops/triton/gluon/pa_decode_gluon.py:4438` | `paged_attention_kernel` | Can select `paged_attention_decode_v2_gluon_large_block_dot_kernel`, which references `COMPUTE` and `VMEM_LOAD` globals that intj refuses. |
| `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:604` | `_moe_gemm_a4w4_prefill` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a8w4.py:710` | `_moe_gemm_a8w4_prefill_gluon` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py:591` | `kernel` | Can select `paged_attention_decode_v2_gluon_large_block_dot_kernel`, which references `COMPUTE` and `VMEM_LOAD` globals that intj refuses. |
| `csrc/cpp_itfs/utils.py:515` | `self.triton_kernel` | `HsacoKernel._call` preserves the returned kernel alongside its separate HSACO path. |
| `op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py:37` | `_rmsnorm_constexpr_kernel` | Test checks Triton launch behavior under `torch.compile`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5828` | `empty_kernel` | Test asserts the returned `CompiledKernel.metadata.hash`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5866` | `kernel` | Explicit precompiled `triton.compile` result, not a `JITFunction`. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:656` | `_causal_conv1d_update_kernel_sglang` | `**pdl_kwargs` can include `launch_pdl=True`, which intj cannot launch. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:1199` | `_causal_conv1d_update_kernel_vllm` | `launch_pdl=_is_arch_support_pdl()` can request PDL. |

One cached `CompiledKernel.run` call also bypasses bracket syntax:
`aiter/ops/triton/fusions/attn_res.py:426`, part of the custom cache described
above. The 144 FlyDSL `.launch(grid=..., block=..., stream=...)` sites
in 77 files are excluded: they compile FlyDSL kernels with a different launch
ABI, not Triton/Gluon `JITFunction`s.

## Direct handles

[`aiter/ops/triton/utils/intj_handle.py`](../aiter/ops/triton/utils/intj_handle.py)
declares a handle next to its kernel (after the kernel's definition when the
module defines it, else before the first function launching it):

```python
_foo_launch = intj_handle(_foo_kernel, grid_arg=1, options={"num_warps": 4})

dev, stream = current_device_stream()
_foo_launch(dev)(stream, grid_x, *public_args)
```

Aiter imports kernel modules without a GPU, and `make_launcher` builds an
extension and queries the target, so `intj_handle` builds
`make_launcher(kernel, bind_device=True, ...).bind_device(dev)` on the first
launch on each device (a `functools.cache`), not at import; intj's
`@make_launcher(...)` decorator form builds when it decorates, so it would
need a GPU at import. The call is
make_launcher's positional form: defaults are passed explicitly, `str` /
dtype constexprs fixed at the site are `baked=`, and the values
`@triton.autotune` / `@triton.heuristics` assign are not passed. Grids are
`grid_arg` for tuple literals, make_launcher's default grid for a named
tuple, and `grid_cpp` for callable grids that read tuned values; values the
grid takes from the caller's frame become keyword-only grid arguments,
passed before the kernel arguments. `options=` may be a callable evaluated on
the device when the handle is built (options that depend on `get_arch()`).
A parameter that is also a compile option (`num_warps: tl.constexpr`) is
passed as both, as Triton does; `compat.launch` compiled those with default
options. Sites that pick one of several kernels with one signature (Gluon or
Triton, `solve_tril`'s three merge kernels) pick between handles.

190 of the 324 `compat.launch` sites and 48 of the 95 `launch_tuned` sites
moved. Warmed, gfx942, Triton 3.8.0, host time per launch statement:

| Site | direct handle | before | `kernel[grid]` |
| --- | ---: | ---: | ---: |
| `fused_recurrent_gated_delta_rule` (heuristics) | 4.7 us | 10.6 us (`launch_tuned`) | 38.3 us |
| `chunk_gated_delta_rule_fwd_h` (autotune, `grid_cpp`) | 4.6 us | 13.5 us (`launch_tuned`) | 40.2 us |
| `l2norm_bwd` (autotune, `grid_cpp`) | 4.3 us | 10.8 us (`launch_tuned`) | 26.1 us |
| `rms_norm` (plain JIT) | 4.4 us | 28.0 us (`compat.launch`) | 22.1 us |

Each spelling launched the same captured arguments, 200 launches per batch
without synchronizing, median of 40 batches, three processes. The direct
number includes `current_device_stream()` (0.36 us) and the HIP launch.

Unlike `compat.launch`, a handle does not check for Triton launch hooks,
CPU tensors, `TensorWrapper` arguments or interpreter mode (make_launcher
refuses `TRITON_INTERPRET=1` when the handle is built), and it is not keyed on
the JIT source or Triton's debug/instrumentation knobs: changing those after
the first launch keeps the handle built before. Like `compat.launch`, a
handle launch graph-breaks under `torch.compile` (Dynamo cannot trace
`_cuda_getCurrentRawStream`).

On gfx942 with Triton 3.8.0 and intj `39f2547`, evenly spaced samples (at
most 120 tests per file) of all 129 files under `op_tests/triton_tests`:
4586 passed and 2907 skipped. The rest fail with the same test ids on the
pre-conversion revision (`c26cf130d`): `test_layernorm.py` (48) and
`test_rmsnorm.py` (78) need `aiter.pertoken_quant`, three files (40 errors
each) need other ops `AITER_TRITON_ONLY` mode skips, and
`test_fav3_sage_compile.py` (17) hits a `fullgraph=True` graph break. Eleven files
also passed with Triton 3.7.1. A pytest plugin that re-ran every handle
launch as `kernel.run(...)` on cloned storages (tuned kernels with intj's
chosen config) found every output bitwise equal except where a kernel reads
out of bounds (`_layernorm_bwd_dwdb_triton_v2` reads the stride-0 `dy` of
`y.sum().backward()` with `x`'s stride; `chunk_delta_attn_gate_fwd` with a
`[H]` `dt_bias`). All 228 handles construct on gfx942; handles that only
gfx950/gfx1250 or Iris/FlyDSL paths reach are constructed only, not run.

### Sites left on `compat.launch` (134)

`compat.launch` serves sites whose launch varies per call in a way one
handle cannot: 

- Per-call compile options (58): `num_warps` / `num_stages` / `waves_per_eu`
  computed from the shape or a config lookup (cross_entropy, fused_fp8_quant,
  rope's thd-cached kernels, the MoE GEMMs, sparse MLA training/backward,
  gather_kv_b_proj, ...).
- Per-shape `**config` dicts carrying options and constexprs (52): conv,
  unified_attention, mhc, mla, pa_mqa_logits, quant_fp8_blockwise, hstu,
  mha_fused_bwd, fav3_sage, ....
- Per-call `str` / dtype / JIT constexprs (13): `pa_decode.py`'s
  `compute_type` (6), the moe topk kernels' `SCORE_MODE` (3), `mla_gluon`'s
  `REGIME`, and the split-K reduce `activation` in `gemm_a16w16.py` (2) and
  `fused_gemm_a16w16_quant_x.py`.
- A kernel chosen at runtime (11) together with one of the above, or with
  Gluon layout constexprs (`gemm_a16w16.py`, `batched_gemm_bf16.py`, gfx1250
  `gemm_afp4wfp4.py`); `fused_kv_cache.py`'s reshape-and-cache kernels differ
  in their extra constexprs.

### Sites left on `launch_tuned` (47)

`compat.launch` refuses `@triton.autotune` and `@triton.heuristics`
wrappers, so decorated kernels with a per-call variant stay on `launch_tuned`
from [`aiter/ops/triton/utils/intj_tuned.py`](../aiter/ops/triton/utils/intj_tuned.py):
it keeps the Triton call spelling, drops the values the decorators assign,
bakes `str`/dtype constexprs, and caches one bound `make_launcher` -- and so
one intj tuner cache -- per kernel, device, grid shape, compile options and
baked values. Triton still tunes on a miss; a hit launches natively.

- Per-shape `**config` dicts (40): the GEMM wrappers (a8w8, a16w16, fp4, fp8,
  batched, fused, feed-forward), `gmm.py`, `attn_res.py`'s
  `_launch_tune_kwargs`, and the gfx1250 Gluon branch of
  `fused_rms_mxfp4_quant`.
- Per-call `num_warps` (5): `activation.py`, `quant.py` (2),
  `fused_bmm_rope_kv_cache.py` (2).
- Per-call activation JIT constexpr (2): `activation.py`,
  `fused_reduce_act_mul_and_mxfp4_quant`.

One site stays on Triton: the tuple argument of `attn_res.py:543` (see the
table). The `fused_mxfp4_quant.py` heuristics that used
`functools.partial(_even_m_n, ...)` were rewritten as module-level,
closure-free defs (`even_m_n1`/`even_m_n2`/`even_m_n3`/`even_m_n1_iter` in
`aiter/ops/triton/utils/mxfp4_heuristics.py`) that intj's subset accepts.
`op_tests/triton_tests/quant/test_mxfp4_intj.py` asserts `make_launcher`
accepts them and that the intj launches match raw Triton launches. The Gluon
kernels these wrappers can also select (`_gluon_fused_rms_mxfp4_quant_kernel`,
`_gluon_fused_reduce_rms_mxfp4_quant_kernel`, gfx1250-only) keep their own
`functools.partial(_even_m_n, ...)` heuristics, so their handle or
`launch_tuned` raises `UnsupportedKernel` if that branch is ever taken; this
is untested here (no gfx1250 hardware).

## FlashKDA

`_FastLaunch` (`chunk_delta_attn/fast_launch.py`) is gone. `flash_kda_fwd`
validates that every input is on the current GPU, then launches each kernel
through `_intj_launcher`, a `functools.cache` of bound handles keyed by kernel,
device, compile options, baked cache modifiers, grid, the JIT source key and
the Triton debug/instrumentation/fpsan knobs; it reads the current stream
before every launch, so CUDA-graph capture works. The autotuned K1 and K2 pass
their decorated kernels, so intj tunes on a miss (the tuner configs, the
opt-in `CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1` space included) and launches
natively on a hit: K1's `IS_VARLEN`/`HAS_BIAS` and the scan's `HAS_H0` stay
`is not None` heuristics, lowered to C by intj; K2's grid reads the tuned `BW`
through `grid_cpp`. The Gluon JITs build their layouts inside the kernel
(K2's from `gl.num_warps()`), which intj requires; their gfx950 ISA is
unchanged. See
[the KDA direct-launch plan](superpowers/plans/2026-09-25-kda-direct-intj-launch.md).
Needs intj `develop` at or after `ad8c832`: earlier, a tuned handle retained
the argument tensors of its last tuning miss, which
`test_cached_native_handle_does_not_retain_tensor` catches.

On gfx942 with Triton 3.8.0 and intj `b4e9f1f`, evenly spaced samples (at most
120 tests per file) of 41 affected test files passed; mxfp4/fp4 and other
gfx950-only cases skip there and are untested. Selected files also passed with
Triton 3.7.1. The gated delta rule prefill and decode entry points, which have
no runnable test here, returned bitwise-equal outputs to upstream Triton
launches. `test_flash_attn_kvcache_sliding_window` in `test_mha_v3.py` is
excluded: it reads uninitialized workspace, and fails with NaNs on the
unconverted upstream revision too once that memory holds NaNs.

For a local Triton 3.7.1 gfx942 check (this ran against the earlier
callable-grid intj checkout; the path now points at intj `develop`):

```sh
AITER_TRITON_ONLY=1 PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/intj" \
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

After the FlashKDA direct-launch change (intj `develop`), `test_intj_launch.py`,
`test_flash_kda.py` and `test_chunk_delta_attn_fwd.py` pass on gfx942 with
Triton 3.7.1 and 3.8.0, and on 3.8.0 also with
`CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1` (four K1 and 24 K2 candidates, tuned by
intj). Forced-Triton FlashKDA outputs are
bitwise equal to the `_FastLaunch` revision on every `test_flash_kda.py`
case, default and autotuned. The Gluon kernels were compile-checked for
gfx950 only; the gfx950 numbers below predate this change, which has not run
on gfx950.

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
