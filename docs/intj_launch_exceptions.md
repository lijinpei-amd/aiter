# intj launch migration exceptions

Triton and Gluon `JITFunction` launches in this branch go through intj's
`make_launcher` (see [Launchers](#launchers)). This branch needs the sibling
intj checkout, `develop` at or after `429d8e1` (lazy build, `dynamic_options`,
object constexprs looked up in C; `9b789f5` or later for the kernel modules
to import under `TRITON_INTERPRET=1`), installed or on `PYTHONPATH`; Aiter
does not yet publish an intj dependency. Declaring a launcher does no GPU
work, but needs a host C compiler once per interpreter (intj builds its
generic lazy stub).

What still launches through Triton, and why: the 32 tracked Triton bracket
launches in the table below, each needing a runtime capability or launch
behavior intj cannot preserve. No `intj.compat.launch` call is left; the
last eight, gfx1250 Gluon paths that passed a tuple or a Gluon layout object
as a constexpr, now build those values in the kernel (see
[gfx1250 Gluon kernels](#gfx1250-gluon-kernels)).

Line numbers refer to this branch. The direct Iris kernels also have
unsupported `tl.tensor` annotations; the fused Iris path remains pending
Iris-enabled multi-GPU validation.

| Site | Target | Reason |
| --- | --- | --- |
| `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py:409` | `torch.library.wrap_triton(_rotary_kernel)` | `torch.library.wrap_triton` is required for `torch.compile` behavior. |
| `aiter/ops/triton/attention/fp8_mqa_logits.py:321` | `_gluon_fp8_mqa_logits_kernel` (gfx950) | intj refuses its `used_global_vals` reference to `_MAX_PROPAGATE_NAN_ALL`. |
| `aiter/ops/triton/attention/mha.py:745` | `_attn_fwd` | intj cannot resolve its inline `torch.Tensor` parameter annotations, starting with `q_ptr`. |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py:280` | `bwd_kernel_causal` | intj refuses its `used_global_vals` reference to `tl_DROPOUT_USE_PYTORCH`. |
| `aiter/ops/triton/attention/mha_onekernel_bwd.py:337` | `bwd_kernel_noncausal` | intj refuses its `used_global_vals` reference to `tl_DROPOUT_USE_PYTORCH`. |
| `aiter/ops/triton/attention/pa_decode_sparse.py:786` | `_sparse_mla_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/pa_decode_sparse.py:862` | `_sparse_mla_reduce_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:594` | `kernel` | A precompiled `triton.compile` result, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:632` | `kernel` | A precompiled Gluon/AOT kernel, not a `JITFunction`. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:664` | `_deepgemm_fp8_paged_mqa_logits` | Caller reads the returned kernel's `hash` for Triton's cache key. |
| `aiter/ops/triton/attention/pod_attention.py:199` | `pod_persistent` | Caller prints the returned kernel's `n_regs` and `n_spills`. |
| `aiter/ops/triton/attention/sparse_mla.py:607` | `_sparse_mla_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/sparse_mla.py:685` | `_sparse_mla_reduce_gfx950` | intj refuses its `used_global_vals` reference to `_MAX_PROP_NAN`. |
| `aiter/ops/triton/attention/unified_attention.py:759` | `_unified_attention_kernel_2d_gfx1250` | intj refuses its `used_global_vals` reference to `_MAX_PROPAGATE_NAN_ALL`. |
| `aiter/ops/triton/comms/all_gather.py:226` | `_all_gather_kernel` | intj cannot resolve `heap_bases: tl.tensor`; its nested JIT helper also calls Iris `put`. |
| `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py:368` | `fused_pipeline_kernel` | Nested JIT helpers call Iris `load` and `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/reduce_scatter.py:236` | `_reduce_scatter_kernel` | intj cannot resolve `heap_bases: tl.tensor`; its nested JIT helper also calls Iris `load`. |
| `aiter/ops/triton/fusions/attn_res.py:459` | `attnres_fwd_kernel` | Custom cache stores the returned `CompiledKernel` and later calls `cached.run`. |
| `aiter/ops/triton/fusions/attn_res.py:544` | `attnres_fwd_kernel` | `_run_sequence` passes `res`, a tuple of residual tensors; intj raises `TypeError: unsupported argument 'res' of type tuple` (tuple arguments are unsupported, with or without `ATTN_RES_TRITON_AUTOTUNE`). |
| `aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py:212` | `_gemm_a8w8_blockscale_kernel` (gfx950 Gluon) | intj refuses its `used_global_vals` reference to `_SUPPORTED_TILES`. |
| `aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py:438` | `_PRESHUFFLE_KERNEL_MAP[kernel_type]` | gfx1250 preshuffle can use `num_ctas > 1` (CGA multicast); intj refuses it. |
| `aiter/ops/triton/gluon/pa_decode_gluon.py:4366` | `paged_attention_kernel` | Both possible targets annotate `softmax_scale: float`, which intj cannot resolve; `paged_attention_decode_sliding_window_head_1` also reads `MFMA` and `VMEM_LOAD` globals. |
| `aiter/ops/triton/gluon/pa_decode_gluon.py:4435` | `paged_attention_kernel` | Can select `paged_attention_decode_v2_gluon_large_block_dot_kernel`, which references `COMPUTE` and `VMEM_LOAD` globals that intj refuses. |
| `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:620` | `_moe_gemm_a4w4_prefill` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a8w4.py:749` | `_moe_gemm_a8w4_prefill_gluon` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `csrc/cpp_itfs/pa_gluon_aot/pa_attention_kernel_test.py:591` | `kernel` | Can select `paged_attention_decode_v2_gluon_large_block_dot_kernel`, which references `COMPUTE` and `VMEM_LOAD` globals that intj refuses. |
| `csrc/cpp_itfs/utils.py:515` | `self.triton_kernel` | `HsacoKernel._call` preserves the returned kernel alongside its separate HSACO path. |
| `op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py:37` | `_rmsnorm_constexpr_kernel` | Test checks Triton launch behavior under `torch.compile`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5828` | `empty_kernel` | Test asserts the returned `CompiledKernel.metadata.hash`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5866` | `kernel` | Explicit precompiled `triton.compile` result, not a `JITFunction`. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:656` | `_causal_conv1d_update_kernel_sglang` | `**pdl_kwargs` can include `launch_pdl=True`, which intj cannot launch. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:1199` | `_causal_conv1d_update_kernel_vllm` | `launch_pdl=_is_arch_support_pdl()` can request PDL. |

One cached `CompiledKernel.run` call also bypasses bracket syntax:
`aiter/ops/triton/fusions/attn_res.py:427`, part of the custom cache described
above. The 144 FlyDSL `.launch(grid=..., block=..., stream=...)` sites
in 77 files are excluded: they compile FlyDSL kernels with a different launch
ABI, not Triton/Gluon `JITFunction`s.

## Launchers

`make_launcher` builds nothing until the first call (it validates arguments,
resolves annotations and analyzes `@triton.autotune`/`@triton.heuristics`
at declaration), so launchers are declared at module level, in modules that
Aiter imports without a GPU. Three forms, in order of preference:

```python
@make_launcher(grid_arg=1, options={"num_warps": 4})   # 248 kernels
@triton.jit
def _foo_kernel(...): ...

_bar_launch = make_launcher(_bar_kernel, dynamic_options=("num_warps",))  # 96

@functools.cache                                        # 60 factories
def _baz_launch(num_warps):
    return make_launcher(_baz_kernel, options={"num_warps": num_warps})
```

- **Decorator** on the kernel when that kernel has one launcher and nothing
  else uses it (no second launch configuration, test, re-export or
  `getattr` by name). The kernel name is then the launcher. A `grid_cpp`
  function sits above the kernel it decorates.
- **Named launcher** next to the wrapper that launches it otherwise, or when
  its arguments (a module constant, a `grid_cpp` def) live in the wrapper's
  module.
- **`functools.cache` factory** in three cases: a kernel parameter doubles as
  a compile option (`num_warps: tl.constexpr`, see below); the kernel is
  imported inside the wrapper (gfx950/gfx1250 Gluon modules); the kernel can
  be `None` when its Gluon module fails to import.

Every call is positional:

```python
dev, stream = current_device_stream()   # aiter/ops/triton/utils/device_info.py
_foo_kernel(dev, stream, grid_x, *public_args)
_bar_launch(dev, stream, grid, num_warps, *public_args)
_baz_launch(num_warps)(dev, stream, grid, *public_args)
```

Defaults are passed explicitly, values that `@triton.autotune` /
`@triton.heuristics` assign are not passed, and literal arguments carry a
`# <parameter>` comment. `current_device_stream()` returns the current
device and its raw stream, as Triton launches on.

- **Per-call compile options** (`num_warps`, `num_stages`, `waves_per_eu`,
  `matrix_instr_nonkdim`, `kpack`, and `num_ctas` for `mha`) are
  `dynamic_options`, passed right after the grid. A site that passed a
  `**config` dict now passes `config.get(name, <Triton's default>)` for every
  one of those options and each kernel value by name (`config.get(name,
  <parameter default>)` where the parameter has a default), so a config
  entry that omits a key still compiles with Triton's default.
- **A kernel parameter that is also a compile option** (`num_warps:
  tl.constexpr`: many Gluon kernels, several GEMMs, mla, rope's thd-cached
  kernels) cannot be a dynamic option (intj refuses it), so its launcher is a
  factory keyed on the option values, and the value is passed both to the
  factory and as the parameter, as Triton does. Before this change
  `compat.launch` compiled these sites with the default option instead.
- **`str` / `tl.dtype` / JIT-function constexprs** (pa_decode's
  `compute_type`, mla_gluon's `REGIME`, the split-K reduce `activation`,
  moe topk's `SCORE_MODE`, ...) are passed positionally and keyed by value.
  Values fixed at a site (the split-K reduces' `KERNEL_NAME`) are baked with
  `extra_annotation={name: Constexpr(value=...)}`.
- **Gluon layouts and tuples** are never arguments: intj bakes scalars, not
  layout objects or tuples. A kernel builds them from its constexpr ints and
  strings, inline or through a `gluon.constexpr_function` it calls (the
  gfx1250 GEMMs' warp bases from `num_warps`, the MXFP4 and MoE decode layout
  sets returned as a `SimpleNamespace`). Such a function imports what it
  needs locally: a global it reads (a class, a module constant) joins the
  kernel's `used_global_vals`, which intj refuses.
- **Tuned kernels.** The values a tuning layer assigns (`BL`, `num_warps`
  and `num_stages` under `ATTN_RES_TRITON_AUTOTUNE`, the chunk-delta
  autotune spaces) come from the tuner; they are neither passed nor
  dynamic options. `attn_res` picks its launcher by
  `ATTN_RES_TRITON_AUTOTUNE`.
- **A kernel picked at runtime** (`impl = gluon if ... else triton`) is one
  launcher per kernel and an `if` at the launch (gemm_a8w8_blockscale,
  gemm_afp4wfp4, fused_gemm_afp4wfp4_a16w16, mla, pa_decode_sparse,
  fused_kv_cache, causal_conv1d_decode, solve_tril, fused_mxfp4_quant).
- **Callable grids** that read only caller values became tuples (conv's
  `_mn_grid(M, K, config)` and friends); grids that read tuned values use
  `grid_cpp`, with caller values as keyword-only grid arguments passed
  before the kernel arguments.

Unlike `compat.launch`, a launcher does not check for Triton launch hooks,
CPU tensors, `TensorWrapper` arguments or interpreter mode, and it reads
Triton's debug/instrumentation/fpsan knobs once, at its first call. Like
`compat.launch`, a launch graph-breaks under `torch.compile` (Dynamo cannot
trace `_cuda_getCurrentRawStream`).

## gfx1250 Gluon kernels

The eight gfx1250 sites that stayed on `intj.compat.launch` now use
factories (`num_warps` is a parameter of most of these kernels), with the
compile options the old call passed and nothing else:

| Site | Kernels | Was passed | Now built in the kernel from |
| --- | --- | --- | --- |
| `gemm_a16w16.py` persistent | `gemm_a16w16_persistent{,_compute_bound}_kernel_` | tuple `WARP_BASES` | `num_warps` |
| `gemm_a8w8_blockscale.py` (plain and preshuffle) | four `_gemm_a8w8_blockscale*_kernel`s | tuple `warp_bases` | `num_warps` |
| `gemm_a16w16.py` | `_gemm_a16w16_{bandwidth,compute}_bound_kernel` | 5 layouts | `LAYOUT`, tiles, `gl.num_warps()` |
| `batched_gemm_bf16.py` | `_batched_gemm_bf16_{bandwidth,compute}_bound_kernel` | 5 layouts | `LAYOUT`, tiles, `num_warps` (shares the a16w16 helpers) |
| `gemm_afp4wfp4.py` preshuffle | `gemm_mxfp4_preshuffle_gfx1250` | 10 layouts | tiles, `num_warps` |
| `moe_op_gemm_a4w4.py` decode | `_moe_gemm_a4w4_decode` | 12 layouts | tiles, `num_warps`, swizzle/preshuffle flags, `GatherIndx`'s element width |
| `bench_cache_copy.py` | `simple_tdm_kernel` | `WMMA_LAYOUT` | `use_tdm` (WMMA on gfx1250, MFMA elsewhere), `num_warps` |

The host-side layout helpers became those constexpr functions
(`create_*_layouts` → `shared_layout_a/b`, `wmma_layout`), so nothing
outside the kernels builds these layouts any more.

No gfx1250 GPU was available. Verified instead (Triton 3.8.0, intj
`develop` `9075eaa`):

- **Identical code.** A pytest plugin made Aiter report gfx1250 and turned
  every launch of a `_gluon_kernels/gfx1250` kernel (and `simple_tdm_kernel`)
  into a warmup compile with the target forced to `hip:gfx1250`, leaving
  every other kernel to launch on the local gfx942. Run on this revision and
  its parent over the gluon cases of `test_gemm_a16w16.py`,
  `test_gemm_a8w8_blockscale.py`, `test_batched_gemm_bf16.py`,
  `test_moe_gemm_a4w4.py` and the preshuffle cases of `test_gemm_afp4wfp4.py`,
  plus sweeps the tests do not reach (batched and plain blockscale
  `compute_bound`; MoE decode over `num_warps`, swizzle, preshuffle, gather
  width, SwiGLU and tiles; MXFP4 over `num_warps`, tiles and buffers;
  `bench_cache_copy` configs): 1,475 compiles, 600 distinct gfx1250 binaries
  across the 13 kernels. The ttgir, LLVM IR and AMDGCN of every one are
  identical to the parent's once locations and debug info are stripped, and
  so are the compiled options.
- **Construction.** Every new launcher is declared without
  `UnsupportedKernel`, and its extension builds and decodes the test and
  sweep calls that go through a launcher on the host (`no_gpu=True`,
  heuristics applied by hand). Each call
  passes exactly the parameters the kernel takes, bound to the values the
  old call bound, plus compile options now passed at Triton's default
  (`kpack=1`, ...) and `MAYBE_LOOP_UNROLL=False`.
- **gfx950 (MI350X).** Only `simple_tdm_kernel` has a non-gfx1250 path
  (MFMA, async copy). Its intj launch returns bitwise the output of the
  parent's kernel under a plain Triton launch (the parent's `compat.launch`
  raised on the layout), within bf16 rounding of an fp32 reference, and the
  benchmark script runs.

Untested: running any of these kernels on gfx1250 hardware (outputs,
performance). The compile comparison covers the constexpr values the tests
and sweeps produce; the MoE decode tests mostly fail before launching on
both revisions (`moe_shuffle_scale` asserts a 2-D scale).

## Verification

With intj `develop` `429d8e1`, Triton 3.8.0, `AITER_TRITON_ONLY=1`, and a
test-harness plugin that binds `aiter.pertoken_quant` (a torch reference that
triton-only mode leaves unbound):

- **gfx942, tests.** Evenly spaced samples (at most 120 tests per file) of all
  129 files under `op_tests/triton_tests`: 5565 passed, 3185 skipped, 17
  failed, 2 errors. The pre-migration revision (`f6566d0db`, same intj and
  harness): 5557 passed, 25 failed, 2 errors; the 8 extra failures are
  `test_pa_decode.py` cases whose grid got a 0-d tensor, fixed here. Every
  failure and error has the same test id on both: `test_fav3_sage_compile.py`
  (17, `fullgraph=True` graph break) and two files that import HIP ops
  triton-only mode skips.
- **gfx950 (MI350X), tests.** At most 20 tests per file, all 129 files:
  1875 passed, 21 failed, 220 skipped, 2 errors; the pre-migration revision
  1829 passed, 67 failed. This change fixes `test_fp8_mqa_logits.py` (20),
  `test_fused_gemm_afp4wfp4_a16w16.py` (20, `compat.launch` refused the
  `@triton.heuristics` kernel it picks),
  `test_gemm_a8w8_blockscale.py` (5, the gfx950 Gluon kernel) and one
  `test_pa_decode.py` case; the remaining failures have the same ids on
  both (`test_fav3_sage_compile.py`, one `test_mha_with_sink.py` case, a
  HIP build in `test_fused_mxfp4_quant.py`, and two cases of
  `test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`
  whose kernel also differs from itself when Triton launches it twice).
- **intj `9b789f5`** (develop HEAD at the end), gfx942, at most 15 tests per
  file: 1173 passed, 460 skipped; failures only in the same three files.
- **Triton 3.7.1** (gfx942): 12 affected files, at most 10 tests each: 104
  passed, 16 skipped. Changed files parse under Python 3.10.
- **Differential against Triton.** A pytest plugin wrapped every launcher:
  each call cloned its tensor arguments' storages (aliasing kept), launched
  intj on the originals and `kernel.run(grid=..., **named, **options)` on the
  clones, then compared every tensor bitwise; a third, `return_compiled`
  launch compared the compiled kernel's options and ISA with Triton's.
  Over the gfx942 sample (at most 20 per file, plus larger rope, mhc, conv,
  hstu, rmsnorm, mla_decode_rope and causal_conv1d runs) and the gfx950
  sample, 247 distinct kernels and about 7,300 launches: every output
  bitwise equal, except kernels that accumulate with `atomic_add` (within 2%
  of the reference's range), `_layernorm_bwd_*`, which read the stride-0
  `dy` of `y.sum().backward()` out of bounds on both revisions, and two
  gfx950 calls of the prequant batched GEMM, which differ between two
  Triton launches as well. Every compiled kernel has Triton's
  options; the ISA is identical except where intj specializes a parameter
  annotated `int` / `tl.int64` that equals 1 as a constant, which Triton
  does not (an intj discrepancy; correct, but a different binary).
- **Construction.** Every module that declares a launcher imports, and
  declaring does no GPU work (Aiter's own `arch_info` still needs a GPU, or
  `jax`, at import, on both revisions); on gfx942 all 385 launchers
  (factories called with typical option
  values) build their extension with no `UnsupportedKernel`. Every launcher
  call without starred arguments passes exactly the arguments its launcher
  takes (checked against each launcher's grid form, dynamic options, baked
  and tuned names).
- **Host time**, warmed wrapper call, median of 40 batches of 200 calls,
  three alternating rounds (gfx942):

  | Site | before | after |
  | --- | ---: | ---: |
  | `gemm_a8w8` (`launch_tuned` → dynamic options) | 31.8 us | 26.2 us |
  | `rope_cached_thd_positions_2c_fwd` (`compat.launch` → factory) | 68.8 us | 19.1 us |
  | `rms_norm` (bound handle → `@make_launcher`) | 18.4 us | 18.3 us |

Not run: 127 of the 374 launched kernels, which no sampled test reaches
(constructed and built only). 26 of them sit behind the 181 sites converted
here from `launch_tuned` / `compat.launch`: the gfx1250 Gluon kernels,
`hstu` backward, `pa_mqa_logits`' Triton kernels, the flash-attention
`fused_atomic` backward, `mla_decode`, `conv2d_3x3_nchw`,
`gemm_afp4wfp4_preshuffled_scales` and a few more. CUDA is untested (no
NVIDIA GPU).

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
  'op_tests/triton_tests/attention/test_pa_decode_sparse.py::test_pa_decode_
  sparse_vs_reference[False-False-True-136-512-16-1]' \
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
