# intj launch migration exceptions

Triton and Gluon `JITFunction` launches in this branch go through intj's
`make_launcher` (see [Launchers](#launchers)). This branch needs the sibling
intj checkout, `develop` at or after `2783079`, installed or on `PYTHONPATH`;
Aiter does not yet publish an intj dependency. Earlier revisions brought the
lazy build, `dynamic_options`, object constexprs looked up in C,
`assume_constant_globals`, `torch.Tensor` and `torch.Tensor | None`
annotations, Triton's conversion of annotated scalars and a `tl.constexpr`
parameter that sets the compile option it is named after. `2783079` adds the
launcher object, `kernel.__self__` (an `intj.launcher.Launcher`):
`make_launcher` only records its keywords as `__self__.options`, a
`MakeLauncherOption` whose `jit_func` is the kernel it launches.
`__self__.check()` runs a `delayed=` launcher's function, which supplies more
keywords and may query the GPU, then every check that needs neither the GPU
nor the target. The checks that need the target (an unknown `options=` name,
an invalid compile option in an autotune config, the C-side checks on bound
values) wait for `__self__.build()` or the first call (see
[intj 2783079](#intj-2783079)). Declaring a launcher does no GPU work, but
needs a host C compiler once per interpreter (intj builds its generic lazy
stub).

What still launches through Triton, and why: the 14 tracked Triton bracket
launches in the table below, each needing a runtime capability or launch
behavior intj cannot preserve. No `intj.compat.launch` call is left; the
last eight, gfx1250 Gluon paths that passed a tuple or a Gluon layout object
as a constexpr, now build those values in the kernel (see
[gfx1250 Gluon kernels](#gfx1250-gluon-kernels)). pod_attention and the metadata-redirect
test's `empty_kernel`, which read the returned `CompiledKernel`, now use a
`return_compiled=True` launcher.

Line numbers refer to this branch. The three Iris kernels are blocked only
by their Iris helpers (intj takes their `tl.tensor` annotations) and remain
pending Iris-enabled multi-GPU validation.

| Site | Target | Reason |
| --- | --- | --- |
| `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py:409` | `torch.library.wrap_triton(_rotary_kernel)` | `torch.library.wrap_triton` is required for `torch.compile` behavior. |
| `aiter/ops/triton/attention/pa_mqa_logits.py:789` | `kernel` | A precompiled Gluon/AOT kernel, not a `JITFunction`. |
| `aiter/ops/triton/comms/all_gather.py:226` | `_all_gather_kernel` | Blocked by Iris helpers only: its nested JIT helper calls Iris `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py:368` | `fused_pipeline_kernel` | Nested JIT helpers call Iris `load` and `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/reduce_scatter.py:236` | `_reduce_scatter_kernel` | Blocked by Iris helpers only: its nested JIT helper calls Iris `load`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py:386` | `_PRESHUFFLE_KERNEL_MAP[kernel_type]` | gfx1250 preshuffle can use `num_ctas > 1` (CGA multicast); intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:595` | `_moe_gemm_a4w4_prefill` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a8w4.py:717` | `_moe_gemm_a8w4_prefill_gluon` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/utils/device_info.py:57` | `kernel.__self__.options.jit_func` | Only while `torch.compile` traces: Dynamo cannot trace an intj launcher, and `fav3_sage_wrapper_func` must compile with `fullgraph=True` (see [torch.compile of fav3_sage](#torchcompile-of-fav3_sage)). |
| `csrc/cpp_itfs/utils.py:515` | `self.triton_kernel` | `HsacoKernel._call` preserves the returned kernel alongside its separate HSACO path. |
| `op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py:37` | `_rmsnorm_constexpr_kernel` | Test checks Triton launch behavior under `torch.compile`. |
| `op_tests/triton_tests/triton_metadata_redirect/test_metadata_redirect.py:5875` | `kernel` | Explicit precompiled `triton.compile` result, not a `JITFunction`. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:656` | `_causal_conv1d_update_kernel_sglang` | `**pdl_kwargs` can include `launch_pdl=True`, which intj cannot launch. |
| `op_tests/triton_tests/utils/causal_conv1d_update_refs.py:1199` | `_causal_conv1d_update_kernel_vllm` | `launch_pdl=_is_arch_support_pdl()` can request PDL. |

The 144 FlyDSL `.launch(grid=..., block=..., stream=...)` sites
in 77 files are excluded: they compile FlyDSL kernels with a different launch
ABI, not Triton/Gluon `JITFunction`s.

## Launchers

`make_launcher` only records its keywords; the first call checks them
(arguments, annotations, `@triton.autotune`/`@triton.heuristics`), builds
and launches; `kernel.__self__.check()` runs the checks that need no target
earlier (see above). So a kernel can carry its launcher as a decorator in a
module Aiter imports without a GPU. That is the default form:

```python
@make_launcher(grid_arg=1, options={"num_warps": 4})   # 418 kernels
@triton.jit
def _foo_kernel(...): ...

_foo_kernel(dev, stream, grid_x, *public_args)          # the kernel name launches
_foo_kernel.__self__.options.jit_func                   # the JIT function (tuner chain) it launches
```

The kernel's name is then the launcher. Code that needs the `JITFunction`
reads `kernel.__self__.options.jit_func`: the pa_decode_gluon AOT compile
tools and warmup, and the tests that read `arg_names`, `Heuristics`
predicates or source, or launch a Triton reference. Arguments the decorator
needs (a `grid_cpp` def, an option constant such as `mla_decode_ref`'s,
whose `kpack` depends on the arch) sit above the kernel. Keywords computed
only when the launcher is first used come from `delayed=`, a function the
launcher runs on its first `check()` or call: chunked_pa_prefill's gfx950
decode options (its `arch_info` import queries the GPU) and pa_mqa_logits'
per-mode annotations and fp8 pointer types. Until then a `delayed=`
launcher's `options` is an `intj.launcher.Delayed`, with no `jit_func`. A
kernel that is `None` when its Gluon module fails to import is decorated
inside that module, so the imported name is a launcher or `None`.
Launchers that used to differ only in something now passed per call are one
decorated kernel: the split-K reduce takes `KERNEL_NAME` and `activation` as
str constexprs; sage_quant_v, `_combined_routing`, `_moe_gemm_a8w8` and
`_sparse_attn_prefill_kernel` each serve two call sites.

Three other forms remain, 13 launchers, only where a decorator cannot give
what one kernel needs; no `functools.cache` factory is left:

```python
_k = make_launcher(_k, ...)                    # beside the kernel, over a wrapper applied after it
_bar_launch = make_launcher(_bar_kernel, ...)  # the JIT function keeps its own name
_TABLE = {mode: make_launcher(_kernel(mode), delayed=partial(_keywords, mode)) for mode in _MODES}
```

| Launcher | Form | Why not a decorator |
| --- | --- | --- |
| `attn_res` `attnres_fwd_kernel`, `attnres_fwd_sequence_kernel` | beside the kernel | Under `ATTN_RES_TRITON_AUTOTUNE` the kernel module wraps the kernel in `@triton.autotune` after defining it, and the launchers must wrap the tuner. Both launch one kernel, each baking what its layout never varies (see [attn_res's sequence layout](#attn_ress-sequence-layout)). |
| `aiter/mla.py` `_fwd_kernel_stage2_asm_launch` | named | `aiter/aot/triton/decode_mla.py` and `csrc/cpp_itfs/mla/asm_mla_decode_fwd.py` compile `_fwd_kernel_stage2_asm` ahead of time with Triton's own `triton.tools.compile`, which loads `aiter/mla.py` and takes that attribute as a `JITFunction`; its generated loader looks the kernel up in the HSACO by the same name, so the JIT function cannot move to another one. |
| `test_mla_stage2_merge.py` `_fwd_kernel_stage2_asm_launch` | named | Sweeps `num_warps`, `num_stages` and `waves_per_eu` per call over `aiter.mla`'s JIT function (above). |
| `rmsnorm` `_rms_norm_kernel_launch` | named | The kernel is also a device function: the Iris `reduce_scatter_rmsnorm_quant_all_gather` kernel calls it, and a launcher cannot be called from kernel code. |
| `pa_attention_kernel_test` `_direct_dot_launch`, `_direct_large_block_dot_launch` | named | Launch pa_decode_gluon's dot kernels' JIT functions with other options (`num_stages=1`, per-call `waves_per_eu`) than their decorators. |
| `pa_mqa_logits` `_GLUON_PAGED_MQA_LOGITS_LAUNCHERS` (6) | table | One launcher per (Preshuffle, VarCtx, KV-padded) mode over three Gluon kernels, each with its own hand-written types and `Assume` facts, from `delayed=` (they need `get_gfx()`); `return_compiled=True`. Each JIT function backs two launchers (KV-padded and unpadded facts), which one decorator cannot declare. |

Every call is positional. `current_device_stream()`
(`aiter/ops/triton/utils/device_info.py`) returns the current device and
its raw stream, as Triton launches on.

Defaults are passed explicitly, values that `@triton.autotune` /
`@triton.heuristics` assign are not passed, and literal arguments carry a
`# <parameter>` comment.

- **Per-call compile options** (`num_warps`, `num_stages`, `waves_per_eu`,
  `matrix_instr_nonkdim`, `kpack`, and `num_ctas` for `mha`) are
  `dynamic_options`, passed right after the grid. A site that passed a
  `**config` dict now passes `config.get(name, <Triton's default>)` for every
  one of those options and each kernel value by name (`config.get(name,
  <parameter default>)` where the parameter has a default), so a config
  entry that omits a key still compiles with Triton's default.
- **A kernel parameter that is also a compile option** (`num_warps:
  tl.constexpr`: many Gluon kernels, several GEMMs, mla, rope's thd-cached
  kernels) is passed once, as the parameter: intj sets the option from it on
  every call, as Triton does for the keyword. Neither `options=` nor
  `dynamic_options` names it (intj refuses the first; the second is a no-op).
  Such kernels used to have a `functools.cache` factory keyed on the option
  values (intj before `7e4acc9` refused the name as a dynamic option), and
  before that `compat.launch` compiled them with the default option.
- **`str` / `tl.dtype` / JIT-function constexprs** (pa_decode's
  `compute_type`, mla_gluon's `REGIME`, the split-K reduce `activation`,
  moe topk's `SCORE_MODE`, the split-K reduce's `KERNEL_NAME`, ...) are
  passed positionally and keyed by value. A value is baked with
  `extra_annotation={name: Constexpr(value=...)}` only where one decorated
  kernel has a single caller (the batched split-K reduce, the chunk-delta
  solves' `DOT_PRECISION`), or where a launcher serves one layout of a
  shared kernel (attn_res's two launchers).
- **Gluon layouts and tuples** are never arguments: intj bakes scalars, not
  layout objects or tuples. A kernel builds them from its constexpr ints and
  strings, inline or through a `gluon.constexpr_function` it calls (the
  gfx1250 GEMMs' warp bases from `num_warps`, the MXFP4 and MoE decode layout
  sets returned as a `SimpleNamespace`). Such a function imports what it
  needs locally: a global it reads (a class, a module constant) joins the
  kernel's `used_global_vals`, which intj refuses unless the launcher
  promises `assume_constant_globals=True`. A tuple of tensors becomes one
  pointer parameter per element, which the kernel packs back into a tuple
  (attn_res's sequence layout: `res0`..`res15`).
- **Kernels that read globals** get `assume_constant_globals=True` when every
  global is assigned once at import and never rebound (no `global`,
  `setattr`, `monkeypatch` or `reload` touches it): the one-kernel MHA
  backward (`tl_DROPOUT_USE_PYTORCH`), the gfx950 a8w8 blockscale GEMM
  (`_SUPPORTED_TILES`), pa_decode_gluon's head-1 sliding-window and
  large-block kernels (`MFMA`, `VMEM_LOAD`, `COMPUTE`) and the three Gluon
  pa_mqa_logits kernels (`_Use_2d_instr_shape_mfma_layout`). A global that
  only named `PropagateNan.ALL` is gone instead: those helpers write
  `tl.PropagateNan.ALL`, a module attribute Triton does not record (intj at
  `e2e75eb` cannot key a recorded enum member; see the report).
- **A hand-specialized kernel.** pa_mqa_logits used to compile its Gluon
  kernels with `triton.compile(ASTSource(signature, attrs))`; its launchers
  state the same types and asserted facts in `extra_annotation`
  (`Argument(type=..., specialize=Assume(Aligned(16), PointerRange(32)))`,
  `NEVER` where the attrs had none), one launcher per
  (Preshuffle, VarCtx, KV-padded) mode.
- **A caller that needs the compiled kernel** uses `return_compiled=True`:
  pa_mqa_logits returns the kernel's Triton cache key (the AOT benchmark
  zips that directory). With `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` the
  launch runs inside `AOTMetadataContext`, so the miss that compiles loads
  the prebuilt hsaco/json, as `triton.compile` did. pod_attention prints
  the returned kernel's `n_regs`/`n_spills`, and the metadata-redirect
  test checks its `metadata.hash`, the same way.
- **`torch.Tensor` / `tl.tensor` annotations** mean "exactly a tensor" to
  intj (Triton ignores them). A parameter that can be `None` (MHA forward's
  descale, alibi, dropout-mask and sink pointers) is annotated
  `torch.Tensor | None`: a tensor or `None`, keyed as Triton keys it.
- **Tuned kernels.** The values a tuning layer assigns (`BL`, `num_warps`
  and `num_stages` under `ATTN_RES_TRITON_AUTOTUNE`, the chunk-delta
  autotune spaces) come from the tuner; they are neither passed nor
  dynamic options. `attn_res`'s launchers take `num_warps` and `num_stages`
  per call unless `ATTN_RES_TRITON_AUTOTUNE` is set.
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
`compat.launch`, a launch graph-breaks under `torch.compile`: Dynamo traces
neither `_cuda_getCurrentRawStream` nor the launcher, a C function. The
kernels `fav3_sage_wrapper_func` launches, whose `fullgraph=True` support
`test_fav3_sage_compile.py` checks, go through
`device_info.launch_on_current_stream`, which launches the launcher's JIT
function through Triton while `torch.compiler.is_compiling()` (see
[torch.compile of fav3_sage](#torchcompile-of-fav3_sage)).

## gfx1250 Gluon kernels

The eight gfx1250 sites that stayed on `intj.compat.launch` are now
decorated kernels (`num_warps`, a parameter of most of them, sets the
option), compiled with the options the old call passed and nothing else:

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

## Sites converted after the exception survey

Sixteen of the 32 Triton launches this table used to list now go through
launchers (intj `develop` `e2e75eb`, Triton 3.8.0): the gfx950 Gluon
fp8_mqa_logits, sparse-MLA (pa_decode_sparse and sparse_mla, two kernels
each) and a8w8 blockscale kernels, the gfx1250 Gluon 2d unified attention,
the one-kernel MHA backward (two), MHA's `_attn_fwd`, pa_decode_gluon's two
launches (four kernels) and the AOT test's direct launch, pa_mqa_logits'
hand-compiled Gluon kernels and its Triton fallback, and `attn_res_gate`,
whose hand-written `CompiledKernel` cache is gone.

- **Tests.** gfx942: `test_attn_res.py` (251), `test_pa_mqa_logits_offset.py`
  (105), `test_mha.py` (200 sampled, plus every backward/dropout/int64 case)
  pass; pa_decode_gluon passes with its PS reduce on the Triton kernel (the
  C++ reduce needs CK headers, on the parent too). gfx950 (MI350X): the
  affected files (fp8_mqa_logits, gemm_a8w8_blockscale, pa_decode_sparse,
  sparse_mla, sparse_mla_dsv4_train, mla_v4_triton, mha, unified_attention,
  mla, attn_res, pa_mqa_logits_offset, pa_decode_gluon's three sets; at
  most 120-200 per file) pass exactly as on the parent.
- **Differential against Triton** (the plugin above): every call of the
  converted kernels is bitwise equal with identical ISA on gfx942 and
  gfx950, including a reduced pa_decode_gluon sweep that reaches all four
  of its kernels (832-1664 calls each per GPU).
- **pa_mqa_logits against the hand-compiled parent**, 155 calls (the offset
  test plus varctx, padded/unpadded KV, MTP and the Triton fallback): outputs
  bitwise equal and AMDGCN identical on gfx942 and gfx950. With
  `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` a launch loads the prebuilt
  hsaco/json (a corrupted one fails to load), as on the parent.
- **gfx1250** (no hardware): the 45 gluon unified-attention test cases that
  reach the 2d kernel compile to the parent's ttgir, LLVM IR and AMDGCN.
- **Host time** (gfx942, as above):

  | Site | before | after |
  | --- | ---: | ---: |
  | `attn_res_gate` (hand-written cache) | 20.0 us | 12.2 us |
  | `flash_attn_func` forward | 102.7 us | 45.1 us |
  | `deepgemm_fp8_paged_mqa_logits` (precompiled kernel) | 45.1 us | 33.2 us |

Found on the way: the earlier conversion passed `config.get("FP8_MIN",
-240.0)` / `("FP8_MAX", 240.0)` at seven unified-attention and MLA sites,
replacing the kernels' own `torch.finfo(e4m3_dtype)` defaults, so FP8 output
clamped at +-240 on gfx950/gfx1250 (OCP e4m3, +-448). They default to
`torch.finfo(e4m3_dtype)` again; a gfx950 output of 400 now comes back as
416 (its fp8 value) instead of 240.

## Decorators as the default form

100 kernels moved from a named launcher or a factory onto a
`@make_launcher` decorator (252 decorated kernels before, 352 after), and
pod_attention and the metadata-redirect test left the exception table
(intj `develop` `027dacd`, Triton 3.8.0). The launchers that are still not
decorators, with the reason for each, are listed under
[Launchers](#launchers).

- **Tests, gfx942.** At most 30 tests per file over all 129 files under
  `op_tests/triton_tests`: 2058 passed, 19 failed, 1 error; the parent
  (`490c589f8`): 2054 passed and the same 19 failures and error by test id
  (`test_fav3_sage_compile.py`, the two pa_decode_gluon sliding-window sets
  that need CK headers, the HIP-rope import in
  `test_fused_reduce_qk_norm_rope_swa_write.py`). The extra passes are
  `test_sage_quant_intj.py`'s new launcher check and `test_gmm.py` cases the
  parent's run skipped for memory. The FlashKDA, Sage, MXFP6/MXFP8,
  fast-transpose and metadata-redirect files in full: 226 passed (parent 225).
- **Tests, gfx950 (MI350X).** At most 20 tests per file, all 129 files:
  1877 passed, 23 failed, 1 error on this revision (`test_sparse_mla.py`
  rerun after fixing its import of a removed launcher); 1874 passed with
  the same failures and error by test id on the parent.
- **Differential against Triton** (the plugin above), gfx942 and gfx950:
  89 of the 100 newly decorated kernels ran (with small driver calls for
  the gated-delta-rule, mla_decode, sparse-prefill, MXFP6-pack and test
  reference kernels the samples miss), every call bitwise equal with
  identical ISA. The metadata test's `empty_kernel` runs in its own test.
  Not run: the 8 gfx1250 Gluon kernels (declared, and their extensions
  built on gfx942), `_conv2d_3x3_nchw_kernel` (no arch has a
  `conv_3x3_nchw` config, so the router never picks it; built only) and
  `pod_persistent` (its module does not import on the parent either).
- **Arity.** Every call without starred arguments passes exactly the
  arguments its launcher takes (317 calls, checked against each launcher's
  grid form, dynamic options and tuned or baked names).
- **Host time** (gfx942, as above): no change, as expected for the same
  launcher under another name.

  | Site | before | after |
  | --- | ---: | ---: |
  | `fast_transpose_2d` (bound handle → decorator) | 18.7 us | 18.5 us |
  | `layer_norm` (named → decorator) | 18.8 us | 18.4 us |
  | `gemm_a8w8`, split-K (baked `KERNEL_NAME` → str constexpr) | 30.8 us | 29.8 us |

## Option parameters

intj `develop` `7e4acc9` and later feed a `tl.constexpr` parameter named like a
compile option into that option on every call, and refuse a static `options=`
entry that names one. The 46 `functools.cache` factories that fixed such an
option per launcher are gone: their 51 kernels are decorated and called
directly, the value passed once, as the parameter (see
[Launchers](#launchers)). Nothing else changed at the call sites; options that
are not parameters stay `dynamic_options`. The gfx942 int8 smoothquant Gluon
kernel, whose decorator fixed `options={"num_warps": 4}` over its own
`num_warps` parameter, no longer imported under those intj revisions; it now
takes the value from the parameter. None of the 51 kernels is autotuned, so no
tuner is shared that used to be separate.

Verified with intj `develop` `ff4d3f1` against the parent (`8294a80a0`, run
with intj `0f2226f`, the last revision it imports under), Triton 3.8.0,
`AITER_TRITON_ONLY=1`:

- **Tests.** The 54 test files that reach a converted wrapper, at most 300
  tests per file (full runs of the moe a8w8-blockscale and int8-smoothquant
  files), plus drivers for kernels no test reaches (the non-GQA thd-cached
  rope kernels, the unified-attention 3d kernel on gfx942, the gfx942 int8
  Gluon kernel, the gfx950 Gluon afp4wfp4 kernel, `gemm_afp4wfp4_preshuffled_scales`,
  `bench_cache_copy`'s kernel), on both revisions.
  gfx942, the same counts in every file: 4951 passed, 5964 skipped, 1 error (the HIP-rope import).
  gfx950 (MI350X): 7898 passed, 18 failed, 2527 skipped, 1 error; the parent
  7897 passed, 19 failed. The failures have the same ids (`test_mha_with_sink.py`
  cases and the HIP build in `test_fused_mxfp4_quant.py`) except one
  `test_mha_with_sink.py` dropout case that failed only in the parent's
  sampled run; alone it fails 4 of 4 times on both revisions.
- **Outputs.** Every tensor a public wrapper call returned, and every tensor
  argument after the call, hashed per test and call index (inputs seeded per
  test): 3,059 calls of 48 wrappers on gfx942 and 5,222 of 61 on gfx950 are
  bitwise equal to the parent. The other 8 (gfx950) differ only in
  uninitialized memory: the `torch.empty` padding rows of
  `fused_clamp_act_mul`'s preshuffled scale (an unconverted Triton kernel), and
  a `y` buffer `fused_gemm_afp4wfp4_a16w16` does not write under
  `skip_reduce`.
- **Code.** The AMDGCN of every compile of the 29 converted kernels that run
  on these GPUs (plus the int8 kernel), on each GPU they ran on, is identical
  to the parent's, with the same compile options, once debug info is stripped (a decorator line shifts the source
  line numbers, so the Triton cache keys differ).
- **Host and GPU time.** Warmed wrapper calls (17 cases on gfx942, 26 on
  gfx950; 3-9 alternating rounds, one process each, pinned GPU and core 16)
  and the bare launch (the parent's factory call included; 5-20 rounds): no
  consistent change. Host time is equal or lower (the factory lookup is
  gone), e.g. launch 4.06 → 3.94 us (`pa_decode_sparse`, gfx942); GPU time is
  equal. Per-process launch times are bimodal on both revisions, so a few
  3-6 round medians differed by up to 0.3 us in either direction; rerun
  with 20 processes each, they match.

Not run: the 22 gfx1250-only Gluon kernels (declared, and their extensions
built on gfx942); each call site passes, as the parameter, the value the
factory used to fix as the option.

## intj 2783079

Ported from intj `d0389ee` to `develop` `2783079`, Triton 3.8.0. Nothing in
the tree used what `2783079` removed or changed (`LauncherFactory`,
`bind_device=True`, module-level `spec_key`/`init_bound`, a positional
option, a `try` around `make_launcher`), so every launcher keeps its
keywords. What changed:

- **Refusals moved from decoration to `check()`.** The tests that asserted
  that declaring a launcher succeeds (`test_mxfp4_intj.py`'s heuristics
  tests) call `kernel.__self__.check()`; the launcher tests check
  `isinstance(kernel.__self__, intj.launcher.Launcher)`. `mla_prefill_reduce`
  (`aiter/mla.py`), which falls back to a PyTorch loop when its Triton path
  raises, re-raises `UnsupportedKernel`, now raised by the first call.
- **`options.jit_func` replaces named launchers.** pa_decode_gluon's three
  kernels, attn_res and the five fused MXFP4 kernels launch under their
  kernel's name; their JIT-function consumers read it from the launcher (see
  [Launchers](#launchers)). Of the 13 named launchers, the three whose JIT
  function must keep its own name remain. The AOT tools
  (`csrc/cpp_itfs/gluon_aot_tools/compile*.py`) take the JIT function from a
  launcher they load by name; the HSACO symbol is the JIT function's name,
  which is the attribute's.
- **No factory is left.** `mla_decode_ref`'s two kernels are decorated with
  option constants computed at import (the module already queries the GPU
  there, through `arch_info` and `is_hip()`); pa_mqa_logits' six Gluon
  launchers are a module-level table whose annotations come from
  `delayed=`; pa_attention_kernel_test's two direct launchers are module
  globals. chunked_pa_prefill's kernel module, which queried the GPU at
  import (through `arch_info`) to pick its gfx950 decode options and so no
  longer imported without a GPU as it did before the migration, picks them in
  `delayed=`.

Verified on gfx942 (MI300X) and gfx950 (MI350X), `AITER_TRITON_ONLY=1`, with
the dtypes plugin and the torch `per_tensor_quant`/`pertoken_quant` shims:

- **Every launcher checks and builds.** A sweep imports every module under
  `aiter/ops/triton` and every other module that declares a launcher (414),
  first with no visible GPU, then on one GPU, where it calls `check()` and
  then `build()` (which runs the checks that need the target) on every
  launcher reachable from a module global (also in module-level dicts, lists
  and namespaces). gfx942: 419 launchers, 7 of them `delayed=`, none
  refused; before this change (intj `2783079`), 409, none refused. The 11
  declared launchers it does not reach are in modules that fail to import
  here, on the base tree too (FlyDSL, `mori` and `matplotlib` missing;
  pod_attention), plus the metadata-redirect test's in-function kernel and
  `test_mla_sparse.py`, a script. gfx950, where FlyDSL and `matplotlib` are
  installed: 426 launchers, 7 of them `delayed=`, none refused; 4 not
  reached.
- **Imports without a GPU.** 148 of the 414 modules import with
  `HIP_VISIBLE_DEVICES=""` on gfx942 (151 on gfx950); each of the others
  fails on the base tree too (`arch_info` needs a GPU or `jax`, plus the
  dependencies above), except `kda_launchers.py`, a test helper the base tree
  does not have, which imports `arch_info`. Before this change chunked_pa_prefill's kernel module
  was the one module that imported on the base tree and not here.
- **Tests**, gfx942 / gfx950: `test_intj_launch.py` 24 passed, 1 skipped
  (two GPUs) on both; `test_flash_kda.py` 65 passed, 4 skipped (gfx950 Gluon
  K2) / 69 passed; `test_sage_quant_intj.py` 8 passed, 1 skipped (gfx950 FP4
  pack) / 9 passed; `test_mxfp4_intj.py` 13, `test_mxfp4_jit_import.py` 61,
  `test_attn_res.py` 251, `test_mla_decode_rope.py` 240,
  `test_pa_mqa_logits_offset.py` 105 and `test_chunked_pa_prefill.py` 16
  passed on both; `test_pa_decode_gluon.py` 1 passed, 2 failed on both (the
  sliding-window sets copy CK headers this checkout lacks; the base tree
  fails them the same way); `test_fused_mxfp4_quant.py` skips all 2356 cases
  on gfx942, and on gfx950 1356 passed, 952 skipped, 48 failed: the
  `test_fused_dynamic_mxfp4_quant_moe_sort` cases, which build the HIP
  `module_quant` (no CK headers), the same 48 ids as on the base tree.
- **Arch-dependent keywords resolve per GPU.** gfx942 gives `kpack=2` for
  both `mla_decode_ref` kernels and, after `check()`, no chunked_pa_prefill
  options and `*fp8e4b8` for pa_mqa_logits' pointers; gfx950 gives
  `kpack=1`, `num_warps=1, num_stages=1, waves_per_eu=2` and `*fp8e4nv`, the
  values the factories and import-time code computed.
- **pa_decode_gluon's AOT scripts** (gfx942). `pa_decode_gluon_aot_warmup`
  warms up the JIT functions through the launchers' options: the disassembly
  of its HSACOs (dot and large-block dot, each with the reduce kernel) equals
  the previous revision's.
  The direct launches of `pa_reduce_kernel_test.py` and
  `pa_attention_kernel_test.py` return bitwise the base tree's outputs (MD5).
  Their compiled paths stop where the base tree's do: the reduce kernel's C
  is generated, then the build needs CK headers; the attention kernel's
  `triton.compile` raises the same `CompilationError` on both trees.

Not run: `op_tests/cpp/test_pa_decode_gluon_aot.py` (its builds need CK
headers), `aiter/aot/triton/decode_mla.py` (unchanged: it still compiles the
JIT function `_fwd_kernel_stage2_asm`).

## torch.compile of fav3_sage

Upstream compiles `fav3_sage_wrapper_func` with `torch.compile(fullgraph=True)`
(`test_fav3_sage_compile.py`); since the migration its 17 tests failed with
`Unsupported`, first on `_cuda_getCurrentRawStream`, then, with the stream
passed in, on the launcher itself. The seven launches the wrapper reaches
(`sage_fwd`, `sage_quant_kernel`, `_q_smooth_int8_kernel`, `_rot_q_kernel`,
`_rot_k_only_kernel`, and `_compute_delta_s_kernel` at two sites) go through
`launch_on_current_stream(kernel, grid, *args)`
(`aiter/ops/triton/utils/device_info.py`). In eager mode it calls the
launcher on the current device and stream. While
`torch.compiler.is_compiling()`, it launches `kernel.__self__.options.jit_func`
through Triton with the launcher's `options` and its `dynamic_options`
values by name, which Dynamo traces into the graph. It suits only a
launcher that takes its grid as one tuple and has no `delayed=`,
`bind_values` or knob in `dynamic_options`, as these seven do. The Sage paths the
wrapper does not reach (`smooth_rotate_downcast_qk`, the MXFP4/MXFP6/F4F4 V
packers, `sage_fwd_mxfp4`) still call their launchers directly and
graph-break under `torch.compile`.

Verified with intj `2783079`, Triton 3.8.0, `AITER_TRITON_ONLY=1`:

- **gfx942.** `test_fav3_sage_compile.py` 17 passed;
  `test_fav3_sage.py` and `test_sage_quant_intj.py` 410 passed, 572 skipped,
  as before. All 16 variants of the wrapper (default, `return_lse`,
  `q_smooth`, both, Hadamard rotation with and without `q_smooth`, causal,
  `smooth_k=False`, in both layouts) compile with `fullgraph=True`, and each
  one's compiled-vs-eager maximum difference equals the base tree's.
- **gfx950 (MI350X)**, on a copy with the same code: the three files 999
  passed (the two eager files 982 before); the 16 variants compile, with the
  base tree's differences.
- **Host time** (gfx942): about 0.35 us more per launch than a direct call
  (4.66 vs 5.03 us, toy kernel); the default wrapper path launches twice
  (`sage_quant_kernel`, `sage_fwd`). Not measured on gfx950.

## FlashKDA

`_FastLaunch` (`chunk_delta_attn/fast_launch.py`) is gone, and so is the
`_intj_launcher` handle cache that replaced it. `flash_kda_fwd` validates that
every input is on the current GPU, then calls the five kernels, each decorated
with its intj launcher, as `kernel(device, stream, *grid, *args)`; it reads the
current stream before every launch, so CUDA-graph capture works. Gluon K1
fixes `num_warps=2` as an option; Gluon K2 and the segment scan take
`num_warps` per call (`dynamic_options`); the cache modifiers are str
constexprs, keyed by value. The autotuned K1 and K2 carry their tuners, so
intj tunes on a miss (the tuner configs, the opt-in
`CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1` space included) and launches natively on a
hit: K1's `IS_VARLEN`/`HAS_BIAS` and the scan's `HAS_H0` stay `is not None`
heuristics, lowered to C by intj; K2's grid reads the tuned `BW` through
`grid_cpp`. Triton's debug/instrumentation/fpsan knobs and `TRITON_INTERPRET`
are read at each launcher's first call and fixed after it. Tests that need a
never-called launcher, other tuner configs, a spy or a Triton reference
re-run the kernel modules under a stand-in `make_launcher`
(`op_tests/triton_tests/chunk_delta_attn/kda_launchers.py`). The Gluon JITs
build their layouts inside the kernel (K2's from `gl.num_warps()`), which intj
requires; their gfx950 ISA is unchanged. See
[the KDA direct-launch plan](superpowers/plans/2026-09-25-kda-direct-intj-launch.md).
K1's and K2's autotune keys include `CM_WS` / `CM_OUT`, so segmented and
unsegmented launches still tune apart as they did under per-modifier handles.
Verified with intj `develop` `0f2226f`, Triton 3.8.0, on gfx942 and gfx950
(MI350X): the three KDA test files pass in all four modes (default,
`AITER_FDA_USE_GLUON=0`, `CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1`, both); o and
final state are bitwise equal to the handle-cache revision on all 10
`_CASES` x {forced Triton, Gluon K1+K2}, default and autotuned; warmed
`flash_kda_fwd` host time is 5-12% lower on every route and GPU time is
unchanged.
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

## attn_res's sequence layout

`_run_sequence` passed its residuals to `attnres_fwd_kernel` as one tuple
(`res`), which intj does not decode, so it launched the JIT function through
Triton. The kernel now takes 16 pointer parameters, `res0`..`res15`, and packs
them back into the tuple its gather loop indexes: the residuals, padded to
`L2` with the first one as before, then `None`. More than 16 residuals
(`ATTN_RES_SEQ_SLOTS`), which the tuple also took, are stacked into one
tensor and take the packed layout: one device copy of the residuals, and
outputs within tolerance of the tuple launch's but not bitwise equal to them
(29 of 68 cases equal on gfx942, L 17-64). The copy and the packed kernel
take 0.05-0.47x the GPU time of the tuple launch, whose 32-row tile spills
(gfx942, L 17 and 32, the N and D of the GPU times below, bf16). Two
launchers launch the one kernel,
each baking what its layout never varies: `attnres_fwd_kernel`
(`attn_res_gate`, `layout="packed"`) bakes the 16 slots as `None`, so the
decode path passes no slot argument (passing them measured 0.4 us more per
launch on gfx942); `attnres_fwd_sequence_kernel` bakes the packed-only
pointers and flags. A device table of base pointers was not used: it needs a
host-to-device copy per call, or a cache keyed by pointer values, where 16
kernel arguments need neither. The sequence entry also takes the
Python fast paths `attn_res_gate` uses (no reshape of a 2-D contiguous
residual, memoized `next_power_of_2`, no INFO-log arguments below that level).

Verified with intj `develop` `2783079`, Triton 3.8.0, `AITER_TRITON_ONLY=1`,
on gfx942 and gfx950 (MI350X):

- **Tests.** `test_attn_res.py`: 255 passed on both GPUs (the base tree's 251
  plus the sequence launch checked bitwise against a Triton launch of the same
  kernel across L2 buckets, dtypes, output norm, scalars equal to 1.0 and
  token counts, and every residual count from 1 to 17, and 33, against the
  torch reference), and the same with
  `ATTN_RES_TRITON_AUTOTUNE=1`, except on gfx942
  `test_attn_res_gate_out_quant_matches_unfused_quant[64]`, which fails there
  on the base tree too. `test_mhc.py`, which imports the same kernel package:
  406 passed, 12 skipped on both GPUs, as on the base tree. Under
  `torch.compile` the sequence entry works without `fullgraph` and fails
  with it, before and after (gfx942).
- **Outputs.** 519 sequence-layout cases (L 1-16, fp32/bf16/fp16, with and
  without the output norm, eps and scale values, leading dimensions, and the
  shape sweep below) are bitwise equal to the base tree's and to upstream
  `874d253dd3`'s, on both GPUs.
- **Code.** The AMDGCN and kernel descriptor (kernarg size, registers) of all
  168 sequence compiles and 186 gate/packed compiles the comparison reaches
  are identical to the base tree's, on both GPUs.
- **Host time**, us per call: N=1, D=1024, bf16, output norm (`attn_res_gate`:
  N=1, B=8, D=7168, quant and `close_block`); median of 2000 batches of 8
  calls per process, median (and range) of 5 alternating processes, pinned to
  the GPU's NUMA node. "Launch" is the kernel call alone, arguments prebuilt.
  Upstream's numbers are the base tree's within 1.5% for the sequence layout.

  | | gfx942 before | gfx942 after | gfx950 before | gfx950 after |
  | --- | ---: | ---: | ---: | ---: |
  | sequence entry, L=2 | 66.0 | 11.9 (11.9-12.3) | 33.6 | 7.4 (7.4-7.6) |
  | sequence entry, L=16 | 110.4 | 18.5 (18.4-18.7) | 61.4 | 10.9 (10.8-11.0) |
  | sequence launch, L=2 | 37.2 | 4.6 (4.6-4.7) | 22.7 | 3.3 (3.3-3.4) |
  | sequence launch, L=16 | 62.2 | 5.5 (5.3-5.5) | 39.4 | 3.4 (3.3-3.6) |
  | packed entry, L=4 | 26.8 | 25.4 (24.8-25.5) | 18.0 | 16.9 (16.7-17.0) |
  | `attn_res_gate` | 17.3 | 17.0 (16.9-17.2) | 10.7 | 10.7 (10.6-10.9) |

- **GPU time** (`torch.cuda.Event` over 50 back-to-back launches queued
  behind a sleep, 3 alternating processes; rocprofv3 aborts under this torch):
  L in {2, 4, 8, 16}, N in {1, 64, 4096}, D in {1024, 4096, 7168}, bf16. The
  after/before ratio of the 36 shapes is 0.995-1.007 on gfx950 and
  0.995-1.028 on gfx942, median 1.000 on both; the gfx942 outlier (N=1,
  D=1024, L=2, 3.99 vs 4.10 us) re-measured equal over 200-launch batches
  (3.96-3.99 vs 3.96-3.97 us), as did the same launch through Triton.

Not run: CUDA.
