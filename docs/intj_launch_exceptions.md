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

What still launches through Triton, and why: the 11 tracked Triton bracket
launches in the table below, each needing a runtime capability or launch
behavior intj cannot preserve. No `intj.compat.launch` call is left; the
last eight, gfx1250 Gluon paths that passed a tuple or a Gluon layout object
as a constexpr, now build those values in the kernel (see
[gfx1250 Gluon kernels](#gfx1250-gluon-kernels)). pod_attention and both
metadata-redirect tests' `empty_kernel`, which read the returned
`CompiledKernel`, now use `return_compiled=True` launchers (see
[The metadata-redirect separate-compile test](#the-metadata-redirect-separate-compile-test)).
So does `csrc/cpp_itfs/utils.py`'s `HsacoKernel` (see
[HsacoKernel](#hsacokernel)).

Line numbers refer to this branch. The three Iris kernels are blocked only
by their Iris helpers (intj takes their `tl.tensor` annotations) and remain
pending Iris-enabled multi-GPU validation.

| Site | Target | Reason |
| --- | --- | --- |
| `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py:409` | `torch.library.wrap_triton(_rotary_kernel)` | `torch.library.wrap_triton` is required for `torch.compile` behavior. |
| `aiter/ops/triton/comms/all_gather.py:226` | `_all_gather_kernel` | Blocked by Iris helpers only: its nested JIT helper calls Iris `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py:370` | `fused_pipeline_kernel` | Nested JIT helpers call Iris `load` and `put`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/comms/reduce_scatter.py:236` | `_reduce_scatter_kernel` | Blocked by Iris helpers only: its nested JIT helper calls Iris `load`; intj compatibility lacks Iris-enabled validation. |
| `aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py:386` | `_PRESHUFFLE_KERNEL_MAP[kernel_type]` | gfx1250 preshuffle can use `num_ctas > 1` (CGA multicast); intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:595` | `_moe_gemm_a4w4_prefill` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/moe/moe_op_gemm_a8w4.py:717` | `_moe_gemm_a8w4_prefill_gluon` | Selected config can use `num_ctas > 1`; intj refuses it. |
| `aiter/ops/triton/utils/device_info.py:57` | `kernel.__self__.options.jit_func` | Only while `torch.compile` traces: Dynamo cannot trace an intj launcher, and `fav3_sage_wrapper_func` must compile with `fullgraph=True` (see [torch.compile of fav3_sage](#torchcompile-of-fav3_sage)). |
| `op_tests/triton_tests/torch_compile/test_compile_constexpr_mutation.py:37` | `_rmsnorm_constexpr_kernel` | Test checks Triton launch behavior under `torch.compile`. |
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
@make_launcher(grid_arg=1, options={"num_warps": 4})   # 426 kernels
@triton.jit
def _foo_kernel(...): ...

_foo_kernel(dev, stream, grid_x, *public_args)          # the kernel name launches
_foo_kernel.__self__.options.jit_func                   # the JIT function (tuner chain) it launches
```

The kernel's name is then the launcher. Code that needs the `JITFunction`
reads `kernel.__self__.options.jit_func`: the pa_decode_gluon AOT compile
tools and warmup, the mla AOT tools (through `aiter/aot/triton/mla_jit.py`),
the Iris kernel that calls rmsnorm's kernel as a device function (as
`_rms_norm`), and the tests that read `arg_names`, `Heuristics` predicates
or source, or launch a Triton reference. Arguments the decorator
needs (a `grid_cpp` def, an option constant such as `mla_decode_ref`'s,
whose `kpack` depends on the arch) sit above the kernel. Keywords computed
only when the launcher is first used come from `delayed=`, a function the
launcher runs on its first `check()` or call: chunked_pa_prefill's gfx950
decode options (its `arch_info` import queries the GPU) and pa_mqa_logits'
per-mode annotations and fp8 pointer types. Until then a `delayed=`
launcher's `options` is an `intj.launcher.Delayed`, with no `jit_func`, so
pa_mqa_logits records each Gluon kernel's JIT function with a small
decorator under the launcher's, for the other launcher it makes over it. A
tuning layer applied only under a knob is a decorator too, between the two
(attn_res's `_attn_res_tuning`, `@triton.autotune` under
`ATTN_RES_TRITON_AUTOTUNE`, else nothing). A
kernel that is `None` when its Gluon module fails to import is decorated
inside that module, so the imported name is a launcher or `None`.
Launchers that used to differ only in something now passed per call are one
decorated kernel: the split-K reduce takes `KERNEL_NAME` and `activation` as
str constexprs; sage_quant_v, `_combined_routing`, `_moe_gemm_a8w8` and
`_sparse_attn_prefill_kernel` each serve two call sites.

One other form remains, 11 launchers: more launchers over a decorated
kernel's JIT function, where one kernel needs several launchers (a decorator
declares one). For attn_res, `test_mla_stage2_merge.py` and
`pa_attention_kernel_test`, taking what they differ in per call measured
slower production launches; for pa_mqa_logits it would change the binaries
of a KV mode (reasoned, not measured). `HsacoKernel` makes its launchers at
run time (last row); no `functools.cache` factory is left:

```python
_k_other = make_launcher(_k.__self__.options.jit_func, ...)   # other options or baked values
_TABLE = {key: make_launcher(_k.__self__.options.jit_func, extra_annotation=...) for key in _KEYS}
```

A `delayed=` decorator's launcher has no `jit_func` before `check()`, so the
second launcher of such a kernel takes the JIT function a decorator recorded
under it (pa_mqa_logits).

| Launcher | Form | Why not a decorator |
| --- | --- | --- |
| `attn_res` `attnres_fwd_sequence_kernels` (5) | table over the decorated `attnres_fwd_kernel`'s JIT function (or tuner) | One launcher per `L2` bucket of the sequence layout, each baking that bucket's unused residual slots as `None` and what the packed layout adds; the decorator is `attn_res_gate`'s and the packed layout's. One sequence launcher passing all 16 slots measured 0.27 us more per launch at L=2 and 0.10 us at L=8 on gfx942 (see [attn_res's sequence layout](#attn_ress-sequence-layout)). |
| `test_mla_stage2_merge.py` `_fwd_kernel_stage2_asm_launch` | over a decorated kernel's JIT function | The script sweeps `num_warps`, `num_stages` and `waves_per_eu`, which `aiter.mla`'s decorator bakes. Taking them per call there makes each production call 31 positional arguments, past the 30 CPython passes directly: +0.33-0.47 us per launch on gfx942. With the grid as one tuple it stays at 30 and costs +22-46 ns, which production would pay for this script's sweep (see [The named launchers as decorators](#the-named-launchers-as-decorators)). |
| `pa_attention_kernel_test` `_direct_dot_launch`, `_direct_large_block_dot_launch` | over a decorated kernel's JIT function | Launch pa_decode_gluon's dot kernels' JIT functions with other options (`num_stages=1`, per-call `waves_per_eu`) than their decorators. Taking both per call on the production dot-kernel decorator added 0.05-0.11 us to each of its launches on gfx942 (the large-block kernel was not timed). |
| `pa_mqa_logits` other-KV-mode launchers (3): plain KV-padded, preshuffle and preshuffle-varctx unpadded | table (`_GLUON_PAGED_MQA_LOGITS_LAUNCHERS`, beside the three decorated Gluon kernels) over the JIT functions recorded under their decorators | Each Gluon kernel has a launcher for a KV cache whose row stride is a multiple of 16 and one for any stride, with different hand-written `Assume` facts (`delayed=`: they need `get_gfx()`). The decorator is the mode 132-byte KV tokens take in each kernel's default page (the benchmark's): unpadded for the plain kernel's page of one token, padded for every preshuffle page (8 or a multiple of 16 tokens). The table launchers run for a padded plain cache (a page of a multiple of 4 such tokens, or tokens padded to 16 bytes, the benchmark's `-p`) and for a preshuffle page of 8 odd-sized tokens. One launcher measuring the KV facts per call (`AUTO`) would compile the unpadded mode with two facts on `KV_buffer` and `scale_buffer` that it now omits (`NEVER`), and that hold for a 16-byte-aligned cache under 2 GiB: 16-byte alignment and the <2 GiB pointer range. That changes today's unpadded binaries, which the AOT mode's prebuilt artifacts hold. Reasoned, not measured: no such launcher was built or timed. |
| `csrc/cpp_itfs/utils.py` `_IntjKernel.launchers` | made on first call | `jit(kernel)` wraps whatever JIT function its caller passes; one `return_compiled=True` launcher per set of option keywords, taken as `dynamic_options` (see [HsacoKernel](#hsacokernel)). |

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
  solves' `DOT_PRECISION`), or where a launcher serves one layout, or one
  `L2` bucket, of a shared kernel (attn_res's launchers).
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
  (Preshuffle, VarCtx, KV-padded) mode. An `Assume` leaves the facts it
  omits measured, and intj cannot assume one fact and refuse another, so
  the launchers compile with facts the attrs lacked when they hold:
  `tt.divisibility` on `context_len_ptr` and `kv_indices`, and
  `tt.pointer_range` on the padded KV and scale pointers.
- **A caller that needs the compiled kernel** uses `return_compiled=True`:
  pa_mqa_logits returns the kernel's Triton cache key (the AOT benchmark
  zips that directory). With `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1` a
  launch that misses compiles and then loads the prebuilt hsaco/json, as
  `triton.compile` did (see [pa_mqa_logits' AOT mode](#pa_mqa_logits-aot-mode)).
  pod_attention prints the returned kernel's `n_regs`/`n_spills`, and the
  metadata-redirect tests check its `metadata.hash`, the same way.
  `HsacoKernel` returns it, as Triton's launch did. Such a
  launcher compiles without running on a zero-volume grid:
  `test_separate_compile_and_run` compiles that way inside the context and
  runs outside it.
- **`torch.Tensor` / `tl.tensor` annotations** mean "exactly a tensor" to
  intj (Triton ignores them). A parameter that can be `None` (MHA forward's
  descale, alibi, dropout-mask and sink pointers) is annotated
  `torch.Tensor | None`: a tensor or `None`, keyed as Triton keys it.
- **Tuned kernels.** The values a tuning layer assigns (`BL`, `num_warps`
  and `num_stages` under `ATTN_RES_TRITON_AUTOTUNE`, the chunk-delta
  autotune spaces) come from the tuner; they are neither passed nor
  dynamic options. `attn_res`'s launchers take `num_warps` and `num_stages`
  per call unless `ATTN_RES_TRITON_AUTOTUNE` is set, when `_attn_res_tuning`
  stacks the tuner under the decorator.
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
  function had to keep its own name remained; `aiter.mla`'s stage-2 merge
  and rmsnorm's kernel are decorated since, and `test_mla_stage2_merge.py`'s
  stays (see
  [The named launchers as decorators](#the-named-launchers-as-decorators)).
  The AOT tools
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
headers), `aiter/aot/triton/decode_mla.py` (then unchanged; it now compiles
the JIT function through `aiter/aot/triton/mla_jit.py`, run in
[The named launchers as decorators](#the-named-launchers-as-decorators)).

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

## Four sites converted after `ba40fc7e2`

Four rows left the table (15 to 11): attn_res's sequence layout, which
passed a tuple, and three launches that needed a `CompiledKernel`: two of a
kernel `triton.compile` built, and `HsacoKernel`'s, which returns the one
Triton's launch returned. Host time is per warmed call, the
median of per-process medians, against the caching each site used before
(spreads, setups and GPU-time checks in each site's section):

| Site | Was | Now | Host time, before → after |
| --- | --- | --- | --- |
| [attn_res sequence layout](#attn_ress-sequence-layout) | Triton launch of the JIT function, residuals as a tuple `res` | `res0`..`res15` pointer parameters, one launcher per `L2` bucket; more than 16 residuals take the packed layout | entry, L=2: 66.2 → 11.6 us (gfx942), 33.7 → 7.0 us (gfx950); launch, L=16: 62.5 → 5.4, 40.0 → 3.1 us |
| [pa_mqa_logits AOT mode](#pa_mqa_logits-aot-mode) | a Triton < 3.4 AOT loader: `kernel[grid]` on a `functools.cache`d `triton.compile` (upstream launched that `CompiledKernel` in both modes on >= 3.5 too) | loader deleted (unreachable: intj needs Triton >= 3.7); AOT mode enters its metadata context only on a miss | gfx950, AOT entry, plain: 23.14 upstream, 20.18 at `ba40fc7e2` → 18.42 us; launch alone 6.62-6.75 (`CompiledKernel`) → 2.57-2.60 us |
| [`HsacoKernel`](#hsacokernel), default mode | `JITFunction[grid]`, returning the `CompiledKernel` | `return_compiled=True` launchers per option set, a call plan per call shape; honours `stream` | plain call: 10.76 → 2.90 us (gfx950), 17.17 → 4.44 us (gfx942) |
| [metadata-redirect separate-compile test](#the-metadata-redirect-separate-compile-test) | `triton.compile(ASTSource)`, then `CompiledKernel[(1, 1, 1)](x)` | `return_compiled=True` launcher, compiled on a zero-volume grid inside the context | gfx942: 7.33 → 3.73 us with `current_device_stream()` per call, 3.11 us with both prebuilt; first call 1.9 → 63.3 ms (warm caches) |

GPU time is unchanged where measured: attn_res's after/before ratio has
median 1.000 over 36 shapes on both GPUs, pa_mqa_logits 17.32 vs 17.39 us
and 10.35 vs 10.30 us (rocprofv3), the redirect test's kernel 2.51-2.53 us on
both sides. `HsacoKernel` launches the binary Triton compiles (identical
AMDGCN in its tests); its GPU time was not measured.

What callers see change: attn_res's sequence layout with more than 16
residuals matches the tuple launch within tolerance, not bitwise;
`HsacoKernel` launches on the `stream` it was given, raises
`UnsupportedKernel` for an unknown keyword or a kernel intj refuses, and
fails under `torch.compile(fullgraph=True)`; `AITER_USE_HSACO=1` is
unchanged.

On the integrated tree (gfx942, intj `2783079`, Triton 3.8.0,
`AITER_TRITON_ONLY=1`): `test_attn_res.py` 272 passed;
`test_pa_mqa_logits_offset.py` 105 passed, 1 skipped (JIT mode);
`test_hsaco_kernel.py` 22 passed, 1 skipped (hip-python) on two GPUs;
`test_metadata_redirect.py` 4 passed. Each changed module imports on one
GPU, and the 25 launchers reachable from them (pa_mqa_logits' 11, attn_res's
6, the fusions package's 8 mHC ones) check and build; `HsacoKernel`'s, made
on first use, check on a toy kernel. Without a GPU the changed modules fail
to import for the reasons they do on `ba40fc7e2` (`arch_info` needs a GPU or
`jax`; `chip_info`, and with it the dtypes plugin, needs `rocminfo` to list
a GPU), except
`aiter.utility.triton.triton_metadata_redirect` and the pa_mqa_logits kernel
module, which import, and whose five launchers check, on both. Not rerun on
the integrated tree: gfx950, AOT mode, the host and GPU timings above.

## attn_res's sequence layout

`_run_sequence` passed its residuals to `attnres_fwd_kernel` as one tuple
(`res`), which intj does not decode, so it launched the JIT function through
Triton. The kernel now takes 16 pointer parameters, `res0`..`res15`, and packs
them back into the tuple its gather loop indexes: the residuals, padded to
`L2` with the first one as before, then `None`. More than 16 residuals
(`ATTN_RES_SEQ_SLOTS`), which the tuple also took, are stacked into one
tensor and take the packed layout: one device copy of the residuals, and
outputs within tolerance of the tuple launch's but not bitwise equal to them
(29 of 68 cases equal on gfx942, 27 on gfx950, L 17-64). GPU time per call
at L 17 and 32 (the N and D of the GPU times below, bf16), copy and packed
kernel against the tuple launch: 0.05-0.47x on gfx942; 0.06-0.66x on gfx950
except at D=1024 with N of 1 or 64, where it is 1.2-2.0x (N=1: 13.6 vs 16.7
us at L=17, 13.7 vs 26.0 us at L=32).

Six launchers launch the one kernel, each baking what its layout never
varies, so a call neither passes nor decodes it. `attnres_fwd_kernel`
(`attn_res_gate`, `layout="packed"`) bakes the 16 slots as `None` (passing
them measured 0.4 us more per launch on gfx942).
`attnres_fwd_sequence_kernels` holds one launcher per `L2` bucket (1, 2, 4,
8, 16); each bakes the packed-only pointers and flags, `L2`, and the slots
from `L2` up as `None` (one launcher passing all 16 slots measured 0.27 us
more per launch at L=2 and 0.10 us at L=8 on gfx942, none at L=16). A device
table of base pointers was not used: it needs a host-to-device copy per
call, or a cache keyed by pointer values, where kernel arguments need
neither. The sequence entry also takes the Python fast paths
`attn_res_gate` uses (no reshape of a 2-D contiguous residual, memoized
`next_power_of_2`, no INFO-log arguments below that level).

Verified with intj `develop` `2783079`, Triton 3.8.0, `AITER_TRITON_ONLY=1`,
on gfx942 and gfx950 (MI350X):

- **Tests.** `test_attn_res.py`: 272 passed on both GPUs (the base tree's 251
  plus the sequence launch checked bitwise against a Triton launch of the same
  kernel across L2 buckets, dtypes, output norm, scalars equal to 1.0 and
  token counts, and every residual count from 1 to 17, and 33, against the
  torch reference), and the same with `ATTN_RES_TRITON_AUTOTUNE=1`, except on
  gfx942 `test_attn_res_gate_out_quant_matches_unfused_quant[64]`, which
  fails there on the base tree too. `test_mhc.py`, which imports the same
  kernel package: 406 passed, 12 skipped on both GPUs, as on the base tree.
  Under `torch.compile` the sequence entry works without `fullgraph` and
  fails with it, before and after (gfx942).
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
  In the same rounds, one sequence launcher passing all 16 slots
  (`9b60a4ddc`) measured 12.1 (gfx942) and 7.25 (gfx950) for the L=2 entry,
  and within 0.1 us of the numbers below at L=16.

  | | gfx942 before | gfx942 after | gfx950 before | gfx950 after |
  | --- | ---: | ---: | ---: | ---: |
  | sequence entry, L=2 | 66.2 | 11.6 (11.5-11.7) | 33.7 | 7.0 (7.0-7.3) |
  | sequence entry, L=16 | 111.7 | 18.6 (18.5-18.7) | 62.6 | 10.7 (10.6-11.0) |
  | sequence launch, L=2 | 37.2 | 4.4 (4.4-4.5) | 22.6 | 2.9 (2.9-3.2) |
  | sequence launch, L=16 | 62.5 | 5.4 (5.3-5.4) | 40.0 | 3.1 (3.1-3.4) |
  | packed entry, L=4 | 26.5 | 25.4 (25.1-25.6) | 18.0 | 17.0 (16.8-17.3) |
  | `attn_res_gate` | 17.0 | 17.2 (17.0-17.4) | 10.5 | 10.5 (10.4-10.8) |

- **GPU time** (`torch.cuda.Event` over 50 back-to-back launches queued
  behind a sleep, 3 alternating processes; rocprofv3 aborts under this torch):
  L in {2, 4, 8, 16}, N in {1, 64, 4096}, D in {1024, 4096, 7168}, bf16. The
  after/before ratio of the 36 shapes is 0.989-1.007 on gfx950 and
  0.982-1.014 on gfx942, median 1.000 on both. Over 200-launch batches the
  gfx942 extremes re-measured equal (N=64, D=1024, L=2: 4.28 vs 4.28-4.29
  us, as did the same launch through Triton), or swapped sides between runs
  (N=64, D=4096, L=16: 23.2-23.3 vs 24.1-24.7 us, then 24.2-24.6 vs
  23.0-23.1 us).

Not run: CUDA.

## pa_mqa_logits' AOT mode

`deepgemm_fp8_paged_mqa_logits` had three Gluon paths: JIT mode and AOT mode
(`AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`) on Triton >= 3.5, both through the
six launchers above, and an AOT loader for Triton < 3.4, the last Triton
launch in the file: `kernel[grid](...)` on a `functools.cache`d
`triton.compile` of a signature-only stand-in kernel (with an extra dummy
pointer argument), its hsaco/json swapped in by `AOTMetadataContext`.
Upstream launched the same cached `triton.compile` result as a
`CompiledKernel` in both modes on Triton >= 3.5.

- **The Triton < 3.4 loader is deleted, not migrated: it cannot run.** intj
  needs Triton >= 3.7, and `make_launcher` refuses an older one already at
  decoration (`UnsupportedKernel: Triton >=3.7 is required`), so `import
  aiter` fails at its first module-level launcher (Triton 3.5.1:
  `_triton_kernels/quant/fast_transpose.py`); this file's own kernel module
  declares launchers too. Upstream's loader was broken as well: the < 3.5
  import branch never imported `get_cdna_version`, so the first call raised
  `NameError` (reached on Triton 3.8 with its version string set to 3.3.1),
  and the stand-ins lack the `num_block`, `CDNA_VERSION` and `ARCH`
  parameters its signature names. Nothing on Triton >= 3.7 reached
  `_compile_deepgemm_fp8_paged_mqa_logits` or the stand-ins; both are gone,
  with the per-call `triton_version >= Version("3.5.0")` test (0.65 us on
  MI350X: it parses a `Version` every call).
- **The AOT context is entered only on a miss.** Every AOT-mode call used to
  enter `AOTMetadataContext`, though only a launch that misses intj's cache
  compiles and loads a binary. In AOT mode the module now passes its three
  Gluon kernels to `require_metadata_redirect` (`triton_metadata_redirect.py`):
  building one's `CompiledKernel` with no redirect registered on the thread
  raises `MetadataRedirectMissing`, after Triton's `__init__` has read the
  files and before anything is loaded on the GPU. The entry launches outside
  the context and retries only on that exception, inside it; the retry finds
  the first attempt's compile in Triton's cache, and intj keeps nothing from
  the failed attempt. A compile of those kernels anywhere else in an
  AOT-mode process raises instead of loading the binary it compiled.
  `test_metadata_redirect.py::test_required_redirect_only_on_miss` covers the
  mechanism, on gfx942 only (its prebuilt kernel is a gfx942 binary). In AOT
  mode `test_pa_mqa_logits_offset.py` covers this file's use of it on gfx942
  and gfx950: the wide-output test checks that the kernels it ran were
  loaded from `configs/paged_mqa_logits/aot/`, and
  `test_paged_mqa_logits_aot_never_runs_its_own_compile` that a configuration
  with no prebuilt files raises. Outputs alone cannot tell: the compiled and
  the prebuilt binary compute the same logits. With the
  `require_metadata_redirect` call removed, these three tests fail on gfx942
  and gfx950, while on gfx942 the tests that predate them still pass. With
  the raise in `triton_metadata_redirect.py` removed, they fail on gfx942.

Verified with intj `2783079`, Triton 3.8.0, `AITER_TRITON_ONLY=1`, on gfx942
(MI300X) and gfx950 (MI350X, GPU 4), against upstream `874d253dd3`, in both
modes. No AOT artifact ships in the tree: per GPU, they were made from
upstream the way `bench_deepgemm_attention.py -aot` makes them (run JIT mode,
copy the returned key's Triton cache directory to
`configs/paged_mqa_logits/aot/<name>`), for the configurations below and the
offset test's.

- **Tests**, the same on both trees and GPUs: `test_pa_mqa_logits_offset.py`
  105 passed in JIT mode, 2 passed and 103 skipped in AOT mode (only the
  wide-output test runs there; this branch adds the AOT test above, which
  passes in AOT mode and is skipped in JIT mode); `test_fp8_mqa_logits.py`
  152 passed;
  `test_metadata_redirect.py` 4 passed on gfx942 (3 upstream), skipped on
  gfx950. `test_pa_mqa_logits_opus.py` (2) and
  `test_pa_mqa_logits_mxfp4_opus.py` (4) fail on both trees: their `test_*`
  functions take arguments pytest does not supply, and they do not reach
  this kernel.
- **Outputs**: 16 calls (plain, preshuffle and preshuffle-varctx at batch 1,
  the offset test's configuration; two seeds; a miss and a hit) are bitwise
  equal to upstream's in both modes on both GPUs.
- **Binary**: in AOT mode a configuration's first call enters the context
  once and the next 200 not at all; the launched kernel's hsaco is the
  artifact's and its json hash differs from the compile's. A direct launcher
  call outside the context raises `MetadataRedirectMissing`; a configuration
  with no artifact raises `FileNotFoundError`, as before. An artifact made by
  `bench_deepgemm_attention.py -aot` on this tree loads in AOT mode on
  gfx950 with the JIT run's `logits_diff`.
- **Host time** (MI350X GPU 4, `taskset` to NUMA node 1; batch 1, one token,
  64 heads, 128-dim, 256-token context, ChunkK 256, WavePerEU 2): median us
  per call of six alternating rounds, one process per tree and mode, each
  the median of 2000 batches of 8 calls with a synchronize after each batch
  outside the timer; in parentheses the range of the six process medians.
  "Before" is this branch at `ba40fc7e2`.

  | Mode | `deepgemm_fp8_paged_mqa_logits` | upstream | before | now |
  | --- | --- | ---: | ---: | ---: |
  | AOT | plain, page 1 | 23.14 (22.66-23.45) | 20.18 (20.00-20.30) | 18.42 (18.14-18.44) |
  | AOT | preshuffle, page 16 | 23.37 (23.06-23.78) | 20.33 (20.12-20.43) | 18.62 (18.24-18.74) |
  | AOT | preshuffle varctx, page 16 | 23.64 (23.48-24.07) | 20.27 (20.21-20.67) | 18.66 (18.19-18.73) |
  | JIT | plain, page 1 | 23.02 (22.84-23.22) | 19.36 (19.12-19.62) | 18.54 (18.40-18.68) |
  | JIT | preshuffle, page 16 | 23.33 (22.79-24.24) | 19.48 (19.20-19.83) | 18.68 (18.41-18.80) |
  | JIT | preshuffle varctx, page 16 | 23.38 (23.18-23.70) | 19.51 (19.23-19.80) | 18.68 (18.40-19.02) |

  The launch alone, replaying the entry's arguments, is 6.62-6.75 us
  upstream (`CompiledKernel[grid](*args)`) and 2.57-2.60 us through the
  launcher, before and now, in every row; inside a per-call
  `AOTMetadataContext` it is 3.03-3.06 us. Not measured on gfx942.
- **GPU kernel time** (rocprofv3, 500 launches, JIT mode), upstream vs now:
  17.32 vs 17.39 us (plain, batch 16, 16384-token context) and 10.35 vs
  10.30 us (preshuffle, page 64). AOT mode loads the same artifact on both
  trees. No kernel code changed.

Not verified: Triton 3.7 (intj's floor; only 3.8.0 here), a real Triton
< 3.4, gfx1250, concurrent misses on several threads in AOT mode, and
plain-kernel artifacts for a padded KV cache (`-p`, `PT` in the name).

## HsacoKernel

`csrc.cpp_itfs.utils.jit(kernel, stream=None)` returns an `HsacoKernel`,
called with Triton's spelling, `kernel[grid](*args, **kwargs)`. Nothing in
the tree calls it (`aiter`, `aiter/aot`, `csrc`, `op_tests` grepped; only
`aiter/aot/test/test_hsaco.py` uses the HSACO helpers beneath it), so it is
kept as a public helper with the same call signature.

**Default mode: intj.** The first call makes the launchers (construction
does no GPU work). Each call shape, the number of positional arguments and
the keyword names in order, is bound once with the kernel's signature into a
plan; a launch then only picks its arguments out of one tuple:

- keywords that name no parameter are compile options, as in Triton: one
  launcher per set of them, which it takes as `dynamic_options` (sorted, so
  keyword order does not matter);
- the parameters follow in declaration order, defaults applied;
- `return_compiled=True`, so the call still returns the `CompiledKernel`
  (no measurable cost);
- `grid` is an int, a 1-3 tuple or list of ints, or a callable returning
  one. The callable gets every parameter by name, as Triton's does, and is
  evaluated in Python before the launch: a `grid_py` launcher (with the
  grid passed through a thread-local, as `intj.compat` does) measured
  within 0.1 us of it, in either direction (gfx950 3.55 vs 3.63 us, gfx942
  6.84 vs 6.72 us), and Python evaluation keeps one launcher for every grid
  form;
- the launch goes on `stream` when one was given, else on the current
  device's current stream. A raw handle launches on the current device; a
  `torch.cuda.Stream` launches on its own device, made current around the
  launch when it is not. intj launches only on the current device: on
  another one a kernel-cache miss raises `UnsupportedKernel` and a hit fails
  in `hipModuleLaunchKernel` ("invalid resource handle"), although intj's
  `Usage.md` asks for the device to be current only the first time. The
  Triton path ignored `stream`; the HSACO path used it.

Launchers and plans belong to the JIT function, not to the wrapper, so
`jit(k)[grid](...)` built per call keeps hitting their kernel caches.
Refusals are loud: an unknown keyword raises `UnsupportedKernel` (Triton
raised `KeyError`), `device=`/`stream=` keywords and what intj cannot
launch (tuple arguments, kernels reading globals, `num_ctas > 1`, ...) raise
instead of launching through Triton. So do some arguments Triton launched
(gfx942, checked against upstream's `HsacoKernel`):

- a grid that is a `tuple` subclass, such as `torch.Size` (intj:
  `TypeError: grid must be an int, tuple or list of ints`);
- numpy integer grid dimensions, also when a callable grid returns them
  (`ValueError: grid dimensions must be ints in [0, 2**32)`, misleading for
  `np.int64(4)`);
- more than three grid dimensions, where Triton ignores the fourth on
  (`ValueError: grid must have between 1 and 3 dimensions`);
- a numpy integer for a constexpr (`TypeError: unsupported argument`).

A callable grid returning a bare int, which Triton refused (`TypeError`),
now launches.

**`AITER_USE_HSACO=1` stays as it was.** That mode is for deployment
without compiling at run time: it compiles an HSACO from Triton once, keeps
it under `BUILD_DIR` keyed by constexpr values and argument types, and later
processes load and launch that file through hip-python without Triton
compiling anything. intj cannot serve it: a launcher compiles through
`triton.compiler.compile` on a miss and builds its own C extension with the
host compiler on first use, and takes no prebuilt HSACO. Its run-time cost
lies elsewhere too: before launching, the HSACO path calls
`inspect.signature` and binds every call, stats the HSACO and reads the
metadata JSON every call, and logs two INFO records per launch once `aiter`
is imported. Without the logging and the launch itself that is 26-61 us
per call (the lower bound below), against 2.9-6.4 us for the whole intj
launch, and would be fixed in that code, not by replacing the launcher.

hip-python is installed on neither machine, so the HSACO path was run with
a stand-in `hip.hip` module (ctypes over `libamdhip64`, the argument buffer
packed at natural alignment and passed as `extra`). With it, the path
returns the intj path's bytes for the cases it can launch, and shows three
defects of the mode on both GPUs (not fixed here): the grid must be a 3-tuple
(an int, a 1- or 2-tuple or a callable raises); every launch uses 256
threads whatever `num_warps` compiled (`num_warps=8`, `BLOCK=1024`: 512 of
1024 outputs written); compile options are not in the HSACO key, so a later
`num_warps=4` call launches the 8-warp HSACO (same 512 of 1024). A fourth
is read in the code, not run on two GPUs: `HsacoLauncher` always loads the
module on device 0 (`hipSetDevice(0)`).

**Host time**, per call, warmed, 8-call batches with a synchronize after
each outside the timer, median of 3000 batches per process (the cases take
turns in chunks of 250), 9 processes per column alternating, pinned to one
core on the GPU's NUMA node (gfx942: MI300X GPU 5, core 100; gfx950:
MI350X GPU 5, core 300). Median [min-max] over the processes, us. "Triton"
is upstream's `HsacoKernel` (`874d253dd3`); "HSACO" is upstream with
`AITER_USE_HSACO=1`, with the stand-in above and `aiter` imported (INFO
logging); its lower bound stubs the hip launch out and sets
`AITER_LOG_LEVEL=WARNING`. The raw rows time the launch under the wrapper:
the JIT function's own `k[grid](...)`, and the intj launcher `HsacoKernel`
holds, called with the current device and stream.

| gfx950 | Triton | HSACO | HSACO lower bound | intj |
| --- | ---: | ---: | ---: | ---: |
| plain, `jit(k)[(1, 1, 1)](x, y, out, n, 256)` | 10.76 [10.71-10.90] | 55.1 [53.9-55.8] | 25.8 [25.4-26.7] | 2.90 [2.89-2.91] |
| two constexpr keywords, one default, `num_warps`, `waves_per_eu` | 10.39 [10.36-10.49] | 60.8 [59.7-61.8] | 31.0 [30.9-32.1] | 3.31 [3.30-3.33] |
| callable grid | 11.55 [11.50-11.73] | raises | raises | 4.16 [4.06-4.20] |
| raw launch, plain | 10.58 [10.48-10.68] | | | 2.47 [2.46-2.48] |
| raw launch, options | 10.15 [10.09-10.18] | | | 2.49 [2.49-2.50] |

| gfx942 | Triton | HSACO | HSACO lower bound | intj |
| --- | ---: | ---: | ---: | ---: |
| plain | 17.17 [16.91-17.74] | 101.6 [101.0-103.0] | 51.9 [51.1-52.5] | 4.44 [4.41-5.61] |
| keywords and options | 17.00 [16.75-17.22] | 109.9 [108.9-111.2] | 60.7 [59.4-61.5] | 5.17 [5.05-6.17] |
| callable grid | 18.61 [18.41-19.38] | raises | raises | 6.41 [6.33-7.62] |
| raw launch, plain | 16.82 [16.59-17.71] | | | 3.77 [3.72-5.24] |
| raw launch, options | 16.49 [16.04-16.80] | | | 3.74 [3.71-5.25] |

Against the Triton path, intj saves 7.1-7.9 us per call on gfx950 (64-73%)
and 11.8-12.7 us on gfx942 (66-74%). The wrapper adds 0.43 us (plain),
0.82 us (keywords and options) and 1.69 us (callable grid) to the raw
launch on gfx950. Two of the nine gfx942 intj processes ran 1.1-1.5 us
slower in every row, raw launch included (the bimodal per-process launch
time noted above); the gfx942 machine was shared with other jobs. The
kernels are the ones Triton compiles (the tests check the AMDGCN is
identical), so GPU time was not measured. Making a stream's device current
(above) changed the closures; re-measured the same way against the code
before it (7 alternating rounds), intj moved by at most 0.02 us on gfx950
and 0.08 us on gfx942, where the unchanged raw rows moved up to 0.05 us.

**torch.compile.** Under `torch.compile(fullgraph=True)` a call fails with
`Unsupported`, where upstream's `HsacoKernel` compiles and matches eager
(gfx942): on a cold kernel at the lock in `_IntjKernel.of` ("Unsupported
context manager"), after an eager warm-up at the launcher ("Unsupported
call to a callable object with no traceable `__call__`"). With
`fullgraph=False` it graph-breaks and matches eager. Not fixed: the
`torch.compiler.is_compiling()` branch of `launch_on_current_stream`, at
the top of `__getitem__`, cost 0.08-0.11 us per call on gfx950 and
0.12-0.20 us on gfx942 (same process, 9 rounds of 1000 batches), on every
call of a helper nothing compiles; the tree adds that branch only where a
caller needs `fullgraph=True` (fav3_sage).

**Tests.** `op_tests/cpp/test_hsaco_kernel.py`: positional, keyword and
defaulted arguments, option keywords (compiled options and AMDGCN equal to
Triton's), every grid form, the grid function's arguments against
Triton's, a zero grid, a lone argument by keyword, the stream (current,
`with torch.cuda.stream`, a `torch.cuda.Stream`, one on another device for
a kernel-cache miss and a hit, a raw handle, a bad one), launchers shared
across wrappers, and the refusals, with no launch reaching
`JITFunction.run`; plus outputs equal to a Triton launch and, where `hip`
imports, to the HSACO path. gfx942 on two GPUs: 22 passed, 1 skipped (no
hip-python), 23 passed with the stand-in on `PYTHONPATH`. gfx950 (MI350X),
one GPU: 20 passed, 3 skipped (also the two-GPU test), 21 passed and 2
skipped with the stand-in. On gfx942, each of seven injected mistakes
fails at least one test: ignoring `stream`, unsorted option names, the
grid's `meta` built from the wrong arguments, a wrong default, no device
switch for another device's stream, launching it with the current device,
and `itemgetter` for a one-value plan.

Not verified: real hip-python, and two GPUs on gfx950.

## The metadata-redirect separate-compile test

`test_separate_compile_and_run` compiled `empty_kernel` with
`triton.compile(ASTSource(signature={"x_ptr": "*fp32"}, constexprs={"SIZE": 64}),
target=GPUTarget("hip", "gfx942", 64))` inside `AOTMetadataContext`, then
launched the returned `CompiledKernel` with `kernel[(1, 1, 1)](x)`. It now
declares a `return_compiled=True` launcher whose `extra_annotation` states
that source: `x_ptr` an `*fp32` `Argument` with `specialize=NEVER`, `SIZE` a
baked `Constexpr(value=64)`.

- **Compile.** The first call, with a zero-volume grid, runs inside the
  context. It misses intj's cache, so `triton.compiler.compile` runs on this
  thread and the context's patched `CompiledKernel.__init__` swaps in the
  prebuilt hsaco/json. The call returns that kernel without running it.
- **Run.** A `(1, 1, 1)` call outside the context finds the kernel, launches
  it and must return the same object.
- **Checks.** The json's hash (as before), the hsaco path from this test's
  directory, and the prebuilt bytes as the loaded binary. Each bypass fails
  the test: compiling outside the context, launching once before it, or
  redirecting to another directory holding the same files (caught by the
  path check only). In a full-file run the first two fail with
  `FileNotFoundError` instead: `test_jit`'s `with_custom_metadata_path`
  leaves `empty_kernel` registered on the thread, pointing at `test_jit`'s
  deleted directory (on the parent too).

What the launcher cannot reproduce: compiling needs the launch's arguments
(device, stream, `x`) and loads the binary right away rather than at the
first run, and the target is the current device's rather than an explicit
`GPUTarget` (the file runs only on gfx942, where they are equal). The miss
compiles with the same options, target, argument types, constexpr and (no)
attributes as `triton.compile` did (checked by intercepting it), but its
`ASTSource` signature also lists `SIZE` as `constexpr`, as Triton's JIT
does, so Triton caches the result under a different key. The redirect
replaces that result either way.

Verified on gfx942 (MI300X, `HIP_VISIBLE_DEVICES=5`, intj `2783079`, Triton
3.8.0): the file passes in full, 3 of 3, on this change and on its parent;
4 of 4 on the integrated tree, which adds pa_mqa_logits'
`test_required_redirect_only_on_miss`.

Host time per launch of the empty kernel, grid `(1, 1, 1)`: batches of 8
calls with a synchronize after each batch outside the timer, median of 3000
batches per process, seven alternating parent/intj rounds in separate
processes pinned to the GPU's NUMA node; median of the round medians
(range).

| Launch | us/call |
| --- | ---: |
| `kernel[(1, 1, 1)](x)` (parent) | 7.33 (7.24-7.42) |
| `runner = kernel[(1, 1, 1)]` hoisted, `runner(x)` | 6.89 (6.80-7.12) |
| `launch(dev, stream, (1, 1, 1), x)` | 3.11 (3.02-3.16) |
| `launch(*current_device_stream(), (1, 1, 1), x)` | 3.73 (3.59-3.83) |

GPU time is unchanged: 2.51-2.53 us per launch on both sides, the same
prebuilt binary (`torch.cuda.Event` around 200 launches queued behind
`torch.cuda._sleep`, three runs each; rocprofv3 aborts against torch's
bundled rocprofiler).

The first call costs more. Compile through the redirect plus the first run,
in a fresh process with the CUDA context, Triton's driver and Triton's
`triton_key()` hash already done (0.50 s per process; parent 502 ms and
intj 568 ms on warm caches with it, 572 ms and 2499 ms cold), median ms
(range) of seven alternating rounds warm, five for the other two rows:

| Disk caches | parent | intj |
| --- | ---: | ---: |
| Warm Triton and intj caches | 1.9 (1.9-2.2) | 63.3 (62.9-64.6) |
| Warm, `torch.utils.cpp_extension` already imported | 2.2 (2.0-2.3) | 8.2 (8.0-9.1) |
| Cold (empty `TRITON_HOME`) | 62.0 (61.3-63.7) | 1982 (1966-2027) |

intj's warm overhead is mostly its first import of `torch.utils.cpp_extension`
(once per process, when the first launcher is checked); cold, it builds the
launcher's extension (1.74 s in the host C compiler and 0.26 s rendering, in
one profiled run). The parent's first run also loads the binary (0.7-1.1
ms); intj loads it while compiling, so its first run takes under 0.12 ms.

## The named launchers as decorators

Four named-launcher sites were left at `00df7c810`: the three of the 13
named launchers whose JIT function had to keep its name (see
[intj 2783079](#intj-2783079)) and pa_attention_kernel_test's two direct
launchers, module globals since its factory went. Two became decorators; the
other two launch a decorated kernel's JIT function with other options, and
stay for a measured cost. Base `ba40fc7e2` (none of these files changed up
to `00df7c810`); gfx942 (triton 3.8.0, torch 2.14+rocm7.2, intj
`2783079`), gfx950 compiled offline.

| Launcher (before) | Now | Why |
| --- | --- | --- |
| `aiter/mla.py` `_fwd_kernel_stage2_asm_launch` | `@make_launcher(grid_arg=2, options=...)` on `_fwd_kernel_stage2_asm` | The AOT tools (`aiter/aot/triton/decode_mla.py`, `csrc/cpp_itfs/mla/asm_mla_decode_fwd.py`) point `triton.tools.compile` at `aiter/aot/triton/mla_jit.py`, which binds `_fwd_kernel_stage2_asm` to the launcher's `__self__.options.jit_func`. `compile_kernel` takes `kernel_name` from that file and its loader looks the HSACO symbol up by the same name, so the name stays `_fwd_kernel_stage2_asm`. |
| `rmsnorm` `_rms_norm_kernel_launch` | `@make_launcher` on `_rms_norm_kernel` | The Iris kernel `fused_pipeline_kernel` calls `_rms_norm = _rms_norm_kernel.__self__.options.jit_func` as its device function. The kernel stays one JIT function, so both kernels compile the same code. |
| `test_mla_stage2_merge.py` `_fwd_kernel_stage2_asm_launch` | still `make_launcher(_fwd_kernel_stage2_asm.__self__.options.jit_func, dynamic_options=(num_warps, num_stages, waves_per_eu))` | See [Launchers](#launchers) and the measurements below. |
| `pa_attention_kernel_test` `_direct_dot_launch`, `_direct_large_block_dot_launch` | unchanged (over pa_decode_gluon's `jit_func`), reason now in the comment | Making `num_stages` and `waves_per_eu` dynamic on pa_decode_gluon's dot-kernel decorator added 0.05-0.11 us to each production launch (below). |

After these two conversions (`210897263`; the current counts, 426, 5 and
11, are under
[attn_res and pa_mqa_logits as decorators](#attn_res-and-pa_mqa_logits-as-decorators)),
`grep -rE '^\s*@(intj\.)?make_launcher\b' --include=*.py aiter op_tests csrc`
counts 422 decorator lines (420 at `00df7c810`). The other
`make_launcher(` calls (`grep -rn 'make_launcher(' --include=*.py aiter
op_tests csrc` minus the decorator lines; `kda_launchers.py` only re-runs
decorated kernel modules under a stand-in) went from 8 to 6, declaring 15
launchers (17 before) plus `HsacoKernel`'s, made at run time. Already
decorators but inside a function, in `test_metadata_redirect.py`, each over
a prebuilt binary in the test's own temp dir: `test_jit`'s `empty_kernel`
(under `with_custom_metadata_path(temp_dir)`),
`test_separate_compile_and_run`'s `empty_kernel` and
`test_required_redirect_only_on_miss`'s `required_redirect_kernel`.

**Codegen and AOT identity.** A decorator line above a kernel moves its
`def`, and Triton records source lines in the TTGIR locations and the
HSACO's DWARF. So "identical" was checked two ways: old vs new with line
numbers set aside, and new against a control tree (`ba40fc7e2` plus an
identity decorator `@(lambda f: f)` and comment lines at the same lines as
the new decorator), which must be byte-identical. All runs compile from one
fixed path (the file path is in the debug info), each with a fresh
`TRITON_CACHE_DIR`.

- **mla AOT** (both tools, gfx942 native and gfx950 offline; the C++ build
  of `asm_mla_decode_fwd` stubbed, everything before it real). With the
  tools' own signatures (15 and 17 entries for a 24-parameter kernel),
  `ba40fc7e2` and new fail identically: `CompilationError ... 'NoneType'
  object has no attribute '_unflatten_ir'`; these signatures predate the
  current kernel, not fixed here. With a valid 24-entry signature (the
  tools' last three constexprs kept), new vs control is byte-identical:
  generated `.c`/`.h`, HSACO, amdgcn, ttgir, metadata json including
  `hash`. New vs `ba40fc7e2`: `.h` identical; `.c` differs only in the
  embedded HSACO; HSACO identical after `llvm-objcopy --strip-debug`;
  disassembly identical; amdgcn differs only in `.loc` and one
  `DW_AT_call_line` (+5 lines); ttgir only in `mla.py` line numbers; json
  only in `hash` (Triton's `cache_key` includes the first decorator's
  line). Symbol `_fwd_kernel_stage2_asm`, func names
  `decode_mla_stage2_asm_f0fd6ad4_...`/`_7932f740_...`, and the loader's
  `hipModuleGetFunction(..., "_fwd_kernel_stage2_asm")` are unchanged.
- **rmsnorm** (`_rmsnorm_forward` on 6 shapes, which compile 6
  configurations, blocked and not, `BLOCK_SIZE` 1024-32768; the same
  annotated sources compiled for gfx950): new vs control byte-identical on
  both targets (ttgir, amdgcn, HSACO, json). New vs `ba40fc7e2`: identical
  except line numbers (+1); the json, `hash` included, is identical:
  Triton's cache key takes the first decorator's line, 56 in both trees
  (`@triton.jit(...)` before, `@make_launcher` now).
- **Iris `fused_pipeline_kernel`**: `iris` is not installed, so it was
  compiled with a stub `iris` package (`load`/`put` as plain loads/stores)
  for 3 configurations on gfx942 and gfx950. Its ttgir, amdgcn and stripped
  HSACO equal `ba40fc7e2`'s except for line numbers. Its `cache_key`
  differs (`da620d7bb68f` before, `d1720a2b3ae7` now): its source now calls
  `_rms_norm`.

**Measurements** (gfx942, host time per launch, rotated order, medians):

- `aiter.mla` stage-2, production shape: old launcher 5016 ns, new
  decorated launcher 4995 ns. Same JIT function with `num_warps`,
  `num_stages`, `waves_per_eu` dynamic and `grid_arg=2`: +334 to +403 ns
  (20-call batches), +380 to +470 ns (single calls); rerun +389, +391, +411
  ns. `num_stages` and `waves_per_eu` dynamic, or `num_warps` alone: within
  5 ns of baked. The compiled kernel (HSACO) is the same either way.
- The cause is the argument count, not the options: three options make the
  call 31 positional arguments (device, stream, 2 grid values, 3 options,
  24 kernel arguments), and CPython compiles a call with more than 30 to
  `CALL_FUNCTION_EX` (`dis` shows it). A 27-argument test kernel costs 4123
  ns with no dynamic option (30 positional) and 4514-4566 ns with 1 to 3
  (31-33); a 23-argument one 4031-4069 ns with 0 to 3 (26-29). `math.hypot`
  shows the same switch from 30 to 31 arguments. With the grid as one tuple
  (30 positional) the three options cost +22, +31, +46 ns over the current
  launcher (3 runs; a review measured +30 to +43 ns).
- pa_decode_gluon dot kernel, arguments captured from
  `pa_attention_kernel_test`: `dynamic_options=("num_stages",
  "waves_per_eu")` over the default-option launcher: +105, +50, +81 ns;
  rerun +90, +73, +62, +109, +57 ns (one more run, in which every launcher
  was 2 us slower, gave +377 ns). Its 54 parameters put every call past 30
  positional arguments already, so this is intj's own cost. Only this
  kernel was timed; the large-block kernel has the same 54 parameters.

**Tests** (gfx942, `ba40fc7e2` vs new):

| | `ba40fc7e2` | new |
| --- | --- | --- |
| check sweep: launchers / check ok / build ok | 419 / 419 / 419 | 419 / 419 / 419 |
| GPU-free imports (414 modules) | 148, same failure set | 148, same failure set |
| `triton_tests/normalization/test_rmsnorm.py` | 450 passed, 2 failed | 450 passed, 2 failed |
| `triton_tests/normalization/test_intj_launches.py` | 1 passed | 1 passed |
| `test_mla_decode_gate.py`, `test_mla_v4_nm.py` | 21 skipped (gfx950 only) | 21 skipped |
| `test_mla_stage2_merge.py` (32 configs) | err 0 everywhere | err 0 everywhere (rerun, 36 configs: the same) |
| `triton_tests/attention/test_intj_attention_launches.py` | not present | 4 passed |
| production stage-2 call, 3 shapes | output MD5s | the same MD5s |
| `pa_attention_kernel_test` direct | passed, MD5s | passed, same MD5s |

The 2 rmsnorm failures are the HIP reference `aiter.rmsnorm2d_fwd_with_add`,
absent under `AITER_TRITON_ONLY`. `test_mla.py` fails on both trees before
reaching `mla_decode_fwd` (`aiter.flash_attn_varlen_func`, HIP).
`test_mla_stage2_merge.py` is a script (under pytest it fails on both trees:
`test_stage2_merge() missing 5 required positional arguments`); its markdown
summary needs `tabulate` (not installed), so it was printed with
`to_string`.

`test_mla_stage2_merge_launcher_matches_torch` calls the decorated launcher
as `mla_decode_fwd_v4_nm` does, on three sequences that merge N, N - 1 and 1
splits, against torch; `test_mla_aot_tools_load_the_stage2_jit_function`
checks that `mla_jit.py` yields the JIT function under the kernel's name.
Before them no pytest test that runs on gfx942 under `AITER_TRITON_ONLY`
reached this launcher. Each of these mutations fails them: the decorator
dropped, `grid_arg=1`, `mla_jit.py` binding the launcher, the kernel
ignoring `valid_split_count`, its `stride_mid_oh`/`stride_mid_os`
parameters swapped. They cannot catch a wrong option or argument order
inside `mla_decode_fwd` / `mla_decode_fwd_v4_nm`; that needs
`test_mla_v4_nm.py` (gfx950) or `test_mla.py` (HIP ops).

Not verified: the mla AOT C++ build (`compile_template_op` needs CK
headers) and running the generated loader; gfx1250 (the stage-1 asm target)
not compiled; `mla_decode_fwd` end to end (its stage 1 is HIP asm; the
stage-2 launch was run directly with production arguments instead); the
Iris kernel against the real `iris`, and multi-GPU; gfx950 at runtime
(compiled offline only); the dynamic-options cost on pa_decode_gluon's
large-block kernel.

## attn_res and pa_mqa_logits as decorators

The last launchers declared beside their kernel are decorators now; what is
left over a kernel's JIT function is listed under [Launchers](#launchers).
Measured on gfx942 (MI300X, triton 3.8.0, torch 2.14+rocm7.2, intj
`2783079`) against `00df7c810` (before the named-launcher commits were
cherry-picked); "new" is `73e982ef0`, which after review moved the
preshuffle kernels' decorators to the KV-padded mode (first `4bd3633f3`,
then `8d3fbaf06` with this record, decorated the unpadded mode of all three).

| Launcher (before) | Now |
| --- | --- |
| `attn_res` `attnres_fwd_kernel = make_launcher(_attnres_fwd, ...)`, after an optional `triton.autotune` rebind | `@make_launcher(...)` over `@_attn_res_tuning` (the `ATTN_RES_TRITON_AUTOTUNE` tuner, or nothing) over `@triton.jit`; the five `attnres_fwd_sequence_kernels` are made over its `__self__.options.jit_func` |
| `pa_mqa_logits` `_GLUON_PAGED_MQA_LOGITS_LAUNCHERS`, six launchers from one `make_launcher` comprehension over `_gluon_kernel(mode)` | each Gluon kernel's decorator, `@make_launcher(..., delayed=partial(_gluon_launch_keywords, ...))`, is the launcher of the KV mode 132-byte tokens take in its default page: unpadded for the plain kernel (a page of one token), KV-padded for the two preshuffle kernels (every page they take is padded with such tokens). It sits in `aiter/ops/triton/gluon/pa_mqa_logits.py` with the keywords function; `@_jit_function_of(Preshuffle, VarCtxOpt)` under it records the JIT function, over which the table makes each kernel's other-mode launcher; the six-entry table, `require_metadata_redirect` and `AOTMetadataContext` take the recorded JIT functions (names unchanged) |
| `aiter/mla.py` `_fwd_kernel_stage2_asm_launch`, `rmsnorm.py` `_rms_norm_kernel_launch` | gone (see [The named launchers as decorators](#the-named-launchers-as-decorators)) |

Every call site keeps its arguments, their count and order. 426 decorator
lines; the other `make_launcher(` calls are 5: four declare 11 launchers,
and `HsacoKernel`'s makes its own at run time.

**Same binaries.** A decorator above a kernel moves source lines, and
Triton keys a JIT function on the line of its first decorator and those of
the JIT functions it calls, and records lines in the TTGIR locations and the
HSACO's DWARF. Each kernel's compiles, run from both trees with a fresh
`TRITON_CACHE_DIR`, were compared as multisets: the metadata json without
`hash`, the TTGIR without locations, the AMDGCN without `.loc`/`.file` and
debug sections, and the HSACO after `llvm-objcopy --strip-debug`.

| Kernel | Compiles compared | Equal | Triton hash |
| --- | --- | --- | --- |
| `attnres_fwd_kernel` | `test_attn_res.py`: 441, and 436 with `ATTN_RES_TRITON_AUTOTUNE=1` | all | differs |
| the three Gluon pa_mqa_logits kernels | `test_pa_mqa_logits_offset.py` (JIT mode): 21 plain, 28 preshuffle; the AOT driver below: 2, 3, 2 | all | differs |
| `_fwd_kernel_stage2_asm` | 2 (4 launches as `mla_decode_fwd_v4_nm` makes them, outputs bitwise equal); `test_mla_stage2_merge.py`'s 50 configurations: 2 | all | differs |
| `_rms_norm_kernel` | `test_rmsnorm.py`: 36 | all | equal (first decorator still on line 56) |

The pa_mqa_logits hash difference is the line numbers alone: with every
JIT function of the Gluon module set back to its starting line in
`00df7c810` (the three kernels, and the helpers above them, 4 lines down
after the new imports), the 7 configurations below compile to
`00df7c810`'s Triton cache keys.

Moving the preshuffle decorators to the padded mode moves no kernel's lines
and changes no mode's keywords. Against `8d3fbaf06` (unpadded
decorators), the offset test's 49 compiles and the driver's 7 have the same
Triton hashes, and equal json, TTGIR, AMDGCN and stripped HSACO. The
driver's, compiled from one path, are byte-identical; the offset test's
differ only in the tree's path in the DWARF (pytest resolves the symlinked
path, so the two trees' paths differ there). Outputs, the launcher each call
took and the returned cache keys equal `8d3fbaf06`'s in JIT and AOT mode.
The other kernels' rows were compiled at `8d3fbaf06`; `73e982ef0` changes
none of their files.

**pa_mqa_logits' AOT path.** A driver runs `deepgemm_fp8_paged_mqa_logits`
on 7 configurations that reach all six launchers (batch 1, 64 heads,
128-dim, 256-token context, ChunkK 256, WavePerEU 2): plain page 1 with a
132- and a 144-byte KV row (unpadded, padded), preshuffle and
preshuffle-varctx page 16 (padded), the same at page 8 with a 133-byte row
(unpadded: the standard 132-byte row makes every preshuffle page padded),
and the offset test's preshuffle page 256 with 128 heads. Outputs are
bitwise equal between the trees, in both modes. AOT artifacts made from
`00df7c810`'s JIT compiles, copied to `configs/paged_mqa_logits/aot/<name>`
as `bench_deepgemm_attention.py -aot` does, load on the new tree: for every
configuration the first call (a miss) and the second (a hit) return a
kernel whose json and hsaco are the artifact's (its json `hash` is
`00df7c810`'s), and the outputs equal JIT mode's. Same on `00df7c810`
and `8d3fbaf06`.

**Check sweep** (`intj-aiter-check-sweep.py --build`, both trees): 415
modules, 383 import on one GPU; 424 launchers (7 `delayed=`), every one
checks and builds; the same 14 declared launchers not reached. Without a
GPU, 148 of 415 import on each tree and the same 267 fail (`00df7c810`'s
count from a GPU-free sweep of its own; the new tree's sweep re-imports on
the base only the modules that fail on the new one).

**Tests** (gfx942; `aiter_triton_only_dtypes`, `aiter_pertoken_shim`,
`aiter_quant_shim` plugins; failures by test id). The pa_mqa_logits, metadata
redirect and `test_intj_attention_launches.py` rows ran on `73e982ef0`; the
others on `8d3fbaf06`, whose code they share (the later commits change only
the Gluon pa_mqa_logits module and that test file):

| | `00df7c810` | new |
| --- | --- | --- |
| `triton_tests/fusions/test_attn_res.py` | 272 passed | 272 passed |
| the same, `ATTN_RES_TRITON_AUTOTUNE=1` | 271 passed, 1 failed | the same |
| `triton_tests/fusions/test_mhc.py` | 406 passed, 12 skipped | the same |
| `test_pa_mqa_logits_offset.py`, JIT / AOT mode | 105 passed, 1 skipped / 3 passed, 103 skipped | the same |
| `triton_tests/triton_metadata_redirect/test_metadata_redirect.py`, JIT / AOT mode | 4 / 4 passed | the same |
| `triton_tests/normalization/test_rmsnorm.py` | 450 passed, 2 failed | the same |
| `triton_tests/normalization/test_intj_launches.py` | 1 passed | 1 passed |
| `triton_tests/attention/test_intj_attention_launches.py`, JIT / AOT mode | 1 passed / not run | 11 passed (adds the stage-2 and pa_mqa_logits tests) / 5 passed, 6 skipped |
| `test_mla_decode_gate.py`, `test_mla_v4_nm.py` | 21 skipped (gfx950 only) | the same |
| `test_mla_stage2_merge.py` (a script), 50 configurations | err 0 | err 0 |

`test_pa_mqa_logits_launchers_carry_their_modes_keywords` checks, after
`check()`, that each of the six table entries launches the JIT function
recorded for its (Preshuffle, VarCtxOpt) with `_gluon_launch_keywords`'s
annotation and options for its mode, and that the six modes' keywords differ.
`test_pa_mqa_logits_launches_each_mode_through_its_launcher` runs each mode
through `deepgemm_fp8_paged_mqa_logits` (2 sequences of 2 tokens, 64 heads;
132-, 144- and 133-byte tokens), checks which launcher ran and compares with
torch. Before them the offset tests reached three modes (plain unpadded and
padded, preshuffle padded) and no pytest test the VarCtx kernel. Five mutations
fail the first test: a decorator's `delayed=` with another `VarCtxOpt`, the
plain decorator padded, a JIT function recorded under another mode, a table
key with the other KV mode, the table's launchers with their decorator's KV
mode. The last three pass every launch test: with the wrong facts the logits
are still right on gfx942. The launch test's output width is a multiple of
16: the VarCtx kernel stores logits up to a sequence's length rounded up to
16, so with a `max_model_len` below that it writes past the row, into the
next row's first logits (300 tokens and `max_model_len` 300: 4 stores past
the row; 290 and 290: 14; 300 and 304: none; upstream `874d253dd3` stores
the same 4 at 300 tokens; not fixed here).

The autotune failure is `test_attn_res_gate_out_quant_matches_unfused_quant[64]`;
the rmsnorm ones are `test_fused_add_rmsnorm_gemma_norm[bf16|fp16]`, whose
HIP reference `aiter.rmsnorm2d_fwd_with_add` `AITER_TRITON_ONLY` leaves
unbound.

**Host time** per launch (us), new vs `00df7c810`, every production
launcher whose declaration changed: the decorated ones, all five sequence
launchers and the six pa_mqa_logits ones (`test_mla_stage2_merge.py`'s
script launcher, now made over the decorator's `jit_func`, was not timed).
Each case calls the launcher alone with the arguments its production wrapper
passed (captured from a real call; a site that passes them positionally is
replayed positionally, a starred one starred), in batches of 8 calls with a
synchronize after each batch outside the timer, median of 3000 batches; each
batch is followed by one of an unchanged reference launcher (pa_mqa_logits'
VarCtx schedule kernel). One process per tree and round with all 14 cases,
the trees' order alternating, two runs of 60 rounds, pinned to a core of the
GPU's NUMA node with memory bound to it. Process times are bimodal on both
trees: in 37 of 120 `00df7c810` processes and 38 of 120 new ones every
launch is slower, the reference's too (4.0-5.2 us against 3.3-3.5 us). The
table is over the others (reference under 4 us; 83 and 82 processes): the
median of the process medians, the difference, and its bootstrap 95%
interval.

| Launcher (case) | `00df7c810` | new | new - `00df7c810` [95% CI] | the same, each batch minus the reference's |
| --- | ---: | ---: | ---: | ---: |
| `_fwd_kernel_stage2_asm` (28 positional) | 3.551 | 3.551 | -0.001 [-0.021, +0.016] | -0.005 [-0.021, +0.009] |
| `_rms_norm_kernel` (15 positional) | 3.222 | 3.218 | -0.004 [-0.020, +0.021] | +0.001 [-0.027, +0.016] |
| `attnres_fwd_kernel`, `attn_res_gate` (starred) | 3.674 | 3.670 | -0.004 [-0.020, +0.027] | -0.000 [-0.017, +0.014] |
| `attnres_fwd_sequence_kernels[1]`, L=1 | 3.344 | 3.342 | -0.002 [-0.022, +0.026] | +0.006 [-0.015, +0.021] |
| `attnres_fwd_sequence_kernels[2]`, L=2 | 3.442 | 3.463 | +0.021 [-0.014, +0.042] | +0.000 [-0.017, +0.020] |
| `attnres_fwd_sequence_kernels[4]`, L=4 | 3.479 | 3.483 | +0.004 [-0.005, +0.020] | +0.010 [-0.023, +0.029] |
| `attnres_fwd_sequence_kernels[8]`, L=8 | 3.833 | 3.822 | -0.011 [-0.023, +0.013] | +0.001 [-0.017, +0.018] |
| `attnres_fwd_sequence_kernels[16]`, L=16 | 4.240 | 4.236 | -0.004 [-0.022, +0.019] | +0.006 [-0.004, +0.017] |
| pa_mqa_logits plain, unpadded (decorator) | 3.454 | 3.458 | +0.004 [-0.019, +0.015] | +0.001 [-0.019, +0.017] |
| pa_mqa_logits plain, padded | 3.482 | 3.493 | +0.010 [-0.015, +0.025] | -0.003 [-0.025, +0.015] |
| pa_mqa_logits preshuffle, padded (decorator) | 3.499 | 3.491 | -0.008 [-0.029, +0.017] | -0.000 [-0.026, +0.023] |
| pa_mqa_logits preshuffle-varctx, padded (decorator) | 3.547 | 3.536 | -0.011 [-0.028, +0.011] | -0.013 [-0.032, +0.013] |
| pa_mqa_logits preshuffle, unpadded | 3.445 | 3.447 | +0.003 [-0.022, +0.021] | -0.003 [-0.018, +0.017] |
| pa_mqa_logits preshuffle-varctx, unpadded | 3.534 | 3.538 | +0.005 [-0.014, +0.023] | -0.023 [-0.038, +0.022] |

No difference is resolved: every interval includes zero, raw and against the
reference. The point estimates are -0.011 to +0.021 us raw and -0.023
to +0.010 us against the reference, and the intervals are 0.02-0.06 us wide, so
a change of a few hundredths of a microsecond is not excluded (up to +0.042
us on L=2's raw interval). A single run is noisier: in the first, new-tree
processes ran the unchanged reference 0.017 us slower, and two raw
intervals excluded zero on the slower side (`attn_res_gate` +0.034, plain
padded +0.026; -0.014 and -0.012 in the second run).

Not verified: gfx950 (runtime and compile), CUDA; the Iris
`fused_pipeline_kernel` (no `iris`; see the previous section's stub
compile); real AOT artifacts other than these, and concurrent misses in AOT
mode. The reason pa_mqa_logits keeps a table (one launcher measuring the KV
facts would change the unpadded binaries; see [Launchers](#launchers)) is
reasoned: no such launcher was built, compared or timed.
