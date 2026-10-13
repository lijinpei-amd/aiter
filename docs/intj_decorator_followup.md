# intj decorators: the named launchers (follow-up)

To be merged into `intj_launch_exceptions.md` ("Launchers") once the
concurrent changes to that file land. Base: `ba40fc7e2`; measured on gfx942
(triton 3.8.0, torch 2.14+rocm7.2, intj `2783079`), gfx950 compiled offline.

## Outcome per launcher

| Launcher (before) | Now | Why |
| --- | --- | --- |
| `aiter/mla.py` `_fwd_kernel_stage2_asm_launch` | `@make_launcher(grid_arg=2, options=...)` on `_fwd_kernel_stage2_asm` | The AOT tools (`aiter/aot/triton/decode_mla.py`, `csrc/cpp_itfs/mla/asm_mla_decode_fwd.py`) point `triton.tools.compile` at `aiter/aot/triton/mla_jit.py`, which binds `_fwd_kernel_stage2_asm` to the launcher's `__self__.options.jit_func`. `compile_kernel` takes `kernel_name` from that file and its loader looks the HSACO symbol up by the same name, so the name stays `_fwd_kernel_stage2_asm`. |
| `test_mla_stage2_merge.py` `_fwd_kernel_stage2_asm_launch` | still `make_launcher(_fwd_kernel_stage2_asm.__self__.options.jit_func, dynamic_options=(num_warps, num_stages, waves_per_eu))` | The script sweeps the three options the production decorator bakes. Taking them per call there makes each production call 31 positional arguments, past the 30 CPython passes directly: +0.33-0.47 us per launch (below). With the grid as one tuple the call stays at 30 and costs +22-46 ns: under the 0.05 us limit, but production would pay it for this script's sweep, and one more kernel parameter would cross the step again. Kept. |
| `rmsnorm` `_rms_norm_kernel_launch` | `@make_launcher` on `_rms_norm_kernel` | The Iris kernel `fused_pipeline_kernel` calls `_rms_norm = _rms_norm_kernel.__self__.options.jit_func` as its device function. No body split: the kernel stays one JIT function, so both kernels compile the same code. |
| `pa_attention_kernel_test` `_direct_dot_launch`, `_direct_large_block_dot_launch` | unchanged (built over pa_decode_gluon's `jit_func`), reason now in the comment | Making `num_stages` and `waves_per_eu` dynamic on pa_decode_gluon's dot-kernel decorator added 0.05-0.11 us to each production launch (below; the large-block kernel was not timed). Rejected. |

Not converted, in the excluded files (plan below): `attn_res`
`attnres_fwd_kernel` (beside the kernel), `pa_mqa_logits`
`_GLUON_PAGED_MQA_LOGITS_LAUNCHERS` (6, a dict comprehension). Already a
decorator but inside a function: `test_metadata_redirect.py` `test_jit`'s
`empty_kernel`, under `with_custom_metadata_path(temp_dir)` of the test's own
temp dir, so it stays there. `kda_launchers.py` only re-runs decorated
kernel modules with a stand-in `make_launcher`; it declares none.

No `functools.cache` / `lru_cache` factory or dict cache of intj launchers is
left (`_LAUNCHER_CACHE`, `_launch_fns` and `get_hsaco_launcher` are FlyDSL
and HSACO launchers).

## Forms count (aiter/, op_tests/, csrc/)

```
grep -rE '^\s*@(intj\.)?make_launcher\b' --include=*.py aiter op_tests csrc | wc -l
grep -rnE '(^|[^@.\w])make_launcher\(' --include=*.py aiter op_tests csrc   # minus comments, docstrings, kda_launchers
```

| | `ba40fc7e2` | now |
| --- | --- | --- |
| decorator lines | 418 | 420 |
| other `make_launcher(` call sites | 6 | 4 |
| launchers they declare | 12 | 10 (7 in excluded files) |

The four left: `attn_res.py:399` and `pa_mqa_logits.py:423` (excluded),
`test_mla_stage2_merge.py:173`, `pa_attention_kernel_test.py:484`.

## Codegen and AOT identity

Adding a decorator line above a kernel moves its `def`, and Triton records
source lines in the TTGIR locations and the HSACO's DWARF. So "identical"
was checked two ways: old vs new with line numbers set aside, and new
against a control tree (`ba40fc7e2` plus an identity decorator `@(lambda f:
f)` and comment lines at the same lines as the new decorator), which must be
byte-identical. All runs compile from one fixed path (the file path is in
the debug info), each with a fresh `TRITON_CACHE_DIR`.

**mla AOT** (both tools, gfx942 native and gfx950 offline; the C++ build of
`asm_mla_decode_fwd` stubbed, everything before it real):

- With the tools' own signatures (15 and 17 entries for a 24-parameter
  kernel), `ba40fc7e2` and new fail identically:
  `CompilationError ... 'NoneType' object has no attribute '_unflatten_ir'`.
  These signatures predate the current kernel; not fixed here.
- With a valid 24-entry signature (the tools' last three constexprs kept):
  new vs control is byte-identical: generated `.c`/`.h`, HSACO, amdgcn,
  ttgir, metadata json including `hash`. New vs `ba40fc7e2`: `.h` identical;
  `.c` differs only in the embedded HSACO; HSACO identical after
  `llvm-objcopy --strip-debug`; disassembly identical; amdgcn differs only in
  `.loc` and one `DW_AT_call_line` (+5 lines); ttgir only in `mla.py` line
  numbers; json only in `hash` (Triton's `cache_key` includes the first
  decorator's line). Symbol `_fwd_kernel_stage2_asm`, func names
  `decode_mla_stage2_asm_f0fd6ad4_...`/`_7932f740_...`, and the loader's
  `hipModuleGetFunction(..., "_fwd_kernel_stage2_asm")` are unchanged.

**rmsnorm** (`_rmsnorm_forward` on 6 shapes, which compile 6 configurations,
blocked and not, `BLOCK_SIZE` 1024-32768; the same annotated sources
compiled for gfx950): new vs control byte-identical on both targets
(ttgir, amdgcn, HSACO, json). New vs `ba40fc7e2`: identical except line
numbers (+1); the json, `hash` included, is identical. Triton's cache key
takes the first decorator's line, 56 in both trees (`@triton.jit(...)`
before, `@make_launcher` now), so `_rms_norm_kernel`'s `cache_key` is the
same in both trees (`1d150fa3ec66`).

**Iris `fused_pipeline_kernel`**: `iris` is not installed, so it was compiled
with a stub `iris` package (`load`/`put` as plain loads/stores) for 3
configurations on gfx942 and gfx950. It compiles in the new tree; ttgir,
amdgcn and stripped HSACO equal `ba40fc7e2`'s except for line numbers. Its
`cache_key` differs (`da620d7bb68f` before, `d1720a2b3ae7` now): its source
now calls `_rms_norm`.

## Measurements (gfx942, host time per launch, rotated order, medians)

- `aiter.mla` stage-2, production shape: old launcher 5016 ns, new decorated
  launcher 4995 ns. Same JIT function with `num_warps`, `num_stages`,
  `waves_per_eu` dynamic and `grid_arg=2`: +334 to +403 ns (20-call
  batches), +380 to +470 ns (single calls); rerun +389, +391, +411 ns.
  `num_stages` and `waves_per_eu` dynamic, or `num_warps` alone: within 5 ns
  of baked. The compiled kernel (HSACO) is the same either way.
- The cause is the argument count, not the options: three options make the
  call 31 positional arguments (device, stream, 2 grid values, 3 options, 24
  kernel arguments), and CPython compiles a call with more than 30 to
  `CALL_FUNCTION_EX` (`dis` shows it). A 27-argument test kernel costs 4123
  ns with no dynamic option (30 positional) and 4514-4566 ns with 1 to 3
  (31-33); a 23-argument one 4031-4069 ns with 0 to 3 (26-29). `math.hypot`
  shows the same switch from 30 to 31 arguments. With the grid as one tuple
  (30 positional) the three options cost +22, +31, +46 ns over the current
  launcher (3 runs; the review measured +30 to +43 ns).
- pa_decode_gluon dot kernel, args captured from
  `pa_attention_kernel_test`: `dynamic_options=("num_stages", "waves_per_eu")`
  over the default-option launcher: +105, +50, +81 ns; rerun +90, +73, +62,
  +109, +57 ns (one more run, in which every launcher was 2 us slower, gave
  +377 ns). Its 54 parameters put every call past 30 positional arguments
  already, so this is intj's own cost. Only this kernel was timed; the
  large-block kernel has the same 54 parameters.

## Tests (gfx942, `ba40fc7e2` vs new)

| | `ba40fc7e2` | new |
| --- | --- | --- |
| check sweep: launchers / check ok / build ok | 419 / 419 / 419 | 419 / 419 / 419 |
| GPU-free imports (414 modules) | 148, same failure set | 148, same failure set |
| `triton_tests/normalization/test_rmsnorm.py` | 450 passed, 2 failed | 450 passed, 2 failed |
| `triton_tests/normalization/test_intj_launches.py` | 1 passed | 1 passed |
| `test_mla_decode_gate.py`, `test_mla_v4_nm.py` | 21 skipped (gfx950 only) | 21 skipped |
| `test_mla_stage2_merge.py` (32 configs) | err 0 everywhere | err 0 everywhere (rerun at the end, 36 configs: the same) |
| `triton_tests/attention/test_intj_attention_launches.py` (new stage-2 tests) | not present | 4 passed |
| production stage-2 call, 3 shapes | output MD5s | the same MD5s |
| `pa_attention_kernel_test` direct | passed, MD5s | passed, same MD5s |

The 2 rmsnorm failures are the HIP reference `aiter.rmsnorm2d_fwd_with_add`,
absent under `AITER_TRITON_ONLY`. `test_mla.py` fails on both trees before
reaching `mla_decode_fwd` (`aiter.flash_attn_varlen_func`, HIP). The
markdown summary of `test_mla_stage2_merge.py` needs `tabulate` (not
installed); it was printed with `to_string`. Under pytest that script fails
on both trees (`test_stage2_merge() missing 5 required positional
arguments`): it is a script.

`test_mla_stage2_merge_launcher_matches_torch` calls the decorated launcher
as `mla_decode_fwd_v4_nm` does, on three sequences that merge N, N - 1 and 1
splits, against torch; `test_mla_aot_tools_load_the_stage2_jit_function`
checks that `mla_jit.py` yields the JIT function under the kernel's name.
Before them no pytest test that runs on gfx942 under `AITER_TRITON_ONLY`
reached this launcher (the review's mutation run). Each of these mutations
fails them: the decorator dropped, `grid_arg=1`, `mla_jit.py`
binding the launcher, the kernel ignoring `valid_split_count`, its
`stride_mid_oh`/`stride_mid_os` parameters swapped. They cannot catch a wrong
option or argument order inside `mla_decode_fwd` / `mla_decode_fwd_v4_nm`;
that needs `test_mla_v4_nm.py` (gfx950) or `test_mla.py` (HIP ops).

## Plan for the excluded files (after the concurrent work lands)

- **`attn_res` `attnres_fwd_kernel`**: move the tuner into a decorator
  stacked under the launcher, so the kernel is decorated in one place:

  ```python
  def _attn_res_tuning(kernel):  # the ATTN_RES_TRITON_AUTOTUNE layer, or none
      if not ATTN_RES_TRITON_AUTOTUNE:
          return kernel
      return triton.autotune(configs=..., key=[...], cache_results=True)(kernel)

  @make_launcher(dynamic_options=() if ATTN_RES_TRITON_AUTOTUNE else ("num_warps", "num_stages"))
  @_attn_res_tuning
  @triton.jit(repr=_attnres_fwd_kernel_repr, do_not_specialize=["L"])
  def attnres_fwd_kernel(...): ...
  ```

  The configs list moves above the kernel. Check: `test_attn_res.py` under
  both values of the knob, and the sequence layout's Triton launch through
  `jit_func`.
- **`pa_mqa_logits` (6 launchers)**: each JIT function needs two launchers
  that differ only in baked `Assume(Aligned(16))` facts (KV-padded or not),
  and a decorator declares one. Line numbers are `ba40fc7e2`'s
  `aiter/ops/triton/attention/pa_mqa_logits.py`.
  1. Move `_gluon_launch_keywords` into
     `aiter/ops/triton/gluon/pa_mqa_logits.py` and decorate each of the
     three Gluon kernels with its unpadded launcher (`delayed=partial(...,
     is_padded_mode=False)`, `return_compiled=True`,
     `assume_constant_globals=True`, `dynamic_options=("waves_per_eu",)`).
  2. Between `@make_launcher` and `@gluon.jit`, stack a decorator that
     records the JIT function in a module dict and returns it; build the
     3-entry padded-mode table from that dict. Not from
     `kernel.__self__.options.jit_func`: until `check()` a `delayed=`
     launcher's `options` is an `intj.launcher.Delayed`, with no `jit_func`
     (checked: `AttributeError`), and `check()` runs `get_gfx()`, which
     needs the GPU, at import.
  3. `:758` passes `_gluon_kernel(...).fn.__name__` to `AOTMetadataContext`;
     a launcher has no `.fn` (checked: `AttributeError`; its `__name__` is
     `launch`). Take the name from the recorded JIT function.
  4. On Triton < 3.5 (`:66-77`) the three names are undecorated
     `@triton.jit` placeholders from
     `_triton_kernels/attention/pa_mqa_logits.py:558/590/622`, and only the
     prebuilt AOT path runs (`enable_jit_gluon_pa_mqa_logits_kernel =
     False`). Build both tables only on Triton >= 3.5. `:575`
     (`kernel_fn.fn.__name__`) runs only on Triton < 3.5, so it keeps
     working on the placeholders.

  Letting intj measure the KV facts per call instead (one launcher per
  kernel) would compile the unpadded mode with facts it does not have now,
  and the AOT path loads prebuilt binaries by hash; so not that. Check: the
  hsaco/json hashes of the AOT benchmark and `test_pa_mqa_logits*`, and a
  GPU-free import.
- **`test_metadata_redirect.py` `empty_kernel`**: nothing; it is a decorator
  inside `test_jit` because it wraps the test's temp dir.
- **`docs/intj_launch_exceptions.md`**: replace the three "named" rows with
  the outcomes above, keep `pa_attention_kernel_test` with the measured
  reason, and fold in this file. These passages are also stale now (line
  numbers at `ba40fc7e2`):
  - `:67` `# 418 kernels`: 420.
  - `:94` "Three other forms remain, 12 launchers": 10. The form at `:99`,
    `_bar_launch = make_launcher(_bar_kernel, ...)  # the JIT function
    keeps its own name`, describes no launcher now; the two sites it stood
    for build a second launcher over a decorated kernel's JIT function,
    `make_launcher(_k.__self__.options.jit_func, ...)`.
  - `:72-79`, the code that reads `options.jit_func`: add the mla AOT tools
    (through `aiter/aot/triton/mla_jit.py`) and the Iris kernel's
    `_rms_norm`.
  - `:41` `reduce_scatter_rmsnorm_quant_all_gather.py:368`: now `:370`.
  - `:487` "the three whose JIT function must keep its own name remain":
    none does now. The three named launchers left
    (`test_mla_stage2_merge`'s, `pa_attention_kernel_test`'s two) launch a
    decorated kernel's JIT function with other options.
  - `:551-553` "`aiter/aot/triton/decode_mla.py` (unchanged: it still
    compiles the JIT function `_fwd_kernel_stage2_asm`)": both mla AOT
    tools now load it through `aiter/aot/triton/mla_jit.py`. Both were run
    (`asm_mla_decode_fwd` up to its C++ build): with their own signatures
    they fail as on `ba40fc7e2`, with a valid one they compile (above).

## Not verified

- The mla AOT C++ build (`compile_template_op`: needs CK headers) and running
  the generated loader; gfx1250 (the stage-1 asm target) not compiled.
- `mla_decode_fwd` end to end (its stage 1 is HIP asm); the stage-2 launch was
  run directly with production arguments instead.
- The Iris kernel against the real `iris` (not installed), and multi-GPU.
- gfx950 at runtime (compiled offline only).
- The dynamic-options cost on pa_decode_gluon's large-block kernel (only the
  dot kernel was timed).
