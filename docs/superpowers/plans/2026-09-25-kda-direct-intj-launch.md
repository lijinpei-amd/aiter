# KDA Direct intj Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace KDA's `_tensor_key` and `_FastLaunch` with direct intj launches for both Gluon kernels and each single-config Triton kernel, while retaining decorated Triton K1/K2 calls for multi-config tuning.

**Architecture:** Build one cached, device-bound native handle per raw JIT, compile option set, baked constant set, and Triton knob state. Each KDA invocation validates input devices, computes its current grid and stream, and supplies current tensors positionally. Only multi-config K1/K2 autotuning calls the existing decorated Triton kernel.

**Tech Stack:** Aiter Python 3.10+, Triton 3.7.1/3.8.0, intj `make_launcher`, PyTorch ROCm, gfx942 and gfx950.

**Spec:** `docs/superpowers/specs/2026-09-25-kda-direct-intj-launch-design.md`

> **Amendment (2026-09-27): multi-config K1/K2 also launch through intj.** intj
> `develop` now accepts `@triton.autotune`/`@triton.heuristics` chains in
> `make_launcher` (intj `docs/Usage.md`, "Autotune and heuristics"), so the
> decorated-Triton branch below is not built. Task 1 (in-JIT Gluon layouts),
> the cached device-bound handles, and the device/stream rules stand as
> written. What changes in Tasks 2-4:
>
> - K1, K2 and the scan pass their decorated kernel (`_flash_kda_prepare_kernel`,
>   `_flash_kda_segment_kernel`, `_flash_kda_seg_scan_kernel`) to
>   `_intj_launcher(kernel, device, options=(), baked=(), grid_cpp=None)`,
>   whatever the config count; there is no `configs[0]` branch. intj tunes on a
>   miss and drops tuned and heuristic values from the call. K1's
>   `IS_VARLEN`/`HAS_BIAS` and the scan's `HAS_H0` stay heuristics
>   (`x is not None`, lowered to C). K2 bakes only `CM_OUT`; its grid is
>   `grid_cpp=_k2_grid` reading the tuned `BW`, with `num_segs * H` as an extra
>   launcher argument. Gluon handles use `grid_arg=2`; Gluon K2 passes `BW`.
> - `return_compiled` is left at its default (`False`).
> - Each handle owns private tuner copies. `test_tuner_keeps_the_two_schedules_apart`
>   collects keys from the private K2 tuners (spying `Autotuner.run`), and
>   swapping a tuner's configs requires `_cached_intj_launcher.cache_clear()`.
> - Tests: `test_intj_launch.py` forbids `JITFunction.run` for both one and
>   several configs; its `ordinary_launches()` adapter also handles decorated
>   kernels and `grid_cpp`. The retention test needs intj `ad8c832`
>   or later; before it, a tuned handle kept its last miss's arguments.
> - Task 4: no multi-config decorated Triton calls remain to record.

## Global Constraints

- Use `grid_arg=2`, `bind_device=True`, and `return_compiled=False` for every native handle.
- Keep the current GPU equal to `q.device` through the whole call, and reject every non-null input pointer tensor on another device before the first native launch.
- Key native handles by the JIT source key, device, options, baked values, the JIT debug flag, and current Triton runtime debug, instrumentation, and fpsan knobs. Reject interpreter mode before checking this cache. Never retain an input or workspace tensor in the handle factory.
- Select `configs[0]` only when a K1/K2 tuner has one config. With more than one, execute the existing decorated kernel; default gfx950 has one K1 config and one fallback K2 config, while opt-in tuning has four K1 and 24 K2 candidates and tests may install the six published K2 candidates. Do not copy Triton's tuning-key algorithm or use mutable `best_config`. A separate intj native-autotune design will replace the warmed decorated path after this phase.
- Bypassing K1's outer `@triton.heuristics` requires passing `IS_VARLEN = cu_seqlens is not None` and `HAS_BIAS = dt_bias is not None` explicitly to its raw JIT. The multi-config branch still calls the outer wrapper.
- Keep intj's global-value refusal, Gluon source mode, and constant decoder unchanged. Preserve Aiter Python 3.10 and Triton 3.7.1 support; do not silently fall back on a single-config intj refusal.
- Read the current stream immediately before each native call. Build kernel arguments from the current invocation, not a prior call's tensors. Compare baseline and final with the same resolved intj `TorchAccess` mode; time warmed full-forward host calls with synchronization outside short measured blocks.

## File map

| Repository | File | Responsibility |
| --- | --- | --- |
| Aiter | `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py` | Make K1 layouts JIT-local; remove its host wrapper. |
| Aiter | `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k2.py` | Derive K2 layouts from `gl.num_warps()` inside JIT; remove layout args and host wrapper. |
| Aiter | `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py` | Validate devices, cache native handles, dispatch all five kernels, preserve multi-config tuning. |
| Aiter | `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py` | Delete when no callers remain. |
| Aiter | `op_tests/triton_tests/chunk_delta_attn/test_fast_launch.py`, `test_flash_kda.py` | Replace wrapper tests with direct-launch, graph, device, and route checks. |
| Aiter | `docs/intj_launch_exceptions.md` | Record the two intentional multi-config Triton calls and measured results. |
| intj | No file changes | Existing `GluonASTSource` and native specialization are sufficient. |

---

### Task 1: Make the two Gluon JIT sources self-contained

**Files:**
- Modify: `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py:12-30,108`
- Modify: `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k2.py:12-48,133-153`
- Modify: `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:1076-1097`
- Test: `op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py`

**Interfaces:** K1 retains the same JIT signature. K2 removes the eight layout parameters `MMA`, `A_OP`, `B_OP`, `MMA_B`, `A_OP_B`, `B_OP_B`, `BLK`, `SH_KR`; callers keep `BW` and `num_warps`.

- [ ] **Step 1: Add a failing source-safety test** to `test_flash_kda.py`:

```python
def test_gluon_kda_jits_have_no_captured_layout_globals():
    from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k1 import (
        k1_prepare_gluon,
    )
    from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k2 import (
        k2_ab_fused_gluon,
    )

    for kernel in (k1_prepare_gluon, k2_ab_fused_gluon):
        kernel.cache_key  # Triton populates used_global_vals lazily.
        assert not kernel.used_global_vals
```

- [ ] **Step 2: Run the test and see K1's global layouts reported.** Run: `AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" /tmp/gb2/bin/python -m pytest -q op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py::test_gluon_kda_jits_have_no_captured_layout_globals`. Expected: FAIL on K1.

- [ ] **Step 3: Move K1's 12 declarations from module scope to the start of `k1_prepare_gluon`, just after `gl.static_assert`.** Keep their values exactly:

```python
    _BLK_WARP_K: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [1, 0])
    _BLK1: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
    _BLK_CC: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [2, 1], [1, 0])
    _MMA_F16: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[2, 1]
    )
    _AF16: gl.constexpr = gl.DotOperandLayout(0, _MMA_F16, 1)
    _BF16: gl.constexpr = gl.DotOperandLayout(1, _MMA_F16, 1)
    _MMA_B16: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 1]
    )
    _A8_16: gl.constexpr = gl.DotOperandLayout(0, _MMA_B16, 8)
    _B8_16: gl.constexpr = gl.DotOperandLayout(1, _MMA_B16, 8)
    _SH_A: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    _SH_B: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
    _SH_CC_F: gl.constexpr = gl.SwizzledSharedLayout(1, 2, 8, [0, 1])
```

Keep the temporary old host wrapper working by setting `_NUM_WARPS = 2`; remove its now-unused `math` import. Task 2 removes that wrapper and constant.

- [ ] **Step 4: Replace K2's `build_layouts` and eight JIT parameters with JIT-local values.** Put this at the start of `k2_ab_fused_gluon`, before `i_w = gl.program_id(0)`:

```python
    nw: gl.constexpr = gl.num_warps()
    MMA: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, nw]
    )
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MMA, 8)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MMA, 8)
    MMA_B: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, nw]
    )
    A_OP_B: gl.constexpr = gl.DotOperandLayout(0, MMA_B, 8)
    B_OP_B: gl.constexpr = gl.DotOperandLayout(1, MMA_B, 8)
    BLK: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0])
    SH_KR: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
```

Delete K2's `KW`, `KW_BIG`, `build_layouts`, and unused `functools` import. Remove `**_g2.build_layouts(nw)` from the existing Gluon K2 call in `flash_kda.py`; leave `num_warps=nw` intact for now.

- [ ] **Step 5: Run the source-safety test on both supported Triton versions.** Run the Step 2 command with `/tmp/intj-compat-py312/bin/python`, then `/tmp/gb2/bin/python`. Expected: PASS twice. Run Appendix A with both interpreters; expected: K1, narrow K2, and four-warp wide K2 compile on both versions with no tracked globals.

- [ ] **Step 6: Commit the independently compilable source change.** Run `git add aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k2.py aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py`, then `git diff --cached --check`, then `git commit -m 'Build KDA Gluon layouts inside JIT kernels'`.

### Task 2: Launch both Gluon routes through native intj

**Files:**
- Modify: `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:50-55,847-849,896-900,951-983,1071-1098`
- Modify: `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py:287-348`
- Modify: `aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k2.py:222`
- Test: `op_tests/triton_tests/chunk_delta_attn/test_fast_launch.py`, `test_flash_kda.py`

**Interfaces:** `flash_kda._intj_launcher(jit, device, options, baked=())` returns a `(stream, gx, gy, *kernel_args)` callable. Its cache identity also includes source and Triton knobs; no tensor enters that cache. K1/K2 Gluon host wrappers disappear; `flash_kda_fwd` is their only in-tree caller.

- [ ] **Step 1: Add a failing input-device test** to `test_fast_launch.py`:

```python
def test_rejects_cpu_input_pointer_before_launch():
    args = make_inputs(1, 512, 12)
    args["A_log"] = args["A_log"].cpu()
    with pytest.raises(ValueError, match="same GPU"):
        run(args)


def test_cached_native_handle_refuses_interpreter_mode(monkeypatch):
    from intj.launcher import UnsupportedKernel
    from triton import knobs
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    device = torch.cuda.current_device()
    fk._intj_launcher(_write_one, device, ())
    monkeypatch.setattr(knobs.runtime, "interpret", True)
    with pytest.raises(UnsupportedKernel, match="TRITON_INTERPRET"):
        fk._intj_launcher(_write_one, device, ())
```

Run: `AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" /tmp/gb2/bin/python -m pytest -q op_tests/triton_tests/chunk_delta_attn/test_fast_launch.py -k 'rejects_cpu_input_pointer or cached_native_handle_refuses_interpreter_mode'`. Expected: both fail before the new guards exist.

- [ ] **Step 2: Add the factory and device guard in `flash_kda.py`.** Import `Constexpr` and `make_launcher` from `intj`, `UnsupportedKernel` from `intj.launcher`, plus `knobs` and `driver` from Triton; keep `functools` and `torch` imports already in the file. The two-level cache key is intentional: intj snapshots debug and instrumentation values when the handle is made.

```python
@functools.cache
def _cached_intj_launcher(jit, source_key, device, options, baked,
                          jit_debug, debug, instrumentation, fpsan):
    del source_key, jit_debug, debug, instrumentation, fpsan
    with torch.cuda.device(device):
        return make_launcher(
            jit,
            grid_arg=2,
            bind_device=True,
            return_compiled=False,
            options=dict(options),
            extra_annotation={name: Constexpr(value=value) for name, value in baked},
        ).bind_device(device)


def _intj_launcher(jit, device, options, baked=()):
    if knobs.runtime.interpret:
        raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
    return _cached_intj_launcher(
        jit, jit.cache_key, device, options, baked,
        jit.debug,
        knobs.runtime.debug,
        knobs.compilation.instrumentation_mode,
        getattr(knobs.compilation, "fpsan_homomorphic_casts", None),
    )


def _validated_device(q, *inputs):
    if q.device.type != "cuda":
        raise ValueError("FlashKDA requires GPU tensors on the same GPU")
    device = q.get_device()
    if driver.active.get_current_device() != device:
        raise ValueError("FlashKDA requires the input GPU to be current")
    if any(t is not None and t.device != q.device for t in inputs):
        raise ValueError("FlashKDA requires input tensors on the same GPU")
    return device
```

At the start of `flash_kda_fwd`, after reading `q.shape`, call `_validated_device(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, chunk_indices)` and keep its returned `device` for every native handle and stream read.

- [ ] **Step 3: Replace the Gluon K1 wrapper call with a direct native call.** Remove `gluon_k1_prepare`, `_k1_fast`, `_NUM_WARPS`, and the `fast_launch` import from `flash_kda_k1.py`. In the `use_gluon_k1` arm of `flash_kda_fwd`, import `k1_prepare_gluon` and call:

```python
        _intj_launcher(
            k1_prepare_gluon, device, (("num_warps", 2),),
            (("CM_WS", CM_STORE), ("CM_LOAD", CM_LOAD)),
        )(
            driver.active.get_current_stream(device),
            total_tiles if cu_seqlens is not None else NT, B * H,
            q, k, g, beta, A_log, dt_bias,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv_mqk,
            cu_seqlens, chunk_indices, scale, lower_bound,
            T, NT, total_tiles, H, K, C, inv_block,
            cu_seqlens is not None, dt_bias is not None,
        )
```

- [ ] **Step 4: Replace the Gluon K2 wrapper call with a direct native call.** Remove `k2_ab_fused_fast` and its `fast_launch` import from `flash_kda_k2.py`. Keep `_k2_gluon_schedule` unchanged. In `flash_kda_fwd`'s Gluon K2 arm, call:

```python
            bw, nw = _k2_gluon_schedule(V, num_segs, H)
            _intj_launcher(
                _g2.k2_ab_fused_gluon, device, (("num_warps", nw),),
                (("BW", bw),),
            )(
                driver.active.get_current_stream(device),
                triton.cdiv(V, bw), num_segs * H,
                ws_kd, ws_kr, ws_gt, ws_inv_mqk,
                v, beta, b_seg, A_seg,
                seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end,
                total_tiles, H, K, V, C,
            )
```

- [ ] **Step 5: Adapt the transitional tests and run the gfx950 route checks.** In `test_fast_launch.py::_wrapped`, remove the two Gluon imports and their `_k1_fast`/`k2_ab_fused_fast` candidates; the three Triton wrappers still exist until Task 3. In `test_flash_kda.py::test_cases_reach_the_gluon_k2`, replace the old `_g2.k2_ab_fused_fast` spy with this spy around `flash_kda._intj_launcher`:

```python
    original = _flash_kda._intj_launcher

    def counted(jit, device, options, baked=()):
        native = original(jit, device, options, baked)
        if jit is not _g2.k2_ab_fused_gluon:
            return native

        def launch(*args):
            reached.add(current)
            return native(*args)

        return launch

    _flash_kda._intj_launcher = counted
    try:
        for current, make in _CASES.items():
            args, kw = make()
            with _route(k1=True, k2=True):
                run_flash(*args, **kw)
    finally:
        _flash_kda._intj_launcher = original
```

Add one test in `test_flash_kda.py` that actually launches the published four-warp K2 schedule on gfx950:

```python
def test_wide_gluon_k2_launch_matches_triton(monkeypatch):
    if not _flash_kda._gluon_k2_usable(FLASH_KDA_CHUNK, K_DIM, K_DIM):
        pytest.skip("Gluon K2 requires gfx950")
    from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn import (
        flash_kda_k2 as _g2,
    )

    args = make_inputs(1, 512, 64)
    assert _flash_kda._k2_gluon_schedule(K_DIM, 16, 64) == (64, 4)
    with _route(k1=True, k2=False):
        want, want_state = run_flash(
            *args, chunks_per_seg=1, output_final_state=True
        )

    seen = []
    original = _flash_kda._intj_launcher

    def counted(jit, device, options, baked=()):
        if jit is _g2.k2_ab_fused_gluon:
            seen.append((dict(options), dict(baked)))
        return original(jit, device, options, baked)

    monkeypatch.setattr(_flash_kda, "_intj_launcher", counted)
    with _route(k1=True, k2=True):
        got, got_state = run_flash(
            *args, chunks_per_seg=1, output_final_state=True
        )
    assert seen == [({"num_warps": 4}, {"BW": 64})]
    assert rel_err(got, want) < 2e-3
    assert rel_err(got_state, want_state) < 1e-4
```

Run both new Task 2 tests on gfx942:

```sh
AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 \
  PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" \
  /tmp/gb2/bin/python -m pytest -q \
  op_tests/triton_tests/chunk_delta_attn/test_fast_launch.py \
  -k 'rejects_cpu_input_pointer or cached_native_handle_refuses_interpreter_mode'
```

Sync only this isolated Aiter worktree and the unchanged intj worktree to a new temporary directory on the gfx950 host; do not touch its dirty `/raid/jinpli/workspace/home01/jinpli/development/aiter` checkout. This setup is local to Task 2; Task 4 stages a fresh baseline/final pair:

```sh
KDA_GLUON_STAGE=$(ssh -p 30004 jinpli@localhost 'mktemp -d /tmp/kda-gluon.XXXXXX')
ssh -p 30004 jinpli@localhost "mkdir -p '$KDA_GLUON_STAGE/aiter' '$KDA_GLUON_STAGE/intj'"
rsync -a --exclude .git -e 'ssh -p 30004' ./ "jinpli@localhost:$KDA_GLUON_STAGE/aiter/"
rsync -a --exclude .git -e 'ssh -p 30004' /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid/ "jinpli@localhost:$KDA_GLUON_STAGE/intj/"
ssh -p 30004 jinpli@localhost "cd '$KDA_GLUON_STAGE/aiter' && AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 AITER_FDA_USE_GLUON=1 HIP_VISIBLE_DEVICES=0 PYTHONPATH='$KDA_GLUON_STAGE/aiter:$KDA_GLUON_STAGE/intj' /raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python -m pytest -q op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py -k 'cases_reach_the_gluon_k2 or routes_agree or wide_gluon_k2_launch_matches_triton'"
```

Expected: both gfx942 guard tests and the gfx950 route tests pass; the wide-route assertion proves the four-warp native handle was used.

- [ ] **Step 6: Commit the Gluon launch change.** Stage only the three KDA sources and the two changed tests; run `git diff --cached --check`; commit with `git commit -m 'Launch KDA Gluon kernels through intj'`.

### Task 3: Replace the three Triton wrappers and delete `_FastLaunch`

**Files:**
- Modify: `aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py:67,847-849,985-1012,1034-1055,1061-1064,1100-1151`
- Delete: `aiter/ops/triton/_triton_kernels/chunk_delta_attn/fast_launch.py`
- Replace: `op_tests/triton_tests/chunk_delta_attn/test_fast_launch.py` with direct-launch tests (`test_intj_launch.py`)
- Test: `op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py`

**Interfaces:** K1 raw JIT is `_flash_kda_prepare_kernel.fn.fn`; K2 raw JIT is `_flash_kda_segment_kernel.fn`; scan raw JIT is `_flash_kda_seg_scan_kernel.fn`. In the single-config path, `BW` and string modifiers are baked and therefore omitted from positional kernel arguments.

- [ ] **Step 1: Write a test that forbids the single-config decorated launch.** In the replacement test file, force the two Gluon route flags off, set each tuner to its first config, replace K1/K2/scan decorated `.run` with a function that raises, and spy on `_intj_launcher`. The spy makes this a reliable red test even if an old `_FastLaunch` entry is already warm and bypasses decorated `.run`:

```python
def test_single_config_never_enters_decorated_run(monkeypatch):
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    monkeypatch.setattr(fk, "AITER_FDA_USE_GLUON_K1", False)
    monkeypatch.setattr(fk, "AITER_FDA_USE_GLUON_K2", False)
    k1 = fk._flash_kda_prepare_kernel.fn
    k2 = fk._flash_kda_segment_kernel
    scan = fk._flash_kda_seg_scan_kernel
    monkeypatch.setattr(k1, "configs", [k1.configs[0]])
    monkeypatch.setattr(k2, "configs", [k2.configs[0]])
    expected = {k1.fn, k2.fn, scan.fn}
    seen = set()
    original = fk._intj_launcher

    def counted(jit, device, options, baked=()):
        if jit in expected:
            seen.add(jit)
        return original(jit, device, options, baked)

    monkeypatch.setattr(fk, "_intj_launcher", counted)

    def fail(*args, **kwargs):
        raise AssertionError("single-config path used decorated Triton launch")

    for wrapped in (k1, k2, scan):
        monkeypatch.setattr(wrapped, "run", fail)
    run(make_inputs(1, 512, 4), chunks_per_seg=4)
    assert seen == expected
```

- [ ] **Step 2: Use the raw K1 JIT only when the tuner has one config.** Put the current keyword arguments at `flash_kda.py:985-1012` under `if len(_flash_kda_prepare_kernel.fn.configs) > 1:`, but change its callee from `_prepare_fast[grid]` to `_flash_kda_prepare_kernel[grid]`. Keep the outer wrapper on that multi-config branch. Add this `else` arm: it supplies the two heuristic flags explicitly because the raw JIT bypasses `@triton.heuristics`, and bakes the two cache modifiers.

```python
        config = _flash_kda_prepare_kernel.fn.configs[0]
        _intj_launcher(
            _flash_kda_prepare_kernel.fn.fn,
            device,
            (("num_warps", config.num_warps), ("num_stages", config.num_stages)),
            (("CM_QKG", CM_LOAD), ("CM_WS", CM_STORE)),
        )(
            driver.active.get_current_stream(device),
            total_tiles if cu_seqlens is not None else NT, B * H,
            q, k, g, beta, A_log, dt_bias,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv_mqk,
            cu_seqlens, chunk_indices, scale, lower_bound,
            T, NT, total_tiles, H, K, C, inv_block,
            inv_block.bit_length() - 2, (C // inv_block).bit_length() - 1,
            cu_seqlens is not None, dt_bias is not None,
        )
```

- [ ] **Step 3: Replace the scan wrapper with its raw JIT.** The scan always has one host-selected schedule:

```python
        BV_SCAN, SCAN_WARPS = _scan_bv(N, H, V)
        _intj_launcher(
            _flash_kda_seg_scan_kernel.fn, device,
            (("num_warps", SCAN_WARPS),),
        )(
            driver.active.get_current_stream(device),
            triton.cdiv(V, BV_SCAN), N * H,
            A_seg, b_seg, h_in, h0, seq_seg_off,
            H, K, V, BV_SCAN, h0 is not None,
        )
```

- [ ] **Step 4: Replace `_launch_k2` with a current-argument tuple and a config-count branch.** Remove the unconditional `common` dictionary at `flash_kda.py:1034-1055`. The following nested helper has the exact raw JIT argument order; it constructs a dictionary only for the multi-config Triton branch:

```python
    def _launch_k2(*, W, out, h_in, h_out, final_state,
                   INIT_IDENTITY, HAS_H_IN, HAS_V, COMPUTE_OUTPUT,
                   STORE_H_OUT, STORE_FINAL):
        values = (
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv_mqk,
            v, beta, out, h_in, h_out, final_state,
            seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end,
            seg_seq, seg_is_last, total_tiles, _seg_occupancy_class(num_segs),
            H, K, V, W, C,
            INIT_IDENTITY, HAS_H_IN, HAS_V, COMPUTE_OUTPUT,
            STORE_H_OUT, STORE_FINAL, state_v_first,
        )
        tuner = _flash_kda_segment_kernel
        if len(tuner.configs) > 1:
            raw = tuner.fn
            names = tuple(p.name for p in raw.params if p.name not in ("BW", "CM_OUT"))
            assert len(names) == len(values)
            return tuner[
                lambda meta: (triton.cdiv(W, meta["BW"]), num_segs * H)
            ](**dict(zip(names, values)), CM_OUT=CM_OUT_STORE)
        config = tuner.configs[0]
        bw = config.kwargs["BW"]
        return _intj_launcher(
            tuner.fn, device,
            (("num_warps", config.num_warps), ("num_stages", config.num_stages)),
            (("BW", bw), ("CM_OUT", CM_OUT_STORE)),
        )(
            driver.active.get_current_stream(device),
            triton.cdiv(W, bw), num_segs * H,
            *values,
        )
```

Change pass A's `_launch_k2` call to pass `out=None, h_in=None, h_out=buf, final_state=None, W=width, INIT_IDENTITY=identity, HAS_H_IN=False, HAS_V=has_v, COMPUTE_OUTPUT=False, STORE_H_OUT=True, STORE_FINAL=False`. Change pass C's call to pass `out=o, h_in=h_in, h_out=None, final_state=final_state, W=V, INIT_IDENTITY=False, HAS_H_IN=h_in is not None, HAS_V=True, COMPUTE_OUTPUT=True, STORE_H_OUT=False, STORE_FINAL=output_final_state`. The Gluon pass A, segment scan, and K2 output pass C retain their existing ordering.

- [ ] **Step 5: Replace wrapper-specific tests with direct-launch checks.** Rename `test_fast_launch.py` to `test_intj_launch.py`. Keep its input builder and bitwise direct-vs-Triton route cases, but compare through this test-only adapter around `_intj_launcher`; do not ship a production fallback:

```python
@contextlib.contextmanager
def ordinary_launches():
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    saved = fk._intj_launcher

    def ordinary(jit, device, options, baked=()):
        del device
        baked_values = dict(baked)
        names = tuple(p.name for p in jit.params if p.name not in baked_values)

        def launch(stream, gx, gy, *values):
            del stream
            assert len(values) == len(names)
            return jit[(gx, gy)](
                **dict(zip(names, values)), **baked_values, **dict(options)
            )

        return launch

    fk._intj_launcher = ordinary
    try:
        yield
    finally:
        fk._intj_launcher = saved
```

Adapt the existing parity tests to `with ordinary_launches():` for the expected result. Remove assertions about `_FastLaunch._cache`, `_MAX_ENTRIES`, and `recording()`; intj's `tests/test_launcher.py::test_spec_key_is_never_coarser_than_triton` covers the replacement specialization cache. Keep the changed-shape, misalignment, variable-length, segmented, bias, and state parity cases. Add this full-pipeline graph replay check; the call inside capture must read the capture stream at every native launch:

```python
def test_warmed_flash_kda_replays_from_graph(monkeypatch):
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    monkeypatch.setattr(fk, "AITER_FDA_USE_GLUON_K1", False)
    monkeypatch.setattr(fk, "AITER_FDA_USE_GLUON_K2", False)
    k1 = fk._flash_kda_prepare_kernel.fn
    k2 = fk._flash_kda_segment_kernel
    monkeypatch.setattr(k1, "configs", [k1.configs[0]])
    monkeypatch.setattr(k2, "configs", [k2.configs[0]])
    args = make_inputs(1, 512, 4)

    def call():
        return run(args, chunks_per_seg=4)

    eager = call()  # Warm every KDA native handle before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, state = call()
    graph.replay()  # Capture records work; the first replay executes it.
    torch.cuda.synchronize()
    assert torch.equal(out, eager[0])
    assert torch.equal(state, eager[1])
    expected_out, expected_state = out.clone(), state.clone()
    out.zero_()
    state.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected_out)
    assert torch.equal(state, expected_state)
```

Keep the Step 1 CPU-pointer test. Add these focused checks; they replace the wrapper-cache shape-count tests and do not implement another specialization map in Aiter:

```python
def test_same_specialization_uses_current_input_tensors():
    first = make_inputs(1, 512, 12, seed=0)
    second = make_inputs(1, 512, 12, seed=1)
    assert first["q"].data_ptr() % 16 == second["q"].data_ptr() % 16
    first_o, _ = run(first)
    got_o, got_s = run(second)
    with ordinary_launches():
        want_o, want_s = run(second)
    assert not torch.equal(first_o, got_o)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_input_gpu_must_be_current():
    current = torch.cuda.current_device()
    other = 1 if current == 0 else 0
    first = make_inputs(1, 512, 12)
    run(first)  # Warm the same shape on the current GPU.
    args = {
        name: value.to(f"cuda:{other}") if isinstance(value, torch.Tensor) else value
        for name, value in first.items()
    }
    with pytest.raises(ValueError, match="current"):
        run(args)
    with torch.cuda.device(other):
        got_o, got_s = run(args)
        with ordinary_launches():
            want_o, want_s = run(args)
        assert torch.equal(got_o, want_o)
        assert torch.equal(got_s, want_s)


def test_cached_native_handle_does_not_retain_tensor():
    import gc
    import weakref

    args = make_inputs(1, 512, 12)
    ref = weakref.ref(args["q"])
    run(args, chunks_per_seg=4)
    torch.cuda.synchronize()
    del args
    gc.collect()
    assert ref() is None


def test_native_handle_changes_with_debug_knob(monkeypatch):
    from triton import knobs
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    device = torch.cuda.current_device()
    first = fk._intj_launcher(_write_one, device, ())
    monkeypatch.setattr(knobs.runtime, "debug", not knobs.runtime.debug)
    second = fk._intj_launcher(_write_one, device, ())
    assert second is not first


def test_native_handle_changes_with_jit_debug(monkeypatch):
    from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk

    device = torch.cuda.current_device()
    first = fk._intj_launcher(_write_one, device, ())
    monkeypatch.setattr(_write_one, "debug", not bool(_write_one.debug))
    second = fk._intj_launcher(_write_one, device, ())
    assert second is not first
```

- [ ] **Step 6: Remove the obsolete module and verify both modes.** Delete `fast_launch.py` and its import plus `_prepare_fast`, `_segment_fast`, `_seg_scan_fast` definitions. Inspect `rg -n '_FastLaunch|_tensor_key|fast_launch\(' aiter op_tests` and confirm no production import or launch remains. Run `test_intj_launch.py` and `test_flash_kda.py` on gfx942 under `/tmp/intj-compat-py312/bin/python` and `/tmp/gb2/bin/python`. On gfx950, rerun both route modes and the forced-Triton autotuning cases in Task 4. Expected: all existing numerical thresholds hold, and single-config calls reach intj on their first invocation.

- [ ] **Step 7: Commit the direct Triton migration.** Stage only `flash_kda.py`, the deleted `fast_launch.py`, the renamed direct-launch test, and any required `test_flash_kda.py` edit. Run `git diff --cached --check`; commit with `git commit -m 'Launch KDA Triton kernels directly through intj'`.

### Task 4: Record exceptions and benchmark the final route

**Files:**
- Modify: `docs/intj_launch_exceptions.md:13-30,61-66,84-124`
- Test: `op_tests/triton_tests/chunk_delta_attn/test_intj_launch.py`, `test_flash_kda.py`, `test_chunk_delta_attn_fwd.py`

**Interfaces:** No code API change. The deliverable is a verified exception inventory and reproducible baseline/final performance table.

- [ ] **Step 1: Update the exception manifest.** Remove the eight obsolete KDA `_FastLaunch`/Gluon-layout rows and its `CompiledKernel.run` reference. Record the two explicit multi-config decorated Triton calls in `flash_kda.py` and the test-only ordinary Triton comparison launch. Recount the tracked bracket sites from the final source instead of carrying forward the old `37` count.

- [ ] **Step 2: Run final local compatibility checks.** From the Aiter worktree, run the following, with no GPU work inside the Python 3.10 parse check:

```sh
PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" \
  /tmp/intj-compat-py310/bin/python -m compileall -q \
  aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py \
  aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py \
  aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k2.py
AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 \
  PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" \
  /tmp/intj-compat-py312/bin/python -m pytest -q \
  op_tests/triton_tests/chunk_delta_attn/test_intj_launch.py \
  op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py \
  op_tests/triton_tests/chunk_delta_attn/test_chunk_delta_attn_fwd.py
AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 \
  PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" \
  /tmp/gb2/bin/python -m pytest -q \
  op_tests/triton_tests/chunk_delta_attn/test_intj_launch.py \
  op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py \
  op_tests/triton_tests/chunk_delta_attn/test_chunk_delta_attn_fwd.py
```

Run the intj checkout's existing `tests/test_launcher.py::test_gluon_launch_matches_triton` and `::test_spec_key_is_never_coarser_than_triton` on both Python 3.12 environments. Run Aiter's CI-pinned Ruff on the changed Python files:

```sh
uvx --from ruff==0.16.0 ruff check \
  aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_kda.py \
  aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k1.py \
  aiter/ops/triton/_gluon_kernels/gfx950/chunk_delta_attn/flash_kda_k2.py \
  op_tests/triton_tests/chunk_delta_attn/test_intj_launch.py \
  op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py
```

Stop on a real failure; do not convert it into an exception entry.

- [ ] **Step 3: Stage isolated source copies on gfx950 and run correctness.** The remote machine's main Aiter checkout has unrelated dirty work; leave it untouched. Use `ssh -p 30004 jinpli@localhost` and its verified `/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python` (Python 3.14, Torch 2.15, Triton 3.8, MI350X). From the local Aiter worktree, create an isolated baseline from `c204770da58dc3e3d419fcd6d7821df56daac433` and copy it, the final worktree, and intj `8619f45503b182fb24898376c8267bc5881f8e39` to new remote temporary directories. One exact setup pattern is:

```sh
git worktree add --detach /tmp/kda-direct-before c204770da58dc3e3d419fcd6d7821df56daac433
KDA_STAGE=$(ssh -p 30004 jinpli@localhost 'mktemp -d /tmp/kda-direct.XXXXXX')
printf '%s\n' "$KDA_STAGE" > /tmp/kda-direct-stage-path
ssh -p 30004 jinpli@localhost "mkdir -p '$KDA_STAGE/before' '$KDA_STAGE/after' '$KDA_STAGE/intj'"
rsync -a --exclude .git -e 'ssh -p 30004' /tmp/kda-direct-before/ "jinpli@localhost:$KDA_STAGE/before/"
rsync -a --exclude .git -e 'ssh -p 30004' ./ "jinpli@localhost:$KDA_STAGE/after/"
rsync -a --exclude .git -e 'ssh -p 30004' /mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid/ "jinpli@localhost:$KDA_STAGE/intj/"
```

Do not run benchmark cases until gfx950 `test_flash_kda.py`, `test_intj_launch.py`, and `test_chunk_delta_attn_fwd.py` pass for both `AITER_FDA_USE_GLUON=1` and `0`. Use separate processes for those route values and for `CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1`, because the flags are read at import. From the remote stage, the command pattern is:

```sh
KDA_STAGE=$(cat /tmp/kda-direct-stage-path)
ssh -p 30004 jinpli@localhost "cd '$KDA_STAGE/after' && AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 AITER_FDA_USE_GLUON=1 PYTHONPATH='$KDA_STAGE/after:$KDA_STAGE/intj' /raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python -m pytest -q op_tests/triton_tests/chunk_delta_attn/test_intj_launch.py op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py op_tests/triton_tests/chunk_delta_attn/test_chunk_delta_attn_fwd.py"
```

Repeat the command with `AITER_FDA_USE_GLUON=0`. In a third process, run the tuner-key check plus numerical cases for an unsegmented tail and a segmented pass; with Gluon forced off and opt-in tuning enabled, these exercise both K1 and K2 multi-config branches:

```sh
ssh -p 30004 jinpli@localhost "cd '$KDA_STAGE/after' && AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 AITER_FDA_USE_GLUON=0 CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1 HIP_VISIBLE_DEVICES=0 PYTHONPATH='$KDA_STAGE/after:$KDA_STAGE/intj' /raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python -m pytest -q 'op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py::test_triton_route_matches_reference[tail chunk]' 'op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py::test_triton_route_matches_reference[segmented]' op_tests/triton_tests/chunk_delta_attn/test_flash_kda.py::test_tuner_keeps_the_two_schedules_apart"
```

The baseline still has `test_fast_launch.py`; use it for baseline correctness if needed. The tuner-key test above inspects the original `tuner.cache` and applies only while this plan's multi-config path uses decorated Triton. When the separate native-autotune design replaces that branch, change the test to inspect intj's private selection keys and winners or create a fresh launcher and count native selections; clearing `tuner.cache` will not reset native selection state.

- [ ] **Step 4: Benchmark matched baseline and final revisions.** From the local Aiter worktree, run this loop against the two isolated remote trees. It uses GPU 0, a fresh process per route and opt-in setting, five alternating baseline/final rounds, and `--warmup-ms 300 --rep-ms 500` for each GPU-event benchmark:

```sh
KDA_STAGE=$(cat /tmp/kda-direct-stage-path)
set -o pipefail
for KDA_ROUND in 1 2 3 4 5; do
  if [ $((KDA_ROUND % 2)) -eq 1 ]; then set -- before after; else set -- after before; fi
  for KDA_TREE in "$@"; do
    for KDA_CASE in 1:0 0:0 0:1; do
      KDA_ROUTE=${KDA_CASE%%:*}
      KDA_TUNE=${KDA_CASE#*:}
      for KDA_T in 512 16384; do
        ssh -p 30004 jinpli@localhost "cd '$KDA_STAGE/$KDA_TREE' && AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 AITER_FDA_USE_GLUON=$KDA_ROUTE CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=$KDA_TUNE HIP_VISIBLE_DEVICES=0 PYTHONPATH='$KDA_STAGE/$KDA_TREE:$KDA_STAGE/intj' /raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python op_tests/op_benchmarks/triton/bench_flash_kda.py --shape 1 $KDA_T 12 128 128 --warmup-ms 300 --rep-ms 500" \
          | tee "/tmp/kda-event-$KDA_ROUND-$KDA_TREE-$KDA_ROUTE-$KDA_TUNE-$KDA_T.log"
      done
    done
    ssh -p 30004 jinpli@localhost "cd '$KDA_STAGE/$KDA_TREE' && AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 AITER_FDA_ENABLE=0 HIP_VISIBLE_DEVICES=0 PYTHONPATH='$KDA_STAGE/$KDA_TREE:$KDA_STAGE/intj' /raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python op_tests/op_benchmarks/triton/bench_chunk_delta_attn.py --shape 2 4096 16 64 64 --warmup-ms 300 --rep-ms 500" \
      | tee "/tmp/kda-event-$KDA_ROUND-$KDA_TREE-control.log"
  done
done
```

The last command is an unchanged default-pipeline control: this benchmark fixes `CHUNK_SIZE=64`, so it does not route through FlashKDA. For host-call timing, run Appendix B's standalone script in each staged tree for all three route/tuning cases, alternating baseline/final processes five times. Keep first compilation and synchronization outside the timed blocks; report each block, median, and actual route. The full-forward host interval includes argument assembly, allocation, stream reads, and launches. GPU-event results include host gaps in eager runs, so do not attribute their full difference to one kernel.

- [ ] **Step 5: Apply the performance gate and commit the inventory.** Add a dated baseline/final table, environment versions, command lines, access mode, and any inconclusive/noisy rows to `docs/intj_launch_exceptions.md`. If any gfx950 FlashKDA route or shape has a repeatable warmed full-forward host-call regression beyond run-to-run variation, stop acceptance, attribute the cost (especially the opt-in multi-config K2 decorated path), revise Task 3's dispatch, and rerun correctness and matched benchmarks. Otherwise, run `git diff --check`; stage only that document; commit with `git commit -m 'Record KDA direct-launch results'`. Verify `git status --short` is clean in Aiter and intj remains unchanged.

## Appendix A: Offline gfx950 Gluon compile check

Run the following from the Aiter root. It compiles K1 and both published K2 warp-count variants under Triton 3.7.1 and 3.8.0 without executing a GPU kernel:

```sh
for KDA_PY in /tmp/intj-compat-py312/bin/python /tmp/gb2/bin/python; do
AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 \
  PYTHONPATH="$PWD:/mnt/nvme2/jinpli/workspace/home/jinpli/development/workspace/intj/callable_grid" \
  "$KDA_PY" - <<'PY'
from triton.backends.compiler import GPUTarget
from triton.compiler import compile as triton_compile, make_backend
from triton.experimental.gluon._runtime import GluonASTSource
from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k1 import k1_prepare_gluon
from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k2 import k2_ab_fused_gluon

target = GPUTarget("hip", "gfx950", 64)

def check(jit, pointers, scalars, constants, num_warps):
    signature = {}
    for name in jit.arg_names:
        signature[name] = (
            "constexpr" if name in constants else
            pointers[name] if name in pointers else scalars[name]
        )
    jit.cache_key
    assert not jit.used_global_vals
    options = make_backend(target).parse_options({"num_warps": num_warps}).__dict__
    compiled = triton_compile(
        GluonASTSource(jit, signature, constants), target=target, options=options
    )
    print(jit.__name__, "warps", num_warps, "BW", constants.get("BW"),
          "shared", compiled.metadata.shared)

check(
    k1_prepare_gluon,
    {**dict.fromkeys("q k ws_kd ws_qd ws_kr".split(), "*bf16"),
     **dict.fromkeys("g_raw beta_raw A_log dt_bias ws_gt".split(), "*fp32"),
     "ws_inv_mqk": "*fp16",
     **dict.fromkeys("cu_seqlens chunk_indices".split(), "*i32")},
    {"scale": "fp32", "lower_bound": "fp32", "T": "i32", "NT": "i32", "TOTAL_TILES": "i32"},
    {"H": 12, "K": 128, "C": 32, "BC": 16,
     "IS_VARLEN": False, "HAS_BIAS": True, "CM_WS": "", "CM_LOAD": ".cg"},
    2,
)
k2_pointers = {
    **dict.fromkeys("ws_kd ws_kr v_input h_out_a".split(), "*bf16"),
    **dict.fromkeys("ws_gt beta_raw h_out_b".split(), "*fp32"),
    "ws_inv_mqk": "*fp16",
    **dict.fromkeys("seg_chunk_base seg_nchunks seg_tok_base seg_tok_end".split(), "*i32"),
}
for h, bw, warps in ((12, 32, 2), (64, 64, 4)):
    check(k2_ab_fused_gluon, k2_pointers, {"TOTAL_TILES": "i32"},
          {"H": h, "K": 128, "V": 128, "C": 32, "BW": bw}, warps)
PY
done
```

## Appendix B: Host-enqueue timing boundary

Measure the same production `flash_kda_fwd` call in both trees. Capturing and replaying only K1/K2 arguments would exclude different work in the two revisions: the baseline would still run `_FastLaunch`, while the final replay would omit its handle lookup, stream read, and tuple construction. The script below keeps input creation and synchronization outside each timed block, but includes output/workspace allocations and every KDA launch. The short case runs K1 plus unsegmented K2; the long pinned case runs K1, segmented pass A, the scan, and pass C. Label these results **full-forward host-call time**, not isolated K1/K2 cost.

```sh
cat > /tmp/bench_kda_host.py <<'PY'
import os
import statistics
import time

import torch
from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk
from op_tests.triton_tests.chunk_delta_attn.test_flash_kda import LOWER_BOUND, make_inputs


def bench(B, T, H, seg):
    q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H)

    def launch():
        return fk.flash_kda_fwd(
            q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
            scale=scale, lower_bound=LOWER_BOUND, chunks_per_seg=seg,
        )

    for _ in range(3):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(9):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(8):
            launch()
        elapsed = time.perf_counter_ns() - start
        torch.cuda.synchronize()
        samples.append(elapsed / 8_000)  # microseconds per forward call
    k1 = "gluon" if fk.AITER_FDA_USE_GLUON_K1 and fk._gluon_k1_usable(32, 128) else "triton"
    k2a = ("gluon" if fk.AITER_FDA_USE_GLUON_K2 and fk._gluon_k2_usable(32, 128, 128)
           else "triton") if seg else "none"
    print(f"B={B} T={T} H={H} seg={seg} gluon={os.getenv('AITER_FDA_USE_GLUON')} "
          f"autotune={os.getenv('CHUNK_DELTA_ATTN_TRITON_AUTOTUNE')} "
          f"route=K1:{k1},K2A:{k2a},K2C:triton "
          f"k1_configs={len(fk._flash_kda_prepare_kernel.fn.configs)} "
          f"k2_configs={len(fk._flash_kda_segment_kernel.configs)} "
          f"median_us={statistics.median(samples):.3f} "
          f"blocks_us={[round(x, 3) for x in samples]}", flush=True)


bench(1, 512, 12, 0)
bench(1, 4096, 12, 4)
PY
KDA_STAGE=$(cat /tmp/kda-direct-stage-path)
scp -P 30004 /tmp/bench_kda_host.py "jinpli@localhost:$KDA_STAGE/bench_kda_host.py"
set -o pipefail
for KDA_ROUND in 1 2 3 4 5; do
  if [ $((KDA_ROUND % 2)) -eq 1 ]; then set -- before after; else set -- after before; fi
  for KDA_CASE in 1:0 0:0 0:1; do
    KDA_ROUTE=${KDA_CASE%%:*}
    KDA_TUNE=${KDA_CASE#*:}
    for KDA_TREE in "$@"; do
      ssh -p 30004 jinpli@localhost "cd '$KDA_STAGE/$KDA_TREE' && AITER_TRITON_ONLY=1 AITER_USE_SYSTEM_TRITON=1 AITER_FDA_USE_GLUON=$KDA_ROUTE CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=$KDA_TUNE HIP_VISIBLE_DEVICES=0 PYTHONPATH='$KDA_STAGE/$KDA_TREE:$KDA_STAGE/intj' /raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python '$KDA_STAGE/bench_kda_host.py'" \
        | tee "/tmp/kda-host-$KDA_ROUND-$KDA_ROUTE-$KDA_TUNE-$KDA_TREE.log"
    done
  done
done
```

Compare medians and the nine block samples within each route and shape. If queue backpressure shows up in the host interval, report it as host wall time including stalls. For separate K1/K2 attribution, insert temporary timers around each actual call expression in both source revisions, including each revision's handle lookup and stream read, then remove the timers before committing. Do not compare a captured `_FastLaunch` wrapper with a captured raw intj callable.
