# TODO

Open work in this folder. Each item says what to change, why, and how to check it.

## Close the GEMM2 gap to FlyDSL stage 2

Priority: high. Target: per dtype, geomean Gluon/FlyDSL <= 1.00 over the 72-cell matrix
(a4w4, a8w4, a8w8 x dsv4-flash, dsv4-pro, glm52-base, minimax-m3 x T 1..16384), no cell
above 1.10, bit-identical output.

Done 2026-11-02: GEMM2 has its own launch config (`_gemm2_launch_config`): shuffled B
scales at block_m 16/32, and the BLOCK_N 256 16x16x128 tile at block_m 64/128. Kernel
time on HIP 7, median of 4 back-to-back pairs, new/old geomean 0.82 (a4w4), 0.87
(a8w4), 0.85 (a8w8). Gluon/FlyDSL geomean is now 1.06, 1.12 and 1.04 (was 1.29, 1.29
and 1.23 in the same runs). Table: `bench_out/gluon_moe_regression/gemm2/
compare_gemm2_20261102_r0.md`.

Remaining gap, largest first:

- T=1024 (1.16-1.38 in all dtypes). The router's `block_m = next_pow2(T * topk / E)`
  splits experts when the mean is a power of two: dsv4-pro (mean 16), glm52-base and
  minimax-m3 (mean 32). Forcing block_m 32 at dsv4-pro measured 1795 -> 1213 us
  (Gluon) and 1540 -> 998 us (Triton) before this config. `routing.py` is shared with
  Triton, so measure both.
- a8w4 at block_m 128 (1.13-1.26 at T 4096/16384): BLOCK_K 256 does not fit LDS with B
  staged. Try GEMM1's a8w4 block_m 128 tile (B in registers, preshuffled weights).
- The separate `_reduce_grouped` kernel (434 us at glm52-base a8w8 T16384). FlyDSL adds
  the rows in its epilogue (`mode="atomic"`). A fused epilogue changes the summation
  order, so it needs a bit-exact design or an agreed tolerance.
- T=1: a8w4 minimax-m3 is 17 us against 10 us; the shuffle does not help a8w4 at T=1.

Check: kernel time with `aiter.test_common.run_perftest` on HIP 7 against FlyDSL's
`us_stage2` (`op_tests/test_moe_2stage.py --kernel`), and the sha256 of every cell's
output against the previous commit. The scripts are in the untracked
`bench_out/gluon_moe_regression/gemm2/tools/` (`gluon_perftest.py`, `out_hash.py`).

## Refill an LDS scale with `R > 1` right after its `ds_read`: tried, no win

Tried 2026-10-21, then dropped. An LDS component was copied `D * R` steps before its
MFMAs (`F = D * R` instead of `(D - 1) * R + 1`), so a scale at `R > 1` refilled its
buffer the step after its `ds_read` instead of idling `R - 1` steps.

- In-process A/B behind a temporary knob (`a8small/ab.sh`, `BLOCK=40 WARM=15`, four
  weight placements): a8w8 T1024 350.4 -> 352.6 us (+0.6%), T4096 1147.2 -> 1151.3 us
  (+0.4%), slower in all 8 placement pairs.
- Why: every loop wait there also needs a payload copy (`F = 3`), newer than the
  scale's (`F` 5 -> 6), and loads retire in order, so the earlier scale copy changes no
  `vmcnt`: the loop's 24 waits stay identical. The loop body grew by 29 instructions
  (`s_nop`, `v_accvgpr` moves) and the pipeline by one step.
- The schedule itself was right: T0 changed in exactly the 4 predicted cases, and the
  four schedule test files (1370 tests) passed with the knob on.

Revisit only for a config where an LDS scale at `R > 1` is the newest copy some wait
needs: its `F` below its payload's, or its payload in registers. The change is small. In
`Schedule.__init__`, an LDS component's `F` becomes `num_buffers * R` and `depth` the
largest of `L` and `F`; the oracles that derive `F` themselves
(`_audit_buffer_schedule`, `_Machine.load`,
`test_scale_k_step_ratio_sets_buffer_span_and_unroll`) change the same way.

## Keep a wait's immediate under 17 after a join

Priority: high for any config that reaches it. No production pin does: their largest
immediate is 11.

LLVM's `SIInsertWaitcnts` keeps the 16 newest asyncmarks where it merges paths
(`mergeAsyncMarks`), but `recordAsyncMark` pushes with no cap, and `determineAsyncWait`
clamps the immediate only while exactly 16 marks are held. After a join that dropped
marks, one more commit leaves 17. A wait with an immediate `n >= 17` then finds no more
than `n` marks and emits no `s_waitcnt vmcnt`, although the mark it needs was dropped by
the merge, not retired. Found in review, 2026-10-21: in the kernel of
`test_rectangular_read_schedule[4x2-PER_OP-none]` (`PER_OP`, 4x2 panes, bias and gammas
staged through LDS), 16 of the 54 waits with `n >= 16` emit no `vmcnt`, for example a
`wait_asyncmark(17)`. Immediates that large need `PER_OP` with many panes, where each
copy is its own group.

Options, safest first:

1. In `Schedule`, clamp every wait immediate after a join (remainder and drain) to 15,
   as LLVM does with exactly 16 marks held. It stays safe: waiting on a newer group
   retires the older ones in order. It tightens only waits above 15.
2. Fix LLVM: remember that a merge dropped marks, and clamp `N` to the marks held minus
   one. Report it upstream either way.
3. Reject configs whose waits exceed 15 in `validate_pipeline`.

Check: in that kernel's asm every `wait_asyncmark` gets a `vmcnt`; production asm is
unchanged; the suite passes.

## Emit only this `K`'s remainder steps and drain phase

Priority: low. It costs a few loads once per tile.

The static remainder emits `unroll - 1` steps, each behind a runtime `if`, and the
static drain one copy per value of `main % buffer_period`, behind `if phase == p`. LLVM
merges the asyncmarks over every path through them (PIPELINE.md, Wait plan), including
remainder counts this `K` never runs. With a component at `R > 1`, an odd count moves
that component's copies by one step, so the merge lowers some executed waits.

Measured 2026-10-21 with a model of LLVM's asyncmark merge over the real control flow,
which reproduces all 187 asyncmark waits of the 11 distinct production kernels:

- Prod a8w8 T1024/T4096 (`PER_OP`, LDS scales at `R = 2`, `rem = 2`): 7 of the 24 drain
  waits get `vmcnt` 17-22 where this `K` needs 20-25, or no wait for 3 of them.
- The same pin under `PER_STAGE_WHOLE` emits `vmcnt(12)` at remainder step 1, where 16
  is needed.
- At `R = 1` nothing changes: every other production cell, and every static drain
  (`buffer_period <= 3` rules out `R > 1`). In T0, only the three `R > 1` cases.
- The loop-skip path changes no wait. LLVM keeps the zero-trip guard only in the
  static-drain kernels (a8w4 and a8w8 T16-T256, all bf16).

Peeling cannot fix this: these are not loop steps.

Plan: `K` is constexpr and the host passes `NUM_K = num_k_tiles(K)` (Wait plan,
Precondition), so `rem` and `main % buffer_period` are compile-time values. Emit exactly
this `K`'s `rem` remainder steps and its drain phase, without the runtime `if`s, which
also drops their dead code. Nesting the `if`s is not enough: it keeps the odd counts.
`NUM_K` is a runtime argument on purpose
(`test_num_k_is_a_runtime_scalar_without_specialization`), so decide whether the kernel
asserts `NUM_K == num_k_tiles(K)` or keeps a runtime path for other values.

Check: in prod a8w8 T1024 the 7 drain waits rise to the executed path's `vmcnt`, no
other production kernel's waits change, the outputs stay bit-identical and the suite
passes.

## Count register buffer loads in a wait's `vmcnt`: verify, then drop

This item assumed that Triton's `UpdateAsyncWaitCount` sets the `vmcnt` and counts only
direct-to-LDS copies, so a wait would block on every register `buffer_load` issued after
its needed copy. On gfx950 that pass skips its count: Triton lowers waits to LLVM
asyncmarks, and LLVM's `vmcnt` counts every load after the needed group's last LDS
copy, register loads included (PIPELINE.md, Wait plan).

Evidence, 2026-10-21: in prod a8w4 T1024, B payload in registers, each loop step issues
12 LDS-copy instructions and 8 register `buffer_load`s, and its `wait_group(1)` becomes
`vmcnt(20)`. A model of LLVM's asyncmark merge that counts register loads reproduces all
187 asyncmark waits of the 11 distinct production kernels.

To close it, about 30 minutes: compile the T0 register cases `breg`, `ascale-reg`,
`bscale-reg` and `bscale-reg-d2`, and check that each loop `s_waitcnt vmcnt(N)` after a
`; wait_asyncmark(n)` equals the LDS-copy plus register-load instructions issued after
its needed group's last LDS copy. If they all do, delete this item.

## Make `K` a runtime argument

Priority: low until one tuning config must serve many `K`. Audited 2026-10-21; nothing
changed yet.

The schedule already runs any valid `NUM_K` with exact waits (PIPELINE.md, Wait plan,
Precondition). The constexpr `K` (`CFG_K`) still reaches the live kernel in four places:

1. `validate` and `validate_pipeline`, as `static_assert`s: `K % BLOCK_K`, whole
   `SCALE_SLOT_K` tiles, the minimum `num_k`. The host already runs both for each `K`
   (`_validate_launch_config`), so the kernel can drop the parts that read `K`.
2. `Schedule(tc, layout, K)`: build it for any valid `num_k`, a multiple of `lcm(R)` of
   at least `depth + peeled + unroll`. `num_k` reaches the code only modulo `R`
   (`final_register_index`, the drain's and final MFMA's `PHASE` and `K_PHASE`), and
   `validate` makes `num_k % R == 0`.
3. HBM offsets: `K` is the stripe stride of a shuffled scale
   (`_shuffled_scale_register_offsets`, `_shuffled_scale_stage_offsets`) and of
   preshuffled B (`K_STORAGE` in `b_payload_hbm_offsets`). They are computed once per
   tile, so a runtime multiply is cheap; expect asm changes outside the K loop only.
4. Raw scale row strides: `a_scale_stride_m` and `B_SCALE_STRIDE_N` are `K / 32`, kept
   constexpr so direct-to-LDS scale copies keep their sub-16-byte alignment
   (`test_a_scale_row_stride_preserves_sub_16_byte_alignment`). A runtime stride needs a
   constexpr alignment hint (`gl.multiple_of`), so one kernel serves each alignment.

Also:

- `A_HIDDEN_DIM` and `B_HIDDEN_DIM` are constexpr `K` that nothing in the kernel
  reads.
- The launch metadata reads `CFG_K` for the kernel name and flops.
- The host picks the tuning config by `(M, N, K)`, so one binary serves only the `K`
  that share a config.
- This works against "Emit only this `K`'s remainder steps and drain phase" above,
  which specializes on `K` further.

Check: T2d asm unchanged inside the K loop, the determinism shas, the suite, and one
build that serves two `K` values bit-exactly.

## Probe the sorted-scale aux table exactly

Priority: low. `_sorted_shuffle_a_scales` (`moe_op_gemm_gluon.py`) checks
`aiter/ops/moe_mxfp4_aux.py::MXFP4_MOE_SUPPORTED_SHAPES` before it calls
`mxfp4_moe_sort_scales`, because the aux module is torch-free since the 2026-11-02
rebase and a missing instance now aborts the process instead of raising. That set is
a hand-kept subset of `gen_instances.py::SHAPES`: on 2026-11-02 it lacked the
`(NE, H)` pairs `(256, 6144)`, `(384, 6144)` and `(512, 2048)`, which then take the raw
scale path although an instance exists. No benchmarked shape is affected (all use
`NE=33, H=7168`).

- Fix: add a C++ `mxfp4_moe_sort_scales_is_supported(NE, D_HIDDEN, MB)` probe beside
  `mxfp4_moe_sort_internal_is_supported`, and call it instead of the set.
- Check: `op_tests/triton_tests/moe/test_moe_gemm_gluon_isa.py` still passes, and the
  T2d asm hashes stay the same.
