# TODO

Open work in this folder. Each item says what to change, why, and how to check it.

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
