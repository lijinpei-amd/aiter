# TODO

Open work in this folder. Each item says what to change, why, and how to check it.

## Refill an LDS scale with `R > 1` right after its `ds_read`

Through LDS, a buffer is free once its `ds_read` has run, at step `t - 1`, but the next
copy into it comes at `s + D * R = t + R - 1`, `R` steps later (`F = L = (D - 1) * R + 1`).
The buffer idles `R - 1` of every `D * R` steps: in PIPELINE.md's `D = 3`, `R = 2` table it
is free at `t` and refilled at `t + 1`. A payload, at `R = 1`, is already refilled at `t`.

Issuing the copy at `t` gives `F = D * R` through LDS, `R - 1` more steps of prefetch from
the same LDS. LDS copies then issue at multiples of `R`, as register copies already do. At
`R = 1` nothing changes.

Affected today (T0): the production pins a8w8 T1024 and T4096 (A and B scale at `D = 3`,
`R = 2`: `F` 5 -> 6, depth 5 -> 6), and the T0 cases `mxfp4-bk128-sk8-R2` (depth 5 -> 6)
and `mxfp4-bk128-sk16-R4` (`R = 4`: depth 9 -> 12).

Plan, about half a day including the A/B:

1. In `Schedule.__init__`, set `load_to_mfma_distance` of an LDS component to
   `num_buffers * R`, and `depth` to the largest of `L` and `F` over the present
   components (an LDS `F` now exceeds its `L` when `R > 1`). Register `F` stays
   `(D - 1) * R`. Put the change behind a temporary constexpr knob for the A/B in step 6.
2. Confirm that every consumer takes `F` as an input and needs no change of its own:
   `_step_plan` (`active`, and `loads` at `(x + F) % R == 0`), `_replay` (the copy at
   `x + 1 - F`), `_index` / `static_lds_index` (`(x + F) // R`), `scale_soff_steps`
   (producer phase `-F % R`), and the peeling bounds.
3. Ordering: the copy at step `t` now overwrites the buffer the `ds_read` at `t - 1`
   read, as a payload copy already does. The pipeline issues no barrier of its own;
   Triton's membar pass inserts one. In the asm of the affected cells, confirm an
   `s_barrier` separates every scale `ds_read` from the next copy into its buffer, and
   count the `s_barrier` and `s_waitcnt` changes.
4. Test oracles: `_audit_buffer_schedule` in `test_moe_gemm_gluon_config.py` and the
   symbolic machine in `test_moe_gemm_gluon_buffered_schedule.py` derive `F` on their
   own; change both. Add a test that an LDS scale at `R > 1` copies at `x % R == 0` and
   refills its buffer at the step after the `ds_read`.
5. The minimum K (`num_k >= depth + peeled + unroll`) grows with the depth. Check that
   the ladder's and the tests' smallest K per dtype still validate.
6. A/B the knob in-process on a8w8 T1024 and T4096 (`bench_out/gluon_moe_regression/
   a8small/ab.sh`, `BLOCK=40 WARM=15`). Keep the change only if it wins beyond noise,
   then remove the knob; otherwise record the result here and drop it.
7. Docs: in PIPELINE.md, the `F` definition, the `D = 3`, `R = 2` table (the LDS copy
   moves to `t - 6` and its refill to `t`), the copy residues mod `R`, and the
   `s + D * R` paragraph. Decide whether `L` stays `(D - 1) * R + 1` or becomes the true
   LDS live span, `D * R`. Then delete this item.

Check:

- T0 differs only for the four cases above; T0b SAME.
- T2d and the production asm (`/tmp/prodasm`-style, 20 cells): only cells with an LDS
  scale at `R > 1` change. The two determinism shas are unchanged.
- The affected cells' outputs are bit-identical before and after: the MFMAs and their
  order do not change, only when the scale arrives.
- Suite: no new failures against the latest base.

## Drop empty commit groups

Priority: low. Today's configs have empty groups only in drain steps, never in the
prologue or the loop, so no steady-state wait changes. Raise it if a config puts one in
the prologue or the loop.

`PER_PANE` commits one group per pane per step, and the per-stage schemes one per step,
even when nothing was copied through LDS (PIPELINE.md, Commit plan: "empty included").
An empty group counts in a wait's immediate, but it repeats the previous asyncmark, so
it moves no `vmcnt` on a straight path (PIPELINE.md, Wait plan). Dropping empty groups
makes the plan simpler and the emitter commit exactly `commits`.

Affected today (T0, emitted steps): only drain steps, once no LDS component copies any
more. That covers the 10 production pins under `PER_STAGE_WHOLE` (a4w4, a8w4 and bf16 at
T256-T4096, a8w8 T256) and the T0 cases `tuned-mxfp4-PER_PANE`,
`tuned-mxfp4-PER_STAGE_WHOLE` and `tuned-mxfp4-PER_STAGE_WARP_PIPELINE`. In production
no prologue step is empty: some LDS component copies at every step from `-depth` on.
Other ring depths, placements or pane ownerships can add empty groups elsewhere.

Plan, about half a day:

1. In `_step_plan`, close a group only when it holds a copy: under `PER_PANE`
   `(ops,) if ops else ()`, and under the per-stage schemes `(whole,)` only if `whole`.
   `_replay` counts the groups in `commits`, so the immediates follow.
2. `_pipeline.py` commits by scheme flag, not from `commits`: in `_buffer_load_pane`
   (`commit_per_pane`, the prologue's `MARK_TILE`) and in `_step_live` (both per-stage
   commit points). Gate each on a non-empty `Schedule.commits(STEP, pane_index)`, so
   that the plan and the emitted program stay one.
3. Waits: with a varying group count per step, early steps can have immediates that
   differ from the steady ones. The window peel already compares immediates; check that
   `test_peeled_wait_prefix_is_minimal` still holds.
4. Test oracles that assume a fixed group count per pane or step:
   `_audit_buffer_schedule` in `test_moe_gemm_gluon_config.py` and the symbolic machine
   in `test_moe_gemm_gluon_buffered_schedule.py`. Add a test that no step commits an
   empty group.
5. Docs, PIPELINE.md: the Commit plan table ("empty included" three times, and "An empty
   group still takes a FIFO slot") and the Wait plan's empty-group bullet. Then delete
   this item.

Check:

- T0: only the cases above change, in their commits and waits. T0b SAME.
- Asm, the 20 production cells: no `s_waitcnt vmcnt` should change; count any that do.
  Instruction order may still move where an empty asyncmark disappears.
- The affected outputs are bit-identical, because only waits move. The suite has no new
  failures against the latest base.
- If the loop's waits change, A/B in-process (`a8small/ab.sh`). Drain-only changes are
  unlikely to show up in timing; keep them for the simpler plan.

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
