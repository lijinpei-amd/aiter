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
Empty groups make the compiler's wait counts conservative. `UpdateAsyncWaitCount`
(Triton, `third_party/amd/lib/TritonAMDGPUTransforms/UpdateAsyncWaitCount.cpp`) turns
`wait_group(n)` into a `vmcnt` by walking back `n + 1` commit groups and summing their
LDS-copy instructions, taking the minimum over every control-flow path. An empty group
uses up one of the `n` but adds no instruction.

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
3. Waits: with a varying group count per step, a non-`PER_OP` scheme can have early steps
   whose immediates differ from the steady ones, so `peeled` can exceed 1. The peeling
   scan already handles it; check that `test_peeled_wait_prefix_is_minimal` still holds.
4. Test oracles that assume a fixed group count per pane or step:
   `_audit_buffer_schedule` in `test_moe_gemm_gluon_config.py` and the symbolic machine
   in `test_moe_gemm_gluon_buffered_schedule.py`. Add a test that no step commits an
   empty group.
5. Docs, PIPELINE.md: the Commit plan table ("empty included" three times, and "An empty
   group still takes a FIFO slot"), the `PER_OP` bullet in Steps, step 2, and Peeling's
   "Only `PER_OP` peels more than one step". Then delete this item.

Check:

- T0: only the cases above change, in their commits and waits. T0b SAME.
- Asm, the 20 production cells: confirm the claim first. Every changed
  `s_waitcnt vmcnt` should be larger (less conservative), never smaller; count them.
  Unaffected cells are identical.
- The affected outputs are bit-identical, because only waits move. The suite has no new
  failures against the latest base.
- If the loop's waits change, A/B in-process (`a8small/ab.sh`). Drain-only changes are
  unlikely to show up in timing; keep them for the simpler plan.

## Peel on the copies in a wait's window, not only its group count

Priority: low. Checked 2026-10-20 over all 35 T0 cases: at every loop wait, the window on
the first iteration holds the same copies as in steady state, so no `peeled` changes.
Two things would raise it: a config that breaks that, typically an LDS component with
`F < depth` under `PER_PANE`, or wanting the constexpr time back. The peeling replay is
skipped in 94% of the sweep's configs, and is at most `depth - 3 + async_period` steps
long instead of `3 * depth + async_period`.

The peeling scan compares each early step's wait immediate `n` (in commit groups) with
its steady twin's. The compiler waits on a `vmcnt`, though. `UpdateAsyncWaitCount` sums
the LDS-copy instructions in the `n` newest groups (the window) and takes the minimum
over every control-flow path, and a loop-body wait has a path that enters the loop from
the peeled steps and the prologue. If a window on that path holds fewer copies, because
an LDS component with `F < depth` has not started copying, every iteration gets the
smaller `vmcnt` even though `n` matches. The non-`PER_OP` schemes are the likely case:
their `n` never changes, so they never peel. On the test grids only `PER_PANE` hits it
(below).

**Upper bound.** A wait is per pane (`PER_OP`, `PER_PANE`) or per step (per-stage). At
step `x` it needs, for each LDS component `c` it DS-reads (`(x + 1) % R(c) == 0`), the
copy issued at step `x + 1 - F(c)`.

1. History. The window starts right after the newest needed copy, the one with the
   smallest `F` in the wait's set, and ends at the wait. So it lies in steps
   `x + 1 - G(x) .. x`, where `G(x)` is the largest, over the waits of step `x`, of the
   smallest `F` that wait needs. Both `n` and the window depend on the commits of these
   steps only. `G` repeats every `async_period` steps. The most history a wait depends
   on is `H = max G`, at most the largest LDS `F`, at most `depth`.
2. First steady step. A step commits like the loop step at the same phase once every
   LDS component is active (`x + F(c) >= 0`). Register components commit nothing. That
   is every step from `y0 = -F_lds` on, where `F_lds` is the smallest `F` over the LDS
   components. `F_lds >= 2`, so `y0 <= -2`. If every LDS component has `F = depth`,
   `y0 = -depth` and every step commits like the loop, as in all of today's T0 cases
   under non-`PER_OP` schemes.
3. Bound. Step `x`'s window matches the steady one if it starts at `y0` or later:
   `x + 1 - G(x) >= y0`, that is `x >= G(x) - 1 - F_lds`. Every step from `peeled_f` on
   passes:

   ```
   peeled_f = max(1, 1 + max{x >= 0 : x < G(x) - 1 - F_lds})
   ```

   Since `G(x) <= H`, `peeled_f <= max(1, H - 1 - F_lds) <= max(1, depth - 3)`.

The bound is not tight. It counts all of step `s = x + 1 - G(x)` as inside the window,
but the window starts after the needed copy's group, inside step `s`.

- Under `PER_OP` and `PER_PANE`, the groups committed before it in step `s` are outside
  the window: earlier panes, and under `PER_OP` also earlier components of its pane. A
  copy missing there does not matter.
- Under the per-stage schemes, the needed copy's group is all of step `s`, so the tight
  rule is `x >= G(x) - 2 - F_lds`. The grid never has `G(x) > F_lds + 1` there, so both
  rules give 1.

A tight closed form for the per-pane schemes needs the commit order inside a step, which
`_replay` already walks. The gap grows with `async_period`: `G(x)` peaks once per
`async_period` steps, so wrongly failing a peak step can move the last failing step by
up to `async_period`. That bound is observed, not proven.

**Tight peel factor.** Replay only the candidates below the bound:

1. If `peeled_f == 1`, then `peeled = 1`, with no peeling replay: the only candidate,
   step 0, is always peeled.
2. Otherwise, replay the prologue, the warm-up and steps
   `0 .. peeled_f + async_period - 1`, so `settled = peeled_f + async_period`. Steps
   from `peeled_f` on are steady, and windows repeat every `async_period` steps. So the
   twin of a candidate `x < peeled_f` is the step in
   `[peeled_f, peeled_f + async_period - 1]` congruent to `x` modulo `async_period`.
3. Peel through the last candidate whose `(n, window)` differs from its twin's.

Checked 2026-10-20 with a throwaway sweep over 17920 configs: 4 schemes, 7 ring-depth
sets, 5 pane shapes, both `middle`, 16 placements, 4 `R` mixes. 1120 of them have no
LDS component.

- The two-stage peel equals the exact one in all 16800 configs with an LDS component.
  The exact peel compares each step with a steady step far out.
- The replay is skipped in 15864 of them. When it runs, it reaches at most step 15.
- The bound alone peels too many steps as follows. The columns count configs per
  scheme, for `PER_OP` and `PER_PANE` each; the per-stage schemes are exact throughout.
  The uniform bound `H - 1 - F_lds` peels 1-5 steps too many in 928 configs.

  | `async_period` | configs | exact | +1 | +2 | +3 | +4 |
  |---|---|---|---|---|---|---|
  | 1 | 1960 | 1910 | 50 | - | - | - |
  | 2 | 1120 | 1080 | 20 | 20 | - | - |
  | 4 | 1120 | 1080 | 24 | 2 | 4 | 10 |

- The exact window peel exceeds today's `peeled` only under `PER_PANE`, in 414 of 4200
  configs. `PER_OP` and the per-stage schemes never exceed it.

Example: `PER_PANE`, 2x2, ring depths `(2, 4, 6, 3)`, all through LDS, both scales at
`R = 2`, so `F = (2, 7, 6, 5)`, `F_lds = 2` and `y0 = -2`. A B pane needs the B payload
(`F = 6`), plus the B scale (`F = 5`) on scale steps, so `G` alternates between 6 and 5
and `H = 6`. Steps 0, 1 and 2 fail the bound and differ from their twins, so
`peeled = 3`. Today it is 1.

Plan, about 3 hours:

1. In `_replay`, return each wait's window next to its immediate: the multiset of
   `(component_index, slot)` copies in the `n` newest groups, per pane or per step.
2. In `Schedule.__init__`, compute `F_lds`, `G(x)` over one `async_period` (from
   `ds_read_slots`, the DS-read phases and `F`) and `peeled_f`. Then run the two-stage
   peel above, and assert `peeled <= peeled_f`. This replaces the scan over
   `[0, depth - 3]` and `settled = 3 * depth + async_period`.
3. Remainder and drain waits also have a path that skips the loop, since the compiler
   does not know `iters >= 1`. Check whether those windows differ, and record the
   result here if peeling cannot fix them.
4. Tests: over the grids of `test_peeled_wait_prefix_is_minimal`
   (`test_moe_gemm_gluon_buffered_schedule.py`: commit schemes, register masks, `R`) and
   `test_independent_buffer_waits_match_copy_history` (`test_moe_gemm_gluon_config.py`:
   ring depths, placements), assert that:
   - the first-iteration window of every loop wait equals the steady one;
   - `peeled` equals the exact peel from a comparison against far steady steps;
   - `peeled <= peeled_f`.
5. Docs: PIPELINE.md's Peeling (the rule, and "Bounds", including "The code uses
   `3 * depth + async_period`") and Steps, step 2. Then delete this item.

Check:

- `tests/gate.sh <label> --fast`: T0 SAME (it prints `peeled` and the waits), T1 passes.
- `tests/gate.sh <label>`: T2d asm x20 SAME.
- A throwaway sweep over the two test grids lists the configs whose `peeled` grows. For
  one of them, the loop's `s_waitcnt vmcnt` in the asm matches the steady value only
  with the change.

## Count register buffer loads in a wait's `vmcnt`

The hardware `vmcnt` counts every outstanding vector-memory load, but
`UpdateAsyncWaitCount` counts only the direct-to-LDS ones (its header: "We ignore loads
to registers since we do not control the vectorization"). Take a wait whose needed copy
is followed by `n_lds` newer LDS-copy instructions and `n_reg` newer register
`buffer_load`s. The safe `vmcnt` is `n_lds + n_reg`, but the pass emits `n_lds`. Loads
retire in order, so the wait blocks on `n_reg` more loads than the DS read needs. This
happens in the loop, so it costs every iteration.

Affected today (T0): the production pins a8w4 T1024 and T4096 (B payload in registers,
`F = 2`), and the T0 cases `breg`, `ascale-reg`, `bscale-reg` and `bscale-reg-d2`.

Plan, about half a day to measure; the fix depends on step 2:

1. Measure: in the asm of prod-a8w4-t1024, for each loop `s_waitcnt vmcnt(N)`, count
   the LDS and register loads issued between its needed copy and the wait, and compare
   `N` with `n_lds + n_reg`. If they already match, record that here and drop the item.
2. Bound the gain: raise the loop's `vmcnt`s to the safe values by hand in the
   `.amdgcn` (`TRITON_KERNEL_OVERRIDE=1`, `TRITON_OVERRIDE_DIR`), then A/B in-process
   against the unpatched kernel (`a8small/ab.sh`, `BLOCK=40 WARM=15`). Stop here if it
   does not win beyond noise.
3. Fix, preferred first:
   - Upstream: let the pass count `amdgpu.buffer_load` to registers as well. Its
     instruction count is fixed by the lowering, the same way the pass already counts
     `BufferLoadToLocalOp`.
   - Locally: Gluon's `wait_group` takes only a group count, so a local fix needs a raw
     `vmcnt` wait. That means duplicating the lowering's instruction counts.
4. Docs: a note in PIPELINE.md's Wait plan that the immediate in groups is not the
   final `vmcnt`. Then delete this item.

Check:

- In the affected cells' asm, every changed loop `vmcnt` rises to `n_lds + n_reg` and
  none goes above it, since that would be a race. Other cells are identical.
- The affected outputs are bit-identical; the suite has no new failures.
- The in-process A/B on a8w4 T1024 and T4096 wins beyond noise.
