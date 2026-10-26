# Pipeline

The pipelined K loop is split in two files. `_schedule.py` plans it:
`Schedule(tuning_cfg, layout, K)` evaluates every per-step decision once, at build time,
into tables keyed by emitted step, a `Step`. `_pipeline.py` emits it: its `@gluon.jit`
functions read those tables through constexpr lookups and decide nothing themselves.

Step, pane, slot, component, ring, `D`, `R`, `Q` and DS read are defined in CONCEPTS.md.

## The life of a K tile

Two per-component distances, in steps, set every timing below:

- **Live span** `L = (D - 1) * R + 1` (`buffer_live_span`): how many steps one value of
  the ring lives. The pipeline depth (`Schedule.depth`) is the largest `L` over the
  present components.
- **Load-to-MFMA distance** `F` (`Schedule.load_to_mfma_distance`): how many steps after
  a copy is issued its MFMAs issue. `F = L` through LDS; `F = L - 1` in registers, which
  issue no `ds_read`. `D >= 2` (`validate_buffer_counts`), so `F >= 2` through LDS and
  `F >= 1` in registers.

Throughout, `s` is the step that issues a copy and `t` the step that issues its MFMAs, so
`t = s + F`, and an LDS component's DS read is at step `t - 1`.

Both cases for a scale at `D = 3`, `R = 2` (so `L = 5`). One value covers two K tiles, and
its MFMAs issue at steps `t` and `t + 1` either way. `x` marks an instruction, `=` a step
in which that row's storage holds the value:

| step | t-5 | t-4 | t-3 | t-2 | t-1 | t | t+1 | t+2 |
|---|---|---|---|---|---|---|---|---|
| **Through LDS, `F = L = 5`, `s = t - 5`** | | | | | | | | |
| `buffer_load` the value -> LDS | x | | | | | | | |
| LDS buffer holds it | = | = | = | = | = | | | |
| `ds_read` -> registers, which hold it | | | | | = | = | = | |
| MFMA of its two K tiles | | | | | | x | x | |
| `buffer_load` the value `D` later -> same buffer | | | | | | | x | |
| **In registers, `F = L - 1 = 4`, `s = t - 4`** | | | | | | | | |
| `buffer_load` the value -> registers | | x | | | | | | |
| register buffer holds it | | = | = | = | = | = | = | |
| MFMA of its two K tiles, from the buffer | | | | | | x | x | |
| `buffer_load` the value `D` later -> same buffer | | | | | | | | x |

A value is copied `L - 1` steps before the first instruction that reads its buffer. Through
LDS that is the `ds_read` at step `s + L - 1`, and the MFMA follows at `t = s + L`, so
`F = L`. In registers nothing is copied out, and the first reader is the MFMA itself, at
`t = s + L - 1`, so `F = L - 1`.

A component copies one value every `R` steps and its `D` buffers rotate, so the next copy
into the same buffer comes at `s + D * R`: `t + 1` through LDS, `t + 2` in registers. It
has to come after the buffer's last reader, and the readers hold the buffer for different
lengths. Through LDS the last reader is the `ds_read` at `t - 1`, so the buffer sits idle
for `R - 1` steps (step `t` here). In registers every MFMA of the value reads the buffer,
the last at `t + R - 1`, so the refill comes on the next step and the buffer never idles.
A value's first MFMA step `t` is a multiple of `R`, so through LDS `s % R == R - 1` and in
registers `s` is a multiple of `R`. The copy in a
step never fills the buffer that step's MFMAs read, since `F = (D - 1) * R` puts them
`D - 1` buffers apart; at `D = 2`, `R = 1` (`F = 1`) a copy is read by the next step's
MFMA.

## Steps

A step is numbered by the K tile its MFMAs consume: step `x` issues the MFMAs of K tile
`x`, DS-reads tile `x + 1`, and copies tile `x + F(c)` of each component `c`. A scale value
covers `R(c)` K tiles, from a multiple of `R(c)`. Before peeling and unrolling, the
pipelined K loop is:

```
def step(x):
    for each component c:
        k = x + F(c)
        if 0 <= k < num_k and k % R(c) == 0:   # active and loads in _step_plan
            buffer_load K tile k of c
    if 0 <= x + 1 < num_k:
        ds_read K tile x + 1                   # LDS components with (x + 1) % R(c) == 0
    if x >= 0:
        mfma K tile x                          # a register component: from its ring

for x in -depth .. num_k - 1:
    step(x)
```

Unrolling and then peeling turn this loop into the emitted program; Step kinds lists
the parts they leave.

### Unroll

`step(x)` depends on `x` through its rings:

- the LDS buffer it copies into and DS-reads from, `((x + F) // R) % D`;
- the register buffer it copies into and its MFMA reads, a tuple index that must be
  compile-time;
- for a scale with `R > 1`, whether it copies and DS-reads at this step, and which K
  tile of the value its MFMA takes, `x % R`.

All of these repeat every `buffer_period = lcm(D * R)` steps over the present
components. The unroll factor is the smallest multiple of `buffer_period` that is also a
multiple of the `K_UNROLL` knob: `unroll = lcm(K_UNROLL, buffer_period)`. Body step `u`
runs at `x = -depth + i * unroll + u`, congruent to `-depth + u` modulo every ring
period, so its rings fold to constants, which also gives immediate LDS offsets. The trip
count depends on the runtime `num_k`, so a remainder runs the steps that do not fill a
body:

```
n = num_k + depth
iters, rem = n // unroll, n % unroll
for i in 0 .. iters - 1:                       # a runtime loop
    for u in 0 .. unroll - 1:                  # unrolled: rings depend on u only
        step(-depth + i * unroll + u)
for u in 0 .. rem - 1:                         # remainder
    step(-depth + iters * unroll + u)
```

The guards in `step` still test `x`, which now depends on the runtime `i`.

### Peeling

Peeling moves steps off the front and the back of the loop, as straight-line code,
until every guard holds in every iteration. Given `1 <= F(c) <= depth`:

- Front: the MFMA's `x >= 0` and each copy's `x + F(c) >= 0` hold from step 0 on. The
  first copy, tile 0 by a component with `F = depth`, is at step `-depth`: `depth` steps,
  `-depth .. -1`.
- Back: every copy's `x + F(c) < num_k` holds through step `main - 1`, where
  `main = num_k - depth`, the last copy of a component with `F = depth`. The DS read's
  `x + 1 < num_k` holds through step `num_k - 2`. So `depth` steps, `main .. num_k - 1`.

Two more reasons peel the first main steps, `peeled >= 1` of them:

- Step 0's MFMA accumulates into zeros. Out of the loop, the compiler sees the zero
  accumulator.
- An early step's wait can differ from its loop step's steady wait. In the loop it
  would set the `vmcnt` of every iteration (Wait plan).

How many steps the second reason peels:

1. **First steady step: `-min(F)`.** From it on every component copies, so each step
   issues and commits what a loop step of its phase does.
2. **History: `reach(x)` steps.** Step `x`'s wait counts groups and loads from its
   newest needed copy, issued at `x + 1 - reach(x)`. `reach(x)` is the smallest `F` it
   DS-reads, the largest over the step's waits (0 if none).
3. **Bound.** The wait is steady once its history is: `x + 1 - reach(x) >= -min(F)`,
   or `x >= reach(x) - 1 - min(F)`. Peeling `bound` steps, one past the last that
   fails, makes every loop wait steady. `reach(x) <= max(F)`, so
   `1 <= bound <= max(1, max(F) - min(F) - 1)`, at most `max(1, depth - 2)`.

`bound` only caps the peel: a step below it can still match its steady twin. A replay
finds the exact `peeled <= bound`:

- **Rule.** Compare each wait's immediate and window (per pane, or per step under the
  per-stage schemes) with its steady twin's, a later step congruent to it modulo
  `copy_period`, `lcm(R)` over the present components. Register copies count: they are
  in the window. `peeled` is one past the last step that differs, and at least 1, so
  the first MFMA keeps its visible zero accumulator.
- **Two stages.** At `bound = 1` there is no replay: `peeled = 1`. Otherwise
  `_compute_wait_plan(..., windows=True)` runs the prologue and steps
  `0 .. bound + copy_period - 1`, and each step below `bound` is compared with its twin in
  `[bound, bound + copy_period - 1]`. A symbolic sweep of 17920 configs (4 schemes,
  7 ring-depth sets, 5 pane shapes, both `SCALE_FILL_MID`, 16 placements, 4 `R` mixes)
  has 16800 with an LDS component. In those the replay is skipped 77% of the time and
  never goes past step 15, and the peel equals the one found by comparing every step
  with a steady step far out.
- **Last, not first.** The differing steps can have gaps, so a first-match rule would
  leave a differing step in the loop. Example: `PER_OP`, 2x2, ring depths `(2, 4, 6, 3)`
  in `COMPONENTS` order, both scales at `R = 2`, B payload in registers: depth 7, step 0
  matches, step 1 differs, `peeled = 2`.
- **Any scheme can peel more than one step.** Example: `PER_PANE`, 2x2, ring depths
  `(2, 4, 6, 3)`, all through LDS, both scales at `R = 2`, so `F = (2, 7, 6, 5)`: every
  immediate matches its twin's, but the windows of steps 0, 1 and 2 hold fewer copies,
  so `peeled = 3`. Under the per-stage schemes, a register ring with a small `F` does
  the same.

In all, `depth + peeled` steps come off the front and `depth` off the back. The loop
keeps steps `peeled .. main - 1`, unrolled as in Unroll:

```
for x in -depth .. peeled - 1:                 # front: straight-line
    step(x)
iters, rem = (main - peeled) // unroll, (main - peeled) % unroll
for i in 0 .. iters - 1:                       # no guards; rings depend on u only
    for u in 0 .. unroll - 1:
        step(peeled + i * unroll + u)
for u in 0 .. rem - 1:                         # remainder
    step(peeled + iters * unroll + u)
for x in main .. num_k - 1:                    # back: straight-line
    step(x)
```

A front step has a compile-time `x`, so its guards and rings fold. A back step `main + j`
still depends on the runtime `num_k`, but its guards do not: `x + F < num_k` is
`j < depth - F`, and the others hold. Its rings depend on `main % buffer_period`; see
Drain under Step kinds.

## Step kinds

The DS read's guard `x + 1 < num_k` splits the back once more: only step `num_k - 1`
fails it, and issues only its MFMAs. Each kind is a range of steps. `executed` counts the
steps that run; `emitted` counts the step bodies in the code, each one entry in the
`Schedule` tables:

| kind | steps `x` | executed | emitted | `buffer_load` | `ds_read` | MFMA | emitted as |
|---|---|---|---|---|---|---|---|
| prologue | `-depth .. -1` | `depth` | `depth` | components with `x >= -F` | at `x = -1` | - | see Prologue below |
| peeled | `0 .. peeled - 1` | `peeled` | `peeled` | all | yes | yes | straight-line `_step`s |
| loop | `peeled .. peeled + iters * unroll - 1` | `iters * unroll` | `unroll` | all | yes | yes | a `tl.range` loop whose body is `unroll` steps |
| remainder | `peeled + iters * unroll .. main - 1` | `rem` | `unroll - 1` if `static_remainder`, else 1 | all | yes | yes | see Remainder below |
| drain | `main .. num_k - 2` | `depth - 1` | `drain_phases * (depth - 1)` | components with `x < num_k - F` | yes | yes | `_drain_buffered_pipeline`; see Drain below |
| final MFMA | `num_k - 1` | 1 | - | - | - | yes | `_last_mfma` or `_drain_last_fused`; no wait |

- Prologue step `-1` copies for every component, since `x + F >= 0` at `F >= 1`, and is
  the only prologue step that passes the DS read's `x + 1 >= 0`: it DS-reads K tile 0.
- The prologue is the `depth`-step lead of an `F = depth` copy over its MFMA; the drain
  and the final MFMA are the same `depth` steps at the back. In all, `num_k + depth`
  steps, the last one the final MFMA.
- `depth` is the largest `L`, not the largest `F`: if the deepest ring is in registers
  and no component reaches `F = depth`, prologue step `-depth` copies nothing.

All together, with the `Schedule` accessor that indexes each kind:

```
for x in -depth .. -1:                         # prologue: prologue_step(x)
    step(x)
for x in 0 .. peeled - 1:                      # peeled: peeled_step(x)
    step(x)
for i in 0 .. iters - 1:                       # loop: a runtime tl.range loop
    for u in 0 .. unroll - 1:                  #   whose body is unrolled: loop_step(u)
        step(peeled + i * unroll + u)
for u in 0 .. rem - 1:                         # remainder: remainder_step(u)
    step(peeled + iters * unroll + u)
for j in 0 .. depth - 2:                       # drain: drain_step(p, j)
    step(main + j)
step(num_k - 1)                                # final MFMA
```

`validate_pipeline` requires `num_k >= depth + peeled + unroll`, so the loop body runs at
least once.

These sketches show the steps that execute. `_run_buffered_pipeline` and
`_drain_buffered_pipeline` emit them, and `moe_gemm.py` then issues the final MFMA. Each
emitted step is a `Step(kind, index)`, `STEP`, from the accessor in its comment: a
`StepKind` and the body's place among its kind's emitted bodies, from 0 in emission
order. Prologue step `x` is `x + depth`, peeled step `x` is `x`, loop and remainder step
`u` is `u`, and drain step `(p, j)` is `p * (depth - 1) + j`, so no index depends on
`num_k`. `_step` also takes `step`, the step number `x`, which may be a runtime value.
Its LDS ring indices count from `ds_read_tile = step + 1`, the tile it DS-reads.

Per emitted step, `Schedule` stores its `stage`, `phase` (the step number its ring
arithmetic uses) and unroll position `ki`. `in_loop`, `dot` (false in the prologue) and
`is_drain` read the step's kind:

- `stage` sets which components copy: `x` on prologue and peeled steps, `j` on drain
  step `main + j`, none (every component) on loop and remainder steps.
- `phase` is `x` on prologue, peeled and drain steps; `peeled + u` on loop step `u` and
  static-remainder step `u`, congruent to `x` modulo `unroll`; 0 on the dynamic
  remainder.
- `ki` is `u` on loop and static-remainder steps, `p + j - peeled` on static-drain step
  `(p, j)`, so `peeled + ki` is congruent to `x` modulo `buffer_period` in the phase
  `p` that runs, and 0 elsewhere.
- `static_lds_index` requires a loop, static-remainder or static-drain step.

- **Prologue.** Steps `-depth .. -2` only copy, so `_run_buffered_pipeline` issues their
  copies inline, not through `_step`. Step `-1` also waits and DS-reads K tile 0: it is
  one `_step`.
- **Remainder.** With `UNROLL_EPILOGUE` (`static_remainder`): `unroll - 1` straight-line
  steps, each behind a runtime `if`, with compile-time LDS indices. Without it: one step
  body in a runtime loop at phase 0, with runtime indices, which needs every scale at
  `R == 1` (`validate`).
- **Drain.** `static_drain` (`buffer_period <= 3`, `reg_buffer_period == 1`, no output
  quant) emits one copy of the drain per value of `main % buffer_period`, picked by a
  runtime `if`, with compile-time indices. Otherwise one copy with runtime indices.

## The pane walk

`_step_live` emits one step as a walk over the panes in pane-index order (N outer,
M inner):

1. The tile wait, `wait(STEP)`, under a per-stage commit scheme; then the `SCHED_MODE`
   hint.
2. For each pane:
   1. The pane wait, `wait(STEP, pane_index)`, under `PER_OP` and `PER_PANE`.
   2. `mem` region: the pane's DS reads placed in the memory region, then its copies
      (`issued`) in `COMPONENTS` order, with their commits.
   3. `mfma` region: the pane's MFMAs of K tile `x` (when `dot`), then the DS reads placed
      in the MFMA region. The last pane advances the HBM pointers.
   4. `commit` region: the `PER_STAGE_WHOLE` commit, after the last pane, when the step
      copied through LDS.
3. A `sched_barrier` after a loop step that holds a register ring (not under the warp
   pipeline). Straight-line steps rotate the register rings.

Prologue step `-1`, the one `_step` without MFMAs, also issues a `sched_barrier` before
its first pane's `mem` region.

The regions are `TritonAMDGPUWarpPipeline` stages only in loop steps with
`WARP_PIPELINE=1`; elsewhere `pick_warp_pipeline_stage` makes them no-ops. Waits sit
outside every region because the pass rejects a wait inside a stage.

## Ownership: which pane copies and DS-reads each slot

Resolved once in `Schedule.__init__`:

- **Buffer-load pane** (`buffer_load_pane`). Payload: at 2x2, `B0, A0, A1, B1` over panes
  0-3; with both axes split, `A0, B0, A1, B1, ...` over pane indices; with one axis
  unsplit, slot `i` of both operands at pane `i`. A scale follows its payload, except
  that `SCALE_FILL_MID` at 2x2 moves slot 0 of each scale to pane 1 and slot 1 to
  pane 2; the flag does nothing at other shapes. A scale slot spanning `Q > 1` slots is copied by its first slot only.
- **Validity** (asserted in `Schedule.__init__`): every slot maps to an existing pane,
  and no pane copies two slots of one component. Different components may share a pane.
- **DS-read pane** (`ds_read_pane`): the pane that copies the same operand's payload slot,
  in the region `DS_READ_*_IN_MFMA` picks. Only LDS components have one; a register
  component's MFMA reads its ring directly.
- **Fragment order.** A step appends each pane's DS-read fragments in pane order
  (`a += am + af`), and `_take_operand_pairs` indexes them by slot, so a component's
  DS-read panes must increase with its slot. Every current ownership rule satisfies this,
  but nothing asserts it: check it when changing ownership.

## Ring indices

A copy issued at step `s` fills buffer `((s + F) // R) % D`, which is `(t // R) % D`, the
buffer of K tile `t`: the `ds_read` at step `t - 1` reads it through LDS, and the MFMA at
step `t` in registers.

- **LDS ring** (`_index`): compile-time when `static_lds_index` has a value (loop, static
  remainder and static drain steps), which gives immediate LDS offsets. Otherwise the
  same formula at runtime from `ds_read_tile = step + 1`, with `gl.assume(k_tile >= 0)`
  so the modulo needs no signed fix-up. `_step_live` adds the 1 once, up front: adding
  it inside `_index` changed the drain's codegen.
- **Register ring**: a tuple has no runtime index. Straight-line steps rotate it
  (`_rotate_buffers`) so index 0 holds the value of K tile `x + 1`, the next step's MFMA
  operand. Loop steps never rotate; they index from the loop's first step, as if at step
  `peeled`, which needs `unroll` to be a multiple of `reg_buffer_period`.
  `register_index(step, component)` is the buffer a step's MFMAs read and
  `register_index(step, component, True)` the one its copy fills; `final_register_index`
  is the buffer the final MFMA reads.

## Commit plan

Each LDS copy belongs to one commit group. Register copies are not asynchronous and
commit nothing. `_step_plan` returns the groups each pane closes (`commits`), and the
emitter closes exactly those: every commit point is gated on `closes_group`. A group
holds at least one copy, so a pane or step that copies nothing through LDS commits
nothing.

| `WAIT_COMMIT_SCHEME` | commit point | groups per step | waits |
|---|---|---|---|
| `PER_OP` (1) | after each LDS copy | one per LDS copy | one per pane |
| `PER_PANE` (2) | after each pane's copies | one per pane that copies through LDS | one per pane |
| `PER_STAGE_WHOLE` (3) | after the last pane's MFMA region, in its own `commit` region | one, if the step copies through LDS | one per step |
| `PER_STAGE_WARP_PIPELINE` (4) | end of the last pane's `mem` region | one, if the step copies through LDS | one per step |

In the inline prologue steps (before `-1`), both per-stage schemes commit after the last
pane's copies (`_buffer_load_pane`, `MARK_TILE`). Under `PER_OP` a pane's groups follow
`COMPONENTS` order, the order `_buffer_load_pane` copies in; reorder both together.

An LDS component copies only at steps `-F <= x < num_k - F`, and a scale with `R > 1`
only one step in `R`. So a step can copy nothing through LDS anywhere: drain steps once
every LDS component has stopped, prologue steps before or between the first LDS copies
(a deep scale at `R = 4`), and, with both payloads in registers, every step off the LDS
scales' phase, loop steps included. Dropping their groups shortened drain immediates but
moved no `vmcnt`: an empty group repeats the previous mark (Wait plan).

## Wait plan

`wait_group(n)` blocks until at most `n` groups are outstanding, retiring the oldest
first. For a DS read whose newest needed copy is in group `j` of the `count` groups
committed so far, the tightest safe immediate is `n = count - j - 1`. It stays safe when
group `j` has already retired, because the outstanding groups are always the newest.

**From immediate to `vmcnt`.** On gfx950, Triton lowers a commit to
`llvm.amdgcn.asyncmark` and `wait_group(n)` to `llvm.amdgcn.wait.asyncmark(n)`, and
LLVM's `SIInsertWaitcnts` computes the `s_waitcnt vmcnt`; Triton's
`UpdateAsyncWaitCount` skips its own count on this target. Each mark remembers the
newest LDS copy issued before it. The wait leaves in flight every load issued after the
`(n + 1)`-th newest mark's copy, the last LDS copy of group `j`: that is the wait's
**window**. It holds register `buffer_load`s too, since `vmcnt` retires every vector
memory load in order. The schedule takes the issue order as emitted (panes in order,
`COMPONENTS` order in a pane); loads never cross a mark, but LLVM's scheduler can move
them between two marks, so a window is a model of the `vmcnt`, not its exact value. Example, prod a8w4 T1024: each loop step issues 6 LDS copies as 12
`buffer_load ... lds` instructions and 2 register B-payload copies as 8 `buffer_load`s,
and its `wait_group(1)` becomes `vmcnt(20)`. `vmcnt` counts instructions; the schedule
compares windows as copies, which is enough because a copy issues the same instructions
at every step.

- Where paths join (the loop header, after a remainder `if`), LLVM lines the marks up
  newest first and keeps the smaller window, so one emitted wait gets the smallest
  `vmcnt` over the paths into it. Peeling depends on this. The remainder `if`s also
  merge remainder counts this `K` never runs (TODO.md).
- An empty group would repeat the previous mark: it would count in the immediate but
  change no `vmcnt` on a straight path. The schedule commits none (Commit plan).
- A join keeps at most the 16 newest marks. With exactly 16 held, LLVM clamps a larger
  immediate to 15, which waits for more than needed. But a commit after a join that
  dropped marks leaves 17 or more, and a wait with an immediate at least that large
  then gets no `s_waitcnt vmcnt` at all, although its group may still be in flight
  (TODO.md). Production immediates are at most 11.

`_compute_wait_plan` computes every immediate:

- **Demand.** A pane (`PER_OP`, `PER_PANE`) or a step (per-stage schemes) needs, for each
  LDS component it DS-reads at this phase (step `t - 1`), the copy issued at `s = t - F`. A
  register ring needs nothing. No demand, no wait (`None`).
- **Execution order.** The replay walks the commit plan in execution order for this `K`:
  prologue, peeled steps, every loop iteration, the remainder, the epilogue's commit
  groups, the drain. It asserts that each needed copy was committed earlier, and that no
  copy is committed twice.
- **One immediate per emitted step.** A loop step runs many times and keeps the smallest
  immediate over its executions.
- **Emitted, not executed.** Remainder steps this `K` does not run take their immediates
  from a second replay with the longest remainder: static-remainder steps past `rem`,
  and the dynamic remainder step when `rem == 0`. The drain copies for the other phases
  borrow the executed phase's immediates, which are exact for them too (Precondition).
- **Epilogue groups.** `_stage_epilogue_inputs` commits `epilogue_groups` groups (bias
  and gammas staged in LDS) between the main loop and the drain. The replay counts them
  there, so they lengthen only the drain waits whose newest needed copy is older. After
  the drain, `wait_group(epilogue_wait())` and a barrier retire them.
- **Precondition.** The replay uses `num_k_tiles(K)` with `K` constexpr, while the loop
  bound `NUM_K` is a runtime argument. The host passes `num_k_tiles(K)`, but any `NUM_K`
  that `validate` accepts, at least `depth + peeled + unroll` and a multiple of every
  `R`, gets exact immediates too. Loop steps are steady from `peeled` on, and remainder
  steps this `K` skips take the longest replay's. With `NUM_K % R == 0`, a drain step's
  copies, DS reads and the history its waits reach back into depend only on `j`, so
  every drain phase's borrowed immediates are its own. Between such `K` the tables
  differ only in the drain rows' `phase`, which the emitter uses modulo `R`. Checked
  2026-10-21: the symbolic emitter (`_execute` in `test_moe_gemm_gluon_buffered_schedule`)
  ran a schedule built for one `num_k` at every other valid `num_k` over two unroll
  periods, for 26 configs under all 4 schemes, and every wait was exact.

## Emission invariants

Changes to `_pipeline.py` keep these. The codegen gate catches most of them; it cannot
see the last two, which only the symbolic tests cover.

- Waits outside every region.
- `_buffer_load_pane` copies in `COMPONENTS` order, the `PER_OP` group order.
- DS-read fragments appended in pane order (`a += am + af`).
- `_rotate_buffers` only on straight-line steps. Rotating inside the loop needs guarded
  tuple updates and raises register pressure.
- `_init_buffers` allocates register rings only for components in registers; a spill
  multiplies the kernel time.
- `gl.assume(main >= peeled + unroll)` in `_run_buffered_pipeline` and
  `gl.assume(k_tile >= 0)` in `_index`.
- Both `sched_barrier` sites with their conditions; the symbolic tests pin the stream.
- `SOFF_UNROLL`: loop copies take a constant `soff`, and `_advance` moves the pointers
  once per loop body, at its last step. One without the other advances twice. It is off
  by default (a measured regression).
- `_step = _step_live`: the symbolic tests replace `_step`.
