# Pipeline

The pipelined K loop is split in two files. `_schedule.py` plans it:
`Schedule(tuning_cfg, layout, K)` evaluates every per-step decision once, at build time,
into tables indexed by an emitted step index. `_pipeline.py` emits it: its `@gluon.jit`
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

Three transformations turn this loop into the emitted program.

**1. Split by kind.** Every condition in `step` depends on `x` alone, and each stays
fixed over one of five ranges of `x`. Split at those bounds, each part knows at compile
time what its steps issue:

```
for x in -depth .. -2:                         # prologue: buffer_load only
    step(x)
step(-1)                                       # warm-up: buffer_load and ds_read
for x in 0 .. main - 1:                        # main: all three, every component
    step(x)
for j in 0 .. depth - 2:                       # drain: ds_read and mfma; buffer_load only
    step(main + j)                             #   by components with j < depth - F(c)
step(num_k - 1)                                # final MFMA: mfma only
```

The ranges, given `1 <= F(c) <= depth`:

| kind | steps `x` | count | `buffer_load` | `ds_read` | MFMA |
|---|---|---|---|---|---|
| prologue | `-depth .. -2` | `depth - 1` | components with `x >= -F` | - | - |
| warm-up | `-1` | 1 | all | yes | - |
| main | `0 .. main - 1` | `main = num_k - depth` | all | yes | yes |
| drain | `main .. num_k - 2` | `depth - 1` | components with `x < num_k - F` | yes | yes |
| final MFMA | `num_k - 1` | 1 | - | - | yes |

- The prologue starts at the first copy: tile 0, by a component with `F = depth`, at
  step `-depth`.
- Main ends at that component's last copy, of tile `num_k - 1`, at step `main - 1`. A
  component with a smaller `F` keeps copying through drain step `depth - F - 1`.
- Prologue and drain both have `depth - 1` steps, the lead of an `F = depth` copy over
  its `ds_read`. In all, `num_k + depth` steps, the last one the final MFMA.
- `depth` is the largest `L`, not the largest `F`: if the deepest ring is in registers
  and no component reaches `F = depth`, prologue step `-depth` copies nothing.

Only main has a runtime trip count. The other parts have at most `depth - 1` steps and
are emitted straight-line. The drain's steps `main + j` still depend on the runtime
`num_k`; see Drain below.

**2. Unroll main.** A main step still depends on `x` through its rings:

- the LDS buffer it copies into and DS-reads from, `((x + F) // R) % D`;
- the register buffer it copies into and its MFMA reads, a tuple index that must be
  compile-time;
- for a scale with `R > 1`, whether it copies and DS-reads at this step, and which K
  tile of the value its MFMA takes, `x % R`.

All of these repeat every `buffer_period = lcm(D * R)` steps over the present
components. Unrolled by `unroll = lcm(K_UNROLL, buffer_period)`, they depend on `u` alone
and fold to constants, which also gives immediate LDS offsets:

```
iters, rem = main // unroll, main % unroll
for i in 0 .. iters - 1:                       # a runtime loop
    for u in 0 .. unroll - 1:                  # unrolled: rings depend on u only
        step(i * unroll + u)
for u in 0 .. rem - 1:                         # remainder
    step(iters * unroll + u)
```

`main` depends on the runtime `num_k`, so the remainder runs the `rem` steps that do not
fill a loop body.

**3. Peel the first main steps.** Two things set them apart:

- Step 0's MFMA accumulates into zeros. Out of the loop, the compiler sees the zero
  accumulator.
- Under `PER_OP`, an early step can need a smaller wait immediate than later iterations
  of its loop step (see Peeling). A loop step takes the smallest immediate over its
  iterations, so in the loop it would tighten the wait of every iteration.

So `peeled >= 1` main steps run straight-line before the loop:

```
for x in 0 .. peeled - 1:                      # peeled: straight-line
    step(x)
iters, rem = (main - peeled) // unroll, (main - peeled) % unroll
for i in 0 .. iters - 1:
    for u in 0 .. unroll - 1:
        step(peeled + i * unroll + u)
for u in 0 .. rem - 1:
    step(peeled + iters * unroll + u)
```

All together, with the `Schedule` accessor that indexes each part:

```
for x in -depth .. -2:                         # prologue: prologue_step(x)    
    step(x)
step(-1)                                       # warm-up: warmup_step()
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
emitted step has an index into the `Schedule` tables, `STEP`, from the accessor in its
comment. `_step` also takes `step`, the step number `x`, which may be a runtime value.
Its LDS ring indices count from `ds_read_tile = step + 1`, the tile it DS-reads.

| kind | emitted | executed | emitted as |
|---|---|---|---|
| prologue | `depth - 1` | `depth - 1` | copies inline in `_run_buffered_pipeline`, not `_step` |
| warm-up | 1 | 1 | one `_step` |
| peeled | `peeled` | `peeled` | straight-line `_step`s |
| loop | `unroll` | `iters * unroll` | a `tl.range` loop whose body is `unroll` steps |
| remainder | `unroll - 1` if `static_remainder`, else 1 | `rem` | see Remainder below |
| drain | `drain_phases * (depth - 1)` | `depth - 1` | `_drain_buffered_pipeline`; see Drain below |
| final MFMA | - | 1 | `_last_mfma` or `_drain_last_fused`; no copy, DS read or wait |

Per emitted step, `Schedule` stores its kind, `phase` (the step number its ring
arithmetic uses), unroll position `ki`, `in_loop`, and `dot` (false for prologue and
warm-up). `phase` is `x` on straight-line steps and `peeled + u` on loop step `u`, which
is congruent to it modulo `unroll`.

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
   4. `commit` region: the `PER_STAGE_WHOLE` commit, after the last pane.
3. A `sched_barrier` after a loop step that holds a register ring (not under the warp
   pipeline). Straight-line steps rotate the register rings.

The warm-up step also issues a `sched_barrier` before its first pane's `mem` region.

The regions are `TritonAMDGPUWarpPipeline` stages only in loop steps with
`WARP_PIPELINE=1`; elsewhere `pick_warp_pipeline_stage` makes them no-ops. Waits sit
outside every region because the pass rejects a wait inside a stage.

## Ownership: which pane copies and DS-reads each slot

Resolved once in `Schedule.__init__`:

- **Buffer-load pane** (`buffer_load_pane`). Payload: at 2x2, `B0, A0, A1, B1` over panes
  0-3; with both axes split, `A0, B0, A1, B1, ...` over pane indices; with one axis
  unsplit, slot `i` of both operands at pane `i`. A scale follows its payload, except
  that `SCALE_FILL_MID` at 2x2 moves both scales to panes 1 and 2; the flag does nothing
  at other shapes. A scale slot spanning `Q > 1` slots is copied by its first slot only.
- **Validity** (`buffer_load_panes_valid`, asserted by `validate_pipeline`): every slot
  maps to an existing pane, and no pane copies two slots of one component. Different
  components may share a pane.
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
emitter closes exactly those.

| `WAIT_COMMIT_SCHEME` | commit point | groups per step | waits |
|---|---|---|---|
| `PER_OP` (1) | after each LDS copy | one per LDS copy | one per pane |
| `PER_PANE` (2) | after each pane's copies | one per pane, empty included | one per pane |
| `PER_STAGE_WHOLE` (3) | after the last pane's MFMA region, in its own `commit` region | one, empty included | one per step |
| `PER_STAGE_WARP_PIPELINE` (4) | end of the last pane's `mem` region | one, empty included | one per step |

In the prologue, both per-stage schemes commit after the last pane's copies
(`_buffer_load_pane`, `MARK_TILE`). An empty group still takes a FIFO slot. Under
`PER_OP` a pane's groups follow `COMPONENTS` order, the order `_buffer_load_pane` copies
in; reorder both together.

## Wait plan

`wait_group(n)` blocks until at most `n` groups are outstanding, retiring the oldest
first. For a DS read whose newest needed copy is in group `j` of the `count` groups
committed so far, the tightest safe immediate is `n = count - j - 1`. It stays safe when
group `j` has already retired, because the outstanding groups are always the newest.

`_replay` computes every immediate:

- **Demand.** A pane (`PER_OP`, `PER_PANE`) or a step (per-stage schemes) needs, for each
  LDS component it DS-reads at this phase (step `t - 1`), the copy issued at `s = t - F`. A
  register ring needs nothing. No demand, no wait (`None`).
- **Execution order.** The replay walks the commit plan in execution order for this `K`:
  prologue, warm-up, peeled steps, every loop iteration, the remainder, the epilogue's
  commit groups, the drain. It asserts that each needed copy was committed earlier, and
  that no copy is committed twice.
- **One immediate per emitted step.** A loop step runs many times and keeps the smallest
  immediate over its executions.
- **Emitted, not executed.** Static-remainder steps past `rem` take their immediates from
  a second replay with the longest remainder. The drain copies for the other phases
  borrow the executed phase's immediates.
- **Epilogue groups.** `_stage_epilogue_inputs` commits `epilogue_groups` groups (bias
  and gammas staged in LDS) between the main loop and the drain. The replay counts them
  there, so they lengthen only the drain waits whose newest needed copy is older. After
  the drain, `wait_group(epilogue_wait())` and a barrier retire them.
- **Precondition.** The replay uses `num_k_tiles(K)` with `K` constexpr, while the loop
  bound `NUM_K` is a runtime argument. The host passes `num_k_tiles(K)`; any other
  `NUM_K` can run a drain phase on borrowed immediates.

## Peeling

The loop body's immediates are the minimum over all its iterations. Under `PER_OP`, an
early step can need a smaller one than the loop: its needed copies were issued in the
prologue, before components with a smaller `F` started copying (see Steps), so fewer
groups separate them from the step. Those steps are peeled out of the loop instead of
tightening every iteration.

- **Rule.** Replay the prologue, the warm-up and steps `0 .. settled - 1` with no end.
  The peeled steps run through the last step in `[0, depth - 3]` whose immediates differ
  from its steady twin, the first step `>= 2 * depth - 1` congruent to it modulo
  `async_period` (`lcm(R)` over the LDS components). Step 0 is always peeled, so the
  first MFMA keeps its visible zero accumulator.
- **Last, not first.** The differing steps can have gaps, so a first-match rule would
  leave a differing step in the loop with a too-large immediate: a race. Example:
  `PER_OP`, 2x2, ring depths `(2, 4, 6, 3)` in `COMPONENTS` order, both scales at
  `R = 2`, B payload in registers: depth 7, step 0 matches, step 1 differs, `peeled = 2`.
- **Only `PER_OP` peels more than one step.** The other schemes commit a fixed number of
  groups per pane or step, whether a component copies or not.
- **Bounds.** From step `depth - 2` on, every needed copy was issued at the warm-up or
  later, where all LDS components copy, so the candidates stop at `depth - 3`. The replay
  must reach the largest twin, step `2 * depth + async_period - 2`, so
  `settled >= 2 * depth + async_period - 1`; a shorter trace compares against a missing
  twin and peels for nothing. The code uses `3 * depth + async_period`.

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
