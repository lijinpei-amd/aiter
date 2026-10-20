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
`t = s + F`, and the DS read is at step `t - 1`.

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
| DS read: pick the buffer | | | | | = | = | = | |
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
Through LDS `s` is therefore a multiple of `R`; in registers `s % R == 1`. At `D = 2`,
`R = 1` a register ring has `F = 1`: its DS read falls in its copy's own step and is placed
after the copy (see DS read in CONCEPTS.md).

A copy issued at step `s` carries K tile `s + F - 1`. That tile is DS-read at step
`t - 1 = s + F - 1`, and its MFMAs issue at step `t`. So every step DS-reads the K tile
with its own number and issues the MFMAs of the tile before, from fragments the previous
step DS-read: the whole `BLOCK_K`
tile is carried in registers (`VGPR_PREFETCH_K == BLOCK_K`, asserted), and a step's MFMAs
never wait on its own DS reads.

Depth 3, every component `F = 3`:

| step | -2 | -1 | 0 | 1 | ... | t | ... | num_k-2 | num_k-1 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| kind | prologue | prologue | warm-up | peeled | | loop / remainder | | drain | drain | final MFMA |
| copies tile | 0 | 1 | 2 | 3 | | t+2 | | - | - | - |
| DS-reads tile | - | - | 0 | 1 | | t | | num_k-2 | num_k-1 | - |
| MFMAs tile | - | - | - | 0 | | t-1 | | num_k-3 | num_k-2 | num_k-1 |

A component copies at step `s` only when both hold (`_step_plan`):

- `active`: its target tile lies in `[0, num_k)`. A component with a smaller `F` starts
  copying later in the prologue and keeps copying into the drain, through drain step
  `depth - F - 1`.
- `loads`: its ring advances, `(s + F - 1) % R == 0`. A payload copies at every active
  step, a scale once every `R`.

It DS-reads at step `t - 1` when `(t - 1) % R == 0`, so only at multiples of `R`
(`Schedule.ds_reads`).

## Steps

`_run_buffered_pipeline` and `_drain_buffered_pipeline` emit the steps; `moe_gemm.py`
then issues the final MFMA. Every emitted step has an index into the `Schedule` tables,
`STEP`, from `prologue_step`, `warmup_step`, `peeled_step`, `loop_step`,
`remainder_step` or `drain_step`. `_step` also takes `step`, the absolute step number,
which may be a runtime value and feeds the runtime LDS ring indices.

| kind | emitted | executed | does |
|---|---|---|---|
| prologue | `depth - 1` | `depth - 1` | steps `1 - depth .. -1`, copies only; inline in `_run_buffered_pipeline`, not `_step` |
| warm-up | 1 | 1 | step 0: copies and DS reads, no MFMA |
| peeled | `peeled` | `peeled` | steps `1 .. peeled`, straight-line |
| loop | `unroll` | `iters * unroll` | a `tl.range` loop whose body is `unroll` steps |
| remainder | `unroll - 1` if `static_remainder`, else 1 | `rem` | the steps the loop leaves over |
| drain | `drain_phases * (depth - 1)` | `depth - 1` | DS-reads the last tiles; shallower components finish their copies |
| final MFMA | - | 1 | `_last_mfma` or `_drain_last_fused`: the last tile's MFMAs from registers; no copy, DS read or wait |

With `main = num_k - depth`, `iters = (main - peeled) // unroll` and
`rem = (main - peeled) % unroll`. In all, `num_k + depth - 1` steps plus the final MFMA.
`validate_pipeline` requires `num_k >= depth + peeled + unroll`.

Per emitted step, `Schedule` stores its kind, `phase` (the step number its ring arithmetic
uses), unroll position `ki`, `in_loop`, and `dot` (false for prologue and warm-up).
`phase` is the absolute step number on straight-line steps and `peeled + u + 1` on loop
step `u`, which is congruent to it modulo `unroll`.

- **Unroll.** `unroll = lcm(K_UNROLL, buffer_period)`: every ring is back at buffer 0
  after one loop body, so each loop step's ring indices are compile-time.
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
   3. `mfma` region: the pane's MFMAs of the previous K tile (when `dot`), then the DS reads
      placed in the MFMA region. The last pane advances the HBM pointers.
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
  in the region `DS_READ_*_IN_MFMA` picks. A register ring at `F == 1` moves later (see
  DS read in CONCEPTS.md).
- **Fragment order.** A step appends each pane's DS-read fragments in pane order
  (`a += am + af`), and `_take_operand_pairs` indexes them by slot, so a component's
  DS-read panes must increase with its slot. Every current ownership rule satisfies this,
  but nothing asserts it: check it when changing ownership.

## Ring indices

A copy issued at step `s` fills buffer `((s + F - 1) // R) % D`, which is
`((t - 1) // R) % D`, the buffer the DS read at step `t - 1` reads.

- **LDS ring** (`_index`): compile-time when `static_lds_index` has a value (loop, static
  remainder and static drain steps), which gives immediate LDS offsets. Otherwise the
  same formula at runtime from `step`, with `gl.assume(k_tile >= 0)` so the modulo needs
  no signed fix-up.
- **Register ring**: a tuple has no runtime index. Straight-line steps rotate it
  (`_rotate_buffers`) so index 0 is the current value. Loop steps never rotate; they index
  from the loop's first step (`register_index`), which needs `unroll` to be a multiple of
  `reg_buffer_period`.

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
prologue, before components with a smaller `F` started copying, so fewer groups separate
them from the step. Those steps are peeled out of the loop instead of tightening every
iteration.

- **Rule.** Replay the prologue, the warm-up and steps `1 .. settled` with no end.
  `peeled` is the last step in `[1, depth - 2]` whose immediates differ from its
  steady twin, the first step `>= 2 * depth` with the same phase modulo `async_period`
  (`lcm(R)` over the LDS components). It is at least 1, so the first main step keeps its
  visible zero accumulator.
- **Last, not first.** The differing steps can have gaps, so a first-match rule would
  leave a differing step in the loop with a too-large immediate: a race. Example:
  `PER_OP`, 2x2, ring depths `(2, 4, 6, 3)` in `COMPONENTS` order, both scales at
  `R = 2`, B payload in registers: depth 7, step 1 matches, step 2 differs, `peeled = 2`.
- **Only `PER_OP` peels more than one step.** The other schemes commit a fixed number of
  groups per pane or step, whether a component copies or not.
- **Bounds.** From step `depth - 1` on, every needed copy was issued at a step `>= 0`,
  where all LDS components copy, so the candidates stop at `depth - 2`. `settled` must
  reach the largest twin, `2 * depth + async_period - 1`; a shorter trace compares
  against a missing twin and peels for nothing. The code uses `3 * depth + async_period`.

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

## Where to change what

| change | edit |
|---|---|
| which pane copies or DS-reads a slot | `buffer_load_pane` / `ds_read_pane` in `Schedule.__init__`; keep the fragment order |
| what a step copies, commits or needs | `_step_plan`, and the emitter's copy and commit points with it |
| step kinds, counts or phases | the `steps` table in `Schedule.__init__` and the `*_step` accessors, then `_run_buffered_pipeline` / `_drain_buffered_pipeline` |
| a commit scheme | `_step_plan`'s grouping, the commit points in `_buffer_load_pane` / `_step_live`, `WaitCommitScheme` |

Verify per REGRESSION_TESTS.md. T0 shows any change to a schedule table. T1 runs the real
`_pipeline.py` on a symbolic machine (`test_moe_gemm_gluon_buffered_schedule.py`) and
checks the emitted groups and waits against a reference
(`test_moe_gemm_gluon_wait_commit.py`); it is the only coverage of the register rings.
