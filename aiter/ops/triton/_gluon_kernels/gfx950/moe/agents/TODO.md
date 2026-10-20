# TODO

Open work in this folder. Each item says what to change, why, and how to check it.

## Tighten `settled` in the peeling replay

To pick `peeled`, `Schedule.__init__` replays the warm-up and steps `0 .. settled - 1`, then
compares each candidate step `x` in `[0, depth - 3]` with its steady twin, the first step
`>= 2 * depth - 1` congruent to `x` modulo `async_period`. The largest twin is step
`2 * depth + async_period - 2`, so `settled = 2 * depth + async_period - 1` is enough. The
code uses `3 * depth + async_period`: `depth + 1` steps of replay that no comparison reads,
paid in constexpr time at every build, and a bound nothing explains.

Plan, about 20 minutes:

1. In `Schedule.__init__`, set `settled = 2 * depth + async_period - 1`, and assert that
   every twin lies inside the replay (`steady < settled`).
2. In PIPELINE.md's Peeling "Bounds", drop "The code uses `3 * depth + async_period`".

Check:

- `peeled` and every wait are unchanged: `tests/gate.sh <label> --fast` shows T0 SAME (it
  prints `peeled` and the waits for all 20 production pins), and T1 passes.
- A throwaway sweep over the grids of `test_peeled_wait_prefix_is_minimal`
  (`test_moe_gemm_gluon_buffered_schedule.py`: commit schemes, register masks, `R`) and
  `test_independent_buffer_waits_match_copy_history` (`test_moe_gemm_gluon_config.py`:
  ring depths, placements) finds the same `peeled` under both bounds.
- `tests/gate.sh <label>`: T2d asm x20 SAME.

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
