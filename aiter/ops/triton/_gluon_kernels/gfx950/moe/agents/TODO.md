# TODO

Open work in this folder. Each item says what to change, why, and how to check it.

## Tighten the LDS ring for scales with `R > 1`

Through LDS, a buffer is free once its `ds_read` has run, at step `t - 1`, but the next
copy into it comes at `s + D * R = t + R - 1`, `R` steps later (`F = L = (D - 1) * R + 1`).
The buffer idles `R - 1` of every `D * R` steps: in PIPELINE.md's `D = 3`, `R = 2` table it
is free at `t` and refilled at `t + 1`.

Issuing the copy on the step after the `ds_read` gives `F = D * R` through LDS, `R - 1`
more steps of prefetch from the same LDS. LDS copies would then issue at multiples of `R`,
as register copies already do. At `R = 1` nothing changes.

- Change: `load_to_mfma_distance` for LDS components in `Schedule.__init__`, and the
  pipeline depth, which must cover the new `F`; peeling and waits follow from both. The
  test oracles in `test_moe_gemm_gluon_config.py` and
  `test_moe_gemm_gluon_buffered_schedule.py` derive `F` on their own and need the same
  change, as do the `F` definition and the table in PIPELINE.md.
- Cost: when such a scale sets the depth, the depth grows by up to `R - 1`, so the
  prologue and drain get longer and the minimum K rises.
- Check: codegen changes only for builds with an LDS scale at `R > 1`; T0 lists them.
  Then the correctness suite, and a cold A/B on the affected cells.
