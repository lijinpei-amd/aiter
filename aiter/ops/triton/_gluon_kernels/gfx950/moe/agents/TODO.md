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
and 1.23 in the same runs). Table (diagnostic: its source identity was not
recorded, see its header): `bench_out/gluon_moe_regression/gemm2/
compare_gemm2_20261102_r0.md`.

Done 2026-11-02 (round 1): `routing.py::routing_block_m` takes one more power of two
when the mean rows per expert is exactly 16 or 32, so those experts no longer split
into a second M block. At dsv4-pro, glm52-base and minimax-m3 T=1024, GEMM2 new/old is
0.70-0.91 (Gluon) and 0.67-0.98 (Triton); GEMM1 at the same cells is also faster
(screened on HIP 4: Gluon -12..-32%, Triton -1..-36%). Gluon/FlyDSL geomean is now
1.04, 1.10 and 1.01 (control in the same runs: 1.07, 1.12, 1.04). Table:
`bench_out/gluon_moe_regression/gemm2/compare_gemm2_20261102_r1.md`.

Done 2026-11-02 (round 2): the GEMM2 tuning is in the `A4W4-GEMM2`, `A8W4-GEMM2` and
`A8W8-GEMM2` JSON families. B is in registers (preshuffled weights) at block_m 16, 32
and 128; block_m 128 also takes the 4x2 warp tile, one wave per SIMD and per-stage
commits, with BLOCK_K 256 (a4w4) or 128 (a8w4, a8w8). block_m 64 keeps B in LDS. Cost:
the launch keeps a preshuffled copy of each GEMM2 weight on the tensor (twice the weight
memory). Gluon/FlyDSL geomean is now 1.013, 1.015 and 0.978 (control in the same runs:
1.030, 1.098, 1.012). Table: `bench_out/gluon_moe_regression/gemm2/
compare_gemm2_20261102_r2.md`.

Done 2026-11-03 (round 3): a `bm16_small` entry (small grid, T=1 here) gives a4w4 and
a8w4 a 128-column N tile; the block_m 128 entries read B scales from registers (a4w4,
a8w4), address unrolled steps through soffset (a4w4) and read A scales in the MFMA
region (a8w8); `routing_block_m` also raises an exact mean of 64 (dsv4-pro T=4096 now
runs block_m 128: GEMM2 Gluon -2.5..-12.5%, Triton -2..-8%, GEMM1 faster in a screen).
Gluon/FlyDSL geomean is now 1.0014, 1.0019 and 0.9739 (control in the same runs:
1.0229, 1.0250, 0.9847). Table: `bench_out/gluon_moe_regression/gemm2/
compare_gemm2_20261103_r3.md`.

Done 2026-11-03 (round 4): when the experts average two or more M blocks (T=16384 in
the matrix) a `bm<block_m>_multi` entry streams the result (`.cs`) and reads the
weights with the default cache policy (Gluon -4.8..+0.1%); block_m 64 holds B in
registers like block_m 128. Final acceptance against dac620634 (four HIP 7 pairs, 14
triggered cells re-timed alone): Gluon/FlyDSL geomean 0.9918, 0.9965 and 0.9673, no
non-exempt cell above 1.10 (a8w4 dsv4-flash T=16384 1.095; T=1 cells above 1.10 have
gaps below 5 us, at most 3.85 us in the matrix and 3.98 us re-timed alone), output
bit-identical in 72/72 cells. The two isolated Triton T=1 cells at 1.022 are noise
(balanced A/A treatment, identical compiled Triton kernels; round 5). Table and
protocol: `bench_out/gluon_moe_regression/gemm2/compare_gemm2_20261103_r4.md`.

Tried 2026-11-03 (rounds 3-5), not kept: the in-kernel last-producer combine (the CTA
whose counter add is the last for a token sums its rows in gate order). It is
bit-identical to `reduce_grouped` and was qualified in round 5 (33 tests, a padding-row
race found and fixed, 72/72 cells exact with it forced on), but it is slower in every
fair measurement: fused/plain 1.02-1.11 at topk 4 / block_m 64, including minimax-m3
T=1024 (a8w4 1.071, a8w8 1.051); the round 3-4 replays of other cells were already
slower or flat although they favoured the fused path. Those replays' "gains" (0.95 at
minimax-m3 T=1024) were an artifact: the fused path spent ~6 ms of host time
per launch on a layout check, the GPU sat idle, and an idle GPU runs the next kernels
about 10% faster. Compare kernel times only between paths with the same host cost per
launch. The register-B tiles hold one CTA per CU, so each CTA's tail runs exposed.
Design, numbers and the final patch: `bench_out/gluon_moe_regression/gemm2/r3/
T7_RESULT.md`, `r5/t7_combine_final.patch`.

Done 2026-11-04 (round 6): a8w4 dsv4-pro T=4096 (a mean of exactly 64 rows per
expert, so the routing gives block_m 128) runs every M block of at most 64 rows on a
second, half-height tile, the block_m 64 config (`HALF_M_TILE`, CONCEPTS.md). Only
launches whose experts average at most half an M block get the second path: in every
block_m 128 launch it changed the schedule of the main loop and cost 1-5% where it never
ran. On HIP 4 (3 paired rounds), GEMM2 + reduce 1530 -> 1311 us (even routing) and
1529 -> 1421 us (random), output bit-identical. Table: `bench_out/gluon_moe_regression/
gemm2/compare_a8w4pro_20261104_c1.md`.

Done 2026-11-04 (round 7): a launch whose experts average at most half an M block takes
the `bm<block_m>_half` entry. For a8w4 it runs the linear tile order (`TILE_SCHED` 0)
and streams the result (`.cs`); the a8w4 block_m 64 tile also streams the result. On
HIP 7 (3 paired rounds) a8w4 dsv4-pro T=4096 is 1323 -> 1255 us (even) and 1464 ->
1407 us (random); a8w4 glm52-base and minimax-m3 T=1024 0.95-0.99. Output
bit-identical in 72/72 cells. Gluon / tuned FlyDSL at that cell is 1.16 on HIP 7.
Table: `bench_out/gluon_moe_regression/gemm2/compare_a8w4pro_20261104_c2.md`.

Done 2026-11-04 (round 8): the half-height tile of a `_half` launch takes the
`bm<block_m>_half_tile` entry, and its `BLOCK_N` can be a whole multiple of the launch's;
the 128-row tile then covers the wider N block in passes (`f8bebc5c7`). For a8w4 the
tile is 512 columns wide, which halves its A reads and CTAs, and the `bm128_half` launch
walks one M block at a time on each XCD (`d65b654c8`). On HIP 7 (3 paired rounds) a8w4
dsv4-pro T=4096 is 1255 -> 1141 us (even) and 1403 -> 1388 us (random). Output
bit-identical in 72/72 cells. Table:
`bench_out/gluon_moe_regression/gemm2/compare_a8w4pro_20261104_l3.md`.

Remaining gap, largest first:

- a8w4 dsv4-pro T=4096 (even routing) is 1.058x the tuned FlyDSL stage 2 on HIP 7 at
  `fded59f68` with the task's exact protocol (Gluon 1142.5 us, FlyDSL atomic 1079.6 us,
  medians of 3 alternating rounds; per round 1.050, 1.059, 1.062;
  `compare_a8w4pro_20261104_l4.md`, which also has the 72-cell HIP 7 matrix). An earlier
  table that screened FlyDSL's reduce candidate too gave 1.077. Seventh pass, not kept:
  a tile-contiguous B weight copy (bit-exact, no change) and other result/weight cache
  policies (GEMM2 slower, reduce unchanged). Gluon's GEMM2 kernel alone now takes about
  as long as FlyDSL's whole stage 2; the separate reduce (~100 us) is the difference.
  Sixth pass (`compare_a8w4pro_20261104_l3.md`): the GPU
  runs at its 1000 W cap during these calls (GFX clock ~1.1 GHz of 2.2), so a call's time
  follows its energy, and removing stalls alone does not help: a persistent kernel that
  issues the next tile's prologue before the store is bit-exact and uses 4% fewer cycles
  on a traced CU but is 2.5-4% slower. Both backends are timed inside a power transient
  (a spike at calls 4-6, then ~30 calls to settle); Gluon's protocol window sits in the
  spike. Diagnostics: no store at all -88 us, but a store into a MALL-resident 50 MB
  window costs the same as the real store, so the store cost is not HBM traffic. Not
  kept: 7 column chunks (the protocol gain is the GPU cooling between launches; steady
  state +2.5%, random +8%), chunk orders and layouts in one launch, reduce launch shapes
  (within 1%), deeper B rings, 8 warps. The separate reduce (~100 us) is the largest
  part of the bit-exact design that FlyDSL's atomic epilogue does not pay. Next
  experiment: read the amd-smi energy counter per call for the GEMM2 and the reduce
  separately, then run the reduce of each column chunk inside the persistent GEMM2
  (per-XCD tile counters exist in `l2/po3_persistent_overlap_prototype.patch`), gated by
  a per-chunk done counter, so the chunk's intermediate is read back while it is in
  MALL. The text below is the state before round 8. These are
  results of the variants tested so far, not a proven limit. In them, most of the gap
  sits in the deterministic epilogue: removing the intermediate store takes the GEMM2
  kernel from ~1096 to 956 us (HIP 2), and the separate reduce adds ~100 us. A
  memory-only probe of the same traffic (256 persistent programs, no MFMA, HIP 3) takes
  811 us for B + A + B scales and ~995 us with the 352 MB bf16 store. FlyDSL instead adds
  into the 59 MB output with atomics, which the bit-exact rule excludes. Measured and not
  kept (screens on HIP 2/3, the HIP 7 ones marked): XCD-local tile order (-0.5%), a
  persistent loop (HIP 7: -0.9% even, +1.4% random), 8 warps at 2 waves/SIMD (HIP 7:
  +0.9% even, +4.1% random,
  +3-6% on the a8w4 block_m 64 cells), streaming hints in `_reduce_grouped` (reduce
  -11 us in the pipeline, GEMM2 +7 us; HIP 7: 1.002/0.992), a 2-way column split so
  each intermediate half stays in MALL (reduce 2x36 us, net -0.8% with `.wt` stores),
  and the pipeline knobs (SCHED_MODE, WAIT_COMMIT_SCHEME, DS_READ_IN_MFMA, K_UNROLL,
  B_NUM_BUFFER, SCALE_SLOT_K, SOFF_UNROLL: all within +-1% or slower). Output sha256
  equal to the base was checked for the persistent loop, the 8-warp, XCD-order,
  SOFF_UNROLL and SCALE_SLOT_K configs and the column split; the reduce hints were
  checked bit-exact standalone; the env knob sweep was timing only. Diagnostic only, not
  a candidate: a tile-contiguous B copy read with a timing hack that loads the wrong
  bytes (-1.6% on HIP 2), so it says nothing about a correct implementation. Third pass:
  the round-5 last-producer in-kernel combine (bit-exact, on its own tree 6e9ee75f5)
  measured 1.163x (even) and 1.193x (random) against the separate reduce at this cell
  (HIP 3); a half-height tile at occupancy 3 spills 8-147 VGPRs in every tested layout
  (occupancy 2 without spills: -2% with half tiles only, but the full tile does not fit);
  8 warps with compiler warp pipelining spills; overlapping one column half's reduce with
  the next half's GEMM2 on a second stream is slower than running them in sequence.
  Diagnostic attribution (wrong-data hacks, HIP 3): B scales ~16 us, A rows ~31 us of the
  GEMM2. Not built: the fixed-role ordered combine (slot 5 waits for slots 0-4); it keeps
  the publication costs round 5 measured as dominant. Fourth pass: FlyDSL's own tuned
  tile with its non-atomic `reduce` epilogue (rows stored, then `moe_reduction`) takes
  1215.6 us against 1092.3 us atomic on HIP 7 (3 alternating rounds); Gluon (1244.9 us
  in the same rounds) is 1.024x that deterministic FlyDSL path and 1.140x the atomic one.
  Diagnostic only: fly_tune.py validates that path by cosine >= 0.999, not bit identity,
  and it does not bound a bit-exact Gluon kernel; criterion 1 stays the atomic kernel.
  Implemented and measured, not kept: a dynamic persistent GEMM2 for the half class (256
  CTAs fetch tiles from a self-resetting device counter, no padded CTAs; bit-exact; HIP 3
  GEMM2 -25 us even, HIP 7 paired 1.000 even / 1.007 random) and half-height tiles only
  (bit-exact; slower: the doubled grid's empty CTAs cost ~60 us at one CTA per CU).
  Diagnostics: twice the MFMA work adds 160-260 us (the MFMAs are mostly exposed at one
  wave per SIMD); launching only the real tiles saves ~20 us on HIP 3. Fifth pass: the
  one-A-fragment pipeline (DS-read each A slot in the last pane that multiplies it, in
  its MFMA region) is bit-exact, cuts the half-height tile's AGPRs 73 -> 44 and brings an
  occupancy-3 tile to 1-2 spilled VGPRs (`TRITON_HIP_AGPR_ALLOC=64`), but HIP 7 paired
  3 rounds give 1.009 even / 1.011 random in the half class (not kept); the occupancy-3
  tile was no faster on HIP 3 either. HIP 3 screens have not transferred to HIP 7 three
  times (8 warps, dynamic persistent, one A fragment): screen on HIP 7 directly. The
  2-way column split with `.wt` stores (each half's intermediate read back from MALL) is
  bit-exact and flat on HIP 7: 1.000 even, 1.006 random (3 alternating rounds). No
  remaining candidate has a positive HIP 7 measurement. Data, probes and the per-variant
  tables:
  `bench_out/gluon_moe_regression/gemm2/compare_a8w4pro_20261104_l2.md`.
- a8w4 dsv4-flash T=16384 is 1.095x FlyDSL, close to the 1.10 limit. The GEMM (~1245
  us) is faster than FlyDSL's whole stage 2 (1355 us, a tuned atomic kernel), but the
  separate deterministic reduce adds ~237 us at ~4.6 TB/s. No reduce launch shape, a
  token-major intermediate (-4% of the reduce) or a constexpr split-K count moves it,
  and no GEMM knob on top of the cache policy helps (dsas, SCHED_MODE 4, SOFF_UNROLL,
  SCALE_FILL_MID, B scales in registers: flat to +1.8%).
- T=1 (up to 1.28, gaps 1.2-3.9 us): the Gluon GEMM alone is at or below FlyDSL's stage
  2; the gap is the ~2.5 us `reduce_grouped` launch (index load, row loads, store).
- The full matrix in one process moves some cells by up to 6% on both backends (weight
  placement), and T=1 cells drift together. A T=1 GEMM2 kernel can sit in a slower
  state there: a8w8 dsv4-flash T=1 is 12.9 or 13.9 us in the matrix (GEMM2 kernel
  10.4 vs 11.4 us) and 12.8-13.1 us alone in every pair. Isolated runs of the same
  kernels agree within 2%, so measure suspect cells alone.

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
