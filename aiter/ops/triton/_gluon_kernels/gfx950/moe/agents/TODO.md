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

Remaining gap, largest first:

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

## Close the GEMM1 gap to FlyDSL stage 1 at an exact mean of 64 rows

Priority: high. Target: a8w4 dsv4-pro T=4096 GEMM1 (N 6144, K 7168, E 384, top-k 6),
Gluon / FlyDSL best stage 1 <= 1.00 on HIP 2, even routing (`N_ACTIVE=max`).
Diagnosis: `bench_out/gluon_moe_regression/gemm2/gemm1_a8w4_dsv4pro_t4096.md`.

Done 2026-11-04: the gap was block_m. The mean is exactly 64 rows per expert, so
`routing_block_m` gives block_m 128 and, with even routing, every M tile is half
empty. FlyDSL's best kernels use a 64-row tile. A GEMM1 launch whose experts average
at most half an M block now runs each M block of at most 64 rows on a half-height tile,
the block_m 64 config (`HALF_M_TILE`, CONCEPTS.md); only the `A8W4-GEMM1` family's
`bm128_any` entry asks for it. GEMM1 kernel on HIP 2: 2504 -> 1977 us (even
routing), 2513 -> 2321 us (random routing); `gluon_perftest` in 4 paired rounds: 0.796
(even), 0.925 (random).
Output bit-identical in all 72 cells, both routings. Forcing block_m 64 in the routing
was -24% (even) but +2% (random) for Gluon and +5.5% (random) for Triton.

Done 2026-11-04: the family's `bm128_half` entry streams the result (`.cs`) of a
half-height-tile launch; both tiles use it. GEMM1 kernel 1985 -> 1870 us (even), about
-0.5% (random). Target cell, 4 paired rounds: 0.765 (even), 0.915 (random). Against
FlyDSL's best bf16 stage 1, 5 alternating rounds on HIP 2: 0.992-1.001, median 0.998.

Remaining margin, largest first:

- The margin is inside the noise. FlyDSL's best bf16 time mostly reads 1997-2012 us,
  but 2 of 15 readings on HIP 2 were 1937-1939 us; there Gluon is 1.03x.
- The protocol call adds two kernels that FlyDSL's stage-1 time does not have: the
  sorted A-scale pre-pass (`moe_sort_scales`, about 30 us; FlyDSL sorts in its quant
  step) and a top-k `reduce_grouped` (39 us with the streamed result). The second is
  an artifact of `bench_moe_gemm_gluon.py`: it passes `scatter_indx` to GEMM1 too, and
  no model does that (see `bench_moe_gemm_a8w4_cudagraph.py`). The GEMM kernel alone is
  6-9% faster than FlyDSL's (1870 vs 1983-2060 us, rocprofv3, same session).
- One wave per SIMD. The block_m 128 tile needs 452 VGPRs, and the half-height tile in
  the same kernel inherits that occupancy. Alone, the block_m 64 tile at two waves per
  SIMD (B in registers, BLOCK_K 128, 192 VGPRs) is 3% faster. Tried and dropped: a
  second launch for that tile (half-tile-only, then full-tile-only): -1.3% even, +27%
  random (the full-tile-only launch is badly balanced, and an empty 9216-CTA launch
  costs 19-28 us); the block_m 128 tile at two waves per SIMD (all six variants spill
  316-576 B, 1.8-2.6x slower); 8-warp tiles (block_m 64 at 3354 us); raw A scales
  (+60 us, more than the pre-pass); `.wt` instead of `.cs` (flat).
- CTA start: about 2.1 us of each 57 us CTA has no copy in flight: the routing scalar
  chain, then about 166 address instructions (A, B and scale offsets) after an 88-cycle
  gather wait and before the first B copy. Issuing B copies before the A address math
  can recover only part of that 3.7%; it changes the code of every kernel.
- Cross-tile prefetch (a persistent CTA that issues the next tile's copies during the
  drain and epilogue) would hide the CTA start and end at one wave per SIMD. It is a
  new pipeline feature; not tried.

Extend the half-height tile, each its own change: a4w4 dsv4-pro T=4096 (its block_m 64
and 128 configs share BLOCK_N, warps and waves; with `AITER_TRITON_MOE_GLUON_HALF_M_TILE=1`
the output is bit-identical in the test, and a HIP 6 screen of the kernel gave 2088 ->
1880 us even and 2079 -> 1996 us random), and the block_m 64 launches at a mean of 32
rows (glm52-base and minimax-m3 T=1024), which need a block_m 32 config with the same
grid and warps. Merge with GEMM2's half-height tile (`../aiter-gemm2-a8w4`, 8e8fd5457):
same field and kernel split; GEMM2 asserts raw A scales, GEMM1 addresses sorted A
scales at `2 * pid_m`.

Check: `gluon_perftest.py STAGE=1` paired against the base, both routings, and the
output sha256 of all 72 cells (`gemm2/g1pro/out_hash2.py`).

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
