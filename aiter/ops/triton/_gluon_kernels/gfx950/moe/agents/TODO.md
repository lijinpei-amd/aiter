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

Done 2026-11-04 (prefill generalization of rounds 6-8): the half-block recipe runs in
every family. a4w4 and a8w8 `bm128_half` (dsv4-pro T=4096) take the 512-column
half-height tile in XCD order, one M block at a time, with `.cs` stores (a8w8: BLOCK_K
128 in that tile). The `bm64_half` class (glm52-base and minimax-m3 T=1024, a mean of
exactly 32 rows) gets a 32-row half-height tile (`bm64_half_tile`, WAVES_PER_EU 1), the
linear tile order and `.cs` stores; the tile is 512 columns wide for a8w4 only. Every
block_m 32 launch (T=1024 of dsv4-flash and dsv4-pro) streams its result in the linear
order, and a8w8 streams the result of every block_m 128 launch. HIP 7 / HIP 1, 3 paired
rounds against 946f4a07f, new/old even / random: a4w4 dsv4-pro T4096 0.796 / 0.916,
a8w8 0.847 / 0.934; the bm64 half cells 0.83-0.93 / 0.89-0.95; block_m 32 cells
0.97-1.00 / 0.97-0.99; a8w8 T4096 any 0.975-0.998 / 0.930-0.976. Output bit-identical.
Not kept, measured: a 16-row half-height tile at `bm32_half` (0.990-0.999); `.cs` on
a4w4 and a8w4 `bm128_any` (1.00-1.05); a 512-column 32-row tile for a4w4 (slower under
random routing) and a8w8 (spills 29-35 VGPRs); the XCD order at block_m 64 (slower
under random routing); per-shape `reduce_grouped` tiles (512 or 2048 columns, 1 or 4
warps, no load hint; a8w4 at every prefill shape, a4w4 and a8w8 at T=4096/16384). In the
screens, 15 variant / shape pairs, all at T=4096, read a whole call at least 2% faster
in one routing and at most 2% slower in the other. Re-timed alone in 4 fresh-process
pairs per routing, their medians are 0.985-1.022, apart from one borderline 0.980. 8 more
pairs put that one at 1.010 / 1.016, and an A/A control of the same cell spans
0.951-1.035 per pair. No load hint makes the reduce up to 1.38x slower (a8w4 glm52-base
T=4096). Table:
`bench_out/gluon_moe_regression/gemm2/gemm2_prefill_generalize.md`.

Remaining gap, largest first:

- a8w4 dsv4-pro T=4096 (even routing) is 1.050x the tuned FlyDSL stage 2 on HIP 7 at
  `db1ba831e` (kernel code unchanged since; `ca7d68ae6` only renames the reduce in
  traces) with the task's exact protocol (Gluon 1132.1 us, FlyDSL atomic 1078.1 us,
  medians of 3 alternating rounds; per round 1.054, 1.046, 1.069). At `ee0de5558`
  (same kernels): 1.042 (Gluon 1133.8 us, FlyDSL 1088.4 us). Gluon has measured
  1132-1134 us in each session since `ad8958e72`; FlyDSL 1078-1088 us. At `ad8958e72`
  (same kernels): 1.051 (Gluon 1133.5 us, FlyDSL 1078.2 us;
  `compare_a8w4pro_20261104_l5.md`, with the 72-cell HIP 7 matrix of that commit).
  `ad8958e72` streams the reduce_grouped rows (`.cg`) and widens its column block to
  1024 (HIP 7: 0.986 even, 0.990 random; balanced order 0.979 / 0.988). At
  `fded59f68` it was 1.058 (`compare_a8w4pro_20261104_l4.md`). The reduce tile is
  read from the MOE-REDUCE config family since `374b6bcad` (same kernel for the
  measured cells). Tenth pass, steady-state screen (2000 back-to-back calls, HIP 6,
  repeat spread ~0.2%; a screen only, not acceptance timing) of the 512-column tile:
  SCHED_MODE 1-4, WAIT_COMMIT_SCHEME 1/2/4, K_UNROLL 6, DS_READ_A_*_IN_MFMA,
  tiles_per_warp [2,4], B scales via LDS, SOFF_UNROLL and SCALE_FILL_MID are all
  within 0.3% of the current tile; GROUP_M 2 and SLOT_N 512 are +1%; BLOCK_N 1024
  spills 697 VGPRs. A single-process HIP 6 run first showed SCHED_MODE 1 at -2.8%; it
  did not reproduce. Eleventh pass: measured for the two-launch split only (8+6 N
  tiles, `.wt` stores, whole waves): its GEMM2 costs +11 us over the whole call (the
  column-chunk tile order, the `.wt` stores and the second launch together), and the
  split is flat on HIP 7 with equal energy per call. Thirteenth pass, in-kernel
  chunked reduce (prototype, not kept): one launch with grid [chunk 0 tiles][chunk 1
  tiles][256 reduce CTAs]; chunk 1's CTAs reduce chunk 0 after their tile, the last CTAs
  reduce chunk 1, gated by per-chunk tile counters (`.wt` stores, `s_waitcnt vmcnt(0)`,
  barrier, relaxed agent-scope atomic; `.cv` loads). Output equal to b38920e09 in both
  routings. Steady state (HIP 6, 2000 calls, 2 rounds): 1196 us vs 1041-1047 us for
  the current two launches, +150 mJ per call; 6/10 chunk-0 N tiles 1240/1167 us, 512
  reduce CTAs 1208 us. Cause: the reduce code runs at GEMM2's occupancy (512 VGPRs,
  one CTA per CU), so each reducer holds a CU through latency-bound loads. Among the
  kernels measured, the reduce is as fast as plain streaming (`l6/reduce_floor.py`,
  HIP 6): 83.1 us for 352 MB in and 59 MB out (4.95 TB/s), torch's contiguous `sum`
  over the same bytes 92.6 us, a 704 MB copy 147.3 us (4.8 TB/s); a footprint folded
  into 59 MB (MALL-resident) is only 4 us faster (`l6/reduce_footprint.py`).
  Fourteenth pass, diagnostics on HIP 6 with checked snapshots (`l6/diag14.sh`, record
  `l6/diag14/`, HEAD ca7d68ae6; they do not decide criterion 1): in one protocol run
  the GPU has no kernels for 146 ms before Gluon's first GEMM2 (Triton's JIT cache key
  for the first Gluon launches), the warmup GEMM2 calls take 970-1228 us and the 20
  timed ones average 1023.7 us (reduce 92.0 us). The same kernels give 1109-1127 us
  plain and 1080-1086 us under cProfile (3 alternating pairs), so the protocol result
  depends on host timing. A plain Triton read of 4.23 GB takes 791-878 us; GEMM2's
  bytes (4.49 GB read, 352 MB written) at the measured rates take ~907 us, against
  ~960 us steady state (rates of these kernels, not a proven limit). Standalone reduce
  experiment (`diag14/reduce_layouts.out`): token-major rows with identity indices
  77.8 us vs 82.5-83.3 us for today's gathered rows; the GEMM2 epilogue that would
  write token-major rows is not built. Fifteenth pass, two CTAs per CU (not kept;
  `l6/w2_probe.txt`, `l6/w2c_screen.txt`): at `WAVES_PER_EU` 2 both tiles must fit 256
  registers; the 512-column half tile spills (650 VGPRs), and the one spill-free form
  (128 x 128 tile in 2 passes, 64 x 256 half tile) is +10.5% at steady state (HIP 5,
  1177 vs 1061-1066 us, +120 mJ): A is read for 28 N tiles per expert instead of 14.
  The B payload loads already stream (`sc0 nt`) and A loads are cached. The
  bf16 store's row-segment length (128-512 B per instruction) makes no difference,
  nor does `global_store` in place of `buffer_store`. Twelfth pass (HIP 7 kernel trace,
  `l2/kt_fly_both_h7/`): FlyDSL's atomic kernel 1076.7 us, its row-storing kernel of the
  same tile 1124.1 us plus `moe_reduction` 92.4 us. Gluon's row-storing GEMM2 is faster
  than FlyDSL's; the gap is the deterministic combine (rows stored + reduce) against
  FlyDSL's atomic one, which costs it 47 us less than storing rows and needs no reduce. An earlier
  table that screened FlyDSL's reduce candidate too gave 1.077. Seventh pass, not kept:
  a tile-contiguous B weight copy (bit-exact, no change) and other result/weight cache
  policies (GEMM2 slower, reduce unchanged). Gluon's GEMM2 kernel alone now takes about
  as long as FlyDSL's whole stage 2; the separate reduce (~100 us) is the difference.
  Eighth pass: the deterministic chunked reduce (GEMM2 + reduce per column chunk, `.wt`
  stores, the chunk read back from MALL) is bit-exact and cuts the reduce by 30%, but
  paired on HIP 7 it is 0.995 (flat) and its energy per call equals the whole call's;
  not kept. Energy per call at the cap: GEMM2 1031 mJ, reduce 105 mJ; the intermediate
  store is ~9% of the call (not DRAM: a MALL-resident window costs the same), B HBM
  traffic ~2.4%. Stores straight from the accumulator layout are 11% slower. The reduce
  candidate (BLOCK_N 1024, 2 warps, `.cg` loads) was confirmed with balanced order on
  HIP 7 and kept in `ad8958e72` (above). Sixth pass (`compare_a8w4pro_20261104_l3.md`): the GPU
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
  part of the bit-exact design that FlyDSL's atomic epilogue does not pay. The energy
  measurement and the separate-launch chunked reduce were done in the eighth pass (above).
  The in-kernel chunked reduce was built and measured in the thirteenth pass (above):
  +15% at steady state. The text below is the state before round 8. These are
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
- Prefill cells after the generalization: the same-lock table (HIP 3 / HIP 7, 3 rounds)
  is in `gemm2_prefill_generalize.md`. Gluon/FlyDSL geomean over the prefill cells, base
  -> final: 1.120 -> 1.072 a4w4, 1.104 -> 1.067 a8w4, 1.065 -> 1.035 a8w8. For the 18
  cells still above 1.05x, one lock (HIP 1, 2 rounds) timed the GEMM2 kernel, the
  reduce and FlyDSL's stage 2:
  - At T=4096 and T=16384, the GEMM2 kernel alone is 0.89-1.005x FlyDSL, and the call
    is 1.01-1.12x. The rest is the separate bit-exact reduce (7-18% of the call).
  - At T=1024, the kernel alone is 0.94-1.11x. The largest gaps are a8w8 dsv4-flash
    1.11, a8w8 glm52-base 1.09, and a4w4 dsv4-flash and minimax-m3 1.06; the reduce
    adds 3-7%.
  - Untested hypothesis for dsv4-flash T=1024 (block_m 32, not a half launch): it runs
    the Python ladder's block_m 32 tile, which was never tuned for prefill; screen
    `bm32_any` knobs.
  - The short-K explanation for glm52-base T=1024 does not hold. In the same lock,
    minimax-m3 T=1024 (K 3072) trails at least as much in a4w4 / a8w4.
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

## Close the GEMM1 gap to FlyDSL stage 1 at few rows per expert

Priority: high. Target: a8w4 dsv4-flash T=1024 (256 experts x 24 rows, block_m 32),
Gluon/FlyDSL <= 1.00 with the same BF16 output, warm, on HIP 3.

Done 2026-11-04: the `A8W4-GEMM1` config family
(`configs/gfx950/gluon/moe/a8w4_gemm1/`), entry `bm32_any`: B in registers, shuffled
B scales without `.cg`, BLOCK_N 256, WAVES_PER_EU 2. The cause was the config: no
override matched, and the ladder tile read 1.26x the ideal HBM bytes (unshuffled B
scales with `.cg`) and held one CTA per CU. dsv4-flash T=1024: 617.9 ->
431.3 us, Gluon/FlyDSL 1.36 -> 0.950 (FlyDSL BF16 out 454.0 us). dsv4-pro T=1024:
2083.1 -> 1644.3 us. Factor table and PMC data: `docs/perf_gemm1_a8w4_bm32.md`.

Done 2026-11-04: entry `bm16_any` (T=8 and T=32 in the model matrix), with
BLOCK_N 128. In the 72-cell matrix (HIP 3, production path) the two entries change
10 a8w4 GEMM1 cells, all 16-32% faster with even and random routing; every other
cell keeps its config and its output. Table: `docs/perf_gemm1_a8w4_bm32.md`.

Remaining:

- The benchmark (`bench_moe_gemm_gluon.py::_run_one`) passes `scatter_indx` to
  GEMM1, so each timed GEMM1 call also runs a `_reduce_grouped` of 7.0-7.5 us that a
  real GEMM1 does not run. FlyDSL's stage-1 time has no such kernel.
- The Gluon grid is `routing_data.n_blocks` x grid_n, an upper bound: at this cell
  2944 of the 7040 CTAs exit at once (FlyDSL launches exactly 8192 at BLOCK_N 128).
- Other a8w4 cells the rule changed, Gluon/FlyDSL best BF16 stage 1 on HIP 3: dsv4-pro
  T=1024 1.009, glm52-base T=32 1.020, dsv4-flash T=32 1.028, minimax-m3 T=8 1.040,
  dsv4-flash T=8 1.071. FlyDSL wins with tile_m 16 at dsv4-pro T=1024 (16 rows per
  expert; the router gives block_m 32) and with a K split across waves at T=8.
- a4w4 dsv4-flash T=1024 is 1.40x FlyDSL (2026-11-03, production path). It is also
  block_m 32 and also takes the ladder; a8w8 at block_m 16/32 too.
- The older GEMM1 overrides in `_default_launch_config` (block_m 64 and 128, the
  September-shape tiles) still hold tuning values in Python. Move them into GEMM1
  families as `A8W4-GEMM1` does; check that the effective configs do not change.

## B scales from registers fault in a GEMM1 block_m 32 tile

Priority: medium. Found 2026-11-04, not fixed. GEMM1 a8w4 dsv4-flash T=1024 with
BLOCK_N 128, BLOCK_K 256, SLOT_N 64, warps (1, 4), tiles (2, 1), B preshuffled in
registers, shuffled B scales and `ALLOW_B_SCALE_VIA_LDS=False` gives
`hipErrorIllegalAddress` in `_moe_gluon_gemm1`. No production config uses it. GEMM2
block_m 128 reads B scales from registers without a fault. Reproduce with
`bench_out/gluon_moe_regression/gemm2/g1/sweep_g1.py` (variant `bscale_reg` in
`v_screen4.json`), then check the B-scale register offsets for SLOT_N 64.

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

Done 2026-11-04: a4w4 and a8w8 at block_m 128 take the tile too, from the new
`A4W4-GEMM1` and `A8W8-GEMM1` families (`bm128_any`, `bm128_half` `.cs`). The a8w8
block_m 64 config has another BLOCK_N and warp split, so its family also sets the
half-height tile's own keys (`bm128_half_tile`: the block_m 128 tile at 64 rows). The
family launch keys now apply at every block_m and dtype pair. dsv4-pro T=4096, 4
fresh-process pairs (`gluon_perftest.py`): a4w4 0.908 (even, HIP 1) and 0.935 (random,
HIP 5); a8w8 0.894 and 0.938. Output bit-identical in all 72 cells, both routings.

Remaining: the block_m 64 launches at a mean of 32 rows (glm52-base and minimax-m3
T=1024), which need a block_m 32 config with the same grid and warps. Merge with
GEMM2's half-height tile (`../aiter-gemm2-a8w4`, 8e8fd5457): same field and kernel
split; GEMM2 asserts raw A scales, GEMM1 addresses sorted A scales at `2 * pid_m`.

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
