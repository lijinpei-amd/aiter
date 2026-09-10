# gfx950 Gluon MoE implementation review findings

- Date: 2026-09-10
- Reviewed source commit: `f42ad9715406816ea626572f4a3eb12a33052aab`
- Scope: implementation files in `aiter/ops/triton/_gluon_kernels/gfx950/moe/`
  only

This report records findings for later triage; it does not implement the suggested
changes. No correctness or determinism failure was confirmed for the currently
validated production configurations. Reachability matters: several findings require
an explicit custom tuning, malformed low-level entry arguments, or unusually large
tensors. Those are called out separately from issues reachable through an otherwise
valid supported launch.

## Summary

| Category | Count | Main themes |
|---|---:|---|
| High impact | 3 | Routing-ID overflow, unsafe accepted `k_width`, buffer-offset window |
| Medium/configuration hardening | 12 | Runtime launch metadata, validation, scale shapes, routing trust, extreme quantization |
| Performance and measurement | 10 | Cache policy, scale traffic, empty work, runtime grid arithmetic, scheduling |
| Maintainability and testing | 3 | Frozen-path scope, duplicated schedule models, implicit synchronization |

The labels rank impact and practical reachability together. H1 and H3 can affect
otherwise structurally valid launches at extreme sizes. H2 requires an explicit
custom tuning override, but the accepted setting is documented in source as silently
wrong. Most M findings are defensive hardening for private/direct entry calls or the
custom tuning surface.

## High-impact findings

### H1. Packed routing block IDs become negative above the signed 16-bit range

**Reachability:** otherwise normal routing when one expert receives at least 32,768
M-blocks; malformed routing metadata reaches it sooner.

**Evidence:**

- `_types.py:376-387` documents each int32 map entry as
  `(block_id << 16) | expert_id`.
- `moe_gemm.py:518-525` and `_frozen.py:1692-1699` recover `block_id` with a
  signed arithmetic right shift.
- For `block_id >= 32768`, the decoded value is negative. `_offsets.py:60-71`
  tests only `offs < M_e`, so negative gathered rows remain live. Output paths such
  as `_epilogue.py:445-450` and `_epilogue.py:538-558` likewise omit a lower-bound
  row check.
- Expert IDs above 65,535 are truncated by the low-half mask.

**Impact:** reads can address before A/gamma storage and stores can address before the
expert's result base. Multiple wrapped blocks can overlap, creating races and
nondeterministic corruption. The per-expert row threshold is about 1.05M, 2.10M, or
4.19M rows for `BLOCK_M` 32, 64, or 128.

**Suggested fix:** use an unsigned logical shift when decoding, enforce both packed
field limits, and check `0 <= block_id * BLOCK_M + row < M_e`. Prefer separate int32
fields or an int64 representation if the 16-bit limits are not contractual.

### H2. A user-supplied `k_width` can select a combination already known to be wrong

**Reachability:** explicit custom tuning. Default configurations use automatic
selection.

**Evidence:** `_layout.py:475-490` says that `k_width=8` with a 16x16x128
instruction compiles and runs but computes an incorrect result. The method still
returns every non-`None` override unchanged, and `_layout.py:555-560` passes it into
`DotOperandLayout`. The validation block at `_layout.py:1149-1317` has no whitelist
for the override.

**Impact:** an accepted tuning can silently return wrong numerical results rather
than failing at construction or compile time.

**Suggested fix:** remove the override or whitelist proven
`(operand storage type, MFMA shape, k_width)` tuples. Require a correctness test for
every accepted tuple.

### H3. Multi-byte operand/result offsets can exceed the hardware buffer-operation window

**Reachability:** very large but otherwise structurally valid tensors. Current common
model sizes are well below the boundary; the normal caller's size guard was inspected
only to establish that two-byte tensors are not fully excluded.

**Evidence:**

- `moe_gemm.py:533-546` explicitly documents the roughly 2 GiB vector-offset
  window and folds the expert dimension of B into a 64-bit scalar base. Within that
  expert, however, plain and preshuffled B retain the N-plane origin in vector offsets
  at `_offsets.py:125-130` and `_layout.py:201-207`; raw and shuffled B scales do the
  same at `_offsets.py:282-288` and `_layout.py:253-255`.
- Gathered A offsets retain a global token row multiplied by `a.stride_m` in
  `_offsets.py:60-71` and `_offsets.py:95-98`; raw A-scale offsets retain the same
  kind of global-row vector term in `_offsets.py:210-225`.
- Result bases fold `start_m` into a 64-bit scalar pointer at
  `moe_gemm.py:655-659`, but expert-local block rows remain vector offsets in
  `_epilogue.py:445-450` and `_epilogue.py:538-558`. The quantized scale stores use
  the same pattern.
- The implementation itself has no byte-span check for these vector offsets.

**Impact:** once a byte offset crosses the buffer-instruction range, it can wrap or
truncate and access the wrong rows. BF16/FP16 payloads reach a byte limit at half the
element count of byte-stored FP8/FP4 data.

**Suggested fix:** fold each CTA's row/K/N origin into 64-bit scalar bases for A, B,
their scales, Y, and Y scales, leaving only tile-local vector offsets. At the calling
boundary, validate every addressed tensor in bytes, including access width and all
payload/scale strides.

## Medium-severity and configuration-hardening findings

### M1. Runtime grid metadata is trusted by the device body

**Reachability:** malformed private/direct entry calls. Supported callers derive the
launch extent and `grid_n` together.

**Evidence:** `moe_gemm.py:502-518` divides/modulos by runtime `grid_n` and loads
`rt.expt_block_pid_map + pid_m` without proving `grid_n > 0`, `pid_m < grid_m`, or
that the N-grid equals `N / BLOCK_N`. Excess N work reaches unmasked B and bias
addresses in `_offsets.py:104-133` and `_epilogue.py:127-137`; result stores assume
the same exact grid at `_epilogue.py:438-450`.

**Impact:** invalid metadata can divide by zero, omit output, read beyond routing or
weight/bias buffers, or store beyond the intended N extent.

**Suggested fix:** keep one authoritative launch specification and validate
`grid_m`, `grid_n`, routing-map length, and `grid_n == N / BLOCK_N` before launch.

### M2. Runtime `NUM_K` is not checked against the compile-time K extent

**Reachability:** malformed private/direct entry calls. Supported callers pass the
full `K / BLOCK_K` count.

**Evidence:** `NUM_K` remains unspecialized in `_entry.py:105-108` and
`_entry.py:224-227`. `_pipeline.py:184-195` validates a count derived from
compile-time K, while `_pipeline.py:833-849` drives the runtime loop from the
independent value and introduces `gl.assume(main >= ...)`. The frozen driver has the
same split at `_frozen.py:1336-1347` and `_frozen.py:1478-1490`.

**Impact:** a short count can violate pipeline prologue/drain assumptions; a long
count advances payload and scale pointers beyond the compile-time extent.

**Suggested fix:** if prefix-K execution is intended, enforce
`pipeline_min <= NUM_K <= K / BLOCK_K`; otherwise require equality. Do not use
`gl.assume` as input validation.

### M3. `B_SCALE_SHUFFLED` lacks the shape validation its fixed physical tile needs

**Reachability:** malformed or experimental custom tuning. Current supported callers
sanitize the flag and shape.

**Evidence:**

- `_layout.py:493-509` sends every shuffled scale through LDS.
- `_layout.py:803-820` represents it as `[nonk // 32, 256]`, which requires whole
  32-row stripes and implements one K256 physical run.
- The A sorted/shuffled branch has a `sorted_shuffled_ok()` assertion at
  `_offsets.py:168-178`; the B branch at `_offsets.py:259-277` has no equivalent.
- `_layout.py:1338-1344` provides `make_scale_swizzle_check`, but nothing in this
  directory calls it, and that helper checks only a lower K bound.

**Impact:** `MINI_BLOCK_N < 32` creates a zero-height allocation, nonmultiples of 32
truncate rows, and `BLOCK_K > 256` can underallocate or copy only part of a stage.

**Suggested fix:** validate B shuffled scales in `validate_layout`: positive non-K
extent divisible by 32 and only the explicitly implemented K128/K256 cases, including
the existing packed-K128 pairing rules.

### M4. Safety validation is disabled by optimized Python

**Reachability:** an optimized Python process plus an invalid custom/environmental
configuration. Valid generated configs are unaffected.

**Evidence:** `require_constexpr` uses a Python `assert` at `_lang.py:215-228`.
Configuration, shape, and resource checks use ordinary asserts throughout
`_config.py:516-592` and `_layout.py:1149-1334`; the pipeline lower-bound check does
the same at `_pipeline.py:184-195`. `moe_gemm.py:499-500` wraps only the helpers'
returned booleans in `gl.static_assert`, so `python -O` removes the substantive
checks before that wrapper sees them.

**Impact:** invalid buffer counts, layouts, K bounds, and enum values can reach code
that assumes they were rejected, resulting in trace failures, out-of-bounds accesses,
or wrong code.

**Suggested fix:** replace safety checks with explicit non-optimizable exceptions or
a Gluon validation primitive. Run validation tests once with `PYTHONOPTIMIZE=1`.

### M5. Fused MXFP4 output disagrees with its stated reference at extreme FP32 values

**Reachability:** the normal fused-output path if an epilogue produces values near the
top of FP32, including infinity; ordinary model ranges are unlikely to do so.

**Evidence:** `_quant.py:35-50` states that the upper E8M0 clamp cannot bind. In both
paths, `_quant.py:76-83` and `_quant.py:105-110` masks the rounded exponent to eight
bits and clamps only the lower bound. A finite exponent-254 value whose mantissa
rounding carries, max-finite FP32, or infinity yields `e_biased == 255`, hence scale
byte 253. The stated reference first forms infinity and clamps the unbiased exponent
to 127, yielding byte 254.

**Impact:** the stored scale and applied inverse scale differ by a factor of two for
these groups. NaN behavior is also not covered by the existing equivalence claim.

**Suggested fix:** implement explicit reference-compatible upper saturation and add
tests for exponent carry, max finite, infinity, and NaN groups.

### M6. The low-level entry ABI carries multiple unchecked sources of truth

**Reachability:** stale or malformed direct calls. Supported wrappers currently build
the duplicated fields coherently.

**Evidence:** `_entry.py:139-220` and `_entry.py:258-339` reconstruct pointer-backed
tensor records alongside independent `CFG_FUNC`, `CFG_N`, and `CFG_K` values.
`dtype_quant`, hidden dimensions, top-k, expert count, result dimensions, and scale
swizzles are not cross-validated. The result pointer's element type controls the cast
at `_epilogue.py:360-363`, while `func_cfg.output_quant` independently selects the
storage path at `_epilogue.py:439-453`.

**Impact:** stale metadata can select a layout or instruction inconsistent with the
actual buffers, reinterpret storage, or address beyond an allocation.

**Suggested fix:** establish one authoritative source for each property or validate
all duplicates before launch. If these kernels remain private implementation details,
document that the wrapper is the only supported ABI.

### M7. Routing metadata contents and cardinalities are trusted

**Reachability:** malformed direct calls. The ordinary routing builder is expected to
produce coherent values.

**Evidence:**

- `A_TOPK` is accepted at `_entry.py:141` and `_entry.py:260` and stored in the
  token record, but row decoding instead divides by the independent
  `rt.n_expts_act` at `_offsets.py:63-68`.
- `a.num_token`, `b.num_expert`, and `res.out_dim` are carried by `_types.py` but are
  not used to bound accesses in the kernel body.
- `moe_gemm.py:518-525` immediately uses decoded expert/block IDs for histograms,
  offsets, weights, scales, and output bases. `_offsets.py:60-71` accepts loaded
  gather entries without checking the resulting token row.
- Duplicate map entries are not rejected.

**Impact:** zero or stale routing cardinality can divide by zero or map gates to wrong
tokens; invalid expert/gather entries can access unrelated storage; duplicate output
tiles can race and become nondeterministic.

**Suggested fix:** validate expert IDs, block bounds, gather-row bounds, sentinel
encoding, positive cardinality, and uniqueness when constructing the launch. Remove
dead routing metadata or make it authoritative.

### M8. Tile-scheduling parameters are only partially validated

**Reachability:** custom tuning.

**Evidence:** valid modes are defined in `_types.py:77-80`, but unknown `TILE_SCHED`
values fall through to LINEAR behavior in `moe_gemm.py:503-516`. Positivity and
mode-specific consistency of `GROUP_M` and `NUM_XCDS` are not checked in
`_config.py:516-592`.

**Impact:** invalid values can silently select another schedule or cause division and
modulo by zero in scheduling helpers.

**Suggested fix:** validate the tile-schedule enum and require positive,
mode-appropriate `GROUP_M` and `NUM_XCDS`.

### M9. Source-level LDS/VGPR validation is only an estimate

**Reachability:** custom tuning near architectural limits.

**Evidence:** `_layout.py:1126-1142` counts logical payload/scale elements, while
physical padding is introduced by layouts such as `_layout.py:115-172`. Epilogue and
compiler scratch are represented by a fixed empirical 4 KiB reserve at
`_layout.py:41-50`. `_layout.py:1144-1146` and `_layout.py:1320-1334` count the
accumulator but not all simultaneously live VGPRs.

**Impact:** a source-valid configuration can still exceed LDS limits, fail at launch,
or spill enough registers to regress badly.

**Suggested fix:** account for physical padding and maximum live scratch where
possible. Treat source VGPR estimates as advisory and gate promoted configurations on
compiled resource metadata.

### M10. Top-level validation and selected pipeline support do not fully agree

**Reachability:** custom tuning and benchmarking experiments.

**Evidence:** `_config.py:543-575` admits `WarpPipeline.MANUAL`, and when warp
pipelining is disabled it also admits zero or partial `VGPR_PREFETCH_K`. Entry-body
selection is controlled independently by `FROZEN_STEP` at `_entry.py:23-74`. The live
driver rejects MANUAL and requires a full-stage prefetch at `_pipeline.py:833-843`;
the frozen driver also requires full-stage prefetch at `_frozen.py:1478-1484` and
ignores several live tuning knobs.

**Impact:** configurations pass initial validation and then fail later during tracing,
or carry labels/knobs that do not describe the selected implementation.

**Suggested fix:** validate against the selected body. Route MANUAL only to an
implementation that uses it, reject unsupported prefetch modes early, and reject or
strip knobs ignored by the frozen path.

### M11. Quantization and format enums are not exhaustively validated

**Reachability:** custom or stale specifications.

**Evidence:** every non-`None` `output_quant` is treated as fused quantization during
layout validation around `_layout.py:1296-1315`, but `_epilogue.py:453-474` later
accepts only MXFP4. Unknown `DtypeQuant` values fall through to uint8/no-scale-like
behavior in `_config.py:181-216`. Stored and online formats are described as equal for
v1 at `_config.py:56-62`, but that relationship is not asserted.

**Impact:** bad values fail late with obscure trace errors or can select nonsensical
storage, packing, and MFMA combinations.

**Suggested fix:** whitelist every enum and enforce the v1 stored/online relationship.
Validate `output_quant` against `{None, MXFP4}` before layout construction.

### M12. The documented mixed-BF16 upcast path is not implemented

**Reachability:** latent/private configuration. Current supported capability checks
reject mixed BF16/microscaled pairs.

**Evidence:** `_lang.py:99-105` and `_config.py:142-162` describe
`UPCAST_MFMA` as applying `scaled_upcast` to the low-precision operand. The live dot at
`moe_gemm.py:137-156` and frozen dot at `_frozen.py:72-104` instead call plain MFMA
without applying the microscale.

**Impact:** if the capability gate is relaxed or the private entry is called directly,
the path will either fail to compile or compute the wrong value.

**Suggested fix:** implement the documented upcast and scale application, or reject
`UPCAST_MFMA` in in-directory validation.

## Performance and measurement findings

### P1. Quantized result stores ignore result cache modifiers

**Evidence:** `result_cache_modifier` and `result_scale_cache_modifier` are fields at
`_config.py:247-248`. Only the nonquantized store applies the payload modifier at
`_epilogue.py:446-452`. Quantized stores at `_epilogue.py:239-251`,
`_epilogue.py:294-306`, `_epilogue.py:542-558`, and `_epilogue.py:562-576` apply
neither modifier; the scale modifier has no use site.

**Impact:** tuning metadata does not match generated quantized-store policy, and cache
policy experiments for the fused output have no effect.

**Suggested fix:** pass the payload and scale modifiers to their respective stores, or
remove the fields until supported.

### P2. `EpilogueMode.NOP` still stages unused bias and gamma vectors

**Evidence:** `_epilogue.py:53-145` allocates, copies, commits, and later waits for
vectors based on `has_bias`/`has_gammas`. Arithmetic use is separately disabled by
`func_cfg.epilogue < 2` at `_epilogue.py:377-383` and `_epilogue.py:423-433`.

**Impact:** NOP measurements can still include LDS allocation, global copies, waits,
barriers, and pointer dereferences for inputs whose arithmetic is disabled.

**Suggested fix:** include the epilogue mode in the staging and group-count predicates.

### P3. Raw A payload and A-scale addressing repeat routing-table work

**Evidence:** `_a_payload_hbm_offsets` gathers rows at `_offsets.py:75-100`.
The raw A-scale branch repeats `_gather_rows` at `_offsets.py:207-225` in a different
layout. `_offsets.py:33-58` already documents the exposed latency and the possible
block-aligned shared-table approach.

**Impact:** this duplicates routing loads and dependent address arithmetic. It is
relevant to the measured MXFP8 x MXFP4 A8W4 path, where A has E8M0 scales and the raw
scale branch is active; it would not apply to unscaled `FP8_E4M3` A.

**Suggested fix:** calculate rows once in a common layout or stage one block-aligned
int32 row table in LDS for both consumers.

### P4. Profiling launch metadata can add GPU work and host synchronization

**Evidence:** `_entry.py:77-101` computes `hist.sum()`, `(hist > 0).sum()`, and converts
device results through `float(...)` and `int(...)` when launch metadata is requested.

**Impact:** profiler-enabled measurements of short kernels can include extra reductions
and synchronization unrelated to GEMM execution.

**Suggested fix:** derive metadata from existing host scalars, cache routing
statistics, or omit expensive fields for device-resident tensors.

### P5. Narrow E8M0 output-scale tiles issue duplicate same-address stores

**Evidence:** `_layout.py:969-995` notes that `[64, 2]` has fewer elements than the CTA
has threads and remains two-times over-covered by the generic result layout. Scale
stores use that layout at `_epilogue.py:231`, `_epilogue.py:284`, and
`_epilogue.py:510`.

**Impact:** values are identical, so this is not a confirmed determinism problem, but
the kernel spends extra store traffic and relies on overlapping writes.

**Suggested fix:** add a scale-specific store layout or predicate surplus lanes/warps.

### P6. Padded routing launches CTAs that only load a sentinel and return

**Evidence:** `_types.py:376-387` defines `-1` map entries as empty. The live body
loads the map and returns at `moe_gemm.py:502-521`; the frozen body follows the same
pattern at `_frozen.py:1679-1695`. XCD scheduling can reject a padded suffix at
`moe_gemm.py:503-511`, but that check still occurs inside a physically launched CTA.

**Impact:** small-token, many-expert workloads can spend a meaningful fraction of
time dispatching and starting waves that perform no GEMM work.

**Suggested fix:** size the physical launch to the live block count when it is
available, or use a persistent work queue. Report launched and live CTA counts when
attributing small-token overhead.

### P7. Fixed scheduling-hint recipes are not tied to emitted geometry

**Evidence:** `_sched_hint` hardcodes MFMA, DS-read, and VMEM group sizes at
`moe_gemm.py:93-112`. `_config.py:527-533` validates only the enum, and
`_pipeline.py:747-749` applies the recipe to arbitrary non-compiler-pipelined
geometries. Mini-tile sizes, LDS/register placement, scale presence, and ring depth
all change the instruction stream under the same mode name.

**Impact:** a hint can describe a different stage from the one emitted and become
ineffective or counterproductive for a new configuration.

**Suggested fix:** derive barrier counts from the authoritative emitted schedule or
whitelist each recipe for the exact validated geometry/storage modes it targets.

### P8. MXFP4 amax lane splitting is a layout-sensitive heuristic

**Evidence:** `_layout.py:1089-1114` describes `quant_amax_lane_elems` as an arithmetic
proxy for a measured accumulator layout. Every fused MXFP4 output uses it at
`_epilogue.py:473-481`.

**Impact:** another accepted tile/warp arrangement or compiler layout can reintroduce
cross-lane permutations, canonicalization, and copies without a source-level signal.
The reduction remains mathematically complete; this is a performance risk, not a
known correctness issue.

**Suggested fix:** derive the split from actual layout bases when an API is available.
Until then, constrain it to validated layouts and track instruction counts/latency for
every production quantized-output configuration.

### P9. `grid_n` remains runtime-valued although N and `BLOCK_N` are compile-time

**Evidence:** `_layout.py:323-333` can derive the exact N grid, but both entries keep
`grid_n` unspecialized at `_entry.py:105-108` and `_entry.py:224-227`. The live body
then performs runtime divide/modulo or group mapping at `moe_gemm.py:502-516`; the
frozen body does the same near `_frozen.py:1679-1691`.

**Impact:** each CTA pays avoidable prologue arithmetic, proportionally most visible
for short small-token kernels. The separate runtime value also creates the mismatch
risk in M1.

**Suggested fix:** derive constexpr `GRID_N = tuning_cfg.grid_N(N)` inside the
entry/body and remove `grid_n` from the unspecialized ABI unless runtime N-grid changes
are a real requirement.

### P10. Small scale-copy layouts can over-cover tiles and duplicate memory work

**Evidence:** `_layout.py:958-977` maps each raw scale copy across every CTA warp but
does not cap the warp-axis coverage to the tile's non-K extent. For example, a
four-wave `[64, 8]` B-scale tile uses `size_per_thread=[1,4]`,
`threads_per_warp=[32,2]`, and `warps_per_cta=[4,1]`, which covers `[128,8]` and
therefore replicates the tile twice. `_layout.py:752-778` explicitly addresses the
analogous surplus-warp replication for shuffled A scales by allowing a larger shared
fill tile, but there is no B-side counterpart.

**Impact:** small B-scale tiles can issue redundant direct-to-LDS traffic and consume
extra async-copy capacity. Separately, A tiles too small for LDS fall back via
`_layout.py:493-509` to register loads, where A-scale values are replicated across
the N-warps that consume them. Both effects are plausible contributors in small-token
scaled-input kernels; neither has yet been isolated by a one-factor benchmark.

**Suggested fix:** add exact-coverage scale copy layouts, merge adjacent B mini-scale
tiles into one shared fill where legal, and evaluate a shared 32-row A-scale fill. Use
an isolated ISA/latency experiment before promoting any change.

## Maintainability and testing caveats

### T1. The frozen snapshot covers the pipeline body, not the complete kernel

**Evidence:** `_frozen.py:4-20` explicitly freezes the performance-sensitive pipeline
while retaining shared helpers. It imports live config/layout, offset, schedule, and
epilogue code at `_frozen.py:33-47`; `_buffer_load_order` is used around
`_frozen.py:653-695`. Layout behavior can also depend on the import-time environment
value at `_layout.py:34-39` and `_layout.py:752-778`.

**Impact:** a shared regression can change both live and frozen outputs/code, so the
frozen path is not an independent whole-kernel oracle. This is a scope caveat, not a
contradiction of `_frozen.py`'s current documentation.

**Suggested fix:** describe regression claims as pipeline-body isolation, and
snapshot/hash shared helpers plus relevant environment inputs whenever a whole-kernel
comparison is required.

### T2. Copy/wait schedule policy has two implementations

**Evidence:** `_schedule.py:76-165` contains `_buffer_load_ops`, group construction,
and wait accounting used by tests but not by the live implementation. The live driver
independently implements operation selection and wait arithmetic at
`_pipeline.py:31-181`.

**Impact:** the analytical model and emitted schedule can drift while their tests
continue exercising only one side. No current mismatch was confirmed.

**Suggested fix:** make one schedule representation authoritative and derive both
emission and tests from it, or remove the non-authoritative model.

### T3. Correct LDS ordering relies partly on compiler-inserted synchronization

**Evidence:** the live step's async-copy waits and shared reads at
`_pipeline.py:719-826` do not spell an explicit source `gl.barrier()`. The frozen path
documents a load-bearing barrier around `_frozen.py:1200-1221`. At the pipeline to
quantized-epilogue boundary, the explicit final wait/barrier in `moe_gemm.py:632-639`
is conditional on staged bias/gamma, while the quant buffers allocated at
`_epilogue.py:630-666` intentionally overlay dead pipeline LDS. Current lowering
inserts the required memory barriers.

**Impact:** this is not a confirmed bug under the current compiler, but correctness is
tied to memory-barrier insertion and LDS-liveness behavior that is not obvious from
the source.

**Suggested fix:** retain code-generation regression tests that assert both the
pipeline read-after-copy boundary and the pipeline-to-quant-epilogue reuse boundary
remain fenced after lowering.

## Suggested triage order

1. Fix H1 and H3 because they can affect structurally valid large workloads.
2. Reject known-wrong `k_width` overrides (H2) before exposing more custom tuning.
3. Make validation non-optimizable (M4); it is the backstop for most custom-layout and
   schedule assumptions.
4. Decide whether private/direct kernel invocation is contractual. If it is, address
   M1-M3, M6-M8, M10-M12 as one ABI/validation hardening effort.
5. Resolve the extreme-value MXFP4 mismatch (M5) before claiming arbitrary-FP32
   reference equivalence.
6. For small-token work, prioritize P3 and P10 alongside the already proven B-payload
   cache-policy and occupancy work; then evaluate P6-P9.
7. Preserve the synchronization/codegen invariants in T3 and clarify the frozen
   oracle's scope in T1.
