# Gluon MoE scheduling configuration

`moe_gemm.py` takes its options from `KernelFuncConfig` and
`KernelTuningConfig`; it does not read the environment. The host launcher still
translates legacy benchmark environment variables, so `~/gluon_cold_bench.sh`
continues to select the recorded impl and frozen configurations.

The low-level entry points `_moe_gluon_gemm1` and `_moe_gluon_gemm2` and launch
metadata live in `_entry.py`. Each entry constructs `KernelFuncConfig` and
`KernelTuningConfig` from the launch specs and dispatches once to the live body in
`moe_gemm.py` or the self-contained reference body in `_frozen.py`, alongside
constexpr `N` and `K` and a runtime `NUM_K` scalar.
The host computes `NUM_K = K // BLOCK_K` and disables specialization of that
argument. Shape and layout calculations may still use constexpr `K`; the K-loop
bounds derive from the runtime scalar. The A-scale row stride is constexpr so
raw rows retain their 4- or 8-byte alignment when they are not 16-byte aligned.
The host caches the two specs and
dimensions as four constexpr arguments. `_buffered.py` implements the live driver for
shared and independent component depths, and `_buffered_schedule.py` computes
the compile-time issue history and waits. `_frozen.py` owns the preserved LDS,
dot, pipeline-state, loop, drain, and final-MFMA path.
`_offsets.py` computes A/B HBM offsets and mini-tile
indices, and `_epilogue.py` owns output activation, quantization, staging, and
stores. The host launch API is unchanged.

## Tuning options

Pass these fields in the launcher's `config` dictionary. They are carried in
`TuningSpec`, with defaults for omitted optional fields.
The config's `BLOCK_M` must match `routing_data.block_m`: the router builds
padded offsets and the block map for that geometry, and the host rejects a
mismatch before preparing scales or launching the GEMM.

| Field | Values and meaning |
| --- | --- |
| `DS_READ_IN_MFMA` | `DSReadOperand` bitmask: `A=1`, `B=2`, `A_SCALE=4`, `B_SCALE=8`. Set bits place the corresponding read in the MFMA region; unset bits place it in the memory region. `ALL=15`. |
| `SCHED_MODE` | `SchedMode.NONE`, `IGLP_0`, `IGLP_1`, `MFMA_16`, or `MFMA_8`. Hints apply outside compiler warp-pipeline regions. |
| `FROZEN_STEP` | Select the preserved reference body in `_frozen.py`, with an unfused final MFMA. Its existing restrictions on component overrides, register flags and packed K128 scales remain. |
| `SOFF_UNROLL` | Advance HBM pointers once per unrolled body and use scalar offsets within it. With packed K128 scales and odd unroll, use per-fill pointer advances so scale-word phases remain correct. |
| `SCALE_FILL_MID` | At a 2x2 mini-tile split, fill scales in the middle slots; wait counts use the same setting. |
| `WAIT_COMMIT_SCHEME` | `PER_OP=1`, `PER_SLOT=2`, `PER_STAGE_WHOLE=3`, or `PER_STAGE_WARP_PIPELINE=4`; see the boundaries below. |
| `B_IN_REG` | Load the B payload into registers with `buffer_load` at its normal global-load slot, bypassing LDS staging. Requires `B_PRESHUFFLED=True`. Defaults to `False`. |
| `A_SCALE_IN_REG`, `B_SCALE_IN_REG` | Load the selected scale component directly into registers. Both options are independent of each other and of `B_IN_REG`; neither requires preshuffled B payload. Defaults to `False`. |
| `A_NUM_BUFFER`, `B_NUM_BUFFER` | Number of pipeline buffers for each payload component. Omitted or zero values inherit `NUM_LDS_BUFFER`; each resolved active depth must be at least 2. |
| `A_SCALE_NUM_BUFFER`, `B_SCALE_NUM_BUFFER` | Number of buffers for each active scale component, independent of the payload counts and at least 2 after inheritance. Absent scales do not participate in validation or scheduling. `A_SCALE_NUMB_BUFFER` is accepted as a dictionary/environment alias for `A_SCALE_NUM_BUFFER`. |
| `K_UNROLL` | Requested main-loop unroll, at least 1. The effective unroll is rounded up to the next multiple of all active register-ring depths' least common multiple. |

The same component driver runs whether depths are inherited or explicit. LDS
indices may vary at runtime, so LDS-only components do not constrain unrolling.
Register tuples use static ring indices in each unrolled body; its length must
return every register ring to its initial slot. If the active register depths
are 2 and 3, their period is 6: requested `K_UNROLL=4` becomes `UNROLL=6`, and
requested `K_UNROLL=7` becomes `UNROLL=12`. With all active components in LDS,
`UNROLL=K_UNROLL`. Scale components that fall back to register loads because of
their layout also participate; absent scales never do. `FROZEN_STEP` retains
its existing rejection of explicit component-depth and register-storage options.

Let `NB_MAX` be the largest resolved active depth and `PEELED` the number of
initial main iterations needed before invariant steady-state waits. The first
MFMA is always peeled, so `PEELED >= 1`; unequal-depth warmup may require more.
The host and device use the same issue-history helper to determine this value.
The exact host requirement is:

```text
NUM_K = K / BLOCK_K
NUM_K >= NB_MAX + PEELED + UNROLL
```

This guarantees at least one complete runtime unrolled body. Equal depths need
only the first peel, giving `NUM_K >= NB_MAX + UNROLL + 1`. `K % BLOCK_K == 0`
and all layout-specific divisibility rules still apply, including complete K256
words for packed K128 scales. `KernelTuningConfig.min_num_k()` exposes the
threshold and `validate_pipeline(K)` applies the resolved-depth and K checks.
The launcher checks depths and K divisibility before preparation, then validates
the complete effective configuration before launch. The final check matters
when scale preparation succeeds or falls back: moving scales between LDS and
registers can change `UNROLL`, `PEELED`, and the minimum accepted K.

All seven new fields also accept the `AITER_TRITON_MOE_GLUON_` environment prefix
used by the benchmark scripts. A config dictionary with `B_PRESHUFFLED=True`
means the caller supplied weights in the existing 16-column-blocked order;
`AITER_TRITON_MOE_GLUON_B_PRESHUFFLED=1` instead requests host preparation of raw
weights. `B_IN_REG` without preshuffled weights raises an error.

`PER_STAGE_WARP_PIPELINE` commits inside the last memory region in the K stage,
after its memory work and before the final MFMA region. `PER_STAGE_WHOLE` commits
outside MFMA, after the full stage's work. With compiler pipelining enabled, this
post-MFMA commit has its own `commit` region so the next wait stays outside a
region. The region wrappers emit no borders with `WARP_PIPELINE=NONE`. Value 3
preserves the whole-stage boundary used by the cold-bench recipes; callers that
previously selected `PER_STAGE` with the compiler warp pipeline should select
`PER_STAGE_WARP_PIPELINE` for the memory-region boundary.

Per-stage schemes wait before the `ni`/`mi` slot walk; `PER_OP` and `PER_SLOT`
wait before each slot's LDS reads. Wait counts follow each component's producer
and the actual committed groups, including empty slot/stage groups during the
prologue and drain. Direct-register loads own no LDS async-copy groups. Their
register dependencies provide VMEM completion before use.

Payload and scale placement are independent. Missing components emit no reads;
components loaded directly into registers follow the same region selection.
For example, after obtaining a normal supported tuning dictionary:

```python
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    DSReadOperand,
    EpilogueMode,
    SchedMode,
)

tuning = dict(tuning)
tuning.update(
    DS_READ_IN_MFMA=int(DSReadOperand.A | DSReadOperand.B_SCALE),
    SCHED_MODE=int(SchedMode.NONE),
)
# Pass config=tuning and epilogue=EpilogueMode.DEFAULT to moe_gemm_gluon.
```

The legacy `DS_IN_MFMA=1` maps to mask 15. `DS_MOVE=1` maps to mask 5;
`DS_MOVE>=2` maps to mask 15. An explicit
`AITER_TRITON_MOE_GLUON_DS_READ_IN_MFMA` takes precedence over both.

`MANUAL_PP` has been removed from the tuning configuration; its legacy environment
variable and dictionary key are ignored. The new storage and buffer fields are
trailing optional fields of `TuningSpec`, so older positional constructions keep
their meaning. `B_PRESHUFFLED` selects the permuted weight layout for either LDS
staging or `B_IN_REG` loads.

## Tuned register/buffer recipes

The opt-in recipes in
[`scripts/gluon_moe_register_tuned`](../scripts/gluon_moe_register_tuned/README.md)
cover GEMM1 on MI350X with T=1024, N=2048, K=7168, E=33, top-k=8,
preshuffled B, split gate/up SiLU, and FP32 output. Both the best B-register
and best independent LDS-buffer recipes are retained for each dtype.

| Dtype | Legacy baseline | Tuned B in registers | Tuned B in LDS |
| --- | ---: | ---: | ---: |
| a4w4 / MXFP4 | 145.201 us | 139.741 us | 136.521 us |
| a8w8 / MXFP8 | 256.992 us | 228.941 us | 219.642 us |
| a16w16 / BF16 | 500.864 us | 421.983 us | 470.983 us |

These historical measurements precede the unified runtime driver. They are
medians from five interleaved rounds on one GPU, with 40 warmups and
100 measured dispatches per recipe per round. A 768 MiB flush precedes each
wrapper call; MX scale sorting follows the flush. Timings contain only GEMM1.
Every final recipe passed 16 exact-output cold replays per round and has zero
VGPR spills/private memory. The tuned B-register recipes lower latency by 3.8%,
10.9%, and 15.7% against the respective original recipes. Independently buffered
LDS B is the faster choice for the two MX dtypes on this workload.

The [four-wave MXFP4-output follow-up](gluon_moe_gfx950_register_perf.md)
records the separate T4096/N4096 benchmark near 615 us. The transferred recipes
regress on that workload; preshuffling B with the original LDS pipeline improves
its measured latency. The results above apply to the smaller FP32-output workload.

The winning B-register recipes use `expert_mod=".cg"` and effective buffer counts
`(4,2,3,3)` for MXFP4 and `(3,3,3,3)` for MXFP8/BF16. The measurements used the
former GCD policy, with unrolls 1, 3, and 3. Replaying these dictionaries now
applies the register-ring LCM policy above and requires new performance
measurements. The scale-register flags remain independent; the selected MX
recipes use LDS scales. Cache policy accounted for much of the MX improvement: tuned legacy
controls measured 136.941 us for MXFP4 and 230.401 us for MXFP8.

The reusable runner is
[`scripts/gluon_moe_register_tune.py`](../scripts/gluon_moe_register_tune.py).
It accepts candidate dictionaries and records correctness, determinism, compiler
resources, source hashes, and cold profiler traces. The linked recipe README
contains replay commands and the complete measurement protocol.

Tuning exposed two configuration/scheduling defects that are now covered by
regressions: unquantized output must unwrap `output_quant` before checking for
None, and pointer advances must stay before the final compiler-pipeline stage
border so a subsequent unrolled wait remains at a region head. At the time of
those measurements, the default kernels matched their preserved encoded
machine-code hashes and resource metadata for all three dtypes.

## Pipeline state

The driver takes invariant data in `_PipelineConst`. Each active stream has its
own depth `NB`, LDS ring or register queue, and HBM pointer. B register loads
occupy the same logical fill slot as B LDS copies. Scale register loads also use
their configured fill slots, including `SCALE_FILL_MID`.

The prologue has `NB_MAX - 1` fill-only stages. A component starts at
`NB_DELTA = NB_MAX - NB`, loading tiles 0 through `NB - 2`. A seed stage then
reads tile 0 and loads tile `NB - 1`, without an MFMA. Each logical main
iteration `x` accumulates tile `x`, reads tile `x + 1`, and loads tile `x + NB`
for each stream. `MAIN = NUM_K - NB_MAX` remains a runtime value.

After the peeled iterations, the driver executes complete statically unrolled
bodies up to the runtime bound. Short bodies use statically expanded guarded
remainder steps. When `UNROLL > 6` and the active register period exceeds one,
a runtime remainder loop avoids register spills caused by the long chain of
guarded tuple updates. Register rings keep distinct SSA slots through a whole unrolled body;
the finite prologue, remainder and drain rotate their logical queues. Packed
K128 scale words retain both halves until the read selects the current half.
Ring and packed-scale phases continue across every phase boundary.

The pipeline drain has `NB_MAX` MFMAs. Drain iteration `j` continues a
component's fill only while `j < NB_DELTA`, so every active stream loads exactly
`NUM_K` tiles. The first `NB_MAX - 1` drain iterations still read the next tile;
the separate final MFMA consumes the prefetched operands without another read
or fill. Eligible output epilogues fuse with that last MFMA; other paths execute
it once before the output epilogue. Epilogue input copies overlap the drain and
participate in its group history.

The K loop carries two aggregates plus the component register rings:

- `_PipelinePointers`, named `hbm_ptrs` at call sites, contains the A/B payload
  HBM pointers and one scale HBM pointer per operand, shared by the mutually
  exclusive LDS-staging and direct-register scale paths.
- `_PipelineRegFragments` contains separate A/B payload and scale tuples and the
  MFMA accumulators. Operand tuples run by mini block, then mini-K step;
  accumulators follow the N-outer, M-inner slot traversal.

`_PipelineConst.lds_ptrs` holds the `LDSManager` descriptors, named
`a_payload_lds_ptr`, `a_scale_lds_ptr`, `b_payload_lds_ptr`, and `b_scale_lds_ptr`.
LDS tile `t` occupies slot `t % NB` for that component. `_fill_slot` issues
copies and commits, `_read_slot` supplies the next register fragments, and
`_advance` moves only pointers whose streams actually filled. With
`SOFF_UNROLL`, the last step advances by the complete body's fill count.
Cooperative LDS barriers and read-result dependencies protect buffer reuse.

## Epilogue function configuration

`moe_gemm_gluon` and `moe_gemm1_a4w4_mxfp4_out` accept `epilogue=`. It becomes a
`FuncSpec` field and participates in the launch cache key.

| Mode | Arithmetic |
| --- | --- |
| `EpilogueMode.DEFAULT` | Normal bias, activation, gate-times-up, and gamma multiplication. |
| `EpilogueMode.NOP_ACTIVATION` | Skip activation, clipping, and activation residual addition; retain bias, gate-times-up, and gammas. |
| `EpilogueMode.NOP` | Also skip bias and gammas. |

Both NOP modes retain the output shape, quantization, stores, and operand
dequantization, including a per-tensor input scale. They are benchmark ablations.
If `epilogue` is omitted, the host accepts the legacy `NO_EPI` value; an explicit
mode overrides it.

## Fixed behavior and the frozen reference

`WAIT_BIAS`, `NO_MFMA`, and `MEM_PRIO` no longer control the live kernel.
The live kernel always uses late LDS-read retirement and relaxed epilogue loads,
retaining the mandatory staging barriers and read-result dependencies. Quantized
interleaved output uses rotating staging buffers; gate/up split output already
stages the whole block before its flush. The interleaved activation passes
`GROUP=3` (the existing ordinary interleaved packing; only `GROUP=2` selected the
alternate packing). Eligible live gate/up split kernels fuse the final activation
into the MFMA drain.

The frozen body keeps its fixed prologue fence and literal priority instructions,
its runtime loop and drain, and its original LDS and dot-selection behavior in
`_frozen.py`. It retains an unfused final MFMA and its existing restrictions. It
does not acquire the live step's read-placement or scheduling options. Use the
complete frozen recipe from the cold benchmark wrapper when comparing it with the
best impl recipe.

The encoded-code comparisons and timing results below describe earlier
refactors, before the unified runtime loop. The current refactor's source
snapshot and validation artifacts are under `bench_out/independent_iter_20260908/`.

The preceding scheduling-configuration refactor was checked against commit
`ad02ed5b7`. The best impl and frozen compiled `.text` sections match byte for byte
for both BF16 and MXFP4 output. Reproduction scripts, baseline sources and
binaries, correctness logs, and relaxed-LDS investigation artifacts are retained
under `bench_out/gluon_refactor_20260906/` in the working checkout.

The follow-up cold benchmark on GPU 4 used six alternating paired rounds, with
40 warmups and 100 measured dispatches per arm per round. Each repetition flushed
768 MiB before the existing scale sorter and GEMM1; timing includes only GEMM1.
At T=4096, K=7168, N=4096, E=33 and top-k=8, with MXFP4 output and no bias/gammas:

| Kernel | Median of six round medians |
| --- | ---: |
| impl | 617.534 us |
| frozen | 614.6445 us |

The mean paired difference was impl +2.5967 us (0.422%), with a paired Student-t
95% interval of +1.3934 to +3.8000 us. Both freshly compiled binaries matched the
preserved best code. Full traces and their independent audit are retained under
`bench_out/cold_impl_frozen_20260906_213142/` in the working checkout.

The subsequent pointer/register refactor uses `fe5ec1ffc` as its preserved
baseline. Both frozen output variants still compile to identical encoded `.text`.
Removing the live manual prologue fence changes impl code generation. The live
per-stage prologue also drops its duplicate outer wait: the stage head already
waits for the same buffer before any copies or reads. Other commit schemes keep
their outer wait, and the frozen prologue is unchanged. A compiler scheduling
boundary immediately after the first live wait limits scalar temporary lifetimes
across the prefetch. It adds no hardware rendezvous; the normal wait and
shared-memory barrier supply cooperative synchronization.

Validation passed 43 CPU configuration/aggregate cases, 50 GPU pipeline and
epilogue cases, and three additional preshuffled-weight/direct-scale probes.
Both impl and frozen passed 30 exact BF16 and MXFP4 replays with the preserved
output hashes. Sources, commands, caches, and results are retained under
`bench_out/gluon_pipeline_refactor_20260906_215026/`.

The final cold comparison used 12 interleaved rounds on idle GPU 4, with the same
shape, 40 warmups, 100 measured dispatches, and 768 MiB flush per repetition as
above. The numbers below are medians of the 12 round medians:

| Kernel | Baseline `fe5ec1ffc` | Refactored |
| --- | ---: | ---: |
| impl | 616.804 us | 613.96425 us |
| frozen | 615.58425 us | 616.01425 us |

The paired impl change was -2.77325 us (-0.45%), with a Student-t 95% interval of
-5.05598 to -0.49052 us. The byte-identical frozen control changed by +1.191375 us,
with an interval of -1.44691 to +3.82966 us. All 48 processes and 4,800 measured
dispatches passed an independent trace, binary, and launch-metadata audit; every
process started after two clean GPU-idle readings. Full results are in
`bench_out/gluon_pipeline_refactor_20260906_215026/cold_final/`.

The buffer-load/DS-read naming and module extraction refactor was checked against
`0d016f483`. Both impl and frozen retain identical encoded `.text` and launch
metadata for BF16 and MXFP4 output, with 30 exact replays of each variant.
Validation also passed 43 CPU cases, 50 GPU pipeline cases, four epilogue staging
cases, three repeated preshuffled-weight/direct-scale probes, and six GEMM2
dtype/reference cases. The GEMM2 cases use explicit supported split tiling with
three LDS buffers; their original inputs and Torch reference assertions remain
unchanged.

Using the same cold protocol above, 12 interleaved baseline/current rounds gave:

| Kernel | Baseline `0d016f483` | Refactored |
| --- | ---: | ---: |
| impl | 615.574 us | 614.38375 us |
| frozen | 616.624 us | 617.13375 us |

The paired current-minus-baseline mean was -0.750125 us for impl, with a
Student-t 95% interval of -2.88330 to +1.38305 us; frozen was +0.284958 us,
with an interval of -1.78503 to +2.35495 us. Neither change is statistically
significant. All 48 processes, 4,800 samples, unchanged binaries, launch metadata,
and 49 idle guards passed independent audit. A detected foreign GPU allocation
was confined to a pause between benchmark processes; no samples were discarded.
Evidence and source snapshots are retained under
`bench_out/gluon_module_refactor_20260906_235642/`.
