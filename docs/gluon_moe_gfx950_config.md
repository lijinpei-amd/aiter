# Gluon MoE scheduling configuration

`moe_gemm.py` takes its options from `KernelFuncConfig` and
`KernelTuningConfig`; it does not read the environment. The host launcher still
translates legacy benchmark environment variables, so `~/gluon_cold_bench.sh`
continues to select the recorded impl and frozen configurations.

The low-level entry points `_moe_gluon_gemm1` and `_moe_gluon_gemm2`, their
`MoeKernelConfig` argument, and launch metadata live in `_entry.py`. The shared
GEMM body and pipeline remain in `moe_gemm.py`; `_offsets.py` computes A/B HBM
offsets and mini-tile indices, and `_epilogue.py` owns output activation,
quantization, staging, and stores. The host launch API is unchanged.

## Tuning options

Pass these fields in the launcher's `config` dictionary. They are also trailing
fields of `TuningSpec`, with defaults for omitted optional fields.

| Field | Values and meaning |
| --- | --- |
| `DS_READ_IN_MFMA` | `DSReadOperand` bitmask: `A=1`, `B=2`, `A_SCALE=4`, `B_SCALE=8`. Set bits place the corresponding read in the MFMA region; unset bits place it in the memory region. `ALL=15`. |
| `SCHED_MODE` | `SchedMode.NONE`, `IGLP_0`, `IGLP_1`, `MFMA_16`, or `MFMA_8`. Hints apply outside compiler warp-pipeline regions. |
| `FROZEN_STEP` | Select the preserved reference step and its original unfused drain. |
| `SOFF_UNROLL` | Advance HBM pointers once per unrolled body and use scalar offsets within it. |
| `SCALE_FILL_MID` | At a 2x2 mini-tile split, fill scales in the middle slots; wait counts use the same setting. |

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

`MANUAL_PP` and `B_IN_REG` have been removed from the tuning configuration. Their
legacy environment variables and dictionary keys are ignored. Positional
`TuningSpec` construction must use the current field order. `B_PRESHUFFLED` still
selects the permuted weight layout staged through LDS.

## Pipeline state

Each step takes invariant data in `_PipelineConst` and carries two separate
aggregates through the K loop:

- `_PipelinePointers`, named `hbm_ptrs` at call sites, contains the A/B payload
  and scale HBM pointers plus the direct scale HBM pointers used by register loads.
- `_PipelineRegFragments` contains separate A/B payload and scale tuples and the
  MFMA accumulators. Operand tuples run by mini block, then mini-K step;
  accumulators follow the N-outer, M-inner slot traversal.

`_PipelineConst.lds_ptrs` holds the `LDSManager` descriptors, named
`a_payload_lds_ptr`, `a_scale_lds_ptr`, `b_payload_lds_ptr`, and `b_scale_lds_ptr`.
`BUFFER_LOAD_IDX` selects the LDS destination buffer and `DS_READ_IDX` selects
the LDS source buffer. Direct scale fallbacks still address HBM.

`_buffer_load` only issues copies and commit markers. It has no `ADVANCE`
argument or pointer return value. Its caller uses `_advance_hbm_ptrs` after the
last slot, inside the memory region. With `SOFF_UNROLL`, this happens once per
unrolled body. `_ds_read` loads the next operands, and
`_advance_direct_scale_hbm_ptrs` advances direct scale pointers after every stage
read. The drain advances neither copy pointers nor pointers for a stage it does
not read.

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

The frozen step keeps its recorded schedule, including its fixed prologue fence,
literal priority instructions, and original unfused drain. This is necessary for `FROZEN_STEP` to
remain the same reference kernel. It does not acquire the live step's read-placement
or scheduling options. Use the complete frozen recipe from the cold benchmark
wrapper when comparing it with the best impl recipe.

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
`PER_STAGE` prologue also drops its duplicate outer wait: the first slot already
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
