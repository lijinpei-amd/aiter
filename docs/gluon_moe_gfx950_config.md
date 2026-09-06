# Gluon MoE scheduling configuration

`moe_gemm.py` takes its options from `KernelFuncConfig` and
`KernelTuningConfig`; it does not read the environment. The host launcher still
translates legacy benchmark environment variables, so `~/gluon_cold_bench.sh`
continues to select the recorded impl and frozen configurations.

## Tuning options

Pass these fields in the launcher's `config` dictionary. They are also trailing
fields of `TuningSpec`, with defaults for older dictionaries and positional specs.

| Field | Values and meaning |
| --- | --- |
| `DS_READ_IN_MFMA` | `DSReadOperand` bitmask: `A=1`, `B=2`, `A_SCALE=4`, `B_SCALE=8`. Set bits place the corresponding read in the MFMA region; unset bits place it in the memory region. `ALL=15`. |
| `SCHED_MODE` | `SchedMode.NONE`, `IGLP_0`, `IGLP_1`, `MFMA_16`, or `MFMA_8`. Hints apply outside compiler warp-pipeline regions. |
| `MANUAL_PP` | Emit the manual prologue fence. This is independent of `WARP_PIPELINE`; the frozen step owns its recorded per-slot rendezvous. |
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
    MANUAL_PP=True,
)
# Pass config=tuning and epilogue=EpilogueMode.DEFAULT to moe_gemm_gluon.
```

The legacy `DS_IN_MFMA=1` maps to mask 15. `DS_MOVE=1` maps to mask 5;
`DS_MOVE>=2` maps to mask 15. An explicit
`AITER_TRITON_MOE_GLUON_DS_READ_IN_MFMA` takes precedence over both.

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

The frozen step keeps its recorded schedule, including its literal priority
instructions and original unfused drain. This is necessary for `FROZEN_STEP` to
remain the same reference kernel. It does not acquire the live step's read-placement
or scheduling options. Use the complete frozen recipe from the cold benchmark
wrapper when comparing it with the best impl recipe.

The September 6 refactor was checked against the pre-refactor commit
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
