# A4W4 T=4096 FlyDSL TM128 recheck

- Date: 2026-09-10
- Repository HEAD: `591175e3a7f77edd8077737ab8b83ed51f54a62a`
- Shape: T=4096, H/K=7168, raw N=4096, intermediate=2048,
  E=33, top-k=8

## Conclusion

The remembered approximately 620 us FlyDSL result is real. It came from the
native cached `TM128 x TN256 x TK256` A4W4 kernel, not from the `TM64` kernel
used in the later dtype/token matrix.

However, TM128 is intermittently incorrect. A fresh 512-replay poisoned-output
validation reproduced exact-output nondeterminism at replay 210. The compiled
50,328-byte HSACO has SHA256
`8e4ecc8008d1150167bad26c6b2fb0421f0025d84c4d256dd6b3136efe392854`,
which is exactly the same binary as the earlier failing diagnostic. Therefore
TM128 must not replace the production/safe FlyDSL baseline until the kernel race
is fixed and long replay validation passes.

The September 9 tuning process did not benchmark and defeat TM128. It explicitly
removed TM128 from the candidate set based on the earlier nondeterminism report,
then selected `TM64/XCD4` as the best member of the retained safe subset. The
selection rationale was substantively correct, but its fresh supporting failure
artifact was not retained with that tuning run, which made the result look like a
performance-tuning omission.

## Session-history evidence

The September 6 cold K sweep fixed both implementations at 128x256x256 and
recorded, at K=7168:

- supplied harness FlyDSL median: 620.714 us
- matched-routing strict-cold FlyDSL median: 615.854 us

The supplied-harness result used six interleaved rounds, 40 warmups and 100
measured dispatches per run, with a 768 MiB fill. See
`../cold_k_sweep_20260906_162758/report.md`.

Older tuning documentation also records `g1_a4w4_128x256x256` at 574.6 us, but
that was produced by a different rocprof/report workflow and is not the best
apples-to-apples number for the final strict-cold protocol.

## Fresh same-protocol performance recheck

The normal 16-replay validation passed, followed by four cold processes with 40
warmups and 100 samples each. Every measured dispatch was immediately preceded
by one 768 MiB streaming fill; routing, quantization, scale sorting, shuffling,
allocation, compilation, and launch preparation were outside the timed loop.

| Kernel | Mean (us) | Median (us) | P99 (us) | Status |
|---|---:|---:|---:|---|
| FlyDSL TM128 x TN256 x TK256, cached, XCD0 | 607.198 | 604.124 | 656.445 | Performance valid; correctness-disqualified |
| Final-matrix FlyDSL TM64 x TN256 x TK256, cached, XCD4 | 682.887 | 682.926 | 704.566 | Validated safe baseline |
| Final-matrix Gluon BM128 x BN256 x BK256 | 611.884 | 610.585 | 646.685 | Validated safe baseline |

Relative to the final-matrix TM64 result, TM128 reduced mean/median/p99 latency
by 11.08% / 11.54% / 6.83%. Its mean and median were also 0.77% and 1.06%
lower than Gluon, while its p99 was 1.51% higher. These performance numbers do
not make TM128 deployable because the long correctness replay failed.

### 2026-09-10 direct TM64/TM128 rerun

A fresh alternating-order rerun measured both FlyDSL configurations together
under the same protocol. TM128 recorded 608.379 / 606.904 / 655.204 us and
TM64/XCD4 recorded 685.480 / 685.485 / 707.485 us (mean/median/p99, 400 samples
per case). The independent audit passed all 8 processes, 1,120 fill/GEMM pairs,
and 800 measured samples. TM128 then reproduced its intermittent correctness
failure in a separate extended validation at replay 185. See the
[direct TM64/TM128 comparison](cold_a4_tm64_tm128_rerun_20260910/COMPARISON.md).

The TM128 launch used 264 active routing blocks, 289 allocated routing blocks,
and 4,624 native grid blocks. The TM64 matrix kernel used twice as many active M
blocks and 8,720 grid workgroups, which explains most of the latency difference.

## Correctness evidence

Historical retained evidence in
`../gluon_flydsl_serial_20260907_Fxq0r9/METHODS.md` reports:

- an initial reference failure;
- several short validations that passed;
- a later 16-replay failure at replay 7 on another GPU;
- a 256-replay diagnostic with failures at replays 52, 162, and 226, changing
  payload and scale bytes while all input hashes remained unchanged.

The fresh recheck behaved the same way:

- 16 poisoned-output replays passed;
- 512 requested replays failed exact equality at replay 210;
- the current and historical failing kernels have the identical HSACO SHA256
  shown above.
- the selected TM64/XCD4 kernel passed the independent reference and all 256
  poisoned-output replays in a matching control run.

This also demonstrates that a 16-replay gate is too weak for this intermittent
failure mode.

## Likely implementation issue and next step

The retained source/ISA review identifies a plausible BM128-only LDS pipeline
race: the A-payload ring has two slots and a prefetch distance of two, so a wave
can refill the same slot while other waves are still reading it. The generated
ISA has no workgroup barrier between the relevant LDS reads and the following
direct-to-LDS refill. This is a strong hypothesis, not yet a proven root cause.

Before reconsidering TM128:

1. Add or restructure synchronization/ring ownership so the consumed A slot
   cannot be overwritten early.
2. Fix the validator's poison-byte generation for replay indices above 255,
   then prove the kernel change with at least 1,024 poisoned cold replays on
   multiple GPUs.
3. Re-run independent-reference validation and the audited four-round cold
   benchmark.

## Artifacts

- Fresh cold result: `cold_a4_tm128_recheck_20260910/`
- Independent trace/statistics audit:
  `cold_a4_tm128_recheck_20260910/independent_audit.json`
- Short validation that passed:
  `validation_a4_tm128_recheck_20260910/`
- Long validation that failed at replay 210:
  `validation_a4_tm128_long_20260910/`
- TM64/XCD4 256-replay control that passed:
  `validation_a4_tm64_xcd4_256_20260910/`
- The separate `validation_a4_tm64_xcd4_long_20260910/` attempt is not a
  kernel failure: the validator itself overflows its uint8 poison value when
  it reaches replay 256.
- Historical failure analysis:
  `../gluon_flydsl_serial_20260907_Fxq0r9/METHODS.md`
- Historical 256-replay failure details:
  `../gluon_flydsl_serial_20260907_Fxq0r9/replay_diagnostic_gpu1/result.json`
