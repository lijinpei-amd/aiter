# A8W4 NT geometry finalist analysis

## Audit status

- Validation: 8/8 cells passed an FP32 reference and 16 poisoned exact-output
  replays per cell.
- Cold: 32/32 processes passed, with 4,480 independently verified
  fill-to-GEMM pairs and 3,200 measured samples.
- Each cell pools 400 samples from four alternating-order rounds. Each sample is
  one 768 MiB flush followed by one prepared GEMM1 launch; preprocessing and
  launch preparation are outside the timed loop.
- `validation/failures.json` and `cold/failures.json` are both `[]`.
- The independent trace result is `cold/independent_audit.json`.

## One-factor B-payload cache-policy isolation

Before changing geometry, a separate audited experiment kept the production
BN256/depth3/WPE1 kernel fixed and changed only its B-payload loads from cached
to NT-only. All times are microseconds; p99 is nearest-rank.

| Tokens | B-payload policy | Mean | Median | P99 |
|---:|---|---:|---:|---:|
| 16 | Cached | 200.667 | 200.522 | 205.442 |
| 16 | NT-only | 153.673 | 153.601 | 157.562 |
| 64 | Cached | 205.923 | 205.802 | 209.842 |
| 64 | NT-only | 159.997 | 159.902 | 163.081 |

The NT-only change reduced mean latency by 46.994 us / 23.42% at T16 and
45.925 us / 22.30% at T64. ISA comparison found the same instruction stream
except that exactly 72 direct-to-LDS B-payload loads acquired the NT bit; no
other instruction changed. The validation, trace audit, and duplicate FlyDSL
controls are retained under `../a8w4_nt_isolation_20260910/`.

## Finalist results

All times are microseconds; p99 is nearest-rank.

| Tokens | Case | Mean | Median | P99 |
|---:|---|---:|---:|---:|
| 16 | BN256 / depth3 / WPE1 / NT | 153.409 | 153.041 | 157.921 |
| 16 | BN128 / depth3 / WPE2 / NT | 146.859 | 146.721 | 151.321 |
| 16 | FlyDSL control A | 131.703 | 131.601 | 135.881 |
| 16 | FlyDSL control B | 131.910 | 131.801 | 136.201 |
| 64 | BN256 / depth3 / WPE1 / NT | 160.125 | 160.001 | 163.121 |
| 64 | BN128 / depth3 / WPE2 / NT | 156.209 | 156.041 | 160.761 |
| 64 | FlyDSL control A | 133.099 | 132.881 | 137.721 |
| 64 | FlyDSL control B | 134.163 | 134.001 | 138.441 |

BN128/WPE2 improves over BN256/WPE1 by:

| Tokens | Mean | Median | P99 |
|---:|---:|---:|---:|
| 16 | 6.550 us / 4.270% | 6.320 us / 4.130% | 6.600 us / 4.179% |
| 64 | 3.917 us / 2.446% | 3.960 us / 2.475% | 2.360 us / 1.447% |

The per-round mean improvements were `[7.059, 6.077, 6.226, 6.837]` us at
T16 and `[3.642, 3.680, 4.058, 4.285]` us at T64. A seeded bootstrap over the
four paired round means gives 95% intervals `[6.152, 6.948]` us and
`[3.661, 4.172]` us respectively. All eight round-level comparisons favor the
candidate.

The duplicate FlyDSL controls differ by only 0.207 us mean / 0.200 us median
at T16 and 1.064 us mean / 1.120 us median at T64. Pooling both controls gives:

| Tokens | Mean | Median | P99 |
|---:|---:|---:|---:|
| 16 | 131.807 | 131.721 | 136.201 |
| 64 | 133.631 | 133.481 | 138.121 |

The finalist still trails pooled FlyDSL by 15.052 / 15.000 / 15.120 us at T16
and 22.577 / 22.560 / 22.640 us at T64 (mean / median / p99). Relative to the
BN256 NT control, the finalist closes 30.32% of the remaining mean gap at T16
and 14.78% at T64.

## Resource and code audit

T16 and T64 produce identical machine text within each case. No audited kernel
has scratch, private memory, SGPR/VGPR spills, or a dynamic stack.

| Case | Dynamic LDS | ELF VGPR | AGPR | SGPR | next-free VGPR | `.text` bytes | `.text` SHA256 |
|---|---:|---:|---:|---:|---:|---:|---|
| BN256 d3 WPE1 | 132,432 | 268 | 52 | 90 | 268 | 15,360 | `cc50ff092f8776c6739f0a4d8353c2252cee95a3f76c7b1a0d4b4cfb2e91a663` |
| BN128 d3 WPE1 | 78,672 | 224 | 32 | 102 | 257 | 11,392 | `66ade82c5018b2456ae85ae0c8477c6035f01d50769b5c07c837a96d3af3f2b9` |
| BN128 d3 WPE2 | 78,672 | 206 | 78 | 102 | 206 | 11,776 | `77d89e199a81840cc30513f3fabf9e3e2a0652afca7451fc2b6877406abea626` |
| BN128 direct B | 28,016 | 182 | 0 | 89 | 257 | 9,920 | `644fb2d8c24a3d57e4ef8ef611cc0498ae402655c7dda3851f2b3f85288d5a4d` |
| BN128 direct B + scale | 24,944 | 206 | 0 | 62 | 257 | 9,216 | `6eb4b8d6ed7a79b7fe181ef4213ff51e97bd12ef85fb58736c4f85e7b76f0cc0` |
| BN256 direct B | 31,088 | 264 | 8 | 74 | 264 | 13,504 | `5d814d261a3bb137fcc71a503722f79fd3617cfd41c0934f34f92e146983349f` |
| Tuned FlyDSL | 32,896 | 134 | 0 | 48 | n/a | n/a | ELF `ddd87558daab769f465d0556ea33e6765db70283da2b2a0f5732282f5c405326` |

BN128 WPE1 and WPE2 are genuinely different programs, not metadata-only
variants. Function sizes are 10,344 and 10,736 bytes, exact function hashes are
`059ec2b5a3e8bbc34030deddee6b39f6f913ebd5b1709f14ebd7ceb7003ed34f`
and `0af0cc928ae5de8f8ef16b135f6d6151bf0e30ce745ee7f8bb6a086170b7a75f`,
and WPE2 changes the allocation from next-free VGPR 257 / accum offset 192 to
VGPR 206 / accum offset 128. Combined with the LDS reduction from 132,432 to
78,672 bytes against gfx950's 163,840-byte LDS capacity, BN128+WPE2 removes both
resource barriers to two resident workgroups. BN128 alone and WPE2 alone were
neutral in the screen; the interaction is what matters.

## Root-cause ranking

1. **B-payload cache policy is the dominant proven cause.** The 462 MiB packed
   weight payload is effectively single-use at T<=256. Changing exactly 72
   direct-to-LDS B-load instructions from cached to NT-only, with no other ISA
   change, cut mean latency by 23.42% at T16 and 22.30% at T64 and closed 63.3%
   and 57.0% of the original mean gap.
2. **A resource/occupancy interaction is a smaller proven cause.** BN128+WPE2
   produces the repeatable finalist gain above. BN128 by itself was neutral,
   WPE2 on BN256 was neutral, and two-buffer pipelines were much slower, so the
   win is not simply less LDS or less work per CTA.
3. **FlyDSL's specialized direct-B schedule explains much of the residual.**
   FlyDSL loads packed B through 128-bit register copies and B scales through
   32-bit register copies while staging A through LDS. It uses 32,896 B LDS and
   134 VGPR. Gluon's generic direct-B modes reduce LDS but regress performance:
   BN128 direct B measured 162.326/171.909 us mean at T16/T64, direct B+scale
   179.197/190.075 us, and BN256 direct B 162.475/168.658 us. The missing piece
   is therefore the specialized register lifetime/scheduling pipeline, not a
   boolean LDS bypass.
4. **Routing padding is not the residual cause.** With BN128 both finalists use
   the same 32x128x256 tile and the same 33 active M blocks / 1,056 padded rows
   as FlyDSL. The timed path also excludes sorting and layout conversion.

## Recommended solution order

1. Select B-payload NT-only for small-token/single-use regimes, but retain cached
   B when multiple M blocks per expert can reuse it.
2. Add the validated BN128/depth3/WPE2 small-token configuration; keep three
   stages because every two-stage screen point regressed substantially.
3. Build a dedicated FlyDSL-like direct-B pipeline with explicit packed
   128-bit B loads, tightly scheduled register rings, and 32-bit packed scale
   loads. Do not enable the current generic `B_IN_REG`/`B_SCALE_IN_REG` path as
   a production fix.
4. Independently investigate the BM32 A-scale sorted/shuffled restriction and a
   shared 32-row scale fill before attributing the entire residual to B; this is
   a plausible additional instruction-path reduction but is not yet benchmarked.

## Cache-artifact disclosure

A resource-audit command using `llvm-objcopy --dump-section .text=/dev/stdout`
unexpectedly normalized ELF section/string tables in 12 **untracked geometry
cache** HSACOs. No tracked source, finalist cache, screen result, saved
`gluon_artifact`, JSON, trace, or provenance file was changed. Raw function
bytes, `.text` hashes, AMDGPU metadata/resources, and generated `.amdgcn` files
remain unchanged. Per parent direction, these cache files were not restored.

Affected full-ELF SHA changes (T16 and T64 each):

| Cache case | Cache key | Original SHA256 | Current SHA256 |
|---|---|---|---|
| `g_bn256_d3_w1_t16`, `g_bn256_d3_w1_t64` | `CNNDSVKBHXZQD6UKRTQJYL7WQOKR6M6JZZQU3UOOMBBSXXW6O36Q` | `1a37e53eef4e68aff295fb5ce116ea3b93bdfe4b2676754a055eafc03069d9b8` | `ea8a30e7eb3da7fce56bbfdec54df2277ec6cbe0c899d2d31b6ef16f6faa69dc` |
| `g_bn128_d3_w1_t16`, `g_bn128_d3_w1_t64` | `456SUPN75ZOQKMOAK3YVENRAPFYWL2YNGYDZ4LLCRME4DEVE7CAQ` | `2316bcce53a67a0dbf6dbc9692d6ea32c8c88f9b8e4e7d0811b6e07a314be3ae` | `11ac027b8b888d4bc34b38b464f40da6056ce99763f83209dc460936a62d960c` |
| `g_bn128_d3_w2_t16`, `g_bn128_d3_w2_t64` | `N4SJ33EDCI73WJTJYEGTJCCN6J3TX7X5HUYZPLP4UFR6SSE3XLNQ` | `3a1da6376d3f0123ecd0f84a1e6fdfd6d4cdea5ca28968d57fefaa6687989832` | `2fcd41078d04bb8dbd261f84862c4aa16660f8a37960d61a8643396419f49ad1` |
| `g_bn128_bp_breg3_t16`, `g_bn128_bp_breg3_t64` | `ZE2Y6F7UKFAP4VDOKLB62Q36LXVIO2HUTNBWIHFO77B2GXBVDEOQ` | `79d1f6a0263320c9bc14c3a18d04cce54c3a2302a1334bd3dea476b138bd61ba` | `4c2d4c2017f8980b1f84bd345a9cbb3cde610132dc0c231cbcceb065fd264dae` |
| `g_bn128_bp_breg_bscale3_t16`, `g_bn128_bp_breg_bscale3_t64` | `MGDDV5FRWDZDOEOJBC7S3U3PJQ2P7ZYAMG4U3NTYUXDRQIHA5FZQ` | `8b38d83a7560498d27b7cdb06318541b410bb646fa85388df11449d6decfe455` | `de581ebad4a5756ede1bb28e88a6f2c4c7aa206cd7c0575019676e63fcbdfac2` |
| `g_bn256_bp_breg3_t16`, `g_bn256_bp_breg3_t64` | `IP3RXGUTC4BITINJJUUFY6J47YENNMUB7LM64GISC2T3SE6LIEBQ` | `7de90c6269a25a516d7c88fb60970cbfe582354bd8b410a18cfa037a036ae5ba` | `38067b55bbd1e2d180bc510244fea45d6ad487522aa17afef56cd663595d4f25` |

Each full path is
`/raid/jinpli/workspace/home01/jinpli/development/aiter/bench_out/a8w4_nt_geometry_20260910/cache/<case>/<cache-key>/_moe_gluon_gemm1.hsaco`.
