# Optimized Gluon MoE GEMM kernels for gfx950

Implement optimized Gluon MoE GEMM kernels for gfx950 (CDNA4 / MI355).

The kernels must cover real model usage — dimensions, dtypes and activations as recorded in
`/raid/jinpli/workspace/home/jinpli/development/moe-summary-dsv4-glm52-m3.md` (DeepSeek-V4-Flash,
DeepSeek-V4-Pro, GLM-5.2, MiniMax-M3). Test cases for those shapes landed in commit `83450d31f`:

- `op_tests/triton_tests/moe/moe_model_recipes.py` — single source of truth for shapes, swiglu
  parameters and router config.
- `op_tests/triton_tests/moe/test_moe_gemm_a4w4.py::test_model_shapes` — 16 cases
  (4 models × 2 stages × m ∈ {16, 1024}), already gfx950-gated.

Put the implementation under `aiter/ops/triton/_gluon_kernels/gfx950/moe` and wire it into the
existing interface at `aiter/ops/triton/moe/`.

Use Gluon named-tuple, aggregate and func/constexpr-func syntax to write kernels that are easy to
read and maintain while aiming at top performance (bandwidth for the decode kernel, TFLOPs /
MFMA utilization for the prefill kernel). A summary of the syntax is at
`/raid/jinpli/workspace/home/jinpli/development/triton/gluon-metaprogramming.md`; study the Triton
source repo (implementation and examples) for exact usage. The closest in-tree reference for house
style is `aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py` — copy its aggregate,
async-copy and layout idioms rather than inventing new ones.

## Scope and milestones

| Milestone | Content |
|---|---|
| **M1** | A4W4 MXFP4 (E2M1 packed 2/byte, group 32 along K, uint8 E8M0 scales), fp32 accumulate, both stages, fused MXFP4 output quant on gemm1, bias + gammas, non-persistent, prefill shapes. Gated on `test_model_shapes`. |
| **M2** | Decode path: `TILE_SCHED`, small-`BLOCK_M` tuning, N-tile fusion for L2 reuse. |
| **M3** | BF16 × BF16, then MXFP8 (E4M3 + E8M0/32). |
| **M4** | DeepSeek-V4 native A8W4 (fp8 activations block-128 × fp4 weights group-32). Needs
`gl.amd.cdna4.scaled_upcast` + plain `mfma`; **it is not expressible with `mfma_scaled`**, whose
CDNA4 implementation asserts `scale_factor == 32` and E8M0 scales. |

Out of scope: NVFP4 (group 16, E4M3 scales — **no hardware path on CDNA4**), hash routing for
DeepSeek-V4 layers 0–2, the router GEMM, the dense (non-MoE) layers 0–2 of GLM-5.2 / MiniMax-M3,
`SPLIT_K`, and the **shared expert**.

The shared expert is left alone: it stays on whatever path serves it today and does *not* ride this
grouped GEMM as an always-on expert `E`. Recorded here because it is not free — every one of the
four models has exactly one shared expert at full `I`, all three AMD repacks quantize it to MXFP4,
and on MiniMax-M3 it is 1 of 5 activated experts (~20% of MoE FLOPs). So the perf numbers this
kernel reports cover the routed experts only, and the end-to-end MoE layer time will not move by the
same factor.

## Key kernel design points

### Implement 2 kernels

- **kernel 1**: gemm1 + activation
- **kernel 2**: gemm2 + router-combine-weight multiply

The gemm1 kernel gathers tokens according to the routing metadata passed to it and writes its
output **contiguously** (dense, expert-sorted `[n_gates, I]`). The gemm2 kernel reads contiguously,
multiplies the result by the router combine weight (`gammas`), and writes contiguously. A third
existing kernel (`reduce_grouped`, `aiter/ops/triton/moe/reduce.py`) does the reduce/combine and is
outside our scope.

**One `@gluon.jit` body, two entry points.** Both kernels share `KernelFuncConfig`,
`KernelTuningConfig` and `LDSManager`; they differ only in the `KernelFuncConfig` instantiated
in-kernel. Note an `@gluon.aggregate` **cannot be a kernel argument** — the class is passed as a
`gl.constexpr` and instantiated inside the kernel from scalar constexprs.

#### Why two kernels, and what gemm1 must emit

The split is cheap at decode (weight traffic dominates: one GLM-5.2 expert's gemm2 weight strip is
`I*H/2` = 6.3 MB of MXFP4 against ~32 KB of intermediate per token) but not free at prefill. GLM-5.2
at T=8192, topk=8 gives an `M × I` intermediate of 65536 × 2048: in bf16 that is 268 MB written plus
268 MB read back, against ~4.8 GB of expert weights per layer — about 11%. Today's flow then inserts
a **third** full pass (`mxfp4_quant`, `aiter/ops/triton/moe/moe_op_gemm_a4w4.py:147-158`, which even
upcasts to fp32 first), i.e. 268 MB write + 268 MB read + 67 MB write + 67 MB read — roughly 2.5× the
necessary intermediate traffic.

**Therefore gemm1's epilogue emits exactly gemm2's operand-A format: MXFP4 E2M1 payload +
uint8 E8M0 scale, group 32 along the emitted N axis.** The emitted scale tensor must be
bit-identical to what `mxfp4_quant` produces today, so gemm2 needs no change and the wrapper drops
the standalone quant launch. Consequences, all of which are requirements:

- The MX group runs along gemm1's N (= gemm2's K), so the epilogue computes an **amax over 32
  emitted columns**. In an MFMA accumulator those 32 columns are spread across ~8 lanes, so this is
  a cross-lane reduction — **this is the reason `out_buf` / `store_res_lds` exist** (see
  "LDS Management"). It is the only thing that justifies LDS-staging the result on CDNA4, which has
  no direct-from-LDS global store (`async_copy.shared_to_global` exists on CDNA5/GFX1250, not CDNA4).
- `BLOCK_N % 64 == 0` in **raw, pre-halving** terms (32 emitted columns = 64 raw columns after the
  interleaved gate/up halving), so the MX group is always tile-local.
- `SPLIT_K == 1`, which is unconditional here anyway (see "No split-K").

#### Entry-point contract

Both kernels take the Family-A routing convention (`RoutingData` + `reduce_grouped`) — line 18's
"a third existing kernel will do reduce combine" already pins it, since `reduce_grouped` is the only
separate combine kernel in the tree. Write the routing metadata as a first-class NamedTuple:

```python
class RoutingMeta(NamedTuple):
    # packed (block_id << 16) | expt_id per launched pid; -1 means "no work, skip"
    expt_block_pid_map: gl.tensor
    expt_hist: gl.tensor          # int32 [n_expts_tot]
    expt_offs_raw: gl.tensor      # int32 [n_expts_tot + 1]
    expt_offs_sum: gl.tensor      # int32 scalar
    gather_indx: gl.tensor        # uint16 if n_gates <= 65535 else int32; divide by n_expts_act
    scatter_indx: gl.tensor       # may be None
    gammas: gl.tensor             # fp32 [n_gates]; may be None
    n_expts_act: gl.constexpr
    INDEX_DTYPE: gl.constexpr     # uint16 | int32
```

Grid: `grid_m = routing_data.n_blocks(M, block_m)` (host-side; **not** `cdiv`), `grid_n =
tuning_cfg.grid_N(N)`, `grid = grid_m * grid_n`. There is no split-K (see "No split-K"). Both
kernels must publish a verbatim Python signature next to their `@gluon.jit` definition, including
every stride and the `bias` / `gammas` pointers.

### Tensor dtype and quant method

Use an enum to represent tensor dtype and quant method. Members in scope:

| Member | Payload | Scale |
|---|---|---|
| `BF16` | bf16 | none |
| `FP8_E4M3` | fp8 e4m3 | none (unit scales) or block-128 |
| `MXFP4` | E2M1, 2 per byte, packed along K | uint8 E8M0, group 32 along K |
| `MXFP8` | E4M3 | uint8 E8M0, group 32 along K |

Live as an `IntEnum` in the new `gfx950/moe` package, carried as an int-valued `gl.constexpr`.
Note that `FP8 × FP8` must still route through `mfma_scaled` with `a_scale=None, b_scale=None` —
the backend folds the synthesized unit scales back into `V_MFMA_*_F8F6F4` and only that path reaches
the double-rate K=64/128 pipes; plain `mfma` selects the CDNA3-class K=16/32.

Aggregate each tensor and its quant scale as a named tuple. One common `gl.constexpr` field of the
tuple is `dtype_quant`; the remaining fields depend on it. The four variants are selected by a
**host-side branch** (four distinct launch sites), composed never subclassed — NamedTuple field
inheritance does not work in Triton (see `gluon-metaprogramming.md` §4, traps 1–3).

```python
class NonQuantTokenTensor(NamedTuple):
    dtype_quant: gl.constexpr
    ptr: gl.tensor
    num_token: gl.tensor          # runtime; do_not_specialize
    stride_m: gl.tensor
    hidden_dim: gl.constexpr      # logical extent
    topk: gl.constexpr


class QuantTokenTensor(NamedTuple):
    dtype_quant: gl.constexpr
    ptr: gl.tensor
    scale_ptr: gl.tensor
    num_token: gl.tensor
    stride_m: gl.tensor           # in payload elements, i.e. hidden_dim // 2 for MXFP4
    scale_stride_m: gl.tensor
    scale_stride_k: gl.tensor
    hidden_dim: gl.constexpr      # logical extent; packed extent derived from dtype_quant
    topk: gl.constexpr
    scale_swizzle: gl.constexpr   # NONE | CDNA4_SCALE


class NonQuantExpertTensor(NamedTuple):
    dtype_quant: gl.constexpr
    ptr: gl.tensor                # already re-based to the current expert, see below
    stride_e: gl.tensor
    stride_k: gl.tensor
    stride_n: gl.tensor
    num_expert: gl.tensor         # runtime: 128 / 256 / 384 across models
    hidden_dim: gl.constexpr
    fused_intermediate_dim: gl.constexpr   # 2*I for stage 1, I for stage 2


class QuantExpertTensor(NamedTuple):
    dtype_quant: gl.constexpr
    ptr: gl.tensor
    scale_ptr: gl.tensor
    stride_e: gl.tensor
    stride_k: gl.tensor
    stride_n: gl.tensor
    scale_stride_e: gl.tensor
    scale_stride_n: gl.tensor
    scale_stride_k: gl.tensor
    num_expert: gl.tensor
    hidden_dim: gl.constexpr
    fused_intermediate_dim: gl.constexpr
    scale_swizzle: gl.constexpr
```

#### Memory layout contract

Not simply "contiguous" — name the axis order, because the existing wrapper hard-asserts a
non-obvious one:

- **Activations**: row-major, `stride(-1) == 1`.
- **Weights**: `(E, K/2, N)` with **`stride(-2) == 1`** — `moe_op_gemm_a4w4.py:203-208` asserts
  "`w` must be column-major when it has data-type mxfp". MXFP4 packs 2 elements per byte along K, so
  the stored K extent is `hidden_dim // 2`; `hidden_dim` in the tuples above is always the **logical**
  extent.
- **Scales**: `(E, N, K/32)` uint8, strides carried explicitly. If `scale_swizzle == CDNA4_SCALE`
  the physical layout is the preshuffled one from `aiter/ops/triton/utils/shuffle.py:196-198`, which
  is not contiguous in the naive sense and **forces `BLOCK_K >= 256`**.
- **Per-expert re-basing is mandatory.** Buffer ops carry a 32-bit offset (2 GB window); V4-Pro's
  stacked gemm1 weight is ~8.5 GB. Re-base `ptr` per expert on the host (any single expert is 22 MB
  and fits), or select the 64-bit `gl.load` fallback via a `USE_BUFFER_LOAD: gl.constexpr` as in
  `gfx950/attention/pa_decode_sparse.py:26-34`. The repo already has `can_overflow_int32`
  (`moe_op_gemm_a4w4.py:23`) for the host-side check.

#### Stage orientation

`hidden_dim` and `fused_intermediate_dim` swap roles between the stages, and the same tuple type
serves both — state it explicitly rather than leaving it to be inferred:

| | N | K |
|---|---|---|
| stage 1 (gate/up) | `2*I` | `H` |
| stage 2 (down) | `H` | `I` |

Host precondition for stage 1: gate and up arrive **pre-fused and column-interleaved (even = gate,
odd = up)**, matching `_triton_kernels/moe/activations.py:21`
(`tl.split(tl.reshape(x, (M, N//2, 2)))`). Interleaving is free for MXFP4 since packing is along K.
`BLOCK_N` tiles the **raw `2*I`**; the epilogue halves it and stores `BLOCK_N/2` columns.

### Kernel Functionality Config

All fields of `KernelFuncConfig` are constexpr. Optional fields are `None` sentinels arriving as
`gl.constexpr(None)` — an aggregate has fixed fields, so "does not exist for gemm2" is not
expressible; `activation is None` is.

```python
@aggregate
@strip_annotate
class KernelFuncConfig:
    # token dtype and quant-scheme as stored in tensor
    token_dtype_quant: gl.constexpr
    # expert dtype and quant-scheme as stored in tensor
    expert_dtype_quant: gl.constexpr
    # MMA operand-A / operand-B representation. v1: format selection only, no in-kernel
    # quantization; expert_online_quant == expert_dtype_quant always. Selects mfma vs mfma_scaled.
    token_online_quant: gl.constexpr
    expert_online_quant: gl.constexpr
    # mma accumulate dtype (fp32 for every in-scope format)
    mma_acc_dtype: gl.constexpr
    # activation for gemm1; None for gemm2
    activation: gl.constexpr        # ActivationSpec | None
    # output quantization scheme; MXFP4 for gemm1 in M1, None for gemm2
    output_quant: gl.constexpr
    # per-expert fp32 bias of shape (E, N)
    has_bias: gl.constexpr
    # multiply by router combine weight (gemm2)
    has_gammas: gl.constexpr
```

`activation` is not a bare tag — the three models need three different parameterisations, and every
`moe_gemm_*` wrapper defaults to `alpha=1.0, limit=1.0, swiglu_add_residual=True`, which is no real
model's activation:

```python
class ActivationSpec(NamedTuple):
    kind: gl.constexpr          # SILU | SWIGLU_OAI
    alpha: gl.constexpr         # 1.0 (GLM, DSv4) | 1.702 (M3)
    limit: gl.constexpr         # None (GLM) | 10.0 (DSv4) | 7.0 (M3)
    add_residual: gl.constexpr  # False | False | True
```

Clamp and multiply happen in **fp32, before the cast to the output dtype**. With a transposed MFMA
accumulator each lane holds 4 consecutive N elements, so an even/odd gate-up pair is lane-local and
the activation is a pure register operation.

**Fixed epilogue order**, matching `_triton_kernels/moe/moe_op_gemm_a4w4.py:463-497`:

> bias (fp32, expert-indexed, only when `pid_k == 0`) → activation → gammas → output_quant → store

Bias and gammas are not optional in practice: `test_moe_gemm_a4w4.py:91,348` allocates and passes
both unconditionally for all 16 model cases.

### Kernel Tuning Config

All fields of `KernelTuningConfig` are constexpr. It holds the func config, so the layout methods
can see the operand dtypes (an aggregate may hold another aggregate as a typed field — see
`MQAAsyncKVLoader.kv_cfg` in `gfx950/attention/fp8_mqa_logits.py:190`).

```python
@aggregate
@strip_annotate
class KernelTuningConfig:
    func_cfg: KernelFuncConfig
    # tiling of token dim; fixed by the moe router, see below
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    # unroll factor of the OUTER (inter-BLOCK_K) k-loop; see "The K loop"
    K_UNROLL: gl.constexpr
    # Within BLOCK_K, run an unrolled loop, each iter of which issues MINI_BLOCK_K amount of
    # lds-load + mma
    MINI_BLOCK_K: gl.constexpr
    # Prefetch amount of the MINI_BLOCK_K loop
    MINI_PREFETCH_K: gl.constexpr
    # tiling of BLOCK_M / BLOCK_N for the final streaming write of results
    MINI_BLOCK_M: gl.constexpr
    MINI_BLOCK_N: gl.constexpr
    # Number of mini (MINI_BLOCK_M, MINI_BLOCK_N) tiles to stage in lds in advance.
    # This is the knob that bounds live accumulator registers.
    MINI_PRESTORE_MN: gl.constexpr
    # number of lds buffers to use, determines pipeline depth
    NUM_LDS_BUFFER: gl.constexpr
    # MFMA shape. NUM_WARPS is derived as prod(warps_per_cta), not carried separately.
    mfma_instr_shape: gl.constexpr   # [32,32,64] or [16,16,128] for the CDNA4 f8f6f4 pipes
    warps_per_cta: gl.constexpr
    tiles_per_warp: gl.constexpr
    k_width: gl.constexpr
    transposed: gl.constexpr         # pin True, see "dot_result_fragment_layout"
    # num of waves per EU (also a launch option, must be duplicated at the launch site)
    WAVES_PER_EU: gl.constexpr
    # tile scheduler: XCD swizzle policy
    TILE_SCHED: gl.constexpr
    # cache modifiers
    token_mod: gl.constexpr
    token_scale_mod: gl.constexpr
    expert_mod: gl.constexpr
    expert_scale_mod: gl.constexpr
    result_mod: gl.constexpr
    result_scale_mod: gl.constexpr
    # warp pipelining. NOT a placeholder: the mechanism is
    #   with gl.amd.warp_pipeline_stage(label, priority): ...
    WARP_PIPELINE: gl.constexpr

    @gluon.constexpr_function
    def grid_N(self, N):
        # Callable from BOTH host and device (verified on gfx950). Host: for the grid tuple,
        # take .value off the returned constexpr. grid-M comes from the routing metadata.
        ...

    @gluon.constexpr_function
    def dot_result_fragment_layout(self) -> gl.constexpr:
        # AMDMFMALayout for dot/scaled-dot results in register.
        # Pin transposed=True: it gives each lane 4 consecutive N elements (a 16 B contiguous
        # store) instead of 4 strided M rows.
        ...

    @gluon.constexpr_function
    def dot_result_lds_layout(self) -> gl.constexpr:
        # layout for dot results staged in LDS. Only used when output_quant needs the
        # cross-lane amax; None otherwise.
        ...

    @gluon.constexpr_function
    def dot_operand_fragment_layout(self, idx: gl.constexpr) -> gl.constexpr:
        # DotOperandLayout for operands in register. idx == 0 is LHS, 1 is RHS.
        ...

    @gluon.constexpr_function
    def dot_operand_lds_layout(self, idx: gl.constexpr) -> gl.constexpr:
        # Use gl.amd.cdna4.compute_efficient_padded_shared_layout(dot_layout, shape, dtype,
        # is_k_contig) and handle its documented None return (unsupported k_width / instr shape).
        ...

    @gluon.constexpr_function
    def dot_operand_scale_fragment_layout(self, idx: gl.constexpr) -> gl.constexpr:
        # Use gl.amd.cdna4.get_mfma_scale_layout(dot_operand_layout, shape, 32).
        # Only valid when the operand is of quant dtype.
        ...

    @gluon.constexpr_function
    def dot_operand_scale_lds_layout(self, idx: gl.constexpr) -> gl.constexpr:
        # layout for scaled-dot operand scales in LDS. See the scale discussion below.
        ...
```

#### BLOCK_M is an input, not a knob

`BLOCK_M == routing_data.block_m` — assert it. The router picks
`max(16, min(next_pow2(M // n_expts_tot), 128))` (`moe_routing/routing.py:305-307`), so the kernel
must serve `block_m ∈ {16, 32, 64, 128}`. The tuner keys on `(block_m, N, K, dtype)` and never
overrides it.

#### TILE_SCHED

v1: a `gl.constexpr` enum selecting an XCD-swizzle policy over the existing free functions
`remap_xcd(pid, GRID_MN, NUM_XCDS)` and `pid_grid(...)`
(`aiter/ops/triton/utils/_triton/pid_preprocessing.py:27,57`). There is no reusable tile-scheduler
class in the repo to copy. Constraints:

- It **may not permute across expert boundaries** — an M-tile belongs to exactly one expert.
- A grid-persistent mode forfeits the `expt_block_pid_map == -1` early return, which is the cheap
  way to drop empty blocks. Persistent is out of scope for M1.

#### The K loop

`K_UNROLL` unrolls the **outer, inter-`BLOCK_K`** loop; `MINI_BLOCK_K` is the inner static unroll
*within* one `BLOCK_K`. They are orthogonal. The loop is a four-part decomposition:

```python
prologue                                # pipeline fill: NUM_LDS_BUFFER-1 stages, no mma
for k in range(..., ..., K_UNROLL):     # steady state, body unrolled K_UNROLL times
    ...
for k in range(..., ..., 1):            # remainder (0 .. K_UNROLL-1 iterations)
    ...
epilogue                                # NUM_LDS_BUFFER-1 mma-only iterations, no fill
```

- **No divisibility requirement** between `cdiv(K, BLOCK_K)` and `K_UNROLL` — the step-1 loop
  absorbs the remainder, so `K_UNROLL` is a free knob.
- The payoff is the **constexpr rotating buffer index**. In-tree the steady-state index is dynamic —
  `w_buffer.index(write_idx % NUM_BUFFERS)`
  (`_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:554-558`); only the prologue and epilogue use
  `gl.static_range`. Inside the unrolled body `(k_base + i) % NUM_LDS_BUFFER` folds to
  `i % NUM_LDS_BUFFER` **iff `K_UNROLL % NUM_LDS_BUFFER == 0`** (naturally `K_UNROLL ==
  NUM_LDS_BUFFER`), turning the descriptor index into a static offset and constant-folding the
  `wait_group` counts and the address arithmetic. Enforce it as a constexpr assert, not a
  convention — without it the unroll buys nothing but code size.
- The step-1 remainder loop and the drain epilogue run at an arbitrary buffer phase and therefore
  use a dynamic `.index()`. That is fine; they are off the critical path, and the alternative
  (`gl.static_range(K_UNROLL)` with an exit predicate) is pure code bloat.
- `MINI_PREFETCH_K > 0` needs the *next* stage's buffer index statically, so it implies
  `K_UNROLL >= 2`.
- Cost to bound: the fully unrolled body is `K_UNROLL × (BLOCK_K / MINI_BLOCK_K)` mma groups. That
  is the I-cache and live-range pressure knob — budget it against the VGPR target below.
- The M-axis mask is a separate matter: the ragged last block of each expert needs operand-A masking
  in **every** body, unrolled or not.

**K divisibility is a host precondition, not a kernel branch.** Assert `K % BLOCK_K == 0` on the
host and assume it in the kernel: **only the even-K case is implemented**. There is no masked tail
body, no `EVEN_K` constexpr, and no K-mask computed anywhere — the same treatment `N % BLOCK_N == 0`
gets (see "Edge cases"). The wrapper must fall back to the Triton kernel rather than mis-compute if
a caller ever presents a non-divisible K.

The assert is satisfiable for every in-scope shape: stage-1 `K = H ∈ {4096, 6144, 7168}` and stage-2
`K = I ∈ {2048, 3072}` are all multiples of 1024, so any `BLOCK_K ∈ {128, 256, 512}` divides them —
and `CDNA4_SCALE` already forces `BLOCK_K >= 256`.

This removes the fill/consume masking hazard rather than solving it, but the **drain still has to
exist**. Under an `NUM_LDS_BUFFER`-deep pipeline the fill for a K-tile is issued `NUM_LDS_BUFFER-1`
iterations before it is consumed, so the last `NUM_LDS_BUFFER-1` stages are consumed by an
mma-only epilogue that issues **no fill at all** — that, not masking, is what keeps the loads inside
`K`. Getting this wrong reads past the end of the K strip while every correctness case still passes,
because the over-read lands in the next expert's weights and is multiplied by an accumulator that is
never stored. Assert the fill count equals `cdiv(K, BLOCK_K)` in the constexpr layer.

Divisibility lattice for the rest: `BLOCK_K % MINI_BLOCK_K == 0`; `MINI_BLOCK_K % mfma_k == 0`
(`mfma_k` = 64 or 128 for the CDNA4 scaled f8f6f4 pipes, 16/32 for bf16);
`MINI_PREFETCH_K < BLOCK_K / MINI_BLOCK_K`;
`MINI_PRESTORE_MN <= NUM_MINI_BLOCK_M * NUM_MINI_BLOCK_N`; and `BLOCK_K >= 256` whenever
`scale_swizzle == CDNA4_SCALE`.

#### Resource budgets

Both must be constexpr asserts, not comments.

**LDS** — cap is 160 KiB on gfx950 (`aiter/ops/triton/utils/_triton/arch_info.py:35`,
`_LDS_CAP_BYTES["gfx950"] = 163840`):

```
NUM_LDS_BUFFER * (A + A_scale + B + B_scale) + out_buf + out_scale_buf <= 163840
```

Worked example, BLOCK_M=128 / BLOCK_N=256 / BLOCK_K=256 MXFP4: A = 16 KiB, A_scale = 1 KiB,
B = 32 KiB, B_scale = 2 KiB → **51 KiB per stage**, so 3 buffers already take 153 KiB. A *full-tile*
`out_buf` would be another 64 KiB bf16 / 128 KiB fp32 and does not fit — which is exactly why the
mini-blocks exist. At MINI_BLOCK_M=128, MINI_BLOCK_N=32 emitted, fp32, `out_buf` is 16 KiB and fits
with `NUM_LDS_BUFFER=2`. `pick_gemm_num_stages(..., use_async_padding=True)`
(`aiter/ops/triton/utils/gemm_config_utils.py:294`) already does this arithmetic and can be reused.

**VGPR** — nothing in the design is register-costed today, yet three knobs are register knobs. The
fp32 accumulator alone is `BLOCK_M * BLOCK_N / (NUM_WARPS * 64)` VGPR/lane = 64 at
128×256 with 8 warps, 128 at BLOCK_N=512. `MINI_PREFETCH_K` holds a tuple of dot-operand fragments
live across the mini-K loop on top of that, and `WAVES_PER_EU = 2` is only reachable at ≤256 VGPR.
State a target VGPR/lane and waves/EU per regime, and note that `MINI_PRESTORE_MN` exists precisely
to drain accumulator registers early so the next tile can start.

### LDS layout goals

The LDS layouts should be designed with the following goals:

- for global→LDS load, use `gl.amd.cdna4.async_copy.buffer_load_to_shared` at dwordx4 width
- load directly from LDS into the fragment layout, no `convert_layout`
- maximize LDS load vector width

**These three are not simultaneously satisfiable for the raw E8M0 scale operands**, so resolve them
per operand:

- **Data operands (fp4/fp8/bf16)**: all three goals are reachable. Use
  `gl.amd.cdna4.compute_efficient_padded_shared_layout(...)`. Under the layout contract above both
  operands are already K-packed, so the plain `smem.load(dot_layout)` /
  `async_copy.load_shared_relaxed` path applies;
  `gl.amd.cdna4.load_shared_fp4_repacked(mem_desc, layout)` is only needed if a checkpoint ever
  arrives M/N-packed, and should not be reached for the four models in scope.
- **E8M0 scales**: `get_mfma_scale_layout` gives each lane K-scale elements strided by 2 (32×32) or
  4 (16×16). Read straight from an identity LDS tile that is one `ds_read_u8` per element. Worse,
  a `[BLOCK_N, BLOCK_K/32]` uint8 tile needs `BLOCK_K >= 512` for 16 B/lane, drops to the 32-bit
  path at `BLOCK_K ∈ {128, 256}`, and at `BLOCK_K = 64` **cannot be lowered at all** — CDNA4's
  direct-to-LDS path supports only 128-bit or 32-bit per lane. Resolution: consume the existing
  `CDNA4_SCALE` preshuffle (`utils/shuffle.py:196-198` + `unswizzle_mx_scale_cdna4`), which keeps the
  direct-to-LDS write coalesced and moves the reordering into a free descriptor reshape/permute, at
  the cost of `BLOCK_K >= 256`. Alternative if the preshuffle is not adopted: hoist the full-K scale
  strip for the tile into LDS once (`BLOCK_N * K / 32` bytes) and accept `ds_read_u8`.

Note the ordering hazard: on CDNA4 `buffer_load_to_shared` completes **in order with ordinary
`load` / `store` / `buffer_load` / `buffer_store`**. Any register-path global access inside the
K-loop (gather indices, expert metadata, epilogue stores) makes every `wait_group` conservative.
Hoist them out of the loop.

#### Verifying the zero-cost claims

All three goals are claims about generated code, and **no correctness test can detect them
failing** — a silent fall back from 128-bit to 32-bit direct-to-LDS, or an inserted
`convert_layout`, passes all 16 model cases while losing most of the performance. Required:

- constexpr asserts: `gl.bank_conflicts(distr_ty, shared_ty) == 0`,
  `compute_efficient_padded_shared_layout(...) is not None`, per-lane width == 128 bits.
- one ISA-assertion test per kernel grepping the AMDGCN for `buffer_load_dwordx4 … lds` and for the
  **absence** of `ds_bpermute` / `v_permlane` inside the K-loop, mirroring upstream
  `python/test/gluon/test_core.py:1622-1655`.

You may need the syntax of Tuples — the fixed-length array of values — to implement the static
unroll of mini `BLOCK_K` and to hold the prefetched fragments.

### LDS Management

Use an aggregate `LDSManager` to handle allocation and loading of LDS. Model it on
`MQAAsyncKVLoader` in `gfx950/attention/fp8_mqa_logits.py:186-290`, which is the same shape.

```python
@aggregate
@strip_annotate
class LDSManager:
    func_cfg: KernelFuncConfig
    tuning_cfg: KernelTuningConfig
    token_buf: gl.shared_memory_descriptor
    # None if the token dtype carries no scale
    token_scale_buf: gl.shared_memory_descriptor | gl.constexpr
    weight_buf: gl.shared_memory_descriptor
    # None if the expert dtype carries no scale
    weight_scale_buf: gl.shared_memory_descriptor | gl.constexpr
    # non-None only when output_quant needs the cross-lane amax (i.e. gemm1 in M1)
    out_buf: gl.shared_memory_descriptor | gl.constexpr
    out_scale_buf: gl.shared_memory_descriptor | gl.constexpr

    @gluon.jit
    def alloc(func_cfg, tuning_cfg):
        # static factory, invoked as LDSManager.alloc(...) — no self, by design.
        ...
        return LDSManager(...)

    @gluon.jit
    def load_a_frag(self, idx, mini_idx: gl.constexpr):
        # idx is the buffer index (runtime), mini_idx the constexpr mini-BLOCK_K index
        ...
        # a_scale_val is None when the token dtype carries no scale
        return a_val, a_scale_val

    @gluon.jit
    def load_b_frag(self, idx, mini_idx: gl.constexpr):
        ...
        return b_val, b_scale_val

    @gluon.jit
    def fill_a_lds(self, idx, a: QuantTokenTensor, a_coord, mini_idx: gl.constexpr):
        # issue buffer_load_to_shared for token and token scale. The scale coordinate is derived
        # from a_coord. Must pass mask=/other= for the ragged tail (see "Edge cases").
        ...

    @gluon.jit
    def fill_b_lds(self, idx, b: QuantExpertTensor, b_coord, mini_idx: gl.constexpr):
        # same for expert / expert-scale
        ...

    @gluon.jit
    def commit_fill_lds(self):
        # exactly one gl.amd.cdna4.async_copy.commit_group() per pipeline stage, issued after
        # BOTH fill_a_lds and fill_b_lds. Without this the group count is silently 2 per stage.
        ...

    @gluon.jit
    def wait_fill_lds_num_buf(self, num_buf: gl.constexpr):
        # thin wrapper over async_copy.wait_group(num_buf): block until the number of outstanding
        # COMMIT GROUPS is <= num_buf. One commit group == one pipeline stage.
        ...

    @gluon.jit
    def reshape_result(self, res):
        # split the [BLOCK_M, BLOCK_N] accumulator into a tuple of
        # NUM_MINI_BLOCK_M * NUM_MINI_BLOCK_N tiles of shape [MINI_BLOCK_M, MINI_BLOCK_N],
        # as a register-only view: no cross-lane data movement, no instruction emitted.
        # Implemented with gl.amd.slice, NOT reshape/permute/split — see below.
        ...

    @gluon.jit
    def store_res_lds(self, res_mini_blocks: gl.tuple,
                      mini_m_idx: gl.constexpr, mini_n_idx: gl.constexpr):
        # stage one mini-block in LDS so the epilogue can read it back grouped by 32 emitted
        # columns for the MX amax. Followed by gl.barrier().
        ...

    @gluon.jit
    def store_res_hbm(self, res_ptr, res_scale_ptr, bias_ptr, gammas_ptr, res_coord,
                      res_mini_blocks: gl.tuple,
                      mini_m_idx: gl.constexpr, mini_n_idx: gl.constexpr):
        # apply the epilogue in order (bias -> activation -> gammas -> output_quant) and
        # buffer_store the mini-block. The scale coordinate is derived from res_coord.
        ...
```

Three corrections to the earlier draft of this API, all forced by what CDNA4 and Gluon actually
provide:

**`wait_store_res_lds` is deleted.** There is no counted wait on LDS stores anywhere in Gluon — the
only shared-store op is `smem.store()` and the only synchronization is the full block barrier
`gl.barrier()`. ds_write counts are not predictable after scheduling anyway. Use `gl.barrier()`
inside `store_res_lds`.

**`wait_fill_lds_num_buf` counts commit groups, not buffers.** There is no "wait tensor count" on
CDNA4; `wait_group(n)` waits until at most `n` *committed groups* remain outstanding, lowering to
`ROCDL::WaitAsyncmarkOp` from which LLVM derives vmcnt. So the abstraction only holds if all loads
of a stage are issued back to back followed by exactly one `commit_group()`. Note the number of
loads per stage is **dtype-dependent** (bf16 has no scale tensors, so it is 2 not 4) — derive the
group from `KernelFuncConfig` rather than hardcoding 4.

**`reshape_result` must use `gl.amd.slice`.** Gluon's `reshape` / `permute` / `split` do not accept
a layout — every result goes through `_wrap_tensor_infer_layout` — and `gl.split` only splits a
trailing axis of size 2. The primitive that does what is wanted is
`gl.amd.slice(source, shape, offsets)`: a register-only, layout-preserving view. Its verifier
requires the lane and warp dim bases to match between source and destination, which means:

```
MINI_BLOCK_M % (instr_shape[0] * warps_per_cta[0] * tiles_per_warp[0]) == 0
MINI_BLOCK_N % (instr_shape[1] * warps_per_cta[1] * tiles_per_warp[1]) == 0
BLOCK_M % MINI_BLOCK_M == 0    and    BLOCK_N % MINI_BLOCK_N == 0
```

With 32×32 MFMA and `warps_per_cta=[4,1]` that pins `MINI_BLOCK_M` at 128 and steps `MINI_BLOCK_N`
in 32s. Non-divisible values are **illegal, not merely wasteful**, so implement it as a
`gl.static_range` over `gl.amd.slice` with the above as constexpr asserts.

Note the mini M/N block machinery is **not** optional under the fused-quant decision: with
`MINI_BLOCK_M = BLOCK_M` and `MINI_BLOCK_N = BLOCK_N` a full-tile `out_buf` does not fit in LDS
alongside the operand buffers (see the budget above). If you need a simplification to get a first
kernel running, drop `output_quant` (emit bf16, keep the standalone `mxfp4_quant` launch) and set
`out_buf = None` — then, and only then, `MINI_BLOCK_* = BLOCK_*` is viable.

### Edge cases

The load/store methods take coordinates but no masks; the boundary rules must be explicit.

- **Ragged M**: the last block of each expert is partially filled. `buffer_load_to_shared` takes
  `mask=` / `other=`; hardware OOB via the buffer descriptor is the cheaper route where the
  coordinate can be clamped.
- **Empty experts / padded pids**: `expt_block_pid_map == -1` signals no work
  (`_gluon_kernels/gfx1250/moe/moe_op_gemm_a8w4.py:295-297`). The early return must not leave
  uncommitted async groups outstanding.
- **N tails: none.** Assert `N % BLOCK_N == 0` on the host and assume it in the kernel — the store
  is an unmasked full-`BLOCK_N` store into an exactly-N buffer, and no N-mask is ever computed. The
  assert is satisfiable for every in-scope shape: stage-1 `N = 2I ∈ {4096, 6144}` and stage-2
  `N = H ∈ {4096, 6144, 7168}` are all multiples of 1024, so any `BLOCK_N ∈ {128 … 1024}` divides
  them. The wrapper must refuse (fall back to Triton) rather than silently mis-store if a future
  shape violates it.
- **K tails: none.** Assert `K % BLOCK_K == 0` on the host, same as N. Only the even-K case is
  implemented; the pipeline drain, not a mask, is what keeps loads inside `K` (see "The K loop").
- **Index dtype**: `gather_indx` is uint16 when `n_gates <= 65535`, else int32
  (`moe_routing/routing.py:98-101`), and is divided by `n_expts_act`.
- **2 GB buffer window**: per-expert re-basing, or the `USE_BUFFER_LOAD` / 64-bit fallback.

### Acceptance criteria

- **Gate**: `op_tests/triton_tests/moe/test_moe_gemm_a4w4.py` — reference `moe_gemm_torch`,
  maxtol 2e-2 / rmstol 4e-3, `do_gather = do_scatter = has_y_gammas = True`, swiglu on stage 1 only.
- Two consequences of arch-gating inside the existing wrapper: **every other case in that file** also
  starts running on the gluon kernel, so either they all pass or the wrapper needs an explicit
  capability predicate; and the gfx1250 a8w4 exemption that skips "swiglu AND gammas together"
  (`test_moe_gemm_a8w4.py:316-317`) is **not** available here — all 16 model cases use both.
- Fused MXFP4 output quant has no existing test. The nearest precedent, a8w4's `out_mx_quant`, emits
  MXFP8 only and is gated on `scatter_indx is None` (`moe_op_gemm_a8w4.py:405-424`). Add a test that
  compares gemm1's emitted payload+scale against the standalone `mxfp4_quant` bit-for-bit.
- `test_moe_model_recipes.py` numerics (the three activations) must keep passing.

### Performance targets

**The target is to beat the tuned FlyDSL / HIP MoE implementation.** Not a %-of-peak number: the
bar is an existing, tuned kernel on the same shapes, which is a harder and less gameable target.

The FlyDSL path is the right comparison because it has the same decomposition — two stages,
separately compiled and separately tuned: `compile_flydsl_moe_stage1` / `compile_flydsl_moe_stage2`
(`aiter/ops/flydsl/moe_kernels.py:547,663`), with the mxfp4 stage-2 gfx950 optimization from commit
`a43fe2589` and tuned configs under `aiter/configs/model_configs/*_a4w4_tuned_fmoe.csv`. The HIP
comparison is `fused_moe` (`aiter/fused_moe.py`).

Measurement:

- `op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4.py --model {dsv4-flash,dsv4-pro,glm52-base,
  minimax-m3}` already fills shapes from `moe_model_recipes.py` and computes a Proton roofline.
- decode T ∈ {1, 8, 32} and prefill T ∈ {1k, 4k, 16k}, all four models, **both stages reported
  separately** — stage 1 and stage 2 have different bottlenecks and FlyDSL tunes them separately.
- report the current Triton a4w4 kernel alongside, so the regression risk on the fallback path is
  visible.

Two things to settle before the numbers mean anything:

- The tuned fmoe CSVs in-tree cover **Kimi-K2**, not these four models
  (`aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv` is the only a4w4 tuned fmoe file). Either
  tune FlyDSL for the four target shapes first, or state explicitly that the baseline is running
  untuned/interpolated — beating an untuned baseline proves nothing.
- The shared expert is out of scope here (see Scope) but is inside some end-to-end FlyDSL paths.
  Compare routed-expert GEMM time against routed-expert GEMM time, not against a fused MoE layer.

### Decode vs prefill

The two stated objectives are decode bandwidth and prefill MFMA utilization, but the only
decomposition given is by MoE stage. State whether one kernel serves both regimes (an
`IS_DECODE: gl.constexpr`) or two exist, and enumerate what differs: N-tile fusion for L2 reuse,
`NUM_LDS_BUFFER`, `K_UNROLL`, MFMA instruction shape.

The decode regime is qualitatively different and the doc should say so. `BLOCK_M` is pinned at 16 by
the router, and `grid_m = routing_data.n_blocks(M, block_m)` **returns `M` whenever
`M <= n_expts_tot`** (`moe_routing/routing.py:69-77`). DeepSeek-V4-Pro at T=16, topk=6 gives M=96
gate rows against 384 experts → 96 blocks of `BLOCK_M=16` holding 96 rows in total, i.e. **~6%
M-occupancy**. The pathology is intra-block M padding, not empty blocks, and there is no
MFMA-utilization story at all: it is a weight-streaming, bandwidth-bound skinny GEMM over K=7168
whose levers — split-K being off the table — are N-tile fusion and minimizing per-tile scalar
metadata loads.

### No split-K

`SPLIT_K` is not a knob and does not appear in `KernelTuningConfig`; the kernel is always
`split_k == 1`. `moe_op_gemm_a4w4.py:79` already hardcodes it today, so nothing is lost.

Two consequences to keep in mind rather than rediscover:

- The fused activation depends on it. When `split_k > 1` the existing wrapper moves the swiglu out of
  the GEMM and into `reduce_grouped` (`moe_op_gemm_a4w4.py:225-229`), because partial sums cannot be
  activated. Fusing the activation into gemm1's epilogue therefore forecloses ever turning split-K
  back on for this path — that is an accepted trade, not an oversight. Fused output quant has the
  same dependency, and more strongly: it needs the final value to compute the group amax.
- The output-buffer contract still carries the leading axis. `matmul_shape = (split_k, M,
  N // reduction_n_matmul)` (`moe_op_gemm_a4w4.py:61`) and `reduce_grouped` indexes accordingly, so
  gemm1/gemm2 must still write into a `(1, M, N)` buffer — do not silently drop the axis.

The wrapper must fall back to the Triton kernel if a caller ever asks for `split_k > 1`.

### Wiring

`aiter/ops/triton/moe/__init__.py` is **0 bytes** — there is no export surface to wire into; callers
import dotted module paths. There is also no per-stage op: `moe_gemm_a4w4` serves both stages,
distinguished only by its arguments. So the two-kernel split is expressed as *one* wrapper choosing
between two gluon kernels. Specify:

- which wrapper file is edited (`moe_op_gemm_a4w4.py`, or a new `moe_op_gemm_a4w4_gluon.py`);
- the arch gate — three incompatible idioms exist in-tree: unconditional import plus
  `use_gluon = get_arch() == "gfx950"` (`moe_op_gemm_a8w4.py:10-18,349`), lazy import, and the
  defensive `try/except ImportError -> None` of `moe/reduce.py:7-16`. Pick one;
- an explicit `_gluon_supported(dtypes, apply_swiglu, split_k, out_quant, swizzle, N, K)`
  predicate — it must refuse `split_k > 1`, `N % BLOCK_N != 0` and `K % BLOCK_K != 0` — with
  the existing Triton kernel as the fallback for every unsupported combination, so no existing
  caller silently changes behaviour. Do not repeat the a8w4 precedent of a hard assert
  (`moe_op_gemm_a8w4.py:351-353`).

### Tuning config plumbing

`aiter/ops/triton/configs/gfx950/gluon/moe/` exists and is **empty**. Three incompatible mechanisms
exist in-tree and none is MoE-capable and modern: a hardcoded Python heuristic
(`moe_op_gemm_a4w4.py:71-116`), a `bm{block_m}_n{n}_k{k}` JSON (`moe_op_gemm_a8w4.py:30-37,98`), and
the nested resolver hardcoded to the `gemm/` subtree (`utils/gemm_config_utils.py:63-70`). Specify:

- where tuned configs live (extend the nested resolver to `{arch}/gluon/moe/{dtype}/`, or start with
  an explicit Python ladder);
- the host→device handoff, which is **verified on gfx950**, not assumed (probe run on an MI350X
  against this Triton):
  - host-constructing an all-constexpr aggregate works, and the *same*
    `@gluon.constexpr_function` method runs there — `TuningCfg(128, 3).grid_N(6144)` returns
    `constexpr[48]`. Take `.value` before putting it in a grid tuple.
  - passing that instance as a kernel argument is rejected:
    `TypeError: failed to specialize argument of type: TuningCfg`.
  - passing the **class** as a `gl.constexpr` and instantiating in-kernel works, and `grid_N` called
    on device returns the same 48.

  So one method definition serves both sides and the config is simply **constructed twice**: once on
  the host for the grid tuple, once in-kernel from the same scalar constexprs. No plain-Python
  mirror of the grid math is needed — keep the arithmetic in the aggregate so the two cannot drift.
- `NUM_WARPS` / `WAVES_PER_EU` are launch options and must also be passed as launch kwargs;
- `do_not_specialize` for `num_token` and every other dynamic scalar — this is the house pattern
  (`gfx1250/moe/moe_op_gemm_a8w4.py:100-103`) and the fix in commit `760a07977`
  ("Remove BATCH_NUM constexpr type and add do_not_specialize clause to fix Triton cache flooding").

### Conventions

Copy from the sibling `gfx950/attention` kernels: SPDX header, module docstring,
`_leading_underscore` kernel names, `make_kernel_repr` + `@gluon.jit(repr=)`, `launch_metadata`,
`__init__.py` in the new package, and `@aggregate` + `@strip_annotate` on every aggregate
(`from triton.language.core import _aggregate as aggregate`;
`from aiter.ops.triton.utils.common_utils import strip_annotate`).
