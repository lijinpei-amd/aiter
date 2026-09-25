# KDA Direct intj Launch Design

## Goal

Remove `_tensor_key` and `_FastLaunch` from FlashKDA. Single-config Triton kernels and both gfx950 Gluon kernels must launch through bound `intj.make_launcher` functions. Keep decorated Triton launches for K1/K2 whenever their tuner has multiple configs. Preserve KDA's route choices, kernel results, graph-capture behavior, and autotuning.

The current wrapper caches a second, Python-level approximation of Triton's specialization key. It binds keyword arguments and performs a Python dictionary lookup on each hit; a miss launches through Triton before installing a native launcher. KDA allocates new workspaces and outputs for each invocation, so a cached argument tuple cannot safely hold tensors from a previous call.

## Existing capability and Gluon boundary

intj already accepts a Gluon `JITFunction` and compiles it with `GluonASTSource`; `tests/test_launcher.py::test_gluon_launch_matches_triton` exercises that route. These two KDA kernels fail for more specific reasons:

- `k1_prepare_gluon` reads 12 module-level layout objects. `intj._check_kernel` rejects any captured global because intj cannot revalidate it on subsequent launches.
- `k2_ab_fused_gluon` takes eight layout objects as `constexpr` launch arguments. intj's baked-value identity and native decoder cover scalars, strings, dtypes, and JIT functions, but not layout objects.

Construct the same layouts as local `gl.constexpr` values inside each Gluon JIT function. K1 uses its fixed two-warp layouts. K2 derives its layouts from `gl.num_warps()`, while the host still selects `num_warps=2` or `4` with `_k2_gluon_schedule`. Remove K2's `build_layouts` call and eight layout parameters. Keep the global-value refusal and intj's constant decoder unchanged. A compile-only transformation of the actual K1 and both K2 schedule variants passed for offline gfx950 on Triton 3.7.1 and 3.8.0; GPU correctness and speed remain acceptance checks.

## Launch contract

Every native handle uses `grid_arg=2`, `bind_device=True`, and `return_compiled=False`. The current raw stream, two integer grid dimensions, and current kernel arguments are passed positionally on every invocation. A small cached factory in `flash_kda.py` owns handles by raw JIT function, device ordinal, compile options, and baked constant values; it retains no input or workspace tensors. Create a handle under `torch.cuda.device(device)` and allow intj to own the dynamic tensor/scalar specialization cache inside that handle. Before the first launch, require every non-null input pointer tensor to be on `q.device` and require that GPU to be current. All five native calls originate in `flash_kda_fwd`; remove the unused Gluon host wrappers. Keep that device current throughout the launch sequence. intj rejects a bound-device mismatch on a compile miss but not necessarily on a cache hit, and its direct tensor decoder does not reject an inaccessible host tensor pointer. The previous Triton first-call route did reject inaccessible pointers.

| Kernel | Raw JIT and grid | Fixed choices and arguments |
| --- | --- | --- |
| Triton K1 | `_flash_kda_prepare_kernel.fn.fn`; `(total_tiles if varlen else NT, B * H)` | A single-config tuner supplies `num_warps` and `num_stages`. Pass `IS_VARLEN`, `HAS_BIAS`, `NUM_DOUBLING`, and `NUM_MERGE` explicitly. Bake `CM_QKG=".cg"` and `CM_WS=""` or `".wt"`. |
| Triton K2 | `_flash_kda_segment_kernel.fn`; `(ceildiv(W, BW), num_segs * H)` | A single-config tuner supplies `BW`, `num_warps`, and `num_stages`; bake `BW` so grid and binary use the same tile. Bake `CM_OUT=""` or `".cs"`. Pass the current buffers and flags for each pass. |
| Triton segment scan | `_flash_kda_seg_scan_kernel.fn`; `(ceildiv(V, BV), N * H)` | `_scan_bv` selects `(BV, num_warps)`. Pass `HAS_H0 = h0 is not None` explicitly. |
| Gluon K1 | `k1_prepare_gluon`; same grid as Triton K1 | Use two warps. Pass `IS_VARLEN` and `HAS_BIAS` explicitly; bake `CM_LOAD=".cg"` and `CM_WS=""` or `".wt"`. |
| Gluon K2 | `k2_ab_fused_gluon`; `(ceildiv(V, BW), num_segs * H)` | `_k2_gluon_schedule` selects `BW` and `num_warps`; bake `BW`, while layouts inside the JIT read the warp count. Only segmented pass A uses this kernel. |

K2's Triton pass A runs twice when Gluon K2 is unavailable; the Gluon kernel combines those passes. The Triton segment scan runs only in segmented mode. K2's output pass C remains Triton. No in-tree KDA caller uses a returned `CompiledKernel`.

## Configuration and cache safety

When a K1 or K2 autotuner has exactly one config, select `configs[0]` and launch its raw JIT through intj from the first invocation. When it has multiple configs, call the existing decorated Triton kernel. That includes the `CHUNK_DELTA_ATTN_TRITON_AUTOTUNE=1` mode, gfx950's published six-config Triton K2 shortlist, and tests that temporarily install multiple configs. Triton then retains its current tuning key, candidate benchmarking, hooks, disk cache, and selected launch. Do not use the mutable `best_config` as a steady-state selector or duplicate Triton's tuning-key algorithm.

The native factory's key separates the JIT source key, devices, compile options, baked cache modifiers or `BW`, the JIT function's debug flag, and the current Triton runtime debug, instrumentation, and fpsan knobs. `intj.make_launcher` snapshots these values when it creates a handle, so a warmed handle must not survive a change. Check `knobs.runtime.interpret` before looking up a cached native handle and raise `UnsupportedKernel` in interpreter mode. intj's own specialization key covers each current dynamic tensor's dtype, alignment, and storage-range facts and each dynamic scalar/`constexpr`. Its module key includes the JIT source key, a Gluon discriminator, target, canonical compile options, and baked values. Source-local layout constructors become part of the JIT source key; K2's `num_warps` is part of the compile-option identity. A single-config intj refusal raises `UnsupportedKernel` instead of silently routing to Triton.

Read the current stream immediately before each native call, from that active device. This matters under `torch.cuda.graph`, which redirects launches to a capture stream. Rebuild positional arguments from the current invocation: workspaces, output, segment buffers, and optional state are newly created or may differ between calls.

## Files and compatibility

In Aiter, change `flash_kda.py`, `flash_kda_k1.py`, and `flash_kda_k2.py`; delete the now-unused `fast_launch.py` and Gluon host wrappers; replace `test_fast_launch.py` with direct-launch tests and adapt the Gluon route spy in `test_flash_kda.py`; update `docs/intj_launch_exceptions.md`. No intj core change is needed because its existing Gluon path handles the normalized kernels. Keep Aiter's Python 3.10 floor and Triton 3.7.1 support. Use the same resolved `TorchAccess` mode in baseline and final performance comparisons; Torch 2.15 does not currently have a verified intj CXX fast-access layout.

## Acceptance evidence

1. The changed Gluon JITs have no tracked layout globals; K1 and both K2 warp-count variants compile for gfx950 under Triton 3.7.1 and 3.8.0. intj's existing key-invariant and Gluon tests remain green.
2. On gfx942, KDA's single-config Triton route matches the existing route across the current correctness suite. On gfx950, run both forced Triton and normal Gluon routes, including segmentation, weak gates, variable lengths, bias, state, and tails. Execute the published four-warp Gluon K2 schedule in a segmented case.
3. A warmed native call survives CUDA/HIP graph capture and replay using the capture stream. Changed tensors, alignment, and a second device do not reuse an incorrect specialization or hold stale tensors alive; a current-device mismatch or a CPU input pointer fails before launching.
4. A multi-config K2 test still produces distinct Triton tuner-cache entries for segmented and unsegmented schedules; a forced-Triton autotuning run exercises K1's multi-config branch. Default single-config calls reach native intj without a first-call Triton launch. A warmed native handle refuses interpreter mode even if it was cached before the mode switch.
5. Report five alternating baseline/final measurements on gfx950 for the short and long FlashKDA Gluon and forced-Triton cases, plus the unchanged default chunk-delta pipeline as a control. Measure warmed full `flash_kda_fwd` host-call time with synchronization outside each short block. If per-kernel attribution is needed, time the actual K1/K2 call expressions in both source revisions; replaying captured arguments crosses different host-work boundaries. Treat changes within run-to-run variation as inconclusive. A repeatable warmed host-call regression beyond that variation in any FlashKDA route or shape blocks acceptance; attribute it, especially to gfx950's six-config K2 path, and revise dispatch before accepting removal of `_FastLaunch`.
