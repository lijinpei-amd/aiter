#!/usr/bin/env python
"""Replay every captured routing-only snapshot, benchmarking the A16W4 MoE GEMM
memory-access-only kernel in place of the real one.

This is ``benchmark_captures.py`` with exactly one thing changed: the GEMM kernel
that actually runs. It replays the same ``full/call_*.pt`` snapshots through the
same ``triton_kernel_fused_experts`` path (real routing + random-synthesized
weights/activations), times each with CUDA events, and writes the same CSV, but
before timing it (a) forces the A16W4 wrapper onto its gfx1250 Gluon path and
(b) swaps the Gluon ``_moe_gemm_a16w4`` kernel for its memory-access-only twin
``_moe_gemm_a16w4_memacc`` -- identical global-memory access pattern (per-K-tile
TDM loads of X, packed-fp4 W and the e8m0 W-scale, plus the Y store) but no LDS
operand loads, fp4 upcast or WMMA.

Diff the resulting CSV against ``benchmark_captures.py``'s ``benchmark_results.csv``
to see, per captured shape, how much of the fused-MoE latency is the GEMM's
memory movement vs. its compute.

Run inside the serving container, same CUDA_VISIBLE_DEVICES as serve.sh, with the
triton-MoE env flags (the GEMM imports are gated on them):

    CUDA_VISIBLE_DEVICES=3 ATOM_USE_TRITON_MOE=1 ATOM_USE_TRITON_GEMM=1 \
        python bench_moe_gemm_a16w4_memacc.py --warmup 5 --iters 20

Pass ``--kernel gluon`` to time the *real* Gluon kernel under these identical
conditions (useful because production decode shapes normally take the Triton
path, so this is the apples-to-apples baseline for the memacc numbers).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
from pathlib import Path

FUNC_CAPTURE_ROOT = "/app/minimax-m3/scripts/py_func_capture"
if FUNC_CAPTURE_ROOT not in sys.path:
    sys.path.insert(0, FUNC_CAPTURE_ROOT)

# The A16W4 GEMM is imported by atom only when these are set; force the Gluon
# backend so the kernel we swap in is the one that runs. Set before importing atom.
os.environ.setdefault("ATOM_USE_TRITON_MOE", "1")
os.environ.setdefault("ATOM_USE_TRITON_GEMM", "1")
os.environ["AITER_MOE_A16W4_BACKEND"] = "gluon"

CAP_DIR = Path(__file__).resolve().parent
FULL_DIR = (
    CAP_DIR
    / "atom.model_ops.fused_moe_triton.triton_kernel_fused_experts"
    / "full"
)


def _call_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=5, help="warmup iters per snapshot")
    ap.add_argument("--iters", type=int, default=20, help="timed iters per snapshot")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel", choices=["memacc", "gluon"], default="memacc",
                    help="'memacc' (memory-only twin, default) or 'gluon' (real kernel)")
    ap.add_argument("--out", default=str(CAP_DIR / "memacc_benchmark_results.csv"))
    args = ap.parse_args()

    os.environ.pop("FUNC_CAPTURE", None)  # never re-capture during replay

    import torch

    torch.manual_seed(args.seed)

    from func_capture.instruments.tensor_args import load_full_call
    from atom.model_ops.fused_moe_triton import triton_kernel_fused_experts

    # ---- swap the kernel that is benchmarked -------------------------------
    # atom's fused_experts calls aiter's moe_gemm_a16w4 wrapper, which (with the
    # forced Gluon backend) dispatches to the module-global _moe_gemm_a16w4_gluon.
    # Rebind that global to the memory-access-only twin.
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as agemm
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w4 import (
        _moe_gemm_a16w4_memacc,
    )

    if args.kernel == "memacc":
        agemm._moe_gemm_a16w4_gluon = _moe_gemm_a16w4_memacc
    print(f"benchmarking kernel: {args.kernel} (Gluon backend forced)\n")

    fn = getattr(triton_kernel_fused_experts, "__wrapped__", triton_kernel_fused_experts)

    snaps = sorted(glob.glob(str(FULL_DIR / "call_*.pt")), key=lambda p: _call_index(Path(p)))
    if not snaps:
        raise SystemExit(f"no snapshots under {FULL_DIR}")
    print(f"benchmarking {len(snaps)} snapshots on {args.device} "
          f"(warmup={args.warmup}, iters={args.iters})\n")

    rows = []
    for i, p in enumerate(snaps):
        path = Path(p)
        try:
            call_args, call_kwargs = load_full_call(
                path, tensor_device=args.device, map_location=args.device
            )
            hs = call_args[1]      # hidden_states (synthesized)
            w1 = call_args[2]      # [E, 2N, K]
            w2 = call_args[3]      # [E, N, K]
            gather = call_args[5]  # real routing gather indices
            M, K = int(hs.shape[0]), int(hs.shape[1])
            E = int(w1.shape[0])
            w1_2n = int(w1.shape[1])
            w2_n = int(w2.shape[1])
            topk = call_kwargs.get("topk")
            activation = str(call_kwargs.get("activation")).split(".")[-1]
            act_quant = str(call_kwargs.get("act_quant")).split(".")[-1]
            gather_n = int(gather.numel())

            def once():
                return fn(*call_args, **call_kwargs)

            for _ in range(max(0, args.warmup)):
                out = once()
            torch.cuda.synchronize()

            starts, ends = [], []
            for _ in range(args.iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = once()
                e.record()
                starts.append(s)
                ends.append(e)
            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(starts, ends)]

            row = {
                "call_index": _call_index(path),
                "M_tokens": M,
                "K_hidden": K,
                "experts": E,
                "w1_2N": w1_2n,
                "w2_N": w2_n,
                "topk": topk,
                "routed_tokens": gather_n,
                "activation": activation,
                "act_quant": act_quant,
                "avg_ms": round(statistics.mean(times), 4),
                "min_ms": round(min(times), 4),
                "max_ms": round(max(times), 4),
                "median_ms": round(statistics.median(times), 4),
                "snapshot": path.name,
            }
            rows.append(row)
            print(f"[{i+1:3d}/{len(snaps)}] call={row['call_index']:9d} "
                  f"M={M:6d} routed={gather_n:7d} {activation:7s} "
                  f"-> avg {row['avg_ms']:8.3f} ms  min {row['min_ms']:8.3f} ms")

            del call_args, call_kwargs, out, hs, w1, w2, gather
            torch.cuda.empty_cache()
        except Exception as exc:  # keep going if one snapshot fails
            print(f"[{i+1:3d}/{len(snaps)}] {path.name}: FAILED: "
                  f"{type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no snapshots benchmarked successfully")

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")

    by_time = sorted(rows, key=lambda r: r["avg_ms"])
    print(f"fastest: call {by_time[0]['call_index']} "
          f"M={by_time[0]['M_tokens']} -> {by_time[0]['avg_ms']} ms")
    print(f"slowest: call {by_time[-1]['call_index']} "
          f"M={by_time[-1]['M_tokens']} -> {by_time[-1]['avg_ms']} ms")
    print(f"total kernel avg across captures: "
          f"{round(statistics.mean(r['avg_ms'] for r in rows), 3)} ms")


if __name__ == "__main__":
    main()
