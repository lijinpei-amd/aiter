#!/usr/bin/env python
"""Replay a captured ``triton_kernel_fused_experts`` call, benchmarking the A16W4
MoE GEMM memory-access-only kernel in place of the real one.

This is ``replay_fused_experts.py`` with exactly one thing changed: the GEMM
kernel that actually runs. It rehydrates a single snapshot (real routing +
random-synthesized weights/activations) and times ``triton_kernel_fused_experts``
the same way, but first (a) forces the A16W4 wrapper onto its gfx1250 Gluon path
and (b) swaps the Gluon ``_moe_gemm_a16w4`` kernel for its memory-access-only twin
``_moe_gemm_a16w4_memacc`` -- identical global-memory access pattern (per-K-tile
TDM loads of X, packed-fp4 W and the e8m0 W-scale, plus the Y store) but no LDS
operand loads, fp4 upcast or WMMA.

Usage:

    # newest snapshot under the default capture dir, timed on the visible GPU
    python replay_moe_gemm_a16w4_memacc.py

    # a specific snapshot, 50 timed iters after 10 warmup iters
    python replay_moe_gemm_a16w4_memacc.py \
        atom.model_ops.fused_moe_triton.triton_kernel_fused_experts/full/call_XXXX_000004000.pt \
        --iters 50 --warmup 10

    # time the *real* Gluon kernel under identical conditions (baseline)
    python replay_moe_gemm_a16w4_memacc.py --kernel gluon

Run it inside the same environment/container that serves the model (so ``atom``
and ``triton_kernels`` import) and with the same ``CUDA_VISIBLE_DEVICES`` as the
server, so ``cuda:0`` maps to the intended physical GPU.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Make the func_capture package importable regardless of CWD.
FUNC_CAPTURE_ROOT = "/app/minimax-m3/scripts/py_func_capture"
if FUNC_CAPTURE_ROOT not in sys.path:
    sys.path.insert(0, FUNC_CAPTURE_ROOT)

# The A16W4 GEMM is imported by atom only when these are set; force the Gluon
# backend so the kernel we swap in is the one that runs. Set before importing atom.
os.environ.setdefault("ATOM_USE_TRITON_MOE", "1")
os.environ.setdefault("ATOM_USE_TRITON_GEMM", "1")
os.environ["AITER_MOE_A16W4_BACKEND"] = "gluon"

DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parent
DEFAULT_FUNC_DIR = (
    DEFAULT_CAPTURE_DIR
    / "atom.model_ops.fused_moe_triton.triton_kernel_fused_experts"
    / "full"
)


def _find_latest_snapshot() -> Path:
    candidates = sorted(glob.glob(str(DEFAULT_FUNC_DIR / "call_*.pt")))
    if not candidates:
        raise SystemExit(
            f"no snapshots found under {DEFAULT_FUNC_DIR}; pass a .pt path explicitly"
        )
    return Path(candidates[-1])


def _resolve_function():
    from atom.model_ops.fused_moe_triton import triton_kernel_fused_experts

    # Call the undecorated function so replay never re-enters capture.
    return getattr(
        triton_kernel_fused_experts, "__wrapped__", triton_kernel_fused_experts
    )


def _swap_kernel(which: str):
    """Rebind aiter's Gluon A16W4 kernel to the memory-only twin (or leave the
    real kernel in place for ``which == 'gluon'``)."""
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as agemm
    from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w4 import (
        _moe_gemm_a16w4_memacc,
    )

    if which == "memacc":
        agemm._moe_gemm_a16w4_gluon = _moe_gemm_a16w4_memacc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        nargs="?",
        help="path to a call_*.pt snapshot (default: newest in the capture dir)",
    )
    parser.add_argument(
        "--device",
        default="original",
        help="'original' (recorded device, default), 'cpu', or e.g. 'cuda:0'",
    )
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations")
    parser.add_argument(
        "--seed", type=int, default=0, help="torch RNG seed for synthesized tensors"
    )
    parser.add_argument(
        "--kernel", choices=["memacc", "gluon"], default="memacc",
        help="'memacc' (memory-only twin, default) or 'gluon' (real kernel)",
    )
    args = parser.parse_args()

    # Never let an inherited FUNC_CAPTURE re-instrument the function during replay.
    os.environ.pop("FUNC_CAPTURE", None)

    import torch

    torch.manual_seed(args.seed)

    from func_capture.instruments.tensor_args import load_full_call

    snapshot_path = Path(args.snapshot) if args.snapshot else _find_latest_snapshot()
    print(f"snapshot : {snapshot_path}")
    print(f"size     : {snapshot_path.stat().st_size / 1e6:.1f} MB")

    # Resolve a concrete device. Captures are recorded on cuda:0 (the visible
    # GPU); "original" replays there when CUDA is present. ``map_location`` must
    # match so tensors nested inside RoutingData load onto the GPU too, not CPU.
    device = args.device
    if device == "original":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device   : {device}")

    fn = _resolve_function()
    _swap_kernel(args.kernel)
    print(f"kernel   : {args.kernel} (Gluon backend forced)")

    call_args, call_kwargs = load_full_call(
        snapshot_path, tensor_device=device, map_location=device
    )

    def _describe(value):
        if isinstance(value, torch.Tensor):
            return f"Tensor{tuple(value.shape)} {value.dtype} {value.device}"
        return type(value).__name__

    names = [
        "output_tensor",
        "hidden_states",
        "w1",
        "w2",
        "routing_data",
        "gather_indx",
        "scatter_indx",
    ]
    print("args:")
    for i, value in enumerate(call_args):
        label = names[i] if i < len(names) else f"arg{i}"
        print(f"  {label:14s}: {_describe(value)}")

    use_cuda = torch.cuda.is_available() and args.device != "cpu"

    def run_once():
        return fn(*call_args, **call_kwargs)

    # Warmup (also triggers triton autotune/JIT so it is excluded from timing).
    for _ in range(max(0, args.warmup)):
        out = run_once()
    if use_cuda:
        torch.cuda.synchronize()

    if args.iters <= 0:
        print("no timed iterations requested; warmup only.")
        return

    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            out = run_once()
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
    else:
        import time

        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = run_once()
        total_ms = (time.perf_counter() - t0) * 1e3

    per_iter = total_ms / args.iters
    print(f"output   : {_describe(out)}")
    print(f"iters    : {args.iters} (warmup {args.warmup})")
    print(f"avg      : {per_iter:.3f} ms/call   total {total_ms:.1f} ms")


if __name__ == "__main__":
    main()
