# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-stage benchmark of the gfx950 Gluon MoE A4W4 grouped GEMM.

``bench_moe_gemm_a4w4.py`` measures a whole MLP behind a Proton roofline; this one
isolates the two grouped GEMMs and reports them **separately**, because stage 1 and
stage 2 have different bottlenecks (stage 1 is MFMA bound at prefill, stage 2 is weight
streaming at decode) and any baseline worth beating tunes them separately.

Baseline is the in-tree Triton ``_moe_gemm_a4w4`` on the identical inputs, selected by
``AITER_TRITON_MOE_DISABLE_GLUON=1``, so the regression risk on the fallback path is
visible in the same table.

    python op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_gluon.py
    python op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_gluon.py --model glm52-base
"""

import argparse
import os
import sys

import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, mxfp4_quant
from aiter.ops.triton.moe.moe_op_gemm_a4w4_gluon import gluon_supported
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.moe_model_recipes import MODEL_RECIPES, get_recipe

DECODE_T = (1, 8, 32)
PREFILL_T = (1024, 4096, 16384)


def _time(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1e3  # us


def _build(t, n, k, n_expts_tot, n_expts_act, device):
    # The bf16 staging weights for dsv4-pro at E=384 are ~17 GB; without releasing the
    # previous case's arena first, the allocator falls back to fragmented reuse and the
    # timings for the largest shapes become meaningless (observed 3.5x noise).
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    logits = torch.randn((t, n_expts_tot), dtype=torch.float16, device=device)
    rdata, gindx, sindx = routing(logits, n_expts_act)
    rdata.gate_scal = None
    x = torch.randn((t, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n_expts_tot, k, n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((n_expts_tot, n), device=device, dtype=torch.float32)
    gammas = torch.rand((gindx.shape[0],), device=device, dtype=torch.float32)
    w, w_scale = downcast_to_mxfp(w, torch.uint8, axis=1)
    x, x_scale = mxfp4_quant(x)
    return rdata, gindx, sindx, x, x_scale, w, w_scale, bias, gammas


def _run_one(recipe, stage, t, device="cuda"):
    shape = recipe.gemm_shape(stage, t)
    n, k = shape.n, shape.k
    try:
        built = _build(t, n, k, shape.n_expts_tot, shape.n_expts_act, device)
    except torch.OutOfMemoryError:
        return None
    rdata, gindx, sindx, x, xs, w, ws, bias, gammas = built
    swiglu = stage == 1
    act = recipe.swiglu

    M = gindx.shape[0]
    ok, why = gluon_supported(
        x=x,
        w=w,
        x_scales=xs,
        w_scales=ws,
        y=torch.empty(
            (1, M, n // (2 if swiglu else 1)), dtype=torch.bfloat16, device=device
        ),
        bias=bias,
        routing_data=rdata,
        swizzle_mx_scale=None,
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=None,
        N=n,
        K=k,
    )

    def call():
        return moe_gemm_a4w4(
            x,
            w,
            xs,
            ws,
            None,
            None,
            bias,
            rdata,
            gindx,
            sindx,
            gammas,
            None,
            torch.bfloat16,
            apply_swiglu=swiglu,
            alpha=act.alpha,
            limit=act.limit,
            swiglu_add_residual=act.add_residual,
        )

    os.environ["AITER_TRITON_MOE_DISABLE_GLUON"] = "1"
    t_triton = _time(call)
    os.environ["AITER_TRITON_MOE_DISABLE_GLUON"] = "0"
    t_gluon = _time(call) if ok else float("nan")

    n_tokens = float(rdata.expt_hist.sum().item())
    flops = 2.0 * n_tokens * n * k
    n_active = int((rdata.expt_hist > 0).sum().item())
    w_bytes = w.numel() * w.element_size() / w.shape[0] * n_active
    w_bytes += ws.numel() * ws.element_size() / ws.shape[0] * n_active
    # the locals die with the frame; _build() releases the arena on the way in
    return {
        "n": n,
        "k": k,
        "t_triton": t_triton,
        "t_gluon": t_gluon,
        "tflops_gluon": flops / (t_gluon * 1e-6) / 1e12 if ok else float("nan"),
        "tflops_triton": flops / (t_triton * 1e-6) / 1e12,
        "gbps_gluon": w_bytes / (t_gluon * 1e-6) / 1e9 if ok else float("nan"),
        "gbps_triton": w_bytes / (t_triton * 1e-6) / 1e9,
        "why": "" if ok else why,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench_moe_gemm_a4w4_gluon")
    p.add_argument("--model", choices=sorted(MODEL_RECIPES), action="append")
    p.add_argument("--regime", choices=("decode", "prefill", "both"), default="both")
    args = p.parse_args(argv)
    if get_arch() != "gfx950":
        print(f"gfx950 required, got {get_arch()}", file=sys.stderr)
        return 1
    models = args.model or sorted(MODEL_RECIPES)
    ts = ()
    if args.regime in ("decode", "both"):
        ts += DECODE_T
    if args.regime in ("prefill", "both"):
        ts += PREFILL_T

    hdr = (
        f"| {'model':<12} | {'st':<2} | {'T':>6} | {'N':>5} | {'K':>5} "
        f"| {'gluon us':>9} | {'triton us':>10} | {'speedup':>7} "
        f"| {'gluon TF/s':>10} | {'gluon GB/s':>10} |"
    )
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for name in models:
        recipe = get_recipe(name)
        for t in ts:
            for stage in (1, 2):
                r = _run_one(recipe, stage, t)
                if r is None:
                    print(f"| {name:<12} | {stage:<2} | {t:>6} | OOM")
                    continue
                sp = r["t_triton"] / r["t_gluon"] if r["t_gluon"] == r["t_gluon"] else 0
                note = f"  ({r['why']})" if r["why"] else ""
                print(
                    f"| {name:<12} | {stage:<2} | {t:>6} | {r['n']:>5} | {r['k']:>5} "
                    f"| {r['t_gluon']:>9.1f} | {r['t_triton']:>10.1f} | {sp:>6.2f}x "
                    f"| {r['tflops_gluon']:>10.1f} | {r['gbps_gluon']:>10.1f} |{note}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
