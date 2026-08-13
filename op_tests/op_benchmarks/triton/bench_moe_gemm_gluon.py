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

    python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py --op a4w4 --model glm52-base
    python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py --op a4w4 --shape 7168,2048,32,8

If `import triton` fails with "module 'triton' has no attribute 'language'", the venv has
a stale site-packages/triton/ shadowing an editable Triton build; prepend the real one:

    PYTHONPATH=<triton>/python python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py ...
"""

import argparse
import os
import sys

# Run directly (`python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py`) and
# sys.path[0] is *this* directory, not the repo root -- so an editable `aiter` install
# pointing at a different checkout wins the import and you silently benchmark the wrong
# tree. Put the repo root first. (The stale site-packages/triton shadow is a separate,
# environment-level problem; see the module docstring.)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_op_gemm_gluon import gluon_supported
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.moe_model_recipes import MODEL_RECIPES, get_recipe

DECODE_T = (1, 8, 32)
PREFILL_T = (1024, 4096, 16384)

#: op name -> (wrapper, x storage dtype, w storage dtype). The Gluon launcher infers the
#: operand dtypes from the tensors, so the only thing that changes per op is how the
#: inputs are quantised and which Triton kernel is the fallback baseline.
_OPS = {
    "a4w4": (moe_gemm_a4w4, torch.uint8, torch.uint8),
    "a8w8": (moe_gemm_a8w8, torch.float8_e4m3fn, torch.float8_e4m3fn),
    "a8w4": (moe_gemm_a8w4, torch.float8_e4m3fn, torch.uint8),
}


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


def _balanced_logits(t, n_expts_tot, n_expts_act, n_active, device):
    """``op_tests/test_moe_2stage.py``'s ``AITER_MOE_NUM_EXPERT_ACTIVATED`` score.

    Random logits activate a number of experts that depends on the seed, and the weight
    traffic of a MoE GEMM is set almost entirely by *how many* experts are active -- so
    two harnesses that seed independently are not measuring the same workload. At T=8 on
    H7168-I2048-E33-k8 that difference alone was 17% of stage-1 HBM traffic, larger than
    any kernel effect being compared. This reproduces the other harness's forced-activation
    score so both read the same weights: -inf everywhere, 1.0 at round-robin slots over
    the active set, giving exactly ``n_active`` experts with balanced load.

    Which experts are chosen does not matter, only the count and the balance, so this
    takes the first ``n_active`` rather than replicating the other harness's ``randperm``
    (that would require matching its whole RNG call sequence).
    """
    lo, hi = n_expts_act, min(n_expts_tot, t * n_expts_act)
    if not lo <= n_active <= hi:
        raise ValueError(f"--n-active {n_active} outside [{lo}, {hi}] for T={t}")
    score = torch.full((t, n_expts_tot), float("-inf"), dtype=torch.float16)
    slot = torch.arange(t * n_expts_act) % n_active
    rows = torch.arange(t).repeat_interleave(n_expts_act)
    score[rows, torch.arange(n_active)[slot]] = 1.0
    return score.to(device)


def _build(t, n, k, n_expts_tot, n_expts_act, device, x_dtype, w_dtype, n_active=None):
    # The bf16 staging weights for dsv4-pro at E=384 are ~17 GB; without releasing the
    # previous case's arena first, the allocator falls back to fragmented reuse and the
    # timings for the largest shapes become meaningless (observed 3.5x noise).
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    if n_active is None:
        logits = torch.randn((t, n_expts_tot), dtype=torch.float16, device=device)
    else:
        logits = _balanced_logits(t, n_expts_tot, n_expts_act, n_active, device)
    rdata, gindx, sindx = routing(logits, n_expts_act)
    rdata.gate_scal = None
    x = torch.randn((t, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n_expts_tot, k, n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((n_expts_tot, n), device=device, dtype=torch.float32)
    gammas = torch.rand((gindx.shape[0],), device=device, dtype=torch.float32)
    w, w_scale = downcast_to_mxfp(w, w_dtype, axis=1)
    x, x_scale = downcast_to_mxfp(x, x_dtype, axis=-1)
    return rdata, gindx, sindx, x, x_scale, w, w_scale, bias, gammas


def _run_one(recipe, stage, t, op, device="cuda", n_active=None):
    wrapper, x_dtype, w_dtype = _OPS[op]
    shape = recipe.gemm_shape(stage, t)
    n, k = shape.n, shape.k
    try:
        built = _build(
            t,
            n,
            k,
            shape.n_expts_tot,
            shape.n_expts_act,
            device,
            x_dtype,
            w_dtype,
            n_active,
        )
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
        # keyword args past w_scales: moe_gemm_a8w8 has an extra `w_static_scale`
        # parameter, so the positional orders of the three ops do not line up.
        return wrapper(
            x,
            w,
            xs,
            ws,
            bias=bias,
            routing_data=rdata,
            gather_indx=gindx,
            scatter_indx=sindx,
            gammas=gammas,
            out_dtype=torch.bfloat16,
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

    # Bytes actually moved, the same accounting the Triton kernels' launch_metadata
    # uses: every *activated* expert's weights (plus their scales) read once, the
    # gathered activation rows read once, the result written once. Counting weights
    # alone -- which is all that matters at decode -- understates prefill by a lot,
    # and reporting a bandwidth that ignores the intermediate is how a bandwidth-bound
    # regression hides.
    n_active = int((rdata.expt_hist > 0).sum().item())
    w_bytes = (w.numel() * w.element_size() / w.shape[0]) * n_active
    w_bytes += (ws.numel() * ws.element_size() / ws.shape[0]) * n_active
    x_bytes = n_tokens * x.shape[-1] * x.element_size()
    x_bytes += n_tokens * xs.shape[-1] * xs.element_size()
    y_bytes = n_tokens * (n // (2 if swiglu else 1)) * 2  # bf16 out
    total_bytes = w_bytes + x_bytes + y_bytes
    # the locals die with the frame; _build() releases the arena on the way in
    return {
        "n": n,
        "k": k,
        "t_triton": t_triton,
        "t_gluon": t_gluon,
        "tflops_gluon": flops / (t_gluon * 1e-6) / 1e12 if ok else float("nan"),
        "tflops_triton": flops / (t_triton * 1e-6) / 1e12,
        "gbps_gluon": total_bytes / (t_gluon * 1e-6) / 1e9 if ok else float("nan"),
        "gbps_triton": total_bytes / (t_triton * 1e-6) / 1e9,
        "why": "" if ok else why,
    }


def _explicit_recipe(hidden: int, inter: int, n_expts: int, topk: int):
    """A one-off MoeRecipe for a shape that is not one of the four model recipes."""
    from dataclasses import replace

    base = get_recipe("glm52-base")
    return replace(
        base,
        name=f"H{hidden}-I{inter}-E{n_expts}-k{topk}",
        hidden_size=hidden,
        intermediate_size=inter,
        n_routed_experts=n_expts,
        topk=topk,
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="bench_moe_gemm_gluon")
    p.add_argument("--model", choices=sorted(MODEL_RECIPES), action="append")
    p.add_argument("--regime", choices=("decode", "prefill", "both"), default="both")
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        help="explicit token counts, overriding --regime; mirrors the -t flag of "
        "op_tests/test_moe_2stage.py so the two harnesses can be pointed at the "
        "same single point",
    )
    p.add_argument("--op", choices=sorted(_OPS), action="append")
    p.add_argument(
        "--n-active",
        type=int,
        help="force exactly this many active experts with a balanced round-robin "
        "assignment, mirroring op_tests/test_moe_2stage.py's "
        "AITER_MOE_NUM_EXPERT_ACTIVATED. Use it when comparing against that harness: "
        "random routing activates a seed-dependent number of experts, and MoE GEMM "
        "traffic is set by that count, so unaligned runs do not measure the same work.",
    )
    p.add_argument(
        "--shape",
        help="explicit H,I,E,topk instead of a model recipe -- for comparing against "
        "the tuned fused_moe path, whose CSVs cover different shapes than the four "
        "model recipes do",
    )
    args = p.parse_args(argv)
    if get_arch() != "gfx950":
        print(f"gfx950 required, got {get_arch()}", file=sys.stderr)
        return 1
    models = args.model or sorted(MODEL_RECIPES)
    ops = args.op or ["a4w4"]
    explicit = None
    if args.shape:
        h, i, e, tk = (int(v) for v in args.shape.split(","))
        explicit = _explicit_recipe(h, i, e, tk)
        models = [explicit.name]
    if args.tokens:
        ts = tuple(args.tokens)
    else:
        ts = ()
        if args.regime in ("decode", "both"):
            ts += DECODE_T
        if args.regime in ("prefill", "both"):
            ts += PREFILL_T

    hdr = (
        f"| {'op':<5} | {'model':<12} | {'st':<2} | {'T':>6} | {'N':>5} | {'K':>5} "
        f"| {'gluon us':>9} | {'triton us':>10} | {'spdup':>6} "
        f"| {'gl TF/s':>8} | {'tr TF/s':>8} | {'gl GB/s':>8} | {'tr GB/s':>8} |"
    )
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for op in ops:
        for name in models:
            recipe = explicit if explicit is not None else get_recipe(name)
            for t in ts:
                for stage in (1, 2):
                    r = _run_one(recipe, stage, t, op, n_active=args.n_active)
                    if r is None:
                        print(f"| {op:<5} | {name:<12} | {stage:<2} | {t:>6} | OOM")
                        continue
                    sp = (
                        r["t_triton"] / r["t_gluon"]
                        if r["t_gluon"] == r["t_gluon"]
                        else 0
                    )
                    note = f"  ({r['why']})" if r["why"] else ""
                    print(
                        f"| {op:<5} | {name:<12} | {stage:<2} | {t:>6} | {r['n']:>5} "
                        f"| {r['k']:>5} "
                        f"| {r['t_gluon']:>9.1f} | {r['t_triton']:>10.1f} | {sp:>5.2f}x "
                        f"| {r['tflops_gluon']:>8.1f} | {r['tflops_triton']:>8.1f} "
                        f"| {r['gbps_gluon']:>8.1f} | {r['gbps_triton']:>8.1f} |{note}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
