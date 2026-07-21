# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py

from itertools import chain
from pathlib import Path
import os
import statistics
import triton.profiler as proton
import torch
import argparse
import csv
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    moe_gemm_a16w4,
)
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
from aiter.ops.triton.utils._triton.arch_info import get_arch
import tempfile
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
import inspect


def parse_profile(profile_path, useful_op_regex, reps):
    """
    construct a PerfRecord from a (proton) profile path and a regex for useful operations
    """
    from triton.profiler import viewer

    gf, _, _, _ = viewer.read(profile_path)

    # aggregate "useful" flops + bytes
    useful = gf.filter(
        f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF"
    ).dataframe
    bytes_ = int(useful["bytes"].sum())
    flops = int(
        sum(useful[[c for c in ["flops8", "flops16"] if c in useful.columns]].sum())
    )

    # take all ops (incl. "not useful" ones) when computing total time
    allops = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe
    total_time_ns = allops["time (ns)"].sum()
    kernel_time_ns = useful["time (ns)"].sum()
    return {
        "total_time_ns": total_time_ns,
        "kernel_time_ns": kernel_time_ns,
        "flops": flops,
        "bytes": bytes_,
        "reps": reps,
    }


def compute_roofline(
    *args, bench_fn, intensity_proxy_name, intensity_proxy_values, out_path, **kwargs
):
    """
    Sweeps intensity_proxy_values by injecting them into bench_fn, prints summary, and writes a CSV to out_path.
    """
    # validate input args
    if not isinstance(intensity_proxy_name, str):
        raise TypeError(
            "intensity_proxy must be a string naming a parameter in target_fn"
        )

    # determine position of intensity_proxy in target_fn signature
    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(
            f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature"
        )
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    # wrapper to inject intensity proxy into target_fn and call it
    def inject_proxy_and_call(val, args_, kwargs_):
        args_list = list(args_)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs_)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # collect performance data
    results: list[tuple[str, dict[str, int | float]]] = []
    print("=========================================")
    print(f"{out_path}...")
    print("=========================================")

    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        results.append((val, perf))

        kt = perf["kernel_time_ns"] or float("nan")
        tflops = perf["flops"] / kt * 1e-3
        tbps = perf["bytes"] / kt * 1e-3
        kernel_latency_us = perf["kernel_time_ns"] / 1e3 / perf["reps"]
        # events mode carries explicit latency stats; proton mode does not
        if "avg_ms" in perf:
            lat = (
                f"avg: {perf['avg_ms']:.4f} ms | min: {perf['min_ms']:.4f} ms | "
                f"active experts: {perf.get('active_experts', '-')}"
            )
        else:
            lat = f"Kernel latency (us): {kernel_latency_us:.2f}"
        print(
            f"{intensity_proxy_name}: {val:6d} | {lat} | "
            f"TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f} | "
            f"grid[{perf.get('launched_grid', '-')}]"
        )

    # write CSV
    fieldnames = [
        intensity_proxy_name,  # e.g. "batch"
        "avg_ms",
        "min_ms",
        "max_ms",
        "median_ms",
        "active_experts",
        "launched_grid",
        "kernel_latency_us",
        "tflops",
        "tbps",
        "total_time_ns",
        "kernel_time_ns",
        "flops",
        "bytes",
        "reps",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for val, perf in results:
            kt = perf["kernel_time_ns"] or float("nan")
            w.writerow(
                {
                    intensity_proxy_name: val,
                    "avg_ms": perf.get("avg_ms", ""),
                    "min_ms": perf.get("min_ms", ""),
                    "max_ms": perf.get("max_ms", ""),
                    "median_ms": perf.get("median_ms", ""),
                    "active_experts": perf.get("active_experts", ""),
                    "launched_grid": perf.get("launched_grid", ""),
                    "kernel_latency_us": perf["kernel_time_ns"] / 1e3 / perf["reps"],
                    "tflops": perf["flops"] / kt * 1e-3,
                    "tbps": perf["bytes"] / kt * 1e-3,
                    "total_time_ns": perf["total_time_ns"],
                    "kernel_time_ns": perf["kernel_time_ns"],
                    "flops": perf["flops"],
                    "bytes": perf["bytes"],
                    "reps": perf["reps"],
                }
            )


def check_and_shuffle_scales(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        scale = shuffle_scale_moe(
            scale, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
        return scale, "CDNA4_SCALE"
    else:
        return scale, None


def quantize(x, dtype):
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    elif dtype == "fp8":
        scale = x.abs().max().item() / 448.0
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x = x.to(fp8e4_dtype)
        return x, scale
    elif dtype == "mx8":
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x, scale = downcast_to_mxfp(x, fp8e4_dtype, axis=1)
        return x, scale
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def make_routing(batch, n_expts_tot, n_expts_act, routing_mode, skew, dev, seed=None):
    """Build (routing_data, gather_indx, scatter_indx) for one shape.

    ``uniform``  -> independent random logits, so expert load is ~uniform
                    (many experts active), matching the old bench behaviour.
    ``skewed``   -> add a per-expert bias so a subset of experts dominates the
                    top-k selection, mimicking the concentrated load seen in
                    real captured routing (far fewer experts active).

    When ``seed`` is not None the logits are drawn from a dedicated generator
    keyed on ``(seed, batch)``, so the routing (and thus the active-expert count
    that drives latency) is identical for a given M across process runs and is
    independent of any other RNG consumption (e.g. weight init). This is what
    makes A/B comparisons (e.g. --gluon-stage 1 vs 2) fair. With seed=None the
    global RNG is used, reproducing the old non-deterministic behaviour.
    """
    gen = None
    if seed is not None:
        # 100003 is a prime multiplier so distinct M values never collide on the
        # same sub-seed; the generator lives on the routing device.
        gen = torch.Generator(device=dev).manual_seed(int(seed) * 100003 + int(batch))
    logits = torch.randn((batch, n_expts_tot), device=dev, generator=gen)
    if routing_mode == "skewed":
        expert_bias = torch.randn((n_expts_tot,), device=dev, generator=gen) * skew
        logits = logits + expert_bias[None, :]
    return routing(logits, n_expts_act)


def dump_routing_blocks(batch, routing_data):
    """Print, per launched M-block, the expert id and non-padding token count.

    Decodes routing_data.expt_data.block_pid_map exactly as the kernel does:
    val==-1 is an idle block; else expt_id = val & 0xFFFF, block_id = val >> 16,
    and tokens = min(block_m, hist[expt_id] - block_id*block_m).
    """
    block_m = int(routing_data.block_m)
    ed = routing_data.expt_data
    hist = ed.hist.tolist()
    bpm = ed.block_pid_map.tolist()
    active = []
    for pid, val in enumerate(bpm):
        if val == -1:
            continue
        expt = val & 0xFFFF
        blk = (val >> 16) & 0xFFFF
        active.append((pid, expt, blk, min(block_m, hist[expt] - blk * block_m)))
    idle = len(bpm) - len(active)
    print(
        f"  [blocks] M={batch} block_m={block_m} grid_m={len(bpm)} "
        f"active={len(active)} idle={idle} "
        f"experts_with_work={len({e for _, e, _, _ in active})}"
    )
    print(f"    {'pid':>5} {'expert':>7} {'blk_in_expert':>13} {'tokens':>7}")
    for pid, expt, blk, tok in active:
        print(f"    {pid:>5} {expt:>7} {blk:>13} {tok:>7}")


def op_bytes_flops(batch, dim1, dim2, n_expts_act, n_active):
    """Analytic byte / flop model for the two-GEMM MoE layer (mxfp4 weights).

    Weight traffic is scaled by the number of *actually active* experts, so it
    reflects the real routing distribution (this is what makes uniform vs
    skewed routing show up in TBPS).
    """
    R = batch * n_expts_act
    half = dim2 // 2
    # mxfp4 weights: 0.5 B/elem + 1 B e8m0 scale per 32 elems along K
    w1_bytes = n_active * (dim1 * dim2 * 0.5 + (dim1 // 32) * dim2)
    w2_bytes = n_active * (half * dim1 * 0.5 + (half // 32) * dim1)
    # bf16 activations: x in, intermediate (write+read), output
    act_bytes = batch * dim1 * 2 + R * half * 2 * 2 + batch * dim1 * 2
    total_bytes = w1_bytes + w2_bytes + act_bytes
    flops = 2 * R * dim1 * dim2 + 2 * R * half * dim1
    return int(total_bytes), int(flops)


def time_with_events(fn, warmup, iters):
    """Repeated-run CUDA-event timing (same method as the capture replay).

    Returns a list of per-call latencies in milliseconds.
    """
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    starts, ends = [], []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        starts.append(s)
        ends.append(e)
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


# --- launched-grid capture ------------------------------------------------
# Patch KernelInterface.__getitem__ (shared by the triton & gluon kernels) to
# record each launch's decomposed grid (grid_m x grid_n, from the named kernel
# args) while capture is on -- so the bench can report the real grid per shape.
_GRID_CAP = {"on": False, "grids": []}


def _install_grid_capture():
    from triton.runtime.jit import KernelInterface

    if getattr(KernelInterface.__getitem__, "_grid_cap", False):
        return
    orig = KernelInterface.__getitem__

    def patched(self, grid):
        launcher = orig(self, grid)
        if not _GRID_CAP["on"]:
            return launcher
        arg_names = getattr(self, "arg_names", None)
        name = getattr(self, "__name__", None) or type(self).__name__
        total = grid[0] if isinstance(grid, (tuple, list)) and grid else grid

        def wrap(*a, **k):
            gm = gn = None
            if arg_names:
                nd = dict(zip(arg_names, a))
                nd.update(k)
                gm, gn = nd.get("grid_m"), nd.get("grid_n")
            _GRID_CAP["grids"].append((name, total, gm, gn))
            return launcher(*a, **k)

        return wrap

    patched._grid_cap = True
    KernelInterface.__getitem__ = patched


def _capture_grids(fn):
    """Run fn once with capture on; return 'gemm=64x48 gemm=64x48 reduce=16'."""
    _install_grid_capture()
    _GRID_CAP["grids"] = []
    _GRID_CAP["on"] = True
    try:
        fn()
    finally:
        _GRID_CAP["on"] = False
    parts = []
    for name, total, gm, gn in _GRID_CAP["grids"]:
        low = name.lower()
        if "gemm" in low:
            parts.append(
                f"gemm={gm}x{gn}" if gm is not None and gn is not None
                else f"gemm={total}"
            )
        elif "reduce" in low:
            parts.append(f"reduce={total}")
    return " ".join(parts) if parts else "-"


def bench_mlp_single_weight_init(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    timing="events",
    routing_mode="uniform",
    skew=4.0,
    warmup=5,
    iters=20,
    dump_blocks=False,
    seed=None,
):
    rank = 0
    dev = f"cuda:{rank}"

    # Make weight/activation init reproducible too when a seed is given. Weight
    # *values* don't affect latency, but this keeps the whole run deterministic.
    if seed is not None:
        torch.manual_seed(int(seed))

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
    # weights
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev)
    # biases
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot, dim1), device=dev)

    # -- numerics --
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    w1_scale, swizzle_mx_scale1 = check_and_shuffle_scales(w1_scale, dim2 // TP, dim1)
    w2_scale, swizzle_mx_scale2 = check_and_shuffle_scales(
        w2_scale, dim1, dim2 // TP // 2
    )

    x_dtype_torch = torch.bfloat16 if x_dtype == "bf16" else torch.float16
    x = torch.randn((batch, dim1), dtype=x_dtype_torch, device=dev)

    # routing computed once and held fixed across timed iters, exactly like the
    # capture replay (which replays fixed real gather/scatter indices).
    rdata, gather_indx, scatter_indx = make_routing(
        batch, n_expts_tot, n_expts_act, routing_mode, skew, dev, seed=seed
    )
    gammas = rdata.gate_scal
    n_active = int((rdata.expt_hist > 0).sum())

    if dump_blocks:
        dump_routing_blocks(batch, rdata)

    def run_layer():
        # GEMM1 (gate/up): gather + swiglu -> intermediate (M*topk, dim2/2)
        interm = moe_gemm_a16w4(
            x,
            w1,
            None,
            w1_scale,
            None,
            None,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            out_dtype=x_dtype_torch,
            apply_swiglu=True,
        )
        # GEMM2 (down): scatter-reduce with router weights, no swiglu -> (M, dim1)
        # This matches the production fused_experts down-projection that the
        # capture replay exercises (previously this used gather + swiglu).
        return moe_gemm_a16w4(
            interm,
            w2,
            None,
            w2_scale,
            None,
            None,
            b2,
            rdata,
            scatter_indx=scatter_indx,
            gammas=gammas,
            swizzle_mx_scale=swizzle_mx_scale2,
            out_dtype=x_dtype_torch,
        )

    total_bytes, flops = op_bytes_flops(
        batch, dim1, dim2 // TP, n_expts_act, n_active
    )

    # capture the actual launched grid(s) once (GEMM1, GEMM2, reduce)
    launched_grid = _capture_grids(run_layer)

    if timing == "proton":
        reps = 100
        fpath = Path(tempfile.mktemp())
        proton.start(str(fpath), hook="triton")
        for _ in range(reps):
            run_layer()
        proton.finalize()
        perf = parse_profile(
            fpath.with_suffix(".hatchet"), useful_op_regex=op_regex, reps=reps
        )
        perf["launched_grid"] = launched_grid
        return perf

    # events: repeated-run CUDA-event timing, same as the capture replay
    times_ms = time_with_events(run_layer, warmup, iters)
    mean_ms = statistics.mean(times_ms)
    return {
        "total_time_ns": mean_ms * 1e6,
        "kernel_time_ns": mean_ms * 1e6,
        "flops": flops,
        "bytes": total_bytes,
        "reps": 1,
        "avg_ms": mean_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": statistics.median(times_ms),
        "active_experts": n_active,
        "launched_grid": launched_grid,
    }


def bench_mlp(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    num_weight_inits=1,
    timing="events",
    routing_mode="uniform",
    skew=4.0,
    warmup=5,
    iters=20,
    dump_blocks=False,
    seed=None,
):
    all_results = []
    for init_idx in range(num_weight_inits):
        # Distinct-but-reproducible seed per weight init so multiple inits still
        # sample different routing while staying stable across process runs.
        init_seed = None if seed is None else int(seed) + init_idx
        result = bench_mlp_single_weight_init(
            batch,
            dim1,
            dim2,
            n_expts_tot,
            n_expts_act,
            x_dtype,
            w_dtype,
            TP,
            op_regex,
            timing=timing,
            routing_mode=routing_mode,
            skew=skew,
            warmup=warmup,
            iters=iters,
            dump_blocks=dump_blocks,
            seed=init_seed,
        )
        all_results.append(result)

    num_runs = len(all_results)
    aggregated = {
        "total_time_ns": sum(r["total_time_ns"] for r in all_results) / num_runs,
        "kernel_time_ns": sum(r["kernel_time_ns"] for r in all_results) / num_runs,
        "flops": sum(r["flops"] for r in all_results) / num_runs,
        "bytes": sum(r["bytes"] for r in all_results) / num_runs,
        "reps": all_results[0]["reps"],
    }
    # average the optional (events-mode) latency fields when present
    for k in ["avg_ms", "min_ms", "max_ms", "median_ms", "active_experts"]:
        if k in all_results[0]:
            aggregated[k] = sum(r[k] for r in all_results) / num_runs
    # launched_grid is a string (same across inits) -- carry the first
    if "launched_grid" in all_results[0]:
        aggregated["launched_grid"] = all_results[0]["launched_grid"]

    return aggregated


def roofline_mlp(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    name="",
    num_weight_inits=1,
    timing="events",
    routing_mode="uniform",
    skew=4.0,
    warmup=5,
    iters=20,
    dump_blocks=False,
    seed=None,
):
    # Avoid creating an empty directory named like the output CSV stem.
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"{x_dtype}x-{w_dtype}w-TP{TP}-{routing_mode}.csv"

    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        TP,
        op_regex,  # fixed args
        num_weight_inits,
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_csv,
        timing=timing,
        routing_mode=routing_mode,
        skew=skew,
        warmup=warmup,
        iters=iters,
        dump_blocks=dump_blocks,
        seed=seed,
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="Benchmark MoE")

    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=None,
        help="MoE batch sizes M (one or more integers). "
        "If not set, a predermined list of values will be used.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Input feature dimensions of MoE layers. Must be two integers.",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Number of total and active experts in [total experts, active experts] order.",
    )
    parser.add_argument(
        "--op-regex",
        type=str,
        default=".*moe_gemm.*",
        help="Regex to find perf for specific operation by its kernel name.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "gluon"],
        help="moe_gemm_a16w4 backend to benchmark. 'gluon' targets the tuned "
        "gfx1250 gluon kernel; 'auto' (default) lets the op pick per-shape.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Number of different weight initializations to run for more stable results (default: 1). "
        "Each initialization runs 100 iterations. Use higher values (e.g., 10) for more stable benchmarks.",
    )
    parser.add_argument(
        "--timing",
        type=str,
        default="events",
        choices=["events", "proton"],
        help="Timing method. 'events' (default) uses repeated CUDA-event runs "
        "like the capture replay and needs no profiler; 'proton' uses the "
        "triton proton profiler (requires librocprofiler-sdk).",
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="uniform",
        choices=["uniform", "skewed"],
        help="Synthetic routing distribution. 'uniform' (default) spreads tokens "
        "over many experts; 'skewed' biases a subset of experts to mimic the "
        "concentrated load of real captured routing (fewer experts active).",
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=4.0,
        help="Strength of per-expert bias when --routing skewed (larger => more "
        "concentrated, fewer active experts). Default: 4.0.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per shape for --timing events (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed iterations per shape for --timing events (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the routing RNG seed so the active-expert distribution (which "
        "drives latency) is identical per M across process runs, making A/B "
        "comparisons (e.g. --gluon-stage 1 vs 2) fair. Routing is keyed on "
        "(seed, M) via a dedicated generator; weight init uses torch.manual_seed"
        "(seed). Default (None) reproduces the old non-deterministic behaviour.",
    )
    parser.add_argument(
        "--dump-blocks",
        action="store_true",
        help="For each shape, print the per-launched-M-block routing layout "
        "(expert id + non-padding token count) decoded from block_pid_map.",
    )
    parser.add_argument(
        "--gluon-stage",
        type=int,
        default=None,
        help="Force the gluon pipeline stage (= NUM_BUFFERS) for every shape: "
        "1 = single-buffer (stage-1), 2 = LDS-prefetch double-buffer (stage-2), "
        "3 = triple-buffer (stage-3). Default (None) uses the per-shape tuned "
        "value from get_kernel_config_gluon. Capped at cdiv(K, block_k) at launch.",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)

    # Select the moe_gemm_a16w4 backend; read at call time in _selected_backend().
    os.environ["AITER_MOE_A16W4_BACKEND"] = parsed_args.backend
    print(f"moe_gemm_a16w4 backend: {parsed_args.backend}")

    # Optionally force the gluon pipeline stage (NUM_BUFFERS) for all shapes; the
    # config reads this env var per launch (get_kernel_config_gluon).
    if parsed_args.gluon_stage is not None:
        os.environ["AITER_MOE_A16W4_GLUON_NUM_BUFFERS"] = str(parsed_args.gluon_stage)
        print(f"gluon stage (NUM_BUFFERS) forced to: {parsed_args.gluon_stage}")

    dim1, dim2 = parsed_args.shape
    total_experts, active_experts = parsed_args.experts
    if parsed_args.M is None:
        batch_ranges_moe = [
            (1, 2, 1),
            (2, 5, 2),
            (8, 18, 8),
            (32, 65, 32),
            (128, 257, 128),
            (1024, 1200, 200),
            (4096, 8200, 4096),
        ]
        batch_sizes_moe = list(chain(*[range(*r) for r in batch_ranges_moe]))
    else:
        batch_sizes_moe = parsed_args.M

    quantized_dtypes = ["bf16", "mx4"]

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        quantized_dtypes[0],
        quantized_dtypes[1],
        TP=1,
        op_regex=parsed_args.op_regex,
        name=f"gpt-oss-x2-{parsed_args.backend}",
        num_weight_inits=parsed_args.num_weight_inits,
        timing=parsed_args.timing,
        routing_mode=parsed_args.routing,
        skew=parsed_args.skew,
        warmup=parsed_args.warmup,
        iters=parsed_args.iters,
        dump_blocks=parsed_args.dump_blocks,
        seed=parsed_args.seed,
    )


if __name__ == "__main__":
    main()
