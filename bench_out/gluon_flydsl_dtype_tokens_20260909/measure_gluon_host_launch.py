"""Measure CPU submission latency of the prepared T=4096 Gluon benchmark calls."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CASES = ("gluon_a4w4", "gluon_a8w4", "gluon_a8w8", "gluon_a16w16")


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p90_us": nearest_rank(values, 0.90),
        "p99_us": nearest_rank(values, 0.99),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_us": min(values),
        "max_us": max(values),
    }


def child(case: str, tokens: int, samples: int, batch_repeats: int, out: Path) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(REPO))

    from case_config import load_cases

    config = load_cases(tokens)[case]
    assert config["backend"] == "gluon"
    for name in list(os.environ):
        if name.startswith(("AITER_TRITON_MOE_", "AITER_FLYDSL_")):
            os.environ.pop(name)
    os.environ.update(config.get("env", {}))
    os.environ["AITER_TRITON_USE_HERD"] = "0"

    import bench_common as bc
    import torch
    import worker

    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.startswith("gfx950"), props.gcnArchName
    pci_bus_id = worker.verify_device_pci()
    bc.configure(7168, config["tuning"]["BLOCK_M"], tokens=tokens)
    data = bc.make_inputs(config["precision"])
    built = worker.build_gluon(data, config["tuning"], out.parent / f"{case}_artifact")
    launch = built["call"]

    # This is the same cache-pollution dispatch used immediately before every GEMM
    # in worker.py. Its submission and execution are deliberately outside the timer.
    flush = torch.empty(768 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(20):
        flush.zero_()
        launch()
    torch.cuda.synchronize()

    perf_counter_ns = time.perf_counter_ns

    def noop():
        return None

    timer_samples_ns = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(max(10_000, samples * 20)):
            start = perf_counter_ns()
            noop()
            timer_samples_ns.append(perf_counter_ns() - start)

        launch_samples_ns = []
        for _ in range(samples):
            torch.cuda.synchronize()
            flush.zero_()
            start = perf_counter_ns()
            launch()
            launch_samples_ns.append(perf_counter_ns() - start)
        torch.cuda.synchronize()

        timer_median_ns = statistics.median(timer_samples_ns)
        raw_us = [value / 1000 for value in launch_samples_ns]
        corrected_us = [max(0.0, value - timer_median_ns) / 1000 for value in launch_samples_ns]

        batches = {}
        for batch in (1, 2, 4, 8, 16, 32, 64, 100, 128, 256):
            raw_batch_us = []
            for _ in range(batch_repeats):
                torch.cuda.synchronize()
                elapsed_ns = 0
                for _ in range(batch):
                    flush.zero_()
                    start = perf_counter_ns()
                    launch()
                    elapsed_ns += perf_counter_ns() - start
                torch.cuda.synchronize()
                raw_batch_us.append(elapsed_ns / batch / 1000)
            batches[str(batch)] = {
                "repeats": batch_repeats,
                "raw_per_call_us": summarize(raw_batch_us),
                "timer_corrected_median_per_call_us": max(
                    0.0, statistics.median(raw_batch_us) - timer_median_ns / 1000
                ),
            }
    finally:
        if gc_was_enabled:
            gc.enable()

    result = {
        "case": case,
        "tokens": tokens,
        "precision": config["precision"],
        "device": {
            "name": props.name,
            "arch": props.gcnArchName,
            "physical_gpu": os.environ.get("BENCH_PHYSICAL_GPU"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "pci_bus_id": pci_bus_id,
        },
        "kernel": built["metadata"],
        "method": {
            "callable": "worker.build_gluon(...)[\"call\"]",
            "preceding_dispatch": "768 MiB flush.zero_() on the same stream",
            "synchronization": "before every primary sample and after each batch; outside timer",
            "timer": "time.perf_counter_ns",
            "timer_baseline": "one no-op Python call inside the same timer bracket",
            "samples": samples,
            "batch_repeats": batch_repeats,
            "cpp_launcher": bool(
                int(os.environ.get("AITER_TRITON_MOE_GLUON_CPP_LAUNCH", "0"))
            ),
        },
        "timer_baseline_us": summarize([value / 1000 for value in timer_samples_ns]),
        "raw_host_submission_us": summarize(raw_us),
        "timer_corrected_host_submission_us": summarize(corrected_us),
        "raw_samples_us": raw_us,
        "batch_queue_check": batches,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    stats = result["timer_corrected_host_submission_us"]
    print(
        f"DONE {case}: mean={stats['mean_us']:.3f} us "
        f"median={stats['median_us']:.3f} us p99={stats['p99_us']:.3f} us",
        flush=True,
    )


def parent(gpu: int, tokens: int, samples: int, batch_repeats: int, out: Path) -> None:
    sys.path.insert(0, str(ROOT))
    import run_bench

    run_bench.GPU_MAP = run_bench.gpu_inventory()
    if gpu not in run_bench.GPU_MAP:
        raise SystemExit(f"Physical GPU {gpu} is unavailable")
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in CASES:
        env = run_bench.environment(case, tokens, gpu, None)
        target = out / f"{case}.json"
        command = [
            str(run_bench.PY),
            str(Path(__file__).resolve()),
            "--case",
            case,
            "--tokens",
            str(tokens),
            "--samples",
            str(samples),
            "--batch-repeats",
            str(batch_repeats),
            "--output",
            str(target),
        ]
        subprocess.run(command, cwd=REPO, env=env, check=True)
        results.append(json.loads(target.read_text()))

    aggregate = {
        "gpu": gpu,
        "tokens": tokens,
        "samples_per_case": samples,
        "batch_repeats": batch_repeats,
        "results": results,
    }
    (out / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print("\ncase             mean us   median us    p99 us")
    for row in results:
        stats = row["timer_corrected_host_submission_us"]
        print(
            f"{row['case']:16s} {stats['mean_us']:9.3f} "
            f"{stats['median_us']:11.3f} {stats['p99_us']:9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--batch-repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args()
    if args.case:
        child(args.case, args.tokens, args.samples, args.batch_repeats, args.output)
    else:
        parent(args.gpu, args.tokens, args.samples, args.batch_repeats, args.output)


if __name__ == "__main__":
    main()
