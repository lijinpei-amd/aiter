# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Screen gfx950 MoE register/buffer recipes with cold, GEMM-only timings.

Candidates are a JSON list of {"name": str, "config": dict}. Each config overrides
the dtype's register-regression baseline. Inputs are built once per process. A
768 MiB flush precedes every warmup and measured launch on the same stream;
rocprofv3 supplies device dispatch durations without host or preprocessing time.
Output bytes must match the legacy pipeline before a candidate is timed.

Example (use the Python environment containing torch, Triton and rocprofv3)::

    python scripts/gluon_moe_register_tune.py --dtype mxfp4 --gpu 4 \
        --candidates candidates.json --output bench_out/register_tuning

Use --warmup 40 --reps 100 --determinism 16 --rounds 3 for final comparisons.
Environment overrides apply to the entire batch, including its baseline.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TUNING_PREFIXES = ("AITER_TRITON_MOE_GLUON_", "TRITON_HIP_", "TRITON_MEMBAR_")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("mxfp4", "mxfp8", "bf16"), required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--determinism", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--flush-mib", type=int, default=768)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--experts", type=int, default=33)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("reps", "determinism", "rounds", "flush_mib"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    args.overrides = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not key or not separator:
            parser.error(f"--env expects KEY=VALUE, got {item!r}")
        args.overrides[key] = value
    args.output = args.output.resolve()
    args.candidates = args.candidates.resolve()
    candidates = json.loads(args.candidates.read_text())
    if not isinstance(candidates, list):
        parser.error("candidates must be a JSON list")
    names = set()
    for candidate in candidates:
        # Accept label as well for compatibility with older measurement scripts.
        name = candidate.get("name", candidate.get("label"))
        if not isinstance(name, str) or not isinstance(candidate.get("config"), dict):
            parser.error("each candidate needs a name and a config dictionary")
        if name in names or name.startswith("__baseline"):
            parser.error(f"duplicate or reserved candidate name: {name}")
        candidate["name"] = name
        names.add(name)
    args.recipes = candidates
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def resources(compiled):
    assembly = compiled.asm["amdgcn"]
    result = {"shared": compiled.metadata.shared, "kernel_hash": compiled.hash}
    for field in (
        "vgpr_count",
        "agpr_count",
        "sgpr_count",
        "vgpr_spill_count",
        "sgpr_spill_count",
        "private_segment_fixed_size",
    ):
        match = re.search(rf"\.{field}:\s*(\d+)", assembly)
        result[field] = int(match[1]) if match else None
    result["assembly_sha256"] = hashlib.sha256(assembly.encode()).hexdigest()
    result["hsaco_sha256"] = hashlib.sha256(compiled.asm["hsaco"]).hexdigest()
    return result


def run_worker(args):
    # Set import-time configuration before importing AITER or the test helpers.
    for key in list(os.environ):
        if key.startswith(TUNING_PREFIXES):
            del os.environ[key]
    os.environ["AITER_TRITON_MOE_GLUON_B_PRESHUFFLED"] = "1"
    os.environ.update(args.overrides)
    sys.path.insert(0, str(REPO))
    import torch
    import triton

    from op_tests.triton_tests.moe.test_moe_gemm_gluon_registers import (
        _build_register_case,
        _launch_register_case,
        _register_config,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dtype": args.dtype,
        "shape": {
            key: getattr(args, key) for key in ("m", "n", "k", "experts", "topk")
        },
        "seed": 72,
        "output_dtype": "float32",
        "gate_up_split": True,
        "warmup": args.warmup,
        "reps": args.reps,
        "determinism": args.determinism,
        "cold_determinism": True,
        "rounds": args.rounds,
        "flush_mib": args.flush_mib,
        "gpu": args.gpu,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(TUNING_PREFIXES + ("TRITON_CACHE_DIR",))
        },
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {},
    }
    sources = list((REPO / "aiter/ops/triton/_gluon_kernels/gfx950/moe").glob("*.py"))
    sources += [
        REPO / "aiter/ops/triton/moe/moe_op_gemm_gluon.py",
        Path(__file__),
        REPO / "op_tests/triton_tests/moe/test_moe_gemm_gluon_registers.py",
    ]
    for source in sorted(sources):
        manifest["source_sha256"][str(source.relative_to(REPO))] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "candidates.json", args.recipes)
    case = _build_register_case(
        args.dtype, m=args.m, n=args.n, k=args.k, experts=args.experts, topk=args.topk
    )
    baseline_config = _register_config(args.dtype)
    dispatch = 0

    def launch(config):
        nonlocal dispatch
        compiled = _launch_register_case(case, config)
        dispatch += 1
        return compiled

    launch(baseline_config)
    torch.testing.assert_close(
        case.output[0],
        case.expected,
        rtol=3e-4,
        atol=3e-4 if args.dtype == "mxfp8" else 3e-5,
    )
    baseline = case.output.clone()
    baseline_bits = baseline.view(torch.int32)
    baseline_hash = hashlib.sha256(baseline.cpu().numpy().tobytes()).hexdigest()
    manifest["baseline_sha256"] = baseline_hash
    write_json(args.output / "manifest.json", manifest)
    flush = torch.empty(args.flush_mib << 20, device="cuda", dtype=torch.uint8)
    baseline_recipe = {"name": "__baseline__", "config": {}}
    with (args.output / "dispatches.jsonl").open("w", buffering=1) as records:
        for round_index in range(args.rounds):
            candidates = args.recipes if round_index % 2 == 0 else args.recipes[::-1]
            for recipe in [
                baseline_recipe,
                *candidates,
                {"name": "__baseline_end__", "config": {}},
            ]:
                config = {**baseline_config, **recipe["config"]}
                record = {
                    "name": recipe["name"],
                    "round": round_index,
                    "config": config,
                    "overrides": recipe["config"],
                    "dispatch_before": dispatch,
                    "status": "error",
                }
                try:
                    case.output.fill_(float("nan"))
                    compiled = launch(config)
                    record["resources"] = resources(compiled)
                    # Equality synchronizes the stream and checks every output bit.
                    if not torch.equal(case.output.view(torch.int32), baseline_bits):
                        raise AssertionError(
                            "output differs from legacy pipeline bytes"
                        )
                    for repeat in range(args.determinism):
                        case.output.fill_(float("nan"))
                        flush.zero_()
                        launch(config)
                        if not torch.equal(
                            case.output.view(torch.int32), baseline_bits
                        ):
                            raise AssertionError(
                                f"nondeterministic output at repeat {repeat}"
                            )
                    record.update(
                        sha256=baseline_hash,
                        correctness=True,
                        determinism_repeats=args.determinism,
                    )
                    for _ in range(args.warmup):
                        flush.zero_()
                        launch(config)
                    record["timed_start"] = dispatch
                    for _ in range(args.reps):
                        flush.zero_()
                        launch(config)
                    record["timed_end"] = dispatch
                    torch.cuda.synchronize()
                    if not torch.equal(case.output.view(torch.int32), baseline_bits):
                        raise AssertionError(
                            "output differs after cold timing launches"
                        )
                    record["status"] = "ok"
                except Exception as error:  # noqa: BLE001
                    # Failed candidates remain in the results with their errors.
                    record["error"] = f"{type(error).__name__}: {error}"
                    record["traceback"] = traceback.format_exc()
                    # Compilation/validation errors are recoverable. A device fault
                    # invalidates this process; stop before corrupting trace mapping.
                    torch.cuda.synchronize()
                finally:
                    record["dispatch_after"] = dispatch
                    records.write(json.dumps(record, default=str) + "\n")
                    print(
                        json.dumps(
                            {
                                key: record[key]
                                for key in (
                                    "name",
                                    "round",
                                    "status",
                                    "dispatch_after",
                                    "error",
                                    "resources",
                                )
                                if key in record
                            },
                            default=str,
                        ),
                        flush=True,
                    )
    manifest["total_gemm_dispatches"] = dispatch
    write_json(args.output / "manifest.json", manifest)


def collect(args, returncode):
    manifest = json.loads((args.output / "manifest.json").read_text())
    rows = []
    for trace in args.output.glob("trace/**/*kernel_trace.csv"):
        with trace.open() as stream:
            for row in csv.DictReader(stream):
                if "_moe_gluon_gemm1" in row["Kernel_Name"]:
                    start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                    rows.append((start, (end - start) / 1000))
    rows.sort()
    if len(rows) != manifest.get("total_gemm_dispatches"):
        raise RuntimeError(
            f"trace dispatch count {len(rows)} != manifest "
            f"{manifest.get('total_gemm_dispatches')}; worker exit {returncode}"
        )
    results = []
    for line in (args.output / "dispatches.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["status"] == "ok":
            samples = [
                value for _, value in rows[record["timed_start"] : record["timed_end"]]
            ]
            if len(samples) != args.reps:
                raise RuntimeError(f"incomplete timing samples for {record['name']}")
            record.update(
                samples_us=samples,
                median_us=statistics.median(samples),
                min_us=min(samples),
                max_us=max(samples),
                mean_us=statistics.mean(samples),
            )
        results.append(record)
    write_json(args.output / "results.json", {"manifest": manifest, "results": results})
    with (args.output / "timings.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "name",
                "round",
                "status",
                "median_us",
                "min_us",
                "max_us",
                "vgpr",
                "agpr",
                "vgpr_spills",
                "lds_bytes",
                "error",
            )
        )
        for result in results:
            resource = result.get("resources", {})
            writer.writerow(
                [
                    result.get(key, "")
                    for key in (
                        "name",
                        "round",
                        "status",
                        "median_us",
                        "min_us",
                        "max_us",
                    )
                ]
                + [
                    resource.get(key, "")
                    for key in (
                        "vgpr_count",
                        "agpr_count",
                        "vgpr_spill_count",
                        "shared",
                    )
                ]
                + [result.get("error", "")]
            )
    good = [record for record in results if record["status"] == "ok"]
    for record in sorted(good, key=lambda record: record["median_us"]):
        print(
            f"{record['name']:48s} r{record['round']} {record['median_us']:9.3f} us",
            flush=True,
        )
    failures = [record for record in results if record["status"] != "ok"]
    print(
        f"{len(good)} measurements, {len(failures)} rejected; {args.output / 'results.json'}",
        flush=True,
    )


def main():
    args = arguments()
    if args._worker:
        run_worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError(f"output already contains a run: {args.output}")
    profiler = Path(sys.executable).parent / "rocprofv3"
    if not profiler.exists():
        profiler = shutil.which("rocprofv3")
    if not profiler:
        raise FileNotFoundError("rocprofv3 is required for GEMM-only timings")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(TUNING_PREFIXES)
    }
    env.update(HIP_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(REPO))
    if args.cache:
        env["TRITON_CACHE_DIR"] = str(args.cache.resolve())
    else:
        cache_tag = hashlib.sha256(
            json.dumps(args.overrides, sort_keys=True).encode()
        ).hexdigest()[:12]
        env["TRITON_CACHE_DIR"] = (
            f"/tmp/aiter_register_tune_{os.getuid()}/{args.dtype}_{cache_tag}"
        )
    command = [
        str(profiler),
        "--kernel-trace",
        "-d",
        str(args.output / "trace"),
        "-o",
        "trace",
        "--output-format",
        "csv",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--_worker",
    ]
    print(
        f"Profiling {len(args.recipes)} candidates on GPU {args.gpu}; log: {args.output / 'run.log'}",
        flush=True,
    )
    with (args.output / "run.log").open("w") as log:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    collect(args, process.returncode)
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
