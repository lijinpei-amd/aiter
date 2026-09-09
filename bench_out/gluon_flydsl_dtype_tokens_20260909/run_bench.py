"""Serialized, restartable cold Gluon/FlyDSL dtype and token comparisons."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
UTILS = REPO / "bench_out/gluon_scale_ptr_unify_20260907/a4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UTILS))

import bench_utils as common
from case_config import load_cases, source_files

PY = common.PY
TOKEN_COUNTS = (16, 64, 256, 1024, 4096)
GPU_MAP = {}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    common.write_json(path, data)


def nearest_rank(values, percentile):
    """Return the nearest-rank percentile: sorted[ceil(p*n)-1]."""
    if not values:
        raise ValueError("nearest_rank requires at least one sample")
    if not 0 < percentile <= 1:
        raise ValueError(percentile)
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def gpu_inventory():
    physical = json.loads(
        subprocess.check_output(
            [str(PY.parent / "amd-smi"), "list", "--json"], text=True
        )
    )
    code = """import ctypes,json,sysconfig
from pathlib import Path
hip=ctypes.CDLL(str(Path(sysconfig.get_path('purelib'))/'_rocm_sdk_core/lib/libamdhip64.so.7'))
n=ctypes.c_int()
assert hip.hipGetDeviceCount(ctypes.byref(n))==0
result={}
for index in range(n.value):
    bus=ctypes.create_string_buffer(64)
    assert hip.hipDeviceGetPCIBusId(bus,len(bus),index)==0
    result[bus.value.decode().lower()]=index
print(json.dumps(result))
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        )
    }
    ordinals = json.loads(
        subprocess.check_output([str(PY), "-c", code], env=env, text=True)
    )
    return {
        item["gpu"]: {
            **item,
            "hip_ordinal": ordinals[item["bdf"].lower()],
        }
        for item in physical
    }


def telemetry(path, gpu):
    command = [
        str(PY.parent / "rocm-smi"),
        "-d",
        str(gpu),
        "--showbus",
        "--showuniqueid",
        "--showserial",
        "--showclocks",
        "--showtemp",
        "--showpower",
        "--json",
    ]
    write(path, json.loads(subprocess.check_output(command, text=True)))


def gpu_idle_state(gpu):
    smi = subprocess.run(
        [
            str(PY.parent / "rocm-smi"),
            "-d",
            str(gpu),
            "--showuse",
            "--showmeminfo",
            "vram",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    mapping = subprocess.run(
        [str(PY.parent / "rocm-smi"), "--showpidgpus"],
        capture_output=True,
        text=True,
        check=True,
    )
    processes = subprocess.run(
        [str(PY.parent / "amd-smi"), "process", "-g", str(gpu), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    card = json.loads(smi.stdout)[f"card{gpu}"]
    mapped_pids = common.gpu_pids_from_mapping(mapping.stdout, gpu)
    device_processes = json.loads(processes.stdout)
    allocations = []
    for device in device_processes:
        if device.get("gpu") != gpu:
            continue
        for item in device.get("process_list", []):
            info = item["process_info"]
            if info in ("N/A", "No running processes detected"):
                continue
            if not isinstance(info, dict):
                raise TypeError(f"Unrecognized amd-smi process entry: {info!r}")
            usage = info.get("memory_usage", {}).get("vram_mem", {})
            vram = usage.get("value", 0) if isinstance(usage, dict) else usage
            if isinstance(vram, (int, float)) and vram > 0:
                allocations.append({"pid": info["pid"], "vram_bytes": vram})
    use = int(card["GPU use (%)"])
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "gpu": gpu,
        "gpu_use_percent": use,
        "vram_used_bytes": int(card["VRAM Total Used Memory (B)"]),
        "mapped_pids": mapped_pids,
        "foreign_allocations": allocations,
        "idle": use == 0 and not mapped_pids and not allocations,
        "raw_process_mapping": mapping.stdout,
        "raw_processes": device_processes,
    }


common.gpu_idle_state = gpu_idle_state


def environment(case, tokens, gpu, specs):
    env = common.base_env(REPO, ROOT / "job", gpu)
    for key in list(env):
        if (
            key.startswith(
                (
                    "AITER_TRITON_MOE_",
                    "AITER_FLYDSL_",
                    "TRITON_HIP_",
                    "TRITON_MEMBAR_",
                    "FLYDSL_",
                    "ROCPROF_",
                )
            )
            or key == "GPU_DEVICE_ORDINAL"
        ):
            env.pop(key)
    cache = ROOT / "cache" / f"{case}_t{tokens}"
    env.update(
        BENCH_REPO=str(REPO),
        PYTHONPATH=str(REPO),
        AITER_TRITON_USE_HERD="0",
        TRITON_AMD_NT_ONLY="0",
        CK_DIR=str(REPO / "3rdparty/composable_kernel"),
        HIP_VISIBLE_DEVICES=str(GPU_MAP[gpu]["hip_ordinal"]),
        PYTHONDONTWRITEBYTECODE="1",
        BENCH_EXPECT_PCI_BUS_ID=GPU_MAP[gpu]["bdf"].lower(),
        BENCH_PHYSICAL_GPU=str(gpu),
        TRITON_CACHE_DIR=str(cache),
        FLYDSL_RUNTIME_CACHE_DIR=str(cache),
        FLYDSL_GPU_ARCH="gfx950",
        TRITON_MEMBAR_DEDUP_BARE="1",
        TRITON_HIP_EXTERNAL_LLC=str(common.LLC),
        TRITON_HIP_EXTERNAL_LLC_FLAGS=common.LLC_FLAGS,
    )
    env["PATH"] = f"{env['ROCM_PATH']}/lib/llvm/bin:{env['ROCM_PATH']}/bin:" + env.get(
        "PATH", ""
    )
    env.update(load_cases(tokens)[case].get("env", {}))
    return env


def provenance():
    harness = [
        ROOT / name
        for name in (
            "README.md",
            "worker.py",
            "bench_common.py",
            "fly_build.py",
            "case_config.py",
            "cases.json",
            "run_bench.py",
            "audit_results.py",
        )
    ]
    dependencies = source_files() + [
        UTILS / "bench_utils.py",
        REPO / "bench_out/gluon_flydsl_small_tokens_20260908/bench_common.py",
        REPO / "bench_out/gluon_flydsl_small_tokens_20260908/fly_build.py",
        REPO / "bench_out/gluon_flydsl_a8a16_20260907/bench_common.py",
    ]
    guarded = []
    for directory in (
        "aiter/ops/triton/_gluon_kernels/gfx950/moe",
        "aiter/ops/triton/moe",
        "aiter/ops/triton/_triton_kernels/moe",
        "aiter/ops/flydsl",
    ):
        guarded.extend((REPO / directory).rglob("*.py"))
    guarded += [
        REPO / "aiter/fused_moe.py",
        REPO / "aiter/ops/shuffle.py",
        REPO / "aiter/utility/fp4_utils.py",
        REPO / "aiter/ops/quant.py",
        REPO / "aiter/ops/moe_mxfp4_aux.py",
        REPO / "op_tests/triton_tests/moe/test_moe_gemm_a4w4.py",
        REPO / "op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py",
    ]
    guarded.extend((REPO / "csrc/kernels/mxfp4_moe/moe_aux").rglob("*.*"))
    guarded.extend((REPO / "aiter/jit").glob("module_moe_mxfp4_aux*.so"))
    guarded.extend((REPO / "aiter/jit").glob("module_quant*.so"))
    unique_harness = sorted(set(harness + dependencies))
    unique_source = sorted(set(guarded))
    return {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "tracked_status": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=REPO,
            text=True,
        ).splitlines(),
        "harness": {str(path): common.sha(path) for path in unique_harness},
        "source": {
            str(path.relative_to(REPO)): common.sha(path) for path in unique_source
        },
        "llc_sha256": common.sha(common.LLC),
        "libtriton_sha256": common.sha(
            REPO.parent / "triton/python/triton/_C/libtriton.so"
        ),
    }


def stable_inputs(value):
    value = copy.deepcopy(value)
    value.get("shape", {}).pop("block_m", None)
    for key in (
        "useful_m_blocks",
        "gluon_grid_m",
        "unit_scales",
        "routing_layout",
    ):
        value.pop(key, None)
    return value


def stable_kernel(value):
    value = copy.deepcopy(value)
    value.pop("saved_executable", None)
    if "launch_metadata" in value:
        value["launch_metadata"].pop("hash", None)
    if "executable" in value:
        executable = value["executable"]
        for key in ("cache_file", "saved_cache_file"):
            executable.pop(key, None)
        for binary in executable.get("binaries", []):
            binary.pop("path", None)
        for item in executable.get("source_modules", {}).values():
            item.pop("path", None)
    return value


def trace_values(run, pattern, warmups, sample_count):
    rows = sorted(
        (
            row
            for path in (run / "trace").rglob("*kernel_trace.csv")
            for row in csv.DictReader(path.open())
        ),
        key=lambda row: int(row["Start_Timestamp"]),
    )
    indices = [index for index, row in enumerate(rows) if pattern in row["Kernel_Name"]]
    total = warmups + sample_count
    assert len(indices) >= total, (pattern, len(indices), total)
    cold = indices[-total:]
    assert len({rows[index]["Kernel_Name"] for index in cold}) == 1
    assert (
        len(
            {
                tuple(rows[index]["Grid_Size_" + axis] for axis in "XYZ")
                for index in cold
            }
        )
        == 1
    )
    assert (
        len(
            {
                tuple(rows[index]["Workgroup_Size_" + axis] for axis in "XYZ")
                for index in cold
            }
        )
        == 1
    )
    section = rows[cold[0] - 1 : cold[-1] + 1]
    assert len(section) == 2 * total, "Unexpected kernels between flush and GEMM"
    assert all(
        int(row["End_Timestamp"]) > int(row["Start_Timestamp"]) for row in section
    )

    node = int(rows[cold[0]]["Agent_Id"].split()[-1])
    agents = [
        row
        for path in (run / "trace").rglob("*agent_info.csv")
        for row in csv.DictReader(path.open())
    ]
    agent = next(row for row in agents if int(row["Node_Id"]) == node)
    location = int(agent["Location_Id"])
    pci = (
        f"{int(agent['Domain']):04x}:{location >> 8:02x}:"
        f"{(location >> 3) & 31:02x}.{location & 7}"
    )
    expected_pci = read(run / "command.json")["env"]["BENCH_EXPECT_PCI_BUS_ID"]
    assert pci == expected_pci, "Profiler ran on a different physical GPU"

    overlaps = []
    for ordinal, index in enumerate(cold):
        fill, gemm = rows[index - 1 : index + 1]
        assert "FillFunctor<unsigned char>" in fill["Kernel_Name"]
        assert tuple(int(fill["Grid_Size_" + axis]) for axis in "XYZ") == (
            50331648,
            1,
            1,
        )
        assert pattern in gemm["Kernel_Name"]
        assert index == cold[0] + 2 * ordinal
    for first, second in pairwise(section):
        assert all(
            first[key] == second[key] for key in ("Agent_Id", "Queue_Id", "Stream_Id")
        )
        assert int(second["Dispatch_Id"]) == int(first["Dispatch_Id"]) + 1
        overlap = int(first["End_Timestamp"]) - int(second["Start_Timestamp"])
        if overlap > 0:
            overlaps.append(
                {
                    "earlier_dispatch": first["Dispatch_Id"],
                    "later_dispatch": second["Dispatch_Id"],
                    "overlap_ns": overlap,
                }
            )
        assert overlap <= 1000, ("Timestamp overlap above audit allowance", overlap)

    measured = cold[-sample_count:]
    samples = [
        {
            "sample": sample,
            "start_ns": int(rows[index]["Start_Timestamp"]),
            "duration_us": (
                int(rows[index]["End_Timestamp"]) - int(rows[index]["Start_Timestamp"])
            )
            / 1000,
        }
        for sample, index in enumerate(measured)
    ]
    durations = [sample["duration_us"] for sample in samples]
    assert all(duration > 0 for duration in durations)
    return {
        "mean_us": st.mean(durations),
        "median_us": st.median(durations),
        "p99_us": nearest_rank(durations, 0.99),
        "stdev_us": st.pstdev(durations),
        "setup_gemm_dispatches": len(indices) - total,
        "warmup_samples": warmups,
        "measured_samples": sample_count,
        "verified_flush_gemm_pairs": total,
        "retained_timestamp_overlaps": overlaps,
        "profiled_pci_bus_id": pci,
        "profiled_node_id": node,
        "grid": [int(rows[measured[-1]]["Grid_Size_" + axis]) for axis in "XYZ"],
        "workgroup": [
            int(rows[measured[-1]]["Workgroup_Size_" + axis]) for axis in "XYZ"
        ],
    }, samples


def execute(
    case,
    tokens,
    gpu,
    run,
    specs,
    mode,
    identities,
    warmups,
    samples,
    validation_replays,
):
    run.mkdir(parents=True, exist_ok=True)
    assert provenance() == identities, "Source or harness changed"
    env = environment(case, tokens, gpu, specs)
    worker = [
        str(PY),
        str(ROOT / "worker.py"),
        "--case",
        case,
        "--tokens",
        str(tokens),
        "--out",
        str(run / "worker.json"),
        "--mode",
        mode,
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
        "--validation-replays",
        str(validation_replays),
    ]
    command = (
        worker
        if mode == "validate"
        else [
            str(PY.parent / "rocprofv3"),
            "--kernel-trace",
            "--agent-index",
            "absolute",
            "-d",
            str(run / "trace"),
            "-o",
            "kt",
            "--output-format",
            "csv",
            "--",
            *worker,
        ]
    )
    command_record = {
        "command": command,
        "cwd": str(REPO),
        "case": case,
        "tokens": tokens,
        "gpu": gpu,
        "mode": mode,
        "warmups": warmups,
        "samples": samples,
        "validation_replays": validation_replays,
        "env": {
            key: value
            for key, value in env.items()
            if key.startswith(("BENCH_", "AITER_", "TRITON_", "FLYDSL_", "HIP_"))
            or key in ("ROCM_PATH", "PATH", "PYTHONPATH", "CK_DIR")
        },
    }
    completed = run / "complete.json"
    if completed.exists():
        assert read(completed) == {"pass": True, "provenance": identities}
        assert read(run / "command.json") == command_record
        return read(run / "worker.json")
    assert not (run / "run.log").exists(), f"Retained incomplete attempt: {run}"
    write(run / "command.json", command_record)
    common.wait_for_gpu_idle(run / "idle_before.jsonl", gpu)
    common.snapshot_gpu(run / "gpu_before.txt", gpu)
    telemetry(run / "gpu_before.json", gpu)
    print(f"START gpu={gpu} {case} T={tokens} {mode}", flush=True)
    start = time.monotonic()
    with (run / "run.log").open("w") as log:
        process = subprocess.run(
            command,
            env=env,
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    write(
        run / "returncode.json",
        {"returncode": process.returncode, "wall_seconds": time.monotonic() - start},
    )
    common.snapshot_gpu(run / "gpu_after.txt", gpu)
    telemetry(run / "gpu_after.json", gpu)
    assert provenance() == identities, "Source or harness changed during run"
    if process.returncode:
        raise RuntimeError(f"Run failed rc={process.returncode}: {run / 'run.log'}")
    result = read(run / "worker.json")
    assert result["pass"] and result["mode"] == mode
    if mode == "validate":
        assert result["validation"]["pass"]
        assert result["validation"]["cold_exact_replays"] == validation_replays
    elif mode == "smoke":
        assert result["smoke"]["pass"]
    write(completed, {"pass": True, "provenance": identities})
    return result


def comparisons(summary):
    groups = {}
    for value in summary.values():
        groups.setdefault((value["precision"], value["tokens"]), {}).setdefault(
            value["backend"], []
        ).append(value)
    result = []
    for (precision, tokens), backends in sorted(groups.items()):
        for gluon in backends.get("gluon", []):
            for flydsl in backends.get("flydsl", []):
                metrics = {}
                for field in ("mean_us", "median_us", "p99_us"):
                    g_value, f_value = gluon[field], flydsl[field]
                    metrics[field.removesuffix("_us")] = {
                        "gluon_us": g_value,
                        "flydsl_us": f_value,
                        "gluon_minus_flydsl_us": g_value - f_value,
                        "flydsl_speedup": g_value / f_value,
                        "flydsl_latency_reduction_percent": 100
                        * (g_value - f_value)
                        / g_value,
                    }
                result.append(
                    {
                        "precision": precision,
                        "tokens": tokens,
                        "gluon_case": gluon["case"],
                        "flydsl_case": flydsl["case"],
                        "metrics": metrics,
                    }
                )
    return result


def precision_name(value):
    return {
        "a4w4": "A4W4",
        "a8w4": "A8W4",
        "a8w8": "A8W8",
        "a16w16": "BF16",
    }.get(value, value)


def tile_text(value):
    tuning = value["effective_tuning"]
    if value["backend"] == "gluon":
        return f"BM{tuning['BLOCK_M']}xBN{tuning['BLOCK_N']}xBK{tuning['BLOCK_K']}"
    return f"TM{tuning['tile_m']}xTN{tuning['tile_n']}xTK{tuning['tile_k']}"


def flag_text(value):
    tuning = value["effective_tuning"]
    if value["backend"] == "gluon":
        keys = (
            ("A_SCALE_SORTED_SHUFFLED", "A-scale-shuffle"),
            ("B_SCALE_SHUFFLED", "B-scale-shuffle"),
            ("B_PRESHUFFLED", "B-preshuffle"),
        )
        flags = [label for key, label in keys if tuning.get(key)]
        flags.append(f"MFMA={tuning.get('mfma_instr_shape')}")
        flags.append(f"warps={tuning.get('warps_per_cta')}")
        if "TRITON_AMD_NT_ONLY" in value["env"]:
            flags.append(f"NT={value['env']['TRITON_AMD_NT_ONLY']}")
        return ", ".join(flags)
    fields = (
        f"wpe={tuning.get('waves_per_eu')}",
        f"b_nt={tuning.get('b_nt')}",
        f"nt={tuning.get('use_nt')}",
        f"gate={tuning.get('gate_mode')}",
        f"async={tuning.get('use_async_copy')}",
    )
    return ", ".join(fields)


def validation_summary(out, plan):
    label = plan.get("validation_label")
    if not label:
        return []
    root = ROOT / label
    groups = {}
    for case, spec in plan["cases"].items():
        key = (spec["precision"], spec["backend"])
        group = groups.setdefault(
            key,
            {
                "precision": spec["precision"],
                "backend": spec["backend"],
                "passed": 0,
                "expected": 0,
                "replays": [],
                "max_errors": [],
                "rms_errors": [],
                "references": set(),
            },
        )
        for tokens in sorted({item["tokens"] for item in plan["runs"]}):
            group["expected"] += 1
            path = root / f"{case}_t{tokens}" / "worker.json"
            if not path.exists():
                continue
            result = read(path)
            validation = result.get("validation", {})
            if result.get("pass") and validation.get("pass"):
                group["passed"] += 1
            if "cold_exact_replays" in validation:
                group["replays"].append(validation["cold_exact_replays"])
            max_error = validation.get(
                "max_normalized_error",
                validation.get("max_normalized_excess_error"),
            )
            rms_error = validation.get(
                "rms_normalized_error",
                validation.get("rms_normalized_excess_error"),
            )
            if max_error is not None:
                group["max_errors"].append(max_error)
            if rms_error is not None:
                group["rms_errors"].append(rms_error)
            if validation.get("reference"):
                group["references"].add(validation["reference"])
    result = []
    for group in groups.values():
        group["min_replays"] = min(group.pop("replays"), default=0)
        group["worst_max_error"] = max(group.pop("max_errors"), default=None)
        group["worst_rms_error"] = max(group.pop("rms_errors"), default=None)
        group["references"] = sorted(group["references"])
        result.append(group)
    return sorted(result, key=lambda item: (item["precision"], item["backend"]))


def write_report(out, summary, comparison):
    plan = read(out / "plan.json")
    provenance_data = read(out / "provenance.json")
    failures = read(out / "failures.json") if (out / "failures.json").exists() else []
    expected_runs = len(plan["runs"])
    complete = len(summary) == len(plan["cases"]) * len(
        {item["tokens"] for item in plan["runs"]}
    ) and not failures
    first = next(iter(summary.values()), None)
    gpu_info = read(out / "gpu_inventory.json").get(str(plan["gpu"]), {})
    device = first.get("device", {}) if first else {}
    versions = first.get("versions", {}) if first else {}
    title_mode = plan["mode"].capitalize()
    lines = [
        f"# {title_mode} Gluon versus FlyDSL GEMM1 results",
        "",
        (
            f"Status: **{'complete' if complete else 'partial'}**; "
            f"{len(read(out / 'records.json')) if (out / 'records.json').exists() else 0}/"
            f"{expected_runs} processes recorded; {len(failures)} failures."
        ),
        "",
        "## Scope and protocol",
        "",
        (
            "The logical GEMM1 shape is H/K=7168, intermediate=2048 "
            "(raw N=4096), E=33, top-k=8, with `SiLU(gate) * up` and BF16 "
            "output except for quantized A4W4 output."
        ),
        "",
        (
            f"Each process uses {plan['warmups']} cold warmups and {plan['samples']} "
            f"measured samples. Cold results use {plan['rounds']} alternating-order "
            f"rounds and pool all {plan['rounds'] * plan['samples']} device-duration "
            f"samples per cell. Every sample is exactly a {plan['flush_MiB']} MiB "
            "streaming fill followed by one prepared GEMM1 dispatch on the same "
            "stream; the trace parser rejects extra kernels."
        ),
        "",
        (
            "Routing, input quantization, A-scale sorting, weight/scale shuffle or "
            "preshuffle, output allocation, JIT compilation, and launch-argument "
            "preparation are setup work outside the measured loop. P99 is "
            "nearest-rank: `sorted[ceil(0.99*n)-1]`."
        ),
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | {device.get('name', 'pending')} |",
        f"| Architecture | `{device.get('arch', 'pending')}` |",
        f"| Physical / HIP index | {plan['gpu']} / {gpu_info.get('hip_ordinal', 'pending')} |",
        f"| PCI BDF / UUID | `{gpu_info.get('bdf', 'pending')}` / `{gpu_info.get('uuid', 'pending')}` |",
        f"| Commit / branch | `{provenance_data['head']}` / `{provenance_data.get('branch', '')}` |",
        f"| Tracked tree | {'clean' if not provenance_data.get('tracked_status') else 'modified'} |",
        f"| Python | `{str(versions.get('python', 'pending')).splitlines()[0]}` |",
        (
            f"| Torch / HIP / Triton | `{versions.get('torch', 'pending')}` / "
            f"`{versions.get('hip', 'pending')}` / "
            f"`{versions.get('triton', 'pending')}` |"
        ),
        "",
        "## Median winner matrix",
        "",
        "A cell shows the lower-latency backend and its speedup over the other backend.",
        "",
    ]
    tokens = sorted({item["tokens"] for item in comparison})
    lines += [
        "| Precision | " + " | ".join(str(value) for value in tokens) + " |",
        "|---|" + "---:|" * len(tokens),
    ]
    by_pair = {(item["precision"], item["tokens"]): item for item in comparison}
    for precision in sorted({item["precision"] for item in comparison}):
        cells = []
        for token_count in tokens:
            item = by_pair.get((precision, token_count))
            if item is None:
                cells.append("pending")
                continue
            ratio = item["metrics"]["median"]["flydsl_speedup"]
            cells.append(
                f"FlyDSL {ratio:.3f}x" if ratio >= 1 else f"Gluon {1 / ratio:.3f}x"
            )
        lines.append(f"| {precision_name(precision)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Backend results",
        "",
        "| Precision | Tokens | Backend | Samples | Mean (us) | Median (us) | P99 (us) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for value in sorted(
        summary.values(),
        key=lambda item: (
            item["precision"],
            item["tokens"],
            item["backend"],
            item["case"],
        ),
    ):
        lines.append(
            f"| {precision_name(value['precision'])} | {value['tokens']} | "
            f"{value['backend']} | {value['n']} | {value['mean_us']:.3f} | "
            f"{value['median_us']:.3f} | {value['p99_us']:.3f} |"
        )
    lines += [
        "",
        "## Pairwise comparison",
        "",
        "Positive delta and G/F ratio above 1 mean FlyDSL is faster.",
        "",
        "| Precision | Tokens | Statistic | Gluon (us) | FlyDSL (us) | Delta G-F (us) | G/F ratio | Winner |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in comparison:
        for statistic in ("mean", "median", "p99"):
            values = item["metrics"][statistic]
            lines.append(
                f"| {precision_name(item['precision'])} | {item['tokens']} | {statistic} | "
                f"{values['gluon_us']:.3f} | {values['flydsl_us']:.3f} | "
                f"{values['gluon_minus_flydsl_us']:+.3f} | "
                f"{values['flydsl_speedup']:.4f}x | "
                f"{'FlyDSL' if values['flydsl_speedup'] >= 1 else 'Gluon'} |"
            )
    lines += [
        "",
        "## Effective configurations and routing padding",
        "",
        "Active rows include per-expert padding; extra rows are not useful token/top-k rows.",
        "",
        "| Precision | Tokens | Backend | Tile | Key flags | Useful rows | Active rows | Extra | Padding | Blocks | Output layout / policy |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for value in sorted(
        summary.values(),
        key=lambda item: (item["precision"], item["tokens"], item["backend"]),
    ):
        extra = value["padded_rows"] - value["useful_rows"]
        padding = 100 * extra / value["useful_rows"]
        policy = value["output_quantization_policy"] or "none"
        lines.append(
            f"| {precision_name(value['precision'])} | {value['tokens']} | "
            f"{value['backend']} | `{tile_text(value)}` | {flag_text(value)} | "
            f"{value['useful_rows']} | {value['padded_rows']} | {extra} | "
            f"{padding:.2f}% | {value['active_blocks']} | "
            f"{value['output_layout']} / {policy} |"
        )
    validation = validation_summary(out, plan)
    if validation:
        lines += [
            "",
            "## Validation",
            "",
            (
                f"Cold measurements are gated by [`{plan['validation_label']}`]"
                f"(../{plan['validation_label']}/). The timed kernel, full logical "
                "inputs, and output hashes must match validation."
            ),
            "",
            "| Precision | Backend | Passed cells | Replays/cell | Worst max normalized error | Worst RMS normalized error |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for item in validation:
            max_error = item["worst_max_error"]
            rms_error = item["worst_rms_error"]
            lines.append(
                f"| {precision_name(item['precision'])} | {item['backend']} | "
                f"{item['passed']}/{item['expected']} | {item['min_replays']} | "
                f"{max_error:.6g} | {rms_error:.6g} |"
            )
    lines += [
        "",
        "## Interpretation notes",
        "",
        (
            "A4W4 native output quantization uses round-to-even scales in Gluon and "
            "round-up scales in FlyDSL; each backend is checked against the matching "
            "independent reference. Gluon emits expert-sorted rows, while FlyDSL may "
            "emit token-slot or sorted native layouts; layout conversion is excluded "
            "because only the prepared GEMM1 is measured."
        ),
        "",
        "## Artifacts",
        "",
        (
            "[`plan.json`](plan.json) · [`provenance.json`](provenance.json) · "
            "[`gpu_inventory.json`](gpu_inventory.json) · "
            "[`summary.json`](summary.json) · "
            "[`comparison.json`](comparison.json) · [`records.json`](records.json) · "
            "[`samples.csv`](samples.csv) · [`samples.json`](samples.json) · "
            "[`failures.json`](failures.json) · "
            "[`independent_audit.json`](independent_audit.json)"
        ),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def summarize(out, records, samples):
    write(out / "records.json", records)
    write(out / "samples.json", samples)
    summary = {}
    groups = sorted({(r["precision"], r["tokens"], r["case"]) for r in records})
    for precision, tokens, case in groups:
        selected = [
            sample["duration_us"]
            for sample in samples
            if sample["tokens"] == tokens and sample["case"] == case
        ]
        rounds = [
            record
            for record in records
            if record["tokens"] == tokens and record["case"] == case
        ]
        backend = rounds[0]["backend"]
        stable_fields = (
            "tuning",
            "env",
            "effective_tuning",
            "useful_rows",
            "padded_rows",
            "allocated_rows",
            "active_blocks",
            "output_layout",
            "output_quantization_policy",
            "device",
            "versions",
            "setup_excluded_from_timing",
        )
        for field in stable_fields:
            assert all(record[field] == rounds[0][field] for record in rounds), (
                "metadata changed across rounds",
                precision,
                tokens,
                case,
                field,
            )
        summary[f"{precision}_t{tokens}_{case}"] = {
            "precision": precision,
            "tokens": tokens,
            "case": case,
            "backend": backend,
            "n": len(selected),
            "mean_us": st.mean(selected),
            "median_us": st.median(selected),
            "p99_us": nearest_rank(selected, 0.99),
            "stdev_us": st.pstdev(selected),
            "round_means_us": [record["mean_us"] for record in rounds],
            "round_medians_us": [record["median_us"] for record in rounds],
            "round_p99_us": [record["p99_us"] for record in rounds],
            **{field: rounds[0][field] for field in stable_fields},
        }
    write(out / "summary.json", summary)
    comparison = comparisons(summary)
    write(
        out / "comparison.json",
        {
            "p99_definition": "nearest rank: sorted[ceil(0.99*n)-1]",
            "comparisons": comparison,
        },
    )
    write_report(out, summary, comparison)
    if samples:
        with (out / "samples.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    return summary, comparison


def build_plan(mode, rounds, tokens, cases):
    plan = []
    for round_index in range(1, rounds + 1):
        token_order = tokens if round_index % 2 else list(reversed(tokens))
        case_order = cases if round_index % 2 else list(reversed(cases))
        for token_count in token_order:
            for case in case_order:
                plan.append({"round": round_index, "tokens": token_count, "case": case})
    if mode == "validate":
        assert rounds == 1
    return plan


def require_paired_case_order(cases, specs):
    """Require adjacent same-precision Gluon/FlyDSL pairs for balanced AB/BA rounds."""
    assert len(cases) % 2 == 0, "cold comparison requires backend pairs"
    for offset in range(0, len(cases), 2):
        first, second = cases[offset : offset + 2]
        assert specs[first]["precision"] == specs[second]["precision"], (
            "adjacent cold cases must have the same precision",
            first,
            second,
        )
        assert {specs[first]["backend"], specs[second]["backend"]} == {
            "gluon",
            "flydsl",
        }, ("adjacent cold cases must be a Gluon/FlyDSL pair", first, second)


def main(argv=None):
    global GPU_MAP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "validate", "cold"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--validation-label", default="validation")
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        choices=TOKEN_COUNTS,
        default=None,
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--backends", nargs="+", choices=("gluon", "flydsl"))
    parser.add_argument(
        "--precisions", nargs="+", choices=("a4w4", "a8w4", "a8w8", "a16w16")
    )
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--validation-replays", type=int, default=16)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    specs = load_cases()
    if (
        args.mode == "smoke"
        and args.cases is None
        and args.backends is None
        and args.precisions is None
    ):
        selected = ["gluon_a4w4", "flydsl_a4w4"]
    else:
        selected = list(specs) if args.cases is None else args.cases
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")
    if args.backends:
        selected = [
            case for case in selected if specs[case]["backend"] in args.backends
        ]
    if args.precisions:
        selected = [
            case for case in selected if specs[case]["precision"] in args.precisions
        ]
    if not selected:
        parser.error("case/backend/precision filters selected no cases")
    token_counts = (
        args.tokens
        if args.tokens is not None
        else ([16] if args.mode == "smoke" else list(TOKEN_COUNTS))
    )
    if len(set(token_counts)) != len(token_counts):
        parser.error("--tokens contains duplicates")

    rounds = (
        args.rounds if args.rounds is not None else (4 if args.mode == "cold" else 1)
    )
    warmups = (
        args.warmups
        if args.warmups is not None
        else (1 if args.mode == "smoke" else 40)
    )
    samples = (
        args.samples
        if args.samples is not None
        else (3 if args.mode == "smoke" else 100)
    )
    if args.mode == "validate":
        rounds = 1
    if rounds <= 0 or warmups < 0 or samples <= 0 or args.validation_replays <= 0:
        parser.error("rounds/samples/replays must be positive and warmups non-negative")

    if args.mode == "cold":
        require_paired_case_order(selected, specs)
    plan = build_plan(args.mode, rounds, token_counts, selected)
    plan_record = {
        "runs": plan,
        "mode": args.mode,
        "rounds": rounds,
        "warmups": warmups,
        "samples": samples,
        "validation_replays": args.validation_replays,
        "validation_label": args.validation_label if args.mode == "cold" else None,
        "flush_MiB": 768,
        "preparation": "setup only; one captured GEMM launch after each flush",
        "gpu": args.gpu,
        "cases": {case: specs[case] for case in selected},
        "resolved_cases": {
            f"{case}_t{tokens}": load_cases(tokens)[case]
            for tokens in token_counts
            for case in selected
        },
        "p99_definition": "nearest rank: sorted[ceil(0.99*n)-1]",
    }
    if args.dry_run:
        print(json.dumps(plan_record, indent=2))
        return 0

    with (ROOT / "serial.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = ROOT / args.label
        out.mkdir(exist_ok=True)
        GPU_MAP = gpu_inventory()
        if args.gpu not in GPU_MAP:
            parser.error(f"physical GPU {args.gpu} is unavailable")
        write(out / "gpu_inventory.json", GPU_MAP)
        identities = provenance()
        if (out / "provenance.json").exists():
            assert read(out / "provenance.json") == identities
        write(out / "provenance.json", identities)
        write(out / "plan.json", plan_record)

        validation_root = ROOT / args.validation_label
        if args.mode == "cold":
            assert read(validation_root / "provenance.json") == identities, (
                "validation provenance differs from the timed harness/source"
            )
            assert read(validation_root / "failures.json") == [], (
                "validation set contains failures"
            )

        records, sample_rows, validation_records, failures = [], [], [], []
        input_identities, kernel_identities = {}, {}
        for item in plan:
            case, tokens = item["case"], item["tokens"]
            run = out / (
                f"{case}_t{tokens}"
                if args.mode == "validate"
                else f"r{item['round']:02d}_{case}_t{tokens}"
            )
            try:
                checked = None
                if args.mode == "cold":
                    validation_run = validation_root / f"{case}_t{tokens}"
                    assert read(validation_run / "complete.json") == {
                        "pass": True,
                        "provenance": identities,
                    }
                    checked = read(validation_run / "worker.json")
                    assert checked["pass"] and checked["mode"] == "validate"
                    assert checked["validation"]["pass"]
                    assert (
                        checked["validation"]["cold_exact_replays"]
                        >= args.validation_replays
                    )
                result = execute(
                    case,
                    tokens,
                    args.gpu,
                    run,
                    specs,
                    args.mode,
                    identities,
                    warmups,
                    samples,
                    args.validation_replays,
                )
                precision = specs[case]["precision"]
                inputs = stable_inputs(result["inputs"])
                input_key = f"{precision}_t{tokens}"
                if input_key in input_identities:
                    assert inputs == input_identities[input_key], (
                        "Logical inputs differ"
                    )
                input_identities[input_key] = inputs
                kernel = stable_kernel(result["kernel"])
                kernel_key = (tokens, case)
                if kernel_key in kernel_identities:
                    assert kernel == kernel_identities[kernel_key], (
                        "Kernel changed across rounds"
                    )
                kernel_identities[kernel_key] = kernel

                if args.mode == "validate":
                    validation_records.append(
                        {
                            **item,
                            "precision": precision,
                            "backend": specs[case]["backend"],
                            "validation": result["validation"],
                        }
                    )
                    write(out / "validation_records.json", validation_records)
                    print(
                        f"PASS {case} T={tokens}: reference + "
                        f"{args.validation_replays} cold replays",
                        flush=True,
                    )
                    continue

                if checked is not None:
                    assert kernel == stable_kernel(checked["kernel"]), (
                        "Timed kernel differs from validation"
                    )
                    assert result["inputs"] == checked["inputs"], (
                        "Timed inputs differ from validation"
                    )
                    assert (
                        result["output_hashes"]
                        == checked["validation"]["output_hashes"]
                    ), "Timed output differs from validation"
                metadata, shots = trace_values(
                    run, result["kernel_pattern"], warmups, samples
                )
                routing = result.get("routing", {})
                input_routing = result["routing_metadata"]
                padded_rows = routing.get(
                    "native_padded_rows", input_routing["padded_rows"]
                )
                allocated_rows = routing.get(
                    "native_allocated_rows", input_routing["allocated_rows"]
                )
                block_m = result["shape"]["block_m"]
                record = {
                    **item,
                    "precision": precision,
                    "backend": specs[case]["backend"],
                    "tuning": result["case_spec"]["tuning"],
                    "env": result["case_spec"].get("env", {}),
                    "effective_tuning": result["kernel"].get(
                        "effective_tuning", result["kernel"]["config"]
                    ),
                    "useful_rows": input_routing["useful_rows"],
                    "padded_rows": padded_rows,
                    "allocated_rows": allocated_rows,
                    "active_blocks": padded_rows // block_m,
                    "output_layout": result["kernel"]["output_layout"],
                    "output_quantization_policy": result["kernel"].get(
                        "output_quantization_policy"
                    ),
                    "device": result["device"],
                    "versions": result["versions"],
                    "setup_excluded_from_timing": result[
                        "setup_excluded_from_timing"
                    ],
                    **metadata,
                }
                records.append(record)
                sample_rows.extend(
                    {
                        **item,
                        "precision": precision,
                        "backend": specs[case]["backend"],
                        **shot,
                    }
                    for shot in shots
                )
                summarize(out, records, sample_rows)
                print(
                    f"DONE {case} T={tokens} mean={metadata['mean_us']:.3f} us "
                    f"median={metadata['median_us']:.3f} us "
                    f"p99={metadata['p99_us']:.3f} us",
                    flush=True,
                )
            except Exception as exc:
                import traceback

                failures.append(
                    {**item, "error": str(exc), "traceback": traceback.format_exc()}
                )
                write(out / "failures.json", failures)
                print(f"FAIL {case} T={tokens}: {exc}", flush=True)
                if not args.keep_going:
                    raise

        write(out / "logical_inputs.json", input_identities)
        write(out / "failures.json", failures)
        common.wait_for_gpu_idle(out / "idle_after.jsonl", args.gpu)
        assert provenance() == identities
        if records:
            summary, comparison = summarize(out, records, sample_rows)
            print(
                json.dumps({"summary": summary, "comparisons": comparison}, indent=2),
                flush=True,
            )
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
