"""Shared setup for the preserved-source cold and correctness runners."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# The interpreter run_bench.py runs under; amd-smi, rocm-smi and rocprofv3 sit next to it.
PY = Path(sys.executable)
# Locally built llc with the out-of-tree AMDGPU flags below (tests/_env.sh sets it).
LLC = Path(os.environ["TRITON_HIP_DS_AGPR_LLC"])
LLC_FLAGS = "-amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd -amdgpu-no-sched-revert -misched-pin-critical-res=HWXDL"
PREFIXES = ("AITER_TRITON_MOE_GLUON_", "TRITON_HIP_", "TRITON_MEMBAR_")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def base_env(repo, gpu):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(PREFIXES)
        and k
        not in (
            "OV",
            "TRITON_CACHE_DIR",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
    }
    env.update(
        GLUON_VALIDATION_REPO=str(repo),
        BENCH_H="7168",
        ROCM_PATH=str(
            PY.parent.parent / "lib/python3.14/site-packages/_rocm_sdk_devel"
        ),
        TRITON_HIP_DS_AGPR_LLC=str(LLC),
        HIP_VISIBLE_DEVICES=str(gpu),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def snapshot_gpu(path, gpu):
    result = subprocess.run(
        [
            str(PY.parent / "rocm-smi"),
            "-d",
            str(gpu),
            "--showuse",
            "--showmemuse",
            "--showclocks",
            "--showtemp",
            "--showpids",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    Path(path).write_text(result.stdout + result.stderr)


def gpu_pids_from_mapping(text, gpu):
    lines = text.splitlines()
    pids = []
    for i, line in enumerate(lines):
        match = re.match(r"PID (\d+) is using (\d+) DRM device\(s\)", line)
        if not match or not int(match.group(2)):
            continue
        devices = []
        for following in lines[i + 1 :]:
            if not re.fullmatch(r"[\d\s]+", following):
                break
            devices.extend(int(value) for value in following.split())
            if len(devices) >= int(match.group(2)):
                break
        if len(devices) != int(match.group(2)):
            raise ValueError(f"Could not parse process-to-GPU mapping: {line}")
        if gpu in devices:
            pids.append(int(match.group(1)))
    return pids


def gpu_idle_state(gpu):
    """Replaced by run_bench.gpu_idle_state at import time."""
    raise NotImplementedError


def wait_for_gpu_idle(path, gpu, expected_idle_vram=None):
    """Require two idle readings before launching; never disturb another process."""
    consecutive = 0
    last = None
    with Path(path).open("w") as log:
        while consecutive < 2:
            state = gpu_idle_state(gpu)
            if (
                expected_idle_vram is not None
                and state["vram_used_bytes"] > expected_idle_vram
            ):
                state["idle"] = False
                state["extra_vram_bytes"] = (
                    state["vram_used_bytes"] - expected_idle_vram
                )
            log.write(json.dumps(state) + "\n")
            log.flush()
            if state["idle"]:
                consecutive += 1
            else:
                consecutive = 0
                print(
                    f"WAIT GPU {gpu}: use={state['gpu_use_percent']}%, pids={state['mapped_pids']}, allocations={state['foreign_allocations']}, extra_vram={state.get('extra_vram_bytes', 0)} B",
                    flush=True,
                )
            last = state
            if consecutive < 2:
                time.sleep(2 if state["idle"] else 5)
    return last
