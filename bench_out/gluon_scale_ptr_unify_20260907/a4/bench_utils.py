"""Shared setup for the preserved-source cold and correctness runners."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

BASE = Path(__file__).resolve().parent
CURRENT = BASE.parents[2]
BASELINE = BASE / "baseline/source"
RUNTIME = BASE / "preserved_harness/runtime"
PY = Path("/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python")
LLC = Path("/raid/jinpli/workspace/home01/jinpli/development/llvm-project/build/bin/llc")
LLC_FLAGS = "-amdgpu-ds-read-agpr -amdgpu-mfma-tied-cd -amdgpu-no-sched-revert -misched-pin-critical-res=HWXDL"
OVERRIDES = {
    "impl": {
        "AITER_TRITON_MOE_GLUON_WARP_PIPELINE": "0",
        "AITER_TRITON_MOE_GLUON_GU_SPLIT": "1",
        "AITER_TRITON_MOE_GLUON_WAVES_PER_EU": "1",
        "AITER_TRITON_MOE_GLUON_FUSE_ACT_MFMA": "1",
        "AITER_TRITON_MOE_GLUON_WAIT_COMMIT_SCHEME": "3",
        "TRITON_HIP_EXTERNAL_LLC_FLAGS": LLC_FLAGS + " -amdgpu-force-dynamic-lds-size=161216 -disable-post-ra -amdgpu-sched-strategy=coexec -amdgpu-coexec-no-regpressure",
    },
    "frozen": {
        "AITER_TRITON_MOE_GLUON_WARP_PIPELINE": "1",
        "AITER_TRITON_MOE_GLUON_FROZEN_STEP": "1",
        "AITER_TRITON_MOE_GLUON_GU_SPLIT": "1",
        "TRITON_HIP_EXTERNAL_LLC_FLAGS": LLC_FLAGS,
    },
}
PREFIXES = ("AITER_TRITON_MOE_GLUON_", "TRITON_HIP_", "TRITON_MEMBAR_")
SOURCE_DIRS = (
    "aiter/ops/triton/_gluon_kernels/gfx950/moe",
    "aiter/ops/triton/moe",
)
SOURCE_FILES = (
    "scripts/gluon_moe_gemm1_best.py",
    "aiter/ops/triton/_triton_kernels/moe/activations.py",
    "aiter/ops/triton/utils/_triton/pid_preprocessing.py",
    "aiter/ops/triton/utils/common_utils.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def repositories(values):
    if not values:
        return {"baseline": BASELINE, "current": CURRENT}
    result = {}
    for value in values:
        label, path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in result:
            raise ValueError(f"Invalid or duplicate repository label: {label}")
        result[label] = Path(path).resolve()
    return result


def source_identity(repo):
    files = set(SOURCE_FILES)
    for directory in SOURCE_DIRS:
        files.update(str(path.relative_to(repo)) for path in (repo / directory).rglob("*.py"))
    return {
        "repo": str(repo),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "files": {name: sha(repo / name) for name in sorted(files)},
    }


def base_env(repo, job, gpu):
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(PREFIXES)
        and k not in ("OV", "TRITON_CACHE_DIR", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    }
    env.update(
        GLUON_VALIDATION_REPO=str(repo),
        BENCH_H="7168",
        CLAUDE_JOB_DIR=str(job),
        ROCM_PATH=str(PY.parent.parent / "lib/python3.14/site-packages/_rocm_sdk_devel"),
        TRITON_HIP_DS_AGPR_LLC=str(LLC),
        HIP_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(RUNTIME) + os.pathsep + str(repo),
        PYTHONDONTWRITEBYTECODE="1",
        CK_DIR=str(CURRENT / "3rdparty/composable_kernel"),
    )
    return env


def correctness_env(repo, cache, gpu, arm):
    spec = importlib.util.spec_from_file_location("preserved_best", BASELINE / "scripts/gluon_moe_gemm1_best.py")
    best = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(best)
    env = base_env(repo, cache.parent / "job", gpu)
    env.update(best.BEST)
    env.update(best.PER_WARPS[4])
    env.update(OVERRIDES[arm])
    env["AITER_TRITON_MOE_GLUON_FLY_WARPS_N"] = best.WARPS_N[4]
    env["TRITON_HIP_EXTERNAL_LLC"] = str(LLC)
    env["TRITON_CACHE_DIR"] = str(cache)
    return env


def relevant_env(env):
    names = ("BENCH_H", "CLAUDE_JOB_DIR", "ROCM_PATH", "TRITON_CACHE_DIR", "HIP_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CK_DIR", "AITER_MOE_NUM_EXPERT_ACTIVATED")
    return {k: v for k, v in env.items() if k.startswith(PREFIXES + ("GLUON_",)) or k in names}


def snapshot_gpu(path, gpu):
    result = subprocess.run(
        [str(PY.parent / "rocm-smi"), "-d", str(gpu), "--showuse", "--showmemuse", "--showclocks", "--showtemp", "--showpids"],
        capture_output=True, text=True, check=False,
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
        for following in lines[i + 1:]:
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
    """Read utilization and map foreign processes to this physical GPU."""
    smi = subprocess.run(
        [str(PY.parent / "rocm-smi"), "-d", str(gpu), "--showuse", "--showmeminfo", "vram", "--json"],
        capture_output=True, text=True, check=True,
    )
    mapping = subprocess.run(
        [str(PY.parent / "rocm-smi"), "--showpidgpus"],
        capture_output=True, text=True, check=True,
    )
    processes = subprocess.run(
        [str(PY.parent / "amd-smi"), "process", "-g", str(gpu), "--json"],
        capture_output=True, text=True, check=True,
    )
    card = json.loads(smi.stdout)[f"card{gpu}"]
    mapped_pids = gpu_pids_from_mapping(mapping.stdout, gpu)
    device_processes = json.loads(processes.stdout)
    allocations = []
    for device in device_processes:
        if device.get("gpu") != gpu:
            continue
        for item in device.get("process_list", []):
            info = item["process_info"]
            vram = info.get("memory_usage", {}).get("vram_mem", {}).get("value", 0)
            if isinstance(vram, (int, float)) and vram > 0:
                allocations.append({"pid": info["pid"], "vram_bytes": vram})
    use = int(card["GPU use (%)"])
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "gpu": gpu, "gpu_use_percent": use,
        "vram_used_bytes": int(card["VRAM Total Used Memory (B)"]),
        "mapped_pids": mapped_pids, "foreign_allocations": allocations,
        "idle": use == 0 and not mapped_pids and not allocations,
        "raw_process_mapping": mapping.stdout, "raw_processes": device_processes,
    }


def wait_for_gpu_idle(path, gpu, expected_idle_vram=None):
    """Require two idle readings before launching; never disturb another process."""
    consecutive = 0
    last = None
    with Path(path).open("w") as log:
        while consecutive < 2:
            state = gpu_idle_state(gpu)
            if expected_idle_vram is not None and state["vram_used_bytes"] > expected_idle_vram:
                state["idle"] = False
                state["extra_vram_bytes"] = state["vram_used_bytes"] - expected_idle_vram
            log.write(json.dumps(state) + "\n")
            log.flush()
            if state["idle"]:
                consecutive += 1
            else:
                consecutive = 0
                print(f"WAIT GPU {gpu}: use={state['gpu_use_percent']}%, pids={state['mapped_pids']}, allocations={state['foreign_allocations']}, extra_vram={state.get('extra_vram_bytes', 0)} B", flush=True)
            last = state
            if consecutive < 2:
                time.sleep(2 if state["idle"] else 5)
    return last


def elf_text(path):
    data = Path(path).read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected little-endian ELF64: {path}")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", data, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    n = headers[names_index]
    names = data[n[4]:n[4] + n[5]]
    return next(data[h[4]:h[4] + h[5]] for h in headers if names[h[0]:].split(b"\0", 1)[0] == b".text")


def code_identity(cache, arm, output):
    kernels = list(Path(cache).rglob("_moe_gluon_gemm1.hsaco"))
    if len(kernels) != 1:
        raise ValueError(f"Expected exactly one GEMM1 binary in {cache}, found {len(kernels)}")
    binary = kernels[0]
    payload = elf_text(binary)
    metadata = json.loads(binary.with_suffix(".json").read_text())
    full_launch = {key: value for key, value in metadata.items() if key != "hash"}
    digest = hashlib.sha256(payload).hexdigest()
    launch = {k: metadata[k] for k in ("num_warps", "shared", "waves_per_eu")}
    result = {
        "binary": str(binary), "text_sha256": digest, "text_bytes": len(payload),
        "binary_sha256": sha(binary),
        "launch_metadata": launch,
        "full_launch_metadata": full_launch,
        "same_text": None, "same_launch_metadata": None, "same_full_launch_metadata": None,
        "identity_required": arm == "frozen",
    }
    fingerprints = BASE / "baseline/fingerprints.json"
    if fingerprints.exists():
        expected = next(r for r in json.loads(fingerprints.read_text()) if r["arm"] == arm and r["output"] == output)
        expected_metadata = json.loads((BASE / "baseline/binaries" / f"{arm}_{output}" / binary.with_suffix(".json").name).read_text())
        expected_full_launch = {key: value for key, value in expected_metadata.items() if key != "hash"}
        result.update(
            same_text=digest == expected["text_sha256"] and len(payload) == expected["text_bytes"],
            same_binary=sha(binary) == expected.get("binary_sha256"),
            same_launch_metadata=launch == expected["launch_metadata"],
            same_full_launch_metadata=full_launch == expected_full_launch,
            expected_text_sha256=expected["text_sha256"],
        )
    return result
