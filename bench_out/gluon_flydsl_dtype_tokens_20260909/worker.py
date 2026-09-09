"""Prepare one MoE GEMM1, then validate it or launch it with cold-cache fills."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import struct
import sys
import sysconfig
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("BENCH_REPO", ROOT.parents[1])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))


def unwrap(value):
    return value.value if hasattr(value, "value") else value


def verify_device_pci():
    """Match visible HIP device zero to the requested physical device."""
    libpath = (
        Path(sysconfig.get_path("purelib")) / "_rocm_sdk_core/lib/libamdhip64.so.7"
    )
    hip = ctypes.CDLL(str(libpath))
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(64)
    status = hip.hipDeviceGetPCIBusId(buffer, len(buffer), 0)
    assert status == 0, ("hipDeviceGetPCIBusId", status)

    def normalize(value):
        domain, bus, devfn = value.lower().split(":")
        device, function = devfn.split(".")
        return (
            f"{int(domain, 16):04x}:{int(bus, 16):02x}:"
            f"{int(device, 16):02x}.{int(function, 16):x}"
        )

    actual = normalize(buffer.value.decode())
    expected = normalize(os.environ["BENCH_EXPECT_PCI_BUS_ID"])
    assert actual == expected, (
        "requested physical GPU does not match HIP device zero",
        expected,
        actual,
    )
    return actual


def elf_text(blob):
    assert blob[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", blob, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", blob, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", blob, offset + index * size)
        for index in range(count)
    ]
    names = headers[names_index]
    names_blob = blob[names[4] : names[4] + names[5]]
    for header in headers:
        if names_blob[header[0] :].split(b"\0", 1)[0] == b".text":
            return blob[header[4] : header[4] + header[5]]
    raise ValueError("No .text section")


def build_gluon(data, tuning, artifact_dir):
    import bench_common as bc
    import torch

    from aiter.ops.triton._gluon_kernels.gfx950.moe._config import (
        KernelFuncConfig,
        KernelTuningConfig,
    )
    from aiter.ops.triton.moe import moe_op_gemm_gluon as host

    assert Path(host.__file__).resolve().is_relative_to(REPO), host.__file__
    quantized_output = data["identity"]["precision"] == "a4w4"
    y = torch.empty(
        (1, bc.T * bc.TOPK, bc.N // (4 if quantized_output else 2)),
        device="cuda",
        dtype=torch.uint8 if quantized_output else torch.bfloat16,
    )
    ys = (
        torch.empty((bc.T * bc.TOPK, bc.N // 64), device="cuda", dtype=torch.uint8)
        if quantized_output
        else None
    )
    w = data["w"].transpose(-1, -2)
    ws = None if data["ws"] is None else data["ws"].transpose(-1, -2)
    captured = []
    original = host._fast_launch

    def capture(*args):
        captured.append(args)
        return original(*args)

    host._fast_launch = capture
    try:
        kernel = host.moe_gemm_gluon(
            y,
            data["x"],
            w,
            data["xs"],
            ws,
            None,
            None,
            data["route"],
            data["gather"],
            None,
            bc.N,
            bc.K,
            True,
            1.0,
            None,
            False,
            y_scales=ys,
            config=tuning,
            gate_up_split=True,
        )
    finally:
        host._fast_launch = original
    assert len(captured) == 1
    assert kernel.name == "_moe_gluon_gemm1"
    launch = captured[0]

    def call():
        compiled = original(*launch)
        assert compiled is not None
        return compiled

    named = dict(zip(launch[0].arg_names, launch[2]))
    func_spec, tuning_spec = unwrap(named["CFG_FUNC"]), unwrap(named["CFG_TUNING"])
    func_config = KernelFuncConfig(*func_spec)
    tuning_config = KernelTuningConfig(func_config, *tuning_spec)
    assert tuning_config.validate(bc.N, bc.K)
    activation = func_spec.activation
    assert activation.alpha == 1.0 and activation.limit is None
    assert not activation.add_residual
    assert not func_spec.has_bias and not func_spec.has_gammas
    assert func_spec.gate_up_split
    assert (func_spec.output_quant is not None) == quantized_output
    effective = tuning_spec._asdict()
    for field, requested in tuning.items():
        assert field in effective, ("requested tuning field was dropped", field)
        actual = effective[field]
        if isinstance(actual, tuple):
            actual = list(actual)
        if isinstance(requested, tuple):
            requested = list(requested)
        assert actual == requested, (
            "requested tuning changed before launch",
            field,
            requested,
            actual,
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    binary = bytes(kernel.asm["hsaco"])
    (artifact_dir / "kernel.hsaco").write_bytes(binary)
    (artifact_dir / "kernel.amdgcn").write_text(kernel.asm["amdgcn"])
    text = elf_text(binary)
    outputs = (y, ys) if quantized_output else (y,)

    def sorted_components():
        return (y[0], ys) if quantized_output else (y[0],)

    return {
        "call": call,
        "output": y,
        "outputs": outputs,
        "sorted_components": sorted_components,
        "sorted_output": (
            (lambda: bc.decode_mxfp4(y[0], ys)) if quantized_output else (lambda: y[0])
        ),
        "kernel_pattern": "_moe_gluon_gemm1",
        "keepalive": (launch, data, w, ws, outputs),
        "metadata": {
            "config": tuning,
            "kernel": kernel.name,
            "effective_tuning": tuning_spec._asdict(),
            "func_spec": func_spec._asdict(),
            "launch_metadata": kernel.metadata._asdict(),
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "text_sha256": hashlib.sha256(text).hexdigest(),
            "text_bytes": len(text),
            "saved_executable": str(artifact_dir / "kernel.hsaco"),
            "output_layout": "dense expert-sorted rows",
            "output_quantization_policy": "even" if quantized_output else None,
            "scale_sort_timing": "setup only; prepared launch arguments captured",
            "prepared_gemm_dispatches_per_call": 1,
            "grid_m": data["routing_metadata"]["gluon_grid_m"],
        },
    }


def build_flydsl(data, config, artifact_dir):
    import bench_common as bc
    import torch
    from fly_build import build_fly, routing_from_topk

    tile_m = config["tuning"]["tile_m"]
    routing = routing_from_topk(
        data["topk_ids"], bc.E, tile_m, logical_order=data["order"]
    )
    case = build_fly(
        config["precision"],
        data["x"],
        data["w"],
        data["topk_ids"],
        x_scales=data["xs"],
        w_scales=data["ws"],
        routing=routing,
        **config["tuning"],
    )

    def sorted_components():
        return tuple(
            value.reshape(bc.T * bc.TOPK, value.shape[-1])[data["order"]]
            for value in case.token_order_components()
        )

    def sorted_output():
        return case.token_order().reshape(bc.T * bc.TOPK, bc.N // 2)[data["order"]]

    ids = case.sorted_token_ids.to(torch.int64)
    tokens, slots = ids & 0xFFFFFF, ids >> 24
    valid = (tokens < bc.T) & (slots >= 0) & (slots < bc.TOPK)
    valid &= torch.arange(ids.numel(), device=ids.device) < case.num_valid_ids[0]
    row_experts = case.sorted_expert_ids.repeat_interleave(tile_m)[: ids.numel()]
    assert torch.equal(
        data["topk_ids"][tokens[valid], slots[valid]].int(), row_experts[valid].int()
    )
    assert int(valid.sum()) == bc.T * bc.TOPK
    logical_rows = tokens[valid] * bc.TOPK + slots[valid]
    assert torch.equal(
        torch.sort(logical_rows).values, torch.arange(bc.T * bc.TOPK, device="cuda")
    )
    assert torch.equal(logical_rows, data["order"])
    pattern = {
        "a4w4": "gemm1_a4w4_port",
        "a8w4": "mfma_moe1_silu_mul_afp8_wfp4_",
        "a8w8": "mfma_moe1_silu_mul_afp8_wfp8_",
        "a16w16": "gemm1_a16w4_port_a16w4_bf16_",
    }[config["precision"]]
    fingerprint = case.fingerprint(artifact_dir)
    return {
        "call": case.call,
        "output": case.out,
        "outputs": tuple(case.outputs),
        "sorted_components": sorted_components,
        "sorted_output": sorted_output,
        "kernel_pattern": pattern,
        "keepalive": case,
        "metadata": {
            "config": case.config,
            "output_layout": case.layout,
            "output_quantization_policy": (
                "round_up" if config["precision"] == "a4w4" else None
            ),
            "executable": fingerprint,
            "native_padded_rows": int(case.num_valid_ids[0]),
            "routing_membership_verified": True,
            "scale_sort_timing": "setup only; prepared launch arguments captured",
            "prepared_gemm_dispatches_per_call": 1,
            "exe_type": str(type(case.exe)),
        },
        "routing": {
            "tile_m": tile_m,
            "useful_rows": bc.T * bc.TOPK,
            "native_padded_rows": int(case.num_valid_ids[0]),
            "native_allocated_rows": ids.numel(),
            "native_allocated_blocks": case.sorted_expert_ids.numel(),
        },
        "saved_executable": fingerprint,
    }


def main():
    from case_config import load_cases

    # Parse first so token-specific tuning is resolved before the selected case is built.
    base_cases = load_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(base_cases), required=True)
    parser.add_argument(
        "--tokens", type=int, choices=(16, 64, 256, 1024, 4096), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "validate", "cold"), required=True)
    parser.add_argument("--warmups", type=int, default=40)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--validation-replays", type=int, default=16)
    args = parser.parse_args()
    assert args.warmups >= 0 and args.samples > 0 and args.validation_replays > 0
    config = load_cases(args.tokens)[args.case]

    # Dtype/config environment is constexpr state and must be clean before aiter imports.
    for name in list(os.environ):
        if name.startswith(("AITER_TRITON_MOE_", "AITER_FLYDSL_")):
            os.environ.pop(name)
    os.environ.update(config.get("env", {}))
    os.environ["AITER_TRITON_USE_HERD"] = "0"

    import bench_common as bc
    import torch
    import triton

    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.startswith("gfx950"), props.gcnArchName
    pci_bus_id = verify_device_pci()
    block_m = config["tuning"]["BLOCK_M" if config["backend"] == "gluon" else "tile_m"]
    bc.configure(7168, block_m, tokens=args.tokens)
    result = {
        "case": args.case,
        "case_spec": config,
        "source": str(REPO),
        "shape": dict(bc.SHAPE),
        "precision": config["precision"],
        "backend": config["backend"],
        "mode": args.mode,
        "pass": False,
        "device": {
            "name": props.name,
            "arch": props.gcnArchName,
            "physical_gpu": os.environ.get("BENCH_PHYSICAL_GPU"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "pci_bus_id": pci_bus_id,
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
        },
        "setup_excluded_from_timing": [
            "routing",
            "input quantization",
            "A-scale sorting",
            "weight and scale shuffle/preshuffle",
            "output allocation",
            "JIT compilation",
            "launch argument preparation",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bc.write_json(args.out, result)
    try:
        print(f"PREPARE {args.case} T={args.tokens}", flush=True)
        data = bc.make_inputs(config["precision"])
        result["inputs"] = data["identity"]
        result["routing_metadata"] = data["routing_metadata"]
        built = (
            build_gluon(data, config["tuning"], args.out.parent / "gluon_artifact")
            if config["backend"] == "gluon"
            else build_flydsl(data, config, args.out.parent / "fly_artifact")
        )
        result.update(kernel=built["metadata"], kernel_pattern=built["kernel_pattern"])
        for key in ("routing", "saved_executable"):
            if key in built:
                result[key] = built[key]
        bc.write_json(args.out, result)

        if args.mode == "validate":
            print(f"VALIDATE {args.case} T={args.tokens}", flush=True)
            result["validation"] = bc.validate(
                built, data, replays=args.validation_replays
            )
        else:
            if args.mode == "smoke":
                result["smoke"] = bc.smoke_validate(built)
            flush = torch.empty(768 << 20, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            print(
                f"COLD {args.case} T={args.tokens}: "
                f"{args.warmups} warmups, {args.samples} measured",
                flush=True,
            )
            for _ in range(args.warmups):
                flush.zero_()
                built["call"]()
            torch.cuda.synchronize()
            for _ in range(args.samples):
                flush.zero_()
                built["call"]()
            torch.cuda.synchronize()
            result["timing_protocol"] = {
                "warmup": args.warmups,
                "samples": args.samples,
                "flush_MiB": 768,
                "sequence": "flush -> one prepared GEMM1",
                "scale_sort_timing": "setup only",
                "timing": "rocprofv3 kernel trace",
            }
            result["output_hashes"] = bc.component_hashes(built["sorted_components"]())
        result["pass"] = True
        bc.write_json(args.out, result)
        print(f"DONE {args.case} T={args.tokens}", flush=True)
    except Exception:
        result["error"] = traceback.format_exc()
        bc.write_json(args.out, result)
        raise


if __name__ == "__main__":
    main()
