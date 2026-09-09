"""Unified canonical inputs and validation for the cold dtype/token sweep.

The preserved A4 and A8/A16 input builders already encode the independently
audited routing, quantization, output-layout, and reference logic.  This module
sets their shape globals before use and normalizes their result contracts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TOKEN_COUNTS = (16, 64, 256, 1024, 4096)
PRECISIONS = ("a4w4", "a8w4", "a8w8", "a16w16")
T, N, K, E, TOPK, BM = 16, 4096, 7168, 33, 8, 128
SHAPE = {"T": T, "N": N, "K": K, "E": E, "topk": TOPK, "block_m": BM}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_A4 = _load(
    "_dtype_tokens_legacy_a4_common",
    REPO / "bench_out/gluon_flydsl_small_tokens_20260908/bench_common.py",
)
_DENSE = _load(
    "_dtype_tokens_legacy_dense_common",
    REPO / "bench_out/gluon_flydsl_a8a16_20260907/bench_common.py",
)


def configure(k: int = 7168, block_m: int = 128, tokens: int = 16):
    global T, K, BM
    T, K, BM = int(tokens), int(k), int(block_m)
    assert T in TOKEN_COUNTS
    assert K == 7168
    assert BM in (16, 32, 64, 128)
    SHAPE.update(T=T, N=N, K=K, E=E, topk=TOPK, block_m=BM)
    for module in (_A4, _DENSE):
        module.T, module.N, module.K = T, N, K
        module.E, module.TOPK, module.BM = E, TOPK, BM
        module.SHAPE.clear()
        module.SHAPE.update(SHAPE)


def sha_tensor(tensor):
    """Hash every logical byte with bounded host temporary storage."""
    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), 64 << 20):
        digest.update(memoryview(flat[offset : offset + (64 << 20)].cpu().numpy()))
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def make_inputs(precision: str):
    assert precision in PRECISIONS
    if precision == "a8w4":
        data = _make_a8w4_inputs()
    else:
        module = _A4 if precision == "a4w4" else _DENSE
        data = module.make_inputs(precision)
    data["identity"] = dict(data["identity"])
    data["identity"]["shape"] = {
        key: value for key, value in SHAPE.items() if key != "block_m"
    }
    if precision in ("a8w8", "a16w16"):
        # The preserved dense builder hashes only small edge samples.  Hash the
        # complete payload so separately prepared Gluon/FlyDSL processes prove
        # they benchmark the same weights.
        data["identity"]["full_weight_payload_sha256"] = sha_tensor(data["w"])
    if "routing_metadata" not in data:
        useful_blocks = int(data["identity"]["useful_m_blocks"])
        grid_m = int(data["identity"]["gluon_grid_m"])
        data["routing_metadata"] = {
            "block_m": BM,
            "useful_m_blocks": useful_blocks,
            "gluon_grid_m": grid_m,
            "useful_rows": T * TOPK,
            "padded_rows": useful_blocks * BM,
            "allocated_rows": grid_m * BM,
        }
    return data


def _make_a8w4_inputs():
    """Build matched MXFP8 activations and MXFP4 weights for both backends."""
    from aiter.ops.triton._triton_kernels.moe.activations import gate_up_split_perm
    from aiter.ops.triton.moe.moe_routing.routing import (
        _USE_HERD,
        ExptData,
        RoutingData,
    )
    from op_tests.op_benchmarks.triton.bench_moe_gemm_gluon import _build

    assert not _USE_HERD, "Disable HERD before importing the routing module"
    native, gather, _scatter, x, xs, w, ws, _bias, _gammas = _build(
        T,
        N,
        K,
        E,
        TOPK,
        "cuda",
        torch.float8_e4m3fn,
        torch.uint8,
        n_active=E,
    )
    topk_cpu, order_cpu, hist_cpu, raw_cpu, pad_cpu, block_map = (
        _A4._routing_arrays(native, gather)
    )
    device = x.device
    hist = hist_cpu.to(device)
    route = RoutingData(BM, None, hist, E, TOPK)
    route.expt_data = ExptData(
        hist, raw_cpu.to(device), pad_cpu.to(device), block_map.to(device)
    )

    # Preserve K-contiguous storage while changing the logical N axis from the
    # public interleaved gate/up layout to [gate | up]. Each backend performs its
    # own native weight/scale transform once during setup.
    perm = gate_up_split_perm(N).to(device)

    def repack(value):
        out = torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device=device
        )
        out.copy_(value[..., perm])
        return out

    w, ws = repack(w), repack(ws)
    assert w.stride(-2) == ws.stride(-2) == 1
    w, ws = w.transpose(-1, -2), ws.transpose(-1, -2)
    identity = {
        "shape": {key: value for key, value in SHAPE.items() if key != "block_m"},
        "precision": "a8w4",
        "seed": 0,
        "expert_selection_seed": 0,
        "builder": "bench_moe_gemm_gluon._build(n_active=33)",
        "input_distribution": "standard-normal BF16 staging tensors",
        "topk_sha256": sha_tensor(topk_cpu),
        "histogram": hist_cpu.tolist(),
        "raw_offsets_sha256": sha_tensor(raw_cpu),
        "gather_sha256": sha_tensor(gather),
        "x_sha256": sha_tensor(x),
        "x_scales_sha256": sha_tensor(xs),
        "w_scales_sha256": sha_tensor(ws),
        "full_weight_payload_sha256": sha_tensor(w),
        "input_dtype": str(x.dtype),
        "weight_dtype": str(w.dtype),
        "scale_dtype": str(xs.dtype),
        "input_quantization": "group32 MXFP8 A and MXFP4 W with E8M0 scales",
        "unit_scales": False,
        "weight_layout": "logical E,N,K/2; gate then up; canonical N-axis permutation",
        "activation": "silu(gate) * up; alpha=1; no clamp; no residual",
        "output_dtype": "bfloat16",
        "bias": False,
        "gammas": False,
    }
    routing_metadata = {
        "block_m": BM,
        "native_block_m": native.block_m,
        "useful_m_blocks": int(pad_cpu[-1]),
        "gluon_grid_m": block_map.numel(),
        "useful_rows": T * TOPK,
        "padded_rows": int(pad_cpu[-1]) * BM,
        "allocated_rows": block_map.numel() * BM,
        "block_pid_map_sha256": sha_tensor(block_map),
        "padded_offsets_sha256": sha_tensor(pad_cpu),
    }
    return {
        "x": x,
        "w": w,
        "xs": xs,
        "ws": ws,
        "topk_ids": topk_cpu.to(device),
        "route": route,
        "gather": gather,
        "order": order_cpu.to(device),
        "hist_cpu": hist_cpu,
        "raw_cpu": raw_cpu,
        "identity": identity,
        "routing_metadata": routing_metadata,
    }


decode_mxfp4 = _A4.decode_mxfp4


def component_hashes(components):
    names = ("payload", "scales") if len(components) == 2 else ("output",)
    hashes = {name: sha_tensor(value) for name, value in zip(names, components)}
    combined = hashlib.sha256()
    for value in components:
        flat = value.detach().contiguous().view(torch.uint8).reshape(-1)
        for offset in range(0, flat.numel(), 64 << 20):
            combined.update(
                memoryview(flat[offset : offset + (64 << 20)].cpu().numpy())
            )
    return {"components": hashes, "combined_sha256": combined.hexdigest()}


def validate(built, data, replays: int = 16):
    precision = data["identity"]["precision"]
    result = (
        _A4.validate(built, data, replays=replays)
        if precision in ("a4w4", "a8w4")
        else _DENSE.validate(built, data, replays=replays)
    )
    hashes = component_hashes(built["sorted_components"]())
    result["output_hashes"] = hashes
    result["output_sha256"] = hashes["combined_sha256"]
    return result


def smoke_validate(built):
    """One prepared launch and a cheap output sanity/hash check."""
    built["call"]()
    torch.cuda.synchronize()
    components = tuple(built["sorted_components"]())
    assert components
    for component in components:
        if component.is_floating_point():
            assert bool(torch.isfinite(component).all())
    return {
        "pass": True,
        "prepared_launches": 1,
        "output_hashes": component_hashes(components),
    }
