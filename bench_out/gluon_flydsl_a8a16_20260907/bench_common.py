"""Shared inputs and routing for the cold MXFP8/BF16 MoE GEMM1 comparison."""
import hashlib
import json
from pathlib import Path

import torch

T, N, K, E, TOPK, BM = 4096, 4096, 7168, 33, 8, 128
SHAPE = dict(T=T, N=N, K=K, E=E, topk=TOPK, block_m=BM)


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def make_inputs(precision):
    from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp

    # Identical token/expert/slot assignments in both implementations. Each token
    # has eight distinct experts; all 33 experts receive 992 or 993 token rows.
    selected = torch.randperm(E, generator=torch.Generator().manual_seed(0))
    topk_cpu = selected[torch.arange(T * TOPK).reshape(T, TOPK) % E].to(torch.int32)
    order_cpu = torch.argsort(topk_cpu.flatten(), stable=True)
    hist_cpu = torch.bincount(topk_cpu.flatten().long(), minlength=E).to(torch.int32)
    raw_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), hist_cpu.cumsum(0).to(torch.int32)))
    blocks_cpu = (hist_cpu + BM - 1) // BM
    pad_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), blocks_cpu.cumsum(0).to(torch.int32)))
    route = RoutingData(BM, None, hist_cpu.cuda(), E, TOPK)
    grid_m = route.n_blocks(T * TOPK, BM)
    block_map = torch.full((grid_m,), -1, dtype=torch.int32)
    offset = 0
    for expert, count in enumerate(blocks_cpu.tolist()):
        block_map[offset:offset + count] = (torch.arange(count, dtype=torch.int32) << 16) | expert
        offset += count
    route.expt_data = ExptData(route.expt_hist, raw_cpu.cuda(), pad_cpu.cuda(), block_map.cuda())
    # Gluon gather indices enumerate the flattened [token, top-k slot] domain.
    gather = order_cpu.to(torch.uint16).cuda()
    torch.manual_seed(123)
    x = torch.randn((T, K), device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn((E, N, K), device="cuda", dtype=torch.bfloat16) * 0.1
    if precision == "a8w8":
        x, xs = downcast_to_mxfp(x, torch.float8_e4m3fn, axis=-1)
        w, ws = downcast_to_mxfp(w, torch.float8_e4m3fn, axis=-1)
    else:
        xs = ws = None
    sample_w = torch.cat((w.flatten()[:65536].view(torch.uint8), w.flatten()[-65536:].view(torch.uint8)))
    identity = {
        "shape": SHAPE, "precision": precision, "seed": 123,
        "topk_sha256": sha_tensor(topk_cpu), "histogram": hist_cpu.tolist(),
        "gather_sha256": sha_tensor(gather), "x_sha256": sha_tensor(x),
        "x_scales_sha256": None if xs is None else sha_tensor(xs),
        "w_scales_sha256": None if ws is None else sha_tensor(ws),
        "weight_first_last_65536_elements_sha256": sha_tensor(sample_w),
        "useful_m_blocks": offset, "gluon_grid_m": grid_m,
        "weight_layout": "logical E,N,K; gate then up", "activation": "silu(gate) * up",
        "output_dtype": "bfloat16", "bias": False, "gammas": False,
    }
    return dict(x=x, w=w, xs=xs, ws=ws, topk_ids=topk_cpu.cuda(), route=route,
                gather=gather, order=order_cpu.cuda(), hist_cpu=hist_cpu,
                raw_cpu=raw_cpu, identity=identity)


def build_gluon(data, config):
    from aiter.ops.triton.moe import moe_op_gemm_gluon as host

    y = torch.empty((1, T * TOPK, N // 2), device="cuda", dtype=torch.bfloat16)
    w = data["w"].transpose(-1, -2)
    ws = None if data["ws"] is None else data["ws"].transpose(-1, -2)

    def call():
        return host.moe_gemm_gluon(y, data["x"], w, data["xs"], ws, None, None,
                                  data["route"], data["gather"], None, N, K,
                                  True, 1.0, None, False, config=config,
                                  gate_up_split=True)

    kernel = call()
    assert kernel.name == "_moe_gluon_gemm1"
    return {"call": call, "output": y, "sorted_output": lambda: y[0],
            "kernel_pattern": "_moe_gluon_gemm1", "metadata": {
                "config": config, "kernel": kernel.name,
                "launch_metadata": kernel.metadata._asdict(),
                "binary_sha256": hashlib.sha256(kernel.asm["hsaco"]).hexdigest(),
                "output_layout": "dense expert-sorted rows",
            }}


def reference_sorted(data):
    from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp

    x, w = data["x"], data["w"]
    if data["xs"] is not None:
        x = upcast_from_mxfp(x, data["xs"], torch.bfloat16, axis=-1)
    # FP32 operands request FP32 accumulation and avoid BF16 intermediate rounding
    # between GEMM and the fused SiLU. Compute one expert at a time.
    torch.backends.cuda.matmul.allow_tf32 = False
    result = torch.empty((T * TOPK, N // 2), device="cuda", dtype=torch.float32)
    for expert in range(E):
        lo, hi = data["raw_cpu"][expert:expert + 2].tolist()
        xe = x[data["order"][lo:hi] // TOPK].float()
        we = w[expert]
        if data["ws"] is not None:
            we = upcast_from_mxfp(we, data["ws"][expert], torch.bfloat16, axis=-1)
        product = xe @ we.float().T
        result[lo:hi] = torch.nn.functional.silu(product[:, :N // 2]) * product[:, N // 2:]
    return result


def validate(built, data, replays=16):
    from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import assert_close

    built["call"]()
    actual = built["sorted_output"]().clone()
    ref = reference_sorted(data)
    assert torch.isfinite(actual).all()
    assert_close(ref, actual, description="matched FP32 GEMM + SiLU reference")
    error = (ref - actual.float()).abs()
    relative = error / torch.maximum(ref.square().mean().sqrt(), ref.abs())
    metrics = {"max_absolute_error": float(error.max()),
               "max_normalized_error": float(relative.max()),
               "rms_normalized_error": float(relative.square().mean().sqrt()),
               "tolerances": {"max": 0.02, "rms": 0.004}}
    flush = torch.empty(768 << 20, dtype=torch.uint8, device="cuda")
    for replay in range(replays):
        built["output"].fill_(float("nan"))
        flush.zero_()
        built["call"]()
        assert torch.equal(actual, built["sorted_output"]()), ("replay", replay)
    return {"pass": True, "reference": "FP32 GEMM and SiLU on dequantized common inputs",
            "cold_exact_replays": replays, "output_sha256": sha_tensor(actual), **metrics}
