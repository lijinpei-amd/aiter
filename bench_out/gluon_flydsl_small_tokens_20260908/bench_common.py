"""Canonical seed-0 inputs and references for the T16/T64 A4 GEMM1 comparison."""
import hashlib
import json
from pathlib import Path

import torch

T, N, K, E, TOPK, BM = 16, 4096, 7168, 33, 8, 128
SHAPE = dict(T=T, N=N, K=K, E=E, topk=TOPK, block_m=BM)


def configure(k, block_m=128, tokens=16):
    global T, K, BM
    T, K, BM = int(tokens), int(k), int(block_m)
    assert T in (16, 64) and K == 7168 and BM in (32, 64, 128)
    SHAPE.update(T=T, K=K, block_m=BM)


def sha_tensor(tensor):
    """Hash every logical byte with bounded host-side temporary storage."""
    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for offset in range(0, flat.numel(), 64 << 20):
        chunk = flat[offset:offset + (64 << 20)].cpu().numpy()
        digest.update(memoryview(chunk))
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def _routing_arrays(native, gather):
    """Recover native top-k slots and reblock their expert-sorted rows on CPU."""
    assert native.n_expts_tot == E and native.n_expts_act == TOPK
    hist_cpu = native.expt_data.hist.detach().cpu().to(torch.int32)
    raw_cpu = native.expt_data.token_offs_raw.detach().cpu().to(torch.int32)
    order_cpu = gather.detach().cpu().to(torch.int64)
    assert hist_cpu.shape == (E,) and raw_cpu.shape == (E + 1,)
    assert order_cpu.shape == (T * TOPK,)
    assert torch.equal(torch.sort(order_cpu).values, torch.arange(T * TOPK))
    assert torch.equal(raw_cpu, torch.cat((torch.zeros(1, dtype=torch.int32),
                                          hist_cpu.cumsum(0).to(torch.int32))))
    assert int(hist_cpu.sum()) == T * TOPK and bool((hist_cpu > 0).all())
    assert int(hist_cpu.max()) - int(hist_cpu.min()) <= 1

    # Native top-k may order tied scores differently from the score builder's
    # round-robin slots. Recover those exact slots from the gather permutation.
    topk_flat = torch.full((T * TOPK,), -1, dtype=torch.int32)
    for expert in range(E):
        lo, hi = raw_cpu[expert:expert + 2].tolist()
        topk_flat[order_cpu[lo:hi]] = expert
    topk_cpu = topk_flat.reshape(T, TOPK)
    selected = torch.randperm(E, generator=torch.Generator().manual_seed(0))
    balanced = selected[torch.arange(T * TOPK).reshape(T, TOPK) % E].to(torch.int32)
    assert torch.equal(torch.sort(topk_cpu, dim=1).values,
                       torch.sort(balanced, dim=1).values)
    blocks_cpu = (hist_cpu + BM - 1) // BM
    pad_cpu = torch.cat((torch.zeros(1, dtype=torch.int32), blocks_cpu.cumsum(0).to(torch.int32)))
    grid_m = native.n_blocks(T * TOPK, BM)
    block_map = torch.full((grid_m,), -1, dtype=torch.int32)
    offset = 0
    for expert, count in enumerate(blocks_cpu.tolist()):
        block_map[offset:offset + count] = (torch.arange(count, dtype=torch.int32) << 16) | expert
        offset += count
    assert offset == int(pad_cpu[-1]) and offset <= grid_m
    return topk_cpu, order_cpu, hist_cpu, raw_cpu, pad_cpu, block_map


def make_inputs(precision):
    from aiter.ops.triton._triton_kernels.moe.activations import gate_up_split_perm
    from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData, _USE_HERD
    from op_tests.op_benchmarks.triton.bench_moe_gemm_gluon import _build

    assert precision == "a4w4", "This comparison uses canonical A4W4/MXFP4 output"
    assert not _USE_HERD, "Disable HERD before importing the routing module"
    # Preserve the archived cold benchmark's seed, standard-normal BF16 staging
    # tensors, E,K,N random-weight draw order, and MXFP4 quantization exactly.
    native, gather, _scatter, x, xs, w, ws, _bias, _gammas = _build(
        T, N, K, E, TOPK, "cuda", torch.uint8, torch.uint8, n_active=E)
    topk_cpu, order_cpu, hist_cpu, raw_cpu, pad_cpu, block_map = _routing_arrays(native, gather)
    device = x.device
    hist = hist_cpu.to(device)
    route = RoutingData(BM, None, hist, E, TOPK)
    route.expt_data = ExptData(hist, raw_cpu.to(device), pad_cpu.to(device), block_map.to(device))

    # Identical to cold_bench._repack_weights with GU_SPLIT=1: preserve the
    # K-contiguous strides while permuting N from interleaved to [gate | up].
    perm = gate_up_split_perm(N).to(device)

    def repack(value):
        out = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
        out.copy_(value[..., perm])
        return out

    w, ws = repack(w), repack(ws)
    assert w.stride(-2) == ws.stride(-2) == 1
    # Both backend adapters take logical E,N,K/pack tensors. These views retain
    # exactly the same payloads; the Gluon adapter transposes them back.
    w, ws = w.transpose(-1, -2), ws.transpose(-1, -2)
    identity = {
        "shape": {key: value for key, value in SHAPE.items() if key != "block_m"},
        "precision": precision, "seed": 0, "expert_selection_seed": 0,
        "builder": "bench_moe_gemm_gluon._build(n_active=33)",
        "input_distribution": "standard-normal BF16 staging tensors; no extra scaling",
        "topk_sha256": sha_tensor(topk_cpu), "histogram": hist_cpu.tolist(),
        "raw_offsets_sha256": sha_tensor(raw_cpu),
        "gather_sha256": sha_tensor(gather), "x_sha256": sha_tensor(x),
        "x_scales_sha256": None if xs is None else sha_tensor(xs),
        "w_scales_sha256": None if ws is None else sha_tensor(ws),
        "full_weight_payload_sha256": sha_tensor(w),
        "input_dtype": str(x.dtype), "scale_dtype": None if xs is None else str(xs.dtype),
        "input_quantization": None if xs is None else "group32 E8M0 ceil scale; shared quantized tensors",
        "unit_scales": False if xs is not None else None,
        "weight_layout": "logical E,N,K/2; gate then up; canonical N-axis permutation",
        "activation": "silu(gate) * up; alpha=1; no clamp; no residual",
        "output_dtype": "mxfp4_e2m1_e8m0" if precision == "a4w4" else "bfloat16",
        "bias": False, "gammas": False,
    }
    routing_metadata = {
        "block_m": BM, "native_block_m": native.block_m,
        "useful_m_blocks": int(pad_cpu[-1]), "gluon_grid_m": block_map.numel(),
        "useful_rows": T * TOPK, "padded_rows": int(pad_cpu[-1]) * BM,
        "allocated_rows": block_map.numel() * BM,
        "block_pid_map_sha256": sha_tensor(block_map),
        "padded_offsets_sha256": sha_tensor(pad_cpu),
    }
    return dict(x=x, w=w, xs=xs, ws=ws, topk_ids=topk_cpu.to(device), route=route,
                gather=gather, order=order_cpu.to(device), hist_cpu=hist_cpu,
                raw_cpu=raw_cpu, identity=identity, routing_metadata=routing_metadata)


def decode_mxfp4(payload, scales):
    assert payload.dtype == scales.dtype == torch.uint8
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         device=payload.device, dtype=torch.float32)
    values = table[codes.long()]
    assert values.shape[-1] == scales.shape[-1] * 32
    return values * e8m0_to_float(scales).repeat_interleave(32, dim=-1)


def e8m0_to_float(scales):
    bits = scales.to(torch.int32) << 23
    bits = torch.where(scales == 0, 0x00400000, bits)
    bits = torch.where(scales == 255, 0x7F800001, bits)
    return bits.view(torch.float32)


def reference_sorted(data):
    from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp

    x, w = data["x"], data["w"]
    if data["xs"] is not None:
        x = upcast_from_mxfp(x, data["xs"], torch.bfloat16, axis=-1)
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
    assert torch.isfinite(result).all()
    return result


def error_metrics(reference, actual):
    absolute = (reference - actual.float()).abs()
    denom = torch.maximum(reference.square().mean().sqrt(), reference.abs()).clamp_min(1e-30)
    normalized = absolute / denom
    return {
        "max_absolute_error": float(absolute.max()),
        "max_normalized_error": float(normalized.max()),
        "rms_normalized_error": float(normalized.square().mean().sqrt()),
    }


def even_e8m0(amax):
    """Independent torch expression of the fused output's documented EVEN scale."""
    biased = (((amax.contiguous().view(torch.int32) + 0x200000) >> 23) & 255).clamp_min(2)
    return (biased - 2).to(torch.uint8)


def output_scales(amax, policy):
    if policy == "even":
        return even_e8m0(amax)
    if policy == "round_up":
        bits = (amax * (1.0 / 6.0)).contiguous().view(torch.int32)
        return (((bits + 0x7FFFFF) >> 23) & 255).clamp_max(254).to(torch.uint8)
    raise ValueError(f"Unknown output quantization policy: {policy}")


def quantize_given_scales(reference, scales):
    """Independent torch nearest-even E2M1 quantization for reference checks."""
    scaled = reference / e8m0_to_float(scales).repeat_interleave(32, dim=-1)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=reference.device, dtype=torch.float32)
    magnitude = scaled.abs().contiguous()
    # bucketize chooses the lower code at a midpoint; only odd lower codes
    # advance to the upper even code for the IEEE ties-to-even rule.
    code = torch.bucketize(magnitude, boundaries, right=False)
    tie = (code < 7) & (magnitude == boundaries[code.clamp_max(6)])
    code = code + (tie & ((code & 1) == 1)).to(code.dtype)
    code = (code | (torch.signbit(scaled).to(code.dtype) << 3)).to(torch.uint8)
    return code[..., ::2] | (code[..., 1::2] << 4)


def validate_mxfp4(reference, components, policy="even"):
    """Check quantized output with an FP32 arithmetic allowance before rounding.

    FP4's unavoidable rounding error is not charged against the BF16 arithmetic
    threshold. The actual code must be a nearest code for a nearby FP32 value,
    and its E8M0 exponent must be attainable within the same numerical allowance.
    Quantization-reference byte agreement is reported separately from this bound.
    """
    payload, scales = components
    actual = decode_mxfp4(payload, scales)
    assert torch.isfinite(actual).all()
    reference_scales = output_scales(reference.abs().reshape(T * TOPK, -1, 32).amax(-1), policy)
    reference_payload = quantize_given_scales(reference, reference_scales)
    max_tol, rms_tol = 0.02, 0.004
    denom = torch.maximum(reference.square().mean().sqrt(), reference.abs()).clamp_min(1e-30)
    allowance = max_tol * denom
    lower_amax = (reference.abs() - allowance).clamp_min(0).reshape(T * TOPK, -1, 32).amax(-1)
    upper_amax = (reference.abs() + allowance).reshape(T * TOPK, -1, 32).amax(-1)
    lower_scale, upper_scale = output_scales(lower_amax, policy), output_scales(upper_amax, policy)
    invalid_scales = (scales < lower_scale) | (scales > upper_scale) | (scales == 255)
    assert not bool(invalid_scales.any()), f"{int(invalid_scales.sum())} E8M0 scales outside reference allowance"
    nearest = decode_mxfp4(quantize_given_scales(reference, scales), scales)
    excess = ((actual - reference).abs() - (nearest - reference).abs()).clamp_min(0) / denom
    max_excess = float(excess.max())
    rms_excess = float(excess.square().mean().sqrt())
    assert max_excess <= max_tol, ("MXFP4 excess quantization error", max_excess, max_tol)
    assert rms_excess <= rms_tol, ("MXFP4 RMS excess quantization error", rms_excess, rms_tol)
    payload_equal = int((payload == reference_payload).sum())
    scales_equal = int((scales == reference_scales).sum())
    return {
        "reference": f"independent FP32 GEMM and SiLU, torch {policy.upper()} E8M0 + round-to-even E2M1",
        "output_quantization_policy": policy,
        "tolerances": {"max_normalized_prequant_allowance": max_tol, "rms_normalized_excess": rms_tol},
        "max_normalized_excess_error": max_excess,
        "rms_normalized_excess_error": rms_excess,
        "invalid_scale_count": int(invalid_scales.sum()),
        "dequantized_error": error_metrics(reference, actual),
        "reference_payload_equal_bytes": payload_equal,
        "reference_payload_bytes": payload.numel(),
        "reference_payload_agreement": payload_equal / payload.numel(),
        "reference_scale_equal_bytes": scales_equal,
        "reference_scale_bytes": scales.numel(),
        "reference_scale_agreement": scales_equal / scales.numel(),
        "quantized_reference_payload_sha256": sha_tensor(reference_payload),
        "quantized_reference_scales_sha256": sha_tensor(reference_scales),
    }


def component_hashes(components):
    names = ("payload", "scales") if len(components) == 2 else ("output",)
    hashes = {name: sha_tensor(value) for name, value in zip(names, components)}
    combined = hashlib.sha256()
    for value in components:
        flat = value.detach().contiguous().view(torch.uint8).reshape(-1)
        for offset in range(0, flat.numel(), 64 << 20):
            combined.update(memoryview(flat[offset:offset + (64 << 20)].cpu().numpy()))
    return {"components": hashes, "combined_sha256": combined.hexdigest()}


def validate(built, data, replays=16):
    from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import assert_close

    built["call"]()
    first = tuple(t.clone() for t in built["sorted_components"]())
    reference = reference_sorted(data)
    if data["identity"]["precision"] == "a4w4":
        metrics = validate_mxfp4(reference, first, built["metadata"]["output_quantization_policy"])
    else:
        actual = first[0]
        assert torch.isfinite(actual).all()
        assert_close(reference, actual, description="matched FP32 GEMM + SiLU reference")
        metrics = {"reference": "FP32 GEMM and SiLU on dequantized common inputs",
                   "tolerances": {"max": 0.02, "rms": 0.004},
                   **error_metrics(reference, actual)}
    flush = torch.empty(768 << 20, dtype=torch.uint8, device="cuda")
    for replay in range(replays):
        for index, output in enumerate(built["outputs"]):
            if output.dtype == torch.uint8:
                output.fill_((0xA5 if index == 0 else 0x5A) ^ replay)
            else:
                output.fill_(float("nan"))
        flush.zero_()
        built["call"]()
        current = built["sorted_components"]()
        assert len(current) == len(first)
        assert all(torch.equal(expected, actual) for expected, actual in zip(first, current)), ("cold replay", replay)
    hashes = component_hashes(first)
    return {"pass": True, "cold_exact_replays": replays,
            "poisoned_output_components": len(built["outputs"]),
            "output_sha256": hashes["combined_sha256"], "output_hashes": hashes,
            **metrics}
