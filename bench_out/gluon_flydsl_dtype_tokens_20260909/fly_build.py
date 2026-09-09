"""Re-export the preserved captured FlyDSL builders used by this sweep."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "gluon_flydsl_small_tokens_20260908/fly_build.py"
SPEC = importlib.util.spec_from_file_location("_dtype_tokens_fly_build", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FlyCase = MODULE.FlyCase
fingerprint_ir = MODULE.fingerprint_ir
routing_from_topk = MODULE.routing_from_topk
unshuffle_mxfp4_scales = MODULE.unshuffle_mxfp4_scales


def build_fly(
    kind,
    x,
    w,
    topk_ids,
    *,
    x_scales=None,
    w_scales=None,
    routing=None,
    tile_m=128,
    tile_n=None,
    tile_k=None,
    waves_per_eu=None,
    b_nt=2,
    use_nt=False,
    gate_mode=None,
    use_async_copy=True,
    xcd_swizzle=0,
    k_wave=1,
):
    """Add captured A8W4 stage-1 support to the preserved builder."""
    if kind != "a8w4":
        return MODULE.build_fly(
            kind,
            x,
            w,
            topk_ids,
            x_scales=x_scales,
            w_scales=w_scales,
            routing=routing,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            use_nt=use_nt,
            gate_mode=gate_mode,
            use_async_copy=use_async_copy,
            xcd_swizzle=xcd_swizzle,
            k_wave=k_wave,
        )

    import os

    import torch

    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl import moe_kernels as mk
    from aiter.ops.quant import mxfp4_moe_sort_fwd, mxfp4_moe_sort_hip
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    assert os.environ.get("AITER_FLYDSL_NO_ACT", "0") == "0"
    tile_n = 128 if tile_n is None else tile_n
    tile_k = 256 if tile_k is None else tile_k
    waves_per_eu = 4 if waves_per_eu is None else waves_per_eu
    gate_mode = "interleave" if gate_mode is None else gate_mode
    t, k = x.shape
    e, n, packed_k = w.shape
    assert packed_k * 2 == k and n % 2 == 0
    topk = topk_ids.shape[1]
    intermediate = n // 2
    assert topk_ids.shape[0] == t
    assert x.device == w.device == topk_ids.device
    assert x.dtype == torch.float8_e4m3fn
    assert w.dtype in (torch.uint8, torch.float4_e2m1fn_x2)
    assert x_scales is not None and w_scales is not None
    assert tuple(x_scales.shape) == (t, k // 32)
    assert tuple(w_scales.shape) == (e, n, k // 32)
    assert k % (tile_k * k_wave) == 0 and intermediate % tile_n == 0
    keepalive = [x, w, x_scales, w_scales, topk_ids]

    if routing is None:
        topk_weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=x.device)
        sti, sorted_weights, sei, nvi, moe_buf = moe_sorting(
            topk_ids,
            topk_weights,
            e,
            k,
            torch.bfloat16,
            block_size=tile_m,
            accumulate=False,
        )
        keepalive += [topk_weights, sorted_weights, moe_buf]
    else:
        sti = routing["sorted_token_ids"]
        sei = routing["sorted_expert_ids"]
        nvi = routing["num_valid_ids"]
    keepalive += [sti, sei, nvi]

    x = x.contiguous()
    w = w.contiguous().view(torch.uint8)
    xs = x_scales.contiguous().view(torch.float8_e8m0fnu)
    ws = (
        w_scales.contiguous()
        .view(torch.float8_e8m0fnu)
        .reshape(e * n, k // 32)
    )
    if k <= 16384:
        sorted_scales = mxfp4_moe_sort_fwd(xs, sti, nvi, t, k)
    else:
        from aiter.utility.fp4_utils import moe_mxfp4_sort

        sorted_scales = moe_mxfp4_sort(xs, sti, nvi, t, block_size=tile_m)
    decoded_scales = unshuffle_mxfp4_scales(sorted_scales, sti.numel(), k // 32)
    token_indices = sti.long() & 0xFFFFFF
    valid_scales = (token_indices < t) & (
        torch.arange(sti.numel(), device=x.device) < nvi[0]
    )
    assert torch.equal(
        decoded_scales[valid_scales], xs.view(torch.uint8)[token_indices[valid_scales]]
    ), "prepared A scale layout mismatch"

    if gate_mode == "interleave":
        wp = shuffle_weight_a16w4(w, 16, True)
        wsp = shuffle_scale_a16w4(ws, e, True)
    elif gate_mode == "separated":
        wp = shuffle_weight(w, (16, 16))
        wsp = e8m0_shuffle(ws)
    else:
        raise ValueError(gate_mode)

    out = torch.empty(
        (t, topk, intermediate), dtype=torch.bfloat16, device=x.device
    )
    captured = []
    real_run = mk._run_compiled

    def capture(exe, args):
        captured.append((exe, args))
        return real_run(exe, args)

    mk._run_compiled = capture
    try:
        mk.flydsl_moe_stage1(
            a=x,
            w1=wp,
            sorted_token_ids=sti,
            sorted_expert_ids=sei,
            num_valid_ids=nvi,
            out=out,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            act="silu",
            a1_scale=sorted_scales,
            w1_scale=wsp,
            sorted_weights=None,
            persist_m=1,
            use_async_copy=use_async_copy,
            k_batch=1,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_mode=gate_mode,
            xcd_swizzle=xcd_swizzle,
            k_wave=k_wave,
            swiglu_limit=float("inf"),
        )
    finally:
        mk._run_compiled = real_run
    assert len(captured) == 1, f"expected one GEMM1 launch, got {len(captured)}"
    exe, args = captured[0]
    assert getattr(exe, "_cf", None) is not None
    compiled = exe._cf

    def call():
        compiled(*args)

    def sort_scales():
        if k <= 16384:
            mxfp4_moe_sort_hip(sorted_scales, xs, sti, nvi, t, k)
        else:
            from aiter.utility.fp4_utils import moe_mxfp4_sort

            sorted_scales.copy_(
                moe_mxfp4_sort(xs, sti, nvi, t, block_size=tile_m)
            )

    keepalive += [x, w, xs, ws, sorted_scales, wp, wsp, out]
    config = {
        "kind": kind,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_per_eu": waves_per_eu,
        "b_nt": b_nt,
        "gate_mode": gate_mode,
        "use_async_copy": use_async_copy,
        "xcd_swizzle": xcd_swizzle,
        "k_wave": k_wave,
        "act": "silu",
        "alpha": 1.0,
        "clamp": None,
        "bias": False,
        "routing_weights": False,
        "output": "bf16",
        "output_layout": "token_slot",
        "shape": {"T": t, "K": k, "N": n, "E": e, "topk": topk},
        "preprocessing": "setup only; retained compiled callable and prepared arguments",
        "allocated_routing_blocks": sei.numel(),
        "active_routing_blocks": int(nvi[0].item()) // tile_m,
    }
    return FlyCase(
        call,
        out,
        kind,
        "token_slot",
        sti,
        sei,
        nvi,
        topk,
        t,
        exe,
        args,
        keepalive,
        config,
        sort_scales,
        None,
    )

__all__ = [
    "FlyCase",
    "build_fly",
    "fingerprint_ir",
    "routing_from_topk",
    "unshuffle_mxfp4_scales",
]
