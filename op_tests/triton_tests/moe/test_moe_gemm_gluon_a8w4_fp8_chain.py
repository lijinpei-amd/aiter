# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The a8w4 GEMM1 -> GEMM2 hand-off without a bf16 round trip.

GEMM1 quantizes its swiglu output to MXFP8 in the epilogue
(``moe_gemm1_a8w4_mxfp8_out``), optionally writing the E8M0 scales in the
sorted-shuffled layout GEMM2 reads (``scale_layout="sorted_shuffled"``), and GEMM2
takes that buffer as is (``moe_gemm_a8w4(..., x_scales_sorted_shuffled=True)``).
"""

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.moe_model_recipes import ACT_RECIPES
from op_tests.triton_tests.moe.test_moe_gemm_a4w4 import (
    init_compute_data,
    init_routing_data,
)

# (E, topk) = (33, 8) makes the routing pick BLOCK_M 128, the block the GEMM1 split
# layout and the GEMM2 half-height tile need (see test_moe_gemm_a4w4).
SHAPES = [(512, 4096, 4096, 33, 8), (1024, 6144, 7168, 33, 8)]


def _skip_unless_gfx950():
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")


def _gemm1_inputs(m, n, k, n_expts_tot, n_expts_act, device):
    torch.manual_seed(0)
    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather=True, do_scatter=True, device=device
    )
    x, w, bias, _ = init_compute_data(
        m, n, k, gindx, None, n_expts_tot, n_expts_act,
        torch.bfloat16, torch.bfloat16, False, device=device,
    )
    w, w_scale = downcast_to_mxfp(w, torch.uint8, axis=1)
    x, x_scale = downcast_to_mxfp(x, torch.float8_e4m3fn, axis=-1)
    return rdata, gindx, sindx, x, x_scale, w, w_scale, bias


def _split(w, w_scale, bias):
    from aiter.ops.triton._triton_kernels.moe.activations import gate_up_split_perm

    perm = gate_up_split_perm(w.shape[-1]).to(w.device)

    def n_perm(t):
        # Keep the K-contiguous ((E, K/pack, N), stride(-2) == 1) layout.
        return t[..., perm].transpose(-1, -2).contiguous().transpose(-1, -2)

    return n_perm(w), n_perm(w_scale), None if bias is None else bias[..., perm].contiguous()


def _shuffle_reference(raw, rdata):
    """Host form of moe_sort_scales.cuh's layout: raw (sorted, unpadded) scale rows
    moved to their padded routing position, then one 256 B run per 32-row stripe and
    K256. Returns (bytes, mask of the bytes a routed row owns)."""
    bm = int(rdata.block_m)
    ed = rdata.expt_data
    offs_raw = ed.token_offs_raw.long()
    pad = ed.token_offs_pad
    if isinstance(pad, dict):
        pad = pad[bm] if bm in pad else next(iter(pad.values()))
    pad = pad.long()
    m, cols = raw.shape
    rows = torch.arange(m, device=raw.device)
    e = torch.searchsorted(offs_raw[1:].contiguous(), rows, right=True)
    dense = torch.zeros(
        (rdata.n_blocks(m, bm) * bm, cols), dtype=torch.uint8, device=raw.device
    )
    dense[pad[e] * bm + (rows - offs_raw[e])] = raw
    live = torch.zeros_like(dense, dtype=torch.bool)
    live[pad[e] * bm + (rows - offs_raw[e])] = True

    def shuffle(t):
        return t.view(-1, 2, 16, cols // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).reshape(-1)

    return shuffle(dense), shuffle(live)


def _fused(args, act, **kw):
    from aiter.ops.triton.moe.moe_op_gemm_gluon import moe_gemm1_a8w4_mxfp8_out

    return moe_gemm1_a8w4_mxfp8_out(
        *args,
        alpha=act.alpha,
        limit=act.limit,
        swiglu_add_residual=act.add_residual,
        **kw,
    )


@pytest.mark.parametrize("m, n, k, n_expts_tot, n_expts_act", SHAPES)
@pytest.mark.parametrize("act", ACT_RECIPES, ids=[a.name for a in ACT_RECIPES])
def test_gemm1_fused_mxfp8_out(m, n, k, n_expts_tot, n_expts_act, act, device="cuda"):
    """The fused MXFP8 output is the bf16 output quantized like dynamic_mxfp8_quant,
    and the gate/up split layout gives the same bytes as the interleaved one."""
    _skip_unless_gfx950()
    rdata, gindx, _, x, xs, w, ws, bias = _gemm1_inputs(
        m, n, k, n_expts_tot, n_expts_act, device
    )
    kw = dict(alpha=act.alpha, limit=act.limit, swiglu_add_residual=act.add_residual)
    y_bf16 = moe_gemm_a8w4(
        x, w, xs, ws, None, None, bias, rdata, gather_indx=gindx,
        out_dtype=torch.bfloat16, apply_swiglu=True, **kw,
    ).reshape(gindx.shape[0], -1)
    y8, s8 = _fused((x, w, xs, ws, bias, rdata, gindx), act)
    ref_q, ref_s = downcast_to_mxfp(
        y_bf16, torch.float8_e4m3fn, axis=-1, pow2_scale=True
    )
    assert y8.shape == ref_q.shape and s8.shape == ref_s.shape
    # The fused path quantizes the fp32 accumulator, the reference its bf16 copy. A
    # group whose amax sits next to the power-of-two rounding point can take the next
    # scale: with swigluoai the output saturates at limit * (limit + 1) = 56, which bf16
    # rounds up to 1.75 * 2**5 (scale rounds up) while fp32 keeps 55.9995 (it does not),
    # so nearly every group differs there. Both encodings are valid; bound the scale
    # difference to one step and the error below, which is what matters.
    assert (s8.int() - ref_s.int()).abs().max().item() <= 1

    def deq(q, s):
        return q.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim=-1)

    ref = y_bf16.float()
    err_fused = (deq(y8, s8) - ref).abs().max()
    err_ref = (deq(ref_q, ref_s) - ref).abs().max()
    assert err_fused <= 2 * err_ref + 1e-6, (err_fused, err_ref)

    w_s, ws_s, b_s = _split(w, ws, bias)
    y8_s, s8_s = _fused((x, w_s, xs, ws_s, b_s, rdata, gindx), act, gate_up_split=True)
    assert torch.equal(y8_s.view(torch.uint8), y8.view(torch.uint8))
    assert torch.equal(s8_s, s8)


@pytest.mark.parametrize("m, n, k, n_expts_tot, n_expts_act", SHAPES)
def test_gemm1_sorted_shuffled_scales(m, n, k, n_expts_tot, n_expts_act, device="cuda"):
    """scale_layout="sorted_shuffled" writes the raw scales' bytes in
    moe_sort_scales.cuh's layout and leaves the payload alone."""
    _skip_unless_gfx950()
    act = ACT_RECIPES[0]
    rdata, gindx, _, x, xs, w, ws, bias = _gemm1_inputs(
        m, n, k, n_expts_tot, n_expts_act, device
    )
    w, ws, bias = _split(w, ws, bias)
    args = (x, w, xs, ws, bias, rdata, gindx)
    y_raw, s_raw = _fused(args, act, gate_up_split=True)
    y_shf, s_shf = _fused(args, act, gate_up_split=True, scale_layout="sorted_shuffled")
    assert torch.equal(y_shf.view(torch.uint8), y_raw.view(torch.uint8))
    ref, live = _shuffle_reference(s_raw, rdata)
    got = s_shf.reshape(-1)
    assert got.numel() == ref.numel()
    # Only routed rows' bytes are written; the rest of the buffer is uninitialised.
    assert torch.equal(got[live], ref[live])


@pytest.mark.parametrize("m, n, k, n_expts_tot, n_expts_act", SHAPES)
def test_gemm2_reads_sorted_shuffled_scales(
    m, n, k, n_expts_tot, n_expts_act, device="cuda"
):
    """GEMM2 on GEMM1's fp8 output gives the same bf16 result whether it reads the
    raw scales or the sorted-shuffled buffer GEMM1 wrote."""
    _skip_unless_gfx950()
    act = ACT_RECIPES[0]
    rdata, gindx, sindx, x, xs, w, ws, bias = _gemm1_inputs(
        m, n, k, n_expts_tot, n_expts_act, device
    )
    w, ws, bias = _split(w, ws, bias)
    args = (x, w, xs, ws, bias, rdata, gindx)
    y8, s_raw = _fused(args, act, gate_up_split=True)
    _, s_shf = _fused(args, act, gate_up_split=True, scale_layout="sorted_shuffled")

    # GEMM2: (E, n/2 -> k), the same routing, rows already sorted.
    torch.manual_seed(1)
    k2, n2 = n // 2, k
    w2 = torch.randn((n_expts_tot, k2, n2), device=device, dtype=torch.bfloat16)
    w2, w2_scale = downcast_to_mxfp(w2, torch.uint8, axis=1)
    b2 = torch.randn((n_expts_tot, n2), device=device, dtype=torch.float32)
    gammas = 2 ** torch.randint(
        -5, 0, (gindx.shape[0],), device=device, dtype=torch.float32
    )

    def gemm2(scales, shuffled):
        return moe_gemm_a8w4(
            y8, w2, scales, w2_scale, None, None, b2, rdata,
            gather_indx=None, scatter_indx=sindx, gammas=gammas,
            out_dtype=torch.bfloat16, apply_swiglu=False,
            x_scales_sorted_shuffled=shuffled,
        )

    ref = gemm2(s_raw, False)
    got = gemm2(s_shf, True)
    assert torch.equal(got, ref), (got.float() - ref.float()).abs().max()
