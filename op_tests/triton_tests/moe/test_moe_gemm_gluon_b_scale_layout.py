# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the CDNA4_SCALE weight-scale entry validation."""

from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe


def _tile(n=256, k=512, experts=2):
    raw = torch.randint(0, 255, (experts, k // 32, n), dtype=torch.uint8, device="cpu")
    return shuffle_scale_moe(raw, arch="gfx950")


def _inputs(n=256, k=512, experts=2, m=4, topk=2):
    expt_data = SimpleNamespace(
        hist=torch.tensor([4, 4], dtype=torch.int32, device="cpu"),
        token_offs_raw=torch.tensor([0, 4, 8], dtype=torch.int32, device="cpu"),
        token_offs_pad=torch.tensor([0, 1, 2], dtype=torch.int32, device="cpu"),
        block_pid_map=torch.tensor([0, 1], dtype=torch.int32, device="cpu"),
    )
    routing_data = SimpleNamespace(
        block_m=128,
        n_expts_tot=experts,
        n_expts_act=topk,
        expt_data=expt_data,
        n_blocks=lambda *_: experts,
    )
    return SimpleNamespace(
        n=n,
        k=k,
        routing_data=routing_data,
        x=torch.empty(m, k, dtype=torch.float8_e4m3fn, device="cpu"),
        w=torch.empty(
            experts, n, k // 2, dtype=torch.uint8, device="cpu"
        ).transpose(1, 2),
        xs=torch.empty(m, k // 32, dtype=torch.uint8, device="cpu"),
        y=torch.empty(1, m * topk, n // 2, dtype=torch.bfloat16, device="cpu"),
        gather=torch.arange(m * topk, dtype=torch.int32, device="cpu"),
    )


def _supported(case, ws, swizzle):
    return host.gluon_supported(
        x=case.x,
        w=case.w,
        x_scales=case.xs,
        w_scales=ws,
        y=case.y,
        bias=None,
        routing_data=case.routing_data,
        swizzle_mx_scale=swizzle,
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=None,
        N=case.n,
        K=case.k,
        apply_swiglu=True,
    )


def _launch(case, ws, swizzle):
    return host.moe_gemm_gluon(
        case.y,
        case.x,
        case.w,
        case.xs,
        ws,
        None,
        None,
        case.routing_data,
        case.gather,
        None,
        case.n,
        case.k,
        True,
        1.0,
        None,
        False,
        swizzle_mx_scale=swizzle,
    )


def test_cdna4_scale_view_is_accepted():
    assert host._cdna4_scale_error(_tile(), 256, 512) is None


def test_e8m0_dtype_is_accepted():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch has no float8_e8m0fnu")
    tile = _tile().view(torch.float8_e8m0fnu)
    assert host._cdna4_scale_error(tile, 256, 512) is None


def test_other_dtype_is_refused():
    tile = _tile().to(torch.float32)
    assert "dtype" in host._cdna4_scale_error(tile, 256, 512)


def test_contiguous_copy_of_the_view_is_refused():
    tile = shuffle_scale_moe(torch.zeros(2, 16, 256, dtype=torch.uint8), arch="gfx950")
    assert "contiguous" in host._cdna4_scale_error(tile.contiguous(), 256, 512)


@pytest.mark.parametrize(
    "n,k,rule", [(256 + 16, 512, "N % 32"), (256, 512 + 128, "K % 256")]
)
def test_cdna4_needs_n32_and_k256(n, k, rule):
    error = host._cdna4_scale_error(_tile(), n, k)
    assert rule in error


def test_raw_tensor_declared_as_cdna4_is_refused():
    raw = torch.zeros(2, 512 // 32, 256, dtype=torch.uint8, device="cpu")
    assert "shape" in host._cdna4_scale_error(raw, 256, 512)


def test_non_3d_tensor_is_refused():
    assert "3" in host._cdna4_scale_error(_tile()[0], 256, 512)


def test_gluon_supported_refuses_unreadable_cdna4_and_unknown_swizzle(monkeypatch):
    monkeypatch.setattr(host, "get_arch", lambda: "gfx950")
    case = _inputs()
    ok, why = _supported(case, _tile(), "GFX1250_SCALE")
    assert not ok and "GFX1250_SCALE" in why
    ok, why = _supported(case, _tile().contiguous(), "CDNA4_SCALE")
    assert not ok and "contiguous" in why
    ok, why = _supported(case, _tile(), "CDNA4_SCALE")
    assert "scale swizzle" not in why


@pytest.mark.parametrize("swizzle", ["CDNA4_SCALE", "GFX1250_SCALE"])
def test_moe_gemm_gluon_raises_on_unreadable_layout(swizzle, monkeypatch):
    def launch(*_args, **_kwargs):
        raise AssertionError("must fail before the launch")

    monkeypatch.setattr(host, "_fast_launch", launch)
    case = _inputs()
    ws = _tile().contiguous() if swizzle == "CDNA4_SCALE" else _tile()
    with pytest.raises(ValueError):
        _launch(case, ws, swizzle)
