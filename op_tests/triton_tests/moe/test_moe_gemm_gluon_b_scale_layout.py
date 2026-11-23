# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the CDNA4_SCALE weight-scale layout of the Gluon MoE host."""

import os
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DtypeQuant,
    ScaleSwizzle,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe


@pytest.fixture(autouse=True)
def _clean_tuning_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    host._launch_spec.cache_clear()
    host._validate_launch_config.cache_clear()
    conf._get_gluon_config_cached.cache_clear()
    yield
    host._launch_spec.cache_clear()
    host._validate_launch_config.cache_clear()
    conf._get_gluon_config_cached.cache_clear()


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
        w=torch.empty(experts, n, k // 2, dtype=torch.uint8, device="cpu").transpose(
            1, 2
        ),
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
    assert host._swizzle_error("CDNA4_SCALE", _tile(), case.n, case.k) is None


def test_swizzle_check_needs_scales_only_for_cdna4():
    assert "needs weight scales" in host._swizzle_error("CDNA4_SCALE", None, 256, 512)
    assert host._swizzle_error(None, None, 256, 512) is None


@pytest.mark.parametrize("swizzle", ["CDNA4_SCALE", "GFX1250_SCALE"])
def test_moe_gemm_gluon_raises_on_unreadable_layout(swizzle, monkeypatch):
    def launch(*_args, **_kwargs):
        raise AssertionError("must fail before the launch")

    monkeypatch.setattr(host, "_fast_launch", launch)
    case = _inputs()
    ws = _tile().contiguous() if swizzle == "CDNA4_SCALE" else _tile()
    with pytest.raises(ValueError):
        _launch(case, ws, swizzle)


# MXFP8 x MXFP8 GEMM1 launches, as in test_moe_gemm_gluon_scale_prep.py.
_N, _K, _E = 512, 2560, 2


def _mx8_inputs():
    m, topk = 4, 2
    case = _inputs(n=_N, k=_K, experts=_E, m=m, topk=topk)
    case.w = torch.empty(_E, _N, _K, dtype=torch.float8_e4m3fn).transpose(1, 2)
    case.ws = torch.empty(_E, _N, _K // 32, dtype=torch.uint8).transpose(1, 2)
    case.y = torch.empty(1, m * topk, _N // 2, dtype=torch.bfloat16)
    return case


def _mx8_tile():
    raw = torch.randint(0, 255, (_E, _K // 32, _N), dtype=torch.uint8)
    return shuffle_scale_moe(raw, arch="gfx950")


def _mx8_config(block_k, a_shuffled, b_shuffled):
    cfg = dict(
        host._launch_spec(
            128,
            _N,
            _K,
            DtypeQuant.MXFP8,
            DtypeQuant.MXFP8,
            False,
            False,
            False,
            True,
            False,
            ActivationSpec(int(ActKind.SILU), 1.0, None, False),
            None,
            None,
        )[4],
        BLOCK_K=block_k,
        VGPR_PREFETCH_K=block_k,
    )
    if block_k == 256:
        cfg.update(BLOCK_N=128, SLOT_N=64, NUM_LDS_BUFFER=2, warps_per_cta=(2, 2))
    cfg.update(A_SCALE_SORTED_SHUFFLED=a_shuffled, B_SCALE_SHUFFLED=b_shuffled)
    return cfg


def _capture(case, ws, swizzle, config, monkeypatch):
    captured = {}
    sentinel = object()

    def launch(kernel, grid, args, *metadata):
        captured.update(zip(kernel.arg_names, args))
        return sentinel

    monkeypatch.setattr(host, "_fast_launch", launch)
    result = host.moe_gemm_gluon(
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
        config=config,
        swizzle_mx_scale=swizzle,
    )
    assert result is sentinel
    return captured


def _no_launch(monkeypatch):
    def launch(*_args, **_kwargs):
        raise AssertionError("must fail before the launch")

    monkeypatch.setattr(host, "_fast_launch", launch)


def _a_sorter(available, monkeypatch):
    packed = torch.empty(_E * 128 * _K // 32, dtype=torch.uint8)
    monkeypatch.setattr(
        host, "_sorted_shuffle_a_scales", lambda *_: packed if available else None
    )
    return packed


def _assert_b_scales(args, ws, cdna4):
    tuning = conf._cval(args["CFG_TUNING"])
    assert tuning.B_SCALE_SHUFFLED is cdna4
    swizzle = ScaleSwizzle.CDNA4_SCALE if cdna4 else ScaleSwizzle.NONE
    assert conf._cval(args["B_SCALE_SWIZZLE"]) == swizzle
    if cdna4:
        tile = ws.transpose(-1, -2)
        assert args["b_scale_ptr"].data_ptr() == ws.data_ptr()
        assert args["b_scale_ptr"].shape == tile.shape
        assert args["b_scale_ptr"].stride() == tile.stride()
        assert args["B_SCALE_STRIDE_K"] == 32
    else:
        assert args["b_scale_ptr"] is ws
        assert args["B_SCALE_STRIDE_K"] == ws.stride(1)


@pytest.mark.parametrize("a_shuffled", [False, True])
@pytest.mark.parametrize("tuned_flag", [True, False])
def test_cdna4_input_is_read_zero_copy(tuned_flag, a_shuffled, monkeypatch):
    case = _mx8_inputs()
    tile = _mx8_tile()
    _a_sorter(True, monkeypatch)
    cfg = _mx8_config(256, a_shuffled, tuned_flag)
    if not tuned_flag:
        _no_launch(monkeypatch)
        with pytest.raises(ValueError, match="B_SCALE_SHUFFLED"):
            _capture(case, tile, "CDNA4_SCALE", cfg, monkeypatch)
        return
    args = _capture(case, tile, "CDNA4_SCALE", cfg, monkeypatch)
    _assert_b_scales(args, tile, cdna4=True)
    tuning = conf._cval(args["CFG_TUNING"])
    assert tuning.A_SCALE_SORTED_SHUFFLED is a_shuffled


def test_raw_input_with_shuffled_explicit_config_raises(monkeypatch):
    case = _mx8_inputs()
    _no_launch(monkeypatch)
    with pytest.raises(ValueError, match="B_SCALE_SHUFFLED"):
        _capture(case, case.ws, None, _mx8_config(256, False, True), monkeypatch)


def test_tuned_shuffled_entry_with_raw_input_reads_raw(monkeypatch):
    case = _mx8_inputs()
    entry = _mx8_config(256, False, True)
    monkeypatch.setattr(host, "_default_launch_config", lambda *_: dict(entry))
    args = _capture(case, case.ws, None, None, monkeypatch)
    _assert_b_scales(args, case.ws, cdna4=False)


def test_default_entry_with_cdna4_input_reads_the_tile(monkeypatch):
    case = _mx8_inputs()
    tile = _mx8_tile()
    entry = _mx8_config(256, False, False)
    monkeypatch.setattr(host, "_default_launch_config", lambda *_: dict(entry))
    args = _capture(case, tile, "CDNA4_SCALE", None, monkeypatch)
    _assert_b_scales(args, tile, cdna4=True)


@pytest.mark.parametrize("a_available", [False, True])
def test_k128_cdna4_needs_shuffled_a(a_available, monkeypatch):
    case = _mx8_inputs()
    tile = _mx8_tile()
    _a_sorter(a_available, monkeypatch)
    cfg = _mx8_config(128, True, True)
    if not a_available:
        _no_launch(monkeypatch)
        with pytest.raises(ValueError, match="BLOCK_K 128"):
            _capture(case, tile, "CDNA4_SCALE", cfg, monkeypatch)
        return
    args = _capture(case, tile, "CDNA4_SCALE", cfg, monkeypatch)
    _assert_b_scales(args, tile, cdna4=True)


def test_config_that_cannot_read_cdna4_is_refused(monkeypatch):
    case = _mx8_inputs()
    _no_launch(monkeypatch)
    monkeypatch.setattr(host, "_b_scale_shuffle_fits", lambda _cfg: False)
    with pytest.raises(ValueError, match="cannot read"):
        _capture(
            case, _mx8_tile(), "CDNA4_SCALE", _mx8_config(256, False, True), monkeypatch
        )


def _mx8_supported(case, ws, monkeypatch, cfg):
    monkeypatch.setattr(host, "get_arch", lambda: "gfx950")
    monkeypatch.setattr(host, "_default_launch_config", lambda *_: dict(cfg))
    return host.gluon_supported(
        x=case.x,
        w=case.w,
        x_scales=case.xs,
        w_scales=ws,
        y=case.y,
        bias=None,
        routing_data=case.routing_data,
        swizzle_mx_scale="CDNA4_SCALE",
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=None,
        N=case.n,
        K=case.k,
        apply_swiglu=True,
        gather_indx=case.gather,
    )


def test_gluon_supported_refuses_cdna4_that_the_config_cannot_read(monkeypatch):
    case = _mx8_inputs()
    monkeypatch.setattr(host, "_b_scale_shuffle_fits", lambda _cfg: False)
    ok, why = _mx8_supported(
        case, _mx8_tile(), monkeypatch, _mx8_config(256, False, True)
    )
    assert not ok and "cannot read" in why


def test_gluon_supported_refuses_k128_cdna4_without_shuffled_a(monkeypatch):
    from aiter.ops import moe_mxfp4_aux

    # No aux instance for this shape: the launch cannot sort the A scales.
    monkeypatch.setattr(moe_mxfp4_aux, "MXFP4_MOE_SUPPORTED_SHAPES", set())
    case = _mx8_inputs()
    ok, why = _mx8_supported(
        case, _mx8_tile(), monkeypatch, _mx8_config(128, True, True)
    )
    assert not ok and "BLOCK_K 128" in why


def test_try_gluon_falls_back_on_k128_cdna4_without_shuffled_a(monkeypatch):
    from aiter.ops import moe_mxfp4_aux

    monkeypatch.setattr(moe_mxfp4_aux, "MXFP4_MOE_SUPPORTED_SHAPES", set())
    _no_launch(monkeypatch)
    case = _mx8_inputs()
    monkeypatch.setattr(host, "get_arch", lambda: "gfx950")
    entry = _mx8_config(128, True, True)
    monkeypatch.setattr(host, "_default_launch_config", lambda *_: dict(entry))
    assert not host.try_gluon_grouped_gemm(
        op_name="test",
        y=case.y,
        x=case.x,
        w=case.w,
        x_scales=case.xs,
        w_scales=_mx8_tile(),
        bias=None,
        gammas=None,
        routing_data=case.routing_data,
        gather_indx=case.gather,
        scatter_indx=None,
        N=case.n,
        K=case.k,
        apply_swiglu=True,
        alpha=1.0,
        limit=None,
        swiglu_add_residual=False,
        split_k=1,
        swizzle_mx_scale="CDNA4_SCALE",
    )


def test_cdna4_scales_do_not_set_direct_b():
    case = _mx8_inputs()
    with torch.inference_mode():
        raw = torch.zeros(_E, _K // 32, _N, dtype=torch.uint8)
        tile = shuffle_scale_moe(raw, arch="gfx950")
    assert tile.is_inference()
    assert not host._reads_b_directly(case.w, _N, _K, DtypeQuant.MXFP8)


def test_scale_copy_and_env_override_are_gone(monkeypatch):
    assert not hasattr(host, "_shuffled_b_scales")
    assert not hasattr(host, "_B_SCALE_SHUFFLE_COPY")
    case = _mx8_inputs()
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SHUFFLED_W_SCALES", "1")
    args = _capture(case, case.ws, None, _mx8_config(256, False, False), monkeypatch)
    _assert_b_scales(args, case.ws, cdna4=False)


@pytest.mark.parametrize("cdna4", [False, True])
def test_k128_scales_are_both_shuffled_or_both_raw(cdna4, monkeypatch):
    case = _mx8_inputs()
    ws = _mx8_tile() if cdna4 else case.ws
    packed_a = _a_sorter(True, monkeypatch)
    cfg = _mx8_config(128, True, cdna4)
    args = _capture(case, ws, "CDNA4_SCALE" if cdna4 else None, cfg, monkeypatch)
    _assert_b_scales(args, ws, cdna4=cdna4)
    tuning = conf._cval(args["CFG_TUNING"])
    assert tuning.A_SCALE_SORTED_SHUFFLED is cdna4
    a_swizzle = ScaleSwizzle.SORTED_SHUFFLED if cdna4 else ScaleSwizzle.NONE
    assert conf._cval(args["A_SCALE_SWIZZLE"]) == a_swizzle
    assert args["a_scale_ptr"] is (packed_a if cdna4 else case.xs)


def test_k128_caller_shuffled_a_with_raw_b_is_refused(monkeypatch):
    case = _mx8_inputs()
    case.gather = None
    _no_launch(monkeypatch)
    with pytest.raises(ValueError, match="BLOCK_K 128"):
        host.moe_gemm_gluon(
            case.y,
            case.x,
            case.w,
            case.xs,
            case.ws,
            None,
            None,
            case.routing_data,
            None,
            None,
            case.n,
            case.k,
            True,
            1.0,
            None,
            False,
            config=_mx8_config(128, True, False),
            x_scales_sorted_shuffled=True,
        )
    monkeypatch.setattr(host, "get_arch", lambda: "gfx950")
    entry = _mx8_config(128, True, False)
    monkeypatch.setattr(host, "_default_launch_config", lambda *_: dict(entry))
    ok, why = host.gluon_supported(
        x=case.x,
        w=case.w,
        x_scales=case.xs,
        w_scales=case.ws,
        y=case.y,
        bias=None,
        routing_data=case.routing_data,
        swizzle_mx_scale=None,
        split_k=1,
        x_static_scale=None,
        quant_static_scale=None,
        out_quant=None,
        N=case.n,
        K=case.k,
        apply_swiglu=True,
        x_scales_sorted_shuffled=True,
    )
    assert not ok and "BLOCK_K 128" in why
