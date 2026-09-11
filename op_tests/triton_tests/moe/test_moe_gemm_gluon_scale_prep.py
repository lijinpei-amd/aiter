# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for packed-scale preparation and the public raw-scale launch API."""

import os
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe._pipeline import _pipeline_peeled
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    DtypeQuant,
    ScaleSwizzle,
)
from aiter.ops.triton.moe import moe_op_gemm_gluon as host


@pytest.fixture(autouse=True)
def _clean_tuning_environment(monkeypatch):
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    host._launch_spec.cache_clear()
    host._validate_launch_config.cache_clear()
    host._get_gluon_config_cached.cache_clear()
    yield
    host._launch_spec.cache_clear()
    host._validate_launch_config.cache_clear()
    host._get_gluon_config_cached.cache_clear()


def _spec(**overrides):
    args = {
        "block_m": 128, "N": 512, "K": 2560,
        "dq_a": DtypeQuant.MXFP8, "dq_b": DtypeQuant.MXFP8,
        "small_grid": False, "has_bias": False, "has_gammas": False,
        "has_gather": True, "has_x_static_scale": False,
        "act": ActivationSpec(int(ActKind.SILU), 1.0, None, False),
        "out_quant": None, "config_items": None, "gate_up_split": False,
    }
    args.update(overrides)
    return host._launch_spec(**args)


def _config(block_k=128):
    config = dict(_spec()[4], BLOCK_K=block_k, MINI_BLOCK_K=block_k,
                  VGPR_PREFETCH_K=block_k)
    if block_k == 256:
        config.update(BLOCK_N=128, MINI_BLOCK_N=64, NUM_LDS_BUFFER=2,
                      warps_per_cta=(2, 2))
    return config


@pytest.mark.parametrize("split", [False, True])
def test_mxfp8_gemm1_default_uses_paired_k128_scales(split):
    spec = _spec(gate_up_split=split)
    cfg = spec[4]
    assert tuple(cfg[name] for name in (
        "BLOCK_M", "BLOCK_N", "BLOCK_K", "MINI_BLOCK_M", "MINI_BLOCK_N",
        "MINI_BLOCK_K", "NUM_LDS_BUFFER", "K_UNROLL", "VGPR_PREFETCH_K",
    )) == (128, 256, 128, 64, 128, 128, 3, 6, 128)
    assert cfg["mfma_instr_shape"] == (16, 16, 128)
    assert cfg["warps_per_cta"] == (1, 4)
    assert cfg["tiles_per_warp"] == (2, 2)
    assert cfg["A_SCALE_SORTED_SHUFFLED"] and cfg["B_SCALE_SHUFFLED"]
    assert not cfg["B_PRESHUFFLED"]
    assert host._cval(spec[1][0]).gate_up_split == split


@pytest.mark.parametrize("overrides", [
    {"act": None},
    {"dq_b": DtypeQuant.MXFP4},
    {"block_m": 64},
    {"N": 384},
    {"K": 1408},
])
def test_other_launches_keep_the_existing_tuning_ladder(overrides):
    cfg = _spec(**overrides)[4]
    expected = host.get_gluon_config(
        overrides.get("block_m", 128), overrides.get("N", 512),
        overrides.get("K", 2560), overrides.get("dq_a", DtypeQuant.MXFP8),
        overrides.get("dq_b", DtypeQuant.MXFP8), False,
    )
    assert cfg == expected


@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
@pytest.mark.parametrize("small_grid", [False, True])
def test_a4_defaults_are_unchanged(block_m, small_grid):
    expected = host.get_gluon_config(
        block_m, 4096, 7168, DtypeQuant.MXFP4, DtypeQuant.MXFP4, small_grid
    )
    cfg = _spec(block_m=block_m, N=4096, K=7168, small_grid=small_grid,
                dq_a=DtypeQuant.MXFP4, dq_b=DtypeQuant.MXFP4)[4]
    assert cfg == expected
    assert not cfg["A_SCALE_SORTED_SHUFFLED"] and not cfg["B_SCALE_SHUFFLED"]


@pytest.mark.parametrize("k", [1280, 2560, 7168])
def test_stage1_capability_uses_the_selected_default_geometry(k):
    cfg = _spec(K=k)[4]
    tc = _spec(K=k)[5]
    assert host._cval(tc.validate(512, k))
    assert host._probe_lds_bytes(cfg, DtypeQuant.MXFP8, DtypeQuant.MXFP8) <= host.LDS_USABLE_BYTES
    case = _inputs(k=k)
    ok, why = host.gluon_supported(
        x=case.x, w=case.w, x_scales=case.xs, w_scales=case.ws,
        y=case.y, bias=None, routing_data=case.routing_data,
        swizzle_mx_scale=None, split_k=1, x_static_scale=None,
        quant_static_scale=None, out_quant=None, N=case.n, K=case.k,
        apply_swiglu=True,
    )
    assert ok, why


def test_explicit_and_frozen_configs_keep_their_geometry(monkeypatch):
    cfg = host.get_gluon_config(128, 512, 2560, DtypeQuant.MXFP8, DtypeQuant.MXFP8)
    explicit = _spec(config_items=tuple(sorted(cfg.items())))[4]
    assert explicit == cfg
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_FROZEN_STEP", "1")
    host._launch_spec.cache_clear()
    host._get_gluon_config_cached.cache_clear()
    frozen = _spec()[4]
    assert frozen["FROZEN_STEP"]
    assert not frozen["A_SCALE_SORTED_SHUFFLED"] and not frozen["B_SCALE_SHUFFLED"]


@pytest.mark.parametrize("field,value", [
    ("MINI_BLOCK_K", 64), ("FROZEN_STEP", True),
    ("mfma_instr_shape", (32, 32, 64)),
])
def test_incompatible_k128_geometry_refuses_scale_preparation(field, value):
    cfg = dict(_config(), **{field: value})
    assert not host._scale_shuffle_supported(cfg, 0)
    assert not host._scale_shuffle_supported(cfg, 1)


@pytest.mark.parametrize("operand", [0, 1])
def test_k128_packing_requires_both_non_k_tiles_in_the_wave(operand):
    tiles = [2, 2]
    tiles[operand] = 1
    cfg = dict(_config(), tiles_per_warp=tuple(tiles))
    assert not host._scale_shuffle_supported(cfg, operand)
    assert host._scale_shuffle_supported(cfg, 1 - operand)


def test_b_preshuffle_preserves_n_order_and_reuses_the_k256_format():
    experts, n, k = 2, 128, 512
    raw = (torch.arange(experts * n * (k // 32), dtype=torch.int64) % 251).to(
        torch.uint8
    ).reshape(experts, n, k // 32).transpose(1, 2)
    original = raw.clone()
    packed = host._shuffled_b_scales(raw, _config())
    assert packed.shape == (experts, n // 32, k)
    flat = packed.reshape(experts, -1)
    # Independent byte address: the two low bits select non-K+16 and K+4.
    # Rows span both gate/up halves, whose original N ordering must be retained.
    for e in range(experts):
        for row in range(n):
            for group in range(k // 32):
                offset = ((row // 32) * k + (group // 8) * 256
                          + (group % 4) * 64 + (row % 16) * 4
                          + ((group % 8) // 4) * 2 + (row % 32) // 16)
                assert flat[e, offset] == raw[e, group, row]
    torch.testing.assert_close(raw, original)
    assert host._shuffled_b_scales(raw, _config(256)) is packed


def _inputs(k=2560):
    m, n, experts, topk = 4, 512, 2, 2
    expt_data = SimpleNamespace(
        hist=torch.tensor([4, 4], dtype=torch.int32),
        token_offs_raw=torch.tensor([0, 4, 8], dtype=torch.int32),
        token_offs_pad=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_pid_map=torch.tensor([0, 1], dtype=torch.int32),
    )
    routing_data = SimpleNamespace(
        block_m=128, n_expts_tot=experts, n_expts_act=topk, expt_data=expt_data,
        n_blocks=lambda *_: experts,
    )
    return SimpleNamespace(
        n=n, k=k, routing_data=routing_data,
        x=torch.empty(m, k, dtype=torch.float8_e4m3fn),
        w=torch.empty(experts, n, k, dtype=torch.float8_e4m3fn).transpose(1, 2),
        xs=torch.empty(m, k // 32, dtype=torch.uint8),
        ws=torch.empty(experts, n, k // 32, dtype=torch.uint8).transpose(1, 2),
        y=torch.empty(1, m * topk, n // 2),
        gather=torch.arange(m * topk, dtype=torch.int32),
    )


def _capture_launch(case, config, split, monkeypatch):
    captured = {}
    sentinel = object()

    def launch(kernel, grid, args, *metadata):
        captured.update(zip(kernel.arg_names, args))
        return sentinel

    monkeypatch.setattr(host, "_fast_launch", launch)
    result = host.moe_gemm_gluon(
        case.y, case.x, case.w, case.xs, case.ws, None, None,
        case.routing_data, case.gather, None, case.n, case.k,
        True, 1.0, None, False, config=config, gate_up_split=split,
    )
    assert result is sentinel
    assert captured["a_ptr"] is case.x and captured["b_ptr"] is case.w
    assert captured["NUM_K"] == case.k // host._cval(captured["CFG_TUNING"]).BLOCK_K
    assert host._cval(captured["CFG_FUNC"]).gate_up_split == split
    return captured


@pytest.mark.parametrize("routing_block_m,config_block_m", [(64, 128), (128, 64)])
def test_block_m_mismatch_is_rejected_before_preparation(
    routing_block_m, config_block_m, monkeypatch
):
    case = _inputs()
    case.routing_data.block_m = routing_block_m
    cfg = dict(_config(), BLOCK_M=config_block_m)

    def unexpected_work(*_):
        pytest.fail("incompatible routing must be rejected before preparation")

    monkeypatch.setattr(host, "_sorted_shuffle_a_scales", unexpected_work)
    monkeypatch.setattr(host, "_shuffled_b_scales", unexpected_work)
    monkeypatch.setattr(host, "_fast_launch", unexpected_work)
    with pytest.raises(AssertionError, match="BLOCK_M.*must match routing block_m"):
        host.moe_gemm_gluon(
            case.y, case.x, case.w, case.xs, case.ws, None, None,
            case.routing_data, case.gather, None, case.n, case.k,
            True, 1.0, None, False, config=cfg,
        )


@pytest.mark.parametrize("k", [0, 256, 640, 768])
def test_short_strips_are_rejected_before_launch(k, monkeypatch):
    case = _inputs(k=k)
    cfg = _config()

    def unexpected_launch(*_):
        pytest.fail("an invalid pipeline must be rejected before launch")

    monkeypatch.setattr(host, "_sorted_shuffle_a_scales", lambda *_: None)
    monkeypatch.setattr(host, "_shuffled_b_scales", lambda *_: None)
    monkeypatch.setattr(host, "_fast_launch", unexpected_launch)
    with pytest.raises(AssertionError, match="NUM_K.*NB_MAX.*PEELED.*UNROLL"):
        host.moe_gemm_gluon(
            case.y, case.x, case.w, case.xs, case.ws, None, None,
            case.routing_data, case.gather, None, case.n, case.k,
            True, 1.0, None, False, config=cfg,
        )


@pytest.mark.parametrize("requested,available", [(True, False), (False, True)])
def test_minimum_uses_resolved_scale_storage_after_preparation(
    requested, available, monkeypatch
):
    case = _inputs(k=768)
    cfg = dict(_config(), K_UNROLL=1, A_SCALE_NUM_BUFFER=4, B_SCALE_NUM_BUFFER=4,
               A_SCALE_SORTED_SHUFFLED=requested, B_SCALE_SHUFFLED=requested)
    raw_tc = host._probe_tuning_config(
        dict(cfg, A_SCALE_SORTED_SHUFFLED=False, B_SCALE_SHUFFLED=False),
        DtypeQuant.MXFP8, DtypeQuant.MXFP8,
    )
    assert (
        raw_tc.pipeline_depth()
        + _pipeline_peeled(raw_tc)
        + raw_tc.pipeline_unroll()
        == 9
    )
    packed_tc = host._probe_tuning_config(
        dict(cfg, A_SCALE_SORTED_SHUFFLED=True, B_SCALE_SHUFFLED=True),
        DtypeQuant.MXFP8, DtypeQuant.MXFP8,
    )
    assert (
        packed_tc.pipeline_depth()
        + _pipeline_peeled(packed_tc)
        + packed_tc.pipeline_unroll()
        == 6
    )
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SORTED_SCALES", "1")
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SHUFFLED_W_SCALES", "1")
    packed_a = torch.empty(2 * 128 * case.k // 32, dtype=torch.uint8)
    packed_b = torch.empty(2, case.n // 32, case.k, dtype=torch.uint8)
    monkeypatch.setattr(host, "_sorted_shuffle_a_scales",
                        lambda *_: packed_a if available else None)
    monkeypatch.setattr(host, "_shuffled_b_scales", lambda *_: packed_b)
    if available:
        args = _capture_launch(case, cfg, True, monkeypatch)
        assert args["NUM_K"] == 6
        assert host._cval(args["CFG_TUNING"]).A_SCALE_SORTED_SHUFFLED
    else:
        with pytest.raises(AssertionError, match="NUM_K.*UNROLL \\(4\\)"):
            _capture_launch(case, cfg, True, monkeypatch)


@pytest.mark.parametrize("k", [256, 640, 768])
def test_capability_rejects_strips_below_selected_pipeline_minimum(k, monkeypatch):
    monkeypatch.setattr(host, "get_arch", lambda: "gfx950")
    case = _inputs(k=k)
    ok, why = host.gluon_supported(
        x=case.x, w=case.w, x_scales=case.xs, w_scales=case.ws,
        y=case.y, bias=None, routing_data=case.routing_data,
        swizzle_mx_scale=None, split_k=1, x_static_scale=None,
        quant_static_scale=None, out_quant=None, N=case.n, K=case.k,
        apply_swiglu=True,
    )
    assert not ok
    assert "NUM_K" in why and "PEELED" in why and "UNROLL" in why


@pytest.mark.parametrize("block_k", [128, 256])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("requested,available", [
    ((True, True), (True, True)), ((True, True), (False, True)),
    ((True, True), (True, False)), ((True, True), (False, False)),
    ((True, False), (True, True)), ((False, True), (True, True)),
    ((False, False), (True, True)),
])
def test_raw_scale_inputs_are_prepared_or_restored_atomically(
    block_k, split, requested, available, monkeypatch
):
    case = _inputs()
    cfg = dict(_config(block_k), A_SCALE_SORTED_SHUFFLED=requested[0],
               B_SCALE_SHUFFLED=requested[1])
    prepared_a = torch.empty(2 * 128 * case.k // 32, dtype=torch.uint8)
    prepared_b = torch.empty(2, case.n // 32, case.k, dtype=torch.uint8)
    seen = []

    def prepare_a(raw, *_):
        assert raw is case.xs
        seen.append(0)
        return prepared_a if available[0] else None

    def prepare_b(raw, *_):
        assert raw is case.ws
        seen.append(1)
        return prepared_b if available[1] else None

    monkeypatch.setattr(host, "_sorted_shuffle_a_scales", prepare_a)
    monkeypatch.setattr(host, "_shuffled_b_scales", prepare_b)
    args = _capture_launch(case, cfg, split, monkeypatch)
    assert seen == [idx for idx in (0, 1) if requested[idx]]
    effective = tuple(want and have for want, have in zip(requested, available))
    if block_k == 128 and not all(effective):
        effective = (False, False)
    tuning = host._cval(args["CFG_TUNING"])
    assert (tuning.A_SCALE_SORTED_SHUFFLED, tuning.B_SCALE_SHUFFLED) == effective
    assert args["a_scale_ptr"] is (prepared_a if effective[0] else case.xs)
    assert args["b_scale_ptr"] is (prepared_b if effective[1] else case.ws)
    assert args["a_scale_stride_m"] == (0 if effective[0] else case.xs.stride(0))
    assert args["A_SCALE_STRIDE_K"] == (32 if effective[0] else case.xs.stride(1))
    assert args["B_SCALE_STRIDE_K"] == (32 if effective[1] else case.ws.stride(1))
    assert host._cval(args["A_SCALE_SWIZZLE"]) == (
        ScaleSwizzle.SORTED_SHUFFLED if effective[0] else ScaleSwizzle.NONE
    )
    assert host._cval(args["B_SCALE_SWIZZLE"]) == (
        ScaleSwizzle.CDNA4_SCALE if effective[1] else ScaleSwizzle.NONE
    )


@pytest.mark.parametrize("block_k", [128, 256])
def test_legacy_environment_switches_still_prepare_raw_scales(block_k, monkeypatch):
    case = _inputs()
    cfg = dict(_config(block_k), A_SCALE_SORTED_SHUFFLED=False, B_SCALE_SHUFFLED=False)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SORTED_SCALES", "1")
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_SHUFFLED_W_SCALES", "1")
    packed_a = torch.empty(2 * 128 * case.k // 32, dtype=torch.uint8)
    packed_b = torch.empty(2, case.n // 32, case.k, dtype=torch.uint8)
    monkeypatch.setattr(host, "_sorted_shuffle_a_scales", lambda *_: packed_a)
    monkeypatch.setattr(host, "_shuffled_b_scales", lambda *_: packed_b)
    args = _capture_launch(case, cfg, True, monkeypatch)
    assert args["a_scale_ptr"] is packed_a and args["b_scale_ptr"] is packed_b


@pytest.mark.parametrize("block_k", [128, 256])
@pytest.mark.parametrize("available", [False, True])
def test_a_sorter_uses_the_same_k256_allocation(block_k, available, monkeypatch):
    from aiter.ops import moe_mxfp4_aux

    case = _inputs()
    seen = []

    def sort_scales(raw, token_ids, cumsum, out, *shape):
        assert raw is case.xs
        assert token_ids.numel() == 2 * 128
        assert cumsum.item() == 2 * 128
        seen.append(shape)
        if not available:
            raise RuntimeError("no generated scale-sort instance")
        out.fill_(47)

    monkeypatch.setattr(moe_mxfp4_aux, "mxfp4_moe_sort_scales", sort_scales)
    out = host._sorted_shuffle_a_scales(
        case.xs, case.routing_data, case.gather, case.k, _config(block_k)
    )
    assert seen == [(2, 2, case.k, 128, 2 * 128)]
    if available:
        assert out.shape == (2 * 128 * case.k // 32,)
        assert torch.all(out == 47)
    else:
        assert out is None


def test_a_sorter_refuses_noncontiguous_raw_scales():
    case = _inputs()
    raw = torch.empty(case.xs.shape[0], case.k // 16, dtype=torch.uint8)[:, ::2]
    assert not raw.is_contiguous()
    assert host._sorted_shuffle_a_scales(
        raw, case.routing_data, case.gather, case.k, _config()
    ) is None
