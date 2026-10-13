# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime loop bounds and frozen/manual modes of the common component driver."""

import os

import pytest
import torch

from aiter.ops.triton._gluon_kernels.gfx950.moe import _frozen
from aiter.ops.triton._gluon_kernels.gfx950.moe._frozen import (
    _buffer_load_order_frozen,
    _buffer_load_tile_frozen,
    _compute_frozen_layout,
    _pipeline_peeled_frozen,
    _validate_frozen_pipeline,
    pipeline_depth_frozen,
    pipeline_unroll_frozen,
)
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import DtypeQuant, WarpPipeline
from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_torch
from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.moe.test_moe_gemm_gluon_registers import (
    _build_register_case,
    _launch_register_case,
    _register_config,
)

#: ``_buffer_load_order_frozen`` for every 1..4 x 1..4 slot grid: payload copies as
#: ``(is_a, tile)`` in slot order. 2x2 is the tuned interleave, the rest alternate A
#: and B from A(0). Pinned when the frozen module stopped sharing code with the live
#: schedule, so the snapshot no longer has a live counterpart to agree with.
_FROZEN_FILL_ORDER = {
    (1, 1): [(1, 0), (0, 0)],
    (1, 2): [(1, 0), (0, 0), (0, 1)],
    (1, 3): [(1, 0), (0, 0), (0, 1), (0, 2)],
    (1, 4): [(1, 0), (0, 0), (0, 1), (0, 2), (0, 3)],
    (2, 1): [(1, 0), (0, 0), (1, 1)],
    (2, 2): [(0, 0), (1, 0), (1, 1), (0, 1)],
    (2, 3): [(1, 0), (0, 0), (1, 1), (0, 1), (0, 2)],
    (2, 4): [(1, 0), (0, 0), (1, 1), (0, 1), (0, 2), (0, 3)],
    (3, 1): [(1, 0), (0, 0), (1, 1), (1, 2)],
    (3, 2): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2)],
    (3, 3): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)],
    (3, 4): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2), (0, 3)],
    (4, 1): [(1, 0), (0, 0), (1, 1), (1, 2), (1, 3)],
    (4, 2): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (1, 3)],
    (4, 3): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2), (1, 3)],
    (4, 4): [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2), (1, 3), (0, 3)],
}


@pytest.mark.parametrize(
    "dtype,depth,unroll", [("mxfp4", 3, 3), ("bf16", 3, 3), ("mxfp8", 5, 6)]
)
def test_frozen_schedule_helpers_hold_their_pinned_values(dtype, depth, unroll):
    """``_frozen.py`` owns private copies of every helper it uses.

    The frozen body is a verbatim snapshot whose acceptance test is identical
    assembly, and it shares no code with the live modules, which may diverge from it
    freely. So these values are pinned rather than compared against live
    counterparts: if one moves, the snapshot moved.
    """
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    quant = {"mxfp4": DtypeQuant.MXFP4, "bf16": DtypeQuant.BF16,
             "mxfp8": DtypeQuant.MXFP8}[dtype]
    tc = host._probe_tuning_config(_register_config(dtype), quant, quant)
    layout = _compute_frozen_layout(tc)
    assert pipeline_depth_frozen(tc, layout) == depth
    assert pipeline_unroll_frozen(tc, layout) == unroll
    assert _pipeline_peeled_frozen(tc) == 1
    assert host._cval(layout.pipeline_register_period()) == 1

    # N outer, M inner: with two slots per axis the visit order is
    # (0,0), (1,0), (0,1), (1,1).
    for nm in range(1, 5):
        for nn in range(1, 5):
            for mi in range(nm):
                for ni in range(nn):
                    assert _frozen._slot_index_frozen(mi, ni, nm, nn) == (
                        ni * nm + mi
                    ), (mi, ni, nm, nn)
    for (nm, nn), order in _FROZEN_FILL_ORDER.items():
        assert _buffer_load_order_frozen(nm, nn) == order, (nm, nn)
        for pos, (is_a, tile) in enumerate(order):
            assert _buffer_load_tile_frozen(pos, nm, nn, True) == (
                tile if is_a else None
            ), (nm, nn, pos)
            assert _buffer_load_tile_frozen(pos, nm, nn, False) == (
                None if is_a else tile
            ), (nm, nn, pos)


@pytest.fixture(scope="module", params=["mxfp4", "bf16"])
def frozen_case(request):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    with pytest.MonkeyPatch.context() as patch:
        for name in os.environ:
            if name.startswith("AITER_TRITON_MOE_GLUON_"):
                patch.delenv(name)
        patch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
        case = _build_register_case(
            request.param, k=960 if request.param == "bf16" else 7168
        )
        assert case.route.block_m == _register_config(case.dtype)["BLOCK_M"]
        _launch_register_case(case, _register_config(case.dtype))
        torch.testing.assert_close(case.output[0], case.expected, rtol=3e-4, atol=3e-5)
        case.baseline = case.output.clone()
        yield case


@pytest.mark.parametrize("mode", list(WarpPipeline))
def test_frozen_step_pipeline_modes_are_bitwise_deterministic(frozen_case, mode):
    config = dict(
        _register_config(frozen_case.dtype), FROZEN_STEP=True, WARP_PIPELINE=int(mode)
    )
    for repeat in range(8):
        frozen_case.output.fill_(float("nan"))
        _launch_register_case(frozen_case, config)
        assert torch.equal(
            frozen_case.output.view(torch.int32), frozen_case.baseline.view(torch.int32)
        ), (mode, repeat)


def test_frozen_unshuffled_k128_scales_load_directly(monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
    case = _build_register_case("mxfp8", m=257, n=512, k=2048, experts=4, topk=2)
    config = dict(
        _register_config("mxfp8"),
        FROZEN_STEP=True,
        A_SCALE_SORTED_SHUFFLED=False,
        B_SCALE_SHUFFLED=False,
    )
    layout = _compute_frozen_layout(
        host._probe_tuning_config(config, DtypeQuant.MXFP8, DtypeQuant.MXFP8)
    )
    assert host._cval(layout.scale_in_reg(0))
    assert host._cval(layout.scale_in_reg(1))
    _launch_register_case(case, config)
    torch.testing.assert_close(case.output[0], case.expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("gated", [False, True], ids=["gemm2", "gemm1"])
@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("dtype", ["bf16", "mxfp4"])
def test_same_compiled_entry_honors_runtime_num_k(gated, depth, dtype, monkeypatch):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    for name in os.environ:
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_B_PRESHUFFLED", "1")
    if dtype == "bf16":
        case = _build_register_case(dtype, m=257, n=512, k=2048, experts=4, topk=2)
        x_ref, w_ref = case.x, case.w
        quant = DtypeQuant.BF16
    else:
        case = _build_register_case(dtype)
        x_ref = upcast_from_mxfp(case.x, case.xs, torch.bfloat16, axis=-1)
        w_ref = upcast_from_mxfp(case.w, case.ws, torch.bfloat16, axis=1)
        quant = DtypeQuant.MXFP4
    if not gated:
        case.output = torch.empty_like(case.raw).unsqueeze(0)
    config = dict(
        _register_config(dtype),
        NUM_LDS_BUFFER=0,
        A_NUM_BUFFER=depth,
        B_NUM_BUFFER=depth,
        A_SCALE_NUM_BUFFER=depth,
        B_SCALE_NUM_BUFFER=depth,
    )
    assert case.route.block_m == config["BLOCK_M"]
    tc = host._probe_tuning_config(config, quant, quant)
    layout = _compute_frozen_layout(tc)
    full_num_k = case.k // config["BLOCK_K"]
    minimum = (
        pipeline_depth_frozen(tc, layout)
        + _pipeline_peeled_frozen(tc)
        + pipeline_unroll_frozen(tc, layout)
    )
    # Depth three exercises every specialized drain phase in the same binary.
    short_counts = (minimum + 1,) if depth == 2 else range(minimum, minimum + depth)
    references = {full_num_k: case.expected if gated else case.raw}
    for short_num_k in short_counts:
        short_k = short_num_k * config["BLOCK_K"]
        assert short_num_k < full_num_k
        assert _validate_frozen_pipeline(tc, short_k)
        prefix = moe_gemm_torch(
            x_ref[:, :short_k].float(),
            w_ref[:, :short_k, :].float(),
            None,
            case.route,
            case.gather,
        )
        if gated:
            gate, up = prefix.chunk(2, dim=-1)
            prefix = torch.nn.functional.silu(gate) * up
        references[short_num_k] = prefix
    captured = {}

    def capture_launch(kernel, grid, args, *_metadata):
        captured.update(kernel=kernel, grid=grid, args=args)

    monkeypatch.setattr(host, "_fast_launch", capture_launch)
    compiled = _launch_register_case(case, config, swiglu=gated, split=gated)
    torch.testing.assert_close(
        case.output[0], references[full_num_k], rtol=3e-4, atol=3e-5
    )
    kernel = captured["kernel"]
    launch_args = list(captured["args"])
    num_k_arg = kernel.arg_names.index("NUM_K")
    cfg_k_arg = kernel.arg_names.index("CFG_K")
    assert host._cval(launch_args[cfg_k_arg]) == case.k
    function = compiled.function
    run = compiled.run
    stream = torch.cuda.current_stream().cuda_stream

    def unexpected_jit(*_args, **_kwargs):
        pytest.fail("changing NUM_K must execute the same compiled entry")

    monkeypatch.setattr(kernel, "run", unexpected_jit)
    previous = {}
    for num_k in (*short_counts, full_num_k, *short_counts, full_num_k):
        case.output.fill_(float("nan"))
        launch_args[num_k_arg] = num_k
        # Keep every layout/shape argument fixed, including CFG_K, A/B strides,
        # and pointers. This invokes the already-loaded binary without JIT entry.
        run(
            captured["grid"],
            1,
            1,
            stream,
            function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *launch_args,
        )
        torch.testing.assert_close(
            case.output[0], references[num_k], rtol=3e-4, atol=3e-5
        )
        if num_k in previous:
            assert torch.equal(case.output.view(torch.int32), previous[num_k])
        previous[num_k] = case.output.view(torch.int32).clone()
        assert compiled.function == function
        assert host._cval(launch_args[cfg_k_arg]) == case.k
