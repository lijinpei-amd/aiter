# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gfx950 Gluon MoE GEMM2 reads the weights that the caller passes now.

The launch keeps a preshuffled copy of the B payload and of the B scales on the
weight tensors. A strided weight view and an in-place weight or scale update must
give the same result as fresh contiguous tensors with the same values. This also
applies to the replay of a CUDA graph, which does not run the host code that checks
the copies.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

M, N, K, EXPERTS = 32, 256, 2048, 2
_OPS = {
    "a8w8": (moe_gemm_a8w8, torch.float8_e4m3fn),
    "a4w4": (moe_gemm_a4w4, torch.uint8),
}


@pytest.fixture(scope="module")
def case():
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    with pytest.MonkeyPatch.context() as patch:
        for name in os.environ:
            if name.startswith("AITER_TRITON_MOE_GLUON_"):
                patch.delenv(name)
        patch.delenv("AITER_TRITON_MOE_DISABLE_GLUON", raising=False)
        host._launch_spec.cache_clear()
        conf._get_gluon_config_cached.cache_clear()
        # Rows 0-15 go to expert 0 and rows 16-31 to expert 1, one M block each.
        logits = torch.zeros((M, EXPERTS), dtype=torch.float16, device="cuda")
        logits[: M // 2, 0] = 1
        logits[M // 2 :, 1] = 1
        route, gather, scatter = routing(logits, 1)
        torch.manual_seed(5)
        x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
        yield SimpleNamespace(
            route=route,
            gather=gather,
            scatter=scatter,
            x={op: downcast_to_mxfp(x, q, axis=-1) for op, (_, q) in _OPS.items()},
        )
        host._launch_spec.cache_clear()
        conf._get_gluon_config_cached.cache_clear()


@pytest.fixture
def launches(monkeypatch):
    """The (B_PRESHUFFLED, B_SCALE_SHUFFLED) flags of every Gluon launch."""
    seen = []
    original = host._fast_launch

    def capture(kernel, grid, args, *metadata):
        tuning = conf._cval(dict(zip(kernel.arg_names, args))["CFG_TUNING"])
        seen.append((bool(tuning.B_PRESHUFFLED), bool(tuning.B_SCALE_SHUFFLED)))
        return original(kernel, grid, args, *metadata)

    monkeypatch.setattr(host, "_fast_launch", capture)
    return seen


def _weights(op, experts, seed):
    """Column-major MX weights (E, K, N) and their scales."""
    torch.manual_seed(seed)
    w = torch.randn((experts, N, K), dtype=torch.bfloat16, device="cuda")
    return downcast_to_mxfp(w.transpose(1, 2), _OPS[op][1], axis=1)


def _gemm2(op, case, w, ws):
    fn = _OPS[op][0]
    x, xs = case.x[op]
    return fn(
        x,
        w,
        xs,
        ws,
        routing_data=case.route,
        gather_indx=case.gather,
        scatter_indx=case.scatter,
    )


def _fresh(op, case, w, ws):
    """The result for new contiguous copies, which have no cached data."""
    return _gemm2(op, case, w.transpose(1, 2).contiguous().transpose(1, 2), ws.clone())


@pytest.mark.parametrize("op", list(_OPS))
def test_strided_weight_view(op, case, launches):
    full, full_scales = _weights(op, 2 * EXPERTS, seed=1)
    w, ws = full[::2], full_scales[::2]
    out = _gemm2(op, case, w, ws)
    assert launches and all(flags == (True, True) for flags in launches), launches
    assert torch.equal(out, _fresh(op, case, w, ws))


@pytest.mark.parametrize("inference", [False, True], ids=["tensor", "inference"])
@pytest.mark.parametrize("op", list(_OPS))
def test_in_place_weight_and_scale_updates(op, inference, case, launches):
    mode = torch.inference_mode() if inference else torch.no_grad()
    with mode:
        w, ws = _weights(op, EXPERTS, seed=2)
        first = _gemm2(op, case, w, ws)
        assert torch.equal(_gemm2(op, case, w, ws), first)

        new_w, new_ws = _weights(op, EXPERTS, seed=3)
        w.copy_(new_w)
        after_w = _gemm2(op, case, w, ws)
        assert not torch.equal(after_w, first)
        assert torch.equal(after_w, _fresh(op, case, w, ws))

        ws.copy_(new_ws)
        out = _gemm2(op, case, w, ws)
        assert not torch.equal(out, after_w)
        assert torch.equal(out, _fresh(op, case, w, ws))
    assert launches and all(flags == (True, True) for flags in launches), launches


@pytest.mark.parametrize("op", list(_OPS))
def test_graph_replay_reads_updated_weights_and_scales(op, case, launches):
    w, ws = _weights(op, EXPERTS, seed=4)
    # Warm up on a side stream, as torch.cuda.graph requires. This makes the copies.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _gemm2(op, case, w, ws)
    torch.cuda.current_stream().wait_stream(stream)
    eager = len(launches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _gemm2(op, case, w, ws)
    captured = launches[eager:]
    assert launches[:eager] and all(f == (True, True) for f in launches[:eager])
    # The captured launch must not read a copy that a replay cannot refresh.
    assert captured and all(f == (False, False) for f in captured), captured

    graph.replay()
    first = out.clone()
    assert torch.equal(first, _fresh(op, case, w, ws))

    new_w, new_ws = _weights(op, EXPERTS, seed=5)
    w.copy_(new_w)
    graph.replay()
    after_w = out.clone()
    assert not torch.equal(after_w, first)
    assert torch.equal(after_w, _fresh(op, case, w, ws))

    ws.copy_(new_ws)
    graph.replay()
    assert not torch.equal(out, after_w)
    assert torch.equal(out, _fresh(op, case, w, ws))
