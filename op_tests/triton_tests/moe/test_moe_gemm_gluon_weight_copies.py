# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The gfx950 Gluon MoE GEMM2 reads the weights that the caller passes now.

The launch keeps a preshuffled copy of the B payload and of the B scales on the
weight tensors. A strided weight view and an in-place weight or scale update must
give the same result as fresh contiguous tensors with the same values. This also
applies to the replay of a CUDA graph, which does not run the host code that checks
the copies, to a copy used from another stream, and to padded weights (unpadded_N,
unpadded_K), which keep their stored layout.
"""

import copy
import io
import os
from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.moe import moe_op_gemm_gluon as host
from aiter.ops.triton.moe import moe_op_gemm_gluon_config as conf
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

M, N, K, EXPERTS = 32, 256, 2048, 2
K_PAD = 3072  # stored K of the padded-K weights
_FP8, _FP4 = torch.float8_e4m3fn, torch.uint8
#: op -> (public wrapper, activation storage, weight storage)
_OPS = {
    "a8w8": (moe_gemm_a8w8, _FP8, _FP8),
    "a8w4": (moe_gemm_a8w4, _FP8, _FP4),
    "a4w4": (moe_gemm_a4w4, _FP4, _FP4),
}


def _routes(rows):
    """Rows split evenly over the two experts, one M block each."""
    logits = torch.zeros((rows, EXPERTS), dtype=torch.float16, device="cuda")
    logits[: rows // 2, 0] = 1
    logits[rows // 2 :, 1] = 1
    return routing(logits, 1)


def _activations(rows, k):
    x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda")
    return {op: downcast_to_mxfp(x, q, axis=-1) for op, (_, q, _) in _OPS.items()}


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
        route, gather, scatter = _routes(M)
        torch.manual_seed(5)
        # 16 rows give block_m 16, the only block_m where unpadded_N/K apply.
        routes16 = _routes(16)
        assert routes16[0].block_m == 16
        yield SimpleNamespace(
            route=route,
            gather=gather,
            scatter=scatter,
            x=_activations(M, K),
            routes16=routes16,
            x16=_activations(16, K_PAD),
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


def _weights(op, experts, seed, n=N, k=K):
    """Column-major MX weights (E, K, N) and their scales."""
    torch.manual_seed(seed)
    w = torch.randn((experts, n, k), dtype=torch.bfloat16, device="cuda")
    return downcast_to_mxfp(w.transpose(1, 2), _OPS[op][2], axis=1)


def _call(op, routes, x, xs, w, ws, **padding):
    route, gather, scatter = routes
    return _OPS[op][0](
        x,
        w,
        xs,
        ws,
        routing_data=route,
        gather_indx=gather,
        scatter_indx=scatter,
        **padding,
    )


def _gemm2(op, case, w, ws):
    routes = case.route, case.gather, case.scatter
    return _call(op, routes, *case.x[op], w, ws)


def _has_copy(t):
    return (id(t), host._B_PRESHUFFLE_COPY) in host._WEIGHT_COPIES


def _dense(t):
    """A new K-contiguous copy of ``t``, with no cached data."""
    return t.transpose(1, 2).contiguous().transpose(1, 2)


def _fresh(op, case, w, ws):
    """The result for new contiguous copies, which have no cached data."""
    return _gemm2(op, case, _dense(w), ws.clone())


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
    # An inference tensor has no version counter, so GEMM2 reads it directly.
    copies = not inference
    assert launches and all(f == (copies, copies) for f in launches), launches
    assert _has_copy(w) == copies


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


@pytest.mark.parametrize("write", ["data", "graph"])
@pytest.mark.parametrize("op", list(_OPS))
def test_drop_weight_copies_after_untracked_writes(op, write, case, launches):
    w, ws = _weights(op, EXPERTS, seed=6)
    new_w, new_ws = _weights(op, EXPERTS, seed=7)
    if write == "graph":
        # A graph that only writes the weights: its replay does not count the write.
        src_w, src_ws = w.clone(), ws.clone()
        update = torch.cuda.CUDAGraph()
        with torch.cuda.graph(update):
            w.copy_(src_w)
            ws.copy_(src_ws)
    first = _gemm2(op, case, w, ws)  # makes the copies
    if write == "graph":
        src_w.copy_(new_w)
        src_ws.copy_(new_ws)
        update.replay()
    else:
        w.data.copy_(new_w)
        ws.data.copy_(new_ws)
    host.drop_weight_copies(w, ws, None)
    out = _gemm2(op, case, w, ws)
    assert not torch.equal(out, first)
    assert torch.equal(out, _fresh(op, case, w, ws))
    assert launches and all(flags == (True, True) for flags in launches), launches


def _clear_config_caches():
    for cached in (
        host._launch_spec,
        conf._get_gluon_config_cached,
        conf._gemm2_launch_config,
    ):
        cached.cache_clear()


@pytest.mark.parametrize("op", list(_OPS))
def test_allow_b_via_lds_environment_skips_the_packed_copy(
    op, case, launches, monkeypatch
):
    w, ws = _weights(op, EXPERTS, seed=8)
    default = _gemm2(op, case, w, ws)
    assert launches[-1][0], "the family entry reads a packed B"
    plain = w.transpose(1, 2).contiguous().transpose(1, 2)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_ALLOW_B_VIA_LDS", "1")
    _clear_config_caches()
    try:
        out = _gemm2(op, case, plain, ws)
    finally:
        monkeypatch.delenv("AITER_TRITON_MOE_GLUON_ALLOW_B_VIA_LDS")
        _clear_config_caches()
    assert not launches[-1][0]
    assert not _has_copy(plain)
    assert torch.equal(out, default)


@pytest.mark.parametrize("op", list(_OPS))
def test_unpadded_n_keeps_the_stored_layout(op, case, launches):
    x, xs = (t[:, : t.shape[1] * K // K_PAD] for t in case.x16[op])
    w, ws = _weights(op, EXPERTS, seed=10, n=2 * N)
    out = _call(op, case.routes16, x, xs, w, ws, unpadded_N=N)
    assert launches[-1] == (False, False), "a padded tensor gets no copy"
    full = _call(op, case.routes16, x, xs, w, ws)  # now with copies of all 2N columns
    assert launches[-1] == (True, True)
    again = _call(op, case.routes16, x, xs, w, ws, unpadded_N=N)
    ref = _call(op, case.routes16, x, xs, _dense(w[..., :N]), _dense(ws[..., :N]))
    assert torch.equal(out[:, :N], ref)
    assert torch.equal(again[:, :N], ref)
    assert torch.equal(full[:, :N], ref)


@pytest.mark.parametrize("op", list(_OPS))
def test_smaller_unpadded_k_after_a_full_k_call(op, case, launches):
    x, xs = case.x16[op]
    w, ws = _weights(op, EXPERTS, seed=11, k=K_PAD)
    _call(op, case.routes16, x, xs, w, ws)  # the full K: makes the copies
    assert launches[-1] == (True, True)
    out = _call(op, case.routes16, x, xs, w, ws, unpadded_K=K)
    assert launches[-1] == (False, False), "a padded tensor gets no copy"
    kx, kw = x.shape[1] * K // K_PAD, w.shape[1] * K // K_PAD
    ref = _call(
        op,
        case.routes16,
        x[:, :kx].contiguous(),
        xs[:, : K // 32].contiguous(),
        _dense(w[:, :kw]),
        _dense(ws[:, : K // 32]),
    )
    assert torch.equal(out, ref)


@pytest.mark.parametrize("op", list(_OPS))
def test_copy_made_on_one_stream_is_complete_on_another(
    op, case, launches, monkeypatch
):
    # Two streams may share a hardware queue, so an unordered read does not always
    # show; the wait itself is checked too.
    waits = []
    wait_event = torch.cuda.Stream.wait_event

    def recording_wait(stream, event):
        waits.append(stream.cuda_stream)
        return wait_event(stream, event)

    monkeypatch.setattr(torch.cuda.Stream, "wait_event", recording_wait)
    # Compile first: a compile on the timed path outlasts the sleep below, and the
    # copy is then complete before the second stream asks for it.
    _gemm2(op, case, *_weights(op, EXPERTS, seed=99))
    w, ws = _weights(op, EXPERTS, seed=12)
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    for stream in (first, second):
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):
        torch.cuda._sleep(1 << 27)  # holds back the stream that makes the copies
        _gemm2(op, case, w, ws)
    waits.clear()  # wait_stream above also waits on an event
    with torch.cuda.stream(second):
        out = _gemm2(op, case, w, ws)
    assert second.cuda_stream in waits, "the second stream did not wait"
    torch.cuda.synchronize()
    assert launches and all(flags == (True, True) for flags in launches), launches
    assert torch.equal(out, _fresh(op, case, w, ws))


def test_versionless_copy_is_kept_until_dropped():
    with torch.inference_mode():
        src = torch.zeros(4, device="cuda")
    made = []

    def make():
        made.append(src + len(made))
        return made[-1]

    first = host._cached_copy(src, host._B_PRESHUFFLE_COPY, make)
    assert host._cached_copy(src, host._B_PRESHUFFLE_COPY, make) is first
    host.drop_weight_copies(src)
    assert host._cached_copy(src, host._B_PRESHUFFLE_COPY, make) is not first
    assert len(made) == 2


def test_copy_made_during_graph_capture_is_not_kept():
    src = torch.zeros(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        host._cached_copy(src, host._B_PRESHUFFLE_COPY, lambda: src + 1)
    assert not _has_copy(src)


@pytest.mark.parametrize("op", list(_OPS))
def test_dropped_copy_stays_allocated_for_a_consumer_stream(
    op, case, launches, monkeypatch
):
    # Two streams may share a hardware queue, so a reused block does not always
    # show; the record on the consumer stream is checked too.
    recorded = []
    record_stream = torch.Tensor.record_stream

    def recording(tensor, stream):
        recorded.append(stream.cuda_stream)
        return record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", recording)
    w, ws = _weights(op, EXPERTS, seed=13)
    _gemm2(op, case, w, ws)  # makes the copies on this stream
    torch.cuda.synchronize()
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(1 << 27)  # the GEMM below runs after the drop
        out = _gemm2(op, case, w, ws)
    host.drop_weight_copies(w, ws)
    # Blocks of the copies' sizes, written on this stream: freed memory goes here.
    reuse = [
        torch.full(
            (t.numel() * t.element_size(),), 0x7F, dtype=torch.uint8, device="cuda"
        )
        for t in (w, ws)
    ]
    assert recorded.count(consumer.cuda_stream) == 2, "one record per copy"
    torch.cuda.synchronize()
    del reuse
    assert launches and all(flags == (True, True) for flags in launches), launches
    assert torch.equal(out, _fresh(op, case, w, ws))


@pytest.mark.parametrize("op", list(_OPS))
def test_weights_with_copies_save_and_copy(op, case, launches):
    w, ws = _weights(op, EXPERTS, seed=14)
    weight = torch.nn.Parameter(w, requires_grad=False)
    module = torch.nn.Module()
    module.weight = weight
    _gemm2(op, case, weight, ws)
    assert launches[-1] == (True, True)
    for obj in (weight, ws, module, {"w": weight, "ws": ws}):
        torch.save(obj, io.BytesIO())
    buffer = io.BytesIO()
    torch.save(ws, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cuda")
    assert torch.equal(loaded, ws) and not vars(loaded)
    assert torch.equal(copy.deepcopy(ws), ws)
