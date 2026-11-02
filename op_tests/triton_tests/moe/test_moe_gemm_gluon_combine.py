# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The in-kernel combine of the gfx950 Gluon MoE GEMM2 (CombineMode.LAST_PRODUCER).

With the combine, the GEMM2 kernel itself adds each token's gate rows: the CTA whose
counter increment is the last for a (token, N slot) reads the rows back and writes the
token row. It must give the bits of the kernel plus ``reduce_grouped``: bf16 rows as
stored, summed in fp32 in gate order. The tests call the public wrappers and force the
combine on or off with ``AITER_TRITON_MOE_GLUON_COMBINE``.
"""

import re

import pytest
import torch

import aiter.ops.triton.moe.moe_op_gemm_gluon as G
from aiter.ops.triton._gluon_kernels.gfx950.moe._types import (
    ActivationSpec,
    ActKind,
    CombineMode,
    DtypeQuant,
)
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_routing.routing import (
    RoutingData,
    compute_expt_data_torch,
    routing,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="Gluon MoE kernels are gfx950 only."
)

_OPS = {
    "a4w4": (moe_gemm_a4w4, torch.uint8, torch.uint8),
    "a8w4": (moe_gemm_a8w4, torch.float8_e4m3fn, torch.uint8),
    "a8w8": (moe_gemm_a8w8, torch.float8_e4m3fn, torch.float8_e4m3fn),
}
_N, _K = 512, 2048

# (id, tokens, experts, topk): one per routing block_m, plus topk 8 at block_m 64.
_SHAPES = [
    ("t1-bm16", 1, 64, 4),
    ("bm32", 400, 64, 4),
    ("bm64", 1024, 128, 4),
    ("bm64-topk8", 1024, 256, 8),
    ("bm128-multi", 4096, 64, 4),
]


def _routing_from(expt_indx, n_expts_tot, block_m):
    """RoutingData for given per-token expert lists (repeats allowed), the way
    routing_torch sorts them."""
    n_expts_act = expt_indx.shape[1]
    flat = expt_indx.reshape(-1).to(torch.int32)
    topk_indx = torch.argsort(flat, stable=True)
    gate_indx = torch.argsort(topk_indx, stable=True)
    gate_scal = torch.rand(flat.shape, dtype=torch.float32, device=flat.device)
    hist = torch.histc(flat, bins=n_expts_tot, max=n_expts_tot - 1).int()
    expt_data = compute_expt_data_torch(hist, n_expts_tot, flat.numel(), block_m)
    rdata = RoutingData(
        block_m, gate_scal[topk_indx], hist, n_expts_tot, n_expts_act, expt_data
    )
    return rdata, topk_indx.int(), gate_indx.int()


def _inputs(op, m, n_expts_tot, n_expts_act, expt_indx=None, block_m=None, n=_N, k=_K):
    torch.manual_seed(0)
    if expt_indx is None:
        logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device="cuda")
        rdata, gindx, sindx = routing(logits, n_expts_act)
    else:
        rdata, gindx, sindx = _routing_from(expt_indx, n_expts_tot, block_m)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n_expts_tot, n, k), dtype=torch.bfloat16, device="cuda").transpose(
        1, 2
    )
    _, x_dtype, w_dtype = _OPS[op]
    w, w_scale = downcast_to_mxfp(w, w_dtype, axis=1)
    x, x_scale = downcast_to_mxfp(x, x_dtype, axis=-1)
    return {
        "x": x,
        "w": w,
        "xs": x_scale,
        "ws": w_scale,
        "bias": torch.randn((n_expts_tot, n), dtype=torch.float32, device="cuda"),
        "gammas": torch.rand(gindx.shape, dtype=torch.float32, device="cuda"),
        "rdata": rdata,
        "gindx": gindx,
        "sindx": sindx,
    }


def _call(op, t, **kwargs):
    return _OPS[op][0](
        t["x"],
        t["w"],
        t["xs"],
        t["ws"],
        bias=t["bias"],
        routing_data=t["rdata"],
        gather_indx=t["gindx"],
        scatter_indx=t["sindx"],
        gammas=t["gammas"],
        out_dtype=torch.bfloat16,
        apply_swiglu=False,
        **kwargs,
    )


@pytest.fixture
def paths(monkeypatch):
    """Counts the two ways a launch can sum the rows, and keeps the combine counters
    of every fused launch."""
    seen = {"fused": [], "reduce": 0}
    workspace, reduce = G._combine_workspace, G.reduce_grouped

    def counted_workspace(*args):
        out = workspace(*args)
        seen["fused"].append(out[0])
        return out

    def counted_reduce(*args, **kwargs):
        seen["reduce"] += 1
        return reduce(*args, **kwargs)

    monkeypatch.setattr(G, "_combine_workspace", counted_workspace)
    monkeypatch.setattr(G, "reduce_grouped", counted_reduce)
    return seen


def _bits(t):
    return t.view(torch.int16)


def _run_both(op, t, monkeypatch, paths, **kwargs):
    """(reduce_grouped output, fused output) of one call, after checking the path
    each one took and that every (token, N slot) counter saw exactly topk rows."""
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "0")
    ref = _call(op, t, **kwargs).clone()
    assert not paths["fused"] and paths["reduce"] == 1
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
    out = _call(op, t, **kwargs).clone()
    torch.cuda.synchronize()
    assert len(paths["fused"]) == 1 and paths["reduce"] == 1, "the combine did not run"
    counters = paths["fused"][-1]
    assert torch.all(counters == t["rdata"].n_expts_act), "a row was not published once"
    return ref, out


@pytest.mark.parametrize("op", sorted(_OPS))
@pytest.mark.parametrize("shape", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_combine_equals_reduce_grouped(op, shape, monkeypatch, paths):
    _, m, n_expts_tot, n_expts_act = shape
    ref, out = _run_both(
        op, _inputs(op, m, n_expts_tot, n_expts_act), monkeypatch, paths
    )
    assert torch.equal(_bits(out), _bits(ref))


def test_combine_sums_repeated_experts_of_one_token(monkeypatch, paths):
    # Every token routes twice to one expert, so two rows of the same token sit next to
    # each other in one block: one CTA (often one wave) publishes both.
    m, n_expts_tot, n_expts_act = 96, 8, 4
    g = torch.Generator(device="cuda").manual_seed(1)
    expt = torch.randint(0, n_expts_tot, (m, 3), device="cuda", generator=g)
    expt_indx = torch.cat([expt, expt[:, :1]], dim=1)
    t = _inputs("a8w4", m, n_expts_tot, n_expts_act, expt_indx, block_m=64)
    ref, out = _run_both("a8w4", t, monkeypatch, paths)
    assert torch.equal(_bits(out), _bits(ref))


def test_combine_ignores_padded_rows(monkeypatch, paths):
    # A padding row (past its expert's rows in a block) publishes nothing, so it must
    # never count as a token's last producer. Its masked counter add returns whatever
    # the register held, often an earlier add's count, and a padding row reads token
    # 0. This many blocks with padding (block_m 128, mean 128 rows) and N tiles make
    # such a stray write land after token 0's real one often enough to see.
    t = _inputs("a8w4", 4096, 128, 4, n=6144, k=3072)
    assert t["rdata"].block_m == 128
    ref, out = _run_both("a8w4", t, monkeypatch, paths)
    assert torch.equal(_bits(out), _bits(ref))
    for _ in range(20):
        assert torch.equal(_bits(_call("a8w4", t)), _bits(ref))


@pytest.mark.parametrize("group_m", [1, 4, 8])
def test_combine_is_independent_of_the_cta_order(group_m, monkeypatch, paths):
    # GROUP_M reorders which CTAs run first, so a different producer is last.
    t = _inputs("a8w8", 1024, 128, 4)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_GROUP_M", str(group_m))
    G._launch_spec.cache_clear()
    ref, out = _run_both("a8w8", t, monkeypatch, paths)
    assert torch.equal(_bits(out), _bits(ref))
    for _ in range(5):
        assert torch.equal(_bits(_call("a8w8", t)), _bits(ref))


def test_combine_ignores_poisoned_buffers(monkeypatch, paths):
    # Every torch.empty of the call (scratch rows, output, counters, inverse scatter)
    # starts as garbage: the result must not depend on it.
    t = _inputs("a8w4", 1024, 128, 4)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "0")
    ref = _call("a8w4", t).clone()
    empty = torch.empty

    def poisoned(*args, **kwargs):
        out = empty(*args, **kwargs)
        if out.is_cuda and out.numel() and out.is_contiguous():
            out.view(torch.uint8).fill_(0xA5)
        return out

    monkeypatch.setattr(torch, "empty", poisoned)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
    out = _call("a8w4", t).clone()
    monkeypatch.setattr(torch, "empty", empty)
    assert paths["fused"], "the combine did not run"
    assert torch.equal(_bits(out), _bits(ref))


def test_combine_repeats_and_replays_in_a_cuda_graph(monkeypatch, paths):
    op = "a8w4"
    t = _inputs(op, 1024, 128, 4)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
    first = _call(op, t).clone()
    second = _call(op, t).clone()
    torch.cuda.synchronize()
    assert torch.equal(_bits(first), _bits(second))
    assert all(torch.all(c == 4) for c in paths["fused"]), "counters were not reset"

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _call(op, t)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = _call(op, t)
    for seed in (2, 3):
        new = _inputs(op, 1024, 128, 4)
        torch.manual_seed(seed)
        fresh = torch.randn(new["x"].shape, dtype=torch.bfloat16, device="cuda")
        x, xs = downcast_to_mxfp(fresh, _OPS[op][1], axis=-1)
        t["x"].copy_(x)
        t["xs"].copy_(xs)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "0")
        ref = _call(op, t)
        monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
        assert torch.equal(_bits(static_out), _bits(ref)), f"replay with seed {seed}"


def test_combine_inside_a_compiled_graph(monkeypatch, paths):
    # Dynamo cannot trace the Gluon launch itself, with or without the combine (it
    # fails on a gluon constexpr), so a compiled model holds the MoE GEMM as an opaque
    # custom op. The fused path must give the same bits there.
    op = "a8w8"
    t = _inputs(op, 1024, 128, 4)

    @torch.library.custom_op("aiter_test::gemm2_combine", mutates_args=())
    def gemm2(x: torch.Tensor) -> torch.Tensor:
        return _call(op, dict(t, x=x)).clone()

    @gemm2.register_fake
    def _(x):
        return x.new_empty((x.shape[0], _N), dtype=torch.bfloat16)

    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "0")
    ref = _call(op, t).clone()
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
    compiled = torch.compile(lambda x: gemm2(x) * 1.0, fullgraph=True)
    out = compiled(t["x"])
    torch.cuda.synchronize()
    assert paths["fused"], "the combine did not run"
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize(
    "op, m, n, k, n_expts_tot, n_expts_act, fused",
    [
        # block_m 64, topk 4, N and K in the measured range: faster, so selected
        ("a8w4", 512, 4096, 3072, 64, 4, True),
        ("a8w8", 512, 4096, 2048, 64, 4, True),
        # outside that range
        ("a8w4", 512, 4096, 2048, 64, 4, False),  # a8w4 K 2048 is slower
        ("a8w8", 512, 2048, 2048, 64, 4, False),  # N below the measured range
        ("a4w4", 512, 4096, 3072, 64, 4, False),  # a4w4 measured no gain
        ("a8w4", 512, 4096, 3072, 128, 8, False),  # topk 8 at block_m 64 is slower
        ("a8w4", 2048, 4096, 3072, 64, 4, False),  # block_m 128 is slower
        ("a8w4", 8, 4096, 3072, 64, 4, False),  # small T is slower
    ],
)
def test_config_selects_the_combine(
    op, m, n, k, n_expts_tot, n_expts_act, fused, monkeypatch, paths
):
    # Without the environment switch the GEMM2 family's LAST_PRODUCER_* keys decide.
    t = _inputs(op, m, n_expts_tot, n_expts_act, n=n, k=k)
    monkeypatch.delenv("AITER_TRITON_MOE_GLUON_COMBINE", raising=False)
    out = _call(op, t).clone()
    assert bool(paths["fused"]) == fused and paths["reduce"] == (0 if fused else 1)
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "0")
    assert torch.equal(_bits(out), _bits(_call(op, t)))


def test_gate_mask_keeps_reduce_grouped(monkeypatch, paths):
    # Masked gates (expert parallelism) are summed by reduce_grouped with indx_valid;
    # the wrapper never offers that contract to the combine.
    t = _inputs("a4w4", 1024, 128, 4)
    gate_valid = (torch.arange(t["sindx"].numel(), device="cuda") % 3 != 0).int()
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
    out = _call("a4w4", t, gate_valid=gate_valid).clone()
    assert not paths["fused"]
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "0")
    assert torch.equal(_bits(out), _bits(_call("a4w4", t, gate_valid=gate_valid)))


def test_combine_rejects_other_tensor_contracts():
    t = _inputs("a8w4", 64, 8, 4)
    rdata, sindx = t["rdata"], t["sindx"]
    y = torch.empty((1, sindx.numel(), _N), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((64, _N), dtype=torch.bfloat16, device="cuda")
    assert G.gluon_combine_supported(y, sindx, out, rdata)[0]
    bad = [
        (y, None, out),
        (y, sindx.long(), out),
        (y, sindx.view(-1, 4).t().reshape(-1)[::2], out),
        (y, sindx, out.float()),
        (y.float(), sindx, out),
        (y, sindx, out[:32]),
        (y, sindx, out.t().contiguous().t()),
    ]
    for args in bad:
        assert not G.gluon_combine_supported(*args, rdata)[0]


def test_combine_rejects_other_kernel_contracts():
    # An activation or an MXFP4 output changes what the rows are; the layout check
    # (_combine_error) refuses them before anything is launched.
    common = (64, _N, _K, DtypeQuant.MXFP8, DtypeQuant.MXFP4, False, True, True, True)
    for act, out_quant in (
        (ActivationSpec(int(ActKind.SILU), 1.0, 7.0, False), None),
        (None, int(DtypeQuant.MXFP4)),
    ):
        spec = G._launch_spec(
            *common,
            False,
            act,
            out_quant,
            None,
            combine=int(CombineMode.LAST_PRODUCER),
        )
        assert G._combine_error(spec[-1], _N)
    plain = G._launch_spec(
        *common, False, None, None, None, combine=int(CombineMode.LAST_PRODUCER)
    )
    assert G._combine_error(plain[-1], _N) is None


def test_combine_publication_order_in_the_isa(monkeypatch):
    # Producer: scratch rows stored with sc0 sc1, then s_waitcnt vmcnt(0), then an
    # agent-scope counter add. Consumer: the gate rows loaded sc0 sc1. In LLVM's GFX942
    # memory model (gfx950 uses it) sc0 sc1 are the bits of a system-scope atomic
    # store or load, and an agent-scope monotonic atomicrmw is coherent across the
    # agent's L2s without extra bits; vmcnt(0) is the wait a release uses. So the count
    # never runs ahead of the rows, without buffer_wbl2 or buffer_inv.
    t = _inputs("a8w4", 1024, 128, 4)
    y = torch.empty((1, t["sindx"].numel(), _N), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((1024, _N), dtype=torch.bfloat16, device="cuda")
    monkeypatch.setenv("AITER_TRITON_MOE_GLUON_COMBINE", "1")
    pgm = G.moe_gemm_gluon(
        y,
        t["x"],
        t["w"],
        t["xs"],
        t["ws"],
        t["bias"],
        t["gammas"],
        t["rdata"],
        t["gindx"],
        t["sindx"],
        _N,
        _K,
        False,
        1.0,
        None,
        False,
        combine_out=out,
    )
    asm = pgm.asm["amdgcn"]
    lines = [ln.strip() for ln in asm.splitlines()]
    atomics = [
        i for i, ln in enumerate(lines) if re.match(r"(global|buffer)_atomic_add", ln)
    ]
    assert atomics, "no counter atomic in the kernel"
    adds = re.findall(r"atomicrmw add ptr addrspace\(1\)[^\n]*", pgm.asm["llir"])
    assert adds and all(
        'syncscope("agent") monotonic' in a for a in adds
    ), "the counter add is not an agent-scope atomic"
    first = atomics[0]
    scratch = [
        i
        for i, ln in enumerate(lines[:first])
        if ln.startswith("buffer_store") and " sc0 sc1" in ln
    ]
    assert scratch, "the scratch rows are not stored write-through"
    assert any(
        ln.startswith("s_waitcnt") and "vmcnt(0)" in ln
        for ln in lines[scratch[-1] : first]
    ), "no vmcnt(0) between the last scratch store and the counter add"
    gate_loads = [
        ln for ln in lines[first:] if ln.startswith("buffer_load") and " sc0 sc1" in ln
    ]
    assert gate_loads, "the last producer does not read the rows sc0 sc1"
