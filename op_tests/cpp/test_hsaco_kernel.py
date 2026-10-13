# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for `csrc.cpp_itfs.utils.jit`, whose `kernel[grid](...)` launches
through intj unless AITER_USE_HSACO=1.

They check the mapping of a Triton-style call onto the launcher's positional
form (positional, keyword and defaulted arguments, compile-option keywords,
every grid form, the stream), against the same call launched by Triton, and
that no launch goes through Triton's `JITFunction.run`.
"""

import pytest
import torch
import triton
import triton.language as tl
from intj.launcher import UnsupportedKernel
from triton.runtime.jit import JITFunction

from csrc.cpp_itfs import utils
from csrc.cpp_itfs.utils import HsacoKernel, jit

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HsacoKernel launches need a GPU"
)

N = 1000


@triton.jit
def _add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=m) + tl.load(y + offs, mask=m), mask=m)


@triton.jit
def _scale_kernel(
    x,
    out,
    n,
    alpha,
    BLOCK: tl.constexpr = 128,
    NEG: tl.constexpr = False,
    CLAMP: tl.constexpr = False,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x + offs, mask=m) * alpha
    if NEG:
        v = -v
    if CLAMP:
        v = tl.minimum(v, 1.0)
    tl.store(out + offs, v, mask=m)


@triton.jit
def _grid_kernel(out):
    # out[z, y, x] = 1 for every program in the grid
    x = tl.program_id(0)
    y = tl.program_id(1)
    z = tl.program_id(2)
    nx = tl.num_programs(0)
    ny = tl.num_programs(1)
    tl.store(out + (z * ny + y) * nx + x, 1)


@pytest.fixture
def no_triton_run(monkeypatch):
    """Fails any launch through Triton's JITFunction.run."""

    def run(self, *args, **kwargs):
        raise AssertionError(f"{self} launched through Triton")

    monkeypatch.setattr(JITFunction, "run", run)


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(N, device="cuda", generator=g)
    y = torch.randn(N, device="cuda", generator=g)
    return x, y


def _scale_reference(x, alpha, neg=False, clamp=False):
    v = x * alpha
    if neg:
        v = -v
    if clamp:
        v = torch.minimum(v, torch.ones_like(v))
    return v


def test_construction_does_no_work():
    k = jit(_grid_kernel, stream=7)
    assert isinstance(k, HsacoKernel)
    assert k.triton_kernel is _grid_kernel and k.stream == 7
    assert id(_grid_kernel) not in utils._INTJ_KERNELS
    with pytest.raises(ValueError, match="not a triton.JITFunction"):
        jit(lambda: None)


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((256,), {}),  # positional
        ((), {"BLOCK": 256}),  # keyword
    ],
)
def test_positional_and_keyword_arguments(no_triton_run, args, kwargs):
    x, y = _inputs()
    out = torch.empty_like(x)
    k = jit(_add_kernel)
    compiled = k[(4,)](x, y, out, N, *args, **kwargs)
    assert torch.equal(out, x + y)
    out.zero_()
    k[4](x=x, y=y, out=out, n=N, BLOCK=256)
    assert torch.equal(out, x + y)
    out.zero_()
    k[(4,)](x, y, n=N, out=out, BLOCK=256)
    assert torch.equal(out, x + y)
    assert compiled.metadata.num_warps == 4


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, {}),  # every default
        ({"BLOCK": 256}, {}),
        ({"NEG": True}, {"neg": True}),
        ({"CLAMP": True, "NEG": True}, {"neg": True, "clamp": True}),
    ],
)
def test_defaults_are_applied(no_triton_run, kwargs, expected):
    x, _ = _inputs()
    out = torch.empty_like(x)
    grid = (triton.cdiv(N, kwargs.get("BLOCK", 128)),)
    jit(_scale_kernel)[grid](x, out, N, 2.0, **kwargs)
    assert torch.equal(out, _scale_reference(x, 2.0, **expected))


def test_compile_option_keywords_match_triton():
    x, _ = _inputs()
    out, ref = torch.empty_like(x), torch.empty_like(x)
    options = {"num_warps": 8, "num_stages": 1, "waves_per_eu": 2}
    k = jit(_scale_kernel)
    compiled = k[(4,)](x, out, N, 2.0, BLOCK=256, NEG=True, **options)
    triton_compiled = _scale_kernel[(4,)](
        x, ref, N, 2.0, BLOCK=256, NEG=True, **options
    )
    assert torch.equal(out, ref)
    for name, value in options.items():
        assert getattr(compiled.metadata, name) == value
    assert compiled.asm["amdgcn"] == triton_compiled.asm["amdgcn"]
    # The same keywords in another order share the plan's launcher.
    k[(4,)](x, out, N, 2.0, waves_per_eu=2, BLOCK=256, num_warps=8, num_stages=1)
    state = utils._INTJ_KERNELS[id(_scale_kernel)]
    assert ("num_stages", "num_warps", "waves_per_eu") in state.launchers
    assert len({id(plan[0]) for plan in state.plans.values() if plan[3] == 3}) == 1


@pytest.mark.parametrize(
    "grid, shape",
    [
        (3, (1, 1, 3)),
        ((3,), (1, 1, 3)),
        ([3, 2], (1, 2, 3)),
        ((3, 2, 2), (2, 2, 3)),
        (lambda meta: (meta["out"].shape[2], meta["out"].shape[1]), (1, 2, 3)),
    ],
)
def test_grid_forms(no_triton_run, grid, shape):
    out = torch.zeros(shape, dtype=torch.int32, device="cuda")
    jit(_grid_kernel)[grid](out)
    assert torch.equal(out, torch.ones_like(out))


def test_grid_function_gets_tritons_meta():
    x, _ = _inputs()
    out = torch.empty_like(x)
    seen = []

    def grid(meta):
        seen.append(meta)
        return (triton.cdiv(meta["n"], meta["BLOCK"]),)

    jit(_scale_kernel)[grid](x, out, N, 2.0, NEG=True, num_warps=4)
    assert torch.equal(out, _scale_reference(x, 2.0, neg=True))
    _scale_kernel[grid](x, out, N, 2.0, NEG=True, num_warps=4)
    ours, triton_meta = seen
    assert list(ours) == list(triton_meta)
    assert all(ours[k] is triton_meta[k] for k in ours)


def test_zero_grid_skips_the_launch(no_triton_run):
    x, y = _inputs()
    out = torch.zeros_like(x)
    compiled = jit(_add_kernel)[(0,)](x, y, out, N, 256)
    assert compiled is not None
    assert not out.any()


def _record_launches(monkeypatch):
    """Fresh launcher state whose launchers record (device, stream)."""
    calls = []
    make = utils._IntjKernel.launcher

    def launcher(self, options):
        real = make(self, options)

        def record(device, stream, *rest):
            calls.append((device, stream))
            return real(device, stream, *rest)

        return record

    monkeypatch.setattr(utils, "_INTJ_KERNELS", {})
    monkeypatch.setattr(utils._IntjKernel, "launcher", launcher)
    return calls


def test_stream(no_triton_run, monkeypatch):
    calls = _record_launches(monkeypatch)
    x, y = _inputs()
    out = torch.empty_like(x)
    device = torch.cuda.current_device()
    side = torch.cuda.Stream()

    jit(_add_kernel)[(4,)](x, y, out, N, 256)
    assert calls[-1] == (device, torch.cuda.current_stream().cuda_stream)
    with torch.cuda.stream(side):
        jit(_add_kernel)[(4,)](x, y, out, N, 256)
    assert calls[-1] == (device, side.cuda_stream)

    torch.cuda.synchronize()
    out.zero_()
    torch.cuda._sleep(10_000_000)  # the current stream is busy; side is not
    jit(_add_kernel, stream=side)[(4,)](x, y, out, N, 256)
    assert calls[-1] == (side.device_index, side.cuda_stream)
    side.synchronize()
    assert torch.equal(out, x + y)
    jit(_add_kernel, stream=side.cuda_stream)[(4,)](x, y, out, N, 256)
    assert calls[-1] == (device, side.cuda_stream)
    torch.cuda.synchronize()

    with pytest.raises(TypeError, match="stream must be"):
        jit(_add_kernel, stream="s")[(4,)](x, y, out, N, 256)


def test_launchers_are_shared_across_wrappers(no_triton_run, monkeypatch):
    calls = _record_launches(monkeypatch)
    x, y = _inputs()
    out = torch.empty_like(x)
    for _ in range(3):
        jit(_add_kernel)[(4,)](x, y, out, N, 256)
    state = utils._INTJ_KERNELS[id(_add_kernel)]
    assert len(calls) == 3
    assert list(state.launchers) == [()] and list(state.plans) == [5]


def test_bad_calls_raise(no_triton_run):
    x, _ = _inputs()
    out = torch.empty_like(x)
    k = jit(_scale_kernel)
    with pytest.raises(UnsupportedKernel, match="unknown compile option"):
        k[(4,)](x, out, N, 2.0, bogus=1)
    with pytest.raises(TypeError, match="alpha"):
        k[(4,)](x, out, N)
    with pytest.raises(TypeError, match="multiple values"):
        k[(4,)](x, out, N, 2.0, x=x)
    with pytest.raises(TypeError, match="too many"):
        k[(4,)](x, out, N, 2.0, 128, False, False, 1)


@pytest.mark.parametrize("path", ["triton", "hsaco"])
def test_outputs_equal_the_other_paths(monkeypatch, tmp_path, path):
    x, y = _inputs()
    calls = [
        (_add_kernel, (4, 1, 1), lambda out: (x, y, out, N, 256), {}),
        (
            _scale_kernel,
            (4, 1, 1),
            lambda out: (x, out, N, 2.0),
            {"BLOCK": 256, "NEG": True, "num_warps": 4, "waves_per_eu": 2},
        ),
    ]
    expected = []
    for kernel, grid, args, kwargs in calls:
        out = torch.zeros_like(x)
        compiled = jit(kernel)[grid](*args(out), **kwargs)
        expected.append((out, compiled))
    if path == "hsaco":
        pytest.importorskip(
            "hip", reason="AITER_USE_HSACO=1 launches through hip-python"
        )
        monkeypatch.setattr(utils, "AITER_USE_HSACO", 1)
        monkeypatch.setattr(utils, "BUILD_DIR", str(tmp_path))
    for (kernel, grid, args, kwargs), (want, compiled) in zip(calls, expected):
        out = torch.zeros_like(x)
        if path == "hsaco":
            jit(kernel)[grid](*args(out), **kwargs)
        else:
            triton_compiled = kernel[grid](*args(out), **kwargs)
            assert compiled.asm["amdgcn"] == triton_compiled.asm["amdgcn"]
        torch.cuda.synchronize()
        assert torch.equal(out, want)
