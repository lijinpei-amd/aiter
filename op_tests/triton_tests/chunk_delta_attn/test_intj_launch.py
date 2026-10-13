# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for FlashKDA's direct intj launches.

Every FlashKDA kernel is decorated with its intj launcher; the autotuned K1/K2
carry their tuner, so intj tunes on a miss. intj's own
``test_spec_key_is_never_coarser_than_triton`` covers its specialization key.
These tests check what FlashKDA adds on top: that the results are the bytes an
ordinary Triton launch produces, that no launch goes through Triton's
``JITFunction.run``, and the device, stream and lifetime rules of the
launchers.
"""

import contextlib
import gc
import weakref

import pytest
import torch
import triton
from intj.launcher import Launcher, module_of
from triton.runtime.jit import JITFunction

from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as fk
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import flash_kda_fwd
from op_tests.triton_tests.chunk_delta_attn import kda_launchers

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash_kda needs a GPU"
)

device = "cuda"
dtype = torch.bfloat16
K_DIM = 128
LOWER_BOUND = -5.0


def make_inputs(B, T, H, seed=0, bias=False, state=False, varlen=False):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, device=device, dtype=dt)

    args = {
        "q": rnd(B, T, H, K_DIM),
        "k": rnd(B, T, H, K_DIM),
        "v": rnd(B, T, H, K_DIM),
        "g": rnd(B, T, H, K_DIM, dt=torch.float32),
        "beta": rnd(B, T, H, dt=torch.float32),
        "A_log": rnd(H, dt=torch.float32),
        "dt_bias": rnd(H * K_DIM, dt=torch.float32) if bias else None,
        "initial_state": (rnd(B, H, K_DIM, K_DIM, dt=torch.float32) if state else None),
    }
    if varlen:
        # One packed sequence per batch entry, which is how the varlen path is
        # reached: B collapses to 1 and the bounds carry the split.
        args["q"], args["k"], args["v"], args["g"] = (
            args[n].reshape(1, B * T, H, -1) for n in ("q", "k", "v", "g")
        )
        args["beta"] = args["beta"].reshape(1, B * T, H)
        args["cu_seqlens"] = torch.arange(
            0, B * T + 1, T, device=device, dtype=torch.int32
        )
        if args["initial_state"] is not None:
            args["initial_state"] = args["initial_state"][:B]
    return args


def run(args, **kw):
    return flash_kda_fwd(
        **args,
        scale=K_DIM**-0.5,
        lower_bound=LOWER_BOUND,
        output_final_state=True,
        **kw,
    )


@contextlib.contextmanager
def ordinary_launches():
    """Route FlashKDA's launchers through ``kernel[grid](...)``, for tests."""
    with kda_launchers.relaunched(kda_launchers.triton_launcher):
        yield


@pytest.fixture
def triton_route(monkeypatch):
    """The Triton K1/K2 (the Gluon ones need gfx950)."""
    monkeypatch.setattr(fk, "AITER_FDA_USE_GLUON_K1", False)
    monkeypatch.setattr(fk, "AITER_FDA_USE_GLUON_K2", False)


def _assert_same(args, **kw):
    with ordinary_launches():
        want_o, want_s = run(args, **kw)
    got_o, got_s = run(args, **kw)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


@pytest.mark.parametrize("B,T,H", [(1, 512, 12), (1, 4096, 12), (2, 1024, 4)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("state", [False, True])
def test_matches_ordinary_path(B, T, H, bias, state):
    _assert_same(make_inputs(B, T, H, bias=bias, state=state))


@pytest.mark.parametrize("chunks_per_seg", [0, 4])
def test_matches_ordinary_path_when_segmented(chunks_per_seg):
    _assert_same(make_inputs(1, 4096, 12), chunks_per_seg=chunks_per_seg)


def test_matches_ordinary_path_varlen():
    _assert_same(make_inputs(3, 512, 12, varlen=True))


def test_alignment_is_part_of_the_key():
    """A tensor off a 16-byte boundary must not reach a kernel told otherwise."""
    B, T, H = 1, 512, 12
    args = make_inputs(B, T, H)
    run(args)
    wide = torch.randn(B * T * H * K_DIM + 1, device=device, dtype=dtype)
    skewed = wide[1:].view(B, T, H, K_DIM)
    assert skewed.data_ptr() % 16 != 0
    _assert_same(dict(args, q=skewed))


def test_no_launch_goes_through_triton(monkeypatch, triton_route):
    """All three Triton kernels reach intj, tuned or not, first call included."""
    seen = set()

    def fail(*args, **kwargs):
        raise AssertionError("FlashKDA launched through JITFunction.run")

    # Never-called launchers: first calls too, not just hits.
    with kda_launchers.relaunched(kda_launchers.spying(lambda n, _: seen.add(n))):
        monkeypatch.setattr(JITFunction, "run", fail)
        run(make_inputs(1, 512, 4, seed=3), chunks_per_seg=4)
    assert seen == {
        "_flash_kda_prepare_kernel",
        "_flash_kda_segment_kernel",
        "_flash_kda_seg_scan_kernel",
    }


def test_multi_config_tuning_stays_native(monkeypatch, triton_route):
    """With several candidates, intj tunes K1 and K2 itself; still no Triton launch."""
    args = make_inputs(1, 1024, 4)
    want_o, want_s = run(args, chunks_per_seg=4)

    configs = kda_launchers.with_configs(
        {
            "_flash_kda_prepare_kernel": [
                triton.Config({}, num_warps=w, num_stages=1) for w in (2, 4)
            ],
            "_flash_kda_segment_kernel": [
                triton.Config({"BW": bw}, num_warps=2, num_stages=2) for bw in (16, 32)
            ],
        }
    )
    with kda_launchers.relaunched(configs) as kernels:
        monkeypatch.setattr(
            JITFunction, "run", lambda *a, **k: pytest.fail("Triton run")
        )
        got_o, got_s = run(args, chunks_per_seg=4)
        k1 = kernels["_flash_kda_prepare_kernel"].fn
        k2 = kernels["_flash_kda_segment_kernel"]
        assert len(k1.configs) == len(k2.configs) == 2
        assert k1.best_config in k1.configs and k2.best_config in k2.configs
    # Other configs reorder no reduction the result depends on beyond rounding.
    assert (got_o.float() - want_o.float()).abs().max() < 1e-2
    assert (got_s.float() - want_s.float()).abs().max() < 1e-3


def test_warmed_flash_kda_replays_from_graph(triton_route):
    args = make_inputs(1, 512, 4)

    def call():
        return run(args, chunks_per_seg=4)

    eager = call()  # Warm every KDA native handle before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, state = call()
    graph.replay()  # Capture records work; the first replay executes it.
    torch.cuda.synchronize()
    assert torch.equal(out, eager[0])
    assert torch.equal(state, eager[1])
    out.zero_()
    state.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager[0])
    assert torch.equal(state, eager[1])


def test_same_specialization_uses_current_input_tensors():
    first = make_inputs(1, 512, 12, seed=0)
    second = make_inputs(1, 512, 12, seed=1)
    first_o, _ = run(first)
    got_o, got_s = run(second)
    with ordinary_launches():
        want_o, want_s = run(second)
    assert not torch.equal(first_o, got_o)
    assert torch.equal(got_o, want_o)
    assert torch.equal(got_s, want_s)


def test_rejects_cpu_input_pointer_before_launch():
    args = make_inputs(1, 512, 12)
    args["A_log"] = args["A_log"].cpu()
    with pytest.raises(ValueError, match="same GPU"):
        run(args)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_input_gpu_must_be_current():
    current = torch.cuda.current_device()
    other = 1 if current == 0 else 0
    first = make_inputs(1, 512, 12)
    run(first)  # Warm the same shape on the current GPU.
    args = {
        name: value.to(f"cuda:{other}") if isinstance(value, torch.Tensor) else value
        for name, value in first.items()
    }
    with pytest.raises(ValueError, match="current"):
        run(args)
    with torch.cuda.device(other):
        got_o, got_s = run(args)
        with ordinary_launches():
            want_o, want_s = run(args)
        assert torch.equal(got_o, want_o)
        assert torch.equal(got_s, want_s)


def test_cached_native_handle_does_not_retain_tensor():
    args = make_inputs(1, 512, 12)
    ref = weakref.ref(args["q"])
    run(args, chunks_per_seg=4)
    torch.cuda.synchronize()
    del args
    gc.collect()
    assert ref() is None


def _launchers():
    from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k1 import (
        k1_prepare_gluon,
    )
    from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k2 import (
        k2_ab_fused_gluon,
    )

    return (
        k1_prepare_gluon,
        k2_ab_fused_gluon,
        fk._flash_kda_prepare_kernel,
        fk._flash_kda_segment_kernel,
        fk._flash_kda_seg_scan_kernel,
    )


def test_kernels_are_intj_launchers():
    """Each module name is the launcher, not a JITFunction (that is
    `__self__.options.jit_func`)."""
    for kernel in _launchers():
        assert not isinstance(kernel, JITFunction)
        assert isinstance(kernel.__self__, Launcher)


def test_every_launcher_builds():
    """Including both Gluon kernels, which only run on gfx950: no refusal here."""
    for kernel in _launchers():
        module_of(kernel)
