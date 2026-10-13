# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fresh FlashKDA launchers for tests.

Each FlashKDA kernel is decorated with its intj launcher, so its name is the
launcher; ``kernel.__self__.options.jit_func`` is the kernel it launches (the
tuner chain included), shared with the module. A test that needs a first call,
other tuner configs, a spy or a Triton reference re-runs the kernel modules
instead, with its own stand-in for ``intj.make_launcher``, and gets a launcher
that has never been called over a tuner of its own.
"""

import contextlib
import importlib
import inspect

import intj
import torch
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction

from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn import (
    flash_kda_k1 as _g1,
)
from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn import (
    flash_kda_k2 as _g2,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_kda as _fk

KERNELS = (
    "k1_prepare_gluon",
    "k2_ab_fused_gluon",
    "_flash_kda_prepare_kernel",
    "_flash_kda_segment_kernel",
    "_flash_kda_seg_scan_kernel",
)
_MODULES = (_g1, _g2, _fk)  # flash_kda imports the Gluon modules per call
_ROUTE_FLAGS = ("AITER_FDA_USE_GLUON_K1", "AITER_FDA_USE_GLUON_K2")
_REAL = intj.make_launcher


def jit_of(kernel):
    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel


def _reload(make):
    flags = {name: getattr(_fk, name) for name in _ROUTE_FLAGS}
    intj.make_launcher = make
    try:
        for module in _MODULES:
            importlib.reload(module)
    finally:
        intj.make_launcher = _REAL
        for name, value in flags.items():  # Keep a test's pinned route.
            setattr(_fk, name, value)


@contextlib.contextmanager
def relaunched(make=_REAL):
    """Re-run the KDA kernel modules with ``make`` standing in for make_launcher.

    Every kernel's decorator calls ``make(kernel, **kwargs)`` with the fresh
    kernel (tuner chain included) the module just built, and its name becomes
    whatever that returns. Yields ``{kernel name: kernel}``. Leaving re-runs
    the modules under the real make_launcher, so later tests start from
    never-called launchers too; a test's pinned route flags survive both.
    """
    kernels = {}

    def recording(kernel=None, **kwargs):
        if kernel is None:  # the decorator-factory form
            return lambda k: recording(k, **kwargs)
        kernels[jit_of(kernel).__name__] = kernel
        return make(kernel, **kwargs)

    _reload(recording)
    try:
        assert set(kernels) == set(KERNELS), sorted(kernels)
        yield kernels
    finally:
        _reload(_REAL)


def spying(record):
    """A make_launcher whose launchers call ``record(name, args)`` first."""

    def make(kernel, **kwargs):
        native, name = _REAL(kernel, **kwargs), jit_of(kernel).__name__

        def launch(*args):
            record(name, args)
            return native(*args)

        return launch

    return make


def with_configs(configs):
    """A make_launcher that first gives the named kernels' tuners ``configs``.

    The launcher reads them at its first check() or call, to tell which
    parameters are tuned, and tunes on its own copy of the tuner.
    """

    def make(kernel, **kwargs):
        name = jit_of(kernel).__name__
        if name in configs:
            tuner = kernel
            while type(tuner) is not Autotuner:
                tuner = tuner.fn
            tuner.configs = list(configs[name])
        return _REAL(kernel, **kwargs)

    return make


def triton_launcher(
    kernel, *, grid_arg=None, grid_cpp=None, dynamic_options=(), options=None
):
    """A make_launcher that launches through ``kernel[grid](...)`` instead.

    Takes the intj call and undoes what intj changes about it: tuned and
    heuristic values are not passed, dynamic options follow the grid, and a
    ``grid_cpp`` grid becomes the Triton grid callback over the tuned values.
    """
    jit, assigned = kernel, set()
    while not isinstance(jit, JITFunction):
        if type(jit) is Autotuner:
            assigned.update(n for c in jit.configs for n in c.all_kwargs())
        elif type(jit) is Heuristics:
            assigned.update(jit.values)
        jit = jit.fn
    names = [p.name for p in jit.params if p.name not in assigned]
    if grid_cpp is not None:
        params = inspect.signature(grid_cpp).parameters.values()
        meta_names = [p.name for p in params if p.kind is not p.KEYWORD_ONLY]
        extra_names = [p.name for p in params if p.kind is p.KEYWORD_ONLY]

    def launch(device, stream, *rest):
        assert device == torch.cuda.current_device()
        del stream  # Triton reads the current stream itself.
        if grid_cpp is None:
            grid, rest = rest[:grid_arg], rest[grid_arg:]
        else:
            extra = dict(zip(extra_names, rest))
            rest = rest[len(extra_names) :]

            def grid(meta):
                return grid_cpp(*(meta[n] for n in meta_names), **extra)

        opts = dict(options or {})
        opts.update(zip(dynamic_options, rest))
        values = rest[len(dynamic_options) :]
        assert len(values) == len(names)
        kernel[grid](**dict(zip(names, values)), **opts)

    return launch
