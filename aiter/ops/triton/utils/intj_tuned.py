# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch ``@triton.autotune``/``@triton.heuristics`` kernels through intj.

``intj.compat.launch`` refuses decorated kernels; ``intj.make_launcher`` takes
them, runs Triton's tuner on a miss and launches natively on a hit. This keeps
the Triton call spelling of the ``intj.compat.launch`` sites: it binds the
arguments, drops the values the decorators assign, bakes ``str``/dtype/JIT
constexprs, and reuses one bound launcher -- and so one tuner cache -- per
kernel, device, grid shape, compile options and baked values. Unsupported
kernels raise ``UnsupportedKernel``; nothing falls back to Triton.
"""

import dataclasses
import inspect
import threading
from functools import cache

from intj import Constexpr, make_launcher
from intj.compat import launch as compat_launch
from triton import language as tl
from triton.compiler import make_backend
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction

_grid = threading.local()
_ARG, _KWARG, _DEFAULT = range(3)


def _current_grid(meta):
    # make_launcher fixes grid_py per launcher; the caller's grid closes over
    # per-call locals, so each call installs its own for the duration.
    return _grid.fn(meta)


@cache
def _plan(kernel, nargs, names):
    """Where each kept parameter comes from, for one call shape of ``kernel``."""
    jit, tuned = kernel, set()
    while type(jit) in (Autotuner, Heuristics):
        if type(jit) is Autotuner:
            tuned.update(n for c in jit.configs for n in c.all_kwargs())
        else:
            tuned.update(jit.values)
        jit = jit.fn
    if not isinstance(jit, JITFunction):
        raise TypeError(f"expected a decorated @triton.jit kernel, got {kernel!r}")
    params = jit.signature.parameters
    if nargs > len(params) or set(names) & set(tuple(params)[:nargs]):
        raise TypeError(f"{jit.__name__}: bad arguments: {nargs} positional, {names}")
    target = driver.active.get_current_target()
    option_names = {f.name for f in dataclasses.fields(make_backend(target).parse_options({}))}
    sources = []
    for i, p in enumerate(jit.params):
        if p.name in tuned:
            continue  # the decorators assign it; Triton's heuristics even overwrite a caller's value
        if i < nargs:
            sources.append((p.name, p.is_constexpr, _ARG, i))
        elif p.name in names:
            sources.append((p.name, p.is_constexpr, _KWARG, p.name))
        elif params[p.name].default is not inspect.Parameter.empty:
            sources.append((p.name, p.is_constexpr, _DEFAULT, params[p.name].default))
        else:
            raise TypeError(f"{jit.__name__}: missing argument {p.name!r}")
    # Triton parses compile options from every keyword, so one that names a
    # kernel parameter too (``num_warps: tl.constexpr``) is also an option.
    options = tuple(
        sorted(n for n in names if n not in tuned and (n not in params or n in option_names))
    )
    return tuple(sources), options


@cache
def _launcher(kernel, device, dims, options, baked):
    return make_launcher(
        kernel,
        **({"grid_py": _current_grid} if dims is None else {"grid_arg": dims}),
        options=dict(options),
        extra_annotation={n: Constexpr(value=v) for n, v in baked},
        bind_device=True,
    ).bind_device(device)


def _bakes(value):
    return type(value) is str or isinstance(value, (tl.dtype, JITFunction))


def launch_tuned(kernel, grid, *args, **kwargs):
    """``kernel[grid](*args, **kwargs)`` for a decorated kernel, through intj.

    A bare ``@triton.jit`` kernel goes to ``intj.compat.launch``, so a call
    site that picks either kind at runtime can use this for both.
    """
    if type(kernel) not in (Autotuner, Heuristics):
        return compat_launch(kernel, grid, *args, **kwargs)
    sources, option_names = _plan(kernel, len(args), tuple(kwargs))
    values = [
        args[key] if kind == _ARG else kwargs[key] if kind == _KWARG else key
        for _, _, kind, key in sources
    ]
    baked = tuple(
        (name, v)
        for (name, constexpr, _, _), v in zip(sources, values)
        if constexpr and _bakes(v)
    )
    public = (
        [v for (_, c, _, _), v in zip(sources, values) if not (c and _bakes(v))]
        if baked
        else values
    )
    options = tuple((n, kwargs[n]) for n in option_names)
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    if callable(grid):
        native = _launcher(kernel, device, None, options, baked)
        _grid.fn, previous = grid, getattr(_grid, "fn", None)
        try:
            return native(stream, *public)
        finally:
            _grid.fn = previous
    dims = (grid,) if type(grid) is int else tuple(grid)
    return _launcher(kernel, device, len(dims), options, baked)(stream, *dims, *public)
