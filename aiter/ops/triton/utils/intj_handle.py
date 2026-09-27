# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lazily built, device-bound ``intj.make_launcher`` handles.

``make_launcher`` builds a native extension and queries the GPU target, and
Aiter imports its kernel modules without a GPU, so a handle is declared next
to its kernel and built on the first launch on each device::

    _foo_launch = intj_handle(_foo_kernel, grid_arg=1, options={"num_warps": 4})

    device, stream = current_device_stream()
    _foo_launch(device)(stream, grid_x, *public_args)

The call is ``make_launcher``'s bound form: positional, baked values omitted,
defaults not filled in, decorated values (``@triton.autotune`` /
``@triton.heuristics``) not passed. Like ``kernel[grid](...)`` it launches on
the current device and stream. Unlike ``intj.compat.launch`` it does not
check for Triton launch hooks, CPU tensors or ``TensorWrapper`` arguments.
"""

import functools

import torch
from intj import Constexpr, make_launcher


def intj_handle(kernel, *, baked=None, **kwargs):
    """``make_launcher(kernel, bind_device=True, **kwargs).bind_device(device)``,
    built once per device on first use.

    ``baked`` maps constexpr names to fixed values (``str``, ``tl.dtype``, JIT
    functions, ...) and removes them from the call. ``options`` may be a
    zero-argument callable, evaluated on the device when its handle is built
    (for options that depend on the GPU architecture).
    """
    if baked:
        kwargs["extra_annotation"] = {
            name: Constexpr(value=value) for name, value in baked.items()
        }

    @functools.cache
    def bound(device):
        kw = dict(kwargs)
        if callable(kw.get("options")):
            with torch.cuda.device(device):
                kw["options"] = kw["options"]()
        return make_launcher(kernel, bind_device=True, **kw).bind_device(device)

    return bound


_raw_stream = torch._C._cuda_getCurrentRawStream  # what Triton's driver reads


def current_device_stream():
    """The current device ordinal and its raw stream, as Triton launches on."""
    device = torch.cuda.current_device()
    return device, _raw_stream(device)
