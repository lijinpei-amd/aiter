# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Make Triton survive being profiled by ``rocprofv3``.

``rocprofv3`` ``LD_PRELOAD``s ``librocprofiler-sdk-tool.so`` and
``librocprofiler-sdk.so``; both pull in ``libamd_comgr`` and, through it, a full
``libLLVM.so.23.0git`` -- loaded before the interpreter starts, so every LLVM symbol in
it is global and takes precedence over anything loaded later.  ``libtriton.so``
statically links its *own* (older, differently laid out) LLVM.  When Python later
``dlopen``s it, its static initialisers resolve LLVM symbols to the preloaded copy and
run against objects of the wrong layout::

    #0  llvm::DenseMapBase<...>::LookupBucketFor<llvm::StringRef>   (this = garbage)
    #2  llvm::MapVector<..., DebugCounter::CounterInfo*>::try_emplace_impl
    #3  _GLOBAL__sub_I_PassBuilder.cpp () from triton/_C/libtriton.so
    #10 _dl_open (".../triton/_C/libtriton.so")

which is a SIGSEGV inside ``import triton`` -- and therefore inside ``import aiter``,
before a single kernel has run.

Importing ``triton`` here, with ``RTLD_DEEPBIND``, fixes it: the loader then satisfies
libtriton's LLVM references from libtriton itself, which is what happens anyway when no
profiler is attached.  It has to happen before anything else imports Triton normally,
hence ``sitecustomize``: put this directory on ``PYTHONPATH`` and the interpreter runs it
during start-up, ahead of the benchmark's own imports.

    PYTHONPATH=op_tests/op_benchmarks/triton/rocprof_shim \
    rocprofv3 --pmc TCC_HIT_sum TCC_MISS_sum --truncate-kernels -d out \
      -- python op_tests/op_benchmarks/triton/bench_moe_gemm_gluon.py ...

Outside a profiled run this is a no-op: no rocprofiler in ``LD_PRELOAD``, no early
import, no deep binding.  Doing it unconditionally would change how Triton binds in
*every* run, which is not something a benchmark harness should do silently.

Note that this file shadows the distro ``/usr/lib/pythonX.Y/sitecustomize.py`` (the
apport hook) for the processes that opt in through ``PYTHONPATH``.
"""

import os
import sys


def _rocprofiler_preloaded() -> bool:
    return any(
        "librocprofiler-sdk" in entry
        for entry in os.environ.get("LD_PRELOAD", "").replace(" ", ":").split(":")
    )


if _rocprofiler_preloaded():
    _flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_LOCAL | os.RTLD_DEEPBIND)
    try:
        import triton  # noqa: F401
    except ImportError:
        pass
    finally:
        sys.setdlopenflags(_flags)
