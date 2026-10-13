# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.mla's stage-2 merge JIT function, for ``triton.tools.compile``.

``compile_kernel`` executes the file at ``path``, takes ``kernel_name`` from it
as a ``JITFunction``, and its generated loader looks the kernel up in the HSACO
by that same name. In ``aiter/mla.py`` that name is the kernel's intj launcher,
so the AOT tools point ``path`` here, where it is the JIT function.
"""

from aiter.mla import _fwd_kernel_stage2_asm as _launcher

_fwd_kernel_stage2_asm = _launcher.__self__.options.jit_func
