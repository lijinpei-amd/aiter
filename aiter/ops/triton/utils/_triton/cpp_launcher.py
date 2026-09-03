# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generate a C++ launcher for one specialization of a Triton/Gluon kernel.

Triton's own launch path costs ~6.4 us for the gfx950 Gluon MoE gemm1, of which only
~3.4 us is ``hipModuleLaunchKernel`` itself. The rest is a Python wrapper
(``HIPLauncher.__call__``), ``PyArg_ParseTuple``, a generic walk over the signature that
flattens tuples and skips constexprs, and -- the expensive part -- one
``PyObject_CallMethodNoArgs(obj, "data_ptr")`` per pointer argument.

None of that has to happen per call. Once a kernel is compiled, its argument types are
fixed, so a launcher can be *generated* for exactly that signature: straight-line
unpacking, ``METH_FASTCALL`` instead of tuple packing, and ``THPVariable_Unpack`` instead
of a Python method call per tensor.

Usage::

    launch = make_kernel(jit_fn, args, kwargs, num_warps=8, waves_per_eu=0)
    launch(grid_x, stream, *runtime_args)

``make_kernel`` compiles the kernel through Triton (so the hsaco, the module load and
the metadata are all Triton's), then builds only the dispatch shim.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from functools import cache

#: signature token -> C scalar type. Pointers are handled separately.
_CTYPE = {
    "i1": "int8_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "uint8_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

_TEMPLATE = pathlib.Path(__file__).with_name("cpp_launcher.cpp.j2")


def _describe_args(compiled):
    """Locate every runtime argument, in ABI order, as a path into the call arguments.

    Triton's rule (third_party/amd/backend/driver.py::make_kernel_signature) is: flatten
    tuples, then drop every ``constexpr``. Rather than flatten the *values* in Python on
    each call, walk the *signature* once here and record where each surviving leaf lives
    -- ``top`` is its index in the top-level argument list, ``path`` the tuple indices
    below that. The template turns those into straight-line ``PyTuple_GET_ITEM`` chains.
    """
    args = []

    def leaf(tok, top, path):
        if isinstance(tok, str) and tok.startswith("*"):
            desc = {"is_ptr": True, "ctype": "void *"}
        elif tok in ("fp32", "f32", "fp16", "bf16", "fp64"):
            raise NotImplementedError(
                f"float kernel argument {tok!r} is not supported by the C++ launcher; "
                "it would need a separate unpack path"
            )
        else:
            ctype = _CTYPE.get(tok)
            if ctype is None:
                raise NotImplementedError(f"unsupported kernel argument type {tok!r}")
            desc = {"is_ptr": False, "ctype": ctype}
        desc.update(top=top, path=list(path), nested=bool(path))
        args.append(desc)

    def walk(sig, top, path):
        if isinstance(sig, str):
            if sig != "constexpr":
                leaf(sig, top, path)
            return
        items = sig.values() if isinstance(sig, dict) else sig
        for i, sub in enumerate(items):
            walk(sub, top, path + [i])

    for top, sig in enumerate(compiled.src.signature.values()):
        walk(sig, top, [])
    return args


def _build_module(source: str, module_name: str, build_dir: str):
    from torch.utils.cpp_extension import load_inline

    rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    if not rocm:
        raise RuntimeError("ROCM_PATH or HIP_PATH must be set to build the launcher")
    return load_inline(
        name=module_name,
        cpp_sources=source,
        # The template supplies its own PyInit_; load_inline would emit a second one.
        functions=None,
        extra_include_paths=[os.path.join(rocm, "include")],
        # No -std here: torch's headers require C++20 and cpp_extension already passes
        # -std=c++20; a flag of ours would come later on the command line and win.
        extra_cflags=["-O3", "-D__HIP_PLATFORM_AMD__=1", "-fno-plt"],
        extra_ldflags=[f"-L{os.path.join(rocm, 'lib')}", "-lamdhip64", "-ltorch_python"],
        build_directory=build_dir,
        is_python_module=True,
        verbose=False,
    )


@cache
def _render_and_build(
    sig_hash: str, kernel_name: str, args_key: tuple, n_top_level: int, build_root: str
):
    import jinja2

    args = [
        {"is_ptr": p, "ctype": c, "top": t, "path": list(path), "nested": bool(path)}
        for p, c, t, path in args_key
    ]
    module_name = f"aiter_launch_{sig_hash}"
    source = jinja2.Template(
        _TEMPLATE.read_text(), trim_blocks=False, lstrip_blocks=False
    ).render(
        kernel_name=kernel_name,
        module_name=module_name,
        sig_hash=sig_hash,
        args=args,
        n_top_level=n_top_level,
    )
    build_dir = os.path.join(build_root, module_name)
    os.makedirs(build_dir, exist_ok=True)
    (pathlib.Path(build_dir) / "launcher.cpp").write_text(source)
    return _build_module(source, module_name, build_dir)


def make_kernel(jit_fn, args, kwargs=None, *, grid=(1,), build_root: str | None = None):
    """Compile ``jit_fn`` for these arguments and return ``(launch, compiled)``.

    ``launch(grid_x, stream, *runtime_args)`` takes the kernel's non-constexpr arguments
    in signature order -- tensors, ``None``, or raw device addresses for pointers, and
    Python ints for scalars.

    The Triton compile happens first and unchanged, so the hsaco, the loaded module and
    the metadata are exactly what the normal path would produce; only the per-call
    dispatch shim is ours.
    """
    kwargs = dict(kwargs or {})
    # jit_fn[grid] bakes warmup=False, so go through run() to compile without launching.
    compiled = jit_fn.run(*args, grid=grid, warmup=True, **kwargs)
    if compiled is None:
        compiled = jit_fn[grid](*args, **kwargs)

    return build_for(compiled, build_root=build_root), compiled


def build_for(compiled, *, build_root: str | None = None):
    """Return a ``launch(grid_x, stream, *top_level_args)`` for an already compiled kernel.

    ``top_level_args`` is exactly what ``kernel[grid](...)`` takes -- aggregates still
    packed, constexprs still present. The generated code indexes into the aggregates and
    drops the constexprs itself, so nothing is flattened in Python per call.
    """
    md0 = compiled.metadata
    # The template emits null scratch pointers; a kernel that actually wants scratch
    # would need it allocated per launch, which this launcher does not do.
    for field in ("global_scratch_size", "profile_scratch_size"):
        if getattr(md0, field, 0):
            raise NotImplementedError(
                f"{compiled.name}: {field}={getattr(md0, field)}; the C++ launcher only "
                "supports kernels with no scratch"
            )
    if getattr(md0, "num_ctas", 1) != 1:
        raise NotImplementedError(f"{compiled.name}: num_ctas > 1 is not supported")
    if getattr(md0, "launch_cooperative_grid", False):
        raise NotImplementedError(f"{compiled.name}: cooperative launch not supported")

    arg_desc = _describe_args(compiled)
    args_key = tuple(
        (a["is_ptr"], a["ctype"], a["top"], tuple(a["path"])) for a in arg_desc
    )
    n_top_level = len(compiled.src.signature)
    sig_hash = hashlib.sha256(
        repr((compiled.name, args_key, n_top_level)).encode()
    ).hexdigest()[:16]

    if build_root is None:
        build_root = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "aiter_cpp_launchers"
        )
    mod = _render_and_build(
        sig_hash, compiled.name, args_key, n_top_level, build_root
    )

    md = compiled.metadata
    block_x = md.warp_size * md.num_warps
    shared = compiled.shared if hasattr(compiled, "shared") else md.shared
    mod.init(int(compiled.function), int(shared), int(block_x))
    return mod.launch
