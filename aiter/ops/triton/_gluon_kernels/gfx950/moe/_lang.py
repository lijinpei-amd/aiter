# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton/Gluon language plumbing shared by the gfx950 Gluon MoE modules.

Nothing here knows anything about MoE; it is the constexpr boxing and unboxing every
other module in the package needs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage

__all__ = [
    "MX_GROUP",
    "MX_GROUP_CE",
    "WARP_SIZE",
    "WARP_SIZE_CE",
    "const",
    "constexpr_fields",
    "nop_warp_pipeline_stage",
    "optional",
    "pick_warp_pipeline_stage",
    "require_constexpr",
    "unwrap",
    "unwrap_attr",
]

#: MX group size along K, and the CDNA4 wave width. Plain ints: every use is host-side
#: or inside a ``constexpr_function``, where they take part in ordinary Python
#: arithmetic and f-string formatting.
MX_GROUP = 32
WARP_SIZE = 64

#: The same group size for ``@gluon.jit`` bodies, which may only read ``gl.constexpr``
#: globals -- a plain module-level int is rejected at compile time.
MX_GROUP_CE: gl.constexpr = gl.constexpr(MX_GROUP)
WARP_SIZE_CE: gl.constexpr = gl.constexpr(WARP_SIZE)


@gluon.constexpr_function
def unwrap(x):
    """Unwrap a ``gl.constexpr``. Must be a ``constexpr_function``, not a plain one:
    Triton's aggregate hash walker rejects any bare callable an aggregate method
    references ("Unsupported function referenced"). Called from Python it returns the
    raw value, so ``list()``/indexing on the result still work."""
    return x.value if isinstance(x, gl.constexpr) else x


def unwrap_attr(x):
    """Duck-typed unwrap for the host-side launch-metadata callback, whose arguments are
    not all ``gl.constexpr`` but any of which may carry a ``.value``."""
    return x.value if hasattr(x, "value") else x


def const(v):
    """Wrap a host value so it lands as a compile-time NamedTuple field."""
    return gl.constexpr(v)


@gluon.constexpr_function
def constexpr_fields(spec):
    """Box spec fields so starred arguments preserve nested tuples and None."""
    return tuple(gl.constexpr(value) for value in unwrap(spec))


@gluon.constexpr_function
def optional(buf):
    """Wrap an optional shared buffer so an absent one is a constexpr, not raw None."""
    if buf is None or (isinstance(buf, gl.constexpr) and buf.value is None):
        return gl.constexpr(None)
    return buf


@gluon.constexpr_function
def require_constexpr(cond):
    """Return ``cond``, failing the compile if it is not compile-time.

    Tests for a ``gl.tensor`` rather than a ``gl.constexpr``: the frontend unwraps
    constexpr arguments before a ``constexpr_function`` sees them, so a compile-time
    value arrives as a plain ``bool``/``int``. ``gl.static_assert`` is unusable here --
    it needs the ``@gluon.jit`` semantic context -- but a plain ``assert`` in this body
    already runs at compile time.
    """
    assert not isinstance(
        cond, gl.tensor
    ), f"expected a compile-time value, got a runtime {type(cond).__name__}"
    return cond


class nop_warp_pipeline_stage:
    """``warp_pipeline_stage``'s do-nothing twin: a ``with`` block that emits no IR.

    ``warp_pipeline_stage`` is not a region -- it emits a single *border* op on
    ``__exit__`` and nothing on entry, so the ops it wraps are emitted exactly where they
    were written. A stand-in that emits no border therefore leaves the instruction stream
    untouched, which is what lets one body serve both the pipelined and the un-pipelined
    kernel: same source, same emission order, and the only difference is whether the
    borders that group them into stages are there. Without it, "no warp pipeline" needs a
    second copy of the schedule, and the two drift.

    ``__triton_builtin__`` is what makes the class reachable from a ``@gluon.jit`` body;
    the frontend rejects any other non-constexpr global. It constructs every context
    manager with ``_semantic=`` injected, hence the ``**kwargs``, and passes ``label`` and
    ``priority`` already boxed as ``gl.constexpr`` -- all ignored here.
    """

    __triton_builtin__ = True
    __slots__ = ()

    def __init__(self, label=None, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@gluon.constexpr_function
def pick_warp_pipeline_stage(enabled):
    """The stage context manager to use: the real one, or :class:`nop_warp_pipeline_stage`.

    Written at the use site as ``with pick_warp_pipeline_stage(FLAG)("mfma", priority=0):``
    -- the frontend evaluates the picker at trace time and constructs whichever class it
    returned, so an off switch costs nothing at all in the emitted IR.
    """
    return warp_pipeline_stage if enabled else nop_warp_pipeline_stage
