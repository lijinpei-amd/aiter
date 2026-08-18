# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton/Gluon language plumbing shared by the gfx950 Gluon MoE modules.

Nothing here knows anything about MoE; it is the constexpr boxing and unboxing every
other module in the package needs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

__all__ = [
    "MX_GROUP",
    "MX_GROUP_CE",
    "WARP_SIZE",
    "const",
    "field_at",
    "optional",
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


@gluon.constexpr_function
def field_at(spec, i):
    """Index one field out of a plain-Python NamedTuple carried inside a constexpr."""
    spec = spec.value if isinstance(spec, gl.constexpr) else spec
    return spec[i]
