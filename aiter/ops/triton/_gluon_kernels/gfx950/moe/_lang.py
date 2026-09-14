# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton/Gluon constants and language plumbing shared by the gfx950 MoE modules."""

from enum import IntEnum

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage

from aiter.ops.triton.utils.common_utils import strip_annotate

__all__ = [
    "MX_GROUP",
    "MX_GROUP_CE",
    "WARP_SIZE",
    "WARP_SIZE_CE",
    "DotKind",
    "DtypeQuant",
    "EpilogueMode",
    "ScaleSwizzle",
    "SchedMode",
    "WaitCommitScheme",
    "WarpPipeline",
    "const",
    "constexpr_fields",
    "dq_has_scale",
    "dq_mx_format",
    "dq_pack_divisor",
    "dq_uses_mfma_scaled",
    "nop_warp_pipeline_stage",
    "optional",
    "pick_warp_pipeline_stage",
    "require_constexpr",
    "strip_annotate",
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


class DtypeQuant(IntEnum):
    """Tensor payload dtype together with its quantisation scheme."""

    BF16 = 0  # bf16, no scale
    FP8_E4M3 = 1  # fp8 e4m3, unit scales (folded into V_MFMA_*_F8F6F4)
    MXFP4 = 2  # E2M1, 2 per byte packed along K, uint8 E8M0 group 32 along K
    MXFP8 = 3  # E4M3, uint8 E8M0 group 32 along K


class ScaleSwizzle(IntEnum):
    NONE = 0
    # TODO name what the swizzle does, not the arch
    CDNA4_SCALE = 1  # utils/shuffle.py:_shuffle_scale_tile_gfx950 preshuffle
    # csrc/kernels/mxfp4_moe/moe_aux/moe_sort_scales.cuh, run per call rather than
    # offline: the token scales are gathered into routing order *and* permuted into the
    # MFMA scale-fragment order, so a stage's scales are a contiguous slice instead of
    # BLOCK_M rows of 8 bytes at a K/32 stride. Only meaningful for operand A, and only
    # for the layout the shuffle was written against -- see
    # KernelTuningConfig.sorted_shuffled_ok().
    SORTED_SHUFFLED = 2


class EpilogueMode(IntEnum):
    """Epilogue arithmetic, including shape-preserving benchmark ablations.

    Both NOP modes keep the gated reduction (gate * up), output quantisation and
    stores, so they preserve the output shape and traffic. NOP_ACTIVATION omits the
    activation function; NOP also omits bias and gammas. Operand dequantisation,
    including a per-tensor activation scale, still applies in every mode.
    """

    DEFAULT = 0
    NOP_ACTIVATION = 1
    NOP = 2


class SchedMode(IntEnum):
    """Backend scheduling hint for a K stage without compiler warp pipelining."""

    NONE = 0
    IGLP_0 = 1
    IGLP_1 = 2
    MFMA_16 = 3  # 4 x (16 MFMA, 6 LDS reads, 4 VMEM operations)
    MFMA_8 = 4  # 8 x (8 MFMA, 3 LDS reads, 2 VMEM operations)


class DotKind(IntEnum):
    """Which CDNA4 matrix instruction an operand pair maps onto."""

    MFMA = 0  # bf16 x bf16
    MFMA_SCALED = 1  # any fp4/fp8 pair, incl. FP8 x FP8 with unit scales
    UPCAST_MFMA = 2  # bf16 x microscaled: scaled_upcast, then plain mfma


class WarpPipeline(IntEnum):
    """Which inter-wave ping-pong the K-loop step is built for.

    ``NONE`` and ``COMPILER`` use the live component-pipeline driver. The live step
    places MFMA and memory work in regions selected by
    ``_lang.pick_warp_pipeline_stage``; the separate frozen body retains the
    reference's manual rendezvous sequence.

    * ``NONE`` -- no borders. One wave group, the MFMAs and the memory work overlapped
      only by the machine scheduler.
    * ``COMPILER`` -- hand the ``mfma``/``mem`` halves to
      ``TritonAMDGPUWarpPipeline``, which turns the interleave into a two-wave-group
      ping-pong. Needs ``VGPR_PREFETCH_K == BLOCK_K`` and only applies inside the
      ``tl.range`` body.
    * ``MANUAL`` -- the hand-emitted rendezvous instead of the pass. It is implemented
      only by the frozen pipeline.
    """

    NONE = 0
    COMPILER = 1
    MANUAL = 2


class WaitCommitScheme(IntEnum):
    """Commit granularity and wait placement for the live HBM-to-LDS pipeline.

    ``PER_OP`` commits each asynchronous copy separately, ``PER_SLOT`` commits after
    each slot, and the per-stage modes commit once at their respective pipeline
    boundary. Direct-register loads contribute no asynchronous group.
    """

    PER_OP = 1
    PER_SLOT = 2
    # Preserve the whole-stage setting used by existing cold-bench recipes.
    PER_STAGE_WHOLE = 3
    PER_STAGE_WARP_PIPELINE = 4


@gluon.constexpr_function
def unwrap(x):
    """Unwrap a ``gl.constexpr``. Must be a ``constexpr_function``, not a plain one:
    Triton's aggregate hash walker rejects any bare callable an aggregate method
    references ("Unsupported function referenced"). Called from Python it returns the
    raw value, so ``list()``/indexing on the result still work."""
    return x.value if isinstance(x, gl.constexpr) else x


@gluon.constexpr_function
def dq_has_scale(dq):
    return unwrap(dq) in (int(DtypeQuant.MXFP4), int(DtypeQuant.MXFP8))


@gluon.constexpr_function
def dq_pack_divisor(dq):
    """Logical elements per stored container along K."""
    return 2 if unwrap(dq) == int(DtypeQuant.MXFP4) else 1


@gluon.constexpr_function
def dq_mx_format(dq):
    """The ``a_format`` / ``b_format`` string ``mfma_scaled`` wants."""
    dq = unwrap(dq)
    if dq == int(DtypeQuant.MXFP4):
        return "e2m1"
    if dq in (int(DtypeQuant.MXFP8), int(DtypeQuant.FP8_E4M3)):
        return "e4m3"
    return None


@gluon.constexpr_function
def dq_uses_mfma_scaled(dq_a, dq_b):
    """Whether the pair uses the CDNA4 f8f6f4 MFMA path.

    FP8 x FP8 must still use ``mfma_scaled`` with synthesized unit scales; only that
    path reaches the double-rate K=64/128 instructions. Only bf16 x bf16 uses plain
    ``mfma``.
    """
    return not (
        unwrap(dq_a) == int(DtypeQuant.BF16)
        and unwrap(dq_b) == int(DtypeQuant.BF16)
    )


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
