# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed scales: the i32 layout of a shuffled E8M0 scale and its MFMA byte selectors.

A shuffled scale (``agents/CONCEPTS.md``, CDNA4 MX-scale order) is a run of *units*,
each 32 non-K rows by 256 payload K. A unit is 64 dwords, and dword ``l`` is what MFMA
lane ``l`` supplies: row ``l % 16`` and scale column ``l // 16`` of one K128 step. Its
four bytes are the two 16-row halves by the two K128 steps, byte
``2 * kstep + rowhalf``.

Viewed as i32, a ``nonk x k`` scale is ``[nonk // 32, k // 4]``: one row per 32-row
stripe, the stripe's units back to back along it, which is the HBM and LDS byte order
itself. :func:`packed_scale_layout` lays the dwords out so that lane ``l`` holds dword
``l`` of every unit it touches and the register and warp bases step whole units.

``gl.amd.cdna4.mfma_scaled_packed`` hands those dwords to the matrix instruction
untouched. Its lowering issues instruction ``i = non_k * num_rep_k + k`` with byte
``sel[i % len(sel)]`` of the scale tensor's register ``i // len(sel)``, and checks the
scale's shape as ``[nonk, k // 32 // len(sel)]``. :func:`packed_scale_sel` derives
``sel`` from the layouts, and :func:`packed_scale_dot_layout` gives the dot's view of
the same registers in the shape the instruction checks.

These belong beside ``get_mfma_scale_layout`` in ``gl.amd.cdna4``; they live here so
the Triton tree stays untouched.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._lang import MX_GROUP

__all__ = [
    "packed_scale_dot_layout",
    "packed_scale_dot_shape",
    "packed_scale_layout",
    "packed_scale_lds_view_layout",
    "packed_scale_sel",
    "packed_scale_shape",
]

#: Non-K rows of one unit.
UNIT_NONK = 32
#: Payload K of one unit.
UNIT_K = 256
#: Scale columns of one unit.
_UNIT_COLS = UNIT_K // MX_GROUP


@gluon.constexpr_function
def _byte(row, col):
    """Byte of scale ``(row, col)`` within its dword."""
    return 2 * (col % _UNIT_COLS // 4) + row % UNIT_NONK // 16


@gluon.constexpr_function
def _dword(row, col):
    """Dword of scale ``(row, col)`` within its unit: the lane that supplies it."""
    return col % 4 * 16 + row % 16


@gluon.constexpr_function
def _word(row, col):
    """Packed coordinate ``[stripe, dword]`` of the dword holding scale ``(row, col)``."""
    return [row // UNIT_NONK, col // _UNIT_COLS * 64 + _dword(row, col)]


@gluon.constexpr_function
def _span(bases, index):
    """The coordinate the set bits of ``index`` select from ``bases``."""
    out = [0, 0]
    for bit, basis in enumerate(bases):
        if index >> bit & 1:
            out = [out[0] ^ basis[0], out[1] ^ basis[1]]
    return out


@gluon.constexpr_function
def _unit_basis(basis):
    """A u8 basis that steps whole units, as a packed basis; ``None`` otherwise."""
    row, col = basis
    if row % UNIT_NONK or col % _UNIT_COLS:
        return None
    return [row // UNIT_NONK, col // _UNIT_COLS * 64]


@gluon.constexpr_function
def packed_scale_shape(nonk, k):
    """The i32 shape of a ``nonk x k`` shuffled scale, ``k`` in payload elements."""
    assert nonk % UNIT_NONK == 0 and k % UNIT_K == 0, (
        f"a packed scale holds whole {UNIT_NONK} x {UNIT_K} units, "
        f"got {nonk} x {k}"
    )
    return [nonk // UNIT_NONK, k // 4]


@gluon.constexpr_function
def packed_scale_layout(dot_operand_layout, nonk, k):
    """The i32 layout of a ``nonk x k`` shuffled scale for ``dot_operand_layout``.

    Derived from the u8 layout the MFMA takes the scale in, over
    :func:`packed_scale_shape`. Each lane basis must name one dword bit of a unit, and
    the two bases that cross a dword's bytes (``[16, 0]``, ``[0, 4]``) must be register
    bases: they fold into the dword. Every other register and warp basis must step
    whole units. ``None`` when the u8 layout breaks any of that, e.g. when a warp basis
    is ``[16, 0]``, which puts a dword's row halves in different warps.
    """
    u8 = gl.amd.cdna4.get_mfma_scale_layout(
        dot_operand_layout, [nonk, k // MX_GROUP], MX_GROUP
    )
    lane_bases = []
    for row, col in u8.lane_bases:
        if row >= UNIT_NONK or col >= _UNIT_COLS or _byte(row, col):
            return None
        lane_bases.append([0, _dword(row, col)])
    reg_bases, folded = [], []
    for basis in u8.reg_bases:
        row, col = basis
        if row < UNIT_NONK and col < _UNIT_COLS and not _dword(row, col):
            folded.append(_byte(row, col))
            continue
        packed = _unit_basis(basis)
        if packed is None:
            return None
        reg_bases.append(packed)
    if sorted(folded) != [1, 2]:
        return None
    warp_bases = []
    for basis in u8.warp_bases:
        packed = [0, 0] if list(basis) == [0, 0] else _unit_basis(basis)
        if packed is None:
            return None
        warp_bases.append(packed)
    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=packed_scale_shape(nonk, k),
    )


@gluon.constexpr_function
def packed_scale_lds_view_layout(nonk, k):
    """The LDS view of a packed scale: :func:`packed_scale_shape`, row-major i32."""
    stripes, dwords = packed_scale_shape(nonk, k)
    bases = [[0, 1 << bit] for bit in range(dwords.bit_length() - 1)]
    bases += [[1 << bit, 0] for bit in range(stripes.bit_length() - 1)]
    return gl.SharedLinearLayout(offset_bases=bases)


@gluon.constexpr_function
def packed_scale_sel(dot_operand_layout, nonk, k, k_offset=0):
    """Byte selectors for one ``nonk x k`` dot, or ``None`` if none exist.

    The dot reads the whole units that hold it, in :func:`packed_scale_layout`, and
    sits ``k_offset`` payload K into them: a K128 dot uses half of every dword, and
    ``k_offset`` says which half. Instruction ``i`` reads scale register ``i`` of the
    u8 layout, so it needs byte ``_byte`` of the packed register holding ``_word``.
    The lowering reads register ``i // len(sel)``, so the instructions one register
    serves must be consecutive and repeat the same bytes; otherwise, e.g. a K512 dot,
    whose consecutive instructions alternate K units, there is no selector list.
    """
    assert nonk % UNIT_NONK == 0, f"a packed dot spans whole 32-row stripes, got {nonk}"
    unit_k = max(k, UNIT_K)
    assert unit_k % k == 0 and k_offset % k == 0 and k_offset < unit_k
    packed = packed_scale_layout(dot_operand_layout, nonk, unit_k)
    if packed is None:
        return None
    u8 = gl.amd.cdna4.get_mfma_scale_layout(
        dot_operand_layout, [nonk, k // MX_GROUP], MX_GROUP
    )
    held = {}
    for reg in range(1 << len(packed.reg_bases)):
        held[tuple(_span(packed.reg_bases, reg))] = reg
    col0 = k_offset // MX_GROUP
    regs, sel = [], []
    for i in range(1 << len(u8.reg_bases)):
        row, col = _span(u8.reg_bases, i)
        regs.append(held[tuple(_word(row, col + col0))])
        sel.append(_byte(row, col + col0))
    period = regs.count(0)
    for i in range(len(regs)):
        if regs[i] != i // period or sel[i] != sel[i % period]:
            return None
    return sel[:period]


@gluon.constexpr_function
def packed_scale_dot_shape(nonk, k, sel):
    """The shape ``mfma_scaled_packed`` checks a ``nonk x k`` dot's packed scale in."""
    return [nonk, k // MX_GROUP // len(sel)]


@gluon.constexpr_function
def packed_scale_dot_layout(dot_operand_layout, nonk, k, sel):
    """:func:`packed_scale_layout` of the dot's units, reshaped to the dot's shape.

    A row-major reshape: the registers, and so the bytes ``sel`` names, do not move.
    """
    unit_k = max(k, UNIT_K)
    packed = packed_scale_layout(dot_operand_layout, nonk, unit_k)
    shape = packed_scale_dot_shape(nonk, k, sel)
    stripes, old_cols = packed_scale_shape(nonk, unit_k)
    assert stripes * old_cols == shape[0] * shape[1], (nonk, k, sel)

    def reshape(bases):
        out = []
        for row, col in bases:
            flat = row * old_cols + col
            out.append([flat // shape[1], flat % shape[1]])
        return out

    return gl.DistributedLinearLayout(
        reg_bases=reshape(packed.reg_bases),
        lane_bases=reshape(packed.lane_bases),
        warp_bases=reshape(packed.warp_bases),
        block_bases=[],
        shape=shape,
    )
