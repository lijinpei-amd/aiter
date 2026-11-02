# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""In-kernel combine of GEMM2 rows: the last producer of a token's gates sums them.

GEMM2 writes one bf16 row per (token, gate) in expert-sorted order, and
``reduce_grouped`` then adds each token's ``topk`` rows in gate order in fp32. Under
:attr:`CombineMode.LAST_PRODUCER` the kernel does that sum itself, deterministically:

1. Every pane stores its gamma-weighted rows to the scratch buffer as before, with
   the same bf16 rounding, but write-through (``.wt``, sc0 sc1).
2. The wave waits until its stores complete (``s_waitcnt vmcnt(0)``). Then, for each
   row it stored, one of its lanes adds 1 to the counter of (token, output N slot)
   with a device-scope atomic. A completed write-through store is visible to every
   XCD, so the count never runs ahead of the rows.
3. The one producer whose add returns ``topk - 1`` reads the token's ``topk`` rows of
   that slot in ``scatter_indx`` order with uncached loads (``.cv``, sc0 sc1), adds
   them in fp32 from zero exactly as ``_reduce_grouped`` does, and stores the token
   row.

No CTA waits for another: a producer that is not last is done after its increment.
The counters and the inverse of ``scatter_indx`` are built per launch by
``_entry._combine_prep``, so a repeated call or a graph replay starts from zero.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

__all__ = ["_combine_rows"]


@gluon.jit
def _scratch_stores_done():
    """``s_waitcnt vmcnt(0)``: this wave's stores have completed. Returns 1.

    The scratch rows are stored write-through (``.wt``, system scope), so completion
    means every CU can read them. The generic agent-scope release fence would also
    write back and wait on the whole L2 (``buffer_wbl2 sc1``), which measured 16x
    slower here. The memory clobber and the returned value keep the counter
    increment, which adds it, behind the wait.
    """
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\n v_mov_b32 $0, 1",
        "=v,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _combine_rows(
    res,
    rt,
    cmb,
    block_id,
    pid_n,
    start_m,
    M_e,
    tuning_cfg,
    layout,
):
    """Publish this CTA's stored rows and finish the token rows it completes.

    Runs after ``_epilogue_store`` has stored every pane write-through. Each row's
    counter is incremented by one lane of the wave that stored the row
    (``combine_publish_layout``); ``validate_combine`` asserts that one wave stores a
    whole row. A token row is then summed by the producer whose increment was the
    last, reading every gate's row with system-coherent loads (``.cv``) issued after
    the count is known, so they cannot see stale lines.
    """
    BM: gl.constexpr = tuning_cfg.BLOCK_M
    SLOT_M: gl.constexpr = tuning_cfg.SLOT_M
    NM: gl.constexpr = layout.num_m_slots
    NN: gl.constexpr = layout.num_n_slots
    OUT_SLOT_N: gl.constexpr = layout.output_n_slot
    TOPK: gl.constexpr = rt.n_expts_act
    BITS: gl.constexpr = res.ptr.dtype.element_ty.primitive_bitwidth
    out_ty: gl.constexpr = cmb.out_ptr.dtype.element_ty
    SL: gl.constexpr = layout.result_store_layout(SLOT_M, OUT_SLOT_N, BITS)
    PL: gl.constexpr = layout.combine_publish_layout(SLOT_M, OUT_SLOT_N, BITS)
    LN: gl.constexpr = layout.combine_publish_lanes(SLOT_M, OUT_SLOT_N, BITS)
    ROWS: gl.constexpr = gl.SliceLayout(1, SL)

    one = _scratch_stores_done()
    owner = (gl.arange(0, LN, layout=gl.SliceLayout(0, PL)) == 0)[None, :]
    # All increments first: their round trips overlap each other and the gate-row
    # index loads below.
    toks = ()
    olds = ()
    for mi in gl.static_range(NM):
        lr = (
            BM * block_id
            + mi * SLOT_M
            + gl.arange(0, SLOT_M, layout=gl.SliceLayout(1, PL))
        )
        valid = lr < M_e
        tok = gl.load(cmb.row_gate + start_m + lr, mask=valid, other=0) // TOPK
        toks = toks + (gl.convert_layout(tok, ROWS, assert_trivial=True),)
        for ni in gl.static_range(NN):
            zero = gl.zeros([SLOT_M, LN], gl.int32, PL)
            old = gl.atomic_add(
                cmb.counters + (tok * cmb.n_slots + pid_n * NN + ni)[:, None] + zero,
                zero + one,
                mask=valid[:, None] & owner,
                sem="relaxed",
                scope="gpu",
            )
            # A masked lane's result is whatever its register held, often an earlier
            # add's count, so a padding row must not be able to look last.
            old = gl.max(gl.where(owner & valid[:, None], old, -1), axis=1)
            olds = olds + (gl.convert_layout(old, ROWS, assert_trivial=True),)

    sn = gl.arange(0, OUT_SLOT_N, layout=gl.SliceLayout(0, SL))
    for mi in gl.static_range(NM):
        tok = toks[mi]
        srcs = ()
        for g in gl.static_range(TOPK):
            srcs = srcs + (gl.load(rt.scatter_indx + tok * TOPK + g).to(gl.int32),)
        for ni in gl.static_range(NN):
            last = olds[mi * NN + ni] == TOPK - 1
            cols = (pid_n * NN + ni) * OUT_SLOT_N + sn
            acc = gl.zeros([SLOT_M, OUT_SLOT_N], gl.float32, SL)
            for g in gl.static_range(TOPK):
                v = gl.amd.cdna4.buffer_load(
                    res.ptr,
                    srcs[g][:, None] * res.stride_m + cols[None, :] * res.stride_n,
                    mask=last[:, None],
                    other=0.0,
                    cache=".cv",
                )
                # _reduce_grouped's per-gate step: curr = 0 + row, then acc += curr.
                curr = gl.zeros([SLOT_M, OUT_SLOT_N], gl.float32, SL) + v.to(gl.float32)
                acc = acc + curr
            gl.amd.cdna4.buffer_store(
                acc.to(out_ty),
                cmb.out_ptr,
                tok[:, None] * cmb.out_stride_m + cols[None, :] * cmb.out_stride_n,
                mask=last[:, None],
            )
