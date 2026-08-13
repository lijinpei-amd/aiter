# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ISA assertions for the gfx950 Gluon MoE GEMM.

The three LDS goals -- 128-bit direct-to-LDS on the global->LDS copies, no
``convert_layout`` between LDS and the MFMA operands, maximum LDS load width -- are all
claims about *generated code*, and **no correctness test can detect them failing**. A
silent fall back from ``buffer_load_dwordx4 ... lds`` to a 32-bit copy, or an inserted
``convert_layout`` in the K loop, passes every one of the 16 model cases while losing
most of the performance. So they get asserted on the AMDGCN and the TTGIR directly,
mirroring the upstream pattern in ``python/test/gluon/test_core.py``.
"""

import re

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import mxfp4_quant
from aiter.ops.triton.moe.moe_op_gemm_a4w4_gluon import moe_gemm_a4w4_gluon
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch

_CASES = [
    # (m, n, k, n_expts_tot, n_expts_act, apply_swiglu) -- one per BLOCK_M regime
    pytest.param(16, 4096, 4096, 256, 8, False, id="bm16-decode"),
    pytest.param(1024, 4096, 4096, 128, 4, False, id="bm32"),
    pytest.param(4096, 6144, 4096, 128, 4, True, id="bm128-swiglu"),
]


def _build(m, n, k, n_expts_tot, n_expts_act, device="cuda"):
    torch.manual_seed(0)
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    rdata, gindx, _ = routing(logits, n_expts_act)
    rdata.gate_scal = None
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n_expts_tot, k, n), device=device, dtype=torch.bfloat16)
    bias = torch.randn((n_expts_tot, n), device=device, dtype=torch.float32)
    gammas = torch.rand((gindx.shape[0],), device=device, dtype=torch.float32)
    w, w_scale = downcast_to_mxfp(w, torch.uint8, axis=1)
    x, x_scale = mxfp4_quant(x)
    return rdata, gindx, x, x_scale, w, w_scale, bias, gammas


def _isa_blocks_with(asm: str, mnemonic: str):
    """Split AMDGCN into label-delimited basic blocks and return those containing
    ``mnemonic``. The K loop is exactly the block(s) that hold the MFMAs."""
    blocks, cur = [], []
    for line in asm.splitlines():
        if re.match(r"^\S+:\s*(;.*)?$", line) and not line.startswith("\t"):
            blocks.append("\n".join(cur))
            cur = []
        cur.append(line)
    blocks.append("\n".join(cur))
    return [b for b in blocks if mnemonic in b]


@pytest.mark.parametrize("m, n, k, n_expts_tot, n_expts_act, swiglu", _CASES)
def test_gluon_moe_isa(m, n, k, n_expts_tot, n_expts_act, swiglu, device="cuda"):
    if get_arch() != "gfx950":
        pytest.skip("Gluon MoE kernels are gfx950 only.")
    rdata, gindx, x, x_scale, w, w_scale, bias, gammas = _build(
        m, n, k, n_expts_tot, n_expts_act, device
    )
    M = gindx.shape[0]
    out_n = n // 2 if swiglu else n
    y = torch.empty((1, M, out_n), dtype=torch.bfloat16, device=device)
    pgm = moe_gemm_a4w4_gluon(
        y,
        x,
        w,
        x_scale,
        w_scale,
        bias,
        gammas,
        rdata,
        gindx,
        None,
        n,
        k,
        swiglu,
        1.0,
        None,
        False,
    )
    assert pgm is not None, "the wrapper must return the compiled kernel handle"
    amdgcn = pgm.asm["amdgcn"]
    ttgir = pgm.asm["ttgir"]

    # 1. the global->LDS copies must be 128-bit direct-to-LDS, not a 32-bit fallback
    #    and not a register round trip.
    assert re.search(r"buffer_load_dwordx4[^\n]*lds", amdgcn), (
        "no `buffer_load_dwordx4 ... lds` in the ISA: the direct-to-LDS copy fell back "
        "to a narrower width or to a register round trip"
    )

    # 2. the MFMA operands must come straight out of LDS in the dot-operand layout.
    #    Anything else shows up as a cross-lane shuffle in the loop body.
    mfma_blocks = _isa_blocks_with(amdgcn, "v_mfma")
    assert mfma_blocks, "no MFMA found in the ISA"
    for b in mfma_blocks:
        for bad in ("ds_bpermute", "v_permlane"):
            assert bad not in b, (
                f"{bad} inside a block containing v_mfma -- a convert_layout was "
                f"inserted between LDS and the dot operand"
            )

    # 3. no convert_layout inside the K loop at the TTGIR level either. The epilogue is
    #    allowed to convert (the store layout differs from the MFMA layout by design),
    #    so scope the check to the scf.for regions.
    depth, in_loop = 0, []
    for line in ttgir.splitlines():
        if "scf.for" in line:
            depth += 1
        if depth:
            in_loop.append(line)
        if depth and re.match(r"^\s*\}", line):
            depth -= 1
    loop_text = "\n".join(in_loop)
    assert (
        "ttg.convert_layout" not in loop_text
    ), "ttg.convert_layout inside the K loop:\n" + "\n".join(
        li for li in in_loop if "convert_layout" in li
    )
