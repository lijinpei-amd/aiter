"""Determinism of the MXFP4-out epilogue at the benchmarked shape.

``corr_probe.py`` only covers the bf16 epilogue. The fused-quant epilogue is a
different code path -- LDS staging of the fp4 payload, the hardware
``v_cvt_scalef32_pk_fp4_f32`` pack and the E8M0 scale reduction -- and it is the arm
``cold_bench --target g4q`` times, so it needs its own bit-exactness check. Two output
tensors (packed payload and E8M0 scales) are hashed together.

    python tests/probes/corr_probe_q.py 30

Prints a sha over both outputs; a pure refactor must keep it at the recorded value.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[8]))

import torch

from aiter.ops.triton.moe.moe_op_gemm_gluon import (
    moe_gemm1_a4w4_mxfp4_out,
)
from op_tests.op_benchmarks.triton.bench_moe_gemm_gluon import (
    _OPS,
    _build,
    _explicit_recipe,
)

H, I, E, TOPK, T = 7168, 2048, 33, 8, 4096


def build():
    import os

    torch.manual_seed(1234)
    recipe = _explicit_recipe(H, I, E, TOPK)
    _w, xd, wd = _OPS["a4w4"]
    sh = recipe.gemm_shape(1, T)
    rdata, gindx, _si, x, xs, w, ws, _bias, _gammas = _build(
        T, sh.n, sh.k, sh.n_expts_tot, sh.n_expts_act, "cuda", xd, wd, n_active=E
    )
    # GU_SPLIT is an operand-layout contract, not an internal option: the caller must
    # hand over [gate | up] halves. Permute here so the probe exercises the same thing
    # cold_bench does -- and note the sha must come out UNCHANGED, because the emitted
    # output order is identical either way. A differing sha means the permutation and
    # the kernel disagree, which is exactly what this is here to catch.
    if os.environ.get("AITER_TRITON_MOE_GLUON_GU_SPLIT", "0") != "0":
        from aiter.ops.triton._triton_kernels.moe.activations import gate_up_split_perm

        perm = gate_up_split_perm(w.shape[-1]).to(w.device)

        def keep_layout(t):
            out = torch.empty_strided(
                t.shape, t.stride(), dtype=t.dtype, device=t.device
            )
            out.copy_(t[..., perm])
            return out

        w, ws = keep_layout(w), keep_layout(ws)

    def call():
        return moe_gemm1_a4w4_mxfp4_out(
            x,
            w,
            xs,
            ws,
            None,
            rdata,
            gindx,
            None,
            alpha=recipe.swiglu.alpha,
            limit=recipe.swiglu.limit,
        )

    return call


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def as_list(x):
    return list(x) if isinstance(x, (tuple, list)) else [x]


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    call = build()

    ref = [t.clone() for t in as_list(call())]
    torch.cuda.synchronize()
    print(
        f"outputs={len(ref)} shapes={[tuple(t.shape) for t in ref]}"
        f" sha={digest(ref)}",
        flush=True,
    )

    bad = 0
    for i in range(reps):
        got = as_list(call())
        torch.cuda.synchronize()
        if any(not torch.equal(a, b) for a, b in zip(got, ref)):
            bad += 1
            print(f"  rep {i:3d} MISMATCH sha={digest(got)}")
    print(
        f"determinism(mxfp4-out): {reps - bad}/{reps} bit-identical"
        f"  -> {'PASS' if not bad else 'FAIL'}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
