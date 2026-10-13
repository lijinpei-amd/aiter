"""Correctness probe at the *benchmarked* shape, which the bit-exact test misses.

test_gemm1_fused_mxfp4_out only parametrizes (1024,4096,4096,128,4),
(1024,6144,7168,256,8) and (16,4096,4096,128,4) -- none of them is
H=7168 I=2048 E=33 topk=8 T=4096. Since the MANUAL_PP race is nondeterministic and
only bit-exact comparisons catch it, this does two things at the traced shape:

  --mode determinism : run the kernel N times on identical inputs, require every
                       output to be bit-identical to the first.
  --mode dump        : write the output tensor so two configs can be compared.
                       WARP_PIPELINE only reorders independent instructions; it does
                       not change the MFMA accumulation order, so WP=0 and WP=1 must
                       agree bit-for-bit.
"""

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[8]
sys.path.insert(0, str(REPO))

H, I, E, TOPK, T = 7168, 2048, 33, 8, 4096


def build():
    import torch

    from aiter.ops.triton.moe.moe_op_gemm_gluon import moe_gemm_gluon
    from op_tests.op_benchmarks.triton.bench_moe_gemm_gluon import (
        _OPS,
        _build,
        _explicit_recipe,
    )

    torch.manual_seed(1234)
    recipe = _explicit_recipe(H, I, E, TOPK)
    _w, xd, wd = _OPS["a4w4"]
    sh = recipe.gemm_shape(1, T)
    n, k = sh.n, sh.k
    rdata, gindx, _si, x, xs, w, ws, bias, gammas = _build(
        T, n, k, sh.n_expts_tot, sh.n_expts_act, "cuda", xd, wd, n_active=E
    )
    M = gindx.shape[0]

    def call(y):
        moe_gemm_gluon(
            y,
            x,
            w,
            xs,
            ws,
            bias,
            gammas,
            rdata,
            gindx,
            None,
            n,
            k,
            True,
            recipe.swiglu.alpha,
            recipe.swiglu.limit,
            False,
        )
        return y

    return call, M, n


def digest(t):
    import torch

    return hashlib.sha256(t.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["determinism", "dump"], required=True)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default="")
    a = p.parse_args()
    import torch

    call, M, n = build()
    print(f"shape M={M} N={n} K={H}", flush=True)

    if a.mode == "dump":
        y = call(torch.empty((1, M, n // 2), dtype=torch.bfloat16, device="cuda"))
        torch.cuda.synchronize()
        torch.save(y.cpu(), a.out)
        print(f"wrote {a.out}  sha={digest(y)}")
        return 0

    ref = call(torch.empty((1, M, n // 2), dtype=torch.bfloat16, device="cuda"))
    torch.cuda.synchronize()
    ref = ref.clone()
    rd = digest(ref)
    print(f"reference sha={rd}")
    bad = 0
    for i in range(a.reps):
        # fresh NaN-filled buffer each time: a partial write would then be visible
        y = torch.full(
            (1, M, n // 2), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        call(y)
        torch.cuda.synchronize()
        if not torch.equal(y, ref):
            bad += 1
            diff = (y != ref).sum().item()
            print(f"  rep {i:3d} MISMATCH  {diff:,} elements  sha={digest(y)}")
    print(
        f"determinism: {a.reps - bad}/{a.reps} bit-identical"
        f"  -> {'PASS' if bad == 0 else 'FAIL'}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
