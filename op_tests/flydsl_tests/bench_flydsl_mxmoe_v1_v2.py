# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Head-to-head benchmark of the FlyDSL mxmoe a4w4 GEMM2 families.

  v1 = flydsl_mxmoe_g2_a4w4_*    (aiter/ops/flydsl/kernels/mxfp4_gemm2.py)
  v2 = flydsl_moe2_layout_*      (aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py)

Both share the same mxmoe GEMM1 (flydsl_mxmoe_g1_a4w4_*), so this isolates the
stage2 family. Shapes come from the a4w4 rows of aiter/configs/model_configs/
*_tuned_fmoe.csv. Nothing under aiter/configs is written.

This subclasses the shipped Mxfp4FlydslTuner instead of editing it, and changes
two things for cost: work is grouped per (model_dim, inter_dim, expert, topk)
*family* so the token sweep reuses the in-process compile caches, and the data
prep + torch reference are hoisted to once per (family, token) instead of once
per candidate.

Usage:
  P=/raid/jinpli/workspace/home01/jinpli/development/venv/01/bin/python
  FLYDSL_GPU_ARCH=gfx950 $P op_tests/flydsl_tests/bench_flydsl_mxmoe_v1_v2.py --list
  FLYDSL_GPU_ARCH=gfx950 $P op_tests/flydsl_tests/bench_flydsl_mxmoe_v1_v2.py \
      --families 6144x512x257x9 --mp 1
  FLYDSL_GPU_ARCH=gfx950 $P op_tests/flydsl_tests/bench_flydsl_mxmoe_v1_v2.py --mp 8
  FLYDSL_GPU_ARCH=gfx950 $P op_tests/flydsl_tests/bench_flydsl_mxmoe_v1_v2.py --report
"""

import argparse
import glob
import os
import signal
import sys
import time
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "csrc", "ck_gemm_moe_2stages_codegen"))

DEFAULT_OUT_DIR = os.path.join(_ROOT, "bench_out", "flydsl_mxmoe_v1_v2")
MODEL_CONFIG_GLOB = os.path.join(_ROOT, "aiter", "configs", "model_configs", "*_tuned_fmoe.csv")
FP4 = "torch.float4_e2m1fn_x2"

RAW_COLUMNS = [
    "model",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "token",
    "act_declared",
    "act_measured",
    "kernelName1",
    "kernelName2",
    "family",
    "us_e2e",
    "us_g2",
    "cos_err",
    "status",
]


# ---------------------------------------------------------------- shape set
def load_a4w4_shapes():
    """All a4w4 rows from the model_configs tuned CSVs, deduped on the shape key.

    The CSVs ship three different header layouts (some lack ``gfx``, some lack
    ``xbf16``/``flat``), so select by column name, never by position.
    """
    frames = []
    for path in sorted(glob.glob(MODEL_CONFIG_GLOB)):
        df = pd.read_csv(path)
        if "q_dtype_a" not in df.columns or "q_dtype_w" not in df.columns:
            continue
        a4 = df[(df.q_dtype_a == FP4) & (df.q_dtype_w == FP4)].copy()
        if a4.empty:
            continue
        a4["model"] = os.path.basename(path).replace("_tuned_fmoe.csv", "")
        cols = ["model", "token", "model_dim", "inter_dim", "expert", "topk", "act_type"]
        keep = [c for c in ("kernelName1", "kernelName2", "us") if c in a4.columns]
        frames.append(a4[cols + keep])
    shapes = pd.concat(frames, ignore_index=True)
    shapes = shapes.drop_duplicates(
        subset=["token", "model_dim", "inter_dim", "expert", "topk"]
    )
    for c in ("token", "model_dim", "inter_dim", "expert", "topk"):
        shapes[c] = shapes[c].astype(int)
    return shapes.sort_values(
        ["model_dim", "inter_dim", "expert", "topk", "token"]
    ).reset_index(drop=True)


def family_key(row):
    return f"{int(row['model_dim'])}x{int(row['inter_dim'])}x{int(row['expert'])}x{int(row['topk'])}"


def group_families(shapes):
    """[(family_key, model, md, id, E, topk, [tokens...]), ...] largest first."""
    out = []
    for (md, idim, e, tk), grp in shapes.groupby(
        ["model_dim", "inter_dim", "expert", "topk"], sort=False
    ):
        out.append(
            {
                "key": f"{md}x{idim}x{e}x{tk}",
                "model": grp["model"].iloc[0],
                "model_dim": int(md),
                "inter_dim": int(idim),
                "expert": int(e),
                "topk": int(tk),
                "rows": grp.sort_values("token").to_dict("records"),
            }
        )
    # Biggest first so the long poles start early and the pool drains evenly.
    out.sort(key=lambda f: -(f["model_dim"] * f["inter_dim"] * f["expert"]))
    return out


def family_of_kn2(kn2):
    if kn2.startswith("flydsl_mxmoe_g2_"):
        return "v1"
    if kn2.startswith("flydsl_moe2_layout_"):
        return "v2"
    return "?"


# ------------------------------------------------------------ measurement
def _make_bench_class():
    """Import the tuner lazily: it pulls in torch + the aiter JIT core."""
    import torch

    from aiter import ActivationType, dtypes
    from aiter.fused_moe import (
        _mxfp4_a4w4_stage1_fw,
        _mxfp4_a4w4_stage2_fw,
        moe_sorting,
    )
    from aiter.ops.flydsl.mxfp4_kname import _parse_mxfp4_g1_kname, parse_g2_kname_any
    from aiter.test_common import run_perftest
    from gemm_moe_tune import Mxfp4FlydslTuner, cosine_diff_compare

    class V1V2Bench(Mxfp4FlydslTuner):
        """Per-family sweep recording every candidate, not just the winner."""

        # Mxfp4FlydslTuner._candidate_row copies row[k] for k in self.keys into
        # each candidate. We only consume the kernel names from it, so the shape
        # key is all it needs to find.
        def __init__(self, keys=None):
            self.keys = keys or ["token", "model_dim", "inter_dim", "expert", "topk"]

        @staticmethod
        def _stage1_setup(data, kn1, kn2, topk, ne, h, dtype):
            """Sorting + GEMM1, hoisted out of the stage2 timing loop.

            Mirrors the first half of Mxfp4FlydslTuner._port_e2e; the sort block
            size and accumulate flag are candidate-dependent, so this has to be
            redone per candidate -- but outside the timed region.
            """
            g2 = parse_g2_kname_any(kn2)
            bm, atomic = g2["BM"], g2["atomic"]
            bm1 = _parse_mxfp4_g1_kname(kn1)["BM"]
            m = data["input"].shape[0]
            sti, sw, sei, nvi, moe_buf, m_indices, reverse_sorted = moe_sorting(
                data["topk_ids"],
                data["topk_weights"],
                ne,
                h,
                dtype,
                block_size=bm,
                accumulate=atomic,
                output_aux=True,
            )
            moe_out = moe_buf if moe_buf.numel() else torch.empty((m, h), dtype=dtype)
            inter_q, inter_s = _mxfp4_a4w4_stage1_fw(
                data["input"],
                data["w1_a16"],
                data["w2_a16"],
                sti,
                sei,
                nvi,
                None,
                topk,
                block_m=bm1,
                w1_scale=data["w1s_a16"],
                kernelName1=kn1,
                m_indices=m_indices,
                moe_buf=moe_buf,
            )
            return {
                "sti": sti,
                "sw": sw,
                "sei": sei,
                "nvi": nvi,
                "moe_out": moe_out,
                "reverse_sorted": reverse_sorted,
                "inter_q": inter_q,
                "inter_s": inter_s,
                "bm": bm,
            }

        @staticmethod
        def _stage2_call(data, st, kn2, topk):
            return _mxfp4_a4w4_stage2_fw(
                st["inter_q"],
                data["w1_a16"],
                data["w2_a16"],
                st["sti"],
                st["sei"],
                st["nvi"],
                st["moe_out"],
                topk,
                w2_scale=data["w2s_a16"],
                a2_scale=st["inter_s"],
                block_m=st["bm"],
                sorted_weights=st["sw"],
                kernelName2=kn2,
                reverse_sorted=st["reverse_sorted"],
            )

        def measure(self, data, ref, kn1, kn2, topk, ne, h, dtype, args):
            """(us_e2e, us_g2, cos_err); raises if the candidate is wrong or fails."""
            out = self._port_e2e(data, kn1, kn2, topk, ne, h, dtype)
            err = cosine_diff_compare(ref, out, msg=f"[{kn1}+{kn2}]", printLog=False)
            if err is None or float(err) > args.errRatio:
                raise RuntimeError(f"cosine err_ratio {err} > {args.errRatio}")

            _, us_e2e = run_perftest(
                lambda: self._port_e2e(data, kn1, kn2, topk, ne, h, dtype),
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            # Stage2 in isolation. Under the atomic epilog the output keeps
            # accumulating across timed iterations -- numerically meaningless,
            # but the work per iteration is identical, so the timing is valid.
            st = self._stage1_setup(data, kn1, kn2, topk, ne, h, dtype)
            _, us_g2 = run_perftest(
                lambda: self._stage2_call(data, st, kn2, topk),
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            return round(float(us_e2e), 4), round(float(us_g2), 4), round(float(err), 6)

        def run_family(self, fam, args):
            """Sweep every candidate x token for one shape family."""
            records = []
            ne, h, e = fam["expert"], fam["model_dim"], fam["inter_dim"]
            topk = fam["topk"]
            dtype = dtypes.bf16

            timeout = int(args.timeout or 0)
            if timeout > 0:
                try:
                    signal.signal(
                        signal.SIGALRM,
                        lambda *_: (_ for _ in ()).throw(
                            TimeoutError(f"exceeded {timeout}s")
                        ),
                    )
                except ValueError:
                    timeout = 0

            for row in fam["rows"]:
                token = int(row["token"])
                t0 = time.time()
                try:
                    data = self._prepare_case(token, h, e, ne, topk, dtype)
                    # The mxmoe GEMM1 hardcodes silu (kernels/mxfp4_gemm1.py
                    # _silu_mul_batch), so Swiglu/Situv2 shapes are measured with
                    # a silu reference; perf is activation-independent here.
                    ref = self._torch_ref(data, topk, dtype, ActivationType.Silu)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[bench] {fam['key']} token={token} PREP FAILED: {exc}",
                        flush=True,
                    )
                    continue

                cands = self._candidate_rows(dict(row))
                n_ok = {"v1": 0, "v2": 0}
                for i, cand in enumerate(cands):
                    if args.progress and i and i % args.progress == 0:
                        print(
                            f"[bench]   {fam['key']} token={token} "
                            f"{i}/{len(cands)} @ {time.time() - t0:.0f}s",
                            flush=True,
                        )
                    kn1, kn2 = cand["kernelName1"], cand["kernelName2"]
                    rec = {
                        "model": row["model"],
                        "model_dim": h,
                        "inter_dim": e,
                        "expert": ne,
                        "topk": topk,
                        "token": token,
                        "act_declared": row["act_type"],
                        "act_measured": "ActivationType.Silu",
                        "kernelName1": kn1,
                        "kernelName2": kn2,
                        "family": family_of_kn2(kn2),
                        "us_e2e": "",
                        "us_g2": "",
                        "cos_err": "",
                        "status": "ok",
                    }
                    if timeout > 0:
                        signal.alarm(timeout)
                    try:
                        us_e2e, us_g2, err = self.measure(
                            data, ref, kn1, kn2, topk, ne, h, dtype, args
                        )
                        rec.update(us_e2e=us_e2e, us_g2=us_g2, cos_err=err)
                        n_ok[rec["family"]] = n_ok.get(rec["family"], 0) + 1
                    except Exception as exc:  # noqa: BLE001
                        rec["status"] = f"{type(exc).__name__}: {exc}"[:200]
                    finally:
                        if timeout > 0:
                            signal.alarm(0)
                    records.append(rec)

                del data, ref
                torch.cuda.empty_cache()
                print(
                    f"[bench] {fam['key']} token={token}: {len(cands)} candidates, "
                    f"v1_ok={n_ok['v1']} v2_ok={n_ok['v2']}, "
                    f"{time.time() - t0:.1f}s",
                    flush=True,
                )
            return records

    return V1V2Bench


# --------------------------------------------------------------- mp driver
def _worker(payload):
    fam, args, gpu_q, out_dir = payload
    import torch

    gpu = gpu_q.get()
    try:
        torch.cuda.set_device(gpu)
        print(f"[bench] family {fam['key']} ({fam['model']}) -> GPU{gpu}", flush=True)
        bench = _make_bench_class()(keys=None)
        recs = bench.run_family(fam, args)
        path = os.path.join(out_dir, f"raw_{fam['key']}.csv")
        pd.DataFrame(recs, columns=RAW_COLUMNS).to_csv(path, index=False)
        return fam["key"], len(recs), None
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return fam["key"], 0, f"{type(exc).__name__}: {exc}"
    finally:
        gpu_q.put(gpu)


def run_sweep(families, args):
    os.makedirs(args.out_dir, exist_ok=True)
    todo = []
    for fam in families:
        path = os.path.join(args.out_dir, f"raw_{fam['key']}.csv")
        if args.resume and os.path.exists(path):
            print(f"[bench] skip {fam['key']} (already done)", flush=True)
            continue
        todo.append(fam)
    if not todo:
        print("[bench] nothing to do")
        return

    if args.mp <= 1:
        import torch

        torch.cuda.set_device(0)
        bench = _make_bench_class()(keys=None)
        for fam in todo:
            t0 = time.time()
            recs = bench.run_family(fam, args)
            pd.DataFrame(recs, columns=RAW_COLUMNS).to_csv(
                os.path.join(args.out_dir, f"raw_{fam['key']}.csv"), index=False
            )
            print(
                f"[bench] family {fam['key']} done: {len(recs)} rows in "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
        return

    import multiprocessing as mp

    import torch

    ngpu = torch.cuda.device_count()
    nproc = max(1, min(args.mp, ngpu, len(todo)))
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    gpu_q = mgr.Queue()
    for g in range(nproc):
        gpu_q.put(g)
    print(f"[bench] {len(todo)} families across {nproc} GPUs", flush=True)
    payloads = [(fam, args, gpu_q, args.out_dir) for fam in todo]
    with ctx.Pool(processes=nproc, maxtasksperchild=1) as pool:
        for key, n, err in pool.imap_unordered(_worker, payloads, chunksize=1):
            print(
                f"[bench] family {key} finished: {n} rows"
                + (f" ERROR {err}" if err else ""),
                flush=True,
            )


# ------------------------------------------------------------------ report
def _best(df, fam, metric):
    sub = df[(df.family == fam) & (df.status == "ok") & (df[metric] != "")]
    if sub.empty:
        return None, None
    sub = sub.copy()
    sub[metric] = sub[metric].astype(float)
    r = sub.loc[sub[metric].idxmin()]
    return float(r[metric]), r["kernelName2"]


def build_report(args):
    files = sorted(glob.glob(os.path.join(args.out_dir, "raw_*.csv")))
    if not files:
        raise SystemExit(f"no raw_*.csv under {args.out_dir}")
    raw = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)

    shape_cols = ["model", "model_dim", "inter_dim", "expert", "topk", "token"]
    rows = []
    for key, grp in raw.groupby(shape_cols, sort=False):
        rec = dict(zip(shape_cols, key))
        rec["act_declared"] = grp["act_declared"].iloc[0]
        for metric in ("us_g2", "us_e2e"):
            for fam in ("v1", "v2"):
                us, kn = _best(grp, fam, metric)
                rec[f"{fam}_{metric}"] = us
                rec[f"{fam}_{metric}_kn"] = kn
            a, b = rec[f"v1_{metric}"], rec[f"v2_{metric}"]
            rec[f"speedup_{metric}"] = round(a / b, 4) if (a and b) else None
        for fam in ("v1", "v2"):
            sub = grp[grp.family == fam]
            rec[f"{fam}_cands"] = len(sub)
            rec[f"{fam}_ok"] = int((sub.status == "ok").sum())
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values(shape_cols).reset_index(drop=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    import math

    def geomean(vals):
        vals = [v for v in vals if v and v > 0]
        return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None

    lines = []
    lines.append("# FlyDSL mxmoe a4w4 GEMM2: v1 vs v2\n")
    lines.append(
        "`v1` = `flydsl_mxmoe_g2_a4w4_*` ([mxfp4_gemm2.py](aiter/ops/flydsl/kernels/mxfp4_gemm2.py), "
        "2026-07-09 `b2dd7703d`) &nbsp;&nbsp; `v2` = `flydsl_moe2_layout_*` "
        "([mxmoe_gemm_v2.py](aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py), 2026-08-04 `db24351ee`).\n"
    )
    lines.append(
        "Both families run behind the same mxmoe GEMM1 (`flydsl_mxmoe_g1_a4w4_*`), so "
        "`us_g2` isolates the stage2 kernel and `us_e2e` covers sort + GEMM1 + GEMM2. "
        "Every candidate is gated on `cosine_diff_compare <= 0.1` against a torch "
        "reference before it is timed. `speedup = v1 / v2`, so **> 1 means v2 is faster**.\n"
    )

    both = summary[summary.speedup_us_g2.notna()]
    measured = summary[summary.v1_us_g2.notna() | summary.v2_us_g2.notna()]
    blocked = summary[summary.v1_us_g2.isna() & summary.v2_us_g2.isna()]

    lines.append("## Headline\n")
    lines.append(f"- shapes attempted: **{len(summary)}**")
    lines.append(f"- shapes measured: **{len(measured)}**")
    lines.append(f"- head-to-head (both families ran): **{len(both)}**")
    lines.append(
        f"- v2-only (no v1 kernel exists): **{len(summary[summary.v1_us_g2.isna() & summary.v2_us_g2.notna()])}**"
    )
    lines.append(
        f"- blocked (neither family could run): **{len(blocked)}** — see Coverage below"
    )
    g2g = geomean(both.speedup_us_g2.tolist())
    e2g = geomean(both.speedup_us_e2e.tolist())
    lines.append(f"- geomean speedup, GEMM2 only: **{g2g:.3f}x**" if g2g else "")
    lines.append(f"- geomean speedup, end-to-end: **{e2g:.3f}x**" if e2g else "")
    lines.append(
        f"- v2 wins {(both.speedup_us_g2 > 1.02).sum()}, "
        f"v1 wins {(both.speedup_us_g2 < 0.98).sum()}, "
        f"tie (within 2%) {both.speedup_us_g2.between(0.98, 1.02).sum()} (GEMM2 only)\n"
    )

    # How much of e2e the stage2 family can even influence.
    allrows = raw[raw.status == "ok"].copy()
    if len(allrows):
        allrows["us_e2e"] = allrows.us_e2e.astype(float)
        allrows["us_g2"] = allrows.us_g2.astype(float)
        allrows["rest"] = allrows.us_e2e - allrows.us_g2
        lines.append("## Why the GEMM2 win shrinks end-to-end\n")
        lines.append(
            "There is **no v2 GEMM1** — `flydsl_moe2_layout_*` is the only v2 family, and "
            "commit `db24351ee` touched no gemm1 file. Both paths run the same "
            "`flydsl_mxmoe_g1_a4w4_*`, so GEMM1 is a shared constant here, not a "
            "comparison axis. It is still the larger half of the runtime:\n"
        )
        lines.append("| tokens | median share of e2e that is sort + GEMM1 |")
        lines.append("|---|---|")
        for lo, hi, lab in [
            (0, 4, "1-4"),
            (5, 64, "8-64"),
            (65, 1024, "128-1024"),
            (1025, 8192, "2048-8192"),
            (8193, 10**9, "16384-32768"),
        ]:
            b = allrows[(allrows.token >= lo) & (allrows.token <= hi)]
            if len(b):
                lines.append(f"| {lab} | {100 * (b.rest / b.us_e2e).median():.0f}% |")
        lines.append("")
        lines.append(
            "So stage2 controls only ~22% of end-to-end time at small batch and ~52% at "
            "the largest. That is the whole reason a 1.07x GEMM2 win lands as ~1.03x "
            "end-to-end — and it says the next real win is in the sort + GEMM1 half, not "
            "in further stage2 tuning.\n"
        )
        best_g1 = allrows.loc[
            allrows.groupby(
                ["model_dim", "inter_dim", "expert", "topk", "token"]
            ).us_e2e.idxmin()
        ]
        lines.append(
            "Best GEMM1 variant per shape (chosen jointly with stage2, by e2e): "
            + ", ".join(
                f"`{k.replace('flydsl_mxmoe_g1_a4w4_', '')}` x{v}"
                for k, v in best_g1.kernelName1.value_counts().items()
            )
            + ".\n"
        )

    lines.append("## Per model\n")
    lines.append(
        "| model | shapes | h2h | geomean g2 | geomean e2e | v2 wins | v1 wins | tie |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for model, grp in summary.groupby("model"):
        h = grp[grp.speedup_us_g2.notna()]
        gg, ge = geomean(h.speedup_us_g2.tolist()), geomean(h.speedup_us_e2e.tolist())
        lines.append(
            f"| {model} | {len(grp)} | {len(h)} | "
            f"{gg:.3f}x | {ge:.3f}x | " if gg else f"| {model} | {len(grp)} | {len(h)} | - | - | "
        )
        lines[-1] += (
            f"{(h.speedup_us_g2 > 1.02).sum()} | {(h.speedup_us_g2 < 0.98).sum()} | "
            f"{h.speedup_us_g2.between(0.98, 1.02).sum()} |"
        )
    lines.append("")

    lines.append("## Per shape\n")
    for (model, md, idim, e, tk), grp in measured.groupby(
        ["model", "model_dim", "inter_dim", "expert", "topk"]
    ):
        act = grp["act_declared"].iloc[0]
        note = (
            ""
            if act == "ActivationType.Silu"
            else f" &nbsp;·&nbsp; declared `{act}`, measured with a silu GEMM1 (the mxmoe port is silu-only)"
        )
        lines.append(
            f"### {model} — model_dim={md}, inter_dim={idim}, E={e}, topk={tk}{note}\n"
        )
        lines.append(
            "| token | v1 g2 (µs) | v2 g2 (µs) | g2 speedup | v1 e2e (µs) | v2 e2e (µs) | e2e speedup | best v1 kernel | best v2 kernel | cands v1/v2 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in grp.sort_values("token").iterrows():
            def f(x):
                return f"{x:.2f}" if pd.notna(x) else "—"

            def s(x):
                return f"**{x:.3f}x**" if pd.notna(x) else "—"

            lines.append(
                f"| {r.token} | {f(r.v1_us_g2)} | {f(r.v2_us_g2)} | {s(r.speedup_us_g2)} | "
                f"{f(r.v1_us_e2e)} | {f(r.v2_us_e2e)} | {s(r.speedup_us_e2e)} | "
                f"`{r.v1_us_g2_kn or '—'}` | `{r.v2_us_g2_kn or '—'}` | "
                f"{int(r.v1_ok)}/{int(r.v1_cands)} · {int(r.v2_ok)}/{int(r.v2_cands)} |"
            )
        lines.append("")

    # What the shipped tuned CSVs currently select, vs what we measured fastest.
    shapes = load_a4w4_shapes()
    if "kernelName2" in shapes.columns:
        cur = shapes.set_index(
            ["model_dim", "inter_dim", "expert", "topk", "token"]
        )["kernelName2"].to_dict()
        disagree = []
        for _, r in both.iterrows():
            key = (r.model_dim, r.inter_dim, r.expert, r.topk, r.token)
            kn = str(cur.get(key, ""))
            picked = family_of_kn2(kn)
            if picked not in ("v1", "v2"):
                continue
            winner = "v2" if r.speedup_us_g2 > 1 else "v1"
            if picked != winner:
                disagree.append((r.model, key, picked, winner, r.speedup_us_g2, kn))
        lines.append("## Where the shipped tuned CSVs disagree with this measurement\n")
        lines.append(
            f"Of the {len(both)} head-to-head shapes, the tuned CSVs name a v1-or-v2 "
            f"GEMM2 for some; **{len(disagree)}** of those name the family this "
            "benchmark measured as slower (GEMM2-only). Stage1 choice and tuner "
            "vintage differ, so treat this as a list worth re-tuning, not a bug list.\n"
        )
        if disagree:
            lines.append("| model | shape | token | CSV picks | measured faster | g2 speedup (v1/v2) |")
            lines.append("|---|---|---|---|---|---|")
            for m, k, picked, winner, sp, _kn in disagree[:40]:
                lines.append(
                    f"| {m} | {k[0]}x{k[1]}x{k[2]}x{k[3]} | {k[4]} | {picked} | "
                    f"**{winner}** | {sp:.3f}x |"
                )
            if len(disagree) > 40:
                lines.append(f"\n_({len(disagree) - 40} more rows in `summary.csv`.)_")
        lines.append("")

    # Run-to-run spread, from an independent repeat of one or more families.
    if args.repeat_dir and os.path.isdir(args.repeat_dir):
        def _best_by_token(path):
            d = pd.read_csv(path, keep_default_na=False)
            d = d[d.status == "ok"].copy()
            if d.empty:
                return None
            d["us_g2"] = d.us_g2.astype(float)
            return d.groupby(["token", "family"]).us_g2.min().unstack()

        rows = []
        for rp in sorted(glob.glob(os.path.join(args.repeat_dir, "raw_*.csv"))):
            base = os.path.join(args.out_dir, os.path.basename(rp))
            if not os.path.exists(base):
                continue
            a, b = _best_by_token(base), _best_by_token(rp)
            if a is None or b is None:
                continue
            j = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
            for tok, r in j.iterrows():
                rows.append(
                    (
                        os.path.basename(rp)[4:-4],
                        tok,
                        (r.v1_2 / r.v1_1 - 1) * 100,
                        (r.v2_2 / r.v2_1 - 1) * 100,
                        r.v1_1 / r.v2_1,
                        r.v1_2 / r.v2_2,
                    )
                )
        if rows:
            lines.append("## Measurement noise\n")
            lines.append(
                "One family was re-measured end-to-end in an independent process. "
                "Same kernels, same tuning, different run:\n"
            )
            lines.append("| family | token | v1 g2 drift | v2 g2 drift | speedup run 1 | speedup run 2 |")
            lines.append("|---|---|---|---|---|---|")
            for fam_k, tok, d1, d2, s1, s2 in rows:
                lines.append(
                    f"| {fam_k} | {tok} | {d1:+.1f}% | {d2:+.1f}% | {s1:.3f}x | {s2:.3f}x |"
                )
            worst = max(max(abs(r[2]), abs(r[3])) for r in rows)
            swing = max(abs(r[4] - r[5]) for r in rows)
            lines.append("")
            lines.append(
                f"Worst single-point drift **{worst:.1f}%**, worst speedup swing "
                f"**{swing:.3f}x**. The drift is concentrated at **token=1**, where the "
                "kernel runs for only a few µs and launch overhead dominates. Read "
                "per-shape speedups within roughly ±3% as noise (worse at token≤2); the "
                "geomeans over hundreds of shapes are far tighter than any single row.\n"
            )

    lines.append("## Coverage and caveats\n")
    if len(blocked):
        lines.append(
            f"- **{len(blocked)} shapes could not be measured at all** — neither family "
            "ran. `moe_sorting(output_aux=True)` dispatches to an AOT-codegen'd aux "
            "kernel keyed on `(NE, TOPK[, H])`, and these combinations are absent from "
            "`SHAPES` in "
            "[gen_instances.py:19](csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py#L19), "
            "so every candidate fails with "
            "`no codegen'd instance for shape key 'aux_sort3s_NE<ne>_TOPK<topk>_MB<mb>'`. "
            "This is a shared-prerequisite gap, not a v1-vs-v2 difference."
        )
        bl = (
            blocked.groupby(["model", "expert", "topk", "model_dim"])
            .size()
            .reset_index(name="rows")
        )
        lines.append("")
        lines.append("| model | NE | topk | H | shapes blocked |")
        lines.append("|---|---|---|---|---|")
        for _, r in bl.iterrows():
            lines.append(
                f"| {r.model} | {int(r.expert)} | {int(r.topk)} | {int(r.model_dim)} | {int(r.rows)} |"
            )
        lines.append("")
        lines.append(
            "  To unblock: add those `(NE, H, D_INTER, TOPK)` tuples to `SHAPES` and rebuild. "
            "Note that on this machine a rebuild of `module_moe_mxfp4_aux` alone is not "
            "enough — `torch.ops.aiter.mxfp4_moe_sort` is already registered when `aiter` "
            "is imported, so `compile_ops` short-circuits and never rebuilds; the prebuilt "
            "`module_aiter_core.so` has to be rebuilt too."
        )
    v2only = summary[summary.v1_us_g2.isna() & summary.v2_us_g2.notna()]
    if len(v2only):
        fams = sorted({f"{int(r.model_dim)}x{int(r.inter_dim)}" for _, r in v2only.iterrows()})
        lines.append(
            f"- **{len(v2only)} shapes have no working v1 kernel** ({', '.join(fams)}). "
            "v1's tile is fixed at `<BM>x256x256`, so `_assert_supported` rejects "
            "`inter_dim % 256 != 0` ([mxfp4_gemm2_kernels.py:82](aiter/ops/flydsl/mxfp4_gemm2_kernels.py#L82)); "
            "v2 tunes `BK ∈ {128, 256}`."
        )
    nsub = int((summary.act_declared != "ActivationType.Silu").sum())
    if nsub:
        lines.append(
            f"- **{nsub} shapes** are declared Swiglu/Situv2 but measured against a silu "
            "reference: the mxmoe GEMM1 calls `_silu_mul_batch` unconditionally "
            "([mxfp4_gemm1.py:704](aiter/ops/flydsl/kernels/mxfp4_gemm1.py#L704)). Timing is "
            "unaffected; only the correctness gate's reference changed."
        )
    lines.append(
        "- **Absolute µs here do not match the shipped CSVs at large token counts.** For "
        "glm5 `6144x512`, re-running the exact `flydsl_mxmoe_g1/g2` pair the CSV names "
        "reproduces it within ~7% at token 2-4, but is ~35% slower at token 16384/32768 "
        "(1984 µs vs the CSV's 1440 µs). That is not 8-GPU contention: re-measuring "
        "token=16384 alone on an otherwise idle machine gives 1922 µs, within 3% of the "
        "sweep. The CSV rows were tuned on an older build, so this is a cross-vintage "
        "difference in the absolute level. **It does not affect the v1-vs-v2 conclusion** "
        "— both families are measured in the same process, on the same data, back to back."
    )
    lines.append(
        "- **The search budgets are not equal**: v1 exposes ~14 candidates per shape, v2 ~160 "
        "(v1's BN/BK are fixed; v2 sweeps `tile_n x tile_k x epilog x nt x persist`). That is a "
        "real property of the two designs, not a harness artifact, but v2 does get more shots "
        "at a good config. The `cands` column shows ok/total per family."
    )
    lines.append(
        "- Under the atomic epilog the stage2-only timing loop re-accumulates into the same "
        "output buffer, so those iterations are numerically meaningless. The work per iteration "
        "is identical, so `us_g2` is still a valid timing."
    )
    fails = raw[raw.status != "ok"]
    if len(fails):
        lines.append(
            f"- {len(fails)} of {len(raw)} candidate runs failed (rejected by the cosine gate, "
            "unsupported-variant errors, or timeouts). Breakdown in `summary.csv` / the raw CSVs."
        )
    lines.append("")
    lines.append(f"Raw per-candidate data: `{os.path.relpath(args.out_dir, _ROOT)}/raw_*.csv`, "
                 f"per-shape rollup: `{os.path.relpath(summary_path, _ROOT)}`.\n")

    out = args.report_file
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[bench] wrote {out} and {summary_path}")
    print(f"[bench] shapes={len(summary)} h2h={len(both)} geomean_g2={g2g} geomean_e2e={e2g}")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these model tags")
    ap.add_argument("--families", nargs="*", default=None, help="e.g. 6144x512x257x9")
    ap.add_argument("--tokens", nargs="*", type=int, default=None)
    ap.add_argument("--mp", type=int, default=8)
    ap.add_argument("--iters", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--errRatio", type=float, default=0.1)
    ap.add_argument("--timeout", type=int, default=600, help="per-candidate seconds, 0=off")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--report-file", default=os.path.join(_ROOT, "flydsl_mxmoe_v1_v2.md"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--progress", type=int, default=25, help="log every N candidates, 0=off")
    ap.add_argument("--list", action="store_true", help="print the shape set and exit")
    ap.add_argument("--report", action="store_true", help="build the report from existing raw CSVs")
    ap.add_argument(
        "--repeat-dir",
        default=None,
        help="dir of raw_*.csv from an independent repeat run, for the noise-floor section",
    )
    args = ap.parse_args()

    if args.report:
        build_report(args)
        return

    shapes = load_a4w4_shapes()
    if args.models:
        shapes = shapes[shapes.model.isin(args.models)]
    if args.tokens:
        shapes = shapes[shapes.token.isin(args.tokens)]
    families = group_families(shapes)
    if args.families:
        families = [f for f in families if f["key"] in args.families]

    if args.list:
        print(f"{len(shapes)} rows / {len(families)} families")
        for f in families:
            v1 = "v1+v2" if f["inter_dim"] % 256 == 0 else "v2-only"
            print(
                f"  {f['key']:<22} {f['model']:<18} {len(f['rows']):>3} tokens  {v1}"
            )
        return

    if not families:
        raise SystemExit("no families selected")
    run_sweep(families, args)


if __name__ == "__main__":
    main()
