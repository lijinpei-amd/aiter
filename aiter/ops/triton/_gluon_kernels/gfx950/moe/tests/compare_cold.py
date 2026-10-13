"""Compare two cold-bench labels of the same harness, cell by cell.

    python compare_cold.py <label_before> <label_after> [--runs DIR]

A label is a directory under --runs (default $HARNESS_OUT, else $OUT/harness, as in
_env.sh; e.g. `cold_before`), the name given to cold_bench.sh (`before` for
`cold_before`), or the path of any cold run directory holding a summary.json.

Prints Gluon and FlyDSL medians, each backend's change, and Gluon's change net of
the FlyDSL control (FlyDSL code is untouched by a Gluon change, so its movement is
the session/device drift). Also prints the Gluon-vs-FlyDSL win/tie/loss count.
Only compare labels measured on the same physical GPU.
"""

import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("before")
ap.add_argument("after")
ap.add_argument("--runs", default=os.environ.get("HARNESS_OUT"))
args = ap.parse_args()
OUT = os.environ.get(
    "OUT", Path(__file__).resolve().parents[7] / "bench_out/gluon_moe_regression"
)
H = Path(args.runs or Path(OUT) / "harness")


def load(label):
    if (Path(label) / "summary.json").is_file():
        d = Path(label)
    else:
        d = H / label if (H / label).is_dir() else H / f"cold_{label}"
    s = json.loads((d / "summary.json").read_text())
    return {
        (v["precision"], v["tokens"], v["backend"]): v["median_us"] for v in s.values()
    }


a, b = load(args.before), load(args.after)
print(
    "| dtype | T | Gluon before | Gluon after | Δ | FlyDSL Δ | Gluon Δ net | Gluon/FlyDSL after |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|")
wtl, dg, df = [0, 0, 0], [], []
for p in ("a4w4", "a8w4", "a8w8", "a16w16"):
    for t in (16, 64, 256, 1024, 4096):
        g0, g1, f0, f1 = (
            a[p, t, "gluon"],
            b[p, t, "gluon"],
            a[p, t, "flydsl"],
            b[p, t, "flydsl"],
        )
        x = g1 / f1
        wtl[0 if x < 0.995 else 2 if x > 1.005 else 1] += 1
        dg.append((g1 / g0 - 1) * 100)
        df.append((f1 / f0 - 1) * 100)
        net = ((g1 / f1) / (g0 / f0) - 1) * 100
        print(
            f"| {p} | {t} | {g0:.1f} | {g1:.1f} | {dg[-1]:+.2f}% | {df[-1]:+.2f}% | {net:+.2f}% | {x:.3f} |"
        )
print(
    f"\nGluon {min(dg):+.2f}%..{max(dg):+.2f}%   FlyDSL control {min(df):+.2f}%..{max(df):+.2f}%"
)
print(f"Gluon vs FlyDSL (after): {wtl[0]} faster / {wtl[1]} tie / {wtl[2]} slower")
