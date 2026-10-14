# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hash the generated assembly of every gemm1 kernel the dispatcher can select.

Vendored copy of ``scripts/gluon_moe_asm_hash.py`` for the regression gate. The
twenty cells are frozen in ``cells.json`` (resolved from the 2026-09-09 dtype/token
harness), and inputs come from ``../harness/bench_common.py``.

``gluon_moe_gemm1_check.py`` compares two tuned builds. That is the arm whose
performance is recorded, but it is two kernels out of the twenty the dtype/token
matrix actually compiles, so a scheduling change could move one of the other
eighteen and leave the check green.

This compiles all twenty and prints one md5 per cell over the stripped instruction
lines. Byte-identical output means the machine code did not move, which is a sharper
and much cheaper statement than a cold median -- those drift ~7 us between sessions
with no code change.

    python tests/asm_hash/asm_hash.py > before.txt
    ...edit...
    python tests/asm_hash/asm_hash.py > after.txt
    diff before.txt after.txt && echo "CODEGEN UNCHANGED"
    python tests/asm_hash/asm_hash.py a4w4:16 a8w4:256   # a subset of the cells
    python tests/asm_hash/asm_hash.py -j 20              # all twenty at once

Compile only: no launch, no timing, so it does not need an idle GPU. Each cell is its
own process with its own Triton cache, so cells compile concurrently: ``-j`` (or
``ASM_HASH_JOBS``) at a time, 4 by default. Each one holds ~5 GB of VRAM while it
compiles (its inputs), so 4 fit beside a nearly full GPU; ``-j 20`` needs ~95 GB free
and takes ~20 s instead of ~1 min. Rows print in the fixed cell order once every cell
is done, so the output does not depend on which finishes first.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[7]
sys.path.insert(0, str(REPO))

TOKENS = [(16, 32), (64, 32), (256, 64), (1024, 128), (4096, 128)]
DTYPES = ["a4w4", "a8w4", "a8w8", "a16w16"]
HARNESS = HERE.parent / "harness"
CELLS = HERE / "cells.json"
TIMEOUT = 1800  # seconds per cell
JOBS = 4  # concurrent cells; ~5 GB of VRAM each

WORKER = r"""
import json, sys, hashlib, glob, os
from pathlib import Path
sys.path.insert(0, {harness!r})
sys.path.insert(0, {repo!r})
import torch
import bench_common as bc
from aiter.ops.triton.moe import moe_op_gemm_gluon as host

case_name, tokens = sys.argv[1], int(sys.argv[2])
tuning = json.loads(Path({cells!r}).read_text())[case_name][str(tokens)]
# Routing block_m is the case's, not the router default: several cells pin a
# larger BLOCK_M than max(16, min(next_pow2(M // E), 128)) would pick.
bc.configure(7168, tuning["BLOCK_M"], tokens=tokens)
data = bc.make_inputs(case_name.split("_", 1)[1])
quantized = case_name.endswith("a4w4")
y = torch.empty((1, bc.T * bc.TOPK, bc.N // (4 if quantized else 2)),
                device="cuda", dtype=torch.uint8 if quantized else torch.bfloat16)
ys = (torch.empty((bc.T * bc.TOPK, bc.N // 64), device="cuda", dtype=torch.uint8)
      if quantized else None)
w = data["w"].transpose(-1, -2)
ws = None if data["ws"] is None else data["ws"].transpose(-1, -2)
original = host._fast_launch
host._fast_launch = lambda *a: None          # compile, never launch
try:
    host.moe_gemm_gluon(
        y, data["x"], w, data["xs"], ws, None, None, data["route"], data["gather"],
        None, bc.N, bc.K, True, 1.0, None, False, y_scales=ys, config=tuning,
        gate_up_split=True)
finally:
    host._fast_launch = original
hits = glob.glob(os.environ["TRITON_CACHE_DIR"] + "/**/_moe_gluon_gemm1.amdgcn",
                 recursive=True)
lines = [l.rstrip() for l in Path(hits[0]).read_text().splitlines()
         if l.startswith("\t") and not l.lstrip().startswith(".")]
print("ASM", hashlib.md5("\n".join(lines).encode()).hexdigest()[:12], len(lines))
"""


def hash_cell(worker, dtype, tokens):
    """Compile one cell in a fresh process and cache; return its output row."""
    cache = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(REPO),
        TRITON_CACHE_DIR=str(cache),
        AITER_TRITON_USE_HERD="0",
        TRITON_AMD_NT_ONLY="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for name in list(env):
        if name.startswith("AITER_TRITON_MOE_GLUON_"):
            env.pop(name)
    try:
        proc = subprocess.run(
            [sys.executable, str(worker), f"gluon_{dtype}", str(tokens)],
            env=env,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"{dtype:7s} t{tokens:<5d} FAILED timeout after {TIMEOUT} s"
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    found = re.search(r"^ASM ([0-9a-f]{12}) (\d+)$", proc.stdout, re.MULTILINE)
    if found:
        return f"{dtype:7s} t{tokens:<5d} md5={found.group(1)} instrs={found.group(2)}"
    tail = (proc.stderr.strip().splitlines() or ["?"])[-1][:100]
    return f"{dtype:7s} t{tokens:<5d} FAILED {tail}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "cells", nargs="*", metavar="dtype:tokens", help="cells; all twenty by default"
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("ASM_HASH_JOBS") or JOBS),
        help=f"cells compiled at once (default: $ASM_HASH_JOBS or {JOBS})",
    )
    args = ap.parse_args()
    cells = [
        (dtype, tokens)
        for dtype in DTYPES
        for tokens, _ in TOKENS
        if not args.cells or f"{dtype}:{tokens}" in args.cells
    ]
    scratch = Path(tempfile.mkdtemp())
    worker = scratch / "w.py"
    worker.write_text(
        WORKER.format(harness=str(HARNESS), repo=str(REPO), cells=str(CELLS))
    )
    try:
        # Threads only wait on the per-cell processes; map keeps the cell order.
        with ThreadPoolExecutor(max(1, args.jobs)) as pool:
            rows = list(pool.map(lambda cell: hash_cell(worker, *cell), cells))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
