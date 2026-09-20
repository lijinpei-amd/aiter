# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hash the generated assembly of every gemm1 kernel the dispatcher can select.

``gluon_moe_gemm1_check.py`` compares two tuned builds. That is the arm whose
performance is recorded, but it is two kernels out of the twenty the dtype/token
matrix actually compiles, so a scheduling change could move one of the other
eighteen and leave the check green.

This compiles all twenty and prints one md5 per cell over the stripped instruction
lines. Byte-identical output means the machine code did not move, which is a sharper
and much cheaper statement than a cold median -- those drift ~7 us between sessions
with no code change.

    python scripts/gluon_moe_asm_hash.py > before.txt
    ...edit...
    python scripts/gluon_moe_asm_hash.py > after.txt
    diff before.txt after.txt && echo "CODEGEN UNCHANGED"

Compile only: no launch, no timing, so it does not need an idle GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TOKENS = [(16, 32), (64, 32), (256, 64), (1024, 128), (4096, 128)]
DTYPES = ["a4w4", "a8w4", "a8w8", "a16w16"]
HARNESS = REPO / "bench_out/gluon_flydsl_dtype_tokens_20260909"

WORKER = r'''
import json, sys, hashlib, glob, os
from pathlib import Path
sys.path.insert(0, {harness!r})
sys.path.insert(0, {repo!r})
import torch
import bench_common as bc
from case_config import load_cases
from aiter.ops.triton.moe import moe_op_gemm_gluon as host

case_name, tokens = sys.argv[1], int(sys.argv[2])
tuning = load_cases(tokens)[case_name]["tuning"]
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
'''


def main():
    worker = Path(tempfile.mkdtemp()) / "w.py"
    worker.write_text(WORKER.format(harness=str(HARNESS), repo=str(REPO)))
    rows = []
    for dtype in DTYPES:
        for tokens, _ in TOKENS:
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
            proc = subprocess.run(
                [sys.executable, str(worker), f"gluon_{dtype}", str(tokens)],
                env=env, cwd=str(REPO), capture_output=True, text=True, timeout=1800,
            )
            found = re.search(r"^ASM ([0-9a-f]{12}) (\d+)$", proc.stdout, re.M)
            if found:
                rows.append(f"{dtype:7s} t{tokens:<5d} md5={found.group(1)} "
                            f"instrs={found.group(2)}")
            else:
                tail = (proc.stderr.strip().splitlines() or ["?"])[-1][:100]
                rows.append(f"{dtype:7s} t{tokens:<5d} FAILED {tail}")
            shutil.rmtree(cache, ignore_errors=True)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
