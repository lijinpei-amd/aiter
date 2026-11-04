# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""reduce_grouped sums the rows of each group in index order, in fp32, then casts.

The top-k combine of a grouped GEMM takes its launch tile from the MOE-REDUCE config
family; every other reduction keeps the default tile. Both tiles must give the sum
of the ordered fp32 additions bit for bit.
"""

import pytest
import torch

from aiter.ops.triton.moe import reduce as reduce_mod
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch


def _ordered_sum(rows):
    """fp32 sum of ``rows`` (a list of bf16 tensors) in list order, cast to bf16."""
    acc = torch.zeros(rows[0].shape, dtype=torch.float32, device=rows[0].device)
    for r in rows:
        acc += r.float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("n", [7168, 1536])
@pytest.mark.parametrize("tuned", [True, False])
def test_reduce_grouped_topk_combine_is_the_ordered_sum(n, tuned, monkeypatch):
    groups, k = 256, 6
    if not tuned:
        monkeypatch.setattr(
            reduce_mod, "_reduce_tile", lambda _: dict(reduce_mod._REDUCE_TILE_DEFAULT)
        )
    x = torch.randn((1, groups * k, n), dtype=torch.bfloat16, device="cuda")
    indx = torch.randperm(groups * k, device="cuda").to(torch.int32).view(groups, k)
    out = torch.empty((groups, n), dtype=torch.bfloat16, device="cuda")
    reduce_grouped(x, indx, out)
    ref = _ordered_sum([x[0][indx[:, i].long()] for i in range(k)])
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_reduce_grouped_split_k_is_the_ordered_sum():
    b, m, n = 3, 64, 2048
    x = torch.randn((b, m, n), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    reduce_grouped(x, None, out)
    ref = _ordered_sum([x[i] for i in range(b)])
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_reduce_tile_tunes_only_the_topk_combine():
    default = reduce_mod._REDUCE_TILE_DEFAULT
    assert reduce_mod._reduce_tile(False) == default
    tuned = reduce_mod._reduce_tile(True)
    if get_arch() == "gfx950":
        assert tuned == {"BLOCK_N": 1024, "num_warps": 2, "X_CACHE_MODIFIER": ".cg"}
    else:
        assert set(tuned) == set(default)
