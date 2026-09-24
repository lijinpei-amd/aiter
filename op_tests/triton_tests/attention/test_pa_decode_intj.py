"""Paged attention uses intj for direct JIT launches."""

import pytest
import torch
import triton.language as tl

from aiter.ops.triton.attention.pa_decode import paged_attention_decode


@pytest.mark.parametrize("max_seq_len", [1, 2048])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("per_token", [False, True])
def test_single_token_uses_intj_and_matches_value(monkeypatch, max_seq_len, heads, per_token):
    from triton.runtime.jit import JITFunction

    def refuse_bracket_launch(self, grid):
        raise AssertionError("paged attention used Triton's bracket launcher")

    monkeypatch.setattr(JITFunction, "__getitem__", refuse_bracket_launch)
    query = torch.randn((1, heads, 64), device="cuda", dtype=torch.float16)
    key_cache = torch.randn((1, 1, 16, 64), device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    output = torch.empty_like(query)
    seq_lens = torch.tensor([1], device="cuda", dtype=torch.int32)
    tables = torch.zeros((1, (max_seq_len + 15) // 16), device="cuda", dtype=torch.int32)
    one = torch.ones((1, 1, 16), device="cuda") if per_token else torch.tensor([1.0])

    paged_attention_decode(
        output, query, key_cache, value_cache, seq_lens, tables,
        1 / 8, max_seq_len, tl.float16, one, one,
    )
    torch.testing.assert_close(output[0], value_cache[0, 0, 0].expand_as(output[0]),
                               rtol=1e-2, atol=1e-2)
