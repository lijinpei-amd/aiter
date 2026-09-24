"""Direct attention launches use intj while preserving their outputs."""

import torch

from aiter.ops.triton.attention.prefill_attention import context_attention_fwd


def test_prefill_uses_intj_and_matches_torch(monkeypatch):
    from triton.runtime.jit import JITFunction

    def refuse_bracket_launch(self, grid):
        raise AssertionError("prefill attention used Triton's bracket launcher")

    monkeypatch.setattr(JITFunction, "__getitem__", refuse_bracket_launch)
    q = torch.randn((8, 1, 32), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = torch.empty_like(q)
    starts = torch.tensor([0], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([8], device="cuda", dtype=torch.int32)

    context_attention_fwd(q, k, v, out, starts, lengths, 8)
    scores = q[:, 0].float() @ k[:, 0].float().T / 32**0.5
    scores.masked_fill_(torch.triu(torch.ones_like(scores, dtype=torch.bool), 1), float("-inf"))
    expected = scores.softmax(-1) @ v[:, 0].float()
    torch.testing.assert_close(out[:, 0].float(), expected, rtol=2e-2, atol=2e-2)
