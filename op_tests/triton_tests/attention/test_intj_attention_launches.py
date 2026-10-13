"""Direct attention launches use intj while preserving their outputs."""

import itertools

import pytest
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
    scores.masked_fill_(
        torch.triu(torch.ones_like(scores, dtype=torch.bool), 1), float("-inf")
    )
    expected = scores.softmax(-1) @ v[:, 0].float()
    torch.testing.assert_close(out[:, 0].float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_heads, num_kv_splits", [(16, 4), (128, 8)])
def test_mla_stage2_merge_launcher_matches_torch(num_heads, num_kv_splits):
    """aiter.mla's decorated stage-2 merge, called as mla_decode_fwd_v4_nm calls it.

    The kernel is pure Triton, so this runs on any target; the callers' stage 1
    is HIP assembly. The three sequences merge N, N - 1 (KV length) and 1
    (valid_split_count) splits, and stale split slots hold NaN.
    """
    from aiter.mla import _fwd_kernel_stage2_asm
    from aiter.ops.triton.utils.device_info import current_device_stream

    num_seqs, n, h, lv, mgc = 3, num_kv_splits, num_heads, 512, 16
    kv_lens = [(n - b) * mgc - b for b in range(num_seqs)]
    valid = [n, n, 1]
    num_valid = [min(n, -(-kv_len // mgc), v) for kv_len, v in zip(kv_lens, valid)]

    logits = torch.randn((num_seqs, n, h, lv), device="cuda")
    attn_lse = torch.randn((num_seqs, n, h, 1), device="cuda") * 2
    for b, nv in enumerate(num_valid):
        logits[b, nv:] = float("nan")
        attn_lse[b, nv:] = float("nan")
    out = torch.empty((num_seqs, h, lv), device="cuda", dtype=torch.bfloat16)
    final_lse_buf = torch.empty((1,), device="cuda")
    i32 = {"device": "cuda", "dtype": torch.int32}
    qo_indptr = torch.arange(num_seqs + 1, **i32)
    kv_indptr = torch.tensor([0, *itertools.accumulate(kv_lens)], **i32)
    kv_last_page_lens = torch.ones(num_seqs, **i32)
    split_indptr = torch.arange(num_seqs + 1, **i32) * n
    valid_split_count = torch.tensor(valid, **i32)

    dev, stream = current_device_stream()
    _fwd_kernel_stage2_asm(
        dev,
        stream,
        num_seqs,
        h,
        logits,
        attn_lse,
        out,
        final_lse_buf,
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        split_indptr,
        valid_split_count,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        out.stride(0),
        out.stride(1),
        0,  # stride_lse_bs
        1,  # page_size
        False,  # KV_INDPTR_IS_PAGE_LEVEL
        True,  # MAYBE_FINAL_OUT
        False,  # HAS_FINAL_LSE
        int(n > 1),  # USE_VALID_SPLIT_COUNT_REDUCE
        num_seqs,  # BATCH_NUM
        lv,  # BLOCK_DV
        lv,  # Lv
        mgc,
    )

    expected = torch.stack(
        [
            (attn_lse[b, :nv].softmax(0) * logits[b, :nv]).sum(0)
            for b, nv in enumerate(num_valid)
        ]
    )
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


def test_mla_aot_tools_load_the_stage2_jit_function():
    """triton.tools.compile takes `_fwd_kernel_stage2_asm` from mla_jit.py as
    a JITFunction, and the loader finds the HSACO symbol by that name."""
    from triton.runtime.jit import JITFunction

    import aiter.mla
    from aiter.aot.triton.mla_jit import _fwd_kernel_stage2_asm

    assert isinstance(_fwd_kernel_stage2_asm, JITFunction)
    assert _fwd_kernel_stage2_asm.__name__ == "_fwd_kernel_stage2_asm"
    launcher = aiter.mla._fwd_kernel_stage2_asm
    assert _fwd_kernel_stage2_asm is launcher.__self__.options.jit_func
