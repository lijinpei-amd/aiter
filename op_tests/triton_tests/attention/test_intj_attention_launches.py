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


# pa_mqa_logits' launchers by (Preshuffle, VarCtxOpt, KV-padded) mode, each
# with the page (KVBlockSize) and bytes per token that reach it: 128 fp8
# values and a 4-byte scale make a 132-byte token, and a page's stride is
# padded (a multiple of 16) unless it is one such token or 8 of 133 bytes.
_PA_MQA_LOGITS_MODES = {
    (False, False, False): (1, 132),
    (False, False, True): (1, 144),
    (True, False, False): (8, 133),
    (True, False, True): (16, 132),
    (True, True, False): (8, 133),
    (True, True, True): (16, 132),
}


def _pa_mqa_logits():
    from aiter.ops.triton.attention import pa_mqa_logits

    if not pa_mqa_logits.enable_gluon_pa_mqa_logits:
        pytest.skip("pa_mqa_logits' Gluon kernels need gfx942, gfx950 or gfx1250")
    return pa_mqa_logits


def test_pa_mqa_logits_launchers_carry_their_modes_keywords():
    """Each launcher has its mode's JIT function and keywords.

    A kernel's mode is written by hand three times (the decorator's `delayed=`
    keywords, the JIT function recorded under it, the table key), and a
    launcher assuming another KV mode's facts can still compute the right
    logits, so the keywords are compared, not just the outputs.
    """
    from aiter.ops.triton.gluon import pa_mqa_logits as gluon

    launchers = _pa_mqa_logits()._GLUON_PAGED_MQA_LOGITS_LAUNCHERS
    assert sorted(launchers) == sorted(_PA_MQA_LOGITS_MODES)
    assert sorted(gluon._GLUON_JIT_FUNCTIONS) == sorted(
        {mode[:2] for mode in _PA_MQA_LOGITS_MODES}
    )
    keywords = {mode: gluon._gluon_launch_keywords(*mode) for mode in launchers}
    for a, b in itertools.combinations(launchers, 2):
        assert keywords[a] != keywords[b], (a, b)
    for mode, launcher in launchers.items():
        launcher.__self__.check()
        options = launcher.__self__.options
        assert options.jit_func is gluon._GLUON_JIT_FUNCTIONS[mode[:2]], mode
        assert options.extra_annotation == keywords[mode]["extra_annotation"], mode
        assert options.options == keywords[mode]["options"], mode


def _mode_id(mode):
    return (
        "-".join(
            name for name, on in zip(("preshuffle", "varctx", "padded"), mode) if on
        )
        or "plain"
    )


def _launch_pa_mqa_logits_mode(mode, monkeypatch):
    """Runs deepgemm_fp8_paged_mqa_logits once on `mode`'s inputs (2 sequences
    of 2 tokens, 64 heads, the mode's page and token size). Returns the
    launches it made, as (launcher key, returned kernel), and the output with
    its torch reference."""
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.shuffle import shuffle_weight

    m = _pa_mqa_logits()
    if get_gfx() not in ("gfx942", "gfx950") or m.enable_aot_gluon_pa_mqa_logits:
        pytest.skip(
            "needs gfx942 or gfx950 in JIT mode (gfx1250 lacks VarCtx and pages under 16)"
        )
    preshuffle, varctx, _ = mode
    page, token_bytes = _PA_MQA_LOGITS_MODES[mode]
    hidden, heads, next_n, context_lengths = 128, 64, 2, (300, 77)
    # A multiple of 16: the VarCtx kernel stores logits up to a sequence's
    # length rounded up to 16, past the row if max_model_len is below that
    # (upstream too).
    batch, max_model_len = len(context_lengths), 320
    num_blocks = 2 * -(-max(context_lengths) // page)
    g = torch.Generator(device="cuda").manual_seed(4517)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    q = randn(batch, next_n, heads, hidden).to(dtypes.fp8)
    kv = randn(num_blocks, page, hidden).to(dtypes.fp8)
    scales = 0.25 + torch.rand(num_blocks, page, generator=g, device="cuda")
    weights = randn(batch * next_n, heads)
    packed = torch.zeros(
        num_blocks, page * token_bytes, dtype=torch.uint8, device="cuda"
    )
    values = shuffle_weight(kv, layout=(min(page, 16), 16)) if preshuffle else kv
    packed[:, : page * hidden] = values.reshape(num_blocks, -1).view(torch.uint8)
    packed[:, page * hidden : page * (hidden + 4)] = scales.view(torch.uint8)
    block_tables = torch.zeros(batch, num_blocks // 2, dtype=torch.int32, device="cuda")
    for b, length in enumerate(context_lengths):
        pages = -(-length // page)
        block_tables[b, :pages] = torch.randperm(
            num_blocks, generator=g, device="cuda"
        )[:pages]
    context_lens = torch.tensor(context_lengths, dtype=torch.int32, device="cuda")
    # Every pointer 16-byte aligned and under 2 GiB, so a fact the launcher
    # measured instead of stating would show in what it compiled.
    for t in (q, packed, weights, block_tables, context_lens):
        assert t.data_ptr() % 16 == 0 and t.untyped_storage().nbytes() < 2**31
    schedule = (
        m.deepgemm_fp8_paged_mqa_logits_schedule(
            batch, next_n, context_lens, max_model_len
        )
        if varctx
        else None
    )

    launched = []
    launchers = m._GLUON_PAGED_MQA_LOGITS_LAUNCHERS
    for key, launch in list(launchers.items()):

        def record(*args, key=key, launch=launch):
            kernel = launch(*args)
            launched.append((key, kernel))
            return kernel

        monkeypatch.setitem(launchers, key, record)
    out = torch.full((batch * next_n, max_model_len), float("-inf"), device="cuda")
    m.deepgemm_fp8_paged_mqa_logits(
        q,
        packed.view(num_blocks, page, 1, token_bytes),
        weights,
        out,
        context_lens,
        block_tables,
        max_model_len,
        Preshuffle=preshuffle,
        KVBlockSize=page,
        VarCtxSchedule=schedule,
    )

    reference = torch.full_like(out, float("-inf"))
    keys_all = kv.float() * scales[..., None]
    for b, length in enumerate(context_lengths):
        pages = block_tables[b, : -(-length // page)].long()
        keys = keys_all[pages].reshape(-1, hidden)[:length]
        rows = slice(b * next_n, (b + 1) * next_n)
        logits = ((q[b].float() @ keys.T).relu() * weights[rows, :, None]).sum(1)
        query_positions = length - next_n + torch.arange(next_n, device="cuda")
        future = torch.arange(length, device="cuda") > query_positions[:, None]
        reference[rows, :length] = logits.masked_fill(future, float("-inf"))
    return launched, out, reference


@pytest.mark.parametrize("mode", list(_PA_MQA_LOGITS_MODES), ids=_mode_id)
@torch.inference_mode()
def test_pa_mqa_logits_launches_each_mode_through_its_launcher(mode, monkeypatch):
    """deepgemm_fp8_paged_mqa_logits picks the mode's launcher and matches
    torch; the offset tests reach only three of the six modes."""
    launched, out, reference = _launch_pa_mqa_logits_mode(mode, monkeypatch)
    assert [key for key, _ in launched] == [mode]
    torch.testing.assert_close(out, reference, rtol=1e-2, atol=1e-2)


def _upstream_compile(mode, gfx):
    """The signature and the facts per parameter (nonempty ones only) of
    upstream's hand-written `triton.compile` (874d253dd3
    `_compile_deepgemm_fp8_paged_mqa_logits`, less the false 16-divisibility
    of `stride_scale_seq` dropped by the upstream fix
    fix/pa-mqa-logits-scale-stride-divisibility); its attrs are the same in
    every mode and arch but for the KV-padded three."""
    preshuffle, varctx, padded = mode
    fp8 = "*fp8e4b8" if gfx == "gfx942" else "*fp8e4nv"
    kv_stride = "i32" if preshuffle else "i64"
    signature = {
        "batch_size": "i32",
        "next_n": "i32",
        "heads_num": "i32",
        "Q_buffer": fp8,
        "stride_q_batch": "i32",
        "stride_q_next_n": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": fp8,
        "stride_k_seq": kv_stride,
        "scale_buffer": "*fp32",
        "stride_scale_seq": kv_stride,
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i64",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "num_block": "i32",
        **({"safe_chunks_per_cta_ptr": "*i32"} if varctx else {"SplitKV": "i32"}),
        **dict.fromkeys(
            ("ChunkQ", "ChunkK", "KVBlockSize", "HiddenDim", "CDNA_VERSION", "ARCH"),
            "constexpr",
        ),
    }
    d16, r32 = ("tt.divisibility", 16), ("tt.pointer_range", 32)
    kv = {d16} if padded else set()
    facts = {
        "heads_num": {d16},
        "Q_buffer": {d16, r32},
        "stride_q_batch": {d16},
        "stride_q_next_n": {d16},
        "stride_q_heads": {d16},
        "KV_buffer": kv,
        "stride_k_seq": kv,
        "scale_buffer": kv,
        "context_len_ptr": {r32},
        "kv_indices": {r32},
        "weights": {d16, r32},
        "stride_w_batch": {d16},
    }
    return signature, {name: f for name, f in facts.items() if f}


@pytest.mark.parametrize("mode", list(_PA_MQA_LOGITS_MODES), ids=_mode_id)
@torch.inference_mode()
def test_pa_mqa_logits_compiles_upstreams_facts(mode, monkeypatch):
    """Each launcher states exactly the facts upstream's hand-written attrs
    stated and compiles them under upstream's signature: the binary is
    upstream's.

    The keywords are checked as well as the compile: a fact left measured
    (`AUTO`, or an `Assume` short of upstream's list) compiles only for
    inputs that have it, and most of these integers are neither 1 nor
    multiples of 16. The compile shows the stated facts became upstream's
    attrs and no runtime parameter became a constexpr; with every pointer
    aligned and under 2 GiB, a measured pointer fact would show there too."""
    from intj import NEVER, Aligned, Exactly, PointerRange

    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.triton.gluon import pa_mqa_logits as gluon

    launched, _, _ = _launch_pa_mqa_logits_mode(mode, monkeypatch)
    ((_, kernel),) = launched
    signature, facts = _upstream_compile(mode, get_gfx())

    tt_fact = {
        Aligned(16): ("tt.divisibility", 16),
        PointerRange(32): ("tt.pointer_range", 32),
    }
    annotation = gluon._gluon_launch_keywords(*mode)["extra_annotation"]
    assert sorted(annotation) == sorted(
        name for name, ty in signature.items() if ty != "constexpr"
    )
    for name, argument in annotation.items():
        spec = argument.specialize
        assert spec is NEVER or isinstance(spec, Exactly), (name, spec)
        stated = {tt_fact[fact] for fact in getattr(spec, "facts", ())}
        assert stated == facts.get(name, set()), name

    names = kernel.src.fn.arg_names
    compiled = {
        names[path[0]]: {tuple(fact) for fact in path_facts}
        for path, path_facts in kernel.src.attrs.items()
        if path_facts
    }
    assert compiled == facts
    assert kernel.src.signature == signature
