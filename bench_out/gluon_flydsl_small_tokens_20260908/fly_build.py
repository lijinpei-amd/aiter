"""Artifact-only FlyDSL MoE GEMM1 builders for a matched dtype benchmark.

Inputs are shared logical tensors: X[T,K], W[E,2*I,K], W stored as [gate | up].
A4W4 uses packed E2M1 payloads and fused MXFP4 output. A8W8 uses E4M3
payloads and BF16 output. Both scaled formats use raw E8M0 scales per 32 K.
BF16 inputs/weights produce BF16 output. All variants fuse plain SiLU(gate)*up.
Preprocessing and first compilation happen in build_fly, outside ``call``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Callable

import torch


def fingerprint_ir(ir_text, dump_dir=None):
    """Decode actual gpu.binary ELF objects using MLIR's string parser.

    This only parses metadata on the CPU; it neither recompiles nor loads modules.
    """
    from flydsl._mlir import ir

    dump_dir = None if dump_dir is None else Path(dump_dir)
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / "compiled.mlir").write_text(ir_text)
    # MLIR escapes binary bytes as two hex digits and escapes quote/backslash;
    # accepting any escaped character lets the native parser enforce the grammar.
    strings = re.findall(r'\bbin = ("(?:\\.|[^"\\])*")', ir_text)
    assert strings, "compiled FlyDSL artifact does not contain gpu.binary bin strings"
    binaries = []
    with ir.Context():
        for index, quoted in enumerate(strings):
            blob = ir.StringAttr(ir.Attribute.parse(quoted)).value_bytes
            assert blob.startswith(b"\x7fELF"), "expected ROCm ELF in gpu.binary"
            binary = {
                "index": index, "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
            if dump_dir is not None:
                path = dump_dir / f"module{index}.hsaco"
                path.write_bytes(blob)
                binary["path"] = str(path.resolve())
            binaries.append(binary)
    kernel_metadata = []
    for match in re.finditer(
        r'#gpu\.kernel_metadata<"([^"]+)".*?metadata = (\{.*?\})', ir_text,
        flags=re.DOTALL,
    ):
        raw = match.group(2)
        attrs = {name: int(value) for name, value in re.findall(r'(\w+) = (-?\d+) : i\d+', raw)}
        for name, values in re.findall(r'(\w+) = array<i\d+: ([^>]+)>', raw):
            attrs[name] = [int(v.strip()) for v in values.split(",")]
        kernel_metadata.append({"name": match.group(1), "resources": attrs, "raw": raw})
    assert kernel_metadata, "compiled FlyDSL artifact is missing GPU kernel metadata"
    return {
        "compiled_ir_sha256": hashlib.sha256(ir_text.encode()).hexdigest(),
        "compiled_ir_bytes": len(ir_text.encode()),
        "binaries": binaries,
        "kernels": kernel_metadata,
    }


def routing_from_topk(
    topk_ids, num_experts, tile_m=128, allocated_blocks=None, logical_order=None
):
    """Build stable expert-sorted FlyDSL metadata using only CPU arithmetic.

    The returned tensors are transferred to ``topk_ids.device`` after construction.
    Useful rows follow ``logical_order`` when supplied: flattened token*topk+slot
    indices grouped by ascending expert, preserving the caller's within-expert
    order. Otherwise a stable expert argsort is used. Each expert's tail is padded
    to tile_m. Default allocation follows the native GEMM1 grid upper bound;
    ``allocated_blocks`` overrides allocation without changing the launch grid.

    num_valid_ids is [padded sorted row count, token count], matching Opus sorting.
    Sentinel rows contain token_num in low24 and slot=0; unused expert blocks -1.
    """
    ids = topk_ids.detach().to(device="cpu", dtype=torch.int64)
    t, topk = ids.shape
    assert 0 < t < (1 << 24) and 0 < topk < 128
    assert 0 < tile_m and 0 < num_experts
    assert bool(((ids >= 0) & (ids < num_experts)).all())
    if logical_order is None:
        order = torch.argsort(ids.flatten(), stable=True)
    else:
        order = torch.as_tensor(logical_order, device="cpu", dtype=torch.int64).flatten()
        assert torch.equal(torch.sort(order).values, torch.arange(t * topk))
        ordered_experts = ids.flatten()[order]
        assert bool((ordered_experts[1:] >= ordered_experts[:-1]).all())
    histogram = torch.bincount(ids.flatten(), minlength=num_experts)
    block_counts = (histogram + tile_m - 1) // tile_m
    useful_blocks = int(block_counts.sum())
    if allocated_blocks is None:
        # Match kernels.mxfp4_gemm1.gemm1_grid, including its BM128 special case.
        routes = t * topk
        active = num_experts if tile_m == 128 else min(routes, num_experts)
        allocated_blocks = (
            routes + active * (tile_m - 1) + tile_m - 1
        ) // tile_m
    assert allocated_blocks >= useful_blocks
    sorted_ids = torch.full((allocated_blocks * tile_m,), t, dtype=torch.int32)
    expert_ids = torch.full((allocated_blocks,), -1, dtype=torch.int32)
    raw_start = block_start = 0
    for expert, (count, blocks) in enumerate(zip(histogram.tolist(), block_counts.tolist())):
        routes = order[raw_start:raw_start + count]
        packed = ((routes // topk) | ((routes % topk) << 24)).to(torch.int32)
        sorted_ids[block_start * tile_m:block_start * tile_m + count] = packed
        expert_ids[block_start:block_start + blocks] = expert
        raw_start += count
        block_start += blocks
    assert raw_start == t * topk and block_start == useful_blocks
    nvi = torch.tensor([useful_blocks * tile_m, t], dtype=torch.int32)
    device = topk_ids.device
    return {
        "sorted_token_ids": sorted_ids.to(device),
        "sorted_expert_ids": expert_ids.to(device),
        "num_valid_ids": nvi.to(device),
    }


@dataclass
class FlyCase:
    call: Callable
    out: torch.Tensor
    kind: str
    layout: str
    sorted_token_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    topk: int
    token_num: int
    exe: object
    args: tuple
    keepalive: list
    config: dict
    sort_scales: Callable | None = None
    out_scales: torch.Tensor | None = None

    @property
    def outputs(self):
        """Raw native output buffers; padding must be excluded from replay hashes."""
        return (self.out,) if self.out_scales is None else (self.out, self.out_scales)

    def fingerprint(self, dump_dir=None):
        """Save/hash the exact artifact retained by this kernel's callable.

        If dump_dir is provided, ``cache/<manager-dir>/<entry>.pkl`` is a copy
        of the exact compiled cache entry; using that cache root avoids recompiling.
        """
        artifact = self.exe._cf._keepalive
        result = fingerprint_ir(artifact.ir, dump_dir)
        result["host_entry"] = artifact._entry
        result["explicit_module_loader"] = artifact._uses_explicit_module
        assert not artifact._post_load_processors, "hash needs to account for post-load binary patching"
        manager = self.exe.cache_manager
        keys = [key for key, value in self.exe._mem_cache.items() if value is artifact]
        assert len(keys) == 1, ("ambiguous compiled cache entry", keys)
        cache_file = manager._cache_file(self.exe._cache_key_to_str(keys[0]))
        result["cache_file"] = str(cache_file.resolve())
        result["cache_manager_key"] = self.exe.manager_key
        if cache_file.is_file():
            result["cache_sha256"] = hashlib.sha256(cache_file.read_bytes()).hexdigest()
            if dump_dir is not None:
                saved = Path(dump_dir) / "cache" / cache_file.parent.name / cache_file.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cache_file, saved)
                result["saved_cache_file"] = str(saved.resolve())
        modules = (
            "flydsl", "flydsl.compiler.jit_function", "flydsl.compiler.jit_executor",
            "aiter.ops.flydsl.moe_kernels", "aiter.ops.flydsl.kernels.tensor_shim",
            "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage",
            "aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage_common",
            "aiter.ops.flydsl.kernels.moe_2stage_a16wmix",
            "aiter.ops.flydsl.kernels.moe_2stage_a16wmix.gemm1",
            "aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils",
            "aiter.ops.flydsl.mxfp4_gemm1_kernels",
            "aiter.ops.flydsl.kernels.mxfp4_gemm1",
            "aiter.ops.flydsl.kernels.mxfp4_gemm_common",
            "aiter.ops.moe_mxfp4_aux", "aiter.ops.shuffle",
            "aiter.utility.fp4_utils",
        )
        result["source_modules"] = {}
        for name in modules:
            mod = sys.modules.get(name)
            file = getattr(mod, "__file__", None)
            if file is not None:
                path = Path(file)
                result["source_modules"][name] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        try:
            result["flydsl_version"] = importlib.metadata.version("flydsl")
        except importlib.metadata.PackageNotFoundError:
            result["flydsl_version"] = getattr(sys.modules["flydsl"], "__version__", None)
        return result

    def _token_order(self, value):
        if self.layout == "token_slot":
            return value
        ids = self.sorted_token_ids.to(torch.int64)
        tokens, slots = ids & 0xFFFFFF, ids >> 24
        valid = (
            (torch.arange(ids.numel(), device=ids.device) < self.num_valid_ids[0])
            & (tokens < self.token_num) & (slots >= 0) & (slots < self.topk)
        )
        out = torch.empty(
            (self.token_num, self.topk, value.shape[-1]),
            dtype=value.dtype, device=value.device,
        )
        out[tokens[valid], slots[valid]] = value[: ids.numel()][valid]
        return out

    def token_order_components(self):
        """Logical [T,topk,...] output bytes, excluding all native padding.

        A4 returns (packed E2M1 [T,topk,I/2], raw E8M0 [T,topk,I/32]).
        Other dtypes return the single BF16 [T,topk,I] tensor in a tuple.
        This layout conversion belongs strictly outside the timed queue.
        """
        if self.out_scales is None:
            return (self._token_order(self.out),)
        rows, packed_columns = self.out.shape
        scales = unshuffle_mxfp4_scales(
            self.out_scales, rows, packed_columns * 2 // 32
        )
        return self._token_order(self.out), self._token_order(scales)

    def token_order(self):
        """Return logical [T,topk,I]; dequantize A4 to FP32 outside timing."""
        components = self.token_order_components()
        if self.out_scales is None:
            return components[0]
        from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

        payload, scales = components
        return mxfp4_to_f32(payload) * e8m0_to_f32(scales).repeat_interleave(32, dim=-1)


def unshuffle_mxfp4_scales(buffer, rows, columns):
    """Invert the native 32-row/8-scale byte permutation; ignore trailing padding.

    The GEMM writes byte offset (r//32)*columns*32 + (c//8)*256 +
    (c%4)*64 + (r%16)*4 + ((c%8)//4)*2 + ((r%32)//16).
    Its public wrapper overallocates the scale buffer, so only the first
    rows*columns bytes represent the logical sorted output.
    """
    assert rows % 32 == 0 and columns % 8 == 0
    raw = buffer.view(torch.uint8).flatten()[: rows * columns]
    assert raw.numel() == rows * columns
    return (
        raw.view(rows // 32, columns // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(rows, columns)
    )


def build_fly(
    kind,
    x,
    w,
    topk_ids,
    *,
    x_scales=None,
    w_scales=None,
    routing=None,
    tile_m=128,
    tile_n=None,
    tile_k=None,
    waves_per_eu=None,
    b_nt=2,
    use_nt=False,
    gate_mode=None,
    use_async_copy=True,
    xcd_swizzle=0,
    k_wave=1,
):
    """Build a kernel-only closure using existing public FlyDSL entry points.

    Optional ``routing`` supplies sorted_token_ids (packed low24 token/high8 slot),
    sorted_expert_ids (one per tile_m block), and num_valid_ids (padded row count).
    Otherwise native aiter sorting runs at setup. All routing, quantization-layout
    transforms and scale sorting finish during setup; ``call`` retains precisely
    the compiled single GEMM launch and its prepared arguments.
    Every variant fuses SiLU(gate)*up, alpha=1, no clamp/bias/routing weight.
    Native prequantized A4 supports BM32 cached/NT and BM64/BM128 cached;
    ``use_nt`` selects its B cache policy independently of other dtype controls.
    """
    from aiter.fused_moe import moe_sorting
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    kind = {"bf16": "a16w16"}.get(kind, kind)
    assert kind in ("a4w4", "a8w8", "a16w16")
    assert os.environ.get("AITER_FLYDSL_NO_ACT", "0") == "0"
    defaults = {
        "a4w4": (256, 256, None, "separated"),
        "a8w8": (128, 256, 2, "interleave"),
        "a16w16": (128, 128, 1, "separated"),
    }[kind]
    tile_n = defaults[0] if tile_n is None else tile_n
    tile_k = defaults[1] if tile_k is None else tile_k
    waves_per_eu = defaults[2] if waves_per_eu is None else waves_per_eu
    gate_mode = defaults[3] if gate_mode is None else gate_mode
    t, packed_k = x.shape
    k = packed_k * 2 if kind == "a4w4" else packed_k
    e, n, wk = w.shape
    assert wk == packed_k and n % 2 == 0
    topk = topk_ids.shape[1]
    i = n // 2
    assert topk_ids.shape[0] == t
    assert x.device == w.device == topk_ids.device
    keepalive = [x, w, x_scales, w_scales, topk_ids]
    if routing is None:
        topk_weights = torch.ones(topk_ids.shape, dtype=torch.float32, device=x.device)
        sti, sw, sei, nvi, moe_buf = moe_sorting(
            topk_ids, topk_weights, e, k, torch.bfloat16,
            block_size=tile_m, accumulate=False,
        )
        keepalive += [topk_weights, sw, moe_buf]
    else:
        sti = routing["sorted_token_ids"]
        sei = routing["sorted_expert_ids"]
        nvi = routing["num_valid_ids"]
    keepalive += [sti, sei, nvi]
    captured = []
    sort_scales = None
    out_scales = None
    native_grid = None

    if kind == "a4w4":
        from aiter.ops.flydsl import moe_kernels as mk
        from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
        from aiter.ops.moe_mxfp4_aux import mxfp4_moe_sort_scales

        assert (tile_m, use_nt) in ((32, False), (32, True), (64, False), (128, False))
        assert (tile_n, tile_k) == (256, 256)
        assert waves_per_eu is None, "native A4 port does not expose an occupancy override"
        assert gate_mode == "separated" and k_wave == 1
        assert x.element_size() == w.element_size() == 1
        assert x.dtype in (torch.uint8, torch.float4_e2m1fn_x2)
        assert w.dtype in (torch.uint8, torch.float4_e2m1fn_x2)
        assert k % 256 == 0 and i % 256 == 0
        assert x_scales is not None and w_scales is not None
        assert tuple(x_scales.shape) == (t, k // 32)
        assert tuple(w_scales.shape) == (e, n, k // 32)
        x = x.contiguous().view(torch.uint8)
        w = w.contiguous().view(torch.uint8)
        xs = x_scales.contiguous().view(torch.uint8)
        ws = w_scales.contiguous().view(torch.uint8).reshape(e * n, k // 32)
        wp = shuffle_weight(w, (16, 16))
        wsp = e8m0_shuffle(ws).view(torch.uint8)
        max_sorted = sti.numel()
        assert max_sorted % tile_m == 0 and sei.numel() * tile_m >= max_sorted
        # The native A4 GEMM takes plain token indices; the sorter takes packed
        # token/slot indices. Keep both, since token_order needs the slot bits.
        m_indices = (sti & 0xFFFFFF).contiguous()
        sorted_scales = torch.empty(
            (((max_sorted + 31) // 32) * 32 * (k // 32) * 2,),
            dtype=torch.uint8, device=x.device,
        )
        mxfp4_moe_sort_scales(
            xs, sti, nvi, sorted_scales, e, topk, k, tile_m, max_sorted
        )
        out = torch.empty((max_sorted, i // 2), dtype=torch.uint8, device=x.device)
        scale_cols = i // 32
        # Match _mxfp4_a4w4_stage1's native buffer sizing. Trailing padding
        # remains exposed for poisoning but never enters correctness hashes.
        scale_bytes = max_sorted * max((1024 // 64) * 4, scale_cols * 2)
        scale_rows = (scale_bytes + scale_cols - 1) // scale_cols
        scale_rows = (scale_rows + 31) // 32 * 32
        out_scales = torch.empty(
            (scale_rows, scale_cols), dtype=torch.uint8, device=x.device
        )
        hidden_unused = torch.empty((0,), dtype=torch.bfloat16, device=x.device)
        real_run = mk._run_compiled

        def capture(exe, args):
            captured.append((exe, args))
            return real_run(exe, args)

        mk._run_compiled = capture
        try:
            flydsl_mxfp4_gemm1(
                a_quant=x, a_scale_sorted_shuffled=sorted_scales,
                w1_u8=wp, w1_scale_u8=wsp, sorted_expert_ids=sei,
                cumsum_tensor=nvi, m_indices=m_indices,
                inter_sorted_quant=out, inter_sorted_shuffled_scale=out_scales,
                hidden_states=hidden_unused, n_tokens=t,
                BM=tile_m, use_nt=use_nt, inline_quant=False,
                NE=e, D_HIDDEN=k, D_INTER=i, topk=topk,
                BN=tile_n, BK=tile_k, interleave=False, xcd_swizzle=xcd_swizzle,
            )
        finally:
            mk._run_compiled = real_run
        native_grid = int(captured[0][1][8])
        keepalive += [x, w, xs, ws, wp, wsp, m_indices, sorted_scales,
                      out, out_scales, hidden_unused]
        layout = "sorted"

    elif kind == "a8w8":
        from aiter.ops.flydsl import moe_kernels as mk
        from aiter.ops.quant import mxfp4_moe_sort_fwd, mxfp4_moe_sort_hip

        assert x.dtype == w.dtype == torch.float8_e4m3fn
        assert x_scales is not None and w_scales is not None
        assert tuple(x_scales.shape) == (t, k // 32)
        assert tuple(w_scales.shape) == (e, n, k // 32)
        x = x.contiguous()
        w = w.contiguous()
        xs = x_scales.contiguous().view(torch.float8_e8m0fnu)
        ws = w_scales.contiguous().view(torch.float8_e8m0fnu).reshape(e * n, k // 32)
        if k <= 16384:
            sorted_scales = mxfp4_moe_sort_fwd(xs, sti, nvi, t, k)
        else:
            # The legacy HIP sorter aborts above K16384. The repository's
            # general sorter writes the same byte layout, during setup only.
            from aiter.utility.fp4_utils import moe_mxfp4_sort
            sorted_scales = moe_mxfp4_sort(xs, sti, nvi, t, block_size=tile_m)
        decoded_scales = unshuffle_mxfp4_scales(sorted_scales, sti.numel(), k // 32)
        token_indices = sti.long() & 0xFFFFFF
        valid_scales = (token_indices < t) & (torch.arange(sti.numel(), device=x.device) < nvi[0])
        assert torch.equal(decoded_scales[valid_scales], xs.view(torch.uint8)[token_indices[valid_scales]]), "prepared A scale layout mismatch"
        if gate_mode == "interleave":
            wp = shuffle_weight_a16w4(w, 16, True)
            wsp = shuffle_scale_a16w4(ws, e, True)
        elif gate_mode == "separated":
            wp = shuffle_weight(w, (16, 16))
            wsp = e8m0_shuffle(ws)
        else:
            raise ValueError(gate_mode)
        out = torch.empty((t, topk, i), dtype=torch.bfloat16, device=x.device)
        real_run = mk._run_compiled

        def capture(exe, args):
            captured.append((exe, args))
            return real_run(exe, args)

        mk._run_compiled = capture
        try:
            mk.flydsl_moe_stage1(
                a=x, w1=wp, sorted_token_ids=sti, sorted_expert_ids=sei,
                num_valid_ids=nvi, out=out, topk=topk,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                a_dtype="fp8", b_dtype="fp8", out_dtype="bf16", act="silu",
                a1_scale=sorted_scales, w1_scale=wsp, sorted_weights=None,
                persist_m=1, use_async_copy=use_async_copy, k_batch=1,
                waves_per_eu=waves_per_eu, b_nt=b_nt, gate_mode=gate_mode,
                xcd_swizzle=xcd_swizzle, k_wave=k_wave, swiglu_limit=float("inf"),
            )
        finally:
            mk._run_compiled = real_run
        keepalive += [x, w, xs, ws, sorted_scales, wp, wsp, out]

        def sort_scales():
            if k <= 16384:
                mxfp4_moe_sort_hip(sorted_scales, xs, sti, nvi, t, k)
            else:
                sorted_scales.copy_(moe_mxfp4_sort(xs, sti, nvi, t, block_size=tile_m))

        layout = "token_slot"
    else:
        from aiter.ops.flydsl.kernels import moe_2stage_a16wmix as mix

        assert x.dtype == w.dtype == torch.bfloat16
        x = x.contiguous()
        wp = shuffle_weight(w.contiguous(), (16, 16))
        empty_scale = torch.empty(0, dtype=torch.uint8, device=x.device)
        out = torch.empty(
            (max(sti.numel(), sei.numel() * tile_m), i),
            dtype=torch.bfloat16, device=x.device,
        )
        real_run = mix._run_compiled

        def capture(exe, *args):
            captured.append((exe, args))
            return real_run(exe, *args)

        mix._run_compiled = capture
        try:
            mix.flydsl_a16w4_gemm1(
                a_bf16=x, w1_u8=wp, w1_scale_u8=empty_scale,
                sorted_expert_ids=sei, cumsum_tensor=nvi, m_indices=sti,
                inter_sorted_bf16=out, n_tokens=t, NE=e, D_HIDDEN=k,
                D_INTER=i, topk=topk, tile_m=tile_m, tile_n=tile_n,
                tile_k=tile_k, waves_per_eu=waves_per_eu, k_batch=1,
                k_wave=k_wave, b_nt=b_nt, xcd_swizzle=xcd_swizzle,
                gate_mode="separated", act="silu", swiglu_limit=float("inf"),
                w_dtype="bf16", w_layout="standard",
            )
        finally:
            mix._run_compiled = real_run
        keepalive += [x, wp, empty_scale, out]
        layout = "sorted"
        gate_mode = "separated"

    assert len(captured) == 1, f"expected one GEMM1 launch, got {len(captured)}"
    exe, args = captured[0]
    assert getattr(exe, "_cf", None) is not None
    cf = exe._cf

    def call():
        cf(*args)

    config = dict(
        kind=kind, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        waves_per_eu=waves_per_eu, b_nt=b_nt, gate_mode=gate_mode,
        use_async_copy=use_async_copy, xcd_swizzle=xcd_swizzle, k_wave=k_wave,
        act="silu", alpha=1.0, clamp=None, bias=False, routing_weights=False,
        output="mxfp4" if kind == "a4w4" else "bf16", output_layout=layout,
        shape={"T": t, "K": k, "N": n, "E": e, "topk": topk},
        preprocessing="setup only; retained compiled callable and prepared arguments",
        allocated_routing_blocks=sei.numel(),
        active_routing_blocks=int(nvi[0].item()) // tile_m,
        native_grid_blocks=native_grid,
    )
    if kind == "a4w4":
        config.update(use_nt=use_nt, inline_quant=False, b_nt=2 if use_nt else 0,
                      native_m_block_upper_bound=native_grid // (n // tile_n),
                      scale_layout="native shuffled; logical payload/scale accessor excludes padding")
    return FlyCase(
        call, out, kind, layout, sti, sei, nvi, topk, t, exe, args,
        keepalive, config, sort_scales, out_scales,
    )
