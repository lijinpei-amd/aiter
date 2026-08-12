# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single source of truth for the MoE shapes/dtypes/activations of the models the
Triton MoE kernels are expected to serve.

Both the pytest suites under ``op_tests/triton_tests/moe/`` and the benchmarks under
``op_tests/op_benchmarks/triton/`` consume this module, so a shape only ever has to be
written down once.

Covers DeepSeek-V4-{Flash,Pro}, GLM-5.2 and MiniMax-M3 (see
``moe-summary-dsv4-glm52-m3.md``): all three ship AMD Quark MXFP4 repacks that are
structurally identical (E2M1 packed 2/byte, group 32 along K, one E8M0 shared exponent
per group, weights static / activations dynamic), so one kernel path covers all of
them and only the shapes and the epilogue change.

Two conventions are easy to get wrong and are therefore centralised here:

``intermediate_size`` in ``op_benchmarks/triton/utils/model_configs.json`` is the
**fused gate+up width, i.e. 2*I**, not the per-projection ``I``. ``bench_moe.py``
reads it as stage-1 ``N`` and derives stage-2 ``K = intermediate_size // 2``, and the
silu-fused path writes ``N // 2`` columns. Storing ``I`` there silently halves the
stage-1 FLOPs and makes stage-2 ``K = I/2``. ``MoeRecipe.intermediate_size`` below is
``I``; ``to_model_config()`` is the only place that doubles it.

The swiglu epilogue knobs (``alpha``/``limit``/``add_residual``) default to
``alpha=1.0, limit=1.0, swiglu_add_residual=True`` in every ``moe_gemm_*`` wrapper,
which is not any real model's activation -- it is silu clamped at 1.0 *with* a
residual. Always splat :meth:`SwigluRecipe.op_kwargs` rather than relying on defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ACT_RECIPES",
    "MODEL_RECIPES",
    "SWIGLU_CLAMP10",
    "SWIGLU_OAI",
    "SWIGLU_PLAIN",
    "MoeGemmShape",
    "MoeRecipe",
    "SwigluRecipe",
    "all_recipes",
    "gemm_shapes",
    "get_recipe",
    "recipe_ids",
    "skip_reason",
    "weight_bytes",
]


# --------------------------------------------------------------------------------
# Activation recipes
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class SwigluRecipe:
    """A concrete parameterisation of the fused swiglu epilogue.

    Maps onto ``_swiglu`` in
    ``aiter/ops/triton/_triton_kernels/moe/activations.py``::

        g = clamp(gate, max=limit)          # one-sided clamp on the gate
        u = clamp(up, -limit, limit)        # two-sided clamp on the up projection
        s = g * sigmoid(alpha * g)          # note: the clamped g is used in BOTH factors
        out = s * (u + 1) if add_residual else s * u

    Note the gate clamp is one-sided (upper only) and the up clamp is two-sided,
    matching both the DeepSeek-V4 and GPT-OSS/MiniMax definitions.
    """

    name: str
    alpha: float
    limit: float | None
    add_residual: bool

    def op_kwargs(self) -> dict:
        """Kwargs for the ``moe_gemm_*`` Triton wrappers."""
        return {
            "apply_swiglu": True,
            "alpha": self.alpha,
            "limit": self.limit,
            "swiglu_add_residual": self.add_residual,
        }

    def torch_kwargs(self) -> dict:
        """Kwargs for the matching ``moe_gemm_torch`` references.

        The references spell the residual flag ``add_residual`` while the ops spell it
        ``swiglu_add_residual``; keeping the two spellings behind these helpers stops
        that asymmetry leaking into every call site.
        """
        return {
            "apply_swiglu": True,
            "alpha": self.alpha,
            "limit": self.limit,
            "add_residual": self.add_residual,
        }


# GLM-5.2: stock silu_and_mul, the only one of the three that needs no clamp.
SWIGLU_PLAIN = SwigluRecipe("glm_silu", alpha=1.0, limit=None, add_residual=False)
# DeepSeek-V4: clamped SwiGLU, swiglu_limit = 10.0, computed in fp32.
SWIGLU_CLAMP10 = SwigluRecipe("dsv4_clamp10", alpha=1.0, limit=10.0, add_residual=False)
# MiniMax-M3: `swigluoai`, alpha 1.702, limit 7.0, with the (up + 1) residual.
SWIGLU_OAI = SwigluRecipe("m3_swigluoai", alpha=1.702, limit=7.0, add_residual=True)

ACT_RECIPES = (SWIGLU_PLAIN, SWIGLU_CLAMP10, SWIGLU_OAI)


# --------------------------------------------------------------------------------
# Model recipes
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class MoeGemmShape:
    """One grouped-GEMM in the MoE layer, in the ``Case(m, n, k, ...)`` convention
    the ``test_moe_gemm_*`` suites use."""

    model: str
    stage: int  # 1 = fused gate/up, 2 = down
    m: int
    n: int
    k: int
    n_expts_tot: int
    n_expts_act: int

    @property
    def id(self) -> str:
        return f"{self.model}-s{self.stage}-m{self.m}"


@dataclass(frozen=True)
class MoeRecipe:
    name: str
    hidden_size: int  # H
    intermediate_size: int  # I, per projection (NOT the fused 2*I)
    n_routed_experts: int  # E
    topk: int
    swiglu: SwigluRecipe
    # routing
    score_mode: str  # "sqrtsoftplus" | "sigmoid"
    use_grouped_topk: bool
    num_expert_group: int
    topk_group: int
    use_score_bias: bool
    renorm: bool
    routed_scaling_factors: tuple[float, ...]
    # structure
    n_shared_experts: int = 1
    vocab_size: int = 0
    # Attention shape. Not used by the MoE kernels, but model_configs.json is a shared
    # registry: the attention benchmarks (e.g. bench_mha.py) iterate every entry and
    # index these keys unconditionally, so an MoE-only entry breaks them.
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    hash_routing_layers: int = 0  # DeepSeek-V4 layers 0-2 use tid2eid hash routing
    dense_intermediate_size: int = 0  # non-MoE prefix layers, 0 if none
    first_k_dense_replace: int = 0
    # dtypes of the *base* (non-repacked) checkpoint
    base_act_dtype: str = "bf16"
    base_weight_dtype: str = "bf16"
    # every one of these models has an AMD Quark MXFP4 (A4W4) repack
    has_mxfp4_repack: bool = True
    notes: str = field(default="", compare=False)

    # -- shapes ------------------------------------------------------------------
    @property
    def fused_intermediate_size(self) -> int:
        """Gate+up fused width == 2*I. This is what model_configs.json stores."""
        return 2 * self.intermediate_size

    def gemm_shape(self, stage: int, m: int, tp: int = 1) -> MoeGemmShape:
        """(N, K) for one MoE grouped-GEMM stage, optionally tensor-parallel sharded.

        stage 1: X[m, H] @ W1[E, 2*I/tp, H]  -> (n=2*I/tp, k=H)
        stage 2: X[m, I/tp] @ W2[E, H, I/tp] -> (n=H,       k=I/tp)
        """
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        inter = self.intermediate_size // tp
        if inter * tp != self.intermediate_size:
            raise ValueError(
                f"{self.name}: intermediate_size {self.intermediate_size} "
                f"not divisible by tp={tp}"
            )
        n, k = (
            (2 * inter, self.hidden_size) if stage == 1 else (self.hidden_size, inter)
        )
        return MoeGemmShape(
            model=self.name,
            stage=stage,
            m=m,
            n=n,
            k=k,
            n_expts_tot=self.n_routed_experts,
            n_expts_act=self.topk,
        )

    def shared_gemm_shape(self, stage: int, m: int, tp: int = 1) -> tuple[int, int]:
        """(N, K) for the dense shared-expert GEMM (same I, no expert dim)."""
        s = self.gemm_shape(stage, m, tp)
        return s.n, s.k

    # -- bench config --------------------------------------------------------------
    def to_model_config(self) -> dict:
        """The dict shape ``benchmark_utils.get_model_configs`` returns.

        ``intermediate_size`` is deliberately ``2*I`` -- see the module docstring.
        """
        cfg = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.fused_intermediate_size,
            "num_expert": self.n_routed_experts,
            "top_k": self.topk,
            "n_shared_experts": self.n_shared_experts,
            "shared_intermediate_size": self.fused_intermediate_size,
            "act_fn": self.swiglu.name,
            "act_alpha": self.swiglu.alpha,
            "act_limit": self.swiglu.limit,
            "act_add_residual": self.swiglu.add_residual,
            "score_mode": self.score_mode,
            "use_grouped_topk": self.use_grouped_topk,
            "num_expert_group": self.num_expert_group,
            "topk_group": self.topk_group,
            "use_score_bias": self.use_score_bias,
            "renorm": self.renorm,
            "routed_scaling_factor": list(self.routed_scaling_factors),
        }
        cfg["vocab_size"] = self.vocab_size
        if self.hash_routing_layers:
            cfg["hash_routing_layers"] = self.hash_routing_layers
        if self.dense_intermediate_size:
            cfg["dense_intermediate_size"] = 2 * self.dense_intermediate_size
            cfg["first_k_dense_replace"] = self.first_k_dense_replace
        return cfg


MODEL_RECIPES: dict[str, MoeRecipe] = {
    "dsv4-flash": MoeRecipe(
        name="dsv4-flash",
        num_attention_heads=64,
        num_key_value_heads=1,
        hidden_size=4096,
        intermediate_size=2048,
        n_routed_experts=256,
        topk=6,
        swiglu=SWIGLU_CLAMP10,
        score_mode="sqrtsoftplus",
        # DeepSeek-V4 config.json has topk_method=noaux_tc with NO n_group/topk_group,
        # i.e. a flat top-k over all routed experts (the `o_groups` field in those
        # configs is attention, not routing). Verified against the checkpoints.
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        use_score_bias=True,
        renorm=True,
        routed_scaling_factors=(1.5,),
        vocab_size=129280,
        hash_routing_layers=3,
        base_act_dtype="fp8_e4m3",  # dynamic per-token, block 128 along K, ue8m0
        base_weight_dtype="fp4_e2m1",  # group 32 along K, fp8 e8m0 scale
        notes="no dense prefix layers; layers 0-2 use tid2eid[129280,6] hash routing",
    ),
    "dsv4-pro": MoeRecipe(
        name="dsv4-pro",
        num_attention_heads=128,
        num_key_value_heads=1,
        hidden_size=7168,
        intermediate_size=3072,
        n_routed_experts=384,
        topk=6,
        swiglu=SWIGLU_CLAMP10,
        score_mode="sqrtsoftplus",
        # DeepSeek-V4 config.json has topk_method=noaux_tc with NO n_group/topk_group,
        # i.e. a flat top-k over all routed experts (the `o_groups` field in those
        # configs is attention, not routing). Verified against the checkpoints.
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        use_score_bias=True,
        renorm=True,
        routed_scaling_factors=(2.5,),
        vocab_size=129280,
        hash_routing_layers=3,
        base_act_dtype="fp8_e4m3",
        base_weight_dtype="fp4_e2m1",
        notes="E=384; fine on the flat topk path (n_cols<32768), unlike grouped_topk",
    ),
    "glm52-base": MoeRecipe(
        name="glm52-base",
        num_attention_heads=64,
        num_key_value_heads=64,
        hidden_size=6144,
        intermediate_size=2048,
        n_routed_experts=256,
        topk=8,
        swiglu=SWIGLU_PLAIN,
        score_mode="sigmoid",
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        use_score_bias=True,
        renorm=True,
        routed_scaling_factors=(2.5,),
        vocab_size=154880,
        dense_intermediate_size=12288,
        first_k_dense_replace=3,
        base_act_dtype="bf16",
        base_weight_dtype="bf16",
        notes="official FP8 variant at zai-org/GLM-5.2-FP8; 1 expert group",
    ),
    "minimax-m3": MoeRecipe(
        name="minimax-m3",
        num_attention_heads=64,
        num_key_value_heads=4,
        hidden_size=6144,
        intermediate_size=3072,
        n_routed_experts=128,
        topk=4,
        swiglu=SWIGLU_OAI,
        score_mode="sigmoid",
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        use_score_bias=True,
        renorm=True,
        routed_scaling_factors=(2.0,),
        vocab_size=200064,
        dense_intermediate_size=12288,
        first_k_dense_replace=3,
        base_act_dtype="bf16",
        base_weight_dtype="bf16",
        notes="official MXFP8 at MiniMaxAI/MiniMax-M3-MXFP8",
    ),
}


def get_recipe(name: str) -> MoeRecipe:
    try:
        return MODEL_RECIPES[name]
    except KeyError:
        raise KeyError(
            f"unknown MoE model recipe {name!r}; known: {sorted(MODEL_RECIPES)}"
        ) from None


def all_recipes() -> list[MoeRecipe]:
    return list(MODEL_RECIPES.values())


def recipe_ids() -> list[str]:
    return list(MODEL_RECIPES)


def gemm_shapes(
    tokens=(16, 1024),
    models=None,
    stages=(1, 2),
    tp: int = 1,
) -> list[MoeGemmShape]:
    """All (model, stage, m) grouped-GEMM shapes, for pytest parametrisation.

    ``tokens`` is deliberately small by default: the torch references in
    ``moe_gemm_torch`` loop over experts in Python, so large M is expensive.
    """
    names = recipe_ids() if models is None else list(models)
    return [
        get_recipe(name).gemm_shape(stage, m, tp=tp)
        for name in names
        for stage in stages
        for m in tokens
    ]


def weight_bytes(shape: MoeGemmShape, bits_per_elem: int = 16) -> int:
    """Bytes for one copy of the stacked expert weight of ``shape``.

    The a4w4/a8w4 harnesses materialise a bf16 master plus a reference copy plus an
    upcast copy, so budget roughly 3x this at bf16 before deciding to skip.
    """
    return shape.n_expts_tot * shape.n * shape.k * bits_per_elem // 8


# bf16-equivalent copies of the stacked expert weight each harness holds live at
# peak. Measured with torch.cuda.max_memory_allocated, not guessed: the blockscale
# harness is the worst because it keeps an fp8 master, a bf16 dequantised reference
# and per-block scales simultaneously.
HARNESS_WEIGHT_COPIES = {
    "a4w4": 4,
    "a16w4": 4,
    "a8w4": 4,
    "a8w8": 4,
    "a8w8_blockscale": 7,
}


def required_hbm_bytes(shape: MoeGemmShape, copies: int = 4) -> int:
    """Rough peak device memory for one ``test_model_shapes`` case.

    The harnesses hold a bf16 master weight, a bf16 reference clone and a bf16
    upcast-from-mxfp copy live simultaneously; activations and scales are small next
    to those. dsv4-pro stage 1 is the only shape where this matters
    (E=384 x N=6144 x K=7168 = 33.8 GB per bf16 copy).
    """
    return copies * weight_bytes(shape, bits_per_elem=16)


def skip_if_insufficient_hbm(
    shape: MoeGemmShape, copies: int = 4, headroom: float = 1.25
) -> None:
    """``pytest.skip`` when ``shape`` cannot fit in free device memory.

    Imports torch/pytest lazily so this module stays importable (and unit-testable)
    without a GPU.
    """
    import pytest
    import torch

    need = int(required_hbm_bytes(shape, copies) * headroom)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/HIP device")
    # Release the caching allocator's idle blocks first: mem_get_info reports
    # driver-free bytes, so without this a long parametrised run sees almost nothing
    # free once torch has cached a few large weight tensors, and the guard would skip
    # shapes that actually fit.
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(
            f"{shape.id} needs ~{need / 1e9:.0f} GB of free HBM "
            f"(E={shape.n_expts_tot} x N={shape.n} x K={shape.k}, "
            f"{copies} bf16 copies), only {free / 1e9:.0f} GB free"
        )


# --------------------------------------------------------------------------------
# Known kernel gaps
# --------------------------------------------------------------------------------
# Reason strings live here so every suite reports an identical, greppable message.
_SKIPS = {
    # NOTE: the two DeepSeek-V4 models are NOT listed here. They use flat noaux_tc
    # top-k with sqrtsoftplus + fp32 bias, which aiter/ops/triton/moe/moe_routing/
    # topk.py:207-220 supports at any E < 32768 -- including dsv4-pro's E=384. Only
    # grouped_topk carries the E<=256 limit (topk.py:41).
    ("glm52-base", "routing"): (
        "grouped_topk asserts num_expert_group>1 "
        "(aiter/ops/triton/moe/moe_routing/topk.py:44) but glm52 has 1 group, and "
        "routing.py:369 then falls through to flat topk which rejects "
        "score_mode='sigmoid' (topk.py:207-210)"
    ),
    ("minimax-m3", "routing"): (
        "flat topk rejects score_mode='sigmoid' "
        "(aiter/ops/triton/moe/moe_routing/topk.py:207-210); minimax-m3 needs "
        "ungrouped sigmoid + bias + renorm"
    ),
    ("dsv4-flash", "native_a8w4"): (
        "DeepSeek-V4 native A8W4 pairs fp8 activations at block-128 with fp4 weights "
        "at group-32; tl.dot_scaled requires a matched 1x32 granularity "
        "(_triton_kernels/moe/moe_op_gemm_a8w4.py:198). Use the ue8m0 "
        "scale-replication proxy instead"
    ),
}
_SKIPS[("dsv4-pro", "native_a8w4")] = _SKIPS[("dsv4-flash", "native_a8w4")]


def skip_reason(model: str, feature: str) -> str | None:
    """Reason string if ``feature`` is a known kernel gap for ``model``, else None."""
    return _SKIPS.get((model, feature))
