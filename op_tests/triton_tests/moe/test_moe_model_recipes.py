# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only guards for the MoE model recipes.

These run without a GPU and without Triton codegen, so they are the cheapest possible
signal that ``moe_model_recipes`` and ``op_benchmarks/triton/utils/model_configs.json``
still agree with each other and with the published model definitions.
"""

import json
import os

import pytest
import torch

from aiter.ops.triton.moe.moe_op_gemm_a4w4 import swiglu_torch
from op_tests.triton_tests.moe.moe_model_recipes import (
    ACT_RECIPES,
    MODEL_RECIPES,
    SWIGLU_CLAMP10,
    SWIGLU_OAI,
    SWIGLU_PLAIN,
    all_recipes,
    get_recipe,
    recipe_ids,
)

MODEL_CONFIGS_JSON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "op_benchmarks",
    "triton",
    "utils",
    "model_configs.json",
)


# --------------------------------------------------------------------------------
# Independent references, transcribed from moe-summary-dsv4-glm52-m3.md section 3.
# Deliberately NOT written in terms of SwigluRecipe, so that a typo in the recipe
# constants is caught rather than mirrored.
# --------------------------------------------------------------------------------
def _ref_glm_silu(gate, up):
    """GLM-5.2: plain SwiGLU, out = silu(gate) * up."""
    return torch.nn.functional.silu(gate) * up


def _ref_dsv4_clamped(gate, up):
    """DeepSeek-V4: clamp(gate, max=10), clamp(up, -10, 10), silu(gate) * up."""
    gate = gate.clamp(max=10.0)
    up = up.clamp(min=-10.0, max=10.0)
    return torch.nn.functional.silu(gate) * up


def _ref_m3_swigluoai(gate, up):
    """MiniMax-M3: swigluoai, alpha 1.702, limit 7.0, with the (up + 1) residual."""
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return (up + 1.0) * gate * torch.sigmoid(1.702 * gate)


_ACT_REFS = {
    SWIGLU_PLAIN.name: _ref_glm_silu,
    SWIGLU_CLAMP10.name: _ref_dsv4_clamped,
    SWIGLU_OAI.name: _ref_m3_swigluoai,
}


def _interleave(gate, up):
    """Pack gate/up into the interleaved layout the kernels and refs expect."""
    return torch.stack([gate, up], dim=-1).flatten(-2)


# --------------------------------------------------------------------------------
# Activation recipes
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("recipe", ACT_RECIPES, ids=[r.name for r in ACT_RECIPES])
def test_swiglu_recipe_matches_published_formula(recipe):
    """Each recipe drives swiglu_torch to exactly the model's documented activation."""
    torch.manual_seed(0)
    # Range must straddle both clamp limits (7.0 and 10.0) or the clamps are untested.
    gate = torch.empty(512, dtype=torch.float32).uniform_(-20.0, 20.0)
    up = torch.empty(512, dtype=torch.float32).uniform_(-20.0, 20.0)

    got = swiglu_torch(
        _interleave(gate, up),
        alpha=recipe.alpha,
        limit=recipe.limit,
        add_residual=recipe.add_residual,
    )
    want = _ACT_REFS[recipe.name](gate, up)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_swiglu_recipes_are_mutually_distinct():
    """Guards against two recipes silently collapsing onto the same maths."""
    torch.manual_seed(0)
    gate = torch.empty(256, dtype=torch.float32).uniform_(-20.0, 20.0)
    up = torch.empty(256, dtype=torch.float32).uniform_(-20.0, 20.0)
    a = _interleave(gate, up)
    outs = {
        r.name: swiglu_torch(
            a, alpha=r.alpha, limit=r.limit, add_residual=r.add_residual
        )
        for r in ACT_RECIPES
    }
    names = list(outs)
    for i, ni in enumerate(names):
        for nj in names[i + 1 :]:
            assert not torch.allclose(
                outs[ni], outs[nj], atol=1e-4
            ), f"activation recipes {ni} and {nj} produce identical output"


def test_default_swiglu_kwargs_are_not_a_real_activation():
    """Canary on the `limit=1.0, swiglu_add_residual=True` wrapper defaults.

    Every ``moe_gemm_*`` wrapper defaults to ``alpha=1.0, limit=1.0,
    swiglu_add_residual=True`` (e.g. aiter/ops/triton/moe/moe_op_gemm_a4w4.py:194-196),
    which is silu clamped at 1.0 *with* an (up+1) residual -- not the activation of any
    model in MODEL_RECIPES. Tests must therefore always pass the knobs explicitly via
    ``SwigluRecipe.op_kwargs()`` / ``torch_kwargs()``.

    If this test starts failing because the defaults were fixed, delete it and drop the
    warning from the moe_model_recipes docstring.
    """
    torch.manual_seed(0)
    gate = torch.empty(256, dtype=torch.float32).uniform_(-20.0, 20.0)
    up = torch.empty(256, dtype=torch.float32).uniform_(-20.0, 20.0)
    a = _interleave(gate, up)

    defaults = swiglu_torch(a, alpha=1.0, limit=1.0, add_residual=True)
    for r in ACT_RECIPES:
        explicit = swiglu_torch(
            a, alpha=r.alpha, limit=r.limit, add_residual=r.add_residual
        )
        assert not torch.allclose(
            defaults, explicit, atol=1e-4
        ), f"wrapper defaults now coincide with the {r.name} recipe"


# --------------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", recipe_ids())
def test_gemm_shapes_follow_the_two_stage_contract(name):
    r = get_recipe(name)
    s1 = r.gemm_shape(1, m=16)
    s2 = r.gemm_shape(2, m=16)
    # stage 1 consumes hidden, emits the fused gate+up width
    assert s1.k == r.hidden_size
    assert s1.n == 2 * r.intermediate_size
    # stage 2 consumes one projection's width, emits hidden
    assert s2.k == r.intermediate_size
    assert s2.n == r.hidden_size
    # expert counts are carried through unchanged
    for s in (s1, s2):
        assert s.n_expts_tot == r.n_routed_experts
        assert s.n_expts_act == r.topk


@pytest.mark.parametrize("name", recipe_ids())
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_gemm_shapes_shard_on_intermediate_only(name, tp):
    r = get_recipe(name)
    if r.intermediate_size % tp:
        pytest.skip(f"{name} I={r.intermediate_size} not divisible by tp={tp}")
    s1 = r.gemm_shape(1, m=16, tp=tp)
    s2 = r.gemm_shape(2, m=16, tp=tp)
    # TP splits the intermediate dimension; hidden is replicated
    assert s1.n == 2 * r.intermediate_size // tp
    assert s1.k == r.hidden_size
    assert s2.n == r.hidden_size
    assert s2.k == r.intermediate_size // tp


# --------------------------------------------------------------------------------
# model_configs.json consistency
# --------------------------------------------------------------------------------
def test_model_configs_json_matches_recipes():
    """model_configs.json must stay a faithful projection of MODEL_RECIPES.

    In particular ``intermediate_size`` there is the *fused* gate+up width (2*I);
    storing I instead silently halves stage-1 FLOPs in every benchmark.
    """
    with open(MODEL_CONFIGS_JSON) as f:
        cfg = json.load(f)

    for name, recipe in MODEL_RECIPES.items():
        family, variant = name.split("-", 1)
        assert (
            family in cfg
        ), f"{name}: family {family!r} missing from model_configs.json"
        assert variant in cfg[family], f"{name}: variant {variant!r} missing"
        entry = cfg[family][variant]
        expected = recipe.to_model_config()
        for key, want in expected.items():
            assert key in entry, f"{name}: key {key!r} missing from model_configs.json"
            assert entry[key] == want, (
                f"{name}: model_configs.json[{key}] = {entry[key]!r}, "
                f"recipe says {want!r}"
            )
        assert entry["intermediate_size"] == 2 * recipe.intermediate_size


def test_model_config_keys_round_trip_through_get_model_configs():
    """The recipe name must be exactly the key get_model_configs produces.

    get_model_configs splits on '_' if present else '-' and keys results as
    f"{family}-{variant}" (benchmark_utils.py:242-257), so a recipe name containing an
    underscore, or more than one dash, would silently fail to resolve.
    """
    for name in MODEL_RECIPES:
        assert "_" not in name, f"{name}: '_' breaks the family/variant split"
        assert name.count("-") == 1, f"{name}: expected exactly one '-'"


def test_every_recipe_has_a_shared_expert():
    # All four models pair the routed experts with exactly one shared expert of the
    # same intermediate size (moe-summary section 1).
    for r in all_recipes():
        assert r.n_shared_experts == 1


def test_skip_reason_keys_reference_real_recipes():
    """Every skip key must name a recipe that exists.

    A stale key (e.g. left behind after renaming a recipe) silently turns an intended
    skip into a hard failure, so pin it here rather than discovering it in CI.
    """
    from op_tests.triton_tests.moe import moe_model_recipes as m

    for model, feature in m._SKIPS:
        assert (
            model in MODEL_RECIPES
        ), f"skip_reason key ({model!r}, {feature!r}) names an unknown recipe"
