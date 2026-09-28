# SPDX-License-Identifier: MIT
"""Host-side MXFP4 import regressions, including CPython 3.10.9 inspect.

No GPU launch is required. The AST check protects older supported interpreters
even when CI runs on a newer interpreter with a fixed inspect implementation.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

GLUON = "aiter.ops.triton._gluon_kernels.gfx1250.quant.fused_mxfp4_quant"
TRITON = "aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant"
PUBLIC_WRAPPER = "aiter.ops.triton.quant.fused_mxfp4_quant"
KERNELS = [
    pytest.param(GLUON, "_gluon_fused_rms_mxfp4_quant_kernel", 0, id="gluon-rms"),
    pytest.param(
        GLUON, "_gluon_fused_reduce_rms_mxfp4_quant_kernel", 2, id="gluon-reduce"
    ),
    pytest.param(TRITON, "_fused_rms_mxfp4_quant_kernel", 1, id="triton-rms"),
    pytest.param(
        TRITON,
        "_fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel",
        3,
        id="triton-act-num-iter",
    ),
    pytest.param(TRITON, "_fused_reduce_rms_mxfp4_quant_kernel", 2, id="triton-reduce"),
]

# Literal truth tables, independent of the implementation formula.
# Expected groups: Gluon RMS, Triton RMS, reduce RMS, iterative activation.
# Each group's order: EVEN_M_N, EVEN_M_N2, EVEN_M_N3 (where applicable).
CASES = [
    pytest.param(
        (32, 48, 64, 96, 3),
        ((True,), (True, True), (True, True, True), (True,)),
        id="all-aligned",
    ),
    pytest.param(
        (8, 48, 64, 96, 3),
        ((True,), (False, False), (False, False, False), (False,)),
        id="only-rows-per-cta-aligned",
    ),
    pytest.param(
        (16, 48, 64, 96, 3),
        ((True,), (True, True), (True, True, True), (False,)),
        id="block-m1-unaligned",
    ),
    pytest.param(
        (33, 48, 64, 96, 3),
        ((False,), (False, False), (False, False, False), (False,)),
        id="m-unaligned",
    ),
    pytest.param(
        (32, 40, 64, 96, 1),
        ((False,), (False, True), (False, True, True), (True,)),
        id="n1-eight-aligned-iter1",
    ),
    pytest.param(
        (32, 40, 64, 96, 3),
        ((False,), (False, True), (False, True, True), (False,)),
        id="n1-eight-aligned-iter3",
    ),
    pytest.param(
        (32, 16, 64, 96, 3),
        ((True,), (True, True), (True, True, True), (False,)),
        id="n1-sixteen-not-24-aligned",
    ),
    pytest.param(
        (32, 48, 65, 96, 3),
        ((True,), (True, False), (True, False, True), (True,)),
        id="n2-unaligned",
    ),
    pytest.param(
        (32, 48, 64, 97, 3),
        ((True,), (True, True), (True, True, False), (True,)),
        id="n3-unaligned",
    ),
    pytest.param(
        (0, 0, 0, 0, 3),
        ((True,), (True, True), (True, True, True), (True,)),
        id="zero-predicate-only",
    ),
]


@pytest.mark.parametrize("module_name,kernel_name,kind", KERNELS)
def test_jit_import_and_complete_source(module_name, kernel_name, kind):
    kernel = getattr(importlib.import_module(PUBLIC_WRAPPER), kernel_name)
    # Heuristics -> JITFunction -> original Python function.
    assert "def " + kernel_name in inspect.getsource(kernel.fn.fn)


@pytest.mark.parametrize("module_name,kernel_name,kind", KERNELS)
def test_decorators_remain_safe_for_legacy_inspect(module_name, kernel_name, kind):
    # No AITER import: detects reintroduced lambdas on Python 3.10.12/3.12 too.
    path = (
        Path(__file__)
        .resolve()
        .parents[3]
        .joinpath(*module_name.split("."))
        .with_suffix(".py")
    )
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    assert not any(
        isinstance(node, ast.Lambda)
        for decorator in fn.decorator_list
        for node in ast.walk(decorator)
    )


@pytest.mark.parametrize("module_name,kernel_name,kind", KERNELS)
@pytest.mark.parametrize("dimensions,expected", CASES)
def test_alignment_truth_table(module_name, kernel_name, kind, dimensions, expected):
    kernel = getattr(importlib.import_module(PUBLIC_WRAPPER), kernel_name)
    args = dict(zip(("M", "N1", "N2", "N3", "NUM_ITER"), dimensions))
    args.update(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_M1=32,
        ROWS_PER_CTA=8,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_N1=8,
        BLOCK_SIZE_N2=32,
        BLOCK_SIZE_N3=48,
    )
    names = ("EVEN_M_N", "EVEN_M_N2", "EVEN_M_N3")[: len(expected[kind])]
    assert set(kernel.values) == set(names)
    for name, value in zip(names, expected[kind]):
        predicate = kernel.values[name]
        # intj's make_launcher only accepts a lambda or single-return def
        # with no free variables, so both the Triton and the Gluon kernels use
        # module-level defs (`even_m_n1`/`even_m_n1_rows`/`even_m_n2`/
        # `even_m_n3`/`even_m_n1_iter`) instead of a
        # `functools.partial(even_m_n, ...)`.
        assert predicate.__code__.co_freevars == ()
        assert predicate(args) is value, name


def test_new_heuristics_match_old_partial_style():
    """The literal-key defs must agree with the shared, closure-based
    `even_m_n` they replaced on every combination they can be asked about."""
    from aiter.ops.triton.utils.mxfp4_heuristics import (
        even_m_n,
        even_m_n1,
        even_m_n1_iter,
        even_m_n1_rows,
        even_m_n2,
        even_m_n3,
    )

    values = [0, 1, 3, 7, 8, 15, 16, 31, 32, 63, 64]
    for m in values:
        for n in values:
            for block_m in (1, 8, 16, 32):
                for block_n in (1, 8, 16, 32):
                    for num_iter in (1, 2, 3):
                        args = {
                            "M": m,
                            "N1": n,
                            "N2": n,
                            "N3": n,
                            "NUM_ITER": num_iter,
                            "BLOCK_SIZE_M": block_m,
                            "BLOCK_SIZE_M1": block_m,
                            "ROWS_PER_CTA": block_m,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_N1": block_n,
                            "BLOCK_SIZE_N2": block_n,
                            "BLOCK_SIZE_N3": block_n,
                        }
                        assert even_m_n1(args) == even_m_n(
                            args, block_m="BLOCK_SIZE_M", n="N1", block_n="BLOCK_SIZE_N"
                        )
                        assert even_m_n1_rows(args) == even_m_n(
                            args, block_m="ROWS_PER_CTA", n="N1", block_n="BLOCK_SIZE_N"
                        )
                        assert even_m_n2(args) == even_m_n(
                            args,
                            block_m="BLOCK_SIZE_M",
                            n="N2",
                            block_n="BLOCK_SIZE_N2",
                        )
                        assert even_m_n3(args) == even_m_n(
                            args,
                            block_m="BLOCK_SIZE_M",
                            n="N3",
                            block_n="BLOCK_SIZE_N3",
                        )
                        assert even_m_n1_iter(args) == even_m_n(
                            args,
                            block_m="BLOCK_SIZE_M1",
                            n="N1",
                            block_n="BLOCK_SIZE_N1",
                            num_iter="NUM_ITER",
                        )
