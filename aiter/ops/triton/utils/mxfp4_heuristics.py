# SPDX-License-Identifier: MIT
"""Host-side alignment predicates shared by Triton and Gluon MXFP4 kernels."""


def even_m_n(args, block_m, n, block_n, num_iter=None):
    # CPython 3.10.9's inspect.BlockFinder can truncate decorator source at
    # lambdas following nested parentheses (CPython gh-83035). Keep the
    # predicate outside the decorator so JIT source extraction remains valid.
    block_n_size = args[block_n] * (args[num_iter] if num_iter is not None else 1)
    return args["M"] % args[block_m] == 0 and args[n] % block_n_size == 0


# intj's `make_launcher` lowers a heuristic that reads a caller argument (one
# not keyed elsewhere) to C, and only accepts a lambda or a single-return
# `def` with exactly one parameter, no free variables, reading it as
# `a["literal"]` -- a `functools.partial` of `even_m_n` above is refused
# (`UnsupportedKernel: heuristic 'EVEN_M_N' must be a lambda or def`). These
# four cover every distinct (block_m, n, block_n, num_iter) combination the
# Triton kernels in `_triton_kernels/quant/fused_mxfp4_quant.py` pass to
# `even_m_n`.
def even_m_n1(a):
    return a["M"] % a["BLOCK_SIZE_M"] == 0 and a["N1"] % a["BLOCK_SIZE_N"] == 0


def even_m_n2(a):
    return a["M"] % a["BLOCK_SIZE_M"] == 0 and a["N2"] % a["BLOCK_SIZE_N2"] == 0


def even_m_n3(a):
    return a["M"] % a["BLOCK_SIZE_M"] == 0 and a["N3"] % a["BLOCK_SIZE_N3"] == 0


def even_m_n1_iter(a):
    return (
        a["M"] % a["BLOCK_SIZE_M1"] == 0
        and a["N1"] % (a["BLOCK_SIZE_N1"] * a["NUM_ITER"]) == 0
    )
