import torch
import triton
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_preshuffle as triton_gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.triton.gluon.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as gluon_gemm_a8w8_blockscale,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.test_common import checkAllclose
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    run_torch,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object,
    get_shape_benchmark_object,
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    add_argparse_ff,
    get_ff_args,
)
import math

block_shape = (128, 128)


def _generate_inputs(M, N, K, layout, shuffle):
    """Generate inputs using the same recipe as op_tests/test_gemm_a8w8_blockscale.py::test_gemm:

      x:       rand fp32 / 10, cast to aiter.dtypes.fp8, shape (M, K)
      weight:  rand fp32 / 10, cast to aiter.dtypes.fp8, shape (N, K)
      x_scale: rand fp32, shape (M, scale_k)
      w_scale: rand fp32, shape (scale_n, scale_k)
      no fixed seed (the test doesn't set one either)

    The op_test only supports TN; we extend by transposing storage for non-TN
    layouts (same data, different stride) so the bench's --layout still works.

    The shuffle path uses the *triton* preshuffle convention (16x16 shuffle
    with a row-flatten reshape, plus transposed x_scale) because the impls
    being benchmarked are triton/gluon. The CK shuffle from op_tests/test_gemm
    would produce wrong-shaped tensors for these kernels.
    """
    block_shape_n, block_shape_k = block_shape
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float32, device="cuda") / 10).to(
            dtypes.fp8
        )
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float32, device="cuda") / 10)
            .to(dtypes.fp8)
            .T
        )

    if layout[1] == "N":
        weight = (torch.rand((N, K), dtype=torch.float32, device="cuda") / 10).to(
            dtypes.fp8
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float32, device="cuda") / 10)
            .to(dtypes.fp8)
            .T
        )

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
        x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    else:
        weight_shuffled = weight
        x_scale_shuffled = x_scale

    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y


def bench_gemm_fn(
    M: int,
    N: int,
    K: int,
    metric: str,
    layout: str,
    impl: callable,
    shuffle: bool = False,
    test: bool = False,
):
    c_dtype = torch.bfloat16

    x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y = (
        _generate_inputs(M, N, K, layout, shuffle)
    )
    if shuffle:
        bench_weight = weight_shuffled
        bench_x_scale = x_scale_shuffled
    else:
        bench_weight = weight
        bench_x_scale = x_scale

    if test:
        # Correctness check, mirrors op_tests/test_gemm_a8w8_blockscale.py:
        # torch reference runs on un-shuffled inputs, impl runs on whatever
        # layout it expects. checkAllclose's defaults (rtol=1e-2, atol=1e-2)
        # are the same tolerances test_gemm uses.
        ref = run_torch(x, weight, x_scale, w_scale, c_dtype)
        out = impl(x, bench_weight, bench_x_scale, w_scale, c_dtype, y)
        checkAllclose(ref, out, msg=f"M={M},N={N},K={K}")

    # flops
    flops = 2.0 * M * N * K
    # memory transfer
    mem_read = (M * K) * x.element_size() + (N * K) * weight.element_size()
    mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
    mem = mem_read + mem_write

    ms = triton.testing.do_bench(
        lambda: impl(x, bench_weight, bench_x_scale, w_scale, c_dtype, y),  # noqa: E731
        warmup=25,
        rep=100,
    )

    # Return exactly one scalar depending on which metric is active
    if metric == "time":
        return ms
    elif metric == "throughput":
        tflops = flops / ms * 1e-9
        return tflops
    elif metric == "bandwidth":
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
        return bandwidth
    else:
        raise ValueError("Unknown metric: " + metric)


def run_model_benchmark(args, impl, plot_name, shuffle):
    """
    Runs benchmark given a --model argument.
    """
    benchmark = get_model_benchmark_object(plot_name, args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8_blockscale(
        M, hidden_dim, intermediate_dim, metric, layer, model_name=None, **kwargs
    ):
        """
        Fc1:
             M      K                  K           N          M       N
        A = (B, hidden_dim) @ W = (hidden_dim, 2*int_dim) -> (B, 2*int_dim) -> gating -> (B, int_dim)

        Fc2:
             M     K               K          N          M       N
        A = (B, int_dim) @ W = (int_dim, hidden_dim) -> (B, hidden_dim)

        Tensor parallel splits across int_dim (N for fc1, K for fc2)
        """
        if layer == "fc1":
            if args.no_glu:
                N, K = intermediate_dim, hidden_dim
            else:
                N, K = intermediate_dim * 2, hidden_dim
            # Divide N by tensor parallel
            N = math.ceil(N / args.tp)
        elif layer == "fc2":
            N, K = hidden_dim, intermediate_dim
            # Divide K by tensor parallel
            K = math.ceil(K / args.tp)
        # print(f"Layer: {layer}, M: {M}, N: {N}, K: {K}, hidden_dim: {hidden_dim}, intermediate_dim: {intermediate_dim}")

        return bench_gemm_fn(
            M, N, K, metric, args.layout, impl, shuffle=shuffle, test=args.test
        )

    bench_gemm_a8w8_blockscale.run(save_path="." if args.o else None, print_data=True)


def run_shape_benchmark(args, impl, plot_name, shuffle):
    benchmark = get_shape_benchmark_object(plot_name, args)

    @triton.testing.perf_report([benchmark])
    def bench_gemm_a8w8_blockscale(M, N, K, metric, model_name=None, **kwargs):
        # Divide N by tensor parallel
        N = math.ceil(N / args.tp)
        return bench_gemm_fn(
            M, N, K, metric, args.layout, impl, shuffle=shuffle, test=args.test
        )

    bench_gemm_a8w8_blockscale.run(save_path="." if args.o else None, print_data=True)


def _select_impls(args):
    # Each flag activates its own impl; the basic triton impl is the fallback
    # when none was requested. `-gluon -preshuffle` runs both, for comparison.
    impls = []
    if args.preshuffle:
        impls.append(
            ("triton_preshuffle", triton_gemm_a8w8_blockscale_preshuffle, True)
        )
    if args.gluon:
        if not arch_info.is_gluon_avail():
            raise RuntimeError(
                f"-gluon is not available on arch {arch_info.get_arch()!r}."
            )
        impls.append(("gluon", gluon_gemm_a8w8_blockscale, False))
    if not impls:
        impls.append(("triton", triton_gemm_a8w8_blockscale, False))
    return impls


def run_benchmark(args, defaults):
    assert not (args.shape and args.model) or not (
        args.shape and args.M
    ), "User can specify --shape or --model MODEL -M VAL exclusively"

    impls = _select_impls(args)
    base_plot_name = get_caller_name_no_ext()
    multiple_impls = len(impls) > 1

    if args.model:
        unsupported_args = []
        runner = run_model_benchmark
    else:
        unsupported_args = ["fc1", "fc2", "no_glu"]
        runner = run_shape_benchmark
    flag_word = "with" if args.model else "without"
    for arg in unsupported_args:
        if getattr(args, arg, None) != getattr(defaults, arg, None):
            raise Exception(
                f"Argument '{arg}' is not supported for benchmarking {flag_word} the --model flag."
            )
    for name, impl, shuffle in impls:
        plot_name = f"{base_plot_name}_{name}" if multiple_impls else base_plot_name
        if multiple_impls:
            print(f"\n=== Benchmarking impl: {name} ===")
        runner(args, impl, plot_name, shuffle)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="A8W8 GEMM Blockscale")
    parser = add_argparse_ff(parser)
    parser.add_argument(
        "-gluon",
        action="store_true",
        help="Benchmark the Gluon implementation instead of the default triton impl. "
        "Combine with -preshuffle to run both gluon and preshuffle. "
        "(experimental, requires gfx950 and latest Triton from main).",
    )
    parser.add_argument(
        "-preshuffle",
        action="store_true",
        help="Use preshuffle implementation",
    )
    parser.add_argument(
        "-test",
        action="store_true",
        help="Run a correctness check for each benchmarked shape against a "
        "torch reference (mirrors op_tests/test_gemm_a8w8_blockscale.py).",
    )
    return get_ff_args(parser, args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args, defaults = parse_args(args=args)
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        fun = lambda: run_benchmark(parsed_args, defaults)  # noqa: E731
        print_vgpr(fun, get_caller_name_no_ext())
        return
    run_benchmark(parsed_args, defaults)


if __name__ == "__main__":
    main()
