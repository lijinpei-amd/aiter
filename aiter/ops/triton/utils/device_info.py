import functools

import torch


@functools.lru_cache(maxsize=1)
def get_num_sms():
    # Returns the Compute Unit count of the device.
    #
    # Prefer chip_info.get_cu_num(): it honors the CU_NUM env override and is the
    # same value the tuning dispatch keys (gfx, cu_num, M, N, K) are built from,
    # so grid/segment sizing stays consistent with the selected tuned configs.
    # Fall back to torch's multi_processor_count when get_cu_num() is unavailable
    # (e.g. rocminfo missing/unparseable).
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return get_cu_num()
    except Exception:  # noqa: BLE001
        import torch

        current_device_index = torch.cuda.current_device()
        current_device = torch.cuda.get_device_properties(current_device_index)
        return current_device.multi_processor_count


def get_num_xcds():
    # Currently, you can't query this programmatically. For gfx942/gfx950 it's 8, so we hardcode that here.
    return 8


_raw_stream = torch._C._cuda_getCurrentRawStream  # what Triton's driver reads


def current_device_stream():
    """The current device ordinal and its raw stream, as Triton launches on:
    the first two arguments of an intj launcher call."""
    device = torch.cuda.current_device()
    return device, _raw_stream(device)


def launch_on_current_stream(kernel, grid, *args):
    """Calls intj launcher `kernel` as `kernel(device, stream, grid, *args)`
    on the current device and stream.

    While torch.compile traces, launches `kernel`'s Triton function instead,
    with the launcher's `options` and `dynamic_options`: Dynamo traces a
    Triton launch into the graph, but not an intj launcher (a C function), and
    a graph break fails `fullgraph=True`. `args` are the launcher's arguments
    after `grid`: its `dynamic_options` values, then the kernel arguments.
    For a launcher that takes `grid` as one tuple and has no `delayed=`,
    `bind_values` or knob in `dynamic_options`.
    """
    if torch.compiler.is_compiling():
        option = kernel.__self__.options
        names = option.dynamic_options
        option.jit_func[grid](
            *args[len(names) :],
            **(option.options or {}),
            **dict(zip(names, args)),
        )
        return
    device = torch.cuda.current_device()
    kernel(device, _raw_stream(device), grid, *args)
