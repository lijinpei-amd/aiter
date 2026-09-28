import torch
import triton

from aiter.ops.triton._triton_kernels.softmax import _softmax_kernel_online
from aiter.ops.triton.utils.device_info import current_device_stream
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def softmax(x):
    """
    Computes row-wise softmax of a 2D input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (n_rows, n_cols). Must be on GPU.

    Returns:
        torch.Tensor: Output with same shape as x, softmax applied along last dimension.
    """
    _LOGGER.info("SOFTMAX: x=%s", tuple(x.shape))
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    num_programs = n_rows

    grid = (num_programs,)
    dev, stream = current_device_stream()
    _softmax_kernel_online(
        dev,
        stream,
        grid,
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE,
    )

    return y
