from turtle import width
from typing import Any, Optional, Union
import numpy as np
import cupy as cp  # type: ignore[import]
import torch
from torch.nn.common_types import _size_any_t

from halutmatmul.modules import ErrorTuple

MAX_THREADS = 1024
SHARED_MEM_PER_BLOCK = 49152
READ_ACC_LUT_KERNEL_SPLIT_FACTOR = 8


def run_encode_kernel(
    kernel: cp.RawKernel,
    N: int,
    D: int,
    A: torch.Tensor,
    hash_info: torch.Tensor,
    C: int,
) -> torch.Tensor:
    rows_per_block = 64 // (C // 16)
    blocks = N // rows_per_block + (1 if N % rows_per_block else 0)
    block_dim = (rows_per_block, C)
    encoded = cp.zeros((N, C), dtype=cp.int32)
    cupy_A = cp.ascontiguousarray(cp.from_dlpack(A.detach()))
    cupy_hash_info = cp.ascontiguousarray(cp.from_dlpack(hash_info.detach()))
    kernel(
        (blocks,),
        block_dim,
        (cupy_A, cupy_hash_info, encoded, N, D, N * D),
        # shared_mem=4 * (8 + 3) * C * 4,
    )
    torch_A = torch.from_dlpack(encoded)
    return torch_A


def calc_rows_per_block_read_acc_lut_kernel(
    split_factor: int, C: int, K: int
) -> tuple[int, int]:
    rows_per_block = MAX_THREADS // split_factor
    used_shared_mem = rows_per_block * C * 4 + C * K * split_factor * 4
    while used_shared_mem > SHARED_MEM_PER_BLOCK:
        rows_per_block //= 2
        used_shared_mem = rows_per_block * C * 4 + C * K * split_factor * 4
        if rows_per_block == 2:
            split_factor //= 2
            rows_per_block = MAX_THREADS // split_factor
    return (rows_per_block, split_factor)


def run_read_acc_lut_kernel(
    kernel: cp.RawKernel,
    N: int,
    M: int,
    lut: torch.Tensor,
    A_enc: torch.Tensor,
    C: int,
    K: int,
) -> torch.Tensor:
    split_factor = READ_ACC_LUT_KERNEL_SPLIT_FACTOR
    rows_per_block, split_factor = calc_rows_per_block_read_acc_lut_kernel(
        split_factor, C, K
    )
    block_dim = (rows_per_block, split_factor)
    blocks_x = N // rows_per_block + (1 if N % rows_per_block else 0)
    blocks_y = M // split_factor + (1 if M % split_factor else 0)
    grid_dim = (blocks_x, blocks_y)
    result = cp.zeros((N, M), dtype=cp.float32)
    cupy_A_enc = cp.ascontiguousarray(cp.from_dlpack(A_enc.detach()))
    cupy_lut = cp.ascontiguousarray(cp.from_dlpack(lut.detach()))

    used_shared_mem = rows_per_block * C * 4 + C * K * split_factor * 4
    assert used_shared_mem <= SHARED_MEM_PER_BLOCK
    kernel(
        grid_dim,
        block_dim,
        (cupy_lut, cupy_A_enc, result, N, M),
        # shared_mem=4 * (8 + 3) * C * 4,
    )
    torch_res = torch.from_dlpack(result)
    return torch_res


def halutmatmul_gpu(
    encode_kernel: cp.RawKernel,
    read_acc_lut_kernel: cp.RawKernel,
    A: torch.Tensor,
    L: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    # encode
    cupy_A = cp.ascontiguousarray(cp.from_dlpack(A.detach()))
    cupyheight = cp.ascontiguousarray(cp.from_dlpack(H.detach()))
    # read accumulate LUTs
    cupy_L = cp.ascontiguousarray(cp.from_dlpack(L.detach()))

    result = halutmatmul_gpu_cupy(
        encode_kernel=encode_kernel,
        read_acc_lut_kernel=read_acc_lut_kernel,
        A=cupy_A,
        L=cupy_L,
        H=cupyheight,
    )
    torch_res = torch.from_dlpack(result)
    return torch_res


def halutmatmul_gpu_cupy(
    encode_kernel: cp.RawKernel,
    read_acc_lut_kernel: cp.RawKernel,
    A: cp.ndarray,
    L: cp.ndarray,
    H: cp.ndarray,
) -> cp.ndarray:
    N = A.shape[0]
    D = A.shape[1]
    M = L.shape[0]
    C = L.shape[1]
    K = L.shape[2]

    # encode
    rows_per_block_encode = 64 // (C // 16)
    blocks = N // rows_per_block_encode + (1 if N % rows_per_block_encode else 0)
    block_dim_encode = (rows_per_block_encode, C)
    encoded = cp.zeros((N, C), dtype=cp.int32)
    encode_kernel(
        (blocks,),
        block_dim_encode,
        (A, H, encoded, N, D, N * D),
    )

    # read accumulate LUTs
    split_factor = READ_ACC_LUT_KERNEL_SPLIT_FACTOR
    rows_per_block_ral, split_factor = calc_rows_per_block_read_acc_lut_kernel(
        split_factor, C, K
    )
    block_dim_ral = (rows_per_block_ral, split_factor)
    blocks_x = N // rows_per_block_ral + (1 if N % rows_per_block_ral else 0)
    blocks_y = M // split_factor + (1 if M % split_factor else 0)
    grid_dim = (blocks_x, blocks_y)
    result = cp.zeros((N, M), dtype=cp.float32)

    used_shared_mem = rows_per_block_ral * C * 4 + C * K * split_factor * 4
    assert used_shared_mem <= SHARED_MEM_PER_BLOCK
    read_acc_lut_kernel(
        grid_dim,
        block_dim_ral,
        (L, encoded, result, N, M),
    )
    return result


def halut_conv2d_gpu(
    _input: torch.Tensor,
    weights: torch.Tensor,
    encode_kernel: cp.RawKernel,
    read_acc_lut_kernel: cp.RawKernel,
    L: cp.ndarray,
    H: cp.ndarray,
    kernel_size: _size_any_t = (1, 1),
    stride: _size_any_t = (1, 1),
    padding: _size_any_t = 0,
    dilation: _size_any_t = 1,
    bias: Optional[torch.Tensor] = None,
    return_reshaped_inputs: bool = False,  # needed for storage
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    unfold_ops = torch.nn.Unfold(
        kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    unfolded = unfold_ops(_input).transpose(1, 2)
    unfolded = torch.reshape(unfolded, (-1, unfolded.size(2)))

    if return_reshaped_inputs:
        weights_prepared = weights.view(weights.size(0), -1).t()
        return (unfolded, weights_prepared)

    unfolded_cupy = cp.asarray(cp.from_dlpack(unfolded.detach()))
    H_cupy = cp.asarray(cp.from_dlpack(H.detach()))
    L_cupy = cp.asarray(cp.from_dlpack(L.detach()))
    ret = halutmatmul_gpu_cupy(
        encode_kernel=encode_kernel,
        read_acc_lut_kernel=read_acc_lut_kernel,
        A=unfolded_cupy,
        L=L_cupy,
        H=H_cupy,
    )

    batch_size = _input.size(0)
    result_tensor = torch.from_dlpack(ret)
    result_tensor = torch.reshape(
        result_tensor, (batch_size, -1, result_tensor.size(1))
    ).transpose(1, 2)

    stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    dilation = (
        (dilation, dilation)
        if isinstance(dilation, int)
        else (dilation[0], dilation[1])
    )
    padding = (
        (padding,) * 4
        if isinstance(padding, int)
        else (padding[0], padding[0], padding[1], padding[1])
    )

    if padding[0] > 0 or padding[2] > 0:
        _input = _input[
            :,
            :,
            -padding[2] : _input.shape[2] + padding[3],
            -padding[0] : _input.shape[3] + padding[1],
        ]
    cout, _, kernel_height, kernel_width = weights.shape
    stride_y, stride_x = stride
    out_y, out_x = (
        (
            (_input.shape[2] + padding[3] + padding[2])
            - dilation[0] * (kernel_height - stride_y)
            - 1
        )
        // stride_y
        + 1,
        (
            (_input.shape[3] + padding[0] + padding[1])
            - dilation[1] * (kernel_width - stride_x)
            - 1
        )
        // stride_x
        + 1,
    )
    ret = torch.reshape(result_tensor, (batch_size, cout, out_y, out_x))

    if bias is not None:
        bias = torch.broadcast_to(
            bias.repeat(out_y * out_x).reshape((cout, out_y, out_x)),
            (batch_size, cout, out_y, out_x),
        )
        ret = ret + bias
    return ret


def error_cupy(
    actual: torch.Tensor,
    desired: torch.Tensor,
) -> np.ndarray:
    actual_cupy = cp.asarray(cp.from_dlpack(actual.detach()))
    desired_cupy = cp.asarray(cp.from_dlpack(desired.detach()))
    _min = cp.min(desired_cupy)
    _max = cp.max(desired_cupy)
    actual_cupy_std = (actual_cupy - _min) / (_max - _min)
    desired_cupy_std = (desired_cupy - _min) / (_max - _min)
    _range = (-1, 1)
    actual_cupy_scaled = actual_cupy_std * (_range[1] - _range[0]) + _range[0]
    desired_cupy_scaled = desired_cupy_std * (_range[1] - _range[0]) + _range[0]
    mae = cp.asnumpy(cp.mean(cp.abs((actual_cupy - desired_cupy))))
    mse = cp.asnumpy(cp.mean((actual_cupy - desired_cupy) ** 2))
    mape = cp.asnumpy(
        cp.mean(cp.abs(actual_cupy - desired_cupy) / (1 + cp.abs(desired_cupy)))
    )
    scaled_absolut_error = cp.asnumpy(
        cp.mean(cp.abs(actual_cupy_scaled - desired_cupy_scaled))
    )
    scaled_shift = cp.asnumpy(cp.mean(actual_cupy_scaled - desired_cupy_scaled))

    return np.array((mae, mse, mape, scaled_absolut_error, scaled_shift))
