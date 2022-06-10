import math
from typing import Optional, Union
import cupy as cp  # type: ignore[import]
import torch
import numpy as np

from torch.nn.common_types import _size_any_t

from halutmatmul.decision_tree_and_pq import halut_encode_pq_tensor


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


def halut_encode_pq_tensor_interface(
    _blocks: tuple,
    _block_dim_encode: tuple,
    args: tuple[cp.ndarray, cp.ndarray, cp.ndarray, int, int, int],
) -> None:
    A, H, encoded, N, D, _ = args
    C = encoded.shape[1]
    encoded_result = halut_encode_pq_tensor(
        torch.reshape(torch.from_dlpack(A), (N, -1)),
        torch.reshape(torch.from_dlpack(H), (C, -1, D)),
    )
    cp.copyto(encoded, cp.from_dlpack(encoded_result.detach()))


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
    rows_per_block_encode = 64 // ((C // 16) if C >= 16 else 1)
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
    L: torch.Tensor,
    H: torch.Tensor,
    kernel_size: _size_any_t = (1, 1),
    stride: _size_any_t = (1, 1),
    padding: _size_any_t = 0,
    dilation: _size_any_t = 1,
    groups: int = 1,
    bias: Optional[torch.Tensor] = None,
    return_reshaped_inputs: bool = False,  # needed for storage
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    unfold_ops = torch.nn.Unfold(
        kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
    )

    n_size_per_channel = 0
    if groups == 1:
        unfolded = unfold_ops(_input).transpose(1, 2)
        unfolded = torch.reshape(unfolded, (-1, unfolded.size(2)))
    else:
        channels_per_group = weights.shape[1]
        unfolded = unfold_ops(_input[:, :channels_per_group]).transpose(1, 2)
        unfolded = torch.reshape(unfolded, (-1, unfolded.size(2)))
        n_size_per_channel = unfolded.shape[0]
        d_size = unfolded.shape[1]
        result = torch.zeros(
            (groups * n_size_per_channel, d_size), device=weights.device
        )
        result[0:n_size_per_channel] = unfolded
        for g in range(1, groups):
            unfolded = unfold_ops(
                _input[:, g * channels_per_group : (g + 1) * channels_per_group]
            ).transpose(1, 2)
            unfolded = torch.reshape(unfolded, (-1, unfolded.size(2)))
            result[g * n_size_per_channel : (g + 1) * n_size_per_channel] = unfolded
        unfolded = result

    if return_reshaped_inputs:
        weights_prepared = weights.view(weights.size(0), -1).t()
        return (unfolded, weights_prepared)

    unfolded_cupy = cp.asarray(cp.from_dlpack(unfolded.detach()))
    H_cupy = cp.asarray(cp.from_dlpack(H.detach()))
    L_cupy = cp.asarray(cp.from_dlpack(L.detach()))

    if groups == 1:
        ret = halutmatmul_gpu_cupy(
            encode_kernel=encode_kernel,
            read_acc_lut_kernel=read_acc_lut_kernel,
            A=unfolded_cupy,
            L=L_cupy,
            H=H_cupy,
        )
    elif groups > 1:
        # TODO: could be heavily parallelized :-)
        channels_per_group = weights.shape[1]
        M = L_cupy.shape[0]
        assert M % groups == 0
        L_per_channel = M // groups
        ret = halutmatmul_gpu_cupy(
            encode_kernel=encode_kernel,
            read_acc_lut_kernel=read_acc_lut_kernel,
            A=unfolded_cupy[:n_size_per_channel],
            L=L_cupy[:L_per_channel],
            H=H_cupy,
        )

        output_n = ret.shape[0]
        ret_all = torch.zeros((groups * output_n, ret.shape[1]), device=weights.device)
        ret_all = cp.asarray(cp.from_dlpack(ret_all.detach()))
        ret_all[:output_n] = ret

        for g in range(g, groups):
            ret = halutmatmul_gpu_cupy(
                encode_kernel=encode_kernel,
                read_acc_lut_kernel=read_acc_lut_kernel,
                A=unfolded_cupy[g * n_size_per_channel : (g + 1) * n_size_per_channel],
                L=L_cupy[g * L_per_channel : (g + 1) * L_per_channel],
                H=H_cupy,
            )
            ret_all[g * output_n : (g + 1) * output_n] = ret
            ret = ret_all

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

    cout, _, kernel_height, kernel_width = weights.shape
    stride_y, stride_x = stride
    # reference https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    out_y, out_x = (
        math.floor(
            (
                (_input.shape[2] + padding[3] + padding[2])
                - dilation[0] * (kernel_height - 1)
                - 1
            )
            / stride_y
            + 1
        ),
        math.floor(
            (
                (_input.shape[3] + padding[0] + padding[1])
                - dilation[1] * (kernel_width - 1)
                - 1
            )
            / stride_x
            + 1
        ),
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


def halut_linear_gpu(
    _input: torch.Tensor,
    encode_kernel: cp.RawKernel,
    read_acc_lut_kernel: cp.RawKernel,
    L: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    input_cupy = cp.asarray(cp.from_dlpack(_input.detach()))
    H_cupy = cp.asarray(cp.from_dlpack(H.detach()))
    L_cupy = cp.asarray(cp.from_dlpack(L.detach()))
    ret = halutmatmul_gpu_cupy(
        encode_kernel=encode_kernel,
        read_acc_lut_kernel=read_acc_lut_kernel,
        A=input_cupy,
        L=L_cupy,
        H=H_cupy,
    )
    result_tensor = torch.from_dlpack(ret)
    return result_tensor
