from typing import Any, Optional, Union
import numpy as np
import cupy as cp  # type: ignore[import]
import torch
from torch.types import _int, _size

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


def calc_rows_per_block_read_acc_lut_kernel(split_factor: int, C: int, K: int) -> int:
    rows_per_block = MAX_THREADS // split_factor
    used_shared_mem = rows_per_block * C * 4 + C * K * split_factor * 4
    while used_shared_mem > SHARED_MEM_PER_BLOCK:
        rows_per_block //= 2
        used_shared_mem = rows_per_block * C * 4 + C * K * split_factor * 4
    return rows_per_block


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
    rows_per_block = calc_rows_per_block_read_acc_lut_kernel(split_factor, C, K)
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
    rows_per_block_ral = calc_rows_per_block_read_acc_lut_kernel(split_factor, C, K)
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


def calc_newaxes_and_newshape_and_old_cp(
    a: cp.ndarray,
    b: cp.ndarray,
    axes: Union[int, list[int], Any] = 2,
) -> tuple[
    list[int], list[int], tuple[int, int], tuple[int, int], list[int], list[int]
]:
    try:
        iter(axes)  # type: ignore[arg-type]
    # pylint: disable=W0703
    except Exception:
        axes_a = list(range(-axes, 0))  # type: ignore[operator]
        axes_b = list(range(0, axes))  # type: ignore[arg-type]
    else:
        axes_a, axes_b = axes  # type: ignore[misc, assignment]
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]  # type: ignore[list-item]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]  # type: ignore[list-item]
        nb = 1
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    return (newaxes_a, newaxes_b, newshape_a, newshape_b, olda, oldb)


# pylint: disable=R0201
def tensordot_gpu(
    A: cp.ndarray,
    B: cp.ndarray,
    axes: Union[int, list[int], Any],
    encode_kernel: cp.RawKernel,
    read_acc_lut_kernel: cp.RawKernel,
    L: cp.ndarray,
    H: cp.ndarray,
    return_reshaped_inputs: bool = False,
) -> Union[cp.ndarray, tuple[cp.ndarray, cp.ndarray]]:
    # https://github.com/numpy/numpy/blob/145ed90f638c1a12ce5b06e9100421f99783f431/numpy/core/numeric.py#L950

    """Example
    padding=0, kernel_size=(3, 3), stride=1

    IN: (128, 64, 112, 112)
    width: (64, 64, 3, 3)
    after Im2col (np.lib.stride_tricks.as_strided): (128, 64, 110, 110, 3, 3)
    np.tensordot(IN, width, ((1,4,5),(1,2,3)))

    at transpose: (128, 64, 110, 110, 3, 3) -> (128, 110, 110, 64, 3, 3)
    newaxes_a: [0, 2, 3, 1, 4, 5]
    bt transpose: (64, 64, 3, 3) -> (64, 3, 3, 64)
    newaxes_b: [1, 2, 3, 0]
    newshape_a: (1548800, 576)
    newshape_B: (576, 64)

    (1548800, 64) -> (128, 64, 110, 110)
    olda: [128, 110, 110]
    oldb: [64]
    olda + oldb: [128, 110, 110, 64]
    OUT: (128, 110, 110, 64)

    needs to be reshaped later to match conv2d output
    np.moveaxis(ret,4,2).reshape(batch_size, channels_out, out_y, out_x)
    """

    (
        newaxes_a,
        newaxes_b,
        newshape_a,
        newshape_b,
        olda,
        oldb,
    ) = calc_newaxes_and_newshape_and_old_cp(A, B, axes)

    at = A.transpose(newaxes_a).reshape(newshape_a)
    if return_reshaped_inputs:
        bt = B.transpose(newaxes_b).reshape(newshape_b)
        return (at, bt)

    # numpy

    res = halutmatmul_gpu_cupy(
        encode_kernel=encode_kernel,
        read_acc_lut_kernel=read_acc_lut_kernel,
        A=at,
        L=L,
        H=H,
    )
    return res.reshape(olda + oldb)


def halut_conv2d_gpu(
    _input: torch.Tensor,
    weights: torch.Tensor,
    encode_kernel: cp.RawKernel,
    read_acc_lut_kernel: cp.RawKernel,
    L: cp.ndarray,
    H: cp.ndarray,
    kernel_size: Union[_int, _size] = (1, 1),
    stride: Union[_int, _size] = (1, 1),
    padding: Union[_int, _size] = 0,
    groups: int = 1,
    bias: Optional[torch.Tensor] = None,
    return_reshaped_inputs: bool = False,  # needed for storage
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

    input_cupy = cp.asarray(cp.from_dlpack(_input.detach()))
    weights_cupy = cp.asarray(cp.from_dlpack(weights.detach()))
    H_cupy = cp.asarray(cp.from_dlpack(H.detach()))
    L_cupy = cp.asarray(cp.from_dlpack(L.detach()))
    kernel_size = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else (kernel_size[0], kernel_size[1])
    )
    stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    padding = (
        (padding,) * 4
        if isinstance(padding, int)
        else (padding[0], padding[0], padding[1], padding[1])
    )

    if padding[0] > 0 or padding[2] > 0:
        input_cupy = input_cupy[
            :,
            :,
            -padding[2] : input_cupy.shape[2] + padding[3],
            -padding[0] : input_cupy.shape[3] + padding[1],
        ]
    # pylint: disable=C0301
    # inspiration https://github.com/geohot/tinygrad/blob/7ad60eb8b21a3a1f1f538b6e9f216a03d8267e74/tinygrad/ops/ops_cpu.py#L167
    cout, cin, height, width = weights_cupy.shape
    stride_x, stride_y = stride
    batch_size, cin_ = input_cupy.shape[0], input_cupy.shape[1]
    out_y, out_x = (
        (input_cupy.shape[2] - (height - stride_y)) // stride_y,
        (input_cupy.shape[3] - (width - stride_x)) // stride_x,
    )
    assert cin * groups == cin_
    assert cout % groups == 0
    rcout = cout // groups

    input_cupy = input_cupy.reshape(
        batch_size, groups, cin, input_cupy.shape[2], input_cupy.shape[3]
    )

    # im2col
    input_cupy_im2col = cp.lib.stride_tricks.as_strided(
        input_cupy,
        shape=(batch_size, groups, cin, out_y, out_x, height, width),
        strides=(
            *input_cupy.strides[0:3],
            input_cupy.strides[3] * stride_y,
            input_cupy.strides[4] * stride_x,
            *input_cupy.strides[3:5],
        ),
    )
    tensor_weights = weights_cupy.reshape((groups, rcout, cin, height, width))

    ret = cp.zeros((batch_size, groups, out_y, out_x, rcout), dtype=input_cupy.dtype)

    if return_reshaped_inputs:
        (_, _, newshape_a, newshape_b, _, _,) = calc_newaxes_and_newshape_and_old_cp(
            input_cupy_im2col[:, 0], tensor_weights[0], ((1, 4, 5), (1, 2, 3))
        )
        input_a = cp.zeros((groups, *newshape_a))
        input_b = cp.zeros((groups, *newshape_b))
        for g in range(groups):
            (input_a_temp, input_b_temp) = tensordot_gpu(
                input_cupy_im2col[:, 0],
                tensor_weights[0],
                ((1, 4, 5), (1, 2, 3)),
                encode_kernel=encode_kernel,
                read_acc_lut_kernel=read_acc_lut_kernel,
                L=L_cupy,
                H=H_cupy,
                return_reshaped_inputs=return_reshaped_inputs,
            )  # halut does not need to be passed
            input_a[g] += input_a_temp
            input_b[g] += input_b_temp
        return (torch.from_dlpack(input_a[0]), torch.from_dlpack(input_b[0]))
    else:
        for g in range(groups):
            ret[:, g] += tensordot_gpu(
                input_cupy_im2col[:, g],
                tensor_weights[g],
                ((1, 4, 5), (1, 2, 3)),
                encode_kernel=encode_kernel,
                read_acc_lut_kernel=read_acc_lut_kernel,
                L=L_cupy,
                H=H_cupy,
            )

    ret = cp.moveaxis(ret, 4, 2).reshape(batch_size, cout, out_y, out_x)

    if bias is not None:
        bias = cp.broadcast_to(
            cp.repeat(cp.asarray(cp.from_dlpack(bias.detach())), out_y * out_x).reshape(
                (cout, out_y, out_x)
            ),
            (batch_size, cout, out_y, out_x),
        )
        ret = ret + bias
    return torch.from_dlpack(ret)
