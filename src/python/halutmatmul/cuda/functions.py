import cupy as cp  # type: ignore[import]
import torch

from halutmatmul.cuda.kernels import READ_ACC_LUT_KERNEL_SPLIT_FACTOR

MAX_THREADS = 1024
SHARED_MEM_PER_BLOCK = 49152


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
