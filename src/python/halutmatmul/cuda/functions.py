import cupy as cp  # type: ignore[import]
import torch


def run_encode_kernel(
    kernel: cp.RawKernel,
    N: int,
    K: int,
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
        (cupy_A, cupy_hash_info, encoded, N, K, N * K),
        # shared_mem=4 * (8 + 3) * C * 4,
    )
    torch_A = torch.from_dlpack(encoded)
    return torch_A
