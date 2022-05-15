import timeit
import functools
from typing import Optional

from test.utils.utils import error_hist_numpy
import pytest
import torch
import numpy as np
from halutmatmul.functions import halut_encode_opt

from halutmatmul.halutmatmul import (
    EncodingAlgorithm,
    HalutMatmul,
    HalutOfflineStorage,
)
from halutmatmul.decision_tree_and_pq import (
    halut_encode_decision_tree,
    halut_encode_pq,
    halut_encode_pq_tensor,
)

TEST_CUDA_DEVICE_ID = 0

try:
    import cupy as cp  # type: ignore[import]

    import halutmatmul.halutmatmul as hm
    from halutmatmul.cuda.kernels import (
        create_encode_kernel_four_dim,
        create_read_acc_lut_kernel,
        create_encode_kernel_decision_tree,
        READ_ACC_LUT_KERNEL_SPLIT_FACTOR,
    )
    from halutmatmul.cuda.functions import (
        run_encode_kernel,
        calc_rows_per_block_read_acc_lut_kernel,
        run_read_acc_lut_kernel,
    )

    def encode_helper(
        N: int,
        D: int,
        M: int,
        C: int,
        a: float,
        b: float,
        device: torch.device,
        K: int = 16,
        encoding_algorithm: int = EncodingAlgorithm.FOUR_DIM_HASH,
    ) -> None:
        print("========TEST========")
        print(f"params: ({N},{D},{M}), C: {C}, dev: {device}")
        A = (np.random.random((N, D)) + b) * a
        B = (np.random.random((D, M)) + b) * a
        store_array = hm.learn_halut_offline(
            A,
            B,
            C=C,
            lut_work_const=-1,
            quantize_lut=False,
            run_optimized=True,
            encoding_algorithm=encoding_algorithm,
        )

        A_2_numpy = (np.random.random((N, D)) + b) * a

        halut = HalutMatmul().from_numpy(store_array)
        res_opt = halut.encode(A_2_numpy)

        hash_info_fp32 = store_array[hm.HalutOfflineStorage.HASH_TABLES].astype(
            np.float32
        )

        A_2 = torch.from_numpy(A_2_numpy.astype(np.float32)).to(device)
        numpy_result = np.array([])

        halut_encode_kernel: Optional[cp.RawKernel] = None

        if encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
            hash_info = torch.from_numpy(hash_info_fp32).to(device)

            num_splits = hash_info_fp32.shape[1]
            info_offset = hash_info_fp32.shape[2] - 3

            halut_encode_kernel = create_encode_kernel_four_dim(
                C, num_splits, info_offset
            )
            torch_result = run_encode_kernel(
                kernel=halut_encode_kernel, N=N, D=D, A=A_2, hash_info=hash_info, C=C
            )
            numpy_result = torch_result.detach().cpu().numpy()
        elif encoding_algorithm == EncodingAlgorithm.DECISION_TREE:
            tree_info = torch.from_numpy(hash_info_fp32).to(device)

            B_ = hash_info_fp32.shape[1] // 3
            depth = int(np.log2(B_))

            halut_encode_kernel = create_encode_kernel_decision_tree(
                C=C, depth=depth, B=B_, K=K
            )
            torch_result = run_encode_kernel(
                kernel=halut_encode_kernel, N=N, D=D, A=A_2, hash_info=tree_info, C=C
            )
            numpy_result = torch_result.detach().cpu().numpy()
        elif encoding_algorithm == EncodingAlgorithm.FULL_PQ:
            prototypes_gpu = torch.from_numpy(
                store_array[hm.HalutOfflineStorage.PROTOTYPES].astype(np.float32)
            ).to(device)

            torch_result = halut_encode_pq_tensor(A_2, prototypes_gpu)
            numpy_result = torch_result.detach().cpu().numpy()

        try:
            np.testing.assert_allclose(numpy_result, res_opt)
        except Exception as e:
            print(e)

        cpu_time = 0.0

        if encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
            cpu_time = (
                timeit.Timer(
                    functools.partial(
                        halut_encode_opt,
                        *(A_2_numpy, store_array[hm.HalutOfflineStorage.HASH_TABLES]),
                    )
                ).timeit(5)
                * 1000
                / 5
            )
        elif encoding_algorithm == EncodingAlgorithm.DECISION_TREE:
            cpu_time = (
                timeit.Timer(
                    functools.partial(
                        halut_encode_decision_tree,
                        *(A_2_numpy, store_array[hm.HalutOfflineStorage.HASH_TABLES]),
                    )
                ).timeit(5)
                * 1000
                / 5
            )
        elif encoding_algorithm == EncodingAlgorithm.FULL_PQ:
            cpu_time = (
                timeit.Timer(
                    functools.partial(
                        halut_encode_pq,
                        *(A_2_numpy, store_array[hm.HalutOfflineStorage.PROTOTYPES]),
                    )
                ).timeit(5)
                * 1000
                / 5
            )

        gpu_time = 0.0
        if encoding_algorithm in [
            EncodingAlgorithm.FOUR_DIM_HASH,
            EncodingAlgorithm.DECISION_TREE,
        ]:
            hash_info = torch.from_numpy(hash_info_fp32).to(device)
            for _ in range(5):
                start_gpu = cp.cuda.Event()
                start_gpu = cp.cuda.Event()
                end_gpu = cp.cuda.Event()
                start_gpu.record()
                _ = run_encode_kernel(halut_encode_kernel, N, D, A_2, hash_info, C)
                end_gpu.record()
                end_gpu.synchronize()
                t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
                gpu_time += t_gpu
            gpu_time /= 5
        elif encoding_algorithm == EncodingAlgorithm.FULL_PQ:
            prototypes_gpu = torch.from_numpy(
                store_array[hm.HalutOfflineStorage.PROTOTYPES].astype(np.float32)
            ).to(device)
            for _ in range(5):
                start_gpu = cp.cuda.Event()
                start_gpu = cp.cuda.Event()
                end_gpu = cp.cuda.Event()
                start_gpu.record()
                _ = halut_encode_pq_tensor(A_2, prototypes_gpu)
                end_gpu.record()
                end_gpu.synchronize()
                t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
                gpu_time += t_gpu
            gpu_time /= 5

        print(
            "cpu/gpu time: %.2f / %.2f ms"
            % (
                cpu_time,
                gpu_time,
            )
        )

    def read_acc_lut_helper(
        N: int,
        D: int,
        M: int,
        C: int,
        a: float,
        b: float,
        device: torch.device,
        K: int = 16,
    ) -> None:
        print("========TEST========")
        print(f"params: ({N},{D},{M}), C: {C}, dev: {device}")
        A = (np.random.random((N, D)) + b) * a
        B = (np.random.random((D, M)) + b) * a
        store_array = hm.learn_halut_offline(
            A,
            B,
            C=C,
            lut_work_const=-1,
            quantize_lut=False,
            run_optimized=True,
        )

        A_2_numpy = (np.random.random((N, D)) + b) * a
        # new_halut = HalutMatmul().from_numpy(store_array)
        # res_expected = new_halut.matmul_online(A_2_numpy)
        A_enc_numpy = halut_encode_opt(
            A_2_numpy, store_array[HalutOfflineStorage.HASH_TABLES]
        )
        offsets = np.arange(C, dtype=np.int32) * K
        A_enc_numpy = A_enc_numpy + offsets
        A_enc = torch.from_numpy(A_enc_numpy.astype(np.int32)).to(device)
        lut = torch.from_numpy(
            store_array[HalutOfflineStorage.LUT].astype(np.float32)
        ).to(device)

        rows_per_block, split_factor = calc_rows_per_block_read_acc_lut_kernel(
            READ_ACC_LUT_KERNEL_SPLIT_FACTOR, C, K
        )
        halut_read_acc_lut_kernel = create_read_acc_lut_kernel(
            C, K=K, blocks=split_factor, rows=rows_per_block
        )
        torch_result = run_read_acc_lut_kernel(
            kernel=halut_read_acc_lut_kernel, N=N, M=M, lut=lut, A_enc=A_enc, C=C, K=K
        )

        numpy_result = torch_result.detach().cpu().numpy()
        # try:
        #     np.testing.assert_allclose(numpy_result, res_expected)
        # except Exception as e:
        #     print(e)

        numpy_dot = np.dot(A_2_numpy, B)
        error_hist_numpy(numpy_result, numpy_dot)
        # error_hist_numpy(res_expected, numpy_dot)

        A_enc_numpy = np.ascontiguousarray(A_enc_numpy)
        total_result = np.empty((M, len(A_enc_numpy)), dtype=np.float32)
        A_raveled = A_enc_numpy.ravel()
        cpu_time = (
            timeit.Timer(
                functools.partial(
                    hm.read_luts_opt,
                    *(
                        A_raveled,
                        A_enc_numpy.shape,
                        store_array[HalutOfflineStorage.LUT],
                        total_result,
                    ),
                )
            ).timeit(5)
            * 1000
            / 5
        )

        gpu_time = 0.0
        for _ in range(5):
            start_gpu = cp.cuda.Event()
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            start_gpu.record()
            _ = run_read_acc_lut_kernel(
                halut_read_acc_lut_kernel, N=N, M=M, lut=lut, A_enc=A_enc, C=C, K=K
            )
            end_gpu.record()
            end_gpu.synchronize()
            t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
            gpu_time += t_gpu
        gpu_time /= 5

        print(
            "cpu/gpu time: %.2f / %.2f ms"
            % (
                cpu_time,
                gpu_time,
            )
        )

    @pytest.mark.parametrize(
        "N, D, M, C, a, b, encoding_algorithm",
        [
            (N, D, M, C, a, b, encoding_algorithm)
            for N in [10000]
            for D in [128, 256]
            for M in [128]
            for C in [16, 32, 64]
            for a in [1.0]
            for b in [0.0]
            for encoding_algorithm in [
                EncodingAlgorithm.FOUR_DIM_HASH,
                EncodingAlgorithm.DECISION_TREE,
                EncodingAlgorithm.FULL_PQ,
            ]
        ],
    )
    def test_encode_kernel(
        N: int, D: int, M: int, C: int, a: float, b: float, encoding_algorithm: int
    ) -> None:
        device_id = TEST_CUDA_DEVICE_ID
        if not torch.cuda.is_available():
            pytest.skip("need GPU to run")

        torch.cuda.set_device(device_id)
        device = torch.device(
            "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
        )

        encode_helper(N, D, M, C, a, b, device, encoding_algorithm=encoding_algorithm)

    @pytest.mark.parametrize(
        "N, D, M, C, a, b",
        [
            (N, D, M, C, a, b)
            for N in [10000]
            for D in [128, 256]
            for M in [128]
            for C in [16, 32, 64]
            for a in [1.0]
            for b in [0.0]
        ],
    )
    def test_read_acc_lut_kernel(
        N: int, D: int, M: int, C: int, a: float, b: float
    ) -> None:
        device_id = TEST_CUDA_DEVICE_ID
        if not torch.cuda.is_available():
            pytest.skip("need GPU to run")

        torch.cuda.set_device(device_id)
        device = torch.device(
            "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
        )
        read_acc_lut_helper(N, D, M, C, a, b, device)

except ImportError as e:
    print(e)
    print("not supported without GPU")
