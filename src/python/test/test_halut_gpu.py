import functools
import timeit
from test.utils.utils import check_if_error_normal_dist_around_zero, error_hist_numpy
import torch
import numpy as np
import pytest
import halutmatmul.halutmatmul as hm

TEST_CUDA_DEVICE_ID = 0
try:
    import cupy as cp  # type: ignore[import]

    from halutmatmul.cuda.functions import halutmatmul_gpu
    from halutmatmul.cuda.kernels import create_kernels_halutmatmul

    def halut_gpu_helper(
        N: int,
        D: int,
        M: int,
        C: int,
        a: float,
        b: float,
        device: torch.device,
        K: int = 16,
        encoding_algorithm: int = hm.EncodingAlgorithm.FOUR_DIM_HASH,
    ) -> None:
        print("========TEST========")
        print(f"params: ({N},{D},{M}), C: {C}, dev: {str(device)}")
        A = (np.random.random((N, D)) + b) * a
        B = (np.random.random((D, M)) + b) * a
        store_array = hm.learn_halut_offline(
            A,
            B,
            C=C,
            K=K,
            lut_work_const=-1,
            quantize_lut=False,
            run_optimized=True,
            encoding_algorithm=encoding_algorithm,
        )

        learn_halut = hm.HalutMatmul().from_numpy(store_array)

        A_2_numpy = (np.random.random((N, D)) + b) * a

        result_numpy = np.dot(A_2_numpy, B)

        A_2 = A_2_numpy.astype(np.float32)
        if encoding_algorithm in [
            hm.EncodingAlgorithm.FOUR_DIM_HASH,
            hm.EncodingAlgorithm.DECISION_TREE,
        ]:
            hash_info_fp32 = store_array[hm.HalutOfflineStorage.HASH_TABLES]
        elif encoding_algorithm in [hm.EncodingAlgorithm.FULL_PQ]:
            hash_info_fp32 = store_array[hm.HalutOfflineStorage.PROTOTYPES]

        encode_kernel, read_acc_lut_kernel = create_kernels_halutmatmul(
            C, K, encoding_algorithm=encoding_algorithm
        )

        A_2_gpu = torch.from_numpy(A_2).to(device)
        L = torch.from_numpy(
            store_array[hm.HalutOfflineStorage.LUT].astype(np.float32)
        ).to(device)
        H = torch.from_numpy(hash_info_fp32).to(device)

        gpu_result = halutmatmul_gpu(
            encode_kernel=encode_kernel,
            read_acc_lut_kernel=read_acc_lut_kernel,
            A=A_2_gpu,
            L=L,
            H=H,
        )

        gpu_result_numpy = cp.asnumpy(gpu_result)

        error_hist_numpy(gpu_result_numpy, result_numpy)
        check_if_error_normal_dist_around_zero(
            gpu_result_numpy, result_numpy, max_rel_error=0.6
        )

        cpu_time = (
            timeit.Timer(
                functools.partial(
                    learn_halut.matmul_online,
                    *[A_2],
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
            _ = halutmatmul_gpu(
                encode_kernel=encode_kernel,
                read_acc_lut_kernel=read_acc_lut_kernel,
                A=A_2_gpu,
                L=L,
                H=H,
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
        "N, D, M, C, K, a, b, encoding_algorithm",
        [
            (N, D, M, C, K, a, b, e)
            for N in [10000]
            for D in [256]
            for M in [128]
            for C in [16, 32, 64]
            for a in [1.0]  # 5.0
            for b in [0.0]
            for e in [
                hm.EncodingAlgorithm.FOUR_DIM_HASH,
                hm.EncodingAlgorithm.DECISION_TREE,
                hm.EncodingAlgorithm.FULL_PQ,
            ]
            for K in (
                [8, 16, 32]  # 64 uses to much shared memory
                if e == hm.EncodingAlgorithm.FOUR_DIM_HASH
                else [4, 8, 12, 16, 24, 32, 64]
            )
        ],
    )
    def test_halut_gpu(
        N: int,
        D: int,
        M: int,
        C: int,
        K: int,
        a: float,
        b: float,
        encoding_algorithm: int,
    ) -> None:
        np.random.seed(4419)
        torch.manual_seed(4419)
        device_id = TEST_CUDA_DEVICE_ID
        if not torch.cuda.is_available():
            pytest.skip("need GPU to run")
        torch.cuda.set_device(device_id)
        device = torch.device(
            "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
        )
        halut_gpu_helper(
            N,
            D,
            M,
            C,
            a=a,
            b=b,
            device=device,
            K=K,
            encoding_algorithm=encoding_algorithm,
        )

except ImportError as e:
    print(e)
    print("not supported without GPU")
