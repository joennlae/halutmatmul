import timeit
import functools
import pytest
import torch
import numpy as np

try:
    import cupy as cp  # type: ignore[import]

    import halutmatmul.halutmatmul as hm
    from halutmatmul.cuda.kernels import create_encode_kernel
    from halutmatmul.cuda.functions import run_encode_kernel

    def encode_helper(
        N: int, K: int, M: int, C: int, a: float, b: float, device: torch.device
    ) -> None:
        print("========TEST========")
        print(f"params: ({N},{K},{M}), C: {C}, dev: {device}")
        A = (np.random.random((N, K)) + b) * a
        B = (np.random.random((K, M)) + b) * a
        store_array = hm.learn_halut_offline(
            A,
            B,
            C=C,
            lut_work_const=-1,
            quantize_lut=False,
            run_optimized=True,
        )

        A_2_numpy = (np.random.random((N, K)) + b) * a

        res_opt = hm.maddness_encode_opt(
            A_2_numpy, store_array[hm.HalutOfflineStorage.HASH_TABLES]
        )

        offsets = np.arange(C, dtype=np.int32) * 16
        res_opt = res_opt + offsets

        hash_info_fp32 = store_array[hm.HalutOfflineStorage.HASH_TABLES].astype(
            np.float32
        )

        A_2 = torch.from_numpy(A_2_numpy.astype(np.float32)).to(device)
        hash_info = torch.from_numpy(hash_info_fp32).to(device)

        num_splits = hash_info_fp32.shape[1]
        info_offset = hash_info_fp32.shape[2] - 3

        halut_encode_kernel = create_encode_kernel(C, num_splits, info_offset)
        torch_result = run_encode_kernel(
            kernel=halut_encode_kernel, N=N, K=K, A=A_2, hash_info=hash_info, C=C
        )

        numpy_result = torch_result.detach().cpu().numpy()

        try:
            np.testing.assert_allclose(numpy_result, res_opt)
        except Exception as e:
            print(e)

        cpu_time = (
            timeit.Timer(
                functools.partial(
                    hm.maddness_encode_opt,
                    *(A_2_numpy, store_array[hm.HalutOfflineStorage.HASH_TABLES]),
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
            _ = run_encode_kernel(halut_encode_kernel, N, K, A_2, hash_info, C)
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
        "N, K, M, C, a, b",
        [
            (N, K, M, C, a, b)
            for N in [20000, 200000, 1000000]
            for K in [64, 256]
            for M in [64, 128]
            for C in [16, 32, 64]
            for a in [1.0]
            for b in [0.0]
        ],
    )
    def test_encode_kernel(N: int, K: int, M: int, C: int, a: float, b: float) -> None:
        device_id = 1
        if not torch.cuda.is_available():
            pytest.skip("need GPU to run")

        torch.cuda.set_device(1)
        device = torch.device(
            "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
        )

        encode_helper(N, K, M, C, a, b, device)

except ImportError:
    print("not supported without GPU")
