import functools
import timeit
from test.utils.utils import check_if_error_normal_dist_around_zero, error_hist_numpy
import numpy as np
import pytest
import halutmatmul.halutmatmul as hm


def helper_halut(
    N: int = 128,
    D: int = 64,
    M: int = 16,
    C: int = 16,
    lut_work_const: int = -1,
    a: float = 1.0,
    b: float = 0.0,
    K: int = 16,
    quantize_lut: bool = False,
    run_optimized: bool = True,
    encoding_algorithm: int = hm.EncodingAlgorithm.FOUR_DIM_HASH,
) -> None:
    print("=====TEST=====")
    print(
        f"params: ({N}, {D}, {M}), C: {C}, a: {a}, b: {b}, quantize_lut: {quantize_lut}, "
        f"run_optimized: {run_optimized}, K: {K}, encoding_algorithm: {encoding_algorithm}"
    )
    A = (np.random.random((N, D)) + b) * a
    B = (np.random.random((D, M)) + b) * a
    store_array = hm.learn_halut_offline(
        A,
        B,
        C=C,
        K=K,
        lut_work_const=lut_work_const,
        quantize_lut=quantize_lut,
        run_optimized=run_optimized,
        encoding_algorithm=encoding_algorithm,
    )
    new_halut = hm.HalutMatmul()
    new_halut.from_numpy(store_array)

    # time_learning = (
    #     timeit.Timer(functools.partial(hm.learn_halut_offline, *[A, B, C])).timeit(5)
    #     * 1000
    #     / 5
    # )

    # import cProfile
    # from pstats import SortKey
    # with cProfile.Profile() as pr:
    #     hm.learn_halut_offline(A, B, C)
    #     pr.disable()
    #     pr.print_stats(sort=SortKey.CUMULATIVE)

    # print("Time learning: %.2f ms" % (time_learning))
    print(new_halut.stats())
    # print(new_halut.get_params())

    # accuracy test
    A_2 = (np.random.random((N // 4, D)) + b) * a
    res_halut = new_halut.matmul_online(A_2)
    res_numpy = np.matmul(A_2, B)

    error_hist_numpy(res_halut, res_numpy)
    check_if_error_normal_dist_around_zero(res_halut, res_numpy)

    time_halut = (
        timeit.Timer(functools.partial(new_halut.matmul_online, *[A_2])).timeit(5)
        * 1000
        / 5
    )

    time_numpy = (
        timeit.Timer(functools.partial(np.matmul, *[A_2, B])).timeit(5) * 1000 / 5
    )

    print(
        "time calculation numpy/halutmatmul fp: %.2f / %.2f ms"
        % (time_numpy, time_halut)
    )

    mse = np.square(res_halut - res_numpy).mean()
    mae = np.abs(res_halut - res_numpy).mean()
    # pylint: disable=E1307
    print("mse: %.4f / mae: %.4f" % (mse, mae))


@pytest.mark.parametrize(
    "N, D, M, K, C, a, b, encoding_algorithm",
    [
        (N, D, M, K, C, a, b, e)
        for N in [2048]
        for D in [512]
        for M in [64, 128]
        for C in [16, 32, 64]
        for a in [1.0]  # 5.0
        for b in [0.0]
        for e in [
            hm.EncodingAlgorithm.FOUR_DIM_HASH,
            hm.EncodingAlgorithm.DECISION_TREE,
            hm.EncodingAlgorithm.FULL_PQ,
        ]
        for K in ([16] if e == hm.EncodingAlgorithm.FOUR_DIM_HASH else [12, 16, 24])
        # for q in [True, False]
        # for r in [True, False]
    ],
)
def test_learn_offline(
    N: int, D: int, M: int, K: int, C: int, a: float, b: float, encoding_algorithm: int
) -> None:
    np.random.seed(4419)

    quantize_lut = False
    run_optimized = True
    helper_halut(
        N,
        D,
        M,
        C,
        K=K,
        a=a,
        b=b,
        quantize_lut=quantize_lut,
        run_optimized=run_optimized,
        encoding_algorithm=encoding_algorithm,
    )
