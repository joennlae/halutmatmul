# pylint: disable=C0413, E1133
# heavily inspired from https://github.com/dblalock/bolt
from __future__ import annotations
from functools import reduce
from typing import Any, Dict, Optional
from sklearn.cluster import KMeans
import faiss

import numpy as np

import numba

from halutmatmul.functions import (
    get_str_hash_buckets,
    halut_encode_opt,
    read_luts_opt,
    read_luts_quantized_opt,
)
from halutmatmul.maddness_legacy import (
    learn_proto_and_hash_function,
    maddness_lut,
    maddness_quantize_luts,
)


class HalutOfflineStorage:
    HASH_TABLES = 0
    LUT = 1
    CONFIG = 2
    PROTOTYPES = 3
    THRESHOLDS = 4
    DIMS = 5
    SIMPLE_PROTOTYPES = 6
    SIMPLE_LUT = 7
    MAX = 8


class ProtoHashReport:
    MSE_ERROR = 0
    MSV_ORIG = 1
    MSE_ERROR_DIV_MSV_ORIG = 2
    MEAN_X = 3
    MSE_RES = 4
    MSE_RES_DIV_MSV_ORIG = 5
    RAM_USAGE = 6


class EncodingAlgorithm:
    FOUR_DIM_HASH = 0
    DECISION_TREE = 1
    FULL_PQ = 2


class HalutModuleConfig:
    C = 0
    K = 1
    LOOP_ORDER = 2
    USE_PROTOTYPES = 3
    MAX = 4


def learn_halut_offline_report(
    A: np.ndarray,
    B: np.ndarray,
    C: int = 16,
    K: int = 16,
    lut_work_const: int = -1,
    quantize_lut: bool = False,
    run_optimized: bool = True,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mn = HalutMatmul(
        C,
        K=K,
        lut_work_const=lut_work_const,
        quantize_lut=quantize_lut,
        run_optimized=run_optimized,
    )
    mn.learn_offline(A, B)

    # print(mn.get_params())
    # print(mn.get_stats())

    return mn.to_numpy(), mn.get_stats()


def learn_halut_offline(
    A: np.ndarray,
    B: np.ndarray,
    C: int = 16,
    K: int = 16,
    lut_work_const: int = -1,
    quantize_lut: bool = False,
    run_optimized: bool = True,
    niter=2,
    nredo=1,
    min_points_per_centroid=100,
    max_points_per_centroid=1000,
    codebook: int = -1,
    already_learned: Optional[Any] = None,
    only_prototypes: bool = False,  # use to only learn prototype for non maddness versions
) -> np.ndarray:
    mn = HalutMatmul(
        C,
        K=K,
        lut_work_const=lut_work_const,
        quantize_lut=quantize_lut,
        run_optimized=run_optimized,
    )
    if already_learned is not None:
        mn.from_numpy(already_learned)
    mn.learn_offline(
        A,
        B,
        niter=niter,
        nredo=nredo,
        only_prototypes=only_prototypes,
        min_points_per_centroid=min_points_per_centroid,
        max_points_per_centroid=max_points_per_centroid,
        codebook=codebook,
    )
    return mn.to_numpy()


class HalutMatmul:
    def __init__(
        self,
        C: int = 16,
        K: int = 16,
        lut_work_const: int = -1,
        quantize_lut: bool = False,
        run_optimized: bool = True,
    ) -> None:
        self.C = C
        self.K = K
        self.prototypes: np.ndarray = np.array([])
        self.luts: np.ndarray = np.array([])
        self.optimized = run_optimized
        self.thresholds: np.ndarray = np.array([])
        self.dims: np.ndarray = np.array([])
        self.simple_k_mean_prototypes: np.ndarray = np.array([])
        self.simple_lut: np.ndarray = np.array([])

        self.encoding_function = halut_encode_opt
        self.learning_function = learn_proto_and_hash_function

        self.lut_work_const = lut_work_const
        self.A_enc: np.ndarray = np.array([])

        # EncodingAlgorithm.FOUR_DIM_HASH
        self.splits_lists: np.ndarray = np.array([])

        self.quantize_lut = quantize_lut
        self.upcast_every = 1
        self.upcast_every = min(self.C, self.upcast_every)
        self.offset: float = 0.0
        self.scale: float = 1.0
        # important otherwise wrong summation
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = "mean"  # sum

        self.stats_dict: Dict[str, Any] = dict([])

    def __repr__(self) -> str:
        return f"<HalutMatmul {self.get_params()}>"

    def __str__(self) -> str:
        return self.get_params()

    def get_stats(self) -> Dict[str, Any]:
        return self.stats_dict

    def get_params(self) -> str:
        params = "=============== \nHalutmatmul parameters\n"
        params += f"C: {self.C}, K: {self.K}, lut_work_const: {self.lut_work_const} \n"
        params += f"is_learned: {self.is_learned()} \n"

        hash_bucket_strings = ""
        if self.splits_lists is not None:
            if self.prototypes.size > 0:
                D = self.prototypes.shape[2]
            i = 0
            for c in self.splits_lists:
                if self.prototypes.size > 0:
                    hash_bucket_strings += (
                        f"Bucket {i} dims: "
                        f"{int(i * D / self.C)} - {int((i + 1) * D / self.C - 1)} \n"
                    )
                hash_bucket_strings += get_str_hash_buckets(c) + "\n"
                i += 1
        params += (
            f"split_lists: {len(self.splits_lists)}, "
            f"hash_buckets for prototypes: \n"
            f"{hash_bucket_strings} \n"
        )
        if self.prototypes.size > 0:
            params += (
                f"prototypes: {self.prototypes.shape}, " f"{self.prototypes.dtype} \n"
            )
        params += (
            f"luts: {self.luts.shape}, "
            f"{self.luts.dtype if self.luts is not None else ''} \n"
        )
        params += f"lut_offset: {self.offset}, lut_scale: {self.scale} \n"
        params += (
            f"quantize_lut: {self.quantize_lut}, upcast_every: {self.upcast_every} \n"
        )
        params += "===============\n"
        return params

    def is_learned(self) -> bool:
        return (
            self.splits_lists is not None
            # and self.prototypes is not None
            and self.luts is not None
            and self.offset is not None
            and self.scale is not None
        )

    def learn_hash_buckets_and_prototypes(self, A: np.ndarray) -> None:
        D = A.shape[1]
        if D < self.C:
            print("D < C: {} < {}".format(D, self.C))
            print("Autocorrecting C == D == ", D)
            self.C = D
        (
            split_lists,
            self.prototypes,
            report_array,
            self.thresholds,
            self.dims,
        ) = self.learning_function(
            A, self.C, self.K, lut_work_const=self.lut_work_const
        )  # type: ignore[operator]
        self.splits_lists = split_lists

        self.stats_dict["MSE_ERROR"] = report_array[ProtoHashReport.MSE_ERROR]
        self.stats_dict["MSV_ORIG"] = report_array[ProtoHashReport.MSV_ORIG]
        self.stats_dict["MSE_ERROR_DIV_MSV_ORIG"] = report_array[
            ProtoHashReport.MSE_ERROR_DIV_MSV_ORIG
        ]
        self.stats_dict["MSE_RES"] = report_array[ProtoHashReport.MSE_RES]
        self.stats_dict["MEAN_X"] = report_array[ProtoHashReport.MEAN_X]
        self.stats_dict["MSE_RES_DIV_MSV_ORIG"] = report_array[
            ProtoHashReport.MSE_RES_DIV_MSV_ORIG
        ]
        self.stats_dict["RAM_USAGE"] = report_array[ProtoHashReport.RAM_USAGE]

    def _check_if_learned(self) -> None:
        if not self.is_learned():
            raise Exception("Halut online tried but not learned!")

    def to_numpy(self) -> np.ndarray:
        self._check_if_learned()
        splits = self.splits_lists

        store_array = np.array(
            [
                splits.astype(np.float32),
                self.luts.astype(np.float32),
                np.array(
                    [
                        self.offset,
                        self.scale,
                        self.optimized,
                        self.quantize_lut,
                        self.upcast_every,
                    ],
                    dtype=np.float32,
                ),
                self.prototypes.astype(np.float32),
                self.thresholds.astype(np.float32),
                self.dims.astype(np.float32),
                self.simple_k_mean_prototypes.astype(np.float32),
                self.simple_lut.astype(np.float32),
            ],
            dtype=object,
        )
        return store_array

    def from_numpy(self, numpy_array: np.ndarray) -> HalutMatmul:
        self.encoding_function = halut_encode_opt
        self.learning_function = learn_proto_and_hash_function

        splits_numpy = numpy_array[HalutOfflineStorage.HASH_TABLES]
        self.splits_lists = splits_numpy

        self.luts = numpy_array[HalutOfflineStorage.LUT]

        self.prototypes = numpy_array[HalutOfflineStorage.PROTOTYPES]
        self.simple_k_mean_prototypes = numpy_array[
            HalutOfflineStorage.SIMPLE_PROTOTYPES
        ]
        self.simple_lut = numpy_array[HalutOfflineStorage.SIMPLE_LUT]
        _, C, K = self.luts.shape
        self.C = C
        self.K = K
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        return self

    # redefinition for convenience public function
    def learn_A(self, A: np.ndarray) -> None:
        self.learn_hash_buckets_and_prototypes(A)

    def learn_offline(
        self,
        A: np.ndarray,
        B: np.ndarray,
        only_prototypes=False,  # default learn prototypes + maddness
        niter=2,
        nredo=1,
        min_points_per_centroid=100,
        max_points_per_centroid=1000,
        codebook: int = -1,
    ) -> None:
        self.learn_simple_k_means_prototypes(
            A,
            niter=niter,
            nredo=nredo,
            min_points_per_centroid=min_points_per_centroid,
            max_points_per_centroid=max_points_per_centroid,
            codebook=codebook,
        )
        self.calculate_simple_lut(B)
        if not only_prototypes:
            self.learn_hash_buckets_and_prototypes(A)
            self._set_B(B)
        else:
            self.luts = np.zeros((B.shape[1], self.C, self.K), dtype=np.float32)
            self.thresholds = np.zeros(
                (B.shape[1], self.C), dtype=np.float32
            )  # shape wrong but not used
            self.dims = np.zeros(
                (B.shape[1], self.C), dtype=np.float32
            )  # shape wrong but not used
            self.offset = 0
            self.scale = 1
        self._check_if_learned()

    def learn_simple_k_means_prototypes(
        self,
        A: np.ndarray,
        niter=2,
        nredo=1,
        min_points_per_centroid=100,
        max_points_per_centroid=1000,
        codebook: int = -1,
    ) -> None:
        print("Learning simple k-means prototypes", A.shape)
        assert A.shape[1] % self.C == 0
        if len(self.simple_k_mean_prototypes.shape) <= 1:
            print("Initializing simple k-means prototypes with zero")
            self.simple_k_mean_prototypes = np.zeros(
                (self.C, self.K, A.shape[1] // self.C), dtype=np.float32
            )
        subsampled = A.astype(np.float32)
        use_kmeans = False
        if use_kmeans:
            subsampled = subsampled.reshape((A.shape[0], self.C, -1))
            for c in range(self.C):
                if codebook > -1 and c != codebook:
                    continue
                print("Learning simple k-means prototypes for channel {}".format(c))
                kmeans = faiss.Kmeans(
                    subsampled.shape[2],
                    self.K,
                    niter=niter,
                    verbose=True,
                    nredo=nredo,
                    # seed=4419,
                    seed=np.random.randint(1, 2**31 - 1),
                    min_points_per_centroid=min_points_per_centroid,
                    max_points_per_centroid=max_points_per_centroid,
                )
                kmeans.train(subsampled[:, c, :])
                centroids_kmeans = kmeans.centroids
                self.simple_k_mean_prototypes[c] = centroids_kmeans
        else:
            nbit = 4
            pq = faiss.ProductQuantizer(subsampled.shape[1], self.C, nbit)
            pq.verbose = True
            # pylint: disable=no-value-for-parameter
            pq.train(subsampled)
            d = subsampled.shape[1] // self.C
            centroids = faiss.vector_to_array(pq.centroids)
            centroids = centroids.reshape((self.C, 1 << nbit, d))
            print(
                "centroids",
                centroids.shape,
                self.simple_k_mean_prototypes.shape,
                centroids[0],
            )
            self.simple_k_mean_prototypes = centroids
        print("Done learning simple k-means prototypes")

    def calculate_simple_lut(self, B) -> None:
        self.simple_lut = np.zeros((B.shape[1], self.C, self.K), dtype=np.float32)
        B_reshaped = B.T.reshape((B.shape[1], self.C, -1))
        self.simple_lut = np.einsum(
            "CKd, MCd -> MCK", self.simple_k_mean_prototypes, B_reshaped
        )

    def apply_matmul_e2e(
        self, A: np.ndarray, B: np.ndarray, A_learn: np.ndarray = None  # type: ignore
    ) -> np.ndarray:
        if A_learn is None:
            self.learn_hash_buckets_and_prototypes(A)
        else:
            self.learn_hash_buckets_and_prototypes(A_learn)
        self._set_A(A)
        self._set_B(B)
        return self._calc_matmul(
            self.A_enc,
            self.luts,
            offset=self.offset,
            scale=self.scale,
        )

    def encode(self, A: np.ndarray) -> np.ndarray:
        idxs = np.zeros((A.shape[0], self.C), np.int32)
        idxs = halut_encode_opt(A, self.splits_lists)
        # offsets = [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240]
        offsets = np.arange(self.C, dtype=np.int32) * self.K
        return idxs + offsets

    def _set_A(self, A: np.ndarray) -> None:
        self.A_enc = self.encode(A)

    def _set_B(self, B: np.ndarray) -> None:
        self.luts, self.offset, self.scale = self._create_lut(B.T)

    def _create_lut(self, B: np.ndarray) -> tuple[np.ndarray, float, float]:
        # called with B.T
        B = np.atleast_2d(B)
        luts = np.zeros((B.shape[0], self.C, self.K))
        for i, q in enumerate(B):
            luts[i] = maddness_lut(q, self.prototypes)
        if self.quantize_lut:
            luts, offset, scale = maddness_quantize_luts(luts)
            return luts, offset, scale
        return luts, 0, 1

    def _calc_matmul(
        self,
        A_enc: np.ndarray,
        B_luts: np.ndarray,
        offset: float,
        scale: float,
    ) -> np.ndarray:
        A_enc = np.ascontiguousarray(A_enc)
        total_result = np.empty((len(B_luts), len(A_enc)), dtype=np.float32)
        A_raveled = A_enc.ravel()
        if self.optimized:
            if self.upcast_every < 2 or not self.quantize_lut:
                total_result = read_luts_opt(
                    A_raveled, A_enc.shape, B_luts, total_result
                )
            else:
                total_result = read_luts_quantized_opt(
                    A_raveled,
                    A_enc.shape,
                    B_luts,
                    total_result,
                    self.upcast_every,
                    self.C,
                    scale,
                    offset,
                )
        else:
            raise NotImplementedError

        return total_result.T

    def matmul_online(
        self, A: np.ndarray, group: int = 0, groups: int = 1
    ) -> np.ndarray:
        self._check_if_learned()
        numba.set_num_threads(min(32, numba.get_num_threads()))
        self._set_A(A)
        luts = self.luts
        if groups > 1:
            assert self.luts.shape[0] % groups == 0
            m_dim_per_group = self.luts.shape[0] // groups
            luts = self.luts[group * m_dim_per_group : (group + 1) * m_dim_per_group]
        return self._calc_matmul(self.A_enc, luts, offset=self.offset, scale=self.scale)

    def stats(self) -> str:
        if self.is_learned():
            ret_str = f"Shape LUT: {self.luts.shape}, "
            ret_str += f"elements: {reduce(lambda x, y: x * y, self.luts.shape)} \n"
            ret_str += f"Actual storage LUT: {self.luts.nbytes / 1024} KB ({self.luts.dtype}) \n"
            numpy_array = self.splits_lists
            ret_str += f"Shaple splits_list: {numpy_array.shape}, "
            ret_str += f"elements: {reduce(lambda x, y: x * y, numpy_array.shape)} \n"
            ret_str += (
                f"Actual storage splits_list: {numpy_array.nbytes / 1024} KB "
                f"({numpy_array.dtype}) \n"
            )
            return ret_str
        else:
            return "not learned"
