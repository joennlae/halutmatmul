# pylint: disable=C0413, E1133
# heavily inspired from https://github.com/dblalock/bolt
from __future__ import annotations
from functools import reduce
from typing import Any, Dict, List, Optional, Union, Literal
import sys
from pathlib import Path

import numpy as np

import numba

from halutmatmul.decision_tree import (
    halut_encode_decision_tree,
    halut_encode_pq,
    learn_proto_and_hash_function_decision_tree,
)
from halutmatmul.functions import (
    get_str_hash_buckets,
    halut_encode_opt,
    numpy_to_split_list,
    read_luts_opt,
    read_luts_quantized_opt,
    split_lists_to_numpy,
)

sys.path.append(
    str(Path(__file__).parent) + "/../../../maddness/python/"
)  # for maddness import

from maddness.maddness import (
    MultiSplit,
    learn_proto_and_hash_function,
    maddness_lut,
    maddness_quantize_luts,
)


class HalutOfflineStorage:
    HASH_TABLES = 0
    LUT = 1
    CONFIG = 2
    PROTOTYPES = 3
    MAX = 4


class HalutConfig:
    LUT_OFFSET = 0
    LUT_SCALE = 1
    RUN_OPTIMIZED = 2
    QUANTIZE_LUT = 3
    UPCAST_EVERY = 4
    ENCODING_ALGORITHM = 5
    MAX = 6


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


def learn_halut_offline_report(
    A: np.ndarray,
    B: np.ndarray,
    C: int = 16,
    lut_work_const: int = -1,
    quantize_lut: bool = False,
    run_optimized: bool = True,
    encoding_algorithm: int = EncodingAlgorithm.FOUR_DIM_HASH,
) -> tuple[np.ndarray, Dict[str, Any]]:
    mn = HalutMatmul(
        C,
        lut_work_const,
        quantize_lut=quantize_lut,
        run_optimized=run_optimized,
        encoding_algorithm=encoding_algorithm,
    )
    mn.learn_offline(A, B)

    # print(mn.get_params())
    # print(mn.get_stats())

    return mn.to_numpy(), mn.get_stats()


def learn_halut_offline(
    A: np.ndarray,
    B: np.ndarray,
    C: int = 16,
    lut_work_const: int = -1,
    quantize_lut: bool = False,
    run_optimized: bool = True,
    encoding_algorithm: int = EncodingAlgorithm.FOUR_DIM_HASH,
) -> np.ndarray:
    mn = HalutMatmul(
        C,
        lut_work_const,
        quantize_lut=quantize_lut,
        run_optimized=run_optimized,
        encoding_algorithm=encoding_algorithm,
    )
    mn.learn_offline(A, B)
    return mn.to_numpy()


ENCODING_FUNCTIONS = [
    halut_encode_opt,  # FOUR_DIM_HASH
    halut_encode_decision_tree,  # DECISION_TREE
    halut_encode_pq,  # FULL_PQ
]

LEARNING_FUNCTIONS = [
    learn_proto_and_hash_function,  # FOUR_DIM_HASH
    learn_proto_and_hash_function_decision_tree,  # DECISION_TREE
    learn_proto_and_hash_function_decision_tree,  # FULL_PQ
]


class HalutMatmul:
    def __init__(
        self,
        C: int = 16,
        lut_work_const: int = -1,
        quantize_lut: bool = False,
        run_optimized: bool = True,
        encoding_algorithm: int = EncodingAlgorithm.FOUR_DIM_HASH,
    ) -> None:

        self.C = C
        self.K = 16
        self.encoding_algorithm = encoding_algorithm
        self.prototypes: np.ndarray = np.array([])
        self.luts: np.ndarray = np.array([])
        self.optimized = run_optimized

        self.encoding_function = ENCODING_FUNCTIONS[self.encoding_algorithm]
        self.learning_function = LEARNING_FUNCTIONS[self.encoding_algorithm]

        self.lut_work_const = lut_work_const
        self.A_enc: np.ndarray = np.array([])

        # EncodingAlgorithm.FOUR_DIM_HASH
        self.splits_lists: list[list[MultiSplit]] = []

        # EncodingAlgorithm.DECISION_TREE
        self.decision_trees: np.ndarray = np.array([])

        self.quantize_lut = quantize_lut
        self.upcast_every = 16
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
            raise Exception("D < C: {} < {}".format(D, self.C))
        (
            return_split_list_or_decison_trees,
            self.prototypes,
            report_array,
        ) = self.learning_function(
            A, self.C, lut_work_const=self.lut_work_const
        )  # type: ignore[operator]
        if self.encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
            self.splits_lists = return_split_list_or_decison_trees
        elif self.encoding_algorithm in [
            EncodingAlgorithm.DECISION_TREE,
            EncodingAlgorithm.FULL_PQ,
        ]:
            self.decision_trees = return_split_list_or_decison_trees

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
        if self.encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
            splits = split_lists_to_numpy(self.splits_lists)
        elif self.encoding_algorithm in [
            EncodingAlgorithm.DECISION_TREE,
            EncodingAlgorithm.FULL_PQ,
        ]:
            splits = self.decision_trees.astype(np.float32)

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
                        self.encoding_algorithm,
                    ],
                    dtype=np.float32,
                ),
                self.prototypes.astype(np.float32),
            ],
            dtype=object,
        )
        return store_array

    def from_numpy(self, numpy_array: np.ndarray) -> HalutMatmul:
        config = numpy_array[HalutOfflineStorage.CONFIG]
        self.encoding_algorithm = int(config[HalutConfig.ENCODING_ALGORITHM])
        self.encoding_function = ENCODING_FUNCTIONS[self.encoding_algorithm]
        self.learning_function = LEARNING_FUNCTIONS[self.encoding_algorithm]

        if self.encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
            splits_numpy = numpy_array[HalutOfflineStorage.HASH_TABLES]
            self.splits_lists = numpy_to_split_list(splits_numpy)
        elif self.encoding_algorithm in [
            EncodingAlgorithm.DECISION_TREE,
            EncodingAlgorithm.FULL_PQ,
        ]:
            self.decision_trees = numpy_array[HalutOfflineStorage.HASH_TABLES]

        self.luts = numpy_array[HalutOfflineStorage.LUT]
        self.offset = config[HalutConfig.LUT_OFFSET]
        self.scale = config[HalutConfig.LUT_SCALE]
        upcast_every = int(config[HalutConfig.UPCAST_EVERY])
        self.optimized = bool(config[HalutConfig.RUN_OPTIMIZED])
        self.quantize_lut = bool(config[HalutConfig.QUANTIZE_LUT])

        if self.encoding_algorithm == EncodingAlgorithm.FULL_PQ:
            self.prototypes = numpy_array[HalutOfflineStorage.PROTOTYPES]
        # assert self.splits_lists and self.luts.shape[1]
        _, C, K = self.luts.shape
        self.C = C
        self.K = K
        self.upcast_every = min(self.C, upcast_every)
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        return self

    # redefinition for convenience public function
    def learn_A(self, A: np.ndarray) -> None:
        self.learn_hash_buckets_and_prototypes(A)

    def learn_offline(self, A: np.ndarray, B: np.ndarray) -> None:
        self.learn_hash_buckets_and_prototypes(A)
        self._set_B(B)
        self._check_if_learned()

    def apply_matmul_e2e(
        self, A: np.ndarray, B: np.ndarray, A_learn: np.ndarray = None
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
        if self.encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
            idxs = halut_encode_opt(A, split_lists_to_numpy(self.splits_lists))
        elif self.encoding_algorithm == EncodingAlgorithm.DECISION_TREE:
            idxs = halut_encode_decision_tree(A, self.decision_trees)
        elif self.encoding_algorithm == EncodingAlgorithm.FULL_PQ:
            idxs = halut_encode_pq(A, self.prototypes)
        # offsets = [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240]
        offsets = np.arange(self.C, dtype=np.int32) * self.K
        return idxs + offsets

    def _set_A(self, A: np.ndarray) -> None:
        self.A_enc = self.encode(A)

    def _set_B(self, B: np.ndarray) -> None:
        self.luts, self.offset, self.scale = self._create_lut(B.T)

    def _create_lut(self, B: np.ndarray) -> tuple[np.ndarray, float, float]:
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
            for i, lut in enumerate(B_luts):
                read_lut = lut.ravel()[A_raveled].reshape(A_enc.shape)
                if self.upcast_every < 2 or not self.quantize_lut:
                    read_lut = read_lut.sum(axis=-1)
                else:
                    # TODO: there is probably room for improvement here
                    read_lut = read_lut.reshape(
                        read_lut.shape[0], -1, self.upcast_every
                    )
                    if self.accumulate_how == "sum":
                        # sum upcast_every vals, then clip to mirror saturating
                        # unsigned addition, then sum without saturation (like u16)
                        read_lut = read_lut.sum(2)
                        read_lut = np.clip(read_lut, 0, 255).sum(axis=-1)
                    elif self.accumulate_how == "mean":
                        # mirror hierarchical avg_epu8
                        while read_lut.shape[-1] > 2:
                            read_lut = (
                                read_lut[:, :, ::2] + read_lut[:, :, 1::2] + 1
                            ) // 2
                        read_lut = (read_lut[:, :, 0] + read_lut[:, :, 1] + 1) // 2
                        read_lut = read_lut.sum(axis=-1)  # clipping not needed
                        # undo biasing; if low bits are {0,0} or {1,1}, no bias
                        # from the averaging; but if {0,1}, then rounds up by
                        # .5; happens with prob ~=~ .5, so each avg op adds .25;
                        # the other tricky thing here is that rounding up when
                        # you're averaging averages biases it even farther
                        read_lut *= self.upcast_every  # convert mean to sum
                        # I honestly don't know why this is the formula, but wow
                        # does it work well
                        bias = self.C / 4 * np.log2(self.upcast_every)
                        read_lut -= int(bias)
                    else:
                        raise ValueError("accumulate_how must be 'sum' or 'mean'")

                if self.quantize_lut:
                    read_lut = (read_lut / scale) + offset
                total_result[i] = read_lut

        return total_result.T

    def matmul_online(self, A: np.ndarray) -> np.ndarray:
        self._check_if_learned()
        numba.set_num_threads(min(32, numba.get_num_threads()))
        self._set_A(A)
        return self._calc_matmul(
            self.A_enc, self.luts, offset=self.offset, scale=self.scale
        )

    def stats(self) -> str:
        if self.is_learned():
            ret_str = f"Shape LUT: {self.luts.shape}, "
            ret_str += f"elements: {reduce(lambda x, y: x * y, self.luts.shape)} \n"
            ret_str += f"Actual storage LUT: {self.luts.nbytes / 1024} KB ({self.luts.dtype}) \n"
            if self.encoding_algorithm == EncodingAlgorithm.FOUR_DIM_HASH:
                numpy_array = split_lists_to_numpy(self.splits_lists)
                ret_str += f"Shaple splits_list: {numpy_array.shape}, "
                ret_str += (
                    f"elements: {reduce(lambda x, y: x * y, numpy_array.shape)} \n"
                )
                ret_str += (
                    f"Actual storage splits_list: {numpy_array.nbytes / 1024} KB "
                    f"({numpy_array.dtype}) \n"
                )
            elif self.encoding_algorithm in [
                EncodingAlgorithm.FOUR_DIM_HASH,
                EncodingAlgorithm.FULL_PQ,
            ]:
                pass  # TODO: add print function here
            return ret_str
        else:
            return "not learned"
