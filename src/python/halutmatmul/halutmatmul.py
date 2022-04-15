# pylint: disable=C0413, E1133
# heavily inspired from https://github.com/dblalock/bolt
from __future__ import annotations
from functools import reduce
from typing import Any, List, Optional, Union
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent) + "/../../../maddness/python/"
)  # for maddness import
import numpy as np

import numba  # type: ignore [import]
from numba import prange

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


class HalutConfig:
    LUT_OFFSET = 0
    LUT_SCALE = 1
    RUN_OPTIMIZED = 2
    QUANTIZE_LUT = 3
    UPCAST_EVERY = 4
    MAX = 5


def learn_halut_offline(
    A: np.ndarray,
    B: np.ndarray,
    C: int = 16,
    lut_work_const: int = -1,
    quantize_lut: bool = False,
    run_optimized: bool = True,
) -> np.ndarray:
    mn = HalutMatmul(
        C, lut_work_const, quantize_lut=quantize_lut, run_optimized=run_optimized
    )
    mn.learn_offline(A, B)
    return mn.to_numpy()


def calc_newaxes_and_newshape_and_old(
    a: np.ndarray,
    b: np.ndarray,
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


def get_str_hash_buckets(buckets: list[MultiSplit]) -> str:
    ret_str = ""
    for v in buckets:
        ret_str += v.get_params() + "\n"
    return ret_str


def split_lists_to_numpy(buckets: list[list[MultiSplit]]) -> np.ndarray:
    length = 0
    for c in buckets:
        for v in c:
            length = v.vals.shape[0] if v.vals.shape[0] > length else length
    i = k = 0
    ret_array = np.zeros((len(buckets), len(buckets[0]), length + 3), dtype=np.float32)
    for c in buckets:
        k = 0
        for v in c:
            ret_array[i, k, 0 : pow(2, k)] = v.vals
            ret_array[i, k, length] = v.dim
            ret_array[i, k, length + 1] = v.scaleby
            ret_array[i, k, length + 2] = v.offset
            k += 1
        i += 1
    return ret_array


def numpy_to_split_list(numpy_array: np.ndarray) -> list[list[MultiSplit]]:
    splits: list[list[MultiSplit]] = []
    length = numpy_array.shape[2] - 3
    C = numpy_array.shape[0]
    num_splits = numpy_array.shape[1]
    assert num_splits == np.log2(length) + 1
    for c in range(C):
        splits.append([])
        for v in range(num_splits):
            vals = numpy_array[c, v, 0 : pow(2, v)].astype(
                np.int32
            )  # TODO: what when float values allowed??
            dim = int(numpy_array[c, v, length])
            scaleby = numpy_array[c, v, length + 1]
            offset = numpy_array[c, v, length + 2]
            multi_split = MultiSplit(dim=dim, vals=vals, scaleby=scaleby, offset=offset)
            splits[c].append(multi_split)
    return splits


# pylint: disable=R0201
def tensordot(
    a: np.ndarray,
    b: np.ndarray,
    axes: Union[int, list[int], Any] = 2,
    return_reshaped_inputs: bool = False,
    halut: Optional[HalutMatmul] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    # https://github.com/numpy/numpy/blob/145ed90f638c1a12ce5b06e9100421f99783f431/numpy/core/numeric.py#L950

    """Example
    padding=0, kernel_size=(3, 3), stride=1

    IN: (128, 64, 112, 112)
    W: (64, 64, 3, 3)
    after Im2col (np.lib.stride_tricks.as_strided): (128, 64, 110, 110, 3, 3)
    np.tensordot(IN, W, ((1,4,5),(1,2,3)))

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

    a, b = np.asarray(a), np.asarray(b)
    (
        newaxes_a,
        newaxes_b,
        newshape_a,
        newshape_b,
        olda,
        oldb,
    ) = calc_newaxes_and_newshape_and_old(a, b, axes)

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    if return_reshaped_inputs:
        return (at, bt)

    # numpy
    # res = np.dot(at, bt)
    if halut is not None:
        # import cProfile
        # from pstats import SortKey
        # with cProfile.Profile() as pr:
        res = halut.matmul_online(at)
        # pr.disable()
        # pr.print_stats(sort=SortKey.CUMULATIVE)
    else:
        raise Exception("Halut was not passed as argument")
    return res.reshape(olda + oldb)


@numba.jit(parallel=True, nopython=True)
def read_luts_quantized_opt(
    A_raveled: np.ndarray,
    A_shape: tuple[int, int],
    B_luts: np.ndarray,
    total_result: np.ndarray,
    upcast_every: int,
    C: int,
    scale: float,
    offset: float,
) -> np.ndarray:
    for i in prange((len(B_luts))):
        lut = B_luts[i]
        read_lut_1 = lut.ravel()[A_raveled].reshape(A_shape)
        shape_new = (
            read_lut_1.shape[0],
            int(A_shape[0] * A_shape[1] / upcast_every / read_lut_1.shape[0]),
            upcast_every,
        )
        read_lut = read_lut_1.reshape(shape_new)

        # while read_lut.shape[-1] > 2:
        for _ in range(np.log2(read_lut.shape[-1]) - 1):
            read_lut = (read_lut[:, :, ::2] + read_lut[:, :, 1::2] + 1) // 2

        read_lut = (read_lut[:, :, 0] + read_lut[:, :, 1] + 1) // 2
        read_lut = read_lut.sum(axis=-1)  # clipping not needed
        read_lut *= upcast_every  # convert mean to sum

        bias = C / 4 * np.log2(upcast_every)
        read_lut -= int(bias)

        read_lut = (read_lut / scale) + offset
        total_result[i] = read_lut
    return total_result


@numba.jit(parallel=True, nopython=True)
def read_luts_opt(
    A_raveled: np.ndarray,
    A_shape: tuple[int, int],
    B_luts: np.ndarray,
    total_result: np.ndarray,
) -> np.ndarray:
    for i in prange((len(B_luts))):
        read_lut = B_luts[i].ravel()[A_raveled].reshape(A_shape)
        read_lut = read_lut.sum(axis=-1)
        total_result[i] = read_lut
    return total_result


@numba.jit(nopython=True, parallel=False)
def apply_hash_function_opt(X: np.ndarray, splits: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    # original code had a distinction: not sure why
    group_ids = np.zeros(N, dtype=np.int64)  # needs to be int64 because of index :-)
    num_splits = splits.shape[0]
    length = splits.shape[1] - 3
    for i in range(num_splits):
        vals = splits[i, 0 : pow(2, i)].astype(np.int32)
        vals = vals[group_ids]
        dim = int(splits[i, length])
        scaleby = splits[i, length + 1]
        offset = splits[i, length + 2]
        x = X[:, dim] - offset
        x = x * scaleby
        indicators = x > vals
        group_ids = (group_ids * 2) + indicators
    return group_ids


def apply_hash_function(X: np.ndarray, splits: List[MultiSplit]) -> np.ndarray:
    N, _ = X.shape
    nsplits = len(splits)
    assert len(splits) >= 1
    # original code had a distinction: not sure why
    group_ids = np.zeros(N, dtype=np.int32)
    for i in range(nsplits):
        split = splits[i]
        vals = split.vals[group_ids]
        indicators = split.preprocess_x(X[:, split.dim]) > vals
        group_ids = (group_ids * 2) + indicators
    return group_ids


@numba.jit(nopython=True, parallel=True)
def maddness_encode_opt(X: np.ndarray, numpy_array: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    C = numpy_array.shape[0]
    A_enc = np.empty((C, N), dtype=np.int32)  # column-major
    # split_lists = numpy_to_split_list(numpy_array)
    for c in prange(C):
        A_enc[c] = apply_hash_function_opt(X, numpy_array[c])
    return np.ascontiguousarray(A_enc.T)


class HalutMatmul:
    def __init__(
        self,
        C: int = 16,
        lut_work_const: int = -1,
        quantize_lut: bool = False,
        run_optimized: bool = True,
    ) -> None:
        self.splits_lists: list[list[MultiSplit]] = []
        self.prototypes: np.ndarray = np.array([])
        self.luts: np.ndarray = np.array([])
        self.offset: float = 0.0
        self.scale: float = 1.0
        self.lut_work_const = lut_work_const
        self.C = C
        self.K = 16
        self.A_enc: np.ndarray = np.array([])

        self.quantize_lut = quantize_lut
        self.upcast_every = 16
        self.upcast_every = min(self.C, self.upcast_every)
        self.optimized = run_optimized
        # important otherwise wrong summation
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = "mean"  # sum

    def __repr__(self) -> str:
        return f"<HalutMatmul {self.get_params()}>"

    def __str__(self) -> str:
        return self.get_params()

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

    def _learn_hash_buckets_and_prototypes(self, A: np.ndarray) -> None:
        _, D = A.shape
        if D < self.C:
            raise Exception("D < C: {} < {}".format(D, self.C))
        self.splits_lists, self.prototypes = learn_proto_and_hash_function(
            A, self.C, lut_work_const=self.lut_work_const
        )

    def _check_if_learned(self) -> None:
        if not self.is_learned():
            raise Exception("Halut online tried but not learned!")

    def to_numpy(self) -> np.ndarray:
        self._check_if_learned()
        splits = split_lists_to_numpy(self.splits_lists)
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
            ],
            dtype=object,
        )
        return store_array

    def from_numpy(self, numpy_array: np.ndarray) -> HalutMatmul:
        splits_numpy = numpy_array[HalutOfflineStorage.HASH_TABLES]
        self.splits_lists = numpy_to_split_list(splits_numpy)
        self.luts = numpy_array[HalutOfflineStorage.LUT]
        config = numpy_array[HalutOfflineStorage.CONFIG]
        self.offset = config[HalutConfig.LUT_OFFSET]
        self.scale = config[HalutConfig.LUT_SCALE]
        upcast_every = int(config[HalutConfig.UPCAST_EVERY])
        self.optimized = bool(config[HalutConfig.RUN_OPTIMIZED])
        self.quantize_lut = bool(config[HalutConfig.QUANTIZE_LUT])
        assert self.splits_lists and self.luts.shape[1]
        _, C, K = self.luts.shape
        self.C = C
        self.K = K
        self.upcast_every = min(self.C, upcast_every)
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        return self

    # redefinition for convenience public function
    def learn_A(self, A: np.ndarray) -> None:
        self._learn_hash_buckets_and_prototypes(A)

    def learn_offline(self, A: np.ndarray, B: np.ndarray) -> None:
        self._learn_hash_buckets_and_prototypes(A)
        self._set_B(B)
        self._check_if_learned()

    def apply_matmul_e2e(
        self, A: np.ndarray, B: np.ndarray, A_learn: np.ndarray = None
    ) -> np.ndarray:
        if A_learn is None:
            self._learn_hash_buckets_and_prototypes(A)
        else:
            self._learn_hash_buckets_and_prototypes(A_learn)
        self._set_A(A)
        self._set_B(B)
        return self._calc_matmul(
            self.A_enc,
            self.luts,
            offset=self.offset,
            scale=self.scale,
        )

    def _encode_A(self, A: np.ndarray) -> np.ndarray:
        idxs = maddness_encode_opt(A, split_lists_to_numpy(self.splits_lists))
        # offsets = [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240]
        offsets = np.arange(self.C, dtype=np.int32) * self.K
        return idxs + offsets

    def _set_A(self, A: np.ndarray) -> None:
        self.A_enc = self._encode_A(A)

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
            numpy_array = split_lists_to_numpy(self.splits_lists)
            ret_str += f"Shaple splits_list: {numpy_array.shape}, "
            ret_str += f"elements: {reduce(lambda x, y: x * y, numpy_array.shape)} \n"
            ret_str += (
                f"Actual storage splits_list: {numpy_array.nbytes / 1024} KB "
                f"({numpy_array.dtype}) \n"
            )
            return ret_str
        else:
            return "not learned"
