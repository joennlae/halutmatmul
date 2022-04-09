from __future__ import annotations
from functools import reduce
from typing import Any, Optional, Type, TypeVar, Union
import numpy as np

from maddness.maddness import MaddnessMatmul, MultiSplit


class HalutOfflineStorage:
    HASH_TABLES = 0
    LUT = 1
    LUT_OFFSET_SCALE = 2


def learn_halut_offline(
    A: np.ndarray, B: np.ndarray, C: int = 16, lut_work_const: int = -1
) -> np.ndarray:
    mn = HalutMatmul(C, lut_work_const)
    mn.learn_offline(A, B)
    return mn.to_numpy()


def calc_newaxes_and_newshape_and_old(
    a: np.ndarray, b: np.ndarray, axes: Union[int, list[int], Any] = 2,
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
        res = halut.matmul_online(at)
    else:
        raise Exception("Halut was not passed as argument")
    return res.reshape(olda + oldb)


class HalutMatmul(MaddnessMatmul):
    def __init__(self, C: int = 16, lut_work_const: int = -1,) -> None:
        super().__init__(C, lut_work_const)
        self.splits_lists: list[list[MultiSplit]] = []
        self.prototypes: np.ndarray = np.array([])
        self.luts: np.ndarray = np.array([])
        self.offset: float = 0.0
        self.scale: float = 1.0

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

    def _check_if_learned(self) -> None:
        if not self.is_learned():
            raise Exception("Halut online tried but not learned!")

    def to_numpy(self) -> np.ndarray:
        self._check_if_learned()
        splits = split_lists_to_numpy(self.splits_lists)
        store_array = np.array(
            [splits, self.luts, np.array([self.offset, self.scale])], dtype=object
        )
        return store_array

    def from_numpy(self, numpy_array: np.ndarray) -> HalutMatmul:
        splits_numpy = numpy_array[HalutOfflineStorage.HASH_TABLES]
        self.splits_lists = numpy_to_split_list(splits_numpy)
        self.luts = numpy_array[HalutOfflineStorage.LUT]
        offset_scale = numpy_array[HalutOfflineStorage.LUT_OFFSET_SCALE]
        self.offset = offset_scale[0]
        self.scale = offset_scale[1]
        assert self.splits_lists and self.luts.shape[1]
        _, C, K = self.luts.shape
        self.C = C
        self.K = K
        self.upcast_every = min(self.C, self.upcast_every)
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
            self.A_enc, self.luts, offset=self.offset, scale=self.scale,  # type: ignore[arg-type]
        )

    def matmul_online(self, A: np.ndarray) -> np.ndarray:
        self._check_if_learned()
        self._set_A(A)
        return self._calc_matmul(
            self.A_enc, self.luts, offset=self.offset, scale=self.scale  # type: ignore[arg-type]
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
