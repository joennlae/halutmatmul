# pylint: disable=C0413, E1133
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Any, Optional, List, Literal

if TYPE_CHECKING:
    from halutmatmul.halutmatmul import HalutMatmul
from pathlib import Path
import sys

import numpy as np
import numba
from numba import prange

sys.path.append(
    str(Path(__file__).parent) + "/../../../maddness/python/"
)  # for maddness import

from maddness.util.hash_function_helper import MultiSplit  # type: ignore[attr-defined]


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
        vals = splits[i, 0 : pow(2, i)]
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
def halut_encode_opt(X: np.ndarray, numpy_array: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    C = numpy_array.shape[0]
    A_enc = np.empty((C, N), dtype=np.int32)  # column-major
    # split_lists = numpy_to_split_list(numpy_array)
    for c in prange(C):
        A_enc[c] = apply_hash_function_opt(X, numpy_array[c])
    return np.ascontiguousarray(A_enc.T)


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


def create_codebook_start_end_idxs(
    X: np.ndarray, number_of_codebooks: int, algo: Literal["start", "end"] = "start"
) -> np.ndarray:
    """
    returns vector (C, 2)
    [
      start_idx_0, end_idx_0,
      start_idx_1, end_idx_1,
      ...
    ]
    """
    assert algo in ("start", "end")

    _, D = X.shape
    number_of_codebooks = int(number_of_codebooks)
    assert D >= number_of_codebooks

    idxs = np.empty((number_of_codebooks, 2), dtype=np.int64)
    full_subvec_len = D // number_of_codebooks
    start_idx = 0
    for c in range(number_of_codebooks):
        subvec_len = full_subvec_len
        if algo == "start":  # wider codebooks at the start
            if c < (D % number_of_codebooks):
                subvec_len += 1
        elif algo == "end":  # wider codebooks at the end
            if (number_of_codebooks - c - 1) < (D % number_of_codebooks):
                subvec_len += 1
        end_idx = min(D, start_idx + subvec_len)
        idxs[c, 0] = start_idx
        idxs[c, 1] = end_idx

        start_idx = end_idx

    assert idxs[0, 0] == 0
    assert idxs[-1, -1] == D
    return idxs


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
            vals = numpy_array[c, v, 0 : pow(2, v)]
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
    halut: Optional["HalutMatmul"] = None,
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
