# type: ignore
import copy
import numba
import numpy as np


@numba.njit(fastmath=True, cache=True, parallel=False)
def _cumsse_cols(X):
    # TODO: can be optimized with numpy
    N, D = X.shape
    cumsses = np.empty((N, D), X.dtype)
    cumX_row = np.empty(D, X.dtype)
    cumX2_row = np.empty(D, X.dtype)
    for j in range(D):
        cumX_row[j] = X[0, j]
        cumX2_row[j] = X[0, j] * X[0, j]
        cumsses[0, j] = 0  # no err in bucket with 1 element
    for i in range(1, N):
        one_over_count = 1.0 / (i + 1)
        for j in range(D):
            cumX_row[j] += X[i, j]
            cumX2_row[j] += X[i, j] * X[i, j]
            meanX = cumX_row[j] * one_over_count
            cumsses[i, j] = cumX2_row[j] - (cumX_row[j] * meanX)
    return cumsses


# @_memory.cache
def optimal_split_val(
    X,
    dim,
    X_orig=None,
):

    X_orig = X if X_orig is None else X_orig
    if X_orig.shape != X.shape:
        assert X_orig.shape == X.shape

    N, _ = X.shape
    sort_idxs = np.argsort(X_orig[:, dim])
    X_sort = X[sort_idxs]

    # cumulative SSE (sum of squared errors)
    sses_head = _cumsse_cols(X_sort)
    sses_tail = _cumsse_cols(X_sort[::-1])[::-1]
    sses = sses_head
    sses[:-1] += sses_tail[1:]
    sses = sses.sum(axis=1)

    best_idx = np.argmin(sses)
    next_idx = min(N - 1, best_idx + 1)
    col = X_orig[:, dim]
    best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2

    return best_val, sses[best_idx]


class Bucket:
    """
    sumX and sumX2 are (IDxs_per_codebook, 1) e.g (512 // 16, 1)
    """

    __slots__ = "N D id sumX sumX2 point_ids support_add_and_remove".split()

    def __init__(
        self,
        D=None,
        N=0,
        sumX=None,
        sumX2=None,
        point_ids=None,
        bucket_id=0,
        support_add_and_remove=False,
    ):
        # self.reset(D=D, sumX=sumX, sumX2=sumX2)
        # assert point_ids is not None
        if point_ids is None:
            assert N == 0
            point_ids = (
                set() if support_add_and_remove else np.array([], dtype=np.int64)
            )

        self.N = len(point_ids)
        self.id = bucket_id

        # this is just so that we can store the point ids as array instead of
        # set, while still retaining option to run our old code that needs
        # them to be a set for efficient inserts and deletes
        self.support_add_and_remove = support_add_and_remove
        if support_add_and_remove:
            self.point_ids = set(point_ids)
        else:
            self.point_ids = np.asarray(point_ids)

        # figure out D
        if (D is None or D < 1) and (sumX is not None):
            D = len(sumX)
        elif (D is None or D < 1) and (sumX2 is not None):
            D = len(sumX2)
        assert D is not None
        self.D = D

        # figure out + sanity check stats arrays
        self.sumX = np.zeros(D, dtype=np.float32) if (sumX is None) else sumX
        self.sumX2 = np.zeros(D, dtype=np.float32) if (sumX2 is None) else sumX2  # noqa
        # print("D: ", D)
        # print("sumX type: ", type(sumX))
        assert len(self.sumX) == D
        assert len(self.sumX2) == D
        self.sumX = np.asarray(self.sumX).astype(np.float32)
        self.sumX2 = np.asarray(self.sumX2).astype(np.float32)

    def add_point(self, point, point_id=None):
        assert self.support_add_and_remove
        # TODO replace with more numerically stable updates if necessary
        self.N += 1
        self.sumX += point
        self.sumX2 += point * point
        if point_id is not None:
            self.point_ids.add(point_id)

    def remove_point(self, point, point_id=None):
        assert self.support_add_and_remove
        self.N -= 1
        self.sumX -= point
        self.sumX2 -= point * point
        if point_id is not None:
            self.point_ids.remove(point_id)

    def deepcopy(self, bucket_id=None):  # deep copy
        bucket_id = self.id if bucket_id is None else bucket_id
        return Bucket(
            sumX=np.copy(self.sumX),
            sumX2=np.copy(self.sumX2),
            point_ids=copy.deepcopy(self.point_ids),
            bucket_id=bucket_id,
        )

    def split(self, X=None, dim=None, val=None, X_orig=None):
        id0 = 2 * self.id
        id1 = id0 + 1
        if X is None or self.N < 2:  # copy of this bucket + an empty bucket
            return (self.deepcopy(bucket_id=id0), Bucket(D=self.D, bucket_id=id1))
        assert dim is not None
        assert val is not None
        assert self.point_ids is not None
        my_idxs = np.asarray(self.point_ids)

        # print("my_idxs shape, dtype", my_idxs.shape, my_idxs.dtype)
        X = X[my_idxs]
        X_orig = X if X_orig is None else X_orig[my_idxs]
        mask = X_orig[:, dim] < val
        not_mask = ~mask
        X0 = X[mask]
        X1 = X[not_mask]
        ids0 = my_idxs[mask]
        ids1 = my_idxs[not_mask]

        def create_bucket(points, ids, bucket_id):
            sumX = points.sum(axis=0) if len(ids) else None
            sumX2 = (points * points).sum(axis=0) if len(ids) else None
            # return Bucket(N=len(ids), D=self.D, point_ids=ids,
            return Bucket(
                D=self.D, point_ids=ids, sumX=sumX, sumX2=sumX2, bucket_id=bucket_id
            )

        return create_bucket(X0, ids0, id0), create_bucket(X1, ids1, id1)

    def optimal_split_val(self, X, dim, X_orig=None):
        if self.N < 2 or self.point_ids is None:
            return 0, 0
        my_idxs = np.asarray(self.point_ids)
        if X_orig is not None:
            X_orig = X_orig[my_idxs]
        return optimal_split_val(X[my_idxs], dim, X_orig=X_orig)

    def col_means(self):
        return self.sumX.astype(np.float64) / max(1, self.N)

    def col_variances(self, safe=False):
        if self.N < 1:
            return np.zeros(self.D, dtype=np.float32)
        E_X2 = self.sumX2 / self.N
        E_X = self.sumX / self.N
        ret = E_X2 - (E_X * E_X)
        return np.maximum(0, ret) if safe else ret

    def col_sum_sqs(self):
        return self.col_variances() * self.N

    @property
    def loss(self):
        # more stable version, that also clamps variance at 0
        return max(0, np.sum(self.col_sum_sqs()))


def create_codebook_start_end_idxs(X, number_of_codebooks, algo="start"):
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


class MultiSplit:
    __slots__ = "dim vals scaleby offset".split()

    def __init__(self, dim, vals, scaleby=None, offset=None):
        self.dim = dim
        self.vals = np.asarray(vals)
        self.scaleby = scaleby
        self.offset = offset

    def __repr__(self) -> str:
        return f"<{self.get_params()}>"

    def __str__(self) -> str:
        return self.get_params()

    def get_params(self) -> str:
        params = (
            f"Multisplit: dim({self.dim}), vals({self.vals}), "
            f"scaleby({self.scaleby}), offset({self.offset})"
        )
        return params

    def preprocess_x(self, x: np.ndarray) -> np.ndarray:
        if self.offset is not None:
            x = x - self.offset
        if self.scaleby is not None:
            x = x * self.scaleby
        return x
