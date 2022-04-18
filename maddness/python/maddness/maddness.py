# pylint: disable=C0302, C1802, C0209, R1705, W0201
import os, resource
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from maddness.util.least_squares import (  # type: ignore[attr-defined]
    _XW_encoded,
    encoded_lstsq,
    sparse_encoded_lstsq,
)
from maddness.util.hash_function_helper import (  # type: ignore[attr-defined]
    Bucket,
    MultiSplit,
    create_codebook_start_end_idxs,
)

# from joblib import Memory
# _memory = Memory(".", verbose=0)

# @_memory.cache
def learn_binary_tree_splits(
    X: np.ndarray,
    nsplits: int = 4,  # levels of resulting binary hash tree
    return_prototypes: bool = True,
    return_buckets: bool = False,
    X_orig: Optional[np.ndarray] = None,
    check_x_dims: int = 4,  # can be used to check more or less dims with max losses
) -> Tuple[list, int, Union[list, np.ndarray]]:
    assert nsplits <= 4  # >4 splits means >16 split_vals for this func's impl

    X = X.astype(np.float32)
    N, D = X.shape  # D amount of IDx per codebook
    X_orig = X if X_orig is None else X_orig

    # initially, one big bucket with everything
    buckets = [
        Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))
    ]
    # total_loss = sum([bucket.loss for bucket in buckets])

    # print("================================")
    # print("learn_binary_tree_splits(): initial loss:   ", total_loss)

    splits = []
    col_losses = np.zeros(D, dtype=np.float32)
    for _ in range(nsplits):
        # in the original code there are more strategies: eigenvec, bucket_eigenvecs, kurtosis
        # dim_heuristic == "bucket_sse":
        col_losses[:] = 0  # set all zero
        for buck in buckets:
            col_losses += buck.col_sum_sqs()
        try_dims = np.argsort(col_losses)[
            -check_x_dims:
        ]  # choose biggest column losses
        losses = np.zeros(len(try_dims), dtype=X.dtype)
        all_split_vals = []  # vals chosen by each bucket/group for each dim

        # determine for this dim what the best split vals are for each
        # group and what the loss is when using these split vals
        for d, dim in enumerate(try_dims):
            split_vals = []  # each bucket contributes one split val
            for _, buck in enumerate(buckets):
                # X.shape (50000, 32), dim is a number 0-31, val 1D, loss 1D
                val, loss = buck.optimal_split_val(X, dim, X_orig=X_orig)
                losses[d] += loss
                if d > 0 and losses[d] >= np.min(losses[:d]):
                    # early stop
                    break
                split_vals.append(val)
            all_split_vals.append(split_vals)
        # determine best dim to split on, and pull out associated split
        # vals for all buckets
        best_tried_dim_idx = np.argmin(losses)
        best_dim = try_dims[best_tried_dim_idx]
        use_split_vals = all_split_vals[best_tried_dim_idx]
        split = MultiSplit(dim=best_dim, vals=use_split_vals)

        # simple version, which also handles 1 bucket: just set min
        # value to be avg of min splitval and xval, and max value to
        # be avg of max splitval and xval
        x = X[:, best_dim]  # Vector (50000, 1)
        offset = (np.min(x) + np.min(use_split_vals)) / 2
        upper_val = (np.max(x) + np.max(use_split_vals)) / 2 - offset
        # TODO: why this specific scale value??
        scale = 254.0 / upper_val
        # if learn_quantize_params == "int16":
        scale = 2.0 ** int(np.log2(scale))

        split.offset = offset
        split.scaleby = scale
        split.vals = (split.vals - split.offset) * split.scaleby
        # TODO: look at clippings
        split.vals = np.clip(split.vals, 0, 255).astype(np.int32)

        splits.append(split)

        # apply this split to get next round of buckets
        new_buckets = []
        for i, buck in enumerate(buckets):
            val = use_split_vals[i]
            new_buckets += list(buck.split(X, dim=best_dim, val=val, X_orig=X_orig))
        buckets = new_buckets

    loss = sum([bucket.loss for bucket in buckets])
    # print("learn_binary_tree_splits(): returning loss: ", loss)

    if return_prototypes:
        prototypes = np.vstack([buck.col_means() for buck in buckets])
        assert prototypes.shape == (len(buckets), X.shape[1])
        return splits, loss, prototypes
    if return_buckets:
        return splits, loss, buckets
    return splits, loss, buckets


# @_memory.cache
def init_and_learn_hash_function(
    X: np.ndarray, C: int, pq_perm_algo: str = "start"
) -> Tuple[np.ndarray, list, np.ndarray, list]:
    _, D = X.shape
    K = 16

    X = X.astype(np.float32)
    X_error = X.copy().astype(np.float32)
    X_orig = X

    all_prototypes = np.zeros((C, K, D), dtype=np.float32)
    all_splits: List = []
    pq_idxs = create_codebook_start_end_idxs(X, C, algo=pq_perm_algo)

    # ------------------------ 0th iteration; initialize all codebooks
    all_splits = []
    all_buckets = []
    for c in range(C):
        start_idx, end_idx = pq_idxs[c]
        idxs = np.arange(start_idx, end_idx)
        # in original code there is other selections based on PCA and disjoint PCA

        use_X_error = X_error[:, idxs]
        use_X_orig = X_orig[:, idxs]

        # learn codebook to soak current residuals
        multisplits, _, buckets = learn_binary_tree_splits(
            use_X_error, X_orig=use_X_orig, return_prototypes=False, return_buckets=True
        )

        for split in multisplits:
            split.dim = idxs[split.dim]
        all_splits.append(multisplits)
        all_buckets.append(buckets)

        # update residuals and store prototypes
        # idxs = IDs that were look at for current codebook
        # buck.point_ids = rows that landed in certain K
        #   [    0     5    21 ... 99950 99979 99999] (N=100000)
        # X_error = is here still the A input
        # remove centroid from all the points that lie in a certain codebook
        # set prototype value
        centroid = np.zeros(D, dtype=np.float32)
        for b, buck in enumerate(buckets):
            # print(b, idxs, buck.point_ids, centroid, buck.col_means())
            if len(buck.point_ids):
                centroid[:] = 0
                centroid[idxs] = buck.col_means()
                X_error[buck.point_ids] -= centroid
                # update centroid here in case we want to regularize it somehow
                all_prototypes[c, b] = centroid

        # X_error = A_input - all_centroids
        ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            f"Learning progress {X.shape}-{C}-{K}: {c + 1}/{C} "
            f"({(ram_usage / (1024 * 1024)):.3f} GB)"
        )
    return X_error, all_splits, all_prototypes, all_buckets


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


def maddness_encode(
    X: np.ndarray, multisplits_lists: list[list[MultiSplit]]
) -> np.ndarray:
    N, _ = X.shape
    C = len(multisplits_lists)
    A_enc = np.empty((N, C), dtype=np.int32, order="F")  # column-major
    for c in range(C):
        A_enc[:, c] = apply_hash_function(X, multisplits_lists[c])
    return np.ascontiguousarray(A_enc)


# @_memory.cache
def learn_proto_and_hash_function(
    X: np.ndarray, C: int, lut_work_const: int = -1
) -> Tuple[list[list[MultiSplit]], np.ndarray]:
    _, D = X.shape
    K = 16
    used_perm_algo = "start"  # or end
    X_orig = X.astype(np.float32)

    # X_error = X_orig - centroid shape: [N, D]
    X_error, all_splits, all_prototypes, _ = init_and_learn_hash_function(
        X, C, pq_perm_algo=used_perm_algo
    )

    mse_orig = (X_orig * X_orig).mean()
    mse0 = (X_error * X_error).mean()
    print("X_error mse / X mse: ", mse0 / mse_orig)

    mse = np.square(X_orig - X_error).mean()
    print("mse", mse)
    # optimize prototypes discriminatively conditioned on assignments
    # applying g(A) [N, C] with values from 0-K (50000, 16)
    A_enc = maddness_encode(X, all_splits)

    # optimizing prototypes
    if lut_work_const != 1:  # if it's 1, equivalent to just doing PQ
        if lut_work_const < 0:
            # print("fitting dense lstsq to X_error")
            W = encoded_lstsq(A_enc=A_enc, Y=X_error)
        else:
            W, _ = sparse_encoded_lstsq(
                A_enc, X_error, nnz_blocks=lut_work_const, pq_perm_algo=used_perm_algo
            )

        all_prototypes_delta = W.reshape(C, K, D)
        all_prototypes += all_prototypes_delta

        # check how much improvement we got
        X_error -= _XW_encoded(A_enc, W)  # if we fit to X_error
        mse_res = (X_error * X_error).mean()
        print("X_error mse / X mse after lstsq: ", mse_res / mse_orig)

    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"After Ridge regression {X.shape}-{C}-{K}"
        f"({(ram_usage / (1024 * 1024)):.3f} GB)"
    )
    return all_splits, all_prototypes


def maddness_lut(q: np.ndarray, all_prototypes: np.ndarray) -> np.ndarray:
    q = q.reshape(1, 1, -1)  # all_prototypes is shape C, K, D
    return (q * all_prototypes).sum(axis=2)  # C, K


def maddness_quantize_luts(luts: np.ndarray, force_power_of_2: bool = True) -> Any:
    mins = luts.min(axis=(0, 2))
    maxs = luts.max(axis=(0, 2))

    gaps = maxs - mins
    gap = np.max(gaps)
    if force_power_of_2:
        exponent = np.ceil(np.log2(gap))
        scale = 2 ** int(-exponent)  # scale is a power of 2, so can just shift
        scale *= 255.5 - 1e-10  # so max val is at most 255
    else:
        scale = (255.5 - 1e-10) / gap

    offsets = mins[np.newaxis, :, np.newaxis]
    luts_quantized = (luts - offsets) * scale
    luts_quantized = (luts_quantized + 0.5).astype(np.int64)

    assert np.min(luts_quantized) >= 0
    assert np.max(luts_quantized) <= 255.0

    return luts_quantized, offsets.sum(), scale


# pylint: disable=R0902
class MaddnessMatmul:
    def __init__(self, C: int = 16, lut_work_const: int = -1) -> None:
        # checks
        if lut_work_const > 0 and lut_work_const > C:
            raise Exception("lut_work_const > C: {} > {}".format(lut_work_const, C))

        self.lut_work_const = lut_work_const
        self.C = C
        self.K = 16
        self.A_enc: Optional[np.ndarray] = None
        self.luts: Optional[np.ndarray] = None

        self.quantize_lut = False
        self.upcast_every = 16
        self.upcast_every = min(self.C, self.upcast_every)
        # important otherwise wrong summation
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = "mean"  # sum

    def _learn_hash_buckets_and_prototypes(self, A: np.ndarray) -> None:
        _, D = A.shape
        if D < self.C:
            raise Exception("D < C: {} < {}".format(D, self.C))
        self.splits_lists, self.prototypes = learn_proto_and_hash_function(
            A, self.C, lut_work_const=self.lut_work_const
        )

    def _encode_A(self, A: np.ndarray) -> np.ndarray:
        idxs = maddness_encode(A, self.splits_lists)
        # offsets = [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240]
        offsets = np.arange(self.C, dtype=np.int32) * self.K
        return idxs + offsets

    def _create_lut(self, B: np.ndarray) -> Tuple[np.ndarray, float, float]:
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
        for i, lut in enumerate(B_luts):
            read_lut = lut.ravel()[A_enc.ravel()].reshape(A_enc.shape)
            if self.upcast_every < 2 or not self.quantize_lut:
                read_lut = read_lut.sum(axis=-1)
            else:
                # TODO: there is probably room for improvement here
                read_lut = read_lut.reshape(read_lut.shape[0], -1, self.upcast_every)
                if self.accumulate_how == "sum":
                    # sum upcast_every vals, then clip to mirror saturating
                    # unsigned addition, then sum without saturation (like u16)
                    read_lut = read_lut.sum(2)
                    read_lut = np.clip(read_lut, 0, 255).sum(axis=-1)
                elif self.accumulate_how == "mean":
                    # mirror hierarchical avg_epu8

                    while read_lut.shape[-1] > 2:
                        read_lut = (read_lut[:, :, ::2] + read_lut[:, :, 1::2] + 1) // 2

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

    def _set_A(self, A: np.ndarray) -> None:
        self.A_enc = self._encode_A(A)

    def _set_B(self, B: np.ndarray) -> None:
        self.luts, self.offset, self.scale = self._create_lut(B.T)

    # public function
    def learn_A(self, A: np.ndarray) -> None:
        self._learn_hash_buckets_and_prototypes(A)

    def learn_offline(self, A: np.ndarray, B: np.ndarray) -> None:
        self._learn_hash_buckets_and_prototypes(A)
        self._set_B(B)

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
            self.A_enc,  # type: ignore[arg-type]
            self.luts,  # type: ignore[arg-type]
            offset=self.offset,
            scale=self.scale,
        )

    def matmul_online(self, A: np.ndarray) -> np.ndarray:
        self._set_A(A)
        return self._calc_matmul(
            self.A_enc, self.luts, offset=self.offset, scale=self.scale  # type: ignore[arg-type]
        )

    def reset(self) -> None:
        self.A_enc = None
        self.luts = None

    def get_speed_metrics(
        self, A: np.ndarray, B: np.ndarray, fixedA: bool = False, fixedB: bool = False
    ) -> None:
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.K * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.C
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.C
        print("nmuls: ", nmuls, "KEY_NLOOKUPS:", nlookups)


def matmul(
    A: np.ndarray,
    B: np.ndarray,
    C: int = 16,
    lut_work_const: int = -1,
    A_learn: Optional[np.ndarray] = None,
) -> np.ndarray:
    return MaddnessMatmul(C=C, lut_work_const=lut_work_const).apply_matmul_e2e(
        A, B, A_learn=A_learn
    )


if __name__ == "__main__":
    print("only use as import")
