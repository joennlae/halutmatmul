# pylint: disable=C0413, E1133
# heavily inspired from https://github.com/dblalock/bolt
#
from pathlib import Path
import resource
import sys
from typing import List, Optional, Union, Tuple
import numpy as np

from halutmatmul.functions import halut_encode_opt, split_lists_to_numpy


sys.path.append(
    str(Path(__file__).parent) + "/../../../maddness/python/"
)  # for maddness import

from maddness.maddness import (
    MultiSplit,
)

from maddness.util.hash_function_helper import (  # type: ignore[attr-defined]
    Bucket,
    create_codebook_start_end_idxs,
)

from maddness.util.least_squares import (  # type: ignore[attr-defined]
    encoded_lstsq,
    _XW_encoded,
    sparse_encoded_lstsq,
)


def learn_binary_tree_splits(
    X: np.ndarray,
    K: int = 16,
    return_prototypes: bool = True,
    # return_buckets: bool = False,
    X_orig: Optional[np.ndarray] = None,
    check_x_dims: int = 8,  # can be used to check more or less dims with max losses
    learn_quantize_params: bool = False,
) -> Tuple[list, int, Union[list, np.ndarray]]:
    assert K in (4, 8, 16, 32, 64, 128)
    nsplits = int(np.log2(K))

    X = X.copy().astype(np.float32)
    N, D = X.shape  # D amount of IDx per codebook
    X_orig = X.copy() if X_orig is None else X_orig.copy()

    # initially, one big bucket with everything
    buckets = [
        Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))
    ]
    # total_loss = sum([bucket.loss for bucket in buckets])

    # print("================================")
    # print("learn_binary_tree_splits(): initial loss:   ", total_loss)

    splits = []
    col_losses = np.zeros(D, dtype=np.float32)
    OFFSET = 0.0
    SCALE_BY = 1.0
    X = X * SCALE_BY + OFFSET
    # X_orig = X_orig + OFFSET
    for _ in range(nsplits):
        # in the original code there are more strategies: eigenvec, bucket_eigenvecs, kurtosis
        # dim_heuristic == "bucket_sse":
        col_losses[:] = 0  # set all zero
        for buck in buckets:
            col_losses += buck.col_sum_sqs()  # return variance
        try_dims = np.argsort(col_losses)[::-1][
            :check_x_dims
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

        if learn_quantize_params:
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
        else:
            split.offset = OFFSET
            split.scaleby = SCALE_BY
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
    # if return_buckets:
    return splits, loss, buckets


def init_and_learn_hash_function(
    X: np.ndarray, C: int, K: int, pq_perm_algo: str = "start"
) -> Tuple[np.ndarray, list[list[MultiSplit]], np.ndarray, list]:
    _, D = X.shape

    X = X.astype(np.float32)
    X_error = X.copy().astype(np.float32)
    X_orig = X

    all_prototypes = np.zeros((C, K, D), dtype=np.float32)
    all_splits: list[list[MultiSplit]] = []
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
            use_X_error, X_orig=use_X_orig, return_prototypes=False, K=K
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


def learn_proto_and_hash_function(
    X: np.ndarray, C: int, K: int, lut_work_const: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, D = X.shape

    used_perm_algo = "start"  # or end
    X_orig = X.astype(np.float32)

    # X_error = X_orig - centroid shape: [N, D]
    X_error, all_splits, all_prototypes, _ = init_and_learn_hash_function(
        X, C, K, pq_perm_algo=used_perm_algo
    )

    msv_orig = (X_orig * X_orig).mean()
    mse_error = (X_error * X_error).mean()
    print(
        "X_error mse / X mean squared value: ",
        mse_error / msv_orig,
        mse_error,
        msv_orig,
        np.mean(X_orig),
    )

    squared_diff = np.square(X_orig - X_error).mean()
    print("Error to Original squared diff", squared_diff)
    # optimize prototypes discriminatively conditioned on assignments
    # applying g(A) [N, C] with values from 0-K (50000, 16)
    all_splits_np = split_lists_to_numpy(all_splits)
    A_enc = halut_encode_opt(X, all_splits_np)

    # optimizing prototypes
    if lut_work_const != 1:  # if it's 1, equivalent to just doing PQ
        if lut_work_const < 0:
            # print("fitting dense lstsq to X_error")
            W = encoded_lstsq(A_enc=A_enc, Y=X_error, K=K)
        else:
            if K != 16:
                raise Exception("not tested with K != 16")
            W, _ = sparse_encoded_lstsq(
                A_enc, X_error, nnz_blocks=lut_work_const, pq_perm_algo=used_perm_algo
            )

        all_prototypes_delta = W.reshape(C, K, D)
        all_prototypes += all_prototypes_delta

        # check how much improvement we got
        X_error -= _XW_encoded(A_enc, W, K=K)  # if we fit to X_error
        mse_res = (X_error * X_error).mean()

        print("X_error mse / X mse after lstsq: ", mse_res / msv_orig)

    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"After Ridge regression {X.shape}-{C}-{K}"
        f"({(ram_usage / (1024 * 1024)):.3f} GB)"
    )
    report_array = np.array(
        [
            mse_error,
            msv_orig,
            mse_error / msv_orig,
            np.mean(X_orig),
            mse_res,
            mse_res / msv_orig,
            ram_usage / (1024 * 1024),
        ]
    )
    return all_splits_np, all_prototypes, report_array


def maddness_lut(q: np.ndarray, all_prototypes: np.ndarray) -> np.ndarray:
    q = q.reshape(1, 1, -1)  # all_prototypes is shape C, K, D
    return (q * all_prototypes).sum(axis=2)  # C, K


def maddness_quantize_luts(
    luts: np.ndarray, force_power_of_2: bool = True
) -> tuple[np.ndarray, float, float]:
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
