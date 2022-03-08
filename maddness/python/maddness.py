# pylint: disable=C0302, C1802, C0209, R1705, W0201

import abc
import os
import numpy as np

from lib.least_squares import _XW_encoded, encoded_lstsq, sparse_encoded_lstsq
from lib.hash_function_helper import Bucket, MultiSplit, create_codebook_start_end_idxs

# from joblib import Memory

# _memory = Memory(".", verbose=0)
_dir = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(_dir, "..", "assets", "cifar10-softmax")
CIFAR100_DIR = os.path.join(_dir, "..", "assets", "cifar100-softmax")

# pylint: disable=R0902
class MultiCodebookEncoder:
    def __init__(
        self,
        number_of_codebooks,
        number_of_prototypes=256,
        quantize_lut=False,
        upcast_every=-1,
        accumulate_how="sum",
    ):
        self.number_of_codebooks = number_of_codebooks
        self.number_of_prototypes = number_of_prototypes
        self.quantize_lut = quantize_lut
        self.upcast_every = upcast_every if upcast_every >= 1 else 1
        self.upcast_every = min(self.number_of_codebooks, self.upcast_every)
        # important otherwise wrong summation
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = accumulate_how

        self.code_bits = int(np.log2(self.number_of_prototypes))

        # for fast lookups via indexing into flattened array
        self.offsets = (
            np.arange(self.number_of_codebooks, dtype=np.int32)
            * self.number_of_prototypes
        )

    def matmul(self, A_enc, B_luts, offset, scale):
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
                    bias = self.number_of_codebooks / 4 * np.log2(self.upcast_every)
                    read_lut -= int(bias)

                else:
                    raise ValueError("accumulate_how must be 'sum' or 'mean'")

            if self.quantize_lut:
                read_lut = (read_lut / scale) + offset
            total_result[i] = read_lut

        return total_result.T


# @_memory.cache
def learn_binary_tree_splits(
    X,
    nsplits=4,  # levels of resulting binary hash tree
    return_centroids=True,
    return_buckets=False,
    X_orig=None,
    check_x_dims=4,  # can be used to check more or less dims with max losses
):
    assert nsplits <= 4  # >4 splits means >16 split_vals for this func's impl

    X = X.astype(np.float32)
    N, D = X.shape  # D amount of IDx per codebook
    X_orig = X if X_orig is None else X_orig

    # initially, one big bucket with everything
    buckets = [
        Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))
    ]
    total_loss = sum([bucket.loss for bucket in buckets])

    print("================================")
    print("learn_binary_tree_splits(): initial loss:   ", total_loss)

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
    print("learn_binary_tree_splits(): returning loss: ", loss)

    if return_centroids:
        centroids = np.vstack([buck.col_means() for buck in buckets])
        assert centroids.shape == (len(buckets), X.shape[1])
        return splits, loss, centroids
    if return_buckets:
        return splits, loss, buckets


# @_memory.cache
def init_and_learn_hash_function(X, number_of_codebooks, pq_perm_algo="start"):
    _, D = X.shape
    number_of_prototypes_per_codebook = 16

    X = X.astype(np.float32)
    X_res = X.copy()
    X_orig = X

    # TODO: stored with height D but could be stored with height of amount of idx as rest is zero!
    all_centroids = np.zeros(
        (number_of_codebooks, number_of_prototypes_per_codebook, D), dtype=np.float32
    )
    all_splits = []
    pq_idxs = create_codebook_start_end_idxs(X, number_of_codebooks, algo=pq_perm_algo)

    # ------------------------ 0th iteration; initialize all codebooks
    all_splits = []
    all_buckets = []
    for c in range(number_of_codebooks):
        start_idx, end_idx = pq_idxs[c]
        idxs = np.arange(start_idx, end_idx)
        # in original code there is other selections based on PCA and disjoint PCA

        use_X_res = X_res[:, idxs]
        use_X_orig = X_orig[:, idxs]

        # learn codebook to soak current residuals
        multisplits, _, buckets = learn_binary_tree_splits(
            use_X_res, X_orig=use_X_orig, return_centroids=False, return_buckets=True
        )

        for split in multisplits:
            split.dim = idxs[split.dim]
        all_splits.append(multisplits)
        all_buckets.append(buckets)

        # update residuals and store centroids
        centroid = np.zeros(D, dtype=np.float32)
        for b, buck in enumerate(buckets):
            if len(buck.point_ids):
                centroid[:] = 0
                centroid[idxs] = buck.col_means()
                X_res[buck.point_ids] -= centroid
                # update centroid here in case we want to regularize it somehow
                all_centroids[c, b] = centroid

    return X_res, all_splits, all_centroids, all_buckets


def apply_hash_function(X, splits):
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


def maddness_encode(X, multisplits_lists):
    N, _ = X.shape
    number_of_codebooks = len(multisplits_lists)
    A_enc = np.empty(
        (N, number_of_codebooks), dtype=np.int32, order="f"
    )  # column-major
    for c in range(number_of_codebooks):
        A_enc[:, c] = apply_hash_function(X, multisplits_lists[c])
    return np.ascontiguousarray(A_enc)


# @_memory.cache
def learn_proto_and_hash_function(X, number_of_codebooks, lut_work_const=-1):
    _, D = X.shape
    number_of_prototypes_per_codebook = 16
    used_perm_algo = "start"  # or end
    X_orig = X.astype(np.float32)

    X_res, all_splits, all_centroids, _ = init_and_learn_hash_function(
        X, number_of_codebooks, pq_perm_algo=used_perm_algo
    )

    mse_orig = (X_orig * X_orig).mean()
    mse0 = (X_res * X_res).mean()
    print("X_res mse / X mse: ", mse0 / mse_orig)

    mse = np.square(X_orig - X_res).mean()
    print("mse", mse)
    # optimize centroids discriminatively conditioned on assignments
    # applying g(A) [N, C] with values from 0-K (50000, 16)
    A_enc = maddness_encode(X, all_splits)

    # optimizing centroids
    if lut_work_const != 1:  # if it's 1, equivalent to just doing PQ
        if lut_work_const < 0:
            print("fitting dense lstsq to X_res")
            W = encoded_lstsq(A_enc=A_enc, Y=X_res)
        else:
            W, _ = sparse_encoded_lstsq(
                A_enc, X_res, nnz_blocks=lut_work_const, pq_perm_algo=used_perm_algo
            )

        all_centroids_delta = W.reshape(
            number_of_codebooks, number_of_prototypes_per_codebook, D
        )
        all_centroids += all_centroids_delta

        # check how much improvement we got
        X_res -= _XW_encoded(A_enc, W)  # if we fit to X_res
        mse_res = (X_res * X_res).mean()
        print("X_res mse / X mse after lstsq: ", mse_res / mse_orig)

    return all_splits, all_centroids


def maddness_lut(q, all_centroids):
    q = q.reshape(
        1, 1, -1
    )  # all_centroids is shape number_of_codebooks, number_of_prototypes, D
    return (q * all_centroids).sum(axis=2)  # number_of_codebooks, number_of_prototypes


def maddness_quantize_luts(luts, force_power_of_2=True):
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


class MaddnessEncoder(MultiCodebookEncoder):
    def __init__(self, number_of_codebooks, lut_work_const=-1):
        super().__init__(
            number_of_codebooks=number_of_codebooks,
            number_of_prototypes=16,
            quantize_lut=True,
            upcast_every=16,  # can be changed in log2
            accumulate_how="mean",
        )
        self.lut_work_const = lut_work_const

    def fit(self, X):
        self.splits_lists, self.centroids = learn_proto_and_hash_function(
            X, self.number_of_codebooks, lut_work_const=self.lut_work_const
        )

    def encode_A(self, A):
        idxs = maddness_encode(A, self.splits_lists)
        # offsets = [  0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240]
        return idxs + self.offsets

    def encode_B(self, B):
        B = np.atleast_2d(B)
        luts = np.zeros(
            (B.shape[0], self.number_of_codebooks, self.number_of_prototypes)
        )
        for i, q in enumerate(B):
            luts[i] = maddness_lut(q, self.centroids)
        if self.quantize_lut:
            luts, offset, scale = maddness_quantize_luts(luts)
            return luts, offset, scale
        return luts, 0, 1


class MaddnessMatmul:
    def __init__(self, number_of_codebooks, lut_work_const=-1):
        self.lut_work_const = lut_work_const
        if (
            (lut_work_const is not None)
            and (lut_work_const > 0)
            and (lut_work_const > number_of_codebooks)
        ):
            raise Exception(
                "lut_work_const > number_of_codebooks: {} > {}".format(
                    lut_work_const, number_of_codebooks
                )
            )
        self.number_of_codebooks = number_of_codebooks
        self.number_of_prototypes = 16
        self.enc = MaddnessEncoder(
            number_of_codebooks=number_of_codebooks, lut_work_const=lut_work_const
        )
        self.reset_for_new_task()

    def set_A(self, A):
        self.A_enc = self.enc.encode_A(A)

    def set_B(self, B):
        self.luts, self.offset, self.scale = self.enc.encode_B(B.T)

    def fit(self, A):
        _, D = A.shape
        if D < self.number_of_codebooks:
            raise Exception("D < C: {} < {}".format(D, self.number_of_codebooks))
        self.enc.fit(A)

    def predict(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.matmul(
            self.A_enc, self.luts, offset=self.offset, scale=self.scale
        )

    def reset_for_new_task(self):
        self.A_enc = None
        self.luts = None

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.number_of_prototypes * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.number_of_codebooks
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.number_of_codebooks
        print("nmuls: ", nmuls, "KEY_NLOOKUPS:", nlookups)


def load_cifar100_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = "cifar100_softmax_inputs_train.npy"
    SOFTMAX_OUTPUTS_TRAIN_PATH = "cifar100_softmax_outputs_train.npy"
    SOFTMAX_INPUTS_TEST_PATH = "cifar100_softmax_inputs_test.npy"
    SOFTMAX_OUTPUTS_TEST_PATH = "cifar100_softmax_outputs_test.npy"
    SOFTMAX_W_PATH = "cifar100_softmax_W.npy"
    SOFTMAX_B_PATH = "cifar100_softmax_b.npy"
    LABELS_TRAIN_PATH = "cifar100_labels_train.npy"
    LABELS_TEST_PATH = "cifar100_labels_test.npy"

    def load_mat(fname):
        fpath = os.path.join(CIFAR100_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel()
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    return (X_train, Y_train, X_test, Y_test, W, b, lbls_train, lbls_test)


def load_cifar10_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = "cifar10_softmax_inputs_train.npy"
    SOFTMAX_OUTPUTS_TRAIN_PATH = "cifar10_softmax_outputs_train.npy"
    SOFTMAX_INPUTS_TEST_PATH = "cifar10_softmax_inputs_test.npy"
    SOFTMAX_OUTPUTS_TEST_PATH = "cifar10_softmax_outputs_test.npy"
    SOFTMAX_W_PATH = "cifar10_softmax_W.npy"
    SOFTMAX_B_PATH = "cifar10_softmax_b.npy"
    LABELS_TRAIN_PATH = "cifar10_labels_train.npy"
    LABELS_TEST_PATH = "cifar10_labels_test.npy"

    def load_mat(fname):
        fpath = os.path.join(CIFAR10_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel()
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    return (X_train, Y_train, X_test, Y_test, W, b, lbls_train, lbls_test)


def main():
    # pylint: disable=W0612
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        W,
        b,
        lbls_train,
        lbls_test,
    ) = load_cifar100_tasks()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, W.shape)
    print(X_train)

    maddness = MaddnessMatmul(
        number_of_codebooks=16, lut_work_const=-1
    )  # MADDNESS-PQ has lut_work_const=1
    maddness.fit(X_train)

    maddness.reset_for_new_task()
    Y_pred = maddness.predict(X_test, W)

    print(Y_pred)
    print("max_pred", np.max(Y_pred), "min_pred", np.min(Y_pred))
    print(Y_test)
    print("max_test", np.max(Y_test), "min_test", np.min(Y_test))

    mse = (np.square(Y_pred - Y_test)).mean()
    print(mse)

    maddness.reset_for_new_task()
    Y_train_pred = maddness.predict(X_train, W)

    print(Y_train_pred.shape, X_train.shape)
    print(
        "max_train_pred", np.max(Y_train_pred), "min_train_pred", np.min(Y_train_pred)
    )

    mse = (np.square(Y_train_pred - Y_train)).mean()
    print(mse)


if __name__ == "__main__":
    main()
