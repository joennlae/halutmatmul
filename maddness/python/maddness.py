import abc
import copy
import os
import numpy as np
import numba
from joblib import Memory
from sklearn import linear_model

_dir = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(_dir, "..", "assets", "cifar10-softmax")
CIFAR100_DIR = os.path.join(_dir, "..", "assets", "cifar100-softmax")

_memory = Memory(".", verbose=0)


# numpy cumsum in insanely slow; also, having the nested loops is twice
# as fast as assigning rows (ie, X[i] += X[i-1])
@numba.njit(fastmath=True)
def _cumsum_cols(X):
    out = np.empty(X.shape, X.dtype)
    for j in range(X.shape[1]):
        out[0, j] = X[0, j]
    for i in range(1, X.shape[0]):
        for j in range(X.shape[1]):
            out[i, j] = X[i, j] + out[i - 1, j]
    return out


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def _cumsse_cols(X):
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


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def _cumsse_cols(X):
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


# def optimal_split_val(X, dim, possible_vals=None, return_val_idx=False):
# @_memory.cache
def optimal_split_val(
    X,
    dim,
    possible_vals=None,
    X_orig=None,
    # return_possible_vals_losses=False, force_val='median'):
    return_possible_vals_losses=False,
    force_val=None,
    # shrink_towards_median=True):
    shrink_towards_median=False,
):

    X_orig = X if X_orig is None else X_orig
    # X_orig = X # TODO rm
    if X_orig.shape != X.shape:
        print("X orig shape: ", X_orig.shape)
        print("X shape: ", X.shape)
        assert X_orig.shape == X.shape

    if force_val in ("mean", "median"):
        assert not return_possible_vals_losses
        x = X_orig[:, dim]
        val = np.median(x) if force_val == "median" else np.mean(x)
        mask = X_orig < val
        X0 = X[mask]
        errs0 = X0 - X0.mean(axis=0)
        loss0 = np.sum(errs0 * errs0)
        X1 = X[~mask]
        errs = X1 - X1.mean(axis=0)
        loss1 = np.sum(errs * errs)
        return val, loss0 + loss1

    N, D = X.shape
    # sort_idxs = np.argsort(X[:, dim])
    sort_idxs = np.argsort(X_orig[:, dim])
    X_sort = X[sort_idxs]

    # use_jit = False
    use_jit = True
    if use_jit:
        # X_sort = X_sort[:100] # TODO rm
        # X_sort = np.ascontiguousarray(X_sort)
        # N, D = X_sort.shape
        # print("about to call jitted func; N, D = ", N, D)
        sses_head = _cumsse_cols(X_sort)
        # print("got thru first call...")
        # X_sort_rev = np.ascontiguousarray(X_sort[::-1])
        # sses_tail = _cumsse_cols(X_sort_rev)[::-1]
        sses_tail = _cumsse_cols(X_sort[::-1])[::-1]
        # print("returned from jitted func!")
    else:
        X_sort_sq = X_sort * X_sort
        # cumX_head = np.cumsum(X_sort, axis=0)
        # cumX2_head = np.cumsum(X_sort_sq, axis=0)
        # cumX_tail = np.cumsum(X_sort[::-1], axis=0)[::-1]
        # cumX2_tail = np.cumsum(X_sort_sq[::-1], axis=0)[::-1]
        cumX_head = _cumsum_cols(X_sort)
        cumX2_head = _cumsum_cols(X_sort_sq)
        cumX_tail = _cumsum_cols(X_sort[::-1])[::-1]
        cumX2_tail = _cumsum_cols(X_sort_sq[::-1])[::-1]

        all_counts = np.arange(1, N + 1).reshape(-1, 1)
        EX_head = cumX_head / all_counts  # E[X], starting from 0
        EX_tail = cumX_tail / all_counts[::-1]  # E[X], starting from N-1
        # EX2_head = cumX2_head / all_counts          # E[X^2], starting from 0
        # EX2_tail = cumX2_tail / all_counts[::-1]    # E[X^2], starting from N-1
        # mses_head = EX2_head - (EX_head * EX_head)  # mses from 0
        # mses_tail = EX2_tail - (EX_tail * EX_tail)  # mses from N-1
        # sses_head = mses_head * all_counts          #
        # sses_tail = mses_tail * all_counts[::-1]

        # simpler equivalent of above; mse * N reduces to this
        sses_head = cumX2_head - (cumX_head * EX_head)
        sses_tail = cumX2_tail - (cumX_tail * EX_tail)

    # # TODO rm
    # mse_head_diffs = sses_head[1:] - sses_head[:-1]
    # # print("mse_head_diffs[:20]", mse_head_diffs[:20])
    # assert np.all(mse_head_diffs > -.1)  # should be nondecreasing
    # mse_tail_diffs = sses_tail[1:] - sses_tail[:-1]
    # assert np.all(mse_tail_diffs < .1)  # should be nonincreasing

    sses = sses_head
    sses[:-1] += sses_tail[1:]  # sse of X_sort[:i] + sse of X_sort[i:]
    sses = sses.sum(axis=1)

    if shrink_towards_median:
        minsse, maxsse = np.min(sses), np.max(sses)
        scale = maxsse - minsse
        # n_over_2 = N // 2
        # scale = (maxsse - minsse) / n_over_2
        coeffs = np.abs(np.arange(N, dtype=np.float32))
        penalties = coeffs * (scale / np.max(coeffs))
        sses += penalties

    # # TODO rm
    # E_X = X.mean(axis=0)
    # E_X2 = (X * X).mean(axis=0)
    # sse_true = np.sum(E_X2 - (E_X * E_X)) * N
    # print("sses[0], sses[-1], true loss, np.sum(X.var(axis=0)) * N",
    #       sses[0], sses[-1], sse_true, np.sum(X.var(axis=0)) * N)

    # X_orig_sort = X_orig[sort_idxs]
    if possible_vals is None or not len(possible_vals):  # can split anywhere
        best_idx = np.argmin(sses)
        next_idx = min(N - 1, best_idx + 1)
        # best_val = (X_sort[best_idx, dim] + X_sort[next_idx, dim]) / 2.
        # X_orig_sort = X_orig[sort_idxs]
        col = X_orig[:, dim]
        best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2
        # best_val = (X_orig_sort[best_idx, dim] + X_orig_sort[next_idx, dim]) / 2
    else:  # have to choose one of the values in possible_vals
        sorted_col = X_orig[:, dim][sort_idxs]
        idxs = np.searchsorted(sorted_col, possible_vals)
        # idxs = np.unique(idxs)
        idxs = np.maximum(0, idxs - 1)  # searchsorted returns first idx larger
        sses_for_idxs = sses[idxs]
        which_idx_idx = np.argmin(sses_for_idxs)
        best_idx = idxs[which_idx_idx]
        best_val = possible_vals[which_idx_idx]

    # print("return_possible_vals_losses: ", return_possible_vals_losses)
    ret = best_val, sses[best_idx]
    return ret + (sses_for_idxs,) if return_possible_vals_losses else ret


class Bucket(object):
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

    def optimal_split_val(
        self, X, dim, possible_vals=None, X_orig=None, return_possible_vals_losses=False
    ):
        if self.N < 2 or self.point_ids is None:
            if return_possible_vals_losses:
                return 0, 0, np.zeros(len(possible_vals), dtype=X.dtype)
            return 0, 0
        # my_idxs = np.array(list(self.point_ids))
        my_idxs = np.asarray(self.point_ids)
        if X_orig is not None:
            X_orig = X_orig[my_idxs]
        return optimal_split_val(
            X[my_idxs],
            dim,
            possible_vals=possible_vals,
            X_orig=X_orig,
            return_possible_vals_losses=return_possible_vals_losses,
        )

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
        # if self.N < 1:
        #     return 0

        # # less stable version with one less divide and mul
        # return max(0, np.sum(self.sumX2 - (self.sumX * (self.sumX / self.N))))

        # more stable version, that also clamps variance at 0
        return max(0, np.sum(self.col_sum_sqs()))
        # expected_X = self.sumX / self.N
        # expected_X2 = self.sumX2 / self.N
        # return max(0, np.sum(expected_X2 - (expected_X * expected_X)) * self.N)


# @numba.jit(nopython=True)  # don't jit since take like 2.5s
# def top_principal_component(X, niters=50, return_eigenval=False,
def top_principal_component(
    X,
    niters=100,
    return_eigenval=False,
    momentum=0.9,
    nguesses=32,
    learning_rate=1.0,
    # allow_materialize=False):
    allow_materialize_XtX=True,
):
    N, D = X.shape
    X = X.astype(np.float32)
    X = X - X.mean(axis=0)

    if nguesses > 1:
        V = np.random.randn(D, nguesses).astype(X.dtype)
        V /= np.linalg.norm(V, axis=0)
        # norms = np.sqrt((V * V).sum(axis=0))
        # V /= norms
        prods = X.T @ (X @ V)
        new_norms = np.linalg.norm(prods, axis=0)
        # new_norms_sq = (prods * prods).sum(axis=0)
        v = V[:, np.argmax(new_norms)]
        # v = V[:, np.argmax(new_norms_sq)]
        # print("picking v = ", v)
    else:
        v = np.random.randn(D).astype(X.dtype)
    # v = np.ones(D, dtype=np.float32)

    v = v.astype(np.float32)
    prev_v = np.zeros_like(v)
    v_momentum = np.zeros_like(v)
    v /= np.linalg.norm(v) + 1e-20

    materialize_cost = N * D * D
    iter_cost_no_materialize = 2 * N * D
    iter_cost_materialize = D * D

    materialize = materialize_cost + (niters * iter_cost_materialize) < (
        niters * iter_cost_no_materialize
    )
    materialize = materialize and allow_materialize_XtX
    if materialize:
        scaleby = np.max(np.linalg.norm(X, axis=0))
        X *= 1.0 / scaleby  # precondition by setting largest variance to 1
        XtX = X.T @ X

    for i in range(niters):
        if materialize:
            v = XtX @ v
        else:
            v = X.T @ (X @ v)
        v *= 1.0 / (np.linalg.norm(v) + 1e-20)
        # v_momentum = .9 * v_momentum + .5 * (v - prev_v)
        # v_momentum = (.9 * v_momentum + (v - prev_v)).astype(np.float32)
        v_momentum = momentum * v_momentum + learning_rate * (v - prev_v)
        v += v_momentum
        prev_v = v
        # if i % 5 == 0:
        #     print("v: ", v)

    v /= np.linalg.norm(v) + 1e-20
    if return_eigenval:
        new_v = X.T @ (X @ v)
        lamda = np.linalg.norm(new_v)
        return v, lamda
    return v


def extract_random_rows(X, how_many, remove_from_X=True):
    split_start = np.random.randint(len(X) - how_many - 1)
    split_end = split_start + how_many
    rows = np.copy(X[split_start:split_end])
    if remove_from_X:
        return np.vstack((X[:split_start], X[split_end:])), rows
    return X, rows


def _learn_best_quantization(luts):
    assert luts.ndim == 2  # luts can be a bunch of vstacked luts, but not 3D
    best_loss = np.inf
    best_alpha = None
    best_floors = None
    best_scale_by = None
    for alpha in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        # alpha_pct = int(100 * alpha)
        alpha_pct = 100 * alpha

        # compute quantized luts this alpha would yield
        floors = np.percentile(luts, alpha_pct, axis=0)
        luts_offset = np.maximum(0, luts - floors)

        ceil = np.percentile(luts_offset, 100 - alpha_pct)
        scale_by = 255.0 / ceil
        # if only_shift:
        #     scale_by = 1 << int(np.log2(scale_by))
        luts_quantized = np.floor(luts_offset * scale_by).astype(np.int64)
        luts_quantized = np.minimum(255, luts_quantized)

        # compute err
        luts_ideal = (luts - luts_offset) * scale_by
        diffs = luts_ideal - luts_quantized
        loss = np.sum(diffs * diffs)

        if loss <= best_loss:
            best_loss = loss
            best_alpha = alpha
            best_floors = floors
            best_scale_by = scale_by

    return best_floors, best_scale_by, best_alpha


class MultiCodebookEncoder(abc.ABC):
    def __init__(
        self,
        ncodebooks,
        ncentroids=256,
        quantize_lut=False,
        upcast_every=-1,
        accumulate_how="sum",
    ):
        self.ncodebooks = ncodebooks
        self.ncentroids = ncentroids
        self.quantize_lut = quantize_lut
        self.upcast_every = upcast_every if upcast_every >= 1 else 1
        self.upcast_every = min(self.ncodebooks, self.upcast_every)
        assert self.upcast_every in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        self.accumulate_how = accumulate_how

        self.code_bits = int(np.log2(self.ncentroids))

        # for fast lookups via indexing into flattened array
        self.offsets = np.arange(self.ncodebooks, dtype=np.int64) * self.ncentroids

    def name(self):
        return "{}x{}b_quantize={}".format(
            self.ncodebooks, self.code_bits, int(self.quantize_lut)
        )

    def params(self):
        return {
            "ncodebooks": self.ncodebooks,
            "code_bits": self.code_bits,
            "quantize": self.quantize_lut,
        }

    def _learn_lut_quantization(self, X, Q=None):
        if self.quantize_lut:  # TODO put this logic in separate function
            print("learning quantization...")

            # print("initial Q: ", Q)
            if Q is None:
                # num_rows = min(10 * 1000, len(X) // 2)
                # _, queries = extract_random_rows(
                #     X[num_rows:], how_many=1000, remove_from_X=False)
                # X = X[:num_rows]  # limit to first 10k rows of X
                _, Q = extract_random_rows(X, how_many=1000, remove_from_X=False)
                Q = Q.T  # want each row to be one query, not each col

            # Q = self._pad_ncols(Q)
            # if self.preproc == 'OPQ':
            #     Q = pq.opq_rotate(Q, self.R)
            # elif self.preproc == 'BOPQ':
            #     Q = pq.bopq_rotate(Q, self.rotations)
            # elif self.preproc == 'GEHT':
            #     Q = Q[:, self.perm]

            # print("Q shape: ", Q.shape)

            # compute luts for all the queries
            # luts = [self.encode_Q(q, quantize=False) for q in Q]
            # luts = self.encode_Q(Q, quantize=False)
            # luts = np.vstack(luts)
            # print("ncodebooks: ", self.ncodebooks)
            # print("luts shape: ", luts.shape)
            assert luts.shape == (len(Q), self.ncodebooks, self.ncentroids)
            luts = np.moveaxis(luts, 2, 1)
            assert luts.shape == (len(Q), self.ncentroids, self.ncodebooks)
            luts = luts.reshape(len(Q) * self.ncentroids, self.ncodebooks)

            self.lut_offsets, self.scale_by, _ = _learn_best_quantization(luts)
            # print("self.lut_offsets.shape", self.lut_offsets.shape)
            # print("self.scale_by.shape", self.scale_by.shape)
            # print("self.scale_by", self.scale_by)
            assert self.lut_offsets.shape == (self.ncodebooks,)
            # self.lut_offsets = self.lut_offsets[:, np.newaxis]
            self.total_lut_offset = np.sum(self.lut_offsets)
            # print("lut offsets: ", self.lut_offsets)

    def dists_enc(self, X_enc, Q_luts, unquantize=True, offset=None, scale=None):
        X_enc = np.ascontiguousarray(X_enc)

        if unquantize:
            offset = self.total_lut_offset if offset is None else offset
            scale = self.scale_by if scale is None else scale

        all_dists = np.empty((len(Q_luts), len(X_enc)), dtype=np.float32)
        for i, lut in enumerate(Q_luts):
            centroid_dists = lut.ravel()[X_enc.ravel()]
            dists = centroid_dists.reshape(X_enc.shape)
            if self.upcast_every < 2 or not self.quantize_lut:
                dists = dists.sum(axis=-1)
            else:
                dists = dists.reshape(dists.shape[0], -1, self.upcast_every)
                if self.accumulate_how == "sum":
                    # sum upcast_every vals, then clip to mirror saturating
                    # unsigned addition, then sum without saturation (like u16)
                    dists = dists.sum(2)
                    dists = np.clip(dists, 0, 255).sum(axis=-1)
                elif self.accumulate_how == "mean":
                    # mirror hierarchical avg_epu8
                    # print("reducing using mean!")

                    # print("fraction of low bits that are 1: ",
                    #       np.mean(dists % 2 == 1))  # ya, ~.5, or maybe ~.495

                    while dists.shape[-1] > 2:
                        dists = (dists[:, :, ::2] + dists[:, :, 1::2] + 1) // 2
                    dists = (dists[:, :, 0] + dists[:, :, 1] + 1) // 2
                    dists = dists.sum(axis=-1)  # clipping not needed

                    # undo biasing; if low bits are {0,0} or {1,1}, no bias
                    # from the averaging; but if {0,1}, then rounds up by
                    # .5; happens with prob ~=~ .5, so each avg op adds .25;
                    # the other tricky thing here is that rounding up when
                    # you're averaging averages biases it even farther
                    # base_bias = .5 * .5
                    # assert self.upcast_every >= 2
                    # bias_per_upcast = 0
                    # nlevels = int(np.log2(self.upcast_every))
                    # for level in range(nlevels):
                    #     num_avg_ops = self.upcast_every / (2 << level)
                    #     print("num_avg_ops: ", num_avg_ops)
                    #     bias_per_op = (1 << level) * base_bias
                    #     print("level multiplier: ", 1 << level)
                    #     bias_per_upcast += num_avg_ops * bias_per_op

                    # bias = bias_per_upcast * (self.ncodebooks / self.upcast_every)

                    # num_avg_ops = (self.upcast_every - 1) * (
                    #     self.ncodebooks / self.upcast_every)
                    # num_avg_ops = (self.upcast_every - 1) * np.sqrt(
                    #     self.ncodebooks / self.upcast_every)
                    # num_avg_ops = (self.upcast_every - 1)
                    # bias = num_avg_ops * base_bias

                    # bias = (self.ncodebooks / 2) * int(np.log2(self.upcast_every))
                    # bias = (self.ncodebooks / 2) * int(np.log2(self.upcast_every))
                    # bias = 0
                    # dists -= int(bias * self.upcast_every)

                    dists *= self.upcast_every  # convert mean to sum

                    # I honestly don't know why this is the formula, but wow
                    # does it work well
                    bias = self.ncodebooks / 4 * np.log2(self.upcast_every)
                    dists -= int(bias)

                else:
                    raise ValueError("accumulate_how must be 'sum' or 'mean'")

            if self.quantize_lut and unquantize:
                # dists = (dists / self.scale_by) + self.total_lut_offset
                dists = (dists / scale) + offset
            all_dists[i] = dists

        return all_dists.T


def _pq_codebook_start_end_idxs(X, ncodebooks, algo="start"):
    assert algo in ("start", "end")  # TODO do something smarter here

    # D = int(D)
    _, D = X.shape
    ncodebooks = int(ncodebooks)
    assert D >= ncodebooks

    idxs = np.empty((ncodebooks, 2), dtype=np.int64)
    full_subvec_len = D // ncodebooks
    start_idx = 0
    for c in range(ncodebooks):
        subvec_len = full_subvec_len
        if algo == "start":  # wider codebooks at the start
            if c < (D % ncodebooks):
                subvec_len += 1
        elif algo == "end":  # wider codebooks at the end
            if (ncodebooks - c - 1) < (D % ncodebooks):
                subvec_len += 1
        end_idx = min(D, start_idx + subvec_len)
        # print("c, start_idx, end_idx: ", c, start_idx, end_idx)
        # print("start_idx, end_idx: ", c, start_idx, end_idx)
        idxs[c, 0] = start_idx
        idxs[c, 1] = end_idx

        start_idx = end_idx

    assert idxs[0, 0] == 0
    assert idxs[-1, -1] == D
    return idxs


class MultiSplit(object):
    __slots__ = "dim vals scaleby offset".split()

    def __init__(self, dim, vals, scaleby=None, offset=None):
        self.dim = dim
        self.vals = np.asarray(vals)
        self.scaleby = scaleby
        self.offset = offset

    def preprocess_x(self, x):
        if self.offset is not None:
            x = x - self.offset
        if self.scaleby is not None:
            x = x * self.scaleby
        return x


@_memory.cache
def learn_multisplits(
    X,
    nsplits=4,
    return_centroids=True,
    return_buckets=False,
    # learn_quantize_params=False,
    # learn_quantize_params='int16', X_orig=None, try_ndims=1,
    # learn_quantize_params='int16', X_orig=None, try_ndims=2,
    learn_quantize_params="int16",
    X_orig=None,
    try_ndims=4,
    # learn_quantize_params='int16', X_orig=None, try_ndims=8,
    # learn_quantize_params='int16', X_orig=None, try_ndims=16,
    # learn_quantize_params=True,
    # verbose=3):
    # verbose=2):
    verbose=1,
):
    assert nsplits <= 4  # >4 splits means >16 split_vals for this func's impl

    X = X.astype(np.float32)
    N, D = X.shape
    X_orig = X if X_orig is None else X_orig

    X_hat = np.zeros_like(X)

    # initially, one big bucket with everything
    buckets = [
        Bucket(sumX=X.sum(axis=0), sumX2=(X * X).sum(axis=0), point_ids=np.arange(N))
    ]
    total_loss = sum([bucket.loss for bucket in buckets])

    if verbose > 0:
        print("================================")
        # print("learn_multisplits(): initial loss: ", total_loss)
        print("learn_multisplits(): initial loss:   ", total_loss)
        # print("learn_multisplits(): trying ndims:   ", min(D, try_ndims))

    splits = []
    col_losses = np.zeros(D, dtype=np.float32)  # TODO rm?
    for s in range(nsplits):
        if verbose > 1:
            print("------------------------ finding split #:", s)

        # dim_heuristic = 'eigenvec'
        # dim_heuristic = 'bucket_eigenvecs'
        dim_heuristic = "bucket_sse"
        # dim_heuristic = 'kurtosis'
        if dim_heuristic == "eigenvec":
            # compute current reconstruction of X, along with errs
            if s > 0:
                for buck in buckets:
                    # print("point ids: ", buck.point_ids)
                    if len(buck.point_ids):
                        centroid = buck.col_means()
                        # X_hat[np.array(buck.point_ids)] = centroid
                        X_hat[buck.point_ids] = centroid
                X_res = X - X_hat
            else:
                X_res = X
            # pick dims by looking at top principal component
            v = top_principal_component(X_res)
            try_dims = np.argsort(np.abs(v))[-try_ndims:]
        elif dim_heuristic == "bucket_eigenvecs":
            dim_scores = np.zeros(D, dtype=np.float32)
            for buck in buckets:
                if buck.N < 2:
                    continue
                X_buck = X[buck.point_ids]
                v, lamda = top_principal_component(X_buck, return_eigenval=True)
                v *= lamda
                dim_scores += np.abs(v)
                # X_buck -= X_buck.mean(axis=0)
            try_dims = np.argsort(dim_scores)[-try_ndims:]
        elif dim_heuristic == "bucket_sse":
            col_losses[:] = 0
            for buck in buckets:
                col_losses += buck.col_sum_sqs()
            try_dims = np.argsort(col_losses)[-try_ndims:]
        elif dim_heuristic == "kurtosis":
            # compute X_res
            if s > 0:
                for buck in buckets:
                    # print("point ids: ", buck.point_ids)
                    if len(buck.point_ids):
                        centroid = buck.col_means()
                        # X_hat[np.array(buck.point_ids)] = centroid
                        X_hat[buck.point_ids] = centroid
                X_res = X - X_hat
            else:
                X_res = X

            col_losses[:] = 0
            for buck in buckets:
                col_losses += buck.col_sum_sqs()
            try_dims = np.argsort(col_losses)[-try_ndims:]

            from scipy import stats

            col_losses *= col_losses  # just 4th central moment
            col_losses *= stats.kurtosis(X_res, axis=0)
            try_dims = np.argsort(col_losses)[-try_ndims:]

        losses = np.zeros(len(try_dims), dtype=X.dtype)
        all_split_vals = []  # vals chosen by each bucket/group for each dim

        # determine for this dim what the best split vals are for each
        # group and what the loss is when using these split vals
        # print("try_dims: ", try_dims)
        for d, dim in enumerate(try_dims):
            # print("s, d, dim = ", s, d, dim)
            if verbose > 2:
                # print("---------------------- dim = ", dim)
                print(
                    "======== dim = {}, ({:.5f}, {:.5f})".format(
                        dim, np.min(X[:, dim]), np.max(X[:, dim])
                    )
                )
            split_vals = []  # each bucket contributes one split val
            for b, buck in enumerate(buckets):
                val, loss = buck.optimal_split_val(X, dim, X_orig=X_orig)
                losses[d] += loss
                if d > 0 and losses[d] >= np.min(losses[:d]):
                    if verbose > 2:
                        print("early abandoning after bucket {}!".format(b))
                    break  # this dim already can't be the best
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
            x = X[:, best_dim]
            offset = (np.min(x) + np.min(use_split_vals)) / 2
            upper_val = (np.max(x) + np.max(use_split_vals)) / 2 - offset
            scale = 254.0 / upper_val
            if learn_quantize_params == "int16":
                scale = 2.0 ** int(np.log2(scale))

            split.offset = offset
            split.scaleby = scale
            split.vals = (split.vals - split.offset) * split.scaleby
            split.vals = np.clip(split.vals, 0, 255).astype(np.int64)

        splits.append(split)

        # apply this split to get next round of buckets
        new_buckets = []
        for i, buck in enumerate(buckets):
            group_idx = i
            val = use_split_vals[group_idx]
            new_buckets += list(buck.split(X, dim=best_dim, val=val, X_orig=X_orig))
        buckets = new_buckets

        if verbose > 1:
            print("bucket counts: ", [buck.N for buck in buckets])
            # print("loss from buckets: ",
            #       sum([bucket.loss for bucket in buckets]))
            print("dim losses: ", losses)
            if verbose > 2:
                print("loss from sse computation: ", losses[best_tried_dim_idx])
                print("using dim, split_vals:", best_dim, use_split_vals)

    # maybe return centroids in addition to set of MultiSplits and loss
    loss = sum([bucket.loss for bucket in buckets])
    if verbose > 0:
        print("learn_multisplits(): returning loss: ", loss)

    ret = [splits, loss]
    if return_centroids:
        centroids = np.vstack([buck.col_means() for buck in buckets])
        assert centroids.shape == (len(buckets), X.shape[1])
        ret.append(centroids)
        # return splits, loss, centroids
    if return_buckets:
        # print("returning buckets!")
        ret.append(buckets)
    return tuple(ret)


@_memory.cache
def _learn_mithral_initialization(X, ncodebooks, pq_perm_algo="start", **kwargs):
    N, D = X.shape
    ncentroids_per_codebook = 16

    X = X.astype(np.float32)
    X_res = X.copy()
    X_orig = X

    all_centroids = np.zeros((ncodebooks, ncentroids_per_codebook, D), dtype=np.float32)
    all_splits = []
    pq_idxs = _pq_codebook_start_end_idxs(X, ncodebooks, algo=pq_perm_algo)
    subvec_len = int(np.ceil(D / ncodebooks))  # for non-pq heuristics

    nonzeros_heuristic = "pq"

    # ------------------------ 0th iteration; initialize all codebooks
    all_splits = []
    all_buckets = []
    for c in range(ncodebooks):
        if nonzeros_heuristic == "pq":
            start_idx, end_idx = pq_idxs[c]
            idxs = np.arange(start_idx, end_idx)
        elif nonzeros_heuristic == "pca":
            v = top_principal_component(X_res)
            idxs = np.argsort(np.abs(v))[:-subvec_len]
        elif nonzeros_heuristic == "disjoint_pca":
            use_X_res = X_res.copy()
            if c > 0:  # not the first codebook
                use_X_res[:, idxs] = 0  # can't use same subspace
            v = top_principal_component(use_X_res)
            idxs = np.argsort(np.abs(v))[:-subvec_len]

        use_X_res = X_res[:, idxs]
        use_X_orig = X_orig[:, idxs]

        # learn codebook to soak current residuals
        multisplits, _, buckets = learn_multisplits(
            use_X_res,
            X_orig=use_X_orig,
            return_centroids=False,
            return_buckets=True,
            **kwargs
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

        # print("X_res mse / X mse: ",
        #       (X_res * X_res).mean() / (X_orig * X_orig).mean())

    return X_res, all_splits, all_centroids, all_buckets


def assignments_from_multisplits(X, splits):
    N, _ = X.shape
    nsplits = len(splits)
    # indicators = np.zeros((nsplits, len(X)), dtype=np.int64)
    assert len(splits) >= 1
    # dim0 = splits[0].dim
    # assert len(splits[0].vals) == 1  # only 1 initial split
    # indicators[0] = X > splits[0].vals[0]

    max_ngroups = len(splits[-1].vals)
    nsplits_affecting_group_id = int(np.log2(max_ngroups))
    assert 1 << nsplits_affecting_group_id == max_ngroups  # power of 2
    # np.log2(max_nsplits)

    # determine group ids for each point; this is the one that's annoying
    # because the number of bits changes after split
    group_ids = np.zeros(N, dtype=np.int64)
    for i in range(min(nsplits, nsplits_affecting_group_id)):
        split = splits[i]
        vals = split.vals[group_ids]
        # x = X[:, split.dim]
        # if split.offset is not None:
        #     x = x - split.offset
        # if split.scaleby is not None:
        #     x = x * split.scaleby
        # indicators = x > vals
        indicators = split.preprocess_x(X[:, split.dim]) > vals
        group_ids = (group_ids * 2) + indicators

    if nsplits <= nsplits_affecting_group_id:
        return group_ids

    # compute remaining bits
    assignments = np.copy(group_ids)
    for i in range(nsplits_affecting_group_id, nsplits):
        split = splits[i]
        vals = split.vals[group_ids]
        # x = X[:, split.dim]
        # if split.offset is not None:
        #     x = x - split.offset
        # if split.scaleby is not None:
        #     x = x * split.scaleby
        # indicators = x > vals
        indicators = split.preprocess_x(X[:, split.dim]) > vals
        assignments = (assignments * 2) + indicators

    return assignments


def mithral_encode(X, multisplits_lists):
    N, D = X.shape
    ncodebooks = len(multisplits_lists)
    X_enc = np.empty((N, ncodebooks), dtype=np.int64, order="f")
    for c in range(ncodebooks):
        X_enc[:, c] = assignments_from_multisplits(X, multisplits_lists[c])
    return np.ascontiguousarray(X_enc)


@numba.njit(fastmath=True, cache=True)
def _densify_X_enc(X_enc, K=16):
    N, C = X_enc.shape
    D = C * K
    out = np.zeros((N, D), np.int8)
    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            out[n, dim_left] = 1

    return out


def _fit_ridge_enc(X_enc=None, Y=None, K=16, lamda=1, X_bin=None):
    if X_bin is None:
        X_bin = _densify_X_enc(X_enc, K=K)
    est = linear_model.Ridge(fit_intercept=False, alpha=lamda)
    est.fit(X_bin, Y)
    return est.coef_.T


@numba.njit(fastmath=True, cache=True)
def _XtX_encoded(X_enc, K=16):
    N, C = X_enc.shape
    D = C * K  # note that this is total number of centroids, not orig D

    out = np.zeros((D, D), np.int32)
    # out = np.zeros((D, D), np.float32)
    # D = int(C * K)  # note that this is total number of centroids, not orig D
    # out = np.zeros((D, D), np.int8)

    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            out[dim_left, dim_left] += 1
            for cc in range(c + 1, C):
                code_right = X_enc[n, cc]
                dim_right = (K * cc) + code_right
                out[dim_left, dim_right] += 1

    # populate lower triangle
    for d in range(D):
        for dd in range(d + 1, D):
            out[dd, d] = out[d, dd]

    return out


@numba.njit(fastmath=True, cache=True)
def _XtY_encoded(X_enc, Y, K=16):
    N, C = X_enc.shape
    N, M = Y.shape

    D = int(C * K)  # note that this is total number of centroids, not orig D
    out = np.zeros((D, M), Y.dtype)

    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            for m in range(M):
                out[dim_left, m] += Y[n, m]

    return out


@numba.njit(fastmath=True, cache=True)
def _XW_encoded(X_enc, W, K=16):
    N, C = X_enc.shape
    D, M = W.shape

    out = np.zeros((N, M), W.dtype)

    for n in range(N):
        for c in range(C):
            code_left = X_enc[n, c]
            dim_left = (K * c) + code_left
            for m in range(M):
                out[n, m] += W[dim_left, m]

    return out


def encoded_lstsq(
    X_enc=None,
    X_bin=None,
    Y=None,
    K=16,
    XtX=None,
    XtY=None,
    precondition=True,
    stable_ridge=True,
):

    if stable_ridge:
        return _fit_ridge_enc(X_enc=X_enc, Y=Y, X_bin=X_bin, K=K, lamda=1)

    if XtX is None:
        XtX = _XtX_encoded(X_enc, K=K).astype(np.float32)
        lamda = 1  # TODO cross-validate to get lamda

        # N = X_enc.shape[0]
        # # lamda = N / (K * K)
        # Y_bar = Y - Y.mean(axis=0)
        # lamda = N * np.var(Y - Y.mean(axis=0)) / (K * K)
        # # lamda = N * np.var(Y - Y.mean(axis=0)) / K
        # lamda = N * np.var(Y) / K
        # lamda = N * np.var(Y) / (K * K)
        # # lamda = N * 1e4  # should shrink coeffs to almost 0
        # # alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
        # lamda = N / (1e5)  # sorta works
        # lamda = N / (1e4) # sorta works

        lamda = max(1, lamda)
        print("using lamda = ", lamda)

        # lamda = max(1, len(X_enc) / 1e6)
        # lamda = max(1, len(X_enc) / 1e5)
        # lamda = max(1, len(X_enc) / 1e4)
        # lamda = max(1, len(X_enc) / float(K * K))
        # lamda = len(X_enc) / float(K)
        # print("computing and regularizing XtX using lambda = ", lamda)
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge

    if XtY is None:
        XtY = _XtY_encoded(X_enc, Y, K=K)

    XtX = XtX.astype(np.float64)
    XtY = XtY.astype(np.float64)

    # preconditioning to avoid numerical issues (seemingly unnecessary, but
    # might as well do it)
    # scale = 1. / np.std(XtX)
    if precondition:

        # # pretend cols of X were scaled differently
        # xscales = np.linalg.norm(XtX, axis=0) + 1e-20
        # mulby = (1. / xscales)
        # XtX *= mulby * mulby
        # XtY *= mulby.reshape(-1, 1)

        # yscales = np.linalg.norm(XtY, axis=1) + 1e-20
        # yscales = np.linalg.norm(XtY, axis=0) + 1e-20
        # yscales = yscales.reshape(-1, 1)

        # xscales = np.mean(np.linalg.norm(XtX, axis=0))
        # xscales = 7
        # xscales = 1

        # XtY *= (1. / yscales)
        # XtY *= (1. / yscales.reshape(-1, 1))

        # scale = 1. / len(X_enc)
        scale = 1.0 / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

    # W = np.linalg.solve(XtX, XtY)
    W, _, _, _ = np.linalg.lstsq(XtX, XtY, rcond=None)  # doesn't fix it

    # W, _, _, _ = np.linalg.lstsq(X_bin, Y, rcond=None)

    # import torch
    # import torch.nn.functional as F
    # import torch.optim as optim

    # def _to_np(A):
    #     return A.cpu().detach().numpy()

    # niters = 10
    # for it in range(niters):

    # if precondition:
    #     pass
    #     # W *= xscales
    #     # W *= xscales.reshape(-1, 1)
    #     # W /= xscales.reshape(-1, 1)
    #     # W *= yscales.ravel()
    #     # W *= yscales

    # W *= yscales  # undo preconditioning

    # import matplotlib.pyplot as plt
    # _, axes = plt.subplots(2, 2, figsize=(13, 10))
    # axes[0, 0].imshow(_densify_X_enc(X_enc[:1000]), interpolation='nearest')
    # axes[0, 1].imshow(XtX, interpolation='nearest')
    # axes[1, 0].imshow(XtY, interpolation='nearest', cmap='RdBu')
    # axes[1, 1].imshow(W, interpolation='nearest', cmap='RdBu')
    # # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    # import sys; sys.exit()

    return W


# each codebook has const number of nonzero idxs
def _sparse_encoded_lstsq_elim_v2(
    X_enc,
    Y,
    nnz_per_centroid,
    K=16,
    # uniform_sparsity=False):  # never better
    uniform_sparsity=True,
    pq_perm_algo="start",
    stable_ridge=True,
):
    ncodebooks = X_enc.shape[1]
    M = Y.shape[1]
    nnz_per_centroid = min(M, int(nnz_per_centroid))
    nnz_per_centroid = max(1, nnz_per_centroid)
    assert nnz_per_centroid >= int(np.ceil(M / ncodebooks))
    assert nnz_per_centroid <= M

    X_bin = _densify_X_enc(X_enc, K=K)

    if not stable_ridge:
        # precompute XtX and XtY and create initial dense W
        XtX = _XtX_encoded(X_enc, K=K).astype(np.float32)

        lamda = 1
        # # alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
        # # lamda = np.sqrt(ncodebooks)
        # N = XtX.shape[0]
        # lamda = N / (K * K)
        # lamda = max(1, lamda)
        # print("using lamda = ", lamda)

        # lamda = max(1, len(X_enc) / 1e4)
        # lamda = max(1, len(X_enc) / float(K * K))
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge
        # XtX += np.diag(np.ones(XtX.shape[0])).astype(np.float32)  # ridge
        XtY = _XtY_encoded(X_enc, Y, K=K)

        # scale = 1. / len(X_enc)
        scale = 1.0 / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

        W = encoded_lstsq(
            X_bin=X_bin,
            Y=Y,
            XtX=XtX,
            XtY=XtY,
            precondition=False,
            stable_ridge=stable_ridge,
        )  # KC x M

        XtX = np.asfarray(XtX)  # since we'll be slicing columns
    else:  # stable_ridge is True
        W = encoded_lstsq(X_bin=X_bin, Y=Y, stable_ridge=stable_ridge)

    # score all blocks of W
    all_scores = np.empty((ncodebooks, M), dtype=np.float)  # C x M
    for c in range(ncodebooks):
        Xc = X_enc[:, c].reshape(-1, 1)
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]

        Yc = _XtY_encoded(Xc, Wc, K=K)  # N x M
        all_scores[c] = np.linalg.norm(Yc, axis=0)

    # pq_idxs = _pq_codebook_start_end_idxs(M, ncodebooks)
    pq_idxs = _pq_codebook_start_end_idxs(Y, ncodebooks, algo=pq_perm_algo)

    # now pick which cols to keep in each codebook
    keep_mask = np.zeros((ncodebooks, M), dtype=np.bool)
    # subvec_len = int(np.ceil(M / ncodebooks))
    for c in range(ncodebooks):
        # initialize with PQ
        start_idx, end_idx = pq_idxs[c]
        keep_mask[c, start_idx:end_idx] = 1

        subvec_len = end_idx - start_idx
        assert subvec_len >= 1
        keep_nidxs_extra = nnz_per_centroid - subvec_len
        scores = all_scores[c]
        scores[start_idx:end_idx] = 0

        if uniform_sparsity and keep_nidxs_extra > 0:
            # take as many other (best) nonzero idxs as we we're allowed to
            assert len(scores) >= keep_nidxs_extra
            best_idxs = np.argsort(scores)[-keep_nidxs_extra:]
            if len(best_idxs) != keep_nidxs_extra:
                print("len(best_idxs)", len(best_idxs))
                print("keep_nidxs_extra", keep_nidxs_extra)
                assert len(best_idxs) == keep_nidxs_extra
            keep_mask[c, best_idxs] = True

    if not uniform_sparsity:
        scores = all_scores.ravel()
        nkept_idxs = M  # number of nonzeros used already
        keep_nidxs_total = nnz_per_centroid * ncodebooks
        keep_nidxs_extra = keep_nidxs_total - nkept_idxs
        keep_idxs = np.argsort(scores)[-keep_nidxs_extra:]
        flat_mask = keep_mask.ravel()
        flat_mask[keep_idxs] = 1
        keep_mask = flat_mask.reshape(keep_mask.shape)

    # at this point, we have the mask for which cols of each centroid to keep;
    # now we just need to go from a mask to a set of indices and a sparse
    # matrix of centroids
    W_sparse = np.empty((ncodebooks * K, M), dtype=np.float32)
    if uniform_sparsity:
        ret_idxs = np.empty((ncodebooks, nnz_per_centroid), dtype=np.int64)
    else:
        ret_idxs = []
    # else:
    # ret_idxs = np.zeros((ncodebooks, M), dtype=np.int64) - 1
    for c in range(ncodebooks):
        idxs = np.where(keep_mask[c] != 0)[0]
        if uniform_sparsity:
            if len(idxs) != nnz_per_centroid:
                print("c: ", c)
                print("len(idxs): ", len(idxs))
                print("nnz_per_centroid: ", nnz_per_centroid)
                print("keep_mask counts:", keep_mask.sum(axis=1))
                assert len(idxs) == nnz_per_centroid
            ret_idxs[c] = idxs
        else:
            ret_idxs.append(idxs)

        zero_idxs = np.where(keep_mask[c] == 0)[0]
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]
        Wc[:, zero_idxs] = 0
        W_sparse[start_idx:end_idx] = Wc

    # now refit W_sparse to each output col; right now it's just the original
    # W with a bunch of entries zeroed
    for m in range(M):
        w = W_sparse[:, m]
        keep_idxs = np.where(w != 0)[0]

        if stable_ridge:
            X_bin_subs = X_bin[:, keep_idxs]
            w_subs = _fit_ridge_enc(X_bin=X_bin_subs, Y=Y[:, m])
        else:
            xty = XtY[:, m]
            use_XtX = XtX[keep_idxs][:, keep_idxs]
            use_xty = xty[keep_idxs]
            w_subs = np.linalg.solve(use_XtX, use_xty)
        w[:] = 0
        w[keep_idxs] = w_subs
        W_sparse[:, m] = w

    # nnzs = [len(idxs) for idxs in ret_idxs]
    # print("nnzs: ", nnzs)

    # print(f"returning {ret_idxs.shape[1]} nonzeros per centroid...")
    return W_sparse, ret_idxs


def sparse_encoded_lstsq(X_enc, Y, K=16, nnz_blocks=-1, **kwargs):
    ncodebooks = X_enc.shape[1]
    if nnz_blocks < 1:
        # nnz_per_centroid = Y.shape[1]
        # default to returning dense centroids
        W = encoded_lstsq(X_enc, Y, K=16)
        ncodebooks = X_enc.shape[1]
        M = Y.shape[1]
        keep_codebook_idxs = np.empty((ncodebooks, M), dtype=np.int64)
        all_idxs = np.arange(M)
        for c in range(ncodebooks):
            keep_codebook_idxs[c] = all_idxs
        return W, keep_codebook_idxs
    else:
        nnz_per_centroid = int(nnz_blocks * Y.shape[1] / ncodebooks)

        # nnz_blocks = int(np.sqrt(ncodebooks) + .5)

    # return _sparse_encoded_lstsq_backward_elim(
    #     X_enc, Y, nnz_blocks=nnz_blocks, K=K)
    # return _sparse_encoded_lstsq_gomp(X_enc, Y, nnz_blocks=nnz_blocks, K=K)

    # print("nnz_per_centroid: ", nnz_per_centroid)
    return _sparse_encoded_lstsq_elim_v2(
        X_enc, Y, nnz_per_centroid=nnz_per_centroid, K=K, **kwargs
    )


@_memory.cache
def learn_mithral(X, ncodebooks, return_buckets=False, lut_work_const=-1, **kwargs):
    N, D = X.shape
    ncentroids_per_codebook = 16
    X_orig = X.astype(np.float32)

    X_res0, all_splits0, all_centroids0, all_buckets0 = _learn_mithral_initialization(
        X, ncodebooks, pq_perm_algo="start"
    )

    mse_orig = (X_orig * X_orig).mean()
    mse0 = (X_res0 * X_res0).mean()
    print("X_res mse / X mse: ", mse0 / mse_orig)

    used_perm_algo = "start"
    if False:
        # choose between having wider codebooks at the start vs the end (if
        # there might be a meaningful difference)
        (
            X_res1,
            all_splits1,
            all_centroids1,
            all_buckets1,
        ) = _learn_mithral_initialization(X, ncodebooks, pq_perm_algo="end")
        mse1 = (X_res1 * X_res1).mean()

        if mse0 <= mse1:
            X_res, all_splits, all_centroids, all_buckets = (
                X_res0,
                all_splits0,
                all_centroids0,
                all_buckets0,
            )
        else:
            X_res, all_splits, all_centroids, all_buckets = (
                X_res1,
                all_splits1,
                all_centroids1,
                all_buckets1,
            )
            used_perm_algo = "end"

        print("X_res1 mse / X mse: ", mse1 / mse_orig)
    else:
        X_res, all_splits, all_centroids, all_buckets = (
            X_res0,
            all_splits0,
            all_centroids0,
            all_buckets0,
        )

    # optimize centroids discriminatively conditioned on assignments
    X_enc = mithral_encode(X, all_splits)

    if lut_work_const != 1:  # if it's 1, equivalent to just doing PQ
        #
        # shrink W towards 0
        #
        # if lut_work_const < 0:
        #     W = encoded_lstsq(X_enc, X)
        # else:
        #     W, nonzero_blocks = sparse_encoded_lstsq(
        #         X_enc, X, nnz_blocks=lut_work_const)

        #
        # shrink W towards initial centroids
        #
        if lut_work_const < 0:
            print("fitting dense lstsq to X_res")
            W = encoded_lstsq(X_enc=X_enc, Y=X_res)
        else:
            W, _ = sparse_encoded_lstsq(
                X_enc, X_res, nnz_blocks=lut_work_const, pq_perm_algo=used_perm_algo
            )

        all_centroids_delta = W.reshape(ncodebooks, ncentroids_per_codebook, D)
        all_centroids += all_centroids_delta

        # check how much improvement we got
        X_res -= _XW_encoded(X_enc, W)  # if we fit to X_res
        mse_res = (X_res * X_res).mean()
        print("X_res mse / X mse after lstsq: ", mse_res / mse_orig)
        # print("min, median, max, std, of all centroids after lstsq:\n",
        #       all_centroids.min(), np.median(all_centroids),
        #       all_centroids.max(), all_centroids.std())

    if return_buckets:
        return all_splits, all_centroids, all_buckets
    return all_splits, all_centroids


def mithral_lut(q, all_centroids):
    q = q.reshape(1, 1, -1)  # all_centroids is shape ncodebooks, ncentroids, D
    return (q * all_centroids).sum(axis=2)  # ncodebooks, ncentroids


def _mithral_quantize_luts(luts, lut_work_const, force_power_of_2=True):
    nqueries, ncodebooks, ncentroids = luts.shape

    # if lut_work_const < 0:  # not time constrained
    #     assert luts.shape == (nqueries, ncodebooks, ncentroids)
    #     luts2d = np.moveaxis(luts, 2, 1)
    #     assert luts2d.shape == (nqueries, ncentroids, ncodebooks)
    #     luts2d = luts2d.reshape(nqueries * ncentroids, ncodebooks)

    #     # if True:
    #     if False:
    #         # ax = sb.distplot(luts.ravel(), hist=False, rug=True)
    #         _, ax = plt.subplots(1, figsize=(13, 5))
    #         # sb.violinplot(data=luts2d, inner='point', ax=ax)
    #         # sb.boxenplot(data=luts2d, ax=ax)
    #         means = luts2d.mean(axis=0)

    #         # # rm largest and smallest entry in each col
    #         # argmaxs = np.argmax(luts2d, axis=0)
    #         # argmins = np.argmax(luts2d, axis=0)
    #         # for c in range(luts.shape[1]):
    #         #     luts2d[argmins[c], c] = means[c]
    #         #     luts2d[argmaxs[c], c] = means[c]

    #         maxs = luts2d.max(axis=0)
    #         mins = luts2d.min(axis=0)

    #         gaps = maxs - mins
    #         max_idx = np.argmax(gaps)
    #         print(f"biggest gap = {np.max(gaps)} at idx {max_idx}")
    #         gaps[max_idx] = 0
    #         max_idx = np.argmax(gaps)
    #         print(f"2nd biggest gap = {np.max(gaps)} at idx {max_idx}")
    #         gaps[max_idx] = 0
    #         max_idx = np.argmax(gaps)
    #         print(f"3rd biggest gap = {np.max(gaps)} at idx {max_idx}")
    #         gaps[max_idx] = 0
    #         max_idx = np.argmax(gaps)
    #         print(f"4th biggest gap = {np.max(gaps)} at idx {max_idx}")
    #         gaps[max_idx] = 0
    #         max_idx = np.argmax(gaps)
    #         print(f"5th biggest gap = {np.max(gaps)} at idx {max_idx}")

    #         # for i in range(len(luts2d)):
    #         #     row = luts2d[i]
    #         #     luts2d[i, row == mins] = means
    #         #     luts2d[i, row == maxs] = means

    #         luts2d -= mins
    #         # luts2d -= means
    #         # luts2d *= 255 / (maxs - mins).max()
    #         luts2d *= 255 / gaps.max()
    #         luts2d = np.minimum(luts2d, 255)

    #         sb.stripplot(data=luts2d, ax=ax, size=4)
    #         ax.set_xlabel('Query dist to centroids (lut dist histogram)')
    #         ax.set_ylabel('Fraction of queries')
    #         plt.show()
    #         import sys; sys.exit()

    #     offsets, scale, _ = _learn_best_quantization(luts2d)
    #     offsets = offsets[np.newaxis, :, np.newaxis]
    #     luts = np.maximum(0, luts - offsets) * scale
    #     luts = np.floor(luts).astype(np.int64)
    #     luts = np.minimum(255, luts)
    #     return luts, offsets.sum(), scale

    # luts = np.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
    mins = luts.min(axis=(0, 2))
    maxs = luts.max(axis=(0, 2))

    gaps = maxs - mins
    # gaps[np.argmax(gaps)] = 0  # use 2nd highest
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
    # luts_quantized = np.minimum(luts_quantized, 255)

    assert np.min(luts_quantized) >= 0
    assert np.max(luts_quantized) <= 255.0

    # print("total offset: ", mins.sum())

    return luts_quantized, offsets.sum(), scale

    # # compute offset taking into account stuff getting rounded down
    # luts_hat = (luts / scale) + offsets
    # diffs = luts - luts_hat
    # print("mean of diffs: ", diffs.mean())
    # offset = diffs.mean() + offsets.sum()

    # return luts_quantized, offset, scale


class MithralEncoder(MultiCodebookEncoder):
    def __init__(self, ncodebooks, lut_work_const=-1):
        super().__init__(
            ncodebooks=ncodebooks,
            ncentroids=16,
            # quantize_lut=True, upcast_every=64,
            # quantize_lut=True, upcast_every=32,
            quantize_lut=True,
            upcast_every=16,
            # quantize_lut=True, upcast_every=8,
            # quantize_lut=True, upcast_every=4,
            # quantize_lut=True, upcast_every=2,
            # quantize_lut=True, upcast_every=1,
            accumulate_how="mean",
        )
        self.lut_work_const = lut_work_const

    def name(self):
        return "{}_{}".format("mithral", super().name())

    def params(self):
        return {"ncodebooks": self.ncodebooks, "lut_work_const": self.lut_work_const}

    def fit(self, X, Q=None):
        self.splits_lists, self.centroids = learn_mithral(
            X, self.ncodebooks, lut_work_const=self.lut_work_const
        )
        # self._learn_lut_quantization(X, Q)

    def encode_X(self, X):
        idxs = mithral_encode(X, self.splits_lists)
        return idxs + self.offsets

    def encode_Q(self, Q, quantize=True):
        Q = np.atleast_2d(Q)
        luts = np.zeros((Q.shape[0], self.ncodebooks, self.ncentroids))
        for i, q in enumerate(Q):
            luts[i] = mithral_lut(q, self.centroids)
        if self.quantize_lut:
            luts, offset, scale = _mithral_quantize_luts(luts, self.lut_work_const)
            return luts, offset, scale

        return luts, 0, 1


class VQMatmul(abc.ABC):
    def __init__(self, ncodebooks, ncentroids=None):
        self.ncodebooks = ncodebooks
        self.ncentroids = self._get_ncentroids() if ncentroids is None else ncentroids
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    @abc.abstractmethod
    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        pass

    # @abc.abstractmethod
    def _get_ncentroids(self):
        pass

    @abc.abstractmethod
    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        pass

    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def reset_for_new_task(self):
        self.A_enc = None
        self.luts = None

    def fit(self, A, B, Y=None):
        _, D = A.shape
        if D < self.ncodebooks:
            raise Exception("D < C: {} < {}".format(D, self.ncodebooks))
        self.enc.fit(A, B.T)

    def set_A(self, A):
        self.A_enc = self.enc.encode_X(A)

    def set_B(self, B):
        self.luts = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts)

    def get_params(self):
        return {"ncodebooks": self.ncodebooks}


class MithralMatmul(VQMatmul):
    def __init__(self, ncodebooks, lut_work_const=-1):
        self.lut_work_const = lut_work_const
        if (
            (lut_work_const is not None)
            and (lut_work_const > 0)
            and (lut_work_const > ncodebooks)
        ):
            raise Exception(
                "lut_work_const > ncodebooks: {} > {}".format(
                    lut_work_const, ncodebooks
                )
            )
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    def _create_encoder(self, ncodebooks):
        return MithralEncoder(ncodebooks=ncodebooks, lut_work_const=self.lut_work_const)

    def get_params(self):
        return {"ncodebooks": self.ncodebooks, "lut_work_const": self.lut_work_const}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.ncentroids * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.ncodebooks
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        # return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def set_B(self, B):
        self.luts, self.offset, self.scale = self.enc.encode_Q(B.T)

    def predict(self, A, B):
        return self(A, B)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(
            self.A_enc, self.luts, offset=self.offset, scale=self.scale
        )


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

    # # TODO rm all this after debug
    # logits_test = Y_test + b
    # print("logits_test.shape", logits_test.shape)
    # print("lbls_test.shape", lbls_test.shape)
    # lbls_hat_test = np.argmax(Y_test, axis=1)
    # print("lbls_hat_test.shape", lbls_hat_test.shape)
    # acc = np.mean(lbls_hat_test.ravel() == lbls_test.ravel())
    # print("Y_test: ", Y_test[:10])
    # print("Y_train head: ", Y_train[:10])
    # print("Y_train tail: ", Y_train[-10:])
    # print("b:\n", b)
    # # print("lbls hat test:")
    # # print(lbls_hat_test[:100])
    # # print("lbls test:")
    # # print(lbls_test[:100])
    # print("lbls train:")
    # print(lbls_train[:100])
    # print("acc: ", acc)

    info = {
        "problem": "softmax",
        "biases": b,
        "lbls_train": lbls_train,
        "lbls_test": lbls_test,
    }

    return (X_train, Y_train, X_test, Y_test, W)


def main():
    X_train, Y_train, X_test, Y_test, W = load_cifar100_tasks()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, W.shape)
    print(X_train)

    maddness = MithralMatmul(ncodebooks=16, lut_work_const=-1)
    maddness.fit(X_train, W)

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

