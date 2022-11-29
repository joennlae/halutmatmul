# type: ignore
import numba
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import linear_model

from maddness.util.hash_function_helper import create_codebook_start_end_idxs


@numba.njit(fastmath=True, cache=True)
def sparsify_and_int8_A_enc(A_enc, K=16):
    """
    returns X_binary from an encoded Matrix [N, C] vals (0-K)
    to
    [[0 0 0 ... 0 0 0]
     [0 0 1 ... 0 0 0]
     ...
     [0 0 0 ... 0 0 0]
     [0 0 0 ... 1 0 0]
     [0 0 0 ... 0 0 0]]
    """
    N, C = A_enc.shape
    D = C * K
    out = np.zeros((N, D), np.int8)
    for n in range(N):
        for c in range(C):
            code_left = A_enc[n, c]
            dim_left = (K * c) + code_left
            out[n, dim_left] = 1
    return out


def _fit_ridge_enc(A_enc=None, Y=None, K=16, lamda=1, X_binary=None):
    """
    minimize loss of |Y - Xw|^2 + alpha * |w|^2
    X is binary in our case -> w without entry in X
    X [N, C * K]
    Y [N, C]
    W [D, C * K] -> W.T [C * K, D] later reshaped to
    [C, K, D] -> prototype dimensons
    """
    if X_binary is None:
        X_binary = sparsify_and_int8_A_enc(A_enc, K=K)
    print(X_binary.shape, Y.shape)
    # X_binary_sparse = csr_matrix(X_binary) # will change solver from cholesky to sparse_cg
    est = linear_model.Ridge(
        fit_intercept=False, alpha=lamda, solver="auto", copy_X=False
    )
    est.fit(X_binary, Y)
    w = est.coef_.T
    print(est.get_params())
    return w


@numba.njit(fastmath=True, cache=True)
def _XtA_encoded(A_enc, K=16):
    N, C = A_enc.shape
    D = C * K  # note that this is total number of centroids, not orig D

    out = np.zeros((D, D), np.int32)
    # out = np.zeros((D, D), np.float32)
    # D = int(C * K)  # note that this is total number of centroids, not orig D
    # out = np.zeros((D, D), np.int8)

    for n in range(N):
        for c in range(C):
            code_left = A_enc[n, c]
            dim_left = (K * c) + code_left
            out[dim_left, dim_left] += 1
            for cc in range(c + 1, C):
                code_right = A_enc[n, cc]
                dim_right = (K * cc) + code_right
                out[dim_left, dim_right] += 1

    # populate lower triangle
    for d in range(D):
        for dd in range(d + 1, D):
            out[dd, d] = out[d, dd]

    return out


@numba.njit(fastmath=True, cache=True)
def _XtY_encoded(A_enc, Y, K=16):
    N, C = A_enc.shape
    N, M = Y.shape

    D = int(C * K)  # note that this is total number of centroids, not orig D
    out = np.zeros((D, M), Y.dtype)

    for n in range(N):
        for c in range(C):
            code_left = A_enc[n, c]
            dim_left = (K * c) + code_left
            for m in range(M):
                out[dim_left, m] += Y[n, m]

    return out


@numba.njit(fastmath=True, cache=True)
def _XW_encoded(A_enc, W, K=16):
    N, C = A_enc.shape
    _, M = W.shape

    out = np.zeros((N, M), W.dtype)

    encoded_shifted = A_enc + np.repeat(np.arange(C) * K, N).reshape(N, C)
    for n in range(N):
        for c in range(C):
            out[n, :] += W[encoded_shifted[n, c], :]

    return out


# equation 8 in paper
def encoded_lstsq(
    A_enc=None,
    X_binary=None,
    Y=None,
    K=16,
    XtX=None,
    XtY=None,
    precondition=True,
    stable_ridge=True,
):
    if stable_ridge:
        return _fit_ridge_enc(A_enc=A_enc, Y=Y, X_binary=X_binary, K=K, lamda=1)

    if XtX is None:
        XtX = _XtA_encoded(A_enc, K=K).astype(np.float32)
        lamda = 1  # TODO cross-validate to get lamda
        # different lambda have been tested in original code
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge

    if XtY is None:
        XtY = _XtY_encoded(A_enc, Y, K=K)

    XtX = XtX.astype(np.float64)
    XtY = XtY.astype(np.float64)

    # preconditioning to avoid numerical issues (seemingly unnecessary, but
    # might as well do it)
    if precondition:
        # different preconditions have been tested
        scale = 1.0 / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

    W, _, _, _ = np.linalg.lstsq(XtX, XtY, rcond=None)  # doesn't fix it

    return W


# each codebook has const number of nonzero idxs
def _sparse_encoded_lstsq_elim_v2(
    A_enc,
    Y,
    nnz_per_centroid,
    K=16,
    # uniform_sparsity=False):  # never better
    uniform_sparsity=True,
    pq_perm_algo="start",
    stable_ridge=True,
):
    number_of_codebooks = A_enc.shape[1]
    M = Y.shape[1]
    nnz_per_centroid = min(M, int(nnz_per_centroid))
    nnz_per_centroid = max(1, nnz_per_centroid)
    assert nnz_per_centroid >= int(np.ceil(M / number_of_codebooks))
    assert nnz_per_centroid <= M

    X_binary = sparsify_and_int8_A_enc(A_enc, K=K)

    if not stable_ridge:
        # precompute XtX and XtY and create initial dense W
        XtX = _XtA_encoded(A_enc, K=K).astype(np.float32)

        lamda = 1
        # # alpha = unscaled_alpha * np.var(X - X.mean(axis=0)) * N / D
        # # lamda = np.sqrt(number_of_codebooks)
        # N = XtX.shape[0]
        # lamda = N / (K * K)
        # lamda = max(1, lamda)
        # print("using lamda = ", lamda)

        # lamda = max(1, len(A_enc) / 1e4)
        # lamda = max(1, len(A_enc) / float(K * K))
        XtX += np.diag(np.ones(XtX.shape[0]) * lamda).astype(np.float32)  # ridge
        # XtX += np.diag(np.ones(XtX.shape[0])).astype(np.float32)  # ridge
        XtY = _XtY_encoded(A_enc, Y, K=K)

        # scale = 1. / len(A_enc)
        scale = 1.0 / np.linalg.norm(XtX, axis=0).max()
        XtX = XtX * scale
        XtY = XtY * scale

        W = encoded_lstsq(
            X_binary=X_binary,
            Y=Y,
            XtX=XtX,
            XtY=XtY,
            precondition=False,
            stable_ridge=stable_ridge,
        )  # KC x M

        XtX = np.asfarray(XtX)  # since we'll be slicing columns
    else:  # stable_ridge is True
        W = encoded_lstsq(X_binary=X_binary, Y=Y, stable_ridge=stable_ridge)

    # score all blocks of W
    all_scores = np.empty((number_of_codebooks, M), dtype=np.float32)  # C x M
    for c in range(number_of_codebooks):
        Xc = A_enc[:, c].reshape(-1, 1)
        start_idx = c * K
        end_idx = start_idx + K
        Wc = W[start_idx:end_idx]

        Yc = _XtY_encoded(Xc, Wc, K=K)  # N x M
        all_scores[c] = np.linalg.norm(Yc, axis=0)

    # pq_idxs = create_codebook_start_end_idxs(M, number_of_codebooks)
    pq_idxs = create_codebook_start_end_idxs(Y, number_of_codebooks, algo=pq_perm_algo)

    # now pick which cols to keep in each codebook
    keep_mask = np.zeros((number_of_codebooks, M), dtype=np.bool_)
    # subvec_len = int(np.ceil(M / number_of_codebooks))
    for c in range(number_of_codebooks):
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
        keep_nidxs_total = nnz_per_centroid * number_of_codebooks
        keep_nidxs_extra = keep_nidxs_total - nkept_idxs
        keep_idxs = np.argsort(scores)[-keep_nidxs_extra:]
        flat_mask = keep_mask.ravel()
        flat_mask[keep_idxs] = 1
        keep_mask = flat_mask.reshape(keep_mask.shape)

    # at this point, we have the mask for which cols of each centroid to keep;
    # now we just need to go from a mask to a set of indices and a sparse
    # matrix of centroids
    W_sparse = np.empty((number_of_codebooks * K, M), dtype=np.float32)
    if uniform_sparsity:
        ret_idxs = np.empty((number_of_codebooks, nnz_per_centroid), dtype=np.int64)
    else:
        ret_idxs = []
    # else:
    # ret_idxs = np.zeros((number_of_codebooks, M), dtype=np.int64) - 1
    for c in range(number_of_codebooks):
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
            X_binary_subs = X_binary[:, keep_idxs]
            w_subs = _fit_ridge_enc(X_binary=X_binary_subs, Y=Y[:, m])
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


# pylint: disable=R1705
def sparse_encoded_lstsq(A_enc, Y, K=16, nnz_blocks=-1, **kwargs):
    number_of_codebooks = A_enc.shape[1]
    if nnz_blocks < 1:
        # default to returning dense centroids
        W = encoded_lstsq(A_enc, Y, K=16)
        number_of_codebooks = A_enc.shape[1]
        M = Y.shape[1]
        keep_codebook_idxs = np.empty((number_of_codebooks, M), dtype=np.int64)
        all_idxs = np.arange(M)
        for c in range(number_of_codebooks):
            keep_codebook_idxs[c] = all_idxs
        return W, keep_codebook_idxs
    else:
        nnz_per_centroid = int(nnz_blocks * Y.shape[1] / number_of_codebooks)

    return _sparse_encoded_lstsq_elim_v2(
        A_enc, Y, nnz_per_centroid=nnz_per_centroid, K=K, **kwargs
    )
