# pylint: disable=C0413, E1133
from typing import Literal
import resource
import sys
from pathlib import Path
import multiprocessing
from unittest.mock import AsyncMockMixin
import warnings
from joblib import Parallel, delayed

import numpy as np
import numba
from numba import prange

import torch

from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn import tree
from sklearn.tree import _tree
from sklearn.model_selection import cross_val_score

from halutmatmul.functions import create_codebook_start_end_idxs

sys.path.append(
    str(Path(__file__).parent) + "/../../../maddness/python/"
)  # for maddness import


from maddness.util.least_squares import (  # type: ignore[attr-defined]
    encoded_lstsq,
    _XW_encoded,
)


class DecisionTreeOffset:
    DIMS = 0
    THRESHOLDS = 1
    CLASSES = 2
    TOTAL = 3


DEFAULT_NEG_VALUE = -4419


@numba.jit(nopython=True, parallel=False)
def apply_hash_function_decision_tree(
    X: np.ndarray, decision_tree: np.ndarray
) -> np.ndarray:
    N, _ = X.shape
    group_ids = np.zeros(N, dtype=np.int64)  # needs to be int64 because of index :-)
    B = decision_tree.shape[0] // 3
    n_decisions = int(np.log2(B))
    for depth in range(n_decisions):
        index_offet = 2**depth - 1
        split_thresholds = decision_tree[group_ids + B + index_offet]
        dims = decision_tree[group_ids + index_offet].astype(np.int64)
        # x = X[np.arange(N), dims]
        # make it numba compatible
        x = np.zeros(group_ids.shape[0], np.float32)
        for i in range(x.shape[0]):
            x[i] = X[i, dims[i]]
        indicators = x > split_thresholds
        group_ids = (group_ids * 2) + indicators
    group_ids = decision_tree[group_ids + 2 * B].astype(np.int32)
    return group_ids


@numba.jit(nopython=True, parallel=True)
def halut_encode_decision_tree(X: np.ndarray, numpy_array: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    C = numpy_array.shape[0]
    A_enc = np.empty((C, N), dtype=np.int32)  # column-major

    for c in prange(C):
        A_enc[c] = apply_hash_function_decision_tree(X, numpy_array[c])
    return np.ascontiguousarray(A_enc.T)


def apply_hash_function_pq(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    group_ids = np.argsort(
        np.array([np.linalg.norm(X - x, axis=1) for x in prototypes]).T, axis=1
    )[:, :1].flatten()
    return group_ids


def apply_hash_function_pq_tensor(
    X: torch.Tensor, prototypes: torch.Tensor
) -> torch.Tensor:
    group_ids = torch.argsort(
        torch.stack([torch.linalg.norm(X - x, axis=1) for x in prototypes]).T, dim=1
    )[:, :1].flatten()
    return group_ids


def halut_encode_pq(X: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    N, _ = X.shape
    C = prototypes.shape[0]
    A_enc = np.empty((C, N), dtype=np.int32)  # column-major
    pq_idxs = create_codebook_start_end_idxs(X.shape[1], C, algo="start")

    for c in prange(C):
        start_idx, end_idx = pq_idxs[c]
        idxs = np.arange(start_idx, end_idx)
        X_cut = X[:, idxs]
        A_enc[c] = apply_hash_function_pq(X_cut, prototypes[c][:, idxs])
    return np.ascontiguousarray(A_enc.T)


def halut_encode_pq_tensor(X: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    N, _ = X.shape
    C = prototypes.shape[0]
    K = prototypes.shape[1]
    A_enc = torch.empty((C, N), dtype=torch.int32, device=str(X.device))  # column-major
    pq_idxs = create_codebook_start_end_idxs(X.shape[1], C, algo="start")

    for c in prange(C):
        start_idx, end_idx = pq_idxs[c]
        idxs = torch.arange(start_idx, end_idx, device=str(X.device))
        X_cut = X[:, idxs]
        A_enc[c] = apply_hash_function_pq_tensor(X_cut, prototypes[c][:, idxs])

    offsets = torch.arange(C, dtype=torch.int32, device=str(X.device)) * K
    return torch.Tensor.contiguous(A_enc.T) + offsets


def tree_to_numpy(
    decision_tree: tree.DecisionTreeClassifier, depth: int = 4
) -> np.ndarray:
    tree_ = decision_tree.tree_
    class_names = decision_tree.classes_

    B = 2**depth
    total_length = B * DecisionTreeOffset.TOTAL
    numpy_array = np.ones(total_length, np.float32) * DEFAULT_NEG_VALUE

    def _add_leaf(value: int, class_name: int, depth: int, tree_id: int) -> None:
        if tree_id >= B:
            numpy_array[tree_id - B + DecisionTreeOffset.CLASSES * B] = class_name
        else:
            _add_leaf(value, class_name, depth + 1, 2 * tree_id)
            _add_leaf(value, class_name, depth + 1, 2 * tree_id + 1)

    def recurse_tree(node: int, depth: int, tree_id: int) -> None:
        value = None
        if tree_.n_outputs == 1:
            value = tree_.value[node][0]
        else:
            value = tree_.value[node].T[0]
        class_name = np.argmax(value)

        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
            class_name = class_names[class_name]

        # pylint: disable=c-extension-no-member
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            dim = tree_.feature[node]
            threshold = tree_.threshold[node]
            numpy_array[tree_id - 1] = dim
            numpy_array[tree_id - 1 + DecisionTreeOffset.THRESHOLDS * B] = threshold
            recurse_tree(tree_.children_left[node], depth + 1, 2 * tree_id)
            recurse_tree(tree_.children_right[node], depth + 1, 2 * tree_id + 1)
        else:
            _add_leaf(value, class_name, depth, tree_id)  # type: ignore[arg-type]

    recurse_tree(0, 1, 1)

    for i in range(B):
        assert numpy_array[DecisionTreeOffset.CLASSES * B + i] != DEFAULT_NEG_VALUE
        if numpy_array[i] == DEFAULT_NEG_VALUE:
            numpy_array[i] = 0  # adding default dimension TODO: optimize
    return numpy_array


def learn_decision_tree(
    X: np.ndarray, K: int = 16, depth: int = 4, iterations: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    X = X.copy().astype(np.float32)

    decision_tree_args = {
        "min_samples_split": 2,
        "max_depth": depth,
        "min_samples_leaf": 20,
        "max_leaf_nodes": 2**depth,
        "splitter": "best",
        # "criterion": "log_loss",
        # "class_weight": "balanced",
    }
    centroids_list = []
    assignments_list = []
    scores = []

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # ignores empty cluster warning for kmeans
    # pylint: disable=import-outside-toplevel
    from timeit import default_timer as timer

    for _ in range(iterations):
        start = timer()
        centroids_, assignments_ = kmeans2(X, K, minit="points", iter=5)
        end = timer()
        print(f"kmeans time {end - start}")
        # kmeans = KMeans(n_clusters=K, n_init=1).fit(X)
        # kmeans = BisectingKMeans(n_clusters=K, n_init=1).fit(X)
        # centroids_, assignments_ = kmeans.cluster_centers_, kmeans.labels_
        clf_ = tree.DecisionTreeClassifier(**decision_tree_args)
        start = timer()
        score_ = cross_val_score(clf_, X, assignments_, cv=2, n_jobs=2)
        end = timer()
        print(f"cross_val_score time {end - start}", score_)
        centroids_list.append(centroids_)
        assignments_list.append(assignments_)
        scores.append(np.mean(score_))
    best_score = np.argsort(scores)[::-1]
    centroids = centroids_list[best_score[0]]
    assignments = assignments_list[best_score[0]]
    clf = tree.DecisionTreeClassifier(**decision_tree_args)
    clf = clf.fit(X, assignments)

    # additional Infos
    PRINT_DEBUG = False

    numpy_array = tree_to_numpy(clf, depth=depth)

    prediction = clf.predict(X)
    bincount_pred = np.bincount(prediction)
    if PRINT_DEBUG:
        r = tree.export_text(clf)
        print(r)
        hist = np.bincount(assignments)
        print(hist)
        print(bincount_pred)
        l2_error = np.mean(np.sqrt((centroids[prediction] - X) ** 2))
        l1_error = np.mean((centroids[prediction] - X))
        score = cross_val_score(clf, X, assignments, cv=5)
        print("L2 error: ", l2_error)
        print("L1 error: ", l1_error)

    # Rebase
    for i in range(bincount_pred.shape[0]):
        if bincount_pred[i] > 0:
            prediction_where = prediction == i
            select_rows = X[prediction_where]
            new_centroid = np.mean(select_rows, axis=0)
            centroids[i] = new_centroid

    if PRINT_DEBUG:
        l2_error = np.mean(np.sqrt((centroids[prediction] - X) ** 2))
        l1_error = np.mean((centroids[prediction] - X))
        score = cross_val_score(clf, X, assignments, cv=5)
        scores_2 = clf.score(X, assignments)
        print("L2 error after: ", l2_error)
        print("L1 error after: ", l1_error)
        print("Prediction score: ", scores_2, score)

    return centroids, numpy_array


def decision_tree_per_codebook(
    c: int, pq_idxs: np.ndarray, X: np.ndarray, K: int, depth: int, C: int, D: int
) -> tuple[np.ndarray, np.ndarray]:
    start_idx, end_idx = pq_idxs[c]
    idxs = np.arange(start_idx, end_idx)
    X_cut = X[:, idxs]
    centroids, tree = learn_decision_tree(X_cut, K=K, depth=depth, iterations=5)
    for i in range(K):
        tree[i] = idxs[int(tree[i])]

    centroids_extended = np.zeros((K, D), np.float32)
    centroids_extended[:, idxs] = centroids

    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"Learning progress {X.shape}-{C}-{K}: {c + 1}/{C} "
        f"({(ram_usage / (1024 * 1024)):.3f} GB)"
    )

    return tree, centroids_extended


def init_and_learn_hash_function_decision_tree(
    X: np.ndarray,
    C: int,
    pq_perm_algo: Literal["start", "end"] = "start",
    K: int = 16,
    depth: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    D = X.shape[1]

    depth = int(np.ceil(np.log2(K)))
    B = 2**depth

    X = X.astype(np.float32)
    all_prototypes = np.zeros((C, K, D), dtype=np.float32)
    pq_idxs = create_codebook_start_end_idxs(X.shape[1], C, algo=pq_perm_algo)

    decision_trees = np.zeros((C, B * 3), dtype=np.float32)

    num_cores = np.min((4, multiprocessing.cpu_count()))
    results = Parallel(n_jobs=num_cores, max_nbytes=None)(
        delayed(decision_tree_per_codebook)(i, pq_idxs, X, K, depth, C, D)
        for i in range(C)
    )
    # results = []
    # for i in range(C):
    #     results.append(decision_tree_per_codebook(i, pq_idxs, X, K, depth, C, D))
    for c in range(C):
        decision_trees[c] = results[c][0]
        all_prototypes[c] = results[c][1]
    return decision_trees, all_prototypes


def learn_proto_and_hash_function_decision_tree(
    X: np.ndarray,
    C: int,
    K: int = 16,
    # pylint: disable=unused-argument
    lut_work_const: int = -1,  # same interface as other learning functions
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = X.shape[1]
    used_perm_algo: Literal["start", "end"] = "start"  # or end
    X_orig = X.astype(np.float32)
    X = X.astype(np.float32)

    # X_error = X_orig - centroid shape: [N, D]
    decision_trees, all_prototypes = init_and_learn_hash_function_decision_tree(
        X, C, K=K, pq_perm_algo=used_perm_algo
    )
    A_enc = halut_encode_decision_tree(X, decision_trees)
    offsets = np.arange(C, dtype=np.int32) * K
    prototypes_reshape = all_prototypes.reshape((-1, all_prototypes.shape[2]))
    A_enc_offset = A_enc + offsets

    offset = prototypes_reshape[A_enc_offset]
    offset = np.sum(offset, axis=1)
    X_error = X_orig - offset

    msv_orig = (X_orig * X_orig).mean()
    mse_error = (X_error * X_error).mean()
    # mse_error_pq = (X_error_pq * X_error_pq).mean()
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

    # optimizing prototypes
    W = encoded_lstsq(A_enc=A_enc, Y=X_error, K=K)
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
    return decision_trees, all_prototypes, report_array


def centroids_per_codebook(
    c: int, pq_idxs: np.ndarray, X: np.ndarray, K: int, C: int, D: int
) -> np.ndarray:
    start_idx, end_idx = pq_idxs[c]
    idxs = np.arange(start_idx, end_idx)
    X_cut = X[:, idxs]
    centroids, _ = kmeans2(X_cut, K, minit="points", iter=25)

    centroids_extended = np.zeros((K, D), np.float32)
    centroids_extended[:, idxs] = centroids

    ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"Learning progress {X.shape}-{C}-{K}: {c + 1}/{C} "
        f"({(ram_usage / (1024 * 1024)):.3f} GB)"
    )

    return centroids_extended


def init_and_learn_hash_function_full_pq(
    X: np.ndarray,
    C: int,
    pq_perm_algo: Literal["start", "end"] = "start",
    K: int = 16,
) -> np.ndarray:
    D = X.shape[1]

    X = X.astype(np.float32)
    all_prototypes = np.zeros((C, K, D), dtype=np.float32)
    pq_idxs = create_codebook_start_end_idxs(X.shape[1], C, algo=pq_perm_algo)

    num_cores = np.min((2, multiprocessing.cpu_count()))
    print("NUM cores", num_cores)
    results = Parallel(n_jobs=num_cores, max_nbytes=None)(
        delayed(centroids_per_codebook)(i, pq_idxs, X, K, C, D) for i in range(C)
    )
    # results = []
    # for i in range(C):
    #     results.append(centroids_per_codebook(i, pq_idxs, X, K, C, D))
    for c in range(C):
        all_prototypes[c] = results[c]
    return all_prototypes


def learn_proto_and_hash_function_full_pq(
    X: np.ndarray,
    C: int,
    K: int = 16,
    # pylint: disable=unused-argument
    lut_work_const: int = -1,  # same interface as other learning functions
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    D = X.shape[1]
    used_perm_algo: Literal["start", "end"] = "start"  # or end
    X_orig = X.astype(np.float32)
    X = X.astype(np.float32)

    all_prototypes = init_and_learn_hash_function_full_pq(
        X, C, K=K, pq_perm_algo=used_perm_algo
    )

    prototypes_reshape = all_prototypes.reshape((-1, all_prototypes.shape[2]))

    A_enc = halut_encode_pq(X, all_prototypes)
    offsets = np.arange(C, dtype=np.int32) * K
    A_enc_offset_pq = A_enc + offsets
    offset = prototypes_reshape[A_enc_offset_pq]
    offset = np.sum(offset, axis=1)
    X_error = X_orig - offset

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

    # optimizing prototypes
    W = encoded_lstsq(A_enc=A_enc, Y=X_error, K=K)
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
    return np.ndarray([]), all_prototypes, report_array
