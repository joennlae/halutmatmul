import argparse
import os
import csv
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
from sklearn import tree
from sklearn.tree import _tree
import numba

from models.resnet20 import resnet20
from models.helper import get_and_print_layers_to_use_halut, RUN_ALL_SUBSAMPLING
from retraining import load_model
from halutmatmul.model import (
    HalutHelper,
    get_module_by_name,
    check_file_exists_and_return_path,
)
from halutmatmul.modules import HalutConv2d, HalutLinear
from halutmatmul.halutmatmul import HalutModuleConfig


class DecisionTreeOffset:
    DIMS = 0
    THRESHOLDS = 1
    CLASSES = 2
    TOTAL = 3


DEFAULT_NEG_VALUE = -4419


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

        if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:  # type: ignore
            class_name = class_names[class_name]

        # pylint: disable=c-extension-no-member
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # type: ignore
            dim = tree_.feature[node]  # type: ignore
            threshold = tree_.threshold[node]  # type: ignore
            numpy_array[tree_id - 1] = dim
            numpy_array[tree_id - 1 + DecisionTreeOffset.THRESHOLDS * B] = threshold
            recurse_tree(tree_.children_left[node], depth + 1, 2 * tree_id)  # type: ignore
            recurse_tree(tree_.children_right[node], depth + 1, 2 * tree_id + 1)  # type: ignore
        else:
            _add_leaf(value, class_name, depth, tree_id)  # type: ignore[arg-type]

    recurse_tree(0, 1, 1)

    for i in range(B):
        assert numpy_array[DecisionTreeOffset.CLASSES * B + i] != DEFAULT_NEG_VALUE
        if numpy_array[i] == DEFAULT_NEG_VALUE:
            numpy_array[i] = 0  # adding default dimension TODO: optimize
    return numpy_array


@numba.jit(nopython=True, parallel=False)
def apply_decision_tree(X: np.ndarray, decision_tree: np.ndarray) -> np.ndarray:
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


def apply_decision_tree_torch(
    X: torch.Tensor, decision_tree: torch.Tensor
) -> torch.Tensor:
    N = X.shape[0]
    mapping = torch.zeros(
        N, dtype=torch.int64
    )  # needs to be int64 because of index :-)
    B = decision_tree.shape[0] // 3
    n_decisions = int(np.log2(B))
    for depth in range(n_decisions):
        index_offet = 2**depth - 1
        split_thresholds = decision_tree[mapping + B + index_offet]
        dims = decision_tree[mapping + index_offet].long()
        x = X[torch.arange(N), dims]
        indicators = x > split_thresholds
        mapping = (mapping * 2) + indicators
    mapping = decision_tree[mapping + 2 * B].long()
    return mapping


def train_decision_tree(
    args: argparse.Namespace,
):
    BATCH_SIZE = 64
    (
        model_name,
        model,
        state_dict_base,
        data_loader_train,
        data_loader_val,
        _,
        halut_modules,
        _,
    ) = load_model(args.checkpoint, distributed=False, batch_size=BATCH_SIZE)

    learned_path = args.learned
    if model_name not in learned_path.lower():
        learned_path += "/" + model_name
    Path(learned_path).mkdir(parents=True, exist_ok=True)

    halut_data_path = args.halutdata
    if model_name not in halut_data_path.lower():
        halut_data_path += "/" + model_name
    Path(halut_data_path).mkdir(parents=True, exist_ok=True)

    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
        )

    model_base = deepcopy(model)
    halut_model = HalutHelper(
        model_base,
        state_dict_base,  # type: ignore
        data_loader_train,
        data_path=halut_data_path,
        learned_path=learned_path,
        device=device,
    )

    state_dict_to_add = halut_model.model.state_dict()

    modules = halut_modules
    for k, v in modules.items():  # type: ignore
        if len(v) > 3:
            halut_model.activate_halut_module(
                k,
                C=v[HalutModuleConfig.C],
                K=v[HalutModuleConfig.K],
                loop_order=v[HalutModuleConfig.LOOP_ORDER],
                use_prototypes=v[HalutModuleConfig.USE_PROTOTYPES],
            )
        else:
            halut_model.activate_halut_module(
                k,
                C=v[HalutModuleConfig.C],
                K=v[HalutModuleConfig.K],
            )

    layers = get_and_print_layers_to_use_halut(halut_model.model)
    layers_now = layers[:-1]  # layers[:-1]  # layers[:2]
    dict_to_store = {}
    for l in layers_now:
        dict_to_store[l] = RUN_ALL_SUBSAMPLING

    # halut_model.store_inputs(dict_to_store)

    # load data
    centroids = []
    for l in layers_now:
        module = get_module_by_name(halut_model.model, l)
        if isinstance(module, (HalutConv2d, HalutLinear)):
            centroids.append(module.P.detach().cpu().numpy())
    print(centroids, len(centroids), centroids[0].shape)

    rows = []
    max_depth = 4
    for idx, l in enumerate(layers_now):
        stored_paths = check_file_exists_and_return_path(halut_data_path, l, "input")
        print("paths for layer", l, stored_paths)
        if len(stored_paths) != 2:
            raise Exception("not stored")
        layer_input = np.load(stored_paths[0])
        input_layer = layer_input
        centroids_layer = centroids[idx]
        input_layer = input_layer.reshape(
            (-1, centroids_layer.shape[0], centroids_layer.shape[2])
        )
        input_layer = input_layer[: 1024 * 1024]
        mse = np.zeros(
            (input_layer.shape[0], input_layer.shape[1], centroids_layer.shape[1])
        )
        # fixes out of memory problems
        splitter = 32
        num_splits = input_layer.shape[0] // splitter
        assert input_layer.shape[0] % splitter == 0
        for i in range(splitter):
            mse[i * num_splits : (i + 1) * num_splits, :, :] = np.sum(
                np.square(
                    np.expand_dims(
                        input_layer[i * num_splits : (i + 1) * num_splits], 2
                    )
                    - centroids_layer
                ),
                axis=3,
            )
        mapping = np.argmin(mse, axis=2)
        print(l, mapping, mapping.shape)

        print(centroids_layer.shape, input_layer.shape, mapping.shape)
        selected_centroids = np.zeros(input_layer.shape)
        C = input_layer.shape[1]
        for c in range(mapping.shape[1]):
            selected_centroids[:, c, :] = centroids_layer[c, mapping[:, c], :]
        mse = selected_centroids - input_layer
        print(np.sum(np.abs(mse)))

        C = mapping.shape[1]
        trees = []
        for c in range(C):
            counted = np.bincount(mapping[:, c])
            print(counted)
            decision_tree = tree.DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features=None,
                criterion="gini",
            )
            decision_tree.fit(
                input_layer[:, c],
                mapping[:, c],
            )
            print(decision_tree.score(input_layer[:, c], mapping[:, c]))
            print(decision_tree.get_depth())
            print(decision_tree.get_n_leaves())
            numpy_tree = tree_to_numpy(decision_tree, depth=max_depth)
            print(numpy_tree)
            trees.append(torch.from_numpy(numpy_tree))
            # print(tree.export_text(decision_tree=decision_tree))
        total_prediction = np.zeros(mapping.shape, dtype=np.int64)
        trees = torch.vstack(trees)
        print("shape of stacked trees", trees.shape)
        for c in range(C):
            predict = apply_decision_tree_torch(
                torch.from_numpy(input_layer[:, c]), trees[c]
            )
            total_prediction[:, c] = predict
        print(total_prediction, total_prediction.shape)
        print(mapping, mapping.shape)
        selected_centroids2 = np.zeros(input_layer.shape)
        for c in range(total_prediction.shape[1]):
            selected_centroids2[:, c, :] = centroids_layer[c, total_prediction[:, c], :]
        mse2 = selected_centroids2 - input_layer
        print(
            np.sum(np.abs(mse2)),
            np.sum(np.abs(mse)),
            np.sum(np.abs(mse2)) / np.sum(np.abs(mse)),
        )
        print(np.sum(total_prediction == mapping) / (total_prediction.shape[0] * C))
        row = [
            idx,
            l,
            max_depth,
            np.sum(total_prediction == mapping) / (total_prediction.shape[0] * C),
            np.sum(np.abs(mse2)) / np.sum(np.abs(mse)),
            np.sum(np.abs(mse2)),
            np.sum(np.abs(mse)),
        ]
        print("Row", row)
        rows.append(row)
        state_dict_to_add[l + ".DT"] = trees
    print(rows)

    with open(f"result_{max_depth}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    model_base = deepcopy(model)
    halut_model = HalutHelper(
        model_base,
        state_dict_to_add,  # type: ignore
        data_loader_val,
        data_path=halut_data_path,
        learned_path=learned_path,
        device=device,
    )
    halut_model.run_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace layer with halut")
    parser.add_argument("gpu", metavar="N", type=int, help="id of cuda_card", default=0)
    parser.add_argument(
        "-halutdata",
        type=str,
        help="halut data path",
    )
    parser.add_argument(
        "-learned",
        type=str,
        help="halut learned path",
    )
    parser.add_argument("-modelname", type=str, help="model name", default="resnet18")
    parser.add_argument(
        "-resultpath",
        type=str,
        help="result_path",
    )
    parser.add_argument(
        "-checkpoint",
        type=str,
        help="check_point_path",
    )
    args = parser.parse_args()
    train_decision_tree(args)
