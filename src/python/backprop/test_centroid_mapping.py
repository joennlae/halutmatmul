import argparse
import os
import csv
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
from sklearn import tree

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


def train_decision_tree(
    args: argparse.Namespace,
):
    BATCH_SIZE = 64
    (
        model_name,
        model,
        state_dict_base,
        data_loader_train,
        _,
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
    layers_now = layers[:-1]  # layers[:2]
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
    max_depth = 8
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
        print("splitter", splitter, num_splits, input_layer.shape[0])
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
            # mse = np.sum(
            #     np.square(np.expand_dims(input_layer, 2) - centroids_layer), axis=3
            # )
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
            # training_input = centroids_layer[c, :, :]
            # training_index = np.arange(centroids_layer.shape[1])
            # training_index = np.repeat(training_index, 5000, axis=0)
            # training_input = np.repeat(training_input, 5000, axis=0)
            # assert training_input.shape[0] == training_index.shape[0]
            # idxs = np.arange(training_input.shape[0])
            # np.random.shuffle(idxs)
            # training_input = training_input[idxs]
            # training_index = training_index[idxs]
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
            # decision_tree.fit(training_input, training_index)
            print(decision_tree.score(input_layer[:, c], mapping[:, c]))
            print(decision_tree.get_depth())
            print(decision_tree.get_n_leaves())
            # print(decision_tree.get_params())
            trees.append(decision_tree)
            # print(tree.export_text(decision_tree=decision_tree))
        total_prediction = np.zeros(mapping.shape, dtype=np.int64)
        for c in range(C):
            predict = trees[c].predict(input_layer[:, c])
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
    print(rows)

    with open(f"result_{max_depth}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    # halut_model.run_inference()


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
