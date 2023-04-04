import argparse
import os
from pathlib import Path
from copy import deepcopy

import torch

from utils.analysis_helper import get_layers
from retraining import load_model
from models.helper import write_module_back
from models.resnet import END_STORE_A, END_STORE_B
from halutmatmul.model import HalutHelper, get_module_by_name
from halutmatmul.modules import HalutConv2d, HalutLinear

BATCH_SIZE = 256


def calculate_c(model: torch.nn.Module, layer: str) -> int:
    module_ref = get_module_by_name(model, layer)
    if isinstance(module_ref, HalutConv2d):
        inner_dim_im2col = (
            module_ref.in_channels
            * module_ref.kernel_size[0]
            * module_ref.kernel_size[1]
        )
        inner_dim_kn2col = module_ref.in_channels
        loop_order = module_ref.loop_order
        if loop_order == "im2col":
            c_ = inner_dim_im2col // 9  # 9 = 3x3
        else:
            c_ = inner_dim_kn2col // 8  # little lower than 9 but safer to work no
        if "downsample" in layer or "shortcut" in layer:
            loop_order = "im2col"
            c_ = inner_dim_im2col // 4
    if isinstance(module_ref, HalutLinear):
        c_ = module_ref.in_features // 4
    return c_


def train_initialization(
    args: argparse.Namespace,
):
    (
        model_name,
        model,
        state_dict_base,
        data_loader_train,
        _,
        _,
        _,
        _,
    ) = load_model(args.checkpoint, distributed=False, batch_size=BATCH_SIZE)
    learned_path = args.learned
    Path(learned_path).mkdir(parents=True, exist_ok=True)
    halut_data_path = args.halutdata
    Path(halut_data_path).mkdir(parents=True, exist_ok=True)

    model_copy = deepcopy(model)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    # device = torch.device("cpu")
    layers = get_layers(model_name)  # type: ignore

    K = 16
    layer_results = {}
    layers = layers[:-1]
    for idx, layer in enumerate(layers):
        model_base = deepcopy(model_copy)
        model_base.to(device)
        prev_max = 0.0
        resampling = 1
        reseeding = 1
        best_model = None
        codebooks = 0
        for resampled in range(resampling):
            if resampled > 0:
                # delete learned files
                learned_file_path = os.path.join(learned_path, layer + END_STORE_A)
                if os.path.exists(learned_file_path):
                    os.remove(learned_file_path)
                learned_weights_file_path = os.path.join(
                    learned_path, layer + END_STORE_B
                )
                if os.path.exists(learned_weights_file_path):
                    os.remove(learned_weights_file_path)

            niter_to_check = 0
            if model_name == "resnet20":
                if "layer1.1" in layer:
                    niter_to_check = 5
                if "layer1.2" in layer:
                    niter_to_check = 25
                if "layer2" in layer:
                    niter_to_check = 25
                if "layer3" in layer:
                    niter_to_check = 25
                if "linear" in layer:
                    niter_to_check = 25
            else:
                raise NotImplementedError
            for _ in range(reseeding):
                nredo = 1
                min_points_per_centroid = 1
                max_points_per_centroid = 10000
                # for max_points_per_centroid in [20000]:

                c_ = calculate_c(model_base, layer)
                kmeans_options = {
                    "niter": niter_to_check,
                    "nredo": nredo,
                    "min_points_per_centroid": min_points_per_centroid,
                    "max_points_per_centroid": max_points_per_centroid,
                }
                save_path = learned_path + f"/{layer}_{c_}_{K}.npy"
                if os.path.exists(save_path):
                    os.remove(save_path)
                halut_model = HalutHelper(
                    model_base,
                    state_dict_base,  # type: ignore
                    data_loader_train,
                    data_path=halut_data_path,
                    learned_path=learned_path,
                    kmeans_options=kmeans_options,
                    device=device,
                )
                # add previous layers
                for i in range(idx):
                    c_ = calculate_c(model_base, layers[i])
                    halut_model.activate_halut_module(
                        layers[i], c_, use_prototypes=True
                    )
                codebooks = c_
                accuracy = halut_model.run_inference(prev_max=prev_max)

                halut_model.activate_halut_module(layer, c_, use_prototypes=True)
                accuracy = halut_model.run_inference(prev_max=prev_max)
                if accuracy > prev_max:
                    prev_max = accuracy
                    print("NEW MAX", prev_max)
                    best_model = deepcopy(halut_model.model)

        state_dict_new = best_model.state_dict()  # type: ignore
        if prev_max < -2.0:
            # iterative improvement
            for c in range(codebooks):
                for _ in range(5):
                    halut_model = HalutHelper(
                        model_base,
                        state_dict_new,  # type: ignore
                        data_loader_train,
                        data_path=halut_data_path,
                        learned_path=learned_path,
                        kmeans_options=kmeans_options,
                        device=device,
                    )
                    halut_model.activate_halut_module(
                        layer, codebooks, use_prototypes=True
                    )
                    accuracy = halut_model.run_inference(prev_max=prev_max, codebook=c)
                    if accuracy > prev_max:
                        prev_max = accuracy
                        print("NEW MAX", prev_max)
                        best_model = deepcopy(halut_model.model)
                        state_dict_new = best_model.state_dict()

                    write_module_back(best_model, learned_path)  # type: ignore
        write_module_back(best_model, learned_path)  # type: ignore
        print("FINAL MAX", layer, prev_max)
        state_dict_base = best_model.state_dict()  # type: ignore
        layer_results[layer] = prev_max

    halut_model = HalutHelper(
        model_base,
        state_dict_base,  # type: ignore
        data_loader_train,
        data_path=halut_data_path,
        learned_path=learned_path,
        kmeans_options=kmeans_options,
        device=device,
    )
    for layer in layers:
        c_ = calculate_c(model_base, layer)
        halut_model.activate_halut_module(layer, c_, use_prototypes=True)
    accuracy_all = halut_model.run_inference()
    print("FINAL ACCURACY", accuracy_all)
    print("FINISHED INIALIZATION")
    print(layer_results)


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
    train_initialization(args)
