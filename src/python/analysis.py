# pylint: disable=C0209
import glob
from pathlib import Path
import re
import argparse
import json
from typing import Any, Callable, Dict, Literal, OrderedDict, TypeVar
import pandas as pd
import torchvision
import torch

from utils.analysis_helper import (
    get_input_data_amount,
    get_layers,
    json_to_dataframe,
    sys_info,
    available_models,
    all_layers,
)

from models.helper import eval_halut_kws, evaluate_halut_imagenet
from models.dscnn.main import setup_ds_cnn_eval
from models.resnet import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from models.levit.main import run_levit_analysis  # type: ignore[attr-defined]

from halutmatmul.model import HalutHelper, eval_func_type
from halutmatmul.halutmatmul import EncodingAlgorithm, HalutModuleConfig


def model_loader(
    name: available_models,
    dataset_path: str,
) -> tuple[
    torch.nn.Module, Any, OrderedDict[str, torch.Tensor], eval_func_type, int, int
]:
    if name == "resnet-50":
        state_dict = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
        data = torchvision.datasets.ImageNet(
            root=dataset_path,  # "/scratch/janniss/imagenet/",
            split="val",
            transform=ResNet50_Weights.IMAGENET1K_V2.transforms(),
        )
        model = resnet50(weights=state_dict, progress=True)  # type: ignore
        return model, data, state_dict, evaluate_halut_imagenet, 256, 128  # type: ignore
    elif name == "resnet18":
        state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        data = torchvision.datasets.ImageNet(
            root=dataset_path,  # "/scratch/janniss/imagenet/",
            split="val",
            transform=ResNet18_Weights.IMAGENET1K_V1.transforms(),
        )
        checkpoint = torch.load(
            "/usr/scratch2/vilan2/janniss/model_checkpoints/checkpoint_100.pth",
            map_location="cpu",
        )
        model = resnet18(weights=state_dict, progress=True)  # type: ignore
        model.load_state_dict(checkpoint["model"])
        return model, data, state_dict, evaluate_halut_imagenet, 256, 128  # type: ignore
    elif name == "levit":
        model, data_loader, state_dict = run_levit_analysis(  # type: ignore
            [
                "--analysis",
                "True",
                "--eval",
                "--model",
                "LeViT_128S",
                "--data-path",
                "/scratch/ml_datasets/ILSVRC2012/",
            ]
        )
        return model, data_loader, state_dict, evaluate_halut_imagenet, 256, 128  # type: ignore
    elif name == "ds-cnn":
        model, data, state_dict = setup_ds_cnn_eval()  # type: ignore[assignment]
        print("data", data)
        return model, data, state_dict, eval_halut_kws, 0, 0
    else:
        raise Exception("Model name not supported: ", name)


def multilayer_analysis(
    cuda_id: int, halut_data_path: str, dataset_path: str, learned_path: str, C: int
) -> None:
    data_path = "results/data/accuracy/single_layer/training_data"
    dfs = []
    i = 0
    for l in all_layers:
        i = i + 1
        # if i > 6:
        #     break
        df = json_to_dataframe(data_path, l)
        dfs.append(df)

    df = pd.concat(dfs)

    C = 64
    df_64 = df[df["C"] == C]
    df_64.sort_values(
        by=["top_1_accuracy"], inplace=True, ignore_index=True, ascending=False
    )
    print(df_64)
    df_64.to_csv("test.csv")

    result_base_path = "./results/data/accuracy/multi_layer/"

    for i in range(20, 27):
        layer_dict: Dict[str, list[int]] = dict({})
        for k in range(i):
            layer_dict |= {
                df_64.iloc[k]["layer_name"].split(" ")[0]: [
                    C,
                    int(df_64.iloc[k]["rows"]),
                ]
            }
        print(layer_dict)
        res = halut_analysis_helper(
            cuda_id,
            halut_modules=layer_dict,
            halut_data_path=halut_data_path,
            dataset_path=dataset_path,
            learned_path=learned_path,
        )
        with open(
            f'{result_base_path}{i}_{str(C)}_{res["top_1_accuracy"]:2.3f}.json',
            "w",
        ) as fp:
            json.dump(res, fp, sort_keys=True, indent=4)


def halut_analysis_helper(
    cuda_id: int,
    halut_modules: dict[str, list[int]],
    halut_data_path: str,
    dataset_path: str,
    learned_path: str,
    model_name: Literal["resnet-50", "levit", "ds-cnn"] = "resnet-50",
) -> dict[str, Any]:
    torch.cuda.set_device(cuda_id)
    sys_info()
    device = torch.device(
        "cuda:" + str(cuda_id) if torch.cuda.is_available() else "cpu"
    )

    print("model name", model_name)
    model, data, state_dict, eval_func, batch_size_store, batch_size = model_loader(
        model_name, dataset_path=dataset_path
    )

    if model_name not in learned_path.lower():
        learned_path += "/" + model_name
    Path(learned_path).mkdir(parents=True, exist_ok=True)

    if model_name not in halut_data_path.lower():
        halut_data_path += "/" + model_name
    Path(halut_data_path).mkdir(parents=True, exist_ok=True)

    model.cuda()
    model.to(device)

    halut_model = HalutHelper(
        model,
        state_dict,
        data,
        batch_size_inference=batch_size,
        batch_size_store=batch_size_store,
        data_path=halut_data_path,
        device=device,
        learned_path=learned_path,
        report_error=True,
        eval_function=eval_func,
    )
    halut_model.print_available_module()
    for k, v in halut_modules.items():
        print("activate", k, v)
        halut_model.activate_halut_module(
            k,
            C=v[HalutModuleConfig.C],
            rows=v[HalutModuleConfig.ROWS],
            K=v[HalutModuleConfig.K],
            encoding_algorithm=v[HalutModuleConfig.ENCODING_ALGORITHM],
        )
    halut_model.run_inference()
    print(halut_model.get_stats())
    return halut_model.get_stats()


def run_test(
    cuda_id: int,
    halut_data_path: str,
    dataset_path: str,
    learned_path: str,
    C: int,
    layers: list[str],
    model_name: available_models,
    result_base_path: str,
    layer_start_offset: int = 0,
) -> None:

    if model_name not in result_base_path.lower():
        result_base_path += "/" + model_name + "/"

    Path(result_base_path).mkdir(parents=True, exist_ok=True)

    layers_test = layers[layer_start_offset:]
    for l in layers_test:
        input_data_amount = get_input_data_amount(model_name, l)
        for r in input_data_amount:
            # for C in [8, 16, 32, 64]:
            for e in [
                EncodingAlgorithm.FOUR_DIM_HASH,
                EncodingAlgorithm.DECISION_TREE,
                EncodingAlgorithm.FULL_PQ,
            ]:
                for K in (
                    [16]  # [8, 16, 32]
                    if e == EncodingAlgorithm.FOUR_DIM_HASH
                    else [16, 32]  # [4, 8, 12, 16, 24, 32, 64]
                ):
                    files = glob.glob(result_base_path + "/*.json")
                    files_res = []
                    regex = rf"{l}_{C}_{K}_{e}-{r}\.json"
                    pattern = re.compile(regex)
                    files_res = [x for x in files if pattern.search(x)]
                    if len(files_res) == 1:
                        print("already done")
                        continue
                    # learned_files = check_file_exists_and_return_path(
                    #     learned_path, k, "learned", C, r
                    # )
                    # if len(learned_files) == 0:
                    #     print(f"not learned {k} C: {C}, r: {r}")
                    #     continue
                    res = halut_analysis_helper(
                        cuda_id,
                        halut_modules=dict({l: [C, r, K, e]}),
                        halut_data_path=halut_data_path,
                        dataset_path=dataset_path,
                        learned_path=learned_path,
                        model_name=model_name,  # type: ignore[arg-type]
                    )
                    with open(
                        result_base_path
                        + l
                        + "_"
                        + str(C)
                        + "_"
                        + str(K)
                        + "_"
                        + str(e)
                        + "-"
                        + str(r)
                        + ".json",
                        "w",
                    ) as fp:
                        json.dump(res, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    DEFAULT_FOLDER = "/scratch2/janniss/"
    parser = argparse.ArgumentParser(description="Start analysis")
    parser.add_argument("cuda_id", metavar="N", type=int, help="id of cuda_card")
    parser.add_argument(
        "-dataset", type=str, help="dataset path", default=DEFAULT_FOLDER + "imagenet"
    )
    parser.add_argument(
        "-halutdata",
        type=str,
        help="halut data path",
        default=DEFAULT_FOLDER + "/halut",
    )
    parser.add_argument(
        "-learned",
        type=str,
        help="halut learned path",
        default=DEFAULT_FOLDER + "/halut/learned",
    )
    parser.add_argument("-C", type=int, help="C")
    parser.add_argument("-modelname", type=str, help="model name", default="resnet-50")
    parser.add_argument("-offset", type=int, help="start_layer_offset", default=0)
    parser.add_argument(
        "-resultpath",
        type=str,
        help="result_path",
        default="./results/data/accuracy/single_layer",
    )
    args = parser.parse_args()

    layers = get_layers(args.modelname)
    print(layers)
    run_test(
        args.cuda_id,
        args.halutdata,
        args.dataset,
        args.learned,
        args.C,
        layers,
        args.modelname,
        args.resultpath,
        args.offset,
    )

    # run_test(
    #     0,
    #     "/scratch2/janniss/resnet_input_data",
    #     "/scratch2/janniss/imagenet",
    #     "/scratch2/janniss/learned",
    #     C
    # )
    # multilayer_analysis(
    #     args.cuda_id, args.halutdata, args.dataset, args.learned, args.C
    # )
