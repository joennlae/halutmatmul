# pylint: disable=C0209
import os, sys
import argparse
import json
from subprocess import call
from typing import Any
import torchvision
import torch
from torchvision import transforms as T

from ResNet.resnet import ResNet50_Weights, resnet50

from halutmatmul.model import HalutHelper


def sys_info() -> None:
    print("__Python VERSION:", sys.version)
    print("__pyTorch VERSION:", torch.__version__)
    print(
        "__CUDA VERSION",
    )

    # ! nvcc --version
    print("__CUDNN VERSION:", torch.backends.cudnn.version())
    print("__Number CUDA Devices:", torch.cuda.device_count())
    print("__Devices")
    call(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        ]
    )
    print("Active CUDA Device: GPU", torch.cuda.current_device())
    print("Available devices ", torch.cuda.device_count())
    print("Current cuda device ", torch.cuda.current_device())


def halut_helper(
    cuda_id: int,
    batch_size_store: int,
    halut_modules: dict[str, int],
    halut_data_path: str,
    dataset_path: str,
) -> dict[str, Any]:
    torch.cuda.set_device(cuda_id)
    sys_info()
    device = torch.device(
        "cuda:" + str(cuda_id) if torch.cuda.is_available() else "cpu"
    )
    state_dict = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
    imagenet_val = torchvision.datasets.ImageNet(
        root=dataset_path,  # "/scratch/janniss/imagenet/",
        split="val",
        transform=ResNet50_Weights.IMAGENET1K_V2.transforms(),
    )
    model = resnet50(weights=state_dict, progress=True)
    model.cuda()
    model.to(device)

    halut_model = HalutHelper(
        model,
        state_dict,
        imagenet_val,
        batch_size_inference=112,
        batch_size_store=batch_size_store,
        data_path=halut_data_path,
        device=device,
    )
    halut_model.print_available_module()
    for k, v in halut_modules.items():
        print("activate", k, v)
        halut_model.activate_halut_module(k, v)
    halut_model.run_inference()
    print(halut_model.get_stats())
    return halut_model.get_stats()


def run_test(
    C: int = 16,
    cuda_id: int = 1,
    halut_data_path: str = "/scratch/janniss/data",
    dataset_path: str = "/scratch/janniss/imagenet",
) -> None:
    tests = [
        "layer4.2.conv3",
        "layer1.0.conv1",
        "layer1.0.conv3",
        "layer1.1.conv1",
        "layer1.1.conv3",
        "layer1.2.conv1",
        "layer1.2.conv3",
        "layer2.0.conv1",
        "layer2.0.conv3",
        "layer2.1.conv1",
        "layer2.1.conv3",
        "layer2.2.conv1",
        "layer2.2.conv3",
        "layer2.3.conv1",
        "layer2.3.conv3",
        "layer3.0.conv1",
        "layer3.0.conv3",
        "layer3.1.conv1",
        "layer3.1.conv3",
        "layer3.2.conv1",
        "layer3.2.conv3",
        "layer3.3.conv1",
        "layer3.3.conv3",
        "layer3.4.conv1",
        "layer3.4.conv3",
        "layer3.5.conv1",
        "layer3.5.conv3",
        "layer4.0.conv1",
        "layer4.0.conv3",
        "layer4.1.conv1",
        "layer4.1.conv3",
        "layer4.2.conv1",
    ]

    for k in tests:
        res = halut_helper(
            cuda_id,
            6 * 128,
            dict({k: C}),
            halut_data_path=halut_data_path,
            dataset_path=dataset_path,
        )
        with open("./results/" + k + "_" + str(C) + ".json", "w") as fp:
            json.dump(res, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start analysis")
    parser.add_argument("cuda_id", metavar="N", type=int, help="id of cuda_card")
    parser.add_argument("-C", type=int, help="amount of codebooks for halut")
    parser.add_argument("-dataset", type=str, help="dataset path")
    parser.add_argument("-halutdata", type=str, help="halut data path")
    args = parser.parse_args()
    run_test(args.C, args.cuda_id, args.halutdata, args.dataset)
