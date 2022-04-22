# pylint: disable=C0209
import glob
import os, sys
import re
import argparse
import json
from subprocess import call
from typing import Any
import torchvision
import torch
from torchvision import transforms as T

from ResNet.resnet import ResNet50_Weights, resnet50

from halutmatmul.model import HalutHelper, check_file_exists_and_return_path


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


def halut_analysis_helper(
    cuda_id: int,
    batch_size_store: int,
    halut_modules: dict[str, list[int]],
    halut_data_path: str,
    dataset_path: str,
    learned_path: str,
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
        learned_path=learned_path,
        report_error=True,
    )
    halut_model.print_available_module()
    for k, v in halut_modules.items():
        print("activate", k, v)
        halut_model.activate_halut_module(k, v[0], v[1])
    halut_model.run_inference()
    print(halut_model.get_stats())
    return halut_model.get_stats()


def run_test(
    cuda_id: int, halut_data_path: str, dataset_path: str, learned_path: str, C: int
) -> None:
    # pylint: disable=unused-variable
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

    tests = [
        "layer4.0.conv1",
        "layer4.0.conv3",
        "layer4.1.conv1",
        "layer3.0.conv3",
        "layer3.1.conv1",
        "layer3.1.conv3",
        "layer2.0.conv1",
        "layer3.0.conv1",
    ]

    result_base_path = "./results/data/accuracy/single_layer/training_data/"
    # C_all = [16, 32, 64]
    rows = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        # 2048,
        # 4096,
        # 8192,
        # 40 * 256,
    ]

    downsampled = [
        "layer1.0.downsample.0",
        "layer2.0.downsample.0",
        "layer3.0.downsample.0",
        "layer4.0.downsample.0",
    ]

    conv3x3 = [
        "layer1.0.conv2",
        "layer1.1.conv2",
        "layer1.2.conv2",
        "layer2.0.conv2",
        "layer2.1.conv2",
        "layer2.2.conv2",
        "layer2.3.conv2",
        "layer3.0.conv2",
        "layer3.1.conv2",
        "layer3.2.conv2",
        "layer3.3.conv2",
        "layer3.4.conv2",
        "layer3.5.conv2",
        "layer4.0.conv2",
        "layer4.1.conv2",
        "layer4.2.conv2",
    ]

    class ContinueI(Exception):
        pass

    continue_i = ContinueI()

    rows.reverse()
    tests_to_skip = {"layer3.1.conv3": [[128, 1]]}
    # pylint: disable=consider-iterating-dictionary, too-many-nested-blocks
    for k in conv3x3:
        for r in rows:
            try:
                if k in tests_to_skip.keys():
                    to_skip = tests_to_skip[k]
                    for skipper in to_skip:
                        print("skipper", skipper)
                        if C == skipper[0] and r == skipper[1]:
                            print("skipped test")
                            raise continue_i
            except ContinueI:
                continue
            files = glob.glob(result_base_path + "/*.json")
            files_res = []
            regex = rf"{k}_{C}_{r}\.json"
            pattern = re.compile(regex)
            files_res = [x for x in files if pattern.search(x)]
            if len(files_res) == 1:
                print("alread done")
                continue
            # learned_files = check_file_exists_and_return_path(
            #     learned_path, k, "learned", C, r
            # )
            # if len(learned_files) == 0:
            #     print(f"not learned {k} C: {C}, r: {r}")
            #     continue
            res = halut_analysis_helper(
                cuda_id,
                batch_size_store=256,
                halut_modules=dict({k: [C, r]}),
                halut_data_path=halut_data_path,
                dataset_path=dataset_path,
                learned_path=learned_path,
            )
            with open(
                result_base_path + k + "_" + str(C) + "_" + str(r) + ".json", "w"
            ) as fp:
                json.dump(res, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start analysis")
    parser.add_argument("cuda_id", metavar="N", type=int, help="id of cuda_card")
    parser.add_argument("-dataset", type=str, help="dataset path")
    parser.add_argument("-halutdata", type=str, help="halut data path")
    parser.add_argument("-learned", type=str, help="halut learned path")
    parser.add_argument("-C", type=int, help="C")
    args = parser.parse_args()
    run_test(args.cuda_id, args.halutdata, args.dataset, args.learned, args.C)

    # run_test(
    #     1,
    #     "/scratch2/janniss/resnet_input_data",
    #     "/scratch2/janniss/imagenet",
    #     "/scratch2/janniss/learned",
    #     64,
    # )
