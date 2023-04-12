# pylint: disable=no-member
import glob
import json
import re
from subprocess import call
import sys
from typing import Any, Dict, Literal, OrderedDict
import pandas as pd
import torch

available_models = Literal["resnet-50", "levit", "ds-cnn", "resnet18", "resnet20"]


def sys_info() -> None:
    print("__Python VERSION:", sys.version)
    print("__pyTorch VERSION:", torch.__version__)
    print(
        "__CUDA VERSION",
    )

    # ! nvcc --version
    print("__CUDNN VERSION:", torch.backends.cudnn.version())  # type: ignore
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


### ResNet-50 Layers
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

tests_bad = [
    "layer4.0.conv1",
    "layer4.0.conv3",
    "layer4.1.conv1",
    "layer3.0.conv3",
    "layer3.1.conv1",
    "layer3.1.conv3",
    "layer2.0.conv1",
    "layer3.0.conv1",
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

layers_interesting = [
    "layer1.0.conv2",
    "layer1.1.conv3",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.0.downsample.0",
    "layer2.3.conv3",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.conv3",
    "layer3.3.conv3",
    "layer3.4.conv2",
    "layer3.5.conv1",
    "layer3.5.conv2",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.conv3",
    "layer4.1.conv1",
    "layer4.2.conv1",
    "layer4.2.conv3",
]


all_layers = [
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.0.conv3",
    "layer1.0.downsample.0",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer1.1.conv3",
    "layer1.2.conv1",
    "layer1.2.conv2",
    "layer1.2.conv3",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.0.conv3",
    "layer2.0.downsample.0",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer2.1.conv3",
    "layer2.2.conv1",
    "layer2.2.conv2",
    "layer2.2.conv3",
    "layer2.3.conv1",
    "layer2.3.conv2",
    "layer2.3.conv3",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.conv3",
    "layer3.0.downsample.0",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer3.1.conv3",
    "layer3.2.conv1",
    "layer3.2.conv2",
    "layer3.2.conv3",
    "layer3.3.conv1",
    "layer3.3.conv2",
    "layer3.3.conv3",
    "layer3.4.conv1",
    "layer3.4.conv2",
    "layer3.4.conv3",
    "layer3.5.conv1",
    "layer3.5.conv2",
    "layer3.5.conv3",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.conv3",
    "layer4.0.downsample.0",
    "layer4.1.conv1",
    "layer4.1.conv2",
    "layer4.1.conv3",
    "layer4.2.conv1",
    "layer4.2.conv2",
    "layer4.2.conv3",
]


def json_to_dataframe(
    path: str, layer_name: str, max_C: int = 128, prefix: str = ""
) -> pd.DataFrame:
    files = glob.glob(path + "/*.json")
    regex = rf"{layer_name}_.+\.json"
    pattern = re.compile(regex)
    files_res = [x for x in files if pattern.search(x)]

    dfs = []  # an empty list to store the data frames
    for file in files_res:
        data = pd.read_json(file)  # read data frame from json file
        if layer_name + ".learned_n" not in data.columns:
            data[layer_name + ".learned_n"] = data.iloc[0][
                layer_name + ".learned_a_shape"
            ]
            data[layer_name + ".learned_d"] = data.iloc[1][
                layer_name + ".learned_a_shape"
            ]
            K = data.iloc[0][layer_name + ".K"]
            C = data.iloc[0][layer_name + ".C"]
            data[layer_name + ".learned_m"] = int(
                data.iloc[0][layer_name + ".L_size"] / (4 * K * C)
            )
        C = data.iloc[0][layer_name + ".C"]
        if C > max_C:
            continue
        if layer_name + ".learned_a_shape" in data.columns:
            data = data.drop([1])
            data = data.drop(
                columns=[
                    layer_name + ".learned_a_shape",
                    layer_name + ".learned_b_shape",
                ]
            )

        data["hue_string"] = prefix + str(C)

        data["test_name"] = layer_name + "-" + str(data.iloc[0][layer_name + ".C"])
        data["layer_name"] = layer_name + (
            " (3x3)" if "conv2" in layer_name else " (1x1)"
        )
        data["row_name"] = layer_name.split(".")[0]
        data["col_name"] = layer_name[len(layer_name.split(".")[0]) + 1 :]
        dfs.append(data)  # append the data frame to the list

    df = pd.concat(
        dfs, ignore_index=True
    )  # concatenate all the data frames in the list.

    df = df.drop(columns="halut_layers")
    df["top_1_accuracy_100"] = df["top_1_accuracy"] * 100
    final_dfs = []
    for C in [16, 32, 64]:
        df_C = df[df[layer_name + ".C"] == C]
        df_C.sort_values(
            by=["top_1_accuracy"], inplace=True, ignore_index=True, ascending=False
        )
        final_dfs.append(df_C.iloc[[0]])
    df = pd.concat(final_dfs, ignore_index=True)
    df.columns = df.columns.str.replace(layer_name + ".", "")
    return df


ds_cnn_layers = [
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "conv5",
    "conv6",
    "conv7",
    "conv8",
    "conv9",
]

layers_levit = [
    "blocks.0.m.qkv",
    "blocks.0.m.proj.1",
    "blocks.1.m.0",
    "blocks.1.m.2",
    "blocks.2.m.qkv",
    "blocks.2.m.proj.1",
    "blocks.3.m.0",
    "blocks.3.m.2",
    "blocks.4.kv",
    "blocks.4.q.1",
    "blocks.4.proj.1",
    "blocks.5.m.0",
    "blocks.5.m.2",
    "blocks.6.m.qkv",
    "blocks.6.m.proj.1",
    "blocks.7.m.0",
    "blocks.7.m.2",
    "blocks.8.m.qkv",
    "blocks.8.m.proj.1",
    "blocks.9.m.0",
    "blocks.9.m.2",
    "blocks.10.m.qkv",
    "blocks.10.m.proj.1",
    "blocks.11.m.0",
    "blocks.11.m.2",
    "blocks.12.kv",
    "blocks.12.q.1",
    "blocks.12.proj.1",
    "blocks.13.m.0",
    "blocks.13.m.2",
    "blocks.14.m.qkv",
    "blocks.14.m.proj.1",
    "blocks.15.m.0",
    "blocks.15.m.2",
    "blocks.16.m.qkv",
    "blocks.16.m.proj.1",
    "blocks.17.m.0",
    "blocks.17.m.2",
    "blocks.18.m.qkv",
    "blocks.18.m.proj.1",
    "blocks.19.m.0",
    "blocks.19.m.2",
    "blocks.20.m.qkv",
    "blocks.20.m.proj.1",
    "blocks.21.m.0",
    "blocks.21.m.2",
    "head",
    "head_dist",
]

resnet18_layers = [
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.0.downsample.0",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.downsample.0",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.downsample.0",
    "layer4.1.conv1",
    "layer4.1.conv2",
    "fc",
]

resnet20_layers = [
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer1.2.conv1",
    "layer1.2.conv2",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer2.2.conv1",
    "layer2.2.conv2",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer3.2.conv1",
    "layer3.2.conv2",
    "linear",
]
resnet20_b_layers = [
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer1.2.conv1",
    "layer1.2.conv2",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.0.shortcut.0",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer2.2.conv1",
    "layer2.2.conv2",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.shortcut.0",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer3.2.conv1",
    "layer3.2.conv2",
    "linear",
]

"""
layer1.0.conv1   torch.Size([64, 64, 3, 3])
layer1.0.conv2   torch.Size([64, 64, 3, 3])
layer1.1.conv1   torch.Size([64, 64, 3, 3])
layer1.1.conv2   torch.Size([64, 64, 3, 3])
layer2.0.conv1   torch.Size([128, 64, 3, 3])
layer2.0.conv2   torch.Size([128, 128, 3, 3])
layer2.0.downsample.0   torch.Size([128, 64, 1, 1])
layer2.1.conv1   torch.Size([128, 128, 3, 3])
layer2.1.conv2   torch.Size([128, 128, 3, 3])
layer3.0.conv1   torch.Size([256, 128, 3, 3])
layer3.0.conv2   torch.Size([256, 256, 3, 3])
layer3.0.downsample.0   torch.Size([256, 128, 1, 1])
layer3.1.conv1   torch.Size([256, 256, 3, 3])
layer3.1.conv2   torch.Size([256, 256, 3, 3])
layer4.0.conv1   torch.Size([512, 256, 3, 3])
layer4.0.conv2   torch.Size([512, 512, 3, 3])
layer4.0.downsample.0   torch.Size([512, 256, 1, 1])
layer4.1.conv1   torch.Size([512, 512, 3, 3])
layer4.1.conv2   torch.Size([512, 512, 3, 3])
fc   torch.Size([1000, 512])
"""


def get_layers(name: available_models) -> list[str]:
    if name == "resnet-50":
        return layers_interesting
    elif name == "levit":
        return layers_levit
    elif name == "ds-cnn":
        return ds_cnn_layers
    elif name == "resnet18":
        return resnet18_layers
    elif name == "resnet20":
        return resnet20_layers
    else:
        return Exception("Model name not supported: ", name)  # type: ignore


def get_input_data_amount(name: available_models, l: str) -> list[int]:
    if name in ["resnet-50", "resnet18"]:
        layer_loc = l.split(".", maxsplit=1)[0]
        rows_adapted = []
        if layer_loc in ["layer1"]:
            rows_adapted = [1, 2, 4, 8]
        elif layer_loc == "layer2":
            rows_adapted = [2, 4, 8, 16]
        elif layer_loc == "layer3":
            rows_adapted = [8, 16, 32, 64]
        elif layer_loc == "layer4":
            rows_adapted = [32, 64, 128, 256]
        return rows_adapted
    elif name == "levit":
        return [1024]  # [128, 256, 512, 1024]
    elif name == "ds-cnn":
        # TODO: think about learning on training set
        return [1]  # all
    else:
        raise Exception("Model name not supported: ", name)
