import glob
import json
import os
from math import ceil
import re
from typing import Any
import numpy as np

from models.resnet import END_STORE_A, END_STORE_B
import halutmatmul.halutmatmul as hm


def analyze_halut(
    l: str,
    C: int,
    r: int,
    data_path: str,
    batch_size: int,
    # pylint: disable=unused-argument
    K: int = 16,
) -> dict[str, Any]:
    print("start learning", l, C, r)
    files = glob.glob(data_path + f"/{l}_{batch_size}_{0}_*" + END_STORE_A)
    files = [x.split("/")[-1] for x in files]
    print(files)
    if len(files) > 1:
        # will take the one with more
        # ['layer1.0.conv2_256_0_10_A.npy', 'layer1.0.conv2_256_0_4_A.npy']
        files.sort()
    assert len(files) == 1
    configs_reg = re.findall(r"(?<=_)(\d+)", files[0])
    iterations = int(configs_reg[2])
    a_numpy = np.load(data_path + f"/{l}_{batch_size}_{0}_{iterations}" + END_STORE_A)
    files_to_load = ceil(r / batch_size)
    rows_per_batch = a_numpy.shape[0]
    total_rows = ceil(rows_per_batch * r / batch_size)

    for i in range(1, files_to_load):
        a_part = np.load(
            data_path + f"/{l}_{batch_size}_{i}_{iterations}" + END_STORE_A
        )
        a_numpy = np.vstack((a_numpy, a_part))
    a_numpy = a_numpy[:total_rows]
    print(
        "A input: ",
        a_numpy.shape,
        a_numpy.shape[0] * a_numpy.shape[1] * 4 / (1024 * 1024 * 1024),
        " GB",
    )
    b_numpy = np.load(data_path + f"/{l}_{batch_size}_{0}_{iterations}" + END_STORE_B)
    print(
        "B input: ",
        b_numpy.shape,
        b_numpy.shape[0] * b_numpy.shape[1] * 4 / (1024 * 1024),
        " MB",
    )
    print("range A", np.min(a_numpy), np.max(a_numpy))
    print(
        "A == 0",
        np.count_nonzero(a_numpy == 0) / (a_numpy.shape[0] * a_numpy.shape[1]) * 100,
    )
    print("range B", np.min(b_numpy), np.max(b_numpy))
    _, report_dict = hm.learn_halut_offline_report(a_numpy, b_numpy, C)

    report_dict["layer_name"] = l
    report_dict["zeros_percentage"] = (
        np.count_nonzero(a_numpy == 0) / (a_numpy.shape[0] * a_numpy.shape[1]) * 100
    )

    print(report_dict)

    return report_dict


layers = [
    "layer1.1.conv3",
    "layer2.0.conv1",
    "layer2.3.conv3",
    "layer3.0.conv1",
    "layer3.0.conv3",
    "layer3.3.conv3",
    "layer4.0.conv1",
    "layer4.0.conv3",
    "layer4.1.conv1",
    "layer4.2.conv3",
]
if __name__ == "__main__":
    report_dicts = []
    layer_name = layers[6]
    for layer_name in layers:
        report = analyze_halut(
            layer_name,
            64,
            16,
            "/scratch2/janniss/resnet_input_data",
            256,
        )
        report_dicts.append(report)
    print(report_dicts)
    with open(
        "halut_learning_64.json",
        "w",
    ) as fp:
        json.dump(report_dicts, fp, sort_keys=True, indent=4)
