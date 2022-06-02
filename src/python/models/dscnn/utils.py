# Copyright (C) 2021 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


import os
from typing import Any

from sklearn.metrics import confusion_matrix

import torch
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def npy_to_txt(layer_number: int, activations: np.ndarray) -> None:
    # Saving the input

    if layer_number == -1:
        tmp = activations.reshape(-1)
        f = open("input.txt", "a")
        f.write("# input (shape [1, 49, 10]),\\\n")
        for elem in tmp:
            if elem < 0:
                f.write(str(256 + elem) + ",\\\n")
            else:
                f.write(str(elem) + ",\\\n")
        f.close()
    # Saving layers' activations
    # pylint: disable=consider-using-with
    else:
        tmp = activations.reshape(-1)
        f = open("out_layer" + str(layer_number) + ".txt", "a")
        f.write(
            "layers.0.relu1 (shape [1, 25, 5, 64]),\\\n"
        )  # Hardcoded, should be adapted for better understanding.
        for elem in tmp:
            if elem < 0:
                f.write(str(256 + elem) + ",\\\n")
            else:
                f.write(str(elem) + ",\\\n")
        f.close()


def remove_txt() -> None:
    # Removing old activations and inputs

    directory = "."
    files_in_directory = os.listdir(directory)
    filtered_files = [
        file
        for file in files_in_directory
        if (file.startswith("out_layer") or file.startswith("input.txt"))
    ]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


# pylint: disable=unnecessary-comprehension
def conf_matrix(
    labels: torch.Tensor, predicted: torch.Tensor, training_parameters: dict["str", Any]
) -> None:
    # Plotting confusion matrix

    labels = labels.cpu()
    predicted = predicted.cpu()
    cm = confusion_matrix(labels, predicted)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        cm,
        index=[i for i in ["silence", "unknown"] + training_parameters["wanted_words"]],
        columns=[
            i for i in ["silence", "unknown"] + training_parameters["wanted_words"]
        ],
    )
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def parameter_generation() -> tuple[dict[str, Any], dict[str, int]]:
    # Data processing parameters

    data_processing_parameters = {"feature_bin_count": 10}
    time_shift_ms = 200
    sample_rate = 16000
    clip_duration_ms = 1000
    time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
    window_size_ms = 40.0
    window_stride_ms = 20.0
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    data_processing_parameters["desired_samples"] = desired_samples
    data_processing_parameters["sample_rate"] = sample_rate
    data_processing_parameters["spectrogram_length"] = spectrogram_length
    data_processing_parameters["window_stride_samples"] = window_stride_samples
    data_processing_parameters["window_size_samples"] = window_size_samples

    # Training parameters
    training_parameters = {
        "data_dir": "/scratch2/janniss/speech_commands_v0.02",
        "data_url": "https://storage.googleapis.com/download.tensorflow.org"
        "/data/speech_commands_v0.02.tar.gz",
        "epochs": 40,
        "batch_size": 128,
        "silence_percentage": 10.0,
        "unknown_percentage": 10.0,
        "validation_percentage": 10.0,
        "testing_percentage": 10.0,
        "background_frequency": 0.8,
        "background_volume": 0.2,
    }
    target_words = "yes,no,up,down,left,right,on,off,stop,go,"  # GSCv2 - 12 words
    # Selecting 35 words
    # target_words='yes,no,up,down,left,right,on,off,stop,go,backward,
    # bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,
    # nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
    wanted_words = (target_words).split(",")
    wanted_words.pop()
    training_parameters["wanted_words"] = wanted_words
    training_parameters["time_shift_samples"] = time_shift_samples

    return training_parameters, data_processing_parameters
