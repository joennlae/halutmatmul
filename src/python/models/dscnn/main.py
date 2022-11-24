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


from copy import deepcopy

import time
from typing import Any, OrderedDict
import torch
from torchinfo import summary

from models.dscnn.dataset import MODEL_PATH, AudioGenerator, AudioProcessor
from models.dscnn.model import DSCNN
from models.dscnn.utils import remove_txt, parameter_generation

# from pthflops import count_ops
from models.dscnn.train import Train


def run_kws_main() -> None:
    cuda_id = 2
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(cuda_id))
    else:
        device = torch.device("cpu")
    print(torch.version.__version__) # type: ignore
    print(device)

    # Parameter generation
    (
        training_parameters,
        data_processing_parameters,
    ) = parameter_generation()  # To be parametrized

    # Dataset generation
    audio_processor = AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size("training")
    valid_size = audio_processor.get_size("validation")
    test_size = audio_processor.get_size("testing")
    print(
        "Dataset split (Train/valid/test): "
        + str(train_size)
        + "/"
        + str(valid_size)
        + "/"
        + str(test_size)
    )

    # Model generation and analysis
    model = DSCNN(use_bias=True)
    model.to(device)
    summary(
        model,
        input_size=(1, 1, 49, data_processing_parameters["feature_bin_count"]),
        verbose=2,
        col_width=16,
        col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
        device=device,
        row_settings=["var_names"],
    )
    print(model)

    # Training initialization
    trainining_environment = Train(audio_processor, training_parameters, model, device)

    # Removing stored inputs and activations
    remove_txt()

    start = time.clock_gettime(0)
    # trainining_environment.train(model)
    print(
        "Finished Training on GPU in {:.2f} seconds".format(
            time.clock_gettime(0) - start
        )
    )

    # Ignoring training, load pretrained model
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cuda")), strict=False
    )

    state_dict = torch.load(MODEL_PATH)
    print(state_dict.keys())
    # model_int8 = torch.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
    #     dtype=torch.qint8,
    # )
    trainining_environment.validate(model)
    # acc_int8 = trainining_environment.validate(model_int8)


def setup_ds_cnn_eval() -> tuple[torch.nn.Module, Any, OrderedDict[str, torch.Tensor]]:
    (
        training_parameters,
        data_processing_parameters,
    ) = parameter_generation()  # To be parametrized

    # Dataset generation
    training_parameters["batch_size"] = -1
    audio_processor = AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size("training")
    valid_size = audio_processor.get_size("validation")
    test_size = audio_processor.get_size("testing")
    print(
        "Dataset split (Train/valid/test): "
        + str(train_size)
        + "/"
        + str(valid_size)
        + "/"
        + str(test_size)
    )

    data = AudioGenerator("validation", audio_processor, training_parameters)
    model = DSCNN(use_bias=True)
    state_dict = torch.load(MODEL_PATH)

    return model, data, state_dict
