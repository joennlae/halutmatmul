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
import torch
import nemo
from torchinfo import summary

from models.dscnn.dataset import AudioProcessor
from models.dscnn.model import DSCNN
from models.dscnn.utils import remove_txt, parameter_generation

# from pthflops import count_ops
from models.dscnn.train import Train


def run_kws_main() -> None:
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(torch.version.__version__)
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
    model.load_state_dict(torch.load("./model.pth", map_location=torch.device("cuda")))

    # model_int8 = torch.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {torch.nn.Linear, torch.nn.Conv2d},  # a set of layers to dynamically quantize
    #     dtype=torch.qint8,
    # )
    trainining_environment.validate(model)
    # acc_int8 = trainining_environment.validate(model_int8)

    # print("accuracies", acc_fp32, acc_int8)
    # Accuracy on the training set.
    # # print ("Training acc")
    # acc = trainining_environment.validate(model, mode='training', batch_size=-1, statistics=False)
    # # Accuracy on the validation set.
    # print ("Validation acc")
    # acc = trainining_environment.validate(
    # model, mode='validation', batch_size=-1, statistics=False)
    # # Accuracy on the testing set.
    # print ("Testing acc")
    # acc = trainining_environment.validate(model, mode='testing', batch_size=-1, statistics=False)

    # Initiating quantization process: making the model quantization aware
    # quantized_model = nemo.transform.quantize_pact(
    #     deepcopy(model), dummy_input=torch.randn((1, 1, 49, 10)).to(device)
    # )

    # precision_8 = {
    #     "conv1": {"W_bits": 7},
    #     "relu1": {"x_bits": 8},
    #     "conv2": {"W_bits": 7},
    #     "relu2": {"x_bits": 8},
    #     "conv3": {"W_bits": 7},
    #     "relu3": {"x_bits": 8},
    #     "conv4": {"W_bits": 7},
    #     "relu4": {"x_bits": 8},
    #     "conv5": {"W_bits": 7},
    #     "relu5": {"x_bits": 8},
    #     "conv6": {"W_bits": 7},
    #     "relu6": {"x_bits": 8},
    #     "conv7": {"W_bits": 7},
    #     "relu7": {"x_bits": 8},
    #     "conv8": {"W_bits": 7},
    #     "relu8": {"x_bits": 8},
    #     "conv9": {"W_bits": 7},
    #     "relu9": {"x_bits": 8},
    #     "fc1": {"W_bits": 7},
    # }
    # quantized_model.change_precision(
    #     bits=1, min_prec_dict=precision_8, scale_weights=True, scale_activations=True
    # )

    # Calibrating model's scaling by collecting largest activations
    # with quantized_model.statistics_act():
    #     trainining_environment.validate(
    #         model=quantized_model, mode="validation", batch_size=128
    #     )
    # quantized_model.reset_alpha_act()

    # Remove biases after FQ stage
    # quantized_model.remove_bias()

    # print("\nFakeQuantized @ 8b accuracy (calibrated):")
    # acc = trainining_environment.validate(
    #     model=quantized_model, mode="testing", batch_size=-1
    # )

    # quantized_model.qd_stage(eps_in=255.0 / 255)  # The activations are already in 0-255

    # print("\nQuantizedDeployable @ mixed-precision accuracy:")
    # acc = trainining_environment.validate(
    #     model=quantized_model, mode="testing", batch_size=-1
    # )

    # quantized_model.id_stage()

    # print("\nIntegerDeployable @ mixed-precision accuracy:")
    # acc = trainining_environment.validate(
    #     model=quantized_model, mode="testing", batch_size=-1, integer=True
    # )

    # Saving the model
    # nemo.utils.export_onnx("model.onnx", quantized_model, quantized_model, (1, 49, 10))
    # Saving the activations for comparison within Dory
    # acc = trainining_environment.validate(
    #     model=quantized_model, mode="testing", batch_size=1, integer=True, save=True
    # )
