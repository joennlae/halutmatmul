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


import torch
from torch import nn

from models.dscnn.utils import npy_to_txt
from halutmatmul.modules import HalutConv2d, HalutLinear


class DSCNN(torch.nn.Module):
    def __init__(self, use_bias: bool = False) -> None:
        super().__init__()

        self.pad1 = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        self.conv1 = HalutConv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(10, 4),
            stride=(2, 2),
            bias=use_bias,
        )
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()

        self.pad2 = nn.ConstantPad2d((1, 1, 1, 1), value=0.0)
        self.conv2 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=64,
            bias=use_bias,
        )
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=use_bias,
        )
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()

        self.pad4 = nn.ConstantPad2d((1, 1, 1, 1), value=0.0)
        self.conv4 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=64,
            bias=use_bias,
        )
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=use_bias,
        )
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.relu5 = torch.nn.ReLU()

        self.pad6 = nn.ConstantPad2d((1, 1, 1, 1), value=0.0)
        self.conv6 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=64,
            bias=use_bias,
        )
        self.bn6 = torch.nn.BatchNorm2d(64)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=use_bias,
        )
        self.bn7 = torch.nn.BatchNorm2d(64)
        self.relu7 = torch.nn.ReLU()

        self.pad8 = nn.ConstantPad2d((1, 1, 1, 1), value=0.0)
        self.conv8 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            groups=64,
            bias=use_bias,
        )
        self.bn8 = torch.nn.BatchNorm2d(64)
        self.relu8 = torch.nn.ReLU()
        self.conv9 = HalutConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=use_bias,
        )
        self.bn9 = torch.nn.BatchNorm2d(64)
        self.relu9 = torch.nn.ReLU()

        self.avg = torch.nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.fc1 = torch.nn.Linear(64, 12, bias=use_bias)
        # self.soft  = torch.nn.Softmax(dim=1)
        # self.soft = F.log_softmax(x, dim=1)

        # CONV2D replacing Block1 for evaluation purposes
        # self.pad2  = nn.ConstantPad2d((1, 1, 1, 1), value=0.)
        # self.conv2 = HalutConv2d(in_channels = 64,
        # out_channels = 64, kernel_size = (3, 3), stride = (1, 1), groups = 1, bias = use_bias)
        # self.bn2   = torch.nn.BatchNorm2d(64)
        # self.relu2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, save: bool = False) -> torch.Tensor:
        if save:

            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            npy_to_txt(0, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            npy_to_txt(1, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            npy_to_txt(2, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad4(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            npy_to_txt(3, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            npy_to_txt(4, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad6(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.relu6(x)
            npy_to_txt(5, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv7(x)
            x = self.bn7(x)
            x = self.relu7(x)
            npy_to_txt(6, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.pad8(x)
            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)
            npy_to_txt(7, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))
            x = self.conv9(x)
            x = self.bn9(x)
            x = self.relu9(x)
            npy_to_txt(8, x.int().cpu().detach().numpy())
            print("Sum: ", str(torch.sum(x.int())))

            x = self.avg(x)
            npy_to_txt(9, x.int().cpu().detach().numpy())
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            npy_to_txt(10, x.int().cpu().detach().numpy())

        else:

            x = self.pad1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            x = self.pad2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)

            x = self.pad4(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)

            x = self.pad6(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.relu6(x)
            x = self.conv7(x)
            x = self.bn7(x)
            x = self.relu7(x)

            x = self.pad8(x)
            x = self.conv8(x)
            x = self.bn8(x)
            x = self.relu8(x)
            x = self.conv9(x)
            x = self.bn9(x)
            x = self.relu9(x)

            x = self.avg(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)

        return x  # To be compatible with Dory
        # return F.log_softmax(x, dim=1)
        # return F.softmax(x, dim=1)
