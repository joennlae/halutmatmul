import torch
from torch import nn
from torch.nn import functional as F

from halutmatmul.modules import HalutConv2d, HalutLinear

halut_active = False


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            HalutConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                stride=stride,
                halut_active=halut_active,
                split_factor=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            HalutConv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                halut_active=halut_active,
                split_factor=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = HalutConv2d(  # type: ignore
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                halut_active=halut_active,
                split_factor=1,
            )

    def forward(self, inputs):
        x = self.block(inputs)
        y = self.residual(inputs)
        return F.relu(x + y)


class Resnet8v1EEMBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
        )

        self.first_stack = ResNetBlock(in_channels=16, out_channels=16, stride=1)
        self.second_stack = ResNetBlock(in_channels=16, out_channels=32, stride=2)
        self.third_stack = ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.first_stack(x)
        x = self.second_stack(x)
        x = self.third_stack(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = Resnet8v1EEMBC().to("cpu")
    print(model)

    from torchinfo import summary

    summary(model, input_size=(1, 3, 32, 32))

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(
        model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
