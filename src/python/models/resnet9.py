import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torchvision.models.quantization.utils import _fuse_modules, quantize_model
from halutmatmul.modules import HalutConv2d, HalutLinear


def conv_block(
    in_channels, out_channels, pool=False, halut_active=False, use_torch_conv=False
):
    layers = [
        HalutConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            halut_active=halut_active,
            split_factor=4,
        )
        if not use_torch_conv
        else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


halut_active = False
use_torch_conv = False  # use for quantization aware training


def _weights_init(m):
    if isinstance(m, (HalutLinear, HalutConv2d)):
        init.kaiming_normal_(m.weight)
    if isinstance(m, HalutConv2d) and m.halut_active:
        init.kaiming_normal_(m.lut)
        init.normal_(m.thresholds)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()

        self.conv1 = conv_block(in_channels, 64, use_torch_conv=use_torch_conv)
        self.conv2 = conv_block(
            64, 128, pool=True, halut_active=halut_active, use_torch_conv=use_torch_conv
        )
        self.res1 = nn.Sequential(
            conv_block(
                128, 128, halut_active=halut_active, use_torch_conv=use_torch_conv
            ),
            conv_block(
                128, 128, halut_active=halut_active, use_torch_conv=use_torch_conv
            ),
        )
        self.conv3 = conv_block(
            128,
            256,
            pool=True,
            halut_active=halut_active,
            use_torch_conv=use_torch_conv,
        )
        self.conv4 = conv_block(
            256,
            256,
            pool=True,
            halut_active=halut_active,
            use_torch_conv=use_torch_conv,
        )
        self.res2 = nn.Sequential(
            conv_block(
                256, 256, halut_active=halut_active, use_torch_conv=use_torch_conv
            ),
            conv_block(
                256, 256, halut_active=halut_active, use_torch_conv=use_torch_conv
            ),
        )
        self.maxpool = nn.MaxPool2d(4)
        self.classifier = nn.Sequential(
            HalutLinear(256, num_classes)
            if not use_torch_conv
            else nn.Linear(256, num_classes)
        )
        self.apply(_weights_init)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.maxpool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out

    def fuse_model(self, is_qat=False):
        _fuse_modules(
            self,
            [
                ["conv1.0", "conv1.1", "conv1.2"],
                ["conv2.0", "conv2.1", "conv2.2"],
                ["res1.0.0", "res1.0.1", "res1.0.2"],
                ["res1.1.0", "res1.1.1", "res1.1.2"],
                ["conv3.0", "conv3.1", "conv3.2"],
                ["conv4.0", "conv4.1", "conv4.2"],
                ["res2.0.0", "res2.0.1", "res2.0.2"],
                ["res2.1.0", "res2.1.1", "res2.1.2"],
            ],
            is_qat=is_qat,
            inplace=True,
        )
        return self


if __name__ == "__main__":
    model = ResNet9(3, 10)
    print(model)

    from torchinfo import summary

    summary(model, input_size=(1, 3, 32, 32))

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(
        model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
