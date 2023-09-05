import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from halutmatmul.modules import HalutConv2d, HalutLinear


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        HalutConv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def _weights_init(m):
    if isinstance(m, (HalutLinear, HalutConv2d)):
        init.kaiming_normal_(m.weight)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.5, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer("noise", torch.tensor(0))
        self.active = Parameter(torch.ones(1, dtype=torch.bool), requires_grad=False)

    def forward(self, x):
        if self.training and self.sigma != 0 and self.active:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class SimulateQuantError(nn.Module):
    def __init__(self, bitwidth=8):
        super().__init__()
        self.bitwidth = bitwidth
        self.active = Parameter(torch.ones(1, dtype=torch.bool), requires_grad=False)

    def forward(self, x):
        if self.training and self.active:
            scale_range = torch.max(x) - torch.min(x)
            scale = scale_range / (2**self.bitwidth - 1)
            scale = scale.item()
            x = torch.fake_quantize_per_tensor_affine(
                x,
                scale,
                0,
                quant_min=-(2 ** (self.bitwidth - 1)),
                quant_max=2 ** (self.bitwidth - 1) - 1,
            )
        return x


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()

        # bitwidth = 4

        self.conv1 = conv_block(in_channels, 64)
        # self.gause1 = GaussianNoise()
        # self.quant1 = SimulateQuantError(bitwidth=bitwidth)
        self.conv2 = conv_block(64, 128, pool=True)
        # self.gause2 = GaussianNoise()
        # self.quant2 = SimulateQuantError(bitwidth=bitwidth)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        # self.gause3 = GaussianNoise()
        # self.quant3 = SimulateQuantError(bitwidth=bitwidth)

        self.conv3 = conv_block(128, 256, pool=True)
        # self.gause4 = GaussianNoise()
        # self.quant4 = SimulateQuantError(bitwidth=bitwidth)
        self.conv4 = conv_block(256, 256, pool=True)
        # self.gause5 = GaussianNoise()
        # self.quant5 = SimulateQuantError(bitwidth=bitwidth)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        # self.gause6 = GaussianNoise()
        # self.quant6 = SimulateQuantError(bitwidth=bitwidth)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),  # MaxPool2d(4)
            nn.Flatten(),
            nn.Dropout(0.2),
            HalutLinear(256, num_classes),
        )
        self.apply(_weights_init)

    def forward(self, xb):
        out = self.conv1(xb)
        # out = self.gause1(out)
        # out = self.quant1(out)
        out = self.conv2(out)
        # out = self.gause2(out)
        # out = self.quant2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        # out = self.gause4(out)
        # out = self.quant4(out)
        out = self.conv4(out)
        # out = self.gause5(out)
        # out = self.quant5(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    model = ResNet9(3, 10)
    print(model)

    from torchinfo import summary

    summary(model, input_size=(1, 3, 32, 32))
