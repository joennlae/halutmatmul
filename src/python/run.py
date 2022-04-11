# pylint: disable=C0209
import os

import torchvision
import torch
from torchvision import transforms as T

from ResNet.resnet import resnet50

from halutmatmul.model import HalutHelper


def cifar_inference() -> None:
    script_dir = os.path.dirname(__file__)
    state_dict = torch.load(
        script_dir + "/.data/" + "resnet50" + ".pt", map_location="cpu"
    )

    # CIFAR transformation
    val_transform = T.Compose(
        [
            T.Resize(32),
            T.CenterCrop(32),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ]
    )

    cifar_10_val = torchvision.datasets.CIFAR10(
        root="./.data", train=False, transform=val_transform, download=False
    )

    model = resnet50(weights=state_dict, progress=False)

    halut_model = HalutHelper(model, state_dict, cifar_10_val)
    halut_model.print_available_module()
    halut_model.activate_halut_module("fc", 16)
    halut_model.activate_halut_module("layer4.2.conv3", 16)
    halut_model.run_inference()


if __name__ == "__main__":
    cifar_inference()
