# pylint: disable=C0209
import os, sys
from subprocess import call
import torchvision
import torch
from torchvision import transforms as T

from ResNet.resnet import ResNet50_Weights, resnet50

from halutmatmul.model import HalutHelper


def sys_info() -> None:
    print("__Python VERSION:", sys.version)
    print("__pyTorch VERSION:", torch.__version__)
    print(
        "__CUDA VERSION",
    )

    # ! nvcc --version
    print("__CUDNN VERSION:", torch.backends.cudnn.version())
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

    model = resnet50(
        weights=state_dict, progress=False, **{"is_cifar": True, "num_classes": 10}
    )

    halut_model = HalutHelper(model, state_dict, cifar_10_val)
    halut_model.print_available_module()
    halut_model.activate_halut_module("fc", 16)
    halut_model.activate_halut_module("layer4.2.conv3", 16)
    halut_model.run_inference()


def imagenet_inference() -> None:
    torch.cuda.set_device(2)
    sys_info()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    state_dict = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
    imagenet_val = torchvision.datasets.ImageNet(
        root="/scratch/janniss/imagenet/",
        split="val",
        transform=ResNet50_Weights.IMAGENET1K_V2.transforms(),
    )
    model = resnet50(weights=state_dict, progress=True)
    model.cuda()
    model.to(device)

    halut_model = HalutHelper(
        model,
        state_dict,
        imagenet_val,  # Dataloader
        batch_size_inference=128,
        batch_size_store=6 * 128,
        device=device,
    )
    halut_model.print_available_module()
    halut_model.activate_halut_module("layer1.0.conv1", 16)
    halut_model.run_inference()


if __name__ == "__main__":
    imagenet_inference()
