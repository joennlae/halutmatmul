# pylint: disable=C0209
import os, sys, glob, re
from subprocess import call
import torchvision
import torch
from torchvision import transforms as T

from models.resnet import END_STORE_A, END_STORE_B, ResNet50_Weights, resnet50

from halutmatmul.model import HalutHelper
import halutmatmul.halutmatmul as hm
from halutmatmul.learn import learn_halut_multi_core


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
    halut_model.activate_halut_module("fc", 16, 10000)
    halut_model.activate_halut_module("layer4.2.conv3", 16, 10000)
    halut_model.run_inference()


def store_offline_data(
    batch_size: int = 256,
    iterations: int = 40,
    data_path: str = "/scratch2/janniss/resnet_input_data",
    dataset_path: str = "/scratch2/janniss/imagenet/",
    device_id: int = 1,
) -> None:
    torch.cuda.set_device(device_id)
    sys_info()
    device = torch.device(
        "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
    )
    state_dict = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
    imagenet_val = torchvision.datasets.ImageNet(
        root=dataset_path,
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
        batch_size_inference=112,
        batch_size_store=batch_size,
        device=device,
        data_path=data_path,
    )
    halut_model.store_all(iterations=iterations)
    print(halut_model.get_stats())


def learn_halut_from_offline_data(
    data_path: str = "/scratch2/janniss/resnet_input_data",
) -> None:
    files = glob.glob(data_path + "/*.npy")
    files = [x.split("/")[-1] for x in files]
    configs_reg = re.findall(r"(?<=_)(\d+)", files[0])
    batch_size = int(configs_reg[0])
    iterations = int(configs_reg[2])
    print(files[0], configs_reg, batch_size, iterations)

    path = data_path + f"/learned_{batch_size}_{iterations}"
    _exists = os.path.exists(path)
    if not _exists:
        os.makedirs(path)
        print("The new directory is created!")

    layers_to_learn = list(
        {re.search(r"(^[^_]+)", x).group(0) for x in files}  # type: ignore[union-attr]
    )
    layers_to_learn.sort()

    print(layers_to_learn)
    # parameters
    C_all = [16, 32, 64]
    rows = [1, 2, 4, 8]  # [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 40 * 256]

    learn_halut_multi_core(
        C_all=C_all,
        layers_to_learn=layers_to_learn,
        rows=rows,
        data_path=data_path,
        batch_size=batch_size,
        store_path=path,
    )


if __name__ == "__main__":
    # store_offline_data(
    #     batch_size=256,
    #     iterations=40,
    #     data_path="/scratch2/janniss/resnet_input_data",
    #     dataset_path="/scratch2/janniss/imagenet",
    # )
    learn_halut_from_offline_data(data_path="/scratch2/janniss/resnet_input_data")
