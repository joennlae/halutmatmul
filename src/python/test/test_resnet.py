import sys
import tempfile
import requests
import torchvision
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from models.resnet import resnet18
from training import presets

from halutmatmul.model import HalutHelper


def download_weights(path: str, is_ci: bool = False) -> None:
    url = "https://github.com/joennlae/PyTorch_CIFAR10/raw/ver2022/resnet18.pth"

    # Streaming, so we can iterate over the response.
    # pylint: disable=missing-timeout
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2**20  # Mebibyte

    dl = 0
    print("Download ResNet-18 CIFAR-10 weights")
    with open(path + "/" + "resnet18.pth", "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
            dl += len(data)
            done = int(50 * dl / total_size)
            if not is_ci:
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()
    print("\n")


def test_cifar10_inference() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)
        download_weights(tmpdirname, is_ci=True)

        script_dir = tmpdirname
        state_dict = torch.load(
            script_dir + "/" + "resnet18" + ".pth", map_location="cpu"
        )

        # CIFAR transformation
        preprocessing = presets.ClassificationPresetEval(
            crop_size=32,
            resize_size=32,
            interpolation=InterpolationMode("bilinear"),
        )

        cifar_10_val = torchvision.datasets.CIFAR10(
            root=tmpdirname + "/.data",
            train=False,
            transform=preprocessing,
            download=True,
        )

        model = resnet18(progress=True, **{"is_cifar": True, "num_classes": 10})

        halut_model = HalutHelper(
            model,
            state_dict["model"],
            cifar_10_val,
            data_path=tmpdirname + "/.data",
            learned_path=tmpdirname + "/.data/learned",
            workers_offline_training=2,
            num_workers=2,
        )
        halut_model.print_available_module()
        halut_model.activate_halut_module("fc", 16, -1)
        halut_model.activate_halut_module("layer1.1.conv2", 16, -1)
        accuracy = halut_model.run_inference()
        accuracy = halut_model.run_inference()  # check if stored used
        assert accuracy >= 85.0
