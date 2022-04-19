import sys
import tempfile
import requests
import torchvision
import torch
from torchvision import transforms as T

from ResNet.resnet import resnet50

from halutmatmul.model import HalutHelper


def download_weights(path: str) -> None:
    url = "https://github.com/joennlae/PyTorch_CIFAR10/raw/ver2022/resnet50.pt"

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2**20  # Mebibyte

    dl = 0
    print("Download ResNet-50 CIFAR-10 weights")
    with open(path + "/" + "resnet50.pt", "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
            dl += len(data)
            done = int(50 * dl / total_size)
            sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
            sys.stdout.flush()
    print("\n")


def test_cifar10_inference() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("created temporary directory", tmpdirname)
        download_weights(tmpdirname)

        script_dir = tmpdirname
        state_dict = torch.load(
            script_dir + "/" + "resnet50" + ".pt", map_location="cpu"
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
            root=tmpdirname + "/.data",
            train=False,
            transform=val_transform,
            download=True,
        )

        model = resnet50(
            weights=state_dict, progress=False, **{"is_cifar": True, "num_classes": 10}
        )

        halut_model = HalutHelper(
            model,
            state_dict,
            cifar_10_val,
            data_path=tmpdirname + "/.data",
            learned_path=tmpdirname + "/.data/learned",
            workers_offline_training=2,
        )
        halut_model.print_available_module()
        halut_model.activate_halut_module("fc", 16, 10000)
        halut_model.activate_halut_module("layer4.2.conv3", 16, 10000)
        accuracy = halut_model.run_inference()
        accuracy = halut_model.run_inference()  # check if stored used
        assert accuracy >= 0.9318
