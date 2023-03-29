import sys
import tempfile
import requests
import torchvision
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from training import presets
from utils.analysis_helper import resnet20_layers

from models.resnet import resnet18
from models.resnet20 import resnet20
from halutmatmul.model import get_module_by_name
from halutmatmul.modules import HalutConv2d, HalutLinear
from halutmatmul.model import HalutHelper


def download_weights(
    path: str,
    is_ci: bool = False,
    url: str = "https://github.com/joennlae/PyTorch_CIFAR10/raw/ver2022/resnet18.pth",
) -> None:

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
        # halut_model.activate_halut_module("fc", 16, -1)
        halut_model.activate_halut_module("layer1.1.conv2", 16, -1, use_prototypes=True)
        # halut_model.activate_halut_module("layer3.1.conv1", 16, -1, 16, "kn2col")
        accuracy = halut_model.run_inference()
        accuracy = halut_model.run_inference()  # check if stored used
        assert accuracy >= 81.8


def download_weights_resnet20(
    path: str,
    is_ci: bool = False,
    url: str = "https://github.com/joennlae/PyTorch_CIFAR10/raw/ver2022/resnet18.pth",
) -> None:

    # Streaming, so we can iterate over the response.
    # pylint: disable=missing-timeout
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2**20  # Mebibyte

    dl = 0
    print("Download ResNet-20 CIFAR-10 weights")
    with open(path + "/" + "resnet20.th", "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
            dl += len(data)
            done = int(50 * dl / total_size)
            if not is_ci:
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()
    print("\n")


# pylint: disable=unused-argument
def test_cifar10_inference_resnet20(layer: str = "layer1.0.conv2") -> float:
    with tempfile.TemporaryDirectory(dir="/scratch/tmp/") as tmpdirname:
        print("created temporary directory", tmpdirname)
        download_weights_resnet20(
            tmpdirname,
            is_ci=True,
            # pylint: disable=line-too-long
            # url="https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th",
            url="https://iis-people.ee.ethz.ch/~janniss/resnet20.pth",
        )

        script_dir = tmpdirname
        state_dict = torch.load(script_dir + "/" + "resnet20.th", map_location="cpu")

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

        model = resnet20()
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict["state_dict"].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # model.load_state_dict(new_state_dict, strict=False)
        # state_dict = model.state_dict()
        print(state_dict.keys())
        state_dict = state_dict["model"]

        halut_model = HalutHelper(
            model,
            state_dict,
            cifar_10_val,
            data_path=tmpdirname + "/.data",
            learned_path=tmpdirname + "/.data/learned",
            workers_offline_training=2,
            num_workers=2,
            batch_size_inference=128,
        )
        halut_model.print_available_module()
        for _layer in resnet20_layers:
            c_ = 16
            module_ref = get_module_by_name(halut_model.model, _layer)
            if isinstance(module_ref, HalutConv2d):
                inner_dim_im2col = (
                    module_ref.in_channels
                    * module_ref.kernel_size[0]
                    * module_ref.kernel_size[1]
                )
                inner_dim_kn2col = module_ref.in_channels
                loop_order = module_ref.loop_order
                if loop_order == "im2col":
                    c_ = inner_dim_im2col // 9  # 9 = 3x3
                else:
                    c_ = (
                        inner_dim_kn2col // 8
                    )  # little lower than 9 but safer to work now

                if "downsample" in _layer:
                    loop_order = "im2col"
                    c_ = inner_dim_im2col // 4
            if isinstance(module_ref, HalutLinear):
                c_ = module_ref.in_features // 4  # fc.weight = [512, 10]
            print("module_ref", module_ref)
            print("c_", c_)
            halut_model.activate_halut_module(_layer, c_, -1, use_prototypes=True)
        print(model)
        accuracy = halut_model.run_inference()
        # accuracy = halut_model.run_inference()  # check if stored used
        return accuracy
