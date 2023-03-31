import sys, os
import shutil
import csv
from copy import deepcopy
import tempfile
import requests
import torchvision
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from training import presets
from utils.analysis_helper import resnet20_layers, resnet20_b_layers

from models.resnet import resnet18
from models.resnet20 import resnet20
from models.helper import write_module_back
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
        halut_model.activate_halut_module("layer1.1.conv2", 16, use_prototypes=True)
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
    with tempfile.TemporaryDirectory(dir="/scratch2/janniss/tmp/") as tmpdirname:
        print("created temporary directory", tmpdirname)
        download_weights_resnet20(
            tmpdirname,
            is_ci=True,
            # pylint: disable=line-too-long
            # url="https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20-12fca82f.th",
            # url="https://iis-people.ee.ethz.ch/~janniss/resnet20.pth",
            url="https://iis-people.ee.ethz.ch/~janniss/resnet20-B-final.pth",
            # url="https://iis-people.ee.ethz.ch/~janniss/resnet20-A-overtrained.pth",
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
            train=True,
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

        data = []
        device = torch.device("cuda:" + str(3) if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        model.to(device=device)
        prev_max = 0.0
        resampling = 1
        reseeding = 1
        best_model = None
        codebooks = 0
        #
        for resampled in range(resampling):
            if resampled > 0:
                # delete all files in ".data/"
                for path in os.listdir(tmpdirname + "/.data"):
                    full_path = os.path.join(tmpdirname + "/.data", path)
                    if os.path.isfile(full_path):
                        os.remove(full_path)

            niter_to_check = 0
            if "layer1.1" in layer:
                niter_to_check = 5
            if "layer1.2" in layer:
                niter_to_check = 10
            if "layer2" in layer:
                niter_to_check = 25
            if "layer3" in layer:
                niter_to_check = 25
            if "linear" in layer:
                niter_to_check = 25

            for i in range(reseeding):
                for niter in [niter_to_check]:
                    # for nredo in [1]:
                    #    for min_points_per_centroid in [1]:
                    nredo = 1
                    min_points_per_centroid = 1
                    max_points_per_centroid = 20000
                    # for max_points_per_centroid in [20000]:
                    kmeans_options = {
                        "niter": niter,
                        "nredo": nredo,
                        "min_points_per_centroid": min_points_per_centroid,
                        "max_points_per_centroid": max_points_per_centroid,
                    }
                    shutil.rmtree(tmpdirname + "/.data/learned", ignore_errors=True)

                    model_base = deepcopy(model)
                    halut_model = HalutHelper(
                        model_base,
                        state_dict,
                        cifar_10_val,
                        data_path=tmpdirname + "/.data",
                        learned_path=tmpdirname + "/.data/learned",
                        workers_offline_training=2,
                        num_workers=2,
                        batch_size_inference=128,
                        kmeans_options=kmeans_options,
                        device=device,
                    )
                    halut_model.print_available_module()
                    # for _layer in resnet20_layers:
                    c_ = 16
                    # for layer in resnet20_b_layers[:-1]:
                    module_ref = get_module_by_name(halut_model.model, layer)
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
                            )  # little lower than 9 but safer to work no
                        if "downsample" in layer or "shortcut" in layer:
                            loop_order = "im2col"
                            c_ = inner_dim_im2col // 4
                    if isinstance(module_ref, HalutLinear):
                        c_ = module_ref.in_features // 4
                    # if layer == "layer1.2.conv1":
                    #     c_ = c_ * 3
                    halut_model.activate_halut_module(layer, c_, use_prototypes=True)
                    codebooks = c_
                    print(model)
                    accuracy = halut_model.run_inference(prev_max=prev_max)
                    if accuracy > prev_max:
                        prev_max = accuracy
                        print("NEW MAX", prev_max)
                        best_model = deepcopy(halut_model.model)
                    row = [
                        accuracy,
                        niter,
                        nredo,
                        min_points_per_centroid,
                        max_points_per_centroid,
                        i,
                    ]
                    data.append(row)
                    print("accuracy", accuracy)
                    print("row", row)

        state_dict = best_model.state_dict()  # type: ignore
        for c in range(codebooks):
            for _ in range(10):
                halut_model = HalutHelper(
                    model_base,
                    state_dict,  # type: ignore
                    cifar_10_val,
                    data_path=tmpdirname + "/.data",
                    learned_path=tmpdirname + "/.data/learned",
                    workers_offline_training=2,
                    num_workers=2,
                    batch_size_inference=128,
                    kmeans_options=kmeans_options,
                    device=device,
                )
                halut_model.activate_halut_module(layer, codebooks, use_prototypes=True)
                accuracy = halut_model.run_inference(prev_max=prev_max, codebook=c)
                if accuracy > prev_max:
                    prev_max = accuracy
                    print("NEW MAX", prev_max)
                    best_model = deepcopy(halut_model.model)
                    state_dict = best_model.state_dict()

        write_module_back(layer, best_model, tmpdirname + "/.data/learned")  # type: ignore

        torch.save(best_model.state_dict(), f"model_init_{layer}.pt")  # type: ignore
        print("data", data)
        print("MAX", prev_max)

        with open(f"data_{layer}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return prev_max
