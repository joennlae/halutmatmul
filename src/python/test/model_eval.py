# simple model eval script
import sys
import tempfile
import requests

import torch
from torch import nn
from training.train import evaluate  # type: ignore[attr-defined]
from retraining import load_model

from halutmatmul.modules import HalutConv2d, HalutLinear


def download_model(
    path: str,
    is_ci: bool = False,
    url: str = "https://iis-people.ee.ethz.ch/~janniss/resnet9-best.pth",
) -> None:
    # Streaming, so we can iterate over the response.
    # pylint: disable=missing-timeout
    r = requests.get(url, stream=True)

    # Total size in Mebibyte
    total_size = int(r.headers.get("content-length", 0))
    block_size = 2**20  # Mebibyte

    dl = 0
    print("Download ResNet-9 CIFAR-10 luts, thresholds and dims")
    with open(path + "/" + "resnet9-best.pth", "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
            dl += len(data)
            done = int(50 * dl / total_size)
            if not is_ci:
                sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (50 - done)))
                sys.stdout.flush()
    print("\n")


with tempfile.TemporaryDirectory() as tmpdirname:
    print("created temporary directory", tmpdirname)
    download_model(tmpdirname, is_ci=False)

    script_dir = tmpdirname
    state_dict = torch.load(
        script_dir + "/" + "resnet9-best" + ".pth", map_location="cpu"
    )

    (
        model_name,
        model,
        state_dict,
        data_loader_train,
        data_loader_val,
        args_checkpoint,
        halut_modules,
        checkpoint,
    ) = load_model(script_dir + "/" + "resnet9-best" + ".pth")
    print(model)
    print(halut_modules)

    # model.to("cuda")
    # criterion = nn.CrossEntropyLoss()
    # evaluate(
    #     model,
    #     criterion=criterion,
    #     data_loader=data_loader_val,
    #     device="cuda",
    # )

    # int8 quantized model
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model,
        {HalutConv2d, HalutLinear},
        dtype=torch.qint8,
    )

    model_int8.to("cuda")
    criterion = nn.CrossEntropyLoss()

    acc1, acc5, loss = evaluate(
        model_int8,
        criterion=criterion,
        data_loader=data_loader_val,
        device="cuda",
    )

    assert acc1 > 0.90
    assert acc5 > 0.99
    assert loss < 0.5
