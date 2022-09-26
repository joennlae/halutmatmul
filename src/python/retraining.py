import argparse
import os
from pathlib import Path
from typing import Any, Optional

import torch

from models.resnet import ResNet18_Weights, resnet18
from training.utils_train import save_on_master, set_weight_decay  # type: ignore[attr-defined]
from training.train import load_data  # type: ignore[attr-defined]
from utils.analysis_helper import get_input_data_amount, get_layers, sys_info
from halutmatmul.halutmatmul import EncodingAlgorithm, HalutModuleConfig
from halutmatmul.model import HalutHelper


def load_model(
    checkpoint_path: str,
) -> tuple[str, torch.nn.Module, Any, Any, Any, Optional[dict[str, list[int]]], Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    print(args)
    args.distributed = False
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    _, dataset_test, _, test_sampler = load_data(train_dir, val_dir, args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size * 2,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.model == "resnet18":
        state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        model = resnet18(weights=state_dict, progress=True)
    else:
        raise Exception(f"model: {args.model} not supported")
    model.load_state_dict(checkpoint["model"])
    print(checkpoint.keys(), checkpoint["optimizer"].keys())
    print(
        checkpoint["optimizer"]["state"].keys(), checkpoint["optimizer"]["param_groups"]
    )
    print(checkpoint["args"])

    halut_modules = (
        checkpoint["halut_modules"] if "halut_modules" in checkpoint else None
    )
    return (
        args.model,
        model,
        state_dict,
        data_loader_test,
        args,
        halut_modules,
        checkpoint,
    )


def run_retraining(args: Any) -> None:
    (
        model_name,
        model,
        state_dict,
        data,
        args_checkpoint,
        halut_modules,
        checkpoint,
    ) = load_model(args.checkpoint)

    learned_path = args.learned
    if model_name not in learned_path.lower():
        learned_path += "/" + model_name
    Path(learned_path).mkdir(parents=True, exist_ok=True)

    halut_data_path = args.halutdata
    if model_name not in halut_data_path.lower():
        halut_data_path += "/" + model_name
    Path(halut_data_path).mkdir(parents=True, exist_ok=True)

    model.cuda()
    model.to(args.cuda_id)

    torch.cuda.set_device(args.cuda_id)
    sys_info()
    device = torch.device(
        "cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu"
    )

    halut_model = HalutHelper(
        model,
        state_dict,
        data,
        batch_size_inference=args_checkpoint.batch_size,
        batch_size_store=args_checkpoint.batch_size,
        data_path=halut_data_path,
        device=device,
        learned_path=learned_path,
        report_error=True,
    )
    halut_model.print_available_module()
    layers = get_layers(model_name)  # type: ignore[arg-type]
    print("modules", halut_modules, layers)

    if halut_modules is None:
        next_layer_idx = 0
    else:
        next_layer_idx = len(halut_modules.keys())
    next_layer = layers[next_layer_idx]
    C = int(args.C)
    enc = EncodingAlgorithm.FOUR_DIM_HASH
    K = 16
    rows = get_input_data_amount(model_name, next_layer)[-1]  # type: ignore[arg-type]
    modules = {next_layer: [C, rows, K, enc]}
    for k, v in modules.items():
        print("activate", k, v)
        halut_model.activate_halut_module(
            k,
            C=v[HalutModuleConfig.C],
            rows=v[HalutModuleConfig.ROWS],
            K=v[HalutModuleConfig.K],
            encoding_algorithm=v[HalutModuleConfig.ENCODING_ALGORITHM],
        )
    halut_model.run_inference()
    print(halut_model.get_stats())

    checkpoint["halut_modules"] = halut_model.halut_modules
    checkpoint["model"] = halut_model.model.state_dict()
    # update optimizer

    custom_keys_weight_decay = []
    if args_checkpoint.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args_checkpoint.bias_weight_decay))
    parameters = set_weight_decay(
        model,
        args_checkpoint.weight_decay,
        norm_weight_decay=args_checkpoint.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )
    opt_name = args_checkpoint.opt.lower()
    optimizer = torch.optim.SGD(
        parameters,
        lr=args_checkpoint.lr,
        momentum=args_checkpoint.momentum,
        weight_decay=args_checkpoint.weight_decay,
        nesterov="nesterov" in opt_name,
    )
    print("new optimizer", optimizer)
    checkpoint["optimizer"] = optimizer.state_dict()
    save_on_master(
        checkpoint,
        os.path.join(
            args_checkpoint.output_dir,
            f"model_halut_{len(halut_model.halut_modules.keys())}.pth",
        ),
    )


if __name__ == "__main__":
    DEFAULT_FOLDER = "/scratch2/janniss/"
    parser = argparse.ArgumentParser(description="Replace layer with halut")
    parser.add_argument(
        "cuda_id", metavar="N", type=int, help="id of cuda_card", default=0
    )
    parser.add_argument(
        "-halutdata",
        type=str,
        help="halut data path",
        default=DEFAULT_FOLDER + "/halut",
    )
    parser.add_argument(
        "-learned",
        type=str,
        help="halut learned path",
        default=DEFAULT_FOLDER + "/halut/learned",
    )
    parser.add_argument("-C", type=int, help="C", default=32)
    parser.add_argument("-modelname", type=str, help="model name", default="resnet18")
    parser.add_argument(
        "-resultpath",
        type=str,
        help="result_path",
        default="./results/data/resnet18/",
    )
    parser.add_argument(
        "-checkpoint",
        type=str,
        help="check_point_path",
        default="/usr/scratch2/vilan2/janniss/model_checkpoints/checkpoint_100.pth",
    )
    args = parser.parse_args()

    # layers = get_layers(args.modelname)
    # print(layers)
    run_retraining(args)
