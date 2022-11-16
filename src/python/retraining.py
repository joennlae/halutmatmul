import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

import torch

from models.resnet import ResNet18_Weights, resnet18
from training import utils_train
from training.utils_train import save_on_master, set_weight_decay  # type: ignore[attr-defined]
from training.train import load_data, main  # type: ignore[attr-defined]
from utils.analysis_helper import get_input_data_amount, get_layers, sys_info
from halutmatmul.halutmatmul import EncodingAlgorithm, HalutModuleConfig
from halutmatmul.model import HalutHelper


def load_model(
    checkpoint_path: str,
) -> tuple[
    str,
    torch.nn.Module,
    Any,
    Any,
    Any,
    Any,
    Optional[dict[str, list[int]]],
    Any,
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    args.distributed = False
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,  # needs to be lower to work due to error calculations
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,  # needs to be lower to work due to error calculations
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.model == "resnet18":
        if args.cifar100:
            model = resnet18(
                progress=True,
                **{"is_cifar": True, "num_classes": 100},  # type: ignore[arg-type]
            )
        elif args.cifar10:
            model = resnet18(
                progress=True,
                **{"is_cifar": True, "num_classes": 10},  # type: ignore[arg-type]
            )
        else:
            model = resnet18(progress=True)
    else:
        raise Exception(f"model: {args.model} not supported")
    model.load_state_dict(checkpoint["model"])

    halut_modules = (
        checkpoint["halut_modules"] if "halut_modules" in checkpoint else None
    )
    return (
        args.model,
        model,
        checkpoint["model"],
        data_loader,
        data_loader_test,
        args,
        halut_modules,
        checkpoint,
    )


def run_retraining(args: Any, test_only: bool = False) -> tuple[Any, int, int]:
    (
        model_name,
        model,
        state_dict,
        data_loader_train,
        data_loader_val,
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

    model_copy = deepcopy(model)
    model.cuda()
    model.to(args.gpu)

    original_stdout = sys.stdout
    with open("resnet18.txt", "w") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("model", model)
        sys.stdout = original_stdout

    torch.cuda.set_device(args.gpu)
    sys_info()
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    halut_model = HalutHelper(
        model,
        state_dict,
        dataset=data_loader_val,
        dataset_train=data_loader_train,
        batch_size_inference=32,  # will be ignored and taken from data_loader
        batch_size_store=args_checkpoint.batch_size,  # will be ignored and taken from data_loader
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
        halut_modules = {}
    else:
        next_layer_idx = len(halut_modules.keys())
    next_layer = layers[next_layer_idx]
    C = int(args.C)
    print()
    enc = EncodingAlgorithm.FOUR_DIM_HASH
    K = 16
    rows = -1  # subsampling
    if not test_only:
        modules = {next_layer: [C, rows, K, enc]} | halut_modules
    else:
        modules = halut_modules
    for k, v in modules.items():
        halut_model.activate_halut_module(
            k,
            C=v[HalutModuleConfig.C],
            rows=v[HalutModuleConfig.ROWS],
            K=v[HalutModuleConfig.K],
            encoding_algorithm=v[HalutModuleConfig.ENCODING_ALGORITHM],
        )
    halut_model.run_inference()
    print(halut_model.get_stats())

    if not test_only:
        print("modules", halut_model.halut_modules)

        checkpoint["halut_modules"] = halut_model.halut_modules
        checkpoint["model"] = halut_model.model.state_dict()
        # ugly hack to port halut active information
        model_copy.load_state_dict(checkpoint["model"])

        # update optimizer with new parameters
        # pylint: disable=using-constant-test
        if False:
            custom_keys_weight_decay = []
            if args_checkpoint.bias_weight_decay is not None:
                custom_keys_weight_decay.append(
                    ("bias", args_checkpoint.bias_weight_decay)
                )
            parameters = set_weight_decay(
                model_copy,
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
            # optimizer.load_state_dict(checkpoint["optimizer"])
            opt_state_dict = optimizer.state_dict()
            opt_state_dict["param_groups"][0]["weight_decay"] = checkpoint["optimizer"][
                "param_groups"
            ][0]["weight_decay"]
            opt_state_dict["param_groups"][0]["lr"] = checkpoint["optimizer"][
                "param_groups"
            ][0]["lr"]
            opt_state_dict["param_groups"][0]["momentum"] = checkpoint["optimizer"][
                "param_groups"
            ][0]["momentum"]
            print(opt_state_dict["param_groups"])
            # ACTIVATE REPLACE AND FREEZE TRAINING
            # optimizer updates
            # checkpoint["optimizer"] = opt_state_dict

        # freeze learning rate by increasing step size
        # TODO: make learning rate more adaptive
        checkpoint["lr_scheduler"]["step_size"] = 4419

        save_on_master(
            checkpoint,
            os.path.join(
                args_checkpoint.output_dir,
                f"model_halut_{len(halut_model.halut_modules.keys())}.pth",
            ),
        )
        save_on_master(
            checkpoint,
            os.path.join(
                args_checkpoint.output_dir,
                f"retrained_checkpoint_{len(halut_model.halut_modules.keys())}.pth",
            ),
        )

    result_base_path = args.resultpath
    if model_name not in result_base_path.lower():
        result_base_path += "/" + model_name + "/"
    Path(result_base_path).mkdir(parents=True, exist_ok=True)

    with open(
        f"{args.resultpath}/retrained_{len(halut_model.halut_modules.keys())}"
        f"{'_trained' if test_only else ''}.json",
        "w",
    ) as fp:
        json.dump(halut_model.get_stats(), fp, sort_keys=True, indent=4)
    idx = len(halut_model.halut_modules.keys())
    return args_checkpoint, idx, len(layers)


if __name__ == "__main__":
    DEFAULT_FOLDER = "/scratch2/janniss/"
    MODEL_NAME_EXTENSION = "cifar100"
    parser = argparse.ArgumentParser(description="Replace layer with halut")
    parser.add_argument(
        "cuda_id", metavar="N", type=int, help="id of cuda_card", default=0
    )
    parser.add_argument(
        "-halutdata",
        type=str,
        help="halut data path",
        default=DEFAULT_FOLDER + f"/halut/resnet18-{MODEL_NAME_EXTENSION}",
    )
    parser.add_argument(
        "-learned",
        type=str,
        help="halut learned path",
        default=DEFAULT_FOLDER + f"/halut/resnet18-{MODEL_NAME_EXTENSION}/learned",
    )
    parser.add_argument("-C", type=int, help="C", default=64)
    parser.add_argument("-modelname", type=str, help="model name", default="resnet18")
    parser.add_argument(
        "-resultpath",
        type=str,
        help="result_path",
        default=f"./results/data/resnet18-{MODEL_NAME_EXTENSION}-e2e-true/",
    )
    parser.add_argument(
        "-checkpoint",
        type=str,
        help="check_point_path",
        # WILL BE OVERWRITTEN!!!
        default=(
            f"/scratch2/janniss/model_checkpoints/{MODEL_NAME_EXTENSION}/retrained_checkpoint.pth"
        ),
    )
    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    args = parser.parse_args()

    utils_train.init_distributed_mode(args)  # type: ignore[attr-defined]
    # start with retraining checkpoint
    if args.rank == 0:
        args_checkpoint, idx, total = run_retraining(args)
        return_values = [args_checkpoint, idx, total]
    else:
        return_values = [None, None, None]
    torch.cuda.set_device(args.gpu)
    torch.distributed.broadcast_object_list(return_values, src=0)
    TRAIN_EPOCHS = 25
    args_checkpoint = return_values[0]
    idx = return_values[1]
    total = return_values[2]
    # carry over rank, world_size, gpu backend
    args_checkpoint.rank = args.rank
    args_checkpoint.world_size = args.world_size
    args_checkpoint.gpu = args.gpu
    args_checkpoint.distributed = args.distributed
    args_checkpoint.dist_backend = args.dist_backend
    args_checkpoint.workers = 0
    for i in range(idx, total):
        args_checkpoint.epochs = args_checkpoint.epochs + TRAIN_EPOCHS
        args_checkpoint.resume = (
            f"{args_checkpoint.output_dir}/retrained_checkpoint_{i}.pth"
        )
        torch.distributed.barrier()
        # sys_info()
        torch.cuda.empty_cache()
        torch.cuda.set_device(args.gpu)
        torch.distributed.barrier()
        # sys_info()
        main(args_checkpoint)
        torch.distributed.barrier()
        args.checkpoint = (
            f"{args_checkpoint.output_dir}/retrained_checkpoint_{i}_trained.pth"
        )
        if args.rank == 0:
            shutil.copy(
                os.path.join(args_checkpoint.output_dir, "checkpoint.pth"),
                args.checkpoint,
            )
            _, idx, total = run_retraining(args, test_only=True)
            _, idx, total = run_retraining(args)  # do not overwrite args_checkpoint
        torch.distributed.barrier()
