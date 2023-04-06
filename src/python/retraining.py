import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import torchvision

from training.timm_model import convert_to_halut
from training import utils_train
from training.utils_train import save_on_master, set_weight_decay  # type: ignore[attr-defined]
from training.train import load_data, main  # type: ignore[attr-defined]
from utils.analysis_helper import get_input_data_amount, get_layers, sys_info
from models.resnet import resnet18
from models.resnet20 import resnet20
from halutmatmul.halutmatmul import EncodingAlgorithm, HalutModuleConfig
from halutmatmul.model import HalutHelper, get_module_by_name
from halutmatmul.modules import HalutConv2d, HalutLinear


def load_model(
    checkpoint_path: str,
    distributed: bool = False,
    batch_size: int = 32,
) -> tuple[str, torch.nn.Module, Any, Any, Any, Any, Optional[dict[str, list]], Any,]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    args.batch_size = batch_size
    args.distributed = distributed
    args.workers = 4
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )
    num_classes = len(dataset.classes)
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        batch_size=args.batch_size,  # needs to be lower to work due to error calculations
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=None,
    )
    data_loader_test = torch.utils.data.DataLoader(  # type: ignore
        dataset_test,
        batch_size=args.batch_size,  # needs to be lower to work due to error calculations
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
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
        if args.model == "resnet20":
            model = resnet20()
    else:
        # model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
        model = torchvision.models.get_model(
            args.model, pretrained=True, num_classes=num_classes
        )
        convert_to_halut(model)
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


@record
def run_retraining(
    args: Any,
    test_only: bool = False,
    distributed: bool = False,
    batch_size: int = 32,
    lr: float = 0.01,
    lr_step_size: int = 8,
    # pylint: disable=unused-argument
    train_epochs: int = 20,
) -> tuple[Any, int, int]:
    (
        model_name,
        model,
        state_dict,
        data_loader_train,
        data_loader_val,
        args_checkpoint,
        halut_modules,
        checkpoint,
    ) = load_model(args.checkpoint, distributed=distributed, batch_size=batch_size)

    learned_path = args.learned
    if model_name not in learned_path.lower():
        learned_path += "/" + model_name
    Path(learned_path).mkdir(parents=True, exist_ok=True)

    halut_data_path = args.halutdata
    if model_name not in halut_data_path.lower():
        halut_data_path += "/" + model_name
    Path(halut_data_path).mkdir(parents=True, exist_ok=True)

    model_copy = deepcopy(model)
    model.to(args.gpu)

    torch.cuda.set_device(args.gpu)
    sys_info()
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    if not hasattr(args, "distributed"):
        args.distributed = False

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
        report_error=False,
        distributed=args.distributed,
        device_id=args.gpu,
    )
    halut_model.print_available_module()
    layers = get_layers(model_name)  # type: ignore[arg-type]
    print("modules", halut_modules, layers)

    if halut_modules is None:
        next_layer_idx = 0
        halut_modules = {}
    else:
        next_layer_idx = len(halut_modules.keys())
    # C = int(args.C)
    K = 16
    use_prototype = True

    # add all at once
    max = len(layers) - 1
    for i in range(next_layer_idx, max):
        if not test_only:
            next_layer = layers[i]
            c_base = 16
            loop_order = "im2col"
            c_ = c_base
            module_ref = get_module_by_name(halut_model.model, next_layer)
            if isinstance(module_ref, HalutConv2d):
                inner_dim_im2col = (
                    module_ref.in_channels
                    * module_ref.kernel_size[0]
                    * module_ref.kernel_size[1]
                )
                inner_dim_kn2col = module_ref.in_channels
                # if "layer3" in next_layer or "layer4" in next_layer:
                #     loop_order = "im2col"
                if loop_order == "im2col":
                    c_ = inner_dim_im2col // 9  # 9 = 3x3
                else:
                    c_ = (
                        inner_dim_kn2col // 8
                    )  # little lower than 9 but safer to work now

                if "downsample" in next_layer or "shortcut" in next_layer:
                    loop_order = "im2col"
                    c_ = inner_dim_im2col // 4
            print("module_ref", module_ref)
            if "fc" in next_layer:
                c_ = c_base  # fc.weight = [512, 10]
            modules = {next_layer: [c_, K, loop_order, use_prototype]} | halut_modules
        else:
            modules = halut_modules
        for k, v in modules.items():
            if len(v) > 3:
                halut_model.activate_halut_module(
                    k,
                    C=v[HalutModuleConfig.C],
                    K=v[HalutModuleConfig.K],
                    loop_order=v[HalutModuleConfig.LOOP_ORDER],
                    use_prototypes=v[HalutModuleConfig.USE_PROTOTYPES],
                )
            else:
                halut_model.activate_halut_module(
                    k,
                    C=v[HalutModuleConfig.C],
                    K=v[HalutModuleConfig.K],
                )
    if args.distributed:
        dist.barrier()
    halut_model.run_inference()
    if args.distributed:
        dist.barrier()
    print(halut_model.get_stats())

    if not test_only:
        print("modules", halut_model.halut_modules)

        checkpoint["halut_modules"] = halut_model.halut_modules
        checkpoint["model"] = halut_model.model.state_dict()
        # ugly hack to port halut active information
        model_copy.load_state_dict(checkpoint["model"])

        params = {
            "other": [],
            "prototypes": [],
            "temperature": [],
        }

        def _add_params(module, prefix=""):
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:  # removes parameter from optimizer!!
                    # filter(lambda p: p.requires_grad, model.parameters())
                    continue
                if isinstance(module, (HalutConv2d, HalutLinear)):
                    if name == "P":
                        params["prototypes"].append(p)
                        continue
                    if name == "temperature":
                        params["temperature"].append(p)
                        continue

                params["other"].append(p)

            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
                _add_params(child_module, prefix=child_prefix)

        _add_params(model)

        custom_lrs = {
            "temperature": 0.1,
        }
        param_groups = []
        # pylint: disable=consider-using-dict-items
        for key in params:
            if len(params[key]) > 0:
                # pylint: disable=consider-iterating-dictionary
                if key in custom_lrs.keys():
                    param_groups.append({"params": params[key], "lr": custom_lrs[key]})
                else:
                    param_groups.append({"params": params[key]})

        weight_decay = 0.0
        opt_name = "adam"
        lr_scheduler_name = "cosineannealinglr"
        args_checkpoint.lr_scheduler = lr_scheduler_name
        args_checkpoint.opt = opt_name
        # opt_name = args_checkpoint.opt.lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=args_checkpoint.momentum,
                weight_decay=weight_decay,
                nesterov="nesterov" in opt_name,
            )
            opt_state_dict = optimizer.state_dict()
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
            opt_state_dict = optimizer.state_dict()
        else:
            raise ValueError("Unknown optimizer {}".format(opt_name))

        # freeze learning rate by increasing step size
        # TODO: make learning rate more adaptive
        # imagenet 0.001, cifar10 0.01

        # TODO LR scheduler
        # checkpoint["lr_scheduler"]["step_size"] = lr_step_size  # imagenet 1, cifar10 7
        # args_checkpoint.lr_scheduler = "steplr"

        # if args_checkpoint.cifar10:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_epochs
        )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=lr_step_size, gamma=0.1
        # )
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()

        # optimizer updates
        checkpoint["optimizer"] = opt_state_dict

        # sys.exit(0)
        args_checkpoint.output_dir = os.path.dirname(args.checkpoint)  # type: ignore
        if not args.distributed or args.rank == 0:
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
    if not args.distributed or args.rank == 0:
        with open(
            f"{args.resultpath}/retrained_{len(halut_model.halut_modules.keys())}"
            f"{'_trained' if test_only else ''}.json",
            "w",
        ) as fp:
            json.dump(halut_model.get_stats(), fp, sort_keys=True, indent=4)
    idx = len(halut_model.halut_modules.keys())

    del model
    torch.cuda.empty_cache()
    return args_checkpoint, idx, len(layers)


# pylint: disable=consider-iterating-dictionary
def model_analysis(args: Any) -> None:
    (_, model, state_dict, _, _, _, _, _) = load_model(
        args.checkpoint, distributed=False
    )

    total_params = 0
    prev_params = 0
    all_results = {}
    print("model", model)
    for k, v in state_dict.items():
        layer = ".".join(k.split(".")[:-1])
        if (
            "bn" in layer or "downsample.1" in layer or "conv1" == layer
        ):  # downsample.1 = bn
            continue
        if layer not in all_results.keys():
            all_results[layer] = {}
        if ".lut" in k or ".threshold" in k or ".A" in k:
            print(k, v.shape)
            type_ = k.split(".")[-1]
            all_results[layer]["name"] = layer
            all_results[layer][f"{type_}_shape"] = v.shape
            all_results[layer][f"{type_}_size"] = v.numel() * 2 // 1024
            all_results[layer][f"{type_}_params"] = v.numel()
            if type_ == "lut" and len(v.shape) == 3:
                all_results[layer]["C"] = v.shape[1]
                all_results[layer]["K"] = v.shape[2]
                all_results[layer]["M"] = v.shape[0]
            total_params += v.numel()
            if ".A" in k or ".lut" in k:
                print(k, v)
        if ".weight" in k:
            print(k, v.shape)
            params_weight = v.numel() * 2 // 1024
            prev_params += v.numel()
            all_results[layer]["weight_size"] = v.numel() * 2 // 1024
            all_results[layer]["weight_params"] = v.numel()
            all_results[layer]["weight_shape"] = v.shape
            all_results[layer]["in"] = v.shape[1]
            all_results[layer]["out"] = v.shape[0]
            all_results[layer]["kernel_size"] = v.shape[2] if len(v.shape) > 2 else 1
            all_results[layer]["D"] = (
                v.shape[1] * all_results[layer]["kernel_size"] ** 2
            )
            print(k, params_weight)
    # pylint: disable=consider-using-dict-items
    for layer in all_results.keys():
        if "C" in all_results[layer].keys():
            all_results[layer]["CW"] = (
                all_results[layer]["D"] // all_results[layer]["C"]
            )
            all_results[layer]["ratio"] = (
                all_results[layer]["lut_size"] / all_results[layer]["weight_size"]
            )
        else:
            all_results[layer]["CW"] = 0
            all_results[layer]["ratio"] = 0
    print("total params", total_params)
    print("prev params", prev_params)
    print(all_results)
    df = pd.DataFrame(all_results).T
    print(df)
    df_selected = df[
        [
            "name",
            "D",
            "M",
            "C",
            "in",
            "out",
            "kernel_size",
            "CW",
            "lut_size",
            "weight_size",
            # "thresholds_size",
            "ratio",
        ]
    ]
    cidx = pd.Index(
        [
            "Layer Name",
            "D",
            "M",
            "C",
            "In",
            "Out",
            "Kernel",
            "CW",
            "LUT [kB]",
            "Weight [kB]",
            # "Threshold [kB]",
            "Ratio",
        ],
    )
    df_selected.columns = cidx
    styler = df_selected.style
    styler.format(subset="CW", precision=0).format_index(escape="latex", axis=1).format(
        subset="Ratio", precision=2
    ).format_index(escape="latex", axis=0).hide(level=0, axis=0)
    styler.to_latex(
        "table_resnet18_cifar10.tex",
        clines="skip-last;data",
        convert_css=True,
        position_float="centering",
        multicol_align="|c|",
        hrules=True,
        # float_format="%.2f",
    )
    # pylint: disable=import-outside-toplevel
    from torchinfo import summary

    summary(model, input_size=(1, 3, 32, 32), depth=4)


if __name__ == "__main__":
    DEFAULT_FOLDER = "/scratch2/janniss/"
    MODEL_NAME_EXTENSION = "cifar10-halut-resnet20"
    TRAIN_EPOCHS = 400  # imagenet 2, cifar10 max 40 as we use plateaulr
    BATCH_SIZE = 32
    LR = 0.001  # imagenet 0.001, cifar10 0.01
    LR_STEP_SIZE = 20
    GRADIENT_ACCUMULATION_STEPS = 8
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
        default=f"./results/data/resnet18-{MODEL_NAME_EXTENSION}/",
    )
    parser.add_argument(
        "-checkpoint",
        type=str,
        help="check_point_path",
        # WILL BE OVERWRITTEN!!!
        default=(
            f"/scratch2/janniss/model_checkpoints/{MODEL_NAME_EXTENSION}/retrained_checkpoint.pth"
            # f"/scratch2/janniss/model_checkpoints/cifar10/checkpoint.pth"
        ),
    )
    parser.add_argument(
        "-analysis",
        action="store_true",
        help="analysis",
    )
    parser.add_argument(
        "-single",
        action="store_true",
        help="run in non distributed mode",
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

    if args.analysis:
        print(
            args.checkpoint,
        )
        model_analysis(args)
        sys.exit(0)

    if args.single:
        args.gpu = args.cuda_id
        args_checkpoint, idx, total = run_retraining(
            args,
            distributed=False,
            batch_size=BATCH_SIZE,
            lr=LR,
            lr_step_size=LR_STEP_SIZE,
            train_epochs=TRAIN_EPOCHS,
        )
        args.distributed = False
        args_checkpoint.distributed = False
        args_checkpoint.world_size = 1
        args_checkpoint.rank = 0
        args_checkpoint.gpu = args.cuda_id
        args_checkpoint.device = "cuda:" + str(args.cuda_id)
    else:
        utils_train.init_distributed_mode(args)  # type: ignore[attr-defined]
        args_checkpoint, idx, total = run_retraining(
            args,
            distributed=True,
            batch_size=BATCH_SIZE,
            lr=LR,
            lr_step_size=LR_STEP_SIZE,
        )
        torch.cuda.set_device(args.gpu)
        # carry over rank, world_size, gpu backend
        args_checkpoint.rank = args.rank  # type: ignore
        args_checkpoint.world_size = args.world_size  # type: ignore
        args_checkpoint.gpu = args.gpu  # type: ignore
        args_checkpoint.distributed = args.distributed  # type: ignore
        args_checkpoint.dist_backend = args.dist_backend  # type: ignore
    args_checkpoint.workers = 4  # type: ignore
    args_checkpoint.output_dir = os.path.dirname(args.checkpoint)  # type: ignore
    args_checkpoint.simulate = False  # type: ignore
    for i in range(idx, total + 1):  # type: ignore
        args_checkpoint.epochs = args_checkpoint.epochs + TRAIN_EPOCHS  # type: ignore
        args_checkpoint.resume = (  # type: ignore
            f"{args_checkpoint.output_dir}/retrained_checkpoint_{i}.pth"  # type: ignore
        )
        if not args.single:
            dist.barrier()
            torch.cuda.empty_cache()
            torch.cuda.set_device(args.gpu)
            dist.barrier()
        # args_checkpoint.test_only = True
        # main(args_checkpoint, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
        # args_checkpoint.test_only = False
        main(args_checkpoint, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
        if not args.single:
            dist.barrier()
        # pylint: disable=line-too-long
        torch.cuda.empty_cache()
        args.checkpoint = f"{args_checkpoint.output_dir}/retrained_checkpoint_{i}_trained.pth"  # type: ignore
        if args.single or args.rank == 0:
            shutil.copy(
                os.path.join(args_checkpoint.output_dir, "checkpoint.pth"),  # type: ignore
                args.checkpoint,
            )
        if not args.single:
            dist.barrier()
        _, idx, total = run_retraining(
            args,
            test_only=True,
            distributed=not args.single,
            batch_size=BATCH_SIZE,
            lr=LR,
            lr_step_size=LR_STEP_SIZE,
        )
        torch.cuda.empty_cache()
        if args.distributed:
            dist.barrier()
        if idx < total:
            _, idx, total = run_retraining(
                args,
                distributed=not args.single,
                batch_size=BATCH_SIZE,
                lr=LR,
                lr_step_size=LR_STEP_SIZE,
            )  # do not overwrite args_checkpoint
        torch.cuda.empty_cache()
        if not args.single:
            dist.barrier()
