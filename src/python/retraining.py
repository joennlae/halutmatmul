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
from torch.nn.parameter import Parameter
import torchvision

from training.timm_model import convert_to_halut
from training import utils_train
from training.utils_train import save_on_master, set_weight_decay  # type: ignore[attr-defined]
from training.train import load_data, main  # type: ignore[attr-defined]
from utils.analysis_helper import get_layers, sys_info
from models.resnet import resnet18
from models.resnet9 import ResNet9
from models.resnet20 import resnet20
from models.tiny.resnet8 import Resnet8v1EEMBC
from halutmatmul.halutmatmul import HalutModuleConfig
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
        elif args.model == "resnet9":
            model = ResNet9(3, num_classes)  # type: ignore
        elif args.model == "resnet8":
            model = Resnet8v1EEMBC()  # type: ignore
    else:
        # model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
        model = torchvision.models.get_model(
            args.model, pretrained=True, num_classes=num_classes
        )
        convert_to_halut(model)
    manipulated_state_dict = checkpoint["model"]
    for k in manipulated_state_dict.keys():
        if ".S" in k or ".B" in k:
            manipulated_state_dict[k] = torch.zeros(1, dtype=torch.bool)
    new_state_dict = model.state_dict()
    new_state_dict.update(checkpoint["model"])

    model.load_state_dict(new_state_dict, strict=False)

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
    # pylint: disable=unused-argument
    train_epochs: int = 20,
) -> tuple[Any, int, int, int]:
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

    if halut_modules is None:
        next_layer_idx = 0
        halut_modules = {}
    else:
        next_layer_idx = len(halut_modules.keys())
    K = 16
    use_prototype = False

    max = len(layers)
    max = next_layer_idx + 1
    for i in range(next_layer_idx, max):
        if not test_only and i < len(layers):
            next_layer = layers[i]
            c_base = 16
            loop_order = "im2col"  # kn2col only tested experimentally
            c_ = c_base
            module_ref = get_module_by_name(halut_model.model, next_layer)
            if isinstance(module_ref, HalutConv2d):
                inner_dim_im2col = (
                    module_ref.in_channels
                    * module_ref.kernel_size[0]
                    * module_ref.kernel_size[1]
                )
                inner_dim_kn2col = module_ref.in_channels
                if loop_order == "im2col":
                    c_ = inner_dim_im2col // 9  # 9 = 3x3
                    if module_ref.kernel_size[0] * module_ref.kernel_size[1] == 1:
                        c_ = inner_dim_im2col // 4
                else:
                    c_ = (
                        inner_dim_kn2col // 8
                    )  # little lower than 9 but safer to work now
                if "downsample" in next_layer or "shortcut" in next_layer:
                    loop_order = "im2col"
                    c_ = inner_dim_im2col // 4
            print("module_ref", module_ref)
            if isinstance(module_ref, HalutLinear):
                c_ = 256 // 4
            modules = {next_layer: [c_, K, loop_order, use_prototype]} | halut_modules
        else:
            modules = halut_modules
        for k, v in modules.items():
            if len(v) > 3:
                halut_model.activate_halut_module(
                    k,
                    C=v[HalutModuleConfig.C],  # type: ignore
                    K=v[HalutModuleConfig.K],  # type: ignore
                    loop_order=v[HalutModuleConfig.LOOP_ORDER],  # type: ignore
                    use_prototypes=v[HalutModuleConfig.USE_PROTOTYPES],  # type: ignore
                )
            else:
                halut_model.activate_halut_module(
                    k,
                    C=v[HalutModuleConfig.C],  # type: ignore
                    K=v[HalutModuleConfig.K],  # type: ignore
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

        params = {  # type: ignore
            "other": [],
            "thresholds": [],
            "temperature": [],
            "lut": [],
        }

        def _add_params(module, prefix=""):
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if isinstance(module, (HalutConv2d, HalutLinear)):
                    if name == "thresholds":
                        params["thresholds"].append(p)
                        continue
                    if name == "temperature":  # temperature currently not trained
                        # params["temperature"].append(p)
                        continue
                    if name == "lut":
                        params["lut"].append(p)
                        continue
                params["other"].append(p)  # add batch normalization

            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
                _add_params(child_module, prefix=child_prefix)

        _add_params(model)
        params["old_lut"] = params["lut"][:-1]
        params["lut"] = params["lut"][-1:]
        params["old_thresholds"] = params["thresholds"][:-1]
        params["thresholds"] = params["thresholds"][-1:]
        custom_lrs = {
            "temperature": 0.1 * 0.0,
            "thresholds": lr / 2,
            "old_thresholds": lr / 2,
            "old_lut": lr,
            "lut": lr,
            "other": lr,
        }
        args_checkpoint.lr = lr
        param_groups = []
        # pylint: disable=consider-using-dict-items
        for key in params:
            if len(params[key]) > 0:
                # pylint: disable=consider-iterating-dictionary
                if key in custom_lrs.keys():
                    param_groups.append({"params": params[key], "lr": custom_lrs[key]})
                else:
                    param_groups.append({"params": params[key]})

        print("param_groups", len(param_groups))
        weight_decay = 0.0
        opt_name = "adam"
        lr_scheduler_name = "cosineannealinglr"
        args_checkpoint.lr_scheduler = lr_scheduler_name
        args_checkpoint.opt = opt_name
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
            optimizer = torch.optim.Adam(  # type: ignore
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
            )
            opt_state_dict = optimizer.state_dict()
        else:
            raise ValueError("Unknown optimizer {}".format(opt_name))

        if args_checkpoint.lr_scheduler == "cosineannealinglr":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_epochs,
                eta_min=0.0002,  # no eta_min during fine-tuning
            )
        elif args_checkpoint.lr_scheduler == "plateau":
            # this should give us three levels
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # type: ignore
                optimizer,
                mode="min",
                factor=0.5,
                patience=6,
                verbose=True,
                min_lr=0.0002,
            )
            args_checkpoint.min_lr_to_break = 0.0002
        else:
            raise Exception(
                "Unknown lr scheduler {}".format(args_checkpoint.lr_scheduler)
            )

        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        checkpoint["optimizer"] = opt_state_dict

        args_checkpoint.output_dir = args.output_dir
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
    return args_checkpoint, idx, len(layers), checkpoint["epoch"]


if __name__ == "__main__":
    DEFAULT_FOLDER = "/scratch2/janniss/"
    MODEL_NAME_EXTENSION = "cifar10-halut-resnet9"
    TRAIN_EPOCHS = 25  # 25 layer-per-layer, 300 fine-tuning
    BATCH_SIZE = 128  # 128
    LR = 0.001  # 0.001/0.002 layer-per-payer, 0.0005 fine-tuning
    GRADIENT_ACCUMULATION_STEPS = 1
    parser = argparse.ArgumentParser(description="Replace layer with halut")
    parser.add_argument(
        "cuda_id", metavar="N", type=int, help="id of cuda_card", default=0
    )
    parser.add_argument(
        "-testname",
        type=str,
        help="test name",
        default=None,
    )
    parser.add_argument(
        "-halutdata",
        type=str,
        help="halut data path",
        default=DEFAULT_FOLDER + f"/halut/resnet9-{MODEL_NAME_EXTENSION}",
    )
    parser.add_argument(
        "-learned",
        type=str,
        help="halut learned path",
        default=DEFAULT_FOLDER + f"/halut/resnet9-{MODEL_NAME_EXTENSION}/learned",
    )
    parser.add_argument("-C", type=int, help="C", default=64)
    parser.add_argument("-modelname", type=str, help="model name", default="resnet18")
    parser.add_argument(
        "-resultpath",
        type=str,
        help="result_path",
        default=f"./results/data/resnet9-{MODEL_NAME_EXTENSION}/",
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

    print(args)

    if args.testname is not None:
        args.resultpath = f"./results/data/{args.testname}/"
        args.halutdata = DEFAULT_FOLDER + f"/halut/{args.testname}/"
        args.learned = DEFAULT_FOLDER + f"/halut/{args.testname}/learned/"

    output_dir = os.path.dirname(args.checkpoint)
    if args.testname is not None:
        output_dir = DEFAULT_FOLDER + f"/halut/{args.testname}/checkpoints/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    if args.single:
        args.gpu = args.cuda_id
        args_checkpoint, idx, total, epoch = run_retraining(
            args,
            distributed=False,
            batch_size=BATCH_SIZE,
            lr=LR,
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
        args_checkpoint, idx, total, epoch = run_retraining(
            args,
            distributed=True,
            batch_size=BATCH_SIZE,
            lr=LR,
            train_epochs=TRAIN_EPOCHS,
        )
        torch.cuda.set_device(args.gpu)
        # carry over rank, world_size, gpu backend
        args_checkpoint.rank = args.rank  # type: ignore
        args_checkpoint.world_size = args.world_size  # type: ignore
        args_checkpoint.gpu = args.gpu  # type: ignore
        args_checkpoint.distributed = args.distributed  # type: ignore
        args_checkpoint.dist_backend = args.dist_backend  # type: ignore
        args_checkpoint.device = "cuda:" + str(args.gpu)
    args_checkpoint.workers = 4  # type: ignore
    args_checkpoint.simulate = False  # type: ignore
    args_checkpoint.testname = args.testname  # type: ignore
    for i in range(idx, total + 1):  # type: ignore
        args_checkpoint.epochs = epoch + TRAIN_EPOCHS  # type: ignore
        args_checkpoint.resume = (  # type: ignore
            f"{args_checkpoint.output_dir}/retrained_checkpoint_{i}.pth"  # type: ignore
        )
        if not args.single:
            dist.barrier()
            torch.cuda.empty_cache()
            torch.cuda.set_device(args.gpu)
            dist.barrier()
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
        _, idx, total, epoch = run_retraining(
            args,
            test_only=True,
            distributed=not args.single,
            batch_size=BATCH_SIZE,
            lr=LR,
            train_epochs=TRAIN_EPOCHS,
        )
        torch.cuda.empty_cache()
        if args.distributed:
            dist.barrier()
        if idx < total:
            _, idx, total, epoch = run_retraining(
                args,
                distributed=not args.single,
                batch_size=BATCH_SIZE,
                lr=LR,
                train_epochs=TRAIN_EPOCHS,
            )  # do not overwrite args_checkpoint
        torch.cuda.empty_cache()
        if not args.single:
            dist.barrier()
