# source: https://github.com/pytorch/vision/blob/main/references/classification/train.py
# type: ignore
# pylint: disable=line-too-long, import-outside-toplevel, unnecessary-lambda-assignment
import datetime
import os
import sys
import time
import warnings

import torch
import torch.utils.data
import torchvision
from torch import nn
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

sys.path.append(os.getcwd())

# pylint: disable=wrong-import-position
from training import presets, transforms, utils_train
from training.sampler import RASampler
from training.timm_model import convert_to_halut
from models.resnet import resnet18
from models.resnet20 import resnet20

SCRATCH_BASE = "/scratch/janniss"


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    gradient_accumulation_steps=1,
):
    model.train()
    # batch accumulation parameter
    accum_iter = gradient_accumulation_steps
    optimizer.zero_grad()
    loss_total = 0
    reported_loss = 0

    # prev_lut = None
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils_train.SmoothedValue(window_size=1, fmt="{value}")
    )
    metric_logger.add_meter(
        "img/s", utils_train.SmoothedValue(window_size=10 * accum_iter, fmt="{value}")
    )

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image = image.half()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            reported_loss = loss.item()
            loss = loss / accum_iter
            loss.backward()

            # check if diff exists
            # if prev_lut is not None:
            #     diff = model.module.layer1[0].conv1.lut - prev_lut
            #     print("diff", torch.norm(diff).item())
            # prev_lut = model.module.layer1[0].conv1.lut.clone()

            loss_total += loss.item()

            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            # weights update
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
                loss_total = 0

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        # pylint: disable=unbalanced-tuple-unpacking
        acc1, acc5 = utils_train.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=reported_loss, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    return (
        metric_logger.acc1.global_avg,
        metric_logger.acc5.global_avg,
        metric_logger.loss.global_avg,
        metric_logger.lr.global_avg,
        metric_logger.meters["img/s"].global_avg,
    )


def evaluate(model, criterion, data_loader, device, print_freq=1, log_suffix=""):
    model.eval()
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.half()
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            # pylint: disable=unbalanced-tuple-unpacking
            acc1, acc5 = utils_train.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils_train.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return (
        metric_logger.acc1.global_avg,
        metric_logger.acc5.global_avg,
        metric_logger.loss.global_avg,
    )


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        preprocessing = presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        )
        if args.cifar100:
            dataset = torchvision.datasets.CIFAR100(
                root=SCRATCH_BASE + "/datasets",
                train=True,
                transform=preprocessing,
                download=True,
            )
        elif args.cifar10:
            # preprocessing = T.Compose(
            #     [
            #         T.RandomCrop(32, padding=4),
            #         T.RandomHorizontalFlip(),
            #         T.ToTensor(),
            #         T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #     ]
            # )
            dataset = torchvision.datasets.CIFAR10(
                root=SCRATCH_BASE + "/datasets",
                train=True,
                transform=preprocessing,
                download=True,
            )
        else:
            dataset = torchvision.datasets.ImageFolder(traindir, preprocessing)
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils_train.mkdir(os.path.dirname(cache_path))
            utils_train.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
            )
        if args.cifar100:
            dataset_test = torchvision.datasets.CIFAR100(
                root=SCRATCH_BASE + "/datasets",
                train=False,
                transform=preprocessing,
                download=True,
            )
        elif args.cifar10:
            # preprocessing = T.Compose(
            #     [
            #         T.ToTensor(),
            #         T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #     ]
            # )
            dataset_test = torchvision.datasets.CIFAR10(
                root=SCRATCH_BASE + "/datasets",
                train=False,
                transform=preprocessing,
                download=True,
            )
        else:
            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils_train.mkdir(os.path.dirname(cache_path))
            utils_train.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args, gradient_accumulation_steps=1):
    if args.output_dir:
        utils_train.mkdir(args.output_dir)

    if not hasattr(args, "distributed"):
        utils_train.init_distributed_mode(args)

    if args.cifar10 or args.cifar100:
        args.val_resize_size = 32
        args.val_crop_size = 32
        args.train_crop_size = 32

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha)
        )
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
    # if args.model == "resnet18":
    if args.cifar100:
        model = resnet18(
            progress=True, **{"is_cifar": True, "num_classes": num_classes}
        )
    elif args.cifar10:
        # model = resnet18(
        #     progress=True, **{"is_cifar": True, "num_classes": num_classes}
        # )
        model = resnet20()
    else:
        # model = timm.create_model(args.model, pretrained=True, num_classes=num_classes)
        # state_dict_copy = model.state_dict().copy()
        # convert_to_halut(model)
        # model.load_state_dict(state_dict_copy, strict=False)
        model = torchvision.models.get_model(
            args.model, pretrained=True, num_classes=num_classes
        )
        state_dict_copy = model.state_dict().copy()
        convert_to_halut(model)
        model.load_state_dict(state_dict_copy, strict=False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        # load to update halut deactivated layers
        model.load_state_dict(checkpoint["model"])
    model.half()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    # if args.cifar10:
    #     args.weight_decay = 5e-4
    parameters = utils_train.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.cifar10:
        args.lr_scheduler = "plateau"
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "plateau":
        main_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=6, verbose=True
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=args.lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[args.lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils_train.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    halut_modules = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        # if we activate this then the optimizer references get overwritten!!!!!
        # for lut, thresholds and so on
        # model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
        halut_modules = (
            checkpoint["halut_modules"] if "halut_modules" in checkpoint else None
        )

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_test, device=device, log_suffix="EMA"
            )
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    if args.simulate:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if halut_modules is not None:
                checkpoint["halut_modules"] = halut_modules
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils_train.save_on_master(
                checkpoint, os.path.join(args.output_dir, "model_base.pth")
            )
            utils_train.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint_base.pth")
            )
        return  # comment out to test if training is stable after simulation

    print("Start training")
    start_time = time.time()
    best_acc = 0.0
    writer = SummaryWriter(comment=os.path.basename(os.path.normpath(args.output_dir)))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        acc_train, acc5_train, loss_train, lr_train, imgs_train = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        writer.add_scalar("train/time_seconds", time.time() - start_time, epoch)
        writer.add_scalar("train/acc", acc_train, epoch)
        writer.add_scalar("train/acc5", acc5_train, epoch)
        writer.add_scalar("train/loss", loss_train, epoch)
        writer.add_scalar("train/lr", lr_train, epoch)
        writer.add_scalar("train/imgs", imgs_train, epoch)
        if args.lr_scheduler == "plateau":
            # pylint: disable=too-many-function-args
            lr_scheduler.step(loss_train)
        else:
            lr_scheduler.step()
        acc, acc5, loss = evaluate(model, criterion, data_loader_test, device=device)
        writer.add_scalar("test/acc", acc, epoch)
        writer.add_scalar("test/acc5", acc5, epoch)
        writer.add_scalar("test/loss", loss, epoch)
        writer.flush()
        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_test, device=device, log_suffix="EMA"
            )
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if halut_modules is not None:
                checkpoint["halut_modules"] = halut_modules
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if acc > best_acc:
                rm = os.path.join(args.output_dir, f"model_best-{best_acc:.2f}.pth")
                if os.path.exists(rm):
                    try:
                        os.remove(rm)
                    except OSError:
                        pass
                best_acc = acc
                utils_train.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, f"model_best-{best_acc:.2f}.pth"),
                )
            # utils_train.save_on_master(
            #     checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth")
            # )
            utils_train.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )
        optimizer_lr_local = optimizer.param_groups[0]["lr"]
        if args.distributed:
            dist.barrier()
            optimizer_lr_all = [
                torch.zeros((1), dtype=torch.float64, device=device)
                for _ in range(utils_train.get_world_size())
            ]
            dist.all_gather(
                optimizer_lr_all,
                torch.tensor([optimizer_lr_local], device=device, dtype=torch.float64),
            )
            # optimizer_lr_all [[0.0005], [0.0050], [0.0050], [0.0050]]
            optimizer_lr_local = optimizer_lr_all[0].item()
        if optimizer_lr_local < args.lr * 1e-4:
            print("learning rate too small, stop training")
            break

    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help
    )

    parser.add_argument(
        "--data-path",
        default="/scratch/ml_datasets/ILSVRC2012",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)"
    )
    parser.add_argument(
        "--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)"
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-min",
        default=0.0,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir",
        default="/scratch2/janniss/model_checkpoints/imagenet",
        type=str,
        help="path to save outputs",
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--auto-augment",
        default=None,
        type=str,
        help="auto augment policy (default: None)",
    )
    parser.add_argument(
        "--random-erase",
        default=0.0,
        type=float,
        help="random erasing probability (default: 0.0)",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
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
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )

    parser.add_argument(
        "--cifar100",
        action="store_true",
        help="Uses CIFAR100 dataset",
    )

    parser.add_argument(
        "--cifar10",
        action="store_true",
        help="Uses CIFAR10 dataset",
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate the training process",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
