from typing import Optional, TypeVar, Union
import math
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset

from models.resnet import END_STORE_A, END_STORE_B
from timm.utils import accuracy
import models.levit.utils
from models.dscnn.dataset import AudioGenerator
from training import utils_train
from halutmatmul.modules import HalutConv2d, HalutLinear

T_co = TypeVar("T_co", covariant=True)

MAX_ROWS_FOR_SUBSAMPLING = 1024 * 128
RUN_ALL_SUBSAMPLING = 4419 * 4419

# pylint: disable=W0212
def write_inputs_to_disk(
    model: torch.nn.Module,
    batch_size: int,
    iteration: int,
    total_iterations: int,
    store_arrays: dict[str, np.ndarray],
    path: str = ".data/",
    additional_dict: Optional[dict[str, int]] = None,
    total_rows_store: int = MAX_ROWS_FOR_SUBSAMPLING,
    distributed: bool = False,
    device_id: int = 0,
) -> None:
    def store(module: torch.nn.Module, prefix: str = "") -> None:
        if (
            additional_dict is not None
            and prefix[:-1] in additional_dict
            and iteration > additional_dict[prefix[:-1]]
        ):
            return  # do not store to much
        if hasattr(module, "store_input"):
            if module.store_input:
                assert hasattr(module, "input_storage_a") or hasattr(
                    module, "input_storage_b"
                )
                if hasattr(module, "input_storage_a"):
                    rows_to_store_during_current_iter = math.ceil(
                        total_rows_store / total_iterations
                    )
                    rows = module.input_storage_a.shape[0]  # type: ignore[index]
                    effective_store_per_iter = rows_to_store_during_current_iter
                    if effective_store_per_iter > rows:
                        effective_store_per_iter = rows
                    effective_rows_stored = effective_store_per_iter * total_iterations
                    # subsampling
                    idx = np.arange(rows)
                    np.random.shuffle(idx)
                    np_array_a = (
                        module.input_storage_a[  # type: ignore[index]
                            idx[:effective_store_per_iter]
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if prefix + module._get_name() not in store_arrays:
                        store_arrays[prefix + module._get_name()] = np.zeros(
                            (effective_rows_stored, np_array_a.shape[1])
                        )
                    store_arrays[prefix + module._get_name()][
                        iteration
                        * effective_store_per_iter : (iteration + 1)
                        * effective_store_per_iter
                    ] = np_array_a
                    print(
                        "[SUBSAMPLED] store inputs for module",
                        prefix + module._get_name(),
                        np_array_a.shape,
                        np_array_a.shape[0]
                        * np_array_a.shape[1]
                        * 4
                        / (1024 * 1024 * 1024),
                        " GB",
                        iteration * effective_store_per_iter,
                        " / ",
                        effective_rows_stored,
                        f"GPU {device_id}" if distributed else "",
                    )

                    if iteration == total_iterations - 1:
                        np.save(
                            path
                            + "/"
                            + prefix[:-1]
                            + f"_{str(batch_size)}_{str(total_iterations)}"
                            + (f"_gpu_{str(device_id)}" if distributed else "")
                            + END_STORE_A,
                            store_arrays[prefix + module._get_name()],
                        )
                    module.input_storage_a = None  # type: ignore[assignment]
                if hasattr(module, "input_storage_b") and iteration == 0:
                    np_array_b = (
                        module.input_storage_b.detach().cpu().numpy()  # type: ignore[operator]
                    )
                    if (distributed and device_id == 0) or not distributed:
                        np.save(
                            path
                            + "/"
                            + prefix[:-1]
                            + f"_{str(batch_size)}_{str(total_iterations)}"
                            + END_STORE_B,
                            np_array_b,
                        )
                module.input_storage_b = None  # type: ignore[assignment]
        for name, child in module._modules.items():
            if child is not None:
                store(child, prefix + name + ".")

    store(model)
    del store


def get_and_print_layers_to_use_halut(
    model: torch.nn.Module,
) -> list[str]:
    all_layers = []

    def layers(module: torch.nn.Module, prefix: str = "") -> None:
        if isinstance(module, (HalutLinear, HalutConv2d)):
            all_layers.append(prefix[:-1])
        for name, child in module._modules.items():
            if child is not None:
                layers(child, prefix + name + ".")

    layers(model)
    del layers
    print(all_layers)
    return all_layers


def evaluate_distributed(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    is_store: bool,
    data_path: str = "./data",
    device_id: int = 0,
    print_freq: int = 1,
    log_suffix="",
):
    model.eval()
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    iterations = len(data_loader)
    store_arrays = {}
    criterion = torch.nn.CrossEntropyLoss()
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
            if is_store:
                print(
                    "iteration for storage: ",
                    image.shape,
                    device_id,
                    f" {n_iter + 1}/{iterations}",
                )
                write_inputs_to_disk(
                    model,
                    batch_size=batch_size,  # naming only, can have less elements in last iter
                    iteration=n_iter,
                    total_iterations=iterations,
                    store_arrays=store_arrays,
                    path=data_path,
                    total_rows_store=MAX_ROWS_FOR_SUBSAMPLING
                    // torch.distributed.get_world_size(),  # type: ignore
                    distributed=True,
                    device_id=device_id,
                )
            n_iter = n_iter + 1
    # gather the stats from all processes

    num_processed_samples = utils_train.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples  # type: ignore
        and torch.distributed.get_rank() == 0  # type: ignore
    ):
        # See FIXME above
        print(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, "  # type: ignore
            f"but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} "
        f"Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    torch.distributed.barrier()  # type: ignore
    if is_store:
        # merge inputs
        if device_id == 0:
            total_arrays = {}
            for key in store_arrays:
                read_in_arrays = []
                for i in range(torch.distributed.get_world_size()):  # type: ignore
                    read_in_arrays.append(
                        np.load(
                            data_path
                            + "/"
                            + key
                            + f"_{str(batch_size)}_{str(iterations)}"
                            + f"_gpu_{str(i)}"
                            + END_STORE_A
                        )
                    )
                    total_arrays[key] = np.concatenate(read_in_arrays, axis=0)
            for key, value in total_arrays.items():
                np.save(
                    data_path
                    + "/"
                    + key
                    + f"_{str(batch_size)}_{str(iterations)}"
                    + END_STORE_A,
                    value,
                )

    return metric_logger.acc1.global_avg


@torch.no_grad()
def evaluate_halut_imagenet(
    dataset: Union[Dataset[T_co], DataLoader],  # type: ignore
    model: torch.nn.Module,
    device: torch.device,
    is_store: bool = False,
    data_path: str = "./data",
    additional_dict: Optional[dict[str, int]] = None,
    batch_size: int = 128,
    num_workers: int = 8,
) -> tuple[float, float]:
    data_loader = dataset
    if isinstance(data_loader, Dataset):
        data_loader = DataLoader(
            dataset,  # type: ignore [arg-type]
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = models.levit.utils.MetricLogger(delimiter="  ")  # type: ignore[attr-defined]
    header = "Test:"

    iterations = len(data_loader)
    # switch to evaluation mode
    model.eval()
    n_iter = 0
    store_arrays = {}
    with torch.inference_mode():
        for images, target in metric_logger.log_every(data_loader, 1, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():  # type: ignore
                output = model(images)
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=images.shape[0])
            metric_logger.meters["acc5"].update(acc5.item(), n=images.shape[0])
            if is_store:
                print(
                    "iteration for storage: ",
                    images.shape,
                    f" {n_iter + 1}/{iterations}",
                )
                write_inputs_to_disk(
                    model,
                    batch_size=batch_size,  # naming only, can have less elements in last iter
                    iteration=n_iter,
                    total_iterations=iterations,
                    store_arrays=store_arrays,
                    path=data_path,
                    additional_dict=additional_dict,
                )
            n_iter = n_iter + 1

    # gather the stats from all processes
    # FIXME: only supporting non distributed envs for training of halut
    # metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} "
        "loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    print(output.mean().item(), output.std().item())

    return metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def eval_halut_kws(
    data: AudioGenerator,
    model: torch.nn.Module,
    device: torch.device,
    is_store: bool = False,
    data_path: str = "./data",
    additional_dict: Optional[dict[str, int]] = None,
    # pylint: disable=unused-argument
    batch_size: int = 128,
    num_workers: int = 8,
) -> tuple[float, float]:
    # data = AudioGenerator(mode, self.audio_processor, training_parameters)
    model.eval()

    store_arrays = {}
    with torch.no_grad():
        inputs_, labels_ = data[0]
        inputs = torch.Tensor(inputs_[:, None, :, :]).to(device)
        labels = torch.Tensor(labels_).long().to(device)

        # if integer:
        #     model = model.cpu()
        #     inputs = inputs * 255.0 / 255
        #     inputs = inputs.type(torch.uint8).type(torch.float).cpu()

        outputs = torch.nn.functional.softmax(model(inputs), dim=1)
        outputs = outputs.to(device)

        # _, predicted = torch.max(outputs, 1)
        # print(labels.size())
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        if is_store:
            print("iteration for storage: ", inputs.shape)
            write_inputs_to_disk(
                model,
                batch_size=0,  # naming only, can have less elements in last iter
                iteration=0,
                total_iterations=1,
                path=data_path,
                store_arrays=store_arrays,
                additional_dict=additional_dict,
            )

    print("Accuracy of the network on the %s set: %.2f %%" % ("validation", acc1))
    return acc1.cpu().detach().item(), acc5.cpu().detach().item()
