import glob
import os
from math import ceil
import re
from typing import Callable, Literal
from timeit import default_timer as timer
from multiprocessing import Process, JoinableQueue
import numpy as np
import torch

from models.resnet import END_STORE_A, END_STORE_B
import halutmatmul.halutmatmul as hm
from halutmatmul.modules import HalutConv2d

# https://docs.python.org/3/library/asyncio-queue.html#asyncio-queues


def check_is_more_than_enough(C: int, K: int, N: int, D: int) -> bool:
    total_prototypes = C * K
    # dims_per_codebook = ceil(D / C)
    rows_per_prototype = N / total_prototypes
    rows_per_dim = N / D
    print(f"rows_per_prototype: {rows_per_prototype}, rows_per_dim: {rows_per_dim}")
    ram_usage = N * D * 4 * 2  # cumsse reallocates
    print(f"RAM usage {ram_usage / (1024 * 1024 * 1024)} GB")
    # MAX_RAM = 25 * 1024 * 1024 * 1024
    return N > 10000000
    # return ram_usage > MAX_RAM


def learn_halut(
    l: str,
    C: int,
    data_path: str,
    store_path: str,
    K: int = 16,
    loop_order: Literal["im2col", "kn2col"] = "im2col",
    kernel_size: tuple[int, int] = (1, 1),  # only needed for kn2col
    stride: tuple[int, int] = (1, 1),  # only needed for kn2col
    padding: tuple[int, int] = (0, 0),  # only needed for kn2col
    niter=2,
    nredo=1,
    min_points_per_centroid=100,
    max_points_per_centroid=1000,
    codebook: int = -1,
) -> None:
    files = glob.glob(data_path + f"/{l}" + END_STORE_A)
    files = [x.split("/")[-1] for x in files]
    if len(files) > 1:
        raise Exception("more than one file not supported anymore")
    assert len(files) == 1
    a_numpy = np.load(data_path + f"/{l}" + END_STORE_A)

    save_path = store_path + f"/{l}_{C}_{K}.npy"

    b_numpy = np.load(data_path + f"/{l}" + END_STORE_B)
    _exists = os.path.exists(save_path)
    if _exists:
        print("already learned")
        if codebook == -1:
            return

        if loop_order == "kn2col":
            raise Exception("not implemented")

    halut_numpy = None
    if _exists:
        already_learned = np.load(save_path, allow_pickle=True)
        halut_numpy = hm.learn_halut_offline(
            a_numpy,
            b_numpy,
            C,
            K=K,
            niter=niter,
            nredo=nredo,
            min_points_per_centroid=min_points_per_centroid,
            max_points_per_centroid=max_points_per_centroid,
            codebook=codebook,
            already_learned=already_learned,
        )
    else:
        if loop_order == "im2col":
            halut_numpy = hm.learn_halut_offline(
                a_numpy,
                b_numpy,
                C,
                K=K,
                niter=niter,
                nredo=nredo,
                min_points_per_centroid=min_points_per_centroid,
                max_points_per_centroid=max_points_per_centroid,
                codebook=codebook,
            )
        elif loop_order == "kn2col":
            halut_numpy_list = []
            lut_list = []
            dims_list = []
            thresholds_list = []
            conv_layer = HalutConv2d(
                a_numpy.shape[-1],
                b_numpy.shape[-1],
                kernel_size,
                stride,
                padding,
            )
            a_torch = torch.from_numpy(a_numpy)
            for k_x in range(kernel_size[0]):
                for k_y in range(kernel_size[1]):
                    input_slice = conv_layer.kn2col_input_slice(
                        a_torch, a_torch.shape[1], a_torch.shape[2], k_x, k_y
                    )
                    input_slice = input_slice.reshape(-1, input_slice.shape[-1])
                    halut_numpy = hm.learn_halut_offline(
                        input_slice.detach().cpu().numpy(),
                        b_numpy[k_x * kernel_size[0] + k_y],
                        C,
                        K=K,
                    )
                    halut_numpy_list.append(halut_numpy)
                    lut_list.append(halut_numpy[hm.HalutOfflineStorage.LUT])
                    dims_list.append(halut_numpy[hm.HalutOfflineStorage.DIMS])
                    thresholds_list.append(
                        halut_numpy[hm.HalutOfflineStorage.THRESHOLDS]
                    )
            lut = np.array(lut_list)
            dims = np.array(dims_list)
            thresholds = np.array(thresholds_list)
            halut_numpy = halut_numpy_list[0]
            halut_numpy[hm.HalutOfflineStorage.LUT] = lut
            halut_numpy[hm.HalutOfflineStorage.DIMS] = dims
            halut_numpy[hm.HalutOfflineStorage.THRESHOLDS] = thresholds

    if halut_numpy is None:
        raise Exception("halut_numpy is None")
    print(f"Store in {save_path}: {halut_numpy.nbytes / (1024 * 1024)} MB")
    _exists = os.path.exists(store_path)
    if not _exists:
        os.makedirs(store_path)
        print(f"created directory {store_path}")
    np.save(save_path, halut_numpy)


def worker(name: str, queue: JoinableQueue, func: Callable) -> None:
    while True:
        params = queue.get()
        print(f"{name}, start: {params}")
        start = timer()
        try:
            func(*(params))
        except Exception as e:
            print("EXCEPTION: ", e)
        queue.task_done()
        end = timer()
        print(f"{name}, end: {params}")
        print(f"{name}", "Training time: %.2f s" % (end - start))


def learn_halut_multi_core_dict(
    dict_to_learn: dict[str, list],
    data_path: str,
    store_path: str,
    kmeans_options: dict = {},
    codebook: int = -1,
) -> None:
    for k, v in dict_to_learn.items():
        print("learning", k, v)
        conv2d_options = {
            "loop_order": "im2col",
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
        }
        kmeans_options_here = {
            "niter": 25,
            "nredo": 1,
            "min_points_per_centroid": 1,
            "max_points_per_centroid": 20000,
        }
        if len(v) > 2:
            for i in range(2, len(v)):
                conv2d_options[list(conv2d_options.keys())[i - 2]] = v[i]
        if len(kmeans_options) > 0:
            for key in kmeans_options.keys():
                kmeans_options_here[key] = kmeans_options[key]
        if len(kmeans_options) == 0 and codebook > -1:
            raise Exception("codebook is set but kmeans_options is empty")

        learn_halut(
            l=k,
            C=v[hm.HalutModuleConfig.C],
            data_path=data_path,
            store_path=store_path,
            K=v[hm.HalutModuleConfig.K],
            loop_order=conv2d_options["loop_order"],  # type: ignore
            kernel_size=conv2d_options["kernel_size"],  # type: ignore
            stride=conv2d_options["stride"],  # type: ignore
            padding=conv2d_options["padding"],  # type: ignore
            niter=kmeans_options_here["niter"],
            nredo=kmeans_options_here["nredo"],
            min_points_per_centroid=kmeans_options_here["min_points_per_centroid"],
            max_points_per_centroid=kmeans_options_here["max_points_per_centroid"],
            codebook=codebook,
        )

    print("==== FINISHED LEARNING (exited all tasks) =======")
