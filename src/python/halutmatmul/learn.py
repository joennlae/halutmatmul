import glob
import os
from math import ceil
import re
from typing import Callable
from timeit import default_timer as timer
from multiprocessing import Process, JoinableQueue
import numpy as np

from models.resnet import END_STORE_A, END_STORE_B
import halutmatmul.halutmatmul as hm

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
    r: int,
    data_path: str,
    batch_size: int,
    store_path: str,
    K: int = 16,
) -> None:
    print("start learning", l, C, r, K)
    files = glob.glob(data_path + f"/{l}_{batch_size}_*" + END_STORE_A)
    files = [x.split("/")[-1] for x in files]
    print(files)
    print(data_path)
    if len(files) > 1:
        raise Exception("more than one file not supported anymore")
    assert len(files) == 1
    configs_reg = re.findall(r"(?<=_)(\d+)", files[0])
    iterations = int(configs_reg[1])
    a_numpy = np.load(data_path + f"/{l}_{batch_size}_{iterations}" + END_STORE_A)

    save_path = store_path + f"/{l}_{C}_{K}_{r}-{a_numpy.shape[1]}.npy"
    _exists = os.path.exists(save_path)
    if _exists:
        print("already learned")
        return

    print(
        "A input: ",
        a_numpy.shape,
        a_numpy.shape[0] * a_numpy.shape[1] * 4 / (1024 * 1024 * 1024),
        " GB",
    )
    b_numpy = np.load(data_path + f"/{l}_{batch_size}_{iterations}" + END_STORE_B)
    print(
        "B input: ",
        b_numpy.shape,
        b_numpy.shape[0] * b_numpy.shape[1] * 4 / (1024 * 1024),
        " MB",
    )
    halut_numpy = hm.learn_halut_offline(a_numpy, b_numpy, C, K=K)
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


def learn_halut_multi_core(
    C_all: list[int],
    layers_to_learn: list[str],
    rows: list[int],
    data_path: str,
    batch_size: int,
    store_path: str,
) -> None:
    WORKERS = 2
    queue: JoinableQueue = JoinableQueue()

    for l in layers_to_learn:
        for C in C_all:
            for r in rows:
                params = (l, C, r, data_path, batch_size, store_path)
                queue.put_nowait(params)

    processes: list[Process] = []
    for i in range(WORKERS):
        process = Process(
            target=worker,
            args=(f"worker-{i}", queue, learn_halut),
        )
        process.daemon = True
        process.start()
        processes.append(process)

    queue.join()

    for process in processes:
        process.kill()

    print("====")


def learn_halut_multi_core_dict(
    dict_to_learn: dict[str, list[int]],
    data_path: str,
    batch_size: int,
    store_path: str,
    amount_of_workers: int = 1,
) -> None:

    if amount_of_workers > 1:
        print(
            "WARNING: will set to amount_of_workers=1 as there is a "
            "multiprocessing bug in cpython with np.multiply."
        )
        amount_of_workers = 1

    WORKERS = amount_of_workers
    queue: JoinableQueue = JoinableQueue()

    for k, v in dict_to_learn.items():
        params = (
            k,
            v[hm.HalutModuleConfig.C],
            v[hm.HalutModuleConfig.ROWS],
            data_path,
            batch_size,
            store_path,
            v[hm.HalutModuleConfig.K],
        )
        if amount_of_workers == 1:
            learn_halut(*(params))
        else:
            queue.put_nowait(params)

    processes: list[Process] = []
    for i in range(WORKERS):
        process = Process(
            target=worker,
            args=(f"worker-{i}", queue, learn_halut),
        )
        process.daemon = True
        process.start()
        processes.append(process)

    queue.join()

    for process in processes:
        process.kill()

    print("==== FINISHED LEARNING (exited all tasks) =======")
