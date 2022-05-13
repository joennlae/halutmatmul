# pylint: disable=C0209
import glob, re, json
from math import ceil
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Literal, Optional, TypeVar, Union
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ResNet.resnet import END_STORE_A, END_STORE_B
import halutmatmul.halutmatmul as hm
from halutmatmul.learn import learn_halut_multi_core_dict
from halutmatmul.modules import ErrorTuple

T_co = TypeVar("T_co", covariant=True)

DEFAULT_BATCH_SIZE_OFFLINE = 512
DEFAULT_BATCH_SIZE_INFERENCE = 128
DATA_PATH = "/scratch2/janniss/resnet_input_data"


class HalutModuleConfig:
    C = 0
    ROWS = 1
    K = 2


def editable_prefixes(state_dict: "OrderedDict[str, torch.Tensor]") -> list[str]:
    keys_weights = list(filter(lambda k: "weight" in k, state_dict.keys()))
    keys_conv_fc = list(
        filter(
            lambda k: any(v in k for v in ("fc", "conv", "downsample.0")), keys_weights
        )
    )
    keys = [v[: -(len(".weight"))] for v in keys_conv_fc]
    return keys


def check_file_exists(path: str) -> bool:
    file_path = Path(path)
    return file_path.is_file()


def check_file_exists_and_return_path(
    base_path: str,
    layers_name: str,
    _type: Union[Literal["input"], Literal["learned"]],
    C: int = 16,
    rows: int = 256,
    # K: int = 16,
) -> list[str]:
    files = glob.glob(base_path + "/*.npy")
    files_res = []
    if _type == "input":
        regex_a = rf"{layers_name}.+_0_.+{END_STORE_A}"
        regex_b = rf"{layers_name}.+_0_.+{END_STORE_B}"
        pattern_a = re.compile(regex_a)
        pattern_b = re.compile(regex_b)
        files_a = [x for x in files if pattern_a.search(x)]
        files_b = [x for x in files if pattern_b.search(x)]
        files_res = files_a + files_b
        assert len(files_res) == 0 or len(files_res) == 2
    elif _type == "learned":
        regex = rf"{layers_name}_{C}_{rows}-.+\.npy"
        pattern = re.compile(regex)
        files_res = [x for x in files if pattern.search(x)]
        assert len(files_res) == 0 or len(files_res) == 1
    return files_res


class HalutHelper:
    def __init__(
        self,
        model: torch.nn.Module,
        state_dict: "OrderedDict[str, torch.Tensor]",
        dataset: Dataset[T_co],
        data_path: str = DATA_PATH,
        batch_size_store: int = DEFAULT_BATCH_SIZE_OFFLINE,
        batch_size_inference: int = DEFAULT_BATCH_SIZE_INFERENCE,
        learned_path: str = DATA_PATH + "/learned/",
        device: torch.device = torch.device("cpu"),
        workers_offline_training: int = 1,
        report_error: bool = False,
        num_workers: int = 16,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size_store = batch_size_store
        self.batch_size_inference = batch_size_inference
        self.state_dict_base = state_dict
        self.editable_keys = editable_prefixes(state_dict)
        self.learned_path = learned_path
        self.halut_modules: Dict[str, list[int]] = dict([])
        self.data_path = data_path
        self.device = device
        self.stats: Dict[str, Any] = dict([])
        self.workers_offline_training = workers_offline_training
        self.report_error = report_error
        self.num_workers = num_workers

    def activate_halut_module(self, name: str, C: int, rows: int, K: int = 16) -> None:
        if name not in self.editable_keys:
            raise Exception(f"module {name} not in model")

        if name in self.halut_modules.keys():
            print(f"overwrite halut layer {name}")

        self.halut_modules |= dict({name: [C, rows, K]})

    def deactivate_halut_module(self, name: str) -> None:
        if name not in self.halut_modules.keys():
            print(f"layer to remove not a halut layer {name}")
        else:
            del self.halut_modules[name]

    def print_available_module(self) -> None:
        for k in self.editable_keys:
            print(k, " ", self.state_dict_base[k + ".weight"].shape)

    def __str__(self) -> str:
        ret = f"Activated Layers ({len(self.halut_modules.keys())}/{len(self.editable_keys)})"
        for k, v in self.halut_modules.items():
            ret += f"\n{k}: C: {v}, {self.state_dict_base[k + '.weight'].shape}"
        return ret

    def store_inputs(self, dict_to_store: dict[str, int]) -> None:
        print("keys to store", dict_to_store)
        max_iterations = 1
        for _, v in dict_to_store.items():
            if v > max_iterations:
                max_iterations = v
        dict_to_add = OrderedDict(
            [
                (k + ".store_input", torch.ones(1, dtype=torch.bool))
                for k in dict_to_store.keys()
            ]
        )
        state_dict_to_store = OrderedDict(self.state_dict_base | dict_to_add)
        if dict_to_store:
            self.run_for_input_storage(
                state_dict_to_store,
                iterations=max_iterations,
                additional_dict=dict_to_store,
            )

    def store_all(self, iterations: int = 1) -> None:
        dict_to_add = OrderedDict(
            [
                (k + ".store_input", torch.ones(1, dtype=torch.bool))
                for k in self.editable_keys
            ]
        )
        state_dict_to_store = OrderedDict(self.state_dict_base | dict_to_add)
        self.run_for_input_storage(state_dict_to_store, iterations=iterations)

    def run_for_input_storage(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        iterations: int = 1,
        additional_dict: Optional[dict[str, int]] = None,
    ) -> None:
        loaded_data = DataLoader(
            self.dataset,
            batch_size=self.batch_size_store,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        with torch.no_grad():
            for n_iter, (image, _) in enumerate(loaded_data):
                image = image.to(self.device)
                if n_iter > iterations:
                    break
                print(
                    "iteration for storage: ",
                    image.shape,
                    f" {n_iter + 1}/{iterations}",
                )
                self.model(image)
                self.model.write_inputs_to_disk(
                    batch_size=self.batch_size_store,
                    iteration=n_iter,
                    total_iterations=iterations,
                    path=self.data_path,  # type: ignore [operator]
                    additional_dict=additional_dict,
                )

    def run_halut_offline_training(self) -> None:
        dict_to_store: dict[str, int] = dict([])
        dict_to_learn: dict[str, list[int]] = dict([])
        for k, args in self.halut_modules.items():
            learned_files = check_file_exists_and_return_path(
                self.learned_path,
                k,
                "learned",
                args[HalutModuleConfig.C],
                args[HalutModuleConfig.ROWS],
            )
            if len(learned_files) == 1:
                continue
            dict_to_learn[k] = [args[HalutModuleConfig.C], args[HalutModuleConfig.ROWS]]
            paths = check_file_exists_and_return_path(self.data_path, k, "input")
            # TODO: doesn't check if enough data is stored
            if len(paths) != 2:
                dict_to_store[k] = ceil(
                    args[HalutModuleConfig.ROWS] / self.batch_size_store
                )
        self.store_inputs(dict_to_store)
        print(dict_to_learn, dict_to_store)
        learn_halut_multi_core_dict(
            dict_to_learn,
            data_path=self.data_path,
            batch_size=self.batch_size_store,
            store_path=self.learned_path,
            amount_of_workers=self.workers_offline_training,
        )

    def prepare_state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        additional_dict: Dict[str, torch.Tensor] = dict([])
        for k, args in self.halut_modules.items():
            learned_files = check_file_exists_and_return_path(
                self.learned_path,
                k,
                "learned",
                args[HalutModuleConfig.C],
                args[HalutModuleConfig.ROWS],
            )
            if len(learned_files) == 1:
                store_array = np.load(learned_files[0], allow_pickle=True)
                splitted = learned_files[0].split("/")[-1]
                configs_reg = re.findall(r"(?<=-)(\d+)", splitted)
                n = int(configs_reg[0])
                d = int(configs_reg[1])
                m = store_array[hm.HalutOfflineStorage.LUT].shape[0]
                print(f"Use Layer {k}: a: {(n, d)}, b: {(d, m)}")
                self.stats[k + ".learned_a_shape"] = (n, d)
                self.stats[k + ".learned_b_shape"] = (d, m)
                self.stats[k + ".learned_n"] = n
                self.stats[k + ".learned_m"] = m
                self.stats[k + ".learned_d"] = d
                self.stats[k + ".C"] = args[HalutModuleConfig.C]
                self.stats[k + ".rows"] = args[HalutModuleConfig.ROWS]
                self.stats[k + ".K"] = args[HalutModuleConfig.K]
                self.stats[k + ".stored_array_size"] = store_array.nbytes
                self.stats[k + ".L_size"] = (
                    store_array[hm.HalutOfflineStorage.LUT].astype(np.float32).nbytes
                )
                self.stats[k + ".H_size"] = (
                    store_array[hm.HalutOfflineStorage.HASH_TABLES]
                    .astype(np.float32)
                    .nbytes
                )
            else:
                raise Exception("learned file not found!")
            additional_dict = additional_dict | dict(
                {
                    k + ".halut_active": torch.ones(1, dtype=torch.bool),
                    k
                    + ".hash_buckets": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.HASH_TABLES].astype(
                            np.float32
                        )
                    ),
                    k
                    + ".lut": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.LUT].astype(np.float32)
                    ),
                    k
                    + ".halut_config": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.CONFIG].astype(np.float32)
                    ),
                    k + ".store_input": torch.zeros(1, dtype=torch.bool),
                    k + ".report_error": torch.ones(1, dtype=torch.bool)
                    if self.report_error
                    else torch.zeros(1, dtype=torch.bool),
                }
            )
        self.stats["halut_layers"] = json.dumps(self.halut_modules)
        return OrderedDict(self.state_dict_base | additional_dict)

    def get_stats(self) -> Dict[str, Any]:
        if self.report_error:
            for k in self.halut_modules.keys():
                submodule = self.model.get_submodule(k)
                errors = submodule.get_error()  # type: ignore[operator]
                self.stats[k + ".mae"] = errors[ErrorTuple.MAE]
                self.stats[k + ".mse"] = errors[ErrorTuple.MSE]
                self.stats[k + ".mape"] = errors[ErrorTuple.MAPE]
                self.stats[k + ".scaled_error"] = errors[ErrorTuple.SCALED_ERROR]
                self.stats[k + ".scaled_shift"] = errors[ErrorTuple.SCALED_SHIFT]
        return self.stats

    def run_inference(self) -> float:
        print("Start training of Halutmatmul")
        self.run_halut_offline_training()
        print("Start preparing state_dict")
        state_dict_with_halut = self.prepare_state_dict()
        print("Load state dict")
        start = timer()
        self.model.load_state_dict(state_dict_with_halut, strict=False)
        end = timer()
        print("State dict time: %.2f s" % (end - start))
        print("Init dataloader")
        start = timer()
        loaded_data = DataLoader(
            self.dataset,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        end = timer()
        print("Init dataloader time: %.2f s" % (end - start))
        self.model.eval()
        correct_5 = correct_1 = 0
        start = timer()
        with torch.no_grad():
            for n_iter, (image, label) in enumerate(loaded_data):
                image, label = image.to(self.device), label.to(self.device)
                # if n_iter > 10:
                #     continue
                print(
                    "iteration: {}\ttotal {} iterations".format(
                        n_iter + 1, len(loaded_data)
                    )
                )
                # https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b/test.py#L54
                output = self.model(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)
                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()
                correct_5 += correct[:, :5].sum()
                correct_1 += correct[:, :1].sum()
        end = timer()
        self.stats["total_time"] = end - start
        print(correct_1, correct_5)
        print("Top 1 error: ", 1 - correct_1 / len(loaded_data.dataset))  # type: ignore[arg-type]
        print("Top 5 error: ", 1 - correct_5 / len(loaded_data.dataset))  # type: ignore[arg-type]
        print("Top 1 accuracy: ", correct_1 / len(loaded_data.dataset))  # type: ignore[arg-type]
        print("Top 5 accuracy: ", correct_5 / len(loaded_data.dataset))  # type: ignore[arg-type]
        print(
            "Parameter numbers: {}".format(
                sum(p.numel() for p in self.model.parameters())
            )
        )
        self.stats["top_1_accuracy"] = (
            correct_1 / len(loaded_data.dataset)  # type: ignore[arg-type]
        ).item()  # type: ignore[attr-defined]
        self.stats["top_5_accuracy"] = (
            correct_5 / len(loaded_data.dataset)  # type: ignore[arg-type]
        ).item()  # type: ignore[attr-defined]
        return correct_1 / len(loaded_data.dataset)  # type: ignore[arg-type]
