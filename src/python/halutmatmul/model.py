# pylint: disable=C0209
import json
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, TypeVar
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ResNet.resnet import END_STORE_A, END_STORE_B
import halutmatmul.halutmatmul as hm

T_co = TypeVar("T_co", covariant=True)

DEFAULT_BATCH_SIZE_OFFLINE = 512
DEFAULT_BATCH_SIZE_INFERENCE = 128
DATA_PATH = "/scratch/janniss/data"


def editable_prefixes(state_dict: "OrderedDict[str, torch.Tensor]") -> list[str]:
    keys_weights = list(filter(lambda k: "weight" in k, state_dict.keys()))
    keys_conv_fc = list(
        filter(lambda k: any(v in k for v in ("fc", "conv")), keys_weights)
    )
    keys = [v[: -(len(".weight"))] for v in keys_conv_fc]
    return keys


def check_file_exists(path: str) -> bool:
    file_path = Path(path)
    return file_path.is_file()


class HalutHelper:
    def __init__(
        self,
        model: torch.nn.Module,
        state_dict: "OrderedDict[str, torch.Tensor]",
        dataset: Dataset[T_co],
        data_path: str = DATA_PATH,
        batch_size_store: int = DEFAULT_BATCH_SIZE_OFFLINE,
        batch_size_inference: int = DEFAULT_BATCH_SIZE_INFERENCE,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size_store = batch_size_store
        self.batch_size_inference = batch_size_inference
        self.state_dict_base = state_dict
        self.editable_keys = editable_prefixes(state_dict)
        self.halut_modules: Dict[str, int] = dict([])
        self.data_path = data_path
        self.device = device
        self.stats: Dict[str, Any] = dict([])

    def activate_halut_module(self, name: str, C: int) -> None:
        if name not in self.editable_keys:
            raise Exception(f"module {name} not in model")

        if name in self.halut_modules.keys():
            print(f"overwrite halut layer {name}")

        self.halut_modules |= dict({name: C})

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

    def store_inputs(self) -> None:
        keys_to_store = list(
            filter(
                lambda k: any(
                    not check_file_exists(self.data_path + "/" + k + e)
                    for e in (END_STORE_A, END_STORE_B)
                ),
                self.halut_modules.keys(),
            )
        )
        print("keys to store", keys_to_store)
        dict_to_add = OrderedDict(
            [
                (k + ".store_input", torch.ones(1, dtype=torch.bool))
                for k in keys_to_store
            ]
        )
        state_dict_to_store = OrderedDict(self.state_dict_base | dict_to_add)
        self.run_for_input_storage(state_dict_to_store)

    def store_all(self) -> None:
        dict_to_add = OrderedDict(
            [
                (k + ".store_input", torch.ones(1, dtype=torch.bool))
                for k in self.editable_keys
            ]
        )
        state_dict_to_store = OrderedDict(self.state_dict_base | dict_to_add)
        self.run_for_input_storage(state_dict_to_store)

    def run_for_input_storage(
        self, state_dict: "OrderedDict[str, torch.Tensor]"
    ) -> None:
        loaded_data = DataLoader(
            self.dataset,
            batch_size=self.batch_size_store,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        with torch.no_grad():
            for n_iter, (image, _) in enumerate(loaded_data):
                image = image.to(self.device)
                if n_iter > 1:
                    continue
                self.model(image)

        self.model.write_inputs_to_disk(path=self.data_path)  # type: ignore [operator]

    def run_halut_offline_training(self) -> "OrderedDict[str, torch.Tensor]":
        additional_dict: Dict[str, torch.Tensor] = dict([])
        for k, C in self.halut_modules.items():
            input_a = np.load(self.data_path + "/" + k + END_STORE_A)
            input_b = np.load(self.data_path + "/" + k + END_STORE_B)
            print(f"Learn Layer {k}: a: {input_a.shape}, b: {input_b.shape}")
            self.stats[k + ".input_a_shape"] = input_a.shape
            self.stats[k + ".input_b_shape"] = input_b.shape
            start = timer()
            store_array = hm.learn_halut_offline(input_a, input_b, C)
            end = timer()
            self.stats[k + ".halut_learning_time"] = end - start
            additional_dict = additional_dict | dict(
                {
                    k + ".halut_active": torch.ones(1, dtype=torch.bool),
                    k
                    + ".hash_buckets": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.HASH_TABLES]
                    ),
                    k
                    + ".lut": torch.from_numpy(store_array[hm.HalutOfflineStorage.LUT]),
                    k
                    + ".halut_config": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.CONFIG]
                    ),
                }
            )
        self.stats["halut_layers"] = json.dumps(self.halut_modules)
        return OrderedDict(self.state_dict_base | additional_dict)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats

    def run_inference(self) -> float:
        self.store_inputs()
        state_dict_with_halut = self.run_halut_offline_training()
        self.model.load_state_dict(state_dict_with_halut, strict=False)
        loaded_data = DataLoader(
            self.dataset,
            batch_size=self.batch_size_inference,
            num_workers=16,
            drop_last=False,
            pin_memory=True,
        )
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
