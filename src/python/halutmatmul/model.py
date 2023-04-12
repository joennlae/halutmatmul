# pylint: disable=C0209
import glob, re, json
from functools import reduce
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union
from timeit import default_timer as timer
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from models.resnet import END_STORE_A, END_STORE_B
from models.helper import (
    RUN_ALL_SUBSAMPLING,
    evaluate_halut_imagenet,
    get_and_print_layers_to_use_halut,
    evaluate_distributed,
)
import halutmatmul.halutmatmul as hm
from halutmatmul.learn import learn_halut_multi_core_dict
from halutmatmul.modules import ErrorTuple, HalutConv2d, HalutLinear

T_co = TypeVar("T_co", covariant=True)

eval_func_type = Callable[
    [
        Any,
        torch.nn.Module,
        torch.device,
        bool,
        str,
        Optional[dict[str, int]],
        int,
        int,
        float,
    ],
    tuple[float, float],
]

DEFAULT_BATCH_SIZE_OFFLINE = 512
DEFAULT_BATCH_SIZE_INFERENCE = 128
DATA_PATH = "/scratch2/janniss/resnet_input_data"


def get_module_by_name(
    module: Union[torch.Tensor, torch.nn.Module], access_string: str
):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def editable_prefixes(model: torch.nn.Module) -> list[str]:
    keys = get_and_print_layers_to_use_halut(model)
    return keys


def check_file_exists(path: str) -> bool:
    file_path = Path(path)
    return file_path.is_file()


def check_file_exists_and_return_path(
    base_path: str,
    layers_name: str,
    _type: Union[Literal["input"], Literal["learned"]],
    C: int = 16,
    K: int = 16,
) -> list[str]:
    files = glob.glob(base_path + "/*.npy")
    files_res = []
    if _type == "input":
        regex_a = rf"{layers_name}{END_STORE_A}"
        regex_b = rf"{layers_name}{END_STORE_B}"
        pattern_a = re.compile(regex_a)
        pattern_b = re.compile(regex_b)
        files_a = [x for x in files if pattern_a.search(x)]
        files_b = [x for x in files if pattern_b.search(x)]
        files_res = files_a + files_b
        assert len(files_res) == 0 or len(files_res) == 2
    elif _type == "learned":
        regex = rf"{layers_name}_{C}_{K}.npy"
        print("pattern", regex)
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
        dataset_train: Optional[Dataset[T_co]] = None,
        data_path: str = DATA_PATH,
        batch_size_store: int = DEFAULT_BATCH_SIZE_OFFLINE,
        batch_size_inference: int = DEFAULT_BATCH_SIZE_INFERENCE,
        learned_path: str = DATA_PATH + "/learned/",
        device: torch.device = torch.device("cpu"),
        workers_offline_training: int = 1,
        report_error: bool = False,
        num_workers: int = 8,
        eval_function: eval_func_type = evaluate_halut_imagenet,
        distributed: bool = False,
        device_id: int = 0,
        kmeans_options: dict[str, Any] = dict([]),
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.dataset_train = dataset_train
        self.batch_size_store = batch_size_store
        self.batch_size_inference = batch_size_inference
        self.state_dict_base = state_dict
        self.editable_keys = editable_prefixes(self.model)
        self.learned_path = learned_path
        self.halut_modules: Dict[str, list] = dict([])
        self.data_path = data_path
        self.device = device
        self.stats: Dict[str, Any] = dict([])
        self.workers_offline_training = workers_offline_training
        self.report_error = report_error
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.distributed = distributed
        self.device_id = device_id
        self.kmeans_options = kmeans_options

    def activate_halut_module(
        self,
        name: str,
        C: int,
        K: int = 16,
        loop_order: Literal["im2col", "kn2col"] = "im2col",
        use_prototypes: bool = False,
    ) -> None:
        if name not in self.editable_keys:
            raise Exception(f"module {name} not in model")

        if name in self.halut_modules.keys():
            print(f"overwrite halut layer {name}")

        module_ref = get_module_by_name(self.model, name)
        if isinstance(module_ref, HalutConv2d):
            # Conv2d layer
            module_ref.loop_order = loop_order
            module_ref.use_prototypes = use_prototypes
            self.halut_modules |= dict({name: [C, K, loop_order]})
        elif isinstance(module_ref, HalutLinear):
            # Linear layer
            module_ref.use_prototypes = use_prototypes
            self.halut_modules |= dict({name: [C, K]})
        else:
            raise Exception(
                f"module {name} not a HALUT conv or linear layer {type(module_ref)}"
            )

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
        dict_to_add = OrderedDict(
            [
                (k + ".store_input", torch.ones(1, dtype=torch.bool))
                for k in dict_to_store.keys()
            ]
        )
        state_dict_to_store = OrderedDict(self.state_dict_base | dict_to_add)
        if dict_to_store:
            if self.distributed:
                self.run_for_input_storage_distributed(state_dict_to_store)
            else:
                self.run_for_input_storage(
                    state_dict_to_store,
                    additional_dict=dict_to_store,
                )

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
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        additional_dict: Optional[dict[str, int]] = None,
    ) -> None:
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.eval_function(
            self.dataset if self.dataset_train is None else self.dataset_train,
            self.model,
            self.device,
            True,
            self.data_path,
            additional_dict,
            self.batch_size_store,
            self.num_workers,
            0.0,
        )

    def run_for_input_storage_distributed(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
    ) -> None:
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.device_id], find_unused_parameters=True
        )

        evaluate_distributed(
            model=model,
            data_loader=self.dataset_train,  # type: ignore
            device=self.device,
            is_store=True,
            data_path=self.data_path,
            device_id=self.device_id,
        )

    def run_halut_offline_training(self, codebook: int = -1) -> None:
        dict_to_store: dict[str, int] = dict([])
        dict_to_learn: dict[str, list] = dict([])
        for k, args in self.halut_modules.items():
            if len(self.state_dict_base[k + ".lut"].shape) > 1 and codebook == -1:
                continue
            learned_files = check_file_exists_and_return_path(
                self.learned_path,
                k,
                "learned",
                args[hm.HalutModuleConfig.C],
                args[hm.HalutModuleConfig.K],
            )
            if len(learned_files) == 1 and codebook == -1:
                continue
            dict_to_learn[k] = [
                args[hm.HalutModuleConfig.C],
                args[hm.HalutModuleConfig.K],
            ]
            if len(args) > 2:
                # Conv2d layer
                dict_to_learn[k].append(args[hm.HalutModuleConfig.LOOP_ORDER])

            paths = check_file_exists_and_return_path(self.data_path, k, "input")
            print(f"paths {paths}")
            if len(paths) != 2:
                dict_to_store[k] = 1
                dict_to_store[k] = RUN_ALL_SUBSAMPLING
                # just needs to be bigger than in input_images / batch_size
                # used for subsampling
        # pylint: disable=consider-iterating-dictionary, consider-using-dict-items
        for name in dict_to_learn.keys():
            module = get_module_by_name(self.model, name)
            if isinstance(module, HalutConv2d):
                assert dict_to_learn[name][2] == module.loop_order
                dict_to_learn[name].append(module.kernel_size)
                dict_to_learn[name].append(module.stride)
                dict_to_learn[name].append(module.padding)

        self.store_inputs(dict_to_store)
        learn_halut_multi_core_dict(
            dict_to_learn,
            data_path=self.data_path,
            store_path=self.learned_path,
            kmeans_options=self.kmeans_options,
            codebook=codebook,
        )

    def prepare_state_dict(
        self, codebook: int = -1
    ) -> "OrderedDict[str, torch.Tensor]":
        additional_dict: Dict[str, torch.Tensor] = dict([])
        for k, args in self.halut_modules.items():
            # if layer is already learned, skip
            if len(self.state_dict_base[k + ".lut"].shape) != 1 and codebook == -1:
                continue
            learned_files = check_file_exists_and_return_path(
                self.learned_path,
                k,
                "learned",
                C=args[hm.HalutModuleConfig.C],
                K=args[hm.HalutModuleConfig.K],
            )
            print("learned files", learned_files)
            if len(learned_files) == 1:
                store_array = np.load(learned_files[0], allow_pickle=True)
            else:
                raise Exception("learned file not found!")
            additional_dict = additional_dict | dict(
                {
                    k + ".halut_active": torch.ones(1, dtype=torch.bool),
                    k
                    + ".lut": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.LUT].astype(np.float32)
                    )
                    if len(store_array[hm.HalutOfflineStorage.SIMPLE_LUT].shape) == 1
                    else torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.SIMPLE_LUT].astype(
                            np.float32
                        )
                    ),
                    k + ".store_input": torch.zeros(1, dtype=torch.bool),
                    k + ".report_error": torch.ones(1, dtype=torch.bool)
                    if self.report_error
                    else torch.zeros(1, dtype=torch.bool),
                    k
                    + ".thresholds": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.THRESHOLDS].astype(
                            np.float32
                        )
                    ),
                    k
                    + ".dims": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.DIMS].astype(np.int32)
                    ),
                    k
                    + ".P": torch.from_numpy(
                        store_array[hm.HalutOfflineStorage.SIMPLE_PROTOTYPES].astype(
                            np.float32
                        )
                    ),
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

    def run_inference(self, prev_max: float = 0.0, codebook: int = -1) -> float:
        print("Start training of Halutmatmul")
        self.run_halut_offline_training(codebook=codebook)
        print("Start preparing state_dict")
        state_dict_with_halut = self.prepare_state_dict(codebook=codebook)
        print("Load state dict")
        start = timer()
        self.model.load_state_dict(state_dict_with_halut, strict=False)
        end = timer()
        print("State dict time: %.2f s" % (end - start))
        # if self.device_id == 0 or not self.distributed:
        top_1_acc, top_5_acc = self.eval_function(
            self.dataset,
            self.model,
            self.device,
            False,
            "",
            None,
            self.batch_size_inference,
            self.num_workers,
            prev_max,
        )
        self.stats["top_1_accuracy"] = top_1_acc
        self.stats["top_5_accuracy"] = top_5_acc
        return top_1_acc
