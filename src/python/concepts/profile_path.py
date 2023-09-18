from collections import OrderedDict
import gc
import time
import torch
import numpy as np

from halutmatmul.modules import (
    halut_matmul_forward,
    create_bit_matrix,
    create_selection_matrix,
    HalutConv2d,
)
import halutmatmul.halutmatmul as hm

# cuda:2, cpu
torch_device = torch.device("cpu")

C = 512
M = 512
K = 16
batch_size = 64
out_channels = 512
in_channels = 512
groups = 1
kernel_size = 3
stride = 1
image_x_y = 16
a = 1.0
b = 0.0

weights = torch.rand((out_channels, in_channels // groups, kernel_size, kernel_size))
bias_weights = torch.rand((out_channels))
input_learn = (torch.rand((batch_size * 2, in_channels, image_x_y, image_x_y)) + b) * a

input_test = (torch.rand((batch_size, in_channels, image_x_y, image_x_y)) + b) * a
torch_module = torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size=kernel_size,
    stride=stride,
    groups=groups,
)

halutmatmul_module = HalutConv2d(
    in_channels,
    out_channels,
    kernel_size=kernel_size,
    stride=stride,
    groups=groups,
    split_factor=4,
)
input_a = halutmatmul_module.transform_input(input_learn)
input_b = halutmatmul_module.transform_weight(weights)

store_array = np.load("store_array.npy", allow_pickle=True)
weights.to(torch_device)

state_dict = OrderedDict({"weight": weights})
torch_module.load_state_dict(state_dict, strict=False)
if torch_device.type == "cuda":
    torch_module.half().to(torch_device)
    halutmatmul_module.half().to(torch_device)
state_dict = OrderedDict(
    state_dict
    | OrderedDict(
        {
            "halut_active": torch.ones(1, dtype=torch.bool).to(torch_device),
            "lut": torch.from_numpy(store_array[hm.HalutOfflineStorage.LUT])
            .to(torch_device)
            .half(),
            "thresholds": torch.from_numpy(
                store_array[hm.HalutOfflineStorage.THRESHOLDS]
            )
            .to(torch_device)
            .half(),
            "dims": torch.from_numpy(store_array[hm.HalutOfflineStorage.DIMS]).to(
                torch_device
            ),
        }
    )
)
halutmatmul_module.load_state_dict(state_dict, strict=False)


# pylint: disable=dangerous-default-value
def get_tensors(only_cuda=False, omit_objs=[]):
    add_all_tensors = not only_cuda
    # To avoid counting the same tensor twice, create a dictionary of tensors,
    # each one identified by its id (the in memory address).
    tensors = {}

    # omit_obj_ids = [id(obj) for obj in omit_objs]

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, "data") and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return

        if (only_cuda and tensor.is_cuda) or add_all_tensors:
            tensors[id(tensor)] = tensor

    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, "saved_tensors") and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except Exception:
            pass
            # print("Exception: ", ex)
            # logger.debug(f"Exception: {str(ex)}")
    return tensors.values()  # return a list of detected tensors


iterations = 100
times = np.zeros((2, iterations))
memory_allocated = np.zeros((2, iterations))
memory_reserved = np.zeros((2, iterations))
times_backward = np.zeros((2, iterations))
# input_test = input_test.half().to(torch_device)
if torch.cuda.is_available():
    input_test = input_test.half().to(torch_device)
    for idx, module in enumerate([torch_module, halutmatmul_module]):
        for i in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()  # type: ignore
            out = module(input_test)
            end_event.record()  # type: ignore
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            del out
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            out = module(input_test)
            loss = out.sum()
            start_event.record()  # type: ignore
            loss.backward()
            end_event.record()  # type: ignore
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms_backward = start_event.elapsed_time(end_event)
            # print("time", elapsed_time_ms, elapsed_time_ms_backward)
            times[idx, i] = elapsed_time_ms
            times_backward[idx, i] = elapsed_time_ms_backward
            memory_allocated[idx, i] = torch.cuda.max_memory_allocated(torch_device)
            # print(torch.cuda.max_memory_allocated(torch_device))
            memory_reserved[idx, i] = torch.cuda.max_memory_reserved(torch_device)
            # print(torch.cuda.max_memory_reserved(torch_device))
            if i == iterations - 1:
                tensors = get_tensors()
                for tensor in tensors:
                    print(
                        tensor.shape,
                        tensor.dtype,
                        tensor.device,
                        tensor.is_cuda,
                        tensor.numel() * tensor.element_size() / 1024 / 1024,
                    )
                del tensors
                del out

if not torch.cuda.is_available():
    for idx, module in enumerate([torch_module, halutmatmul_module]):
        for i in range(iterations):
            start_time = time.time()
            out = module(input_test)
            elapsed_time_ms = time.time() - start_time
            del out
            out = module(input_test)
            loss = out.sum()
            start_time = time.time()
            loss.backward()
            elapsed_time_ms_backward = time.time() - start_time
            print("time", elapsed_time_ms, elapsed_time_ms_backward)
            times[idx, i] = elapsed_time_ms
            times_backward[idx, i] = elapsed_time_ms_backward
            if i == iterations - 1:
                tensors = get_tensors()
                for tensor in tensors:
                    print(
                        tensor.shape,
                        tensor.dtype,
                        tensor.device,
                        tensor.is_cuda,
                        tensor.numel() * tensor.element_size() / 1024 / 1024,
                    )
                del tensors
                del out


print("mean time", np.mean(times[:, 10:], axis=1))
print("mean time backward", np.mean(times_backward[:, 10:], axis=1))
print("memory reserved", memory_reserved[:, 0] / 1000 / 1000, "MiB")
print("memory allocated", memory_allocated[:, 0] / 1000 / 1000, "MiB")

# current numbers on my machine
# mean time 0.4418311085965898
# memory 10210.524672 MiB

# baseline
# mean time 75.02608726501465
# memory reserved 10708.058112 MiB
# memory allocated 10210.524672 MiB

# remove unsqueeze
# mean time 74.91948002815246
# memory reserved 10708.058112 MiB
# memory allocated 10210.524672 MiB

# without the hard forward path
# mean time 66.74334077715874
# memory reserved 10171.187199999998 mib
# memory allocated 9606.544896 MiB

# without einsum
# mean time 38.73765052258968
# memory reserved 2118.12352 MiB
# memory allocated 1855.471104 MiB

# running with for loop
# mean time 2242.158778076172
# memory reserved 2386.558976 MiB
# memory allocated 2157.4609920000003 MiB

# splitting it into two parts (einsum)
# mean time 77.46310869216919
# memory reserved 6413.090816 MiB
# memory allocated 5932.334592 MiB

# new measurements with backward and requires_grad (without loop)
# mean time 1.0296817600727082
# mean time backward 346.7793290710449
# memory reserved 10708.058112 MiB
# memory allocated 10462.182912 MiB

# new measurements with backward and requires_grad (with loop i = 2)
# mean time 1.3819680002331733
# mean time backward 313.10253044128416
# memory reserved 6413.090816 MiB
# memory allocated 6183.992832000001 MiB

# measurements with backward and requires_grad (with loop i = 4)
# mean time 1.725952000916004
# mean time backward 335.18721862792967
# memory reserved 4265.607168 MiB
# memory allocated 4028.644864 MiB

# mean time [  0.409632   126.24782223]
# mean time backward [0.87060764 1.43258417]
# memory reserved [ 322.961408 8547.991552] MiB
# memory allocated [ 292.353536 8190.75072 ] MiB
