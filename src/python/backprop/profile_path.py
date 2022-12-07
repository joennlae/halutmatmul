import torch
import numpy as np

from halutmatmul.modules import (
    halut_matmul_forward,
    create_bit_matrix,
    create_selection_matrix,
)

# cuda:2, cpu
torch_device = torch.device("cuda:2")

C = 512
M = 512
K = 16
input = (
    torch.rand([64 * 512, 9 * C], requires_grad=True).to(torch.float16).to(torch_device)
)
T = torch.rand([C * 15], requires_grad=True).to(torch.float16).to(torch_device)
idx = np.arange(9 * C)
np.random.shuffle(idx)
dims = torch.Tensor(np.arange(9 * C)[idx][: 4 * C]).to(torch.int64).to(torch_device)
L = torch.rand([M, C, K], requires_grad=True).to(torch.float16).to(torch_device)
S = create_selection_matrix(C=C, dtype=torch.float16).to(torch_device)
B = create_bit_matrix(C=C, dtype=torch.float16).to(torch_device)

print(input.shape, T.shape, dims.shape, L.shape, S.shape, B.shape)

times = []
memory_allocated = []
memory_reserved = []
times_backward = []
for i in range(500):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()  # type: ignore

    out = halut_matmul_forward(input, T, L, S, B, C, K, dims, None)

    end_event.record()  # type: ignore
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(out.shape)
    del out
    del start_event
    del end_event
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    out = halut_matmul_forward(input, T, L, S, B, C, K, dims, None)
    loss = out.sum()
    start_event.record()  # type: ignore
    loss.backward()
    end_event.record()  # type: ignore
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms_backward = start_event.elapsed_time(end_event)
    print("time", elapsed_time_ms, elapsed_time_ms_backward)
    times.append(elapsed_time_ms)
    times_backward.append(elapsed_time_ms_backward)
    memory_allocated.append(torch.cuda.max_memory_allocated(torch_device))
    print(torch.cuda.max_memory_allocated(torch_device))
    memory_reserved.append(torch.cuda.max_memory_reserved(torch_device))
    print(torch.cuda.max_memory_reserved(torch_device))

    del out

print("mean time", np.mean(times[100:]))
print("mean time backward", np.mean(times_backward[100:]))
print("memory reserved", memory_reserved[0] / 1000 / 1000, "MiB")
print("memory allocated", memory_allocated[0] / 1000 / 1000, "MiB")

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
