import torch
import numpy as np

from halutmatmul.modules import (
    halut_matmul_forward,
    create_bit_matrix,
    create_selection_matrix,
)

# cuda:2, cpu
torch_device = torch.device("cuda:2")

C = 256
M = 512
K = 16
input = torch.rand([64 * 512, 9 * C]).to(torch.float16).to(torch_device)
T = torch.rand([C * 15]).to(torch.float16).to(torch_device)
idx = np.arange(9 * C)
np.random.shuffle(idx)
dims = torch.Tensor(np.arange(9 * C)[idx][: 4 * C]).to(torch.int64).to(torch_device)
L = torch.rand([M, C, K]).to(torch.float16).to(torch_device)
S = create_selection_matrix(C=C, dtype=torch.float16).to(torch_device)
B = create_bit_matrix(C=C, dtype=torch.float16).to(torch_device)

print(input.shape, T.shape, dims.shape, L.shape, S.shape, B.shape)

times = []
memory_allocated = []
memory_reserved = []
for i in range(500):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()  # type: ignore

    out = halut_matmul_forward(input, T, L, dims, S, B, C, K)

    end_event.record()  # type: ignore
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(out.shape)
    print("time", elapsed_time_ms)
    times.append(elapsed_time_ms)
    memory_allocated.append(torch.cuda.max_memory_allocated(torch_device))
    print(torch.cuda.max_memory_allocated(torch_device))
    memory_reserved.append(torch.cuda.max_memory_reserved(torch_device))
    print(torch.cuda.max_memory_reserved(torch_device))

    del out

print("mean time", np.mean(times[100:]))
print("memory reserved", memory_reserved[0] / 1000 / 1000, "MiB")
print("memory allocated", memory_allocated[0] / 1000 / 1000, "MiB")

# current numbers on my machine
# mean time 0.4418311085965898
# memory 10210.524672 MiB

# mean time 75.02608726501465
# memory reserved 10708.058112 MiB
# memory allocated 10210.524672 MiB
