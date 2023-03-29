import torch
import numpy as np
from models.resnet20 import resnet20
from halutmatmul.modules import HalutConv2d, HalutLinear


N = 128
D = 64
M = 32
C = 16
d = D // C
K = 16

input = torch.rand(N, D)
prototypes = torch.rand(C, K, d)

input_reshaped = input.view(N, C, d)

mse_base = torch.zeros(N, C, K)
for n in range(N):
    for c in range(C):
        for k in range(K):
            mse_base[n, c, k] = torch.mean(
                (input_reshaped[n, c, :] - prototypes[c, k, :]) ** 2
            )

# mse
mse = torch.mean(torch.square(input_reshaped.unsqueeze(2) - prototypes), dim=3)
print(mse.shape)

print("mse_base == mse: ", torch.all(mse_base == mse))

print("mse_base: ", mse_base[0, 0, :])

softmaxed = torch.softmax(-mse, dim=2)

print("softmaxed: ", softmaxed[0, 0, :], torch.sum(softmaxed[0, 0, :]))

model = resnet20()

checkpoint = torch.load(
    "/usr/scratch2/vilan1/janniss/model_checkpoints/cifar10-halut-resnet20-adam-2/checkpoint.pth",
    map_location=torch.device("cpu"),
)
model.load_state_dict(checkpoint["model"])
print(model.state_dict().keys())
# pylint: disable=unsubscriptable-object
proto = model.state_dict()["layer1.2.conv1.P"]


def loop_module(module, prefix=""):
    for name, p in module.named_parameters(recurse=False):
        if isinstance(module, (HalutConv2d, HalutLinear)):
            if name == "temperature":
                print(f"{prefix}.{name}: {p}")

    for child_name, child_module in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
        loop_module(child_module, prefix=child_prefix)


loop_module(model)
print(proto[0, :, :])
print(model)
