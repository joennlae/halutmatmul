import math
import numpy as np
import torch
import torch.utils.data
from halutmatmul.modules import (
    halut_matmul_forward,
    create_bit_matrix,
    create_selection_matrix,
    create_A_matrix_from_dims,
)
import halutmatmul.halutmatmul as hm

dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "/usr/scratch2/vilan1/janniss/halut/resnet18-cifar10-halut-A-2"

test_layer_A = "layer1.0.conv1_32_391_A.npy"
test_layer_B = "layer1.0.conv1_32_391_B.npy"

I = torch.from_numpy(np.load(data_path + "/" + test_layer_A)).to(dtype).to(device)
W = torch.from_numpy(np.load(data_path + "/" + test_layer_B)).to(dtype).to(device)

index_random = torch.randperm(I.shape[0])
I = I[index_random]

rank = np.linalg.matrix_rank(I.cpu().numpy())
print(f"rank {rank} of {I.shape[1]}")

N = I.shape[0]
# train_input = I[: N - (N // 10)]
# val_input = I[N - (N // 10) :]
train_input = I
val_input = I
C = 16
K = 16
M = W.shape[1]
N = I.shape[0]
D = I.shape[1]

S = create_selection_matrix(C, K, dtype=dtype).to(device)
B = create_bit_matrix(C, K, dtype=dtype).to(device)
T = torch.randn((C * 15), dtype=dtype).to(device)
T = torch.nn.init.uniform_(T, a=0, b=1)
L = torch.randn((M, C, K), dtype=dtype).to(device)

L = torch.nn.init.kaiming_uniform_(L, a=math.sqrt(5))

depth = int(math.sqrt(K))
A = torch.randn((C, D // C, depth), dtype=dtype).to(device)
A = torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))

for c in range(C):
    pca = torch.pca_lowrank(I[:, c * (D // C) : (c + 1) * (D // C)], niter=10)
    print("shapes", pca[0].shape, pca[1].shape, pca[2].shape)
    A[c] = pca[2][:, :depth]
    print("A", A[c])


class HalutMatmul(torch.nn.Module):
    def __init__(self, C, K, S, B, T, L, A):
        super().__init__()
        self.C = C
        self.K = K
        self.A = torch.nn.Parameter(A, requires_grad=False)
        self.S = torch.nn.Parameter(S, requires_grad=False)
        self.B = torch.nn.Parameter(B, requires_grad=False)
        self.T = torch.nn.Parameter(T, requires_grad=True)
        self.L = torch.nn.Parameter(L, requires_grad=True)

    def forward(self, I):
        return halut_matmul_forward(
            I,
            self.T,
            self.L,
            self.S,
            self.B,
            self.C,
            self.K,
            None,
            self.A,
        )


# train
batch_size = 1024 * 4
epochs = 100
model = HalutMatmul(C, K, S, B, T, L, A)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.HuberLoss(reduction="mean")
criterion = torch.nn.MSELoss(reduction="mean")

train_dataset = torch.utils.data.TensorDataset(train_input)
data_loader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataset = torch.utils.data.TensorDataset(val_input)
data_loader_val = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=epochs
)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()

    optimizer.zero_grad()
    total_loss = 0
    for _, (input) in enumerate(data_loader):
        input = input[0].to(device)
        output = model(input)
        target = input.mm(W)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(
        f"({epoch}) Average loss train: {total_loss / len(data_loader)}, "
        f"lr: {optimizer.param_groups[0]['lr']}"
    )


def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for _, (input) in enumerate(data_loader):
            input = input[0].to(device)
            output = model(input)
            target = input.mm(W)
            loss = criterion(output, target)
            total_loss += loss.item()
    print("Average loss evaluate: {}".format(total_loss / len(data_loader)))


for i in range(epochs):
    train_one_epoch(model, criterion, optimizer, data_loader_train, device, i)
    lr_scheduler.step()
    evaluate(model, criterion, data_loader_val, device)

print("Final results", model.A, model.T, model.L)
print("Final max", torch.max(model.A), torch.max(model.T), torch.max(model.L))

output = model(val_input)
target = val_input.mm(W)

print("Final loss", criterion(output, target))
print("Final max", torch.max(output), torch.max(target))

print("Compare", output[0], target[0])
print("Final MSE", torch.nn.MSELoss(reduction="mean")(output, target))
print("Final MAE", torch.nn.L1Loss(reduction="mean")(output, target))
print("Final Huber", torch.nn.HuberLoss(reduction="mean")(output, target))

halut_learned = hm.learn_halut_offline(
    I.detach().cpu().numpy(), W.detach().cpu().numpy(), C, K
)
halut_lut = (
    torch.from_numpy(halut_learned[hm.HalutOfflineStorage.LUT]).to(dtype).to(device)
)
halut_T = (
    torch.from_numpy(halut_learned[hm.HalutOfflineStorage.THRESHOLDS])
    .to(dtype)
    .to(device)
)
halut_dims = (
    torch.from_numpy(halut_learned[hm.HalutOfflineStorage.DIMS])
    .to(torch.int32)
    .to(device)
)
halut_A = create_A_matrix_from_dims(halut_dims, D, C, K, dtype=dtype).to(device)

halut_learned_model = HalutMatmul(C, K, S, B, halut_T, halut_lut, halut_A)
halut_learned_output = halut_learned_model(val_input)
print("Learned loss", criterion(halut_learned_output, target))
print("Learned max", torch.max(halut_learned_output), torch.max(target))
print("Learned MSE", torch.nn.MSELoss(reduction="mean")(halut_learned_output, target))
print("Learned MAE", torch.nn.L1Loss(reduction="mean")(halut_learned_output, target))
print(
    "Learned Huber", torch.nn.HuberLoss(reduction="mean")(halut_learned_output, target)
)

print("Compare", halut_learned_output[0], target[0])
