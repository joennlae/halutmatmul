import math
from functools import partial

import numpy as np
import torch
import torch.utils.data
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from halutmatmul.modules import (
    create_A_matrix_from_dims,
    create_bit_matrix,
    create_selection_matrix,
    halut_matmul_forward,
)
import halutmatmul.halutmatmul as hm

dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "/usr/scratch2/vilan1/janniss/halut/resnet18-cifar10-halut-A-2"

test_layer_A = "layer1.0.conv1_32_391_A.npy"
test_layer_B = "layer1.0.conv1_32_391_B.npy"

# test_layer_A = "layer4.1.conv2_32_391_A.npy"
# test_layer_B = "layer4.1.conv2_32_391_B.npy"

I = torch.from_numpy(np.load(data_path + "/" + test_layer_A)).to(dtype).to(device)
W = torch.from_numpy(np.load(data_path + "/" + test_layer_B)).to(dtype).to(device)

index_random = torch.randperm(I.shape[0])
I = I[index_random]

# rank = np.linalg.matrix_rank(I.cpu().numpy())
# print(f"rank {rank} of {I.shape[1]}")

N = I.shape[0]
# train_input = I[: N - (N // 10)]
# val_input = I[N - (N // 10) :]
train_input = I
val_input = I

C = 64
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
# A = torch.randn((C, D // C, depth), dtype=dtype).to(device)
# A = torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
A = torch.empty((C, D // C, depth), dtype=dtype).to(device)

A_inv = torch.zeros((C, depth, D // C), dtype=dtype).to(device)
for c in range(C):
    pca = torch.pca_lowrank(I[:, c * (D // C) : (c + 1) * (D // C)], niter=10)
    print("shapes", pca[0].shape, pca[1].shape, pca[2].shape)
    A[c] = pca[2][:, :depth]
    print("A", A[c])
    how_close = torch.matmul(pca[2][:, :depth], pca[2][:, :depth].t())
    print("how_close", how_close)
    print("how_close", torch.max(torch.abs(how_close - torch.eye(D // C).to(device))))
    print("pca[2]", pca[2])
    how_close_3 = torch.matmul(pca[2], pca[2].t())
    print("how_close_3", how_close_3)
    print(
        "how_close_3", torch.max(torch.abs(how_close_3 - torch.eye(D // C).to(device)))
    )
    A_inv[c] = torch.pinverse(pca[2][:, :depth])
    print("how_close", torch.matmul(A[c], A_inv[c]) - torch.eye(D // C).to(device))


I_reshaped = I.T.reshape((C, -1, I.shape[0])).transpose(1, 2)
input_tilde = (
    torch.bmm(I_reshaped, A).transpose(1, 2).reshape((C * int(math.sqrt(K)), -1)).T
)
print("input_tilde", input_tilde.shape)
A_inv_used = A.transpose(1, 2)
# A_inv_used = A_inv
weight_tilde = torch.bmm(A_inv_used, W.reshape((C, -1, M))).reshape(
    (C * int(math.sqrt(K)), -1)
)
print("weight_tilde", weight_tilde.shape)
output_tilde = torch.matmul(input_tilde, weight_tilde)
print("output_tilde", output_tilde.shape)

result = torch.matmul(I, W)
print("result", result.shape)
print("result MAE", torch.nn.L1Loss()(result, output_tilde))
print("result MSE", torch.nn.MSELoss()(result, output_tilde))
print("result RMSE", torch.sqrt(torch.nn.MSELoss()(result, output_tilde)))
print("result MAPE", torch.mean(torch.abs(result - output_tilde) / result))
print("result Huber", torch.nn.HuberLoss()(result, output_tilde))

halut_learned_init = hm.learn_halut_offline(
    input_tilde.detach().cpu().numpy(), weight_tilde.detach().cpu().numpy(), C, K
)
halut_lut_init = (
    torch.from_numpy(halut_learned_init[hm.HalutOfflineStorage.LUT])
    .to(dtype)
    .to(device)
)
halut_T_init = (
    torch.from_numpy(halut_learned_init[hm.HalutOfflineStorage.THRESHOLDS])
    .to(dtype)
    .to(device)
)
L = halut_lut_init
T = halut_T_init


class HalutMatmul(torch.nn.Module):
    def __init__(self, C, K, S, B, T, L, A, a_grad=True, t_grad=True, l_grad=True):
        super().__init__()
        self.C = C
        self.K = K
        self.A = torch.nn.Parameter(A, requires_grad=a_grad)
        self.S = torch.nn.Parameter(S, requires_grad=False)
        self.B = torch.nn.Parameter(B, requires_grad=False)
        self.T = torch.nn.Parameter(T, requires_grad=t_grad)
        self.L = torch.nn.Parameter(L, requires_grad=l_grad)

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
batch_size = 256
epochs = 100
model = HalutMatmul(C, K, S, B, T, L, A)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
criterion = torch.nn.HuberLoss(reduction="mean")
criterion = torch.nn.L1Loss(reduction="mean")
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
    return total_loss / len(data_loader)


def train(
    model,
    criterion,
    optimizer,
    data_loader_train,
    data_loader_val,
    device,
    epochs,
    lr_scheduler,
    is_plateu=False,
):
    for i in range(epochs):
        train_one_epoch(model, criterion, optimizer, data_loader_train, device, i)
        if is_plateu:
            lr_scheduler.step(evaluate(model, criterion, data_loader_val, device))
        else:
            lr_scheduler.step()
        evaluate(model, criterion, data_loader_val, device)


# train(model, criterion,
# optimizer, data_loader_train, data_loader_val, device, epochs, lr_scheduler)

configs = {
    "lr": tune.grid_search([0.0005]), # tune.uniform(0.0001, 0.001)
    "batch_size": tune.grid_search([2048]),
    "epochs": tune.grid_search([100]),
    "optimizer": tune.grid_search(["adam"]), # sgd
    "criterion": tune.grid_search(["mse"]), # mae, huber
    "lr_scheduler": tune.grid_search(["cosine"]), # plateau, step
    "learn_tensors": tune.grid_search(
        [
            [True, False, True],
        ]
    ),
}


def train_with_tune(config):
    I = torch.from_numpy(np.load(data_path + "/" + test_layer_A)).to(dtype).to(device)
    train_input = I
    val_input = I
    model = HalutMatmul(C, K, S, B, T, L, A, *config["learn_tensors"])
    batch_size = config["batch_size"]
    train_dataset = torch.utils.data.TensorDataset(train_input)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataset = torch.utils.data.TensorDataset(val_input)
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    epochs = config["epochs"]
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    if config["lr_scheduler"] == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.1, patience=10
        )
    elif config["lr_scheduler"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=epochs // 4, gamma=0.1
        )
    elif config["lr_scheduler"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )

    if config["criterion"] == "mse":
        criterion = torch.nn.MSELoss(reduction="mean")
    elif config["criterion"] == "l1":
        criterion = torch.nn.L1Loss(reduction="mean")
    elif config["criterion"] == "huber":
        criterion = torch.nn.HuberLoss(reduction="mean")

    train(
        model,
        criterion,
        optimizer,
        data_loader_train,
        data_loader_val,
        device,
        epochs,
        lr_scheduler,
        is_plateu=config["lr_scheduler"] == "plateau",
    )

    mse = evaluate(model, torch.nn.MSELoss(reduction="mean"), data_loader_val, device)
    mae = evaluate(model, torch.nn.L1Loss(reduction="mean"), data_loader_val, device)
    huber = evaluate(
        model, torch.nn.HuberLoss(reduction="mean"), data_loader_val, device
    )
    tune.report(mse=mse, mae=mae, huber=huber)


reporter = CLIReporter(
    parameter_columns=[
        "lr",
        "batch_size",
        "epochs",
        "optimizer",
        "criterion",
        "lr_scheduler",
        "learn_tensors",
    ],
    metric_columns=["mse", "mae", "huber"],
)
scheduler = ASHAScheduler(
    max_t=200,
    grace_period=10,
    reduction_factor=2,
    brackets=1,
    metric="mse",
    mode="min",
)

analysis = tune.run(
    partial(train_with_tune),
    resources_per_trial={"gpu": 0.3},
    config=configs,
    scheduler=scheduler,
    num_samples=3,
    progress_reporter=reporter,
)

best_trial = analysis.get_best_trial("mse", "min", "last")
print("Best trial config: {}".format(best_trial.config))  # type: ignore
print("Best trial final validation loss: {}".format(best_trial.last_result["mse"]))  # type: ignore
print("Best trial final validation MAE: {}".format(best_trial.last_result["mae"]))  # type: ignore
print(
    "Best trial final validation Huber"
    ": {}".format(best_trial.last_result["huber"])  # type: ignore
)
print("Best trial directory: {}".format(best_trial.logdir))  # type: ignore


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

for t in [4, 6, 7, 8, 10, 16, 32]:
    # convert to empty type as empty_strided is not supported
    scale = torch.max(I) - torch.min(I)
    scale = scale / (2**t - 1)
    I_quant = torch.fake_quantize_per_tensor_affine(
        I, scale, 0, quant_min=-(2 ** (t - 1)), quant_max=2 ** (t - 1) - 1
    )
    W_quant = torch.fake_quantize_per_tensor_affine(
        W, scale, 0, quant_min=-(2 ** (t - 1)), quant_max=2 ** (t - 1) - 1
    )

    result = I_quant.mm(W_quant)
    result_scale_back = result
    print(
        f"MSE INT{t}",
        torch.nn.MSELoss(reduction="mean")(result_scale_back, target),
    )
    print(
        f"MAE INT{t}",
        torch.nn.L1Loss(reduction="mean")(result_scale_back, target),
    )
    print(
        f"Huber INT{t}",
        torch.nn.HuberLoss(reduction="mean")(result_scale_back, target),
    )
