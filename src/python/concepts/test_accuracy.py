# pylint: disable=cell-var-from-loop
import math
from functools import partial

import numpy as np
import torch
import torch.utils.data
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# data to csv
import pandas as pd

from halutmatmul.modules import (
    create_bit_matrix,
    create_selection_matrix,
    halut_matmul_forward,
)
import halutmatmul.halutmatmul as hm

dtype = torch.float32

device = torch.device(
    "cpu"
)  # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_path = "/usr/scratch2/vilan1/janniss/halut/resnet9-tanh-all"

Cs = [64, 32, 16]
data = []

layers = [
    "conv2.0",
    "res1.0.0",
    "res1.1.0",
    "conv3.0",
    "conv4.0",
    "res2.0.0",
    "res2.1.0",
]
layer_factor = {
    "conv2.0": 1,
    "res1.0.0": 2,
    "res1.1.0": 2,
    "conv3.0": 2,
    "conv4.0": 4,
    "res2.0.0": 4,
    "res2.1.0": 4,
}


def getBack(var_grad_fn, all_shapes: list) -> None:
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor.shape)
                print(" - gradient:", tensor.grad.shape)
                if len(tensor.grad.shape) > 1:
                    # count zeros
                    print(
                        "LUT grad - zeros:",
                        torch.sum(tensor.grad == 0),
                        torch.numel(tensor.grad),
                    )
                elif tensor.grad.shape[0] > 1:
                    print("threshold grad", tensor.grad, torch.sum(tensor.grad == 0))
                all_shapes.append(tensor.grad.shape)
                print()
            except AttributeError:
                getBack(n[0], all_shapes)


class HalutMatmul(torch.nn.Module):
    def __init__(self, C, K, S, B, T, L, dims, t_grad=True, l_grad=True):
        super().__init__()
        self.C = C
        self.K = K
        self.S = torch.nn.Parameter(S, requires_grad=False)
        self.B = torch.nn.Parameter(B, requires_grad=False)
        self.T = torch.nn.Parameter(T, requires_grad=t_grad)
        self.L = torch.nn.Parameter(L, requires_grad=l_grad)
        self.D = torch.nn.Parameter(dims, requires_grad=False)
        self.temp = torch.nn.Parameter(torch.ones(1).to(device), requires_grad=False)

    def halut_updates(self):
        self.temp.data = torch.clamp(self.temp.data, 0.1, 1.0)
        # print("temp", self.temp.data)

    def forward(self, I):
        return halut_matmul_forward(
            I,
            self.T,
            self.L,
            self.S,
            self.B,
            self.C,
            self.K,
            self.D,
            temperature=self.temp,
        )


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()

    def halut_updates(module, prefix=""):
        if isinstance(module, (HalutMatmul)):
            module.halut_updates()
            return

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            halut_updates(child_module, prefix=child_prefix)

    optimizer.zero_grad()
    total_loss = 0

    for _, (input) in enumerate(data_loader):
        input = input[0].to(device)
        output = model(input)
        target = input.mm(W)
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        # if idx % 100 == 0:
        # print("gradient threshold", model.T.grad)
        # print("gradient lut", model.L.grad)
        # getBack(loss.grad_fn, [])
        optimizer.step()
        optimizer.zero_grad()
        halut_updates(model)

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
    evaluate(model, criterion, data_loader_val, device)
    for i in range(epochs):
        train_one_epoch(model, criterion, optimizer, data_loader_train, device, i)
        if is_plateu:
            lr_scheduler.step(evaluate(model, criterion, data_loader_val, device))
        else:
            lr_scheduler.step()
        evaluate(model, criterion, data_loader_val, device)


for c_ in Cs:
    for layer in layers:
        row = {}
        test_layer_A = f"{layer}_A.npy"
        test_layer_B = f"{layer}_B.npy"

        I = (
            torch.from_numpy(np.load(data_path + "/" + test_layer_A))
            .to(dtype)
            .to(device)
        )
        W = (
            torch.from_numpy(np.load(data_path + "/" + test_layer_B))
            .to(dtype)
            .to(device)
        )

        index_random = torch.randperm(I.shape[0])
        I = I[index_random]

        N = I.shape[0]
        # train_input = I[: N - (N // 10)]
        # val_input = I[N - (N // 10) :]
        train_input = I
        val_input = I

        # C = 64
        C = c_ * layer_factor[layer]
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

        result = torch.matmul(I, W)

        row["C"] = C
        row["K"] = K
        row["M"] = M
        row["N"] = N
        row["D"] = D
        row["layer"] = test_layer_A  # type: ignore

        halut_learned_init = hm.learn_halut_offline(
            train_input.detach().cpu().numpy(), W.detach().cpu().numpy(), C, K
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
        halut_dims_init = (
            torch.from_numpy(halut_learned_init[hm.HalutOfflineStorage.DIMS])
            .to(torch.int64)
            .to(device)
        )
        L = halut_lut_init
        # L = torch.nn.init.kaiming_uniform_(L)
        T = halut_T_init
        # T = torch.nn.init.uniform_(T)
        dims = halut_dims_init
        # in_channels = I.shape[1] // 9
        # dims = torch.zeros(in_channels * 4, dtype=torch.int64)
        # for i in range(in_channels):
        #     # random select idx out of list with no duplicates
        #     # ensure no duplicates
        #     channel_dims = torch.tensor(
        #         np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8], size=4, replace=False),
        #         dtype=torch.int64,
        #     )
        #     dims[i * 4 : (i + 1) * 4] = channel_dims + i * 9

        # train
        batch_size = 256
        epochs = 250
        model = HalutMatmul(C, K, S, B, T, L, dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00004)
        # criterion = torch.nn.HuberLoss(reduction="mean")
        # criterion = torch.nn.L1Loss(reduction="mean")
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

        # train(model, criterion,
        # optimizer, data_loader_train, data_loader_val, device, epochs, lr_scheduler)

        configs = {
            "lr": tune.grid_search([0.0005]),  # tune.uniform(0.0001, 0.001)
            "batch_size": tune.grid_search([2048]),
            "epochs": tune.grid_search([100]),
            "optimizer": tune.grid_search(["adam"]),  # sgd
            "criterion": tune.grid_search(["mse"]),  # mae, huber
            "lr_scheduler": tune.grid_search(["cosine"]),  # plateau, step
            "learn_tensors": tune.grid_search(
                [
                    [True, False, True],
                ]
            ),
        }

        configs = {
            "lr": 0.001,
            "batch_size": 128,
            "epochs": 20,
            "optimizer": "adam",
            "criterion": "mse",
            "lr_scheduler": "cosine",
            "learn_tensors": [True, True],
        }

        row["epochs"] = configs["epochs"]
        row["batch_size"] = configs["batch_size"]
        row["lr"] = configs["lr"]
        row["optimizer"] = configs["optimizer"]
        row["criterion"] = configs["criterion"]
        row["lr_scheduler"] = configs["lr_scheduler"]

        def train_with_config(config):
            I = (
                torch.from_numpy(np.load(data_path + "/" + test_layer_A))
                .to(dtype)
                .to(device)
            )
            train_input = I
            val_input = I
            model = HalutMatmul(C, K, S, B, T, L, dims, *config["learn_tensors"])
            batch_size = config["batch_size"]
            train_dataset = torch.utils.data.TensorDataset(train_input)
            data_loader_train = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_dataset = torch.utils.data.TensorDataset(val_input)
            data_loader_val = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

            params = {
                "other": [],
                "thresholds": [],
                "temperature": [],
                "lut": [],
            }

            def _add_params(module, prefix=""):
                for name, p in module.named_parameters(recurse=False):
                    if not p.requires_grad:  # removes parameter from optimizer!!
                        # filter(lambda p: p.requires_grad, model.parameters())
                        continue
                    if isinstance(module, (HalutMatmul)):
                        if name == "T":
                            params["thresholds"].append(p)
                            continue
                        if name == "temp":
                            params["temperature"].append(p)
                            continue
                        if name == "L":
                            params["lut"].append(p)
                            continue
                    # if prefix in ("conv1", "linear"):
                    #     continue
                    print("add to other", prefix, name)
                    params["other"].append(p)

                for child_name, child_module in module.named_children():
                    child_prefix = (
                        f"{prefix}.{child_name}" if prefix != "" else child_name
                    )
                    _add_params(child_module, prefix=child_prefix)

            _add_params(model)

            custom_lrs = {
                "other": config["lr"],
                "temperature": config["lr"] * 0,
                "thresholds": config["lr"] * 1,
                "lut": config["lr"],
            }
            param_groups = []
            # pylint: disable=consider-using-dict-items
            for key in params:
                if len(params[key]) > 0:
                    # pylint: disable=consider-iterating-dictionary
                    if key in custom_lrs.keys():
                        param_groups.append(
                            {"params": params[key], "lr": custom_lrs[key]}
                        )
                    else:
                        param_groups.append({"params": params[key]})

            epochs = config["epochs"]
            if config["optimizer"] == "sgd":
                optimizer = torch.optim.SGD(param_groups, lr=config["lr"])
            elif config["optimizer"] == "adam":
                optimizer = torch.optim.Adam(param_groups, lr=config["lr"])

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

            mse = evaluate(
                model, torch.nn.MSELoss(reduction="mean"), data_loader_val, device
            )
            mae = evaluate(
                model, torch.nn.L1Loss(reduction="mean"), data_loader_val, device
            )
            huber = evaluate(
                model, torch.nn.HuberLoss(reduction="mean"), data_loader_val, device
            )
            return mse, mae, huber

        mse, mae, huber = train_with_config(configs)
        print("Best trial final validation loss: {}".format(mse))  # type: ignore
        print("Best trial final validation MAE: {}".format(mae))  # type: ignore
        print("Best trial final validation Huber" ": {}".format(huber))  # type: ignore

        # print("Final results", model.T, model.L)
        print("Final max", torch.max(model.T), torch.max(model.L))

        output = model(val_input)
        target = val_input.mm(W)

        print("Final loss", criterion(output, target))
        print("Final max", torch.max(output), torch.max(target))

        # print("Compare", output[0], target[0])
        print("Final MSE", torch.nn.MSELoss(reduction="mean")(output, target))
        print("Final MAE", torch.nn.L1Loss(reduction="mean")(output, target))
        print("Final Huber", torch.nn.HuberLoss(reduction="mean")(output, target))

        print("config", configs)

        row["mse_backprop"] = mse
        row["mae_backprop"] = mae
        row["huber_backprop"] = huber
        row["max_backprop"] = torch.max(output).item()  # tye: ignore

        halut_learned = hm.learn_halut_offline(
            I.detach().cpu().numpy(), W.detach().cpu().numpy(), C, K
        )
        halut_lut = (
            torch.from_numpy(halut_learned[hm.HalutOfflineStorage.LUT])
            .to(dtype)
            .to(device)
        )
        halut_T = (
            torch.from_numpy(halut_learned[hm.HalutOfflineStorage.THRESHOLDS])
            .to(dtype)
            .to(device)
        )
        halut_dims = (
            torch.from_numpy(halut_learned[hm.HalutOfflineStorage.DIMS])
            .to(torch.int64)
            .to(device)
        )

        halut_learned_model = HalutMatmul(C, K, S, B, halut_T, halut_lut, halut_dims)
        halut_learned_output = halut_learned_model(val_input)
        print("Learned loss", criterion(halut_learned_output, target))
        print("Learned max", torch.max(halut_learned_output), torch.max(target))
        print(
            "Learned MSE",
            torch.nn.MSELoss(reduction="mean")(halut_learned_output, target),
        )
        print(
            "Learned MAE",
            torch.nn.L1Loss(reduction="mean")(halut_learned_output, target),
        )
        print(
            "Learned Huber",
            torch.nn.HuberLoss(reduction="mean")(halut_learned_output, target),
        )

        print("Compare", halut_learned_output[0], target[0])

        row["mse_learned"] = torch.nn.MSELoss(reduction="mean")(
            halut_learned_output, target
        ).item()
        row["mae_learned"] = torch.nn.L1Loss(reduction="mean")(
            halut_learned_output, target
        ).item()
        row["huber_learned"] = torch.nn.HuberLoss(reduction="mean")(
            halut_learned_output, target
        ).item()
        row["max_learned"] = torch.max(halut_learned_output).item()  # type: ignore

        for t in [4, 6, 7, 8, 10, 16, 32]:
            # convert to empty type as empty_strided is not supported
            scale = torch.max(I) - torch.min(I)
            scale = scale / (2**t - 1)
            I_quant = torch.fake_quantize_per_tensor_affine(  # type: ignore
                I, scale, 0, quant_min=-(2 ** (t - 1)), quant_max=2 ** (t - 1) - 1
            )
            W_quant = torch.fake_quantize_per_tensor_affine(  # type: ignore
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
            row[f"int{t}_mse"] = torch.nn.MSELoss(reduction="mean")(
                result_scale_back, target
            ).item()
            row[f"int{t}_mae"] = torch.nn.L1Loss(reduction="mean")(
                result_scale_back, target
            ).item()
            row[f"int{t}_huber"] = torch.nn.HuberLoss(reduction="mean")(
                result_scale_back, target
            ).item()
        data.append(row)


# data to csv
df = pd.DataFrame(data)
df.to_csv("accuracy_comparison_all.csv")
