from collections import OrderedDict
from test.utils.utils import helper_test_module
import torch
import pytest

from halutmatmul.modules import HalutLinear
import halutmatmul.halutmatmul as hm

TEST_CUDA_DEVICE_ID = 0


def linear_helper_gpu(
    in_features: int,
    out_features: int,
    bias: bool,
    C: int,
    device: torch.device,
    a: float = 1.0,
    b: float = 0.0,
    batch_size: int = 1,
) -> None:
    batch_size = 8
    torch.manual_seed(4419)
    n_row = 256
    if batch_size == 1:
        weights = torch.rand((out_features, in_features), device=device)
        bias_weights = torch.rand((out_features), device=device)

        input_learn = (torch.rand((20 * n_row, in_features), device=device) + b) * a
        input_test = (torch.rand((4 * n_row, in_features), device=device) + b) * a
    else:
        weights = torch.rand((out_features, in_features), device=device)
        bias_weights = torch.rand((out_features), device=device)

        input_learn = (
            torch.rand((batch_size * 10, n_row, in_features), device=device) + b
        ) * a
        input_test = (
            torch.rand((batch_size, n_row, in_features), device=device) + b
        ) * a

    learn_numpy = input_learn.detach().cpu().numpy().reshape(-1, input_learn.shape[-1])
    weights_numpy = weights.detach().cpu().numpy().transpose(1, 0)
    store_array = hm.learn_halut_offline(
        learn_numpy, weights_numpy, C=C, lut_work_const=-1
    )

    torch_module = torch.nn.Linear(
        in_features=in_features, out_features=out_features, bias=bias
    )

    halutmatmul_module = HalutLinear(
        in_features=in_features, out_features=out_features, bias=bias
    )
    state_dict = OrderedDict({"weight": weights})
    if bias:
        state_dict = OrderedDict(state_dict | OrderedDict({"bias": bias_weights}))
    torch_module.cuda()
    torch_module.to(device)
    torch_module.eval()
    torch_module.load_state_dict(state_dict, strict=False)
    state_dict = OrderedDict(
        state_dict
        | OrderedDict(
            {
                "store_input": torch.zeros(1, dtype=torch.bool),
                "halut_active": torch.ones(1, dtype=torch.bool),
                "hash_buckets_or_prototypes": torch.from_numpy(
                    store_array[hm.HalutOfflineStorage.HASH_TABLES]
                ),
                "lut": torch.from_numpy(store_array[hm.HalutOfflineStorage.LUT]),
                "halut_config": torch.from_numpy(
                    store_array[hm.HalutOfflineStorage.CONFIG]
                ),
            }
        )
    )
    halutmatmul_module.cuda()
    halutmatmul_module.eval()
    halutmatmul_module.to(device)
    halutmatmul_module.load_state_dict(state_dict, strict=False)

    print("======== TEST =========")
    print(
        f"params: C: {C}, in: {in_features}, out: {out_features}, bias: {bias}, "
        f"a: {a}, b: {b}"
    )
    torch_module.to(device)
    halutmatmul_module.to(device)
    input_test.to(device)
    helper_test_module(
        input_test,
        torch_module,
        halutmatmul_module,
        rel_error=0.0,
        scaled_error_max=0.2,
    )


@pytest.mark.parametrize(
    "in_features, out_features, C, a, b, bias, batch_size",
    [
        (in_features, out_features, C, a, b, bias, batch_size)
        for in_features in [128, 256]
        for out_features in [64, 128]
        for C in [4, 16, 64]
        for a in [1.0]
        for b in [0.0]
        for bias in [False, True]
        for batch_size in [1, 8]
    ],
)
def test_linear_module_gpu(
    in_features: int,
    out_features: int,
    C: int,
    a: float,
    b: float,
    bias: bool,
    batch_size: int,
) -> None:

    device_id = TEST_CUDA_DEVICE_ID
    if not torch.cuda.is_available():
        pytest.skip("need GPU to run")
    torch.cuda.set_device(device_id)

    device = torch.device(
        "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
    )

    linear_helper_gpu(
        in_features,
        out_features,
        bias,
        C,
        device=device,
        a=a,
        b=b,
        batch_size=batch_size,
    )
