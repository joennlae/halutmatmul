from collections import OrderedDict
from test.utils.utils import helper_test_module
import torch
import pytest

from halutmatmul.modules import HalutLinear
import halutmatmul.halutmatmul as hm


def linear_helper(
    in_features: int,
    out_features: int,
    bias: bool,
    n_row_learn: int,
    n_row_test: int,
    C: int,
    a: float = 1.0,
    b: float = 0.0,
    batch_size: int = 1,
) -> None:
    torch.manual_seed(4419)
    if batch_size == 1:
        weights = torch.rand((out_features, in_features))
        bias_weights = torch.rand((out_features))

        input_learn = (torch.rand((n_row_learn, in_features)) + b) * a
        input_test = (torch.rand((n_row_test, in_features)) + b) * a
    else:
        weights = torch.rand((out_features, in_features))
        bias_weights = torch.rand((out_features))

        n_row = 256
        input_learn = (torch.rand((batch_size * 10, n_row, in_features)) + b) * a
        input_test = (torch.rand((batch_size, n_row, in_features)) + b) * a

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
    halutmatmul_module.load_state_dict(state_dict, strict=False)

    print("======== TEST =========")
    print(
        f"params: C: {C}, in: {in_features}, out: {out_features}, bias: {bias}, "
        f"n_row_learn: {n_row_learn}, n_row_test: {n_row_test}, a: {a}, b: {b}"
    )
    helper_test_module(input_test, torch_module, halutmatmul_module)


@pytest.mark.parametrize(
    "in_features, out_features, C, a, b, bias, batch_size",
    [
        (in_features, out_features, C, a, b, bias, batch_size)
        for in_features in [512, 1024]
        for out_features in [32, 128]
        for C in [4, 16, 64]
        for a in [1.0]
        for b in [0.0]
        for bias in [True, False]
        for batch_size in [1, 8]
    ],
)
def test_linear_module(
    in_features: int,
    out_features: int,
    C: int,
    a: float,
    b: float,
    bias: bool,
    batch_size: int,
) -> None:
    n_row_learn = 10000
    n_row_test = 2000
    linear_helper(
        in_features,
        out_features,
        bias,
        n_row_learn,
        n_row_test,
        C,
        a,
        b,
        batch_size=batch_size,
    )
