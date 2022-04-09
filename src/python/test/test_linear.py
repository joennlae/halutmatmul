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
) -> None:
    torch.manual_seed(4419)
    weights = torch.rand((out_features, in_features))
    bias_weights = torch.rand((out_features))

    input_learn = (torch.rand((n_row_learn, in_features)) + b) * a
    input_test = (torch.rand((n_row_test, in_features)) + b) * a

    learn_numpy = input_learn.detach().numpy()
    weights_numpy = weights.detach().numpy().transpose(1, 0)
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
    torch_module.load_state_dict(state_dict)
    state_dict = OrderedDict(state_dict | OrderedDict(
        {
            "store_input": torch.zeros(1, dtype=torch.bool),
            "halut_active": torch.ones(1, dtype=torch.bool),
            "hash_buckets": torch.from_numpy(
                store_array[hm.HalutOfflineStorage.HASH_TABLES]
            ),
            "lut": torch.from_numpy(store_array[hm.HalutOfflineStorage.LUT]),
            "lut_offset_scale": torch.from_numpy(
                store_array[hm.HalutOfflineStorage.LUT_OFFSET_SCALE]
            ),
        }
    ))
    halutmatmul_module.load_state_dict(state_dict)

    print("======== TEST =========")
    print(
        f"params: C: {C}, in: {in_features}, out: {out_features}, bias: {bias}, "
        f"n_row_learn: {n_row_learn}, n_row_test: {n_row_test}, a: {a}, b: {b}"
    )
    helper_test_module(input_test, torch_module, halutmatmul_module)


@pytest.mark.parametrize(
    "in_features, out_features", [(2048, 1000)] # (512, 16), (2048, 100)
)
def test_linear_module(in_features: int, out_features: int) -> None:
    n_row_learn = 10000
    n_row_test = 2000
    for C in [4, 16, 32, 64]:  # [4, 8, 16, 32, 64]
        for a in [1.0, 4.0, 100.0]:
            for b in [0.0, 20.0]:
                for bias in [True, False]:
                    linear_helper(
                        in_features,
                        out_features,
                        bias,
                        n_row_learn,
                        n_row_test,
                        C,
                        a,
                        b,
                    )
