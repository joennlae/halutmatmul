from collections import OrderedDict
from test.utils.utils import helper_test_module
import torch
import pytest
import numpy as np

from halutmatmul.modules import HalutConv2d
import halutmatmul.halutmatmul as hm


def conv2d_helper(
    in_channels: int,
    out_channels: int,
    image_x_y: int,
    bias: bool,
    kernel_size: int,
    stride: int,
    batch_size: int,
    groups: int = 1,
    C: int = 16,
    K: int = 16,
    a: float = 1.0,
    b: float = 0.0,
    use_A: bool = False,
) -> None:
    torch.manual_seed(4419)

    weights = torch.rand(
        (out_channels, in_channels // groups, kernel_size, kernel_size)
    )
    bias_weights = torch.rand((out_channels))

    input_learn = (
        torch.rand((batch_size * 2, in_channels, image_x_y, image_x_y)) + b
    ) * a
    input_test = (torch.rand((batch_size, in_channels, image_x_y, image_x_y)) + b) * a

    torch_module = torch.nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        groups=groups,
    )

    halutmatmul_module = HalutConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        groups=groups,
        split_factor=1,
        use_A=use_A
    )
    input_a = halutmatmul_module.transform_input(input_learn)
    input_b = halutmatmul_module.transform_weight(weights)

    store_array = hm.learn_halut_offline(
        input_a.detach().cpu().numpy(),
        input_b.detach().cpu().numpy(),
        C=C,
        K=K,
        lut_work_const=-1,
    )

    state_dict = OrderedDict({"weight": weights})
    if bias:
        state_dict = OrderedDict(state_dict | OrderedDict({"bias": bias_weights}))
    torch_module.load_state_dict(state_dict, strict=False)
    state_dict = OrderedDict(
        state_dict
        | OrderedDict(
            {
                "halut_active": torch.ones(1, dtype=torch.bool),
                "lut": torch.from_numpy(store_array[hm.HalutOfflineStorage.LUT]),
                "thresholds": torch.from_numpy(
                    store_array[hm.HalutOfflineStorage.THRESHOLDS]
                ),
                "dims": torch.from_numpy(store_array[hm.HalutOfflineStorage.DIMS]),
            }
        )
    )
    halutmatmul_module.load_state_dict(state_dict, strict=False)

    print("======== TEST =========")
    print(
        f"params: C: {C}, in: {in_channels}, out: {out_channels}, bias: {bias}, "
        f"input_learn: {input_learn.shape}, input_test: {input_test.shape}, a: {a}, b: {b} "
    )
    helper_test_module(
        input_test,
        torch_module,
        halutmatmul_module,
        rel_error=-1.0,
        scaled_error_max=0.02,
    )


@pytest.mark.parametrize(
    "in_channels, out_channels, image_x_y, kernel_size, bias, C, K, a, b, groups, use_A",
    [
        (in_channels, out_channels, image_x_y, kernel_size, bias, C, K, a, b, g, use_A)
        for in_channels in [64, 32]
        for out_channels in [64, 32]
        for image_x_y in [7, 14]
        for kernel_size in [1, 3, 5]
        for bias in [True, False]  # True, False
        for C in [16, 32, 64]
        for a in [9.0]
        for b in [-0.35]
        for K in [16]  # [8, 16, 32]
        for g in [1]  # only supporting one group for now
        for use_A in [True, False]
    ],
)
def test_conv2d_module(
    in_channels: int,
    out_channels: int,
    image_x_y: int,
    kernel_size: int,
    bias: bool,
    C: int,
    K: int,
    a: float,
    b: float,
    groups: int,
    use_A: bool,
) -> None:
    batch_size = 32

    if C > out_channels // groups:
        pytest.skip("Not possible due to D < C")

    if use_A and in_channels * kernel_size**2 % C != 0:
        pytest.skip("Not supported yet when usage of A is enabled")

    stride = 1
    conv2d_helper(
        in_channels,
        out_channels,
        image_x_y,
        bias,
        kernel_size,
        stride,
        batch_size,
        groups,
        C,
        K=K,
        a=a,
        b=b,
        use_A=use_A,
    )
