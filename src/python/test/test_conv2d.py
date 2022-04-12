from collections import OrderedDict
from test.utils.utils import helper_test_module
import torch
import numpy as np
import pytest

from halutmatmul.modules import HalutConv2d, halut_conv2d
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
    a: float = 1.0,
    b: float = 0.0,
    rel_error: float = 0.3,
) -> None:
    torch.manual_seed(4419)

    weights = torch.rand(
        (out_channels, in_channels // groups, kernel_size, kernel_size)
    )
    bias_weights = torch.rand((out_channels))

    input_learn = (
        torch.rand((batch_size * 10, in_channels, image_x_y, image_x_y)) + b
    ) * a
    input_test = (torch.rand((batch_size, in_channels, image_x_y, image_x_y)) + b) * a

    learn_numpy = input_learn.detach().cpu().numpy()
    weights_numpy = weights.detach().cpu().numpy()
    bias_numpy = bias_weights.detach().cpu().numpy()

    input_a, input_b = halut_conv2d(
        learn_numpy,
        weights_numpy,
        hm.HalutMatmul(),
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        bias=bias_numpy,
        return_reshaped_inputs=True,
    )

    store_array = hm.learn_halut_offline(input_a, input_b, C=C, lut_work_const=-1)

    torch_module = torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
    )

    halutmatmul_module = HalutConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        groups=groups,
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
                "hash_buckets": torch.from_numpy(
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
        f"params: C: {C}, in: {in_channels}, out: {out_channels}, bias: {bias}, "
        f"input_learn: {input_learn.shape}, input_test: {input_test.shape}, a: {a}, b: {b}"
    )
    helper_test_module(
        input_test, torch_module, halutmatmul_module, rel_error=rel_error
    )


@pytest.mark.parametrize(
    "in_channels, out_channels, image_x_y, kernel_size, bias, C, a, b",
    [
        (in_channels, out_channels, image_x_y, kernel_size, bias, C, a, b)
        for in_channels in [64, 128, 256]
        for out_channels in [64, 128, 256]
        for image_x_y in [7, 14]
        for kernel_size in [1, 3]
        for bias in [False]  # True, False
        for C in [16, 32, 64]
        for a in [1.0]
        for b in [0.0]
    ],
)
def test_conv2d_module(
    in_channels: int,
    out_channels: int,
    image_x_y: int,
    kernel_size: int,
    bias: bool,
    C: int,
    a: float,
    b: float,
) -> None:
    batch_size = 32  # 32, 64

    stride = 1
    groups = 1
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
        a,
        b,
    )
