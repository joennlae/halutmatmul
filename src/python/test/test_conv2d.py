from typing import Literal
import math
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
    loop_order: Literal["im2col", "kn2col"] = "im2col",
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
        use_A=use_A,
        loop_order=loop_order,
    )
    input_a = halutmatmul_module.transform_input(input_learn)
    input_b = halutmatmul_module.transform_weight(weights)

    if loop_order == "im2col":
        store_array = hm.learn_halut_offline(
            input_a.detach().cpu().numpy(),
            input_b.detach().cpu().numpy(),
            C=C,
            K=K,
            lut_work_const=-1,
        )
    elif loop_order == "kn2col":
        C = math.ceil(C / (kernel_size * kernel_size))
        store_arrays = []
        # k_x*k_y,M,C,K
        print("C", C)
        luts = []
        dims_list = []
        thresholds_list = []
        for k_x in range(kernel_size):
            for k_y in range(kernel_size):
                input_slice = halutmatmul_module.kn2col_input_slide(
                    input_learn, input_a, k_x, k_y
                )
                print("input_slice.shape", input_slice.shape)
                input_slice = input_slice.reshape(-1, input_slice.shape[-1])
                print("input_slice.shape", input_slice.shape, input_b.shape)
                store_array = hm.learn_halut_offline(
                    input_slice.detach().cpu().numpy(),
                    input_b[k_x * kernel_size + k_y].detach().cpu().numpy(),
                    C=C,
                    K=K,
                    lut_work_const=-1,
                )
                print("C", C)
                luts.append(store_array[hm.HalutOfflineStorage.LUT])
                dims_list.append(store_array[hm.HalutOfflineStorage.DIMS])
                thresholds_list.append(store_array[hm.HalutOfflineStorage.THRESHOLDS])
                store_arrays.append(store_array)
        luts = np.array(luts)
        dims = np.array(dims_list)
        thresholds = np.array(thresholds_list)
        store_array = store_arrays[0]
        print("dims", dims.shape, thresholds.shape)
        store_array[hm.HalutOfflineStorage.LUT] = luts
        store_array[hm.HalutOfflineStorage.DIMS] = dims
        store_array[hm.HalutOfflineStorage.THRESHOLDS] = thresholds

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
    "in_channels, out_channels, image_x_y, kernel_size, bias, C, K, a, b, "
    "groups, use_A, stride, loop_order",
    [
        (
            in_channels,
            out_channels,
            image_x_y,
            kernel_size,
            bias,
            C,
            K,
            a,
            b,
            g,
            use_A,
            stride,
            loop_order,
        )
        for in_channels in [64, 32]
        for out_channels in [64, 32]
        for image_x_y in [7, 14]
        for kernel_size in [1, 3, 5]
        for bias in [True, False]  # True, False
        for C in [32, 64]
        for a in [9.0]
        for b in [-0.35]
        for K in [16]  # [8, 16, 32]
        for g in [1]  # only supporting one group for now
        for use_A in [False]
        for stride in [1, 2]
        for loop_order in ["kn2col", "im2col"]
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
    stride: int,
    loop_order: Literal["im2col", "kn2col"],
) -> None:
    batch_size = 32

    if C > out_channels // groups:
        pytest.skip("Not possible due to D < C")

    if use_A and in_channels * kernel_size**2 % C != 0:
        pytest.skip("Not supported yet when usage of A is enabled")

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
        loop_order=loop_order,
    )
