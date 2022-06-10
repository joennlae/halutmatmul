from collections import OrderedDict
from test.utils.utils import helper_test_module
import torch
import numpy as np
import pytest

from halutmatmul.modules import HalutConv2d
import halutmatmul.halutmatmul as hm

TEST_CUDA_DEVICE_ID = 0


def conv2d_helper_gpu(
    in_channels: int,
    out_channels: int,
    image_x_y: int,
    bias: bool,
    kernel_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    groups: int = 1,
    C: int = 16,
    K: int = 16,
    a: float = 1.0,
    b: float = 0.0,
    encoding_algorithm: int = hm.EncodingAlgorithm.FOUR_DIM_HASH,
) -> None:
    # pylint: disable=import-outside-toplevel
    from halutmatmul.cuda.functions import halut_conv2d_gpu

    torch.manual_seed(4419)

    weights = torch.rand(
        (out_channels, in_channels // groups, kernel_size, kernel_size)
    )
    bias_weights = torch.rand((out_channels))

    input_learn = (
        torch.rand((batch_size * 2, in_channels, image_x_y, image_x_y)) + b
    ) * a
    input_learn = torch.relu(input_learn)
    print("MIN_MAX", torch.min(input_learn), torch.max(input_learn))
    input_test = (torch.rand((batch_size, in_channels, image_x_y, image_x_y)) + b) * a
    input_test = torch.relu(input_test)

    padding = 1
    input_a_torch, input_b_torch = halut_conv2d_gpu(
        input_learn,
        weights,
        L=None,  # type: ignore[arg-type]
        H=None,  # type: ignore[arg-type]
        read_acc_lut_kernel=None,
        encode_kernel=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias_weights,
        return_reshaped_inputs=True,
    )
    input_a = input_a_torch.detach().cpu().numpy()
    input_b = input_b_torch.detach().cpu().numpy()

    store_array = hm.learn_halut_offline(
        input_a,
        input_b,
        C=C,
        K=K,
        lut_work_const=-1,
        encoding_algorithm=encoding_algorithm,
    )

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

    torch_module.cuda()
    torch_module.to(device)
    torch_module.eval()
    torch_module.load_state_dict(state_dict, strict=False)

    state_dict = OrderedDict(
        state_dict
        | OrderedDict(
            {
                "halut_active": torch.ones(1, dtype=torch.bool),
                "hash_buckets_or_prototypes": torch.from_numpy(
                    store_array[hm.HalutOfflineStorage.HASH_TABLES].astype(np.float32)
                    if encoding_algorithm
                    in [
                        hm.EncodingAlgorithm.FOUR_DIM_HASH,
                        hm.EncodingAlgorithm.DECISION_TREE,
                    ]
                    else store_array[hm.HalutOfflineStorage.PROTOTYPES].astype(
                        np.float32
                    )
                ),
                "lut": torch.from_numpy(
                    store_array[hm.HalutOfflineStorage.LUT].astype(np.float32)
                ),
                "halut_config": torch.from_numpy(
                    store_array[hm.HalutOfflineStorage.CONFIG].astype(np.float32)
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
        f"params: C: {C}, in: {in_channels}, out: {out_channels}, bias: {bias}, "
        f"input_learn: {input_learn.shape}, input_test: {input_test.shape}, a: {a}, b: {b} "
        f"encoding_algorithm: {encoding_algorithm}"
    )
    input_test = input_test.to(device)
    helper_test_module(
        input_test,
        torch_module,
        halutmatmul_module,
        rel_error=-1.0,
        scaled_error_max=0.08,
    )


@pytest.mark.parametrize(
    "in_channels, out_channels, image_x_y, kernel_size, bias, C, K, a, b, encoding_algorithm",
    [
        (in_channels, out_channels, image_x_y, kernel_size, bias, C, K, a, b, e)
        for in_channels in [64]
        for out_channels in [64]
        for image_x_y in [7]  # 14, 28, 56
        for kernel_size in [1, 3]
        for bias in [False]  # True, False
        for C in [4, 8, 16, 32, 64]  # s 96, 128
        for a in [9.0]
        for b in [-0.35]
        for e in [
            hm.EncodingAlgorithm.FOUR_DIM_HASH,
            hm.EncodingAlgorithm.DECISION_TREE,
            hm.EncodingAlgorithm.FULL_PQ,
        ]
        for K in (
            [16]  # 64 uses to much shared memory [8, 16, 32]
            if e == hm.EncodingAlgorithm.FOUR_DIM_HASH
            else [8, 16, 24] # [4, 8, 12, 16, 24, 32, 64]
        )
    ],
)
def test_conv2d_module_gpu(
    in_channels: int,
    out_channels: int,
    image_x_y: int,
    kernel_size: int,
    bias: bool,
    C: int,
    K: int,
    a: float,
    b: float,
    encoding_algorithm: int,
) -> None:

    device_id = TEST_CUDA_DEVICE_ID
    if not torch.cuda.is_available():
        pytest.skip("need GPU to run")
    torch.cuda.set_device(device_id)
    device = torch.device(
        "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
    )
    batch_size = 32  # 32, 64, 128

    stride = 1
    groups = 1  # only 1 supported
    conv2d_helper_gpu(
        in_channels=in_channels,
        out_channels=out_channels,
        image_x_y=image_x_y,
        bias=bias,
        kernel_size=kernel_size,
        stride=stride,
        batch_size=batch_size,
        device=device,
        groups=groups,
        C=C,
        K=K,
        a=a,
        b=b,
        encoding_algorithm=encoding_algorithm,
    )
