# pylint: disable=import-outside-toplevel
import sys
import math
from typing import Any, Optional, OrderedDict, Union, Literal
import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules import Linear

from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter


def create_selection_matrix(
    C: int = 1, K: int = 16, dtype=torch.float16
) -> torch.Tensor:
    depth = int(math.sqrt(K))
    selection_matrix = torch.zeros((C * 15, C * depth), dtype=dtype)
    based_selection_matrix = torch.zeros((K - 1, depth), dtype=dtype)
    for i in range(K - 1):
        if i == 0:
            based_selection_matrix[0, 0] = 1
        else:
            based_selection_matrix[i, int(np.log2(i + 1))] = 1
    for c in range(C):
        selection_matrix[
            c * 15 : (c + 1) * 15, c * depth : (c + 1) * depth
        ] = based_selection_matrix
    return selection_matrix


def create_bit_matrix(C: int = 1, K: int = 16, dtype=torch.float16) -> torch.Tensor:
    # example when using C = 1
    # fmt: off
    bit_matrix_numpy = np.array(
        [
            # 0
            [-1,
             -1, 0,
             -1, 0, 0, 0,
             -1, 0, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             -1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             1, 0, 0, 0,
             0, -1, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             1, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, -1, 0, 0,
             0, 0, -1, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, -1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, 1, 0, 0,
             0, 0, 0, -1, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0],
            # 8
            [1,
             0, -1,
             0, 0, -1, 0,
             0, 0, 0, 0, -1, 0, 0, 0],
            [1,
             0, -1,
             0, 0, -1, 0,
             0, 0, 0, 0, 1, 0, 0, 0],
            [1,
             0, -1,
             0, 0, 1, 0,
             0, 0, 0, 0, 0, -1, 0, 0],
            [1,
             0, -1,
             0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 0, 0],
             # 12
            [1,
             0, 1,
             0, 0, 0, -1,
             0, 0, 0, 0, 0, 0, -1, 0],
            [1,
             0, 1,
             0, 0, 0, -1,
             0, 0, 0, 0, 0, 0, 1, 0],
            [1,
             0, 1,
             0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, -1],
            [1,
             0, 1,
             0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    # fmt: on
    bit_matrix_base = torch.from_numpy(bit_matrix_numpy).to(dtype)
    bit_matrix = torch.ones((C * K, C * (K - 1)), dtype=dtype)
    for c in range(C):
        bit_matrix[
            c * K : (c + 1) * K,
            c * (K - 1) : (c + 1) * (K - 1),
        ] = bit_matrix_base
    return bit_matrix


def apply_decision_tree_torch(
    X: torch.Tensor, decision_tree: torch.Tensor
) -> torch.Tensor:
    N = X.shape[0]
    mapping = torch.zeros(
        N, dtype=torch.int64
    )  # needs to be int64 because of index :-)
    B = decision_tree.shape[0] // 3
    n_decisions = int(np.log2(B))
    for depth in range(n_decisions):
        index_offet = 2**depth - 1
        split_thresholds = decision_tree[mapping + B + index_offet]
        dims = decision_tree[mapping + index_offet].long()
        x = X[torch.arange(N), dims]
        indicators = x > split_thresholds
        mapping = (mapping * 2) + indicators
    mapping = decision_tree[mapping + 2 * B].long()
    return mapping


def halut_matmul_forward(
    input: torch.Tensor,
    T: torch.Tensor,
    L: torch.Tensor,
    S: torch.Tensor,
    B: torch.Tensor,
    C: int = 32,
    K: int = 16,
    dims: Optional[torch.Tensor] = None,
    prototypes: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = torch.ones(1, dtype=torch.float16),
    split_factor: int = 4,
    L_int8: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # encoding
    input_reshaped = input.reshape((input.shape[0], C, -1))
    if dims is not None:  # default maddness
        h = S.mm(input[:, dims].T) - T.unsqueeze(1)
    elif prototypes is not None:  # using argmin
        # input_reshaped = input.reshape((input.shape[0], C, -1))
        mse = torch.sum(torch.square(input_reshaped.unsqueeze(2) - prototypes), dim=3)
        encoding_soft = torch.nn.Softmax(dim=2)(-mse / temperature)
    else:
        raise Exception("Either dims or prototype must be provided")
    if prototypes is None:
        tanh_h = torch.tanh(h / temperature)
        sign_ste = torch.sign(h) - tanh_h.detach() + tanh_h
        b = B.mm(sign_ste)
        b = b.T.reshape((-1, C, K))
        encoding_soft = torch.nn.Softmax(dim=2)(b)
    index = torch.argmax(encoding_soft, dim=2, keepdim=True)
    encoding_hard = torch.zeros_like(
        encoding_soft, memory_format=torch.legacy_contiguous_format
    ).scatter_(2, index, 1.0)
    E = encoding_hard - encoding_soft.detach() + encoding_soft

    # decoding
    result = torch.zeros(
        [input.shape[0], L.size(0)], dtype=input.dtype, device=input.device
    )
    # split_factor only need for memory usage reduction
    assert L.size(0) % split_factor == 0
    for i in range(split_factor):
        M = L.size(0)
        result[
            :, (M // split_factor) * i : (M // split_factor) * (i + 1)
        ] = torch.einsum(
            "nij, kij -> nki",
            [E, L[(M // split_factor) * i : (M // split_factor) * (i + 1)]],
        ).sum(
            dim=2
        )
    if L_int8 is not None and scale is not None:
        result_hard = torch.zeros(
            [input.shape[0], L.size(0)], dtype=input.dtype, device=input.device
        )
        for i in range(split_factor):
            M = L.size(0)
            result_hard[
                :, (M // split_factor) * i : (M // split_factor) * (i + 1)
            ] = torch.einsum(
                "nij, kij -> nki",
                [E, L_int8[(M // split_factor) * i : (M // split_factor) * (i + 1)]],
            ).sum(
                dim=2
            )
        result_hard = result_hard * scale
        result = result_hard - result.detach() + result  # STE
    return result


class HalutLinear(Linear):
    r"""
    Tested is only 1D, and 2D!!
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Union[str, Any] = None,
        dtype: Union[str, Any] = None,
        split_factor: int = 1,
        use_prototypes: bool = False,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.halut_active = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.lut = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.thresholds = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.dims = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.report_error = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.S = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.B = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.P = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.temperature = Parameter(torch.ones(1), requires_grad=True)
        self.errors = [(-1, np.zeros(ErrorTuple.MAX, dtype=np.float64))]

        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

        self.split_factor = split_factor
        self.use_prototypes = use_prototypes
        self._register_load_state_dict_pre_hook(self.state_dict_hook)

    def update_lut(self):
        pass

    def halut_updates(self):
        pass

    # has to be defined twice as we need the self object which is not passed per default to the hook
    def state_dict_hook(
        self, state_dict: "OrderedDict[str, Tensor]", prefix: str, *_: Any
    ) -> None:
        if all(
            k in state_dict.keys()
            for k in (
                prefix + "lut",
                prefix + "thresholds",
                prefix + "dims",
            )
        ):
            if not state_dict[prefix + "halut_active"]:
                return
            # hack to support variable parameter size --> with the cost of double copying :-)
            self.lut = Parameter(
                state_dict[prefix + "lut"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=True,
            )
            self.thresholds = Parameter(
                state_dict[prefix + "thresholds"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=True,
            )
            self.dims = Parameter(
                state_dict[prefix + "dims"]
                .clone()
                .to(torch.int64)
                .to(str(self.weight.device)),
                requires_grad=False,
            )
            self.P = Parameter(
                state_dict[prefix + "P"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=True,
            )
            if len(self.P.shape) > 1:
                self.use_prototypes = False
            if not self.use_prototypes:
                state_dict[prefix + "B"] = create_bit_matrix(
                    self.lut.size(1), self.lut.size(2), self.weight.dtype
                ).to(str(self.weight.device))
                self.B = Parameter(
                    state_dict[prefix + "B"],
                    requires_grad=False,
                )
                state_dict[prefix + "S"] = create_selection_matrix(
                    self.lut.size(1), self.lut.size(2), self.weight.dtype
                ).to(str(self.weight.device))
                self.S = Parameter(
                    state_dict[prefix + "S"],
                    requires_grad=False,
                )
            self.weight.requires_grad = False
        elif any(
            k in state_dict.keys()
            for k in (
                prefix + "lut",
                prefix + "thresholds",
                prefix + "dims",
            )
        ):
            raise Exception(
                f"not all '{prefix}lut', "
                f"'{prefix}thresholds', '{prefix}dims' in state_dict"
            )

    def get_error(self) -> np.ndarray:
        if not self.report_error[0]:
            raise Exception("get_error() called without error reporting active")
        errors = np.zeros(ErrorTuple.MAX, dtype=np.float64)
        total_input_images = 0
        for elem in self.errors:
            if elem[0] > 0:
                total_input_images += elem[0]
                errors += elem[1] * elem[0]
        errors /= total_input_images
        return errors

    def check_store_offline(self, _input: Tensor) -> None:
        if self.store_input[0]:
            if self.input_storage_a is None and self.input_storage_b is None:
                self.input_storage_a = (
                    _input.clone().cpu().detach().reshape((-1, _input.shape[-1]))
                )
                self.input_storage_b = (
                    self.weight.clone().cpu().transpose(1, 0).detach()
                )
            else:
                self.input_storage_a = torch.cat(
                    (
                        self.input_storage_a,
                        _input.clone()
                        .cpu()
                        .detach()
                        .reshape((-1, _input.shape[-1])),  # type: ignore[arg-type]
                    ),
                    0,  # type: ignore
                )

    def forward(self, _input: Tensor) -> Tensor:
        if self.halut_active[0] and not self.store_input[0]:
            input_shape_len = len(_input.shape)
            batch_size = _input.shape[0]
            _input = _input.reshape((-1, _input.shape[-1]))
            output = halut_matmul_forward(
                _input,
                self.thresholds,
                self.lut,
                self.S,
                self.B,
                self.lut.size(1),
                self.lut.size(2),
                self.dims if not self.use_prototypes else None,
                self.P if self.use_prototypes else None,
                temperature=self.temperature,
                split_factor=self.split_factor,
            )
            if self.bias is not None:
                output += self.bias.t().repeat(*(*output.shape[:-1], 1))
            if input_shape_len > 2:
                output = output.reshape(batch_size, -1, output.shape[-1])
            if self.report_error[0]:
                torch_ret = F.linear(_input, self.weight, self.bias)
                output_clone = output.detach()
                if "cuda" in str(_input.device):
                    from halutmatmul.cuda.functions import error_cupy

                    res_error = error_cupy(output_clone, torch_ret)
                    self.errors.append((_input.shape[0], res_error))  # type: ignore
                else:
                    res_error = error_numpy(output_clone, torch_ret)
                    self.errors.append((_input.shape[0], res_error))  # type: ignore

            return output
        else:
            self.check_store_offline(_input)
            return F.linear(_input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "Halut in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def error_numpy(
    actual: torch.Tensor,
    desired: torch.Tensor,
) -> np.ndarray:
    _min = np.min(desired)
    _max = np.max(desired)
    actual_cupy_std = (actual - _min) / (_max - _min)
    desired_cupy_std = (desired - _min) / (_max - _min)
    _range = (-1, 1)
    actual_cupy_scaled = actual_cupy_std * (_range[1] - _range[0]) + _range[0]
    desired_cupy_scaled = desired_cupy_std * (_range[1] - _range[0]) + _range[0]
    mae = np.mean(np.abs((actual - desired)))
    mse = np.mean((actual - desired) ** 2)
    mape = np.mean(np.abs(actual - desired) / (1 + np.abs(desired)))
    scaled_absolut_error = np.mean(np.abs(actual_cupy_scaled - desired_cupy_scaled))
    scaled_shift = np.mean(actual_cupy_scaled - desired_cupy_scaled)

    return np.array((mae, mse, mape, scaled_absolut_error, scaled_shift))


class ErrorTuple:
    MAE = 0
    MSE = 1
    MAPE = 2
    SCALED_ERROR = 3
    SCALED_SHIFT = 4
    MAX = 5


class HalutConv2d(_ConvNd):
    __doc__ = r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
        split_factor: int = 4,
        use_prototypes: bool = False,
        loop_order: Literal["im2col", "kn2col"] = "im2col",
        halut_active: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        self.halut_active = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.lut = Parameter(torch.zeros(1), requires_grad=False)
        self.lut_int8 = Parameter(torch.ones(1), requires_grad=False)
        self.scale = Parameter(torch.ones(1), requires_grad=False)
        self.thresholds = Parameter(torch.zeros(1), requires_grad=False)
        self.dims = Parameter(torch.zeros(1), requires_grad=False)

        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.report_error = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.errors = [(-1, np.zeros(ErrorTuple.MAX, dtype=np.float64))]
        self.S = Parameter(torch.zeros(1), requires_grad=False)
        self.B = Parameter(torch.zeros(1), requires_grad=False)
        self.P = Parameter(torch.zeros(1), requires_grad=False)
        self.temperature = Parameter(torch.ones(1), requires_grad=True)
        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

        self.split_factor = split_factor
        self.use_prototypes = use_prototypes
        self.loop_order = loop_order

        if halut_active:
            self.halut_active = Parameter(
                torch.ones(1, dtype=torch.bool), requires_grad=False
            )

            # M, C, K
            K = 16
            C = self.weight.shape[1]
            M = self.weight.shape[0]
            self.lut = Parameter(torch.zeros((M, C, K)), requires_grad=True)
            self.thresholds = Parameter(torch.zeros((C * 15)), requires_grad=True)
            self.S = Parameter(
                create_selection_matrix(
                    self.lut.size(1), self.lut.size(2), self.weight.dtype
                ).to(str(self.weight.device)),
                requires_grad=False,
            )
            self.B = Parameter(
                create_bit_matrix(
                    self.lut.size(1), self.lut.size(2), self.weight.dtype
                ).to(str(self.weight.device)),
                requires_grad=False,
            )
            self.weight.requires_grad = False

            self.dims = Parameter(
                torch.zeros(in_channels * 4, dtype=torch.int64), requires_grad=False
            )
            for i in range(in_channels):
                # random select idx out of list with no duplicates
                # ensure no duplicates
                channel_dims = torch.tensor(
                    np.random.choice(
                        [0, 1, 2, 3, 4, 5, 6, 7, 8], size=4, replace=False
                    ),
                    dtype=torch.int64,
                )
                self.dims[i * 4 : (i + 1) * 4] = channel_dims + i * 9

        self._register_load_state_dict_pre_hook(self.pre_state_dict_hook)

    def update_lut(self):
        self.scale.data[0] = torch.max(self.lut.data) - torch.min(self.lut.data)
        bits = 8
        self.scale.data[0] = self.scale.data[0] / (2**bits - 1)
        quant_min = -(2 ** (bits - 1))
        quant_max = 2 ** (bits - 1) - 1
        self.lut_int8.data = torch.clamp(
            torch.round(self.lut / self.scale), quant_min, quant_max
        )

    def halut_updates(self):
        if self.halut_active:
            self.update_lut()

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", halut_active={self.halut_active.item()}, loop_order={self.loop_order}, "
            f"split_factor={self.split_factor}, "
            f" use_prototypes={self.use_prototypes}"
        )

    def pre_state_dict_hook(
        self, state_dict: "OrderedDict[str, Tensor]", prefix: str, *_: Any
    ) -> None:
        if all(
            k in state_dict.keys()
            for k in (
                prefix + "lut",
                prefix + "thresholds",
                prefix + "dims",
            )
        ):
            if not state_dict[prefix + "halut_active"]:
                return
            # hack to support variable parameter size --> with the cost of double copying :-)
            self.lut = Parameter(
                state_dict[prefix + "lut"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=True,
            )
            self.lut_int8 = Parameter(
                state_dict[prefix + "lut_int8"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=False,
            )
            self.thresholds = Parameter(
                state_dict[prefix + "thresholds"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=True,
            )
            self.dims = Parameter(
                state_dict[prefix + "dims"]
                .clone()
                .to(torch.int64)
                .to(str(self.weight.device)),
                requires_grad=False,
            )
            self.P = Parameter(
                state_dict[prefix + "P"]
                .clone()
                .to(str(self.weight.device))
                .to(self.weight.dtype),
                requires_grad=True,
            )
            if len(self.P.shape) > 1:
                self.use_prototypes = False
            self.weight.requires_grad = False
            if len(self.lut.shape) > 3:
                self.loop_order = "kn2col"
            if not self.use_prototypes:
                state_dict[prefix + "B"] = create_bit_matrix(
                    self.lut.size(1), self.lut.size(2), self.weight.dtype
                ).to(str(self.weight.device))
                self.B = Parameter(
                    state_dict[prefix + "B"],
                    requires_grad=False,
                )
                state_dict[prefix + "S"] = create_selection_matrix(
                    self.lut.size(1), self.lut.size(2), self.weight.dtype
                ).to(str(self.weight.device))
                self.S = Parameter(
                    state_dict[prefix + "S"],
                    requires_grad=False,
                )
        elif any(
            k in state_dict.keys()
            for k in (
                prefix + "lut",
                prefix + "tresholds",
                prefix + "dims",
            )
        ):
            raise Exception(
                f"not all '{prefix}lut', "
                f"'{prefix}thresholds', '{prefix}dims' in state_dict"
            )

    def get_error(self) -> np.ndarray:
        if not self.report_error[0]:
            raise Exception("get_error() called without error reporting active")
        errors = np.zeros(ErrorTuple.MAX, dtype=np.float64)
        total_input_images = 0
        for elem in self.errors:
            if elem[0] > 0:
                total_input_images += elem[0]
                errors += elem[1] * elem[0]
        errors /= total_input_images
        return errors

    def check_store_offline(
        self, _input: Tensor, transformed_input: Optional[Tensor] = None
    ) -> None:
        if self.store_input[0]:
            input_a = (
                self.transform_input(_input)
                if transformed_input is None
                else transformed_input
            )
            input_b = self.transform_weight(self.weight)
            print(
                "storing input in ram ",
                input_a.shape,
                input_a.shape[0] * input_a.shape[1] * 4 / (1024 * 1024 * 1024),
                " GB",
            )
            if self.input_storage_a is None and self.input_storage_b is None:
                self.input_storage_a = input_a.cpu().detach()
                self.input_storage_b = input_b.cpu().detach()
            else:
                self.input_storage_a = torch.cat(
                    (self.input_storage_a, input_a.cpu().detach()), 0  # type: ignore[arg-type]
                )
                print(
                    f"new storage size: "
                    f"{sys.getsizeof(self.input_storage_a.storage())/ (1024 * 1024 * 1024)} GB"
                    f"size: {self.input_storage_a.size}"
                )

    def transform_input(self, _input: Tensor) -> Tensor:
        if self.loop_order == "im2col":
            unfold_ops = torch.nn.Unfold(
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,  # type: ignore[arg-type]
                stride=self.stride,
            )
            if self.groups == 1:
                unfolded = unfold_ops(_input).transpose(1, 2)
                unfolded = torch.reshape(unfolded, (-1, unfolded.size(2)))
            else:
                raise NotImplementedError("groups > 1 has to be implemented")
        elif self.loop_order == "kn2col":
            # batch size, in channels, height, width
            unfolded = _input.movedim(1, 3)
            # batch size, height, width, in channels
        return unfolded

    def transform_weight(self, weight: Tensor) -> Tensor:
        # weight is passed that the trasnfrom can also be used from test etc.
        if self.loop_order == "im2col":
            weights_prepared = weight.view(weight.size(0), -1).t()
        elif self.loop_order == "kn2col":
            weights_prepared = weight.reshape(
                weight.size(0), weight.size(1), -1
            ).transpose(0, 2)
        return weights_prepared

    def get_H_W_out(self, H_in: int, W_in: int) -> tuple[int, int]:
        # reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H_out, W_out = (
            math.floor(
                (
                    H_in  # type: ignore
                    + 2 * self.padding[0]
                    - self.dilation[0] * (self.kernel_size[0] - 1)
                    - 1
                )
                / self.stride[0]
                + 1
            ),
            math.floor(
                (
                    W_in  # type: ignore
                    + 2 * self.padding[1]
                    - self.dilation[1] * (self.kernel_size[1] - 1)
                    - 1
                )
                / self.stride[1]
                + 1
            ),
        )
        return H_out, W_out

    def transform_output(self, output: Tensor, _input: Tensor) -> Tensor:
        H_out, W_out = self.get_H_W_out(_input.shape[2], _input.shape[3])
        if self.loop_order == "im2col":
            output = output.reshape((_input.shape[0], -1, output.size(1))).transpose(
                1, 2
            )
            # torch.nn.functional.fold had some issues ...
            out = output.reshape((output.size(0), output.size(1), H_out, W_out))
        elif self.loop_order == "kn2col":
            out = output.transpose(1, 2).reshape((_input.shape[0], -1, H_out, W_out))
        return out

    def kn2col_input_slice(
        self, input_transformed: Tensor, H_in: int, W_in: int, k_x: int, k_y: int
    ) -> Tensor:
        H_out, W_out = self.get_H_W_out(H_in, W_in)
        if self.padding[0] > 0 or self.padding[1] > 0:  # type: ignore
            input_transformed = torch.nn.functional.pad(
                input_transformed,
                (
                    0,
                    0,
                    self.padding[1],  # type: ignore
                    self.padding[1],
                    self.padding[0],
                    self.padding[0],
                ),
                mode="constant",
                value=0,
            )
        if len(input_transformed.shape) == 3:
            input_transformed = input_transformed.unsqueeze(0)
        #  padding is already accounted for through H_out, W_out
        input_slice = input_transformed[
            :,
            k_x : H_out * self.stride[0] + k_x : self.stride[0],  # type: ignore
            k_y : W_out * self.stride[1] + k_y : self.stride[1],  # type: ignore
            :,
        ].reshape(-1, H_out * W_out, self.in_channels)

        return input_slice

    def forward(self, _input: Tensor) -> Tensor:
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

        if self.halut_active[0] and not self.store_input[0]:
            transformed_input = self.transform_input(_input)

            if self.loop_order == "im2col":
                if len(self.lut_int8.shape) == 1:  # first time lut update
                    self.update_lut()

                ret_tensor = halut_matmul_forward(
                    transformed_input,
                    self.thresholds,
                    self.lut,
                    self.S,
                    self.B,
                    self.lut.size(1),
                    self.lut.size(2),
                    self.dims if not self.use_prototypes else None,
                    self.P if self.use_prototypes else None,
                    temperature=self.temperature,
                    split_factor=self.split_factor,
                    L_int8=self.lut_int8,
                    scale=self.scale,
                )
            elif self.loop_order == "kn2col":
                H_out, W_out = self.get_H_W_out(_input.shape[2], _input.shape[3])
                ret_tensor = torch.zeros(
                    _input.shape[0],
                    H_out * W_out,
                    self.out_channels,
                    device=_input.device,
                    dtype=_input.dtype,
                )
                for k_x in range(self.kernel_size[0]):
                    for k_y in range(self.kernel_size[1]):
                        if self.dilation[0] > 1 or self.dilation[1] > 1:
                            raise NotImplementedError(
                                "dilation > 1 has to be implemented"
                            )
                        input_slice = self.kn2col_input_slice(
                            transformed_input,
                            _input.shape[2],
                            _input.shape[3],
                            k_x,
                            k_y,
                        )
                        input_slice = input_slice.reshape(-1, input_slice.shape[-1])

                        matmul_result = halut_matmul_forward(
                            input_slice,
                            self.thresholds[k_x * self.kernel_size[0] + k_y],
                            self.lut[k_x * self.kernel_size[0] + k_y],
                            self.S,
                            self.B,
                            self.lut.size(-2),
                            self.lut.size(-1),
                            self.dims[k_x * self.kernel_size[0] + k_y]
                            if not self.use_prototypes
                            else None,
                            self.P if self.use_prototypes else None,
                            temperature=self.temperature,
                            split_factor=self.split_factor,
                        )
                        ret_tensor += matmul_result.reshape(
                            _input.shape[0], -1, self.out_channels
                        )

            output = self.transform_output(ret_tensor, _input)
            if self.bias is not None:
                bias = torch.broadcast_to(
                    self.bias.repeat(output.size(-2), output.size(-1)).reshape(
                        (output.size(-3), output.size(-2), output.size(-1))
                    ),
                    (_input.size(0), output.size(-3), output.size(-2), output.size(-1)),
                )
                output = output + bias

            if self.report_error[0]:
                torch_ret = F.conv2d(
                    _input,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                ret_compare_tensor = output.detach().clone()
                if "cuda" in str(_input.device):
                    # pylint: disable=import-outside-toplevel
                    from halutmatmul.cuda.functions import error_cupy

                    res_error = error_cupy(ret_compare_tensor, torch_ret)
                    self.errors.append((_input.shape[0], res_error))  # type: ignore
                else:
                    res_error = error_numpy(ret_compare_tensor, torch_ret)
                    self.errors.append((_input.shape[0], res_error))  # type: ignore
            return output
        else:
            self.check_store_offline(_input)

            if self.padding_mode != "zeros":
                return F.conv2d(
                    F.pad(
                        _input,
                        self._reversed_padding_repeated_twice,
                        mode=self.padding_mode,
                    ),
                    self.weight,
                    self.bias,
                    self.stride,
                    _pair(0),
                    self.dilation,
                    self.groups,
                )
            return F.conv2d(
                _input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
