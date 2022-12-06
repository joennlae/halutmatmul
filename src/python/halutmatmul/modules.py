# pylint: disable=import-outside-toplevel
import sys
import math
from typing import Any, Optional, OrderedDict, Union
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
    offset = 0
    bit_matrix_numpy = np.array(
        [
            [
                offset,
                offset,
                offset,
                offset,
                offset,
                offset,
                offset,
                offset,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            [offset, offset, offset, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, offset, offset, offset, offset, 1, 1, 1, 1],
            [offset, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, offset, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, offset, offset, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, offset, 1, 1],
            [offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, offset, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, 1],
        ]
    )
    bit_matrix_base = torch.from_numpy(bit_matrix_numpy.T).to(dtype)
    bit_matrix = torch.ones((C * K, C * (K - 1)), dtype=dtype)
    for c in range(C):
        bit_matrix[
            c * K : (c + 1) * K,
            c * (K - 1) : (c + 1) * (K - 1),
        ] = bit_matrix_base
    return bit_matrix


def halut_matmul_forward(
    input: torch.Tensor,
    T: torch.Tensor,
    L: torch.Tensor,
    dims: torch.Tensor,
    S: torch.Tensor,
    B: torch.Tensor,
    C: int = 32,
    K: int = 16,
    split_factor: int = 4,
) -> torch.Tensor:
    # encoding
    h = S.mm(input[:, dims].T) - T.unsqueeze(1)
    b = B.mm(h.relu())
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
    # # for m in range(L.size(0)):
    # #     result[:, m] += (E * L[m].repeat((E.shape[0], 1, 1))).sum(dim=2).sum(dim=1)
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
    # result = torch.einsum("nij, kij -> nki", [E, L])
    # result = result.sum(dim=2)
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
        split_factor: int = 4,
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
        self.errors = [(-1, np.zeros(ErrorTuple.MAX, dtype=np.float64))]

        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

        self.split_factor = split_factor
        self._register_load_state_dict_pre_hook(self.state_dict_hook)

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
                self.dims,
                self.S,
                self.B,
                self.lut.size(1),
                self.lut.size(2),
                self.split_factor,
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
        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

        self.split_factor = split_factor
        self._register_load_state_dict_pre_hook(self.state_dict_hook)

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

    def check_store_offline(self, _input: Tensor) -> None:
        if self.store_input[0]:
            input_a = self.transform_input(_input)
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
        return unfolded

    def transform_weight(self, weight: Tensor) -> Tensor:
        # weight is passed that the trasnfrom can also be used from test etc.
        weights_prepared = weight.view(weight.size(0), -1).t()
        return weights_prepared

    def transform_output(self, output: Tensor, _input: Tensor) -> Tensor:
        output = output.reshape((_input.shape[0], -1, output.size(1))).transpose(1, 2)
        # reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H_out, W_out = (
            math.floor(
                (
                    _input.shape[2]
                    + 2 * self.padding[0]  # type: ignore[arg-type]
                    - self.dilation[0] * (self.kernel_size[0] - 1)
                    - 1
                )
                / self.stride[0]
                + 1
            ),
            math.floor(
                (
                    _input.shape[3]
                    + 2 * self.padding[1]  # type: ignore[arg-type]
                    - self.dilation[1] * (self.kernel_size[1] - 1)
                    - 1
                )
                / self.stride[0]
                + 1
            ),
        )

        # torch.nn.functional.fold had some issues ...
        out = output.reshape((output.size(0), output.size(1), H_out, W_out))
        return out

    def forward(self, _input: Tensor) -> Tensor:
        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html

        if self.halut_active[0] and not self.store_input[0]:
            transformed_input = self.transform_input(_input)

            ret_tensor = halut_matmul_forward(
                transformed_input,
                self.thresholds,
                self.lut,
                self.dims,
                self.S,
                self.B,
                self.lut.size(1),
                self.lut.size(2),
                self.split_factor
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
