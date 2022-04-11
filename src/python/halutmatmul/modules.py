from typing import Any, Optional, OrderedDict, Union
import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules import Linear

from torch.nn.common_types import _size_2_t
from torch.types import _int, _size
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter

from halutmatmul.halutmatmul import (
    HalutMatmul,
    calc_newaxes_and_newshape_and_old,
    tensordot,
)


class HalutLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Union[str, Any] = None,
        dtype: Union[str, Any] = None,
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
        self.hash_buckets = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.lut = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.lut_offset_scale = Parameter(
            torch.zeros(2, dtype=torch.float32), requires_grad=False
        )
        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )

        self.halut: Optional[HalutMatmul] = None

        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

        self._register_load_state_dict_pre_hook(self.state_dict_hook)

    def state_dict_hook(
        self, state_dict: "OrderedDict[str, Tensor]", prefix: str, *_: Any
    ) -> None:
        if all(
            k in state_dict.keys()
            for k in (
                prefix + "hash_buckets",
                prefix + "lut",
                prefix + "lut_offset_scale",
            )
        ):
            # hack to support variable parameter size --> with the cost of double copying :-)
            self.hash_buckets = Parameter(
                state_dict[prefix + "hash_buckets"].clone(), requires_grad=False
            )
            self.lut = Parameter(
                state_dict[prefix + "lut"].clone(), requires_grad=False
            )
            store_array = np.array(
                [
                    state_dict[prefix + "hash_buckets"].clone().detach().cpu().numpy(),
                    state_dict[prefix + "lut"].clone().detach().cpu().numpy(),
                    state_dict[prefix + "lut_offset_scale"].clone().detach().cpu().numpy(),
                ],
                dtype=object,
            )
            self.halut = HalutMatmul().from_numpy(store_array)
        elif any(
            k in state_dict.keys()
            for k in (
                prefix + "hash_buckets",
                prefix + "lut",
                prefix + "lut_offset_scale",
            )
        ):
            raise Exception(
                f"not all '{prefix}hash_buckets', '{prefix}lut', '{prefix}lut_offset_scale' "
                "paramters in state_dict"
            )

    def check_store_offline(self, _input: Tensor) -> None:
        if (
            self.store_input[0]
            and self.input_storage_a is None
            and self.input_storage_b is None
        ):
            self.input_storage_a = _input.clone()
            self.input_storage_b = self.weight.clone().transpose(1, 0)

    # pylint: disable=W0622
    def forward(self, input: Tensor) -> Tensor:
        self.check_store_offline(input)
        if self.halut_active[0]:
            input_numpy = input.detach().cpu().numpy()
            if self.halut is None:
                raise Exception("self.halut is None")
            result = torch.from_numpy(self.halut.matmul_online(input_numpy))
            if self.bias is not None:
                bias_to_add = self.bias.clone().repeat(input.shape[0], 1)
                result += bias_to_add
            return result
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "Halut in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


# pylint: disable=R0201
def halut_conv2d(
    _input: np.ndarray,
    weights: np.ndarray,
    halut: HalutMatmul,
    kernel_size: Union[_int, _size] = (1, 1),
    stride: Union[_int, _size] = (1, 1),
    padding: Union[_int, _size] = 0,
    groups: int = 1,
    bias: Optional[np.ndarray] = None,
    return_reshaped_inputs: bool = False,  # needed for storage
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    kernel_size = (
        (kernel_size, kernel_size)
        if isinstance(kernel_size, int)
        else (kernel_size[0], kernel_size[1])
    )
    stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    padding = (
        (padding,) * 4
        if isinstance(padding, int)
        else (padding[0], padding[0], padding[1], padding[1])
    )

    if padding[0] > 0 or padding[2] > 0:
        _input = _input[
            :,
            :,
            -padding[2] : _input.shape[2] + padding[3],
            -padding[0] : _input.shape[3] + padding[1],
        ]
    # pylint: disable=C0301
    # inspiration https://github.com/geohot/tinygrad/blob/7ad60eb8b21a3a1f1f538b6e9f216a03d8267e74/tinygrad/ops/ops_cpu.py#L167
    cout, cin, H, W = weights.shape
    stride_x, stride_y = stride
    batch_size, cin_ = _input.shape[0], _input.shape[1]
    out_y, out_x = (
        (_input.shape[2] - (H - stride_y)) // stride_y,
        (_input.shape[3] - (W - stride_x)) // stride_x,
    )
    assert cin * groups == cin_
    assert cout % groups == 0
    rcout = cout // groups

    _input = _input.reshape(batch_size, groups, cin, _input.shape[2], _input.shape[3])

    # im2col
    _input_im2col = np.lib.stride_tricks.as_strided(
        _input,
        shape=(batch_size, groups, cin, out_y, out_x, H, W),
        strides=(
            *_input.strides[0:3],
            _input.strides[3] * stride_y,
            _input.strides[4] * stride_x,
            *_input.strides[3:5],
        ),
        writeable=False,
    )
    tensor_weights = weights.reshape(groups, rcout, cin, H, W)

    ret = np.zeros((batch_size, groups, out_y, out_x, rcout), dtype=_input.dtype)

    if return_reshaped_inputs:
        (_, _, newshape_a, newshape_b, _, _,) = calc_newaxes_and_newshape_and_old(
            _input_im2col[:, 0], tensor_weights[0], ((1, 4, 5), (1, 2, 3))
        )
        input_a = np.zeros((groups, *newshape_a))
        input_b = np.zeros((groups, *newshape_b))
        for g in range(groups):
            (input_a_temp, input_b_temp) = tensordot(
                _input_im2col[:, 0],
                tensor_weights[0],
                ((1, 4, 5), (1, 2, 3)),
                return_reshaped_inputs=return_reshaped_inputs,
            )  # halut does not need to be passed
            input_a[g] += input_a_temp
            input_b[g] += input_b_temp
        return (input_a[0], input_b[0])
    else:
        for g in range(groups):
            ret[:, g] += tensordot(
                _input_im2col[:, g],
                tensor_weights[g],
                ((1, 4, 5), (1, 2, 3)),
                halut=halut,
            )

    ret = np.moveaxis(ret, 4, 2).reshape(batch_size, cout, out_y, out_x)

    if bias is not None:
        bias = np.broadcast_to(
            np.repeat(bias, out_y * out_x).reshape((cout, out_y, out_x)),
            (batch_size, cout, out_y, out_x),
        )
        ret = ret + bias
    return ret


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
        self.hash_buckets = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.lut = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.lut_offset_scale = Parameter(
            torch.zeros(2, dtype=torch.float32), requires_grad=False
        )
        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )

        self.halut: Optional[HalutMatmul] = None

        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

        self._register_load_state_dict_pre_hook(self.state_dict_hook)

    def state_dict_hook(
        self, state_dict: "OrderedDict[str, Tensor]", prefix: str, *_: Any
    ) -> None:
        if all(
            k in state_dict.keys()
            for k in (
                prefix + "hash_buckets",
                prefix + "lut",
                prefix + "lut_offset_scale",
            )
        ):
            # hack to support variable parameter size --> with the cost of double copying :-)
            self.hash_buckets = Parameter(
                state_dict[prefix + "hash_buckets"].clone(), requires_grad=False
            )
            self.lut = Parameter(
                state_dict[prefix + "lut"].clone(), requires_grad=False
            )
            store_array = np.array(
                [
                    state_dict[prefix + "hash_buckets"].clone().detach().cpu().numpy(),
                    state_dict[prefix + "lut"].clone().detach().cpu().numpy(),
                    state_dict[prefix + "lut_offset_scale"].clone().detach().cpu().numpy(),
                ],
                dtype=object,
            )
            self.halut = HalutMatmul().from_numpy(store_array)
        elif any(
            k in state_dict.keys()
            for k in (
                prefix + "hash_buckets",
                prefix + "lut",
                prefix + "lut_offset_scale",
            )
        ):
            raise Exception(
                f"not all '{prefix}hash_buckets', '{prefix}lut', '{prefix}lut_offset_scale' "
                "paramters in state_dict"
            )

    def conv2d(
        self,
        _input: Tensor,
        weight: Tensor,
        halut: HalutMatmul,
        bias: Optional[Tensor] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        groups: _int = 1,
        return_reshaped_inputs: bool = False,  # needed for storage
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        assert dilation in (1, (1, 1))

        input_numpy = _input.detach().cpu().numpy()
        weights_numpy = weight.detach().cpu().numpy()
        bias_numpy = bias.detach().cpu().numpy() if bias is not None else None

        ret_numpy = halut_conv2d(
            input_numpy,
            weights_numpy,
            halut,
            self.kernel_size,
            stride,
            padding,
            groups,
            bias_numpy,
            return_reshaped_inputs=return_reshaped_inputs,
        )

        if return_reshaped_inputs:
            return (torch.from_numpy(ret_numpy[0]), torch.from_numpy(ret_numpy[1]))
        else:
            return torch.from_numpy(ret_numpy)

    def check_store_offline(
        self, _input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> None:
        if (
            self.store_input[0]
            and self.input_storage_a is None
            and self.input_storage_b is None
        ):
            (input_a, input_b) = self.conv2d(
                _input,
                weight,
                HalutMatmul(),
                bias,
                self.stride,
                self.padding,  # type: ignore[arg-type]
                self.dilation,
                self.groups,
                return_reshaped_inputs=True,
            )
            self.input_storage_a = input_a.clone()
            self.input_storage_b = input_b.clone()

    def _conv_forward(
        self, _input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        self.check_store_offline(_input, weight, bias)
        if self.halut_active[0]:
            if self.padding_mode != "zeros":
                raise Exception("padding_mode != zeros not supported with Halut")
            elif self.halut is None:
                raise Exception("halut is not set")
            else:
                return self.conv2d(  # type: ignore[return-value]
                    _input,
                    weight,
                    self.halut,
                    bias,
                    self.stride,
                    self.padding,  # type: ignore[arg-type]
                    self.dilation,
                    self.groups,
                )
        else:
            if self.padding_mode != "zeros":
                return F.conv2d(
                    F.pad(
                        _input,
                        self._reversed_padding_repeated_twice,
                        mode=self.padding_mode,
                    ),
                    weight,
                    bias,
                    self.stride,
                    _pair(0),
                    self.dilation,
                    self.groups,
                )
            return F.conv2d(
                _input,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def forward(self, _input: Tensor) -> Tensor:
        return self._conv_forward(_input, self.weight, self.bias)
