from typing import Any, Optional, Union
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

from maddness.maddness import MaddnessMatmul
from halutmatmul.halutmatmul import HalutMatmul, calc_newaxes_and_newshape_and_old


class HalutLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Union[str, Any] = None,
        dtype: Union[str, Any] = None,
        halut_active: bool = False,
        halut_offline_A: Union[np.ndarray, None] = None,
        halut_C: int = 16,
        halut_lut_work_const: int = -1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.halut_active = halut_active
        self.halut: Optional[MaddnessMatmul] = None
        self.halut_offline_learned = False
        self.halut_offline_A = halut_offline_A
        self.halut_C = halut_C
        self.halut_lut_work_const = halut_lut_work_const

        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

    def learn_offline(self) -> None:
        if self.halut_offline_A is None:
            raise Exception("halut A is None: {}".format(self.halut_offline_A))
        self.halut = MaddnessMatmul(
            C=self.halut_C, lut_work_const=self.halut_lut_work_const
        )
        weights_numpy = self.weight.detach().cpu().numpy().transpose(1, 0)
        print(weights_numpy, weights_numpy.shape)
        print(self.halut_offline_A, self.halut_offline_A.shape)
        self.halut.learn_offline(self.halut_offline_A, weights_numpy)
        self.halut_offline_learned = True

    def check_store_offline(self, _input: Tensor) -> None:
        if (
            self.store_input[0]
            and self.input_storage_a is None
            and self.input_storage_b is None
        ):
            self.input_storage_a = _input.clone()
            self.input_storage_b = self.weight.clone()

    # pylint: disable=W0622
    def forward(self, input: Tensor) -> Tensor:
        self.check_store_offline(input)
        if self.halut_active:
            if not self.halut_offline_learned:
                self.learn_offline()
            input_numpy = input.detach().cpu().numpy()
            print(input_numpy.shape)
            if self.halut:
                result = self.halut.matmul_online(input_numpy)
            print(result.shape)
            bias_to_add = self.bias.clone().repeat(input.shape[0], 1)
            print(bias_to_add.shape)
            return torch.tensor(result) + bias_to_add
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "Halut in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


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
        halut_active: bool = False,
        halut_offline_A: Union[np.ndarray, None] = None,
        halut_C: int = 16,
        halut_lut_work_const: int = -1,
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
            **factory_kwargs
        )
        self.halut_active = halut_active
        self.halut: Optional[MaddnessMatmul] = None
        self.halut_offline_learned = False
        self.halut_offline_A = halut_offline_A
        self.halut_C = halut_C
        self.halut_lut_work_const = halut_lut_work_const

        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.input_storage_a: Optional[Tensor] = None
        self.input_storage_b: Optional[Tensor] = None

    # pylint: disable=R0201
    def halut_conv2d(
        self,
        _input: np.ndarray,
        weights: np.ndarray,
        kernel_size: Union[_int, _size] = (1, 1),
        stride: Union[_int, _size] = (1, 1),
        padding: Union[_int, _size] = 0,
        groups: int = 1,
        bias: Optional[Tensor] = None,
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

        _input = _input.reshape(
            batch_size, groups, cin, _input.shape[2], _input.shape[3]
        )

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
                (input_a_temp, input_b_temp) = HalutMatmul().tensordot(
                    _input_im2col[:, 0],
                    tensor_weights[0],
                    ((1, 4, 5), (1, 2, 3)),
                    return_reshaped_inputs=return_reshaped_inputs,
                )
                input_a[g] += input_a_temp
                input_b[g] += input_b_temp
            return (input_a, input_b)
        else:
            for g in range(groups):
                ret[:, g] += HalutMatmul().tensordot(
                    _input_im2col[:, g], tensor_weights[g], ((1, 4, 5), (1, 2, 3))
                )

        ret = np.moveaxis(ret, 4, 2).reshape(batch_size, cout, out_y, out_x)

        assert bias is None
        # TODO: bias not used at the moment so probably false
        return ret if bias is None else np.add(ret, bias)  # type: ignore[arg-type]

    def conv2d(
        self,
        _input: Tensor,
        weight: Tensor,
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
        bias_numpy = bias.detach().cpu().numpy() if bias else None

        ret_numpy = self.halut_conv2d(
            input_numpy,
            weights_numpy,
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
            assert bias is None
            (input_a, input_b) = self.conv2d(
                _input,
                weight,
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
        if not self.halut_active:
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
        else:
            if self.padding_mode != "zeros":
                raise Exception("padding_mode != zeros not supported with Halut")
            else:
                return self.conv2d( # type: ignore[return-value]
                    _input,
                    weight,
                    bias,
                    self.stride,
                    self.padding,  # type: ignore[arg-type]
                    self.dilation,
                    self.groups,
                )

    def forward(self, _input: Tensor) -> Tensor:
        return self._conv_forward(_input, self.weight, self.bias)
