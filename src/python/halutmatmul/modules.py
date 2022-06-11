# pylint: disable=import-outside-toplevel
from typing import Any, Optional, OrderedDict, Union
from timeit import default_timer as timer
import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules import Linear

from torch.nn.common_types import _size_2_t
from torch.types import _int, _size
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_any_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.parameter import Parameter

from halutmatmul.halutmatmul import EncodingAlgorithm, HalutConfig, HalutMatmul
from halutmatmul.functions import (
    calc_newaxes_and_newshape_and_old,
    tensordot,
)


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
        self.hash_buckets_or_prototypes = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.lut = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.halut_config = Parameter(
            torch.zeros(HalutConfig.MAX, dtype=torch.float32), requires_grad=False
        )
        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.report_error = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        # only for FULL PQ
        self.prototypes = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.errors = [(-1, np.zeros(ErrorTuple.MAX, dtype=np.float64))]

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
                prefix + "hash_buckets_or_prototypes",
                prefix + "lut",
                prefix + "halut_config",
            )
        ):
            if not state_dict[prefix + "halut_active"]:
                return
            # hack to support variable parameter size --> with the cost of double copying :-)
            self.hash_buckets_or_prototypes = Parameter(
                state_dict[prefix + "hash_buckets_or_prototypes"]
                .clone()
                .to(str(self.weight.device)),
                requires_grad=False,
            )
            self.lut = Parameter(
                state_dict[prefix + "lut"].clone().to(str(self.weight.device)),
                requires_grad=False,
            )
            print("STATE_DICT", self.weight.device)
            if "cuda" in str(self.weight.device):
                # pylint: disable=import-outside-toplevel, attribute-defined-outside-init
                from halutmatmul.cuda.kernels import create_kernels_halutmatmul

                C = self.lut.shape[1]
                K = self.lut.shape[2]
                (
                    self.encode_kernel,
                    self.read_acc_lut_kernel,
                ) = create_kernels_halutmatmul(
                    C=C,
                    K=K,
                    encoding_algorithm=int(
                        state_dict[prefix + "halut_config"][
                            HalutConfig.ENCODING_ALGORITHM
                        ]
                    ),
                )
            else:
                store_array = np.array(
                    [
                        state_dict[prefix + "hash_buckets_or_prototypes"]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy(),
                        state_dict[prefix + "lut"].clone().detach().cpu().numpy(),
                        state_dict[prefix + "halut_config"]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy(),
                        state_dict[prefix + "hash_buckets_or_prototypes"]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy(),
                    ],
                    dtype=object,
                )
                self.halut = HalutMatmul().from_numpy(store_array)
        elif any(
            k in state_dict.keys()
            for k in (
                prefix + "hash_buckets_or_prototypes",
                prefix + "lut",
                prefix + "halut_config",
            )
        ):
            raise Exception(
                f"not all '{prefix}hash_buckets_or_prototypes', '{prefix}lut', "
                "'{prefix}halut_config' paramters in state_dict"
            )

    def check_store_offline(self, _input: Tensor) -> None:
        if self.store_input[0]:
            if self.input_storage_a is None and self.input_storage_b is None:
                self.input_storage_a = (
                    _input.clone()
                    .cpu()
                    .detach()
                    .reshape(
                        (
                            -1,
                            _input.shape[2]
                            if len(_input.shape) > 2
                            else _input.shape[1],
                        )
                    )
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
                    0,
                )

    def linear_halut(self, _input: Tensor) -> Tensor:
        if "cuda" in str(self.weight.device):
            if self.encode_kernel is None:
                raise Exception("Kernels not compiled")
            from halutmatmul.cuda.functions import halut_linear_gpu

            ret_tensor = halut_linear_gpu(
                _input.reshape((-1, _input.shape[-1])),
                self.encode_kernel,
                self.read_acc_lut_kernel,
                self.lut,
                self.hash_buckets_or_prototypes,
            )
        else:
            if self.halut is None:
                raise Exception("self.halut is None")
            input_numpy = _input.detach().cpu().numpy().reshape(-1, _input.shape[-1])
            ret_tensor = torch.from_numpy(self.halut.matmul_online(input_numpy)).to(
                str(_input.device)
            )

        ret_tensor = ret_tensor.reshape(
            (*_input.shape[:-1], self.weight.shape[0])
        ).squeeze()
        if self.bias is not None:
            print(self.bias.shape)
            print("repeats", (*ret_tensor.shape[:-1], 1))
            bias_to_add = self.bias.clone().t().repeat(*(*ret_tensor.shape[:-1], 1))
            ret_tensor += bias_to_add
        if self.report_error[0]:
            torch_ret = F.linear(_input, self.weight, self.bias)
            if "cuda" in str(_input.device):
                from halutmatmul.cuda.functions import error_cupy

                res_error = error_cupy(ret_tensor, torch_ret)
                self.errors.append((_input.shape[0], res_error))
            else:
                res_error = error_numpy(ret_tensor, torch_ret)
                self.errors.append((_input.shape[0], res_error))

        return ret_tensor

    # pylint: disable=W0622
    def forward(self, input: Tensor) -> Tensor:
        self.check_store_offline(input)
        if self.halut_active[0]:
            return self.linear_halut(input)
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "Halut in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def halut_conv2d_cpu(
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
                group=g,
                groups=groups,
            )  # halut does not need to be passed
            input_a[g] += input_a_temp
            input_b[g] += input_b_temp
            print(
                "SHAPESSSS",
                input_a.shape,
                input_a_temp.shape,
                input_b.shape,
                input_b_temp.shape,
            )
        return (
            input_a.reshape((-1, input_a.shape[2])),
            input_b.reshape((input_b.shape[1], -1)),
        )
    else:
        for g in range(groups):
            ret[:, g] += tensordot(
                _input_im2col[:, g],
                tensor_weights[g],
                ((1, 4, 5), (1, 2, 3)),
                halut=halut,
                group=g,
                groups=groups,
            )

    ret = np.moveaxis(ret, 4, 2).reshape(batch_size, cout, out_y, out_x)

    if bias is not None:
        bias = np.broadcast_to(
            np.repeat(bias, out_y * out_x).reshape((cout, out_y, out_x)),
            (batch_size, cout, out_y, out_x),
        )
        ret = ret + bias
    return ret


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
        self.hash_buckets_or_prototypes = Parameter(torch.zeros(1), requires_grad=False)
        self.lut = Parameter(torch.zeros(1), requires_grad=False)
        self.halut_config = Parameter(
            torch.zeros(HalutConfig.MAX, dtype=torch.float32), requires_grad=False
        )
        self.store_input = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.report_error = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        # only for FULL PQ
        self.prototypes = Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.errors = [(-1, np.zeros(ErrorTuple.MAX, dtype=np.float64))]

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
                prefix + "hash_buckets_or_prototypes",
                prefix + "lut",
                prefix + "halut_config",
            )
        ):
            if not state_dict[prefix + "halut_active"]:
                return
            # hack to support variable parameter size --> with the cost of double copying :-)
            self.hash_buckets_or_prototypes = Parameter(
                state_dict[prefix + "hash_buckets_or_prototypes"]
                .clone()
                .to(str(self.weight.device)),
                requires_grad=False,
            )
            self.lut = Parameter(
                state_dict[prefix + "lut"].clone().to(str(self.weight.device)),
                requires_grad=False,
            )

            if "cuda" in str(self.weight.device):
                # pylint: disable=import-outside-toplevel, attribute-defined-outside-init
                from halutmatmul.cuda.kernels import create_kernels_halutmatmul

                C = self.lut.shape[1]
                K = self.lut.shape[2]
                (
                    self.encode_kernel,
                    self.read_acc_lut_kernel,
                ) = create_kernels_halutmatmul(
                    C=C,
                    K=K,
                    encoding_algorithm=int(
                        state_dict[prefix + "halut_config"][
                            HalutConfig.ENCODING_ALGORITHM
                        ]
                    ),
                )
            else:
                store_array = np.array(
                    [
                        state_dict[prefix + "hash_buckets_or_prototypes"]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy(),
                        state_dict[prefix + "lut"].clone().detach().cpu().numpy(),
                        state_dict[prefix + "halut_config"]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy(),
                        state_dict[prefix + "hash_buckets_or_prototypes"]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy(),
                    ],
                    dtype=object,
                )
                self.halut = HalutMatmul().from_numpy(store_array)
        elif any(
            k in state_dict.keys()
            for k in (
                prefix + "hash_buckets_or_prototypes",
                prefix + "lut",
                prefix + "halut_config",
            )
        ):
            raise Exception(
                f"not all '{prefix}hash_buckets_or_prototypes', '{prefix}lut', "
                "'{prefix}halut_config' paramters in state_dict"
            )

    def conv2d(
        self,
        _input: Tensor,
        weight: Tensor,
        halut: HalutMatmul,
        bias: Optional[Tensor] = None,
        stride: _size_any_t = 1,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        groups: _int = 1,
        return_reshaped_inputs: bool = False,  # needed for storage
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        if "cuda" in str(_input.device):
            if self.halut_active[0] and any(
                not hasattr(self, x) for x in ("encode_kernel", "read_acc_lut_kernel")
            ):
                raise Exception("CUDA kernels not defined or loaded!")
            # pylint: disable=import-outside-toplevel
            from halutmatmul.cuda.functions import halut_conv2d_gpu

            ret_tensor = halut_conv2d_gpu(
                _input=_input,
                weights=weight,
                encode_kernel=self.encode_kernel if self.halut_active[0] else None,
                read_acc_lut_kernel=self.read_acc_lut_kernel
                if self.halut_active[0]
                else None,
                L=self.lut,
                H=self.hash_buckets_or_prototypes,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=self.bias,
                return_reshaped_inputs=return_reshaped_inputs,
            )
            if return_reshaped_inputs:
                return ret_tensor[0], ret_tensor[1]
            return ret_tensor
        else:
            input_numpy = _input.detach().cpu().numpy()
            weights_numpy = weight.detach().cpu().numpy()
            bias_numpy = bias.detach().cpu().numpy() if bias is not None else None

            ret_numpy = halut_conv2d_cpu(
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
                return torch.from_numpy(ret_numpy).to(str(_input.device))

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
        self, _input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> None:
        if self.store_input[0]:
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

    def _conv_forward(
        self, _input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        self.check_store_offline(_input, weight, bias)
        if self.halut_active[0]:
            if self.padding_mode != "zeros":
                raise Exception("padding_mode != zeros not yet supported with Halut")
            elif "cpu" in str(self.weight.device) and self.halut is None:
                raise Exception("halut is not set")
            else:
                ret_tensor = self.conv2d(
                    _input,
                    weight,
                    self.halut,  # type: ignore[arg-type]
                    bias,
                    self.stride,
                    self.padding,  # type: ignore[arg-type]
                    self.dilation,
                    self.groups,
                )
                if self.report_error[0]:
                    torch_ret = F.conv2d(
                        _input,
                        weight,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                    if "cuda" in str(_input.device):
                        # pylint: disable=import-outside-toplevel
                        from halutmatmul.cuda.functions import error_cupy

                        res_error = error_cupy(
                            ret_tensor, torch_ret  # type: ignore[arg-type]
                        )
                        self.errors.append((_input.shape[0], res_error))
                    else:
                        res_error = error_numpy(
                            ret_tensor, torch_ret  # type: ignore[arg-type]
                        )
                        self.errors.append((_input.shape[0], res_error))
                return ret_tensor  # type: ignore[return-value]
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
