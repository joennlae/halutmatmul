# pylint: disable=abstract-method, arguments-differ, import-outside-toplevel
# type: ignore # mypy has issues with something in here
from typing import Optional, Any
import torch
import numpy as np
from torch.functional import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from halutmatmul.modules import HalutLinear

# torch.autograd.Function
# source https://pytorch.org/docs/master/notes/extending.html#example
# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(
        ctx: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        self_module: Optional[HalutLinear] = None,
    ) -> Any:
        ctx.save_for_backward(input, weight, bias)
        if self_module is None or not self_module.halut_active[0]:
            output = input.mm(weight.t())

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output
        # halut
        if self_module is None:
            raise Exception("self_model is None when halut active!!")

        # new lut calculated
        # weight is transposed because pytorch stores it transposed
        # TODO: CPU takes LUT stored in HalutMatmul object
        torch_res = torch.tensordot(
            self_module.prototypes, weight.t(), dims=([2], [0])
        ).permute((2, 0, 1))
        self_module.lut = Parameter(
            torch_res,
            requires_grad=False,
        )
        return self_module.linear_halut(input)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


halutlinear = LinearFunction.apply


class Conv2dFunction(torch.autograd.Function):
    # pylint: disable=line-too-long
    # source: https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7

    @staticmethod
    def forward(
        cxt: Any,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        self_module: Optional[HalutLinear] = None,
    ) -> Tensor:

        confs = torch.from_numpy(np.array([stride, padding, dilation, groups]))
        cxt.save_for_backward(input, weight, bias, confs)

        if self_module is None:
            raise Exception("self_module is None")

        if len(self_module.prototypes.shape) > 1:
            if "cuda" in str(input.device):
                from halutmatmul.cuda.functions import halut_conv2d_gpu

                _, weights_reshaped = halut_conv2d_gpu(
                    input,
                    weight,
                    L=None,
                    H=None,
                    read_acc_lut_kernel=None,
                    encode_kernel=None,
                    kernel_size=self_module.kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                    return_reshaped_inputs=True,
                )
            else:
                from halutmatmul.modules import halut_conv2d_cpu

                input_numpy = input.detach().cpu().numpy()
                weights_numpy = weight.detach().cpu().numpy()
                bias_numpy = bias.detach().cpu().numpy() if bias is not None else None

                ret_numpy = halut_conv2d_cpu(
                    input_numpy,
                    weights_numpy,
                    self_module.halut,
                    self_module.kernel_size,
                    stride,
                    padding,
                    groups,
                    bias_numpy,
                    return_reshaped_inputs=True,
                )
                weights_reshaped = torch.from_numpy(ret_numpy[1]).to(torch.float32)

            # TODO: dont update it on every call do it between epochs
            print("update lut", weights_reshaped.shape, self_module.prototypes.shape)
            torch_res = torch.tensordot(
                self_module.prototypes, weights_reshaped, dims=([2], [0])
            ).permute((2, 0, 1))
            self_module.lut = Parameter(
                torch_res,
                requires_grad=False,
            )
        # pylint: disable=protected-access
        return self_module._conv_forward(input, weight, bias)

    @staticmethod
    def backward(cxt: Any, grad_output: Tensor) -> tuple[Optional[Tensor], ...]:
        input, weight, bias, conf = cxt.saved_variables
        confs = conf.detach().cpu().numpy()
        stride, padding, dilation, groups = confs[0], confs[1], confs[2], confs[3]

        grad_input = grad_weight = grad_bias = None

        if cxt.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, stride, padding, dilation, groups
            )

        if cxt.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, stride, padding, dilation, groups
            )

        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


halutconv2d = Conv2dFunction.apply
