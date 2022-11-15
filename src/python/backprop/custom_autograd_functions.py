# pylint: disable=abstract-method, arguments-differ
from typing import Optional
import torch
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
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        self_module: Optional[HalutLinear] = None,
    ) -> Tensor:
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
        # TODO: check if weight needs to be transposed
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
    def backward(ctx, grad_output):
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
    @staticmethod
    def forward(
        cxt, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
    ):

        cxt.save_for_backward(input, weight, bias)

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(cxt, grad_output):
        input, weight, bias = cxt.saved_variables

        grad_input = grad_weight = grad_bias = None

        if cxt.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output)

        if cxt.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output)

        if bias is not None and cxt.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias


halutconv2d = Conv2dFunction.apply
