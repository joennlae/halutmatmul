import torch

from torch.autograd import gradcheck
from backprop.custom_autograd_functions import halutlinear, halutconv2d

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (
    torch.randn(20, 20, dtype=torch.double, requires_grad=True),
    torch.randn(30, 20, dtype=torch.double, requires_grad=True),
)
test = gradcheck(halutlinear, input, eps=1e-6, atol=1e-4)
print(test)
input_conv = (
    torch.randn(6, 6, 32, 32, dtype=torch.double, requires_grad=True),
    torch.randn(6, 6, 32, 32, dtype=torch.double, requires_grad=True),
    None,
    1,
    0,
    1,
    1,
)
test = gradcheck(halutconv2d, input_conv, eps=1e-6, atol=1e-4)
print(test)
