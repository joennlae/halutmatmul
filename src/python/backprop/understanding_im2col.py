import torch
import numpy as np
from models.resnet import resnet18

in_channels = 3
out_channels = 16
kernel_size = 3
stride = 1
batch_size = 2

image_x_y = 8

input = torch.randn(batch_size, in_channels, image_x_y, image_x_y)
kernels = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

print("kernels", kernels.shape)
print("input", input.shape)

output_compare = torch.nn.functional.conv2d(input, kernels, stride=stride)

# manual conv2d implementation
output = torch.zeros(batch_size, out_channels, image_x_y - 2, image_x_y - 2)

for b in range(batch_size):
    for o_c in range(out_channels):
        for o_x in range(image_x_y - 2):
            for o_y in range(image_x_y - 2):
                sum = 0.0
                for i_c in range(in_channels):
                    for k_x in range(kernel_size):
                        for k_y in range(kernel_size):
                            sum += (
                                input[b, i_c, o_x + k_x, o_y + k_y]
                                * kernels[o_c, i_c, k_x, k_y]
                            )
                output[b, o_c, o_x, o_y] = sum

print("output", output[0, 0], output_compare[0, 0])


print("is equal", torch.allclose(output, output_compare, atol=1e-5))

# manual im2col implementation
im2col = torch.zeros(
    batch_size,
    (image_x_y - 2) * (image_x_y - 2),
    kernel_size * kernel_size * in_channels,
)
reshaped_kernels = torch.reshape(
    kernels, (out_channels, kernel_size * kernel_size * in_channels)
).T

for b in range(batch_size):
    for i_c in range(in_channels):
        for o_x in range(image_x_y - 2):
            for o_y in range(image_x_y - 2):
                for k_x in range(kernel_size):
                    for k_y in range(kernel_size):
                        im2col[
                            b,
                            o_x * (image_x_y - 2) + o_y,
                            i_c * kernel_size * kernel_size + k_x * kernel_size + k_y,
                        ] = input[b, i_c, o_x + k_x, o_y + k_y]

print("reshaped_kernels", reshaped_kernels.shape)
print("im2col", im2col[0, 0])
print("im2col shape", im2col.shape)

output_im2col = torch.matmul(im2col, reshaped_kernels)

# im2row use
transposed_check = torch.matmul(reshaped_kernels.T, im2col.transpose(1, 2))
print("transposed_check", transposed_check.shape)

print("shape output_im2col", output_im2col.shape)

output_im2col_reshaped = torch.reshape(
    output_im2col.transpose(1, 2),
    (batch_size, out_channels, image_x_y - 2, image_x_y - 2),
)
output_im2col_transposed = torch.reshape(
    transposed_check, (batch_size, out_channels, image_x_y - 2, image_x_y - 2)
)
print(
    "compare im2col", torch.allclose(output_im2col_reshaped, output_compare, atol=1e-5)
)
print(
    "im2col transposed",
    torch.allclose(output_im2col_transposed, output_compare, atol=1e-5),
)

# check resnet18 im2col sizes
model = resnet18(
    progress=True,
    **{"is_cifar": True, "num_classes": 10},  # type: ignore[arg-type]
)
print(model)

input = torch.randn(32, 3, 32, 32)
output = model(input)
print("output", output.shape)

"""
unfolded shape torch.Size([32, 576, 1024])
unfolded shape torch.Size([32, 576, 1024])
unfolded shape torch.Size([32, 576, 1024])
unfolded shape torch.Size([32, 576, 1024])
unfolded shape torch.Size([32, 576, 256])
unfolded shape torch.Size([32, 1152, 256])
unfolded shape torch.Size([32, 64, 256])
unfolded shape torch.Size([32, 1152, 256])
unfolded shape torch.Size([32, 1152, 256])
unfolded shape torch.Size([32, 1152, 64])
unfolded shape torch.Size([32, 2304, 64])
unfolded shape torch.Size([32, 128, 64])
unfolded shape torch.Size([32, 2304, 64])
unfolded shape torch.Size([32, 2304, 64])
unfolded shape torch.Size([32, 2304, 16])
unfolded shape torch.Size([32, 4608, 16])
unfolded shape torch.Size([32, 256, 16])
unfolded shape torch.Size([32, 4608, 16])
unfolded shape torch.Size([32, 4608, 16])
output torch.Size([32, 10])
"""
