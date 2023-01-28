import torch
import numpy as np

in_channels = 3
out_channels = 16
kernel_size = 3
stride = 1
batch_size = 1

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
