# pylint: disable=chained-comparison, use-a-generator
from queue import Queue
import torch
import numpy as np


# estimate 3x3 im2col unit
kernel_size = 3
N = 1
H = 16
W = 16
IN_CHANNEL = 16
OUT_CHANNEL = 64

random_array = np.random.rand(N, H, W, IN_CHANNEL) * 2 - 1

random_weights = (
    np.random.rand(IN_CHANNEL, kernel_size, kernel_size, OUT_CHANNEL) * 2 - 1
).reshape((-1, OUT_CHANNEL))

# im2col of random array
L = (H - 2) * (W - 2)
im2col_array = np.zeros(
    (N, L, IN_CHANNEL * kernel_size * kernel_size), dtype=np.float32
)

# buffer approach
buffer = [[] for _ in range(IN_CHANNEL)]
im2col_backup = im2col_array
im2col_array = np.zeros_like(im2col_array)
print(im2col_array)
output_counter = 0
l_counter = 0
n_counter = 0
c_counter = 0
max_len = 0
for n in range(N):
    for h in range(H):
        for w in range(W):
            for c in range(IN_CHANNEL):
                buffer[c].append(random_array[n, h, w, c])
                if len(buffer[c]) > max_len:
                    max_len = len(buffer[c])
                if w >= 2 and h >= 2:
                    print("len buffer", len(buffer[c]))
                    # check for overwrite
                    if not (
                        im2col_array[n_counter, output_counter, c * 9 : c * 9 + 9] == 0
                    ).all():
                        raise ValueError("overwrite detected")
                    im2col_array[n_counter, output_counter, c * 9 + 0] = buffer[c][0]
                    im2col_array[n_counter, output_counter, c * 9 + 1] = buffer[c][1]
                    im2col_array[n_counter, output_counter, c * 9 + 2] = buffer[c][2]
                    im2col_array[n_counter, output_counter, c * 9 + 3] = buffer[c][W]
                    im2col_array[n_counter, output_counter, c * 9 + 4] = buffer[c][
                        W + 1
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 5] = buffer[c][
                        W + 2
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 6] = buffer[c][
                        2 * W
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 7] = buffer[c][
                        2 * W + 1
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 8] = buffer[c][
                        2 * W + 2
                    ]
                    buffer[c].pop(0)
                    if c == IN_CHANNEL - 1:
                        output_counter += 1
                        if (output_counter % (W - (kernel_size - 1))) == 0:
                            for i in range(IN_CHANNEL):
                                for _ in range(kernel_size - 1):
                                    buffer[i].pop(0)
                    if output_counter == L:
                        output_counter = 0
                        n_counter += 1
                    if n_counter == N:
                        n_counter = 0
                        print("resetting n_counter")


print(im2col_array)
print(im2col_array.shape)
print(im2col_backup)
print(im2col_backup.shape)
print(buffer)
print(max_len)
torch_random = torch.from_numpy(random_array).permute(0, 3, 1, 2).to(torch.float32)
# unfold
unfold = torch.nn.Unfold(kernel_size=3, padding=0, stride=1)
unfolded = unfold(torch_random).permute(0, 2, 1)
print(unfolded.shape)
print(unfolded)
unfolded_numpy = unfolded.numpy()
# allclose
print("torch correctness", np.allclose(im2col_array, unfolded_numpy, atol=1e-3))
diff = np.abs(im2col_array - unfolded_numpy)
print(diff)
print("dtypes", im2col_array.dtype, unfolded_numpy.dtype)
out_conv2d = np.zeros((N, L, OUT_CHANNEL), dtype=np.float32)
