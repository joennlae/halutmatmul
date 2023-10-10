# pylint: disable=chained-comparison, use-a-generator
from queue import Queue
import torch
import numpy as np


# estimate 3x3 im2col unit
kernel_size = 3
N = 1
H = 8
W = 8
IN_CHANNEL = 16
OUT_CHANNEL = 64
padding = 1

random_array = np.random.rand(N, H, W, IN_CHANNEL) * 2 - 1

# padding array
padded_array = np.zeros((N, H + padding * 2, W + padding * 2, IN_CHANNEL))
padded_array[:, padding : H + padding, padding : W + padding, :] = random_array
random_array = padded_array

random_weights = (
    np.random.rand(IN_CHANNEL, kernel_size, kernel_size, OUT_CHANNEL) * 2 - 1
).reshape((-1, OUT_CHANNEL))

# im2col of random array
L = (H - 2 + padding * 2) * (W - 2 + padding * 2)
print("L", L)
im2col_array = np.zeros(
    (N, L, IN_CHANNEL * kernel_size * kernel_size), dtype=np.float32
)

# buffer approach
buffer = [[] for _ in range(IN_CHANNEL)]
output_counter = 0
l_counter = 0
n_counter = 0
c_counter = 0
max_len = 0


def pop(array, buffers, n_counter, output_counter, c, W=W):
    # check for overwrite
    print("c", c, "buffers[c]", buffers[c], n_counter, output_counter, c * 9)
    if not (array[n_counter, output_counter, c * 9 : c * 9 + 9] == 0).all():
        raise ValueError("overwrite detected")
    array[n_counter, output_counter, c * 9 + 0] = buffers[c][0]
    array[n_counter, output_counter, c * 9 + 1] = buffers[c][1]
    array[n_counter, output_counter, c * 9 + 2] = buffers[c][2]
    array[n_counter, output_counter, c * 9 + 3] = buffers[c][W]
    array[n_counter, output_counter, c * 9 + 4] = buffers[c][W + 1]
    array[n_counter, output_counter, c * 9 + 5] = buffers[c][W + 2]
    array[n_counter, output_counter, c * 9 + 6] = buffers[c][2 * W]
    array[n_counter, output_counter, c * 9 + 7] = buffers[c][2 * W + 1]
    array[n_counter, output_counter, c * 9 + 8] = buffers[c][2 * W + 2]
    buffers[c].pop(0)


h_padded = H + padding * 2
w_padded = W + padding * 2
for n in range(N):
    for h in range(h_padded):
        for w in range(w_padded):
            for c in range(IN_CHANNEL):
                buffer[c].append(random_array[n, h, w, c])
                if len(buffer[c]) > max_len:
                    max_len = len(buffer[c])
                if w >= 2 and h >= 2:
                    # check for overwrite
                    if not (
                        im2col_array[n_counter, output_counter, c * 9 : c * 9 + 9] == 0
                    ).all():
                        raise ValueError("overwrite detected")
                    im2col_array[n_counter, output_counter, c * 9 + 0] = buffer[c][0]
                    im2col_array[n_counter, output_counter, c * 9 + 1] = buffer[c][1]
                    im2col_array[n_counter, output_counter, c * 9 + 2] = buffer[c][2]
                    im2col_array[n_counter, output_counter, c * 9 + 3] = buffer[c][
                        w_padded
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 4] = buffer[c][
                        w_padded + 1
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 5] = buffer[c][
                        w_padded + 2
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 6] = buffer[c][
                        2 * w_padded
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 7] = buffer[c][
                        2 * w_padded + 1
                    ]
                    im2col_array[n_counter, output_counter, c * 9 + 8] = buffer[c][
                        2 * w_padded + 2
                    ]
                    buffer[c].pop(0)
                    if c == IN_CHANNEL - 1:
                        output_counter += 1
                        if (output_counter % (w_padded - (kernel_size - 1))) == 0:
                            for i in range(IN_CHANNEL):
                                for _ in range(kernel_size - 1):
                                    buffer[i].pop(0)
                    if output_counter == L:
                        output_counter = 0
                        n_counter += 1
                        for i in range(IN_CHANNEL):
                            buffer[i] = []
                    if n_counter == N:
                        n_counter = 0
                        print("resetting n_counter")


torch_random = torch.from_numpy(random_array).permute(0, 3, 1, 2).to(torch.float32)
# unfold
unfold = torch.nn.Unfold(
    kernel_size=3, padding=0, stride=1
)  # padding already inside the random_array
unfolded = unfold(torch_random).permute(0, 2, 1)
unfolded_numpy = unfolded.numpy()
# allclose
print("torch correctness", np.allclose(im2col_array, unfolded_numpy, atol=1e-3))

conv2d_out = np.matmul(im2col_array, random_weights)
print("conv2d_out", im2col_array.shape, random_weights.shape, conv2d_out.shape)

print("max_len", max_len)
# max pooling
pooling_kernel_size = 2
max_pool = torch.nn.MaxPool2d(
    kernel_size=pooling_kernel_size, stride=pooling_kernel_size
)
max_pool_out = max_pool(
    torch.from_numpy(conv2d_out.reshape(N, H, W, OUT_CHANNEL).transpose(0, 3, 1, 2))
).numpy()

print(max_pool_out.shape)
# print(max_pool_out)

max_pooling_buffers = [
    [-10000 for _ in range(W // pooling_kernel_size)] for _ in range(OUT_CHANNEL)
]

max_pool_out_buffer = np.zeros((N, L // (pooling_kernel_size**2), OUT_CHANNEL))

buffer_offset = 0
element_cnt = 0
output_valid = [False for _ in range(W // pooling_kernel_size)]
for n in range(N):
    for l in range(L):  # H * W
        # print("l", l, "buffer_offset", buffer_offset)
        # print("output_valid", output_valid)
        for c in range(OUT_CHANNEL):
            if output_valid[buffer_offset]:
                # check if overwrite
                if max_pool_out_buffer[n, element_cnt, c] != 0.0:
                    raise ValueError("overwrite detected")
                max_pool_out_buffer[n, element_cnt, c] = max_pooling_buffers[c][
                    buffer_offset
                ]
                max_pooling_buffers[c][buffer_offset] = -10000
            if conv2d_out[n, l, c] > max_pooling_buffers[c][buffer_offset]:
                max_pooling_buffers[c][buffer_offset] = conv2d_out[n, l, c]
            if output_valid[buffer_offset] and c == OUT_CHANNEL - 1:
                element_cnt += 1
                if element_cnt == L // (pooling_kernel_size**2):
                    raise ValueError("element_cnt overflow")
                output_valid[buffer_offset] = False
        if l > 0 and (l + 1) % pooling_kernel_size == 0:
            buffer_offset += 1
            if buffer_offset == W // pooling_kernel_size:
                buffer_offset = 0
        if l > 0 and (l + 1) % (pooling_kernel_size * W) == 0:
            output_valid = [True for _ in range(W // pooling_kernel_size)]
    for b in range(W // pooling_kernel_size):
        for c_ in range(OUT_CHANNEL):
            if output_valid[b]:
                # check if overwrite
                if max_pool_out_buffer[n, element_cnt, c_] != 0.0:
                    raise ValueError("overwrite detected")
                max_pool_out_buffer[n, element_cnt, c_] = max_pooling_buffers[c_][b]
                max_pooling_buffers[c_][b] = -10000
        output_valid[b] = False
        element_cnt += 1
    if element_cnt != L // (pooling_kernel_size**2):
        raise ValueError("element_cnt not full", element_cnt)
    element_cnt = 0


print("max_pool_out_buffer", max_pool_out_buffer)
print("max_pool_out", max_pool_out.shape)
print("max_pool_out_buffer", max_pool_out_buffer.shape)
max_pool_out = max_pool_out.reshape(N, OUT_CHANNEL, -1).transpose(0, 2, 1)
# allclose
print(
    "max_pool correctness",
    np.allclose(max_pool_out, max_pool_out_buffer, atol=1e-3),
)

print("diff", np.abs(max_pool_out - max_pool_out_buffer))
