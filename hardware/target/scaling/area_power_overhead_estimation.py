# pylint: disable=chained-comparison, use-a-generator
from queue import Queue
import math
import numpy as np


# estimate 3x3 im2col unit
kernel_size = 3
N = 1
H = 32
W = 32
IN_CHANNEL = 32
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

q = Queue(maxsize=3)
fifo_width = kernel_size * kernel_size
fifos = [[] for _ in range(fifo_width * IN_CHANNEL)]
# select random dims
dims = np.zeros((IN_CHANNEL, 4), dtype=np.int64)
for i in range(IN_CHANNEL):
    channel_dims = np.random.choice(9, 4, replace=False)
    dims[i, :] = channel_dims
print(fifos)
output_counter = 0
n_counter = 0
max_len = 0
maxes = [0 for _ in range(fifo_width * IN_CHANNEL)]
for n in range(N):
    for h in range(H):
        for w in range(W):
            for c in range(IN_CHANNEL):
                cur_val = random_array[n, h, w, c]
                print(f"cur_val: {cur_val}")
                if w < W - (kernel_size - 1) and h < H - (kernel_size - 1):
                    # kernel top left
                    fifos[c * fifo_width + 0 * kernel_size].append(cur_val)
                if w > 0 and w < W - (kernel_size - 2) and h < H - (kernel_size - 1):
                    # kernel top middle
                    fifos[c * fifo_width + 0 * kernel_size + 1].append(cur_val)
                if w > 1 and h < H - (kernel_size - 1):
                    # kernel top right
                    fifos[c * fifo_width + 0 * kernel_size + 2].append(cur_val)
                if w < W - (kernel_size - 1) and h > 0 and h < H - (kernel_size - 2):
                    # kernel middle left
                    fifos[c * fifo_width + 1 * kernel_size + 0].append(cur_val)
                if (
                    w > 0
                    and w < W - (kernel_size - 2)
                    and h > 0
                    and h < H - (kernel_size - 2)
                ):
                    # kernel middle middle
                    fifos[c * fifo_width + 1 * kernel_size + 1].append(cur_val)
                if w > 1 and h > 0 and h < H - (kernel_size - 2):
                    # kernel middle right
                    fifos[c * fifo_width + 1 * kernel_size + 2].append(cur_val)
                if w < W - (kernel_size - 1) and h > 1:
                    # kernel bottom left
                    fifos[c * fifo_width + 2 * kernel_size + 0].append(cur_val)
                if w > 0 and w < W - (kernel_size - 2) and h > 1:
                    # kernel bottom middle
                    fifos[c * fifo_width + 2 * kernel_size + 1].append(cur_val)
                if w > 1 and h > 1:
                    # kernel bottom right
                    fifos[c * fifo_width + 2 * kernel_size + 2].append(cur_val)

                maxes = [
                    max(len(fifos[i]), maxes[i]) for i in range(fifo_width * IN_CHANNEL)
                ]
                if all([len(fifos[c * fifo_width + i]) > 0 for i in range(fifo_width)]):
                    print("all fifos have at least one entry")
                    # if all fifos have at least one entry, then we can pop
                    # overwrite check
                    if not (
                        im2col_array[
                            n_counter,
                            output_counter,
                            c * fifo_width : (c + 1) * fifo_width,
                        ]
                        == 0
                    ).all():
                        print(
                            "overwrite detected", n, h, w, c, output_counter, n_counter
                        )
                        raise ValueError("overwrite detected")
                    im2col_array[
                        n_counter, output_counter, c * fifo_width : (c + 1) * fifo_width
                    ] = [fifos[c * fifo_width + i].pop(0) for i in range(fifo_width)]
                    if c == IN_CHANNEL - 1:
                        output_counter += 1
                        print(f"output_counter: {output_counter}")
                        if output_counter == L:
                            output_counter = 0
                            print("resetting output_counter")
                            n_counter += 1


print(im2col_array)
print(im2col_array.shape)
# check if a value is still zero
if (im2col_array == 0).any():
    raise ValueError("Zero value detected")
# check if all fifos are empty
if any([len(fifo) > 0 for fifo in fifos]):
    raise ValueError("Fifo not empty")
print(fifos)
print(maxes)
out_conv2d = np.zeros((N, L, OUT_CHANNEL), dtype=np.float32)
