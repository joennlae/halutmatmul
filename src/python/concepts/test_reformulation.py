from typing import Tuple, Any
import torch
import numpy as np


def encoding_function(
    threshold_table: np.ndarray, input_a: np.ndarray, tree_depth: int = 4, K: int = 16
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    encoded = np.zeros((input_a.shape[0], input_a.shape[1]), dtype=np.int32)
    kaddr_history = np.zeros(
        (input_a.shape[0], input_a.shape[1], tree_depth), dtype=np.int32
    )
    thresh_mem_history = np.zeros(
        (input_a.shape[0], input_a.shape[1], tree_depth), dtype=np.float16
    )
    # CPerEncUnit = input_a.shape[1]
    caddr_internal_offset = np.arange(input_a.shape[1]) * K
    prototype_addr_internal_offset = 2 ** np.arange(tree_depth + 1) - 1
    for row in range(input_a.shape[0]):
        kaddr = np.zeros(input_a.shape[1], dtype=np.int64)
        prototype_addr_internal = np.zeros(input_a.shape[1], dtype=np.int64)
        for tree_level_cnt in range(tree_depth):
            kaddr_history[row, :, tree_level_cnt] = kaddr
            data_thresh_mem_o = threshold_table[
                prototype_addr_internal + caddr_internal_offset
            ]
            thresh_mem_history[row, :, tree_level_cnt] = data_thresh_mem_o
            data_input_comparision = input_a[row, :, tree_level_cnt]
            fp_16_comparision_o = data_input_comparision > data_thresh_mem_o
            kaddr = (kaddr * 2) + fp_16_comparision_o
            prototype_addr_internal = (
                kaddr + prototype_addr_internal_offset[tree_level_cnt + 1]
            )
        encoded[row] = kaddr
    return encoded, kaddr_history, thresh_mem_history


def decoding_2d(
    lut: np.ndarray, encoded: np.ndarray
) -> "Tuple[np.ndarray, np.ndarray]":
    result = np.zeros((encoded.shape[0], lut.shape[0]), dtype=np.float32)  # [N, M]
    result_history = np.zeros(
        (encoded.shape[0], lut.shape[0], lut.shape[1]), dtype=np.float32
    )
    for m in range(lut.shape[0]):
        for c in range(lut.shape[1]):
            result_history[:, m, c] = result[:, m]
            result[:, m] += lut[m, c, encoded[:, c]]
    return result, result_history


C = 32
K = 16
M = 64
N = 128
np.random.seed(4419)
threshold_table = np.random.random((C * K)).astype(np.float16)
input_a = np.random.random((N, C, 4)).astype(np.float16)

encoded, _, _ = encoding_function(threshold_table, input_a, tree_depth=4, K=K)

lut = np.random.random((M, C, K)).astype(np.float16)

result, _ = decoding_2d(lut, encoded)

input_torch = torch.from_numpy(input_a).to(torch.float32)
threshold_table_torch = torch.from_numpy(threshold_table).to(torch.float32)


def create_selection_matrix(C: int = 1, depth: int = 4) -> torch.Tensor:
    selection_matrix = torch.zeros((C * 15, C * depth), dtype=torch.float32)
    based_selection_matrix = torch.zeros((2**depth - 1, depth), dtype=torch.float32)
    for i in range(2**depth - 1):
        if i == 0:
            based_selection_matrix[0, 0] = 1
        else:
            based_selection_matrix[i, int(np.log2(i + 1))] = 1
    for c in range(C):
        selection_matrix[
            c * 15 : (c + 1) * 15, c * depth : (c + 1) * depth
        ] = based_selection_matrix
    return selection_matrix


def create_bit_matrix(C: int = 1, depth: int = 4) -> torch.Tensor:
    # example when using C = 1
    offset = 0
    bit_matrix_numpy = np.array(
        [
            [
                offset,
                offset,
                offset,
                offset,
                offset,
                offset,
                offset,
                offset,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            [offset, offset, offset, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, offset, offset, offset, offset, 1, 1, 1, 1],
            [offset, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, offset, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, offset, offset, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, offset, 1, 1],
            [offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, offset, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, offset, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, offset, 1],
        ]
    )
    print(bit_matrix_numpy.shape)
    # fmt: off
    bit_matrix_numpy = np.array(
        [
            # 0
            [-1,
             -1, 0,
             -1, 0, 0, 0,
             -1, 0, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             -1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             1, 0, 0, 0,
             0, -1, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             1, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, -1, 0, 0,
             0, 0, -1, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, -1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, 1, 0, 0,
             0, 0, 0, -1, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0],
            # 8
            [1,
             0, -1,
             0, 0, -1, 0,
             0, 0, 0, 0, -1, 0, 0, 0],
            [1,
             0, -1,
             0, 0, -1, 0,
             0, 0, 0, 0, 1, 0, 0, 0],
            [1,
             0, -1,
             0, 0, 1, 0,
             0, 0, 0, 0, 0, -1, 0, 0],
            [1,
             0, -1,
             0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 0, 0],
             # 12
            [1,
             0, 1,
             0, 0, 0, -1,
             0, 0, 0, 0, 0, 0, -1, 0],
            [1,
             0, 1,
             0, 0, 0, -1,
             0, 0, 0, 0, 0, 0, 1, 0],
            [1,
             0, 1,
             0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, -1],
            [1,
             0, 1,
             0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    print(bit_matrix_numpy.shape)
    # fmt: on
    bit_matrix_base = torch.from_numpy(bit_matrix_numpy).to(torch.float32)
    K = 2**depth
    bit_matrix = torch.zeros((C * K, C * (K - 1)), dtype=torch.float32)
    for c in range(C):
        bit_matrix[
            c * K : (c + 1) * K,
            c * (K - 1) : (c + 1) * (K - 1),
        ] = bit_matrix_base
    return bit_matrix


S = create_selection_matrix(C=C)
B = create_bit_matrix(C=C)


def traverse_tree(
    S: torch.Tensor,
    B: torch.Tensor,
    T: torch.Tensor,
    input: torch.Tensor,
    C: int = 32,
    depth: int = 4,
) -> torch.Tensor:
    temperature = 0.5
    print("S", S, S.shape, input.shape)
    selection_tensor = torch.Tensor([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3])
    selection_tensor_all = torch.zeros((C * 15), dtype=torch.int64)
    for c in range(C):
        selection_tensor_all[c * 15 : (c + 1) * 15] = selection_tensor + c * 4
    print("selection_tensor", selection_tensor_all.shape)
    h = input[selection_tensor_all, :] - T.unsqueeze(1)
    h = S.mm(input) - T.unsqueeze(1)
    tanh_h = torch.tanh(h / temperature)
    sign_ste = torch.sign(h) - tanh_h.detach() + tanh_h
    b = B.mm(sign_ste)
    b = b.T.reshape((-1, C, 2**depth))
    encoding_soft = torch.nn.Softmax(dim=2)(b)
    index = torch.argmax(encoding_soft, dim=2, keepdim=True)
    encoding_hard = torch.zeros_like(encoding_soft).scatter_(2, index, 1.0)
    encoding_out = encoding_hard - encoding_soft.detach() + encoding_soft
    return encoding_out.to(torch.float32)


def encode_with_traversal(
    S: torch.Tensor,
    B: torch.Tensor,
    thresholds: torch.Tensor,
    input: torch.Tensor,
    C: int,
) -> torch.Tensor:
    thresholds_reshaped = thresholds.reshape((C, -1))
    encoded_result = torch.zeros((input.shape[0], C), dtype=torch.int32)
    encoded_value = traverse_tree(
        S,
        B,
        thresholds_reshaped[:, :15].flatten(),
        input.reshape((input.shape[0], -1)).T,
        C=C,
    )
    encoded_result = encoded_value
    return encoded_result


input_torch_normal = torch.from_numpy(input_a).to(torch.float32)
threshold_table_torch.requires_grad = True
encoded_new = encode_with_traversal(S, B, threshold_table_torch, input_torch_normal, C)

lut_torch = torch.from_numpy(lut.transpose(0, 1, 2)).to(torch.float32)
lut_torch.requires_grad = True
encoded_new = encoded_new.permute((0, 1, 2))
result_torch = torch.zeros(
    (encoded_new.shape[0], lut_torch.shape[0]), dtype=torch.float32
)
# for m in range(lut_torch.size(0)):
#     result_torch[:, m] += (
#         (encoded_new * lut_torch[m].repeat((encoded_new.shape[0], 1, 1)))
#         .sum(dim=2)
#         .sum(dim=1)
#     )
# result_torch = torch.einsum("nij, kij -> nki", [encoded_new, lut_torch])
# result_torch = result_torch.sum(dim=2)
for i in range(2):
    M = lut_torch.size(0)
    result_torch[:, (M // 2) * i : (M // 2) * (i + 1)] = torch.einsum(
        "nij, kij -> nki", [encoded_new, lut_torch[(M // 2) * i : (M // 2) * (i + 1)]]
    ).sum(dim=2)

loss = result_torch.sigmoid().prod()
loss.backward()


def getBack(var_grad_fn: Any) -> None:
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                # print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad.shape)
                print()
            except AttributeError:
                getBack(n[0])


getBack(loss.grad_fn)

print("important test", np.allclose(result, result_torch.detach().numpy()))
# show how many values are different
print(np.sum(result != result_torch.detach().numpy()))

# old version
embeddings = torch.nn.EmbeddingBag.from_pretrained(
    torch.from_numpy(lut.transpose(1, 2, 0).reshape(-1, M)).to(torch.float32),
    mode="sum",
)
threshold_embeddings = torch.nn.Embedding.from_pretrained(
    threshold_table_torch.flatten().unsqueeze(1)
)
encoded_torch = torch.zeros((N, C), dtype=torch.float32, requires_grad=True).flatten()
prototype_addr_internal_offset = 2 ** torch.arange(4 + 1) - 1
caddr_internal_offset = (torch.arange(C) * K).repeat(N)

kaddr = encoded_torch.to(torch.int32)
input_torch = input_torch.reshape((-1, 4))
prototype_addr_internal = torch.zeros(N * C, dtype=torch.int64)

for tree_level in range(4):
    thresholds = threshold_embeddings(
        prototype_addr_internal + caddr_internal_offset
    ).squeeze()
    data_input_comparision = input_torch[:, tree_level]
    comparison_out = torch.where(
        data_input_comparision > thresholds,
        torch.scalar_tensor(1, dtype=torch.int32),
        torch.scalar_tensor(0, dtype=torch.int32),
    )
    kaddr = (kaddr * 2) + comparison_out
    prototype_addr_internal = kaddr + prototype_addr_internal_offset[tree_level + 1]
encoded_torch = kaddr.reshape((N, C))

print(np.allclose(encoded, encoded_torch.numpy()))
# show how many values are different
print(np.sum(encoded != encoded_torch.numpy()))
# and equal
print(np.sum(encoded == encoded_torch.numpy()))

encoded_torch += torch.arange(C) * K
input = torch.IntTensor(encoded_torch)

encoded += np.arange(C) * K
results_embeddings = embeddings(encoded_torch).detach().numpy()
print(np.allclose(result, results_embeddings))
