from typing import Tuple
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

embeddings = torch.nn.EmbeddingBag.from_pretrained(
    torch.from_numpy(lut.transpose(1, 2, 0).reshape(-1, M)).to(torch.float32),
    mode="sum",
)

input_torch = torch.from_numpy(input_a).to(torch.float32)
threshold_table_torch = torch.from_numpy(threshold_table).to(torch.float32)

threshold_embeddings = torch.nn.Embedding.from_pretrained(
    threshold_table_torch.flatten().unsqueeze(1)
)

encoded_torch = torch.zeros((N, C), dtype=torch.float32, requires_grad=True).flatten()
prototype_addr_internal_offset = 2 ** torch.arange(4 + 1) - 1
caddr_internal_offset = (torch.arange(C) * K).repeat(N)

kaddr = encoded_torch.to(torch.int32)
input_torch = input_torch.reshape((-1, 4))
prototype_addr_internal = torch.zeros(N * C, dtype=torch.int64)


def create_selection_matrix() -> torch.Tensor:
    selection_matrix = torch.zeros((15, 4), dtype=torch.float32)
    for i in range(15):
        if i == 0:
            selection_matrix[0, 0] = 1
        else:
            selection_matrix[i, int(np.log2(i + 1))] = 1
    return selection_matrix


def create_bit_matrix() -> torch.Tensor:
    bit_matrix_numpy = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        ]
    )
    bit_matrix = torch.from_numpy(bit_matrix_numpy.T).to(torch.float32)
    return bit_matrix


S = create_selection_matrix()
B = create_bit_matrix()


def traverse_tree(
    S: torch.Tensor, B: torch.Tensor, T: torch.Tensor, input: torch.Tensor
) -> torch.Tensor:
    h = S.mm(input) - T.unsqueeze(1)
    b = B.mm(h.relu())
    return torch.argmax(b)


def encode_with_traversal(
    S: torch.Tensor,
    B: torch.Tensor,
    thresholds: torch.Tensor,
    input: torch.Tensor,
    C: int,
) -> torch.Tensor:
    thresholds_reshaped = thresholds.reshape((C, -1))
    encoded_result = torch.zeros((input.shape[0], C), dtype=torch.int32)
    for row in range(input.shape[0]):
        for c in range(C):
            encoded_value = traverse_tree(
                S,
                B,
                thresholds_reshaped[c][:15],
                input[row, c, :].unsqueeze(1),
            )
            encoded_result[row, c] = encoded_value
    return encoded_result


input_torch_normal = torch.from_numpy(input_a).to(torch.float32)
encoded_new = encode_with_traversal(S, B, threshold_table_torch, input_torch_normal, C)
print(encoded_new, encoded_new.shape)

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
print(torch.allclose(encoded_torch, encoded_new))

encoded_torch += torch.arange(C) * K
input = torch.IntTensor(encoded_torch)

encoded += np.arange(C) * K
results_embeddings = embeddings(encoded_torch).detach().numpy()
print(np.allclose(result, results_embeddings))

