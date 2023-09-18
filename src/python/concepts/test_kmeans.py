from collections import OrderedDict
import os
from test.test_utils.utils import helper_test_module
import numpy as np
import faiss
import torch
from halutmatmul.modules import HalutLinear
import halutmatmul.halutmatmul as hm

path = "/usr/scratch2/vilan1/janniss/halut/resnet18-cifar10-halut-resnet20-faiss-pq"

layer_name = "layer1.0.conv2"

A = np.load(os.path.join(path, layer_name + "_A.npy"))
B = np.load(os.path.join(path, layer_name + "_B.npy"))

print(A.shape, B.shape)

C = 16
K = 16

simple_k_mean_prototypes = np.zeros((C, K, A.shape[1] // C), dtype=np.float32)

subsampled = A.reshape((A.shape[0], C, -1))

# for c in range(C):
#     print("Learning simple k-means prototypes for channel {}".format(c))
#     kmeans = faiss.Kmeans(
#         subsampled.shape[2],
#         K,
#         niter=1000,
#         verbose=True,
#         nredo=10,
#     )
#     kmeans.train(subsampled[:, c, :])
#     centroids = kmeans.centroids
#     print("centroids", centroids.shape, centroids[0])
#     simple_k_mean_prototypes[c] = centroids
# print("Done learning simple k-means prototypes")
#
# print(
#     "simple_k_mean_prototypes",
#     simple_k_mean_prototypes.shape,
#     simple_k_mean_prototypes[0, 0],
# )

# nbit = 4
# pq = faiss.ProductQuantizer(A.shape[1], C, nbit)
# pq.verbose = True
#
# pq.train(A)
#
# codes = pq.compute_codes(A)
# print("codes", codes.shape, codes[0])
# x2 = pq.decode(codes)
#
# print("x2", x2[0], A[0])
# diff = ((A - x2) ** 2).sum() / (A**2).sum()
# print("diff", diff)
#
# proto_transposed = simple_k_mean_prototypes.transpose(2, 1, 0)
# faiss.copy_array_to_vector(simple_k_mean_prototypes.ravel(), pq.centroids)
#
# codes = pq.compute_codes(A)
# print("codes", codes.shape, codes[0])
# x2 = pq.decode(codes)
#
# print("x2", x2[0], A[0])
# diff = ((A - x2) ** 2).sum() / (A**2).sum()
# print("diff", diff)
A_reshaped = A.reshape((A.shape[0], C, -1))
# for c in range(C):
#     print("Learning simple k-means prototypes for channel {}".format(c))
#     clustering = faiss.Clustering(A_reshaped.shape[2], K)
#     clustering.d = A_reshaped.shape[2]
#     clustering.verbose = True
#     clustering.niter = 1000
#     clustering.train(A_reshaped[:, c, :])
#     centroids = faiss.vector_float_to_array(clustering.centroids)
#     print("centroids", centroids.shape, centroids[0])

store_array = hm.learn_halut_offline(A, B, C=C)

halutmatmul_module = HalutLinear(
    in_features=B.shape[0],
    out_features=B.shape[1],
    split_factor=1,
    use_prototypes=False,
)
torch_linear = torch.nn.Linear(B.shape[0], B.shape[1])
state_dict = OrderedDict({"weight": torch.from_numpy(B.astype(np.float32)).T})
torch_linear.load_state_dict(state_dict, strict=False)
state_dict = OrderedDict(
    state_dict
    | OrderedDict(
        {
            "store_input": torch.zeros(1, dtype=torch.bool),
            "halut_active": torch.ones(1, dtype=torch.bool),
            "lut": torch.from_numpy(
                store_array[hm.HalutOfflineStorage.SIMPLE_LUT].astype(np.float32)
            ),
            "thresholds": torch.from_numpy(
                store_array[hm.HalutOfflineStorage.THRESHOLDS]
            ),
            "dims": torch.from_numpy(store_array[hm.HalutOfflineStorage.DIMS]),
            "P": torch.from_numpy(
                store_array[hm.HalutOfflineStorage.SIMPLE_PROTOTYPES].astype(np.float32)
            ),
        }
    )
)
halutmatmul_module.load_state_dict(state_dict, strict=False)
out = halutmatmul_module(torch.from_numpy(A.astype(np.float32)))
out_2 = torch_linear(torch.from_numpy(A.astype(np.float32)))
compared = ((out_2 - out) ** 2).sum() / (out_2**2).sum()


helper_test_module(
    torch.from_numpy(A.astype(np.float32)),
    torch_linear,
    halutmatmul_module,
    rel_error=-1.0,
    scaled_error_max=0.02,
)
print("compared", compared)
