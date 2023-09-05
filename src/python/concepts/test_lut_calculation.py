import numpy as np
import torch

C = 32
K = 16
D = 64
M = 96

prototypes = np.random.random((C, K, D))
W = np.random.random((D, M))

# Maddness implementation

lut = np.zeros((W.T.shape[0], C, K))
for i, q in enumerate(W.T):
    update = (q.reshape(1, 1, -1) * prototypes).sum(axis=2)
    lut[i] = update

print(lut.shape)

# single operation

results = np.tensordot(prototypes, W, axes=(2, 0))
results = results.transpose(2, 0, 1)
print(results.shape)
print(np.allclose(results, lut))

prop = torch.from_numpy(prototypes)
weights = torch.from_numpy(W)
torch_res = torch.tensordot(prop, weights, dims=([2], [0])).permute((2, 0, 1))  # type: ignore
print(torch_res.shape)
torch_numpy_res = torch_res.detach().numpy()
print(np.allclose(torch_numpy_res, lut))


d = 9
D = C * d
prototypes = np.random.random((C, K, d))
W = np.random.random((D, M))
W_reshaped = W.reshape((M, C, d)).transpose((2, 1, 0))

lut = np.zeros((W.shape[1], C, K))
# manual

lut_very_simple = np.zeros((W.shape[1], C, K))

for m in range(M):
    for c in range(C):
        for k in range(K):
            for i in range(d):
                lut_very_simple[m, c, k] += prototypes[c, k, i] * W_reshaped[i, c, m]

for m in range(M):
    for c in range(C):
        for k in range(K):
            # print("shapes", prototypes[c, k, :].shape, W_reshaped[:, c, m].shape)
            lut[m, c, k] = np.dot(prototypes[c, k, :], W_reshaped[:, c, m])

print("simple == very simple", np.allclose(lut, lut_very_simple))
# single operation
prop = torch.from_numpy(prototypes)
weights = torch.from_numpy(W_reshaped)
print("prop", prop.shape, "weights", weights.shape)
torch_res = torch.einsum("CKd, dCM -> MCK", [prop, weights])  # type: ignore
print(torch_res.shape)
torch_numpy_res = torch_res.detach().numpy()
print(np.allclose(torch_numpy_res, lut))
