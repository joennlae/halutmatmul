import numpy as np

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
