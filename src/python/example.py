import numpy as np
from halutmatmul.halutmatmul import HalutMatmul

A = np.random.random((10000, 512))
A_train = A[:8000]
A_test = A[8000:]
B = np.random.random((512, 10))
C = np.matmul(A_test, B)

hm = HalutMatmul(C=32, K=16)
hm.learn_offline(A_train, B)
C_halut = hm.matmul_online(A_test)

mse = np.square(C_halut - C).mean()
print(mse)
