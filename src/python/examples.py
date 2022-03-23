import numpy as np
import maddness.maddness as mn

A_1 = np.random.random((64, 32))
B = np.random.random((32, 4))
C = np.matmul(A_1, B)

C_halut = mn.matmul(A_1, B, C=8, A_learn=A_1)

print(C)
print(C_halut)

mse = np.square(C_halut - C).mean()
print(mse)
