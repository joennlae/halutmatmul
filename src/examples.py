import numpy as np
import maddness.maddness as mn

if __name__ == "__main__":
    print("test")
    A_1 = np.random.random((50000, 512))
    A_2 = np.random.random((50000, 512))
    print(A_1.shape)
    print(A_2.shape)
    B = np.random.random((512, 50))
    print(B)
    C = np.matmul(A_2, B)
    print(C)

    C_halut = mn.matmul(A_2, B, C=128, A_learn=A_1)
    print(C_halut)

    mse = np.square(C_halut - C).mean()
    print(mse)
