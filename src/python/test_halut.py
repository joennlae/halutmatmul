from test.test_halut import test_learn_offline
from test.test_kernel_gpu import test_read_acc_lut_kernel

if __name__ == "__main__":
    # test_learn_offline(64, 8, 8, 4, 1.0, 0.00, False, True)
    # test_encode_kernel(4096, 256, 128, 64, 1.0, 0.0)
    test_read_acc_lut_kernel(4096, 256, 128, 64, 1.0, 0.0)
