from test.test_conv2d_gpu import test_conv2d_module_gpu
from test.test_halut import test_learn_offline
from test.test_kernel_gpu import test_read_acc_lut_kernel

if __name__ == "__main__":
    test_learn_offline(100000, 64, 64, 64, 1.0, 0.00, False, True)
    # test_encode_kernel(4096, 256, 128, 64, 1.0, 0.0)
    # test_read_acc_lut_kernel(4096, 256, 128, 64, 1.0, 0.0)
    # test_conv2d_module_gpu(128, 128, 7, 1, False, 32, 1.0, 0.0)
