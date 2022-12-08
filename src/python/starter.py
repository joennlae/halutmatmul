from test.test_linear import test_linear_module
from test.test_conv2d import test_conv2d_module

if __name__ == "__main__":
    # test_linear_module(512, 32, 32, 1.0, 0.0, False, 32, True)
    test_conv2d_module(512, 512, 6, 3, False, 64, 16, 1.0, 0.0, 1, True)
