from test.test_linear import test_linear_module
from test.test_conv2d import test_conv2d_module
from test.test_resnet import test_cifar10_inference

if __name__ == "__main__":
    # test_linear_module(512, 32, 32, 1.0, 0.0, False, 32, False, True)
    test_conv2d_module(
        32, 64, 5, 3, False, 32, 16, 1.0, 0.0, 1, False, 1, 1, "im2col", False
    )
    # test_cifar10_inference()
