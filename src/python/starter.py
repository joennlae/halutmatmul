from test.test_linear import test_linear_module
from test.test_conv2d import test_conv2d_module
from test.test_resnet import test_cifar10_inference_resnet20, test_cifar10_inference
from utils.analysis_helper import resnet20_layers, resnet20_b_layers

if __name__ == "__main__":
    # test_linear_module(128, 64, 16, 9.0, -0.35, False, 32, True)
    # test_conv2d_module(
    #     32, 32, 7, 1, False, 32, 16, 9.0, -0.35, 1, 2, 1, "im2col", False
    # )
    # acc = {}
    # # for layer in resnet20_b_layers:
    # #     accuracy = test_cifar10_inference_resnet20(layer)
    # #     acc[layer] = accuracy
    # acc = test_cifar10_inference_resnet20("layer2.0.conv1")
    # print(acc)
    # test_cifar10_inference_resnet20("layer1.0.conv2")
    test_cifar10_inference()
