from test.test_conv2d import test_conv2d_module
from test.test_resnet import test_cifar10_inference
from test.test_conv2d_gpu import test_conv2d_module_gpu
from test.test_halut import test_learn_offline
from models.dscnn.main import run_kws_main
from models.levit.main import run_levit  # type: ignore[attr-defined]
from halutmatmul.halutmatmul import EncodingAlgorithm

# from test.test_kernel_gpu import untest_read_acc_lut_kernel

if __name__ == "__main__":
    # test_learn_offline(
    #     2048, 512, 64, 64, 16, 1.0, 0.00, EncodingAlgorithm.DECISION_TREE
    # )
    # test_encode_kernel(4096, 256, 128, 64, 1.0, 0.0)
    # test_read_acc_lut_kernel(4096, 256, 128, 64, 1.0, 0.0)
    # test_conv2d_module_gpu(128, 128, 7, 1, False, 32, 1.0, 0.0)
    test_conv2d_module(64, 64, 7, 1, False, 16, 16, 1.0, 0.0, 0, 2)
    # test_cifar10_inference()
    # run_kws_main()
    # run_levit()
