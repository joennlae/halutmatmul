import os
import numpy as np
import maddness.maddness as mn

_dir = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DIR = os.path.join(_dir, "..", "assets", "cifar10-softmax")
CIFAR100_DIR = os.path.join(_dir, "..", "assets", "cifar100-softmax")

def load_cifar100_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = "cifar100_softmax_inputs_train.npy"
    SOFTMAX_OUTPUTS_TRAIN_PATH = "cifar100_softmax_outputs_train.npy"
    SOFTMAX_INPUTS_TEST_PATH = "cifar100_softmax_inputs_test.npy"
    SOFTMAX_OUTPUTS_TEST_PATH = "cifar100_softmax_outputs_test.npy"
    SOFTMAX_W_PATH = "cifar100_softmax_W.npy"
    SOFTMAX_B_PATH = "cifar100_softmax_b.npy"
    LABELS_TRAIN_PATH = "cifar100_labels_train.npy"
    LABELS_TEST_PATH = "cifar100_labels_test.npy"

    def load_mat(fname):
        fpath = os.path.join(CIFAR100_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel()
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    return (X_train, Y_train, X_test, Y_test, W, b, lbls_train, lbls_test)


def load_cifar10_tasks():
    SOFTMAX_INPUTS_TRAIN_PATH = "cifar10_softmax_inputs_train.npy"
    SOFTMAX_OUTPUTS_TRAIN_PATH = "cifar10_softmax_outputs_train.npy"
    SOFTMAX_INPUTS_TEST_PATH = "cifar10_softmax_inputs_test.npy"
    SOFTMAX_OUTPUTS_TEST_PATH = "cifar10_softmax_outputs_test.npy"
    SOFTMAX_W_PATH = "cifar10_softmax_W.npy"
    SOFTMAX_B_PATH = "cifar10_softmax_b.npy"
    LABELS_TRAIN_PATH = "cifar10_labels_train.npy"
    LABELS_TEST_PATH = "cifar10_labels_test.npy"

    def load_mat(fname):
        fpath = os.path.join(CIFAR10_DIR, fname)
        return np.load(fpath)

    X_train = load_mat(SOFTMAX_INPUTS_TRAIN_PATH)
    Y_train = load_mat(SOFTMAX_OUTPUTS_TRAIN_PATH)
    X_test = load_mat(SOFTMAX_INPUTS_TEST_PATH)
    Y_test = load_mat(SOFTMAX_OUTPUTS_TEST_PATH)
    W = load_mat(SOFTMAX_W_PATH)
    b = load_mat(SOFTMAX_B_PATH)
    lbls_train = load_mat(LABELS_TRAIN_PATH).ravel()
    lbls_test = load_mat(LABELS_TEST_PATH).ravel()

    # we aren't going to store or approximate the biases, so just subtract
    # off their contributions at the start
    Y_train -= b
    Y_test -= b

    return (X_train, Y_train, X_test, Y_test, W, b, lbls_train, lbls_test)


def cifar_test():
    # pylint: disable=W0612
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        W,
        b,
        lbls_train,
        lbls_test,
    ) = load_cifar100_tasks() # change cifar type here
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, W.shape)
    print(X_train)

    maddness = mn.MaddnessMatmul(
        C=16, lut_work_const=-1
    )  # MADDNESS-PQ has lut_work_const=1
    maddness.learn_A(X_train)
    maddness.reset()
    Y_pred = maddness.apply_matmul(X_test, W)

    print(Y_pred)
    print("max_pred", np.max(Y_pred), "min_pred", np.min(Y_pred))
    print(Y_test)
    print("max_test", np.max(Y_test), "min_test", np.min(Y_test))

    mse = (np.square(Y_pred - Y_test)).mean()
    print(mse)

    maddness.reset()
    Y_train_pred = maddness.apply_matmul(X_train, W)

    print(Y_train_pred.shape, X_train.shape)
    print(
        "max_train_pred", np.max(Y_train_pred), "min_train_pred", np.min(Y_train_pred)
    )

    mse = (np.square(Y_train_pred - Y_train)).mean()
    print(mse)


if __name__ == "__main__":
    cifar_test()
