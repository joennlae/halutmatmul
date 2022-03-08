# pylint: disable=C0413, W0612
import numpy as np

from cifar_example import load_cifar100_tasks, load_cifar10_tasks
import maddness.maddness as mn


def test_maddness_cifar_100():
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        W,
        b,
        lbls_train,
        lbls_test,
    ) = load_cifar100_tasks()
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


def test_maddness_cifar_10():
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        W,
        b,
        lbls_train,
        lbls_test,
    ) = load_cifar10_tasks()
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
