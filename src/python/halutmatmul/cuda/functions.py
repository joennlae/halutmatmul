import cupy as cp  # type: ignore[import]
import torch
import numpy as np


def error_cupy(
    actual: torch.Tensor,
    desired: torch.Tensor,
) -> np.ndarray:
    # absolute memory hog but for speed better on GPU
    actual_cupy = cp.asarray(cp.from_dlpack(actual.detach()))
    desired_cupy = cp.asarray(cp.from_dlpack(desired.detach()))
    _min = cp.min(desired_cupy)
    _max = cp.max(desired_cupy)
    actual_cupy_std = (actual_cupy - _min) / (_max - _min)
    desired_cupy_std = (desired_cupy - _min) / (_max - _min)
    _range = (-1, 1)
    actual_cupy_scaled = actual_cupy_std * (_range[1] - _range[0]) + _range[0]
    desired_cupy_scaled = desired_cupy_std * (_range[1] - _range[0]) + _range[0]
    mae = cp.asnumpy(cp.mean(cp.abs((actual_cupy - desired_cupy))))
    mse = cp.asnumpy(cp.mean((actual_cupy - desired_cupy) ** 2))
    mape = cp.asnumpy(
        cp.mean(cp.abs(actual_cupy - desired_cupy) / (1 + cp.abs(desired_cupy)))
    )
    scaled_absolut_error = cp.asnumpy(
        cp.mean(cp.abs(actual_cupy_scaled - desired_cupy_scaled))
    )
    scaled_shift = cp.asnumpy(cp.mean(actual_cupy_scaled - desired_cupy_scaled))

    return np.array((mae, mse, mape, scaled_absolut_error, scaled_shift))
