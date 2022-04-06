from typing import Any, Union
import numpy as np

from maddness.maddness import MaddnessMatmul


class HalutMatmul(MaddnessMatmul):
    def __init__(self, C: int = 16, lut_work_const: int = -1) -> None:
        super().__init__(C, lut_work_const)

    # pylint: disable=R0201
    def tensordot(
        self, a: np.ndarray, b: np.ndarray, axes: Union[int, list[int], Any] = 2
    ) -> np.ndarray:
        # https://github.com/numpy/numpy/blob/145ed90f638c1a12ce5b06e9100421f99783f431/numpy/core/numeric.py#L950

        """Example
          padding=0, kernel_size=(3, 3), stride=1

          IN: (128, 64, 112, 112)
          W: (64, 64, 3, 3)
          after Im2col (np.lib.stride_tricks.as_strided): (128, 64, 110, 110, 3, 3)
          np.tensordot(IN, W, ((1,4,5),(1,2,3)))

          at transpose: (128, 64, 110, 110, 3, 3) -> (128, 110, 110, 64, 3, 3)
          newaxes_a: [0, 2, 3, 1, 4, 5]
          bt transpose: (64, 64, 3, 3) -> (64, 3, 3, 64)
          newaxes_b: [1, 2, 3, 0]
          newshape_a: (1548800, 576)
          newshape_B: (576, 64)

          (1548800, 64) -> (128, 64, 110, 110)
          olda: [128, 110, 110]
          oldb: [64]
          olda + oldb: [128, 110, 110, 64]
          OUT: (128, 110, 110, 64)

          needs to be reshaped later to match conv2d output
          np.moveaxis(ret,4,2).reshape(batch_size, channels_out, out_y, out_x)
        """
        try:
            iter(axes)  # type: ignore[arg-type]
        # pylint: disable=W0703
        except Exception:
            axes_a = list(range(-axes, 0))  # type: ignore[operator]
            axes_b = list(range(0, axes))  # type: ignore[arg-type]
        else:
            axes_a, axes_b = axes  # type: ignore[misc, assignment]
        try:
            na = len(axes_a)
            axes_a = list(axes_a)
        except TypeError:
            axes_a = [axes_a]  # type: ignore[list-item]
            na = 1
        try:
            nb = len(axes_b)
            axes_b = list(axes_b)
        except TypeError:
            axes_b = [axes_b]  # type: ignore[list-item]
            nb = 1

        a, b = np.asarray(a), np.asarray(b)
        as_ = a.shape
        nda = a.ndim
        bs = b.shape
        ndb = b.ndim
        equal = True
        if na != nb:
            equal = False
        else:
            for k in range(na):
                if as_[axes_a[k]] != bs[axes_b[k]]:
                    equal = False
                    break
                if axes_a[k] < 0:
                    axes_a[k] += nda
                if axes_b[k] < 0:
                    axes_b[k] += ndb
        if not equal:
            raise ValueError("shape-mismatch for sum")

        # Move the axes to sum over to the end of "a"
        # and to the front of "b"
        notin = [k for k in range(nda) if k not in axes_a]
        newaxes_a = notin + axes_a
        N2 = 1
        for axis in axes_a:
            N2 *= as_[axis]
        newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
        olda = [as_[axis] for axis in notin]

        notin = [k for k in range(ndb) if k not in axes_b]
        newaxes_b = axes_b + notin
        N2 = 1
        for axis in axes_b:
            N2 *= bs[axis]
        newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
        oldb = [bs[axis] for axis in notin]

        at = a.transpose(newaxes_a).reshape(newshape_a)
        bt = b.transpose(newaxes_b).reshape(newshape_b)

        print(at.shape, bt.shape)
        # not supporting all different configurations
        res = np.dot(at, bt)
        return res.reshape(olda + oldb)
