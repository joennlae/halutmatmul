# extracted from https://github.com/dblalock/bolt
# SPDX-License-Identifier: MPL-2.0 (as before)
# this file is only needed to resolve a circular dependency

import numpy as np


class MultiSplit:
    __slots__ = "dim vals scaleby offset".split()

    def __init__(self, dim, vals, scaleby=None, offset=None):
        self.dim = dim
        self.vals = np.asarray(vals)
        self.scaleby = scaleby
        self.offset = offset

    def __repr__(self) -> str:
        return f"<{self.get_params()}>"

    def __str__(self) -> str:
        return self.get_params()

    def get_params(self) -> str:
        params = (
            f"Multisplit: dim({self.dim}), vals({self.vals}), "
            f"scaleby({self.scaleby}), offset({self.offset})"
        )
        return params

    def preprocess_x(self, x: np.ndarray) -> np.ndarray:
        if self.offset is not None:
            x = x - self.offset
        if self.scaleby is not None:
            x = x * self.scaleby
        return x
