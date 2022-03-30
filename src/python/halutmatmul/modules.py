import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn.modules import Linear

from maddness.maddness import MaddnessMatmul


class HalutLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        halut_active: bool = False,
        halut_offline_A: np.ndarray = None,
        halut_C: int = 16,
        halut_lut_work_const: int = -1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.halut_active = halut_active
        self.halut = None
        self.halut_offline_learned = False
        self.halut_offline_A = halut_offline_A
        self.halut_C = halut_C
        self.halut_lut_work_const = halut_lut_work_const

    def learn_offline(self) -> None:
        if self.halut_offline_A is None:
            raise Exception("halut A is None: {}".format(self.halut_offline_A))
        self.halut = MaddnessMatmul(
            C=self.halut_C, lut_work_const=self.halut_lut_work_const
        )
        weights_numpy = self.weight.detach().cpu().numpy().transpose(1, 0)
        print(weights_numpy, weights_numpy.shape)
        print(self.halut_offline_A, self.halut_offline_A.shape)
        self.halut.learn_offline(self.halut_offline_A, weights_numpy)
        self.halut_offline_learned = True

    # pylint: disable=W0622
    def forward(self, input: Tensor) -> Tensor:
        if self.halut_active:
            if not self.halut_offline_learned:
                self.learn_offline()
            input_numpy = input.detach().cpu().numpy()
            print(input_numpy.shape)
            result = self.halut.matmul_online(input_numpy)
            print(result.shape)
            bias_to_add = self.bias.clone().repeat(input.shape[0], 1)
            print(bias_to_add.shape)
            return torch.tensor(result) + bias_to_add
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "Halut in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
