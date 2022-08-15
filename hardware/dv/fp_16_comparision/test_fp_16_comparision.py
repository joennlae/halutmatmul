# pylint: disable=no-value-for-parameter, protected-access
from math import log2
from random import getrandbits
import numpy as np
import cocotb
from cocotb.triggers import Timer
from cocotb.binary import BinaryValue
from cocotb.types import LogicArray

from util.helper_functions import (
    float_to_float16_binary,
)


@cocotb.test()
async def fp_16_comparision_test(dut) -> None:  # type: ignore[no-untyped-def]
    # Initial values
    await Timer(5, units="ps")
    dut.operand_a_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.operand_b_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    await Timer(5, units="ps")

    for _ in range(10000):
        random_val_fp16_a = np.float16(np.random.random_sample() * 10 - 5)
        random_val_fp16_b = np.float16(np.random.random_sample() * 10 - 5)
        random_val_bin_fp16_a = float_to_float16_binary(random_val_fp16_a)
        random_val_bin_fp16_b = float_to_float16_binary(random_val_fp16_b)
        # dut._log.info(f"values: {random_val_fp16_a}, {random_val_fp16_b}")
        await Timer(5, units="ps")
        dut.operand_a_i.value = random_val_bin_fp16_a
        dut.operand_b_i.value = random_val_bin_fp16_b
        await Timer(5, units="ps")
        read_out_bin = dut.comparision_o.value

        assert read_out_bin == (
            random_val_fp16_a >= random_val_fp16_b
        ), "output != a>=b"


@cocotb.test()
async def fp_16_comparision_special_values_test(dut) -> None:  # type: ignore[no-untyped-def]
    # Initial values
    await Timer(5, units="ps")
    dut.operand_a_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.operand_b_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    await Timer(5, units="ps")

    a = [0, 2e-14, 2.5]
    b = [0, 1e-14, 2.5]
    for _a, _b in zip(a, b):
        random_val_fp16_a = np.float16(_a)
        random_val_fp16_b = np.float16(_b)
        random_val_bin_fp16_a = float_to_float16_binary(random_val_fp16_a)
        random_val_bin_fp16_b = float_to_float16_binary(random_val_fp16_b)
        # dut._log.info(f"values: {random_val_fp16_a}, {random_val_fp16_b}")
        await Timer(5, units="ps")
        dut.operand_a_i.value = random_val_bin_fp16_a
        dut.operand_b_i.value = random_val_bin_fp16_b
        await Timer(5, units="ps")
        read_out_bin = dut.comparision_o.value

        assert read_out_bin == (
            random_val_fp16_a >= random_val_fp16_b
        ), "output != a>=b"
