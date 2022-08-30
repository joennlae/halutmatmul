# pylint: disable=no-value-for-parameter, protected-access
from math import log2
from random import getrandbits
import numpy as np
import cocotb
from cocotb.triggers import Timer
from cocotb.binary import BinaryValue
from cocotb.types import LogicArray

from util.helper_functions import (
    binary_to_float16,
    binary_to_float32,
    float_to_float16_binary,
    float_to_float32_binary,
)


@cocotb.test()
async def fp_16_32_adder_test(dut) -> None:  # type: ignore[no-untyped-def]

    # Initial values
    dut.operand_fp16_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.operand_fp32_i.value = BinaryValue(0, n_bits=32, bigEndian=True)

    for _ in range(1000):
        random_val_fp16 = np.float16(np.random.random_sample())
        random_val_bin_fp16 = float_to_float16_binary(random_val_fp16)
        random_val_fp32 = np.float32(np.random.random_sample())
        random_val_bin_fp32 = float_to_float32_binary(random_val_fp32)
        # dut._log.info(
        #     f"values: {random_val_fp16}, {random_val_fp32}, "
        #     f"{random_val_bin_fp16}, {random_val_bin_fp32}"
        # )
        await Timer(5, units="ps")
        dut.operand_fp16_i.value = random_val_bin_fp16
        dut.operand_fp32_i.value = random_val_bin_fp32
        await Timer(5, units="ps")
        read_out_bin = dut.result_o.value

        assert (
            binary_to_float32(read_out_bin)
            == np.float32(random_val_fp16) + random_val_fp32
        ), "a + b != result"


@cocotb.test()
async def fp_16_32_adder_with_fp16_denormals_test(dut) -> None:  # type: ignore[no-untyped-def]
    # Initial values
    dut.operand_fp16_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.operand_fp32_i.value = BinaryValue(0, n_bits=32, bigEndian=True)

    for _ in range(1000):
        denormal_fp16_bin_str = LogicArray(
            "0" + "00000" + bin(getrandbits(10))[2:].zfill(10)
        )
        denormal_val_bin_fp16 = binary_to_float16(denormal_fp16_bin_str)
        random_val_fp32 = np.float32(np.random.random_sample() * 1e-6)
        random_val_bin_fp32 = float_to_float32_binary(random_val_fp32)
        # dut._log.info(
        #     f"values: {random_val_fp16}, {random_val_fp32}, "
        #     f"{random_val_bin_fp16}, {random_val_bin_fp32}"
        # )
        await Timer(5, units="ps")
        dut.operand_fp16_i.value = denormal_fp16_bin_str
        dut.operand_fp32_i.value = random_val_bin_fp32
        await Timer(5, units="ps")
        read_out_bin = dut.result_o.value

        assert (
            binary_to_float32(read_out_bin)
            == np.float32(denormal_val_bin_fp16) + random_val_fp32
        ), "a + b != result"
