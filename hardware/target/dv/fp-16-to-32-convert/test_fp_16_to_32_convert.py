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
)


@cocotb.test()
async def fp_16_to_32_convert_test(dut) -> None:  # type: ignore[no-untyped-def]
    # Initial values
    await Timer(5, units="ps")
    dut.operand_fp16_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    await Timer(5, units="ps")

    for _ in range(1000):
        random_val_fp16 = np.float16(np.random.random_sample())
        random_val_bin_fp16 = float_to_float16_binary(random_val_fp16)
        # dut._log.info(f"values: {random_val_fp16}, {random_val_bin_fp16}")
        await Timer(5, units="ps")
        dut.operand_fp16_i.value = random_val_bin_fp16
        await Timer(5, units="ps")
        read_out_bin = dut.result_o.value

        assert binary_to_float32(read_out_bin) == np.float32(
            random_val_fp16
        ), "conversion wrong"


@cocotb.test()
async def fp_16_to_32_convert_denormals_zeros_test(dut) -> None:  # type: ignore[no-untyped-def]
    # Initial values
    await Timer(5, units="ps")
    dut.operand_fp16_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    await Timer(5, units="ps")

    random_val_fp16 = np.float16(0)
    random_val_bin_fp16 = float_to_float16_binary(random_val_fp16)
    dut._log.info(f"values: {random_val_fp16}, {random_val_bin_fp16}")
    await Timer(5, units="ps")
    dut.operand_fp16_i.value = random_val_bin_fp16
    await Timer(5, units="ps")
    read_out_bin = dut.result_o.value

    assert binary_to_float32(read_out_bin) == np.float32(
        random_val_fp16
    ), "zero conversion wrong"

    for _ in range(100):
        denormal_fp16_bin_str = LogicArray(
            "0" + "00000" + bin(getrandbits(10))[2:].zfill(10)
        )
        denormal_val_bin_fp16 = binary_to_float16(denormal_fp16_bin_str)
        # dut._log.info(f"values: {denormal_val_bin_fp16}, {denormal_fp16_bin_str}")
        await Timer(5, units="ps")
        dut.operand_fp16_i.value = denormal_fp16_bin_str
        await Timer(5, units="ps")
        read_out_bin = dut.result_o.value

        assert binary_to_float32(read_out_bin) == np.float32(
            denormal_val_bin_fp16
        ), "denormal conversion wrong"
