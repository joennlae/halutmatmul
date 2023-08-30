# pylint: disable=no-value-for-parameter, protected-access
from math import ceil, floor
from random import getrandbits
import numpy as np
import cocotb
from cocotb.triggers import Timer
from cocotb.clock import Clock
from cocotb.binary import BinaryValue

from util.helper_functions import binary_to_int32, int_8_to_binary, int_32_to_binary


@cocotb.test()
async def int_8_32_adder_test(dut) -> None:  # type: ignore[no-untyped-def]
    # Initial values
    dut.int_short_i.value = BinaryValue(0, n_bits=8, bigEndian=True)
    dut.int_long_i.value = BinaryValue(0, n_bits=32, bigEndian=True)

    await Timer(5, units="ps")
    for _ in range(1000):
        random_val_a = np.int8(floor(np.random.random() * 256 - 127))
        random_val_bin_a = int_8_to_binary(random_val_a)
        random_val_b = np.int32(ceil(np.random.random() * 256 - 127))
        random_val_bin_b = int_32_to_binary(random_val_b)
        dut._log.info(
            f"values: {random_val_a}, {random_val_b}, {random_val_bin_a}, {random_val_bin_b}"
        )
        await Timer(5, units="ps")
        dut.int_short_i.value = random_val_bin_a
        dut.int_long_i.value = random_val_bin_b
        await Timer(5, units="ps")
        read_out_bin = dut.int_long_o.value

        assert (
            binary_to_int32(read_out_bin) == random_val_a + random_val_b
        ), "a + b != result"
