# pylint: disable=no-value-for-parameter, protected-access
from math import log2
import os
from random import getrandbits
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotb.binary import BinaryValue

from util.helper_functions import binary_to_float16, float_to_float16_binary

# be sure to get the correct scm__$hash$ from the generated file !!
DATA_TYPE_WIDTH = int(os.environ.get("DATA_WIDTH", 16))
C = int(os.environ.get("NUM_C", 32))
M = 1
K = 16
SUB_UNIT_ADDR_WIDTH = 5
TOTAL_ADDR_WIDTH = int(log2(C * K))
TOTAL_DATA_WIDTH = M * DATA_TYPE_WIDTH


@cocotb.test()
async def read_write_test(dut) -> None:  # type: ignore[no-untyped-def]
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    # Initial values
    dut.raddr_a_i.value = 0
    dut.waddr_a_i.value = 0
    dut.wdata_a_i.value = 0
    dut.we_a_i.value = 0

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    await RisingEdge(dut.clk_i)
    dut.we_a_i.value = 1
    dut.wdata_a_i.value = 4419

    await RisingEdge(dut.clk_i)
    dut._log.info(f"value {dut.wdata_a_i.value}")

    dut.we_a_i.value = 0
    dut.raddr_a_i.value = 0
    await RisingEdge(dut.clk_i)

    assert dut.rdata_a_o.value == 4419, "read != write"


@cocotb.test()
async def read_write_test_extended(dut) -> None:  # type: ignore[no-untyped-def]
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    # Initial values
    dut.raddr_a_i.value = BinaryValue(0, n_bits=TOTAL_ADDR_WIDTH, bigEndian=True)
    dut.waddr_a_i.value = BinaryValue(0, n_bits=TOTAL_ADDR_WIDTH, bigEndian=True)
    dut.wdata_a_i.value = BinaryValue(0, n_bits=DATA_TYPE_WIDTH, bigEndian=True)
    dut.we_a_i.value = 0

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    for _ in range(1000):
        random_val = np.float16(np.random.random_sample())
        random_val_bin = float_to_float16_binary(random_val)
        random_addr = getrandbits(TOTAL_ADDR_WIDTH - 1)
        # dut._log.info(f"value: {random_val}, {random_val_bin}, addr: {random_addr}")
        await RisingEdge(dut.clk_i)
        dut.waddr_a_i.value = random_addr
        dut.we_a_i.value = 1
        dut.wdata_a_i.value = random_val_bin
        await RisingEdge(dut.clk_i)
        dut.we_a_i.value = 0
        dut.raddr_a_i.value = random_addr
        await RisingEdge(dut.clk_i)
        read_out_bin = dut.rdata_a_o.value
        assert (
            read_out_bin == random_val_bin
        ), f"read != write, {read_out_bin}, {random_val_bin}"
        assert (
            binary_to_float16(read_out_bin) == random_val
        ), "float -> bin -> float != float"
