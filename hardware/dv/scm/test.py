# pylint: disable=no-value-for-parameter, protected-access
from math import log2
from random import getrandbits
import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
from cocotb.binary import BinaryValue, BinaryRepresentation

DATA_TYPE_WIDTH = 16
C = 32
M = 1
K = 16
SUB_UNIT_ADDR_WIDTH = 5
TOTAL_ADDR_WIDTH = int(log2(C * K))
TOTAL_DATA_WIDTH = M * DATA_TYPE_WIDTH


@cocotb.test()
async def read_write_test(dut) -> None:  # type: ignore[no-untyped-def]
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    # Initial values
    dut.test_en_i.value = 0
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
    dut.test_en_i.value = 0
    dut.raddr_a_i.value = 0
    dut.waddr_a_i.value = 0
    dut.wdata_a_i.value = 0
    dut.we_a_i.value = 0

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    for _ in range(100):
        random_val = getrandbits(DATA_TYPE_WIDTH)
        random_addr = getrandbits(TOTAL_ADDR_WIDTH)
        # dut._log.info(f"value: {random_val}, addr: {random_addr}")
        await RisingEdge(dut.clk_i)
        dut.waddr_a_i.value = random_addr
        dut.we_a_i.value = 1
        dut.wdata_a_i.value = random_val
        await RisingEdge(dut.clk_i)
        dut.we_a_i.value = 0
        dut.raddr_a_i.value = random_addr
        await RisingEdge(dut.clk_i)
        assert dut.rdata_a_o.value == random_val, "read != write"
