# pylint: disable=no-value-for-parameter, protected-access
from math import log2
from random import getrandbits
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
from cocotb.binary import BinaryValue, BinaryRepresentation
from cocotb.types import LogicArray

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


def float_to_float16_binary(fl: np.float16) -> BinaryValue:
    # pylint: disable=too-many-function-args
    # fl = 0.33325195 -> '0011010101010101' # big endian flip for little endian
    return LogicArray(bin(np.float16(fl).view("H"))[2:].zfill(16)[::-1]).to_BinaryValue(
        bigEndian=False
    )


def binary_to_float16(binary: BinaryValue) -> np.float16:
    bin_str = binary.binstr[::-1]  # back to big endian
    padded_bits = bin_str + "0" * ((8 - len(bin_str) % 8) if len(bin_str) % 8 else 0)
    bytes_list = list(int(padded_bits, 2).to_bytes(len(padded_bits) // 8, "big"))
    # print(bin_str, padded_bits, bytes_list, bytes(bytes_list))
    dt = np.dtype(np.float16)
    dt = dt.newbyteorder(">")
    return np.frombuffer(bytes(bytes_list), dtype=dt, count=-1)[0]


@cocotb.test()
async def read_write_test_extended(dut) -> None:  # type: ignore[no-untyped-def]
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    # Initial values
    dut.test_en_i.value = 0
    dut.raddr_a_i.value = BinaryValue(0, n_bits=16, bigEndian=False)
    dut.waddr_a_i.value = BinaryValue(0, n_bits=16, bigEndian=False)
    dut.wdata_a_i.value = BinaryValue(0, n_bits=16, bigEndian=False)
    dut.we_a_i.value = 0

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    for _ in range(100):
        random_val = np.float16(np.random.random_sample())
        random_val_bin = float_to_float16_binary(random_val)
        random_addr = getrandbits(TOTAL_ADDR_WIDTH)
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

        assert read_out_bin == random_val_bin, "read != write"
        assert (
            binary_to_float16(read_out_bin) == random_val
        ), "float -> bin -> float != float"
