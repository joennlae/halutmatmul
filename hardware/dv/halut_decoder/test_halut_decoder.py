# pylint: disable=no-value-for-parameter, protected-access
from math import log2
import os
from random import getrandbits
import typing
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotb.binary import BinaryValue


from util.helper_functions import (
    binary_to_float32,
    float_to_float16_binary,
)

DATA_TYPE_WIDTH = 16
C = int(os.environ.get("NUM_C", 32))
K = 16
ROWS = 64


def decoding(
    lut: np.ndarray, encoded: np.ndarray
) -> "typing.Tuple[np.ndarray, np.ndarray]":
    result = np.zeros((encoded.shape[0]), dtype=np.float32)
    result_history = np.zeros((encoded.shape[0], C), dtype=np.float32)
    for c in range(C):
        result_history[:, c] = result
        result += lut[c, encoded[:, c]]
    return result, result_history


@cocotb.test()
async def halut_decoder_test(dut) -> None:  # type: ignore[no-untyped-def]
    # generate threshold table
    lut = np.random.random((C, K)).astype(np.float16)
    encoded = (np.random.random((ROWS, C)) * 16).astype(np.int32)

    # pylint: disable=unused-variable
    result, result_history = decoding(lut, encoded)
    print("encoded", encoded, lut, result)
    cocotb.start_soon(Clock(dut.clk_i, 1, units="ns").start())

    # Initial values
    dut.waddr_i.value = BinaryValue(0, n_bits=int(log2(C * K)), bigEndian=True)
    dut.wdata_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.we_i.value = 0
    dut.decoder_i.value = 0
    dut.c_addr_i.value = BinaryValue(0, n_bits=int(log2(C)))
    dut.k_addr_i.value = BinaryValue(0, n_bits=int(log2(K)))

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # write lut values
    await RisingEdge(dut.clk_i)
    for c in range(C):
        for k in range(K):
            dut.waddr_i.value = c * K + k
            dut.we_i.value = 1
            dut.wdata_i.value = float_to_float16_binary(lut[c, k])
            await RisingEdge(dut.clk_i)

    dut.we_i.value = 0

    await RisingEdge(dut.clk_i)

    dut.decoder_i.value = 1
    for row in range(encoded.shape[0]):
        for c in range(C):
            # dut._log.info(
            #     f"r: {row}, c: {c}, valid_o: {dut.valid_o.value}, \n"
            #     f"res_int: {binary_to_float32(dut.result_int_q.value)}, \n"
            #     f"res_expected: {result_history[row, c - 2]} \n"
            #     f"res: {binary_to_float32(dut.result_o.value)}\n"
            # )
            if row > 0 and c == 2:
                assert dut.valid_o.value == 1, "output not valid"
                assert binary_to_float32(dut.result_o.value) == np.float32(
                    result[row - 1]
                ), "result not correct"
            if row > 0 and c > 1:
                assert (
                    binary_to_float32(dut.result_int_q.value)
                    == result_history[row, c - 2]
                ), "res_history not correct"
            dut.c_addr_i.value = int(c)
            dut.k_addr_i.value = int(encoded[row, c])
            await RisingEdge(dut.clk_i)
