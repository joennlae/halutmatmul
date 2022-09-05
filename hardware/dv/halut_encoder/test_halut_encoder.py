# pylint: disable=no-value-for-parameter, protected-access
from math import log2
from random import getrandbits
import typing
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
from cocotb.binary import BinaryValue


from util.helper_functions import (
    binary_to_float16,
    convert_fp16_array,
    encoding_function,
    float_to_float16_binary,
)

DATA_TYPE_WIDTH = 16
C = 32
K = 16

CLOCK_PERIOD_PS = 3000

EncUnits = 4
CPerEncUnit = C // EncUnits
ThreshMemAddrWidth = int(log2(CPerEncUnit * K))
TreeDepth = int(log2(K))

ROWS = 1


@cocotb.test()
async def halut_encoder_test(dut) -> None:  # type: ignore[no-untyped-def]
    # generate threshold table
    np.random.seed(4419)
    threshold_table = np.random.random((CPerEncUnit * K)).astype(np.float16)
    input_a = np.random.random((ROWS, CPerEncUnit, 4)).astype(np.float16)

    encoded, kaddr_hist, thres_mem_hist = encoding_function(
        threshold_table, input_a, tree_depth=TreeDepth, K=K
    )

    print("encoded", encoded, kaddr_hist, thres_mem_hist)
    cocotb.start_soon(Clock(dut.clk_i, 3.0, units="ns").start())

    # Initial values
    dut.a_input_i.value = BinaryValue(0, n_bits=4 * 16, bigEndian=True)
    dut.waddr_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.wdata_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.we_i.value = 0
    dut.encoder_i.value = 0

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # write threshold values
    await RisingEdge(dut.clk_i)
    for idx in range(threshold_table.shape[0]):
        await Timer(CLOCK_PERIOD_PS / 2, "ps")
        dut.waddr_i.value = idx
        dut.we_i.value = 1
        dut.wdata_i.value = float_to_float16_binary(threshold_table[idx])
        await RisingEdge(dut.clk_i)

    await Timer(CLOCK_PERIOD_PS / 2, "ps")
    dut.we_i.value = 0

    await RisingEdge(dut.clk_i)
    for row in range(input_a.shape[0]):
        for c_ in range(input_a.shape[1]):
            await Timer(CLOCK_PERIOD_PS / 2, "ps")
            dut.a_input_i.value = convert_fp16_array(input_a[row, c_])
            dut.encoder_i.value = 1
            await RisingEdge(dut.clk_i)
            # do asserts
            dut._log.info(
                f"(assert) : {dut.valid_o.value}, {dut.k_addr_o.value}, {dut.c_addr_o.value}"
            )
            if not (row == 0 and c_ == 0):
                assert dut.valid_o.value == 1, "not a valid output"
                read_out_k_addr_bin = dut.k_addr_o.value
                read_out_c_addr_bin = dut.c_addr_o.value

                # check last
                assert (
                    read_out_k_addr_bin.value
                    == encoded[row - (1 if c_ == 0 else 0), c_ - 1]
                ), "encoded value wrong"
                assert (
                    read_out_c_addr_bin.value
                    == (np.arange(input_a.shape[1]) * 4)[c_ - 1]
                ), "c value wrong"
            dut._log.info(f"c_out: {dut.c_addr_o.value.value}")
            # logging
            # dut._log.info(
            #     f"(0) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 0]}\n"
            #     f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
            #     f"thresh_mem_hist: {thres_mem_hist[row, c_, 0]}\n"
            #     f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            # )
            # history asserts
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 0], "(0) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 0]
            ), "(0) thres_mem wrong"
            await RisingEdge(dut.clk_i)
            # dut._log.info(
            #     f"(1) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 1]}\n"
            #     f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
            #     f"thresh_mem_hist: {thres_mem_hist[row, c_, 1]}\n"
            #     f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            # )
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 1], "(1) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 1]
            ), "(1) thres_mem wrong"
            await RisingEdge(dut.clk_i)
            # dut._log.info(
            #     f"(2) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 2]}\n"
            #     f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
            #     f"thresh_mem_hist: {thres_mem_hist[row, c_, 2]}\n"
            #     f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            # )
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 2], "(2) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 2]
            ), "(2) thres_mem wrong"
            await RisingEdge(dut.clk_i)
            # dut._log.info(
            #     f"(3) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 3]}\n"
            #     f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
            #     f"thresh_mem_hist: {thres_mem_hist[row, c_, 3]}\n"
            #     f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            # )
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 3], "(3) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 3]
            ), "(3) thres_mem wrong"
