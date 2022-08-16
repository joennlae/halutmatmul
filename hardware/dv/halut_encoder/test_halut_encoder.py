# pylint: disable=no-value-for-parameter, protected-access
from math import log2
from random import getrandbits
import typing
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotb.binary import BinaryValue


from util.helper_functions import (
    binary_to_float16,
    convert_fp16_array,
    float_to_float16_binary,
)

DATA_TYPE_WIDTH = 16
C = 32
K = 16

EncUnits = 4
CPerEncUnit = C // EncUnits
ThreshMemAddrWidth = int(log2(CPerEncUnit * K))
TreeDepth = int(log2(K))


def encoding_function(
    threshold_table: np.ndarray, input_a: np.ndarray, tree_depth: int = 4
) -> "typing.Tuple[np.ndarray, np.ndarray, np.ndarray]":
    encoded = np.zeros((input_a.shape[0], input_a.shape[1]), dtype=np.int32)
    kaddr_history = np.zeros(
        (input_a.shape[0], input_a.shape[1], tree_depth), dtype=np.int32
    )
    thresh_mem_history = np.zeros(
        (input_a.shape[0], input_a.shape[1], tree_depth), dtype=np.float16
    )
    # CPerEncUnit = input_a.shape[1]
    caddr_internal_offset = np.arange(input_a.shape[1]) * K
    prototype_addr_internal_offset = 2 ** np.arange(tree_depth + 1) - 1
    for row in range(input_a.shape[0]):
        kaddr = np.zeros(input_a.shape[1], dtype=np.int64)
        prototype_addr_internal = np.zeros(input_a.shape[1], dtype=np.int64)
        for tree_level_cnt in range(tree_depth):
            kaddr_history[row, :, tree_level_cnt] = kaddr
            data_thresh_mem_o = threshold_table[
                prototype_addr_internal + caddr_internal_offset
            ]
            thresh_mem_history[row, :, tree_level_cnt] = data_thresh_mem_o
            data_input_comparision = input_a[row, :, tree_level_cnt]
            fp_16_comparision_o = data_input_comparision > data_thresh_mem_o
            kaddr = (kaddr * 2) + fp_16_comparision_o
            prototype_addr_internal = (
                kaddr + prototype_addr_internal_offset[tree_level_cnt + 1]
            )
        encoded[row] = kaddr
    return encoded, kaddr_history, thresh_mem_history


@cocotb.test()
async def halut_encoder_test(dut) -> None:  # type: ignore[no-untyped-def]
    # generate threshold table
    threshold_table = np.random.random((CPerEncUnit * K)).astype(np.float16)
    input_a = np.random.random((64, CPerEncUnit, 4)).astype(np.float16)

    encoded, kaddr_hist, thres_mem_hist = encoding_function(
        threshold_table, input_a, tree_depth=TreeDepth
    )

    print("encoded", encoded, kaddr_hist, thres_mem_hist)
    cocotb.start_soon(Clock(dut.clk_i, 1, units="ns").start())

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
        dut.waddr_i.value = idx
        dut.we_i.value = 1
        dut.wdata_i.value = float_to_float16_binary(threshold_table[idx])
        await RisingEdge(dut.clk_i)

    dut.we_i.value = 0

    await RisingEdge(dut.clk_i)
    for row in range(input_a.shape[0]):
        for c_ in range(input_a.shape[1]):
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
                    read_out_c_addr_bin.value == np.arange(input_a.shape[1])[c_ - 1]
                ), "c value wrong"
            # logging
            dut._log.info(
                f"(0) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 0]}\n"
                f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
                f"thresh_mem_hist: {thres_mem_hist[row, c_, 0]}\n"
                f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            )
            # history asserts
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 0], "(0) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 0]
            ), "(0) thres_mem wrong"
            await RisingEdge(dut.clk_i)
            dut._log.info(
                f"(1) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 1]}\n"
                f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
                f"thresh_mem_hist: {thres_mem_hist[row, c_, 1]}\n"
                f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            )
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 1], "(1) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 1]
            ), "(1) thres_mem wrong"
            await RisingEdge(dut.clk_i)
            dut._log.info(
                f"(2) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 2]}\n"
                f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
                f"thresh_mem_hist: {thres_mem_hist[row, c_, 2]}\n"
                f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            )
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 2], "(2) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 2]
            ), "(2) thres_mem wrong"
            await RisingEdge(dut.clk_i)
            dut._log.info(
                f"(3) k_addr: {dut.k_addr.value.value}, kaddr_hist: {kaddr_hist[row, c_, 3]}\n"
                f"    thresh_mem: {binary_to_float16(dut.data_thresh_mem_o.value)}, "
                f"thresh_mem_hist: {thres_mem_hist[row, c_, 3]}\n"
                f"    read_addr_thresh_mem: {dut.read_addr_thresh_mem.value.value}"
            )
            assert dut.k_addr.value.value == kaddr_hist[row, c_, 3], "(3) k_addr wrong"
            assert (
                binary_to_float16(dut.data_thresh_mem_o.value)
                == thres_mem_hist[row, c_, 3]
            ), "(3) thres_mem wrong"
