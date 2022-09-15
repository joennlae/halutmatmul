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
from cocotb.types import LogicArray


from util.helper_functions import (
    convert_fp16_array,
    convert_int_array_width,
    encoding_function,
)

DATA_TYPE_WIDTH = 16
C = int(os.environ.get("NUM_C", 32))
K = 16

EncUnits = 4
CPerEncUnit = C // EncUnits
ThreshMemAddrWidth = int(log2(CPerEncUnit * K))
TreeDepth = int(log2(K))

ROWS = 64


@cocotb.test()
async def halut_encoder_4_test(dut) -> None:  # type: ignore[no-untyped-def]
    # generate threshold table
    threshold_table = np.random.random((C * K)).astype(np.float16)
    input_a = np.random.random((ROWS, C, TreeDepth)).astype(np.float16)

    encoded, kaddr_hist, thres_mem_hist = encoding_function(
        threshold_table, input_a, tree_depth=TreeDepth, K=K
    )

    print("encoded", encoded, kaddr_hist, thres_mem_hist)
    cocotb.start_soon(Clock(dut.clk_i, 1, units="ns").start())

    # Initial values
    dut.a_input_i.value = BinaryValue(
        0, n_bits=EncUnits * 4 * DATA_TYPE_WIDTH, bigEndian=True
    )
    dut.waddr_i.value = BinaryValue(
        0, n_bits=EncUnits * ThreshMemAddrWidth, bigEndian=True
    )
    dut.wdata_i.value = BinaryValue(
        0, n_bits=EncUnits * DATA_TYPE_WIDTH, bigEndian=True
    )
    dut.we_i.value = BinaryValue(0, n_bits=EncUnits, bigEndian=True)
    dut.encoder_i.value = 0

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(6):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # write threshold values

    await RisingEdge(dut.clk_i)
    for idx in range(threshold_table.shape[0] // EncUnits):
        w_addr_val = convert_int_array_width(
            [idx, idx, idx, idx], n_bits=ThreshMemAddrWidth
        )
        dut.we_i.value = LogicArray([1, 1, 1, 1])
        ids = np.arange(EncUnits) * 16 + (idx // 16 * EncUnits * 16) + (idx % 16)
        dut.waddr_i.value = w_addr_val
        dut.wdata_i.value = convert_fp16_array(threshold_table[ids])
        # dut._log.info(
        #     f"write {dut.waddr_i.value}, {w_addr_val}, {ids}, {dut.wdata_i.value}"
        # )
        await RisingEdge(dut.clk_i)
    dut.we_i.value = LogicArray([0, 0, 0, 0])

    current_encoder_input = np.zeros((EncUnits * 4), dtype=np.float16)
    idx_encoder_input_base = np.arange(4) * 4
    idx_encoder_input_top = np.arange(4) * 4 + 4

    current_encoder_input[
        idx_encoder_input_base[0 % 4] : idx_encoder_input_top[0 % 4]
    ] = input_a[0, 0]
    dut.encoder_i.value = 1
    dut.a_input_i.value = convert_fp16_array(current_encoder_input)
    await RisingEdge(dut.clk_i)
    for row in range(input_a.shape[0]):
        for c_ in range(input_a.shape[1]):
            current_encoder_input[
                idx_encoder_input_base[(c_ + 1) % 4] : idx_encoder_input_top[
                    (c_ + 1) % 4
                ]
            ] = input_a[
                (row + (1 if (c_ + 1) == input_a.shape[1] else 0)) % input_a.shape[0],
                (c_ + 1) % input_a.shape[1],
            ]
            dut.a_input_i.value = convert_fp16_array(current_encoder_input)
            await RisingEdge(dut.clk_i)
            # dut._log.info(
            #     f"(int_vals) : {dut.encoder_int.value}, {dut.k_addr_o_q.value}, "
            #     f"{dut.c_addr_o_q.value}, {dut.valid_counter.value}, {dut.k_addr_int.value}"
            #     f"{dut.valid_int.value}, {dut.c_addr_int.value}, {c_}"
            # )
            # do asserts
            if not (row == 0 and c_ < 5):
                lookup_row = row - 1 if c_ < 5 else row
                lookup_c = np.arange(C)[c_ - 5]
                # dut._log.info(
                #     f"(assert) : {dut.valid_o.value}, {dut.k_addr_o.value}, "
                #     f"{dut.c_addr_o.value}\n"
                #     f"(compare): 1, {encoded[lookup_row, c_ - 5]}, {lookup_c}"
                # )
                assert dut.valid_o.value == 1, "output not valid"
                assert (
                    dut.k_addr_o.value.value == encoded[lookup_row, c_ - 5]
                ), "encoded output wrong"
                assert dut.c_addr_o.value.value == lookup_c, "assert wrong c output"
