# pylint: disable=no-value-for-parameter, protected-access
from math import ceil, log2
from random import getrandbits
import typing
import numpy as np
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotb.binary import BinaryValue
from cocotb.types import LogicArray


from util.helper_functions import (
    binary_to_float32,
    convert_fp16_array,
    convert_int_array_width,
    create_bin_vals_from_binstr,
    decoding_2d,
    encoding_function,
)

DATA_TYPE_WIDTH = 16
C = 32
K = 16
M = 32

EncUnits = 4
DecoderUnits = 16
DecUnitsX = M // DecoderUnits
CPerEncUnit = C // EncUnits
ThreshMemAddrWidth = int(log2(CPerEncUnit * K))
TreeDepth = int(log2(K))
TotalAddrWidth = int(log2(C * K))
DecAddrWidth = int(log2(DecoderUnits))
CAddrWidth = int(log2(C))
MAddrWidth = ceil(log2(M))

ROWS = 64


@cocotb.test()
async def halut_matmul_test(dut) -> None:  # type: ignore[no-untyped-def]
    # generate threshold table
    threshold_table = np.random.random((C * K)).astype(np.float16)
    input_a = np.random.random((ROWS, C, TreeDepth)).astype(np.float16)

    encoded, _, _ = encoding_function(
        threshold_table, input_a, tree_depth=TreeDepth, K=K
    )

    lut = np.random.random((M, C, K)).astype(np.float16)

    result, _ = decoding_2d(lut, encoded)
    print("results", encoded, result)
    cocotb.start_soon(Clock(dut.clk_i, 1, units="ns").start())

    # Initial values
    dut.a_input_enc_i.value = BinaryValue(
        0, n_bits=EncUnits * TreeDepth * DATA_TYPE_WIDTH, bigEndian=True
    )
    dut.waddr_enc_i.value = BinaryValue(
        0, n_bits=EncUnits * DATA_TYPE_WIDTH, bigEndian=True
    )
    dut.wdata_enc_i.value = BinaryValue(
        0, n_bits=EncUnits * DATA_TYPE_WIDTH, bigEndian=True
    )
    dut.we_enc_i.value = BinaryValue(0, n_bits=EncUnits, bigEndian=True)
    dut.encoder_i.value = 0
    dut.waddr_dec_i.value = BinaryValue(
        0, n_bits=DecoderUnits * TotalAddrWidth, bigEndian=True
    )
    dut.wdata_dec_i.value = BinaryValue(
        0, n_bits=DecoderUnits * DATA_TYPE_WIDTH, bigEndian=True
    )
    dut.m_addr_dec_i.value = BinaryValue(
        0, n_bits=DecoderUnits * DecAddrWidth, bigEndian=True
    )
    dut.we_dec_i.value = BinaryValue(0, n_bits=DecoderUnits, bigEndian=True)

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
        dut.we_enc_i.value = LogicArray([1, 1, 1, 1])
        ids = np.arange(EncUnits) * 16 + (idx // 16 * EncUnits * 16) + (idx % 16)
        dut.waddr_enc_i.value = w_addr_val
        dut.wdata_enc_i.value = convert_fp16_array(threshold_table[ids])
        await RisingEdge(dut.clk_i)
    dut._log.info("finished writing threshold values to encoder")
    dut.we_enc_i.value = LogicArray([0, 0, 0, 0])

    await RisingEdge(dut.clk_i)

    await RisingEdge(dut.clk_i)
    for m in range(DecoderUnits):
        for c in range(C):
            for k in range(K):
                dut.waddr_dec_i.value = convert_int_array_width(
                    [c * K + k for _ in range(DecUnitsX)], n_bits=TotalAddrWidth
                )
                dut.m_addr_dec_i.value = convert_int_array_width(
                    [m for _ in range(DecUnitsX)], n_bits=DecAddrWidth
                )
                dut.we_dec_i.value = LogicArray([1 for _ in range(DecUnitsX)])
                m_idx = np.arange(DecUnitsX) * DecoderUnits + m
                dut.wdata_dec_i.value = convert_fp16_array(lut[m_idx, c, k])
                await RisingEdge(dut.clk_i)
        dut._log.info(f"finished writing {m + 1}/{DecoderUnits}")

    dut.we_dec_i.value = 0

    await RisingEdge(dut.clk_i)

    current_encoder_input = np.zeros((EncUnits * TreeDepth), dtype=np.float16)
    idx_encoder_input_base = np.arange(TreeDepth) * TreeDepth
    idx_encoder_input_top = np.arange(TreeDepth) * TreeDepth + TreeDepth

    dut.encoder_i.value = 1
    await RisingEdge(dut.clk_i)
    for row in range(input_a.shape[0] + 1):
        for c_ in range(input_a.shape[1]):
            if row < ROWS:
                dut.encoder_i.value = 1
                current_encoder_input[
                    idx_encoder_input_base[c_ % 4] : idx_encoder_input_top[c_ % 4]
                ] = input_a[row, c_]
                dut.a_input_enc_i.value = convert_fp16_array(current_encoder_input)

            await RisingEdge(dut.clk_i)
            # dut._log.info(
            #     f"(r) : {row}, {c_}, \n"
            #     f"{dut.valid_o.value}, {dut.result_o.value}, {dut.m_addr_o.value}\n"
            #     f"{dut.valid_enc_o.value}"
            # )
            # do asserts encoder
            if (not (row == 0 and c_ < 5)) and (not row == ROWS and c > 5):
                lookup_row = row - 1 if c_ < 5 else row
                lookup_c = np.arange(C)[c_ - 5]
                assert dut.valid_enc_o.value == 1, "enc output not valid"
                assert (
                    dut.k_addr_enc_o.value.value == encoded[lookup_row, c_ - 5]
                ), "encoded output wrong"
                assert (
                    dut.c_addr_enc_o.value.value == lookup_c
                ), "enc assert wrong c output"

            if not (row == 0 or (row == 1 and c_ < 6)) and c_ >= 6:
                lookup_m = c_ - 6
                assert (
                    DecoderUnits < C - 6
                ), "DecoderUnits too high for this logic (and max_fan_out)"
                if lookup_m in range(DecoderUnits):
                    m_addr_out = dut.m_addr_o.value
                    result_o_out = dut.result_o.value
                    m_addr_bin_vals = create_bin_vals_from_binstr(
                        m_addr_out.binstr, DecUnitsX
                    )
                    result_o_bin_vals = create_bin_vals_from_binstr(
                        result_o_out.binstr, DecUnitsX
                    )
                    for i in range(DecUnitsX):
                        assert dut.valid_o[i].value == 1, f"not valid result {i}"
                        assert (
                            m_addr_bin_vals[i] == lookup_m + DecoderUnits * i
                        ), f"m_addr_o[{i}] wrong"
                        assert (
                            binary_to_float32(result_o_bin_vals[i])
                            == result[row - 1, lookup_m + DecoderUnits * i]
                        ), "result_o[{i}] wrong"
                else:
                    assert dut.valid_o.value == BinaryValue(
                        0, n_bits=DecUnitsX
                    ), "should be invalid"
                if row == ROWS and c == 6 + DecoderUnits - 6:
                    dut.encoder_i.value = 0  # turn off
            else:
                assert dut.valid_o.value == BinaryValue(
                    0, n_bits=DecUnitsX
                ), "should be invalid"
        dut._log.info(f"finished applying input row {row + 1}/{ROWS}")
