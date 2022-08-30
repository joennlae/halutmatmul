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
    binary_to_float32,
    decoding_2d,
    float_to_float16_binary,
)

DATA_TYPE_WIDTH = 16
C = 32
K = 16
ROWS = 64

DecoderUnits = 16
TotalAddrWidth = int(log2(C * K))
DecAddrWidth = int(log2(DecoderUnits))
CAddrWidth = int(log2(C))
TreeDepth = int(log2(K))


@cocotb.test()
async def halut_decoder_x_test(dut) -> None:  # type: ignore[no-untyped-def]
    # generate threshold table
    lut = np.random.random((DecoderUnits, C, K)).astype(np.float16)
    encoded = (np.random.random((ROWS, C)) * K).astype(np.int32)

    # pylint: disable=unused-variable
    result, result_history = decoding_2d(lut, encoded)
    print("encoded", encoded, lut, result)
    cocotb.start_soon(Clock(dut.clk_i, 1, units="ns").start())

    # Initial values
    dut.waddr_i.value = BinaryValue(0, n_bits=TotalAddrWidth, bigEndian=True)
    dut.wdata_i.value = BinaryValue(0, n_bits=16, bigEndian=True)
    dut.m_addr_i.value = BinaryValue(0, n_bits=DecAddrWidth, bigEndian=True)
    dut.we_i.value = 0
    dut.decoder_i.value = 0
    dut.c_addr_i.value = BinaryValue(0, n_bits=CAddrWidth, bigEndian=True)
    dut.k_addr_i.value = BinaryValue(0, n_bits=TreeDepth, bigEndian=True)

    # Reset DUT
    dut.rst_ni.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk_i)
    dut.rst_ni.value = 1

    # write lut values
    await RisingEdge(dut.clk_i)
    for m in range(DecoderUnits):
        for c in range(C):
            for k in range(K):
                dut.waddr_i.value = c * K + k
                dut.m_addr_i.value = int(m)
                dut.we_i.value = 1
                dut.wdata_i.value = float_to_float16_binary(lut[m, c, k])
                # dut._log.info(
                #     f"write_input: {dut.gen_decoders[0].sub_unit_decoder.wdata_i.value}\n"
                #     f"write_addr: {dut.gen_decoders[0].sub_unit_decoder.waddr_i.value}\n"
                #     f"write_enable: {dut.gen_decoders[0].sub_unit_decoder.we_i.value}\n"
                #     f"write_enable: {dut.gen_decoders[1].sub_unit_decoder.we_i.value}\n"
                #     f"write_enable: {dut.gen_decoders[2].sub_unit_decoder.we_i.value}\n"
                # )
                await RisingEdge(dut.clk_i)
        dut._log.info(f"finished writing {m + 1}/{DecoderUnits}")

    dut.we_i.value = 0

    await RisingEdge(dut.clk_i)

    dut.decoder_i.value = 1
    for row in range(encoded.shape[0] + 1):
        for c in range(C):
            # dut._log.info(
            #     f"r: {row}, c: {c}, valid_o: {dut.valid_o.value}, \n"
            #     f"dec_0_read: {dut.gen_decoders[0].sub_unit_decoder.rdata_o.value}\n"
            #     f"dec_0_addr: {dut.gen_decoders[0].sub_unit_decoder.raddr.value}\n"
            #     f"dec_1_read: {dut.gen_decoders[1].sub_unit_decoder.rdata_o.value}\n"
            # )
            if row > 0 and c >= 3:
                if c < 2 + DecoderUnits:
                    dut._log.info(f"c:{c - 3}, r:{row - 1}")
                    assert dut.valid_o.value == 1, "output not valid"
                    assert binary_to_float32(dut.result_o.value) == np.float32(
                        result[row - 1, c - 3]  # [N, M]
                    ), "result not correct"
                    assert dut.m_addr_o.value.value == c - 3, " m_addr_o wrong"
                elif c >= 3 + DecoderUnits:
                    assert dut.valid_o.value == 0, "output should be invalid"
                if row == ROWS and c == 3 + DecoderUnits - 3:
                    dut.decoder_i.value = 0  # turn of decoder
            else:
                assert dut.valid_o.value == 0, "output should be invalid"
            if row < encoded.shape[0]:
                dut.c_addr_i.value = int(c)
                dut.k_addr_i.value = int(encoded[row, c])
            else:
                dut.c_addr_i.value = int(0)  # set zero address to garantee running
            await RisingEdge(dut.clk_i)
        dut._log.info(f"finished row {row + 1} / {encoded.shape[0]}")
