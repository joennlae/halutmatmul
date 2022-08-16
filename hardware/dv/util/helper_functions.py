import typing
import numpy as np
from cocotb.binary import BinaryValue
from cocotb.types import LogicArray


def convert_fp16_array(vals: np.ndarray) -> BinaryValue:
    bin_vals = []
    for val in vals:
        bin_vals.append(float_to_float16_binary(val))
    return fuse_binary_values(bin_vals)


def fuse_binary_values(vals: "typing.List[BinaryValue]") -> BinaryValue:
    tot_binstr = ""
    for val in vals:
        tot_binstr += val.binstr
    return LogicArray(tot_binstr).to_BinaryValue(bigEndian=True)


def float_to_float16_binary(fl: np.float16) -> BinaryValue:
    # pylint: disable=too-many-function-args
    # fl = 0.33325195 -> '0011010101010101' # big endian flip for little endian
    return LogicArray(bin(np.float16(fl).view("u2"))[2:].zfill(16)).to_BinaryValue(
        bigEndian=True
    )


def binary_to_float16(binary: BinaryValue) -> np.float16:
    bin_str = binary.binstr  # back to big endian
    padded_bits = bin_str + "0" * ((8 - len(bin_str) % 8) if len(bin_str) % 8 else 0)
    bytes_list = list(int(padded_bits, 2).to_bytes(len(padded_bits) // 8, "big"))
    # print(bin_str, padded_bits, bytes_list, bytes(bytes_list))
    dt = np.dtype(np.float16)
    dt = dt.newbyteorder(">")
    return np.frombuffer(bytes(bytes_list), dtype=dt, count=-1)[0]


def float_to_float32_binary(fl: np.float32) -> BinaryValue:
    # pylint: disable=too-many-function-args
    # fl = 0.33325195 -> '0011010101010101' # big endian flip for little endian
    # numpy could control it with ">H" big-endian, "<H" little-endian
    # u4 -> uint32
    return LogicArray(bin(np.float32(fl).view("u4"))[2:].zfill(32)).to_BinaryValue(
        bigEndian=True
    )


def binary_to_float32(binary: BinaryValue) -> np.float32:
    bin_str = binary.binstr  # back to big endian
    padded_bits = bin_str + "0" * ((8 - len(bin_str) % 8) if len(bin_str) % 8 else 0)
    bytes_list = list(int(padded_bits, 2).to_bytes(len(padded_bits) // 8, "big"))
    # print(bin_str, padded_bits, bytes_list, bytes(bytes_list))
    dt = np.dtype(np.float32)
    dt = dt.newbyteorder(">")
    return np.frombuffer(bytes(bytes_list), dtype=dt, count=-1)[0]
