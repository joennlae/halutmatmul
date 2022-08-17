import typing
import numpy as np
from cocotb.binary import BinaryValue
from cocotb.types import LogicArray


def convert_fp16_array(vals: np.ndarray) -> BinaryValue:
    bin_vals = []
    for val in vals:
        bin_vals.append(float_to_float16_binary(val))
    return fuse_binary_values(bin_vals)


def convert_int_array_width(vals: "typing.List[int]", n_bits: int = 4) -> BinaryValue:
    bin_vals = []
    for val in vals:
        bin_vals.append(
            BinaryValue(val, n_bits=n_bits, bigEndian=False)
        )  # hack to get bigEndian!!
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


def encoding_function(
    threshold_table: np.ndarray, input_a: np.ndarray, tree_depth: int = 4, K: int = 16
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
