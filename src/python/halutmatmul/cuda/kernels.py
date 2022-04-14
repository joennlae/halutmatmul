import os
import re
from pathlib import Path
import cupy as cp  # type: ignore[import]

import halutmatmul.halutmatmul as hm


def create_encode_kernel(
    C: int = 16, num_splits: int = 4, info_offset: int = 8
) -> cp.RawKernel:
    script_folder = Path(os.path.dirname(os.path.abspath(__file__)))
    file_to_open = script_folder / "kernels/encode.cu"
    with open(file_to_open, "r") as f:
        halut_encode_code = f.read()

        halut_encode_code = re.sub(
            r"const int num_splits = \d;\n",
            "const int num_splits = " + str(num_splits) + ";\n",
            halut_encode_code,
            flags=re.MULTILINE,
        )
        halut_encode_code = re.sub(
            r"const int C = \d+;\n",
            "const int C = " + str(C) + ";\n",
            halut_encode_code,
            flags=re.MULTILINE,
        )
        halut_encode_code = re.sub(
            r"const int info_offset = \d;\n",
            "const int info_offset = " + str(info_offset) + ";\n",
            halut_encode_code,
            flags=re.MULTILINE,
        )
        # print(halut_encode_code)
        halut_encode_kernel = cp.RawKernel(
            halut_encode_code,  # defined in file kernels/encode.cu
            "halut_encode",
        )
        return halut_encode_kernel


def create_read_acc_lut_kernel(
    C: int = 16, K: int = 16, blocks: int = 8, rows: int = 128
) -> cp.RawKernel:
    script_folder = Path(os.path.dirname(os.path.abspath(__file__)))
    file_to_open = script_folder / "kernels/read_acc_lut.cu"
    with open(file_to_open, "r") as f:
        halut_read_acc_lut_code = f.read()

        halut_read_acc_lut_code = re.sub(
            r"const int K = \d+;\n",
            "const int K = " + str(K) + ";\n",
            halut_read_acc_lut_code,
            flags=re.MULTILINE,
        )
        halut_read_acc_lut_code = re.sub(
            r"const int C = \d+;\n",
            "const int C = " + str(C) + ";\n",
            halut_read_acc_lut_code,
            flags=re.MULTILINE,
        )
        halut_read_acc_lut_code = re.sub(
            r"const int blocks = \d+;\n",
            "const int blocks = " + str(blocks) + ";\n",
            halut_read_acc_lut_code,
            flags=re.MULTILINE,
        )
        halut_read_acc_lut_code = re.sub(
            r"const int rows = \d+;\n",
            "const int rows = " + str(rows) + ";\n",
            halut_read_acc_lut_code,
            flags=re.MULTILINE,
        )
        # print(halut_read_acc_lut_code)
        halut_read_acc_lut_kernel = cp.RawKernel(
            halut_read_acc_lut_code,  # defined in file kernels/read_acc_lut.cu
            "halut_read_acc_lut",
        )
        return halut_read_acc_lut_kernel
