#!/bin/bash

CONDA_ENV_PATH=/scratch/janniss/conda/halutmatmul_hw
patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64:/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/zachjs-sv2v

patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/verible-verilog-format
patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/verible-verilog-syntax
patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/verible-verilog-lint