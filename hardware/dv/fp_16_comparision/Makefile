# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# Makefile

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(shell cat "files.txt")


ifeq ($(SIM),icarus)
COMPILE_ARGS += -I $(shell cat "incdirs.txt")
else ifeq ($(SIM),verilator)
COMPILE_ARGS += -I$(shell cat "incdirs.txt") # TODO fix when multiple include dirs
# https://docs.cocotb.org/en/stable/simulator_support.html#sim-verilator-waveforms
EXTRA_ARGS += --trace --trace-structs
endif
# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = fp_16_comparision

# MODULE is the basename of the Python test file
MODULE = test_fp_16_comparision

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim