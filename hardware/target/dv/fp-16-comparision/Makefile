# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# Makefile

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES = $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = fp_16_comparision

# MODULE is the basename of the Python test file
MODULE = test_fp_16_comparision

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim