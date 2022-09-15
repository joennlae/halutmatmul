# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# Makefile

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VSIM_ARGS=-t 1ps
WAVES = 1

ifeq ($(SIM),questa)
COMPILE_ARGS += "+nospecify"
else ifeq ($(SIM),icarus)
COMPILE_ARGS += -gno-specify -gno-xtypes -gno-icarus-misc
endif

VERILOG_SOURCES += $(shell cat "files.txt")
# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL = halut_encoder

# MODULE is the basename of the Python test file
MODULE = sim_halut_encoder

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim