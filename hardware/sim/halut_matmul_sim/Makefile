# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# Makefile

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(shell cat "files.txt")

ifeq ($(SIM),questa)
COMPILE_ARGS += "+nospecify"
# WAVES=1 would generate wlf to directly generate vcd add to code!! 
else ifeq ($(SIM),icarus)
COMPILE_ARGS += -gno-specify -gno-xtypes -gno-icarus-misc
endif


TOPLEVEL = halut_matmul

# MODULE is the basename of the Python test file
MODULE = sim_halut_matmul

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim