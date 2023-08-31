# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# Makefile

# defaults
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES = $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v

# TOPLEVEL is the name of the toplevel module in your Verilog or VHDL file
TOPLEVEL_SHORT = mixed_int_adder
TOPLEVEL= $$(sed -n 's|module \($(TOPLEVEL_SHORT)__[[:alnum:]_]*\)\s.*$$|\1|p' $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v 2> /dev/null | tail -1)

# MODULE is the basename of the Python test file
MODULE = test_mixed_int_adder

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim