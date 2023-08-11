# Copyright (c) 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# heavily inspired by https://github.com/pulp-platform/iguana/blob/main/target/ihp13/pickle/pickle.mk
# Authors:
# - Jannis Schönleber  <janniss@iis.ee.ethz.ch>

# First step of preprocessing, put all used RTL into one large file

BENDER ?= bender # https://github.com/pulp-platform/bender
MORTY  ?= morty  # https://github.com/pulp-platform/morty
SVASE  ?= svase  # https://github.com/pulp-platform/svase
SV2V   ?= sv2v   # https://https://github.com/zachjs/sv2v

PICKLE_DIR ?= $(HALUT_ROOT)/target/open-frontend/pickle


#########
# Morty #
#########

# Generate sources manifest for use by Morty
$(PICKLE_DIR)/out/halut_matmul.sources.json: halut-deps $(HALUT_ROOT)/Bender.yml
	mkdir -p $(dir $@)
	$(BENDER) sources -f > $@

# Pickle all synthesizable RTL into a single file
$(PICKLE_DIR)/out/halut_matmul.morty.sv: $(PICKLE_DIR)/out/halut_matmul.sources.json
	$(MORTY) -q -f $< -o $@ -D SYNTHESIS=1 -D MORTY=1 -D COCOTB_SIM=1 --keep_defines --top halut_matmul

halut-morty-all: $(PICKLE_DIR)/out/halut_matmul.morty.sv

#########
# SVase #
#########

# svase not needed at the moment
# Pre-elaborate SystemVerilog pickle
# $(PICKLE_DIR)/out/halut_matmul.svase.sv: $(PICKLE_DIR)/out/halut_matmul.morty.sv
# 	$(SVASE) halut_matmul $@ $<
# 
# halut-svase-all: $(PICKLE_DIR)/out/halut_matmul.svase.sv

########
# SV2V #
########

# Convert pickle to Verilog
$(PICKLE_DIR)/out/halut_matmul.sv2v.v: $(PICKLE_DIR)/out/halut_matmul.morty.sv
	$(SV2V) --oversized-numbers --verbose --write $@ $<

halut-pickle-all: $(PICKLE_DIR)/out/halut_matmul.sv2v.v