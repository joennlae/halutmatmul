HALUT_ROOT ?= $(shell pwd)

$(HALUT_ROOT)/.bender/.done:
	bender checkout
	touch $@

halut-deps: $(HALUT_ROOT)/.bender/.done

include $(HALUT_ROOT)/target/open-frontend/pickle/pickle.mk
include $(HALUT_ROOT)/target/dv/tests.mk

halut-slang-sv: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv
	slang --top halut_matmul $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv

halut-slang-verilog: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
	slang --top halut_matmul $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v

halut-lint-verilator: halut-deps
	verilator --lint-only --top halut_matmul -Wall --timing $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv
