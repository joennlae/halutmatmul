HALUT_ROOT ?= $(shell pwd)

NUM_C ?= 32
NUM_M ?= 32
NUM_DECODER_UNITS ?= 16
DATA_WIDTH ?= 16
# FP32 or INT
ACC_TYPE ?= FP32

$(HALUT_ROOT)/.bender/.done:
	bender checkout
	touch $@

halut-deps: $(HALUT_ROOT)/.bender/.done

include $(HALUT_ROOT)/target/open-frontend/pickle/pickle.mk
include $(HALUT_ROOT)/target/dv/tests.mk
include $(HALUT_ROOT)/target/open-synth-and-pnr/synth-and-pnr.mk

halut-slang-sv: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv
	slang --top halut_matmul $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv

halut-slang-verilog: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
	slang --top halut_matmul $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v

halut-lint-verilator: halut-deps $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv
	verilator --lint-only --top halut_matmul -Wall --timing \
		$(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.morty.sv \
		-Wno-COMBDLY \
		-Wno-UNOPTFLAT \
		-Wno-UNUSEDSIGNAL \
		-Wno-DECLFILENAME \
		-Wno-UNUSEDPARAM
