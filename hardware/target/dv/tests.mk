

TESTS = \
	scm \
	halut-matmul \
	halut-encoder-4 \
	halut-encoder \
	halut-decoder-x \
	halut-decoder \
	fp-adder \
	fp-16-32-adder \
	fp-16-comparision \
	fp-16-to-32-convert \
	mixed-int-adder

test-targets = $(addprefix test-, $(TESTS))
test-targets-questa = $(addprefix test-questa-, $(TESTS))

.PHONY: $(test-targets)

$(test-targets): test-%: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
	make -C $(HALUT_ROOT)/target/dv/$*/ SIM=icarus HALUT_ROOT=$(HALUT_ROOT) \
    PYTHONPATH=$(HALUT_ROOT)/target/dv/ ACC_TYPE=$(ACC_TYPE) DATA_WIDTH=$(DATA_WIDTH) NUM_C=$(NUM_C) \
		NUM_M=$(NUM_M) NUM_DECODER_UNITS=$(NUM_DECODER_UNITS)
	grep -q "failure" $(HALUT_ROOT)/target/dv/$*/results.xml && exit 1 || exit 0

test-all: $(test-targets)

$(test-targets-questa): test-questa-%: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
	make -C $(HALUT_ROOT)/target/dv/$*/ SIM=questa HALUT_ROOT=$(HALUT_ROOT) \
		PYTHONPATH=$(HALUT_ROOT)/target/dv/ ACC_TYPE=$(ACC_TYPE) DATA_WIDTH=$(DATA_WIDTH) NUM_C=$(NUM_C) \
		NUM_M=$(NUM_M) NUM_DECODER_UNITS=$(NUM_DECODER_UNITS) GUI=1
	grep -q "failure" $(HALUT_ROOT)/target/dv/$*/results.xml && exit 1 || exit 0