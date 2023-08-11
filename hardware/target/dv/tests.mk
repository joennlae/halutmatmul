

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
	fp-16-to-32-convert

targets = $(addprefix test-, $(TESTS))

.PHONY: $(targets)

$(targets): test-%: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
	make -C $(HALUT_ROOT)/target/dv/$*/ SIM=icarus HALUT_ROOT=$(HALUT_ROOT) PYTHONPATH=$(HALUT_ROOT)/target/dv/

test-all: $(targets)