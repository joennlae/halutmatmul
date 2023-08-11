HALUT_ROOT ?= $(shell pwd)

$(HALUT_ROOT)/.bender/.done:
	bender checkout
	touch $@

halut-deps: $(HALUT_ROOT)/.bender/.done

include $(HALUT_ROOT)/target/open-frontend/pickle/pickle.mk
include $(HALUT_ROOT)/target/dv/tests.mk