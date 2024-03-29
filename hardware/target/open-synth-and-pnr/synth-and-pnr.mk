

PNR_DIR ?= $(HALUT_ROOT)/target/open-synth-and-pnr

FLOW_HOME ?= $(PNR_DIR)/OpenROAD/flow

TOPLEVELS = \
	halut_matmul \
	halut_encoder_4 \
	halut_decoder \
	fp_16_32_adder

hw-targets = $(addprefix halut-open-synth-and-pnr-, $(TOPLEVELS))

.PHONY: $(hw-targets)

UNIQUE_TOP = $$( \
			if [ "$*" = "halut_matmul" ]; then \
				echo "halut_matmul"; \
			else \
				sed -n 's|module \($*__[[:alnum:]_]*\)\s.*$$|\1|p' $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v 2> /dev/null | tail -1; \
			fi)

$(PNR_DIR)/out/%/.done: $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
	mkdir -p $(dir $@)
	echo $(UNIQUE_TOP)
	make -C $(FLOW_HOME) finish \
		HALUT_ROOT=$(HALUT_ROOT) \
		DESIGN_CONFIG=$(PNR_DIR)/nangate45/config.mk \
		FLOW_HOME=$(FLOW_HOME) \
		SHELL=/bin/bash \
		YOSYS_CMD=yosys \
		OPENROAD_EXE=openroad \
		DESIGN_NAME=$(UNIQUE_TOP)
	python $(FLOW_HOME)/util/genMetrics.py --design $(UNIQUE_TOP)
	python $(FLOW_HOME)/util/genReport.py -vvvv
	python $(FLOW_HOME)/util/genReportTable.py
	mkdir -p $(dir $@)/results
	mkdir -p $(dir $@)/reports
	cp $(FLOW_HOME)/metadata.json $(dir $@)/reports/metadata.json
	cp -r $(FLOW_HOME)/reports/nangate45/$(UNIQUE_TOP)/base/* $(dir $@)/reports
	cp $(FLOW_HOME)/reports/report-gallery-$(UNIQUE_TOP).html $(dir $@)/reports/report-gallery-$*.html
	# sed replace the path to the report-gallery.html
	sed -i 's/nangate45\/$*__.+\/base\///g' $(dir $@)/reports/report-gallery-$*.html
	cp $(FLOW_HOME)/results/nangate45/$(UNIQUE_TOP)/base/6_final.v $(dir $@)/results/6_final.v
	cp $(FLOW_HOME)/results/nangate45/$(UNIQUE_TOP)/base/6_final.gds $(dir $@)/results/6_final.gds
	touch $@

$(hw-targets): halut-open-synth-and-pnr-%: $(PNR_DIR)/out/%/.done