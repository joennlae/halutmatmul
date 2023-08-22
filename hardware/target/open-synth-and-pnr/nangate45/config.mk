export PLATFORM    = nangate45

export VERILOG_FILES = $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
export SDC_FILE      = $(HALUT_ROOT)/target/open-synth-and-pnr/nangate45/constraint.sdc
export SYNTH_HIERARCHICAL = 1

export CORE_UTILIZATION       	= 40
export CORE_ASPECT_RATIO      	= 1
export CORE_MARGIN            	= 2
export PLACE_DENSITY_LB_ADDON   = 0.40

export TNS_END_PERCENT        = 100