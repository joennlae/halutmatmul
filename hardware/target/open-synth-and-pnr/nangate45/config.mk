# export DESIGN_NAME  = register_file_latch # defined via edalize
export PLATFORM    = nangate45

export VERILOG_FILES = $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
export SDC_FILE      = $(HALUT_ROOT)/target/open-synth-and-pnr/nangate45/constraint.sdc
export ABC_AREA      = 1

# export PLACE_DENSITY          = 0.60
export SYNTH_HIERARCHICAL = 1
export MAX_UNGROUP_SIZE = 100
# Adders degrade GCD
export ADDER_MAP_FILE :=

# export HAS_IO_CONSTRAINTS = 1 # fix error [ERROR GPL-0305] RePlAce diverged at newStepLength.

# export DIE_AREA    = 0 0 2000 2000
# export CORE_AREA   = 10 10 1995 1995


export CTS_BUF_CELL           = BUF_X32 BUF_X16 # BUF_X8 BUF_X4

export ABC_AREA               = 1
# export RESYNTH_AREA_RECOVER		= 1
# export RESYNTH_TIMING_RECOVER = 1

export CORE_UTILIZATION       = 40
export CORE_ASPECT_RATIO      = 1
export CORE_MARGIN            = 2
export PLACE_DENSITY          = 0.60
