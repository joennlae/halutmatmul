export PLATFORM    = nangate45

export VERILOG_FILES = $(HALUT_ROOT)/target/open-frontend/pickle/out/halut_matmul.sv2v.v
export SDC_FILE      = $(HALUT_ROOT)/target/open-synth-and-pnr/nangate45/constraint.sdc
export ABC_AREA      = 1

export SYNTH_HIERARCHICAL = 1
export MAX_UNGROUP_SIZE ?= 10000
export RTLMP_FLOW = True
export PLACE_DENSITY_LB_ADDON = 0.05

export SYNTH_HIERARCHICAL = 1

# export DIE_AREA    = 0 0 2000 2000
# export CORE_AREA   = 10 10 1995 1995

export ABC_AREA               = 1
# export RESYNTH_AREA_RECOVER		= 1
# export RESYNTH_TIMING_RECOVER = 1

export CORE_UTILIZATION       = 40
export CORE_ASPECT_RATIO      = 1
export CORE_MARGIN            = 2
# export PLACE_DENSITY          = 0.60

export TNS_END_PERCENT        = 100