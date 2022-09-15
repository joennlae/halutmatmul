export PLATFORM               = asap7

# export DESIGN_NAME            = register_file_latch # defined via edalize

export VERILOG_FILES = $(shell cat "files.txt")
export SDC_FILE      = ./constraint.sdc

# export PLACE_DENSITY          = 0.60
# export DIE_AREA               = 0 0 800 800
# export CORE_AREA              = 2 2 798 798

export SYNTH_HIERARCHICAL = 1
export MAX_UNGROUP_SIZE = 100
# Remove yosys adders
export ADDER_MAP_FILE :=

# export DONT_USE_CELLS          += SDF* ICG* DFFH*
# export DONT_USE_CELLS        = ASYNC_DFFH*

# export CTS_BUF_CELL           = BUFx12_ASAP7_75t_R BUFx10_ASAP7_75t_R BUFx8_ASAP7_75t_R BUFx4_ASAP7_75t_R BUFx2_ASAP7_75t_R
# export CTS_BUF_CELL           = BUFx12_ASAP7_75t_L BUFx10_ASAP7_75t_L BUFx8_ASAP7_75t_L BUFx4_ASAP7_75t_L BUFx2_ASAP7_75t_L
# export CTS_CLUSTER_DIAMETER = 15
# export CTS_CLUSTER_SIZE = 80
# export CTS_BUF_DISTANCE        = 60

export ABC_AREA               = 1
# export RESYNTH_AREA_RECOVER		= 1
# export RESYNTH_TIMING_RECOVER = 1

export CORE_UTILIZATION       = 40
export CORE_ASPECT_RATIO      = 1
export CORE_MARGIN            = 2
export PLACE_DENSITY          ?= 0.60

# export HAS_IO_CONSTRAINTS = 1 # fix error [ERROR GPL-0305] RePlAce diverged at newStepLength.
# export ASAP7_USESLVT					= 1
# export ASAP7_USELVT						= 1
export CORNER									= TC
export DFF_LIB_FILE           = $($(CORNER)_DFF_LIB_FILE)