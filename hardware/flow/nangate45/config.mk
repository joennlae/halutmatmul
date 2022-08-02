# export DESIGN_NAME  = register_file_latch # defined via edalize
export PLATFORM    = nangate45

export VERILOG_FILES = $(shell cat "files.txt")
export SDC_FILE      = ./constraint.sdc
export ABC_AREA      = 1

# Adders degrade GCD
export ADDER_MAP_FILE :=

export DIE_AREA    = 0 0 200 200
export CORE_AREA   = 10 10 195 195
