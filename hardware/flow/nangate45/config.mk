export DESIGN_NAME = sram
export PLATFORM    = nangate45

export VERILOG_FILES = ./designs/src/$(DESIGN_NAME)/sram.v
export SDC_FILE      = ./constraint.sdc
export ABC_AREA      = 1

# Adders degrade GCD
export ADDER_MAP_FILE :=

# These values must be multiples of placement site
# x=0.19 y=1.4
export DIE_AREA    = 0 0 2000 2000
export CORE_AREA   = 10 10 1950 1950

