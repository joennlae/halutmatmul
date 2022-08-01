export PLATFORM               = asap7

export DESIGN_NAME            = register_file_latch

export VERILOG_FILES = $(shell cat "files.txt")
export SDC_FILE      = ./constraint.sdc

export PLACE_DENSITY          = 0.60

export DIE_AREA               = 0 0 30 30
export CORE_AREA              = 2 2 28 28

export HAS_IO_CONSTRAINTS = 1 # fix error [ERROR GPL-0305] RePlAce diverged at newStepLength.

export CORNER									= TC
export DFF_LIB_FILE           = $($(CORNER)_DFF_LIB_FILE)