set clk_name  clk_all
set clk_port_name clk_i
# attention NanGate45 is in ns instead of ps
set clk_period 6.0
set clk_io_pct 0.1

set clk_port [get_ports $clk_port_name]

create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]

set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]

# Driving cells and loads.
set driving_cell     BUF_X4
set driving_cell_clk BUF_X4
set load_cell        BUF_X8

# load of 4x 6.585178fF for BUF_X8
set_load -pin_load [expr 4* 6.585178 ] [all_outputs]
set_driving_cell [all_inputs]      -lib_cell $driving_cell     -pin Z
set_driving_cell [get_ports clk_i] -lib_cell $driving_cell_clk -pin Z

# False path the reset signal
set_false_path -from rst_ni
