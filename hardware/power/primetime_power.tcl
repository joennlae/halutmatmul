

set ptpx_design_name  "halut_matmul"
# $::env(design_name)

# The strip path must be defined!
#
#   export strip_path = th/dut
#
# There must _not_ be any quotes, or read_saif will fail. This fails:
#
#   export strip_path = "th/dut"
#

# set ptpx_strip_path         		$::env(saif_instance)

# set ptpx_analysis_mode				$::env(analysis_mode)
# set ptpx_zero_delay_simulation		$::env(zero_delay_simulation)
# set ptpx_op_condition				$::env(lib_op_condition)

#-------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------

# set adk_dir                       inputs/adk
#
# set ptpx_additional_search_path   $adk_dir
# set ptpx_target_libraries         stdcells.db

# set ptpx_extra_link_libraries     [join "
# [lsort [glob -nocomplain inputs/*.db]]
# [lsort [glob -nocomplain inputs/adk/*.db]]
# "]

#-------------------------------------------------------------------------
# Inputs
#-------------------------------------------------------------------------

set ptpx_gl_netlist         inputs/design.v
set ptpx_sdc                inputs/design.sdc
set ptpx_spef               inputs/design.spef
# set ptpx_saif               inputs/run.saif
set ptpx_vcd				inputs/run.vcd
# set ptpx_namemap			inputs/design.namemap

#-------------------------------------------------------------------------
# Directories
#-------------------------------------------------------------------------

set ptpx_reports_dir	   	reports
set ptpx_logs_dir	   		logs
set ptpx_outputs_dir		outputs

set adk_dir                       inputs/adk

set ptpx_additional_search_path   $adk_dir
set ptpx_target_libraries         stdcells.db

set ptpx_extra_link_libraries     [join "
[lsort [glob -nocomplain inputs/*.lib]]
[lsort [glob -nocomplain inputs/adk/*.lib]]
"]

set_app_var search_path      ". $ptpx_additional_search_path $search_path"
set_app_var target_library   stdcells.db
set_app_var link_library     [join "
*
$ptpx_target_libraries
$ptpx_extra_link_libraries
"]

# Set up power analysis

set_app_var power_enable_analysis true
set_app_var power_analysis_mode   "time_based"
set_app_var report_default_significant_digits 3

read_verilog   $ptpx_gl_netlist
current_design $ptpx_design_name

link_design
#> ${ptpx_logs_dir}/${ptpx_design_name}.link.rpt

# report_activity_file_check $ptpx_vcd
#-strip_path $ptpx_strip_path

echo "Read VCD activity annotation file from RTL simulation."
# -strip_path $ptpx_strip_path
# set time in ns
set start_value   [expr {$::env(START_TIME_NS)}]
set end_value     [expr {$::env(END_TIME_NS)}]
read_vcd -rtl $ptpx_vcd -time [list $start_value $end_value]

read_sdc -echo $ptpx_sdc

read_parasitics -format spef $ptpx_spef
report_annotated_parasitics -check

# power

check_power > $ptpx_reports_dir/$ptpx_design_name.power.check.rpt

# Set power analysis options

set_power_analysis_options 	-include all_without_leaf \
  -npeak 10 -peak_power_instances \
  -npeak_out $ptpx_reports_dir/$ptpx_design_name \
  -waveform_output $ptpx_reports_dir/$ptpx_design_name

# Apply activiy annotations and calculate power values

update_power > $ptpx_logs_dir/$ptpx_design_name.power.update.rpt

#-------------------------------------------------------------------------
# Power Reports
#-------------------------------------------------------------------------

# Report switching activity
report_switching_activity \
  > $ptpx_reports_dir/$ptpx_design_name.activity.post.rpt

# Group-based power report
report_power -nosplit -verbose \
  > $ptpx_reports_dir/$ptpx_design_name.power.rpt

# Cell hierarchy power report
report_power -nosplit -hierarchy -verbose \
  > $ptpx_reports_dir/$ptpx_design_name.power.hier.rpt

report_clock_gate_savings  > $ptpx_reports_dir/$ptpx_design_name.clock-gate-savings.rpt

# Get Leakage for each used library and count of cells

# foreach_in_collection l [get_libs] {
#   if {[get_attribute [get_lib $l] default_threshold_voltage_group] == ""} {
#     set libname [get_object_name [get_lib $l]]
#     set_user_attribute [get_lib $l] default_threshold_voltage_group $libname -class lib
#   }
# }
# report_power -threshold_voltage_group > $ptpx_reports_dir/$ptpx_design_name.power.leakage-per-lib.rpt
# report_threshold_voltage_group > $ptpx_reports_dir/$ptpx_design_name.power.cells-per-vth-group.rpt

exit
