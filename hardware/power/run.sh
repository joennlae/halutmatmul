# Print commands during execution

set -x

# Build directories

rm -rf ./logs
rm -rf ./reports
rm -rf ./outputs

mkdir -p logs
mkdir -p reports
mkdir -p outputs

./pt_shell -f primetime_power.tcl -output_log_file logs/pt.log || exit 1
