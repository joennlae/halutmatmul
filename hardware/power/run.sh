#!/bin/bash
# Print commands during execution
set -x
# Build directories

rm -rf ./logs
rm -rf ./reports
rm -rf ./outputs
rm -rf ./inputs

rm -rf /scratch2/janniss/pt/reports
mkdir -p /scratch2/janniss/pt/reports

mkdir -p logs
# mkdir -p reports
ln -s /scratch2/janniss/pt/reports ./reports
mkdir -p outputs
mkdir -p inputs/adk

export BASE_PATH=$PWD
# techs: asap7, nangate45, gf22_com
export TECH=gf22_com
export DESIGN_NAME=halut_matmul

export START_TIME_NS=0.0
export END_TIME_NS=-1

if [[ "$TECH" == "asap7" ]]; then
  export DESIGN_PATH=/scratch2/janniss/fusesoc/7_4_2_2000/openroad_asap7-openroad
  export SIM_PATH=/scratch2/janniss/fusesoc/7_4_2_2000/questa-cocotb
  export LIB_NAME=asap7_merged
  export START_TIME_NS=2322.00
  echo "ASAP7"
elif [[ "$TECH" == "nangate45" ]]; then
  export DESIGN_PATH=/scratch2/janniss/fusesoc/nangate45_8_4_6000/openroad_nangate45-openroad
  export SIM_PATH=/scratch2/janniss/fusesoc/nangate_8_4_6000/icarus-cocotb
  export LIB_NAME=NangateOpenCellLibrary_typical.lib
elif [[ "$TECH" == "gf22_com" || "$TECH" == "nangate45_com" ]]; then
  export MFLOWGEN_PATH=/scratch2/janniss/mflowgen_build/mflowgen-iis
  export BUILD_FOLDER=build_8_4_1750
  export SIM_PATH=/scratch2/janniss/fusesoc/22_32_16_1750/questa-cocotb
  export START_TIME_NS=15342.02
else
  echo "TECH=$TECH not supported" && exit 1
fi

if [[ "$TECH" == "asap7" ]]; then
  # symlink inputs
  cd $BASE_PATH/inputs/adk
  # ln -s $DESIGN_PATH/objects/$TECH/$DESIGN_NAME/base/lib/$LIB_NAME stdcells.lib
  ln -s ../../stdcells_7.db stdcells.db

  cd $BASE_PATH/inputs
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.v design.v
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.spef design.spef
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.sdc design.sdc

  ln -s $SIM_PATH/dump.vcd run.vcd

  cd $BASE_PATH
  # ./lc_shell -f convert_lib.tcl
elif [[ "$TECH" == "nangate45" ]]; then
  # symlink inputs
  cd $BASE_PATH/inputs/adk
  ln -s $DESIGN_PATH/objects/$TECH/$DESIGN_NAME/base/lib/$LIB_NAME stdcells.lib

  cd $BASE_PATH/inputs
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.v design.v
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.spef design.spef
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.sdc design.sdc

  ln -s $SIM_PATH/dump.vcd run.vcd

  cd $BASE_PATH
  ./lc_shell -f convert_lib.tcl
else # gf22
  # mflowgen path
  cd $BASE_PATH/inputs/adk
  # replace with freepdk!!
  ln -s $MFLOWGEN_PATH/adks/gf22/view-standard/stdcells.db stdcells.db
  ln -s $MFLOWGEN_PATH/adks/gf22/view-standard/iocells.db iocells.db
  ln -s $MFLOWGEN_PATH/adks/gf22/view-standard/iocells.lib iocells.lib
  # ln -s $MFLOWGEN_PATH/$BUILD_FOLDER/8-synopsys-ptpx-genlibdb/outputs/design.db stdcells.db

  cd $BASE_PATH/inputs
  # build_8_4/6-cadence-innovus-place-route/outputs/design.vcs.v
  ln -s $MFLOWGEN_PATH/$BUILD_FOLDER/6-cadence-innovus-place-route/outputs/design.vcs.v design.v
  ln -s $MFLOWGEN_PATH/$BUILD_FOLDER/6-cadence-innovus-place-route/outputs/design.spef.gz design.spef.gz
  ln -s $MFLOWGEN_PATH/$BUILD_FOLDER/6-cadence-innovus-place-route/outputs/design.pt.sdc design.sdc

  # if questa is used --> wlf2vcd vsim.wlf > run.vcd
  ln -s $SIM_PATH/dump.vcd run.vcd
fi

cd $BASE_PATH
export SYNOPSYS_LC_ROOT=/usr/pack/synopsys-2022.03-kgf/lc/bin/lc_shell
# python util/markDontUse.py -p "SDFL* SDFH* ICG* " -i inputs/adk/stdcells.lib -o inputs/adk/stripped.lib

# rm inputs/adk/stdcells.lib
./pt_shell -f primetime_power.tcl -output_log_file logs/pt.log || exit 1
