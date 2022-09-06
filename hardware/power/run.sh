#!/bin/bash
# Print commands during execution
set -x
# Build directories

rm -rf ./logs
rm -rf ./reports
rm -rf ./outputs
rm -rf ./inputs

mkdir -p logs
mkdir -p reports
mkdir -p outputs
mkdir -p inputs/adk

export BASE_PATH=$PWD
# techs: asap7, nangate45, gf22_com
export TECH=gf22_com
export DESIGN_NAME=halut_matmul

export START_TIME_NS=0.0

if [[ "$TECH" == "asap7" ]]; then
  export DESIGN_PATH=/scratch2/janniss/fusesoc/asap7_halut_matmaul_8_4_3000_cts_normal/openroad_asap7-openroad
  export SIM_PATH=/scratch2/janniss/fusesoc/asap_8_4_3000/icarus-cocotb
  export LIB_NAME=merged.lib
elif [[ "$TECH" == "nangate45" ]]; then
  export DESIGN_PATH=/scratch2/janniss/fusesoc/nangate45_8_4_6000/openroad_nangate45-openroad
  export SIM_PATH=/scratch2/janniss/fusesoc/nangate_8_4_6000/icarus-cocotb
  export LIB_NAME=NangateOpenCellLibrary_typical.lib
elif [[ "$TECH" == "gf22_com" || "$TECH" == "nangate45_com" ]]; then
  export MFLOWGEN_PATH=/scratch2/janniss/mflowgen_build/mflowgen-iis
  export BUILD_FOLDER=build_8_4_2500
  export SIM_PATH=/scratch2/janniss/fusesoc/22_8_4_2500/questa-cocotb
  export START_TIME_NS=5462.5
else
  echo "TECH=$TECH not supported" && exit 1
fi

if [[ "$TECH" == "asap7" || "$TECH" == "nangate45" ]]; then
  # symlink inputs
  cd $BASE_PATH/inputs/adk
  ln -s $DESIGN_PATH/objects/$TECH/$DESIGN_NAME/base/lib/$LIB_NAME stdcells.lib

  cd $BASE_PATH/inputs
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.v design.v
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.spef design.spef
  ln -s $DESIGN_PATH/results/$TECH/$DESIGN_NAME/base/6_final.sdc design.sdc

  ln -s $SIM_PATH/dump.vcd run.vcd

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
  ln -s $SIM_PATH/run.vcd run.vcd
fi

cd $BASE_PATH
export SYNOPSYS_LC_ROOT=/usr/pack/synopsys-2022.03-kgf/lc/bin/lc_shell
# python util/markDontUse.py -p "SDFL* SDFH* ICG* " -i inputs/adk/stdcells.lib -o inputs/adk/stripped.lib

# rm inputs/adk/stdcells.lib
./pt_shell -f primetime_power.tcl -output_log_file logs/pt.log || exit 1
