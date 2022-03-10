#!/bin/bash

CXX_COMPILER=clang++-12.0.1 #clang++-12.0.1, g++-9.2.0
C_COMPILER=clang-12.0.1 #clang-12.0.1, gcc-9.2.0
CLANG_TIDY=0
UNIT=1

for i in "$@"; do
  case $i in
    -c|--clang-tidy)
      # CXX_COMPILER=clang-13
      # C_COMPILER=clang-13
      CLANG_TIDY=1
      shift
      ;;
    -u|--unit)
      UNIT=1
      shift
      ;;
    *)
      ;;
  esac
done
ORIGIN=$PWD

rm -rf build
mkdir -p build/
cd build/

cmake-3.7.1  ../ \
  -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
  -DCMAKE_C_COMPILER=$C_COMPILER \
  -DORIGIN=$ORIGIN \
  -DCLANG_TIDY=$CLANG_TIDY \
  -DUNIT_TESTS=$UNIT
make -j16
EXIT_CODE=$? # important for CI

cd $ORIGIN
exit $EXIT_CODE # important for CI