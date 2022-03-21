#!/bin/bash

CXX_COMPILER=clang++-12 #clang++-12.0.1, g++-9.2.0
C_COMPILER=clang-12 #clang-12.0.1, gcc-9.2.0
C_LANG_TIDY=clang-tidy-12
IS_CLANG_TIDY=0
UNIT=1
CMAKE=cmake

DIR="/usr/sepp"
if [ -d "$DIR" ]; then
  # we are in IIS environemnt #
  echo "Load IIS env file"
  source halut.env
  # for IIS computing
  CMAKE=cmake-3.18.1
  CXX_COMPILER=clang++-12.0.1
  C_COMPILER=clang-12.0.1
  CLANG_TIDY=clang-tidy-12.0.1
fi

for i in "$@"; do
  case $i in
    -c|--clang-tidy)
      IS_CLANG_TIDY=1
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

$CMAKE ../ \
  -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
  -DCMAKE_C_COMPILER=$C_COMPILER \
  -DORIGIN=$ORIGIN \
  -DCLANG_TIDY=$IS_CLANG_TIDY \
  -DCMAKE_CLANG_TIDY=$CLANG_TIDY \
  -DUNIT_TESTS=$UNIT
make -j16
EXIT_CODE=$? # important for CI

cd $ORIGIN
exit $EXIT_CODE # important for CI