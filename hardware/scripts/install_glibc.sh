
INSTALL_PATH=$HOME/.local
BASE_PATH=/var/tmp/$(whoami)/glibc_install

ORIGIN=`pwd`

BUILD_PATH=$BASE_PATH/glibc
mkdir -p ${BUILD_PATH}

cd $BUILD_PATH
wget https://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
wget https://ftp.gnu.org/gnu/make/make-4.2.tar.gz


tar -xvf make-4.2.tar.gz
cd make-4.2
mkdir build
cd ./build
../configure --prefix=$INSTALL_PATH --disable-werror
make -j `nproc` 
make install

ln -s $HOME/.local/bin/make $HOME/.local/bin/gmake

export PATH=$HOME/.local/bin/:$PATH
make --version
gmake --version

unset LD_LIBRARY_PATH

cd $BUILD_PATH
tar -xvf glibc-2.29.tar.gz
cd glibc-2.29
mkdir build
cd ./build
../configure --prefix=$INSTALL_PATH/custom_glibc --disable-werror
make -j `nproc` 
make install
