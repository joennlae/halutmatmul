# Hardware

## Folder Structure

```yaml
lint/ # linting files
rtl/ # all custom rtl files
vendor/ # ip files
scripts/
util/ 
internal/
  # everything that should not reach the public
  # internal git repo
```

## Setup Environment

```bash
# Conda environment & activate
conda env create -f environment_hw.yml
conda activate halutmatmul_hw

# IIS prefixed env
conda env create -f environment_hw.yml --prefix /scratch/janniss/conda/halutmatmul_hw
```

### If you have a too old GLIBC version

This is a problem with our old CentOS versions mainly inside IIS computing infrastructure. We will build our own toolchain, update make + install a new version of GLIBC and then patch with patchelf.


#### Build GCC toolchain if needed
```bash
# build toolchain if gcc version too old
https://github.com/joennlae/gcc-toolchain-builder.git
cd gcc-toolchain-builder
# default path ${HOME}/.local
./build-native-toolchain.sh
```

### Install GLIBC

```bash
./scripts/install_glibc.sh
```

### Patchelf conda installed binaries

```bash
patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64:/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/zachjs-sv2v

patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/verible-verilog-format

patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/verible-verilog-lint

patchelf --set-rpath ~/.local/custom_glibc/lib:~/.local/lib64 --set-interpreter ~/.local/custom_glibc/lib/ld-linux-x86-64.so.2 /scratch/janniss/conda/halutmatmul_hw/bin/verible-verilog-syntax
```

### vscode setup

Two extensions are recommended:

* [SystemVerilog - Language Support](https://marketplace.visualstudio.com/items?itemName=eirikpre.systemverilog)
* [SystemVerilog and Verilog Formatter](https://marketplace.visualstudio.com/items?itemName=bmpenuelas.systemverilog-formatter-vscode)




