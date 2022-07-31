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

### VSCode setup

Two extensions are recommended:

* [SystemVerilog - Language Support](https://marketplace.visualstudio.com/items?itemName=eirikpre.systemverilog)
* [SystemVerilog and Verilog Formatter](https://marketplace.visualstudio.com/items?itemName=bmpenuelas.systemverilog-formatter-vscode)


## Vendored IP

Use the script `util/vendor.py` to update or added vendored IP.

* [vendor.py Docs](https://docs.opentitan.org/doc/rm/vendor_in_tool/index.html)

#### IIS tricks

Install `GitBSLR` an `LD_Preload` based lib that will make git follow symlinks.

* [StackOverflow](https://superuser.com/a/1318025/1685000)
```bash
git clone https://github.com/Alcaro/GitBSLR.git
cd GitBSLR
./install.sh
# add to .bashrc or .zshrc
alias git="LD_PRELOAD=/path/to/gitbslr.so git"
```
Symlinks for `OpenROAD` flow:
```bash
# set
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1
# and symlink
ln -s /scratch2/janniss/halut_flow flow
# use vendor.py to OpenROAD (normally would use submodule but IIS storage limits)
# symlink option is only needed when not enough space is available
python hardware/util/vendor.py hardware/flow/openroad.vendor.hjson -S /scratch2/janniss/halut_flow/OpenROAD
```