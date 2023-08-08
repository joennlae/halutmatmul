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

## OpenROAD flow

* [The-OpenROAD-Project/OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
* [Web Documentation](https://openroad.readthedocs.io/en/latest/)
* [PDF Documentation](https://openroad.readthedocs.io/_/downloads/en/latest/pdf/)

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
git clone https://github.com/joennlae/gcc-toolchain-builder.git
cd gcc-toolchain-builder
# default path ${HOME}/.local
./build-native-toolchain.sh
```

### Install GLIBC

```bash
# be sure to deactivate conda env
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

* [Verilog-HDL/SystemVerilog/Bluespec SystemVerilog](https://marketplace.visualstudio.com/items?itemName=mshr-h.VerilogHDL)
* [SystemVerilog and Verilog Formatter](https://marketplace.visualstudio.com/items?itemName=bmpenuelas.systemverilog-formatter-vscode)
* [WaveTrace](https://marketplace.visualstudio.com/items?itemName=wavetrace.wavetrace)

Be sure to update the `.vscode` file with your own verilator path:

`/scratch2/janniss/conda/halutmatmul_hw/bin/verilator` -> `verilator` or your custom path
```json
{
  // the default with verilator 
  "verilog.linting.linter": "verilator",
  "verilog.linting.path": "/scratch2/janniss/conda/halutmatmul_hw/bin/",
  "verilog.linting.verilator.runAtFileLocation": false,
  "verilog.linting.verilator.arguments": "--language 1800-2012 --Wall -Ihardware/vendor/lowrisc_ip/ip/prim/rtl -Ihardware/build/halut_ip_halut_top_0.1/src/lowrisc_prim_abstract_and2_0 -Ihardware/build/halut_ip_halut_top_0.1/src/lowrisc_prim_generic_and2_0/rtl/ hardware/build/halut_ip_halut_top_0.1/src/lowrisc_prim_abstract_prim_pkg_0.1/prim_pkg.sv hardware/lint/verilator/verilator_waiver.vlt hardware/rtl/fp_defs_pkg.sv hardware/rtl/halut_pkg.sv",
}
```


## Vendored IP

Use the script `util/vendor.py` to update or added vendored IP.

* [vendor.py Docs](https://docs.opentitan.org/doc/rm/vendor_in_tool/index.html)

**Important** use it from the base git directory otherwise patches will be skipped silently.

```bash
python hardware/util/vendor.py hardware/vendor/lowrisc_ip.vendor.hjson -v
```

#### IIS tricks

```bash
mkdir -p /scratch2/janniss/halut_flow/OpenROAD
# use vendor.py to OpenROAD (normally would use submodule but IIS storage limits)
# symlink option is only needed when not enough space is available
python hardware/util/vendor.py hardware/flow/openroad.vendor.hjson -S /scratch2/janniss/halut_flow/OpenROAD -v
# if using symlinks manually apply patches from flow/patches/OpenROAD
```

## License

### lowRISC IP

* [Apache-2.0 license](https://github.com/lowRISC/opentitan)

### OpenROAD

* [BSD 3-Clause License](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts#license)

But tools sometimes have different licenses.

The OpenROAD-flow-scripts repository (build and run scripts) has a BSD 3-Clause License.
The flow relies on several tools, platforms and designs that each have their own licenses:

- Find the tool license at: `OpenROAD-flow-scripts/tools/{tool}/` or `OpenROAD-flow-scripts/tools/OpenROAD/src/{tool}/`.
- Find the platform license at: `OpenROAD-flow-scripts/flow/platforms/{platform}/`.
- Find the design license at: `OpenROAD-flow-scripts/flow/designs/src/{design}/`.