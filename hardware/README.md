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

### VSCode setup

One extensions are recommended:

* [WaveTrace](https://marketplace.visualstudio.com/items?itemName=wavetrace.wavetrace)


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


### Commercial-14nm

Only works with IIS rights (ask me for git access). But the NDA needs to be signed first.

```bash
cd target
git clone git@iis-git.ee.ethz.ch:janniss/maddness-gf12.git commercial-14nm
```

And read the README inside the repo for more information.