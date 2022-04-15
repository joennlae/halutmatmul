# Halutmatmul

* Based on [MADDness/Bolt](https://github.com/dblalock/bolt).
* More information about the base project is [here](maddness/README.md)
* [arXiv](https://arxiv.org/abs/2106.10860) paper link

## Install

```bash
# install conda environment & activate
conda env create -f environment_gpu.yml
conda activate halutmatmul

# install CLI
./scripts/install-cli.sh

# now use CLI with
halut --help

# or without install
./halut --help
```

## Formalism

Some definitions about the forward path.

### Encode kernel
![](docs/images/encode_kernel.png)
### Read and accumulate LUTs kernel
![](docs/images/read_acc_lut_kernel.png)
### Links

* [Addendum](docs/addendum.md)