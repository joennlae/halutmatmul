# Halutmatmul

Based on [MADDness/Bolt](https://github.com/dblalock/bolt).

More information about the base project is [here](maddness/README.md)

## Install

```bash
# install conda environment & activate
conda env create -f environment.yml
conda activate halutmatmul

# install CLI
./scripts/install-cli.sh

# now use CLI with
halut --help

# or without install
./halut --help
```
### Links

* [Github - Runner - Env](https://github.com/actions/virtual-environments/blob/main/images/linux/Ubuntu2004-Readme.md)


## PyTorch Install

* [Link](https://pytorch.org/get-started/locally/)

Currently we need the nightly build as we need this pull request [##5618](https://github.com/pytorch/vision/pull/5618)
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly # please change back to non nightly when pull is published
```

### CIFAR-10 pretrained model from 

* [CIFAR-10 pretrained](https://github.com/huyvnphan/PyTorch_CIFAR10)