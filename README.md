# Halutmatmul

[![GPU Tests (Vast.ai)](https://github.com/joennlae/halutmatmul/actions/workflows/gpu_tests.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/gpu_tests.yaml)
[![PyTest](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml)
[![Linting](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml)
[![MyPy](https://github.com/joennlae/halutmatmul/actions/workflows/python_mypy.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_mypy.yaml)
[![C++ build](https://github.com/joennlae/halutmatmul/actions/workflows/cpp_testing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/cpp_testing.yaml)

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

## Results

### Single Layer replacement with `C=32` and `K=16`

### LeViT ([Source](https://github.com/facebookresearch/LeViT))

SOTA Vision Transformer on ImageNet 1K
![LeViT Results](https://github.com/joennlae/halutdata/raw/master/figures/levit.png)

### ResNet-50 (only interesting layers in analysis)
Legacy Classifier on ImageNet 1K
![ResNet-50 Results](https://github.com/joennlae/halutdata/raw/master/figures/resnet-50.png)

### Depthwise seperable CNN
on Google Speech v2
![DS-CNN Results](https://github.com/joennlae/halutdata/raw/master/figures/dscnn.png)


### Offline learning convergence on ResNet-50
![ResNet-50 Convergence Results](https://github.com/joennlae/halutdata/raw/master/figures/all_layers.png)

## Formalism

Some definitions about the forward path.

### Encode kernel
![](docs/images/encode_kernel.png)
### Read and accumulate LUTs kernel
![](docs/images/read_acc_lut_kernel.png)
### Links

* [Addendum](docs/addendum.md)