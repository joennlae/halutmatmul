# Halutmatmul

### Algorithmic CI
[![GPU Tests (Vast.ai)](https://github.com/joennlae/halutmatmul/actions/workflows/gpu_tests.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/gpu_tests.yaml)
[![PyTest](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml)
[![Linting](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml)
[![MyPy](https://github.com/joennlae/halutmatmul/actions/workflows/python_mypy.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_mypy.yaml)
[![C++ build](https://github.com/joennlae/halutmatmul/actions/workflows/cpp_testing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/cpp_testing.yaml)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fjoennlae%2Fhalutmatmul.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fjoennlae%2Fhalutmatmul?ref=badge_shield)

### Hardware CI

[![HW Synth + PAR OpenROAD](https://github.com/joennlae/halutmatmul/actions/workflows/hw_openroad.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/hw_openroad.yaml)
[![RTL Linting](https://github.com/joennlae/halutmatmul/actions/workflows/hw_linting.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/hw_linting.yaml)
[![HW Design Verification](https://github.com/joennlae/halutmatmul/actions/workflows/hw_dv.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/hw_dv.yaml)


## General Information

* Based on [MADDness/Bolt](https://github.com/dblalock/bolt).
* More information about the base project is [here](maddness/README.md)
* [arXiv](https://arxiv.org/abs/2106.10860) paper link

This repo is used for the algorithmic exploration. I will try to update this repo with as much hardware information as I am allowed to publish.

## Install

```bash
# install conda environment & activate
conda env create -f environment_gpu.yml
conda activate halutmatmul

# IIS prefixed env
conda env create -f environment_gpu.yml --prefix /scratch/janniss/conda/halutmatmul_gpu

# install CLI
./scripts/install-cli.sh

# now use CLI with
halut --help

# or without install
./halut --help
```

## Hackernews mention (comments only) and discussion

* [HN: Bolt: Faster matrix and vector operations that run on compressed data](https://news.ycombinator.com/item?id=31792206)

## Hardware OpenROAD flow results

| All Designs    | ASAP7         | NanGate45      |
| -------------  | ------------- | -------------  |
| All Report     | [All](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/latest/asap7)  |  [All](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/latest/nangate45)  |
| History        | [History](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/history/asap7)  | [History](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/history/nangate45)  |

### Total Circuit (M=2)
| halut_matmul         | ASAP7         | NanGate45      |
| -------------  | ------------- | -------------  |
| Area [μm^2]    | 9643.6787  | 140647.7656 |
| Freq [Mhz]     | 666.7 | 333.3 |
| GE             | 110.238 kGE | 176.25 kGE |
| Std Cell [#]   | 68186 | 68994 | 
| Voltage [V]    |  0.77         | 1.1             |
| Util [%]       | 45.0 | 59.2 | 
| TNS            | -1086.59   | -0.31 |
| Clock Net      | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/asap7/halut_matmul/reports/asap7/halut_matmul/base/final_clocks.webp.png)  | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_matmul/reports/nangate45/halut_matmul/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_matmul/reports/report-gallery-halut_matmul.html)  | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_matmul/reports/report-gallery-halut_matmul.html)  |
| Metrics        | [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_matmul/metrics.html)  |  [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_matmul/metrics.html)  |
| Report         | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_matmul/reports/report-table.html)  | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_matmul/reports/report-table.html)  |


### Encoder
| halut_encoder_4         | ASAP7         | NanGate45      |
| -------------  | ------------- | -------------  |
| Area [μm^2]    | 4844.5405  | 69711.9531 |
| Freq [Mhz]     | 666.7 | 333.3 |
| GE             | 55.378 kGE | 87.358 kGE |
| Std Cell [#]   | 34334 | 33746 | 
| Voltage [V]    |  0.77         | 1.1             |
| Util [%]       | 45.0 | 58.7 | 
| TNS            | 0.0   | 0.0 |
| Clock Net      | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/asap7/halut_encoder_4/reports/asap7/halut_encoder_4/base/final_clocks.webp.png)  | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_encoder_4/reports/nangate45/halut_encoder_4/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_encoder_4/reports/report-gallery-halut_encoder_4.html)  | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_encoder_4/reports/report-gallery-halut_encoder_4.html)  |
| Metrics        | [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_encoder_4/metrics.html)  |  [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_encoder_4/metrics.html)  |
| Report         | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_encoder_4/reports/report-table.html)  | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_encoder_4/reports/report-table.html)  |


### Decoder
| halut_decoder         | ASAP7         | NanGate45      |
| -------------  | ------------- | -------------  |
| Area [μm^2]    | 4749.8286  | 68923.7891 |
| Freq [Mhz]     | 666.7 | 333.3 |
| GE             | 54.296 kGE | 86.37 kGE |
| Std Cell [#]   | 33709 | 34395 | 
| Voltage [V]    |  0.77         | 1.1             |
| Util [%]       | 44.4 | 58.9 | 
| TNS            | -11340.5098   | -0.66 |
| Clock Net      | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/asap7/halut_decoder/reports/asap7/halut_decoder/base/final_clocks.webp.png)  | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_decoder/reports/nangate45/halut_decoder/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_decoder/reports/report-gallery-halut_decoder.html)  | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_decoder/reports/report-gallery-halut_decoder.html)  |
| Metrics        | [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_decoder/metrics.html)  |  [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_decoder/metrics.html)  |
| Report         | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/asap7/halut_decoder/reports/report-table.html)  | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_decoder/reports/report-table.html)  |

## Progress Slides

* [Week 11 slides](https://github.com/joennlae/halutdata/raw/master/slides/week_11.pdf)
* [Weekly updated slides](http://jsdev.vsos.ethz.ch/maddness/progress-slides.pdf)

![Slides preview](https://github.com/joennlae/halutdata/raw/master/slides/week_11.gif)

## `CUDA` kernels

* [Encode Kernel](src/python/halutmatmul/cuda/kernels/encode.cu)
* [Decode Kernel](src/python/halutmatmul/cuda/kernels/read_acc_lut.cu)

_I am aware that there is still a lot that could be optimized here (warp etc.), but it was only developed for fast analysis_

## Results

Caveats: No retraining and fine-tuning done yet!
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


### `C`, `K` and `encoding_algorithm` parameter sweep for ResNet-50

* [Data visualizer](http://jsdev.vsos.ethz.ch/halut/)
_be sure to select ResNet-50 layers `layerX.X.convX`_

![Data visualizer](https://github.com/joennlae/halutdata/raw/master/figures/halut_viewer.png)
### Offline learning convergence on ResNet-50

The goal was to find out how much offline training data is needed to get the maximum accuracy.

![ResNet-50 Convergence Results](https://github.com/joennlae/halutdata/raw/master/figures/all_layers.png)

## Formalism

Some definitions about the forward path.

### Encode kernel
![](docs/images/encode_kernel.png)
### Read and accumulate LUTs kernel
![](docs/images/read_acc_lut_kernel.png)
### Links

* [Addendum](docs/addendum.md)

## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fjoennlae%2Fhalutmatmul.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fjoennlae%2Fhalutmatmul?ref=badge_large)