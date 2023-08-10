# Halutmatmul

### Algorithmic CI
[![GPU Tests (Vast.ai)](https://github.com/joennlae/halutmatmul/actions/workflows/gpu_tests.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/gpu_tests.yaml)
[![PyTest](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml)
[![Linting](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml)

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
# mamba is recommended for faster install
conda env create -f environment_gpu.yml
conda activate halutmatmul

# IIS prefixed env
conda env create -f environment_gpu.yml --prefix /scratch/janniss/conda/halutmatmul_gpu
```

## Hackernews mention (comments only) and discussion

* [HN: Bolt: Faster matrix and vector operations that run on compressed data](https://news.ycombinator.com/item?id=31792206)

## Hardware OpenROAD flow results

| All Designs    |  NanGate45      |
| -------------  |  -------------  |
| All Report     | [All](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/latest/nangate45)  |
| History        | [History](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/history/nangate45)  |

### Total Circuit (M=2)
| halut_matmul         |  NanGate45      |
| -------------  |  -------------  |
| Area [μm^2]    | 140647.7656 |
| Freq [Mhz]     |  333.3 |
| GE             |  176.25 kGE |
| Std Cell [#]   |  68994 | 
| Voltage [V]    |   1.1             |
| Util [%]       |  59.2 | 
| TNS            |  -0.31 |
| Clock Net      | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_matmul/reports/nangate45/halut_matmul/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_matmul/reports/report-gallery-halut_matmul.html)  |
| Metrics        | [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_matmul/metrics.html)  |
| Report         | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_matmul/reports/report-table.html)  |


### Encoder
| halut_encoder_4         |  NanGate45      |
| -------------  |  -------------  |
| Area [μm^2]    | 69711.9531 |
| Freq [Mhz]     |  333.3 |
| GE             |  87.358 kGE |
| Std Cell [#]   |  33746 | 
| Voltage [V]    |   1.1             |
| Util [%]       |  58.7 | 
| TNS            |  0.0 |
| Clock Net      | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_encoder_4/reports/nangate45/halut_encoder_4/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_encoder_4/reports/report-gallery-halut_encoder_4.html)  |
| Metrics        | [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_encoder_4/metrics.html)  |
| Report         | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_encoder_4/reports/report-table.html)  |


### Decoder
| halut_decoder         |  NanGate45      |
| -------------  |  -------------  |
| Area [μm^2]    | 68923.7891 |
| Freq [Mhz]     |  333.3 |
| GE             |  86.37 kGE |
| Std Cell [#]   |  34395 | 
| Voltage [V]    |   1.1             |
| Util [%]       |  58.9 | 
| TNS            |  -0.66 |
| Clock Net      | ![Clock_net](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_decoder/reports/nangate45/halut_decoder/base/final_clocks.webp)  |
| Gallery        | [Gallery Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_decoder/reports/report-gallery-halut_decoder.html)  |
| Metrics        | [Metrics Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_decoder/metrics.html)  |
| Report         | [Report Viewer](https://htmlpreview.github.io/?https://github.com/joennlae/halutmatmul-openroad-reports/blob/main/latest/nangate45/halut_decoder/reports/report-table.html)  |

### Links

* [Addendum](docs/addendum.md)