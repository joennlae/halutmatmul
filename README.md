<div align="center">

# Stella Nera: A halutmatmul based Accelerator
</div>

<div align="center">

### Algorithmic CI
[![PyTorch Layer Test | PyTest](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_testing.yaml)
[![Python Linting](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/linting.yaml)
[![Mypy - Typechecking](https://github.com/joennlae/halutmatmul/actions/workflows/python_typing.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/python_typing.yaml)

### ML CI
[![ResNet9 - 92%+ accuracy](https://github.com/joennlae/halutmatmul/actions/workflows/resnet9_validation.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/resnet9_validation.yaml)
### Hardware CI

[![HW Synth + PAR OpenROAD](https://github.com/joennlae/halutmatmul/actions/workflows/hw_openroad.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/hw_openroad.yaml)
[![RTL Linting](https://github.com/joennlae/halutmatmul/actions/workflows/hw_linting.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/hw_linting.yaml)
[![HW Design Verification](https://github.com/joennlae/halutmatmul/actions/workflows/hw_dv.yaml/badge.svg)](https://github.com/joennlae/halutmatmul/actions/workflows/hw_dv.yaml)

</div>

## Paper

* [Stella Nera: Achieving 161 TOp/s/W with Multiplier-free DNN Acceleration based on Approximate Matrix Multiplication](https://arxiv.org/abs/2311.10207)

### Abstract


*The recent Maddness method approximates Matrix Multiplication (MatMul) without the need for multiplication by using a hash-based version of product quantization (PQ). The hash function is a decision tree, allowing for efficient hardware implementation, as multiply-accumulate operations are replaced by decision tree passes and LUT lookups. Stella Nera is the first Maddness accelerator achieving 15x higher area efficiency (GMAC/s/mm^2) and 25x higher energy efficiency (TMAC/s/W) than direct MatMul accelerators in the same technology. In a commercial 14 nm technology and scaled to 3 nm, we achieve an energy efficiency of 161 TOp/s/W@0.55V with a Top-1 accuracy on CIFAR-10 of over 92.5% using ResNet9.*


# Algorithmic - Maddness

![Maddness Animation](/docs/images/maddness_animation.webp)


### ResNet-9 LUTs, Thresholds, Dims

* [Download 92%+ Model](https://iis-people.ee.ethz.ch/~janniss/resnet9-best-int8.pth)

### Halutmatmul example

* [example.py](src/python/example.py)

```python
import numpy as np
from halutmatmul.halutmatmul import HalutMatmul

A = np.random.random((10000, 512))
A_train = A[:8000]
A_test = A[8000:]
B = np.random.random((512, 10))
C = np.matmul(A_test, B)

hm = HalutMatmul(C=32, K=16)
hm.learn_offline(A_train, B)
C_halut = hm.matmul_online(A_test)

mse = np.square(C_halut - C).mean()
print(mse)
```

## Installation

```bash
# install conda environment & activate
# mamba is recommended for faster install
conda env create -f environment_gpu.yml
conda activate halutmatmul

# IIS prefixed env
conda env create -f environment_gpu.yml --prefix /scratch/janniss/conda/halutmatmul_gpu
```

### Differentiable Maddness
<div align="center">
<img src="docs/images/code_preview.png" alt="Differentiable Maddness" width="600"> 
</div>

### Hardware - OpenROAD flow results from CI - NOT OPTIMIZED

All completely open hardware results are NOT OPTIMIZED! The results are only for reference and to show the flow works. In the paper results from commercial tools are shown. See this as a community service to make the hardware results more accessible.

| All Designs    |  NanGate45      |
| -------------  |  -------------  |
| All Report     | [All](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/latest/nangate45)  |
| History        | [History](https://github.com/joennlae/halutmatmul-openroad-reports/tree/main/history/nangate45)  |


#### Open Hardware Results Table
| NanGate45      |  halut_matmul  |  halut_encoder_4  |  halut_decoder  |
| -------------  |  -------------  |  -------------  |  -------------  |
| Area [μm^2]    |  128816 |  46782  |  24667.5  | 
| Freq [Mhz]     |  166.7 |  166.7  |  166.7  |
| GE             |  161.423 kGE | 58.624 kGE  |  30.911 kGE  |
| Std Cell [#]   |  65496 |  23130  |  12256  |
| Voltage [V]    |   1.1             | 1.1             | 1.1             |
| Util [%]       |  50.4 |  48.7  |  52.1  |
| TNS            |  0 |  0  |  0  |
| Clock Net      | <img src="https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_matmul/reports/final_clocks.webp" alt="Clock Net" width="150">  | <img src="https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_encoder_4/reports/final_clocks.webp" alt="Clock Net" width="150">  | <img src="https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_decoder/reports/final_clocks.webp" alt="Clock Net" width="150">  |
| Routing        | <img src="https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_matmul/reports/final_routing.webp" alt="Routing" width="150">   | <img src="https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_encoder_4/reports/final_routing.webp" alt="Routing" width="150">   | <img src="https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_decoder/reports/final_routing.webp" alt="Routing" width="150">   |
| GDS            | [GDS Download](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_matmul/results/6_final.gds)  | [GDS Download](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_encoder_4/results/6_final.gds)  | [GDS Download](https://raw.githubusercontent.com/joennlae/halutmatmul-openroad-reports/main/latest/nangate45/halut_decoder/results/6_final.gds)  |


#### Full design (halutmatmul)

Run locally with:
```bash
git submodule update --init --recursive
cd hardware
ACC_TYPE=INT DATA_WIDTH=8 NUM_M=8 NUM_DECODER_UNITS=4 NUM_C=16 make halut-open-synth-and-pnr-halut_matmul
```

### References

* [arXiv](https://arxiv.org/abs/2106.10860) Maddness paper
* Based on [MADDness/Bolt](https://github.com/dblalock/bolt).


## Citation

```bibtex
@article{schonleber2023stella,
  title={Stella Nera: Achieving 161 TOp/s/W with Multiplier-free DNN Acceleration based on Approximate Matrix Multiplication},
  author={Sch{\"o}nleber, Jannis and Cavigelli, Lukas and Andri, Renzo and Perotti, Matteo and Benini, Luca},
  journal={arXiv preprint arXiv:2311.10207},
  year={2023}
}
```