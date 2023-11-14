# pylint: disable=line-too-long
import math


# System parameters
DECODER_BLOCKS = 8
DECODERS_PER_BLOCK = 8
C = 16
K = 16
STELLA_NERA_UNITS = 4

# 128 bit memory bandwidth
TCDM_BANDWIDTH = 4 * 32

FP_16_MAC_UNITS = (
    2 * STELLA_NERA_UNITS
)  # two operations per cycle we add two per stella nera unit

# Input parameters
"""
ResNet9(
  (conv1): Sequential(
    (0): HalutConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
  (conv2): Sequential(
    (0): HalutConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res1): Sequential(
    (0): Sequential(
      (0): HalutConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (1): Sequential(
      (0): HalutConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
  (conv3): Sequential(
    (0): HalutConv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4): Sequential(
    (0): HalutConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res2): Sequential(
    (0): Sequential(
      (0): HalutConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (1): Sequential(
      (0): HalutConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), halut_active=False, loop_order=im2col, split_factor=4,  use_prototypes=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
  (maxpool): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (classifier): Sequential(
    (0): HalutLinear(Halut in_features=256, out_features=10, bias=True)
  )
)
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet9                                  [1, 10]                   --
├─Sequential: 1-1                        [1, 64, 32, 32]           --
│    └─HalutConv2d: 2-1                  [1, 64, 32, 32]           1,804
│    └─BatchNorm2d: 2-2                  [1, 64, 32, 32]           128
│    └─ReLU: 2-3                         [1, 64, 32, 32]           --
├─Sequential: 1-2                        [1, 128, 16, 16]          --
│    └─HalutConv2d: 2-4                  [1, 128, 32, 32]          73,868
│    └─BatchNorm2d: 2-5                  [1, 128, 32, 32]          256
│    └─ReLU: 2-6                         [1, 128, 32, 32]          --
│    └─MaxPool2d: 2-7                    [1, 128, 16, 16]          --
├─Sequential: 1-3                        [1, 128, 16, 16]          --
│    └─Sequential: 2-8                   [1, 128, 16, 16]          --
│    │    └─HalutConv2d: 3-1             [1, 128, 16, 16]          147,596
│    │    └─BatchNorm2d: 3-2             [1, 128, 16, 16]          256
│    │    └─ReLU: 3-3                    [1, 128, 16, 16]          --
│    └─Sequential: 2-9                   [1, 128, 16, 16]          --
│    │    └─HalutConv2d: 3-4             [1, 128, 16, 16]          147,596
│    │    └─BatchNorm2d: 3-5             [1, 128, 16, 16]          256
│    │    └─ReLU: 3-6                    [1, 128, 16, 16]          --
├─Sequential: 1-4                        [1, 256, 8, 8]            --
│    └─HalutConv2d: 2-10                 [1, 256, 16, 16]          295,180
│    └─BatchNorm2d: 2-11                 [1, 256, 16, 16]          512
│    └─ReLU: 2-12                        [1, 256, 16, 16]          --
│    └─MaxPool2d: 2-13                   [1, 256, 8, 8]            --
├─Sequential: 1-5                        [1, 256, 4, 4]            --
│    └─HalutConv2d: 2-14                 [1, 256, 8, 8]            590,092
│    └─BatchNorm2d: 2-15                 [1, 256, 8, 8]            512
│    └─ReLU: 2-16                        [1, 256, 8, 8]            --
│    └─MaxPool2d: 2-17                   [1, 256, 4, 4]            --
├─Sequential: 1-6                        [1, 256, 4, 4]            --
│    └─Sequential: 2-18                  [1, 256, 4, 4]            --
│    │    └─HalutConv2d: 3-7             [1, 256, 4, 4]            590,092
│    │    └─BatchNorm2d: 3-8             [1, 256, 4, 4]            512
│    │    └─ReLU: 3-9                    [1, 256, 4, 4]            --
│    └─Sequential: 2-19                  [1, 256, 4, 4]            --
│    │    └─HalutConv2d: 3-10            [1, 256, 4, 4]            590,092
│    │    └─BatchNorm2d: 3-11            [1, 256, 4, 4]            512
│    │    └─ReLU: 3-12                   [1, 256, 4, 4]            --
├─MaxPool2d: 1-7                         [1, 256, 1, 1]            --
├─Sequential: 1-8                        [1, 10]                   --
│    └─HalutLinear: 2-20                 [1, 10]                   2,580
==========================================================================================
Total params: 2,441,844
Trainable params: 2,441,747
Non-trainable params: 97
Total mult-adds (M): 285.24
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 5.64
Params size (MB): 9.77
Estimated Total Size (MB): 15.42
==========================================================================================

ResNet9(
  2.44 M, 100.000% Params, 286.51 MMac, 100.000% MACs, 
  (conv1): Sequential(
    1.92 k, 0.079% Params, 2.03 MMac, 0.709% MACs, 
    (0): Conv2d(1.79 k, 0.073% Params, 1.84 MMac, 0.640% MACs, 3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, 0.005% Params, 131.07 KMac, 0.046% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0, 0.000% Params, 65.54 KMac, 0.023% MACs, inplace=True)
  )
  (conv2): Sequential(
    74.11 k, 3.035% Params, 76.15 MMac, 26.579% MACs, 
    (0): Conv2d(73.86 k, 3.025% Params, 75.63 MMac, 26.396% MACs, 64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, 0.010% Params, 262.14 KMac, 0.091% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0, 0.000% Params, 131.07 KMac, 0.046% MACs, inplace=True)
    (3): MaxPool2d(0, 0.000% Params, 131.07 KMac, 0.046% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res1): Sequential(
    295.68 k, 12.109% Params, 75.76 MMac, 26.442% MACs, 
    (0): Sequential(
      147.84 k, 6.055% Params, 37.88 MMac, 13.221% MACs, 
      (0): Conv2d(147.58 k, 6.044% Params, 37.78 MMac, 13.187% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, 0.010% Params, 65.54 KMac, 0.023% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0, 0.000% Params, 32.77 KMac, 0.011% MACs, inplace=True)
    )
    (1): Sequential(
      147.84 k, 6.055% Params, 37.88 MMac, 13.221% MACs, 
      (0): Conv2d(147.58 k, 6.044% Params, 37.78 MMac, 13.187% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(256, 0.010% Params, 65.54 KMac, 0.023% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0, 0.000% Params, 32.77 KMac, 0.011% MACs, inplace=True)
    )
  )
  (conv3): Sequential(
    295.68 k, 12.109% Params, 75.83 MMac, 26.465% MACs, 
    (0): Conv2d(295.17 k, 12.088% Params, 75.56 MMac, 26.373% MACs, 128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, 0.021% Params, 131.07 KMac, 0.046% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0, 0.000% Params, 65.54 KMac, 0.023% MACs, inplace=True)
    (3): MaxPool2d(0, 0.000% Params, 65.54 KMac, 0.023% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv4): Sequential(
    590.59 k, 24.187% Params, 37.83 MMac, 13.204% MACs, 
    (0): Conv2d(590.08 k, 24.166% Params, 37.77 MMac, 13.181% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, 0.021% Params, 32.77 KMac, 0.011% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(0, 0.000% Params, 16.38 KMac, 0.006% MACs, inplace=True)
    (3): MaxPool2d(0, 0.000% Params, 16.38 KMac, 0.006% MACs, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (res2): Sequential(
    1.18 M, 48.375% Params, 18.91 MMac, 6.599% MACs, 
    (0): Sequential(
      590.59 k, 24.187% Params, 9.45 MMac, 3.300% MACs, 
      (0): Conv2d(590.08 k, 24.166% Params, 9.44 MMac, 3.295% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, 0.021% Params, 8.19 KMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0, 0.000% Params, 4.1 KMac, 0.001% MACs, inplace=True)
    )
    (1): Sequential(
      590.59 k, 24.187% Params, 9.45 MMac, 3.300% MACs, 
      (0): Conv2d(590.08 k, 24.166% Params, 9.44 MMac, 3.295% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(512, 0.021% Params, 8.19 KMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(0, 0.000% Params, 4.1 KMac, 0.001% MACs, inplace=True)
    )
  )
  (maxpool): MaxPool2d(0, 0.000% Params, 4.1 KMac, 0.001% MACs, kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (classifier): Sequential(
    2.57 k, 0.105% Params, 2.57 KMac, 0.001% MACs, 
    (0): Linear(2.57 k, 0.105% Params, 2.57 KMac, 0.001% MACs, in_features=256, out_features=10, bias=True)
  )
)
Computational complexity:       286.51 MMac
Number of parameters:           2.44 M  

[DEBUG] transformed input shape:  torch.Size([1024, 27]) torch.Size([1, 3, 32, 32]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([27, 64]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([1024, 576]) torch.Size([1, 64, 32, 32]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([576, 128]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([256, 1152]) torch.Size([1, 128, 16, 16]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([1152, 128]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([256, 1152]) torch.Size([1, 128, 16, 16]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([1152, 128]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([256, 1152]) torch.Size([1, 128, 16, 16]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([1152, 256]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([64, 2304]) torch.Size([1, 256, 8, 8]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([2304, 256]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([16, 2304]) torch.Size([1, 256, 4, 4]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([2304, 256]) HalutConv2d cuda:0
[DEBUG] transformed input shape:  torch.Size([16, 2304]) torch.Size([1, 256, 4, 4]) HalutConv2d cuda:0
[DEBUG] transformed weights shape:  torch.Size([2304, 256]) HalutConv2d cuda:0
"""

"""
resnet20
Total number of params 269741
Total layers 20
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   [1, 10]                   --
├─Conv2d: 1-1                            [1, 16, 32, 32]           432
├─BatchNorm2d: 1-2                       [1, 16, 32, 32]           32
├─Sequential: 1-3                        [1, 16, 32, 32]           --
│    └─BasicBlock: 2-1                   [1, 16, 32, 32]           --
│    │    └─HalutConv2d: 3-1             [1, 16, 32, 32]           2,316
│    │    └─BatchNorm2d: 3-2             [1, 16, 32, 32]           32
│    │    └─HalutConv2d: 3-3             [1, 16, 32, 32]           2,316
│    │    └─BatchNorm2d: 3-4             [1, 16, 32, 32]           32
│    │    └─Sequential: 3-5              [1, 16, 32, 32]           --
│    └─BasicBlock: 2-2                   [1, 16, 32, 32]           --
│    │    └─HalutConv2d: 3-6             [1, 16, 32, 32]           2,316
│    │    └─BatchNorm2d: 3-7             [1, 16, 32, 32]           32
│    │    └─HalutConv2d: 3-8             [1, 16, 32, 32]           2,316
│    │    └─BatchNorm2d: 3-9             [1, 16, 32, 32]           32
│    │    └─Sequential: 3-10             [1, 16, 32, 32]           --
│    └─BasicBlock: 2-3                   [1, 16, 32, 32]           --
│    │    └─HalutConv2d: 3-11            [1, 16, 32, 32]           2,316
│    │    └─BatchNorm2d: 3-12            [1, 16, 32, 32]           32
│    │    └─HalutConv2d: 3-13            [1, 16, 32, 32]           2,316
│    │    └─BatchNorm2d: 3-14            [1, 16, 32, 32]           32
│    │    └─Sequential: 3-15             [1, 16, 32, 32]           --
├─Sequential: 1-4                        [1, 32, 16, 16]           --
│    └─BasicBlock: 2-4                   [1, 32, 16, 16]           --
│    │    └─HalutConv2d: 3-16            [1, 32, 16, 16]           4,620
│    │    └─BatchNorm2d: 3-17            [1, 32, 16, 16]           64
│    │    └─HalutConv2d: 3-18            [1, 32, 16, 16]           9,228
│    │    └─BatchNorm2d: 3-19            [1, 32, 16, 16]           64
│    │    └─LambdaLayer: 3-20            [1, 32, 16, 16]           --
│    └─BasicBlock: 2-5                   [1, 32, 16, 16]           --
│    │    └─HalutConv2d: 3-21            [1, 32, 16, 16]           9,228
│    │    └─BatchNorm2d: 3-22            [1, 32, 16, 16]           64
│    │    └─HalutConv2d: 3-23            [1, 32, 16, 16]           9,228
│    │    └─BatchNorm2d: 3-24            [1, 32, 16, 16]           64
│    │    └─Sequential: 3-25             [1, 32, 16, 16]           --
│    └─BasicBlock: 2-6                   [1, 32, 16, 16]           --
│    │    └─HalutConv2d: 3-26            [1, 32, 16, 16]           9,228
│    │    └─BatchNorm2d: 3-27            [1, 32, 16, 16]           64
│    │    └─HalutConv2d: 3-28            [1, 32, 16, 16]           9,228
│    │    └─BatchNorm2d: 3-29            [1, 32, 16, 16]           64
│    │    └─Sequential: 3-30             [1, 32, 16, 16]           --
├─Sequential: 1-5                        [1, 64, 8, 8]             --
│    └─BasicBlock: 2-7                   [1, 64, 8, 8]             --
│    │    └─HalutConv2d: 3-31            [1, 64, 8, 8]             18,444
│    │    └─BatchNorm2d: 3-32            [1, 64, 8, 8]             128
│    │    └─HalutConv2d: 3-33            [1, 64, 8, 8]             36,876
│    │    └─BatchNorm2d: 3-34            [1, 64, 8, 8]             128
│    │    └─LambdaLayer: 3-35            [1, 64, 8, 8]             --
│    └─BasicBlock: 2-8                   [1, 64, 8, 8]             --
│    │    └─HalutConv2d: 3-36            [1, 64, 8, 8]             36,876
│    │    └─BatchNorm2d: 3-37            [1, 64, 8, 8]             128
│    │    └─HalutConv2d: 3-38            [1, 64, 8, 8]             36,876
│    │    └─BatchNorm2d: 3-39            [1, 64, 8, 8]             128
│    │    └─Sequential: 3-40             [1, 64, 8, 8]             --
│    └─BasicBlock: 2-9                   [1, 64, 8, 8]             --
│    │    └─HalutConv2d: 3-41            [1, 64, 8, 8]             36,876
│    │    └─BatchNorm2d: 3-42            [1, 64, 8, 8]             128
│    │    └─HalutConv2d: 3-43            [1, 64, 8, 8]             36,876
│    │    └─BatchNorm2d: 3-44            [1, 64, 8, 8]             128
│    │    └─Sequential: 3-45             [1, 64, 8, 8]             --
├─HalutLinear: 1-6                       [1, 10]                   660
==========================================================================================
Total params: 269,948
Trainable params: 269,741
Non-trainable params: 207
Total mult-adds (M): 40.55
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 3.01
Params size (MB): 1.08
Estimated Total Size (MB): 4.11
==========================================================================================

ResNet(
  269.72 k, 100.000% Params, 40.93 MMac, 100.000% MACs, 
  (conv1): Conv2d(432, 0.160% Params, 442.37 KMac, 1.081% MACs, 3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1): Sequential(
    14.02 k, 5.196% Params, 14.35 MMac, 35.067% MACs, 
    (0): BasicBlock(
      4.67 k, 1.732% Params, 4.78 MMac, 11.689% MACs, 
      (conv1): Conv2d(2.3 k, 0.854% Params, 2.36 MMac, 5.765% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(2.3 k, 0.854% Params, 2.36 MMac, 5.765% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (1): BasicBlock(
      4.67 k, 1.732% Params, 4.78 MMac, 11.689% MACs, 
      (conv1): Conv2d(2.3 k, 0.854% Params, 2.36 MMac, 5.765% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(2.3 k, 0.854% Params, 2.36 MMac, 5.765% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (2): BasicBlock(
      4.67 k, 1.732% Params, 4.78 MMac, 11.689% MACs, 
      (conv1): Conv2d(2.3 k, 0.854% Params, 2.36 MMac, 5.765% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(2.3 k, 0.854% Params, 2.36 MMac, 5.765% MACs, 16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, 0.012% Params, 32.77 KMac, 0.080% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
  )
  (layer2): Sequential(
    51.07 k, 18.935% Params, 13.07 MMac, 31.945% MACs, 
    (0): BasicBlock(
      13.95 k, 5.173% Params, 3.57 MMac, 8.727% MACs, 
      (conv1): Conv2d(4.61 k, 1.708% Params, 1.18 MMac, 2.882% MACs, 16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, 0.024% Params, 16.38 KMac, 0.040% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(9.22 k, 3.417% Params, 2.36 MMac, 5.765% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, 0.024% Params, 16.38 KMac, 0.040% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): LambdaLayer(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (1): BasicBlock(
      18.56 k, 6.881% Params, 4.75 MMac, 11.609% MACs, 
      (conv1): Conv2d(9.22 k, 3.417% Params, 2.36 MMac, 5.765% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, 0.024% Params, 16.38 KMac, 0.040% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(9.22 k, 3.417% Params, 2.36 MMac, 5.765% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, 0.024% Params, 16.38 KMac, 0.040% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (2): BasicBlock(
      18.56 k, 6.881% Params, 4.75 MMac, 11.609% MACs, 
      (conv1): Conv2d(9.22 k, 3.417% Params, 2.36 MMac, 5.765% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, 0.024% Params, 16.38 KMac, 0.040% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(9.22 k, 3.417% Params, 2.36 MMac, 5.765% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, 0.024% Params, 16.38 KMac, 0.040% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
  )
  (layer3): Sequential(
    203.52 k, 75.455% Params, 13.03 MMac, 31.825% MACs, 
    (0): BasicBlock(
      55.55 k, 20.596% Params, 3.56 MMac, 8.687% MACs, 
      (conv1): Conv2d(18.43 k, 6.834% Params, 1.18 MMac, 2.882% MACs, 32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, 0.047% Params, 8.19 KMac, 0.020% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(36.86 k, 13.667% Params, 2.36 MMac, 5.765% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, 0.047% Params, 8.19 KMac, 0.020% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): LambdaLayer(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (1): BasicBlock(
      73.98 k, 27.430% Params, 4.73 MMac, 11.569% MACs, 
      (conv1): Conv2d(36.86 k, 13.667% Params, 2.36 MMac, 5.765% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, 0.047% Params, 8.19 KMac, 0.020% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(36.86 k, 13.667% Params, 2.36 MMac, 5.765% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, 0.047% Params, 8.19 KMac, 0.020% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
    (2): BasicBlock(
      73.98 k, 27.430% Params, 4.73 MMac, 11.569% MACs, 
      (conv1): Conv2d(36.86 k, 13.667% Params, 2.36 MMac, 5.765% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, 0.047% Params, 8.19 KMac, 0.020% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(36.86 k, 13.667% Params, 2.36 MMac, 5.765% MACs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, 0.047% Params, 8.19 KMac, 0.020% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    )
  )
  (linear): Linear(650, 0.241% Params, 650.0 Mac, 0.002% MACs, in_features=64, out_features=10, bias=True)
)
Computational complexity:       40.93 MMac
Number of parameters:           269.72 k

[DEBUG] transformed input shape:  torch.Size([1024, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 16]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([1024, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 16]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([1024, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 16]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([1024, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 16]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([1024, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 16]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([1024, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 16]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([256, 144]) torch.Size([1, 16, 32, 32]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([144, 32]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([256, 288]) torch.Size([1, 32, 16, 16]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([288, 32]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([256, 288]) torch.Size([1, 32, 16, 16]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([288, 32]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([256, 288]) torch.Size([1, 32, 16, 16]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([288, 32]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([256, 288]) torch.Size([1, 32, 16, 16]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([288, 32]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([256, 288]) torch.Size([1, 32, 16, 16]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([288, 32]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([64, 288]) torch.Size([1, 32, 16, 16]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([288, 64]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([64, 576]) torch.Size([1, 64, 8, 8]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([576, 64]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([64, 576]) torch.Size([1, 64, 8, 8]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([576, 64]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([64, 576]) torch.Size([1, 64, 8, 8]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([576, 64]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([64, 576]) torch.Size([1, 64, 8, 8]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([576, 64]) HalutConv2d cpu
[DEBUG] transformed input shape:  torch.Size([64, 576]) torch.Size([1, 64, 8, 8]) HalutConv2d cpu
[DEBUG] transformed weights shape:  torch.Size([576, 64]) HalutConv2d cpu
"""

# lets do inf/sec and then inf/sec/watt
# for ResNet20

# lets get the matmul sizes

resnet20_layer_info = [
    [1024, 27],
    [1, 3, 32, 32],
    [27, 16],
    # second layer
    [1024, 144],  # conv transform in
    [1, 16, 32, 32],  # conv input
    [144, 16],  # conv tranform weight
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [256, 144],
    [1, 16, 32, 32],
    [144, 32],
    [256, 288],
    [1, 32, 16, 16],
    [288, 32],
    [256, 288],
    [1, 32, 16, 16],
    [288, 32],
    [256, 288],
    [1, 32, 16, 16],
    [288, 32],
    [256, 288],
    [1, 32, 16, 16],
    [288, 32],
    [256, 288],
    [1, 32, 16, 16],
    [288, 32],
    [64, 288],
    [1, 32, 16, 16],
    [288, 64],
    [64, 576],
    [1, 64, 8, 8],
    [576, 64],
    [64, 576],
    [1, 64, 8, 8],
    [576, 64],
    [64, 576],
    [1, 64, 8, 8],
    [576, 64],
    [64, 576],
    [1, 64, 8, 8],
    [576, 64],
    [64, 576],
    [1, 64, 8, 8],
    [576, 64],
    # last layer
    [1, 64],
    [1, 64, 1, 1],
    [64, 10],
]

resnet9_layer_info = [
    [1024, 27],
    [1, 3, 32, 32],
    [27, 64],
    # second layer
    [1024, 576],
    [1, 64, 32, 32],
    [576, 128],
    # third
    [256, 1152],
    [1, 128, 16, 16],
    [1152, 128],
    [256, 1152],
    [1, 128, 16, 16],
    [1152, 128],
    [256, 1152],
    [1, 128, 16, 16],
    [1152, 256],
    [64, 2304],
    [1, 256, 8, 8],
    [2304, 256],
    [16, 2304],
    [1, 256, 4, 4],
    [2304, 256],
    [16, 2304],
    [1, 256, 4, 4],
    [2304, 256],
    # linear layer
    [1, 256],
    [1, 256, 1, 1],
    [256, 10],
]

resnet8_layer_info = [
    [1024, 27],
    [1, 3, 32, 32],
    [27, 16],
    # second layer
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [1024, 144],
    [1, 16, 32, 32],
    [144, 16],
    [256, 144],
    [1, 16, 32, 32],
    [144, 32],
    [256, 288],
    [1, 32, 16, 16],
    [288, 32],
    [256, 16],
    [1, 16, 32, 32],
    [16, 32],
    [64, 288],
    [1, 32, 16, 16],
    [288, 64],
    [64, 576],
    [1, 64, 8, 8],
    [576, 64],
    [64, 32],
    [1, 32, 16, 16],
    [32, 64],
    # last layer
    [1, 64],
    [1, 64, 1, 1],
    [64, 10],
]

total_cycles = 0
total_fJ = 0

# we assume everything is loaded into L2/L1 cache on a pulpissimo like architecture
# L1 access 0.9 pJ/Byte https://arxiv.org/pdf/2110.09101.pdf should be lower
mem_access_cost = 900
# FMA operation cost 2 FMA vector units scaled to 14nm and frequency and voltage
# --> 2 MACS/cycle per unit latency 3 cycles
# https://arxiv.org/pdf/2007.01530.pdf 3.43 pJ/(4xMACs) throughput per cycle
# for one unit 1.71 / cycle --> 2 MACS/cycle
mac_unit_cost_per_mac = 1710 // 2
# riscy core overhead for energy


def print_total_info(total_cycles, total_fJ):
    print("mJ", total_fJ / 1e12)
    print("time ms", total_cycles * (1 / (624 * 1e6)))


def execute_fp16_conv2d(input_shape, kernel_size, out_channel, stride=1, padding=1):
    cycles = 0
    fJ = 0
    # pylint: disable=too-many-nested-blocks
    for _ in range(input_shape[0]):  # batch_size
        for _ in range(input_shape[1]):  # in_channel
            for _ in range(0, input_shape[2] + 2 * padding, stride):
                for _ in range(0, input_shape[3] + 2 * padding, stride):
                    for _ in range(out_channel):  # out_cha
                        fJ += kernel_size * kernel_size * 2 * mem_access_cost  # read
                        for _ in range(kernel_size):  # k_x
                            for _ in range(kernel_size):  # k_y
                                fJ += mac_unit_cost_per_mac + mem_access_cost * 2  # MAC
                                cycles += 0.5 / FP_16_MAC_UNITS  # 1 * MAC unit
                        fJ += mem_access_cost * 2  # write result
    return cycles, fJ


def execute_fp16_macs(n, d, m):
    cycles = 0
    fJ = 0
    for _ in range(n):
        for _ in range(m):
            fJ += mem_access_cost * 2
            for _ in range(d):
                fJ += mac_unit_cost_per_mac + (mem_access_cost * 2)
                cycles += 0.5 / (FP_16_MAC_UNITS)
            fJ += mem_access_cost * 2  # write result
    return cycles, fJ


def execute_maddness(n, d, m, input_shape, output_shape):
    # default kernel size is 3
    kernel_size = 3
    div_factor = kernel_size * kernel_size
    c = d // div_factor
    if (d % div_factor) > 0:
        # kernel size 1
        print("assuming 4 four kernel size == 1")
        div_factor = 4
        kernel_size = 1
        c = d // div_factor
        assert (d % div_factor) == 0
    cycles = 0
    fJ = 0
    # mapping to accelerators
    # we first tile over c
    # pooling = n // (output_shape[-2] * output_shape[-1])
    units = math.ceil(c / C)
    mapping = {}
    for i in range(STELLA_NERA_UNITS):
        mapping[i] = []
    units_assignment = {}
    for i in range(STELLA_NERA_UNITS):
        units_assignment[i] = []
    for i in range(units):
        offset = (i // STELLA_NERA_UNITS) * STELLA_NERA_UNITS * C
        current_assignment = []
        for j in range(
            offset,
            offset + C,
            C - 1,
        ):
            current_assignment.append((i * C) + j)
        if i == units - 1:
            current_assignment[1] = c - 1
        units_assignment[i % STELLA_NERA_UNITS].append(current_assignment)
    for i in range(STELLA_NERA_UNITS):
        mapping[i] = {"C": units_assignment[i]}
    # TODO if units < STELLA_NERA_UNITS optimizations are possible
    # double buffering for example

    m_capacity = DECODER_BLOCKS * DECODERS_PER_BLOCK
    m_rounds = math.ceil(m / m_capacity)
    m_mapping = []
    for i in range(m_rounds):
        m_mapping.append([(i * m_capacity), (i * m_capacity) + m_capacity - 1])
        if i == m_rounds - 1:
            m_mapping[i][1] = m - 1
    print(mapping)
    for j in range(STELLA_NERA_UNITS):
        mapping[j]["M"] = m_mapping

    # schedule
    schedule = {}
    for i in range(STELLA_NERA_UNITS):
        schedule_unit = []
        mapping_unit = mapping[i]
        for c_mapping in mapping_unit["C"]:
            for m_mapping in mapping_unit["M"]:
                schedule_unit.append((c_mapping, m_mapping))
        schedule[i] = schedule_unit
    # print(schedule)
    m_length = 0
    c_length = 0
    for i in range(STELLA_NERA_UNITS):
        m_length = max(m_length, len(mapping[i]["M"]))
        c_length = max(c_length, len(mapping[i]["C"]))

    # longest schedule length
    schedule_length = 0
    for i in range(STELLA_NERA_UNITS):
        schedule_length = max(schedule_length, len(schedule[i]))
    # execute schedule
    # print("lengths", m_length, c_length, schedule_length)
    # weight loading
    for i in range(schedule_length):
        for j in range(STELLA_NERA_UNITS):
            if i < len(schedule[j]):
                c_mapping, m_mapping = schedule[j][i]
                # load weights
                for _ in range(c_mapping[0], c_mapping[1] + 1):
                    for _ in range(m_mapping[0], m_mapping[1] + 1):
                        fJ += mem_access_cost  # int 8
                        cycles += 1 / (TCDM_BANDWIDTH // 8)
        cycles += 10  # latency until streaming 7 normally + 3 FMA + 2 to be overcorrect
    # input loading assuming we have a im2col unit
    for _ in range(input_shape[0]):
        for _ in range(input_shape[1]):
            for _ in range(input_shape[2]):
                for _ in range(input_shape[3]):
                    fJ += mem_access_cost * 2  # FP16
                    # no cycles as this is done during execution and wont be bandwidth limited
    # storing weight assuming we have a relu, batchnorm fused into the FMA as described in the paper
    # and pooling unit
    total_output_mem_cost = 0
    for _ in range(output_shape[0]):
        for _ in range(output_shape[1]):
            for _ in range(output_shape[2]):
                for _ in range(output_shape[3]):
                    fJ += mem_access_cost * 2
                    total_output_mem_cost += mem_access_cost * 2
    # additional memory costs when c_length > 1 and we have to swap to L1
    if c_length > 1:
        # without optimizations we have to read and write the whole output
        fJ += total_output_mem_cost * (c_length - 1) * 2  # read and write
    # execute
    for i in range(schedule_length):
        for j in range(STELLA_NERA_UNITS):
            if i < len(schedule[j]):
                c_mapping, m_mapping = schedule[j][i]
                # execution cycles and energy
                for _ in range(n):
                    cycles += C  # takes c cyles to get a value
                    fJ += (
                        23600
                        + mac_unit_cost_per_mac
                        * 4  # per cycles / stella nera unit + 4x MACs
                    ) * C
                    # that is a clear overestimate
                    # as most of the time we sum over C first and then have to do the conversion once
    return cycles, fJ


# output calculation
# 8 outputs per cycles for 8 cycles
# 4 outputs per cycles
# 2 FPUs needed per stella nera accelerator --> could be heavily optimized

# resnet20_layer_info, resnet9_layer_info, resnet8_layer_info
model_info = resnet9_layer_info

# first layer
fp16_macs = model_info[0][0] * model_info[0][1] * model_info[2][1]
print("fp16_macs", fp16_macs)
cycles_fp16, fJ_fp16 = execute_fp16_conv2d(model_info[1], 3, 64)
total_cycles += cycles_fp16
total_fJ += fJ_fp16

print("layer 1 cycles", cycles_fp16, "fJ", fJ_fp16)
print_total_info(total_cycles, total_fJ)

# second layer

for i in range(0, (len(model_info) // 3) - 2):
    print("LAYER", i + 2)
    offset = i * 3
    cycles, fJ = execute_maddness(
        model_info[3 + offset][0],
        model_info[3 + offset][1],
        model_info[5 + offset][1],
        model_info[4 + offset],
        model_info[7 + offset],
    )
    print("layer", i + 2, "cycles", cycles, "fJ", fJ)
    total_cycles += cycles
    total_fJ += fJ
    print_total_info(total_cycles, total_fJ)

cycles, fJ = execute_fp16_macs(model_info[-3][0], model_info[-3][1], model_info[-1][1])
total_cycles += cycles
total_fJ += fJ
print_total_info(total_cycles, total_fJ)

inf_per_sec = 1 / (total_cycles * (1 / (624 * 1e6)))
mJ_per_inf = total_fJ / 1e12
print("inf/sec", inf_per_sec)
print("mJ/inf", mJ_per_inf)

# import numpy as np
#
# A = np.random.random((resnet20_first_layer_info[0][0], resnet20_first_layer_info[0][1]))
# B = np.random.random((resnet20_first_layer_info[0][1], resnet20_first_layer_info[2][1]))
#
# C = np.zeros((resnet20_first_layer_info[0][0], resnet20_first_layer_info[2][1]))
#
# # matmul
# for i in range(resnet20_first_layer_info[0][0]):
#     for j in range(resnet20_first_layer_info[0][1]):
#         for k in range(resnet20_first_layer_info[2][1]):
#             C[i][k] += A[i][j] * B[j][k]
