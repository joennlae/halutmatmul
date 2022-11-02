# Retraining

## Commands

```bash
torchrun --nproc_per_node 2 retraining/train.py
# resume
torchrun --nproc_per_node 2 retraining/train.py --resume /scratch2/janniss/model_checkpoints/checkpoint.pth --epochs 100
# start retraining

```

## Baseline
Test:  Acc@1 70.284 Acc@5 89.472

## Baseline CIFAR-100
Acc@1 68.250 Acc@5 90.510

## Baseline CIFAR-10
Test:  Acc@1 87.130 Acc@5 99.560