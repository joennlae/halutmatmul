# Retraining

## Commands

```bash
torchrun --nproc_per_node 2 retraining/train.py
# resume
torchrun --nproc_per_node 2 retraining/train.py --resume /scratch2/janniss/model_checkpoints/checkpoint.pth --epochs 100
```

## Baseline
Test:  Acc@1 70.284 Acc@5 89.472