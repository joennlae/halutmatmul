# Retraining

## Commands

```bash
python training/train.py --device cuda:3 --opt adam --model resnet9 --cifar10 --lr 0.001 --lr-scheduler cosineannealinglr --epochs 200 --amp --output-dir /scratch2/janniss/model_checkpoints/resnet9-lr-0.001-amp
# start retraining
python retraining.py 3 -single -testname resnet9-lpl-0.001-amp-lut8-25-93.7 -checkpoint /scratch2/janniss/model_checkpoints/resnet9-lr-0.001-amp/model_best-93.69.pth
```