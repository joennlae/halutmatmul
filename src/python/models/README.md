# Models

## Classification

### ImageNet

* ResNet-50 ([Source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py))
* LeViT ([Source](https://github.com/facebookresearch/LeViT))

#### Run LeViT validation

```bash
# with 
# from models.levit.main import run_levit  # type: ignore[attr-defined]
# run_levit() in src/python/starter.py
python src/python/starter.py --eval --model LeViT_128S --data-path /scratch/ml_datasets/ILSVRC2012/
```

### KWS
* Key-Word-Spotting - KWS ([Source](https://github.com/pulp-platform/kws-on-pulp))

#### Run KWS validation

```bash
# with 
# from models.dscnn.main import run_kws_main
# run_kws_main() in src/python/starter.py
python src/python/starter.py
```