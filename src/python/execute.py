# pylint: disable=C0209
import os

import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ResNet.resnet import resnet50

script_dir = os.path.dirname(__file__)
state_dict = torch.load(script_dir + "/.data/" + "resnet50" + ".pt", map_location="cpu")

dict_ = state_dict
print(dict_.keys())

val_transform = T.Compose(
    [
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ]
)

cifar_10_val = torchvision.datasets.CIFAR10(
    root="./.data", train=False, transform=val_transform, download=False
)

loaded_data = DataLoader(
    cifar_10_val, batch_size=128, num_workers=8, drop_last=False, pin_memory=True
)
total_iters = len(loaded_data)
n_rows = 0
for n_iter, (image, label) in enumerate(loaded_data):
    n_rows += label.size()[0]

print("total iters: {}, n_rows: {}".format(total_iters, n_rows))

model = resnet50(
    weights=state_dict,
    progress=False,
    **{"n_rows": n_rows, "n_iter": total_iters, "batch_size": 128}
)
print(model)
model.eval()

correct_1 = 0.0
correct_5 = 0.0
total = 0

with torch.no_grad():
    for n_iter, (image, label) in enumerate(loaded_data):
        # if n_iter > 10:
        #     continue
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(loaded_data)))
        # https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b/test.py#L54

        output = model(image).squeeze(0).softmax(0)

        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)

        correct = pred.eq(label).float()

        # compute top 5
        correct_5 += correct[:, :5].sum()

        # compute top1
        correct_1 += correct[:, :1].sum()

model.write_inputs_to_disk()

print(correct_1, correct_5)
print("Top 1 error: ", 1 - correct_1 / len(loaded_data))
print("Top 5 error: ", 1 - correct_5 / len(loaded_data))
print("Top 1 accuracy: ", correct_1 / len(loaded_data))
print("Top 5 accuracy: ", correct_5 / len(loaded_data))
print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
