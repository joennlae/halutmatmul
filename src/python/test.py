
import torch, torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
imagenet_val = torchvision.datasets.ImageNet(
    root="/scratch2/janniss/imagenet/",
    split="val",
    transform=ResNet50_Weights.IMAGENET1K_V2.transforms()
)
loaded_data = DataLoader(
    imagenet_val,
    batch_size=128,
    num_workers=16,
    drop_last=False,
    pin_memory=True,
)
model.to(device)
model.eval()
correct_5 = correct_1 = 0
with torch.no_grad():
    for n_iter, (image, label) in enumerate(loaded_data):
        image, label = image.to(device), label.to(device)
        # if n_iter > 10:
        #     continue
        print(
            "iteration: {}\ttotal {} iterations".format(
                n_iter + 1, len(loaded_data)
            )
        )
        # https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b/test.py#L54
        output = model(image) # .squeeze(0).softmax(0)
        _, pred = output.topk(5, 1, largest=True, sorted=True)
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        correct_5 += correct[:, :5].sum()
        correct_1 += correct[:, :1].sum()
    print(correct_1, correct_5)
    print("Top 1 error: ", 1 - correct_1 / len(loaded_data.dataset))  # type: ignore[arg-type]
    print("Top 5 error: ", 1 - correct_5 / len(loaded_data.dataset))  # type: ignore[arg-type]
    print("Top 1 accuracy: ", correct_1 / len(loaded_data.dataset))  # type: ignore[arg-type]
    print("Top 5 accuracy: ", correct_5 / len(loaded_data.dataset))  # type: ignore[arg-type]
    print(
        "Parameter numbers: {}".format(
            sum(p.numel() for p in model.parameters())
        )
    )
