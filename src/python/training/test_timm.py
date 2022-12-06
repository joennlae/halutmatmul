import timm
import torch

from halutmatmul.modules import HalutConv2d, HalutLinear

print(timm.list_models("resnet*", pretrained=True))

model = timm.create_model("resnet18", pretrained=True)
state_dict_copy = model.state_dict().copy()
print(model)
print(model.state_dict().keys())


def convert_to_halut(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            halut_module = HalutLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                split_factor=1,
            )
            setattr(model, child_name, halut_module)
        elif isinstance(child, torch.nn.Conv2d):
            halut_module = HalutConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,  # type: ignore
                child.dilation,
                child.groups,
                child.bias is not None,
                child.padding_mode,
                split_factor=1,
            )
            setattr(model, child_name, halut_module)
        else:
            convert_to_halut(child)


convert_to_halut(model)
print(model)

model.load_state_dict(state_dict_copy, strict=False)
