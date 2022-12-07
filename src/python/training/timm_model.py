import timm
import torch

from halutmatmul.modules import HalutConv2d, HalutLinear


def convert_to_halut(model, parent_name=""):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            halut_module = HalutLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                split_factor=1,
            )
            setattr(model, child_name, halut_module)
        elif isinstance(child, torch.nn.Conv2d) and parent_name != "":
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
            convert_to_halut(child, child_name)


def create_timm_checkpoint(model_name, checkpoint_path):
    model = timm.create_model(model_name, pretrained=True)
    state_dict_copy = model.state_dict().copy()
    convert_to_halut(model)

    model.load_state_dict(state_dict_copy, strict=False)
    model.half()
    print(model)
    torch.save(model, checkpoint_path)


if __name__ == "__main__":
    create_timm_checkpoint("resnet18", "resnet18.pth")
