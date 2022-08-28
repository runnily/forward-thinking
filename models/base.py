import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
        self,
        incoming_layers: nn.Module,
        num_classes: int,
        batch_norm: bool,
        in_channels: int,
        init_weights: bool = True,
    ) -> None:
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        self.in_channels = in_channels
        self.incoming_layers = incoming_layers
        if len(incoming_layers) == 0:
            raise ValueError("Error cannot be initalized with size 0")
        self.frozen_layers = nn.ModuleList([])
        self.current_layers = nn.Sequential()

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        self.output = nn.Linear(512, num_classes)


def conv_2d(in_features, out_features, kernel_size, batch_norm, **kwargs):
    conv_2d = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, **kwargs)
    if batch_norm == True:
        batch_norm = nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.05, affine=True)
        return conv_2d, batch_norm
    return (conv_2d,)


def conv_2d_relu(*args, **kwargs):
    return (*conv_2d(*args, **kwargs), nn.ReLU())
