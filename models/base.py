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
        else:
            if isinstance(incoming_layers[0], int):
                self.incoming_layers = self.make_layers(incoming_layers, batch_norm)
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
        self.classifier = nn.Linear(512, num_classes)
