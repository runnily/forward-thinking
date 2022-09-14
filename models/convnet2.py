import torch.nn as nn

try:
    from .base import BaseModel, conv_2d_relu
except ImportError:
    from base import BaseModel, conv_2d_relu


class Convnet2(BaseModel):
    def __init__(self, num_classes: int, batch_norm: bool, init_weights: bool = True, affine: bool = True) -> None:
        in_channels = 3
        net = nn.Sequential(
            # 0 : size: 24x24, channel: 3
            nn.Sequential(*conv_2d_relu(3, 64, (3, 3), batch_norm, padding=1, affine=affine)),
            # 1 : kernel: 3x3, channel: 64, padding: 1
            nn.Sequential(*conv_2d_relu(64, 64, (3, 3), batch_norm, padding=1, affine=affine)),
            # 2 : kernel: 3x3, channel: 64, padding: 1
            nn.Sequential(
                *conv_2d_relu(64, 128, (3, 3), batch_norm, padding=1, affine=affine),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Dropout(p=0.25)
            ),
            # 3 : kernel: 3x3, channel: 128, padding: 1
            nn.Sequential(*conv_2d_relu(128, 128, (3, 3), batch_norm, padding=1, affine=affine)),
            # 4 : kernel: 3x3, channel: 128, padding: 1
            nn.Sequential(
                *conv_2d_relu(128, 256, (3, 3), batch_norm, padding=1, affine=affine),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Dropout(p=0.25)
            ),
            # 5 : kernel: 3x3, channel: 256, padding: 1
            nn.Sequential(*conv_2d_relu(256, 256, (3, 3), batch_norm, padding=1, affine=affine)),
            # 6 : kernel: 3x3, channel: 256, padding: 1
            nn.Sequential(*conv_2d_relu(256, 256, (3, 3), batch_norm, padding=1, affine=affine)),
            # 7 : kernel: 3x3, channel: 256, padding: 1
            nn.Sequential(*conv_2d_relu(256, 256, (3, 3), batch_norm, padding=1, affine=affine)),
            # 8 : kernel: 3x3, channel: 256, padding: 1
            nn.Sequential(
                *conv_2d_relu(256, 256, (3, 3), batch_norm, padding=1, affine=affine),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Dropout(p=0.25)
            ),
        )

        super().__init__(net, num_classes, batch_norm, in_channels, init_weights)

        self.output = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.current_layers(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
