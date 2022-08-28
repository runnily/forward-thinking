import torch.nn as nn
import torch.nn.functional as F

try:
    from .base import BaseModel, conv_2d_relu
except:
    from base import BaseModel, conv_2d_relu


class SimpleNet(BaseModel):
    def __init__(self, num_classes: int, batch_norm: bool, init_weights: bool = True) -> None:
        net = nn.Sequential(
            # 0 : kernel: 3x3, out-channel = 64, padding 1
            nn.Sequential(*conv_2d_relu(3, 64, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))),
            # 1 : kernel: 3x3, out-channel = 128, padding 1
            nn.Sequential(
                *conv_2d_relu(64, 128, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))
            ),
            # 2 : kernel: 3x3, out-channel = 128, padding 1
            nn.Sequential(
                *conv_2d_relu(128, 128, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))
            ),
            # 3 : kernel: 3x3, out-channel = 128, padding 1
            nn.Sequential(
                *conv_2d_relu(128, 128, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1),
            ),
            # 4 : kernel: 3x3, out-channel = 128, padding 1
            nn.Sequential(
                *conv_2d_relu(128, 128, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))
            ),
            # 5 : kernel: 3x3, out-channel = 128, padding 1
            nn.Sequential(
                *conv_2d_relu(128, 128, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))
            ),
            # 6 : kernel: 3x3, out-channel = 256, padding 1
            nn.Sequential(
                *conv_2d_relu(128, 256, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1),
            ),
            # 7 : kernel: 3x3, out-channel = 256, padding 1
            nn.Sequential(
                *conv_2d_relu(256, 256, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))
            ),
            # 8 : kernel: 3x3, out-channel = 256, padding 1
            nn.Sequential(
                *conv_2d_relu(256, 256, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1),
            ),
            # 9 : kernel: 3x3, out-channel = 512, padding 1
            nn.Sequential(
                *conv_2d_relu(256, 512, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1),
            ),
            # 10: kernel: 2048, out-channel = 512, padding 1
            nn.Sequential(
                *conv_2d_relu(512, 2048, (1, 1), batch_norm, stride=(1, 1), padding=(0, 0))
            ),
            # 11 : kernel: 2048, out-channel = 512, padding 1
            nn.Sequential(
                *conv_2d_relu(2048, 256, (1, 1), batch_norm, stride=(1, 1), padding=(0, 0)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                nn.Dropout2d(p=0.1),
            ),
            # 12 : kernel: 256, out-channel = 256, padding 1
            nn.Sequential(
                *conv_2d_relu(256, 256, (3, 3), batch_norm, stride=(1, 1), padding=(1, 1))
            ),
        )
        in_channels = 3
        super().__init__(net, num_classes, batch_norm, in_channels, init_weights)

    def forward(self, x):
        x = self.current_layers(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
