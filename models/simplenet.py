import torch.nn as nn
import torch.nn.functional as F
try:
  from .base import BaseModel
except:
  from base import BaseModel

net = nn.Sequential(
        # 0 : kernel: 3x3, out-channel = 64, padding 1
        nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        ),
        # 1 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        ),
        # 2 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        ),
        # 3 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 4 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        ),
        # 5 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        ),
        # 6 : kernel: 3x3, out-channel = 256, padding 1
        nn.Sequential(
          nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 7 : kernel: 3x3, out-channel = 256, padding 1
        nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        ),
        # 8 : kernel: 3x3, out-channel = 256, padding 1
        nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 9 : kernel: 3x3, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 10: kernel: 2048, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
          nn.ReLU(inplace=True),
        ),
        # 11 : kernel: 2048, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 12 : kernel: 2048, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.ReLU(inplace=True),
        )
)

net_bn = [
        # 0 : kernel: 3x3, out-channel = 64, padding 1
        nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 1 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 2 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 3 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 4 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 5 : kernel: 3x3, out-channel = 128, padding 1
        nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 6 : kernel: 3x3, out-channel = 256, padding 1
        nn.Sequential(
          nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 7 : kernel: 3x3, out-channel = 256, padding 1
        nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 8 : kernel: 3x3, out-channel = 256, padding 1
        nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 9 : kernel: 3x3, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 10 : kernel: 2048, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
          nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        ),
        # 11 : kernel: 2048, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
          nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
          nn.Dropout2d(p=0.1),
        ),
        # 12 : kernel: 2048, out-channel = 512, padding 1
        nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
          nn.ReLU(inplace=True),
        )
]

class SimpleNet(BaseModel):
  def __init__(self, 
    num_classes: int, 
    batch_norm: bool, 
    init_weights : bool = True) -> None:
    in_channels = 3
    if batch_norm:
      super().__init__(net_bn, num_classes, batch_norm, in_channels, init_weights)
    else:
      super().__init__(net, num_classes, batch_norm, in_channels, init_weights)
    self.classifier = nn.LazyLinear(num_classes)

  def forward(self, x):
    x = self.current_layers(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
