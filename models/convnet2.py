import torch.nn as nn
import torch.nn.functional as F
try:
  from .base import BaseModel
except:
  from base import BaseModel



net = nn.Sequential(
        # 0 : size: 24x24, channel: 3
        nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),),
        # 1 : kernel: 3x3, channel: 64, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),),
        # 2 : kernel: 3x3, channel: 64, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2,2),stride=2),
          nn.Dropout(p=0.25),),
        # 3 : kernel: 3x3, channel: 128, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),),
        # 4 : kernel: 3x3, channel: 128, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2,2),stride=2),
          nn.Dropout(p=0.25),),
        # 5 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),),
        # 6 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),),
        # 7 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),),
        # 8 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2,2),stride=2),
          nn.Dropout(p=0.25),)
)

net_bn = [
        # 0 : size: 24x24, channel: 3
        nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),),
        # 1 : kernel: 3x3, channel: 64, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),),
        # 2 : kernel: 3x3, channel: 64, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2,2),stride=2),
          nn.Dropout(p=0.25),),
        # 3 : kernel: 3x3, channel: 128, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),),
        # 4 : kernel: 3x3, channel: 128, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2,2),stride=2),
          nn.Dropout(p=0.25),),
        # 5 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),),
        # 6 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),),
        # 7 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),),
        # 8 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2,2),stride=2),
          nn.Dropout(p=0.25),)
]

class Convnet2(BaseModel):

  def __init__(self, num_classes: int = 10, batch_norm: bool = False, init_weights : bool = True):
    if batch_norm:
      super().__init__(net_bn, num_classes)
    else:
      super().__init__(net, num_classes)

    if init_weights:
      for m in self.modules():
        print(m)
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

    self.classifier = nn.LazyLinear(num_classes)

  
      

  def forward(self, x):
    x = self.current_layers(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

    

