try:
  from .base import BaseModel
except:
  from base import BaseModel

from typing import Sequence
import torch.nn as nn
import torch.nn.functional as F


net = [
        # 0 : size: 24x24, channel: 3
        nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(64)
        ),
        # 1 : kernel: 3x3, channel: 64, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(64)),
        # 2 : kernel: 3x3, channel: 64, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(128)),
        # 3 : kernel: 3x3, channel: 128, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(128)),
        # 4 : kernel: 3x3, channel: 128, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(256)),
        # 5 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(256)),
        # 6 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(256)),
        # 7 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(256)),
        # 8 : kernel: 3x3, channel: 256, padding: 1
        nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(256)),
        # 9 : channel: 1024
        # 10 : channel: 1024
        # 11 : channel: 10
]

class Convnet2BN(BaseModel):

  def __init__(self, num_classes=10, backpropgate=False):
    super(Convnet2BN, self).__init__(net, num_classes, backpropgate=backpropgate)
    self.classifier = nn.Sequential(
      nn.LazyLinear(1024),
      # nn.Linear(1024, 1024),
      nn.Linear(1024, num_classes))
    self.batch_layers = {}
    self.device = None
    self.batch_norm = True
  

  def forward(self, x):
    for i, layer in enumerate(self.current_layers):
      if i in {2, 4, 8}: # 8
        x = F.max_pool2d(x, kernel_size=(2,2),stride=2)
        x = F.dropout2d(x, 0.25)
      x = F.relu(layer(x))

    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x


    

