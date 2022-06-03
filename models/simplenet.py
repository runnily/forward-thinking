try:
  from .base import BaseModel
except:
  from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

net = nn.ModuleList([
  # 1st layer
  nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),

  # 2nd layer
  nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 3rd layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 4th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 5th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 6th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 7th layer
  nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),

  # 8th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),

  # 9th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),

  # 10th layer
  nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),

  # 11th layer
  nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
  nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),

  # 12th layer
  nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
 
  # 13th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
])


class SimpleNet(BaseModel):

  def __init__(self, num_classes=10, backpropgate=False):
      super(SimpleNet, self).__init__(net, num_classes, backpropgate=backpropgate)

  def forward(self, x):

    for i in range(1, len(self.current_layers)+1):
      if i % 2 == 0: # every batch layer is on the eventh index (i) so only apply relu here
        x = F.relu(self.current_layers[i-1](x), inplace=True) 
        if i in [4*2, 7*2, 9*2, 10*2]: # simple net applies this on the following layers
          x = F.max_pool2d(x, kernel_size = (2, 2), stride = (2, 2), dilation = (1, 1), ceil_mode = False)
          x = F.dropout2d(x, p=0.1)
        if i == 26: # on the last layer
          x = F.max_pool2d(x, kernel_size=x.size()[2:]) 
          x = F.dropout2d(x, 0.1, training=True)
      else:
        x = self.current_layers[i-1](x)
    x = x.reshape(x.shape[0], -1) # flatten to go into the linear hidden layer

    x = self.classifier(x)
    return x
      
