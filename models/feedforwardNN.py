try:
  from .base import BaseModel
except:
  from base import BaseModel

import torch.nn as nn
import torch.nn.functional as F


net = [
      nn.Linear(784, 512),
      nn.Linear(150, 100),
      nn.Linear(100, 50),
        # 9 : channel: 1024
        # 10 : channel: 1024
        # 11 : channel: 10
]

class FeedForward(BaseModel):

  def __init__(self, input_size=784, num_classes=10, backpropgate=False):
    net[0].in_features = 784
    super(FeedForward, self).__init__(net, num_classes, backpropgate=backpropgate)
    #self.incoming_layers[0].in_features = input_size
     

  def forward(self, x):
    out = x.view(x.size(0), -1)
    for l in self.layers:
      out = l(out)
      out = F.relu(out)
    out = self.classifier(out)
    return out
