try:
  from .base import BaseModel
except:
  from base import BaseModel

import torch.nn as nn
import torch.nn.functional as F


net = [
  nn.Sequential(
    nn.Linear(784, 150),
    nn.ReLU()),
  nn.Sequential(
    nn.Linear(150, 100),
    nn.ReLU()),
  nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU()),
]

class FeedForward(BaseModel):

  def __init__(self, input_size=784, num_classes=10, backpropgate=False):
    super(FeedForward, self).__init__(net, num_classes, backpropgate=backpropgate)
    self.incoming_layers[0].in_features = input_size
     

  def forward(self, x):
    out = x.view(x.size(0), -1)
    out = self.current_layers(out)
    out = self.classifier(out)
    return out
