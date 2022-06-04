try:
  from .base import BaseModel
except:
  from base import BaseModel

import torch.nn as nn
import torch.nn.functional as F


basicNet = nn.ModuleList([
        nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
        
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
])
        #nn.Linear(25088, 4096),
        #nn.Linear(4096, 4096),
        #nn.Linear(4096, 10)])

class BasicNet(BaseModel):

  def __init__(self, num_classes=10, backpropgate=False):
    super(BasicNet, self).__init__(basicNet, num_classes, backpropgate=backpropgate)

  def forward(self, x):
    for i in range(len(self.current_layers)):
      if i in [2, 4, 7, 10, 12]:
        x = F.max_pool2d(x, kernel_size=2, stride=2)
      x = F.relu(self.current_layers[i](x))
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x



