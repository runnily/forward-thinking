try:
  from .base import BaseModel
except:
  from base import BaseModel

import torch.nn as nn
import torch.nn.functional as F


basicNet = [
        nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1),
        nn.ReLu(),
        

]