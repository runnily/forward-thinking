try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel

import torch.nn as nn


class FeedForward(BaseModel):
    def __init__(
        self, input_size: int = 784, num_classes: int = 10, init_weights: bool = True
    ) -> None:
        super(FeedForward, self).__init__([
          nn.Sequential(nn.Linear(input_size, 150), nn.ReLU()),
          nn.Sequential(nn.Linear(150, 100), nn.ReLU()),
          nn.Sequential(nn.Linear(100, 50), nn.ReLU()),
        ], num_classes, init_weights, in_channels=1)
        self.output = nn.LazyLinear(num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.current_layers(out)
        out = self.output(out)
        return out
