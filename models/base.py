import torch.nn as nn


class BaseModel(nn.Module):

  def __init__(self, incoming_layers, num_classes, in_channels = 3, backpropgate=False, batch_norm=False):
    super(BaseModel, self).__init__()
    self.num_classes = num_classes
    self.classifier = nn.LazyLinear(num_classes)
    self.backpropgate = backpropgate
    self.batch_norm = batch_norm
    self.in_channels = in_channels
    self.incoming_layers = incoming_layers
    if len(incoming_layers) == 0:
      raise ValueError("Error cannot be initalized with size 0")
    else:
      if isinstance(incoming_layers[0], int):
        self.incoming_layers = self.make_layers(incoming_layers, batch_norm)
    self.frozen_layers =  nn.ModuleList([])
    self.current_layers = nn.ModuleList([])
    self.current_features = nn.Sequential(*self.current_layers)
    
  def make_layers(self, cfg, batch_norm=False):
    layers = []
    channels = self.in_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(channels, v, kernel_size=3, padding=1)
            if self.batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            channels = v
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.ModuleList(layers)

    def forward(self, x):
      raise NotImplementedError("You should implement this!")

