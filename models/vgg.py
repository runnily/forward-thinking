"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn

try:
    from .base import BaseModel, conv_2d_relu
except ImportError:
    from base import BaseModel, conv_2d_relu

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(BaseModel):

    def __init__(self, features, num_classes, batch_norm, init_weights):

        in_channels = 3
        
        super().__init__(features, num_classes, batch_norm, in_channels, init_weights)

        self.output = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        output = self.current_layers(x)
        output = output.view(output.size()[0], -1)
        output = self.output(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for layer in cfg:
        seq_layer = nn.Sequential()
        if layer == 'M':
            seq_layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        seq_layer.append(nn.Conv2d(input_channel, layer, kernel_size=3, padding=1))

        if batch_norm:
            seq_layer.append(nn.BatchNorm2d(layer))

        seq_layer.append(nn.ReLU(inplace=True))
        layers.append(seq_layer)
        input_channel = layer

    return nn.Sequential(*layers)

def vgg11(batch_norm, num_classes=100, init_weights=True):
    return VGG(make_layers(cfg['A'], batch_norm),  num_classes, batch_norm, init_weights)

def vgg13(batch_norm, num_classes=100, init_weights=True):
    return VGG(make_layers(cfg['B'], batch_norm),  num_classes, batch_norm, init_weights)

def vgg16(batch_norm, num_classes=100, init_weights=True):
    return VGG(make_layers(cfg['D'], batch_norm),  num_classes, batch_norm, init_weights)

def vgg19(batch_norm, num_classes=100, init_weights=True):
    return VGG(make_layers(cfg['E'], batch_norm),  num_classes, batch_norm, init_weights)
    