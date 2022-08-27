"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

try:
    from .base import BaseModel, conv_2
except:
    from base import BaseModel, conv_2

class BasicBlock(BaseModel):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct

    def __init__(self, in_channels, out_channels, batch_norm, stride=2):
        super().__init__()

        #residual function f0, f1
        self.f0 = nn.Sequential(*conv_2(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False, 
            batch_norm=batch_norm
          ),
          nn.ReLU(inplace=True)
        )

        self.classifer = nn.Sequential(*conv_2(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False, 
            batch_norm=batch_norm
          )
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1:
            self.shortcut = nn.Sequential(*conv_2(
              in_channels, 
              out_channels * BasicBlock.expansion, 
              kernel_size=1, 
              stride=stride, 
              padding=0, 
              bias=False, 
              batch_norm=batch_norm
            ))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.classifer(self.current_layers(x)) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """

    def __init__(self, in_channels, out_channels, batch_norm, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
          *conv_2(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=1, 
            bias=False, 
            batch_norm=batch_norm
          ),
          nn.ReLU(inplace=True),
          *conv_2(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=1, 
            bias=False, 
            batch_norm=batch_norm
          ),
          nn.ReLU(inplace=True),
        )

        self.classifer = nn.Sequential(*conv_2(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=1, 
            bias=False, 
            batch_norm=batch_norm
          )
        )

        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Sequential(*conv_2(
              in_channels, 
              out_channels * BasicBlock.expansion, 
              kernel_size=1, 
              stride=stride, 
              padding=0, 
              bias=False, 
              batch_norm=batch_norm
              )
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.classifer(self.current_layers(x))  + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, batch_norm, num_classes=100):
        super().__init__()

        self.layer_1 = nn.Sequential(
            *conv_2(3, 64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False, 
            batch_norm=batch_norm
            ),
            nn.ReLU(inplace=True)
          )

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        num_features = [64, 64, 128, 256, 512]

        if isinstance(block, BottleNeck) == True:
          num_features = [64, 256, 512, 1024, 2048]
        
        
        self.layer_2 = self._make_layer(block,  num_features[0], num_features[1], num_block[0], batch_norm)
        self.layer_3 = self._make_layer(block,  num_features[1], num_features[2], num_block[1], batch_norm)
        self.layer_4 = self._make_layer(block,  num_features[2], num_features[3], num_block[2], batch_norm)
        self.layer_5 = self._make_layer(block,  num_features[3], num_features[4], num_block[3], batch_norm)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, batch_norm):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = [block(in_channels, out_channels, batch_norm, stride=1)]
        for i in range(1, num_blocks):
            layers.append(block(in_channels, out_channels, batch_norm, stride=2))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.layer_1(x)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)
        output = self.layer_5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.classifer(output)
        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])