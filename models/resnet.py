"""
resnet using forward-thinking
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
[2] This model is inspired from https://github.com/runnily/forward-thinking/blob/main/models/resnet.py
    and https://gist.github.com/liao2000/09fa73d6ee01bed5b1803e0ccda81f6c and is simply built to allow
    the implementation of the forward-thinking algorthium to train a resnet neural network.
"""

import torch.nn as nn

try:
    from .base import BaseModel, conv_2d
except ImportError:
    from base import BaseModel, conv_2d


class BasicBlock(BaseModel):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct

    def __init__(self, in_channels, out_channels, batch_norm, stride=2, init_weights=True):
        # residual function f0, f1
        f0 = nn.Sequential(
            *conv_2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                batch_norm=batch_norm,
            ),
            nn.ReLU(inplace=True),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__(f0, out_channels, batch_norm, in_channels, init_weights)

        self.output = nn.Sequential(
            *conv_2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                batch_norm=batch_norm,
            )
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1:
            self.shortcut = nn.Sequential(
                *conv_2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    batch_norm=batch_norm,
                )
            )
        self.current_layers = nn.Sequential(*self.incoming_layers)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.output(self.current_layers(x)) + self.shortcut(x))


class BottleNeck(BaseModel):
    """
    Residual block for resnet over 50 layers
    """

    def __init__(self, in_channels, out_channels, batch_norm, stride=1, init_weights=True):
        out_channels_f0 = out_channels // 4
        f0 = nn.Sequential(
            *conv_2d(
                in_channels,
                out_channels_f0,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                batch_norm=batch_norm,
            ),
            nn.ReLU(inplace=True),
            *conv_2d(
                out_channels_f0,
                out_channels_f0,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                batch_norm=batch_norm,
            ),
            nn.ReLU(inplace=True),
        )

        super().__init__(f0, out_channels, batch_norm, in_channels, init_weights)

        self.output = nn.Sequential(
            *conv_2d(
                out_channels_f0,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                batch_norm=batch_norm,
            )
        )

        self.shortcut = nn.Sequential()  # essentially an identify function

        if stride != 1:
            self.shortcut = nn.Sequential(
                *conv_2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=True,
                    batch_norm=batch_norm,
                )
            )

        self.current_layers = nn.Sequential(*self.incoming_layers)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.output(self.current_layers(x)) + self.shortcut(x))


class ResNet(BaseModel):
    def __init__(self, block, num_block, batch_norm, num_classes=100, init_weights=True):

        layer_1 = nn.Sequential(
            *conv_2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True, batch_norm=batch_norm),
            nn.ReLU(inplace=True),
        )

        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        num_features = [64, 64, 128, 256, 512]

        if isinstance(block, BottleNeck) is True:
            num_features = [64, 256, 512, 1024, 2048]

        layer_2 = self._make_layer(
            block,
            num_features[0],
            num_features[1],
            num_block[0],
            batch_norm,
            init_weights,
            stride=1,
        )
        layer_3 = self._make_layer(
            block,
            num_features[1],
            num_features[2],
            num_block[1],
            batch_norm,
            init_weights,
            stride=2,
        )
        layer_4 = self._make_layer(
            block,
            num_features[2],
            num_features[3],
            num_block[2],
            batch_norm,
            init_weights,
            stride=2,
        )
        layer_5 = self._make_layer(
            block,
            num_features[3],
            num_features[4],
            num_block[3],
            batch_norm,
            init_weights,
            stride=2,
        )

        super(ResNet, self).__init__(
            nn.Sequential(layer_1, *layer_2, *layer_3, *layer_4, *layer_5),
            num_classes,
            batch_norm,
            3,
            init_weights,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.LazyLinear(num_features[4], num_classes)

    def _make_layer(
        self, block, in_channels, out_channels, num_blocks, batch_norm, init_weights, stride
    ):
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
        inner_stride = 1 if (stride == 2) else 2
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = nn.Sequential(
            block(in_channels, out_channels, batch_norm, stride=stride, init_weights=init_weights)
        )
        for i in range(1, num_blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    batch_norm,
                    stride=inner_stride,
                    init_weights=init_weights,
                )
            )

        return layers

    def forward(self, x):
        output = self.current_layers(x)
        output = output.view(output.size(0), -1)
        output = self.output(output)
        return output


def resnet18(batch_norm, num_classes=100, init_weights=True):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], batch_norm, num_classes=100, init_weights=True)


def resnet34(batch_norm, num_classes=100, init_weights=True):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], batch_norm, num_classes=100, init_weights=True)


def resnet50(batch_norm, num_classes=100, init_weights=True):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], batch_norm, num_classes=100, init_weights=True)


def resnet101(batch_norm, num_classes=100, init_weights=True):
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152(batch_norm, num_classes=100, init_weights=True):
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3], batch_norm, num_classes=100, init_weights=True)
