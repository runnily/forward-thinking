import torch.nn as nn
from paras import IN_CHANNELS

simple_net = [ 
    nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),


    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
    nn.Dropout2d(p=0.1),


    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),



    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
    nn.Dropout2d(p=0.1),


    nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),


    nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),



    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
    nn.Dropout2d(p=0.1),



    nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),



    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
    nn.Dropout2d(p=0.1),


    nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
    nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),



    nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),


    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
    nn.Dropout2d(p=0.1),


    nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
    nn.ReLU(inplace=True),
  ]

            