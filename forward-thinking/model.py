import torch.nn as nn
from paras import IN_CHANNELS

basic = [ 
    nn.Conv2d(in_channels=IN_CHANNELS, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.ReLU(),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.ReLU(),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.ReLU(),
	  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.ReLU(),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.ReLU(),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
		nn.ReLU(),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
	
  ]

model_m5 = [
  nn.Conv2d(1, 32, 5, bias=False),
  nn.BatchNorm2d(32),
  nn.ReLU(),
  nn.Conv2d(32, 64, 5, bias=False),
  nn.BatchNorm2d(64),
  nn.ReLU(),
  nn.Conv2d(64, 96, 5, bias=False),
  nn.BatchNorm2d(96),
  nn.ReLU(),
  nn.Conv2d(96, 128, 5, bias=False),
  nn.BatchNorm2d(128),
  nn.ReLU(),
  nn.Conv2d(128, 160, 5, bias=False),
  nn.BatchNorm2d(160),
  nn.ReLU(),
  nn.Linear(10240, 10, bias=False),
]

simple_net = [
  nn.Conv2d(IN_CHANNELS, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
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

            