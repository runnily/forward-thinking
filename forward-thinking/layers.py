import torch.nn as nn
from paras import IN_CHANNELS

basic = nn.ModuleList([
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
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),]
)


simple_net = nn.ModuleList([

  # 1st layer
  nn.Conv2d(IN_CHANNELS, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 2nd layer
  nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 3rd layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 4th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout2d(p=0.1),

  # 5th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 6th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 7th layer
  nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout2d(p=0.1),

  # 8th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 9th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout2d(p=0.1),

  # 10th layer
  nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout2d(p=0.1),

  # 11th layer
  nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
  nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),

  # 12th layer
  nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout2d(p=0.1),

  # 13th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),


])

            