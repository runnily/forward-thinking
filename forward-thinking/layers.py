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
  nn.Conv2d(IN_CHANNELS, 66, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(66, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 2nd layer
  nn.Conv2d(66, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 3rd layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 4th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 5th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout(0.2),

  # 6th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 7th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 8th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 9th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 10th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
  nn.Dropout(0.2),

  # 11th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 12th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),

  # 13th layer 
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
  nn.ReLU(inplace=True),
  nn.Dropout(0.2),


])

            