import torch.nn as nn
from .paras import in_channels, input_size

basic_net = nn.ModuleList([
    nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
	  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),]
)


simple_net = nn.ModuleList([

  # 1st layer
  nn.Conv2d(in_channels, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),

  # 2nd layer
  nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 3rd layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 4th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 5th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 6th layer
  nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),

  # 7th layer
  nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),

  # 8th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),

  # 9th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),

  # 10th layer
  nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),

  # 11th layer
  nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
  nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),

  # 12th layer
  nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
 
  # 13th layer
  nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
  nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),


])

dense_net = additional_layers = nn.ModuleList([
            nn.Linear(input_size, 150),
            nn.Linear(150, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 10),])
            