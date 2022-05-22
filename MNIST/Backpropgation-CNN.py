import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F

BATCH_SIZE = 64

# Getting Data
train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=ToTensor())
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=ToTensor())

train_loader = torch.utils.DataLoader(train_data, batch_size=BATCH_SIZE)
test_loader = torch.utils.DataLoader(test_data, batch_size=BATCH_SIZE)


class CNN(nn.Module):

  def __init__(self, in_features, out_features):
    super(DNN, self).__init__()

    self.c1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=8, stride=1)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.c1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=1)
    self.fn1 = nn.Linear(16*7*7, num_features)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)

    return x

