from paras import BATCH_SIZE
from torchvision.transforms import ToTensor
import torchvision
import torch
from torch.utils.data import DataLoader


def CIFAR_100():
  train_data = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=ToTensor())
  test_data = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=ToTensor())

  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
  return train_loader, test_loader

def MNIST():
  train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=ToTensor())
  test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=ToTensor())

  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
  return train_loader, test_loader