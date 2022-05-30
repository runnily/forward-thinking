from .paras import batch_size
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision
import torch
from torch.utils.data import DataLoader


def CIFAR_100():
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform)

  train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=2)
  return train_loader, test_loader

def CIFAR_10():
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

  train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=2)
  return train_loader, test_loader

def MNIST():
  transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

  train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

  train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=2)
  return train_loader, test_loader