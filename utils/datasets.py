from torch.functional import split
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

  return loaders(train_data, test_data)

def CIFAR_10():
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

  return loaders(train_data, test_data)

def MNIST():
  transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

  train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

  return loaders(train_data, test_data)

def SVHN():
  """Builds and returns Dataloader for MNIST and SVHN dataset."""

  transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform)
  test_data = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform)
  return loaders(train_data, test_data)


def loaders(train_data, test_data, batch_size=128):
  train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2)
  test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=2)
  return train_loader, test_loader
