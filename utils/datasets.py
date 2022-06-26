from torch.functional import split
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision
import torch
import copy
from torch.utils.data import DataLoader, Subset, ConcatDataset


def CIFAR_100(batch_size=128):
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform)

  return loaders(train_data, test_data, batch_size)

def CIFAR_10(batch_size=128):
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

  return loaders(train_data, test_data, batch_size)

def MNIST(batch_size=128):
  transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

  train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

  return loaders(train_data, test_data, batch_size)

def SVHN(batch_size=128):
  """Builds and returns Dataloader for MNIST and SVHN dataset."""

  transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_data = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform)
  test_data = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform)
  return loaders(train_data, test_data, batch_size)

def divide_data_by_group(dataset, num_data_per_group, batch_size=32, groups={0 : [], 1: [], 2:[], 
                                          3: [], 4: [], 5: [], 6 : [], 7 : [], 8 : [], 9: []}):
 
  targets = set(dataset.targets)
  selected_indices = 0
  for target in targets:
    selected_target_idx = torch.tensor(dataset.targets) == target
    selected_target_idx = selected_target_idx.nonzero().reshape(-1)
    for group in groups:
      group_target_idx = selected_target_idx[selected_indices:selected_indices+num_data_per_group]
      selected_indices += num_data_per_group
      groups[group] += group_target_idx
    selected_indices = 0

  groups_data_loader = {}
  for group in groups:
    data = Subset(dataset, groups[group])
    groups_data_loader[group] = DataLoader(data, batch_size, shuffle=True)
  return groups_data_loader
        

def loaders(train_data, test_data, batch_size=128):
  train_loader = DataLoader(train_data, batch_size=batch_size,num_workers=2, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size,num_workers=2, shuffle=True)
  return train_loader, test_loader, train_data, test_data
