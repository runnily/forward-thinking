from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision.transforms import Compose, Normalize, ToTensor


def get_dataset(name: str, batch_size: int) -> Optional[Tuple[DataLoader, DataLoader]]:
    train_data: Dataset
    test_data: Dataset
    transform = get_transform()

    if name in {"CIFAR100", "CIFAR10", "MNIST"}:
        if name == "MNIST":
            transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

        train_data = globals()[name]("./data", train=True, download=True, transform=transform)
        test_data = globals()[name]("./data", train=False, download=True, transform=transform)

    elif name == "SVHN":
        train_data = globals()[name](
            root="./data", split="train", download=True, transform=transform
        )
        test_data = globals()[name](root="./data", split="test", download=True, transform=transform)

    else:
        return None

    return loaders(train_data, test_data, batch_size)


def get_transform():
    return Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def divide_data_by_group(
    dataset: int, num_data_per_group: int, batch_size: int, groups: Dict[nn.Module, List]
) -> Optional[Dict[nn.Module, DataLoader]]:
    targets = set(dataset.targets)
    selected_indices = 0
    for target in targets:
        selected_target_idx = torch.tensor(dataset.targets) == target
        selected_target_idx = selected_target_idx.nonzero().reshape(-1)
        for group in groups:
            group_target_idx = selected_target_idx[
                selected_indices : selected_indices + num_data_per_group
            ]
            selected_indices += num_data_per_group
            groups[group] += group_target_idx
        selected_indices = 0
    groups_data_loader = {}
    for group in groups:
        if len(groups[group]) > 0:
            data = Subset(dataset, groups[group])
            groups_data_loader[group] = DataLoader(data, batch_size=batch_size, shuffle=True)
    return groups_data_loader


def loaders(train_data, test_data, batch_size=128):
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2, shuffle=True)
    return train_loader, test_loader
