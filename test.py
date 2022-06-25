from torch.functional import split
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import copy

# preprocessor
def divide_data_by_group(dataset, num_data_per_group, batch_size=32, groups={0 : [], 1: [], 2:[], 
                                          3: [], 4: [], 5: [], 6 : [], 7 : [], 8 : [], 9: []}):
  """
    Used to put data into different groups.
  """
  targets = set(dataset.targets)
  selected_indices = 0
  for target in targets:
    selected_target_idx = dataset.targets == target
    selected_target_data = copy.deepcopy(dataset)
    selected_target_data.targets = selected_target_data.targets[selected_target_idx]
    selected_target_data.data = selected_target_data.data[selected_target_idx]
    for group in groups:
      subset = Subset(selected_target_data, list(range(selected_indices, selected_indices+num_data_per_group, 1) ))
      selected_indices += num_data_per_group
      groups[group].append(subset)
    selected_indices = 0
  
  groups_data_loader = {}
  for group in groups:
    data = ConcatDataset(groups[group])
    groups_data_loader[group] = DataLoader(data, batch_size, shuffle=True)

  return groups_data_loader 
        
transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
print("here")
print(len(train_data.targets)/len(set(train_data.targets))/10)

groups = divide_data_by_group(train_data, 500)
print(len(groups))
for group in groups:
  print("number of data per group")
  num = len(groups[group])
  print(num)


