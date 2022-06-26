from torch.functional import split
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import copy

# preprocessor
"""
def divide_data_by_group(dataset, num_data_per_group, batch_size=32, groups={0 : [], 1: [], 2:[], 
                                          3: [], 4: [], 5: [], 6 : [], 7 : [], 8 : [], 9: []}):
  
  targets = set(dataset.targets)
  selected_indices = 0
  for target in targets:
    selected_target_idx = dataset.targets == target
    selected_target_data = copy.deepcopy(dataset)
    y = np.array(selected_target_data.targets)
    selected_target_data.targets = selected_target_data.targets[selected_target_idx]
    selected_target_data.data = selected_target_data.data[selected_target_idx]
    for group in groups:
      selected_group = copy.deepcopy(selected_target_data)
      selected_group.data = selected_group.data[selected_indices:selected_indices+num_data_per_group]
      selected_group.targets = y[selected_indices:selected_indices+num_data_per_group].tolist()
      selected_indices += num_data_per_group
      groups[group].append(selected_group)
    selected_indices = 0
  
  groups_data_loader = {}
  for group in groups:
    #data = ConcatDataset(groups[group])
    groups_data_loader[group] = DataLoader(np.array(groups[group]), batch_size, shuffle=True)

  return groups_data_loader"""

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
        
transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
train_loader= DataLoader(train_data, batch_size=32, shuffle=True)

groups = divide_data_by_group(train_data, 500)
print(len(groups))
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print(labels)
        break
"""for group in groups:
  print("number of data per group")
  num = len(groups[group])
  print(num)"""
