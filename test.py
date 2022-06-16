from torch.functional import split
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# preprocessor
def train_data_for_layers(train_data, split, batch_size=32, groups={0 : None, 1: None, 
                                          2: None, 3: None, 4: None, 5:
                                          None, 6 : None, 7 : None, 8 : None, 9: None}):
  targets = np.array(train_data.targets)
  data = train_data.data
  target_set = set(targets)  
  for i in range(len(split)-1): # go through split array to find where to split the data
    for group in groups: # go through each group defined
      a = np.array([])
      for target in target_set: 
        # Here goes through each target and to distrubute different classes in the 
        # training group dataset
        group_targets_idx = targets==target
        training_group = copy.deepcopy(train_data)
        training_group.targets = targets[group_targets_idx]
        training_group.data = data[group_targets_idx]
        training_group = Subset(training_group, [split[i],split[i+1]])
        a.append(training_group)
    groups[group] = DataLoader(training_group, batch_size, shuffle=True)
  return groups
        
transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)

groups = train_data_for_layers(train_data, [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
print(len(groups))
for group in groups:
  print(len(groups[group]["train"]))