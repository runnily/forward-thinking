import torch.nn as nn
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset, dataloader, Dataset
import torch.optim as optim
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinarySubsetDataset(Dataset):

  def __init__(self, dataset, indices, target):
    self.dataset = dataset
    self.indices = indices
    self.target = target

  def __getitem__(self, idx):
    if isinstance(idx, list):
      return [self.__expand(self.dataset[self.indices[i]]) for i in idx]
    return self.__expand(self.dataset[self.indices[idx]])
  
  def __expand(self, item):
    if item[1] != self.target:
      return (item[0], 0)
    else:
      return (item[0], 1)
    
  def __len__(self):
    return len(self.indices)


def CIFAR100():
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
  return train_data, test_data

class MiniBinaryModel(nn.Module):
  def __init__(self, init_weights : bool = True):
    super(MiniBinaryModel, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)), 
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)), 
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=(2,2),stride=2),
    )
    
    self.classifier = nn.Linear(25088, 2)

    if init_weights:
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
          if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
          nn.init.normal_(m.weight, 0, 0.01)
          nn.init.constant_(m.bias, 0)
    
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

class EnsembleBasedModel(nn.Module):
    def __init__(self, dataset=CIFAR100(), num_classes=100,batch_size=64,epochs=10):
      super(EnsembleBasedModel, self).__init__()
      self.dataset, self.test_dataset = dataset
      self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
      self.num_classes = num_classes
      self.models_and_data = {}
      self.createModels(batch_size)
      self.epochs = epochs

    def forward(self, x):
      return self.ensemble(x)

    def configOutput(self, y):
      softmax_out = F.softmax(y, 1)
      if softmax_out.size(0) > 1:
        return [self.maximum(x) for x in softmax_out]
      return self.maximum(softmax_out)

    def maximum(self,y):
      if y[1] >= y[0]:
        return y[1]
      return 0

    def ensemble(self, x):
      outputs = []
      for model in self.models_and_data:
        outputs.append(self.configOutput(model(x))) 
      output_tensor = torch.tensor(outputs)
      if output_tensor.size(0) > 1:
        x, y = output_tensor.shape
        return output_tensor.reshape(y,x)
      return output_tensor

    def _getSelectedIndicies(self, target):
      selected_target_idx = (torch.tensor(self.dataset.targets) == target).nonzero().reshape(-1)
      selected_target_idx_ops = (torch.tensor(self.dataset.targets) != 5).nonzero().reshape(-1)[torch.randperm(len(selected_target_idx))]
      return torch.cat((selected_target_idx,selected_target_idx_ops ))

    def createModels(self, batch_size):
      """
        createModels:
          Will create a model for every targets
      """
      for _, target in self.dataset.class_to_idx.items(): # should be the same order of test
        data = BinarySubsetDataset(self.dataset, self._getSelectedIndicies(target), target)
        model = MiniBinaryModel().to(DEVICE)
        self.models_and_data[model] = DataLoader(data, batch_size=batch_size, shuffle=True)

    def train_model(self):
      for i, model in enumerate(self.models_and_data):
        batch_data = self.models_and_data[model]
        self.train_inner_models(model, batch_data) # need to train model with full train_data? not just one

    def train_inner_models(self, model, batch_data):
      model.train()
      criterion = nn.CrossEntropyLoss().to(DEVICE)
      optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

      for epoch in range(self.epochs):
        for images, labels in batch_data:
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)

          # zero the parameter gradients
          optimizer.zero_grad()   

          # forward + backward + optimize
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          
          #torch.Size([32, 2]) | outputs.shape
          #torch.Size([32]) | labels.shape
          optimizer.step()

        print("Epoch {}, Loss {}, Train accuracy {}".format(epoch, loss.item(), self.test_model(model)))
      model.eval()
            
    def test_model(self, model=None):
      model, test_loader = (model, self.models_and_data[model]) if (model) else (self, self.test_loader)
      model.eval()
      with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, (images, labels) in enumerate(test_loader):
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)
          outputs = self.forward(images)
          # max returns (value, maximum index value)
          _, predicted = torch.max(outputs.data, 1) 
          n_samples += labels.size(0)  # number of samples in current batch
          n_correct += (
              (predicted.to(DEVICE) == labels).sum().item()
          )  # gets the number of correct
          
      accuracy = n_correct / n_samples
      return accuracy

if __name__ == "__main__":
  ensemble = EnsembleBasedModel().to(DEVICE)
  ensemble.train_model()
  print(ensemble.test_model())
