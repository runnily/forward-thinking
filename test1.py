import torch.nn as nn
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset, dataloader, Dataset # remove subset later
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debugging():
  transform = Compose(
      [ToTensor(),
      Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  test = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
  sub_test = Subset(test, (torch.tensor(test.targets) == 2).nonzero().reshape(-1))
  return DataLoader(sub_test, batch_size=64, shuffle=True)
  
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
      # return input, outputs
      return (item[0].float(), 0)
    else:
      return (item[0].float(), 1)
    
  def __len__(self):
    return len(self.indices)

def CIFAR100():
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
  return train_data, test_data

class BinaryBaseModel(nn.Module):
  def __init__(self, init_weights : bool = True, num_classes : int = 1):
    super(BinaryBaseModel, self).__init__()
    self.num_classes = num_classes
    self.features = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)), 
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)), 
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=(2,2),stride=2),
    )
    self.classifier = nn.Linear(25088, num_classes)

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

    def forward(self):
      # TODO : implement
      pass

class OneTargetModel(BinaryBaseModel):
  def __init__(self):
    super(OneTargetModel, self).__init__(num_classes = 1)
    
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    return torch.sigmoid(self.classifier(x))

class TwoTargetModel(BinaryBaseModel):
  def __init__(self):
    super(TwoTargetModel, self).__init__(num_classes = 2)
    
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

class EnsembleBasedModel(nn.Module):
    def __init__(self, dataset=CIFAR100(), num_classes=100,batch_size=64,epochs=10):
      super(EnsembleBasedModel, self).__init__()
      self.dataset, self.test_dataset = dataset
      self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
      self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
      self.num_classes = num_classes
      self.models_and_data = {}
      self.createModels(batch_size)
      self.epochs = epochs
      self._training_model = TrainAndTest(self.to(DEVICE), self.test_loader, epochs)

    def forward(self, x):
      return self.ensemble(x)

    def configOutput(self, y):
      softmax_out = F.softmax(y, 1)
      if softmax_out.size(0) > 1:
        return [self.maximum(x) for x in softmax_out]
      return self.maximum(softmax_out)

    def maximum(self,y):
      print(y)
      if y.size(0) > 1:
        if y[1] >= y[0]:
          return y[1]
      elif y > 0.65:
        return y
      return 0
      
    def ensemble(self, x):
      outputs = []
      for model in self.models_and_data:
        outputs.append(self.configOutput(model(x))) 
      output_tensor = torch.tensor(outputs)
      if output_tensor.size(0) > 1:
        x, y = output_tensor.shape
        return output_tensor.reshape(y,x)
      return output_tensor.to(DEVICE)

    def _getSelectedIndicies(self, dataset, target):
      selected_target_idx = (torch.tensor(dataset.targets) == target).nonzero().reshape(-1)
      selected_target_idx_ops = (torch.tensor(dataset.targets) != 5).nonzero().reshape(-1)[torch.randperm(len(selected_target_idx))]
      return torch.cat((selected_target_idx,selected_target_idx_ops ))

    def createModels(self, batch_size):
      """
        createModels:
          Will create a model for every targets
      """
      for _, target in self.dataset.class_to_idx.items(): # should be the same order of test
        train_data = BinarySubsetDataset(self.dataset, self._getSelectedIndicies(self.dataset, target), target)
        test_data = BinarySubsetDataset(self.test_dataset, self._getSelectedIndicies(self.test_dataset, target), target)
        model = OneTargetModel().to(DEVICE)
        self.models_and_data[model] = (
          DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True), 
          DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=True)
          )

    def train_model(self):
      #for epochs in range(self.epochs):
      for i, model in enumerate(self.models_and_data):
        data = self.models_and_data[model]
        self._training_model.train(model, data[1], epochs=2)
      
      #self.train_loader

    def test_model(self):
      return self._training_model.test()

class TrainAndTest():
  def __init__(self, model, loader, epochs):
    self.model = model
    self.loader = loader
    self.epochs = epochs

  def train(self, model=None, train_loader=None, epochs=None):
    model, train_loader, epochs = ( model, train_loader, epochs) if (model and train_loader and epochs) else (self.model, self.loader, self.epochs)
    n_total_steps = len(train_loader)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = None
    if model.num_classes > 1:
      criterion = nn.CrossEntropyLoss().to(DEVICE)
    else:
      criterion = nn.BCELoss().to(DEVICE)
    for epoch in range(epochs): 
      model.train()
      for i, (images, labels) in enumerate(
        train_loader, 0):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE) if (model.num_classes > 1) else labels.unsqueeze(1).to(DEVICE).to(torch.float32)
        optimizer.zero_grad()  
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  
        optimizer.step()  
        if (i + 1) % 100 == 0:
          print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(
            epoch + 1, epochs, i + 1, n_total_steps, loss.item()
          )
        )    
      print("Test accuracy {}, at epoch: {}".format(self.test(model, train_loader), epoch))

  def test(self, model=None, test_loader=None):
    model, test_loader = (model, test_loader) if (model and test_loader) else (self.model, debugging())
    model.eval()
    with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        #print("here")
        #print(labels) 
        _, predicted = torch.max(outputs.data, 1)
        if model == self:
          print(labels)
          print(outputs)
        #print(predicted)
        n_samples += labels.size(0) 
        n_correct += (
            (predicted.to(DEVICE) == labels).sum().item()
        )  
        #print(n_correct)
        #print(n_samples)
    accuracy = n_correct / n_samples
    return accuracy

if __name__ == "__main__":
  ensemble = EnsembleBasedModel().to(DEVICE)
  ensemble.train_model()
  print(ensemble.test_model())
