import torch.nn as nn
import torch
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Subset, dataloader
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CIFAR100():
  transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_data = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform)
  return train_data, test_data

class MiniModel(nn.Module):
  def __init__(self, init_weights : bool = True):
    super(MiniModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(720, 1024)
    self.fc2 = nn.Linear(1024, 2)

    if init_weights:
      nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
      nn.init.constant_(self.conv1.weight, 0)
      nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
      nn.init.constant_(self.conv2.weight, 0)
      nn.init.normal_(self.fc1.weight, 0, 0.01)
      nn.init.constant_(self.fc1.bias, 0)
      nn.init.normal_(self.fc2.weight, 0, 0.01)
      nn.init.constant_(self.fc2.bias, 0)
    
  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(x.shape[0],-1)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return x


class EnsembleBasedModel(nn.Module):
    def __init__(self, dataset=CIFAR100(), num_class=100,batch_size=32,epochs=2):
      super(EnsembleBasedModel, self).__init__()
      self.dataset, self.test_dataset = dataset
      self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
      self.num_classes = 100
      self.models_and_data = {}
      self.createModels(batch_size)
      self.epochs = epochs

    def forward(self, x):
      out = self.models_forward(x)
      print(out.shape)
      return out

    def sop(self, y):
      softmax_out = F.softmax(y, 1)
      if softmax_out.size(0) > 1:
        return [self.maximum(x) for x in softmax_out]
      return self.maximum(softmax_out)

    def maximum(self,y):
      if y[1] >= y[0]:
        return y[1]
      return 0

    def models_forward(self, x):
      outputs = []
      for model in self.models_and_data:
        outputs.append(self.sop(model(x)))
      output_tensor = torch.tensor(outputs)
      if output_tensor.size(0) > 1:
        x, y = output_tensor.shape
        return output_tensor.reshape(y,x)
      return output_tensor

    def _getSelectedIndicies(self, target):
      selected_target_idx = (torch.tensor(self.dataset.targets) == target).nonzero().reshape(-1)
      selected_target_idx_ops = (torch.tensor(self.dataset.targets) != target).nonzero().reshape(-1)[torch.randperm(500)]
      return torch.cat((selected_target_idx , selected_target_idx_ops))

    def createModels(self, batch_size):
      """
        createModels:
          Will create a model for every targets
      """
      targets = set(self.dataset.targets)
      for target in targets:
        data = Subset(deepcopy(self.dataset), self._getSelectedIndicies(target))
        labels = data.dataset.targets
        labels[labels != target] = 0
        labels[labels == target] = 1
        data.dataset.targets = labels
        model = MiniModel().to(DEVICE)
        self.models_and_data[model] = DataLoader(data, batch_size=batch_size, shuffle=True)

    def train_model(self):
      for i, model in enumerate(self.models_and_data):
        batch_data = self.models_and_data[model]
        self.train_inner_models(model, batch_data) # need to train model with full train_data? not just one

    def train_inner_models(self, model, batch_data):
      model.train()
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

      for epoch in range(self.epochs):
        for i, (images, labels) in enumerate(batch_data):
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)

          # zero the parameter gradients
          optimizer.zero_grad()   

          # forward + backward + optimize
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

        print("Epoch {}, Loss {}".format(epoch, loss.item()))
      model.eval()
            
    def test_model(self):
      with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, (images, labels) in enumerate(self.test_loader):
          if i == 2:
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
