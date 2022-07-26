from scipy.sparse.extract import tril
import torch.nn as nn
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit
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
  pass

class EnsembleBasedModel(nn.Module):
    def __init__(self, dataset=CIFAR100(), num_class=100,batch_size=32,epochs=2):
      self.dataset, self.test_dataset = dataset
      self.num_classes = 100
      self.createModels(batch_size)
      self.models_and_data = {}
      self.epochs = epochs

    def forward(self, x):
      return self.models_forward(x)

    def sop(self, y):
      softmax_out = F.softmax(y)
      if softmax_out.shape(0) > 1:
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
      if output_tensor.shape(0) > 1:
        x, y = output_tensor.shape
        return output_tensor.reshape(y,x)
      return output_tensor

    def _getSelectedIndicies(self, target):
      selected_target_idx = (torch.tensor(self.dataset.targets) == target).nonzero().reshape(-1)
      selected_target_idx_ops = (torch.tensor(self.dataset.targets) != target).nonzero().reshape(-1)[torch.randperm(500)]
      return torch.cat(selected_target_idx , selected_target_idx_ops)

    def createModels(self, batch_size):
      """
        createModels:
          Will create a model for every targets
      """
      targets = set(self.dataset.targets)
      for target in targets:
        selected_target_idx = self._getSelectedIndicies(self, target) # including a random selection
        data = Subset(deepcopy(self.dataset), selected_target_idx)
        labels = data.dataset.targets
        labels[labels != target] = 0
        labels[labels == target] = 1
        data.dataset.targets = labels
        model = MiniModel()
        self.models_and_data[model] = DataLoader(data, batch_size=batch_size, shuffle=True)

    def train_models(self):
      for i, model in enumerate(self.models_and_data):
        batch_data = self.models_and_data[self.models]
        self.train_a_model(model, batch_data) # need to train model with full train_data? not just one

    def train_a_model(self, model, batch_data):
      model.train()
      criterion = nn.MSELoss()
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

        print("Epoch {}, Loss {}".format(epoch, loss))
            
    def test_model(self):
      self.eval()
      with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in self.test_loader:
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)
          outputs = self.forward(images)

          # max returns (value, maximum index value)
          """_, predicted = torch.max(outputs.data, 1)
          n_samples += labels.size(0)  # number of samples in current batch
          n_correct += (
              (predicted == labels).sum().item()
          )  # gets the number of correct"""

      accuracy = n_correct / n_samples
      return accuracy

if __name__ == "__main__":
  ensemble = EnsembleBasedModel()
  ensemble.test_model()
