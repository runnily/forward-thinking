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

    def maximum(self, y):
      softmax_out = F.log_softmax(y)
      if softmax_out[1] >= softmax_out[0]:
        return softmax_out[1]
      else:
        return softmax_out[0]

    def models_forward(self, x):
      outputs = []
      for model in self.models_and_data:
        outputs.append(self.maximum(model(x)))
      return torch.tensor(outputs)

    def createModels(self, batch_size):
      """
        createModels:
          Will create a model for every targets
      """
      targets = set(self.dataset.targets)
      for target in targets:
        selected_target_idx = torch.tensor(self.dataset.targets) == target # including a random selection
        data = Subset(self.dataset, selected_target_idx)
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
            
    def test_model():
      pass

