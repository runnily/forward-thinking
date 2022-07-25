import torch.nn as nn
import torch
from sklearn.linear_model import OrthogonalMatchingPursuit
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import torchvision
import numpy as np

def MNIST():
  transform = Compose([
    ToTensor(), Normalize((0.5,), (0.5,)), Resize(28)])
  train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
  test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
  return train_data, test_data

class OrthogonalMatchingPursuitBasedModel(nn.Module):

  def __init__(self, dataset=MNIST(), inputs_size=28*28, mapping_size=200):
    super(OrthogonalMatchingPursuitBasedModel, self).__init__()
    self.dataset, self.test_dataset = dataset
    nsamples, nx, ny = self.dataset.data.shape
    self.data = self.dataset.data.reshape((nsamples,nx*ny))
    self.targets = nn.functional.one_hot(self.dataset.targets).numpy()
    self.transform = nn.Linear(inputs_size, mapping_size)
    self.reg = OrthogonalMatchingPursuit(normalize=True, n_nonzero_coefs=mapping_size)
    self.mapping = None
    self.fit = None

  def forward(self, x):
    if not self.mapping:
      self.train_model()
    mapping = self.transform(x)
    return self.reg.predict(mapping)

  def train_model(self):
    # self.mapping is a transformation from input_image_units -> linear_model_output
    self.mapping = np.array([self.transform(image.float()).detach().numpy() for image in self.data])
    self.fit = self.reg.fit(self.mapping, self.targets)
    print(len(self.reg.coef_))
  
  def get_score(self):
    if not self.fit:
      self.train_model()
    return self.fit.score(self.mapping, self.targets)

  def get_accuracy(self):
    if not self.fit:
      self.train_model()
    preds = self.reg.predict(self.mapping)
    torch.argmax(preds, dim=1)
    return preds

  

if __name__ == "__main__":
  model = OrthogonalMatchingPursuitBasedModel()
  print(model.get_accuracy())


  