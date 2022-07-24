import torch.nn as nn
from sklearn.linear_model import OrthogonalMatchingPursuit
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import numpy as np


def MNIST(batch_size=128):
  transform = Compose([
    ToTensor(), Normalize((0.5,), (0.5,))])

  train_data = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
  return train_data

class OrthogonalMatchingPursuitBasedModel(nn.Module):

  def __init__(self, dataset=MNIST, inputs_size=784, mapping_size=2000):
    super.__init__()
    self.dataset = dataset
    nsamples, nx, ny = dataset.shape
    self.data = dataset.data.reshape((nsamples,nx*ny))
    self.target = dataset.targets.numpy()
    self.f1n = nn.linear(inputs_size, mapping_size)
    self.reg = OrthogonalMatchingPursuit(normalize=False)
    self.mapping = []

  def forward(self, x):
    return self.reg.predict(x)

  def train_model(self):
    # self.mapping is a transformation from input_image_units -> linear_model_output
    self.mapping = np.array([self.f1n(image.float()).detach().numpy() for image in self.data])
    self.reg.fit(self.mapping, self.target)
  
  def get_accuracy():
    pass


  