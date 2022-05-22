from os import X_OK
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import paras
import layers
from torch import device, cuda
DEVICE = device('cuda' if cuda.is_available()  else 'cpu')


class FTNN(nn.Module):

  def __init__(self, in_channels = paras.IN_CHANNELS, classes=paras.NUM_CLASSES, layers=layers.simple_net, classifer = nn.LazyLinear(paras.NUM_CLASSES)):
      super(FTNN, self).__init__()

      self.classifer = classifer

      self.additional_layers = layers # layers to be added into our model one at a time
      self.layers = nn.ModuleList([])
      self.frozen_layers = nn.ModuleList([])
      self.classes = classes

  def forward(self, x):

    for l in self.layers:
      x = l(x)

    #x = F.max_pool2d(x, kernel_size=x.size()[2:]) # optional global max pool
    #x = F.dropout2d(x, 0.1, training=True) # optional can be removed 

    x = x.reshape(x.shape[0], -1)
    x = self.classifer(x)
    return x

class Train():

  def __init__(self,  train_loader, test_loader, backpropgate = False, model=FTNN().to(DEVICE), lr=paras.LEARNING_RATE, num_epochs=paras.NUM_EPOCHS):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.lr = lr
    self.num_epochs = num_epochs
    self.backpropgate = backpropgate

  
  def train_(self):

    criterion = nn.CrossEntropyLoss()
    specific_params_to_be_optimized = [{'params': self.model.layers[-1].parameters()},
                                {'params': self.model.classifer.parameters()}]
                                
    optimizer = torch.optim.Adam(specific_params_to_be_optimized, lr=self.lr)

    n_total_steps = len(self.train_loader)

    for epoch in range(self.num_epochs):
      for i, (images, labels) in enumerate(self.train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward pass
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backwards and optimize
        optimizer.zero_grad() # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all backward() calls
        loss.backward() #  computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.
        
        if (i+1) % 100 == 0:
           print("Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(epoch+1,self.num_epochs,i+1,n_total_steps,loss.item()))
      #print("Epoch {} and Loss: {}".format(epoch, loss.item()))
        
  def test_(self):
    with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for images, labels in self.test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = self.model(images)

        # max returns (value, maximum index value)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.shape[0] # number of samples in current batch
        n_correct += (predicted == labels).sum().item() # gets the number of correct

    accuracy = (n_correct / n_samples) * 100
    print("Accuracy is now: {}".format(accuracy))

  def freeze_layers_(self):
    if self.backpropgate == False:
      for l in self.model.frozen_layers:
        l.requires_grad_(False)

  def add_layers(self):

    if self.backpropgate == True:
      self.model.layers = self.model.additional_layers
    
    else:
      
      N = len(self.model.additional_layers)

      for i in range(N):
        layer = self.model.additional_layers[i] # incoming new layer
        # 1. Add new layer to model
        self.model.layers.append(layer)
        # 2. diregarded output as output layer is retrained with every new added layer
        self.model.classifer = nn.LazyLinear(out_features=self.model.classes).to(DEVICE)
        # 3. Train
        self.train_()
        # 4. Get Accuracy
        self.test_()
        # 5. As we have trained add layer to the frozen_layers
        self.model.frozen_layers.append(self.model.layers[-1])
        # 6. Freeze layers
        self.freeze_layers_()  
      

    if len(self.model.additional_layers) < 0:
      pass

train_loader, test_loader = datasets.CIFAR_10()
train = Train(train_loader, test_loader)
train.add_layers()

    
