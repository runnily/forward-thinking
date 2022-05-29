import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import paras
import layers
from torch import device, cuda
DEVICE = device('cuda' if cuda.is_available()  else 'cpu')


class FTNN(nn.Module):

  def __init__(self, in_channels = paras.IN_CHANNELS, classes=paras.NUM_CLASSES, layers=layers.simple_net, hidden_layer_features = paras.HIDDEN_SIZE):
      super(FTNN, self).__init__()

      self.h0 = nn.LazyLinear(hidden_layer_features)
      self.classifer = nn.Linear(hidden_layer_features, classes)
      self.hidden_layer_features = hidden_layer_features
      self.additional_layers = layers # layers to be added into our model one at a time
      self.layers = nn.ModuleList([])
      self.frozen_layers = nn.ModuleList([])
      self.classes = classes

  def forward(self, x):

    for l in self.layers:
      x = l(x)
    

    #x = F.max_pool2d(x, kernel_size=x.size()[2:]) 
    #x = F.dropout2d(x, 0.1, training=True)

    x = x.reshape(x.shape[0], -1) # flatten to go into the linear hidden layer

    x = self.h0(x)
    x = self.classifer(x)
    return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Train():

  def __init__(self,  train_loader, test_loader, backpropgate = False, model=FTNN().to(DEVICE), lr=paras.LEARNING_RATE, num_epochs=paras.NUM_EPOCHS):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.lr = lr
    self.num_epochs = num_epochs
    self.backpropgate = backpropgate


  def optimizer_(self, parameters_to_be_optimized):
    return torch.optim.SGD(parameters_to_be_optimized, lr=self.lr, momentum=0.9)
    #return torch.optim.Adam(parameters_to_be_optimized, lr=self.lr, rho=0.9, eps=1e-3, weight_decay=0.001)
  
  def train_(self):

    specific_params_to_be_optimized = []
    if self.backpropgate == True:
      specific_params_to_be_optimized = self.model.parameters()
    else: 
      specific_params_to_be_optimized = [{'params': self.model.layers[-1].parameters()},
                                {'params': self.model.classifer.parameters()}, {'params': self.model.h1.parameters()}]
    
    n_total_steps = len(self.train_loader)
                                
    optimizer = self.optimizer_(specific_params_to_be_optimized )
    criterion = nn.CrossEntropyLoss()

    # model.train() tells your model that you are training the model.
    # So effectively layers like dropout, batchnorm etc. 
    # which behave different on the train and test procedures 
    # know what is going on and hence can behave accordingly.
    self.model.train()

    for epoch in range(self.num_epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(self.train_loader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(DEVICE)
          labels = labels.to(DEVICE)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = self.model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(epoch+1,self.num_epochs,i+1,n_total_steps,loss.item()))
    
    self.model.eval()
        
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
      self.train_()
      self.test_()
    
    else:
      N = len(self.model.additional_layers)

      for i in range(N):
        layer = self.model.additional_layers[i] # incoming new layer
        # 1. Add new layer to model
        self.model.layers.append(layer)
        # 2. diregarded output as output layer is retrained with every new added layer
        self.model.h0 = nn.LazyLinear(out_features=self.model.hidden_layer_features).to(DEVICE)
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
train = Train(train_loader, test_loader, backpropgate = True)
train.add_layers()

    
