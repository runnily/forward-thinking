import torch
import torch.nn as nn
from torch.nn.functional import relu
import torchvision
import torchvision.transforms as transforms

INPUT_SIZE = 784 # 28 * 28
HIDDEN_SIZE = 150 
NUM_CLASSES = 10
NUM_EPOCHS = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Getting the dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

# Obtaining a data loader: which add batch size, and makes the data iterable
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

class FNN(nn.Module):

  def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES, initial_hidden_size = HIDDEN_SIZE):
    super(FNN, self).__init__()
    # self.l1 = nn.Linear(input_size, initial_hidden_size)
    self.l2 = nn.Linear(initial_hidden_size, num_classes)
    self.layers = [] # self.layers = [self.l1]
    self.frozen_layers = []

  def forward(self, x):
    out = x.reshape(-1, INPUT_SIZE).to(device)
    for l in self.layers:
      out = l(out)
      out = relu(out)
    out = self.l2(out)
    return out


class Train():

  def __init__(self, backpropgate = False, model=FNN().to(device), lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, train_loader=train_loader, test_loader=test_loader,):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.lr = lr
    self.num_epochs = NUM_EPOCHS
    self.backpropgate = backpropgate

  
  def train_(self):

    criterion = nn.CrossEntropyLoss()
    specific_params_to_be_optimized = [{'params': self.model.layers[-1].parameters()},
                                {'params': self.model.l2.parameters()}
            ]
    optimizer = torch.optim.Adam(specific_params_to_be_optimized, lr=self.lr)

    n_total_steps = len(train_loader)

    for epoch in range(self.num_epochs):
      for i, (images, labels) in enumerate(self.train_loader):

        labels = labels.to(device)

        # forward pass
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backwards and optimize
        optimizer.zero_grad() # clears old gradients from the last step (otherwise you’d just accumulate the gradients from all backward() calls
        loss.backward() #  computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.
        
        #if (i+1) % 100 == 0:
           # print("Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(epoch+1,self.num_epochs,i+1,n_total_steps,loss.item()))
      print("Epoch {} and Loss: {}".format(epoch, loss.item()))
        
  def test_(self):
    with torch.no_grad():
      n_correct = 0
      n_samples = 0
      for images, labels in self.test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
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

    def choice():
      add_layer_ques = input("would you like to add a layer? ")
      return ("y" or add_layer_ques == "Y" or add_layer_ques == "Yes")

    add_layer_choice = choice()

    initial_hidden_size = HIDDEN_SIZE

    additional_layers = [
            nn.Linear(INPUT_SIZE, initial_hidden_size),
            nn.Linear(150, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 10),]

    while (len(additional_layers) > 0) and add_layer_choice:
      layer = additional_layers.pop(0) # incoming new layer
      print(additional_layers)
      # 1. Add new layer to model
      self.model.layers.append(layer)
      # 2. diregarded output as output layer is retrained with every new added layer
      self.model.l2 = nn.Linear(layer.out_features, 10) #nn.LazyLinear(10)
      # 3. Train
      self.train_()
      # 4. Get Accuracy
      self.test_()
      # 5. As we have trained add layer to the frozen_layers
      self.model.frozen_layers.append(self.model.layers[-1])
      # 6. Freeze layers
      self.freeze_layers_() 
      
      add_layer_choice = choice()

    if len(additional_layers) < 0:
      pass
    

model = Train()
model.add_layers()

      
    
    

    



    
