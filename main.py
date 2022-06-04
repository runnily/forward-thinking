import torch
import torch.nn as nn
import torch.nn.functional as F
import utils 
import models

DEVICE = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
input_size = 784 
hidden_size = 512
num_classes = 100
num_epochs = 5
batch_size = 128
in_channels = 3 #1
learning_rate = 0.01
model = models.BasicNet(num_classes=num_classes).to(DEVICE)


class Train():

  def __init__(self,  train_loader, test_loader, model=model, lr=learning_rate, num_epochs=num_epochs):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.lr = lr
    self.num_epochs = num_epochs
    self.recordAccuracy = utils.Measure()
    self.__running_time = 0.00


  def optimizer_(self, parameters_to_be_optimized):
    return torch.optim.SGD(parameters_to_be_optimized, lr=self.lr, momentum=0.9)
    #return torch.optim.Adadelta(parameters_to_be_optimized, lr=self.lr, rho=0.9, eps=1e-3, weight_decay=0.001)

  def __accuracy(self, predictions, labels):
    # https://stackoverflow.com/questions/61696593/accuracy-for-every-epoch-in-pytorch
    classes = torch.argmax(predictions, dim=1) 
    return torch.mean((classes == labels).float()) # needs mean for each batch size
  
  def __train(self, specific_params_to_be_optimized, num_epochs):

    n_total_steps = len(self.train_loader)
                                
    optimizer = self.optimizer_(specific_params_to_be_optimized)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # model.train() tells your model that you are training the model.
    # So effectively layers like dropout, batchnorm etc. 
    # which behave different on the train and test procedures 
    # know what is going on and hence can behave accordingly.

    for epoch in range(num_epochs):  # loop over the dataset multiple times
      self.model.train()
      running_loss = 0.00
      running_accuracy = 0.00
      start = torch.cuda.Event(enable_timing=True)
      start.record()
      for i, (images, labels) in enumerate(self.train_loader, 0): # looping over every batch
          # get the inputs; data is a list of [inputs, labels]
         
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)

          outputs = self.model(images)

          # forward + backward + optimize
          optimizer.zero_grad() # zero the parameter gradients
          loss = criterion(outputs, labels)
          loss.backward() #  computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
          optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters.

          running_accuracy += self.__accuracy(outputs, labels)
          # print statistics
          running_loss += loss.item()
          if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(epoch+1,num_epochs,i+1,n_total_steps,loss.item()))
      end = torch.cuda.Event(enable_timing=True)
      end.record()

      torch.cuda.synchronize()

      test_accuracy = self.__test()
      len_self_loader = len(self.train_loader)
      running_accuracy /= len_self_loader
      running_loss /= len_self_loader
      print("Test accuracy: {}, Training accuracy: {}".format(test_accuracy, running_accuracy))
      self.__running_time += start.elapsed_time(end) # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
      self.recordAccuracy(self.__running_time, epoch, running_loss, test_accuracy, running_accuracy.item())

        
  def __test(self):
    self.model.eval()
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

    accuracy = n_correct / n_samples
    return accuracy

  def freeze_layers_(self):
    if self.model.backpropgate == False:
      for l in self.model.frozen_layers:
        l.requires_grad_(False)

  def __getEpochforLayer(self, layer_key, change_epochs_each_layer = False, epochs_each_layer={}):
    if change_epochs_each_layer:
      try:
        return int(input("Number of epoch for {} ".format(layer_key)))
      except:
        pass
    return epochs_each_layer.get(layer_key,  self.num_epochs)


  def add_layers(self, change_epochs_each_layer = False, epochs_each_layer={}):

    N = len(self.model.incoming_layers)
    if self.model.backpropgate == True:
      self.model.current_layers = self.model.incoming_layers
      print(self.model.current_layers)
      params = [{'params': self.model.classifier.parameters()}]
      for l in self.model.current_layers:
        params.append({'params': l.parameters()})
      self.__train(params, self.num_epochs)
   
    else:
      print(self.model.backpropgate)

      for i in range(N):
        layer = self.model.incoming_layers[i] # incoming new layer
        # 1. Add new layer to model
        self.model.current_layers.append(layer)
        # 2. diregarded output as output layer is retrained with every new added layer
        self.model.classifier = nn.LazyLinear(out_features=self.model.num_classes).to(DEVICE)
        if not isinstance(self.model.incoming_layers[i], nn.ReLU) or not isinstance(self.model.incoming_layers[i], nn.MaxPool2d) or not isinstance(self.model.incoming_layers[i], nn.AvgPool2d):
          # 3. defining parameters to be optimized
          specific_params_to_be_optimized = [{'params': self.model.current_layers[-1].parameters()}, {'params': self.model.classifier.parameters()}]
          # 4. Train 
          num_epochs = self.__getEpochforLayer(i, change_epochs_each_layer, epochs_each_layer)
          self.__train(specific_params_to_be_optimized, num_epochs = num_epochs)
          # 5. As we have trained add layer to the frozen_layers
          self.model.frozen_layers.append(self.model.current_layers[-1])
          # 6. Freeze layers
          self.freeze_layers_()
        else:
          pass

    # This part is to train the last layers
    if len(self.model.incoming_layers) == len(self.model.incoming_layers) and self.model.backpropgate==False:
        num_epochs = self.__getEpochforLayer(N, change_epochs_each_layer, epochs_each_layer)
        self.__train([{'params': self.model.classifier.parameters()}], num_epochs = num_epochs)

if __name__ == "__main__":
  train_loader, test_loader = utils.CIFAR_100()
  train = Train(train_loader, test_loader)
  train.add_layers()
  train.recordAccuracy.save()

 