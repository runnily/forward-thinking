import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import batchnorm
import utils
import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784
hidden_size = 512
num_classes = 100
num_epochs = 5
batch_size = 64
in_channels = 3  # 1
learning_rate = 0.01


class Train:
  def __init__(
      self,
      model,
      test_loader,
      train_loader=None,
      train_data=None,
      backpropgate=False,
      freeze_batch_layers=False,
      lr=learning_rate,
      num_epochs=num_epochs,
  ) -> None:
      self.model = model
      self.freeze_batch_layers = freeze_batch_layers
      self.backpropgate = backpropgate
      self.train_loader = train_loader
      self.test_loader = test_loader
      self.lr = lr
      self.num_epochs = num_epochs
      self.recordAccuracy = utils.Measure()
      self.__running_time = 0.00
      self.classifier_train_loader = train_loader
      save_keys = self.model.incoming_layers
      self.get_loader = {}

      # if train_loader is defined than uses whole dataset for each layer
      if train_loader != None and train_data == None:
        for key in save_keys:
          self.get_loader[key] = train_loader

      # if no train_loader and train_data is given uses subset of data for each layer
      if train_data != None and train_loader == None:

        for key in save_keys:
          self.get_loader[key] = []
        self.get_loader[self.model.classifier] = []

        num_data_per_layer = int(
          len(train_data.targets) / self.model.num_classes / len(get_loader)
        )
        self.get_loader = utils.divide_data_by_group(
          train_data,
          num_data_per_layer,
          batch_size=batch_size,
          groups=self.get_loader,
        )

        self.classifier_train_loader = self.get_loader.pop(self.model.classifier)

  def __optimizer(self, parameters_to_be_optimized):
    return optim.SGD(parameters_to_be_optimized, lr=self.lr, momentum=0.9)

  def __accuracy(self, predictions, labels):
    # https://stackoverflow.com/questions/61696593/accuracy-for-every-epoch-in-pytorch
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())  # needs mean for each batch size

  def __train(self, specific_params_to_be_optimized, num_epochs, train_loader):

      n_total_steps = len(train_loader)

      optimizer = self.__optimizer(specific_params_to_be_optimized)
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
        for i, (images, labels) in enumerate(
          train_loader, 0):
          # looping over every batch
          # get the inputs; data is a list of [inputs, labels]

          images = images.to(DEVICE)
          labels = labels.to(DEVICE)

          # forward + backward + optimize
          optimizer.zero_grad()  # zero the parameter gradients
          outputs = self.model(images)
          loss = criterion(outputs, labels)
          loss.backward()  #  computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
          optimizer.step()  # causes the optimizer to take a step based on the gradients of the parameters.

          running_accuracy += self.__accuracy(outputs, labels)
          # print statistics
          running_loss += loss.item()
          if (i + 1) % 100 == 0:
            print(
              "Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(
              epoch + 1, num_epochs, i + 1, n_total_steps, loss.item()
              )
            )
        end = torch.cuda.Event(enable_timing=True)
        end.record()

        torch.cuda.synchronize()

        test_accuracy = self.__test()
        len_self_loader = len(train_loader)
        running_accuracy /= len_self_loader
        running_loss /= len_self_loader
        print(
          "Test accuracy: {}, Training accuracy: {}".format(
            test_accuracy * 100, running_accuracy * 100
            )
        )
        self.__running_time += start.elapsed_time(
            end
        )  # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        self.recordAccuracy(
          self.__running_time,
          epoch,
          running_loss,
          test_accuracy,
          running_accuracy.item(),
        )

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
          _, predicted = torch.max(outputs.data, 1)
          n_samples += labels.size(0)  # number of samples in current batch
          n_correct += (
              (predicted == labels).sum().item()
          )  # gets the number of correct

      accuracy = n_correct / n_samples
      return accuracy

  def freeze_layers_(self):
      if self.backpropgate == False:
        for l in self.model.frozen_layers:
          l.requires_grad_(False)
          if self.model.batch_norm and self.freeze_batch_layers == False:
            # when freeze_batch_layers is False
            print("True")
            l[1].requires_grad_(True) # freeze only conv layer
                  

  def __getEpochforLayer(
    self,
    layer_key: int,
    change_epochs_each_layer: bool = False,
    epochs_each_layer={},
  ):
    print("This is layer {}".format(layer_key))
    if change_epochs_each_layer:
      try:
        return int(input("Number of epoch for layer {} ".format(layer_key)))
      except:
        pass
    return epochs_each_layer.get(layer_key, self.num_epochs)

  def __defineParas(self, idx_layer, layer):
    # defines specific parameters for layer
    specific_params_to_be_optimized = []
    if self.model.batch_norm and self.freeze_batch_layers == False:
      # when freeze_batch_layers is False add batch_parameters parameters to optimise
      specific_params_to_be_optimized = [
        {"params": self.model.current_layers[i][1].parameters()}
        for i in range(0, idx_layer)
      ]
    specific_params_to_be_optimized.append({"params": layer.parameters()})
    specific_params_to_be_optimized.append(
      {"params": self.model.classifier.parameters()}
    )
    return specific_params_to_be_optimized

  def add_layers(self, change_epochs_each_layer=False, epochs_each_layer={}):

      if self.backpropgate == True:
        if self.train_loader == None:
          raise ValueError("You cannot backpropgate with train_loader set as 0")
        self.model.current_layers = nn.Sequential(*self.model.incoming_layers).to(
          DEVICE
        )
        params = [
          {"params": self.model.classifier.parameters()},
          {"params": self.model.current_layers.parameters()},
        ]
        self.__train(params, self.num_epochs, self.train_loader)

      else:

        for i, layer in enumerate(self.model.incoming_layers):
          # 1. Add new layer to model
          self.model.current_layers.append(layer.to(DEVICE))
          # 2. diregarded output as output layer is retrained with every new added layer
          self.model.classifier = nn.LazyLinear(
              out_features=self.model.num_classes
          ).to(DEVICE)
          # 3. defining parameters to be optimized
          specific_params_to_be_optimized = self.__defineParas(i, layer)
          # 4. Train
          # 4a. Get the number of epochs
          num_epochs = self.__getEpochforLayer(
              i, change_epochs_each_layer, epochs_each_layer
          )
          # 4b. Training the model
          self.__train(
              specific_params_to_be_optimized, num_epochs, self.get_loader[layer]
          )
          # 5. As we have trained add layer to the frozen_layers
          self.model.frozen_layers.append(self.model.current_layers[-1])
          # 6. Freeze layers
          self.freeze_layers_()

        incoming_layers_len = len(self.model.incoming_layers)
        if (
          self.backpropgate == False
          and len(self.model.current_layers) == incoming_layers_len
        ):
          # This part is for training the last layers
          num_epochs = self.__getEpochforLayer(
            incoming_layers_len, change_epochs_each_layer, epochs_each_layer
          )
          print("Last layer!!")
          self.__train(
            [{"params": self.model.classifier.parameters()}],
            num_epochs,
            self.classifier_train_loader,
          )


if __name__ == "__main__":
  model = models.Convnet2(num_classes=num_classes, batch_norm=True).to(DEVICE)
  train_loader, test_loader, _, _ = utils.CIFAR_10(batch_size=batch_size)
  # _, test_loader, train_data, _ = utils.CIFAR_10()
  train = Train(
    model,
    test_loader, 
    train_loader=train_loader, 
    )
  train.add_layers()
  train.recordAccuracy.save()
