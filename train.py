import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import batchnorm
from torch.utils.data import DataLoader, Dataset

import models
import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 64
in_channels = 3  # 1
learning_rate = 0.01
MILESTONES = [60, 120, 160]


class Train:
    """
    This is a class used for training. This either uses forward-thinking
    or backpropgation to train the model. The function add_layers() is where
    the forward-thinking algorithm is applied.
    Args:
        model (nn.Module): The model to train
        backpropgation (Bool): whether to use backpropgation (True) or Forward thinking (False)
        Freeze_batch_layers (Bool): Utilised in when using forward-thinking to train, whether to
                                    (True) freeze batch layers when training or not (False)
        learning_rate: The learning_rate used to train the algorthium
        num_epochs: The number of epochs used.
    """

    def __init__(
        self,
        model: models.BaseModel,
        backpropgate: bool,
        freeze_batch_layers: bool,
        learning_rate: int,
        num_epochs: int,
    ) -> None:
        self.model = model.to(DEVICE)
        self.freeze_batch_layers = freeze_batch_layers
        self.backpropgate = backpropgate
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.recordAccuracy = utils.Measure()
        self.__running_time = 0.00
        self.get_loader: Optional[Dict[nn.Module, DataLoader]]

    def get_train_loader(self, layer: nn.Module) -> DataLoader:
        pass

    def _optimizer(self, parameters_to_be_optimized):
        return optim.SGD(
            parameters_to_be_optimized, lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )

    def __accuracy(self, predictions, labels):
        # https://stackoverflow.com/questions/61696593/accuracy-for-every-epoch-in-pytorch
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float())  # needs mean for each batch size

    def __train(self, specific_params_to_be_optimized, num_epochs, train_loader):
        n_total_steps = len(train_loader)
        optimizer = self._optimizer(specific_params_to_be_optimized)
        criterion = nn.CrossEntropyLoss().to(DEVICE)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.00
            running_accuracy = 0.00
            if torch.cuda.is_available() is True:
              start = torch.cuda.Event(enable_timing=True)
              start.record()
            for i, (images, labels) in enumerate(train_loader, 0):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_accuracy += self.__accuracy(outputs, labels)
                running_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(
                            epoch + 1, num_epochs, i + 1, n_total_steps, loss.item()
                        )
                    )
            if torch.cuda.is_available() is True:
              end = torch.cuda.Event(enable_timing=True)
              end.record()
              torch.cuda.synchronize()
              test_accuracy = self.__test()
              len_self_loader = len(train_loader)
              running_accuracy /= len_self_loader
              running_loss /= len_self_loader
              print(
                  "Epoch {} | Test accuracy {} | Training accuracy: {}".format(
                      epoch + 1, test_accuracy * 100, running_accuracy * 100
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
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

        accuracy = n_correct / n_samples
        return accuracy

    def _freeze_layers(self):
        parameters = []
        if self.backpropgate is False:
            for layer in self.model.frozen_layers:
                layer.requires_grad_(False)
                if self.model.batch_norm and self.freeze_batch_layers is False:
                    if isinstance(layer, nn.Sequential) and isinstance(layer[1], nn.BatchNorm2d):
                        layer[1].requires_grad_(True)  # freezes only the conv layer
                        parameters.append({"params": layer[1].parameters()})
                    elif isinstance(layer, nn.BatchNorm2d):
                        layer.requires_grad_(True)
                        parameters.append({"params": layer.parameters()})
                    elif isinstance(layer, models.BasicBlock):
                        layer.current_layers[1].requires_grad_(True)
                        parameters.append({"params": layer.current_layers[1].parameters()})
                        layer.output[1].requires_grad_(True)
                        parameters.append({"params": layer.output[1].parameters()})
                        if len(layer.shortcut) > 1:
                            layer.shortcut[1].requires_grad_(True)
                            parameters.append({"params": layer.shortcut[1].parameters()})
                    elif isinstance(layer, models.BottleNeck):
                        layer.current_layers[1].requires_grad_(True)
                        parameters.append({"params": layer.current_layers[1].parameters()})
                        layer.current_layers[4].requires_grad_(True)
                        parameters.append({"params": layer.current_layers[3].parameters()})
                        layer.output[1].requires_grad_(True)
                        parameters.append({"params": layer.output[1].parameters()})
                        if len(layer.shortcut) > 1:
                            layer.shortcut[1].requires_grad_(True)
                            parameters.append({"params": layer.shortcut[1].parameters()})
                    else:
                        pass

        return parameters

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
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")
        return epochs_each_layer.get(layer_key, self.num_epochs)

    def __defineParas(self, idx_layer, layer, specific_params_to_be_optimized=[]):
        # defines specific parameters for layer
        specific_params_to_be_optimized.append({"params": layer.parameters()})
        specific_params_to_be_optimized.append({"params": self.model.output.parameters()})
        return specific_params_to_be_optimized

    def add_layers(self, change_epochs_each_layer=False, epochs_each_layer={}):

        if self.backpropgate is True:
            if self.get_loader is not None:
                raise ValueError(
                    "You cannot backpropgate when there are different train sets defined for each"
                    " layer"
                )
            self.model.current_layers = nn.Sequential(*self.model.incoming_layers).to(DEVICE)
            params = [
                {"params": self.model.output.parameters()},
                {"params": self.model.current_layers.parameters()},
            ]
            self.__train(params, self.num_epochs, self.train_loader)

        else:

            parameters = []

            for i, layer in enumerate(self.model.incoming_layers):
                # 1. Add new layer to model
                self.model.current_layers.append(layer.to(DEVICE))
                # 2. diregarded output as output layer is retrained with every new added layer
                self.model.output = nn.LazyLinear(out_features=self.model.num_classes).to(DEVICE)
                # 3. defining parameters to be optimized
                specific_params_to_be_optimized = self.__defineParas(i, layer, parameters)
                # 4. Train
                # 4a. Get the number of epochs
                num_epochs = self.__getEpochforLayer(i, change_epochs_each_layer, epochs_each_layer)
                # 4b. Training the model
                self.__train(
                    specific_params_to_be_optimized, num_epochs, self.get_train_loader(layer)
                )
                # 5. As we have trained add layer to the frozen_layers
                self.model.frozen_layers.append(self.model.current_layers[-1])
                # 6. Freeze layers
                parameters = self._freeze_layers()

            incoming_layers_len = len(self.model.incoming_layers)
            if self.backpropgate is False and len(self.model.current_layers) == incoming_layers_len:
                # This part is for training the last layers
                num_epochs = self.__getEpochforLayer(
                    incoming_layers_len, change_epochs_each_layer, epochs_each_layer
                )
                print("Last layer!!")
                self.__train(
                    [{"params": self.model.output.parameters()}],
                    num_epochs,
                    self.get_train_loader(self.model.output),
                )


class TrainWithDataLoader(Train):
    Train.__doc__ += """
            train_loader (nn.DataLoader): This is the train dataloader
            test_loader (nn.DataLoader): This is the test dataloader
    Examples::
    `train = TrainWithDataLoader(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        backpropgate=False,
        freeze_batch_layers=False,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )`
    """

    def __init__(
        self,
        model: models.BaseModel,
        backpropgate: bool,
        freeze_batch_layers: bool,
        learning_rate: int,
        num_epochs: int,
        train_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
    ) -> None:
        super(TrainWithDataLoader, self).__init__(
            model, backpropgate, freeze_batch_layers, learning_rate, num_epochs
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.get_loader = None

    def get_train_loader(self, layer: nn.Module) -> DataLoader:
        return self.train_loader


class TrainWithDataSet(Train):
    Train.__doc__ += """
            train_dataset (nn.DataLoader): This is the train dataset
            test_dataset (nn.DataLoader): This is the test dataset
    
    Notes::
      This is used when for multiple-source learning where the dataset 
      is divided into different sets. This different sets is used when model is
      training with the forward-thinking method, where each layer is given a corresponding
      dataset to train with.
    Examples::
    `train = TrainWithDataLoader(
        model=model,
        test_dataset=test_dataset,
        test_dataset=test_dataset,
        backpropgate=False,
        freeze_batch_layers=False,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )`
    """

    def __init__(
        self,
        model: models.BaseModel,
        train_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        freeze_batch_layers: bool,
        learning_rate: int,
        num_epochs: int,
        batch_size: int,
    ) -> None:
        super().__init__(model, False, freeze_batch_layers, learning_rate, num_epochs)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=2, shuffle=True
        )
        self.get_loader = {}
        for layer_key in self.model.incoming_layers:
            self.get_loader[layer_key] = []
        self.get_loader[self.model.output] = []

        num_data_per_layer = 2000 #int(len(train_dataset.targets) / self.model.num_classes)
        self.get_loader = utils.divide_data_by_group(
            train_dataset,
            num_data_per_layer,
            batch_size=batch_size,
            groups=self.get_loader,
        )

        self.loader_for_last_layer = list(self.get_loader.values())[-1]

    def get_train_loader(self, layer: nn.Module) -> DataLoader:
        if layer not in self.get_loader:
            return self.loader_for_last_layer
        return self.get_loader[layer]


if __name__ == "__main__":
    # model = models.Convnet2(num_classes=num_classes, batch_norm=False, init_weights=False).to(
    # DEVICE
    # )
    model = models.resnet34(batch_norm=False, num_classes=10, init_weights=True)
    # model = models.FeedForward().to(DEVICE)
    train_loader, test_loader = utils.get_dataset(name="CIFAR10", batch_size=batch_size)
    # _, test_loader, train_data, _ = utils.CIFAR_10()
    train = TrainWithDataLoader(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        backpropgate=False,
        freeze_batch_layers=False,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    train.add_layers()
    train.recordAccuracy.save()
