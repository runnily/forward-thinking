import torch
torch.cuda.empty_cache()
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torchvision.models.vgg as vgg


model_urls = {
    'svhn': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/svhn-f564f3d8.pth',
}


DEVICE = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.SVHN(root='./data', split="train",
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.SVHN(root='./data', split="test",
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def test(loader):
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in loader:
          images, labels = data
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)
          # calculate outputs by running images through the network
          outputs = net(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return (correct/total)*100


class Net(nn.Module):

  def __init__(self, num_classes: int = 10):
    super().__init__()
    """self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.AdaptiveAvgPool2d((7, 7))
    )"""
    self.features = nn.Sequential(# 0 : size: 24x24, channel: 3
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 1 : kernel: 3x3, channel: 64, padding: 1
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 2 : kernel: 3x3, channel: 64, padding: 1
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 3 : kernel: 3x3, channel: 128, padding: 1
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 4 : kernel: 3x3, channel: 128, padding: 1
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 5 : kernel: 3x3, channel: 256, padding: 1
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 6 : kernel: 3x3, channel: 256, padding: 1
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 7 : kernel: 3x3, channel: 256, padding: 1
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True),
        # 8 : kernel: 3x3, channel: 256, padding: 1
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.ReLU(inplace=True))
        # 9 : channel: 1024
        # 10 : channel: 1024
        # 11 : channel: 10

    if True:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    self.classifier = nn.Sequential(
            nn.LazyLinear(512 * 7 * 7, 10),
            
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x



net = vgg.VGG(vgg.make_layers(vgg.cfgs["A"]), 
                  num_classes=10, init_weights=True).to(DEVICE)
net = Net().to(DEVICE)
#net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False).to(DEVICE)

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 5
n_total_steps = len(trainloader)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}".format(epoch+1,num_epochs,i+1,n_total_steps,loss.item()))
            running_loss = 0.0
    accuracy = test(testloader)
    print("Test accuracy: {}".format(accuracy))

print('Finished Training')


