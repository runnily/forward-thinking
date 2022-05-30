from sys import argv


input_size = 784 
hidden_size = 10
num_classes = 10
num_epochs = 2
batch_size = 128
in_channels = 3 #1
learning_rate = 0.001

paras = {input_size : input_size, hidden_size: hidden_size, num_classes : num_classes,
          num_epochs : num_epochs, batch_size : batch_size, in_channels : in_channels,
          learning_rate: learning_rate}

def setInputSize(num):
  input_size = num

def setHiddenSize(num):
  hidden_size = num

def setClassesSize(num):
  classes_size = num

def setNumEpochs(num):
  num_epochs = num

def setBatchSize(num):
  batch_size = num

def setInChannels(num):
  in_channels = num

def setLearningRate(num):
  learning_rate = num

