INPUT_SIZE = 784 # 28 * 28
HIDDEN_SIZE = 150 
NUM_CLASSES = 100
NUM_EPOCHS = 2
BATCH_SIZE = 100
IN_CHANNELS = 3 #1
LEARNING_RATE = 0.001
from torch import device, cuda
device = device('cuda' if cuda.is_available()  else 'cpu')
