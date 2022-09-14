![alt text](imgs/title.png)

![GitHub issues](https://img.shields.io/github/issues/runnily/forward-thinking)
![GitHub pull requests](https://img.shields.io/github/issues-pr/runnily/forward-thinking)
![Github workflow](https://github.com/runnily/forward-thinking/actions/workflows/docker-image.yml/badge.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/runnily/forward-thinking)

The code for the forward-thinking algorithm. A forward-thinking algorithm constructs a neural
network layer by layer, choosing the next layer in such a way that it provides the best parameters (weights and biases) that fit the layer. The algorithm can be extended to multiple source transfer learning whereby the sample dataset is subdivided into unique sets for each layer. 

The code allows you to train using the CIFAR10, SVHN and MNIST datasets. 
- All architectures except "FeedForward" can be trained using the CIFAR10, SVHN dataset
- Feedforward can only be trained using the MNIST dataset


## Usage
This is tested to work on python 3.8-3.10. You can either run the code using docker or run it on your local PC. When you run `python main.py ...` make sure your within the forward-thinking folder.

(1) Using Docker
```
git clone git@github.com:runnily/forward-thinking.git
docker build -t <container-name> .
docker run -d <container-name>
python main.py --dataset=cifar10 --model=resnet18 --learning_rate=0.01 --num_classes=10 --batch_size=64 --epochs=5 --forward_thinking=1 --multisource=0 --init_weights=0 --batch_norm=0 --freeze_batch_norm_layers=0
```

(2) Local PC

```
git clone git@github.com:runnily/forward-thinking.git
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py --dataset=cifar10 --model=resnet18 --learning_rate=0.01 --num_classes=10 --batch_size=64 --epochs=5 --forward_thinking=1 --multisource=0 --init_weights=0 --batch_norm=0 --freeze_batch_norm_layers=0
```

other optional arguments:

```
  -h, --help            show this help message and exit
  --dataset {cifar10,cifar100,svhn,mnist}
                        Choose a dataset to use (default: None)
  --num_data_per_layer NUM_DATA_PER_LAYER
                        Defines how number of layers in the neural network when using multisource training (default: 500)
  --learning_rate LEARNING_RATE
                        Choose a learning rate (default: 0.01)
  --model {convnet,simplenet,feedforward,resnet18,resnet34,resnet50,resnet101,resnet152,vgg11,vgg13,vgg16,vgg19}
                        Choose the model architecture (default: None)
  --num_classes NUM_CLASSES
                        Choose the number of classes for model (default: None)
  --batch_size BATCH_SIZE
                        Choose a batch_size (default: 64)
  --epochs EPOCHS       Choose the number of epochs (default: None)
  --forward_thinking FORWARD_THINKING
                        Choose whether you want your model to learn using backpropgate (0) or forwardthinking (1) (default: 1)
  --multisource MULTISOURCE
                        If your model trains using forward thinking, Multisource (1) means to have different training data to train each layer or using the same training data to train each
                        layer (0) (default: 0)
  --init_weights INIT_WEIGHTS
                        Choose whether you want to initialize your weights (1) or not (0) (default: 1)
  --batch_norm BATCH_NORM
                        Choose whether you want your model to include batch normalisation layers (1) or not (0) (default: 0)
  --affine AFFINE       Define whether batch norm has learnable affine parameters. (1) yes for learnable parameters, (0) no learnable affine parameters (default: 1)
  --freeze_batch_norm_layers FREEZE_BATCH_NORM_LAYERS
                        If the model architecture your using includes batch normalisation layers and model is using the forward-thinking method to learn choose whether to freeze those batch
                        layers during training (default: 0)
  --filename FILENAME   where to save the metrics logs (accuracy etc). All files will be saved as .csv typesin the utils/recorded-accuracy folder (default: accuracy)
```

## Additional notes
Performance metrics are created with computers that have GPUs that support CUDA 9.0. 
These performance metrics are usually logged and saved within `utils/recorded-accuracy`.