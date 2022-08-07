from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import get_dataset, get_transform
from models import SimpleNet, Convnet2, BaseModel
from train import TrainWithDataSet, TrainWithDataLoader, Train
from torchvision.datasets import CIFAR100, CIFAR10

if __name__ == '__main__':
  parser = ArgumentParser(description='Forward-thinking', formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--dataset", 
    type=str, 
    choices=['cifar10', 'cifar100', 'svhn', 'mnist'], 
    help="Choose a dataset to use"
    )
  parser.add_argument(
    "--learning_rate", 
    type=float, default=0.01, 
    help="Choose a learning rate "
    )
  parser.add_argument(
    "--model", 
    type=str, 
    choices=['convnet', "simplenet"], 
    help="Choose the model architecture")
  parser.add_argument(
    "--num_classes",
    type=int,
  )
  # Arguments to use when training
  parser.add_argument(
    "--batch_size", 
    type=int, 
    default=64, 
    help="Choose a batch_size")
  parser.add_argument(
    "--epochs",
    type=int,
    help="Choose the number of epochs")
  parser.add_argument(
    "--forward_thinking",
    default=1,
    type=int,
    help="Choose whether you want your model to learn using backpropgate (0) or forwardthinking (1)")
  parser.add_argument(
    "--multisource",
    default=0,
    type=int,
    help="If your model trains using forward thinking, Multisource (1) means to have different training data to train each layer or using the same training data to train each layer (0)")
  parser.add_argument(
    "--init_weights", 
    default=1, 
    type=int, 
    help="Choose whether you want to initialize your weights (1) or not (0)")
  parser.add_argument(
    "--batch_norm", 
    default=0, 
    type=int, 
    help="Choose whether you want your model to include batch normalisation layers (1) or not (0)")
  parser.add_argument(
    "--freeze_batch_norm_layers", 
    default=0, 
    type=int, 
    help="If the model architecture your using includes batch normalisation layers and model is using the forward-thinking method to learn choose whether to freeze those batch layers during training")

  args = parser.parse_args()
  dataset = args.dataset.upper()
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  num_epochs = args.epochs

  model: BaseModel
  train: Train

  if args.model.lower() == "simplenet":
    model = SimpleNet(
      num_classes=args.num_classes, 
      batch_norm=args.batch_norm,
      init_weights=args.init_weights)
  if args.model.lower() == "convnet":
    model = Convnet2(
      num_classes=args.num_classes, 
      batch_norm=args.batch_norm,
      init_weights=args.init_weights)
  if args.multisource == 1 and args.forward_thinking == 1:
    if dataset in {"CIFAR10", "CIFAR100"}:
      train = TrainWithDataSet(
        model = model,
        train_dataset = globals()[dataset]("./data", train=True, download=True, transform=get_transform()),
        test_dataset = globals()[dataset]("./data", train=True, download=True, transform=get_transform()),
        freeze_batch_layers = args.freeze_batch_norm_layers,
        learning_rate = args.learning_rate,
        num_epochs = args.epochs,
        batch_size=batch_size
      )
    else:
      raise ValueError("Can only perform action with the CIFAR datasets")

  if args.multisource and args.backpropgate:
    raise ValueError("Cannot do multisource training without using forward thinking, both must variables be set True")

  if args.multisource == 0:
    if args.multisource == False:
      train_loader, test_loader = get_dataset(
          name="CIFAR10",
          batch_size=batch_size
        )
      backpropgate = False if (args.forward_thinking == 1) else True
      train = TrainWithDataLoader(
            model = model,
            train_loader = train_loader,
            test_loader = test_loader,
            backpropgate = backpropgate ,
            freeze_batch_layers = args.freeze_batch_norm_layers,
            learning_rate = learning_rate,
            num_epochs = num_epochs)

  train.add_layers()
  train.recordAccuracy.save()









