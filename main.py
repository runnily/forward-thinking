from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import ConcatDataset
import torch

from models import (
    BaseModel,
    Convnet2,
    FeedForward,
    SimpleNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
)
from train import Train, TrainWithDataLoader, TrainWithDataSet
from utils import get_dataset, get_transform

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Forward-thinking", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "svhn", "mnist"],
        help="Choose a dataset to use",
    )
    parser.add_argument(
        "--num_data_per_layer",
        default=500,
        type=int,
        help="Defines how number of layers in the neural network when using multisource training",
    )
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Choose a learning rate ")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "convnet",
            "simplenet",
            "feedforward",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
        ],
        help="Choose the model architecture",
    )
    parser.add_argument("--num_classes", type=int, help="Choose the number of classes for model")
    # Arguments to use when training
    parser.add_argument("--batch_size", type=int, default=64, help="Choose a batch_size")
    parser.add_argument("--epochs", type=int, help="Choose the number of epochs")
    parser.add_argument(
        "--forward_thinking",
        default=1,
        type=int,
        help=(
            "Choose whether you want your model to learn using backpropgate (0) or"
            " forwardthinking (1)"
        ),
    )
    parser.add_argument(
        "--multisource",
        default=0,
        type=int,
        help=(
            "If your model trains using forward thinking, Multisource (1) means to have different"
            " training data to train each layer or using the same training data to train each"
            " layer (0)"
        ),
    )
    parser.add_argument(
        "--init_weights",
        default=1,
        type=int,
        help="Choose whether you want to initialize your weights (1) or not (0)",
    )
    parser.add_argument(
        "--batch_norm",
        default=0,
        type=int,
        help=(
            "Choose whether you want your model to include batch normalisation layers (1) or"
            " not (0)"
        ),
    )
    parser.add_argument(
        "--affine", 
        default=1,
        type=int, 
        help=(
              "Define whether batch norm has learnable affine parameters. (1) yes for"
              " learnable parameters, (0) no learnable affine parameters"),
    )
    parser.add_argument(
        "--freeze_batch_norm_layers",
        default=0,
        type=int,
        help=(
            "If the model architecture your using includes batch normalisation layers and model is"
            " using the forward-thinking method to learn choose whether to freeze those batch"
            " layers during training"
        ),
    )
    parser.add_argument(
        "--filename",
        default="accuracy",
        type=str,
        help=(
            "where to save the metrics logs (accuracy etc). All files will be saved as .csv types"
            "in the utils/recorded-accuracy folder"
        ),
    )

    args = parser.parse_args()
    dataset = args.dataset.upper()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    model_choice = args.model.lower()
    filename = args.filename.lower() + ".csv"

    model: BaseModel
    train: Train

    assert (model_choice == "feedforward" and dataset == "MNIST") or (
        model_choice != "feedforward"
    ), f"The choosen model: {model_choice} is not compatible with the dataset {dataset} "

    if model_choice == "simplenet":
        model = SimpleNet(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "convnet":
        model = Convnet2(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "feedforward":
        model = FeedForward()
    if model_choice == "resnet18":
        model = resnet18(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "resnet34":
        model = resnet34(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "resnet50":
        model = resnet50(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "resnet101":
        model = resnet101(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "resnet152":
        model = resnet152(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "vgg11":
        model = vgg11(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "vgg13":
        model = vgg13(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "vgg16":
        model = vgg16(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )
    if model_choice == "vgg19":
        model = vgg19(
            num_classes=args.num_classes, batch_norm=args.batch_norm, init_weights=args.init_weights, affine=args.affine
        )

    if args.multisource == 1 and args.forward_thinking == 1:
        train_dataset = None
        test_dataset = None
        if dataset == "SVHN":
            train_dataset = globals()[dataset](
              root="./data", split="extra", download=True, transform=get_transform()
            )
            test_dataset = globals()[dataset](root="./data", split="test", download=True, transform=get_transform())
        elif dataset in {"CIFAR10", "CIFAR100"}:
            train_dataset=globals()[dataset](
                  "./data", train=True, download=True, transform=get_transform()
            )
            test_dataset=globals()[dataset](
                  "./data", train=False, download=True, transform=get_transform()
            )
        else:
            raise ValueError("Can only perform action with the CIFAR datasets")
        train = TrainWithDataSet(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            freeze_batch_layers=args.freeze_batch_norm_layers,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=batch_size,
            num_data_per_layer=args.num_data_per_layer
        )
        

    assert (args.multisource == 1 and args.forward_thinking != 0) or (
        args.multisource == 0
    ), "Cannot do multisource training without using forward thinking. To use multi source set --forward_thinking = 1"

    if args.multisource == 0:
        train_loader, test_loader = get_dataset(name=dataset, batch_size=batch_size)
        backpropgate = False if (args.forward_thinking == 1) else True
        train = TrainWithDataLoader(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            backpropgate=backpropgate,
            freeze_batch_layers=args.freeze_batch_norm_layers,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )

    train.add_layers()
    train.recordAccuracy.save(filename)
