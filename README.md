![alt text](imgs/title.png)

![GitHub issues](https://img.shields.io/github/issues/runnily/forward-thinking)
![GitHub pull requests](https://img.shields.io/github/issues-pr/runnily/forward-thinking)
![Github workflow](https://github.com/runnily/forward-thinking/actions/workflows/docker-image.yml/badge.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/runnily/forward-thinking)

This alternative approaches of building a neural networks. This project explores greedy approaches to buidling a neural network 
rather than the conventienal heuristic method. A comparasion will then be made to see the affectiveness of both

## Usage

```
python main.py --dataset=cifar10 --model=resnet18 --learning_rate=0.01 --num_classes=10 --batch_size=64 --epochs=5 --forward_thinking=1 --multisource=0 --init_weights=0 --batch_norm=0 --freeze_batch_norm_layers=0
```
