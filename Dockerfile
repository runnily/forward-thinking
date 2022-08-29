FROM python:3.8

ADD requirements.txt /
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

ADD models /
ADD utils /
ADD main.py /
ADD train.py /
ADD README.md /

ENV PYTHONUNBUFFERED=1
CMD [ "python", "python main.py --dataset=cifar10 --model=resnet18 --learning_rate=0.01 --num_classes=10 --batch_size=64 --epochs=5 --forward_thinking=1 --multisource=0 --init_weights=0 --batch_norm=0 --freeze_batch_norm_layers=0" ]
