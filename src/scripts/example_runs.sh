#!/bin/bash

cp configs/train_s4.yaml configs/train.yaml
python src/train.py dataset.name=mnist
python src/train.py dataset.name=cifar dataset.grayscale=true
python src/train.py dataset.name=cifar dataset.grayscale=false

cp configs/train_drnn.yaml configs/train.yaml
# ...