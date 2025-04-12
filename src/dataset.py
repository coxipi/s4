import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        # keeping fixed seed for now
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


class MNISTDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        val_split = 0.1,
        grayscale=True, # just a dummy option to have compatible signatures
   ):
        if not grayscale: 
            raise ValueError("MNIST dataset only supports `grayscale==True`. " \
            f"Got {grayscale}")
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())
        ])
        transform_train = transform_test = transform    
        train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_train)
        train_dataset, _ = split_train_val(train_dataset, val_split=0.1)

        val_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform_test)
        _, val_dataset = split_train_val(val_dataset, val_split=0.1)

        test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform_test)
        
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True, # probably not necessary since the shuffle was done with the dataset?
            collate_fn=None,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

    @beartype
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

class CIFAR10DataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        val_split = 0.1,
        grayscale=True,
   ):   
        self.grayscale = grayscale
        if self.grayscale:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
                transforms.Lambda(lambda x: x.view(1, 1024).t())
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: x.view(3, 1024).t())
            ])

        # S4 is trained on sequences with no data augmentation!
        transform_train = transform_test = transform

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform_train)
        train_dataset, _ = split_train_val(train_dataset, val_split=0.1)

        val_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform_test)
        _, val_dataset = split_train_val(val_dataset, val_split=0.1)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=False, download=True, transform=transform_test)

        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True, # probably not necessary since the shuffle was done with the dataset?
            collate_fn=None,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

    @beartype
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )
