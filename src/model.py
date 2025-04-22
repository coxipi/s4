import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

import numpy as np 
import random
from typing import Any,Union

import torchvision
import torchvision.transforms as transforms

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from tqdm.auto import tqdm
import math

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        N = 64,
        n_layers=4,
        dropout=0.1, # changed the default value in the class to reflect the default value in the script
        prenorm=False,
        lr=0.01, 
        mode = 'dplr'
    ):
        super(S4Model, self).__init__()

        self.prenorm = prenorm
        self.N = N
        self.lr = lr 
        self.mode = mode

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, d_state=self.N, transposed=True, lr=min(0.001, self.lr), mode=self.mode)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Isn't it weird mixing encoding and embedding?
class S4ModelWithEmbedding(S4Model):
    def __init__(
        self,
        d_input,
        embedding_dim,
        d_output=10,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        prenorm=False,
        lr=0.01, 
        mode = 'dplr',
        padding_idx=None,
    ):
        super().__init__(embedding_dim, d_output, d_model, n_layers, dropout, prenorm, lr, mode)
        self.embedding = nn.Embedding(d_input, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = self.embedding(x)
        return super().forward(x)

class S4DModel(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        N = 64,
        n_layers=4,
        dropout=0.1, # changed the default value in the class to reflect the default value in the script
        prenorm=False,
        lr=0.01, 
    ):
        super(S4DModel, self).__init__()

        self.prenorm = prenorm
        self.N = N
        self.lr = lr 

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, d_state=self.N, transposed=True, lr=min(0.001, self.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


class RNNModel(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        hidden_size = 256,
        n_layers=4,
        dropout=0.1,
    ):
        super(RNNModel, self).__init__()
        

        self.encoder = nn.Linear(d_input, d_model)

        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input: x of shape (B, L, d_input)
        Output: (B, d_output)
        """
        x = self.encoder(x)       # (B, L, d_input) -> (B, L, d_model)
        # Not giving any possible hidden state initialization other than all zeroes (default)
        # If the goal is to compare with the given s4 implementation, then it seems fair we should have
        # something more specialized in this respect.
        x, _ = self.rnn(x)        # (B, L, d_model) -> (B, L, d_model)
        # Instead of having dim(y) = dim(x), we now pool, as in S4
        # Our task is classification and not copy, makes sense 
        # (L * d_input) = (N*N pixels * 1 grayshade value) -> (d_ouput) = 10 (digits)
        x = x.mean(dim=1)         # (B, d_model) via average pooling over sequence
        x = self.decoder(x)       # (B, d_model) -> (B, d_output)
        return x


class RNNModelWithEmbedding(RNNModel):
    def __init__(
        self,
        d_input,
        embedding_dim,
        d_output=10,
        d_model=256,
        hidden_size=256,
        n_layers=4,
        dropout=0.1,
        padding_idx=None,
    ):
        super().__init__(embedding_dim, d_output, d_model, n_layers, dropout, prenorm, lr, mode)
        self.embedding = nn.Embedding(d_input, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = self.embedding(x)
        return super().forward(x)
