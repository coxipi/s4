import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from charactertokenizer import CharacterTokenizer
from models.s4.s4 import (
    S4Block as S4,  # Can use full version instead of minimal S4D standalone below
)
from models.s4.s4d import S4D

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split(".")[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        N=64,
        n_layers=4,
        dropout=0.1,
        prenorm=False,
        lr=0.01,
        mode="dplr",
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
                S4(
                    d_model,
                    dropout=dropout,
                    d_state=self.N,
                    transposed=True,
                    lr=min(0.001, self.lr),
                    mode=self.mode,
                )
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
        mode="dplr",
        padding_idx=None,
    ):
        super().__init__(
            embedding_dim, d_output, d_model, n_layers, dropout, prenorm, lr, mode
        )
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
        N=64,
        n_layers=4,
        dropout=0.1,
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
                S4D(
                    d_model,
                    dropout=dropout,
                    d_state=self.N,
                    transposed=True,
                    lr=min(0.001, self.lr),
                )
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
        hidden_size=256,
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
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        # Not giving any possible hidden state initialization other than all zeroes (default)
        # If the goal is to compare with the given s4 implementation,
        # then it seems fair we should have something more specialized in this respect.
        x, _ = self.rnn(x)  # (B, L, d_model) -> (B, L, d_model)
        # Instead of having dim(y) = dim(x), we now pool, as in S4
        # Our task is classification and not copy, makes sense
        # (L * d_input) = (N*N pixels * 1 grayshade value) -> (d_ouput) = 10 (digits)
        x = x.mean(dim=1)  # (B, d_model) via average pooling over sequence
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x


# class RNNModelWithEmbedding(RNNModel):
#     def __init__(
#         self,
#         d_input,
#         embedding_dim,
#         d_output=10,
#         d_model=256,
#         hidden_size=256,
#         n_layers=4,
#         dropout=0.1,
#         padding_idx=None,
#     ):
#         super().__init__(
#             embedding_dim, d_output, d_model, n_layers, dropout, prenorm, lr, mode
#         )
#         self.embedding = nn.Embedding(d_input, embedding_dim, padding_idx=padding_idx)

#     def forward(self, x):
#         x = self.embedding(x)
#         return super().forward(x)


class CPRNN_cell(nn.Module):
    """CP-Factorized LSTM, single cell.

    Args:
        input_size: Input size
        hidden_size: Dimension of hidden features.
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate
        gate: Gate function (activation from t to t+1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int,
        tokenizer,
        batch_first: bool,
        dropout: float,
        gate: callable,
        **kwargs
    ):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rank = rank
        self.gate = gate

        # Encoder using CP factors
        self.A = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.C = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.d = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tokenizer = tokenizer
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(
        self,
        inp: Union[torch.LongTensor, str],
        init_states: tuple = None,
        top_k: int = 1,
        device=torch.device("cpu"),
    ):
        with torch.no_grad():
            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError(
                        "Tokenizer not defined. Please provide a tokenizer to the model."
                    )
                x = (
                    torch.tensor(self.tokenizer.char_to_ix(inp))
                    .reshape(1, 1)
                    .to(device)
                )
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        A_prime = h @ self.A
        B_prime = x @ self.B

        h_next = self.gate(
            torch.einsum("br,br,hr -> bh", A_prime, B_prime, self.C)
            + h @ self.V
            + x @ self.U
            + self.d
        )

        return h_next


class DeepCPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        d_input: Dimension of hidden features.
        embedding_dim: Dimension of the embedding (`rescale`=False) or encoding (`rescale`=True)
        d_output: Dimension of output
        hidden_size: Dimension of hidden features.
        n_layers: Number of layers
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate
        dropout_between_layers: Dropout between layers or not
        activation: Activation function (activation from l to l+1)
        readout_activation: Readout activation function
        gate: Gate function (activation from t to t+1)
        rescale: Whether to rescale the input ([0,255] -> [0,1.0]) or not
    """

    def __init__(
        self,
        d_input,
        embedding_dim,
        d_output=10,
        hidden_size=256,
        n_layers: int = 4,
        rank: int = 8,
        tokenizer: CharacterTokenizer = None,
        batch_first: bool = True,
        dropout: float = 0.5,
        dropout_between_layers: bool = False,
        activation="identity",
        readout_activation="identity",
        gate: str = "identity",
        rescale: bool = True,
    ):
        super().__init__()

        self.dropout = dropout
        self.dropout_between_layers = dropout_between_layers
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.rescale = rescale
        self.d_input = d_input if self.rescale else d_input * 256
        self.d_output = d_output
        self.n_layers = n_layers

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'identity'")

        if readout_activation == "relu":
            self.readout_activation_fn = F.relu
        elif readout_activation == "tanh":
            self.readout_activation_fn = torch.tanh
        elif readout_activation == "identity":
            self.readout_activation_fn = lambda x: x
        else:
            raise ValueError("readout_activation must be 'relu', 'tanh', or 'identity'")

        self.rank = rank
        self.gate = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "identity": lambda x: x,
        }[gate]

        # Define embedding and decoder layers
        if self.rescale:
            self.embedding = nn.Linear(self.d_input, self.embedding_dim)
        else:
            self.embedding = nn.Embedding(self.d_input, self.embedding_dim)
        self.init_weights()

        self.cprnn_cell = CPRNN_cell
        self.cprnn_layers = nn.ModuleList(
            [
                self.cprnn_cell(
                    self.embedding_dim if i == 0 else self.hidden_size,
                    self.hidden_size,
                    self.rank,
                    self.tokenizer,
                    self.batch_first,
                    self.dropout,
                    self.gate,
                )
                for i in range(self.n_layers)
            ]
        )

        self.decoder = nn.Linear(self.hidden_size, self.d_output)

    # what is this used for? not clear
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(
        self,
        inp: Union[torch.LongTensor, str],
        init_states: tuple = None,
        top_k: int = 1,
        device=torch.device("cpu"),
    ):
        with torch.no_grad():
            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError(
                        "Tokenizer not defined. Please provide a tokenizer to the model."
                    )
                x = (
                    torch.tensor(self.tokenizer.char_to_ix(inp))
                    .reshape(1, 1)
                    .to(device)
                )
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, x: torch.LongTensor):
        if self.batch_first:
            x = x.transpose(0, 1)

        if self.embedding_dim is not None:
            # if len(x.shape) != 2:
            #     raise ValueError(
            #         "Expected input tensor of order 2, but got order {} tensor instead".format(
            #             len(x.shape)
            #         )
            #     )
            x = self.embedding(x)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        seq_length, batch_size, _ = x.size()
        device = x.device
        # not input possible for init states for now, like DeepRNN
        h = self.init_hidden(batch_size, device)
        outputs = []

        for t in range(seq_length):
            out, h = self.forward_one_timestep(x[t], h)
            # Below: out.unsqueeze(0) -> out
            # unsqueeze(0) was messing the computation
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)
        # I don't know what this does but I'm keeping it
        outputs = outputs.contiguous()
        # This reproduces CPRNN behaviour: Dropout after last layer, after stacking
        # just before decoding
        if not self.dropout_between_layers:
            outputs = nn.Dropout(self.dropout)(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        outputs = outputs.mean(dim=1)  # (B, d_model) via average pooling over sequence
        outputs = self.decoder(outputs)
        return outputs

    def forward_one_timestep(self, x, h):
        h_depth = []
        h_rec = []
        for i, cprnn in enumerate(self.cprnn_layers):
            h_i = cprnn(x if i == 0 else h_depth[i - 1], h[i])
            h_rec.append(h_i)
            h_i = self.activation_fn(h_i)
            # This reproduces S4 behaviour: Dropout between layers,
            # after activation
            if self.dropout_between_layers:
                h_i = nn.Dropout(self.dropout)(h_i)
            h_depth.append(h_i)
        out = h_depth[-1]
        out = self.readout_activation_fn(out)

        return out, torch.stack(h_rec)

    def init_hidden(self, batch_size, device=torch.device("cpu")):
        return [
            torch.zeros(batch_size, self.hidden_size).to(device)
            for _ in range(self.n_layers)
        ]


# if __name__ == "__main__":
#     # Example usage
#     vocab_size = 100
#     input_size = 50
#     hidden_size = 128
#     rank = 8
#     batch_size = 32
#     seq_len = 10

#     model = CPRNN(input_size, hidden_size, vocab_size, rank=rank)
#     x = torch.randint(0, vocab_size, (batch_size, seq_len))
#     output = model(x)
#     print(output.shape)  # Should be [batch_size, seq_len, vocab_size]
