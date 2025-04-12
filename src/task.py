import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from beartype import beartype
from lightning import Callback, LightningModule
from torch import Tensor
from torch.nn import functional as F
import torch.optim as optim
import torch.nn as nn


def sequential_ce_loss(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -1,
) -> Tensor:
    """
    Compute the negative log likelihood loss for a sequence of predictions.
    Args:
        input (Tensor): The predicted probabilities (shape: batch_size, seq_len, vocab_size).
        target (Tensor): The target indices (shape: batch_size, seq_len, vocab_size).
    Returns:
        Tensor: The computed loss.
    """
    # Reshape input and target to match the expected dimensions
    input = input.float()
    input = input.reshape(-1, input.size(-1))
    target = target.view(-1)

    # Compute the negative log likelihood loss
    loss = F.cross_entropy(input, target, ignore_index=ignore_index)
    return loss

def mse_loss(target: Any, preds: Any) -> Tensor:
    """Loss function to be used in the training loop."""
    loss = F.mse_loss(preds, target)
    return loss

def l1_loss(target: Any, preds: Any) -> Tensor:
    """Loss function to be used in the training loop."""
    loss = F.l1_loss(preds, target)
    return loss

def setup_optimizer_s4(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


class ClassifyTask(LightningModule):
    @beartype
    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
        ignore_index = -1,
        criterion = "CrossEntropy", 
        epochs = 100, 
        weight_decay = 0.01, #useless for RNN
    ):
        super().__init__()
        self.model = model
        self.ignore_index = ignore_index
        self.save_hyperparameters()  # ignore the instance of nn.Module that are already stored ignore=['my_module'])
        self.epochs = epochs
        self.weight_decay = weight_decay
        
        # We could directly ask for loss function instead of passing by
        # the criterion I wanted to keep the same structure as the s4 repo
        self.criterion = {
            "CrossEntropy":nn.CrossEntropyLoss(),
            "MSE":nn.MSELoss(),
            "L1":nn.L1Loss(),
            }.get(criterion)
        self.loss_function = lambda out,tgt : self.criterion(out,tgt)

    @beartype
    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self.model, 'batch_first') and not self.model.batch_first:
            x = x.permute(1, 0, 2)  

        out = self.model(x)

        if isinstance(out, tuple):
            out = out[0]

        if out.size(0) == x.size(1): 
            out = out.permute(1, 0, 2)

        return out

    def configure_optimizers(self) -> None:
        if self.model._get_name() != "S4Model": 
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.model._get_name() == "S4Model": 
            opt,sch = setup_optimizer_s4(self.model, self.hparams.lr, self.weight_decay, self.epochs)
            return {"optimizer": opt, "lr_scheduler":sch}


    @beartype
    def training_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)

        # Compute loss only on nonzero vectors
        loss = self.loss_function(preds, y)

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Compute loss
        loss = self.loss_function(preds, y)
        # Compute mean accuracy
        mask = y != self.ignore_index 
        accuracy = torch.logical_and(preds.argmax(dim=-1) == y, mask).float().sum() / mask.float().sum() if mask.any() else torch.tensor(0.0)
        # Log loss
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Compute loss
        loss = self.loss_function(preds, y)
        # Compute mean accuracy
        mask = y != self.ignore_index 
        accuracy = torch.logical_and(preds.argmax(dim=-1) == y, mask).float().sum() / mask.float().sum() if mask.any() else torch.tensor(0.0)
        # Log loss
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss
