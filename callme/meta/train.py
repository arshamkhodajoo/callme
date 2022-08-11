from typing import Callable, Union

import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["ProtonetLightningTrainer"]


class ProtonetLightningTrainer(pl.LightningModule):
    """Lightning Module for Prototypical Neural Train"""

    def __init__(self, model: Callable, loss_fn: Callable, optimizer: Optimizer, lr_scheduler: Union[_LRScheduler, None]) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer

        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, batch, batch_idx):
        samples, labels = batch
        embedding_output = self.model(samples)
        loss_value, accuracy_value = self.loss_fn(embedding_output, labels)
        self.log("training_loss", loss_value)
        self.log("training_accuracy", accuracy_value)

        return loss_value

    def validation_step(self, batch, batch_idx):
        samples, labels = batch
        embedding_output = self.model(samples)
        loss_value, accuracy_value = self.loss_fn(embedding_output, labels)
        self.log("validation_loss", loss_value)
        self.log("validation_accuracy", accuracy_value)