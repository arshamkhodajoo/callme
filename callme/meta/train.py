from typing import Callable, Union

import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["ProtonetLightningTrainer"]


class ProtonetLightningTrainer(pl.LightningModule):
    """Lightning Module for Prototypical Neural Train"""

    def __init__(self, model: Callable, loss_fn: Callable, optimizer: Optimizer, n_support: int, lr_scheduler: Union[_LRScheduler, None] = None) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.n_support = n_support

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer

        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, batch, batch_idx):
        samples, labels = batch
        embedding_output = self.model(samples)
        loss_value, accuracy_value = self.loss_fn(
            embedding_output, labels, self.n_support)
        self.log("training_loss", loss_value)
        self.log("training_accuracy", accuracy_value)

        return loss_value

    def validation_step(self, batch, batch_idx):
        samples, labels = batch
        embedding_output = self.model(samples)
        loss_value, accuracy_value = self.loss_fn(
            embedding_output, labels, self.n_support)
        self.log("validation_loss", loss_value)
        self.log("validation_accuracy", accuracy_value)


class TripletNetworkdTrainer(pl.LightningModule):
    """Lightning module for Triplet Network"""

    def __init__(self, embedding_model: Callable, loss_fn: Callable, optimizer, scheduler) -> None:
        super().__init__()
        self.model = embedding_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        triplet_samples, labels = batch
        embedding = self.model(*triplet_samples)
        loss_value = self.loss_fn(*embedding)
        self.log("trainig_loss", loss_value)
        return loss_value

    def validation_step(self, batch, batch_idx):
        triplet_samples, labels = batch
        embedding = self.model(*triplet_samples)
        loss_value = self.loss_fn(*embedding)
        self.log("validation_loss", loss_value)


class OnlineTripletNetworkTrainer(TripletNetworkdTrainer):
    """Lightning module for Triplet Network which works on online triplet loss"""

    def training_step(self, batch, batch_idx):
        mfccs, labels = batch
        embedding = self.model(mfccs)
        loss_value = self.loss_fn(embedding, labels)[0]
        self.log("online_training_loss", loss_value)
        return loss_value

    def validation_step(self, batch, batch_idx):
        mfccs, labels = batch
        embedding = self.model(mfccs)
        loss_value = self.loss_fn(embedding, labels)[0]
        self.log("online_validation_loss", loss_value)
