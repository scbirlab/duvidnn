"""Mixins to enable Lightning features."""

from typing import Mapping, Optional

from lightning import LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from ...functions import mse_loss
from ....stateless.typing import Array, ArrayLike


class LightningMixin(LightningModule):

    # optimizer: Optimizer = None
    # learning_rate: float = .01
    model_attr: str = None
    # reduce_lr_on_plateau: bool = False
    # reduce_lr_patience: int = 10

    def _init_lightning(
        self, 
        model_attr: str,
        optimizer: Optional[Optimizer] = None, 
        learning_rate: float = .01,
        reduce_lr_on_plateau: bool = False,
        reduce_lr_patience: int = 10
    ) -> None:
        if issubclass(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(
                "Optimizer must by a PyTorch Optimizer, but "
                f"'{optimizer}' inherits from \'{' / '.join(optimizer.__mro__)}\'.")
        self.learning_rate = learning_rate
        if model_attr in dir(self):
            self.model_attr = model_attr
        else:
            raise KeyError(f"The model attribute '{model_attr}' does not exist!")
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.reduce_lr_patience = reduce_lr_patience
        return None

    def get_loss(
        self, 
        batch: Mapping[str, ArrayLike]
    ) -> Array:
        inputs, outputs = batch['inputs'], batch['labels']
        predicted = self(inputs)
        return mse_loss(predicted, outputs)

    def training_step(
        self, 
        batch: Mapping[str, ArrayLike], 
        batch_idx: int
    ) -> Array:
        loss = self.get_loss(batch)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch['labels'].shape[-1])
        return loss

    def validation_step(
        self, 
        batch: Mapping[str, ArrayLike], 
        batch_idx: int
    ) -> Array:
        loss = self.get_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch['labels'].shape[-1])
        return loss

    def on_validation_epoch_end(self):
        # Log the learning rate.
        if self.reduce_lr_on_plateau:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(
            getattr(self, self.model_attr).parameters(), 
            lr=self.learning_rate,
        )
        if self.reduce_lr_on_plateau:
            return {
                "optimizer": optimizer,
                "lr_scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.reduce_lr_patience,
                ),
                "monitor": "val_loss",
            }
        else:
            return optimizer
