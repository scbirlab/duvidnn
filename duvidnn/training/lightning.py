"""Ephemeral Lightning integration for training PyTorch models."""

from typing import Any

from lightning.pytorch.callbacks import EarlyStopping
from torch.optim import Adam

from collections.abc import Callable, Mapping

import lightning as L
from torch import nn

from ..invoke import ModelInvoker


class LightningTask(L.LightningModule):
    """Adapt an arbitrary PyTorch model for Lightning training."""

    def __init__(
        self,
        model: nn.Module,
        invoker: ModelInvoker,
        loss: Callable,
        optimizer: Callable,
        optimizer_kwargs: Mapping | None = None,
        loss_inputs: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.invoker = invoker
        self.loss_fn = loss
        self.optimizer_cls = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_inputs = dict(loss_inputs or {})

    def _loss_kwargs(self, batch) -> dict:
        return {
            argument: batch[column]
            for argument, column
            in self.loss_inputs.items()
        }

    def loss(self, batch):
        prediction, target = self.invoker.supervised(batch)
        return self.loss_fn(
            prediction,
            target,
            **self._loss_kwargs(batch),
        )

    def _step(
        self,
        batch,
        stage: str
    ):
        loss = self.loss(batch)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(
        self,
        batch,
        batch_idx
    ):
        return self._step(batch, "train")

    def validation_step(
        self,
        batch,
        batch_idx
    ):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return self.optimizer_cls(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )


class Trainer:
    """Train arbitrary PyTorch models through Lightning."""

    def __init__(
        self,
        *,
        max_epochs: int = 100,
        loss: Callable,
        optimizer: Callable = Adam,
        optimizer_kwargs: Mapping | None = None,
        loss_inputs: Mapping[str, str] | None = None,
        early_stopping: int | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        self.max_epochs = max_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_inputs = dict(loss_inputs or {})
        self.early_stopping = early_stopping
        self.trainer_kwargs = trainer_kwargs

        self._trainer = None

    def fit(
        self,
        model: nn.Module,
        invoker: ModelInvoker,
        train_dataloader,
        val_dataloader=None,
        trainer: L.Trainer | None = None
    ) -> nn.Module:
        task = LightningTask(
            model=model,
            invoker=invoker,
            loss=self.loss,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            loss_inputs=self.loss_inputs,
        )

        trainer_kwargs = dict(self.trainer_kwargs)

        callbacks = list(trainer_kwargs.pop("callbacks", []))

        if (
            self.early_stopping is not None
            and val_dataloader is not None
        ):
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping,
                    mode="min",
                )
            )

        if trainer is None:
            self._trainer = L.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                **trainer_kwargs,
            )
        elif isinstance(trainer, L.Trainer):
            self._trainer = trainer
        else:
            raise ValueError(
                "If provided, trainer must be a Lightning Trainer instance, "
                f"but was {type(trainer)}: {trainer}"
            )

        self._trainer.fit(
            task,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        return model