"""Ephemeral Lightning integration for training PyTorch models."""

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn, Tensor
from torch.optim import Adam
from torchmetrics import Metric, MetricCollection

from ..invoke import ModelInvoker

@dataclass
class PredictionTargetLoss:
    prediction: Tensor
    target: Tensor
    loss: Tensor


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
        metrics: Mapping[str, Metric] | Iterable[Metric] | None = None,
        metric_mask: Callable[[Mapping], Tensor] | None = None
    ) -> None:
        super().__init__()

        self.model = model
        self.invoker = invoker
        self.loss_fn = loss
        self.optimizer_cls = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_inputs = dict(loss_inputs or {})
        self.metric_mask = metric_mask

        metrics = metrics or {}
        if isinstance(metrics, (tuple, list)):
            metrics = {f.__name__: f for f in metrics}

        self.train_metrics = MetricCollection(deepcopy(metrics))
        self.val_metrics = MetricCollection(deepcopy(metrics))
        self.test_metrics = MetricCollection(deepcopy(metrics))

    def _loss_kwargs(self, batch) -> dict:
        return {
            argument: batch[column]
            for argument, column
            in self.loss_inputs.items()
        }

    def _prediction_target_loss(self, batch):
        prediction, target = self.invoker.supervised(batch)
        loss = self.loss_fn(
            prediction,
            target,
            **self._loss_kwargs(batch),
        )
        return PredictionTargetLoss(prediction, target, loss)

    def loss(self, batch):
        prediction, target = self.invoker.supervised(batch)
        return self._prediction_target_loss(batch).loss

    def _update_metrics(
        self,
        prediction,
        target,
        batch,
        stage: str
    ) -> None:
        metrics = (
            self.train_metrics
            if stage == "train"
            else self.val_metrics
        )

        if len(metrics) == 0:
            return None

        if self.metric_mask is not None:
            mask = self.metric_mask(batch)

            prediction = prediction[mask]
            target = target[mask]

            if prediction.numel() == 0:
                return None

        metrics.update(prediction, target)
        return metrics

    def _log_metrics(
        self,
        metrics,
        stage: str,
    ) -> None:
        for name, metric in metrics.items():
            self.log(
                f"{stage}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def _step(
        self,
        batch,
        stage: str
    ):
        prediction_target_loss = self._prediction_target_loss(batch)

        self.log(
            f"{stage}_loss",
            prediction_target_loss.loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        metrics = self._update_metrics(
            prediction=prediction_target_loss.prediction,
            target=prediction_target_loss.target,
            batch=batch,
            stage=stage,
        )
        self._log_metrics(metrics, stage=stage)
        return prediction_target_loss.loss

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
        metrics: Mapping[str, Metric] | Iterable[Metric] | None = None,
        metric_mask: Callable[[Mapping], Tensor] | None = None,
        **trainer_kwargs
    ) -> None:
        self.max_epochs = max_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_inputs = dict(loss_inputs or {})
        self.early_stopping = early_stopping
        self.metrics = dict(metrics or {})
        self.metric_mask = metric_mask
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
            metrics=self.metrics,
            metric_mask=self.metric_mask,
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