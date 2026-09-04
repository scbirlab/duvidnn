"""Ephemeral Lightning integration for training PyTorch models."""

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import warnings

import lightning as L
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.warnings import PossibleUserWarning
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
        metric_mask: Callable[[Mapping], Tensor] | None = None,
        regularizer: Callable | Iterable[Callable] | None = None,
        scheduler: Callable | None = None,
        scheduler_kwargs: Mapping | None = None,
        scheduler_monitor: str | None = None
    ) -> None:
        super().__init__()

        self.model = model
        self.invoker = invoker
        self.loss_fn = loss
        self.optimizer_cls = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_inputs = dict(loss_inputs or {})

        if regularizer is None:
            regularizer = ()
        elif callable(regularizer):
            regularizer = (regularizer,)
        else:
            regularizer = tuple(regularizer)
        self.regularizers = regularizer

        self.scheduler_cls = scheduler
        self.scheduler_kwargs = dict(scheduler_kwargs or {})
        self.scheduler_monitor = scheduler_monitor

        self.metric_mask = metric_mask

        metrics = metrics or {}
        if isinstance(metrics, (tuple, list)):
            metrics = {
                (f.__name__ if hasattr(f, "__name__") else str(f)): f 
                for f in metrics
            }

        self.train_metrics = MetricCollection(deepcopy(metrics))
        self.val_metrics = MetricCollection(deepcopy(metrics))

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
        for regularizer in self.regularizers:
            loss = loss + regularizer(self.model)
        return PredictionTargetLoss(prediction, target, loss)

    def loss(self, batch):
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
        metrics: Mapping[str, Metric] | None,
        stage: str,
        batch_size: int
    ) -> None:
        if metrics is None:
            return None
        for name, metric in metrics.items():
            self.log(
                f"{stage}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _step(
        self,
        batch,
        stage: str
    ):
        prediction_target_loss = self._prediction_target_loss(batch)
        batch_size = prediction_target_loss.target.shape[0]

        self.log(
            f"{stage}_loss",
            prediction_target_loss.loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        metrics = self._update_metrics(
            prediction=prediction_target_loss.prediction,
            target=prediction_target_loss.target,
            batch=batch,
            stage=stage,
        )
        self._log_metrics(
            metrics, 
            stage=stage,
            batch_size=batch_size,
        )
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
        optimizer = self.optimizer_cls(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )

        if self.scheduler_cls is None:
            return optimizer

        scheduler = self.scheduler_cls(
            optimizer,
            **self.scheduler_kwargs,
        )
        scheduler_config = {"scheduler": scheduler}
        if self.scheduler_monitor is not None:
            scheduler_config["monitor"] = self.scheduler_monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }


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
        regularizer: Callable | Iterable[Callable] | None = None,
        scheduler: Callable | None = None,
        scheduler_kwargs: Mapping | None = None,
        scheduler_monitor: str | None = None,
        **trainer_kwargs
    ) -> None:
        self.max_epochs = max_epochs
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.loss_inputs = dict(loss_inputs or {})

        self.regularizer = regularizer
        self.scheduler = scheduler
        self.scheduler_kwargs = dict(scheduler_kwargs or {})
        self.scheduler_monitor = scheduler_monitor

        self.early_stopping = early_stopping
        metrics = metrics or {}
        if isinstance(metrics, (tuple, list)):
            metrics = {
                (f.__name__ if hasattr(f, "__name__") else str(f)): f 
                for f in metrics
            }
        self.metrics = metrics
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
            regularizer=self.regularizer,
            scheduler=self.scheduler,
            scheduler_kwargs=self.scheduler_kwargs,
            scheduler_monitor=self.scheduler_monitor,
            metrics=self.metrics,
            metric_mask=self.metric_mask,
        )

        trainer_kwargs = dict(self.trainer_kwargs)
        if val_dataloader is None:
            trainer_kwargs.setdefault("limit_val_batches", 0)
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

        plugins = list(trainer_kwargs.pop("plugins", []))
        if not any(
            isinstance(plugin, LightningEnvironment)
            for plugin in plugins
        ):
            plugins.append(LightningEnvironment())
        if trainer is None:
            self._trainer = L.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                plugins=plugins,
                **trainer_kwargs,
            )
        elif isinstance(trainer, L.Trainer):
            self._trainer = trainer
        else:
            raise ValueError(
                "If provided, trainer must be a Lightning Trainer instance, "
                f"but was {type(trainer)}: {trainer}"
            )

        with warnings.catch_warnings():
            for msg in (
                r".*does not have many workers.*",  # Reduce lightning chatter about dataloaders
                r".*defined a `validation_step` but have no `val_dataloader`.*",
            ):
                warnings.filterwarnings(
                    "ignore",
                    message=msg,
                    category=PossibleUserWarning,
                )
            self._trainer.fit(
                task,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
        return model
