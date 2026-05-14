"""Lasso (L1-regularized linear regression) models."""

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module

from torch.optim import Adam, Optimizer

from .utils.lt import LightningMixin
from .utils.ensemble import TorchEnsembleMixin


class TorchLassoBase(Module):
    """Linear model with L1 (lasso) penalty.

    Notes
    -----
    - This is *not* a different architecture vs linear regression; it's a different loss.
    - By default, we L1-penalize weights but not bias (typical lasso convention).
    """

    def __init__(
        self,
        n_input: int,
        n_out: int = 1,
        l1: float = 0.,
        no_bias: bool = False,
        _init_model: bool = True
    ):
        super().__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.l1 = float(l1)
        self.no_bias = no_bias
        self.linear = self.build_model() if _init_model else None

    def build_model(self):
        return Linear(self.n_input, self.n_out, bias=not self.no_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def l1_penalty(self) -> Tensor:
        # Penalize weights only (not bias) unless you *want* bias penalized too.
        return self.linear.weight.abs().sum()


class TorchLassoLightning(TorchLassoBase, LightningMixin):
    """Lasso linear model with Lightning training.
    
    """

    def __init__(
        self,
        learning_rate: float = 1e-2,
        optimizer: Optimizer = Adam,
        reduce_lr_on_plateau: bool = True,
        reduce_lr_patience: int = 10,
        l1: float = 0.,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # Train the parameter set in `linear`
        self._init_lightning(
            optimizer=optimizer,
            learning_rate=learning_rate,
            model_attr="linear",
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
            l1=l1,
        )


class TorchLassoEnsemble(TorchEnsembleMixin, TorchLassoLightning):
    """Ensemble of lasso linear models, consistent with your TorchEnsembleMixin pattern."""

    def __init__(
        self,
        ensemble_size: int = 1,
        learning_rate: float = 1e-2,
        optimizer: Optimizer = Adam,
        reduce_lr_on_plateau: bool = True,
        reduce_lr_patience: int = 10,
        l1: float = 0.,
        *args,
        **kwargs,
    ):
        super().__init__(_init_model=False, *args, **kwargs)
        self.save_hyperparameters()
        self._init_ensemble(ensemble_size)
        self._init_lightning(
            optimizer=optimizer,
            learning_rate=learning_rate,
            model_attr="_model_ensemble",
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
            l1=l1,
        )

    def l1_penalty(self) -> Tensor:
        s = 0.
        for module in self.model_ensemble:
            s += module.weight.abs().sum()
        return s

    def create_module(self) -> Module:
        return self.build_model()
