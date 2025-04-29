"""Chemprop message-passing neural network."""

from abc import ABC, abstractmethod

from carabiner import print_err
from chemprop.models import MPNN
from chemprop.nn import (
    BondMessagePassing, 
    EvidentialFFN, 
    NormAggregation, 
    RegressionFFN
)
import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer

from .data import (
    DuvidaTrainingBatch, 
    _collate_training_batch_for_forward
)
from ..utils.ensemble import TorchEnsembleMixin
from ..utils.lt import LightningMixin


class ChempropBase(Module, ABC):

    def __init__(
        self, 
        n_input: int, 
        n_hidden: int = 1,
        n_units: int = 16, 
        mp_units: int = 300,
        mp_hidden: int = 3,
        mp_activation: str = "elu",
        learning_rate: float = .01,
        dropout: float = 0., 
        activation: str = "ELU",  # Smooth activation to prevent gradient collapse
        n_out: int = 1, 
        batch_norm: bool = False,
        evidential: bool = False,
        warmup_epochs: int = 2,
        *args, **kwargs
    ):
        super().__init__()
        self._extra_args = args
        self._extra_kwargs = kwargs
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.mp_units = mp_units
        self.mp_hidden = mp_hidden
        self.mp_activation = mp_activation
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.activation = activation
        self.n_out = n_out
        self.batch_norm = batch_norm
        self.evidential = evidential
        self.warmup_epochs = warmup_epochs
        self.n_mp_units: int = None

        if self.batch_norm:
            # TODO: fix this issue.
            print_err("Warning: using batch norm, which will not allow calculation of doubtscore or information sensitivity.")

    def build_model(self) -> Module:
        message_passing_layer = BondMessagePassing(
            d_h=self.mp_units,        
            depth=self.mp_hidden,
            activation=self.mp_activation, 
        )
        self.n_mp_units = message_passing_layer.output_dim
        aggregation_layer = NormAggregation()
        predictor_kwargs = {
            "n_tasks": self.n_out,
            "input_dim": self.n_input + self.n_mp_units,
            "hidden_dim": self.n_units,
            "n_layers": self.n_hidden,
            "dropout": self.dropout,
            "activation": self.activation,
        }
        predictor_layer = (
            EvidentialFFN(**predictor_kwargs) if self.evidential 
            else RegressionFFN(**predictor_kwargs)
        )
        return MPNN(
            message_passing_layer, 
            aggregation_layer, 
            predictor_layer, 
            warmup_epochs=self.warmup_epochs,
            batch_norm=self.batch_norm,
            init_lr=self.learning_rate, 
            max_lr=self.learning_rate * 10., 
            final_lr=self.learning_rate,
        )

    @abstractmethod
    def forward(self, x: DuvidaTrainingBatch) -> torch.Tensor:
        pass


class ChempropEnsemble(TorchEnsembleMixin, ChempropBase, LightningMixin):

    def __init__(
        self, 
        ensemble_size: int = 1,
        learning_rate: float = .01,
        optimizer: Optimizer = Adam,
        reduce_lr_on_plateau: bool = True,
        reduce_lr_patience: int = 10,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._init_ensemble(ensemble_size)
        self._init_lightning(
            optimizer=optimizer, 
            learning_rate=learning_rate, 
            model_attr='_model_ensemble',
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )

    def create_module(self):
        return self.build_model()

    def forward(self, x: DuvidaTrainingBatch) -> torch.Tensor:
        if not isinstance(x, DuvidaTrainingBatch):
            for p in self.parameters():
                device = p.device
                break
            x = _collate_training_batch_for_forward(x, device=device)
        x = (x.bmg, x.V_d, x.X_d)
        # Stack outputs to shape [batch, n_out, ensemble_size]
        out = torch.stack([
            module(*x) for _, module in self.model_ensemble.items()
        ], dim=-1)
        # If n_out == 1 (scalar), squeeze the middle axis to get [batch, ensemble_size]
        if out.size(-2) == 1:
            out = out.squeeze(-2)
        return out
