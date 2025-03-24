"""Chemprop message-passing neural network."""

from typing import Callable, Iterable, List, Mapping, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar

from carabiner import print_err
from chemprop.data import BatchMolGraph, Datum, MolGraph, TrainingBatch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, EvidentialFFN, NormAggregation, RegressionFFN
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer

from .ensemble import TorchEnsembleMixin
from .lt import LightningMixin
from ..nn import mse_loss, ModelBox
from ...base_classes import VarianceMixin
from ...stateless.typing import Array, ArrayLike


@dataclass(repr=False, eq=False, slots=True)
class DuvidaBatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data
    """

    mgs: InitVar[Iterable[MolGraph]]
    """A list of individual :class:`MolGraph`\s to be batched together"""
    V: torch.Tensor = field(init=False)
    """the atom feature matrix"""
    E: torch.Tensor = field(init=False)
    """the bond feature matrix"""
    edge_index: torch.Tensor = field(init=False)
    """an tensor of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: torch.Tensor = field(init=False)
    """A tensor of shape ``E`` that maps from an edge index to the index of the source of the
    reverse edge in the ``edge_index`` attribute."""
    batch: torch.Tensor = field(init=False)
    """the index of the parent :class:`MolGraph` in the batched graph"""

    __size: int = field(init=False)

    def __post_init__(self, mgs: Iterable[MolGraph]):
        self.__size = len(mgs)

        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            # batch_indexes.append(torch.ones((len(mg.V),)) * i)
            batch_indexes.append([i] * len(mg.V))

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.concat(Vs)
        try:
            self.E = torch.concat(Es)
        except TypeError as e:
            Es = [
                E for E in Es if not isinstance(E, list)
            ]
            if len(Es) > 0:
                self.E = torch.concat(Es)
            else:
                self.E = torch.empty((0,))
        self.edge_index = torch.hstack(edge_indexes).long()
        self.rev_edge_index = torch.concat(rev_edge_indexes).long()
        # self.batch = torch.concat(batch_indexes).long()
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long()

    def __len__(self) -> int:
        """the number of individual :class:`MolGraph`\s in this batch"""
        return self.__size

    def to(self, device: Union[str, torch.device]):
        self.V = self.V.to(device)
        self.E = self.E.to(device)
        self.edge_index = self.edge_index.to(device)
        self.rev_edge_index = self.rev_edge_index.to(device)
        self.batch = self.batch.to(device)


class DuvidaTrainingBatch(TrainingBatch):

    def to(self, device: Union[str, torch.device]) -> TrainingBatch:
        for p in self:
            if isinstance(p, (torch.Tensor, DuvidaBatchMolGraph)):
                p.to(device)
        return self


def collate_batch(
    batch: Iterable[Datum], 
    device: Optional[Union[str, torch.device]] = None
) -> DuvidaTrainingBatch:
    """
    """
    
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)
    bmg = DuvidaBatchMolGraph(mgs)
    training_batch = DuvidaTrainingBatch(
        bmg,
        None if V_ds[0] is None else torch.concat(V_ds),
        None if x_ds[0] is None else torch.stack(x_ds),
        None if ys[0] is None else torch.concat(ys).unsqueeze(1),
        weights,
        None if lt_masks[0] is None else lt_masks,
        None if gt_masks[0] is None else gt_masks,
    )
    if device is not None:
        training_batch.to(device)
    return training_batch


def _collate_training_batch_for_forward(
    batch: Iterable[Mapping],
    for_dataloader: bool = False,
    _in_key: str = "inputs",
    _out_key: str = "labels", 
    device: str = None
) -> TrainingBatch:
    new_batch = []
    for datum in batch:
        new_datum = {}
        if for_dataloader and _in_key in datum:
            _datum = datum[_in_key]
        else:
            _datum = datum
        for key, val in _datum.items():
            if isinstance(val, dict):
                new_val = MolGraph(**val)
            else:
                new_val = val
            new_datum[key] = new_val
        new_batch.append(Datum(**new_datum))
    collated = collate_batch(new_batch, device=device)
    if for_dataloader:
        return {
            _in_key: collated,
            _out_key: torch.concat([d[_out_key] for d in batch]).unsqueeze(1),
        }
    else:
        return collated


class ChempropBase(Module, ABC):

    def __init__(
        self, 
        n_input: int, 
        n_hidden: int = 1,
        n_units: int = 16, 
        mp_units: int = 300,
        mp_hidden: int = 3,
        mp_activation: str = "relu",
        learning_rate: float = .01,
        dropout: float = 0., 
        activation: str = "ELU",  # Smooth activation to prevent gradient collapse
        n_out: int = 1, 
        batch_norm: bool = False,
        evidential: bool = False,
        warmup_epochs: int = 2,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
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
            print_err("Warning: using batch norm, which will not allow calculation of doubtscore or information sensitivity.")

    def create_model(self) -> Module:
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
        predictor_layer = EvidentialFFN(**predictor_kwargs) if self.evidential else RegressionFFN(**predictor_kwargs)
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
    def forward(self, x: ArrayLike) -> Array:
        pass


class ChempropEnsemble(ChempropBase, TorchEnsembleMixin, LightningMixin):

    def __init__(
        self, 
        ensemble_size: int = 1,
        learning_rate: float = 1e-4,
        optimizer: Optimizer = Adam,
        reduce_lr_on_plateau: bool = False,
        reduce_lr_patience: int = 10,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._init_ensemble(ensemble_size)
        self._init_lightning(
            optimizer=optimizer, 
            learning_rate=learning_rate, 
            model_attr='_model_ensemble',
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )

    def create_module(self):
        return self.create_model()

    def forward(self, x: ArrayLike) -> Array:
        if not isinstance(x, DuvidaTrainingBatch):
            for p in self.parameters():
                dev = p.device
                break
            x = _collate_training_batch_for_forward(x, device=dev)
        x = (x.bmg, x.V_d, x.X_d)
        return torch.concat([
            module(*x) for _, module in self.model_ensemble.items()
        ], dim=-1)