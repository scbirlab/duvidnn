"""Data structures for using chemprop."""

from typing import Iterable, Mapping, Optional, Union
from dataclasses import dataclass, field, InitVar

from chemprop.data import Datum, MolGraph, TrainingBatch
import numpy as np
import torch


@dataclass(repr=False, eq=False, slots=True)
class DuvidaBatchMolGraph:
    """A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`s.

    It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
    class is intended for use with data loading, so it uses :obj:`~torch.Tensor`s to store data
    """

    mgs: InitVar[Iterable[MolGraph]]
    """A list of individual :class:`MolGraph`s to be batched together"""
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
        except TypeError:
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
        """the number of individual :class:`MolGraph`s in this batch"""
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
