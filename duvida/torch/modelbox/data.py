"""Modelbox data aspects."""

from typing import Callable, Dict, Mapping, Union
from functools import partial
from multiprocessing import cpu_count

from datasets import Dataset, IterableDataset
import torch
from torch.utils.data import DataLoader

from ...base.data import ChemMixinBase, DataMixinBase
from ...stateless.config import config

config.set_backend('torch', precision='float')

from ...stateless.typing import Array, ArrayLike
from ...stateless.utils import vmap
from ..models.chemprop.data import _collate_training_batch_for_forward

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataMixin(DataMixinBase):

    _format = 'pytorch'
    _format_kwargs = {
        'dtype': torch.float32, 
        'device': _DEVICE,
    }

    @staticmethod
    def make_dataloader(
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 16, 
        shuffle: bool = False,
        **kwargs
    ) -> DataLoader:
        # kwargs = {"num_workers": min(4, cpu_count() - 1)} | kwargs
        # kwargs = {"num_workers": 1} | kwargs
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            **kwargs,
        )


class TorchChemMixin(ChemMixinBase, DataMixin):

    @staticmethod
    def _get_max_sim(
        query: ArrayLike, 
        references: ArrayLike,
        aggregator: Callable[[ArrayLike], float] = torch.max
    ) -> Array:
        a_n_b = torch.sum(query.unsqueeze(0) * references, dim=-1, keepdim=True)
        similarities = a_n_b / (torch.sum(query) + torch.sum(references, dim=-1) - torch.sum(a_n_b, dim=-1)).unsqueeze(-1)
        return aggregator(similarities)

    @staticmethod
    def _get_nn_tanimoto(
        x: Mapping[str, ArrayLike],
        refs_data: Mapping[str, ArrayLike],
        results_column: str,
        _in_key: str,
        _sim_fn: Callable[[ArrayLike, ArrayLike], float]
    ) -> Dict[str, Array]:
        query_fps = x[_in_key]
        refs = refs_data[_in_key]
        results = vmap(_sim_fn, in_axes=(0, None))(query_fps, refs)
        x[results_column] = results
        return x


class ChempropDataMixin(TorchChemMixin):

    @staticmethod
    def make_dataloader(
        dataset: Dataset, 
        batch_size: int = 16, 
        shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=partial(
                _collate_training_batch_for_forward, 
                for_dataloader=True,
            ),
        )
