"""ModelBoxes for Chemistry in PyTorch."""

from typing import Callable, Dict, Iterable, Mapping, Optional, Union
from functools import partial

from datasets import Dataset
import torch
from torch.utils.data import DataLoader

from ..base_classes import VarianceMixin
from ..chem import FPModelBoxMixinBase, ChempropModelBoxMixinBase
from ..stateless.config import config

config.set_backend('torch', precision='float')

from ..stateless.hessians import _DEFAULT_APPROXIMATOR, get_approximators
from ..stateless.numpy import numpy as dnp
from ..stateless.typing import Array, ArrayLike, StatelessModel
from ..stateless.utils import grad, vmap
from .nn import DataMixin, DoubtMixin, ModelBox
from .models.cp import _collate_training_batch_for_forward, ChempropEnsemble
from .models.mlp import MLPModelBox

class TorchFPModelBoxMixin(FPModelBoxMixinBase):

    @staticmethod
    def _get_max_sim(
        q: ArrayLike, 
        references: ArrayLike
    ) -> Array:
        a_n_b = torch.sum(q.unsqueeze(0) * references, dim=-1, keepdim=True)
        sum_q = torch.sum(q)
        similarities = a_n_b / (sum_q + torch.sum(references, dim=-1) - torch.sum(a_n_b, dim=-1)).unsqueeze(-1)
        return torch.max(similarities)

    @staticmethod
    def _get_nn_tanimoto(
        queries: Mapping[str, ArrayLike],
        refs_data: Mapping[str, ArrayLike],
        _in_key: str,
        _out_key: str,
        _sim_fn: Callable[[ArrayLike, ArrayLike], ArrayLike]
    ) -> Dict[str, Array]:
        query_fps = queries[_in_key]
        refs = refs_data[_in_key]
        results = vmap(_sim_fn, in_axes=(0, None))(query_fps, refs)
        return dict(tanimoto_nn=results)


class FPMLPModelBox(TorchFPModelBoxMixin, MLPModelBox):

    def __init__(
        self, 
        use_fp: bool = False,
        use_2d: bool = False,
        extra_featurizers: Optional[Union[Iterable[Callable], Callable]] = None,
        *args, **kwargs
    ):
        self.use_fp = use_fp
        self.use_2d = use_2d
        self.extra_featurizers = extra_featurizers
        super().__init__(*args, **kwargs)


class ChempropDataMixin(DataMixin):

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

def parameter_gradient(model: StatelessModel) -> Callable[[ArrayLike, ArrayLike], Array]:

    @grad
    def _f0(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return dnp.sum(model(x, *params))

    def _f(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return [_f0(params, [d]) for d in x]

    return _f


def parameter_hessian_diagonal(
    model, 
    approximator: str = _DEFAULT_APPROXIMATOR, 
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike], Array]:

    @partial(get_approximators(approximator), *args, **kwargs)
    def _f0(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return dnp.sum(model(x, *params))

    def _f(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return [_f0(params, [d]) for d in x]

    return _f


class ChempropDoubtMixin(DoubtMixin):

    def parameter_gradient(
        self,
        model: Optional[StatelessModel] = None
    ) -> Callable[[ArrayLike], Array]:
        stateless_model, params = self.make_stateless_model(model)
        gradient_fn = parameter_gradient(stateless_model)

        def _parameter_gradient_with_params(x: ArrayLike) -> Array:
            g = [_g[0] for _g in gradient_fn((params,), x)]
            return {
                key: torch.stack([_g[key] for _g in g]) 
                for key in g[0]
            }

        return _parameter_gradient_with_params
    
    def parameter_hessian_diagonal(
        self, 
        model: Optional[StatelessModel] = None,
        approximator: str = _DEFAULT_APPROXIMATOR, 
        *args, **kwargs
    ) -> Callable:
        stateless_model, params, flat_params = self.make_stateless_model(model, flat_params=True)
        if approximator in ('bekas', 'exact_diagonal'):
            kwargs['device'] = self.device
        hessian_fn = parameter_hessian_diagonal(
            stateless_model,
            approximator=approximator,
            *args, **kwargs,
        )
        packing_fn = partial(vmap, in_axes=(0, None))(self._pack_params_like)

        def _parameter_hessian_with_params(x):
            return packing_fn(torch.stack(hessian_fn(flat_params, x)), params)

        return _parameter_hessian_with_params


class ChempropModelBox(ChempropDoubtMixin, ChempropDataMixin, ChempropModelBoxMixinBase, TorchFPModelBoxMixin, VarianceMixin, ModelBox):

    def __init__(
        self, 
        use_fp: bool = False,
        use_2d: bool = False,
        extra_featurizers: Optional[Union[Iterable[Callable], Callable]] = None,
        *args, **kwargs
    ):
        super().__init__()
        self.use_fp = use_fp
        self.use_2d = use_2d
        self.extra_featurizers = extra_featurizers
        self._chemprop_kwargs = kwargs

    def create_model(self):
        if self.training_example[self._in_key][0]["x_d"] is None:
            self.input_shape = (0,)
        else:
            self.input_shape = self.training_example[self._in_key][0]["x_d"].shape
        return ChempropEnsemble(
            n_input=self.input_shape[-1],
            n_out=self.output_shape[-1], 
            **self._chemprop_kwargs,
        )