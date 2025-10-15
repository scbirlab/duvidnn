"""Building and managing model ensembles."""

from typing import Iterable, Union

from abc import ABC, abstractmethod

from torch import stack
from torch.nn import Module, ModuleDict

from ....stateless.typing import Array, ArrayLike


class TorchEnsembleMixin(ABC):

    _module_prefix: str = "ensemble_module"

    def _init_ensemble(self, ensemble_size: int) -> None:
        self.ensemble_size = ensemble_size
        self.model_to_use = 'all'
        self.model_keys = None
        self._model_ensemble = self.create_ensemble()
        self.set_model(self.model_to_use)
        return None

    @abstractmethod
    def create_module(self) -> Module:
        pass
        
    def create_ensemble(self) -> ModuleDict:
        return ModuleDict({f"{self._module_prefix}:{i}": self.create_module() for i in range(self.ensemble_size)})

    def set_model(self, n: Union[str, int] = 'all') -> None:
        if n == 'all':
            ensemble = self._model_ensemble
        elif isinstance(n, int) and n < len(self._model_ensemble):
            key = f"{self._module_prefix}:{n}"
            ensemble = ModuleDict({key: self._model_ensemble[key]})
        elif (
            isinstance(n, Iterable) 
            and all(isinstance(_n, int) for _n in n) 
            and all(_n < len(self._model_ensemble) for _n in n)
        ):
            keys = {f"{self._module_prefix}:{i}" for i in n}
            ensemble = ModuleDict({key: self._model_ensemble[key] for key in keys})
        else:
            raise ValueError(f"Cannot set model with n={n} from ensemble with length{(len(self._model_ensemble))}")
        self.model_to_use = n
        self.model_ensemble = ensemble
        self.model_keys = tuple(self.model_ensemble.keys())
        return None

    def forward(self, x: ArrayLike) -> Array:
        # Stack outputs to shape [batch, n_out, ensemble_size]
        out = stack([
            module(x) for _, module in self.model_ensemble.items()
        ], dim=-1)
        # If n_out == 1 (scalar), squeeze the middle axis to get [batch, ensemble_size]
        if out.size(-2) == 1:
            out = out.squeeze(-2)
        return out
