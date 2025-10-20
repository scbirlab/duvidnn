"""Bilinear models."""

from typing import Callable, Iterable, List, Optional, Union
from functools import partial

from duvida.config import config
config.set_backend('torch', precision='float')
from duvida.types import Array, ArrayLike
from duvida import jit
from carabiner import cast, print_err
import torch
from torch.nn import BatchNorm1d, Dropout, Identity, Linear, Module, ModuleDict, SiLU, Sequential
from torch.optim import Adam, Optimizer

from .mlp import TorchMLPBase
from .utils.ensemble import TorchEnsembleMixin
from .utils.lt import LightningMixin

_DEFAULT_ACTIVATION: Callable[..., Module] = SiLU  # Smooth activation to prevent gradient collapse
_DEFAULT_N_UNITS: int = 16


class _TorchBilinearBase(Module):

    """Bilinear multilayer perceptron.

    Examples
    ========
    >>> import torch
    >>> net = TorchBilinearBase(n_input=[2, 10], n_hidden=2, n_units=8, n_out=1)
    >>> net([torch.randn(3, 2), torch.randn(3, 10)]).shape 
    torch.Size([3, 1])
    >>> net = TorchBilinearBase(n_input=[2, 10], n_context=16, n_hidden=2, n_units=8, n_out=1)
    >>> net([torch.randn(3, 2), torch.randn(3, 10), torch.randn(3, 16)]).shape 
    torch.Size([3, 1])
    >>> net = TorchBilinearBase(n_input=[2, 10], n_hidden=2, n_units=8, n_out=1, merge_method="product")
    >>> net([torch.randn(3, 2), torch.randn(3, 10)]).shape 
    torch.Size([3, 1])
    >>> net = TorchBilinearBase(n_input=[2, 10], n_hidden=2, n_units=8, n_out=1, merge_method="sum")
    >>> net([torch.randn(3, 2), torch.randn(3, 10)]).shape 
    torch.Size([3, 1])
    >>> resnet = TorchBilinearBase(n_input=[2, 10], n_hidden=4, n_units=8, n_out=1, residual_depth=2)
    >>> resnet([torch.randn(5, 2), torch.randn(5, 10)]).shape 
    torch.Size([5, 1])

    """

    _module_prefix = "bilinear"
    _fusion_name = "fusion"
    _film_name = "film"

    def __init__(
        self, 
        n_input: Union[int, Iterable[int]], 
        n_units: Union[int, Iterable[int]] = _DEFAULT_N_UNITS,
        n_hidden: int = 1,
        n_hidden_fusion: int = 1,
        n_units_fusion: Optional[int] = None,
        n_context: Optional[int] = None,
        merge_method: str = "concat",
        drop_inputs: bool = False,
        dropout: float = 0., 
        activation: Callable[..., Module] = _DEFAULT_ACTIVATION,
        n_hidden_film = 1,
        n_units_film: Optional[int] = None,
        activation_film: Callable[..., Module] = _DEFAULT_ACTIVATION,
        soft_film: bool = False,
        n_out: int = 1, 
        batch_norm: bool = False,
        residual_depth: Optional[int] = None,
        final_activation: Optional[Callable[..., Module]] = None,
        tower_classes: Union[Callable[..., Module], Iterable[Callable[..., Module]]]  = TorchMLPBase,
        *args, **kwargs
    ):
        super().__init__()
        n_input = tuple(cast(n_input, to=list))
        n_units = tuple(cast(n_units, to=list))
        self.n_towers = len(n_input)
        if len(n_units) == 1 and self.n_towers > 1:
            n_units = n_units * self.n_towers
        elif len(n_units) != self.n_towers:
            raise ValueError(f"Size of {n_units=} and {n_input=} must be equal or 1")
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.residual_depth = residual_depth

        if isinstance(tower_classes, Callable):
            tower_classes = (tower_classes,)
        if len(tower_classes) == 1 and self.n_towers > 1:
            tower_classes = tower_classes * self.n_towers
        elif len(tower_classes) != self.n_towers:
            raise ValueError(f"Size of {tower_classes=} and {n_input=} must be equal or 1")
        self.tower_classes = tower_classes

        self.dropout = dropout
        self.activation = activation
        self.n_out = n_out
        self.batch_norm = batch_norm

        self.n_hidden_fusion = n_hidden_fusion
        self.n_units_fusion = n_units_fusion or self.n_units[-1]
        self.merge_method = merge_method.casefold()
        self.drop_inputs = drop_inputs
        if self.merge_method == "concat":
            self._merge_fn = self._concat
            self.tower_out_sizes = self.n_units
            self.merge_size = sum(self.tower_out_sizes)
        elif self.merge_method == "product":
            self._merge_fn = self._product
            self.tower_out_sizes = tuple(max(self.n_units) for _ in range(self.n_towers))
            self.merge_size = self.tower_out_sizes[0]
        elif self.merge_method == "sum":
            self._merge_fn = self._sum
            self.tower_out_sizes = tuple(max(self.n_units) for _ in range(self.n_towers))
            self.merge_size = self.tower_out_sizes[0]
        else:
            raise ValueError(f"Merge method {self.merge_method} is not implemented.")
        if not self.drop_inputs:
            self.merge_size += sum(self.n_input)

        self.n_context = n_context
        self.n_hidden_film = n_hidden_film
        self.n_units_film = n_units_film or self.n_units[-1]
        self.activation_film = activation_film
        self.soft_film = soft_film
        
        self.model_towers = self.build_towers()

    @staticmethod
    def _product(*x) -> Array:
        return torch.prod(torch.stack(list(x), dim=-1), dim=-1)

    @staticmethod
    def _sum(*x) -> Array:
        return torch.stack(list(x), dim=-1).sum(dim=-1)

    @staticmethod
    def _concat(*x) -> Array:
        return torch.concat(list(x), dim=-1)

    def build_towers(self):
        towers = []
        for n_input, n_units, n_out, layer_class in zip(
            self.n_input, 
            self.n_units, 
            self.tower_out_sizes, 
            self.tower_classes,
        ):
            mlp = layer_class(
                n_input=n_input, 
                n_out=n_out, 
                n_hidden=self.n_hidden,
                n_units=n_units, 
                dropout=self.dropout, 
                activation=self.activation, 
                batch_norm=self.batch_norm,
                residual_depth=self.residual_depth,
                final_activation=self.activation,
            )
            towers.append(mlp)
        towers = {
            f"{self._module_prefix}:{i}_tower": tower 
            for i, tower in enumerate(towers)
        }
        if not len(towers) == self.n_towers:
            # should never happen
            raise ValueError(
                f"Number of towers generated ({len(towers)=}) is not equal to number of inputs ({self.n_towers=})"
            )
        if self.n_context is not None and isinstance(self.n_context, int):
            towers[f"{self._module_prefix}:{self._film_name}"] = TorchMLPBase(
                n_input=self.n_context, 
                n_out=2 * self.merge_size, 
                n_hidden=self.n_hidden_film,
                n_units=self.n_units_film, 
                activation=self.activation_film,
            )
            
        towers[f"{self._module_prefix}:{self._fusion_name}"] = TorchMLPBase(
            n_input=self.merge_size, 
            n_out=self.n_out, 
            n_hidden=self.n_hidden_fusion,
            n_units=self.n_units_fusion, 
            dropout=self.dropout, 
            activation=self.activation, 
            batch_norm=self.batch_norm,
            residual_depth=self.residual_depth,
        )
        
        return ModuleDict(towers)
        
    def forward(self, x: Union[ArrayLike, Iterable[ArrayLike]]) -> Array:
        if isinstance(x, torch.Tensor):
            x = [x]
        elif isinstance(x, tuple):
            x = list(x)
        if self.n_context is not None:
            context_x = x[-1]  # context is the final input
            x = x[:-1]
        else:
            context_x = None
        if len(x) != self.n_towers:
            raise ValueError(
                f"Length of inputs ({len(x)=}: {[_x.shape for _x in x]}) is not equal to "
                f"number of towers ({self.n_towers})."
            )
        towers = [
            module for name, module in self.model_towers.items()
            if name.endswith("_tower")
        ]
        towers = [
            module(_x) for _x, module in zip(x, towers)
        ]
        towers = self._merge_fn(*towers)
        if not self.drop_inputs:
            towers = torch.concat(x + [towers], dim=-1)

        film_key = f"{self._module_prefix}:{self._film_name}"
        if film_key in self.model_towers and context_x is not None:
            gamma_beta = self.model_towers[film_key](context_x)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            if self.soft_film:
                gamma = torch.sigmoid(gamma) * 2.
                beta  = torch.tanh(beta)
            towers = gamma * towers + beta

        fusion_module = self.model_towers[f"{self._module_prefix}:{self._fusion_name}"]
        return fusion_module(towers)


class TorchBilinearBase(Module):

    def __init__(self, _init_model: bool = True, *args, **kwargs):
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        for key, arg in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, arg)
        if _init_model:
            self.bilinear_model = self.build_model()
        else:
            self.bilinear_model = False

    def build_model(self) -> Module:
        return _TorchBilinearBase(*self._args, **self._kwargs)

    def forward(self, x):
        return self.bilinear_model(x)


class TorchBilinearLightning(TorchBilinearBase, LightningMixin):

    """Bilinear multilayer perceptron with Lightning.

    Examples
    ========
    >>> import torch
    >>> net = TorchBilinearLightning(n_input=[4, 10], n_hidden=2, n_units=8, n_out=1)
    >>> net([torch.randn(3, 4), torch.randn(3, 10)]).shape 
    torch.Size([3, 1])

    """

    def __init__(
        self, 
        learning_rate: float = .01,
        optimizer: Optimizer = Adam,
        reduce_lr_on_plateau: bool = True,
        reduce_lr_patience: int = 10,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._init_lightning(
            optimizer=optimizer, 
            learning_rate=learning_rate, 
            model_attr='bilinear_model',
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )


class TorchBilinearEnsemble(TorchEnsembleMixin, TorchBilinearLightning):

    """Bilinear multilayer perceptron ensemble with Lightning.

    Examples
    ========
    >>> import torch
    >>> ensemble = TorchBilinearEnsemble(n_input=[4, 5], n_units=4, n_out=1, ensemble_size=2) 
    >>> ensemble([torch.randn(5, 4), torch.randn(5, 5)]).shape 
    torch.Size([5, 2])
    >>> ensemble.set_model(0)  # keep only the first sub-model
    >>> ensemble([torch.randn(2, 4), torch.randn(2, 5)]).shape 
    torch.Size([2, 1])
    >>> ensemble.set_model("all"); len(ensemble.model_ensemble) 
    2
    >>> list(ensemble.model_keys) == ["ensemble_module:0", "ensemble_module:1"] 
    True

    """

    def __init__(
        self, 
        ensemble_size: int = 1,
        learning_rate: float = .01,
        optimizer: Optimizer = Adam,
        reduce_lr_on_plateau: bool = True,
        reduce_lr_patience: int = 10,
        *args, **kwargs
    ):
        super().__init__(_init_model=False, *args, **kwargs)
        self.save_hyperparameters()
        self._init_ensemble(ensemble_size)
        self._init_lightning(
            optimizer=optimizer, 
            learning_rate=learning_rate, 
            model_attr='_model_ensemble',
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )

    def create_module(self) -> ModuleDict:
        return self.build_model()
