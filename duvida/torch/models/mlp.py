"""Multi-layer perceptrons."""

from typing import Callable, Iterable, List, Mapping, Optional
from abc import ABC, abstractmethod

from carabiner import cast
from torch.nn import BatchNorm1d, Dropout, Linear, Module, SiLU, Sequential

from torch.optim import Adam, Optimizer

from .utils.ensemble import TorchEnsembleMixin
from .utils.lt import LightningMixin
from ...stateless.typing import Array, ArrayLike

class TorchMLPBase(Module, ABC):

    def __init__(
        self, 
        n_input: int, 
        n_hidden: int = 1,
        n_units: int = 16, 
        learning_rate: float = .01,
        dropout: float = 0., 
        activation: Callable = SiLU,  # Smooth activation to prevent gradient collapse
        n_out: int = 1, 
        batch_norm: bool = False,
        skip_length: Optional[int] = None
    ):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.activation = activation
        self.n_out = n_out
        self.batch_norm = batch_norm
        if skip_length is not None and (skip_length > self.n_hidden):
            raise ValueError(
                f"""
                Skip length must be greater than number of hidden layers:
                - Skip length: {skip_length}
                - Number of hidden layers: {self.n_hidden}
                """
            )
        self.skip_length = skip_length
        self._stack_size = 2 + int(self.dropout > 0.)  + int(self.batch_norm)
        if skip_length is not None:
            self._stack_skip_length = self.skip_length * self._stack_size
        else:
            self._stack_skip_length = None
        self.model_layers = self.create_model()

    def _add_layer(
        self, 
        layer_input_size: int,
        layers: Optional[Iterable[Module]] = None, 
    ) -> List[Module]:
        if layers is None:
            layers = []
        layers = cast(layers, to=list)
        layers += [Linear(layer_input_size, self.n_units), self.activation()]
        if self.dropout > 0.:
            layers.append(Dropout(self.dropout))
        if self.batch_norm:
            layers.append(BatchNorm1d(self.n_units))
        return layers

    def create_model(self) -> Module:
        layers = self._add_layer(self.n_input)
        for _ in range(1, self.n_hidden):
            layers = self._add_layer(self.n_units, layers)
        layers.append(Linear(self.n_units, self.n_out))
        return Sequential(*layers)

    def forward(self, x: ArrayLike) -> Array:
        if self.skip_length is None:
            return self.model_layers(x)
        else:
            activations = []
            for i, layer in enumerate(self.model_layers):
                x = layer(x)
                activations.append(x)
                not_input = i >= self._stack_size
                modulo_1 = i % self._stack_skip_length == 1
                make_skip = not_input and modulo_1
                if make_skip:
                    x = x + activations[i - self._stack_skip_length]
            return x


class TorchMLPLightning(TorchMLPBase, LightningMixin):

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
            model_attr='model_layers',
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )


class TorchMLPEnsemble(TorchEnsembleMixin, TorchMLPLightning):

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

    def create_module(self) -> Module:
        return self.create_model()