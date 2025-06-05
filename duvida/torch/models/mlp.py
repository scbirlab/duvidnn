"""Multi-layer perceptrons."""

from typing import Callable, List, Optional

from carabiner import cast, print_err
from torch.nn import BatchNorm1d, Dropout, Identity, Linear, Module, SiLU, Sequential

from torch.optim import Adam, Optimizer

from .utils.ensemble import TorchEnsembleMixin
from .utils.lt import LightningMixin
from ...stateless.config import config

config.set_backend('torch', precision='float')

from ...stateless.typing import Array, ArrayLike
from ...stateless.utils import jit

_DEFAULT_ACTIVATION: Callable[..., Module] = SiLU  # Smooth activation to prevent gradient collapse
_DEFAULT_N_UNITS: int = 16

class LinearStack(Module):

    @staticmethod
    def _add_layer(
        in_features: int,
        out_features: int,
        layers: Optional[list] = None,
        layer_class: Callable[..., Module] = Linear,
        batch_norm: bool = False,
        dropout: float = 0.0,
        activation: Callable[..., Module] = _DEFAULT_ACTIVATION,
        **kwargs
    ) -> List[Module]:
        layers = cast(
            [] if layers is None else layers, 
            to=list,
        )
        layers += [layer_class(in_features, out_features, **kwargs)]
        if batch_norm:
            layers.append(BatchNorm1d(out_features))
        layers.append(activation())
        if dropout > 0.:
            layers.append(Dropout(dropout))
        return layers

    def build_model(
        self, 
        n_input: int,
        n_out: int = 1,
        n_hidden: int = 1,
        n_units: int =_DEFAULT_N_UNITS,
        batch_norm: bool = False,
        dropout: float = 0.,
        activation: Callable[..., Module] = _DEFAULT_ACTIVATION,
        layer_class: Callable[..., Module] = Linear,
        **layer_kwargs
    ) -> Module:
        common_kwargs = {
            "out_features": n_units,
            "layer_class": layer_class,
            "batch_norm": batch_norm,
            "dropout": dropout,
            "activation": activation,
        }
        layers = self._add_layer(
            in_features=n_input,
            **common_kwargs,
            **layer_kwargs,
        )
        for _ in range(1, n_hidden):
            layers = self._add_layer(
                in_features=n_units,
                layers=layers,
                **common_kwargs,
                **layer_kwargs,
            )
        layers.append(
            Linear(in_features=n_units, out_features=n_out)
        )
        return Sequential(*layers)


class TorchResidualBlock(LinearStack):

    """Residual block module.

    Examples
    ========
    >>> import torch
    >>> net = TorchResidualBlock(n_input=4, residual_depth=2, n_units=8, n_out=1)
    >>> net(torch.randn(3, 4)).shape 
    torch.Size([3, 1])

    """

    def __init__(
        self, 
        n_input: int, 
        n_out: int, 
        residual_depth: int = 1,
        n_units: int =_DEFAULT_N_UNITS, 
        dropout: float = 0., 
        activation: Callable = _DEFAULT_ACTIVATION,  
        batch_norm: bool = False
    ):
        super().__init__()
        self.n_input = n_input
        self.residual_depth = residual_depth
        self.n_units = n_units
        self.dropout = dropout
        self.activation = activation
        self.n_out = n_out
        self.batch_norm = batch_norm
        if self.n_input == self.n_out:
            self.projection = Identity()
        else:
            self.projection = Linear(n_input, n_out)
        self.model_layers = self.build_model()

    def build_model(self):
        return super().build_model(
            n_input=self.n_input,
            n_out=self.n_out,
            n_hidden=self.residual_depth,
            n_units=self.n_units,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
            activation=self.activation,
            layer_class=Linear,
        )

    # @jit  # Fails compile
    def forward(self, x: ArrayLike) -> Array:
        residual = self.model_layers(x)
        projection = self.projection(x)
        return self.activation()(residual + projection)


class TorchMLPBase(LinearStack):

    """Multilayer perceptron.

    Examples
    ========
    >>> import torch
    >>> net = TorchMLPBase(n_input=4, n_hidden=2, n_units=8, n_out=1)
    >>> net(torch.randn(3, 4)).shape 
    torch.Size([3, 1])
    >>> resnet = TorchMLPBase(n_input=4, n_hidden=4, n_units=8, n_out=1, residual_depth=2)
    >>> resnet(torch.randn(3, 4)).shape 
    torch.Size([3, 1])

    """

    def __init__(
        self, 
        n_input: int, 
        n_hidden: int = 1,
        n_units: int = _DEFAULT_N_UNITS, 
        dropout: float = 0., 
        activation: Callable = _DEFAULT_ACTIVATION,
        n_out: int = 1, 
        batch_norm: bool = False,
        residual_depth: Optional[int] = None
    ):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.dropout = dropout
        self.activation = activation
        self.n_out = n_out
        self.batch_norm = batch_norm
        if residual_depth is not None and (residual_depth > self.n_hidden):
            print_err(
                f"""
                WARNING: Skip length must be greater than number of hidden layers:
                - Skip length: {residual_depth}
                - Number of hidden layers: {self.n_hidden}
                Falling back to non-residual.
                """
            )
            self.residual_depth = None
        else:
            self.residual_depth = residual_depth
        if self.residual_depth is not None:
            self._layer_class = TorchResidualBlock
            self._layer_kwargs = {
                "residual_depth": self.residual_depth,
            }
            self._n_residual_blocks = self.n_hidden // (self.residual_depth or 1)
            self._n_extra_linear = self.n_hidden % (self.residual_depth or 1)
        else:
            self._layer_class = Linear
            self._layer_kwargs = {}
            self._n_residual_blocks = 0
            self._n_extra_linear = 0
        self.model_layers = self.build_model()

    def build_model(self):
        layers = super().build_model(
            n_input=self.n_input,
            n_out=(self.n_out if self._n_extra_linear == 0 else self.n_units), 
            layer_class=self._layer_class,
            n_hidden=(self._n_residual_blocks if self.residual_depth is not None else self.n_hidden),
            n_units=self.n_units, 
            dropout=self.dropout, 
            activation=self.activation,  
            batch_norm=self.batch_norm,
            **self._layer_kwargs,
        )
        if self._n_extra_linear > 0:
            extra_layers = super().build_model(
                n_input=self.n_units,
                n_out=self.n_out, 
                layer_class=Linear,
                n_hidden=self._n_extra_linear,
                n_units=self.n_units, 
                dropout=self.dropout, 
                activation=self.activation,  
                batch_norm=self.batch_norm,
            )
            layers = Sequential(layers, extra_layers)

        return layers
        
    @jit
    def forward(self, x: ArrayLike) -> Array:
        return self.model_layers(x)


class TorchMLPLightning(TorchMLPBase, LightningMixin):

    """Multilayer perceptron with Lightning.

    Examples
    ========
    >>> import torch
    >>> net = TorchMLPLightning(n_input=4, n_hidden=2, n_units=8, n_out=1)
    >>> net(torch.randn(3, 4)).shape 
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
            model_attr='model_layers',
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )


class TorchMLPEnsemble(TorchEnsembleMixin, TorchMLPLightning):

    """Multilayer perceptron ensemble with Lightning.

    Examples
    ========
    >>> import torch
    >>> ensemble = TorchMLPEnsemble(n_input=4, n_units=4, n_out=1, ensemble_size=2) 
    >>> ensemble(torch.randn(5, 4)).shape 
    torch.Size([5, 2])
    >>> ensemble.set_model(0)  # keep only the first sub-model
    >>> ensemble(torch.randn(2, 4)).shape 
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
        return self.build_model()
