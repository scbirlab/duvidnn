"""Convolutional neural networks."""

from typing import Callable, Iterable, List, Optional, Union

from carabiner import cast

from torch.optim import Adam, Optimizer
from torch.nn import (
    BatchNorm1d, 
    Conv2d, 
    Dropout, 
    Flatten, 
    Linear, 
    MaxPool2d, 
    Module, 
    SiLU, 
    Sequential
)
from torch import nn

from .mlp import LinearStack
from .utils.ensemble import TorchEnsembleMixin
from .utils.lt import LightningMixin

from ...stateless.typing import Array, ArrayLike
from ...stateless.utils import jit

_DEFAULT_ACTIVATION = SiLU  # Smooth activation to prevent gradient collapse
_DEFAULT_N_UNITS: int = 16


class CNNStack(LinearStack):

    @staticmethod
    def _add_layer(
        in_features: int,
        out_features: int,
        layers: Optional[Iterable[Module]] = None,
        layer_class = Linear,
        batch_norm: bool = False,
        dropout: float = 0.0,
        activation = _DEFAULT_ACTIVATION,
        extras: Optional[Iterable[Module]] = None,
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
        if extras is not None:
            layers += list(extras)
        return layers

    def build_model(
        self, 
        n_input: int,  # input channels
        n_out: int = 1,
        n_hidden: int = 1,
        n_units: int = _DEFAULT_N_UNITS,
        batch_norm: bool = False,
        dropout: float = 0.,
        activation = _DEFAULT_ACTIVATION,
        layer_class = Linear,
        extras=None,
        flatten: bool = True,
        infer_projection: bool = False,
        img_shape: Optional[Iterable[int]] = None,
        **layer_kwargs
    ) -> Module:
        common_kwargs = {
            "out_features": n_units,
            "layer_class": layer_class,
            "batch_norm": batch_norm,
            "dropout": dropout,
            "activation": activation,
            "extras": extras,
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
        if img_shape is None:
            img_shape = []
        else:
            img_shape = list(img_shape)
        if flatten:
            layers.append(
                Flatten(
                    # needs to be -ve to count from right for `vmap` compatibility
                    start_dim=1 - (2 + len(img_shape)),
                    end_dim=-1,
                )
            )
        if infer_projection:
            with torch.no_grad():
                dummy = torch.zeros(1, n_input, *img_shape)  # adjust shape to match input
                out = Sequential(*layers)(dummy)
                n_feats = out.view(1, -1).size(1)
        else:
            n_feats = n_units
        layers.append(
            Linear(n_feats, out_features=n_out)
        )
        return Sequential(*layers)
        

class TorchCNN2DBase(CNNStack):
    def __init__(
        self,
        n_input: int, 
        img_shape: Iterable[int],
        n_hidden: int = 1,
        n_units: int = _DEFAULT_N_UNITS, 
        dropout: float = 0., 
        activation: Callable = _DEFAULT_ACTIVATION,
        n_out: int = 1, 
        batch_norm: bool = False,
        n_conv_layers: int = 1,
        filters: int = 32,
        kernel_size: Union[int, Iterable[int]] = 3,
        padding: Union[str, int, Iterable[int]] = "same",
        pool_stride: Union[int, Iterable[int]] = 2,
        *args, **kwargs
    ):
        super().__init__()
        self.n_input = n_input
        self.img_shape = tuple(img_shape)
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.dropout = dropout
        self.activation = activation
        self.n_out = n_out
        self.batch_norm = batch_norm
        self.n_conv_layers = n_conv_layers
        self.filters = filters
        self.kernel_size = kernel_size

        # functorch is not compatible with these strings
        if padding == "same":
            self.padding = self.kernel_size // 2
        elif padding == "valid":
            self.padding = 0
        elif not isinstance(padding, int):
            raise ValueError(f"Value for padding ({padding}) is not acceptable. Use integer or 'same' or 'valid'.")
        else:
            self.padding = padding
        
        self.dropout = dropout
        self.pool_stride = pool_stride
        self._layer_class = Conv2d
        self._layer_kwargs = {
            "kernel_size": self.kernel_size,
            "padding": self.padding,
        }
        self.model_layers = self.build_model()
        
    def build_model(self) -> Module:
        cnn_layers = super().build_model(
            n_input=self.n_input,
            n_out=self.n_units,
            layer_class=self._layer_class,
            n_hidden=self.n_conv_layers,
            n_units=self.filters, 
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            extras=[
                MaxPool2d(
                    self.kernel_size, 
                    stride=self.pool_stride,
                ),
            ],
            infer_projection=True,
            img_shape=self.img_shape,
            **self._layer_kwargs,
        )
        linear_layers = super().build_model(
            n_input=self.n_units,
            n_out=self.n_out,
            layer_class=Linear,
            n_hidden=self.n_conv_layers,
            n_units=self.n_units,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            flatten=False,
        )
        return Sequential(*cnn_layers, self.activation(), *linear_layers)
       
    @jit
    def forward(self, x: ArrayLike) -> Array:
        return self.model_layers(x)


class TorchCNN2DLightning(TorchCNN2DBase, LightningMixin):
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
        # Lightning logic
        self._init_lightning(
            optimizer=optimizer, 
            learning_rate=learning_rate, 
            model_attr='model_layers',  # the attribute containing the model
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            reduce_lr_patience=reduce_lr_patience,
        )


class TorchCNN2DEnsemble(TorchEnsembleMixin, TorchCNN2DLightning):
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
        