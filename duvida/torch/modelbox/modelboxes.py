"""Containers for models and their training data."""

from typing import Tuple, Optional
import os

import torch

from ...base.modelboxes import (
    ChempropModelBoxBase,
    FingerprintModelBoxBase, 
    ModelBoxBase, 
    ModelBoxWithVarianceBase
)
from ...base.modelbox_registry import register_modelbox
from ...checkpoint_utils import load_checkpoint_file
from ...stateless.config import config

config.set_backend('torch', precision='float')

from ...stateless.typing import Array, ArrayLike
from ..models import ChempropEnsemble, TorchMLPEnsemble
from .data import ChempropDataMixin, DataMixin, TorchChemMixin
from .information import DoubtMixin, ChempropDoubtMixin
from .training import ModelTrainer


class TorchModelBoxBase(ModelBoxBase, DataMixin, DoubtMixin):

    def save_weights(
        self,
        checkpoint: str
    ) -> None:
        torch.save(
            self.model.state_dict(), 
            os.path.join(checkpoint, "params.pt"),
        )
        return None

    def load_weights(
        self,
        checkpoint: str,
        cache_dir: Optional[str] = None
    ):
        state_dict = load_checkpoint_file(
            checkpoint, 
            filename="params.pt",
            callback="pt",
            none_on_error=False,
            cache_dir=cache_dir,
        )
        self.model.load_state_dict(state_dict)
        return self

    def eval_mode(self) -> None:
        self.model.eval()
        return None

    def train_mode(self) -> None:
        self.model.train()
        return None

    @property
    def shape(self) -> Tuple[Array]:
        return tuple(p.shape for p in self.model.parameters())
    
    @property
    def size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @staticmethod
    def detach_tensor(x: ArrayLike) -> Array:
        return x.detach().cpu().numpy()

    def create_trainer(
        self, 
        epochs: int = 1, 
        **kwargs
    ) -> ModelTrainer:
        trainer_opts = {
            "logger": True,
            # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "accelerator": "auto",
            "devices": "auto",
        }
        trainer_opts.update(kwargs)
        return ModelTrainer(epochs=epochs, **trainer_opts)


@register_modelbox("mlp")
class TorchMLPModelBox(TorchModelBoxBase, ModelBoxWithVarianceBase):

    """ModelBox for pytorch multilayer perceptron ensemble.

    Examples
    ========
    >>> mb = TorchMLPModelBox(ensemble_size=3) 
    >>> mb.input_shape, mb.output_shape = (4,), (1,) # usually set by .load_training_data() 
    >>> mb.model = mb.create_model() 
    >>> mb.model.n_input, mb.model.n_out 
    (4, 1)

    """

    def create_model(self, *args, **kwargs) -> TorchMLPEnsemble:
        self._model_config.update(kwargs)
        return TorchMLPEnsemble(
            n_input=self.input_shape[-1],
            n_out=self.output_shape[-1], 
            *args, **self._model_config,
        )


@register_modelbox("fingerprint")
class TorchFingerprintModelBox(TorchChemMixin, FingerprintModelBoxBase, TorchMLPModelBox):
    pass


@register_modelbox("chemprop")
class ChempropModelBox(ChempropDataMixin, ChempropDoubtMixin, ChempropModelBoxBase, TorchFingerprintModelBox):

    def create_model(self, *args, **kwargs) -> ChempropEnsemble:
        self._model_config.update(kwargs)
        if self.training_example[self._in_key][0]["x_d"] is None:
            self.input_shape = (0,)
        else:
            self.input_shape = self.training_example[self._in_key][0]["x_d"].shape
        return ChempropEnsemble(
            n_input=self.input_shape[-1],
            n_out=self.output_shape[-1], 
            **self._model_config,
        )
