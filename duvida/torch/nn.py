"""Module classes enabling information sensitivity calculations."""

from typing import Callable, Dict, Iterable, Mapping, Tuple, Optional, Union
from functools import partial
import os

from datasets import Dataset
from lightning import LightningModule, Trainer
from numpy import ndarray
import torch
from torch import as_tensor, float32
from torch.func import functional_call, replace_all_batch_norm_modules_
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader

from ..base_classes import DataMixinBase, DoubtMixinBase, ModelBoxBase, ModelTrainerBase
from ..stateless.config import config

config.set_backend('torch', precision='float')

from .models.ensemble import TorchEnsembleMixin
from ..checkpoint_utils import load_checkpoint_file
from ..stateless.hessians import _DEFAULT_APPROXIMATOR
from ..stateless.information import (
    parameter_gradient, 
    parameter_hessian_diagonal, 
    fisher_score, 
    fisher_information_diagonal, 
    doubtscore, 
    information_sensitivity,
)
from ..stateless.typing import Array, ArrayLike, LossFunction, StatelessModel
from ..stateless.utils import get_eps, vmap

_EPS = get_eps()

def mse_loss(
    y_pred: ArrayLike, 
    y_true: ArrayLike
) -> Array:
    return torch.mean(torch.square(y_pred - y_true))

class DataMixin(DataMixinBase):

    _format = 'pytorch'
    _format_kwargs = {
        'dtype': float32, 
        'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    }

    @staticmethod
    def make_dataloader(
        dataset: Dataset, 
        batch_size: int = 16, 
        shuffle: bool = False,
        **kwargs
    ) -> DataLoader:
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            **kwargs,
        )


class ModelTrainer(ModelTrainerBase):

    _trainer: Trainer = None

    def create_trainer(self) -> None:
        self._trainer = Trainer(
            max_epochs=self.epochs,
            **self._kwargs,
        )
        return None

    def train(
        self, 
        model: LightningModule, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader
    ) -> None:
        if self._trainer is None:
            self.create_trainer()
        self._trainer.fit(model, train_dataloader, val_dataloader)
        return model


class DoubtMixin(DoubtMixinBase):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _get_shape(x: ArrayLike) -> Array:
        try:
            return x.shape
        except AttributeError:
            return torch.ones((1,))

    def to(self, device: str) -> None:
        device = torch.device(device)
        state_dict = self.model.state_dict()
        parameter_names = {name for name, param in self.model.named_parameters()}
        new_state_dict = {
            key: val.to(device)
            for key, val in state_dict.items()
            if key in parameter_names
        }
        new_state_dict.update({
            key: Parameter(
                val.to(device), 
                requires_grad=False,  # otherwise batchnorm integers cause error (not differentiable)
            ) 
            for key, val in state_dict.items()
            if key not in parameter_names
        })
        self.model.load_state_dict(
            new_state_dict, 
            assign=True,  # otherwise new device won't be preserved
        )
        self.device = device
        return None    

    @staticmethod
    def release_memory() -> None:
        torch.cuda.empty_cache()
        return None

    @staticmethod
    def _pack_params_like(
        p: ArrayLike, 
        d: Mapping[str, ArrayLike]
    ) -> Dict[str, Array]:
        index = 0
        output= {}
        for name, param in d.items():
            end_index = index + param.numel()
            output[name] = p[index:end_index].view(-1, *param.shape)
            if output[name].shape[0] == 1:
                output[name] = output[name].squeeze(0)
            index = end_index
        return output

    def make_stateless_model(
        self, 
        model: Optional[Module] = None,
        flat_params: bool = False
    ) -> Union[Tuple[StatelessModel, Dict[str, Array]], Tuple[StatelessModel, Dict[str, Array], Array]]:
        if model is None:
            self.to(self.device)
            model = self.model
        else:
            model.to(self.device)
        model.eval()
        replace_all_batch_norm_modules_(model)
        model_params = {key: val.detach().clone() for key, val in model.named_parameters()}
        if isinstance(model, TorchEnsembleMixin):
            model_params = {
                key: val for key, val in model_params.items()
                if any(model_key in key for model_key in model.model_keys)
            }

        def stateless_model(x, *params):
            # x = x.to(self.device)
            return functional_call(
                module=model, 
                parameter_and_buffer_dicts=params, 
                args=(x,),
                tie_weights=False,
            )

        if flat_params:
            def stateless_model_flat_params(x, *flat_params):
                try:
                    flat_params = torch.concat(flat_params)
                except RuntimeError:  # if flat_params has already gone through *tuple unpacking
                    flat_params = torch.stack(flat_params)
                packed_params = self._pack_params_like(
                    torch.as_tensor(flat_params), 
                    model_params,
                )
                return stateless_model(x, packed_params)
            
            flattened_params = torch.concat([p.flatten() for _, p in model_params.items()])
            return stateless_model_flat_params, model_params, flattened_params
        else:
            return stateless_model, model_params

    def parameter_gradient(
        self,
        model: Optional[StatelessModel] = None
    ) -> Callable[[ArrayLike], Array]:
        stateless_model, params = self.make_stateless_model(model)
        gradient_fn = parameter_gradient(stateless_model)

        def _parameter_gradient_with_params(x: ArrayLike) -> Array:
            return gradient_fn((params,), x)[0]

        return _parameter_gradient_with_params

    def fisher_score(
        self, 
        model: Optional[StatelessModel] = None,
        loss: LossFunction = mse_loss,
    ) -> Callable[[ArrayLike, ArrayLike], Array]:
        stateless_model, params = self.make_stateless_model(model)
        gradient_fn = fisher_score(stateless_model, loss)
        
        def _fisher_score_with_params(x_true: ArrayLike, y_true: ArrayLike) -> Array:
            return gradient_fn((params,), x_true, y_true.to(self.device))[0]
        
        return _fisher_score_with_params
    
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
            return packing_fn(hessian_fn(flat_params, x), params)

        return _parameter_hessian_with_params
    
    def fisher_information_diagonal(
        self, 
        model: Optional[StatelessModel] = None,
        loss: LossFunction = mse_loss,
        approximator: str = _DEFAULT_APPROXIMATOR, 
        *args, **kwargs
    ) -> Callable:
        stateless_model, params, flat_params = self.make_stateless_model(model, flat_params=True)
        if approximator in ('bekas', 'exact_diagonal'):
            kwargs['device'] = self.device
        fisher_info_fn = fisher_information_diagonal(
            stateless_model,
            loss,
            approximator=approximator,
            *args, **kwargs,
        )

        def _fisher_inform_diagonal_with_params(x_true, y_true):
            return self._pack_params_like(fisher_info_fn(flat_params, x_true, y_true.to(self.device)), params)

        return _fisher_inform_diagonal_with_params

    def doubtscore_core(
        self,
        fisher_score: ArrayLike, 
        parameter_gradient: ArrayLike, 
        eps: float = _EPS
    ) -> Array:
        doubtscore = parameter_gradient / (fisher_score.unsqueeze(0).to(self.device) + eps)
        return doubtscore.flatten(start_dim=1, end_dim=-1).detach().cpu()
    
    def information_sensitivity_core(
        self, 
        fisher_score: ArrayLike, 
        fisher_information_diagonal: ArrayLike, 
        parameter_gradient: ArrayLike, 
        parameter_hessian_diagonal: Optional[ArrayLike] = None, 
        eps: float = _EPS,
        optimality_approximation: bool = False
    ) -> Array:
        term1 = fisher_information_diagonal.unsqueeze(0).to(self.device) * parameter_gradient
        if not (optimality_approximation or parameter_hessian_diagonal is None):
            term2 = fisher_score.unsqueeze(0).to(self.device) * parameter_hessian_diagonal
        else:
            term2 = 0.
        information_sensitivity = torch.square(parameter_gradient) / (term1 - term2 + eps)
        return information_sensitivity.flatten(start_dim=1, end_dim=-1).detach().cpu()


class ModelBox(ModelBoxBase, DataMixin, DoubtMixin):

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
            "enable_checkpointing": False, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            "enable_progress_bar": True,
            "accelerator": "auto",
            "devices": "auto",
        }
        trainer_opts.update(kwargs)
        return ModelTrainer(epochs=epochs, **trainer_opts)