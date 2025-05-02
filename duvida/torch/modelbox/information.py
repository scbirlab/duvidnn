"""Enabling information sensitivity calculations."""

from typing import Callable, Dict, Mapping, Tuple, Optional, Union
from functools import partial

import torch
from torch.func import functional_call, replace_all_batch_norm_modules_
from torch.nn import Module, Parameter

from ...base.information import DoubtMixinBase
from ...stateless.config import config

config.set_backend('torch', precision='float')

from ...stateless.hessians import _DEFAULT_APPROXIMATOR
from ...stateless.information import (
    fisher_score, 
    fisher_information_diagonal, 
    parameter_gradient, 
    parameter_hessian_diagonal,
    parameter_gradient_unrolled, 
    parameter_hessian_diagonal_unrolled
)
from ...stateless.typing import Array, ArrayLike, LossFunction, StatelessModel
from ...stateless.utils import get_eps, jit, vmap
from ..functions import mse_loss
from ..models.utils.ensemble import TorchEnsembleMixin

_EPS = get_eps()
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubtMixin(DoubtMixinBase):

    device = _DEVICE

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
        output = {}
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

            # @jit  # Fails compile
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

        @jit
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
        
        @jit
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

        @jit
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

        # @jit  # Doesn't compile
        def _fisher_inform_diagonal_with_params(x_true, y_true):
            return self._pack_params_like(fisher_info_fn(flat_params, x_true, y_true.to(self.device)), params)

        return _fisher_inform_diagonal_with_params

    @jit
    def doubtscore_core(
        self,
        fisher_score: ArrayLike, 
        parameter_gradient: ArrayLike, 
        eps: float = _EPS
    ) -> Array:
        doubtscore = parameter_gradient / (fisher_score.unsqueeze(0).to(self.device) + eps)
        return doubtscore.flatten(start_dim=1, end_dim=-1).detach().cpu()
    
    @jit
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


class ChempropDoubtMixin(DoubtMixin):

    def parameter_gradient(
        self,
        model: Optional[StatelessModel] = None
    ) -> Callable[[ArrayLike], Array]:
        stateless_model, params = self.make_stateless_model(model)
        gradient_fn = parameter_gradient_unrolled(stateless_model)

        # @jit  # fails compile
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
        (
            stateless_model, 
            params, 
            flat_params,
        ) = self.make_stateless_model(model, flat_params=True)
        if approximator in ('bekas', 'exact_diagonal'):
            # these need explicit device placement for hvp
            kwargs['device'] = self.device
        hessian_fn = parameter_hessian_diagonal_unrolled(
            stateless_model,
            approximator=approximator,
            *args, **kwargs,
        )
        packing_fn = partial(vmap, in_axes=(0, None))(self._pack_params_like)

        @jit
        def _parameter_hessian_with_params(x):
            return packing_fn(torch.stack(hessian_fn(flat_params, x)), params)

        return _parameter_hessian_with_params