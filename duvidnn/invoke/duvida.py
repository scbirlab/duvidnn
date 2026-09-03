"""Adapt PyTorch models and ModelInvoker batches to Duvida transforms."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from duvida import (
    fisher_information_diagonal,
    fisher_score,
    information_sensitivity,
    parameter_gradient,
    parameter_hessian_diagonal,
    doubtscore,
)
import torch
from torch import nn, Tensor
from tqdm.auto import tqdm

from .base import ModelInvoker
from .functional import make_stateless_model


ParameterTree = dict[str, Tensor]

DEFAULT_APPROXIMATOR: str = "exact_diagonal"


def _accumulate(
    total: ParameterTree | None,
    value: Mapping[str, Tensor],
) -> ParameterTree:
    if total is None:
        return {
            name: tensor.detach().clone()
            for name, tensor in value.items()
        }

    for name, tensor in value.items():
        total[name].add_(
            tensor.detach()
        )

    return total


def _cpu_tree(
    tree: Mapping[str, Tensor] | None,
) -> ParameterTree | None:
    if tree is None:
        return None

    return {
        name: tensor.detach().cpu()
        for name, tensor in tree.items()
    }


@dataclass
class TrainingDerivatives:
    fisher_score: ParameterTree | None = None
    fisher_information: ParameterTree | None = None
    fisher_information_config: dict[str, Any] | None = None
    n_samples: int | None = None
    loss_reduction: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "fisher_score": _cpu_tree(self.fisher_score),
            "fisher_information": _cpu_tree(self.fisher_information),
            "fisher_information_config": (
                dict(self.fisher_information_config)
                if self.fisher_information_config is not None
                else None
            ),
            "n_samples": self.n_samples,
            "loss_reduction": self.loss_reduction,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "TrainingDerivatives":
        return cls(**dict(state))


class DuvidaModel:
    """Functional Duvida view over a PyTorch model and ModelInvoker."""

    def __init__(
        self,
        model: nn.Module,
        invoker: ModelInvoker
    ):
        self.model = model
        self.invoker = invoker

    @property
    def params(self) -> ParameterTree:
        return dict(self.model.named_parameters())

    def inputs(
        self,
        batch,
    ):
        return self.invoker.inputs(batch)

    def target(
        self,
        batch,
    ):
        return self.invoker.target(batch)

    def stateless_model(self):
        return make_stateless_model(self.model)

    @staticmethod
    def _loss(
        loss: Callable,
        target: Tensor,
        loss_kwargs: Mapping[str, Any],
        loss_reduction: str
    ) -> Callable:
        if loss_reduction not in {
            "mean",
            "sum",
        }:
            raise ValueError(
                "`loss_reduction` must be "
                "'mean' or 'sum'."
            )

        scale = (
            target.numel()
            if loss_reduction == "mean"
            else 1
        )

        def loss_fn(
            prediction,
            observed,
        ):
            return scale * loss(
                prediction,
                observed,
                **loss_kwargs,
            )

        return loss_fn

    @staticmethod
    def _loss_kwargs(
        batch,
        loss_inputs: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        return {
            argument: batch[column]
            for argument, column
            in dict(loss_inputs or {}).items()
        }

    def parameter_gradient(
        self,
        batch,
    ):
        return parameter_gradient(self.stateless_model())(
            (self.params,),
            self.inputs(batch),
        )[0]

    def parameter_hessian(
        self,
        batch,
        *,
        approximator="squared_jacobian",
        **kwargs,
    ):
        return parameter_hessian_diagonal(
            self.stateless_model(),
            approximator=approximator,
            **kwargs,
        )(
            (self.params,),
            self.inputs(batch),
        )[0]

    def fisher_score(
        self,
        batch,
        loss: Callable,
        *,
        target=None,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
    ):
        if target is None:
            target = self.target(batch)

        loss_fn = self._loss(
            loss,
            target,
            self._loss_kwargs(
                batch,
                loss_inputs,
            ),
            loss_reduction,
        )

        return fisher_score(
            self.stateless_model(),
            loss_fn,
        )(
            (self.params,),
            self.inputs(batch),
            target,
        )[0]

    def fisher_information(
        self,
        batch,
        loss: Callable,
        *,
        target=None,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
        approximator=DEFAULT_APPROXIMATOR,
        **kwargs,
    ):
        if target is None:
            target = self.target(batch)

        loss_fn = self._loss(
            loss,
            target,
            self._loss_kwargs(
                batch,
                loss_inputs,
            ),
            loss_reduction,
        )

        return fisher_information_diagonal(
            self.stateless_model(),
            loss_fn,
            approximator=approximator,
            **kwargs,
        )(
            (self.params,),
            self.inputs(batch),
            target,
        )[0]

    def doubtscore(
        self,
        candidate_batch,
        training_batch,
        loss: Callable,
        *,
        target=None,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
        use_reciprocal: bool = False,
    ):
        if target is None:
            target = self.target(training_batch)

        loss_fn = self._loss(
            loss,
            target,
            self._loss_kwargs(
                training_batch,
                loss_inputs,
            ),
            loss_reduction,
        )

        return doubtscore(
            self.stateless_model(),
            loss_fn,
            use_reciprocal=use_reciprocal,
        )(
            (self.params,),
            self.inputs(candidate_batch),
            self.inputs(training_batch),
            target,
        )[0]

    def information_sensitivity(
        self,
        candidate_batch,
        training_batch,
        loss: Callable,
        *,
        target=None,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
        approximator=DEFAULT_APPROXIMATOR,
        use_reciprocal: bool = False,
        **kwargs,
    ):
        if target is None:
            target = self.target(training_batch)

        loss_fn = self._loss(
            loss,
            target,
            self._loss_kwargs(
                training_batch,
                loss_inputs,
            ),
            loss_reduction,
        )

        return information_sensitivity(
            self.stateless_model(),
            loss_fn,
            approximator=approximator,
            use_reciprocal=use_reciprocal,
            **kwargs,
        )(
            (self.params,),
            self.inputs(candidate_batch),
            self.inputs(training_batch),
            target,
        )[0]

    def accumulate_fisher_score(
        self,
        dataloader,
        loss: Callable,
        *,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
        prepare_batch: Callable | None = None,
    ) -> tuple[ParameterTree, int]:

        total = None
        n_samples = 0

        for batch in tqdm(dataloader, desc="Accumulating Fisher score"):
            if prepare_batch is not None:
                batch = prepare_batch(batch)
            target = self.target(batch)
            total = _accumulate(
                total,
                self.fisher_score(
                    batch,
                    loss,
                    target=target,
                    loss_inputs=loss_inputs,
                    loss_reduction=loss_reduction,
                ),
            )
            n_samples += target.shape[0]

        if total is None:
            raise ValueError(
                "Cannot compute Fisher score "
                "from an empty dataset."
            )

        return total, n_samples

    def accumulate_fisher_information(
        self,
        dataloader,
        loss: Callable,
        *,
        loss_inputs: Mapping[str, str] | None = None,
        loss_reduction: str = "mean",
        approximator=DEFAULT_APPROXIMATOR,
        prepare_batch: Callable | None = None,
        **kwargs,
    ) -> tuple[ParameterTree, int]:

        total = None
        n_samples = 0

        for batch in tqdm(dataloader, desc="Accumulating Fisher info."):
            if prepare_batch is not None:
                batch = prepare_batch(batch)

            target = self.target(batch)
            total = _accumulate(
                total,
                self.fisher_information(
                    batch,
                    loss,
                    target=target,
                    loss_inputs=loss_inputs,
                    loss_reduction=loss_reduction,
                    approximator=approximator,
                    **kwargs,
                ),
            )

            n_samples += target.shape[0]

        if total is None:
            raise ValueError(
                "Cannot compute Fisher information "
                "from an empty dataset."
            )

        return total, n_samples
