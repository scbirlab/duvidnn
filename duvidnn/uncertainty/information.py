from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from ..invoke import DuvidaModel
from ..invoke.duvida import DEFAULT_APPROXIMATOR


def _loss_for(
    box,
    explicit_loss: Callable | None,
):
    if explicit_loss is not None:
        return explicit_loss

    if box.trainer is not None:
        return box.trainer.loss

    raise ValueError(
        "No loss was provided, and the Box does not have a trainer. "
        "Either provide loss explicitly, or use the Box for training."
    )


def _loss_inputs_for(
    box,
    explicit_loss_inputs,
):
    if explicit_loss_inputs is not None:
        return explicit_loss_inputs

    if box.trainer is not None:
        return box.trainer.loss_inputs

    return None


@dataclass
class DoubtScore:
    loss: Callable | None = None
    loss_inputs: Mapping[str, str] | None = None
    loss_reduction: str = "mean"
    reciprocal: bool = False

    name = "doubtscore"

    def prepare(
        self,
        box,
        *,
        batch_size: int = 32,
        device=None
    ):
        if (
            box.training_derivatives is None
            or box.training_derivatives.fisher_score is None
        ):
            box.compute_training_derivatives(
                fisher_score=True,
                loss=_loss_for(
                    box,
                    self.loss,
                ),
                loss_inputs=_loss_inputs_for(
                    box,
                    self.loss_inputs,
                ),
                loss_reduction=self.loss_reduction,
                batch_size=batch_size,
                device=device,
            )
        return box.training_derivatives.fisher_score

    def __call__(
        self,
        *,
        box,
        batch,
        prediction=None,
        state
    ):
        functional = DuvidaModel(
            box.model,
            box.invoker,
        )
        score = functional.doubtscore_from_fisher(
            batch,
            state,
            use_reciprocal=self.reciprocal,
        )

        return functional.parameter_rms(score)


@dataclass
class InformationSensitivity:
    loss: Callable | None = None
    loss_inputs: Mapping[str, str] | None = None
    loss_reduction: str = "mean"
    approximator: str = DEFAULT_APPROXIMATOR
    reciprocal: bool = False
    approximator_kwargs: Mapping | None = None

    name = "information_sensitivity"

    def prepare(
        self,
        box,
        *,
        batch_size: int = 32,
        device=None,
    ):
        approximator_kwargs = dict(
            self.approximator_kwargs or {}
        )
        need_score = (
            box.training_derivatives is None
            or box.training_derivatives.fisher_score is None
        )
        need_information = (
            box.training_derivatives is None
            or box.training_derivatives.fisher_information is None
        )

        if need_score:
            box.compute_training_derivatives(
                fisher_score=True,
                loss=_loss_for(
                    box,
                    self.loss,
                ),
                loss_inputs=_loss_inputs_for(
                    box,
                    self.loss_inputs,
                ),
                loss_reduction=self.loss_reduction,
                batch_size=batch_size,
                device=device,
            )

        if need_information:
            box.compute_training_derivatives(
                fisher_information={
                    "approximator": self.approximator,
                    **approximator_kwargs,
                },
                loss=_loss_for(
                    box,
                    self.loss,
                ),
                loss_inputs=_loss_inputs_for(
                    box,
                    self.loss_inputs,
                ),
                loss_reduction=self.loss_reduction,
                batch_size=batch_size,
                device=device,
            )

        return {
            "fisher_score": (
                box.training_derivatives
                .fisher_score
            ),
            "fisher_information": (
                box.training_derivatives
                .fisher_information
            ),
        }

    def __call__(
        self,
        *,
        box,
        batch,
        prediction=None,
        state
    ):
        functional = DuvidaModel(
            box.model,
            box.invoker,
        )
        approx_kwargs = self.approximator_kwargs or {}
        score = (
            functional
            .information_sensitivity_from_fisher(
                batch,
                fisher_score=state["fisher_score"],
                fisher_information=state["fisher_information"],
                approximator=self.approximator,
                use_reciprocal=self.reciprocal,
                **dict(approx_kwargs),
            )
        )

        return functional.parameter_rms(score)
