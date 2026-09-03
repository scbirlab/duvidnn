"""Functional model invocation."""

from collections.abc import Mapping

from torch import nn
from torch.func import functional_call

from .base import ModelInvoker


def functional_predict(
    model: nn.Module,
    invoker: ModelInvoker,
    batch: Mapping,
    params: Mapping | None = None,
    buffers: Mapping | None = None
):
    """Invoke a model with substituted parameters and buffers."""

    inputs = invoker.inputs(batch)

    if params is None:
        params = dict(model.named_parameters())
    if buffers is None:
        buffers = dict(model.named_buffers())

    return functional_call(
        model,
        (
            params,
            buffers,
        ),
        args=(),
        kwargs=inputs,
    )


def make_stateless_model(
    model: nn.Module,
    buffers: Mapping | None = None
):
    """Adapt a module to Duvida's stateless model interface."""

    if buffers is None:
        buffers = dict(model.named_buffers())

    def stateless_model(
        inputs,
        params
    ):
        return functional_call(
            model,
            (params, buffers),
            args=(),
            kwargs=inputs,
        )

    return stateless_model
