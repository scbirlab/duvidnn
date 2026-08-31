"""Interpret declarative PyTorch model specifications."""

from typing import TYPE_CHECKING, Any
from collections.abc import Mapping

if TYPE_CHECKING:
    from torch.nn import Module
else:
    Module = Any


def instantiate_model(
    config: Mapping[str, Any]
) -> Module:
    """Instantiate an nn.Module from a class_path/init_args specification.

    """
    from jsonargparse import ArgumentParser
    from torch import nn

    parser = ArgumentParser()
    parser.add_subclass_arguments(
        nn.Module,
        "model",
    )
    parsed = parser.parse_object({
        "model": dict(config),
    })
    instantiated = parser.instantiate(parsed)
    return instantiated.model
