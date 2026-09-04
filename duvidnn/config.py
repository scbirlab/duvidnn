"""Interpret declarative PyTorch model specifications."""

from typing import TYPE_CHECKING, Any
from collections.abc import Mapping

if TYPE_CHECKING:
    from torch.nn import Module
    from .training import Trainer
else:
    Module = Any
    Trainer = Any


def instantiate_object(
    config: Mapping[str, Any],
):
    """Instantiate one class_path/init_args object."""

    from importlib import import_module

    config = dict(config)
    class_path = config["class_path"]
    init_args = dict(config.get("init_args", {}))

    module_name, class_name = class_path.rsplit(".", 1)

    cls = getattr(
        import_module(module_name),
        class_name,
    )
    return cls(**init_args)


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


def instantiate_trainer(
    config: Mapping[str, Any],
) -> Trainer:
    """Instantiate a Trainer from declarative configuration."""

    from jsonargparse import ArgumentParser

    from .training import Trainer

    parser = ArgumentParser()
    parser.add_class_arguments(
        Trainer,
        "trainer",
    )
    parsed = parser.parse_object({
        "trainer": dict(config),
    })
    instantiated = parser.instantiate(parsed)
    return instantiated.trainer


def instantiate_uncertainty(config):
    if config is None:
        return None

    return {
        name: instantiate_object(method)
        for name, method
        in config.items()
    }