"""Checkpointable composition of data processing and PyTorch models."""

from typing import Any, TypeAlias
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
import json
import os

from aspect.data import DataPipeline
from carabiner import print_err
import torch
from torch import nn

from ..config import instantiate_model
from ..invoke import ModelInvoker
from ..mapping import ColumnMap
from ..checkpoint_utils import save_json, load_json


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pt"
MODEL_FILENAME = "model.pt"
PIPELINE_DIRNAME: str = "data"

PipelineLike: TypeAlias = DataPipeline | Mapping[str, Any] | str | Callable


def _json_copy(
    value: Mapping[Any, Any],
) -> dict[str, ...]:
    """Return a JSON-native copy, validating serializability."""
    return json.loads(
        json.dumps(value)
    )


def _resolve_pipeline(pipeline: PipelineLike | None = None) -> DataPipeline:
    """Resolve a pipeline specification."""
    if isinstance(pipeline, DataPipeline):
        return pipeline

    if callable(pipeline):
        print_err(
            "[WARN] Only aspect.DataPipeline as a data pipeline allows serialization.", 
            f"You are using {pipeline}",
        )
        return pipeline
    
    if isinstance(pipeline, Mapping) or pipeline is None:
        return DataPipeline(
            column_transforms=dict(pipeline or {}),
        )

    if isinstance(pipeline, str):
        if os.path.exists(pipeline):
            return DataPipeline.from_file(pipeline)

        raise FileNotFoundError(
            "`pipeline` was a string, but no file found "
            f"called `{pipeline}`"
        )

    raise ValueError(
        "`pipeline` must be a dict, filename, callable, "
        "or aspect.DataPipeline, "
        f"but was {type(pipeline)}: {pipeline}"
    )


def _materialize_model(module: nn.Module) -> None:
    """Build modules before state restoration.

    Applies mainly to in-package models.

    """
    if hasattr(module, "_built"):
        if getattr(module, "_built") is False:
            build = getattr(module, "build")
            if callable(build):
                build()

    for child in module.children():
        _materialize_model(child)


class Box:
    """Data processing, model invocation, and optional training.

    A Box constructed with :meth:`from_config` retains the declarative model
    specification used to construct its PyTorch module. Such Boxes are saved
    as configuration plus a ``state_dict``.

    A Box constructed directly from an arbitrary existing ``nn.Module`` has
    no generic reconstruction recipe. Such Boxes remain supported and fall
    back to whole-model PyTorch serialization.

    """

    def __init__(
        self,
        model: nn.Module,
        input_map: ColumnMap,
        pipeline: PipelineLike | None = None,
        # TODO: Make type hints more specific
        data: Any | None = None,
        trainer: Any | None = None,
        model_config: Mapping[str, Any] | None = None
    ):
        self.pipeline = _resolve_pipeline(pipeline)
        self.model = model
        self.input_map = input_map
        self.data = data
        self.trainer = trainer

        self.model_config = (
            _json_copy(model_config)
            if model_config is not None
            else None
        )
        self.invoker = ModelInvoker(
            model=self.model,
            input_map=self.input_map,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str
    ) -> "Box":
        """Construct a Box from JSON-compatible configuration."""
        if isinstance(config, str):
            if os.path.exists(config):
                config = load_json(config)
            else:
                raise FileNotFoundError(
                    "`config` was string, but filename "
                    f"called `{config}` not found."
                )

        config = deepcopy(dict(config))
        try:
            model_config = config["model"]
        except KeyError as error:
            raise ValueError(
                f"Box config requires a 'model' specification. Received {config=}"
            ) from error

        if model_config is None:
            raise ValueError(
                f"Box.from_config() requires a model specification. "
                "Opaque models can only be reconstructed from a "
                "saved whole-model checkpoint. Received {config=}."
            )

        try:
            input_map_config = config["input_map"]
        except KeyError as error:
            raise ValueError(
                f"Box config requires an 'input_map'. Received {config=}."
            ) from error

        model_config = _json_copy(model_config)
        model = instantiate_model(model_config)
        
        input_map = ColumnMap.from_config(input_map_config)

        pipeline_config = config.get("pipeline", {})
        pipeline = DataPipeline.from_config(pipeline_config)
        return cls(
            model=model,
            input_map=input_map,
            pipeline=pipeline,
            model_config=model_config,
        )

    def to_config(self, filename: str | None = None) -> dict[str, ...]:
        """Return the durable JSON configuration for this Box."""
        if not isinstance(self.pipeline, DataPipeline):
            raise TypeError(
                "Checkpoint serialization currently requires "
                "an aspect.DataPipeline. Arbitrary callable "
                "pipelines do not have a generic reconstruction "
                "representation."
            )

        config = {
            "model": (
                _json_copy(
                    self.model_config
                )
                if self.model_config is not None
                else None
            ),
            "pipeline": self.pipeline.to_config(),
            "input_map": self.input_map.to_config(),
        }
        if filename is not None:
            save_json(config, filename)
        return config

    def save(
        self, 
        path: str | os.PathLike,
        **pipeline_kwargs
    ) -> None:
        """Save configuration and model state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        config = self.to_config()
        pipeline_config = config.pop("pipeline")

        save_json(config, path / CONFIG_FILENAME)

        self.pipeline.save(
            path / PIPELINE_DIRNAME,
            **pipeline_kwargs,
        )

        if self.model_config is None:
            torch.save(
                self.model,
                path / MODEL_FILENAME,
            )
        else:
            _materialize_model(self.model)
            torch.save(
                self.model.state_dict(),
                path / WEIGHTS_FILENAME,
            )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        map_location: Any = "cpu",
        cache_dir: str | None = None
    ) -> "Box":
        """Restore a Box checkpoint."""
        path = Path(path)
        config = load_json(path / CONFIG_FILENAME)

        input_map = ColumnMap.from_config(config.get("input_map"))
        pipeline = DataPipeline.load(
            path / PIPELINE_DIRNAME,
            cache_dir=cache_dir,
        )
        pipeline_config = pipeline.to_config()
        config["pipeline"] = pipeline_config

        model_config = config.get("model")
        box = cls.from_config(config)
        box.pipeline = pipeline
        
        if model_config is not None:
            _materialize_model(box.model)
            state_dict = torch.load(
                path / WEIGHTS_FILENAME,
                map_location=map_location,
                weights_only=True,
            )
            box.model.load_state_dict(state_dict)
            return box

        model = torch.load(
            path / MODEL_FILENAME,
            map_location=map_location,
            weights_only=False,
        )
        return box

    def prepare(self, data):
        """Apply the recorded Aspect pipeline."""
        return self.pipeline(data)

    def predict_batch(
        self,
        batch,
    ):
        """Predict from an already prepared runtime batch."""
        return self.invoker.predict(batch)

    def supervised_batch(
        self,
        batch,
    ):
        """Return predictions and target from an already prepared batch."""
        return self.invoker.supervised(batch)
