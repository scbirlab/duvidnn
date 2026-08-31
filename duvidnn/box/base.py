""""""
from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Mapping
import os

from aspect.data import DataPipeline
from torch.nn import Module

from ..invoke import ModelInvoker
from ..mapping import ColumnMap


def _resolve_pipeline(
    pipeline: DataPipeline | Mapping[str, ...] | None = None
):
    if isinstance(pipeline, Mapping) or pipeline is None:
        return DataPipeline(column_transforms=pipeline)
    elif isinstance(pipeline, str):
        if os.path.exists(pipeline):
            return DataPipeline.from_file(pipeline)
        else:
            raise FileNotFoundError(
                f"`pipeline` was a string, but no file found called {pipeline}"
            )
    elif isinstance(pipeline, (Callable, DataPipeline)):
        return pipeline
    else:
        raise ValueError(
            "`pipeline` must be a dict, filename, or aspect.DataPipeline, "
            f"but was {type(pipeline)}: {pipeline}"
        )


class Box:
    """Data processing, model invocation, and optional training."""

    def __init__(
        self,
        model: Module,
        input_map: ColumnMap,
        pipeline: Mapping[str, Any] | DataPipeline | None = None,
        data: Any | None = None,
        trainer: Any | None = None
    ) -> None:
        self.pipeline = _resolve_pipeline(pipeline)
        self.model = model
        self.input_map = input_map
        self.data = data
        self.trainer = trainer

        self.invoker = ModelInvoker(
            model=self.model,
            input_map=self.input_map,
        )

    def prepare(self, data):
        """Apply the recorded Aspect pipeline."""
        return self.pipeline(data)

    def predict_batch(self, batch):
        """Predict from an already prepared runtime batch."""
        return self.invoker.predict(batch)

    def supervised_batch(self, batch):
        """Return predictions and target from an already prepared batch."""
        return self.invoker.supervised(batch)
