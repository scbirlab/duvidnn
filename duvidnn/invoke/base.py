"""Model invocation from mapped dataset batches."""

from collections.abc import Mapping

from torch import nn

from ..mapping import ColumnMap


class ModelInvoker:
    """Invoke a model using inputs mapped from a dataset batch."""

    def __init__(
        self,
        model: nn.Module,
        input_map: ColumnMap,
    ) -> None:
        self.model = model
        self.input_map = input_map

    def inputs(
        self,
        batch: Mapping[str, ...],
    ) -> dict[str, ...]:
        return self.input_map.map_inputs(batch)

    def target(
        self,
        batch: Mapping[str, ...],
    ):
        return self.input_map.map_target(batch)

    def predict(
        self,
        batch: Mapping[str, ...],
    ):
        return self.model(**self.inputs(batch))

    def supervised(
        self,
        batch: Mapping[str, ...],
    ):
        prediction = self.predict(batch)
        target = self.target(batch)

        return prediction, target

    def __call__(
        self,
        batch: Mapping[str, ...],
    ):
        return self.predict(batch)
