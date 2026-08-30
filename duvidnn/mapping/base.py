"""Mapping between dataset columns and model inputs."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnMap:
    """Map batch fields onto model inputs.

    Parameters
    ==========
    inputs
        Mapping of model input names to dataset column names.
    target
        Dataset column containing the training target.
    """

    inputs: Mapping[str, str]
    target: str

    def map_inputs(
        self,
        batch: Mapping[str, ...]
    ) -> dict[str, ...]:
        missing = [
            column
            for column in self.inputs.values()
            if column not in batch
        ]

        if missing:
            raise KeyError(
                "Missing input columns from batch: "
                + ", ".join(missing)
            )
        return {
            argument: batch[column]
            for argument, column in self.inputs.items()
        }

    def map_target(
        self,
        batch: Mapping[str, ...]
    ):
        try:
            return batch[self.target]
        except KeyError as e:
            raise KeyError(
                f"Missing target column {self.target!r} from batch: {batch}."
            ) from e

    def map_batch(
        self,
        batch: Mapping[str, ...]
    ) -> tuple[dict[str, ...], ...]:
        return (
            self.map_inputs(batch),
            self.map_target(batch),
        )

    def __call__(*args, **kwargs):
        return self.map_batch(*args, **kwargs)