"""Column-wise dataset collation."""

from collections.abc import Callable, Mapping, Iterable
from typing import Any, TypeAlias


CollateFn: TypeAlias = Callable[[Iterable[Any]], Any]


def identity_collate(values: Iterable[...]) -> list[...]:
    """Return column values unchanged."""
    return list(values)


def _columnize(batch: Mapping[str, Iterable[...]] | Iterable[Mapping[str, ...]]):
    if isinstance(batch, Mapping):
        return batch

    if len(batch) == 0:
        return {}

    return {
        column: [row[column] for row in batch]
        for column in batch[0]
    }


class ColumnCollator:
    """Collate row-oriented dataset examples column by column.

    Parameters
    ==========
    collators
        Optional mapping from dataset column names to custom collators.
    default
        Collator used for columns without an explicit custom collator.
        If ``None``, values are returned as a list.
    """

    def __init__(
        self,
        collators: Mapping[str, CollateFn] | None = None,
        default: CollateFn | None = None,
        columnizer: Callable | None = None
    ):
        self.collators = dict(collators or {})
        self.default = default or identity_collate
        self.columnizer = columnizer or _columnize

    def collate_column(
        self,
        name: str,
        values: Iterable[...]
    ) -> Any:
        collator = self.collators.get(
            name,
            self.default,
        )
        return collator(values)

    def __call__(
        self, 
        batch: Mapping[str, Iterable[...]] | Iterable[Mapping[str, ...]]
    ) -> dict[str, list[...]]:
        columns = self.columnizer(batch)
        return {
            name: self.collate_column(
                name,
                values,
            )
            for name, values in columns.items()
        }
