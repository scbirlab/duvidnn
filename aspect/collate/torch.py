"""PyTorch collation helpers."""

from collections.abc import Mapping, Iterable

from .base import ColumnCollator, CollateFn, _columnize


class TorchColumnCollator(ColumnCollator):
    """Column collator using PyTorch's default collation."""

    def __init__(
        self,
        collators: Mapping[str, CollateFn] | None = None
    ):
        from torch.utils.data import default_collate
        super().__init__(
            collators=collators,
            default=default_collate,
        )

    def __call__(
        self,
        *args, **kwargs
    ):
        import torch
        base_collated = super().__call__(*args, **kwargs)
        collated = {}
        for key, val in base_collated.items():
            if torch.is_tensor(val):
                new_val = val
            elif isinstance(val, (tuple, list)):
                new_val = torch.stack(val, dim=0) 
            else:
                raise ValueError(f"Don't know how to collate {key=}, {val=}")
            collated[key] = new_val
        return collated
