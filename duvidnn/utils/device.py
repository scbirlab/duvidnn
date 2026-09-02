"""Device movement utilities."""

from collections.abc import Mapping

from torch import Tensor


def move_to_device(
    value,
    device,
):
    """Recursively move tensor-like runtime data to a device."""

    if isinstance(value, Tensor):
        return value.to(device)

    if isinstance(value, Mapping):
        return type(value)({
            key: move_to_device(
                item,
                device,
            )
            for key, item
            in value.items()
        })

    if isinstance(value, tuple):
        return tuple(
            move_to_device(
                item,
                device,
            )
            for item in value
        )

    if isinstance(value, list):
        return [
            move_to_device(
                item,
                device,
            )
            for item in value
        ]

    to = getattr(
        value,
        "to",
        None,
    )

    if callable(to):
        return to(device)

    return value