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

    if isinstance(value, (tuple, list)):
        return type(value)(
            move_to_device(
                item,
                device,
            )
            for item in value
        )

    to = getattr(value, "to", None)

    if callable(to):
        moved = to(device)
        return value if moved is None else moved

    return value