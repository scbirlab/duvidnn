""""""

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Mapping, Iterable

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

def _get_name(x: Callable) -> str:
    attributes = (
        "name",
        "__name__",
        "__qualname__",
        "__str__",
        "__repr__",
    )
    for a in attributes:
        if hasattr(x, a):
            this_attribute = getattr(x, a)
            if callable(this_attribute):
                to_return = this_attribute()
            else:
                to_return =  this_attribute
            return str(to_return)
    try:
        to_return = str(type(x))
    except:
        raise AttributeError(
            f"Object {type(x)} has no name, and cannot be converted to a string."
        )
    else:
        return to_return
    

def normalize_uncertainty(
    uncertainty: Callable | Iterable[Callable] | Mapping[str, Callable] | None = None
) -> dict[str, Any]:
    if uncertainty is None:
        return {}

    if isinstance(uncertainty, Mapping):
        return dict(uncertainty)

    if isinstance(uncertainty, (list, tuple)):
        output = {}

        for method in uncertainty:
            name = _get_name(method)

            if name in output:
                raise ValueError(
                    f"Duplicate uncertainty method name: {name!r}. "
                    "Use a dict to provide explicit output names."
                )

            output[name] = method

        return output

    try:
        name = _get_name(uncertainty)
    except AttributeError as error:
        raise TypeError(
            "Uncertainty must be a method, list of methods, "
            "dict of names to methods, or None."
        ) from error

    return {name: uncertainty}
